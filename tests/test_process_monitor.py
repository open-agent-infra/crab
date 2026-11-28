from __future__ import annotations

import unittest
from unittest.mock import patch

from crab.host_inspector import process_monitor


class ResetSoftDirtyStabilizeTest(unittest.TestCase):
    """Covers the post-checkpoint soft-dirty stabilization loop.

    CRIU dumps tasks with --leave-running; its parasite/resume writes can leave
    a few residual soft-dirty pages that land around the reset's clear_refs. The
    stabilize path re-clears until the writable set reads clean (idle process)
    or attempts run out (genuinely busy process, which the next status() poll
    still reports).
    """

    def test_no_stabilize_by_default_leaves_single_clear(self) -> None:
        with patch.object(process_monitor, "clear_soft_dirty") as clear, patch.object(
            process_monitor, "dirty_pids"
        ) as scan:
            result = process_monitor.reset_soft_dirty_for_pids({111, 222})
        self.assertEqual(result, {111, 222})
        # One clear per pid, and NO verification scan when stabilize is off.
        self.assertEqual(clear.call_count, 2)
        scan.assert_not_called()

    def test_idle_criu_residual_is_recleared_until_clean(self) -> None:
        # Sleep-first loop: iter 1 sleeps, sees the residual page, re-clears;
        # iter 2 sleeps, reads clean (idle process, no ongoing writes), stops.
        scan_returns = [{111}, set()]
        with patch.object(process_monitor, "clear_soft_dirty") as clear, patch.object(
            process_monitor, "dirty_pids", side_effect=scan_returns
        ) as scan, patch.object(process_monitor.time, "sleep") as sleep:
            result = process_monitor.reset_soft_dirty_for_pids(
                {111}, stabilize=True, stabilize_attempts=3, stabilize_delay=0.05
            )
        self.assertEqual(result, {111})
        # initial clear (1) + one re-clear of the residual pid (1).
        self.assertEqual(clear.call_count, 2)
        self.assertEqual(scan.call_count, 2)
        # One sleep before each of the two scans (sleep-first ordering).
        self.assertEqual(sleep.call_count, 2)
        sleep.assert_called_with(0.05)

    def test_busy_process_exhausts_attempts_and_is_not_masked(self) -> None:
        # A genuinely busy process keeps re-dirtying: every scan reports dirty,
        # so we re-clear up to the attempt limit and then stop. We must NOT loop
        # forever, and the residual is left for the next status() poll to catch.
        with patch.object(process_monitor, "clear_soft_dirty") as clear, patch.object(
            process_monitor, "dirty_pids", return_value={111}
        ) as scan, patch.object(process_monitor.time, "sleep"):
            result = process_monitor.reset_soft_dirty_for_pids(
                {111}, stabilize=True, stabilize_attempts=3, stabilize_delay=0.0
            )
        self.assertEqual(result, {111})
        # initial clear (1) + one re-clear per attempt (3) = 4; bounded, not infinite.
        self.assertEqual(clear.call_count, 4)
        self.assertEqual(scan.call_count, 3)

    def test_vanished_pid_during_stabilize_is_dropped(self) -> None:
        def clear(pid: int) -> None:
            if pid == 111:
                raise FileNotFoundError

        with patch.object(process_monitor, "clear_soft_dirty", side_effect=clear), patch.object(
            process_monitor, "dirty_pids", return_value={111}
        ), patch.object(process_monitor.time, "sleep"):
            # 111 is cleared once at entry (added to baseline), then vanishes on
            # the re-clear attempt; it must be discarded without raising.
            result = process_monitor.reset_soft_dirty_for_pids(
                {111}, stabilize=True, stabilize_attempts=2, stabilize_delay=0.0
            )
        # The initial clear raised FileNotFoundError, so 111 never entered the
        # cleared set and stabilization is a no-op.
        self.assertEqual(result, set())


if __name__ == "__main__":
    unittest.main()
