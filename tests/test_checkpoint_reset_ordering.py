from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from crab.system import CrabSystem
from crab.models import utc_now


def _succeeded_result() -> MagicMock:
    result = MagicMock()
    result.status.value = "succeeded"
    result.checkpoint_id = "ckpt-1"
    result.failure_code.value = "none"
    result.finished_at = utc_now()
    return result


class CheckpointResetOrderingTest(unittest.TestCase):
    """The inspector re-baseline (clear_refs) must run AFTER the sandbox is
    resumed.

    CRIU dumps the tasks with --leave-running while the container is paused; the
    parasite teardown on resume dirties 1-3 residual soft-dirty pages *after*
    the clear. If the reset runs before resume, those residual pages latch a
    false process_changed=True on an otherwise idle sandbox. So
    inspector.mark_checkpoint_complete must be ordered after _resume_sandbox.
    """

    def _make_system(self, order: list[str]) -> CrabSystem:
        system = CrabSystem.__new__(CrabSystem)
        system.telemetry = MagicMock()
        system._telemetry_attrs = MagicMock(return_value={})
        system._next_pending_live_request = MagicMock(return_value=None)
        system._build_checkpoint_metadata = MagicMock(return_value={})
        system._journal_lifecycle = MagicMock()
        system._release_response_gate = MagicMock()
        system._refresh_interceptor_pending_state = MagicMock()
        system._should_resume_after_checkpoint = MagicMock(return_value=True)

        result = _succeeded_result()
        system.executor = MagicMock()
        system.executor.run_checkpoint = MagicMock(
            side_effect=lambda job: (order.append("run_checkpoint"), result)[1]
        )
        system.executor.submit_checkpoint = MagicMock(
            return_value=MagicMock(result=MagicMock(return_value=result))
        )
        system.scheduler = MagicMock()

        def _pause(_sid: str) -> bool:
            order.append("pause")
            return True

        def _resume(_sid: str) -> None:
            order.append("resume")

        system._pause_for_manual_checkpoint = MagicMock(side_effect=_pause)
        system._resume_sandbox = MagicMock(side_effect=_resume)

        system.inspector = MagicMock()
        system.inspector.mark_checkpoint_complete = MagicMock(
            side_effect=lambda *a, **k: order.append("inspector_reset")
        )
        return system

    def test_checkpoint_once_resets_inspector_after_resume(self) -> None:
        order: list[str] = []
        system = self._make_system(order)
        with patch("crab.system.start_operation", return_value=MagicMock()):
            system.checkpoint_once("sbx-1", leave_running=True)

        self.assertIn("resume", order)
        self.assertIn("inspector_reset", order)
        self.assertLess(
            order.index("resume"),
            order.index("inspector_reset"),
            f"inspector reset must run after resume; got order={order}",
        )
        system.inspector.mark_checkpoint_complete.assert_called_once()

    def test_scheduled_flow_resets_inspector_after_resume(self) -> None:
        order: list[str] = []
        system = self._make_system(order)
        # _execute_checkpoint_flow drives the scheduler-based path.
        decision = MagicMock()
        decision.should_checkpoint = True
        decision.reason = "auto"
        decision.checkpoint_process = True
        decision.checkpoint_filesystem = False
        decision.leave_running = True
        decision.is_incremental_process = False
        decision.parent_process_checkpoint_id = None
        decision.produce_pre_dump = False
        decision.policy_name = "policy"
        decision.metadata = {}
        system.scheduler.query_checkpoint = MagicMock(return_value=decision)
        system._txn_active = MagicMock(return_value=False)
        system._merge_active = MagicMock(return_value=False)

        with patch("crab.system.start_operation", return_value=MagicMock()):
            system._execute_checkpoint_flow("sbx-1", pending_request=None)

        self.assertIn("resume", order)
        self.assertIn("inspector_reset", order)
        self.assertLess(
            order.index("resume"),
            order.index("inspector_reset"),
            f"inspector reset must run after resume; got order={order}",
        )


if __name__ == "__main__":
    unittest.main()
