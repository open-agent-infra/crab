from __future__ import annotations

import unittest

from crab import SchedulerConfig
from crab.ids import CheckpointId, SandboxId
from crab.models import SchedulerCheckpointDecision
from crab.scheduler import (
    InMemorySchedulerStateStore,
    _resolve_incremental_process,
)


def _full_decision(should: bool = True, proc: bool = True) -> SchedulerCheckpointDecision:
    return SchedulerCheckpointDecision(
        should_checkpoint=should,
        checkpoint_process=proc,
        checkpoint_filesystem=False,
        leave_running=True,
        reason="r",
        policy_name="p",
    )


class IncrementalDecisionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sbx = SandboxId("sbx-1")
        self.store = InMemorySchedulerStateStore()
        self.config = SchedulerConfig(
            incremental_process_enabled=True,
            full_process_checkpoint_interval=4,
            max_process_chain_length=10,
        )

    def _resolve(self, decision: SchedulerCheckpointDecision) -> SchedulerCheckpointDecision:
        return _resolve_incremental_process(
            decision,
            config=self.config,
            last_process_checkpoint_id=self.store.get_last_process_checkpoint(self.sbx),
            process_chain_length=self.store.get_process_chain_length(self.sbx),
        )

    def test_scheduler_config_enables_incremental_process_by_default(self) -> None:
        self.assertTrue(SchedulerConfig().incremental_process_enabled)

    def test_first_checkpoint_is_anchor_with_pre_dump(self) -> None:
        # Chain root: not "incremental" (no parent), but produces a pre_dump
        # so the next checkpoint can chain off it.
        decision = self._resolve(_full_decision())
        self.assertFalse(decision.is_incremental_process)
        self.assertIsNone(decision.parent_process_checkpoint_id)
        self.assertTrue(decision.produce_pre_dump)

    def test_subsequent_checkpoints_chain(self) -> None:
        self.store.record_process_checkpoint(
            self.sbx, CheckpointId("c-1"), is_incremental=False
        )
        decision = self._resolve(_full_decision())
        self.assertTrue(decision.is_incremental_process)
        self.assertTrue(decision.produce_pre_dump)
        self.assertEqual(decision.parent_process_checkpoint_id, CheckpointId("c-1"))

    def test_chain_resets_at_interval(self) -> None:
        # interval=4 -> after the chain root (anchor), we expect 3 incrementals
        # (chain_length 1, 2, 3). The 4th would push chain_length to 4 == interval,
        # so the resolver should force a fresh anchor (incremental=False, pair=True).
        self.store.record_process_checkpoint(
            self.sbx, CheckpointId("c-0"), is_incremental=False
        )
        for i in range(1, 4):
            d = self._resolve(_full_decision())
            self.assertTrue(d.is_incremental_process, f"step {i}")
            self.assertTrue(d.produce_pre_dump, f"step {i}")
            self.store.record_process_checkpoint(
                self.sbx, CheckpointId(f"c-{i}"), is_incremental=True
            )
        # Now chain_length=3; next should be a fresh anchor.
        decision = self._resolve(_full_decision())
        self.assertFalse(decision.is_incremental_process)
        self.assertTrue(decision.produce_pre_dump)
        self.assertIsNone(decision.parent_process_checkpoint_id)

    def test_disabled_config_never_returns_incremental(self) -> None:
        config = SchedulerConfig(incremental_process_enabled=False)
        decision = _resolve_incremental_process(
            _full_decision(),
            config=config,
            last_process_checkpoint_id=CheckpointId("c-1"),
            process_chain_length=2,
        )
        self.assertFalse(decision.is_incremental_process)
        self.assertFalse(decision.produce_pre_dump)

    def test_max_chain_length_caps_chain(self) -> None:
        config = SchedulerConfig(
            incremental_process_enabled=True,
            full_process_checkpoint_interval=100,  # interval doesn't trigger
            max_process_chain_length=3,
        )
        decision = _resolve_incremental_process(
            _full_decision(),
            config=config,
            last_process_checkpoint_id=CheckpointId("c-1"),
            process_chain_length=3,
        )
        self.assertFalse(decision.is_incremental_process)
        # Cap-triggered anchors still produce a pre_dump for the next chain.
        self.assertTrue(decision.produce_pre_dump)

    def test_skipped_decision_passes_through_unchanged(self) -> None:
        decision = self._resolve(_full_decision(should=False, proc=False))
        self.assertFalse(decision.should_checkpoint)
        self.assertFalse(decision.is_incremental_process)
        self.assertFalse(decision.produce_pre_dump)

    def test_filesystem_only_decision_stays_full(self) -> None:
        # checkpoint_process=False -> incremental concept doesn't apply.
        decision = self._resolve(_full_decision(should=True, proc=False))
        self.assertFalse(decision.is_incremental_process)
        self.assertFalse(decision.produce_pre_dump)


class StateStoreChainTests(unittest.TestCase):
    def test_record_process_checkpoint_increments_and_resets(self) -> None:
        store = InMemorySchedulerStateStore()
        sbx = SandboxId("s-1")
        self.assertEqual(store.get_process_chain_length(sbx), 0)
        self.assertIsNone(store.get_last_process_checkpoint(sbx))

        store.record_process_checkpoint(sbx, CheckpointId("c-0"), is_incremental=False)
        self.assertEqual(store.get_process_chain_length(sbx), 0)
        self.assertEqual(store.get_last_process_checkpoint(sbx), CheckpointId("c-0"))

        store.record_process_checkpoint(sbx, CheckpointId("c-1"), is_incremental=True)
        self.assertEqual(store.get_process_chain_length(sbx), 1)

        store.record_process_checkpoint(sbx, CheckpointId("c-2"), is_incremental=True)
        self.assertEqual(store.get_process_chain_length(sbx), 2)

        # Full resets to 0.
        store.record_process_checkpoint(sbx, CheckpointId("c-3"), is_incremental=False)
        self.assertEqual(store.get_process_chain_length(sbx), 0)
        self.assertEqual(store.get_last_process_checkpoint(sbx), CheckpointId("c-3"))

        store.set_process_checkpoint_base(
            sbx,
            CheckpointId("c-restored"),
            chain_length=5,
        )
        self.assertEqual(store.get_process_chain_length(sbx), 5)
        self.assertEqual(
            store.get_last_process_checkpoint(sbx),
            CheckpointId("c-restored"),
        )

        with self.assertRaisesRegex(ValueError, "chain_length"):
            store.set_process_checkpoint_base(
                sbx,
                CheckpointId("c-invalid"),
                chain_length=-1,
            )


if __name__ == "__main__":
    unittest.main()
