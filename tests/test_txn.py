"""Unit tests for the transaction API v1 (PR-B2.1): CrabSystem txn core
(adaptive base, commit/abort, scheduler suppression, journal tagging,
teardown), and the SDK Transaction handle. Host-runnable — no runc."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from crab import (
    AdapterFileSystemCWorker,
    AdapterFileSystemRWorker,
    AdapterProcessCWorker,
    AdapterProcessRWorker,
    CRExecutor,
    CRScheduler,
    CrabSystem,
    DefaultCWorker,
    DefaultRWorker,
    EBPFSandboxInspector,
    ExecutorConfig,
    InMemorySchedulerStateStore,
    InMemoryTelemetrySink,
    LocalCheckpointManager,
    RuncRuntime,
    RuncRuntimePaths,
    SandboxId,
    SandboxSnapshot,
    SchedulerConfig,
    StorageConfig,
    Transaction,
    TxnActiveError,
    TxnAbortError,
    TxnMismatchError,
    TxnResolvedError,
)
from crab.ids import CheckpointId
from crab.journal import ActionJournal
from crab.models import JobStatus, RestoreResult, utc_now
from crab.ids import JobId
from crab.models import FailureCode
from crab.runtime import CommandRunner
from crab.sandbox import Sandbox
from crab.scheduler import FaultToleranceCheckpointingPolicy
from crab.txn import TxnDescription


class FakeCommandRunner(CommandRunner):
    def __init__(self) -> None:
        self.commands: list[tuple[str, ...]] = []

    def run(self, command: list[str], *, cwd: Path | None = None, timeout_seconds: float | None = None):
        _ = (cwd, timeout_seconds)
        self.commands.append(tuple(command))
        return type(
            "Result",
            (),
            {"command": tuple(command), "returncode": 0, "stdout": "", "stderr": ""},
        )()


def _restore_result(sandbox_id: SandboxId, checkpoint_id: CheckpointId, *, ok: bool) -> RestoreResult:
    return RestoreResult(
        job_id=JobId.new(),
        sandbox_id=sandbox_id,
        checkpoint_id=checkpoint_id,
        status=JobStatus.SUCCEEDED if ok else JobStatus.FAILED,
        started_at=utc_now(),
        finished_at=utc_now(),
        failure_code=FailureCode.NONE if ok else FailureCode.RUNTIME_ERROR,
        message="" if ok else "restore blew up",
    )


class TxnSystemTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="crab_txn_")
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.system, self.runtime, self.executor, self.inspector, self.journal = self._build(self.root)
        self.addCleanup(self.executor.shutdown)
        self.sid = SandboxId("sbx-txn")
        bundle_dir = self.root / "bundles" / str(self.sid)
        bundle_dir.mkdir(parents=True)
        (bundle_dir / "config.json").write_text(json.dumps({"process": {}}), encoding="utf-8")
        self.runtime.launch("runc", {"sandbox_id": str(self.sid), "bundle_path": str(bundle_dir)})
        self._mark(process_changed=True, filesystem_changed=True)
        registry = self.system.response_gate_registry
        assert registry is not None
        registry.enable()
        self.registry = registry

    def _build(self, root: Path):
        runner = FakeCommandRunner()
        telemetry = InMemoryTelemetrySink()
        inspector = EBPFSandboxInspector()
        journal = ActionJournal(root / "storage" / "journal")
        runtime = RuncRuntime(
            command_runner=runner,
            paths=RuncRuntimePaths(
                state_root=root / "runtime-state",
                bundle_root=root / "bundles",
                checkpoint_root=root / "checkpoints",
                metadata_root=root / "sandbox-metadata",
                zfs_dataset_prefix="pool/crab",
            ),
            action_recorder=journal,
        )
        storage = LocalCheckpointManager(
            StorageConfig(root_dir=root / "storage"),
            destroy_filesystem_ref=runtime.destroy_filesystem_ref,
        )
        executor = CRExecutor(
            ExecutorConfig(max_workers=1),
            DefaultCWorker(
                AdapterProcessCWorker(runtime),
                AdapterFileSystemCWorker(runtime),
                storage,
                runtime,
            ),
            DefaultRWorker(
                AdapterProcessRWorker(runtime),
                AdapterFileSystemRWorker(runtime),
                storage,
            ),
            telemetry,
        )
        scheduler_cfg = SchedulerConfig(
            min_checkpoint_interval_seconds=0.0,
            force_checkpoint_after_seconds=0.0,
            require_change_signal=True,
        )
        scheduler = CRScheduler(
            scheduler_cfg,
            inspector,
            runtime,
            InMemorySchedulerStateStore(),
            telemetry,
            FaultToleranceCheckpointingPolicy(scheduler_cfg),
        )
        system = CrabSystem(
            scheduler=scheduler,
            executor=executor,
            storage=storage,
            inspector=inspector,
            runtime=runtime,
            telemetry=telemetry,
            journal=journal,
        )
        return system, runtime, executor, inspector, journal

    def _mark(self, *, process_changed: bool, filesystem_changed: bool) -> None:
        self.inspector.upsert_snapshot(
            SandboxSnapshot(
                sandbox_id=self.sid,
                runtime_name="runc",
                is_running=True,
                process_changed=process_changed,
                filesystem_changed=filesystem_changed,
                observed_at=utc_now(),
            )
        )

    # ----- begin -----------------------------------------------------

    def test_begin_takes_fresh_base_when_changed(self) -> None:
        txn = self.system.begin_txn(self.sid)
        self.assertTrue(txn.base_was_fresh)
        self.assertIn(
            CheckpointId(txn.base_checkpoint_id),
            self.system.storage.list_checkpoints(self.sid),
        )
        self.assertTrue(self.registry.staging_active(self.sid))
        self.assertEqual(self.journal.active_txn(self.sid), txn.txn_id)

    def test_begin_reuses_full_checkpoint_when_unchanged(self) -> None:
        checkpoint = self.system.checkpoint_once(self.sid, leave_running=True)
        self.assertEqual(checkpoint.status.value, "succeeded")
        # mark_checkpoint_complete cleared the change cursors above.
        txn = self.system.begin_txn(self.sid)
        self.assertFalse(txn.base_was_fresh)
        self.assertEqual(txn.base_checkpoint_id, str(checkpoint.checkpoint_id))
        # No second checkpoint was taken.
        self.assertEqual(len(self.system.storage.list_checkpoints(self.sid)), 1)

    def test_begin_takes_fresh_base_when_unchanged_but_no_checkpoint(self) -> None:
        self._mark(process_changed=False, filesystem_changed=False)
        txn = self.system.begin_txn(self.sid)
        self.assertTrue(txn.base_was_fresh)

    def test_nested_begin_raises(self) -> None:
        self.system.begin_txn(self.sid)
        with self.assertRaises(TxnActiveError):
            self.system.begin_txn(self.sid)

    # ----- commit ----------------------------------------------------

    def test_commit_releases_staged_and_drops_fresh_base(self) -> None:
        txn = self.system.begin_txn(self.sid)
        generation = self.registry.arm(self.sid, "req-1")
        assert generation is not None
        result = self.system.commit_txn(self.sid, txn.txn_id)
        self.assertEqual(result.released_observations, 1)
        self.assertTrue(result.base_dropped)
        self.assertNotIn(
            CheckpointId(txn.base_checkpoint_id),
            self.system.storage.list_checkpoints(self.sid),
        )
        self.assertFalse(self.registry.staging_active(self.sid))
        self.assertIsNone(self.system.current_txn(self.sid))
        self.assertIsNone(self.journal.active_txn(self.sid))

    def test_commit_keeps_reused_base(self) -> None:
        checkpoint = self.system.checkpoint_once(self.sid, leave_running=True)
        txn = self.system.begin_txn(self.sid)
        self.assertFalse(txn.base_was_fresh)
        result = self.system.commit_txn(self.sid, txn.txn_id)
        self.assertFalse(result.base_dropped)
        self.assertIn(
            CheckpointId(str(checkpoint.checkpoint_id)),
            self.system.storage.list_checkpoints(self.sid),
        )

    def test_commit_with_wrong_txn_id_raises(self) -> None:
        self.system.begin_txn(self.sid)
        with self.assertRaises(TxnMismatchError):
            self.system.commit_txn(self.sid, "txn-nope")
        with self.assertRaises(TxnMismatchError):
            self.system.commit_txn(SandboxId("sbx-other"), "txn-nope")

    # ----- abort -----------------------------------------------------

    def test_abort_discards_and_restores_base(self) -> None:
        txn = self.system.begin_txn(self.sid)
        generation = self.registry.arm(self.sid, "req-1")
        assert generation is not None
        with mock.patch.object(
            self.system,
            "restore_once",
            return_value=_restore_result(self.sid, CheckpointId(txn.base_checkpoint_id), ok=True),
        ) as restore:
            result = self.system.abort_txn(self.sid, txn.txn_id)
        restore.assert_called_once()
        self.assertEqual(restore.call_args.args[1], CheckpointId(txn.base_checkpoint_id))
        self.assertEqual(result.discarded_observations, 1)
        self.assertEqual(result.restored_checkpoint_id, txn.base_checkpoint_id)
        # Base checkpoint is kept on abort.
        self.assertIn(
            CheckpointId(txn.base_checkpoint_id),
            self.system.storage.list_checkpoints(self.sid),
        )
        self.assertIsNone(self.system.current_txn(self.sid))

    def test_abort_restore_failure_keeps_txn_open(self) -> None:
        txn = self.system.begin_txn(self.sid)
        with mock.patch.object(
            self.system,
            "restore_once",
            return_value=_restore_result(self.sid, CheckpointId(txn.base_checkpoint_id), ok=False),
        ):
            with self.assertRaises(TxnAbortError):
                self.system.abort_txn(self.sid, txn.txn_id)
        current = self.system.current_txn(self.sid)
        self.assertIsNotNone(current)
        self.assertEqual(current.txn_id, txn.txn_id)
        # Retry with a working restore succeeds.
        with mock.patch.object(
            self.system,
            "restore_once",
            return_value=_restore_result(self.sid, CheckpointId(txn.base_checkpoint_id), ok=True),
        ):
            self.system.abort_txn(self.sid, txn.txn_id)
        self.assertIsNone(self.system.current_txn(self.sid))

    # ----- scheduler suppression --------------------------------------

    def test_auto_checkpoints_suppressed_while_txn_active(self) -> None:
        txn = self.system.begin_txn(self.sid)
        self._mark(process_changed=True, filesystem_changed=True)
        with mock.patch.object(self.system, "_execute_checkpoint_flow") as flow:
            self.assertIsNone(self.system.checkpoint_if_due(self.sid))
        flow.assert_not_called()
        self.assertFalse(self.system._should_coordinate_live_request(self.sid, "req-x"))
        # Direct coordination path is also guarded.
        self.assertIsNone(self.system._execute_checkpoint_flow(self.sid))
        self.system.commit_txn(self.sid, txn.txn_id)
        with mock.patch.object(self.system, "_execute_checkpoint_flow", return_value=None) as flow:
            self.system.checkpoint_if_due(self.sid)
        flow.assert_called_once()

    # ----- journal integration ----------------------------------------

    def test_journal_tags_records_and_markers(self) -> None:
        txn = self.system.begin_txn(self.sid)
        # Any exec recorded while the txn is open picks up the txn_id.
        self.journal.record_exec(
            self.sid,
            argv=["true"],
            cwd=None,
            env=None,
            user=None,
            timeout_s=None,
            capture_output=True,
            returncode=0,
            duration_ms=1.0,
            stdout="",
            stderr="",
            started_at=utc_now().isoformat(),
            finished_at=utc_now().isoformat(),
        )
        self.system.commit_txn(self.sid, txn.txn_id)
        records = self.journal.entries(self.sid)
        tagged = [record for record in records if record.txn_id == txn.txn_id]
        events = {
            record.payload["event"]
            for record in tagged
            if record.kind == "lifecycle"
        }
        self.assertIn("txn_begin", events)
        self.assertIn("txn_commit", events)
        self.assertTrue(any(record.kind == "exec" for record in tagged))
        begin_marker = next(
            record for record in tagged
            if record.kind == "lifecycle" and record.payload["event"] == "txn_begin"
        )
        self.assertEqual(
            begin_marker.payload["metadata"]["base_checkpoint_id"],
            txn.base_checkpoint_id,
        )
        # Post-commit records are untagged again.
        self.journal.record_lifecycle(self.sid, "poke")
        self.assertIsNone(self.journal.entries(self.sid)[-1].txn_id)

    # ----- teardown ----------------------------------------------------

    def test_release_txn_disarms_without_restore(self) -> None:
        self.system.begin_txn(self.sid)
        generation = self.registry.arm(self.sid, "req-1")
        assert generation is not None
        with mock.patch.object(self.system, "restore_once") as restore:
            self.system.release_txn(self.sid)
        restore.assert_not_called()
        self.assertIsNone(self.system.current_txn(self.sid))
        self.assertFalse(self.registry.staging_active(self.sid))
        self.assertIsNone(self.journal.active_txn(self.sid))
        # Idempotent on sandboxes without a txn.
        self.system.release_txn(self.sid)


class TransactionHandleTests(unittest.TestCase):
    class _FakeCommands:
        def __init__(self) -> None:
            self.calls: list = []

        def run(self, cmd=None, *, argv=None, **kwargs):
            self.calls.append((cmd, argv, kwargs))
            return "exec-result"

    class _FakeSystem:
        def __init__(self) -> None:
            self.committed: list = []
            self.aborted: list = []

        def begin_txn(self, sandbox_id, *, label=None):
            return TxnDescription(
                txn_id="txn-fixed",
                sandbox_id=str(sandbox_id),
                base_checkpoint_id="ckpt-base",
                base_was_fresh=True,
                started_at=utc_now().isoformat(),
                label=label,
            )

        def current_txn(self, sandbox_id):
            return None

        def commit_txn(self, sandbox_id, txn_id):
            self.committed.append((str(sandbox_id), txn_id))
            return "commit-result"

        def abort_txn(self, sandbox_id, txn_id):
            self.aborted.append((str(sandbox_id), txn_id))
            return "abort-result"

    class _FakeEngine:
        def __init__(self) -> None:
            self.system = TransactionHandleTests._FakeSystem()

        def _register_sandbox(self, sandbox) -> None:
            pass

    def _sandbox(self):
        engine = self._FakeEngine()
        sandbox = Sandbox.connect(SandboxId("sbx-handle"), engine=engine)
        commands = self._FakeCommands()
        sandbox.commands = commands  # type: ignore[attr-defined]
        return sandbox, engine, commands

    def test_context_manager_commits_on_clean_exit(self) -> None:
        sandbox, engine, commands = self._sandbox()
        with sandbox.begin(label="demo") as txn:
            self.assertEqual(txn.label, "demo")
            txn.exec("echo hi", env={"A": "1"})
        self.assertEqual(engine.system.committed, [("sbx-handle", "txn-fixed")])
        self.assertEqual(engine.system.aborted, [])
        self.assertEqual(commands.calls[0][0], "echo hi")
        self.assertEqual(txn.resolved, "committed")

    def test_context_manager_aborts_on_exception(self) -> None:
        sandbox, engine, _ = self._sandbox()
        with self.assertRaises(ValueError):
            with sandbox.begin() as txn:
                raise ValueError("boom")
        self.assertEqual(engine.system.aborted, [("sbx-handle", "txn-fixed")])
        self.assertEqual(engine.system.committed, [])
        self.assertEqual(txn.resolved, "aborted")

    def test_double_resolve_raises(self) -> None:
        sandbox, engine, _ = self._sandbox()
        txn = sandbox.begin()
        txn.commit()
        with self.assertRaises(TxnResolvedError):
            txn.commit()
        with self.assertRaises(TxnResolvedError):
            txn.abort()
        with self.assertRaises(TxnResolvedError):
            txn.exec("true")
        # Context-manager exit after manual resolve is a no-op.
        with sandbox.begin() as txn2:
            txn2.abort()
        self.assertEqual(engine.system.aborted, [("sbx-handle", "txn-fixed")])
        self.assertEqual(engine.system.committed, [("sbx-handle", "txn-fixed")])

    def test_begin_without_txn_support_raises(self) -> None:
        class _BareEngine:
            system = object()

            def _register_sandbox(self, sandbox) -> None:
                pass

        sandbox = Sandbox.connect(SandboxId("sbx-bare"), engine=_BareEngine())
        with self.assertRaises(NotImplementedError):
            sandbox.begin()
        self.assertIsNone(sandbox.current_txn())


if __name__ == "__main__":
    unittest.main()
