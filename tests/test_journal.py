"""Unit tests for the action journal (PR-B1.1): the ActionJournal store,
the RuncRuntime exec/launch recording hooks, CrabSystem lifecycle markers,
EngineConfig parsing, and Sandbox.actions(). Host-runnable — no runc."""
from __future__ import annotations

import hashlib
import json
import subprocess
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
    SandboxExecTimeout,
    StorageConfig,
)
from crab.engine import EngineConfig
from crab.journal import ActionJournal
from crab.models import utc_now
from crab.runtime import CommandRunner
from crab.sandbox import Sandbox
from crab.scheduler import FaultToleranceCheckpointingPolicy


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


class _FakeProc:
    """Stand-in for subprocess.Popen inside RuncRuntime.exec."""

    def __init__(self, *, stdout: str = "", stderr: str = "", returncode: int = 0, timeout_first: bool = False) -> None:
        self._stdout = stdout
        self._stderr = stderr
        self.returncode = returncode
        self._timeout_first = timeout_first

    def communicate(self, timeout: float | None = None):
        if self._timeout_first:
            self._timeout_first = False
            raise subprocess.TimeoutExpired(cmd="runc exec", timeout=timeout or 0.0)
        return self._stdout, self._stderr

    def poll(self) -> int | None:
        return self.returncode

    def kill(self) -> None:
        self.returncode = -9


# ---------------------------------------------------------------------------
# ActionJournal store
# ---------------------------------------------------------------------------


class ActionJournalTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="crab_journal_")
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.journal = ActionJournal(self.root)
        self.sid = SandboxId("sbx-1")

    def _record_exec(self, **overrides):
        fields = dict(
            argv=["/bin/sh", "-c", "echo hi"],
            cwd="/work",
            env={"MARKER": 1, "PATH": "/usr/bin"},
            user=None,
            timeout_s=5.0,
            capture_output=True,
            returncode=0,
            duration_ms=12.5,
            stdout="hi\n",
            stderr="",
            started_at=utc_now().isoformat(),
            finished_at=utc_now().isoformat(),
        )
        fields.update(overrides)
        return self.journal.record_exec(self.sid, **fields)

    def test_exec_record_round_trips_verbatim(self) -> None:
        self._record_exec()
        records = self.journal.entries(self.sid)
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record.kind, "exec")
        self.assertEqual(record.seq, 0)
        self.assertIsNone(record.txn_id)
        payload = record.payload
        self.assertEqual(payload["argv"], ["/bin/sh", "-c", "echo hi"])
        self.assertEqual(payload["cwd"], "/work")
        # Env values are str-coerced exactly as the runtime passes them.
        self.assertEqual(payload["env"], {"MARKER": "1", "PATH": "/usr/bin"})
        self.assertEqual(payload["returncode"], 0)
        self.assertFalse(payload["timed_out"])
        expected = "hi\n".encode("utf-8")
        self.assertEqual(payload["stdout_len"], len(expected))
        self.assertEqual(payload["stdout_sha256"], hashlib.sha256(expected).hexdigest())
        # Bodies are never journaled.
        raw = self.journal.path_for(self.sid).read_text(encoding="utf-8")
        self.assertNotIn("hi\\n", raw)

    def test_uncaptured_output_has_no_digests(self) -> None:
        self._record_exec(capture_output=False, stdout=None, stderr=None)
        payload = self.journal.entries(self.sid)[0].payload
        self.assertIsNone(payload["stdout_len"])
        self.assertIsNone(payload["stdout_sha256"])
        self.assertIsNone(payload["stderr_sha256"])

    def test_seq_is_monotonic_and_recovers_across_instances(self) -> None:
        self._record_exec()
        self._record_exec()
        reopened = ActionJournal(self.root)
        reopened.record_lifecycle(self.sid, "checkpoint")
        seqs = [record.seq for record in reopened.entries(self.sid)]
        self.assertEqual(seqs, [0, 1, 2])

    def test_kind_and_since_seq_filters(self) -> None:
        self._record_exec()
        self.journal.record_lifecycle(self.sid, "checkpoint", metadata={"checkpoint_id": "ckpt-1"})
        self._record_exec()
        lifecycle = self.journal.entries(self.sid, kind="lifecycle")
        self.assertEqual(len(lifecycle), 1)
        self.assertEqual(lifecycle[0].payload["event"], "checkpoint")
        self.assertEqual(lifecycle[0].payload["metadata"], {"checkpoint_id": "ckpt-1"})
        tail = self.journal.entries(self.sid, since_seq=0)
        self.assertEqual([record.seq for record in tail], [1, 2])

    def test_malformed_lines_are_skipped(self) -> None:
        self._record_exec()
        with self.journal.path_for(self.sid).open("a", encoding="utf-8") as handle:
            handle.write("not-json\n")
        self._record_exec()
        self.assertEqual([record.seq for record in self.journal.entries(self.sid)], [0, 1])

    def test_unknown_sandbox_is_empty(self) -> None:
        self.assertEqual(self.journal.entries(SandboxId("nope")), [])


# ---------------------------------------------------------------------------
# RuncRuntime hooks
# ---------------------------------------------------------------------------


class RuncExecJournalTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="crab_journal_runc_")
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.journal = ActionJournal(self.root / "journal")
        self.runtime = RuncRuntime(
            command_runner=FakeCommandRunner(),
            paths=RuncRuntimePaths(
                state_root=self.root / "runtime-state",
                bundle_root=self.root / "bundles",
                checkpoint_root=self.root / "checkpoints",
                metadata_root=self.root / "sandbox-metadata",
                zfs_dataset_prefix="pool/crab",
            ),
            action_recorder=self.journal,
        )
        self.sid = SandboxId("sbx-exec")

    def _exec(self, proc: _FakeProc, **kwargs):
        # Payload-tree cleanup has dedicated contract tests.  This journal
        # unit uses no real process/PID file, so make termination a no-op and
        # exercise only recording plus the stable timeout type here.
        with mock.patch(
            "crab.runtime.runc.subprocess.Popen", return_value=proc
        ), mock.patch.object(self.runtime, "_terminate_exec_payload"):
            return self.runtime.exec(self.sid, ["echo", "hi"], **kwargs)

    def test_exec_records_success(self) -> None:
        result = self._exec(_FakeProc(stdout="hi\n"), cwd="/work", env={"A": "1"}, timeout_s=3.0)
        self.assertEqual(result.returncode, 0)
        records = self.journal.entries(self.sid, kind="exec")
        self.assertEqual(len(records), 1)
        payload = records[0].payload
        self.assertEqual(payload["argv"], ["echo", "hi"])
        self.assertEqual(payload["cwd"], "/work")
        self.assertEqual(payload["env"], {"A": "1"})
        self.assertEqual(payload["returncode"], 0)
        self.assertGreaterEqual(payload["duration_ms"], 0.0)
        self.assertIsNotNone(records[0].finished_at)

    def test_exec_records_nonzero_returncode(self) -> None:
        self._exec(_FakeProc(stderr="boom", returncode=3))
        payload = self.journal.entries(self.sid, kind="exec")[0].payload
        self.assertEqual(payload["returncode"], 3)

    def test_exec_timeout_records_attempt_and_reraises(self) -> None:
        with self.assertRaises(SandboxExecTimeout):
            self._exec(_FakeProc(timeout_first=True), timeout_s=0.01)
        records = self.journal.entries(self.sid, kind="exec")
        self.assertEqual(len(records), 1)
        payload = records[0].payload
        self.assertTrue(payload["timed_out"])
        self.assertIsNone(payload["returncode"])

    def test_recorder_failure_does_not_break_exec(self) -> None:
        class _Boom:
            def record_exec(self, *_, **__):
                raise RuntimeError("journal down")

            def record_lifecycle(self, *_, **__):
                raise RuntimeError("journal down")

        self.runtime.action_recorder = _Boom()
        result = self._exec(_FakeProc(stdout="ok"))
        self.assertEqual(result.returncode, 0)

    def test_no_recorder_means_no_journal_file(self) -> None:
        self.runtime.action_recorder = None
        self._exec(_FakeProc())
        self.assertEqual(self.journal.entries(self.sid), [])

    def test_launch_records_lifecycle_marker(self) -> None:
        bundle_dir = self.root / "bundles" / "sbx-launch"
        bundle_dir.mkdir(parents=True)
        (bundle_dir / "config.json").write_text(json.dumps({"process": {}}), encoding="utf-8")
        sid = self.runtime.launch("runc", {"sandbox_id": "sbx-launch", "bundle_path": str(bundle_dir)})
        records = self.journal.entries(sid, kind="lifecycle")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].payload["event"], "launch")
        self.assertEqual(records[0].payload["metadata"]["runtime_name"], "runc")


# ---------------------------------------------------------------------------
# CrabSystem lifecycle markers
# ---------------------------------------------------------------------------


class SystemLifecycleJournalTests(unittest.TestCase):
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

    def test_checkpoint_fork_destroy_markers(self) -> None:
        with tempfile.TemporaryDirectory(prefix="crab_journal_sys_") as tmp:
            root = Path(tmp)
            system, runtime, executor, inspector, journal = self._build(root)
            self.addCleanup(executor.shutdown)
            source = SandboxId("sbx-src")
            bundle_dir = root / "bundles" / str(source)
            bundle_dir.mkdir(parents=True)
            (bundle_dir / "config.json").write_text(json.dumps({"process": {}}), encoding="utf-8")
            runtime.launch("runc", {"sandbox_id": str(source), "bundle_path": str(bundle_dir)})
            inspector.upsert_snapshot(
                SandboxSnapshot(
                    sandbox_id=source,
                    runtime_name="runc",
                    is_running=True,
                    process_changed=True,
                    filesystem_changed=True,
                    observed_at=utc_now(),
                )
            )

            checkpoint = system.checkpoint_once(source, leave_running=True)
            self.assertEqual(checkpoint.status.value, "succeeded")

            fork_id = SandboxId("sbx-src-fork-1")
            (root / "bundles" / str(fork_id)).mkdir(parents=True)
            result = system.fork_once(
                source, fork_id, target_rootfs_path=root / "bundles" / str(fork_id) / "rootfs"
            )
            system.prepare_source_destroy(source)

            source_events = [
                (record.payload["event"], record.payload.get("metadata", {}))
                for record in journal.entries(source, kind="lifecycle")
            ]
            event_names = [name for name, _ in source_events]
            self.assertEqual(event_names[0], "launch")
            self.assertIn("checkpoint", event_names)
            self.assertIn("fork_source", event_names)
            self.assertIn("destroy", event_names)
            # fork_once takes its own fresh checkpoint, so pick the first
            # (manual) checkpoint marker for the metadata assertions.
            checkpoint_meta = next(meta for name, meta in source_events if name == "checkpoint")
            self.assertEqual(checkpoint_meta["checkpoint_id"], str(checkpoint.checkpoint_id))
            self.assertEqual(checkpoint_meta["reason"], "manual")

            fork_events = journal.entries(fork_id, kind="lifecycle")
            self.assertEqual(fork_events[0].payload["event"], "fork_created")
            self.assertEqual(
                fork_events[0].payload["metadata"]["source_sandbox_id"], str(source)
            )
            self.assertEqual(
                fork_events[0].payload["metadata"]["checkpoint_id"], str(result.checkpoint_id)
            )


# ---------------------------------------------------------------------------
# EngineConfig parsing + Sandbox.actions
# ---------------------------------------------------------------------------


class EngineConfigJournalTests(unittest.TestCase):
    def test_default_enabled(self) -> None:
        self.assertTrue(EngineConfig().enable_action_journal)

    def test_flat_key(self) -> None:
        cfg = EngineConfig.from_mapping({"enable_action_journal": False})
        self.assertFalse(cfg.enable_action_journal)

    def test_nested_block(self) -> None:
        cfg = EngineConfig.from_mapping({"journal": {"enabled": False}})
        self.assertFalse(cfg.enable_action_journal)


class SandboxActionsTests(unittest.TestCase):
    class _FakeEngine:
        def __init__(self, journal) -> None:
            self.system = type("Sys", (), {"journal": journal})()

        def _register_sandbox(self, sandbox) -> None:
            pass

    def test_actions_reads_journal(self) -> None:
        with tempfile.TemporaryDirectory(prefix="crab_journal_sdk_") as tmp:
            journal = ActionJournal(Path(tmp))
            sid = SandboxId("sbx-sdk")
            journal.record_lifecycle(sid, "launch")
            journal.record_exec(
                sid,
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
            sandbox = Sandbox.connect(sid, engine=self._FakeEngine(journal))
            rows = sandbox.actions()
            self.assertEqual([row["kind"] for row in rows], ["lifecycle", "exec"])
            execs = sandbox.actions(kind="exec")
            self.assertEqual(len(execs), 1)
            self.assertEqual(execs[0]["payload"]["argv"], ["true"])
            self.assertEqual(len(sandbox.actions(limit=1)), 1)

    def test_actions_without_journal_raises(self) -> None:
        engine = self._FakeEngine(None)
        sandbox = Sandbox.connect(SandboxId("sbx-none"), engine=engine)
        with self.assertRaises(NotImplementedError):
            sandbox.actions()


if __name__ == "__main__":
    unittest.main()
