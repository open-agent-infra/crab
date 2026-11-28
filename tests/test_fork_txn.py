"""Unit tests for fork-backed transactions (B3): begin/commit/abort
state machine against a duck-typed runtime with the heavy flows
(checkpoint/restore/replicate) patched at the system boundary, teardown
via release_txn, exec routing, daemon/shim/CLI serialization.
Host-runnable — no runc."""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
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
    SandboxId,
    SchedulerConfig,
    StorageConfig,
)
from crab.daemon.server import _serialize_txn
from crab.daemon.transport import DaemonRequestError
from crab.ids import CheckpointId
from crab.interceptor import SandboxResponseGateRegistry
from crab.journal import ActionJournal
from crab.models import ChangesetEntry, ChangesetResult
from crab.remote_engine import RemoteEngine, _deserialize_txn
from crab.sandbox import Sandbox
from crab.scheduler import FaultToleranceCheckpointingPolicy
from crab.txn import (
    Transaction,
    TxnActiveError,
    TxnCommitConflict,
    TxnDescription,
    TxnError,
    TxnMismatchError,
)


class FakeForkTxnRuntime:
    """Just enough runtime surface for the fork-txn commit swap."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.promote_error: Exception | None = None

    def describe(self, sandbox_id):
        return SimpleNamespace(status="running")

    def stop(self, sandbox_id) -> None:
        self.calls.append(("stop", str(sandbox_id)))

    def delete_runtime(self, sandbox_id, *, force=True, ignore_missing=True) -> None:
        self.calls.append(("delete_runtime", str(sandbox_id)))

    def rootfs_path_for(self, sandbox_id) -> Path:
        return Path(f"/tmp/fake-bundles/{sandbox_id}/rootfs")

    def clone_filesystem_snapshot(self, source_id, checkpoint_id, target_id, *, target_rootfs_path):
        self.calls.append(
            ("clone", str(source_id), str(checkpoint_id), str(target_id), str(target_rootfs_path))
        )
        return f"pool/{target_id}"

    def promote_filesystem_dataset(self, sandbox_id) -> None:
        self.calls.append(("promote", str(sandbox_id)))
        if self.promote_error is not None and not str(sandbox_id).endswith("fork-1"):
            # Failure injection targets the post-swap promote (source).
            raise self.promote_error

    def destroy_filesystem_dataset(self, sandbox_id) -> None:
        self.calls.append(("destroy_fs", str(sandbox_id)))


class ForkTxnSystemBase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="crab_fork_txn_")
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.fake = FakeForkTxnRuntime()
        self.telemetry = InMemoryTelemetrySink()
        self.journal = ActionJournal(self.root / "storage" / "journal")
        storage = LocalCheckpointManager(
            StorageConfig(root_dir=self.root / "storage"),
            destroy_filesystem_ref=lambda fs_ref: None,
        )
        executor = CRExecutor(
            ExecutorConfig(max_workers=1),
            DefaultCWorker(
                AdapterProcessCWorker(self.fake),
                AdapterFileSystemCWorker(self.fake),
                storage,
                self.fake,
            ),
            DefaultRWorker(
                AdapterProcessRWorker(self.fake),
                AdapterFileSystemRWorker(self.fake),
                storage,
            ),
            self.telemetry,
        )
        self.addCleanup(executor.shutdown)
        scheduler_cfg = SchedulerConfig(
            min_checkpoint_interval_seconds=0.0,
            force_checkpoint_after_seconds=0.0,
            require_change_signal=True,
        )
        scheduler = CRScheduler(
            scheduler_cfg,
            EBPFSandboxInspector(),
            self.fake,
            InMemorySchedulerStateStore(),
            self.telemetry,
            FaultToleranceCheckpointingPolicy(scheduler_cfg),
        )
        self.system = CrabSystem(
            scheduler=scheduler,
            executor=executor,
            storage=storage,
            inspector=EBPFSandboxInspector(),
            runtime=self.fake,
            telemetry=self.telemetry,
            response_gate_registry=SandboxResponseGateRegistry(),
            journal=self.journal,
        )
        self.source = SandboxId("sbx-src")
        self.fork = SandboxId("sbx-src-fork-1")
        self.hook_calls: list[tuple] = []

        def fork_hook(source_id):
            self.hook_calls.append(("fork", str(source_id)))
            # Mirror fork_once: the fork carries a fork_created marker
            # naming its source and the fork-point checkpoint.
            self.journal.record_lifecycle(
                self.fork,
                "fork_created",
                metadata={"source_sandbox_id": str(source_id), "checkpoint_id": "base-1"},
            )
            return self.fork

        def destroy_hook(fork_id):
            self.hook_calls.append(("destroy", str(fork_id)))

        def lease_hook(sandbox_id):
            self.hook_calls.append(("lease_repair", str(sandbox_id)))

        # Saved so subclasses can re-wire with an extra hook. `lease_transfer`
        # is deliberately left unwired here: the commit tests below then pin
        # the no-transfer path, which is what an unnetworked promotion does.
        self._fork_hook = fork_hook
        self._destroy_hook = destroy_hook
        self._lease_hook = lease_hook

        self.system.configure_fork_txn_hooks(
            fork=fork_hook, destroy=destroy_hook, lease_repair=lease_hook
        )

    def _begin(self) -> TxnDescription:
        return self.system.begin_txn(self.source, isolation="fork")

    def _patch_commit_flows(
        self,
        *,
        source_entries=(),
        restore_status: str = "succeeded",
        checkpoint_status: str = "succeeded",
    ):
        """Patch the heavy flows at the system boundary; the E2E covers
        the real ones. Returns the mock bundle."""
        checkpoint = mock.patch.object(
            self.system,
            "checkpoint_once",
            return_value=SimpleNamespace(
                status=SimpleNamespace(value=checkpoint_status),
                checkpoint_id=CheckpointId("ckpt-commit"),
            ),
        )
        restore = mock.patch.object(
            self.system,
            "restore_once",
            return_value=SimpleNamespace(
                status=SimpleNamespace(value=restore_status), message=""
            ),
        )
        changeset = mock.patch.object(
            self.system,
            "changeset_since",
            return_value=ChangesetResult(
                sandbox_id=self.source,
                base_checkpoint_id=CheckpointId("base-1"),
                entries=tuple(source_entries),
            ),
        )
        replicate = mock.patch.object(
            self.system, "_replicate_fork_checkpoint", return_value=CheckpointId("ckpt-commit")
        )
        mocks = SimpleNamespace(
            checkpoint=checkpoint.start(),
            restore=restore.start(),
            changeset=changeset.start(),
            replicate=replicate.start(),
        )
        for patcher in (checkpoint, restore, changeset, replicate):
            self.addCleanup(patcher.stop)
        return mocks

    def _lifecycle(self, sandbox_id, event: str) -> list[dict]:
        return [
            record.payload["metadata"]
            for record in self.journal.entries(sandbox_id, kind="lifecycle")
            if record.payload.get("event") == event
        ]


class BeginForkTxnTests(ForkTxnSystemBase):
    def test_begin_forks_and_registers_both_ids(self) -> None:
        description = self._begin()
        self.assertEqual(description.isolation, "fork")
        self.assertEqual(description.fork_sandbox_id, str(self.fork))
        self.assertEqual(description.base_checkpoint_id, "base-1")
        self.assertFalse(description.base_was_fresh)
        self.assertEqual(self.hook_calls, [("fork", str(self.source))])
        self.assertTrue(self.system._txn_active(self.source))
        self.assertTrue(self.system._txn_active(self.fork))
        self.assertIs(self.system.current_txn(self.source), self.system.current_txn(self.fork))
        marker = self._lifecycle(self.source, "txn_begin")[-1]
        self.assertEqual(marker["isolation"], "fork")
        self.assertEqual(marker["fork_sandbox_id"], str(self.fork))

    def test_begin_arms_staging_on_the_fork(self) -> None:
        self._begin()
        registry = self.system.response_gate_registry
        self.assertTrue(registry.staging_active(self.fork))
        self.assertFalse(registry.staging_active(self.source))

    def test_begin_without_hooks_raises(self) -> None:
        self.system._fork_txn_fork = None
        with self.assertRaises(TxnError):
            self._begin()
        self.assertFalse(self.system._txn_active(self.source))

    def test_begin_fork_hook_failure_cleans_reservation(self) -> None:
        self.system.configure_fork_txn_hooks(
            fork=mock.Mock(side_effect=RuntimeError("no fork for you")),
            destroy=mock.Mock(),
        )
        with self.assertRaises(RuntimeError):
            self._begin()
        self.assertFalse(self.system._txn_active(self.source))
        self.assertFalse(self.system._txn_active(self.fork))

    def test_nested_begin_refused_on_source_and_fork(self) -> None:
        self._begin()
        with self.assertRaises(TxnActiveError):
            self.system.begin_txn(self.source)
        with self.assertRaises(TxnActiveError):
            self.system.begin_txn(self.fork, isolation="fork")

    def test_unknown_isolation_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.system.begin_txn(self.source, isolation="overlay")


class CommitForkTxnTests(ForkTxnSystemBase):
    def test_clean_commit_promotes_in_order(self) -> None:
        description = self._begin()
        mocks = self._patch_commit_flows()

        result = self.system.commit_txn(self.source, description.txn_id)

        self.assertEqual(result.promoted_checkpoint_id, "ckpt-commit")
        self.assertFalse(result.base_dropped)
        mocks.changeset.assert_called_once_with(self.source, CheckpointId("base-1"))
        mocks.checkpoint.assert_called_once_with(self.fork, leave_running=True)
        mocks.replicate.assert_called_once_with(
            self.fork, self.source, CheckpointId("ckpt-commit")
        )
        mocks.restore.assert_called_once_with(
            self.source, CheckpointId("ckpt-commit"), restore_metadata=None
        )
        self.assertEqual(
            [(call[0], call[1]) for call in self.fake.calls],
            [
                ("stop", str(self.source)),
                ("delete_runtime", str(self.source)),
                ("promote", str(self.fork)),
                ("destroy_fs", str(self.source)),
                ("clone", str(self.fork)),
                ("promote", str(self.source)),
            ],
        )
        clone = next(call for call in self.fake.calls if call[0] == "clone")
        self.assertEqual(clone[1:4], (str(self.fork), "ckpt-commit", str(self.source)))
        self.assertIn(("lease_repair", str(self.source)), self.hook_calls)
        self.assertIn(("destroy", str(self.fork)), self.hook_calls)
        self.assertFalse(self.system._txn_active(self.source))
        self.assertFalse(self.system._txn_active(self.fork))
        marker = self._lifecycle(self.source, "txn_commit")[-1]
        self.assertEqual(marker["promoted_checkpoint_id"], "ckpt-commit")
        self.assertEqual(marker["isolation"], "fork")
        self.assertTrue(self._lifecycle(self.fork, "txn_fork_committed"))
        events = [attrs for name, attrs in self.telemetry.events if name == "txn.commit"]
        self.assertTrue(events and events[-1]["isolation"] == "fork")

    def test_dirty_source_refuses_commit(self) -> None:
        description = self._begin()
        mocks = self._patch_commit_flows(
            source_entries=(ChangesetEntry(path="/dirty.txt", change="added"),)
        )

        with self.assertRaises(TxnCommitConflict):
            self.system.commit_txn(self.source, description.txn_id)

        mocks.checkpoint.assert_not_called()
        self.assertTrue(self.system._txn_active(self.source))
        self.assertNotIn(("destroy", str(self.fork)), self.hook_calls)
        self.assertEqual(self.fake.calls, [])

    def test_force_commit_skips_dirty_check(self) -> None:
        description = self._begin()
        mocks = self._patch_commit_flows(
            source_entries=(ChangesetEntry(path="/dirty.txt", change="added"),)
        )

        result = self.system.commit_txn(self.source, description.txn_id, force=True)

        mocks.changeset.assert_not_called()
        self.assertEqual(result.promoted_checkpoint_id, "ckpt-commit")
        marker = self._lifecycle(self.source, "txn_commit")[-1]
        self.assertTrue(marker["forced"])

    def test_missing_fork_point_snapshot_means_retry(self) -> None:
        # A previous commit attempt already swapped the source dataset;
        # the dirty check cannot run and must not block the retry.
        description = self._begin()
        mocks = self._patch_commit_flows()
        mocks.changeset.side_effect = FileNotFoundError("snapshot missing")

        result = self.system.commit_txn(self.source, description.txn_id)

        self.assertEqual(result.promoted_checkpoint_id, "ckpt-commit")

    def test_restore_failure_keeps_txn_open_and_fork_alive(self) -> None:
        description = self._begin()
        self._patch_commit_flows(restore_status="failed")

        with self.assertRaises(TxnError):
            self.system.commit_txn(self.source, description.txn_id)

        self.assertTrue(self.system._txn_active(self.source))
        self.assertNotIn(("destroy", str(self.fork)), self.hook_calls)
        # Retry succeeds against the still-live fork.
        self._patch_commit_flows()
        result = self.system.commit_txn(self.source, description.txn_id)
        self.assertEqual(result.promoted_checkpoint_id, "ckpt-commit")

    def test_promote_failure_retains_fork(self) -> None:
        description = self._begin()
        self._patch_commit_flows()
        self.fake.promote_error = RuntimeError("promote boom")

        result = self.system.commit_txn(self.source, description.txn_id)

        self.assertEqual(result.promoted_checkpoint_id, "ckpt-commit")
        self.assertNotIn(("destroy", str(self.fork)), self.hook_calls)
        self.assertFalse(self.system._txn_active(self.source))

    def test_commit_via_fork_id_is_a_mismatch(self) -> None:
        description = self._begin()
        with self.assertRaises(TxnMismatchError):
            self.system.commit_txn(self.fork, description.txn_id)


class PromotionNetnsTransferTests(ForkTxnSystemBase):
    """PR-N1: the fork is dumped inside its own netns, so its sockets are
    bound to the fork's guest IP. Promotion therefore moves that network
    identity onto the source instead of restoring the image against the
    source's netns, where CRIU's soccr cannot bind the addresses back
    (`EADDRNOTAVAIL`)."""

    def _wire_transfer(self, *, result: bool = True, boom: bool = False) -> list:
        """Wire the hook and return the runtime call names observed at
        transfer time, so ordering can be asserted rather than assumed."""
        seen_calls: list[list[str]] = []

        def transfer(fork_id, source_id, *, probe: bool = False):
            if probe:
                # The pre-flight asks whether a transfer would happen; it
                # must not record a call or mutate anything.
                return result
            self.hook_calls.append(("lease_transfer", str(fork_id), str(source_id)))
            seen_calls.append([call[0] for call in self.fake.calls])
            if boom:
                raise RuntimeError("netns busy")
            return result

        self.system.configure_fork_txn_hooks(
            fork=self._fork_hook,
            destroy=self._destroy_hook,
            lease_repair=self._lease_hook,
            lease_transfer=transfer,
        )
        return seen_calls

    def test_transfer_replaces_repair_and_precedes_the_restore(self) -> None:
        description = self._begin()
        mocks = self._patch_commit_flows()
        seen = self._wire_transfer()

        result = self.system.commit_txn(self.source, description.txn_id)

        self.assertEqual(result.promoted_checkpoint_id, "ckpt-commit")
        self.assertIn(
            ("lease_transfer", str(self.fork), str(self.source)), self.hook_calls
        )
        # The transferred netns is already plumbed and was never torn down;
        # repairing the source's released lease would be wrong.
        self.assertNotIn(("lease_repair", str(self.source)), self.hook_calls)
        # The dataset swap is complete and the restore has not started.
        self.assertEqual(
            seen, [["stop", "delete_runtime", "promote", "destroy_fs", "clone"]]
        )
        mocks.restore.assert_called_once()

    def test_a_fork_without_its_own_lease_keeps_the_repair_path(self) -> None:
        # Engine.fork_sandbox only logs lease-allocation failures, so a
        # networked engine can hold a lease-less fork; it shares the source's
        # netns, which is exactly what lease_repair already handles. The
        # probe says "no transfer", so the fork is also dumped
        # leave_running=True and stays retryable.
        description = self._begin()
        mocks = self._patch_commit_flows()
        self._wire_transfer(result=False)

        result = self.system.commit_txn(self.source, description.txn_id)

        self.assertEqual(result.promoted_checkpoint_id, "ckpt-commit")
        self.assertNotIn(
            ("lease_transfer", str(self.fork), str(self.source)), self.hook_calls
        )
        self.assertIn(("lease_repair", str(self.source)), self.hook_calls)
        self.assertTrue(mocks.checkpoint.call_args.kwargs["leave_running"])

    def test_the_transfer_path_dumps_the_fork_stopped(self) -> None:
        """Two live copies of a listener cannot share one netns (CRIU:
        "Can't bind inet socket: Address already in use"), so the fork's
        processes leave the namespace at dump time — the standard CRIU
        migration shape."""
        description = self._begin()
        mocks = self._patch_commit_flows()
        self._wire_transfer()

        self.system.commit_txn(self.source, description.txn_id)

        self.assertFalse(mocks.checkpoint.call_args.kwargs["leave_running"])

    def test_transfer_failure_keeps_the_fork_and_the_txn(self) -> None:
        description = self._begin()
        mocks = self._patch_commit_flows()
        self._wire_transfer(boom=True)

        with self.assertRaises(TxnError):
            self.system.commit_txn(self.source, description.txn_id)

        mocks.restore.assert_not_called()
        self.assertNotIn(("destroy", str(self.fork)), self.hook_calls)
        self.assertTrue(self.system._txn_active(self.source))


class AbortForkTxnTests(ForkTxnSystemBase):
    def test_abort_destroys_fork_without_touching_source(self) -> None:
        description = self._begin()
        with mock.patch.object(self.system, "restore_once") as restore:
            result = self.system.abort_txn(self.source, description.txn_id)

        restore.assert_not_called()
        self.assertIsNone(result.restored_checkpoint_id)
        self.assertIn(("destroy", str(self.fork)), self.hook_calls)
        self.assertFalse(self.system._txn_active(self.source))
        self.assertFalse(self.system._txn_active(self.fork))
        marker = self._lifecycle(self.source, "txn_abort")[-1]
        self.assertEqual(marker["isolation"], "fork")
        self.assertIsNone(marker["restored_checkpoint_id"])
        self.assertTrue(self._lifecycle(self.fork, "txn_fork_discarded"))
        self.assertEqual(self.fake.calls, [])

    def test_abort_via_fork_id_is_a_mismatch(self) -> None:
        description = self._begin()
        with self.assertRaises(TxnMismatchError):
            self.system.abort_txn(self.fork, description.txn_id)


class ReleaseForkTxnTests(ForkTxnSystemBase):
    def test_source_kill_destroys_orphaned_fork(self) -> None:
        self._begin()
        self.system.release_txn(self.source)
        self.assertIn(("destroy", str(self.fork)), self.hook_calls)
        self.assertFalse(self.system._txn_active(self.source))
        self.assertFalse(self.system._txn_active(self.fork))

    def test_fork_kill_clears_txn_without_recursive_destroy(self) -> None:
        self._begin()
        self.system.release_txn(self.fork)
        self.assertNotIn(("destroy", str(self.fork)), self.hook_calls)
        self.assertFalse(self.system._txn_active(self.source))
        self.assertFalse(self.system._txn_active(self.fork))


class ExecRoutingTests(unittest.TestCase):
    class _FakeEngine:
        system = object()

        def _register_sandbox(self, sandbox) -> None:
            pass

    def _description(self, fork_sandbox_id: str | None) -> TxnDescription:
        return TxnDescription(
            txn_id="txn-1",
            sandbox_id="sbx-src",
            base_checkpoint_id="base-1",
            base_was_fresh=False,
            started_at="2026-01-01T00:00:00+00:00",
            isolation="snapshot" if fork_sandbox_id is None else "fork",
            fork_sandbox_id=fork_sandbox_id,
        )

    def test_exec_targets_fork_when_fork_backed(self) -> None:
        engine = self._FakeEngine()
        source = Sandbox.connect("sbx-src", engine=engine)
        txn = Transaction(source, self._description("sbx-src-fork-1"))
        target = txn._exec_target()
        self.assertEqual(str(target.sandbox_id), "sbx-src-fork-1")
        self.assertIs(txn._exec_target(), target)  # cached

    def test_exec_targets_source_for_snapshot_txns(self) -> None:
        engine = self._FakeEngine()
        source = Sandbox.connect("sbx-src", engine=engine)
        txn = Transaction(source, self._description(None))
        self.assertIs(txn._exec_target(), source)


class SerializationTests(unittest.TestCase):
    def test_serialize_deserialize_round_trip(self) -> None:
        description = TxnDescription(
            txn_id="txn-9",
            sandbox_id="src",
            base_checkpoint_id="base-1",
            base_was_fresh=False,
            started_at="2026-01-01T00:00:00+00:00",
            label="demo",
            isolation="fork",
            fork_sandbox_id="src-fork-1",
        )
        payload = _serialize_txn(description)
        self.assertEqual(payload["isolation"], "fork")
        self.assertEqual(payload["fork_sandbox_id"], "src-fork-1")
        self.assertEqual(_deserialize_txn(payload), description)

    def test_deserialize_defaults_for_old_payloads(self) -> None:
        description = _deserialize_txn(
            {
                "txn_id": "txn-old",
                "sandbox_id": "src",
                "base_checkpoint_id": "b",
                "base_was_fresh": True,
                "started_at": "t",
                "label": None,
            }
        )
        self.assertEqual(description.isolation, "snapshot")
        self.assertIsNone(description.fork_sandbox_id)


class _FakeDaemonClient:
    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.responses: dict[str, object] = {}

    def post_json(self, path, payload=None, *, timeout_seconds=None):
        self.requests.append({"path": path, "payload": payload, "timeout": timeout_seconds})
        response = self.responses.get(path)
        if isinstance(response, Exception):
            raise response
        return response or {"ok": True}

    def get_json(self, path, *, timeout_seconds=None):
        self.requests.append({"path": path, "timeout": timeout_seconds})
        response = self.responses.get(path)
        if isinstance(response, Exception):
            raise response
        return response or {"ok": True}


class ShimForkTxnTests(unittest.TestCase):
    _INFO = {"runtime": "runc", "default_image": "ubuntu:22.04"}

    def _engine(self):
        client = _FakeDaemonClient()
        return RemoteEngine(client, info=self._INFO), client

    def test_begin_fork_isolation_payload_and_rehydration(self) -> None:
        engine, client = self._engine()
        client.responses["/sandboxes/src/txn"] = {
            "ok": True,
            "txn": {
                "txn_id": "txn-9",
                "sandbox_id": "src",
                "base_checkpoint_id": "base-1",
                "base_was_fresh": False,
                "started_at": "t",
                "label": None,
                "isolation": "fork",
                "fork_sandbox_id": "src-fork-1",
            },
        }
        description = engine.system.begin_txn(SandboxId("src"), isolation="fork")
        self.assertEqual(description.fork_sandbox_id, "src-fork-1")
        request = client.requests[0]
        self.assertEqual(request["payload"], {"isolation": "fork"})
        self.assertEqual(request["timeout"], 600.0)

    def test_commit_force_payload_and_promoted_checkpoint(self) -> None:
        engine, client = self._engine()
        client.responses["/sandboxes/src/txn/txn-9/commit"] = {
            "ok": True,
            "result": {
                "txn_id": "txn-9",
                "released_observations": 0,
                "base_dropped": False,
                "promoted_checkpoint_id": "ckpt-commit",
            },
        }
        result = engine.system.commit_txn(SandboxId("src"), "txn-9", force=True)
        self.assertEqual(result.promoted_checkpoint_id, "ckpt-commit")
        self.assertEqual(client.requests[0]["payload"], {"force": True})

    def test_commit_conflict_rehydrates_typed_error(self) -> None:
        engine, client = self._engine()
        body = json.dumps(
            {"ok": False, "error": "source dirty", "error_type": "txn_commit_conflict"}
        ).encode("utf-8")
        client.responses["/sandboxes/src/txn/txn-9/commit"] = DaemonRequestError(409, "/x", body)
        with self.assertRaises(TxnCommitConflict):
            engine.system.commit_txn(SandboxId("src"), "txn-9")


class CliForkTxnTests(unittest.TestCase):
    def _run_cli(self, argv: list[str], responses: dict) -> tuple[int, str, list]:
        requests: list[dict] = []

        class _CliClient:
            def __init__(self, socket_path, *, timeout_seconds):
                requests.append({"socket": str(socket_path), "timeout": timeout_seconds})

            def post_json(self, path, payload=None, *, timeout_seconds=None):
                requests.append({"path": path, "payload": payload})
                return responses[path]

            def get_json(self, path, *, timeout_seconds=None):
                requests.append({"path": path})
                return responses[path]

        stdout = io.StringIO()
        from crab.cli import commands

        with mock.patch.object(commands, "DaemonClient", _CliClient):
            with contextlib.redirect_stdout(stdout):
                rc = commands.main(argv)
        return rc, stdout.getvalue(), requests

    def test_begin_isolation_fork_payload(self) -> None:
        rc, out, requests = self._run_cli(
            ["txn", "begin", "sbx-1", "--isolation", "fork"],
            {"/sandboxes/sbx-1/txn": {"ok": True, "txn": {"txn_id": "txn-f"}}},
        )
        self.assertEqual(rc, 0)
        self.assertEqual(out.strip(), "txn-f")
        self.assertEqual(requests[-1]["payload"], {"isolation": "fork"})
        self.assertEqual(requests[0]["timeout"], 600.0)

    def test_commit_force_and_promoted_output(self) -> None:
        rc, out, requests = self._run_cli(
            ["txn", "commit", "sbx-1", "txn-f", "--force"],
            {
                "/sandboxes/sbx-1/txn/txn-f/commit": {
                    "ok": True,
                    "result": {
                        "txn_id": "txn-f",
                        "released_observations": 1,
                        "base_dropped": False,
                        "promoted_checkpoint_id": "ckpt-9",
                    },
                }
            },
        )
        self.assertEqual(rc, 0)
        self.assertIn("promoted=ckpt-9", out)
        self.assertEqual(requests[-1]["payload"], {"force": True})


if __name__ == "__main__":
    unittest.main()
