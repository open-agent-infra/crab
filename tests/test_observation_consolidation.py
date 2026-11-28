"""Unit tests for C3 observation consolidation: the system primitive
(append/dedupe/none policies, summarizer hook, provenance, idempotence),
its wiring into C2 merges and B3 fork-txn commits, the SDK plumbing, and
the daemon routes/shim/CLI incl. the journal read RPC. Host-runnable —
no runc."""
from __future__ import annotations

import contextlib
import io
import json
import shutil
import tempfile
import threading
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
from crab.daemon.server import _build_handler, _Routes
from crab.daemon.transport import DaemonClient, serve_unix_socket
from crab.ids import CheckpointId
from crab.interceptor import SandboxResponseGateRegistry
from crab.journal import ActionJournal
from crab.models import ChangesetEntry, MergeReport, ObservationReport
from crab.remote_engine import RemoteEngine
from crab.sandbox import Sandbox
from crab.scheduler import FaultToleranceCheckpointingPolicy
from crab.txn import TxnDescription


def _record_exec(journal: ActionJournal, sandbox_id, argv, *, returncode=0, stdout="out"):
    return journal.record_exec(
        SandboxId(str(sandbox_id)),
        argv=list(argv),
        cwd="/w",
        env={"K": "V"},
        user=None,
        timeout_s=None,
        capture_output=True,
        returncode=returncode,
        duration_ms=1.0,
        stdout=stdout,
        stderr="",
        started_at="2026-01-01T00:00:00+00:00",
        finished_at="2026-01-01T00:00:01+00:00",
    )


class _SystemHarness(unittest.TestCase):
    """Real CrabSystem + journal on tmpdir; runtime is a bare fake."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="crab_obs_")
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.fake_runtime = SimpleNamespace(name="runc")
        self.telemetry = InMemoryTelemetrySink()
        self.journal = ActionJournal(self.root / "storage" / "journal")
        storage = LocalCheckpointManager(
            StorageConfig(root_dir=self.root / "storage"),
            destroy_filesystem_ref=lambda fs_ref: None,
        )
        executor = CRExecutor(
            ExecutorConfig(max_workers=1),
            DefaultCWorker(
                AdapterProcessCWorker(self.fake_runtime),
                AdapterFileSystemCWorker(self.fake_runtime),
                storage,
                self.fake_runtime,
            ),
            DefaultRWorker(
                AdapterProcessRWorker(self.fake_runtime),
                AdapterFileSystemRWorker(self.fake_runtime),
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
            self.fake_runtime,
            InMemorySchedulerStateStore(),
            self.telemetry,
            FaultToleranceCheckpointingPolicy(scheduler_cfg),
        )
        self.system = CrabSystem(
            scheduler=scheduler,
            executor=executor,
            storage=storage,
            inspector=EBPFSandboxInspector(),
            runtime=self.fake_runtime,
            telemetry=self.telemetry,
            response_gate_registry=SandboxResponseGateRegistry(),
            journal=self.journal,
        )
        self.source = SandboxId("sbx-src")
        self.fork = SandboxId("sbx-src-fork-1")

    def _seed_fork_journal(self) -> None:
        self.journal.record_lifecycle(
            self.fork,
            "fork_created",
            metadata={"source_sandbox_id": str(self.source), "checkpoint_id": "base-1"},
        )
        _record_exec(self.journal, self.fork, ["echo", "one"])
        self.journal.record_lifecycle(self.fork, "staging_begin")  # mechanics: excluded
        _record_exec(self.journal, self.fork, ["echo", "two"], returncode=1)
        self.journal.record_lifecycle(self.fork, "checkpoint", metadata={"id": "c9"})

    def _observations(self):
        return self.journal.entries(self.source, kind="observation")


class ConsolidateObservationsTests(_SystemHarness):
    def test_append_copies_qualifying_records_in_order_with_provenance(self) -> None:
        self._seed_fork_journal()

        report = self.system.consolidate_observations(self.source, self.fork)

        self.assertEqual(report.policy, "append")
        self.assertEqual(report.consolidated, 4)  # fork_created + 2 exec + checkpoint
        self.assertEqual(report.skipped_duplicates, 0)
        self.assertFalse(report.already_consolidated)
        rows = self._observations()
        self.assertEqual(len(rows), 4)
        origin_seqs = [row.payload["origin_seq"] for row in rows]
        self.assertEqual(origin_seqs, sorted(origin_seqs))
        first = rows[0].payload
        self.assertEqual(first["fork_sandbox_id"], str(self.fork))
        self.assertEqual(first["origin_kind"], "lifecycle")
        self.assertEqual(first["origin_payload"]["event"], "fork_created")
        self.assertEqual(first["reason"], "manual")
        exec_row = rows[1].payload
        self.assertEqual(exec_row["origin_kind"], "exec")
        self.assertEqual(exec_row["origin_payload"]["argv"], ["echo", "one"])
        # Mechanics markers never qualify.
        self.assertFalse(
            any(
                row.payload.get("origin_payload", {}).get("event") == "staging_begin"
                for row in rows
            )
        )

    def test_adopted_rows_do_not_requalify(self) -> None:
        self._seed_fork_journal()
        # The fork itself adopted history from a grandchild.
        self.journal.record_observation(
            self.fork, payload={"fork_sandbox_id": "sbx-grand", "origin_kind": "exec"}
        )
        report = self.system.consolidate_observations(self.source, self.fork)
        self.assertEqual(report.consolidated, 4)

    def test_dedupe_skips_identical_source_execs_since_fork_point(self) -> None:
        # Source history before the fork point must not dedupe.
        _record_exec(self.journal, self.source, ["echo", "one"])
        self.journal.record_lifecycle(
            self.source,
            "fork_source",
            metadata={"target_sandbox_id": str(self.fork), "checkpoint_id": "base-1"},
        )
        # After the fork point the source reproduced "echo one" itself
        # and ran a variant of "echo two" with a different outcome.
        _record_exec(self.journal, self.source, ["echo", "one"])
        _record_exec(self.journal, self.source, ["echo", "two"], returncode=0)
        self._seed_fork_journal()  # fork ran "echo one" (rc 0), "echo two" (rc 1)

        report = self.system.consolidate_observations(self.source, self.fork, policy="dedupe")

        self.assertEqual(report.skipped_duplicates, 1)  # echo one
        self.assertEqual(report.consolidated, 3)  # fork_created + echo two(rc1) + checkpoint
        argvs = [
            row.payload["origin_payload"].get("argv")
            for row in self._observations()
            if row.payload["origin_kind"] == "exec"
        ]
        self.assertEqual(argvs, [["echo", "two"]])

    def test_dedupe_without_fork_source_marker_uses_whole_journal(self) -> None:
        _record_exec(self.journal, self.source, ["echo", "one"])  # pre-fork, no marker
        self._seed_fork_journal()
        report = self.system.consolidate_observations(self.source, self.fork, policy="dedupe")
        self.assertEqual(report.skipped_duplicates, 1)

    def test_summarizer_receives_rows_and_writes_last(self) -> None:
        self._seed_fork_journal()
        seen: list[list] = []

        def summarizer(rows):
            seen.append(rows)
            return {"digest": f"{len(rows)} records"}

        report = self.system.consolidate_observations(
            self.source, self.fork, summarizer=summarizer
        )

        self.assertTrue(report.summary_written)
        self.assertEqual(len(seen), 1)
        self.assertEqual(len(seen[0]), 4)
        self.assertEqual(seen[0][0]["kind"], "lifecycle")
        rows = self._observations()
        self.assertEqual(rows[-1].payload["origin_kind"], "summary")
        self.assertEqual(rows[-1].payload["summary"], {"digest": "4 records"})

    def test_summarizer_none_result_writes_nothing(self) -> None:
        self._seed_fork_journal()
        report = self.system.consolidate_observations(
            self.source, self.fork, summarizer=lambda rows: None
        )
        self.assertFalse(report.summary_written)

    def test_policy_none_with_summarizer_is_digest_only(self) -> None:
        self._seed_fork_journal()
        report = self.system.consolidate_observations(
            self.source, self.fork, policy="none", summarizer=lambda rows: "digest"
        )
        self.assertEqual(report.consolidated, 0)
        self.assertTrue(report.summary_written)
        rows = self._observations()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].payload["origin_kind"], "summary")

    def test_summarizer_error_propagates_without_summary(self) -> None:
        self._seed_fork_journal()
        with self.assertRaises(RuntimeError):
            self.system.consolidate_observations(
                self.source,
                self.fork,
                summarizer=mock.Mock(side_effect=RuntimeError("digest boom")),
            )
        self.assertFalse(
            any(row.payload["origin_kind"] == "summary" for row in self._observations())
        )

    def test_marker_telemetry_and_report_serialization(self) -> None:
        self._seed_fork_journal()
        report = self.system.consolidate_observations(self.source, self.fork, reason="merge")
        markers = [
            record.payload["metadata"]
            for record in self.journal.entries(self.source, kind="lifecycle")
            if record.payload.get("event") == "observations_consolidated"
        ]
        self.assertEqual(len(markers), 1)
        self.assertEqual(markers[0]["fork_sandbox_id"], str(self.fork))
        self.assertEqual(markers[0]["consolidated"], 4)
        self.assertEqual(markers[0]["reason"], "merge")
        events = [
            attrs for name, attrs in self.telemetry.events if name == "observations.consolidated"
        ]
        self.assertTrue(events and events[-1]["consolidated"] == 4)
        round_trip = ObservationReport.from_json(report.to_json())
        self.assertEqual(round_trip, report)

    def test_idempotence_for_automatic_reasons_manual_reruns(self) -> None:
        self._seed_fork_journal()
        first = self.system.consolidate_observations(self.source, self.fork, reason="merge")
        self.assertEqual(first.consolidated, 4)

        second = self.system.consolidate_observations(self.source, self.fork, reason="merge")
        self.assertTrue(second.already_consolidated)
        self.assertEqual(second.consolidated, 0)
        self.assertEqual(len(self._observations()), 4)

        third = self.system.consolidate_observations(self.source, self.fork, reason="txn_commit")
        self.assertTrue(third.already_consolidated)

        manual = self.system.consolidate_observations(self.source, self.fork)
        self.assertFalse(manual.already_consolidated)
        self.assertEqual(manual.consolidated, 4)
        self.assertEqual(len(self._observations()), 8)

    def test_guards(self) -> None:
        with self.assertRaises(ValueError):
            self.system.consolidate_observations(self.source, self.fork, policy="merge3")
        self.system.journal = None
        with self.assertRaises(RuntimeError):
            self.system.consolidate_observations(self.source, self.fork)


class CommitConsolidationTests(_SystemHarness):
    """Fork-txn commit adopts the fork's history by default."""

    def setUp(self) -> None:
        super().setUp()
        self.destroy_calls: list[str] = []
        self.system.configure_fork_txn_hooks(
            fork=lambda source_id: self.fork,
            destroy=lambda fork_id: self.destroy_calls.append(str(fork_id)),
        )
        # Runtime surface the commit swap touches.
        self.fake_runtime.describe = lambda sid: SimpleNamespace(status="stopped")
        self.fake_runtime.stop = lambda sid: None
        self.fake_runtime.delete_runtime = lambda sid, **kw: None
        self.fake_runtime.rootfs_path_for = lambda sid: self.root / str(sid) / "rootfs"
        self.fake_runtime.clone_filesystem_snapshot = (
            lambda src, ckpt, dst, *, target_rootfs_path: f"pool/{dst}"
        )
        self.fake_runtime.promote_filesystem_dataset = lambda sid: None
        self.fake_runtime.destroy_filesystem_dataset = lambda sid: None
        self._seed_fork_journal()
        self.description = TxnDescription(
            txn_id="txn-1",
            sandbox_id=str(self.source),
            base_checkpoint_id="base-1",
            base_was_fresh=False,
            started_at="t",
            isolation="fork",
            fork_sandbox_id=str(self.fork),
        )
        with self.system._txn_lock:
            self.system._active_txns[self.source] = self.description
            self.system._active_txns[self.fork] = self.description
        for patcher in (
            mock.patch.object(
                self.system,
                "checkpoint_once",
                return_value=SimpleNamespace(
                    status=SimpleNamespace(value="succeeded"),
                    checkpoint_id=CheckpointId("ckpt-commit"),
                ),
            ),
            mock.patch.object(
                self.system,
                "restore_once",
                return_value=SimpleNamespace(status=SimpleNamespace(value="succeeded"), message=""),
            ),
            mock.patch.object(
                self.system,
                "changeset_since",
                return_value=SimpleNamespace(entries=()),
            ),
            mock.patch.object(
                self.system,
                "_replicate_fork_checkpoint",
                return_value=CheckpointId("ckpt-commit"),
            ),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_commit_appends_fork_history_by_default(self) -> None:
        result = self.system.commit_txn(self.source, "txn-1")
        self.assertEqual(result.observations_consolidated, 4)
        rows = self.journal.entries(self.source, kind="observation")
        self.assertEqual(len(rows), 4)
        self.assertEqual(self.destroy_calls, [str(self.fork)])
        marker = [
            record.payload["metadata"]
            for record in self.journal.entries(self.source, kind="lifecycle")
            if record.payload.get("event") == "txn_commit"
        ][-1]
        self.assertEqual(marker["observations_consolidated"], 4)

    def test_commit_observations_none_skips(self) -> None:
        result = self.system.commit_txn(self.source, "txn-1", observations="none")
        self.assertIsNone(result.observations_consolidated)
        self.assertEqual(self.journal.entries(self.source, kind="observation"), [])

    def test_commit_survives_consolidation_failure(self) -> None:
        with mock.patch.object(
            self.system, "consolidate_observations", side_effect=RuntimeError("obs boom")
        ):
            result = self.system.commit_txn(self.source, "txn-1")
        self.assertIsNone(result.observations_consolidated)
        self.assertEqual(result.promoted_checkpoint_id, "ckpt-commit")


class MergeConsolidationTests(_SystemHarness):
    """merge_from_fork(observations=...) adopts history only on success."""

    def setUp(self) -> None:
        super().setUp()
        self.fake_runtime.describe = lambda sid: SimpleNamespace(status="stopped")
        self.fake_runtime.rootfs_path_for = lambda sid: self._rootfs(sid)
        self.fake_runtime.changeset_since = lambda sid, ckpt: list(
            self.changesets.get(str(sid), [])
        )
        self.fake_runtime.checkpoint_filesystem = self._snapshot
        self.fake_runtime.snapshot_content_root = (
            lambda sid, ckpt: self.root / str(sid) / "snapshots" / str(ckpt)
        )
        self.fake_runtime.discard_partial_checkpoint = lambda sid, ckpt: None
        self.changesets: dict[str, list[ChangesetEntry]] = {}
        for sid in ("sbx-src", "sbx-src-fork-1"):
            (self.root / sid / "rootfs").mkdir(parents=True)
        self.journal.record_lifecycle(
            self.fork,
            "fork_created",
            metadata={"source_sandbox_id": str(self.source), "checkpoint_id": "base-1"},
        )
        _record_exec(self.journal, self.fork, ["make", "thing"])

    def _rootfs(self, sid) -> Path:
        return self.root / str(sid) / "rootfs"

    def _snapshot(self, sid, ckpt):
        snap = self.root / str(sid) / "snapshots" / str(ckpt)
        shutil.copytree(self._rootfs(sid), snap)
        return SimpleNamespace(executed=True, reason=None)

    def test_merge_append_attaches_observation_report(self) -> None:
        self.changesets[str(self.fork)] = [ChangesetEntry(path="/new.txt", change="added")]
        (self._rootfs(self.fork) / "new.txt").write_text("fork\n")

        report = self.system.merge_from_fork(self.source, self.fork, observations="append")

        self.assertIsNotNone(report.observations)
        self.assertEqual(report.observations.consolidated, 2)  # fork_created + exec
        self.assertEqual(report.observations.reason, "merge")
        self.assertTrue(self.journal.entries(self.source, kind="observation"))
        payload = report.to_json()
        self.assertEqual(MergeReport.from_json(payload), report)

    def test_merge_default_keeps_observations_off(self) -> None:
        self.changesets[str(self.fork)] = [ChangesetEntry(path="/new.txt", change="added")]
        (self._rootfs(self.fork) / "new.txt").write_text("fork\n")
        report = self.system.merge_from_fork(self.source, self.fork)
        self.assertIsNone(report.observations)
        self.assertEqual(self.journal.entries(self.source, kind="observation"), [])

    def test_aborted_merge_never_consolidates(self) -> None:
        self.changesets[str(self.fork)] = [ChangesetEntry(path="/shared.txt", change="modified")]
        self.changesets[str(self.source)] = [ChangesetEntry(path="/shared.txt", change="modified")]
        (self._rootfs(self.fork) / "shared.txt").write_text("fork\n")
        (self._rootfs(self.source) / "shared.txt").write_text("src\n")

        report = self.system.merge_from_fork(self.source, self.fork, observations="append")

        self.assertTrue(report.conflicted)
        self.assertIsNone(report.observations)
        self.assertEqual(self.journal.entries(self.source, kind="observation"), [])

    def test_merge_rejects_unknown_observation_policy(self) -> None:
        with self.assertRaises(ValueError):
            self.system.merge_from_fork(self.source, self.fork, observations="summarize")


class SandboxPlumbingTests(unittest.TestCase):
    class _FakeSystem:
        def __init__(self) -> None:
            self.calls: list = []
            self.journal = None

        def consolidate_observations(self, source_id, fork_id, *, policy, summarizer):
            self.calls.append((str(source_id), str(fork_id), policy, summarizer))
            return ObservationReport(
                source_sandbox_id=source_id,
                fork_sandbox_id=fork_id,
                policy=policy,
                consolidated=3,
                skipped_duplicates=0,
            )

    class _FakeEngine:
        def __init__(self, system) -> None:
            self.system = system

        def _register_sandbox(self, sandbox) -> None:
            pass

    def test_consolidate_plumbs_and_accepts_sandbox_instance(self) -> None:
        system = self._FakeSystem()
        engine = self._FakeEngine(system)
        source = Sandbox.connect("sbx-src", engine=engine)
        fork = Sandbox.connect("sbx-fork", engine=engine)
        report = source.consolidate_observations(fork, policy="dedupe")
        self.assertEqual(report.consolidated, 3)
        self.assertEqual(system.calls, [("sbx-src", "sbx-fork", "dedupe", None)])

    def test_bare_engine_raises(self) -> None:
        class _Bare:
            system = object()

            def _register_sandbox(self, sandbox) -> None:
                pass

        sandbox = Sandbox.connect("sbx-src", engine=_Bare())
        with self.assertRaises(NotImplementedError):
            sandbox.consolidate_observations("sbx-fork")


class _FakeDaemonSystem:
    def __init__(self, journal) -> None:
        self.journal = journal
        self.calls: list = []

    def consolidate_observations(self, source_id, fork_id, *, policy, reason):
        self.calls.append((str(source_id), str(fork_id), policy, reason))
        return ObservationReport(
            source_sandbox_id=source_id,
            fork_sandbox_id=fork_id,
            policy=policy,
            consolidated=2,
            skipped_duplicates=1,
            reason=reason,
        )


class _FakeDaemon:
    def __init__(self, engine) -> None:
        self.engine = engine

    def require_engine(self):
        return self.engine

    def register_sandbox(self, sandbox_id) -> None:
        pass

    def unregister_sandbox(self, sandbox_id) -> None:
        pass


class DaemonRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="crab_obsd_")
        self.addCleanup(self._tmp.cleanup)
        self.journal = ActionJournal(Path(self._tmp.name) / "journal")
        self.engine = SimpleNamespace(system=_FakeDaemonSystem(self.journal))
        self.routes = _Routes(_FakeDaemon(self.engine))

    def test_consolidate_route(self) -> None:
        response = self.routes.consolidate_observations_sandbox(
            {"fork_sandbox_id": "sbx-fork", "policy": "dedupe"}, sandbox_id="sbx-src"
        )
        self.assertTrue(response["ok"])
        self.assertEqual(response["report"]["consolidated"], 2)
        self.assertEqual(
            self.engine.system.calls, [("sbx-src", "sbx-fork", "dedupe", "manual")]
        )

    def test_consolidate_requires_fork_id_and_maps_value_error(self) -> None:
        from crab.daemon.server import _BadRequest

        with self.assertRaises(_BadRequest):
            self.routes.consolidate_observations_sandbox({}, sandbox_id="sbx-src")
        self.engine.system.consolidate_observations = mock.Mock(
            side_effect=ValueError("bad policy")
        )
        with self.assertRaises(_BadRequest):
            self.routes.consolidate_observations_sandbox(
                {"fork_sandbox_id": "sbx-fork"}, sandbox_id="sbx-src"
            )

    def test_actions_route_filters_and_limits(self) -> None:
        _record_exec(self.journal, "sbx-src", ["echo", "a"])
        self.journal.record_lifecycle(SandboxId("sbx-src"), "launch")
        _record_exec(self.journal, "sbx-src", ["echo", "b"])

        response = self.routes.sandbox_actions({"kind": "exec"}, sandbox_id="sbx-src")
        argvs = [row["payload"]["argv"] for row in response["records"]]
        self.assertEqual(argvs, [["echo", "a"], ["echo", "b"]])

        limited = self.routes.sandbox_actions({"kind": "exec", "limit": 1}, sandbox_id="sbx-src")
        self.assertEqual(limited["records"][0]["payload"]["argv"], ["echo", "b"])

    def test_actions_route_without_journal(self) -> None:
        from crab.daemon.server import _BadRequest

        self.engine.system.journal = None
        with self.assertRaises(_BadRequest):
            self.routes.sandbox_actions({}, sandbox_id="sbx-src")

    def test_dispatch_over_socket(self) -> None:
        socket_path = Path(self._tmp.name) / "crab.sock"
        server = serve_unix_socket(socket_path, _build_handler(_FakeDaemon(self.engine)))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        client = DaemonClient(socket_path, timeout_seconds=10.0)
        _record_exec(self.journal, "sbx-src", ["echo", "wire"])

        actions = client.post_json("/sandboxes/sbx-src/actions", {})
        self.assertEqual(actions["records"][0]["payload"]["argv"], ["echo", "wire"])
        consolidate = client.post_json(
            "/sandboxes/sbx-src/observations/consolidate", {"fork_sandbox_id": "sbx-fork"}
        )
        self.assertEqual(consolidate["report"]["skipped_duplicates"], 1)


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
        return self.responses.get(path) or {"ok": True}


class ShimTests(unittest.TestCase):
    _INFO = {"runtime": "runc", "default_image": "ubuntu:22.04"}

    def _engine(self):
        client = _FakeDaemonClient()
        return RemoteEngine(client, info=self._INFO), client

    def test_consolidate_proxy_and_summarizer_rejection(self) -> None:
        engine, client = self._engine()
        client.responses["/sandboxes/src/observations/consolidate"] = {
            "ok": True,
            "report": ObservationReport(
                source_sandbox_id=SandboxId("src"),
                fork_sandbox_id=SandboxId("fork"),
                policy="append",
                consolidated=5,
                skipped_duplicates=0,
            ).to_json(),
        }
        report = engine.system.consolidate_observations(SandboxId("src"), SandboxId("fork"))
        self.assertIsInstance(report, ObservationReport)
        self.assertEqual(report.consolidated, 5)
        self.assertEqual(
            client.requests[0]["payload"],
            {"fork_sandbox_id": "fork", "policy": "append", "reason": "manual"},
        )
        with self.assertRaises(NotImplementedError):
            engine.system.consolidate_observations(
                SandboxId("src"), SandboxId("fork"), summarizer=lambda rows: None
            )

    def test_journal_shim_powers_sandbox_actions(self) -> None:
        engine, client = self._engine()
        client.responses["/sandboxes/src/actions"] = {
            "ok": True,
            "records": [
                {
                    "seq": 0,
                    "kind": "observation",
                    "sandbox_id": "src",
                    "txn_id": None,
                    "started_at": "t",
                    "finished_at": "t",
                    "payload": {"fork_sandbox_id": "fork", "origin_kind": "exec"},
                }
            ],
        }
        sandbox = Sandbox.connect("src", engine=engine)
        rows = sandbox.actions(kind="observation")
        self.assertEqual(rows[0]["payload"]["fork_sandbox_id"], "fork")
        self.assertEqual(client.requests[0]["payload"], {"kind": "observation"})

    def test_merge_payload_carries_observations_and_report_nests(self) -> None:
        engine, client = self._engine()
        nested = ObservationReport(
            source_sandbox_id=SandboxId("src"),
            fork_sandbox_id=SandboxId("fork"),
            policy="append",
            consolidated=1,
            skipped_duplicates=0,
            reason="merge",
        )
        merge_report = MergeReport(
            source_sandbox_id=SandboxId("src"),
            fork_sandbox_id=SandboxId("fork"),
            base_checkpoint_id=CheckpointId("base-1"),
            policy="fail_fast",
            applied=(),
            conflicted=(),
            skipped=(),
            observations=nested,
        )
        client.responses["/sandboxes/src/merge"] = {"ok": True, "report": merge_report.to_json()}
        report = engine.system.merge_from_fork(
            SandboxId("src"), SandboxId("fork"), observations="append"
        )
        self.assertEqual(report.observations, nested)
        self.assertEqual(client.requests[0]["payload"]["observations"], "append")
        with self.assertRaises(NotImplementedError):
            engine.system.merge_from_fork(
                SandboxId("src"),
                SandboxId("fork"),
                observation_summarizer=lambda rows: None,
            )

    def test_commit_result_carries_consolidated_count(self) -> None:
        engine, client = self._engine()
        client.responses["/sandboxes/src/txn/txn-9/commit"] = {
            "ok": True,
            "result": {
                "txn_id": "txn-9",
                "released_observations": 0,
                "base_dropped": False,
                "promoted_checkpoint_id": "ckpt-1",
                "observations_consolidated": 4,
            },
        }
        result = engine.system.commit_txn(SandboxId("src"), "txn-9")
        self.assertEqual(result.observations_consolidated, 4)


class CliTests(unittest.TestCase):
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

    def test_actions_rows_and_payload(self) -> None:
        rc, out, requests = self._run_cli(
            ["sandbox", "actions", "sbx-1", "--kind", "observation", "--limit", "5"],
            {
                "/sandboxes/sbx-1/actions": {
                    "ok": True,
                    "records": [
                        {
                            "seq": 3,
                            "kind": "observation",
                            "payload": {
                                "fork_sandbox_id": "sbx-f",
                                "origin_kind": "exec",
                                "origin_payload": {"argv": ["make", "it"]},
                            },
                        }
                    ],
                }
            },
        )
        self.assertEqual(rc, 0)
        self.assertIn("3\tobservation\texec\tfrom=sbx-f\tmake it", out)
        self.assertEqual(
            requests[-1]["payload"], {"kind": "observation", "limit": 5}
        )

    def test_consolidate_summary_line(self) -> None:
        rc, out, requests = self._run_cli(
            ["sandbox", "consolidate", "sbx-src", "sbx-fork", "--policy", "dedupe"],
            {
                "/sandboxes/sbx-src/observations/consolidate": {
                    "ok": True,
                    "report": {
                        "consolidated": 2,
                        "skipped_duplicates": 1,
                        "policy": "dedupe",
                        "already_consolidated": False,
                    },
                }
            },
        )
        self.assertEqual(rc, 0)
        self.assertIn("consolidated=2 skipped_duplicates=1 policy=dedupe", out)
        self.assertEqual(
            requests[-1]["payload"], {"fork_sandbox_id": "sbx-fork", "policy": "dedupe"}
        )


if __name__ == "__main__":
    unittest.main()
