from __future__ import annotations

from datetime import timedelta
import json
import threading
import tempfile
import unittest
from pathlib import Path

from crab import (
    CrabSystem,
    AdapterFileSystemCWorker,
    AdapterFileSystemRWorker,
    AdapterProcessCWorker,
    AdapterProcessRWorker,
    CRExecutor,
    CRScheduler,
    DeleteAfterRestoreCheckpointManager,
    DefaultCWorker,
    DefaultRWorker,
    EBPFSandboxInspector,
    EBPFEvent,
    EBPFEventKind,
    ExecutorConfig,
    FailureCode,
    FaultToleranceCheckpointingPolicy,
    CheckpointId,
    CheckpointJob,
    JobId,
    InMemoryEBPFEventCollector,
    InMemoryRequestStateStore,
    InMemorySchedulerStateStore,
    InMemoryTelemetrySink,
    LocalCheckpointManager,
    RequestContext,
    SandboxResponseGateRegistry,
    RuncRuntime,
    RuncRuntimePaths,
    SandboxId,
    SandboxSnapshot,
    SchedulerConfig,
    SpotPreemptionCheckpointingPolicy,
    StorageConfig,
)
from crab.models import (
    CheckpointManifest,
    JobStatus,
    RecoveryEvent,
    RecoveryRecord,
    SchedulerCheckpointDecision,
    utc_now,
)
from integrations.llm_services.simulated.service import SimulatedLLMState, handle_request
from crab.runtime import CommandRunner

RuncRuntimeAdapter = RuncRuntime
RuncSandboxManager = RuncRuntime
RuncSandboxManagerPaths = RuncRuntimePaths


class FakeCommandRunner(CommandRunner):
    def __init__(self) -> None:
        self.commands: list[tuple[str, ...]] = []
        self._running_sandboxes: set[str] = set()

    def run(self, command: list[str], *, cwd: Path | None = None, timeout_seconds: float | None = None):
        _ = cwd
        _ = timeout_seconds
        self.commands.append(tuple(command))
        if "start" in command:
            self._running_sandboxes.add(command[-1])
        elif "delete" in command or "kill" in command:
            self._running_sandboxes.discard(command[-2] if "kill" in command else command[-1])
        elif "restore" in command:
            self._running_sandboxes.add(command[-1])
        elif "state" in command:
            sandbox_id = command[-1]
            if sandbox_id not in self._running_sandboxes:
                return type(
                    "Result",
                    (),
                    {"command": tuple(command), "returncode": 1, "stdout": "", "stderr": "container does not exist"},
                )()
            payload = {"status": "running", "pid": 123}
            return type(
                "Result",
                (),
                {"command": tuple(command), "returncode": 0, "stdout": json.dumps(payload), "stderr": ""},
            )()
        return type(
            "Result",
            (),
            {"command": tuple(command), "returncode": 0, "stdout": "", "stderr": ""},
        )()


class FailingRestoreRunner(FakeCommandRunner):
    def run(self, command: list[str], *, cwd: Path | None = None, timeout_seconds: float | None = None):
        result = super().run(command, cwd=cwd, timeout_seconds=timeout_seconds)
        if "restore" in command:
            return type(
                "Result",
                (),
                {"command": tuple(command), "returncode": 1, "stdout": "", "stderr": "restore failed"},
            )()
        return result


class RestoreMissingRuntimeRunner(FakeCommandRunner):
    def __init__(self) -> None:
        super().__init__()
        self._restore_attempted = False

    def run(self, command: list[str], *, cwd: Path | None = None, timeout_seconds: float | None = None):
        result = super().run(command, cwd=cwd, timeout_seconds=timeout_seconds)
        if "restore" in command:
            self._restore_attempted = True
            return result
        if self._restore_attempted and len(command) >= 2 and command[-2] == "state":
            return type(
                "Result",
                (),
                {"command": tuple(command), "returncode": 1, "stdout": "", "stderr": "container does not exist"},
            )()
        return result


class EmptyCheckpointManager:
    def list_checkpoints(self, sandbox_id: SandboxId) -> list[object]:
        _ = sandbox_id
        return []


class MinimalExecutor:
    def __init__(self, *, restore_workers: int, coordination_workers: int | None = None) -> None:
        self.config = ExecutorConfig(
            max_workers=restore_workers,
            restore_workers=restore_workers,
            coordination_workers=coordination_workers,
        )


class SystemIntegrationTests(unittest.TestCase):
    def test_incremental_decision_with_missing_parent_manifest_becomes_anchor(self) -> None:
        class MissingStorage:
            def get_manifest(self, sandbox_id, checkpoint_id):
                _ = (sandbox_id, checkpoint_id)
                raise FileNotFoundError(checkpoint_id)

        system = CrabSystem.__new__(CrabSystem)
        system.storage = MissingStorage()
        system.runtime = object()
        parent_id = CheckpointId("ckpt-missing-parent")
        decision = SchedulerCheckpointDecision(
            should_checkpoint=True,
            checkpoint_process=True,
            checkpoint_filesystem=True,
            leave_running=True,
            reason="process_changed",
            policy_name="test",
            is_incremental_process=True,
            parent_process_checkpoint_id=parent_id,
            produce_pre_dump=True,
            metadata={
                "incremental_process": True,
                "incremental_chain_role": "node",
                "process_chain_length_after": 3,
            },
        )

        resolved = CrabSystem._ensure_incremental_parent_available(
            system,
            SandboxId("sbx-missing-parent"),
            decision,
        )

        self.assertFalse(resolved.is_incremental_process)
        self.assertIsNone(resolved.parent_process_checkpoint_id)
        self.assertTrue(resolved.produce_pre_dump)
        self.assertEqual(resolved.metadata["incremental_chain_role"], "anchor")
        self.assertEqual(resolved.metadata["process_chain_length_after"], 0)
        self.assertEqual(
            resolved.metadata["incremental_parent_reset_reason"],
            "missing_parent_manifest",
        )
        self.assertEqual(
            resolved.metadata["incremental_parent_candidate"],
            str(parent_id),
        )
        self.assertNotIn("incremental_process", resolved.metadata)

    def test_build_checkpoint_metadata_captures_spec_pair_group_once(self) -> None:
        request_store = InMemoryRequestStateStore()
        response_gate_registry = SandboxResponseGateRegistry()
        response_gate_registry.enable()
        sandbox_id = SandboxId("sbx-spec-group")
        telemetry = InMemoryTelemetrySink()
        system = CrabSystem.__new__(CrabSystem)
        system.request_state_store = request_store
        system.response_gate_registry = response_gate_registry
        system.extra_checkpoint_metadata_provider = None
        system.telemetry = telemetry

        draft_context = RequestContext(
            request_id="req-draft",
            sandbox_id=sandbox_id,
            started_at=utc_now(),
            metadata={
                "provider": "openai",
                "path": "/v1/chat/completions",
                "response_gate_enabled": True,
                "request_group_kind": "spec_pair",
                "request_group_id": "pair-1",
            },
        )
        oracle_context = RequestContext(
            request_id="req-oracle",
            sandbox_id=sandbox_id,
            started_at=utc_now(),
            metadata={
                "provider": "openai",
                "path": "/v1/chat/completions",
                "response_gate_enabled": True,
                "request_group_kind": "spec_pair",
                "request_group_id": "pair-1",
            },
        )
        request_store.mark_request_start(draft_context)
        request_store.mark_request_start(oracle_context)
        response_gate_registry.arm(
            sandbox_id,
            "req-draft",
            request_group_kind="spec_pair",
            request_group_id="pair-1",
        )
        response_gate_registry.arm(
            sandbox_id,
            "req-oracle",
            request_group_kind="spec_pair",
            request_group_id="pair-1",
        )

        metadata = CrabSystem._build_checkpoint_metadata(system, sandbox_id)

        self.assertTrue(metadata["captures_inflight_llm"])
        self.assertEqual(metadata["captured_request_group_kind"], "spec_pair")
        self.assertEqual(metadata["captured_request_group_id"], "pair-1")
        self.assertEqual(set(metadata["captured_request_ids"]), {"req-draft", "req-oracle"})

    def test_validate_restore_checkpoint_accepts_matching_spec_pair_group(self) -> None:
        response_gate_registry = SandboxResponseGateRegistry()
        response_gate_registry.enable()
        sandbox_id = SandboxId("sbx-spec-group")
        system = CrabSystem.__new__(CrabSystem)
        system.response_gate_registry = response_gate_registry
        response_gate_registry.arm(
            sandbox_id,
            "req-draft",
            request_group_kind="spec_pair",
            request_group_id="pair-1",
        )
        response_gate_registry.arm(
            sandbox_id,
            "req-oracle",
            request_group_kind="spec_pair",
            request_group_id="pair-1",
        )
        manifest = CheckpointManifest(
            schema_version="1",
            checkpoint_id=CheckpointId("ckpt-1"),
            sandbox_id=sandbox_id,
            created_at=utc_now(),
            runtime_name="runc",
            runtime_version="test",
            process_artifacts=[],
            filesystem_artifacts=[],
            metadata={
                "captures_inflight_llm": True,
                "captured_request_id": "req-draft",
                "captured_request_group_kind": "spec_pair",
                "captured_request_group_id": "pair-1",
            },
        ).with_integrity()
        system._resolve_restore_manifest = lambda _sandbox_id, _checkpoint_id: manifest

        message = CrabSystem._validate_restore_checkpoint(system, sandbox_id, CheckpointId("ckpt-1"))

        self.assertIsNone(message)

    def _wait_for_record(self, system: CrabSystem, sandbox_id: SandboxId, expected_event_type: str):
        import time

        deadline = time.time() + 5.0
        while time.time() < deadline:
            record = system.get_last_recovery_record(sandbox_id)
            if record is not None and record.event_type == expected_event_type:
                return record
            time.sleep(0.05)
        self.fail(f"timed out waiting for recovery record {expected_event_type} for {sandbox_id}")

    def _build_runc_system(
        self,
        *,
        root: Path,
        runner: CommandRunner,
        telemetry: InMemoryTelemetrySink,
        inspector: EBPFSandboxInspector,
        request_store: InMemoryRequestStateStore | None = None,
        relaunch_handler=None,
        enforce_restore_checkpoint_validation: bool = False,
        relaunch_on_restore_failure: bool = False,
        scheduler_config: SchedulerConfig | None = None,
    ) -> tuple[CrabSystem, CRExecutor]:
        runtime = RuncRuntimeAdapter(
            command_runner=runner,
            paths=RuncRuntimePaths(
                state_root=root / "runtime-state",
                bundle_root=root / "bundles",
                checkpoint_root=root / "checkpoints",
                zfs_dataset_prefix="pool/crab",
            ),
        )
        storage = LocalCheckpointManager(StorageConfig(root_dir=root / "storage"))
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
        sandbox_manager = RuncSandboxManager(
            command_runner=runner,
            paths=RuncSandboxManagerPaths(
                state_root=root / "runtime-state",
                bundle_root=root / "bundles",
                checkpoint_root=root / "checkpoints",
                metadata_root=root / "sandbox-metadata",
                zfs_dataset_prefix="pool/crab",
            ),
        )
        scheduler_cfg = scheduler_config or SchedulerConfig(
            min_checkpoint_interval_seconds=0.0,
            force_checkpoint_after_seconds=0.0,
            require_change_signal=True,
        )
        scheduler = CRScheduler(
            scheduler_cfg,
            inspector,
            sandbox_manager,
            InMemorySchedulerStateStore(),
            telemetry,
            FaultToleranceCheckpointingPolicy(scheduler_cfg),
        )
        system = CrabSystem(
            scheduler=scheduler,
            executor=executor,
            storage=storage,
            inspector=inspector,
            runtime=sandbox_manager,
            telemetry=telemetry,
            request_state_store=request_store,
            relaunch_handler=relaunch_handler,
            enforce_restore_checkpoint_validation=enforce_restore_checkpoint_validation,
            relaunch_on_restore_failure=relaunch_on_restore_failure,
        )
        return system, executor

    def test_requested_logical_checkpoints_reuse_or_partially_materialize(self) -> None:
        with tempfile.TemporaryDirectory(prefix="crab_logical_ckpt_") as tmp:
            root = Path(tmp)
            runner = FakeCommandRunner()
            telemetry = InMemoryTelemetrySink()
            inspector = EBPFSandboxInspector(InMemoryEBPFEventCollector())
            system, executor = self._build_runc_system(
                root=root,
                runner=runner,
                telemetry=telemetry,
                inspector=inspector,
            )
            sandbox_id = system.sandbox_manager.launch(
                "runc",
                {
                    "sandbox_id": "sbx-logical",
                    "bundle_path": str(root / "bundles" / "sbx-logical"),
                },
            )

            try:
                inspector.upsert_snapshot(
                    SandboxSnapshot(
                        sandbox_id=sandbox_id,
                        runtime_name="runc",
                        is_running=True,
                        process_changed=False,
                        filesystem_changed=False,
                        observed_at=utc_now(),
                    )
                )
                baseline = system.checkpoint_requested(
                    sandbox_id,
                    checkpoint_id="ckpt-logical-1",
                )
                self.assertEqual(baseline.status, JobStatus.SUCCEEDED)
                assert baseline.manifest is not None
                self.assertTrue(baseline.manifest.process_artifacts)
                self.assertTrue(baseline.manifest.filesystem_artifacts)
                self.assertEqual(
                    baseline.manifest.metadata["checkpoint_materialization"],
                    "full",
                )

                physical_commands_before_reuse = [
                    command
                    for command in runner.commands
                    if (command and command[0] == "runc" and "checkpoint" in command)
                    or command[:2] == ("zfs", "snapshot")
                ]
                inspector.upsert_snapshot(
                    SandboxSnapshot(
                        sandbox_id=sandbox_id,
                        runtime_name="runc",
                        is_running=True,
                        process_changed=False,
                        filesystem_changed=False,
                        observed_at=utc_now(),
                        last_checkpoint_at=baseline.finished_at,
                    )
                )
                reused = system.checkpoint_requested(
                    sandbox_id,
                    checkpoint_id="ckpt-logical-2",
                )
                self.assertEqual(reused.status, JobStatus.SUCCEEDED)
                assert reused.manifest is not None
                self.assertEqual(reused.checkpoint_id, CheckpointId("ckpt-logical-2"))
                self.assertEqual(reused.manifest.process_artifacts, [])
                self.assertEqual(reused.manifest.filesystem_artifacts, [])
                self.assertEqual(
                    reused.manifest.metadata["checkpoint_materialization"],
                    "reused",
                )
                self.assertEqual(
                    reused.manifest.metadata["process_restore_checkpoint_id"],
                    "ckpt-logical-1",
                )
                self.assertEqual(
                    reused.manifest.metadata["filesystem_restore_checkpoint_id"],
                    "ckpt-logical-1",
                )
                physical_commands_after_reuse = [
                    command
                    for command in runner.commands
                    if (command and command[0] == "runc" and "checkpoint" in command)
                    or command[:2] == ("zfs", "snapshot")
                ]
                self.assertEqual(
                    physical_commands_after_reuse,
                    physical_commands_before_reuse,
                )
                resolved_reuse = system._resolve_restore_manifest(
                    sandbox_id, reused.checkpoint_id
                )
                self.assertTrue(resolved_reuse.process_artifacts)
                self.assertTrue(resolved_reuse.filesystem_artifacts)

                inspector.upsert_snapshot(
                    SandboxSnapshot(
                        sandbox_id=sandbox_id,
                        runtime_name="runc",
                        is_running=True,
                        process_changed=False,
                        filesystem_changed=True,
                        observed_at=utc_now(),
                        last_checkpoint_at=baseline.finished_at,
                    )
                )
                partial = system.checkpoint_requested(
                    sandbox_id,
                    checkpoint_id="ckpt-logical-3",
                )
                self.assertEqual(partial.status, JobStatus.SUCCEEDED)
                assert partial.manifest is not None
                self.assertEqual(partial.manifest.process_artifacts, [])
                self.assertTrue(partial.manifest.filesystem_artifacts)
                self.assertEqual(
                    partial.manifest.metadata["checkpoint_materialization"],
                    "filesystem_only",
                )
                self.assertEqual(
                    partial.manifest.metadata["process_restore_checkpoint_id"],
                    "ckpt-logical-1",
                )
                self.assertEqual(
                    partial.manifest.metadata["filesystem_restore_checkpoint_id"],
                    "ckpt-logical-3",
                )

                system.changeset_since(
                    sandbox_id,
                    reused.checkpoint_id,
                    use_inspector_gate=False,
                )
                self.assertIn(
                    (
                        "zfs",
                        "diff",
                        "-FH",
                        f"pool/crab/{sandbox_id}@ckpt-logical-1",
                        f"pool/crab/{sandbox_id}",
                    ),
                    runner.commands,
                )

                # Restoring an older logical point must also move Crab's
                # adaptive-reuse cursor backwards. The newer filesystem-only
                # point remains stored, but a clean post-restore logical id
                # must map to the state that is actually running (baseline),
                # not to the newest manifest on disk (partial).
                restored = system.restore_once(
                    sandbox_id, baseline.checkpoint_id
                )
                self.assertEqual(restored.status, JobStatus.SUCCEEDED)
                after_restore = system.checkpoint_requested(
                    sandbox_id,
                    checkpoint_id="ckpt-logical-4",
                )
                self.assertEqual(after_restore.status, JobStatus.SUCCEEDED)
                assert after_restore.manifest is not None
                self.assertEqual(
                    after_restore.manifest.metadata[
                        "checkpoint_materialization"
                    ],
                    "reused",
                )
                self.assertEqual(
                    after_restore.manifest.metadata[
                        "process_restore_checkpoint_id"
                    ],
                    "ckpt-logical-1",
                )
                self.assertEqual(
                    after_restore.manifest.metadata[
                        "filesystem_restore_checkpoint_id"
                    ],
                    "ckpt-logical-1",
                )
            finally:
                executor.shutdown()

    def test_requested_incremental_chain_anchors_restores_and_recovers_from_standalone_full(self) -> None:
        with tempfile.TemporaryDirectory(prefix="crab_logical_incremental_") as tmp:
            root = Path(tmp)
            runner = FakeCommandRunner()
            telemetry = InMemoryTelemetrySink()
            inspector = EBPFSandboxInspector(InMemoryEBPFEventCollector())
            scheduler_config = SchedulerConfig(
                min_checkpoint_interval_seconds=0.0,
                force_checkpoint_after_seconds=0.0,
                require_change_signal=True,
                incremental_process_enabled=True,
                full_process_checkpoint_interval=8,
                max_process_chain_length=16,
            )
            system, executor = self._build_runc_system(
                root=root,
                runner=runner,
                telemetry=telemetry,
                inspector=inspector,
                scheduler_config=scheduler_config,
            )
            sandbox_id = system.sandbox_manager.launch(
                "runc",
                {
                    "sandbox_id": "sbx-logical-incremental",
                    "bundle_path": str(
                        root / "bundles" / "sbx-logical-incremental"
                    ),
                },
            )

            try:
                inspector.upsert_snapshot(
                    SandboxSnapshot(
                        sandbox_id=sandbox_id,
                        runtime_name="runc",
                        is_running=True,
                        process_changed=False,
                        filesystem_changed=False,
                        observed_at=utc_now(),
                    )
                )
                baseline = system.checkpoint_requested(
                    sandbox_id,
                    checkpoint_id="ckpt-incremental-anchor",
                )
                self.assertEqual(baseline.status, JobStatus.SUCCEEDED)
                assert baseline.manifest is not None
                self.assertEqual(baseline.manifest.process_kind, "full")
                self.assertEqual(
                    baseline.manifest.metadata["incremental_chain_role"],
                    "anchor",
                )
                self.assertEqual(
                    baseline.manifest.metadata["checkpoint_materialization"],
                    "full",
                )
                anchor_pre_dump = system.runtime.pre_dump_location(
                    sandbox_id, baseline.checkpoint_id
                )
                self.assertIsNotNone(anchor_pre_dump)
                self.assertTrue(Path(str(anchor_pre_dump)).is_dir())

                inspector.upsert_snapshot(
                    SandboxSnapshot(
                        sandbox_id=sandbox_id,
                        runtime_name="runc",
                        is_running=True,
                        process_changed=True,
                        filesystem_changed=True,
                        observed_at=utc_now(),
                        last_checkpoint_at=baseline.finished_at,
                    )
                )
                incremental = system.checkpoint_requested(
                    sandbox_id,
                    checkpoint_id="ckpt-incremental-node",
                )
                self.assertEqual(incremental.status, JobStatus.SUCCEEDED)
                assert incremental.manifest is not None
                self.assertEqual(incremental.manifest.process_kind, "incremental")
                self.assertEqual(
                    incremental.manifest.parent_checkpoint_id,
                    baseline.checkpoint_id,
                )
                self.assertEqual(
                    incremental.manifest.metadata["checkpoint_materialization"],
                    "incremental",
                )
                self.assertEqual(
                    system._stored_process_chain_length(
                        sandbox_id, incremental.checkpoint_id
                    ),
                    1,
                )

                restored = system.restore_once(
                    sandbox_id,
                    incremental.checkpoint_id,
                )
                self.assertEqual(restored.status, JobStatus.SUCCEEDED)

                standalone = system.checkpoint_once(
                    sandbox_id,
                    leave_running=True,
                    checkpoint_id="ckpt-standalone-full",
                )
                self.assertEqual(standalone.status, JobStatus.SUCCEEDED)
                standalone_pre_dump = system.runtime.pre_dump_location(
                    sandbox_id, standalone.checkpoint_id
                )
                self.assertIsNotNone(standalone_pre_dump)
                self.assertFalse(Path(str(standalone_pre_dump)).exists())

                inspector.upsert_snapshot(
                    SandboxSnapshot(
                        sandbox_id=sandbox_id,
                        runtime_name="runc",
                        is_running=True,
                        process_changed=True,
                        filesystem_changed=True,
                        observed_at=utc_now(),
                        last_checkpoint_at=standalone.finished_at,
                    )
                )
                reset_anchor = system.checkpoint_requested(
                    sandbox_id,
                    checkpoint_id="ckpt-reset-anchor",
                )
                self.assertEqual(reset_anchor.status, JobStatus.SUCCEEDED)
                assert reset_anchor.manifest is not None
                self.assertEqual(reset_anchor.manifest.process_kind, "full")
                self.assertEqual(
                    reset_anchor.manifest.metadata[
                        "incremental_parent_reset_reason"
                    ],
                    "missing_pre_dump",
                )
                self.assertEqual(
                    reset_anchor.manifest.metadata[
                        "incremental_parent_candidate"
                    ],
                    str(standalone.checkpoint_id),
                )
                reset_pre_dump = system.runtime.pre_dump_location(
                    sandbox_id, reset_anchor.checkpoint_id
                )
                self.assertIsNotNone(reset_pre_dump)
                self.assertTrue(Path(str(reset_pre_dump)).is_dir())
            finally:
                executor.shutdown()

    def test_recovery_workers_match_restore_workers_and_run_different_sandboxes_in_parallel(self) -> None:
        class BlockingRecoverySystem(CrabSystem):
            def __post_init__(self) -> None:
                super().__post_init__()
                self.release_event = threading.Event()
                self.started_pair = threading.Event()
                self._active_workers_lock = threading.Lock()
                self.active_workers = 0
                self.max_active_workers = 0
                self.started_sandboxes: list[str] = []

            def _handle_recovery_event(self, event: RecoveryEvent) -> None:
                if not self._acquire_coordination(event.sandbox_id):
                    return
                started = utc_now()
                try:
                    with self._active_workers_lock:
                        self.active_workers += 1
                        self.max_active_workers = max(self.max_active_workers, self.active_workers)
                        self.started_sandboxes.append(str(event.sandbox_id))
                        if len(self.started_sandboxes) >= 2:
                            self.started_pair.set()
                    self.release_event.wait(timeout=5.0)
                finally:
                    finished = utc_now()
                    record = RecoveryRecord(
                        sandbox_id=event.sandbox_id,
                        event_type=event.event_type,
                        started_at=started,
                        finished_at=finished,
                        status="restored",
                    )
                    with self._recovery_lock:
                        self._recovery_records[event.sandbox_id] = record
                    with self._active_workers_lock:
                        self.active_workers -= 1
                    self._release_coordination(event.sandbox_id)

        telemetry = InMemoryTelemetrySink()
        restore_workers = 3
        system = BlockingRecoverySystem(
            scheduler=object(),  # type: ignore[arg-type]
            executor=MinimalExecutor(restore_workers=restore_workers),  # type: ignore[arg-type]
            storage=EmptyCheckpointManager(),  # type: ignore[arg-type]
            inspector=object(),  # type: ignore[arg-type]
            runtime=object(),  # type: ignore[arg-type]
            telemetry=telemetry,
        )

        system.start()
        try:
            self.assertEqual(system._recovery_worker_count, restore_workers)
            self.assertEqual(len(system._recovery_futures), restore_workers)
            for sandbox_name in ("sbx-a", "sbx-b"):
                system._recovery_queue.put(
                    RecoveryEvent(
                        sandbox_id=SandboxId(sandbox_name),
                        event_type="fault",
                        observed_at=utc_now(),
                        reason="fault",
                    )
                )
            self.assertTrue(system.started_pair.wait(timeout=2.0))
            self.assertGreaterEqual(system.max_active_workers, 2)

            system.release_event.set()
            record_a = self._wait_for_record(system, SandboxId("sbx-a"), "fault")
            record_b = self._wait_for_record(system, SandboxId("sbx-b"), "fault")
        finally:
            system.release_event.set()
            system.stop()

        self.assertEqual(record_a.status, "restored")
        self.assertEqual(record_b.status, "restored")

    def test_recovery_queue_wait_metric_is_emitted(self) -> None:
        telemetry = InMemoryTelemetrySink()
        system = CrabSystem(
            scheduler=object(),  # type: ignore[arg-type]
            executor=MinimalExecutor(restore_workers=1),  # type: ignore[arg-type]
            storage=EmptyCheckpointManager(),  # type: ignore[arg-type]
            inspector=object(),  # type: ignore[arg-type]
            runtime=object(),  # type: ignore[arg-type]
            telemetry=telemetry,
        )

        event = RecoveryEvent(
            sandbox_id=SandboxId("sbx-queue-wait"),
            event_type="fault",
            observed_at=utc_now(),
            received_at=utc_now() - timedelta(milliseconds=50),
            reason="fault",
        )

        system._handle_recovery_event(event)

        queue_wait_metrics = [
            (value, attributes)
            for name, value, attributes in telemetry.metrics
            if name == "recovery.queue_wait_ms"
        ]
        self.assertEqual(len(queue_wait_metrics), 1)
        metric_value, metric_attributes = queue_wait_metrics[0]
        self.assertGreaterEqual(metric_value, 40.0)
        self.assertEqual(metric_attributes["sandbox_id"], "sbx-queue-wait")
        self.assertEqual(metric_attributes["event_type"], "fault")
        record = system.get_last_recovery_record(SandboxId("sbx-queue-wait"))
        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record.status, "no_checkpoint")

    def test_runc_system_checkpoint_restore_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory(prefix="crab_system_it_") as tmp:
            root = Path(tmp)
            runner = FakeCommandRunner()
            telemetry = InMemoryTelemetrySink()
            collector = InMemoryEBPFEventCollector()
            inspector = EBPFSandboxInspector(collector)

            runtime_paths = RuncRuntimePaths(
                state_root=root / "runtime-state",
                bundle_root=root / "bundles",
                checkpoint_root=root / "checkpoints",
                zfs_dataset_prefix="pool/crab",
            )
            sandbox_paths = RuncSandboxManagerPaths(
                state_root=root / "runtime-state",
                bundle_root=root / "bundles",
                metadata_root=root / "sandbox-metadata",
                zfs_dataset_prefix="pool/crab",
            )

            runtime = RuncRuntimeAdapter(command_runner=runner, paths=runtime_paths)
            storage_root = root / "storage"
            storage = LocalCheckpointManager(StorageConfig(root_dir=storage_root))
            checkpoint_worker = DefaultCWorker(
                AdapterProcessCWorker(runtime),
                AdapterFileSystemCWorker(runtime),
                storage,
                runtime,
            )
            restore_worker = DefaultRWorker(
                AdapterProcessRWorker(runtime),
                AdapterFileSystemRWorker(runtime),
                storage,
            )
            executor = CRExecutor(ExecutorConfig(max_workers=1), checkpoint_worker, restore_worker, telemetry)
            sandbox_manager = RuncSandboxManager(command_runner=runner, paths=sandbox_paths)
            scheduler = CRScheduler(
                SchedulerConfig(
                    min_checkpoint_interval_seconds=0.0,
                    force_checkpoint_after_seconds=0.0,
                    require_change_signal=True,
                ),
                inspector,
                sandbox_manager,
                InMemorySchedulerStateStore(),
                telemetry,
            )
            system = CrabSystem(
                scheduler=scheduler,
                executor=executor,
                storage=storage,
                inspector=inspector,
                runtime=sandbox_manager,
                telemetry=telemetry,
            )

            sandbox_id = system.sandbox_manager.launch(
                "runc",
                {
                    "sandbox_id": "sbx-int",
                    "bundle_path": str(root / "bundles" / "sbx-int"),
                },
            )
            self.assertEqual(sandbox_id, SandboxId("sbx-int"))

            inspector.upsert_snapshot(
                SandboxSnapshot(
                    sandbox_id=sandbox_id,
                    runtime_name="runc",
                    is_running=True,
                    process_changed=False,
                    filesystem_changed=False,
                    observed_at=utc_now(),
                )
            )
            collector.record(
                EBPFEvent(
                    sandbox_id=sandbox_id,
                    kind=EBPFEventKind.FILE_WRITE,
                    observed_at=utc_now(),
                    metadata={"path": "/workspace/file.txt"},
                )
            )

            checkpoint_result = system.checkpoint_if_due(sandbox_id)
            self.assertIsNotNone(checkpoint_result)
            assert checkpoint_result is not None
            self.assertEqual(checkpoint_result.status, JobStatus.SUCCEEDED)
            self.assertIsNotNone(checkpoint_result.manifest)
            self.assertGreaterEqual(len(checkpoint_result.manifest.process_artifacts), 1)
            self.assertTrue(len(checkpoint_result.manifest.filesystem_artifacts) >= 1)

            restore_result = system.restore_once(sandbox_id, checkpoint_result.checkpoint_id)
            self.assertEqual(restore_result.status, JobStatus.SUCCEEDED)

            description = system.sandbox_manager.describe(sandbox_id)
            self.assertEqual(description.status, "running")

            system.sandbox_manager.stop(sandbox_id)
            self.assertEqual(system.sandbox_manager.describe(sandbox_id).status, "stopped")
            system.sandbox_manager.delete(sandbox_id)

            self.assertEqual(
                runner.commands[:5],
                [
                    ("zfs", "destroy", "-r", "pool/crab/sbx-int"),
                    ("zfs", "create", "-o", f"mountpoint={root / 'bundles' / 'sbx-int' / 'rootfs'}", "pool/crab/sbx-int"),
                    (
                        "runc",
                        "--root",
                        str(root / "runtime-state"),
                        "create",
                        "--bundle",
                        str(root / "bundles" / "sbx-int"),
                        "sbx-int",
                    ),
                    ("runc", "--root", str(root / "runtime-state"), "start", "sbx-int"),
                    ("runc", "--root", str(root / "runtime-state"), "pause", "sbx-int"),
                ],
            )
            self.assertIn(
                (
                    "runc",
                    "--root",
                    str(root / "runtime-state"),
                    "checkpoint",
                    "--image-path",
                    str(root / "checkpoints" / "sbx-int" / str(checkpoint_result.checkpoint_id) / "pre_dump"),
                    "--work-path",
                    str(root / "checkpoints" / "sbx-int" / str(checkpoint_result.checkpoint_id) / "pre_dump_work"),
                    "--pre-dump",
                    "--tcp-established",
                    "--shell-job",
                    "--tcp-skip-in-flight",
                    "--ext-unix-sk",
                    "sbx-int",
                ),
                runner.commands,
            )
            self.assertIn(
                (
                    "runc",
                    "--root",
                    str(root / "runtime-state"),
                    "checkpoint",
                    "--image-path",
                    str(root / "checkpoints" / "sbx-int" / str(checkpoint_result.checkpoint_id) / "process"),
                    "--work-path",
                    str(root / "checkpoints" / "sbx-int" / str(checkpoint_result.checkpoint_id) / "work"),
                    "--leave-running=false",
                    "--parent-path",
                    "../pre_dump",
                    "--tcp-established",
                    "--shell-job",
                    "--tcp-skip-in-flight",
                    "--ext-unix-sk",
                    "sbx-int",
                ),
                runner.commands,
            )
            self.assertIn(
                ("zfs", "snapshot", f"pool/crab/sbx-int@{checkpoint_result.checkpoint_id}"),
                runner.commands,
            )
            self.assertIn(
                ("runc", "--root", str(root / "runtime-state"), "delete", "-f", "sbx-int"),
                runner.commands,
            )
            self.assertIn(
                ("zfs", "rollback", "-r", f"pool/crab/sbx-int@{checkpoint_result.checkpoint_id}"),
                runner.commands,
            )
            self.assertIn(
                (
                    "runc",
                    "--root",
                    str(root / "runtime-state"),
                    "restore",
                    "-d",
                    "--bundle",
                    str(root / "bundles" / "sbx-int"),
                    "--image-path",
                    str(root / "checkpoints" / "sbx-int" / str(checkpoint_result.checkpoint_id) / "process"),
                    "--work-path",
                    str(root / "checkpoints" / "sbx-int" / str(checkpoint_result.checkpoint_id) / "work"),
                    "--tcp-established",
                    "--shell-job",
                    "--ext-unix-sk",
                    "sbx-int",
                ),
                runner.commands,
            )
            self.assertEqual(
                runner.commands[-4:],
                [
                    (
                        "runc",
                        "--root",
                        str(root / "runtime-state"),
                        "kill",
                        "sbx-int",
                        "TERM",
                    ),
                    ("runc", "--root", str(root / "runtime-state"), "state", "sbx-int"),
                    ("runc", "--root", str(root / "runtime-state"), "delete", "-f", "sbx-int"),
                    ("zfs", "destroy", "-r", "pool/crab/sbx-int"),
                ],
            )

            event_names = [name for name, _ in telemetry.events]
            self.assertIn("scheduler.evaluate", event_names)
            self.assertIn("executor.job_finished", event_names)
            executor.shutdown()

    def test_notify_fault_marks_sandbox_not_running_before_recovery_runs(self) -> None:
        with tempfile.TemporaryDirectory(prefix="crab_system_it_") as tmp:
            root = Path(tmp)
            runner = FakeCommandRunner()
            telemetry = InMemoryTelemetrySink()
            collector = InMemoryEBPFEventCollector()
            inspector = EBPFSandboxInspector(collector)
            system, executor = self._build_runc_system(
                root=root,
                runner=runner,
                telemetry=telemetry,
                inspector=inspector,
            )

            sandbox_id = system.sandbox_manager.launch(
                "runc",
                {
                    "sandbox_id": "sbx-faulted",
                    "bundle_path": str(root / "bundles" / "sbx-faulted"),
                },
            )
            inspector.upsert_snapshot(
                SandboxSnapshot(
                    sandbox_id=sandbox_id,
                    runtime_name="runc",
                    is_running=True,
                    process_changed=False,
                    filesystem_changed=False,
                    observed_at=utc_now(),
                )
            )

            system.notify_fault(sandbox_id)

            self.assertFalse(inspector.inspect(sandbox_id).is_running)
            self.assertEqual(system.sandbox_manager.describe(sandbox_id).status, "stopped")
            executor.shutdown()

    def test_fault_tolerance_policy_resumes_sandbox_after_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory(prefix="crab_system_it_") as tmp:
            root = Path(tmp)
            runner = FakeCommandRunner()
            telemetry = InMemoryTelemetrySink()
            collector = InMemoryEBPFEventCollector()
            inspector = EBPFSandboxInspector(collector)

            runtime = RuncRuntimeAdapter(
                command_runner=runner,
                paths=RuncRuntimePaths(
                    state_root=root / "runtime-state",
                    bundle_root=root / "bundles",
                    checkpoint_root=root / "checkpoints",
                    zfs_dataset_prefix="pool/crab",
                ),
            )
            storage = LocalCheckpointManager(StorageConfig(root_dir=root / "storage"))
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
            sandbox_manager = RuncSandboxManager(
                command_runner=runner,
                paths=RuncSandboxManagerPaths(
                    state_root=root / "runtime-state",
                    bundle_root=root / "bundles",
                    metadata_root=root / "sandbox-metadata",
                    zfs_dataset_prefix="pool/crab",
                ),
            )
            scheduler = CRScheduler(
                SchedulerConfig(
                    min_checkpoint_interval_seconds=0.0,
                    force_checkpoint_after_seconds=0.0,
                    require_change_signal=True,
                ),
                inspector,
                sandbox_manager,
                InMemorySchedulerStateStore(),
                telemetry,
                FaultToleranceCheckpointingPolicy(
                    SchedulerConfig(
                        min_checkpoint_interval_seconds=0.0,
                        force_checkpoint_after_seconds=0.0,
                        require_change_signal=True,
                    )
                ),
            )
            system = CrabSystem(
                scheduler=scheduler,
                executor=executor,
                storage=storage,
                inspector=inspector,
                runtime=sandbox_manager,
                telemetry=telemetry,
            )
            sandbox_id = system.sandbox_manager.launch(
                "runc",
                {
                    "sandbox_id": "sbx-ft",
                    "bundle_path": str(root / "bundles" / "sbx-ft"),
                },
            )
            inspector.upsert_snapshot(
                SandboxSnapshot(
                    sandbox_id=sandbox_id,
                    runtime_name="runc",
                    is_running=True,
                    process_changed=True,
                    filesystem_changed=True,
                    observed_at=utc_now(),
                )
            )

            result = system.checkpoint_if_due(sandbox_id)

            self.assertIsNotNone(result)
            self.assertEqual(system.sandbox_manager.describe(sandbox_id).status, "running")
            self.assertTrue(any("--leave-running=true" in command for command in runner.commands))
            executor.shutdown()

    def test_fault_notification_restores_latest_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory(prefix="crab_system_it_") as tmp:
            root = Path(tmp)
            runner = FakeCommandRunner()
            telemetry = InMemoryTelemetrySink()
            collector = InMemoryEBPFEventCollector()
            inspector = EBPFSandboxInspector(collector)
            request_store = InMemoryRequestStateStore()

            runtime = RuncRuntimeAdapter(
                command_runner=runner,
                paths=RuncRuntimePaths(
                    state_root=root / "runtime-state",
                    bundle_root=root / "bundles",
                    checkpoint_root=root / "checkpoints",
                    zfs_dataset_prefix="pool/crab",
                ),
            )
            storage = LocalCheckpointManager(StorageConfig(root_dir=root / "storage"))
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
            sandbox_manager = RuncSandboxManager(
                command_runner=runner,
                paths=RuncSandboxManagerPaths(
                    state_root=root / "runtime-state",
                    bundle_root=root / "bundles",
                    metadata_root=root / "sandbox-metadata",
                    zfs_dataset_prefix="pool/crab",
                ),
            )
            scheduler_cfg = SchedulerConfig(
                min_checkpoint_interval_seconds=0.0,
                force_checkpoint_after_seconds=0.0,
                require_change_signal=True,
            )
            scheduler = CRScheduler(
                scheduler_cfg,
                inspector,
                sandbox_manager,
                InMemorySchedulerStateStore(),
                telemetry,
                FaultToleranceCheckpointingPolicy(scheduler_cfg),
            )
            system = CrabSystem(
                scheduler=scheduler,
                executor=executor,
                storage=storage,
                inspector=inspector,
                runtime=sandbox_manager,
                telemetry=telemetry,
                request_state_store=request_store,
            )
            sandbox_id = system.sandbox_manager.launch(
                "runc",
                {
                    "sandbox_id": "sbx-auto-fault",
                    "bundle_path": str(root / "bundles" / "sbx-auto-fault"),
                },
            )
            inspector.upsert_snapshot(
                SandboxSnapshot(
                    sandbox_id=sandbox_id,
                    runtime_name="runc",
                    is_running=True,
                    process_changed=True,
                    filesystem_changed=True,
                    observed_at=utc_now(),
                )
            )
            checkpoint_result = system.checkpoint_if_due(sandbox_id)
            self.assertIsNotNone(checkpoint_result)
            inspector.upsert_snapshot(
                SandboxSnapshot(
                    sandbox_id=sandbox_id,
                    runtime_name="runc",
                    is_running=False,
                    process_changed=True,
                    filesystem_changed=True,
                    observed_at=utc_now(),
                    last_checkpoint_at=utc_now(),
                )
            )

            system.request_state_store = None
            system.start()
            try:
                system.notify_fault(sandbox_id)
                record = self._wait_for_record(system, sandbox_id, "fault")
            finally:
                system.stop()
                executor.shutdown()

            self.assertEqual(record.status, "restored")
            self.assertIn(
                (
                    "runc",
                    "--root",
                    str(root / "runtime-state"),
                    "restore",
                    "-d",
                    "--bundle",
                    str(root / "bundles" / "sbx-auto-fault"),
                    "--image-path",
                    str(root / "checkpoints" / "sbx-auto-fault" / str(checkpoint_result.checkpoint_id) / "process"),
                    "--work-path",
                    str(root / "checkpoints" / "sbx-auto-fault" / str(checkpoint_result.checkpoint_id) / "work"),
                    "--tcp-established",
                    "--shell-job",
                    "--ext-unix-sk",
                    "sbx-auto-fault",
                ),
                runner.commands,
            )

    def test_restore_releases_buffered_response_for_matching_live_request_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory(prefix="crab_system_it_") as tmp:
            root = Path(tmp)
            runner = FakeCommandRunner()
            telemetry = InMemoryTelemetrySink()
            collector = InMemoryEBPFEventCollector()
            inspector = EBPFSandboxInspector(collector)
            request_store = InMemoryRequestStateStore()
            system, executor = self._build_runc_system(
                root=root,
                runner=runner,
                telemetry=telemetry,
                inspector=inspector,
                request_store=request_store,
            )
            sandbox_id = system.sandbox_manager.launch(
                "runc",
                {
                    "sandbox_id": "sbx-live-restore",
                    "bundle_path": str(root / "bundles" / "sbx-live-restore"),
                },
            )
            inspector.upsert_snapshot(
                SandboxSnapshot(
                    sandbox_id=sandbox_id,
                    runtime_name="runc",
                    is_running=True,
                    process_changed=True,
                    filesystem_changed=True,
                    observed_at=utc_now(),
                )
            )
            assert system.response_gate_registry is not None
            system.response_gate_registry.enable()

            response_payload: dict[str, object] = {}
            request_finished = threading.Event()

            def _run_intercept() -> None:
                from crab import CrabRequestInterceptor

                interceptor = CrabRequestInterceptor(
                    upstream_transport=lambda path, headers, body: (
                        200,
                        [("Content-Type", "application/json")],
                        json.dumps(
                            handle_request(
                                path=path,
                                headers=headers,
                                payload=json.loads(body.decode("utf-8")),
                                state=SimulatedLLMState(),
                            ),
                            sort_keys=True,
                        ).encode("utf-8"),
                    ),
                    request_state_store=request_store,
                    on_response_ready=system.notify_live_response_ready,
                    response_gate_registry=system.response_gate_registry,
                )
                _, _, body = interceptor.intercept(
                    path="/v1/chat/completions",
                    headers={
                        "Content-Type": "application/json",
                        "X-Agent-Sandbox-Id": str(sandbox_id),
                        "X-Request-Id": "req-live",
                    },
                    body=json.dumps({"model": "simulated-openai", "messages": [{"role": "user", "content": "continue"}]}).encode("utf-8"),
                )
                response_payload["body"] = json.loads(body.decode("utf-8"))
                request_finished.set()

            thread = threading.Thread(target=_run_intercept)
            thread.start()
            pending = None
            for _ in range(100):
                pending = system.response_gate_registry.get_pending(sandbox_id)
                if pending is not None:
                    break
                request_finished.wait(0.01)
            self.assertIsNotNone(pending)
            assert pending is not None

            checkpoint_job = CheckpointJob(
                job_id=JobId.new(),
                sandbox_id=sandbox_id,
                requested_at=utc_now(),
                reason="manual",
                leave_running=True,
                metadata=system._build_checkpoint_metadata(sandbox_id),
            )
            checkpoint_result = executor.run_checkpoint(checkpoint_job)
            self.assertEqual(checkpoint_result.status, JobStatus.SUCCEEDED)
            restore_result = system.restore_once(sandbox_id, checkpoint_result.checkpoint_id)
            self.assertEqual(restore_result.status, JobStatus.SUCCEEDED)
            self.assertTrue(system._release_checkpoint_response_gate(sandbox_id, checkpoint_result.checkpoint_id))

            thread.join(timeout=2.0)
            self.assertTrue(request_finished.is_set())
            self.assertIn("body", response_payload)
            self.assertEqual(request_store.get(sandbox_id).completed_llm_requests, 1)
            executor.shutdown()

    def test_restore_validation_can_reject_invalid_live_request_checkpoint_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory(prefix="crab_system_it_") as tmp:
            root = Path(tmp)
            runner = FakeCommandRunner()
            telemetry = InMemoryTelemetrySink()
            collector = InMemoryEBPFEventCollector()
            inspector = EBPFSandboxInspector(collector)
            request_store = InMemoryRequestStateStore()
            system, executor = self._build_runc_system(
                root=root,
                runner=runner,
                telemetry=telemetry,
                inspector=inspector,
                request_store=request_store,
                enforce_restore_checkpoint_validation=True,
            )
            sandbox_id = system.sandbox_manager.launch(
                "runc",
                {
                    "sandbox_id": "sbx-restore-validate",
                    "bundle_path": str(root / "bundles" / "sbx-restore-validate"),
                },
            )
            inspector.upsert_snapshot(
                SandboxSnapshot(
                    sandbox_id=sandbox_id,
                    runtime_name="runc",
                    is_running=True,
                    process_changed=True,
                    filesystem_changed=True,
                    observed_at=utc_now(),
                )
            )

            checkpoint_job = CheckpointJob(
                job_id=JobId.new(),
                sandbox_id=sandbox_id,
                requested_at=utc_now(),
                reason="manual",
                leave_running=True,
                metadata={
                    "captures_inflight_llm": True,
                    "captured_request_id": "req-missing",
                },
            )
            checkpoint_result = executor.run_checkpoint(checkpoint_job)
            self.assertEqual(checkpoint_result.status, JobStatus.SUCCEEDED)

            restore_result = system.restore_once(sandbox_id, checkpoint_result.checkpoint_id)

            self.assertEqual(restore_result.status, JobStatus.FAILED)
            self.assertEqual(restore_result.failure_code, FailureCode.VALIDATION_ERROR)
            self.assertIn("no matching interceptor-held request is pending", restore_result.message)
            self.assertFalse(
                any(command[3] == "restore" for command in runner.commands if len(command) > 3),
            )
            executor.shutdown()

    def test_fault_notification_skips_stale_live_request_checkpoint_for_older_safe_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory(prefix="crab_system_it_") as tmp:
            root = Path(tmp)
            runner = FakeCommandRunner()
            telemetry = InMemoryTelemetrySink()
            collector = InMemoryEBPFEventCollector()
            inspector = EBPFSandboxInspector(collector)
            request_store = InMemoryRequestStateStore()
            system, executor = self._build_runc_system(
                root=root,
                runner=runner,
                telemetry=telemetry,
                inspector=inspector,
                request_store=request_store,
                enforce_restore_checkpoint_validation=True,
            )
            sandbox_id = system.sandbox_manager.launch(
                "runc",
                {
                    "sandbox_id": "sbx-stale-skip",
                    "bundle_path": str(root / "bundles" / "sbx-stale-skip"),
                },
            )
            inspector.upsert_snapshot(
                SandboxSnapshot(
                    sandbox_id=sandbox_id,
                    runtime_name="runc",
                    is_running=True,
                    process_changed=True,
                    filesystem_changed=True,
                    observed_at=utc_now(),
                )
            )
            safe_checkpoint = system.checkpoint_if_due(sandbox_id)
            self.assertIsNotNone(safe_checkpoint)
            assert safe_checkpoint is not None

            assert system.response_gate_registry is not None
            system.response_gate_registry.enable()
            context = RequestContext(
                request_id="req-stale",
                sandbox_id=sandbox_id,
                started_at=utc_now(),
                metadata={"provider": "openai"},
            )
            request_store.mark_request_start(context)
            generation = system.response_gate_registry.arm(sandbox_id, context.request_id)
            self.assertIsNotNone(generation)
            inspector.upsert_snapshot(
                SandboxSnapshot(
                    sandbox_id=sandbox_id,
                    runtime_name="runc",
                    is_running=True,
                    process_changed=True,
                    filesystem_changed=True,
                    observed_at=utc_now(),
                )
            )
            stale_checkpoint = system.checkpoint_if_due(sandbox_id)
            self.assertIsNotNone(stale_checkpoint)
            assert stale_checkpoint is not None
            self.assertNotEqual(safe_checkpoint.checkpoint_id, stale_checkpoint.checkpoint_id)
            request_store.mark_request_end(context)

            inspector.upsert_snapshot(
                SandboxSnapshot(
                    sandbox_id=sandbox_id,
                    runtime_name="runc",
                    is_running=False,
                    process_changed=True,
                    filesystem_changed=True,
                    observed_at=utc_now(),
                    last_checkpoint_at=utc_now(),
                )
            )

            system.request_state_store = None
            system.start()
            try:
                system.notify_fault(sandbox_id)
                record = self._wait_for_record(system, sandbox_id, "fault")
            finally:
                system.stop()
                executor.shutdown()

            self.assertEqual(record.status, "restored")
            self.assertEqual(record.checkpoint_id, safe_checkpoint.checkpoint_id)
            self.assertIn(
                (
                    "runc",
                    "--root",
                    str(root / "runtime-state"),
                    "restore",
                    "-d",
                    "--bundle",
                    str(root / "bundles" / "sbx-stale-skip"),
                    "--image-path",
                    str(root / "checkpoints" / "sbx-stale-skip" / str(safe_checkpoint.checkpoint_id) / "process"),
                    "--work-path",
                    str(root / "checkpoints" / "sbx-stale-skip" / str(safe_checkpoint.checkpoint_id) / "work"),
                    "--tcp-established",
                    "--shell-job",
                    "--ext-unix-sk",
                    "sbx-stale-skip",
                ),
                runner.commands,
            )

    def test_fault_notification_relaunches_when_no_satisfiable_live_request_checkpoint_exists(self) -> None:
        with tempfile.TemporaryDirectory(prefix="crab_system_it_") as tmp:
            root = Path(tmp)
            runner = FakeCommandRunner()
            telemetry = InMemoryTelemetrySink()
            collector = InMemoryEBPFEventCollector()
            inspector = EBPFSandboxInspector(collector)
            request_store = InMemoryRequestStateStore()
            relaunched: list[tuple[SandboxId, str, bool]] = []
            system, executor = self._build_runc_system(
                root=root,
                runner=runner,
                telemetry=telemetry,
                inspector=inspector,
                request_store=request_store,
                relaunch_handler=lambda sandbox_id, event_type, preserve_fs: relaunched.append(
                    (sandbox_id, event_type, preserve_fs)
                ),
                enforce_restore_checkpoint_validation=True,
                relaunch_on_restore_failure=True,
            )
            sandbox_id = system.sandbox_manager.launch(
                "runc",
                {
                    "sandbox_id": "sbx-no-live-match",
                    "bundle_path": str(root / "bundles" / "sbx-no-live-match"),
                },
            )
            assert system.response_gate_registry is not None
            system.response_gate_registry.enable()
            context = RequestContext(
                request_id="req-stale-only",
                sandbox_id=sandbox_id,
                started_at=utc_now(),
                metadata={"provider": "openai"},
            )
            request_store.mark_request_start(context)
            system.response_gate_registry.arm(sandbox_id, context.request_id)
            inspector.upsert_snapshot(
                SandboxSnapshot(
                    sandbox_id=sandbox_id,
                    runtime_name="runc",
                    is_running=True,
                    process_changed=True,
                    filesystem_changed=True,
                    observed_at=utc_now(),
                )
            )
            stale_checkpoint = system.checkpoint_if_due(sandbox_id)
            self.assertIsNotNone(stale_checkpoint)
            request_store.mark_request_end(context)
            inspector.upsert_snapshot(
                SandboxSnapshot(
                    sandbox_id=sandbox_id,
                    runtime_name="runc",
                    is_running=False,
                    process_changed=True,
                    filesystem_changed=True,
                    observed_at=utc_now(),
                    last_checkpoint_at=utc_now(),
                )
            )

            system.start()
            try:
                system.notify_fault(sandbox_id)
                record = self._wait_for_record(system, sandbox_id, "fault")
            finally:
                system.stop()
                executor.shutdown()

            self.assertEqual(record.status, "relaunched")
            self.assertEqual(relaunched, [(sandbox_id, "fault", False)])
            self.assertFalse(any(command[3] == "restore" for command in runner.commands if len(command) > 3))

    def test_fault_notification_relaunches_when_restore_fails(self) -> None:
        with tempfile.TemporaryDirectory(prefix="crab_system_it_") as tmp:
            root = Path(tmp)
            runner = FailingRestoreRunner()
            telemetry = InMemoryTelemetrySink()
            collector = InMemoryEBPFEventCollector()
            inspector = EBPFSandboxInspector(collector)
            request_store = InMemoryRequestStateStore()
            relaunched: list[tuple[SandboxId, str, bool]] = []
            restored_metadata: list[tuple[SandboxId, object]] = []

            runtime = RuncRuntimeAdapter(
                command_runner=runner,
                paths=RuncRuntimePaths(
                    state_root=root / "runtime-state",
                    bundle_root=root / "bundles",
                    checkpoint_root=root / "checkpoints",
                    zfs_dataset_prefix="pool/crab",
                ),
            )
            storage = LocalCheckpointManager(StorageConfig(root_dir=root / "storage"))
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
            sandbox_manager = RuncSandboxManager(
                command_runner=runner,
                paths=RuncSandboxManagerPaths(
                    state_root=root / "runtime-state",
                    bundle_root=root / "bundles",
                    metadata_root=root / "sandbox-metadata",
                    zfs_dataset_prefix="pool/crab",
                ),
            )
            scheduler_cfg = SchedulerConfig(
                min_checkpoint_interval_seconds=0.0,
                force_checkpoint_after_seconds=0.0,
                require_change_signal=True,
            )
            scheduler = CRScheduler(
                scheduler_cfg,
                inspector,
                sandbox_manager,
                InMemorySchedulerStateStore(),
                telemetry,
                FaultToleranceCheckpointingPolicy(scheduler_cfg),
            )
            system = CrabSystem(
                scheduler=scheduler,
                executor=executor,
                storage=storage,
                inspector=inspector,
                runtime=sandbox_manager,
                telemetry=telemetry,
                request_state_store=request_store,
                relaunch_handler=lambda sandbox_id, event_type, preserve_fs: relaunched.append(
                    (sandbox_id, event_type, preserve_fs)
                ),
                restore_metadata_handler=lambda sandbox_id, manifest: restored_metadata.append(
                    (sandbox_id, manifest.checkpoint_id)
                ),
                relaunch_on_restore_failure=True,
            )
            sandbox_id = system.sandbox_manager.launch(
                "runc",
                {
                    "sandbox_id": "sbx-auto-fallback",
                    "bundle_path": str(root / "bundles" / "sbx-auto-fallback"),
                },
            )
            inspector.upsert_snapshot(
                SandboxSnapshot(
                    sandbox_id=sandbox_id,
                    runtime_name="runc",
                    is_running=True,
                    process_changed=True,
                    filesystem_changed=True,
                    observed_at=utc_now(),
                )
            )
            checkpoint_result = system.checkpoint_if_due(sandbox_id)
            self.assertIsNotNone(checkpoint_result)
            inspector.upsert_snapshot(
                SandboxSnapshot(
                    sandbox_id=sandbox_id,
                    runtime_name="runc",
                    is_running=False,
                    process_changed=True,
                    filesystem_changed=True,
                    observed_at=utc_now(),
                    last_checkpoint_at=utc_now(),
                )
            )

            system.start()
            try:
                system.notify_fault(sandbox_id)
                record = self._wait_for_record(system, sandbox_id, "fault")
            finally:
                system.stop()
                executor.shutdown()

            self.assertEqual(record.status, "relaunched")
            self.assertEqual(relaunched, [(sandbox_id, "fault", True)])
            self.assertEqual(restored_metadata, [(sandbox_id, checkpoint_result.checkpoint_id)])

    def test_fault_notification_surfaces_restore_failure_when_relaunch_fallback_is_disabled(self) -> None:
        # With the default `relaunch_on_restore_failure=False`, a restore
        # failure during recovery must NOT silently fall through to
        # relaunch_handler. The system instead records a failed recovery and
        # leaves relaunch_handler untouched, so latent bugs surface loudly.
        with tempfile.TemporaryDirectory(prefix="crab_system_it_") as tmp:
            root = Path(tmp)
            runner = FailingRestoreRunner()
            telemetry = InMemoryTelemetrySink()
            collector = InMemoryEBPFEventCollector()
            inspector = EBPFSandboxInspector(collector)
            request_store = InMemoryRequestStateStore()
            relaunched: list[tuple[SandboxId, str, bool]] = []

            runtime = RuncRuntimeAdapter(
                command_runner=runner,
                paths=RuncRuntimePaths(
                    state_root=root / "runtime-state",
                    bundle_root=root / "bundles",
                    checkpoint_root=root / "checkpoints",
                    zfs_dataset_prefix="pool/crab",
                ),
            )
            storage = LocalCheckpointManager(StorageConfig(root_dir=root / "storage"))
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
            sandbox_manager = RuncSandboxManager(
                command_runner=runner,
                paths=RuncSandboxManagerPaths(
                    state_root=root / "runtime-state",
                    bundle_root=root / "bundles",
                    metadata_root=root / "sandbox-metadata",
                    zfs_dataset_prefix="pool/crab",
                ),
            )
            scheduler_cfg = SchedulerConfig(
                min_checkpoint_interval_seconds=0.0,
                force_checkpoint_after_seconds=0.0,
                require_change_signal=True,
            )
            scheduler = CRScheduler(
                scheduler_cfg,
                inspector,
                sandbox_manager,
                InMemorySchedulerStateStore(),
                telemetry,
                FaultToleranceCheckpointingPolicy(scheduler_cfg),
            )
            system = CrabSystem(
                scheduler=scheduler,
                executor=executor,
                storage=storage,
                inspector=inspector,
                runtime=sandbox_manager,
                telemetry=telemetry,
                request_state_store=request_store,
                relaunch_handler=lambda sandbox_id, event_type, preserve_fs: relaunched.append(
                    (sandbox_id, event_type, preserve_fs)
                ),
                # relaunch_on_restore_failure stays at its default False value
            )
            sandbox_id = system.sandbox_manager.launch(
                "runc",
                {
                    "sandbox_id": "sbx-no-fallback",
                    "bundle_path": str(root / "bundles" / "sbx-no-fallback"),
                },
            )
            inspector.upsert_snapshot(
                SandboxSnapshot(
                    sandbox_id=sandbox_id,
                    runtime_name="runc",
                    is_running=True,
                    process_changed=True,
                    filesystem_changed=True,
                    observed_at=utc_now(),
                )
            )
            checkpoint_result = system.checkpoint_if_due(sandbox_id)
            self.assertIsNotNone(checkpoint_result)
            inspector.upsert_snapshot(
                SandboxSnapshot(
                    sandbox_id=sandbox_id,
                    runtime_name="runc",
                    is_running=False,
                    process_changed=True,
                    filesystem_changed=True,
                    observed_at=utc_now(),
                    last_checkpoint_at=utc_now(),
                )
            )

            system.start()
            try:
                system.notify_fault(sandbox_id)
                record = self._wait_for_record(system, sandbox_id, "fault")
            finally:
                system.stop()
                executor.shutdown()

            self.assertEqual(record.status, "failed")
            self.assertIn("recovery restore failed", (record.message or "").lower())
            self.assertEqual(relaunched, [])

    def test_restore_once_fails_when_runtime_does_not_come_back(self) -> None:
        with tempfile.TemporaryDirectory(prefix="crab_system_it_") as tmp:
            root = Path(tmp)
            runner = RestoreMissingRuntimeRunner()
            telemetry = InMemoryTelemetrySink()
            collector = InMemoryEBPFEventCollector()
            inspector = EBPFSandboxInspector(collector)
            system, executor = self._build_runc_system(
                root=root,
                runner=runner,
                telemetry=telemetry,
                inspector=inspector,
            )
            sandbox_id = system.sandbox_manager.launch(
                "runc",
                {
                    "sandbox_id": "sbx-restore-missing",
                    "bundle_path": str(root / "bundles" / "sbx-restore-missing"),
                },
            )
            inspector.upsert_snapshot(
                SandboxSnapshot(
                    sandbox_id=sandbox_id,
                    runtime_name="runc",
                    is_running=True,
                    process_changed=True,
                    filesystem_changed=True,
                    observed_at=utc_now(),
                )
            )
            checkpoint_result = system.checkpoint_if_due(sandbox_id)
            self.assertIsNotNone(checkpoint_result)

            result = system.restore_once(sandbox_id, checkpoint_result.checkpoint_id)
            executor.shutdown()

            self.assertEqual(result.status, JobStatus.FAILED)
            self.assertEqual(result.failure_code, FailureCode.RUNTIME_ERROR)
            self.assertIn("is not running", result.message or "")

    def test_restore_once_preloads_restore_metadata_before_process_resume(self) -> None:
        class RestoreOrderingRunner(FakeCommandRunner):
            def __init__(self, events: list[str]) -> None:
                super().__init__()
                self._events = events

            def run(self, command: list[str], *, cwd: Path | None = None, timeout_seconds: float | None = None):
                if "restore" in command:
                    self._events.append("restore_command")
                return super().run(command, cwd=cwd, timeout_seconds=timeout_seconds)

        with tempfile.TemporaryDirectory(prefix="crab_system_it_") as tmp:
            root = Path(tmp)
            events: list[str] = []
            runner = RestoreOrderingRunner(events)
            telemetry = InMemoryTelemetrySink()
            collector = InMemoryEBPFEventCollector()
            inspector = EBPFSandboxInspector(collector)
            system, executor = self._build_runc_system(
                root=root,
                runner=runner,
                telemetry=telemetry,
                inspector=inspector,
            )
            system.restore_metadata_handler = lambda sandbox_id, manifest: events.append(
                f"restore_metadata:{sandbox_id}:{manifest.checkpoint_id}"
            )
            sandbox_id = system.sandbox_manager.launch(
                "runc",
                {
                    "sandbox_id": "sbx-preload-restore-metadata",
                    "bundle_path": str(root / "bundles" / "sbx-preload-restore-metadata"),
                },
            )
            inspector.upsert_snapshot(
                SandboxSnapshot(
                    sandbox_id=sandbox_id,
                    runtime_name="runc",
                    is_running=True,
                    process_changed=True,
                    filesystem_changed=True,
                    observed_at=utc_now(),
                )
            )
            checkpoint_result = system.checkpoint_if_due(sandbox_id)
            self.assertIsNotNone(checkpoint_result)

            result = system.restore_once(sandbox_id, checkpoint_result.checkpoint_id)
            executor.shutdown()

            self.assertEqual(result.status, JobStatus.SUCCEEDED)
            self.assertTrue(events)
            self.assertTrue(events[0].startswith("restore_metadata:"))
            self.assertIn("restore_command", events)
            self.assertLess(events.index(next(event for event in events if event.startswith("restore_metadata:"))), events.index("restore_command"))

    def test_preemption_notification_checkpoints_then_restores(self) -> None:
        with tempfile.TemporaryDirectory(prefix="crab_system_it_") as tmp:
            root = Path(tmp)
            runner = FakeCommandRunner()
            telemetry = InMemoryTelemetrySink()
            collector = InMemoryEBPFEventCollector()
            inspector = EBPFSandboxInspector(collector)

            runtime = RuncRuntimeAdapter(
                command_runner=runner,
                paths=RuncRuntimePaths(
                    state_root=root / "runtime-state",
                    bundle_root=root / "bundles",
                    checkpoint_root=root / "checkpoints",
                    zfs_dataset_prefix="pool/crab",
                ),
            )
            storage = LocalCheckpointManager(StorageConfig(root_dir=root / "storage"))
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
            sandbox_manager = RuncSandboxManager(
                command_runner=runner,
                paths=RuncSandboxManagerPaths(
                    state_root=root / "runtime-state",
                    bundle_root=root / "bundles",
                    metadata_root=root / "sandbox-metadata",
                    zfs_dataset_prefix="pool/crab",
                ),
            )
            scheduler_cfg = SchedulerConfig(
                min_checkpoint_interval_seconds=0.0,
                force_checkpoint_after_seconds=0.0,
                require_change_signal=False,
            )
            scheduler = CRScheduler(
                scheduler_cfg,
                inspector,
                sandbox_manager,
                InMemorySchedulerStateStore(),
                telemetry,
                SpotPreemptionCheckpointingPolicy(scheduler_cfg),
            )
            system = CrabSystem(
                scheduler=scheduler,
                executor=executor,
                storage=storage,
                inspector=inspector,
                runtime=sandbox_manager,
                telemetry=telemetry,
            )
            sandbox_id = system.sandbox_manager.launch(
                "runc",
                {
                    "sandbox_id": "sbx-auto-spot",
                    "bundle_path": str(root / "bundles" / "sbx-auto-spot"),
                },
            )
            inspector.upsert_snapshot(
                SandboxSnapshot(
                    sandbox_id=sandbox_id,
                    runtime_name="runc",
                    is_running=True,
                    process_changed=False,
                    filesystem_changed=False,
                    observed_at=utc_now(),
                )
            )

            system.start()
            try:
                system.notify_preemption(sandbox_id, grace_remaining_seconds=30.0)
                record = self._wait_for_record(system, sandbox_id, "preemption")
            finally:
                system.stop()
                executor.shutdown()

            self.assertEqual(record.status, "restored")
            self.assertTrue(any("--leave-running=false" in command for command in runner.commands))
            self.assertTrue(any(command[3] == "restore" for command in runner.commands if len(command) > 3))

    def test_preemption_reuses_recent_checkpoint_without_duplicate_pause(self) -> None:
        with tempfile.TemporaryDirectory(prefix="crab_system_it_") as tmp:
            root = Path(tmp)
            runner = FakeCommandRunner()
            telemetry = InMemoryTelemetrySink()
            collector = InMemoryEBPFEventCollector()
            inspector = EBPFSandboxInspector(collector)

            runtime = RuncRuntimeAdapter(
                command_runner=runner,
                paths=RuncRuntimePaths(
                    state_root=root / "runtime-state",
                    bundle_root=root / "bundles",
                    checkpoint_root=root / "checkpoints",
                    zfs_dataset_prefix="pool/crab",
                ),
            )
            storage = LocalCheckpointManager(StorageConfig(root_dir=root / "storage"))
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
            sandbox_manager = RuncSandboxManager(
                command_runner=runner,
                paths=RuncSandboxManagerPaths(
                    state_root=root / "runtime-state",
                    bundle_root=root / "bundles",
                    metadata_root=root / "sandbox-metadata",
                    zfs_dataset_prefix="pool/crab",
                ),
            )
            scheduler_cfg = SchedulerConfig(
                min_checkpoint_interval_seconds=0.0,
                force_checkpoint_after_seconds=0.0,
                require_change_signal=False,
            )
            scheduler = CRScheduler(
                scheduler_cfg,
                inspector,
                sandbox_manager,
                InMemorySchedulerStateStore(),
                telemetry,
                SpotPreemptionCheckpointingPolicy(scheduler_cfg),
            )
            system = CrabSystem(
                scheduler=scheduler,
                executor=executor,
                storage=storage,
                inspector=inspector,
                runtime=sandbox_manager,
                telemetry=telemetry,
            )
            sandbox_id = system.sandbox_manager.launch(
                "runc",
                {
                    "sandbox_id": "sbx-recent-spot",
                    "bundle_path": str(root / "bundles" / "sbx-recent-spot"),
                },
            )
            inspector.upsert_snapshot(
                SandboxSnapshot(
                    sandbox_id=sandbox_id,
                    runtime_name="runc",
                    is_running=True,
                    process_changed=False,
                    filesystem_changed=False,
                    observed_at=utc_now(),
                    metadata={
                        "preemption_notice": True,
                        "preemption_grace_remaining_seconds": 30.0,
                    },
                )
            )

            observed_at = utc_now()
            checkpoint_result = system.checkpoint_once(sandbox_id, leave_running=False)
            self.assertEqual(checkpoint_result.status.value, "succeeded")

            try:
                system._handle_recovery_event(
                    RecoveryEvent(
                        sandbox_id=sandbox_id,
                        event_type="preemption",
                        observed_at=observed_at,
                        grace_remaining_seconds=30.0,
                        reason="preemption",
                    )
                )
            finally:
                executor.shutdown()

            record = system.get_last_recovery_record(sandbox_id)
            self.assertIsNotNone(record)
            assert record is not None
            self.assertEqual(record.status, "restored")
            pause_commands = [command for command in runner.commands if len(command) > 3 and command[3] == "pause"]
            restore_commands = [command for command in runner.commands if len(command) > 3 and command[3] == "restore"]
            self.assertEqual(len(pause_commands), 1)
            self.assertEqual(len(restore_commands), 1)

    def test_preemption_notification_restores_with_delete_after_restore_storage_policy(self) -> None:
        with tempfile.TemporaryDirectory(prefix="crab_system_it_") as tmp:
            root = Path(tmp)
            runner = FakeCommandRunner()
            telemetry = InMemoryTelemetrySink()
            collector = InMemoryEBPFEventCollector()
            inspector = EBPFSandboxInspector(collector)

            runtime = RuncRuntimeAdapter(
                command_runner=runner,
                paths=RuncRuntimePaths(
                    state_root=root / "runtime-state",
                    bundle_root=root / "bundles",
                    checkpoint_root=root / "checkpoints",
                    zfs_dataset_prefix="pool/crab",
                ),
            )
            storage = DeleteAfterRestoreCheckpointManager(
                LocalCheckpointManager(StorageConfig(root_dir=root / "storage"))
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
            sandbox_manager = RuncSandboxManager(
                command_runner=runner,
                paths=RuncSandboxManagerPaths(
                    state_root=root / "runtime-state",
                    bundle_root=root / "bundles",
                    metadata_root=root / "sandbox-metadata",
                    zfs_dataset_prefix="pool/crab",
                ),
            )
            scheduler_cfg = SchedulerConfig(
                min_checkpoint_interval_seconds=0.0,
                force_checkpoint_after_seconds=0.0,
                require_change_signal=False,
            )
            scheduler = CRScheduler(
                scheduler_cfg,
                inspector,
                sandbox_manager,
                InMemorySchedulerStateStore(),
                telemetry,
                SpotPreemptionCheckpointingPolicy(scheduler_cfg),
            )
            system = CrabSystem(
                scheduler=scheduler,
                executor=executor,
                storage=storage,
                inspector=inspector,
                runtime=sandbox_manager,
                telemetry=telemetry,
            )
            sandbox_id = system.sandbox_manager.launch(
                "runc",
                {
                    "sandbox_id": "sbx-auto-spot-prune",
                    "bundle_path": str(root / "bundles" / "sbx-auto-spot-prune"),
                },
            )
            inspector.upsert_snapshot(
                SandboxSnapshot(
                    sandbox_id=sandbox_id,
                    runtime_name="runc",
                    is_running=True,
                    process_changed=False,
                    filesystem_changed=False,
                    observed_at=utc_now(),
                )
            )

            system.start()
            try:
                system.notify_preemption(sandbox_id, grace_remaining_seconds=30.0)
                record = self._wait_for_record(system, sandbox_id, "preemption")
            finally:
                system.stop()
                executor.shutdown()

            self.assertEqual(record.status, "restored")
            self.assertEqual(storage.list_checkpoints(sandbox_id), [])
            self.assertTrue(any(command[3] == "restore" for command in runner.commands if len(command) > 3))


class QuiesceForVerificationTests(unittest.TestCase):
    def _build_system(self, *, paused: bool, active_job_counts: list[int]):
        from crab.models import SandboxDescription

        sandbox_id = SandboxId("sbx-quiesce")

        class FakeScheduler:
            def __init__(self) -> None:
                self.deactivated: list[SandboxId] = []

            def deactivate_sandbox(self, sid: SandboxId) -> None:
                self.deactivated.append(sid)

            def is_sandbox_deactivated(self, sid: SandboxId) -> bool:
                return sid in self.deactivated

        class FakeExecutor:
            def __init__(self, counts: list[int]) -> None:
                self._counts = list(counts)
                self.polls = 0

            def has_active_job(self, sid: SandboxId) -> bool:
                self.polls += 1
                if not self._counts:
                    return False
                remaining = self._counts.pop(0)
                return remaining > 0

        class FakeRuntime:
            def __init__(self, is_paused: bool) -> None:
                self._status = "paused" if is_paused else "running"
                self.describe_calls: list[SandboxId] = []
                self.resume_calls: list[SandboxId] = []
                self.sync_calls: list[tuple[SandboxId, bool]] = []

            def describe(self, sid: SandboxId) -> SandboxDescription:
                self.describe_calls.append(sid)
                return SandboxDescription(sandbox_id=sid, runtime_name="runc", status=self._status)

            def resume(self, sid: SandboxId) -> None:
                self.resume_calls.append(sid)
                self._status = "running"

            def sync_runtime_state(self, sid: SandboxId, *, is_running: bool) -> None:
                self.sync_calls.append((sid, is_running))

        system = CrabSystem.__new__(CrabSystem)
        system.scheduler = FakeScheduler()
        system.executor = FakeExecutor(active_job_counts)
        system.runtime = FakeRuntime(is_paused=paused)
        system.inspector = object()
        return system, sandbox_id

    def test_quiesce_deactivates_drains_and_resumes_paused_sandbox(self) -> None:
        system, sandbox_id = self._build_system(paused=True, active_job_counts=[2, 1, 0])

        system.quiesce_for_verification(sandbox_id, poll_interval_seconds=0.0)

        self.assertEqual(system.scheduler.deactivated, [sandbox_id])
        self.assertGreaterEqual(system.executor.polls, 3)
        self.assertEqual(system.runtime.resume_calls, [sandbox_id])
        self.assertEqual(system.runtime.sync_calls, [(sandbox_id, True)])

    def test_quiesce_skips_resume_when_not_paused(self) -> None:
        system, sandbox_id = self._build_system(paused=False, active_job_counts=[0])

        system.quiesce_for_verification(sandbox_id, poll_interval_seconds=0.0)

        self.assertEqual(system.scheduler.deactivated, [sandbox_id])
        self.assertEqual(system.runtime.resume_calls, [])
        self.assertEqual(system.runtime.sync_calls, [])

    def test_quiesce_times_out_when_executor_never_drains(self) -> None:
        system, sandbox_id = self._build_system(paused=False, active_job_counts=[])
        system.executor.has_active_job = lambda sid: True  # type: ignore[assignment]

        system.quiesce_for_verification(
            sandbox_id,
            drain_timeout_seconds=0.0,
            poll_interval_seconds=0.0,
        )

        self.assertEqual(system.scheduler.deactivated, [sandbox_id])


if __name__ == "__main__":
    unittest.main()
