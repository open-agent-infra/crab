from __future__ import annotations

from dataclasses import dataclass, field, replace
from concurrent.futures import Future, ThreadPoolExecutor
import logging
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Lock, Thread
import time
from typing import Callable

from .config import ExecutorConfig, SchedulerConfig, StorageConfig, TelemetryConfig
from .contracts import CheckpointManager, Runtime, SandboxInspector, TelemetrySink
from .executor import CRExecutor
from .ids import CheckpointId, JobId
from .inspector import EBPFSandboxInspector
from .remote_inspector import HostInspectorServiceClient, RemoteSandboxInspector
from .interceptor import InMemoryRequestStateStore, RequestAwareSandboxInspector, SandboxResponseGateRegistry
from .models import (
    CheckpointManifest,
    CheckpointJob,
    CheckpointResult,
    FailureCode,
    JobStatus,
    RecoveryEvent,
    RecoveryRecord,
    RestoreJob,
    RestoreResult,
    SandboxId,
    utc_now,
)
from .runtime import InMemoryRuntime, RuncRuntime, RuncRuntimeOptions
from .scheduler import CRScheduler, InMemorySchedulerStateStore, SchedulerPolicy
from .storage import LocalCheckpointManager
from .telemetry import (
    CompositeTelemetrySink,
    ConfiguredTelemetrySink,
    NoopTelemetrySink,
    build_configured_telemetry_sink,
    start_operation,
)
from .workers import (
    AdapterFileSystemCWorker,
    AdapterFileSystemRWorker,
    AdapterProcessCWorker,
    AdapterProcessRWorker,
    DefaultCWorker,
    DefaultRWorker,
)
from .workers.composite import resolve_restore_manifest

logger = logging.getLogger(__name__)

_CAPTURES_INFLIGHT_LLM = "captures_inflight_llm"
_CAPTURED_REQUEST_ID = "captured_request_id"
_CAPTURED_REQUEST_GENERATION = "captured_request_generation"
_CAPTURED_REQUEST_PROVIDER = "captured_request_provider"
_CAPTURED_REQUEST_STARTED_AT = "captured_request_started_at"
_CAPTURED_REQUEST_GROUP_KIND = "captured_request_group_kind"
_CAPTURED_REQUEST_GROUP_ID = "captured_request_group_id"
_CAPTURED_REQUEST_IDS = "captured_request_ids"
_CAPTURED_REQUEST_GROUP_STARTED_AT = "captured_request_group_started_at"
_RESTORE_RUNTIME_READY_ATTEMPTS = 10
_RESTORE_RUNTIME_READY_DELAY_S = 0.1


def _checkpoint_scope(job: CheckpointJob) -> str:
    if job.checkpoint_process and job.checkpoint_filesystem:
        return "full"
    if job.checkpoint_process:
        return "process_only"
    if job.checkpoint_filesystem:
        return "filesystem_only"
    return "none"


def _checkpoint_guard_from_inspector(inspector: SandboxInspector) -> Callable[[CheckpointJob], tuple[bool, str | None]]:
    def guard(job: CheckpointJob) -> tuple[bool, str | None]:
        try:
            snapshot = inspector.inspect(job.sandbox_id)
        except Exception:
            return True, None
        if snapshot.is_running:
            return True, None
        return False, "sandbox_not_running"

    return guard


@dataclass
class CrabSystem:
    scheduler: CRScheduler
    executor: CRExecutor
    storage: CheckpointManager
    inspector: SandboxInspector
    runtime: Runtime
    telemetry: TelemetrySink
    request_state_store: InMemoryRequestStateStore | None = None
    response_gate_registry: SandboxResponseGateRegistry | None = None
    relaunch_handler: Callable[[SandboxId, str, bool], None] | None = None
    extra_checkpoint_metadata_provider: Callable[[SandboxId], dict[str, object]] | None = None
    restore_metadata_handler: Callable[[SandboxId, CheckpointManifest], None] | None = None
    recovery_delay_seconds: float = 0.0
    enforce_restore_checkpoint_validation: bool = False
    # When False (the default), a restore failure during recovery surfaces as a
    # hard error rather than silently falling back to `relaunch_handler`. The
    # relaunch path is intended as an availability backstop, but it tends to
    # mask real bugs (corrupt checkpoints, broken restore plumbing) — especially
    # for callers that set `checkpoint_full_baseline_on_first_checkpoint=true`
    # and therefore expect every recovery to use a complete checkpoint. Set to
    # True to opt back into relaunch on restore failure.
    relaunch_on_restore_failure: bool = False
    _interceptor_lock: Lock = field(init=False, repr=False)
    _interceptor_pending: set[SandboxId] = field(init=False, repr=False)
    _coordination_lock: Lock = field(init=False, repr=False)
    _active_coordination: set[SandboxId] = field(init=False, repr=False)
    _recovery_lock: Lock = field(init=False, repr=False)
    _recovery_queue: Queue[RecoveryEvent | None] = field(init=False, repr=False)
    _recovery_records: dict[SandboxId, RecoveryRecord] = field(init=False, repr=False)
    _stop_event: Event = field(init=False, repr=False)
    _monitor_thread: Thread | None = field(init=False, repr=False, default=None)
    _recovery_pool: ThreadPoolExecutor | None = field(init=False, repr=False, default=None)
    _recovery_futures: list[Future[None]] = field(init=False, repr=False, default_factory=list)
    _recovery_worker_count: int = field(init=False, repr=False, default=0)
    _coordination_pool: ThreadPoolExecutor | None = field(init=False, repr=False, default=None)

    @property
    def sandbox_manager(self) -> Runtime:
        return self.runtime

    @sandbox_manager.setter
    def sandbox_manager(self, value: Runtime) -> None:
        self.runtime = value

    def __post_init__(self) -> None:
        if self.response_gate_registry is None:
            self.response_gate_registry = SandboxResponseGateRegistry()
        self._interceptor_lock = Lock()
        self._interceptor_pending = set()
        self._coordination_lock = Lock()
        self._active_coordination = set()
        self._recovery_lock = Lock()
        self._recovery_queue = Queue()
        self._recovery_records = {}
        self._stop_event = Event()

    def start(self) -> None:
        with self._coordination_lock:
            request_running = self._monitor_thread is not None and self._monitor_thread.is_alive()
            recovery_running = self._recovery_pool is not None and any(
                not future.done() for future in self._recovery_futures
            )
            if request_running and recovery_running:
                return
            self._stop_event.clear()
            if self.response_gate_registry is not None:
                self.response_gate_registry.enable()
            if self._coordination_pool is None:
                self._coordination_pool = ThreadPoolExecutor(
                    max_workers=self.executor.config.resolved_coordination_workers,
                    thread_name_prefix="crab-coordinate",
                )
            if self.request_state_store is not None and not request_running:
                self._monitor_thread = Thread(target=self._run_monitor_loop, name="crab-system", daemon=True)
                self._monitor_thread.start()
            if not recovery_running:
                recovery_workers = self.executor.config.resolved_restore_workers
                self._recovery_worker_count = recovery_workers
                self._recovery_pool = ThreadPoolExecutor(
                    max_workers=recovery_workers,
                    thread_name_prefix="crab-recovery",
                )
                self._recovery_futures = [
                    self._recovery_pool.submit(self._run_recovery_loop)
                    for _ in range(recovery_workers)
                ]
        logger.info("Started CrabSystem background loops")

    def stop(self) -> None:
        self._stop_event.set()
        if self.response_gate_registry is not None:
            self.response_gate_registry.disable()
        if self.request_state_store is not None:
            self.request_state_store.notify_waiters()
        for _ in range(self._recovery_worker_count):
            self._recovery_queue.put(None)
        thread = self._monitor_thread
        if thread is not None:
            thread.join(timeout=5.0)
        self._monitor_thread = None
        recovery_pool = self._recovery_pool
        self._recovery_pool = None
        self._recovery_futures = []
        self._recovery_worker_count = 0
        if recovery_pool is not None:
            recovery_pool.shutdown(wait=True, cancel_futures=False)
        coordination_pool = self._coordination_pool
        self._coordination_pool = None
        if coordination_pool is not None:
            coordination_pool.shutdown(wait=True, cancel_futures=False)
        self.telemetry.flush()
        logger.info("Stopped CrabSystem background loops")

    def _telemetry_attrs(
        self,
        sandbox_id: SandboxId,
        *,
        component: str,
        checkpoint_id: CheckpointId | None = None,
        job_id: JobId | None = None,
        event_type: str | None = None,
        request_id: str | None = None,
        extra: dict[str, object] | None = None,
    ) -> dict[str, object]:
        attributes: dict[str, object] = {
            "component": component,
            "sandbox_id": str(sandbox_id),
        }
        if checkpoint_id is not None:
            attributes["checkpoint_id"] = str(checkpoint_id)
        if job_id is not None:
            attributes["job_id"] = str(job_id)
        if event_type is not None:
            attributes["event_type"] = event_type
        if request_id is not None:
            attributes["request_id"] = request_id
        if extra:
            attributes.update(extra)
        return attributes

    def checkpoint_once(self, sandbox_id: SandboxId, leave_running: bool=False) -> CheckpointResult:
        logger.info("Running manual checkpoint for sandbox %s", sandbox_id)
        operation = start_operation(
            self.telemetry,
            "checkpoint.flow",
            self._telemetry_attrs(sandbox_id, component="system", extra={"reason": "manual"}),
        )
        pending_request = self._next_pending_live_request(sandbox_id)
        paused = self._pause_for_manual_checkpoint(sandbox_id)
        result: CheckpointResult | None = None
        job: CheckpointJob | None = None
        try:
            job = CheckpointJob(
                job_id=JobId.new(),
                sandbox_id=sandbox_id,
                requested_at=utc_now(),
                reason="manual",
                leave_running=leave_running,
                metadata=self._build_checkpoint_metadata(sandbox_id, pending_request=pending_request),
            )
            result = self.executor.run_checkpoint(job)
            if result.status.value == "succeeded":
                self.scheduler.mark_checkpoint_complete(
                    sandbox_id,
                    result.finished_at,
                    process_checkpoint_id=(
                        result.checkpoint_id if job.checkpoint_process else None
                    ),
                    is_incremental_process=job.is_incremental_process,
                )
                self.inspector.mark_checkpoint_complete(
                    sandbox_id,
                    process=job.checkpoint_process,
                    filesystem=job.checkpoint_filesystem,
                    at=result.finished_at,
                )
        finally:
            if paused and self._should_resume_after_checkpoint(job, result):
                self._resume_sandbox(sandbox_id)
            self._release_response_gate(sandbox_id, pending_request)
            self._refresh_interceptor_pending_state(sandbox_id)
        finish_attrs: dict[str, object] = {}
        if result is not None:
            finish_attrs["checkpoint_id"] = str(result.checkpoint_id)
            finish_attrs["failure_code"] = result.failure_code.value
        if job is not None:
            finish_attrs["job_id"] = str(job.job_id)
            finish_attrs["checkpoint_scope"] = _checkpoint_scope(job)
        operation.finish(
            status="failed" if result is None else result.status.value,
            attributes=finish_attrs,
        )
        assert result is not None
        logger.info("Manual checkpoint for sandbox %s finished with status=%s", sandbox_id, result.status.value)
        return result

    def checkpoint_if_due(self, sandbox_id: SandboxId) -> CheckpointResult | None:
        pending_request = self._next_pending_live_request(sandbox_id)
        try:
            logger.debug("Checking whether sandbox %s is due for checkpoint", sandbox_id)
            result = self._execute_checkpoint_flow(sandbox_id, pending_request=pending_request)
            if result is None:
                logger.debug("Sandbox %s is not due for checkpoint", sandbox_id)
                return None
            logger.info("Checkpoint-if-due for sandbox %s finished with status=%s", sandbox_id, result.status.value)
            return result
        finally:
            self._release_response_gate(sandbox_id, pending_request)
            self._refresh_interceptor_pending_state(sandbox_id)

    def checkpoint_due_sandboxes(self, sandbox_ids: list[SandboxId]) -> list[CheckpointResult]:
        results: list[CheckpointResult] = []
        for sandbox_id in sandbox_ids:
            result = self.checkpoint_if_due(sandbox_id)
            if result is not None:
                results.append(result)
        return results

    def restore_once(self, sandbox_id: SandboxId, checkpoint_id) -> RestoreResult:
        logger.info("Running manual restore for sandbox %s checkpoint=%s", sandbox_id, checkpoint_id)
        started = utc_now()
        restore_checkpoint_id = CheckpointId(str(checkpoint_id))
        operation = start_operation(
            self.telemetry,
            "restore.flow",
            self._telemetry_attrs(
                sandbox_id,
                component="system",
                checkpoint_id=restore_checkpoint_id,
                extra={"reason": "manual"},
            ),
        )
        restore_message = (
            self._validate_restore_checkpoint(sandbox_id, restore_checkpoint_id)
            if self.enforce_restore_checkpoint_validation
            else None
        )
        if restore_message is not None:
            logger.warning(
                "Skipping restore for sandbox=%s checkpoint=%s message=%s",
                sandbox_id,
                restore_checkpoint_id,
                restore_message,
            )
            failed_result = RestoreResult(
                job_id=JobId.new(),
                sandbox_id=sandbox_id,
                checkpoint_id=restore_checkpoint_id,
                status=JobStatus.FAILED,
                started_at=started,
                finished_at=utc_now(),
                failure_code=FailureCode.VALIDATION_ERROR,
                message=restore_message,
            )
            operation.finish(
                status=failed_result.status.value,
                attributes={
                    "job_id": str(failed_result.job_id),
                    "failure_code": failed_result.failure_code.value,
                },
            )
            return failed_result
        restore_manifest = None
        if self.restore_metadata_handler is not None:
            restore_manifest = self._resolve_restore_manifest(sandbox_id, restore_checkpoint_id)
            if restore_manifest is not None:
                # Preload replay/router state before the restored process can
                # resume making requests. This avoids a narrow race where the
                # process issues its first post-restore request before the
                # replay service cursor is rewound to the checkpoint position.
                self.restore_metadata_handler(sandbox_id, restore_manifest)
        self.runtime.prepare_for_restore(sandbox_id)
        job = RestoreJob(
            job_id=JobId.new(),
            sandbox_id=sandbox_id,
            checkpoint_id=restore_checkpoint_id,
            requested_at=utc_now(),
            reason="manual",
        )
        result = self.executor.run_restore(job)
        if result.status.value == "succeeded":
            runtime_state = self._wait_for_runtime_running(sandbox_id)
            if runtime_state is None:
                self.runtime.sync_runtime_state(sandbox_id, is_running=False)
                message = f"restore completed but sandbox {sandbox_id} is not running"
                logger.warning(
                    "Restore reported success but sandbox is not running sandbox=%s checkpoint=%s",
                    sandbox_id,
                    restore_checkpoint_id,
                )
                result = replace(
                    result,
                    status=JobStatus.FAILED,
                    finished_at=utc_now(),
                    failure_code=FailureCode.RUNTIME_ERROR,
                    message=message,
                )
            else:
                self.runtime.mark_restored(sandbox_id)
                self._mark_sandbox_running(sandbox_id)
                self.storage.handle_restore_complete(sandbox_id, result.checkpoint_id)
        logger.info(
            "Manual restore for sandbox %s checkpoint=%s finished with status=%s",
            sandbox_id,
            restore_checkpoint_id,
            result.status.value,
        )
        operation.finish(
            status=result.status.value,
            attributes={
                "job_id": str(result.job_id),
                "failure_code": result.failure_code.value,
            },
        )
        return result

    def _wait_for_runtime_running(self, sandbox_id: SandboxId):
        for attempt in range(_RESTORE_RUNTIME_READY_ATTEMPTS):
            try:
                runtime_state = self.runtime.inspect_runtime(sandbox_id)
            except Exception:
                runtime_state = None
            if runtime_state is not None and runtime_state.is_running:
                return runtime_state
            if attempt + 1 < _RESTORE_RUNTIME_READY_ATTEMPTS:
                time.sleep(_RESTORE_RUNTIME_READY_DELAY_S)
        return None

    def notify_fault(self, sandbox_id: SandboxId, *, reason: str = "fault") -> None:
        logger.info("Received fault notification for sandbox=%s reason=%s", sandbox_id, reason)
        self._mark_sandbox_not_running(sandbox_id)
        event = RecoveryEvent(
            sandbox_id=sandbox_id,
            event_type="fault",
            observed_at=utc_now(),
            reason=reason,
        )
        self._recovery_queue.put(event)
        self.telemetry.emit_event(
            "recovery.event_received",
            {"sandbox_id": str(sandbox_id), "event_type": "fault", "reason": reason},
        )

    def notify_preemption(self, sandbox_id: SandboxId, *, grace_remaining_seconds: float) -> None:
        logger.info(
            "Received preemption notification for sandbox=%s grace_remaining_seconds=%.3f",
            sandbox_id,
            grace_remaining_seconds,
        )
        self._merge_snapshot_metadata(
            sandbox_id,
            preemption_notice=True,
            preemption_grace_remaining_seconds=grace_remaining_seconds,
        )
        event = RecoveryEvent(
            sandbox_id=sandbox_id,
            event_type="preemption",
            observed_at=utc_now(),
            grace_remaining_seconds=grace_remaining_seconds,
            reason="preemption",
        )
        self._recovery_queue.put(event)
        self.telemetry.emit_event(
            "recovery.event_received",
            {
                "sandbox_id": str(sandbox_id),
                "event_type": "preemption",
                "grace_remaining_seconds": grace_remaining_seconds,
            },
        )

    def get_last_recovery_record(self, sandbox_id: SandboxId) -> RecoveryRecord | None:
        with self._recovery_lock:
            return self._recovery_records.get(sandbox_id)

    def notify_interceptor_state_change(self, sandbox_id: SandboxId) -> None:
        # Reconcile against the actual gate state instead of blindly adding.
        # The monitor loop (auto_cr only) does the same reconciliation on its
        # own cadence; in manual mode where the monitor doesn't run, this is
        # the *only* path that prunes the set, so an unconditional add would
        # leak the sandbox forever and make has_pending_interceptor_signal
        # return True permanently — which deadlocks wait_for_task_completion's
        # replay-complete short-circuit.
        self._refresh_interceptor_pending_state(sandbox_id)
        with self._interceptor_lock:
            pending = sandbox_id in self._interceptor_pending
        if self.request_state_store is not None:
            self.request_state_store.notify_waiters()
        logger.debug("Recorded interceptor state change for sandbox %s pending=%s", sandbox_id, pending)
        self.telemetry.emit_event(
            "interceptor.state_changed",
            {
                "sandbox_id": str(sandbox_id),
                "pending": pending,
            },
        )

    def notify_live_response_ready(
        self,
        sandbox_id: SandboxId,
        request_id: str,
        generation: int | None = None,
    ) -> None:
        self.executor.notify_live_response_ready(
            sandbox_id,
            request_id,
            generation=generation,
        )

    def has_pending_interceptor_signal(self, sandbox_id: SandboxId) -> bool:
        with self._interceptor_lock:
            return sandbox_id in self._interceptor_pending

    def _run_monitor_loop(self) -> None:
        assert self.request_state_store is not None
        while not self._stop_event.is_set():
            change = self.request_state_store.wait_for_change(timeout=0.5)
            if change is not None:
                self._refresh_interceptor_pending_state(change.sandbox_id)
                if change.event_type == "request_start":
                    coord_decision = self._should_coordinate_live_request(change.sandbox_id, change.request_id)
                    logger.debug(
                        "DIAG.monitor.request_start sandbox=%s request_id=%s should_coord=%s",
                        change.sandbox_id,
                        "" if change.request_id is None else change.request_id,
                        coord_decision,
                    )
                    if coord_decision:
                        self._dispatch_coordination(change.sandbox_id)
                    else:
                        logger.debug(
                            "Skipping stale request_start coordination sandbox=%s request_id=%s",
                            change.sandbox_id,
                            "" if change.request_id is None else change.request_id,
                        )
                else:
                    logger.debug(
                        "DIAG.monitor.event sandbox=%s event_type=%s request_id=%s",
                        change.sandbox_id,
                        change.event_type,
                        "" if change.request_id is None else change.request_id,
                    )
            self._dispatch_pending_coordination()

    def _should_coordinate_live_request(self, sandbox_id: SandboxId, request_id: str | None) -> bool:
        if self.request_state_store is None or self.response_gate_registry is None:
            logger.debug(
                "DIAG.coord.check.no_store sandbox=%s",
                sandbox_id,
            )
            return False
        request_state = self.request_state_store.get(sandbox_id)
        if not request_state.llm_request_in_flight:
            logger.debug(
                "DIAG.coord.check.no_in_flight sandbox=%s request_id=%s active_llm_requests=%d",
                sandbox_id,
                "" if request_id is None else request_id,
                request_state.active_llm_requests,
            )
            return False
        if request_id is None:
            oldest = self.response_gate_registry.get_oldest_pending(sandbox_id)
            logger.debug(
                "DIAG.coord.check.no_request_id sandbox=%s oldest_pending=%s",
                sandbox_id,
                "" if oldest is None else oldest.request_id,
            )
            return oldest is not None
        found = self.response_gate_registry.find_pending_request(sandbox_id, request_id)
        logger.debug(
            "DIAG.coord.check.find_pending sandbox=%s request_id=%s found=%s",
            sandbox_id,
            request_id,
            "" if found is None else found.generation,
        )
        return found is not None

    def _run_recovery_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                event = self._recovery_queue.get(timeout=0.5)
            except Empty:
                continue
            if event is None:
                return
            logger.info(
                "Recovery loop dequeued event sandbox=%s event_type=%s reason=%s",
                event.sandbox_id,
                event.event_type,
                event.reason,
            )
            self._handle_recovery_event(event)

    def _dispatch_coordination(self, sandbox_id: SandboxId) -> None:
        with self._coordination_lock:
            if sandbox_id in self._active_coordination:
                logger.debug(
                    "DIAG.coord.dispatch.skipped_already_active sandbox=%s",
                    sandbox_id,
                )
                return
            self._active_coordination.add(sandbox_id)
            pool = self._coordination_pool
        if pool is None:
            with self._coordination_lock:
                self._active_coordination.discard(sandbox_id)
            raise RuntimeError("coordination pool is not running")
        logger.info("DIAG.coord.dispatch.submitted sandbox=%s", sandbox_id)
        pool.submit(self._coordinate_sandbox_request, sandbox_id)

    def _coordinate_sandbox_request(self, sandbox_id: SandboxId) -> None:
        iteration = 0
        try:
            while not self._stop_event.is_set() and self._should_coordinate_any_pending_request(sandbox_id):
                iteration += 1
                pending_request = self._next_pending_live_request(sandbox_id)
                if pending_request is None:
                    logger.debug(
                        "DIAG.coord.loop.no_pending sandbox=%s iter=%d",
                        sandbox_id,
                        iteration,
                    )
                    break
                logger.debug(
                    "DIAG.coord.loop.execute sandbox=%s iter=%d request_id=%s generation=%s",
                    sandbox_id,
                    iteration,
                    pending_request.request_id,
                    pending_request.generation,
                )
                try:
                    self._execute_checkpoint_flow(sandbox_id, pending_request=pending_request)
                except Exception:
                    logger.exception(
                        "Checkpoint coordination failed for sandbox %s request_id=%s generation=%s",
                        sandbox_id,
                        pending_request.request_id,
                        pending_request.generation,
                    )
                    self._resume_sandbox(sandbox_id)
                finally:
                    self._release_response_gate(sandbox_id, pending_request)
                    self._refresh_interceptor_pending_state(sandbox_id)
        finally:
            should_redispatch = False
            with self._coordination_lock:
                self._active_coordination.discard(sandbox_id)
            self._refresh_interceptor_pending_state(sandbox_id)
            if self._should_coordinate_any_pending_request(sandbox_id):
                should_redispatch = True
                self._dispatch_coordination(sandbox_id)
            logger.info(
                "DIAG.coord.loop.exit sandbox=%s iters=%d redispatched=%s",
                sandbox_id,
                iteration,
                should_redispatch,
            )

    def _handle_recovery_event(self, event: RecoveryEvent) -> None:
        if not self._acquire_coordination(event.sandbox_id):
            return
        operation = start_operation(
            self.telemetry,
            "recovery.total",
            self._telemetry_attrs(
                event.sandbox_id,
                component="recovery",
                event_type=event.event_type,
                extra={"reason": event.reason},
            ),
        )
        started = utc_now()
        checkpoint_id = None
        restore_manifest: CheckpointManifest | None = None
        pinned_restore_ids: list[CheckpointId] = []
        status = "failed"
        message = None
        try:
            logger.info(
                "Handling recovery event sandbox=%s event_type=%s reason=%s",
                event.sandbox_id,
                event.event_type,
                event.reason,
            )
            queue_wait_ms = max(0.0, (started - event.received_at).total_seconds() * 1000.0)
            self.telemetry.emit_metric(
                "recovery.queue_wait_ms",
                queue_wait_ms,
                self._telemetry_attrs(
                    event.sandbox_id,
                    component="recovery",
                    event_type=event.event_type,
                    extra={"reason": event.reason},
                ),
            )
            self.telemetry.emit_event(
                "recovery.started",
                {
                    "sandbox_id": str(event.sandbox_id),
                    "event_type": event.event_type,
                },
            )
            if event.event_type == "preemption":
                selection_operation = start_operation(
                    self.telemetry,
                    "recovery.select_checkpoint",
                    self._telemetry_attrs(event.sandbox_id, component="recovery", event_type=event.event_type),
                )
                checkpoint_id = self._select_recovery_checkpoint_after(
                    event.sandbox_id,
                    observed_after=event.observed_at,
                )
                selection_operation.finish(status="succeeded", attributes={"checkpoint_id": "" if checkpoint_id is None else str(checkpoint_id)})
                if checkpoint_id is not None:
                    logger.info(
                        "Reusing checkpoint already captured after preemption notice sandbox=%s checkpoint=%s",
                        event.sandbox_id,
                        checkpoint_id,
                    )
                else:
                    logger.info("Triggering preemption checkpoint flow for sandbox=%s", event.sandbox_id)
                    self._drain_active_runtime_execs(event.sandbox_id)
                    try:
                        checkpoint_result = self._execute_checkpoint_flow(event.sandbox_id)
                    except Exception:
                        checkpoint_id = self._select_recovery_checkpoint_after(
                            event.sandbox_id,
                            observed_after=event.observed_at,
                        )
                        if checkpoint_id is None:
                            raise
                        logger.warning(
                            "Preemption checkpoint flow failed after a recent checkpoint was captured; continuing with recovery sandbox=%s checkpoint=%s",
                            event.sandbox_id,
                            checkpoint_id,
                        )
                    else:
                        if checkpoint_result is not None and checkpoint_result.status.value == "succeeded":
                            checkpoint_id = checkpoint_result.checkpoint_id
                            logger.info(
                                "Preemption checkpoint completed sandbox=%s checkpoint=%s",
                                event.sandbox_id,
                                checkpoint_id,
                            )
                        elif checkpoint_result is not None:
                            message = checkpoint_result.message
                            logger.warning(
                                "Preemption checkpoint failed sandbox=%s message=%s",
                                event.sandbox_id,
                                checkpoint_result.message,
                            )
            if checkpoint_id is None:
                selection_operation = start_operation(
                    self.telemetry,
                    "recovery.select_checkpoint",
                    self._telemetry_attrs(event.sandbox_id, component="recovery", event_type=event.event_type),
                )
                checkpoint_id = self._select_recovery_checkpoint(event.sandbox_id)
                selection_operation.finish(status="succeeded", attributes={"checkpoint_id": "" if checkpoint_id is None else str(checkpoint_id)})
                logger.info(
                    "Resolved latest checkpoint for recovery sandbox=%s checkpoint=%s",
                    event.sandbox_id,
                    "" if checkpoint_id is None else checkpoint_id,
                )
            if checkpoint_id is not None:
                self.telemetry.emit_event(
                    "recovery.checkpoint_resolved",
                    {
                        "sandbox_id": str(event.sandbox_id),
                        "event_type": event.event_type,
                        "checkpoint_id": str(checkpoint_id),
                    },
                )
            else:
                self.telemetry.emit_event(
                    "recovery.checkpoint_missing",
                    {
                        "sandbox_id": str(event.sandbox_id),
                        "event_type": event.event_type,
                    },
                )
            if checkpoint_id is not None:
                if self.recovery_delay_seconds > 0:
                    logger.info(
                        "Sleeping before restore sandbox=%s delay_seconds=%.3f",
                        event.sandbox_id,
                        self.recovery_delay_seconds,
                    )
                    time.sleep(self.recovery_delay_seconds)
                logger.info(
                    "Starting recovery restore sandbox=%s checkpoint=%s",
                    event.sandbox_id,
                    checkpoint_id,
                )
                restore_manifest, pinned_restore_ids = self._pin_restore_checkpoints(event.sandbox_id, checkpoint_id)
                if restore_manifest is None:
                    raise FileNotFoundError(f"manifest not found: selected checkpoint {checkpoint_id}")
                restore_operation = start_operation(
                    self.telemetry,
                    "recovery.restore",
                    self._telemetry_attrs(
                        event.sandbox_id,
                        component="recovery",
                        event_type=event.event_type,
                        checkpoint_id=checkpoint_id,
                    ),
                )
                try:
                    restore_result = self.restore_once(event.sandbox_id, checkpoint_id)
                except Exception:
                    restore_operation.finish(
                        status="failed",
                        attributes={"checkpoint_id": str(checkpoint_id)},
                    )
                    raise
                restore_operation.finish(
                    status=restore_result.status.value,
                    attributes={
                        "checkpoint_id": str(checkpoint_id),
                        "failure_code": restore_result.failure_code.value,
                        "job_id": str(restore_result.job_id),
                    },
                )
                if restore_result.status.value == "succeeded":
                    self._release_checkpoint_response_gate(
                        event.sandbox_id,
                        checkpoint_id,
                        manifest=restore_manifest,
                    )
                    status = "restored"
                    logger.info(
                        "Recovery restore succeeded sandbox=%s checkpoint=%s",
                        event.sandbox_id,
                        checkpoint_id,
                    )
                elif (
                    self.relaunch_handler is not None
                    and self.relaunch_on_restore_failure
                ):
                    logger.warning(
                        "Recovery restore failed; invoking relaunch handler sandbox=%s checkpoint=%s message=%s",
                        event.sandbox_id,
                        checkpoint_id,
                        restore_result.message,
                    )
                    self.relaunch_handler(
                        event.sandbox_id,
                        event.event_type,
                        True,
                    )
                    status = "relaunched"
                    message = "restore_failed_relaunch_handler_invoked"
                else:
                    # Restore failed and either no relaunch handler is wired or
                    # the relaunch fallback is opted out via
                    # relaunch_on_restore_failure=False (the default). Surface
                    # this as a hard error so latent bugs in checkpoint capture
                    # / restore are not masked by a silent relaunch.
                    logger.error(
                        "Recovery restore failed sandbox=%s checkpoint=%s message=%s "
                        "(set relaunch_on_restore_failure=True to fall back to relaunch_handler)",
                        event.sandbox_id,
                        checkpoint_id,
                        restore_result.message,
                    )
                    raise RuntimeError(
                        f"recovery restore failed sandbox={event.sandbox_id} "
                        f"checkpoint={checkpoint_id} message={restore_result.message}"
                    )
            elif (
                self.relaunch_handler is not None
                and self.relaunch_on_restore_failure
            ):
                logger.info("No checkpoint available; invoking relaunch handler for sandbox=%s", event.sandbox_id)
                self.relaunch_handler(
                    event.sandbox_id,
                    event.event_type,
                    False,
                )
                status = "relaunched"
                message = "relaunch_handler_invoked"
            else:
                status = "no_checkpoint"
                message = "no restorable checkpoint available"
                logger.warning(
                    "No checkpoint available for sandbox=%s and relaunch fallback is disabled "
                    "(set relaunch_on_restore_failure=True to fall back to relaunch_handler)",
                    event.sandbox_id,
                )
            if event.event_type == "preemption":
                self._clear_snapshot_metadata(
                    event.sandbox_id,
                    "preemption_notice",
                    "preemption_grace_remaining_seconds",
                )
        except Exception as exc:
            logger.exception("Recovery handling failed for sandbox %s event=%s", event.sandbox_id, event.event_type)
            status = "failed"
            message = str(exc)
        finally:
            finished = utc_now()
            logger.info(
                "Finished recovery event sandbox=%s event_type=%s status=%s checkpoint=%s message=%s",
                event.sandbox_id,
                event.event_type,
                status,
                "" if checkpoint_id is None else checkpoint_id,
                "" if message is None else message,
            )
            record = RecoveryRecord(
                sandbox_id=event.sandbox_id,
                event_type=event.event_type,
                started_at=started,
                finished_at=finished,
                status=status,
                checkpoint_id=checkpoint_id,
                message=message,
            )
            with self._recovery_lock:
                self._recovery_records[event.sandbox_id] = record
            self.telemetry.emit_event(
                "recovery.finished",
                {
                    "sandbox_id": str(event.sandbox_id),
                    "event_type": event.event_type,
                    "status": status,
                    "checkpoint_id": "" if checkpoint_id is None else str(checkpoint_id),
                },
            )
            operation.finish(
                status=status,
                attributes={"checkpoint_id": "" if checkpoint_id is None else str(checkpoint_id)},
            )
            self._release_coordination(event.sandbox_id)
            self._refresh_interceptor_pending_state(event.sandbox_id)
            if self._should_coordinate_any_pending_request(event.sandbox_id):
                self._dispatch_coordination(event.sandbox_id)
            if pinned_restore_ids:
                self._unpin_restore_checkpoints(event.sandbox_id, pinned_restore_ids)

    def _drain_active_runtime_execs(self, sandbox_id: SandboxId) -> None:
        """Ask the runtime to terminate any in-flight `runc exec`
        subprocesses for `sandbox_id`. Called before the spot
        preemption checkpoint flow so CRIU doesn't trip over half-
        stream unix-socket connections wired by runc-exec stdio."""
        cancel = getattr(self.runtime, "cancel_active_execs", None)
        if not callable(cancel):
            return
        try:
            cancelled = cancel(sandbox_id, timeout_s=2.0)
        except Exception:
            logger.debug(
                "Failed to drain active runtime execs sandbox=%s",
                sandbox_id,
                exc_info=True,
            )
            return
        if cancelled:
            logger.info(
                "Drained %d in-flight runtime exec(s) before preemption checkpoint sandbox=%s",
                cancelled,
                sandbox_id,
            )

    def _execute_checkpoint_flow(
        self,
        sandbox_id: SandboxId,
        *,
        pending_request: PendingSandboxResponse | None = None,
    ) -> CheckpointResult | None:
        operation = start_operation(
            self.telemetry,
            "checkpoint.flow",
            self._telemetry_attrs(
                sandbox_id,
                component="system",
                request_id=None if pending_request is None else pending_request.request_id,
                extra=(
                    None
                    if pending_request is None
                    else {"request_generation": pending_request.generation}
                ),
            ),
        )
        decision = self.scheduler.query_checkpoint(sandbox_id)
        if not decision.should_checkpoint:
            operation.finish(status="skipped", attributes={"reason": decision.reason})
            return None
        checkpoint_metadata = self._build_checkpoint_metadata(sandbox_id, pending_request=pending_request)

        job = CheckpointJob(
            job_id=JobId.new(),
            sandbox_id=sandbox_id,
            requested_at=utc_now(),
            reason=decision.reason,
            checkpoint_process=decision.checkpoint_process,
            checkpoint_filesystem=decision.checkpoint_filesystem,
            leave_running=decision.leave_running,
            is_incremental_process=decision.is_incremental_process,
            parent_process_checkpoint_id=decision.parent_process_checkpoint_id,
            produce_pre_dump=decision.produce_pre_dump,
            metadata={"policy": decision.policy_name, **decision.metadata, **checkpoint_metadata},
        )
        result: CheckpointResult | None = None
        flow_exception: BaseException | None = None
        try:
            result = self.executor.submit_checkpoint(job).result()
            if result.status.value == "succeeded":
                self.scheduler.mark_checkpoint_complete(
                    sandbox_id,
                    result.finished_at,
                    process_checkpoint_id=(
                        result.checkpoint_id if job.checkpoint_process else None
                    ),
                    is_incremental_process=job.is_incremental_process,
                )
                self.inspector.mark_checkpoint_complete(
                    sandbox_id,
                    process=job.checkpoint_process,
                    filesystem=job.checkpoint_filesystem,
                    at=result.finished_at,
                )
            return result
        except BaseException as exc:
            flow_exception = exc
            raise
        finally:
            # Telemetry must be symmetric: every flow.start needs a
            # matching flow.finish, even if post-checkpoint bookkeeping
            # (e.g. daemon-side reset) raised. An orphan flow.start is
            # counted as a failure by the report tooling.
            if result is not None:
                operation.finish(
                    status=result.status.value,
                    attributes={
                        "checkpoint_id": str(result.checkpoint_id),
                        "reason": decision.reason,
                        "job_id": str(job.job_id),
                        "checkpoint_scope": _checkpoint_scope(job),
                        "failure_code": result.failure_code.value,
                    },
                )
            else:
                operation.finish(
                    status="failed",
                    attributes={
                        "reason": decision.reason,
                        "job_id": str(job.job_id),
                        "checkpoint_scope": _checkpoint_scope(job),
                        "error": type(flow_exception).__name__ if flow_exception is not None else "unknown",
                    },
                )
            if self._should_resume_after_checkpoint(job, result):
                self._resume_sandbox(sandbox_id)

    def _latest_checkpoint_id(self, sandbox_id: SandboxId):
        checkpoints = self.storage.list_checkpoints(sandbox_id)
        if not checkpoints:
            return None
        return checkpoints[-1]

    def _build_checkpoint_metadata(
        self,
        sandbox_id: SandboxId,
        *,
        pending_request: PendingSandboxResponse | None = None,
    ) -> dict[str, object]:
        metadata: dict[str, object] = {_CAPTURES_INFLIGHT_LLM: False}
        if self.extra_checkpoint_metadata_provider is not None:
            try:
                metadata.update(self.extra_checkpoint_metadata_provider(sandbox_id))
            except Exception:
                logger.exception("Failed to collect extra checkpoint metadata for sandbox=%s", sandbox_id)
        if self.request_state_store is None or self.response_gate_registry is None:
            return metadata
        pending = pending_request or self.response_gate_registry.get_oldest_pending(sandbox_id)
        if pending is None:
            return metadata
        request_context = self.request_state_store.get_request_context(sandbox_id, pending.request_id)
        if request_context is None:
            return metadata
        if not bool(request_context.metadata.get("response_gate_enabled", True)):
            logger.debug(
                "Skipping live-request checkpoint capture for auxiliary request sandbox=%s request_id=%s kind=%s",
                sandbox_id,
                pending.request_id,
                request_context.metadata.get("request_kind"),
            )
            return metadata
        metadata[_CAPTURES_INFLIGHT_LLM] = True
        metadata[_CAPTURED_REQUEST_ID] = pending.request_id
        metadata[_CAPTURED_REQUEST_GENERATION] = pending.generation
        provider = request_context.metadata.get("provider")
        if provider is not None:
            metadata[_CAPTURED_REQUEST_PROVIDER] = str(provider)
        metadata[_CAPTURED_REQUEST_STARTED_AT] = request_context.started_at.isoformat()
        if pending.request_ids:
            metadata[_CAPTURED_REQUEST_IDS] = list(pending.request_ids)
        if pending.request_group_kind and pending.request_group_id:
            metadata[_CAPTURED_REQUEST_GROUP_KIND] = pending.request_group_kind
            metadata[_CAPTURED_REQUEST_GROUP_ID] = pending.request_group_id
            contexts = self.request_state_store.get_request_contexts_for_group(
                sandbox_id,
                request_group_kind=pending.request_group_kind,
                request_group_id=pending.request_group_id,
            )
            if contexts:
                group_started_at = min(context.started_at for context in contexts)
                metadata[_CAPTURED_REQUEST_GROUP_STARTED_AT] = group_started_at.isoformat()
                metadata[_CAPTURED_REQUEST_IDS] = [context.request_id for context in contexts]
        logger.info(
            "Checkpoint captured live request sandbox=%s request_id=%s generation=%s group_kind=%s group_id=%s",
            sandbox_id,
            pending.request_id,
            pending.generation,
            pending.request_group_kind,
            pending.request_group_id,
        )
        self.telemetry.emit_event(
            "checkpoint.captured_live_request",
            {
                "sandbox_id": str(sandbox_id),
                "request_id": pending.request_id,
                "request_generation": pending.generation,
                "request_group_kind": pending.request_group_kind or "",
                "request_group_id": pending.request_group_id or "",
            },
        )
        return metadata

    def _validate_restore_checkpoint(self, sandbox_id: SandboxId, checkpoint_id: CheckpointId) -> str | None:
        manifest = self._resolve_restore_manifest(sandbox_id, checkpoint_id)
        if not bool(manifest.metadata.get(_CAPTURES_INFLIGHT_LLM, False)):
            return None
        captured_request_id = str(manifest.metadata.get(_CAPTURED_REQUEST_ID, "")).strip()
        captured_group_kind = str(manifest.metadata.get(_CAPTURED_REQUEST_GROUP_KIND, "")).strip()
        captured_group_id = str(manifest.metadata.get(_CAPTURED_REQUEST_GROUP_ID, "")).strip()
        if captured_group_kind and captured_group_id:
            pending = self._find_captured_pending_request(sandbox_id, manifest)
            if pending is None:
                return (
                    f"checkpoint {checkpoint_id} captured live request group "
                    f"{captured_group_kind}:{captured_group_id} but no matching interceptor-held group is pending"
                )
            if pending.request_group_kind != captured_group_kind or pending.request_group_id != captured_group_id:
                return (
                    f"checkpoint {checkpoint_id} captured live request group "
                    f"{captured_group_kind}:{captured_group_id} but current pending group is "
                    f"{pending.request_group_kind}:{pending.request_group_id}"
                )
            return None
        if not captured_request_id:
            return f"checkpoint {checkpoint_id} advertises live-request restore without captured_request_id"
        pending = self._find_captured_pending_request(sandbox_id, manifest)
        if pending is None:
            return (
                f"checkpoint {checkpoint_id} captured live request {captured_request_id} "
                "but no matching interceptor-held request is pending"
            )
        if pending.request_id != captured_request_id:
            return (
                f"checkpoint {checkpoint_id} captured live request {captured_request_id} "
                f"but current pending request is {pending.request_id}"
            )
        return None

    def _select_recovery_checkpoint(self, sandbox_id: SandboxId) -> CheckpointId | None:
        checkpoints = list(reversed(self.storage.list_checkpoints(sandbox_id)))
        for checkpoint_id in checkpoints:
            validation_message = (
                self._validate_restore_checkpoint(sandbox_id, checkpoint_id)
                if self.enforce_restore_checkpoint_validation
                else None
            )
            if validation_message is None:
                return checkpoint_id
            manifest = self._resolve_restore_manifest(sandbox_id, checkpoint_id)
            if not bool(manifest.metadata.get(_CAPTURES_INFLIGHT_LLM, False)):
                logger.warning(
                    "Skipping checkpoint after validation failure sandbox=%s checkpoint=%s message=%s",
                    sandbox_id,
                    checkpoint_id,
                    validation_message,
                )
                continue
            logger.warning(
                "Skipping stale live-request checkpoint sandbox=%s checkpoint=%s message=%s",
                sandbox_id,
                checkpoint_id,
                validation_message,
            )
            self.telemetry.emit_event(
                "recovery.checkpoint_skipped_stale_request",
                {
                    "sandbox_id": str(sandbox_id),
                    "checkpoint_id": str(checkpoint_id),
                    "message": validation_message,
                },
            )
        logger.warning("No restorable checkpoint available for sandbox=%s", sandbox_id)
        self.telemetry.emit_event(
            "recovery.no_satisfiable_checkpoint",
            {"sandbox_id": str(sandbox_id)},
        )
        return None

    def _select_recovery_checkpoint_after(
        self,
        sandbox_id: SandboxId,
        *,
        observed_after,
    ) -> CheckpointId | None:
        checkpoints = list(reversed(self.storage.list_checkpoints(sandbox_id)))
        for checkpoint_id in checkpoints:
            manifest = self._resolve_restore_manifest(sandbox_id, checkpoint_id)
            if manifest.created_at < observed_after:
                break
            validation_message = (
                self._validate_restore_checkpoint(sandbox_id, checkpoint_id)
                if self.enforce_restore_checkpoint_validation
                else None
            )
            if validation_message is None:
                return checkpoint_id
            logger.warning(
                "Skipping recent checkpoint after validation failure sandbox=%s checkpoint=%s message=%s",
                sandbox_id,
                checkpoint_id,
                validation_message,
            )
        return None

    def _resolve_restore_manifest(self, sandbox_id: SandboxId, checkpoint_id: CheckpointId):
        manifest = self.storage.get_manifest(sandbox_id, checkpoint_id)
        return resolve_restore_manifest(self.storage, manifest)

    def _pin_restore_checkpoints(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> tuple[CheckpointManifest | None, list[CheckpointId]]:
        manifest = self._resolve_restore_manifest(sandbox_id, checkpoint_id)
        pin_checkpoint = getattr(self.storage, "pin_checkpoint", None)
        if not callable(pin_checkpoint):
            return manifest, []
        if not pin_checkpoint(sandbox_id, checkpoint_id):
            return None, []
        pinned_ids = [checkpoint_id]
        for metadata_key in ("process_restore_checkpoint_id", "filesystem_restore_checkpoint_id"):
            raw_value = manifest.metadata.get(metadata_key)
            if raw_value is None:
                continue
            candidate = CheckpointId(str(raw_value))
            if candidate in pinned_ids:
                continue
            if pin_checkpoint(sandbox_id, candidate):
                pinned_ids.append(candidate)
        return manifest, pinned_ids

    def _unpin_restore_checkpoints(self, sandbox_id: SandboxId, checkpoint_ids: list[CheckpointId]) -> None:
        unpin_checkpoint = getattr(self.storage, "unpin_checkpoint", None)
        if not callable(unpin_checkpoint):
            return
        for checkpoint_id in reversed(checkpoint_ids):
            unpin_checkpoint(sandbox_id, checkpoint_id)

    def _release_checkpoint_response_gate(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
        *,
        manifest: CheckpointManifest | None = None,
    ) -> bool:
        release_operation = start_operation(
            self.telemetry,
            "recovery.response_release",
            self._telemetry_attrs(sandbox_id, component="recovery", checkpoint_id=checkpoint_id),
        )
        if self.response_gate_registry is None:
            release_operation.finish(status="skipped")
            return False
        if manifest is None:
            manifest = self._resolve_restore_manifest(sandbox_id, checkpoint_id)
        if not bool(manifest.metadata.get(_CAPTURES_INFLIGHT_LLM, False)):
            release_operation.finish(status="skipped")
            return False
        captured_group_kind = str(manifest.metadata.get(_CAPTURED_REQUEST_GROUP_KIND, "")).strip()
        captured_group_id = str(manifest.metadata.get(_CAPTURED_REQUEST_GROUP_ID, "")).strip()
        captured_request_id = str(manifest.metadata.get(_CAPTURED_REQUEST_ID, "")).strip()
        if not captured_request_id:
            release_operation.finish(status="skipped")
            return False
        request_ids = manifest.metadata.get(_CAPTURED_REQUEST_IDS, [])
        if not isinstance(request_ids, list) or not request_ids:
            request_ids = [captured_request_id]
        for request_id in request_ids:
            self.executor.clear_live_response_ready(sandbox_id, str(request_id))
        pending = self._find_captured_pending_request(sandbox_id, manifest)
        if pending is None:
            release_operation.finish(status="skipped")
            return False
        if captured_group_kind and captured_group_id:
            released = self.response_gate_registry.release_pending(
                sandbox_id,
                request_id=pending.request_id,
                generation=pending.generation,
            )
        else:
            released = self.response_gate_registry.release_pending(
                sandbox_id,
                request_id=captured_request_id,
                generation=pending.generation,
            )
        if released:
            logger.info(
                "Released buffered response to restored sandbox=%s request_id=%s checkpoint=%s",
                sandbox_id,
                captured_request_id,
                checkpoint_id,
            )
            self.telemetry.emit_event(
                "recovery.response_released",
                {
                    "sandbox_id": str(sandbox_id),
                    "request_id": captured_request_id,
                    "checkpoint_id": str(checkpoint_id),
                },
            )
        release_operation.finish(
            status="succeeded" if released else "failed",
            attributes={"request_id": captured_request_id},
        )
        return released

    def _merge_snapshot_metadata(self, sandbox_id: SandboxId, **metadata: object) -> None:
        upsert = getattr(self.inspector, "upsert_snapshot", None)
        if upsert is None:
            return
        snapshot = self.inspector.inspect(sandbox_id)
        upsert(
            snapshot.__class__(
                sandbox_id=snapshot.sandbox_id,
                runtime_name=snapshot.runtime_name,
                is_running=snapshot.is_running,
                process_changed=snapshot.process_changed,
                filesystem_changed=snapshot.filesystem_changed,
                observed_at=utc_now(),
                last_checkpoint_at=snapshot.last_checkpoint_at,
                metadata={**snapshot.metadata, **metadata},
            )
        )

    def _clear_snapshot_metadata(self, sandbox_id: SandboxId, *keys: str) -> None:
        upsert = getattr(self.inspector, "upsert_snapshot", None)
        if upsert is None:
            return
        snapshot = self.inspector.inspect(sandbox_id)
        metadata = dict(snapshot.metadata)
        for key in keys:
            metadata.pop(key, None)
        upsert(
            snapshot.__class__(
                sandbox_id=snapshot.sandbox_id,
                runtime_name=snapshot.runtime_name,
                is_running=snapshot.is_running,
                process_changed=snapshot.process_changed,
                filesystem_changed=snapshot.filesystem_changed,
                observed_at=utc_now(),
                last_checkpoint_at=snapshot.last_checkpoint_at,
                metadata=metadata,
            )
        )

    def _acquire_coordination(self, sandbox_id: SandboxId) -> bool:
        while not self._stop_event.is_set():
            with self._coordination_lock:
                if sandbox_id not in self._active_coordination:
                    self._active_coordination.add(sandbox_id)
                    return True
            self._stop_event.wait(0.05)
        return False

    def _release_coordination(self, sandbox_id: SandboxId) -> None:
        with self._coordination_lock:
            self._active_coordination.discard(sandbox_id)

    def _pause_for_manual_checkpoint(self, sandbox_id: SandboxId) -> bool:
        try:
            self.runtime.pause(sandbox_id)
            return True
        except Exception:
            logger.exception("Failed to pause sandbox %s for manual checkpoint", sandbox_id)
            return False

    def _resume_sandbox(self, sandbox_id: SandboxId) -> None:
        try:
            description = self.runtime.describe(sandbox_id)
        except Exception:
            return
        if description.status != "paused":
            return
        try:
            self.runtime.resume(sandbox_id)
        except Exception:
            logger.exception("Failed to resume sandbox %s", sandbox_id)
            return
        self._mark_sandbox_running(sandbox_id)

    def quiesce_for_verification(
        self,
        sandbox_id: SandboxId,
        *,
        drain_timeout_seconds: float = 120.0,
        poll_interval_seconds: float = 0.1,
    ) -> None:
        # Terminal transition from run phase to verification phase. Enforces:
        #   1. scheduler no longer issues checkpoint decisions for this sandbox
        #   2. no executor jobs remain pending or running
        #   3. the container is not paused
        # Any of these held at verify time produce the "cannot exec in a paused
        # container" race we saw in 20260420_123846 spec-91-spec-81.
        self.scheduler.deactivate_sandbox(sandbox_id)
        deadline = time.monotonic() + max(0.0, float(drain_timeout_seconds))
        while self.executor.has_active_job(sandbox_id):
            if time.monotonic() >= deadline:
                logger.warning(
                    "quiesce_for_verification timed out draining executor for sandbox %s; proceeding",
                    sandbox_id,
                )
                break
            time.sleep(poll_interval_seconds)
        self._resume_sandbox(sandbox_id)

    def _mark_sandbox_not_running(self, sandbox_id: SandboxId) -> None:
        upsert = getattr(self.inspector, "upsert_snapshot", None)
        if upsert is not None:
            try:
                snapshot = self.inspector.inspect(sandbox_id)
            except Exception:
                snapshot = None
            if snapshot is not None and snapshot.is_running:
                upsert(
                    replace(
                        snapshot,
                        is_running=False,
                        observed_at=utc_now(),
                    )
                )
        try:
            self.runtime.sync_runtime_state(sandbox_id, is_running=False)
        except Exception:
            logger.debug("Failed to sync runtime state for faulted sandbox %s", sandbox_id, exc_info=True)

    def _mark_sandbox_running(self, sandbox_id: SandboxId) -> None:
        upsert = getattr(self.inspector, "upsert_snapshot", None)
        if upsert is not None:
            try:
                snapshot = self.inspector.inspect(sandbox_id)
            except Exception:
                snapshot = None
            if snapshot is not None:
                upsert(
                    replace(
                        snapshot,
                        is_running=True,
                        observed_at=utc_now(),
                    )
                )
        try:
            self.runtime.sync_runtime_state(sandbox_id, is_running=True)
        except Exception:
            logger.debug("Failed to sync runtime state for running sandbox %s", sandbox_id, exc_info=True)

    def _release_response_gate(
        self,
        sandbox_id: SandboxId,
        pending_request: PendingSandboxResponse | None = None,
    ) -> None:
        if pending_request is None or not pending_request.request_ids:
            self.executor.clear_live_response_ready(
                sandbox_id,
                None if pending_request is None else pending_request.request_id,
            )
        else:
            for request_id in pending_request.request_ids:
                self.executor.clear_live_response_ready(sandbox_id, request_id)
        if self.response_gate_registry is None:
            return
        if pending_request is None:
            self.response_gate_registry.release(sandbox_id)
            return
        self.response_gate_registry.release_pending(
            sandbox_id,
            request_id=pending_request.request_id,
            generation=pending_request.generation,
        )

    def _dispatch_pending_coordination(self) -> None:
        with self._interceptor_lock:
            sandbox_ids = list(self._interceptor_pending)
        for sandbox_id in sandbox_ids:
            if self._should_coordinate_any_pending_request(sandbox_id):
                self._dispatch_coordination(sandbox_id)
                continue
            if not self._has_pending_response_gate(sandbox_id):
                self._refresh_interceptor_pending_state(sandbox_id)

    def _should_coordinate_any_pending_request(self, sandbox_id: SandboxId) -> bool:
        if self.request_state_store is None or self.response_gate_registry is None:
            return False
        if not self.request_state_store.get(sandbox_id).llm_request_in_flight:
            return False
        return self.response_gate_registry.get_oldest_pending(sandbox_id) is not None

    def _next_pending_live_request(self, sandbox_id: SandboxId) -> PendingSandboxResponse | None:
        if self.response_gate_registry is None:
            return None
        return self.response_gate_registry.get_oldest_pending(sandbox_id)

    def _has_pending_response_gate(self, sandbox_id: SandboxId) -> bool:
        return self.response_gate_registry is not None and self.response_gate_registry.get_pending(sandbox_id) is not None

    def _refresh_interceptor_pending_state(self, sandbox_id: SandboxId) -> None:
        with self._interceptor_lock:
            if self._has_pending_response_gate(sandbox_id):
                self._interceptor_pending.add(sandbox_id)
            else:
                self._interceptor_pending.discard(sandbox_id)

    def _find_captured_pending_request(
        self,
        sandbox_id: SandboxId,
        manifest: CheckpointManifest,
    ) -> PendingSandboxResponse | None:
        if self.response_gate_registry is None:
            return None
        captured_group_kind = str(manifest.metadata.get(_CAPTURED_REQUEST_GROUP_KIND, "")).strip()
        captured_group_id = str(manifest.metadata.get(_CAPTURED_REQUEST_GROUP_ID, "")).strip()
        if captured_group_kind and captured_group_id:
            return self.response_gate_registry.find_pending_group(
                sandbox_id,
                request_group_kind=captured_group_kind,
                request_group_id=captured_group_id,
            )
        captured_request_id = str(manifest.metadata.get(_CAPTURED_REQUEST_ID, "")).strip()
        if not captured_request_id:
            return None
        raw_generation = manifest.metadata.get(_CAPTURED_REQUEST_GENERATION)
        if raw_generation is not None:
            try:
                generation = int(raw_generation)
            except (TypeError, ValueError):
                generation = None
            else:
                return self.response_gate_registry.get_pending_generation(sandbox_id, generation)
        return self.response_gate_registry.find_pending_request(sandbox_id, captured_request_id)

    def _should_resume_after_checkpoint(
        self,
        job: CheckpointJob | None,
        result: CheckpointResult | None,
    ) -> bool:
        if job is None:
            return True
        if result is None or result.status.value != "succeeded":
            return True
        if isinstance(self.runtime, InMemoryRuntime):
            return True
        return job.leave_running


def build_default_system(
    *,
    storage_root: str | Path,
    runtime: str = "runc",
    scheduler_config: SchedulerConfig | None = None,
    executor_config: ExecutorConfig | None = None,
    storage_config: StorageConfig | None = None,
    runc_runtime_options: RuncRuntimeOptions | None = None,
    use_in_memory_telemetry: bool = True,
    telemetry_config: TelemetryConfig | None = None,
    request_state_store: InMemoryRequestStateStore | None = None,
    host_inspector_url: str | None = None,
    scheduler_policy: SchedulerPolicy | None = None,
    checkpoint_manager: CheckpointManager | None = None,
    relaunch_handler: Callable[[SandboxId, str, bool], None] | None = None,
    enforce_restore_checkpoint_validation: bool = False,
    relaunch_on_restore_failure: bool = False,
) -> CrabSystem:
    logger.info("Building default crab system with runtime=%s storage_root=%s", runtime, storage_root)
    scheduler_cfg = scheduler_config or SchedulerConfig()
    executor_cfg = executor_config or ExecutorConfig()
    store_cfg = storage_config or StorageConfig(root_dir=Path(storage_root))

    host_inspector_client = (
        HostInspectorServiceClient(host_inspector_url) if host_inspector_url is not None else None
    )

    telemetry_cfg = telemetry_config or TelemetryConfig(enabled=True)
    telemetry = build_configured_telemetry_sink(
        telemetry_cfg,
        keep_in_memory_fallback=use_in_memory_telemetry,
    )

    if runtime == "docker":
        runtime_impl = InMemoryRuntime(name="docker", host_inspector_client=host_inspector_client)
    elif runtime == "runc":
        runtime_impl = RuncRuntime(
            host_inspector_client=host_inspector_client,
            telemetry=telemetry,
            options=runc_runtime_options,
        )
    else:
        raise ValueError(f"unsupported runtime: {runtime}")
    storage = checkpoint_manager or LocalCheckpointManager(
        store_cfg,
        runtime_image_path_in_use=runtime_impl.runtime_image_path_in_use,
    )
    if checkpoint_manager is None:
        # When the caller supplied their own manager (likely wrapped in a
        # retention policy), late-bind the safety predicate so it can
        # still defer pruning a runtime tree with an active lazy-pages
        # daemon. Caller-supplied managers may not be ``LocalCheckpointManager``;
        # ``setattr``-style installation would be wrong on those, so we
        # only call the setter when the manager exposes it.
        pass
    setter = getattr(storage, "set_runtime_image_path_in_use", None)
    if callable(setter):
        setter(runtime_impl.runtime_image_path_in_use)
    request_store = request_state_store or InMemoryRequestStateStore()
    response_gate_registry = SandboxResponseGateRegistry()
    base_inspector: SandboxInspector
    if host_inspector_client is not None:
        base_inspector = RemoteSandboxInspector(host_inspector_client)
    else:
        base_inspector = EBPFSandboxInspector()
    inspector = RequestAwareSandboxInspector(base_inspector, request_store)

    process_c = AdapterProcessCWorker(runtime_impl)
    process_r = AdapterProcessRWorker(runtime_impl)
    fs_c = AdapterFileSystemCWorker(runtime_impl)
    fs_r = AdapterFileSystemRWorker(runtime_impl)

    c_worker = DefaultCWorker(
        process_c,
        fs_c,
        storage,
        runtime_impl,
        checkpoint_guard=_checkpoint_guard_from_inspector(inspector),
        telemetry=telemetry,
        step_workers=executor_cfg.resolved_composite_step_workers,
    )
    r_worker = DefaultRWorker(
        process_r, fs_r, storage, telemetry=telemetry, runtime=runtime_impl
    )

    executor = CRExecutor(executor_cfg, c_worker, r_worker, telemetry)
    scheduler = CRScheduler(
        scheduler_cfg,
        inspector,
        runtime_impl,
        InMemorySchedulerStateStore(),
        telemetry,
        scheduler_policy,
    )

    logger.debug(
        "Constructed crab components runtime=%s telemetry=%s",
        runtime,
        type(telemetry).__name__,
    )
    return CrabSystem(
        scheduler=scheduler,
        executor=executor,
        storage=storage,
        inspector=inspector,
        runtime=runtime_impl,
        telemetry=telemetry,
        request_state_store=request_store,
        response_gate_registry=response_gate_registry,
        relaunch_handler=relaunch_handler,
        recovery_delay_seconds=0.0,
        enforce_restore_checkpoint_validation=enforce_restore_checkpoint_validation,
        relaunch_on_restore_failure=relaunch_on_restore_failure,
    )
