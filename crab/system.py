from __future__ import annotations

from dataclasses import dataclass, field, replace
from concurrent.futures import Future, ThreadPoolExecutor
import logging
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Lock, Thread
import time
import uuid
from typing import Callable, Protocol

from .config import ExecutorConfig, SchedulerConfig, StorageConfig, TelemetryConfig
from .contracts import CheckpointManager, Runtime, SandboxInspector, TelemetrySink
from .executor import CRExecutor
from .ids import CheckpointId, JobId
from .inspector import EBPFSandboxInspector
from .remote_inspector import HostInspectorServiceClient, RemoteSandboxInspector
from .interceptor import InMemoryRequestStateStore, RequestAwareSandboxInspector, SandboxResponseGateRegistry
from .models import (
    ArtifactPayload,
    ChangesetEntry,
    ChangesetResult,
    CheckpointManifest,
    CheckpointJob,
    CheckpointResult,
    FailureCode,
    JobStatus,
    MergeReport,
    EgressFlow,
    EgressLedger,
    ObservationReport,
    EgressReplayReport,
    ProcessMergeReport,
    RecoveryEvent,
    RecoveryRecord,
    RestoreJob,
    RestoreResult,
    SandboxId,
    SandboxSnapshot,
    SchedulerCheckpointDecision,
    utc_now,
)
from . import forking
from .effects import (
    DEFAULT_MAX_QUEUE_BYTES,
    DEFAULT_MAX_QUEUE_ENTRIES,
    EFFECT_POLICIES,
    EffectRule,
    flush_deferred_request,
)
from .egress import classify_flow
from .journal import ActionJournal
from .process_merge import (
    PROCESS_MERGE_STRATEGIES,
    PROCESS_PROBE_ARGV,
    PROCESS_PROBE_BASELINE,
    ProcessMergeConflict,
    replay_fork_execs,
)
from .merging import (
    DEFAULT_MERGE_IGNORE_PREFIXES,
    MERGE_POLICIES,
    MergeApplyError,
    MergeError,
    MergerHook,
    apply_plan,
    build_report,
    plan_merge,
)
from .txn import (
    EffectFlushReport,
    FlushedEffect,
    TxnAbortError,
    TxnAbortResult,
    TxnActiveError,
    TxnCommitConflict,
    TxnCommitResult,
    TxnDescription,
    TxnError,
    TxnMismatchError,
    TxnNotAbortable,
    new_txn_id,
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

# C3 observation consolidation: how a fork's journal history is adopted
# into its source's journal. `SummarizerHook` receives the qualifying
# fork records (`ActionRecord.to_json()` rows) and returns a digest
# (str/dict) or None; local engine only — never crosses the daemon RPC.
OBSERVATION_POLICIES: tuple[str, ...] = ("append", "dedupe", "none")
SummarizerHook = Callable[[list], object]

# C4 promotion policies for the reverse fs apply (source changes onto
# the fork before it takes over); direction-neutral names mapped onto
# the C2 engine to avoid "prefer_fork" ambiguity in this context.
PROMOTION_POLICIES: dict[str, str] = {
    "fail_fast": "fail_fast",
    "prefer_incoming": "prefer_fork",
    "prefer_existing": "prefer_source",
    "text_merge": "text_merge",
}


class ForkPromotionError(RuntimeError):
    """The fork->source identity swap failed mid-flight. On the no-transfer
    path the fork is still running and the operation is retryable against
    it; on the netns-transfer path the fork is dumped-and-stopped but its
    filesystem is intact, so recovery is a restore of the promoted
    checkpoint. Either way the fork's dataset is retained."""


class _LeaseTransfer(Protocol):
    """Engine-provided hook moving a fork's network identity onto the
    source during promotion. ``probe=True`` answers "would a transfer
    happen?" without mutating (used before the fork is dumped); the real
    call performs the move and returns whether it happened."""

    def __call__(
        self, from_sandbox_id: SandboxId, to_sandbox_id: SandboxId, *, probe: bool = ...
    ) -> bool: ...

_CAPTURES_INFLIGHT_LLM = "captures_inflight_llm"
_CAPTURED_REQUEST_ID = "captured_request_id"
_CAPTURED_REQUEST_GENERATION = "captured_request_generation"
_CAPTURED_REQUEST_PROVIDER = "captured_request_provider"
_CAPTURED_REQUEST_STARTED_AT = "captured_request_started_at"
_CAPTURED_REQUEST_GROUP_KIND = "captured_request_group_kind"
_CAPTURED_REQUEST_GROUP_ID = "captured_request_group_id"
_CAPTURED_REQUEST_IDS = "captured_request_ids"
_CAPTURED_REQUEST_GROUP_STARTED_AT = "captured_request_group_started_at"
_LOGICAL_CHECKPOINT = "logical_checkpoint"
_LOGICAL_CHECKPOINT_ID = "logical_checkpoint_id"
_CHECKPOINT_MATERIALIZATION = "checkpoint_materialization"
_PHYSICAL_CHECKPOINT_CREATED = "physical_checkpoint_created"
_PROCESS_RESTORE_CHECKPOINT_ID = "process_restore_checkpoint_id"
_FILESYSTEM_RESTORE_CHECKPOINT_ID = "filesystem_restore_checkpoint_id"
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
    journal: ActionJournal | None = None
    egress_rules: tuple = ()
    """Host-scoped egress classification overrides (D1). Applied when the
    ledger is read, so changing them re-classifies history too — the
    stored row keeps the class it was recorded with, the view reflects
    the current rules."""
    cassette_store: object | None = None
    """Recorded egress bodies (D2), assigned by the engine when recording
    is enabled. Pruned per sandbox on destroy, so a fork's cassettes must
    be replayed before the fork is killed."""
    cassette_replayer: object | None = None
    """Serves recorded reads during a replay window (D2), assigned by the
    engine alongside the proxy."""
    effect_gate: object | None = None
    """Effect policy for in-transaction egress (D3), assigned by the
    engine alongside the proxy. Sessions are opened/closed by the txn
    lifecycle (PR-D3.2); without a session the gate is inert and every
    flow behaves as it did before D3."""
    effect_policy_defaults: dict = field(default_factory=dict)
    """Configured defaults for effect sessions (default/fork policy,
    on_unlisted, opaque_effects, rules)."""
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
    _fork_lock: Lock = field(init=False, repr=False)
    _fork_chain_pins: dict[SandboxId, tuple[SandboxId, CheckpointId]] = field(init=False, repr=False)
    _fork_children: dict[SandboxId, set[SandboxId]] = field(init=False, repr=False)
    _current_restore_lock: Lock = field(init=False, repr=False)
    _current_restore_sources: dict[
        SandboxId,
        tuple[CheckpointId, CheckpointId | None, CheckpointId | None],
    ] = field(init=False, repr=False)
    _txn_lock: Lock = field(init=False, repr=False)
    _active_txns: dict[SandboxId, TxnDescription | None] = field(init=False, repr=False)

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
        self._fork_lock = Lock()
        self._fork_chain_pins = {}
        self._fork_children = {}
        self._current_restore_lock = Lock()
        self._current_restore_sources = {}
        self._txn_lock = Lock()
        self._active_txns = {}
        self._merge_lock = Lock()
        self._active_merges: set[SandboxId] = set()
        # Fork-backed txn hooks (B3): forking needs engine-level lease
        # allocation + bundle replication + restore; teardown mirrors the
        # SDK kill path. Registered by the Engine after construction.
        self._fork_txn_fork: Callable[[SandboxId], SandboxId] | None = None
        self._fork_txn_destroy: Callable[[SandboxId], None] | None = None
        self._fork_txn_lease_repair: Callable[[SandboxId], None] | None = None
        # Promotion moves the fork's network identity onto the source so
        # CRIU can bind its sockets back (see _promote_fork_onto_source).
        self._fork_txn_lease_transfer: "_LeaseTransfer | None" = None

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

    def _journal_lifecycle(
        self,
        sandbox_id: SandboxId,
        event: str,
        *,
        metadata: dict[str, object] | None = None,
    ) -> None:
        """Best-effort lifecycle marker in the action journal (B1)."""
        journal = self.journal
        if journal is None:
            return
        try:
            journal.record_lifecycle(sandbox_id, event, metadata=metadata)
        except Exception:
            logger.exception(
                "Journal lifecycle record failed sandbox=%s event=%s", sandbox_id, event
            )

    def checkpoint_once(self, sandbox_id: SandboxId, leave_running: bool=False, *, checkpoint_id: str | None = None) -> CheckpointResult:
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
            checkpoint_metadata = self._build_checkpoint_metadata(sandbox_id, pending_request=pending_request)
            if checkpoint_id is not None:
                # Client pre-allocated the checkpoint id; the composite worker
                # reads it from job metadata instead of minting a fresh one.
                checkpoint_metadata["checkpoint_id"] = str(checkpoint_id)
            job = CheckpointJob(
                job_id=JobId.new(),
                sandbox_id=sandbox_id,
                requested_at=utc_now(),
                reason="manual",
                leave_running=leave_running,
                metadata=checkpoint_metadata,
            )
            result = self.executor.run_checkpoint(job)
            if result.status.value == "succeeded":
                self._remember_current_restore_sources(
                    sandbox_id,
                    result.checkpoint_id,
                    result.checkpoint_id,
                    result.checkpoint_id,
                )
                self.scheduler.mark_checkpoint_complete(
                    sandbox_id,
                    result.finished_at,
                    process_checkpoint_id=(
                        result.checkpoint_id if job.checkpoint_process else None
                    ),
                    is_incremental_process=job.is_incremental_process,
                )
                self._journal_lifecycle(
                    sandbox_id,
                    "checkpoint",
                    metadata={
                        "checkpoint_id": str(result.checkpoint_id),
                        "reason": "manual",
                        "leave_running": bool(leave_running),
                    },
                )
        finally:
            if paused and self._should_resume_after_checkpoint(job, result):
                self._resume_sandbox(sandbox_id)
            # Re-baseline the inspector AFTER the sandbox is resumed. CRIU dumps
            # the tasks with --leave-running while the container is paused; the
            # parasite teardown on resume dirties 1-3 residual soft-dirty pages
            # *after* the clear. Running the reset before resume (the old
            # ordering) latched those residual pages as a false
            # process_changed=True on an otherwise idle sandbox. Doing it here,
            # once the tasks are running again, clears the residue; a genuinely
            # busy process keeps re-dirtying and is still reported by the next
            # status() poll, so real activity is never masked.
            if (
                job is not None
                and result is not None
                and result.status.value == "succeeded"
            ):
                self.inspector.mark_checkpoint_complete(
                    sandbox_id,
                    process=job.checkpoint_process,
                    filesystem=job.checkpoint_filesystem,
                    at=result.finished_at,
                )
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

    def checkpoint_requested(
        self,
        sandbox_id: SandboxId,
        *,
        leave_running: bool = True,
        checkpoint_id: str | CheckpointId | None = None,
    ) -> CheckpointResult:
        """Create a user-visible logical checkpoint.

        The id is allocated before work starts and remains stable.  Inspector
        state determines whether this turn creates a full/partial physical
        checkpoint or persists a zero-artifact logical node that reuses the
        prior restore sources. Unlike automatic scheduling, a requested
        checkpoint is never skipped merely because a time interval has not
        elapsed: changed state must be materialized to keep the logical point
        exact.
        """
        logical_id = (
            CheckpointId.new()
            if checkpoint_id is None
            else CheckpointId(str(checkpoint_id))
        )
        pending_request = self._next_pending_live_request(sandbox_id)
        try:
            return self._execute_requested_checkpoint_flow(
                sandbox_id,
                logical_id=logical_id,
                leave_running=leave_running,
                pending_request=pending_request,
            )
        finally:
            self._release_response_gate(sandbox_id, pending_request)
            self._refresh_interceptor_pending_state(sandbox_id)

    def checkpoint_if_due(self, sandbox_id: SandboxId) -> CheckpointResult | None:
        if self._txn_active(sandbox_id):
            logger.debug("Skipping checkpoint-if-due; txn active sandbox=%s", sandbox_id)
            return None
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

    def restore_once(
        self,
        sandbox_id: SandboxId,
        checkpoint_id,
        *,
        restore_metadata: dict[str, object] | None = None,
    ) -> RestoreResult:
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
        # Resolve once before the runtime is mutated. Besides feeding replay
        # metadata, the exact component ids become the new *current* baseline
        # after a successful restore. Looking at the newest stored manifest at
        # that point would be wrong when the caller restored an older logical
        # recovery point.
        restore_manifest = self._resolve_restore_manifest(
            sandbox_id, restore_checkpoint_id
        )
        if self.restore_metadata_handler is not None:
            # Preload replay/router state before the restored process can
            # resume making requests. This avoids a narrow race where the
            # process issues its first post-restore request before the replay
            # service cursor is rewound to the checkpoint position.
            self.restore_metadata_handler(sandbox_id, restore_manifest)
        self.runtime.prepare_for_restore(sandbox_id)
        job = RestoreJob(
            job_id=JobId.new(),
            sandbox_id=sandbox_id,
            checkpoint_id=restore_checkpoint_id,
            requested_at=utc_now(),
            reason="manual",
            metadata=dict(restore_metadata or {}),
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
                restored_at = utc_now()
                process_source_raw = restore_manifest.metadata.get(
                    _PROCESS_RESTORE_CHECKPOINT_ID
                )
                filesystem_source_raw = restore_manifest.metadata.get(
                    _FILESYSTEM_RESTORE_CHECKPOINT_ID
                )
                if process_source_raw is None or filesystem_source_raw is None:
                    # Legacy/incomplete restores remain supported, but they
                    # cannot safely seed adaptive reuse. The next requested
                    # checkpoint therefore creates a full baseline.
                    self._mark_current_restore_sources_incomplete(
                        sandbox_id, restore_checkpoint_id
                    )
                else:
                    process_source_id = CheckpointId(str(process_source_raw))
                    filesystem_source_id = CheckpointId(
                        str(filesystem_source_raw)
                    )
                    self._remember_current_restore_sources(
                        sandbox_id,
                        restore_checkpoint_id,
                        process_source_id,
                        filesystem_source_id,
                    )
                    # A restore changes which historical process image the
                    # live sandbox descends from. Reset the incremental
                    # scheduler base to that exact source and its persisted
                    # depth; retaining a newer pre-restore parent creates a
                    # divergent CRIU chain, while resetting the depth to zero
                    # can exceed the configured maximum after a historical
                    # incremental restore.
                    process_chain_length = self._stored_process_chain_length(
                        sandbox_id,
                        process_source_id,
                    )
                    self.scheduler.mark_checkpoint_complete(
                        sandbox_id,
                        restored_at,
                        process_checkpoint_id=process_source_id,
                        is_incremental_process=process_chain_length > 0,
                        process_chain_length=process_chain_length,
                    )
                self.inspector.mark_checkpoint_complete(
                    sandbox_id,
                    process=True,
                    filesystem=True,
                    at=restored_at,
                )
                self.storage.handle_restore_complete(sandbox_id, result.checkpoint_id)
                self._journal_lifecycle(
                    sandbox_id,
                    "restore",
                    metadata={"checkpoint_id": str(restore_checkpoint_id)},
                )
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

    def fork_once(
        self,
        source_sandbox_id: SandboxId,
        target_sandbox_id: SandboxId,
        *,
        checkpoint_id: CheckpointId | None = None,
        target_rootfs_path: Path,
        bundle_root: Path | None = None,
        checkpoint_root: Path | None = None,
    ) -> forking.ForkResult:
        """Clone a source sandbox's checkpoint state onto a new sandbox id.

        Takes a fresh checkpoint when ``checkpoint_id`` is None, clones the
        filesystem via the runtime's provider, copies manifests/artifacts
        with path rewrites, and applies incremental chain sharing (pin +
        ancestor symlinks) when available. The fork is left *stopped*;
        callers restore it (Engine.fork_sandbox does, optionally lazily).
        Mechanics sunk from the benchmark harness's clone_checkpoint_to_fork.
        """
        paths = getattr(self.runtime, "paths", None)
        if bundle_root is None:
            bundle_root = None if paths is None else paths.bundle_root
        if checkpoint_root is None:
            checkpoint_root = None if paths is None else paths.checkpoint_root
        if bundle_root is None or checkpoint_root is None:
            raise ValueError("fork_once requires bundle_root/checkpoint_root (runtime exposes no paths)")

        operation = start_operation(
            self.telemetry,
            "fork.flow",
            self._telemetry_attrs(
                source_sandbox_id,
                component="system",
                extra={"target_sandbox_id": str(target_sandbox_id)},
            ),
        )
        try:
            if checkpoint_id is None:
                checkpoint_result = self.checkpoint_once(source_sandbox_id, leave_running=True)
                if checkpoint_result.status.value != "succeeded" or checkpoint_result.checkpoint_id is None:
                    raise RuntimeError(
                        f"fork checkpoint failed for sandbox {source_sandbox_id}: "
                        f"status={checkpoint_result.status.value}"
                    )
                checkpoint_id = checkpoint_result.checkpoint_id

            manifests = {
                cid: self.storage.get_manifest(source_sandbox_id, cid)
                for cid in self.storage.list_checkpoints(source_sandbox_id)
            }
            if checkpoint_id not in manifests:
                raise ValueError(f"checkpoint {checkpoint_id} not found for sandbox {source_sandbox_id}")
            checkpoint_order = list(manifests.keys())
            copy_plan = forking.resolve_checkpoint_copy_plan(checkpoint_order, manifests, checkpoint_id)
            process_checkpoint_id = next(
                copy_id
                for copy_id, copy_process, _ in reversed(copy_plan)
                if copy_process
            )
            filesystem_checkpoint_id = next(
                copy_id for copy_id, _, copy_filesystem in reversed(copy_plan) if copy_filesystem
            )

            target_rootfs_path.mkdir(parents=True, exist_ok=True)
            target_dataset = self.runtime.clone_filesystem_snapshot(
                source_sandbox_id,
                filesystem_checkpoint_id,
                target_sandbox_id,
                target_rootfs_path=target_rootfs_path,
            )
            inherited_launch_metadata: dict[str, object] = {}
            try:
                source_description = self.runtime.describe(source_sandbox_id)
                for key in (
                    "idle_timeout",
                    "idle_action",
                    "sdk_image",
                    "image_reference",
                    "image_id",
                    "image_digest",
                    "network_mode",
                    "network_requested",
                    "rootfs_preparation_schema",
                ):
                    if key in source_description.metadata:
                        inherited_launch_metadata[key] = source_description.metadata[key]
            except Exception:
                logger.debug(
                    "Unable to read source idle metadata while forking %s",
                    source_sandbox_id,
                    exc_info=True,
                )
            # Restore flows (prepare_for_restore/mark_restored) require a
            # runtime description; forks were never launched, so adopt one.
            self.runtime.adopt_sandbox_description(
                target_sandbox_id,
                runtime_name=self.runtime.name,
                status="stopped",
                metadata={
                    **inherited_launch_metadata,
                    "sandbox_id": str(target_sandbox_id),
                    "bundle_path": str(bundle_root / str(target_sandbox_id)),
                    "rootfs_path": str(target_rootfs_path),
                    "zfs_dataset": target_dataset,
                    "forked_from": str(source_sandbox_id),
                },
            )

            # Chain sharing: only meaningful when the runtime supports
            # incremental process checkpoints and the leaf has ancestors.
            chain_sharing_active = False
            leaf = manifests.get(checkpoint_id)
            try:
                supports_incremental = bool(self.runtime.capabilities().supports_incremental_process)
            except Exception:
                supports_incremental = False
            if supports_incremental and leaf is not None and leaf.parent_checkpoint_id is not None:
                pin_chain = getattr(self.storage, "pin_chain", None)
                if callable(pin_chain) and pin_chain(source_sandbox_id, checkpoint_id):
                    with self._fork_lock:
                        self._fork_chain_pins[target_sandbox_id] = (source_sandbox_id, checkpoint_id)
                    chain_sharing_active = True
                else:
                    logger.warning(
                        "Fork chain-sharing pin unavailable; using copy mode source=%s target=%s checkpoint=%s",
                        source_sandbox_id,
                        target_sandbox_id,
                        checkpoint_id,
                    )

            chain_links, chain_bytes_saved = self._copy_checkpoint_artifacts(
                source_sandbox_id,
                target_sandbox_id,
                manifests=manifests,
                copy_plan=copy_plan,
                leaf_checkpoint_id=checkpoint_id,
                bundle_root=bundle_root,
                checkpoint_root=checkpoint_root,
                chain_sharing_active=chain_sharing_active,
            )

            if chain_sharing_active:
                # Plant ancestor symlinks for the whole parent chain so
                # CRIU's chain walk during restore resolves into the
                # source's bytes without copying. Walk via manifests because
                # the copy plan short-circuits when the leaf carries both
                # process+filesystem artifacts (the common incremental case).
                try:
                    cursor_id = manifests[checkpoint_id].parent_checkpoint_id
                    seen: set[CheckpointId] = set()
                    while cursor_id is not None and cursor_id not in seen:
                        seen.add(cursor_id)
                        self.runtime.link_ancestor_pre_dump(source_sandbox_id, target_sandbox_id, cursor_id)
                        chain_links += 1
                        parent_manifest = manifests.get(cursor_id)
                        if parent_manifest is None:
                            break
                        chain_bytes_saved += sum(
                            int(ref.size_bytes or 0) for ref in parent_manifest.process_artifacts
                        )
                        cursor_id = parent_manifest.parent_checkpoint_id
                except Exception:
                    logger.exception(
                        "Failed to plant ancestor symlinks for chain sharing source=%s target=%s checkpoint=%s",
                        source_sandbox_id,
                        target_sandbox_id,
                        checkpoint_id,
                    )

            with self._fork_lock:
                self._fork_children.setdefault(source_sandbox_id, set()).add(target_sandbox_id)

            self._remember_current_restore_sources(
                target_sandbox_id,
                checkpoint_id,
                process_checkpoint_id,
                filesystem_checkpoint_id,
            )

            inherited_checkpoint_at = manifests[checkpoint_id].created_at
            upsert = getattr(self.inspector, "upsert_snapshot", None)
            if callable(upsert):
                upsert(
                    SandboxSnapshot(
                        sandbox_id=target_sandbox_id,
                        runtime_name=self.runtime.name,
                        is_running=False,
                        process_changed=False,
                        filesystem_changed=False,
                        observed_at=utc_now(),
                        last_checkpoint_at=inherited_checkpoint_at,
                    )
                )
            self.scheduler.mark_checkpoint_complete(target_sandbox_id, inherited_checkpoint_at)

            result = forking.ForkResult(
                source_sandbox_id=source_sandbox_id,
                target_sandbox_id=target_sandbox_id,
                checkpoint_id=checkpoint_id,
                filesystem_checkpoint_id=filesystem_checkpoint_id,
                chain_shared=chain_sharing_active,
                chain_links=chain_links,
                chain_bytes_saved=chain_bytes_saved,
            )
            operation.finish(
                status="succeeded",
                attributes={
                    "checkpoint_id": str(checkpoint_id),
                    "filesystem_checkpoint_id": str(filesystem_checkpoint_id),
                    "chain_shared": chain_sharing_active,
                    "chain_links": chain_links,
                },
            )
            self._journal_lifecycle(
                source_sandbox_id,
                "fork_source",
                metadata={
                    "target_sandbox_id": str(target_sandbox_id),
                    "checkpoint_id": str(checkpoint_id),
                    "chain_shared": chain_sharing_active,
                },
            )
            self._journal_lifecycle(
                target_sandbox_id,
                "fork_created",
                metadata={
                    "source_sandbox_id": str(source_sandbox_id),
                    "checkpoint_id": str(checkpoint_id),
                },
            )
            logger.info(
                "Forked sandbox source=%s target=%s checkpoint=%s chain_shared=%s links=%d bytes_saved=%d",
                source_sandbox_id,
                target_sandbox_id,
                checkpoint_id,
                chain_sharing_active,
                chain_links,
                chain_bytes_saved,
            )
            return result
        except Exception:
            operation.finish(status="failed")
            raise

    def _copy_checkpoint_artifacts(
        self,
        source_sandbox_id: SandboxId,
        target_sandbox_id: SandboxId,
        *,
        manifests: dict[CheckpointId, CheckpointManifest],
        copy_plan,
        leaf_checkpoint_id: CheckpointId,
        bundle_root: Path,
        checkpoint_root: Path,
        chain_sharing_active: bool,
    ) -> tuple[int, int]:
        """Copy a checkpoint chain's manifests + artifacts onto another
        sandbox id with path rewrites. With ``chain_sharing_active`` the
        non-leaf process artifacts become symlink references into the
        source's pre-dumps; without it everything is fully materialized
        (fork-backed txn commits require this: the fork dies afterwards).
        Returns ``(chain_links, chain_bytes_saved)``."""
        chain_links = 0
        chain_bytes_saved = 0
        for copy_id, copy_process, copy_filesystem in copy_plan:
            source_manifest = manifests[copy_id]
            is_leaf = copy_id == leaf_checkpoint_id
            link_this_entry = chain_sharing_active and copy_process and not is_leaf
            process_refs = []
            filesystem_refs = []
            if copy_process:
                if link_this_entry:
                    self.runtime.link_ancestor_pre_dump(source_sandbox_id, target_sandbox_id, copy_id)
                for reference in source_manifest.process_artifacts:
                    payload = self.storage.get_artifact(source_sandbox_id, copy_id, reference)
                    if link_this_entry:
                        rewritten = forking.rewrite_process_artifact_linked(
                            payload,
                            target_sandbox_id=target_sandbox_id,
                            checkpoint_id=copy_id,
                            bundle_root=bundle_root,
                            checkpoint_root=checkpoint_root,
                        )
                        chain_links += 1
                        chain_bytes_saved += int(reference.size_bytes or 0)
                    else:
                        rewritten = forking.rewrite_process_artifact(
                            payload,
                            source_sandbox_id=source_sandbox_id,
                            target_sandbox_id=target_sandbox_id,
                            checkpoint_id=copy_id,
                            bundle_root=bundle_root,
                            checkpoint_root=checkpoint_root,
                            preserve_symlinks=chain_sharing_active and is_leaf,
                        )
                    process_refs.append(
                        self.storage.put_artifact(
                            target_sandbox_id,
                            copy_id,
                            ArtifactPayload(
                                kind=reference.kind,
                                name=reference.name,
                                data=rewritten,
                                metadata=dict(reference.metadata),
                            ),
                        )
                    )
            if copy_filesystem:
                fork_fs_metadata = self.runtime.filesystem_checkpoint_metadata(target_sandbox_id, copy_id)
                for reference in source_manifest.filesystem_artifacts:
                    payload = self.storage.get_artifact(source_sandbox_id, copy_id, reference)
                    filesystem_refs.append(
                        self.storage.put_artifact(
                            target_sandbox_id,
                            copy_id,
                            ArtifactPayload(
                                kind=reference.kind,
                                name=reference.name,
                                data=forking.rewrite_filesystem_artifact(
                                    payload,
                                    target_sandbox_id=target_sandbox_id,
                                    checkpoint_id=copy_id,
                                    filesystem_metadata=fork_fs_metadata,
                                ),
                                metadata=dict(reference.metadata),
                            ),
                        )
                    )
            manifest = CheckpointManifest(
                schema_version=source_manifest.schema_version,
                checkpoint_id=source_manifest.checkpoint_id,
                sandbox_id=target_sandbox_id,
                created_at=source_manifest.created_at,
                runtime_name=source_manifest.runtime_name,
                runtime_version=source_manifest.runtime_version,
                process_artifacts=process_refs,
                filesystem_artifacts=filesystem_refs,
                metadata=dict(source_manifest.metadata),
            ).with_integrity()
            self.storage.put_manifest(manifest)
        return chain_links, chain_bytes_saved

    def release_fork(self, target_sandbox_id: SandboxId) -> None:
        """Reverse fork_once's bookkeeping when a fork is destroyed."""
        # First, and unconditionally: a bare fork's effect session (F1). This
        # cannot ride the chain-pin path below, which returns early when the
        # fork has no pin — the session would then leak onto a dead id.
        self._release_effect_session(target_sandbox_id)
        with self._fork_lock:
            pin = self._fork_chain_pins.pop(target_sandbox_id, None)
            for children in self._fork_children.values():
                children.discard(target_sandbox_id)
        if pin is None:
            return
        source_sandbox_id, leaf_checkpoint_id = pin
        unpin_chain = getattr(self.storage, "unpin_chain", None)
        if not callable(unpin_chain):
            return
        try:
            unpin_chain(source_sandbox_id, leaf_checkpoint_id)
        except Exception:
            logger.exception(
                "Failed to unpin chain for fork=%s source=%s leaf=%s",
                target_sandbox_id,
                source_sandbox_id,
                leaf_checkpoint_id,
            )

    def prepare_source_destroy(self, source_sandbox_id: SandboxId) -> None:
        """Before destroying a sandbox that has live forks, detach them:
        promote one fork's filesystem clone so the source's dataset loses
        its dependents, and replace chain-shared symlinks with real bytes
        (storage artifacts and runtime pre-dump trees)."""
        self._forget_current_restore_sources(source_sandbox_id)
        self._journal_lifecycle(source_sandbox_id, "destroy")
        # Recorded egress bodies die with the sandbox (D2): replay must
        # happen while the recording sandbox is still alive.
        store = self.cassette_store
        if store is not None:
            try:
                store.prune(source_sandbox_id)
            except Exception:
                logger.debug(
                    "Failed to prune cassettes for %s", source_sandbox_id, exc_info=True
                )
        with self._fork_lock:
            live_forks = sorted(self._fork_children.get(source_sandbox_id, set()))
            pinned_forks = [fork for fork, (source, _) in self._fork_chain_pins.items() if source == source_sandbox_id]
        if not live_forks and not pinned_forks:
            return
        if live_forks:
            # Promoting one clone re-parents the source's snapshots (and any
            # sibling clones) onto it; destroying the source then succeeds.
            # Backends without clone-origin dependencies no-op here.
            try:
                self.runtime.promote_filesystem_dataset(live_forks[0])
            except Exception:
                logger.exception(
                    "Failed to promote fork filesystem before source destroy source=%s fork=%s",
                    source_sandbox_id,
                    live_forks[0],
                )
        if pinned_forks:
            materialize = getattr(self.storage, "materialize_linked_artifacts", None)
            if callable(materialize):
                try:
                    materialize(source_sandbox_id)
                except Exception:
                    logger.exception("Failed to materialize linked storage artifacts for source=%s", source_sandbox_id)
            for fork_id in pinned_forks:
                try:
                    self.runtime.materialize_linked_pre_dumps(fork_id)
                except Exception:
                    logger.exception("Failed to materialize linked pre-dumps for fork=%s", fork_id)
        with self._fork_lock:
            self._fork_children.pop(source_sandbox_id, None)

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

    # ----- transactions (B2) -------------------------------------------
    # Snapshot-based, weak isolation: actions run in place; abort rewinds
    # to the base checkpoint; observation staging keeps gated responses
    # from escaping an uncommitted txn. One active txn per sandbox.

    def _txn_active(self, sandbox_id: SandboxId) -> bool:
        with self._txn_lock:
            return sandbox_id in self._active_txns

    def begin_txn(
        self,
        sandbox_id: SandboxId,
        *,
        label: str | None = None,
        isolation: str = "snapshot",
        effects: str | None = None,
    ) -> TxnDescription:
        if isolation not in ("snapshot", "fork"):
            raise ValueError(f"unknown txn isolation: {isolation!r} (expected snapshot or fork)")
        effect_policy = self._resolve_effect_policy(isolation, effects)
        if effect_policy not in EFFECT_POLICIES:
            raise ValueError(
                f"unknown effect policy: {effect_policy!r} "
                f"(expected one of {EFFECT_POLICIES})"
            )
        if isolation == "fork" and effect_policy == "defer":
            # The queue would live on a session whose sandbox the commit
            # destroys; no use case justifies that hand-off yet (D3
            # decision 12).
            raise ValueError(
                "effects='defer' is not supported with isolation='fork': the "
                "fork owning the queue is destroyed by commit. Use 'reject' "
                "(the fork default), 'seal', or 'allow'."
            )
        txn_id = new_txn_id()
        with self._txn_lock:
            existing = self._active_txns.get(sandbox_id)
            if existing is not None:
                raise TxnActiveError(
                    f"transaction already active for {sandbox_id}: {existing.txn_id}"
                )
            if sandbox_id in self._active_txns:
                raise TxnActiveError(f"transaction begin already in flight for {sandbox_id}")
            # Reservation: suppresses auto-checkpoints and locks out
            # concurrent begins while the base checkpoint runs.
            self._active_txns[sandbox_id] = None
        try:
            if self._merge_active(sandbox_id):
                raise TxnError(
                    f"merge in progress for {sandbox_id}; begin the transaction after it completes"
                )
            if isolation == "fork":
                return self._begin_fork_txn(
                    sandbox_id, txn_id, label, effect_policy=effect_policy
                )
            base_checkpoint_id: CheckpointId | None = None
            base_was_fresh = False
            changed = True
            try:
                snapshot = self.inspector.inspect(sandbox_id)
                changed = bool(snapshot.process_changed or snapshot.filesystem_changed)
            except Exception:
                changed = True
            if not changed:
                base_checkpoint_id = self._latest_full_checkpoint_id(sandbox_id)
            if base_checkpoint_id is None:
                result = self.checkpoint_once(sandbox_id, leave_running=True)
                if result.status.value != "succeeded":
                    raise TxnError(
                        f"txn base checkpoint failed for {sandbox_id}: "
                        f"status={result.status.value} message={result.message}"
                    )
                base_checkpoint_id = result.checkpoint_id
                base_was_fresh = True
            self.begin_observation_staging(sandbox_id)
            journal = self.journal
            if journal is not None:
                try:
                    journal.set_active_txn(sandbox_id, txn_id)
                except Exception:
                    logger.exception("Failed to set active txn on journal sandbox=%s", sandbox_id)
            description = TxnDescription(
                txn_id=txn_id,
                sandbox_id=str(sandbox_id),
                base_checkpoint_id=str(base_checkpoint_id),
                base_was_fresh=base_was_fresh,
                started_at=utc_now().isoformat(),
                label=label,
                effects=effect_policy,
            )
            with self._txn_lock:
                self._active_txns[sandbox_id] = description
            self._arm_effect_session(
                sandbox_id, policy=effect_policy, txn_id=txn_id, isolation="snapshot"
            )
            self._journal_lifecycle(
                sandbox_id,
                "txn_begin",
                metadata={
                    "txn_id": txn_id,
                    "base_checkpoint_id": str(base_checkpoint_id),
                    "base_was_fresh": base_was_fresh,
                    **({"label": label} if label else {}),
                },
            )
            self.telemetry.emit_event(
                "txn.begin",
                self._telemetry_attrs(
                    sandbox_id,
                    component="system",
                    extra={
                        "txn_id": txn_id,
                        "base_checkpoint_id": str(base_checkpoint_id),
                        "base_was_fresh": base_was_fresh,
                    },
                ),
            )
            logger.info(
                "Began txn %s for sandbox %s base=%s fresh=%s",
                txn_id,
                sandbox_id,
                base_checkpoint_id,
                base_was_fresh,
            )
            return description
        except Exception:
            with self._txn_lock:
                self._active_txns.pop(sandbox_id, None)
            raise

    def commit_txn(
        self,
        sandbox_id: SandboxId,
        txn_id: str,
        *,
        force: bool = False,
        observations: str = "append",
    ) -> TxnCommitResult:
        active = self._require_txn(sandbox_id, txn_id)
        if active.isolation == "fork":
            if active.sandbox_id != str(sandbox_id):
                raise TxnMismatchError(
                    f"fork-backed txn {active.txn_id} must be committed via its "
                    f"source sandbox {active.sandbox_id}"
                )
            return self._commit_fork_txn(active, force=force, observations=observations)
        registry = self.response_gate_registry
        if registry is not None:
            # Anything still pending (armed but never checkpoint-released)
            # moves into the staged buffer first.
            registry.release(sandbox_id)
        released = self.release_staged_observations(sandbox_id)
        self.end_observation_staging(sandbox_id)
        base_dropped = False
        if active.base_was_fresh and active.base_checkpoint_id is not None:
            try:
                self.storage.delete_checkpoint(
                    sandbox_id, CheckpointId(active.base_checkpoint_id), cascade=False
                )
                base_dropped = True
            except Exception:
                logger.warning(
                    "Keeping txn base checkpoint after commit (delete failed) sandbox=%s ckpt=%s",
                    sandbox_id,
                    active.base_checkpoint_id,
                    exc_info=True,
                )
        self._journal_lifecycle(
            sandbox_id,
            "txn_commit",
            metadata={
                "txn_id": active.txn_id,
                "released": released,
                "base_dropped": base_dropped,
            },
        )
        # Deferred writes fire only now, after the filesystem commit, and
        # a failure here never unwinds it (D3 decision 5).
        flush_report = self._flush_deferred_effects(sandbox_id, active.txn_id)
        self._release_effect_session(sandbox_id)
        self._clear_txn(sandbox_id)
        self.telemetry.emit_event(
            "txn.commit",
            self._telemetry_attrs(
                sandbox_id,
                component="system",
                extra={"txn_id": active.txn_id, "released": released, "base_dropped": base_dropped},
            ),
        )
        logger.info(
            "Committed txn %s for sandbox %s released=%d base_dropped=%s",
            active.txn_id,
            sandbox_id,
            released,
            base_dropped,
        )
        return TxnCommitResult(
            txn_id=active.txn_id,
            released_observations=released,
            base_dropped=base_dropped,
            effects=flush_report,
        )

    def abort_txn(
        self, sandbox_id: SandboxId, txn_id: str, *, force: bool = False
    ) -> TxnAbortResult:
        active = self._require_txn(sandbox_id, txn_id)
        gate = self.effect_gate
        if not force and gate is not None:
            # `seal` traded abortability for letting the write out; refusing
            # here is the whole point of the policy.
            sealed_side = None
            for candidate in (sandbox_id, active.fork_sandbox_id):
                if candidate is not None and gate.sealed(candidate):
                    sealed_side = candidate
                    break
            if sealed_side is not None:
                raise TxnNotAbortable(
                    f"txn {active.txn_id} already sent a mutating request under "
                    "effects='seal'; commit it, or abort with force=True to "
                    "accept that the external write stands"
                )
        if active.isolation == "fork":
            if active.sandbox_id != str(sandbox_id):
                raise TxnMismatchError(
                    f"fork-backed txn {active.txn_id} must be aborted via its "
                    f"source sandbox {active.sandbox_id}"
                )
            return self._abort_fork_txn(active)
        registry = self.response_gate_registry
        if registry is not None:
            registry.release(sandbox_id)
        discarded = self.discard_staged_observations(sandbox_id)
        assert active.base_checkpoint_id is not None
        restore = self.restore_once(sandbox_id, CheckpointId(active.base_checkpoint_id))
        if restore.status.value != "succeeded":
            # Txn stays open: observations are already dropped (idempotent),
            # the caller may retry abort.
            raise TxnAbortError(
                f"txn abort restore failed for {sandbox_id}: "
                f"status={restore.status.value} message={restore.message}",
                restore_result=restore,
            )
        self.end_observation_staging(sandbox_id)
        self._journal_lifecycle(
            sandbox_id,
            "txn_abort",
            metadata={
                "txn_id": active.txn_id,
                "discarded": discarded,
                "restored_checkpoint_id": str(active.base_checkpoint_id),
            },
        )
        self._clear_txn(sandbox_id)
        mutating_egress = self._mutating_egress_in_txn(sandbox_id, active.txn_id)
        dropped = self._drop_deferred_effects(sandbox_id, active.txn_id)
        self._release_effect_session(sandbox_id)
        self.telemetry.emit_event(
            "txn.abort",
            self._telemetry_attrs(
                sandbox_id,
                component="system",
                extra={
                    "txn_id": active.txn_id,
                    "discarded": discarded,
                    "mutating_egress": mutating_egress,
                },
            ),
        )
        logger.info(
            "Aborted txn %s for sandbox %s discarded=%d restored=%s mutating_egress=%d",
            active.txn_id,
            sandbox_id,
            discarded,
            active.base_checkpoint_id,
            mutating_egress,
        )
        return TxnAbortResult(
            txn_id=active.txn_id,
            discarded_observations=discarded,
            restored_checkpoint_id=active.base_checkpoint_id,
            mutating_egress=mutating_egress,
            deferred_dropped=dropped,
        )

    def current_txn(self, sandbox_id: SandboxId) -> TxnDescription | None:
        with self._txn_lock:
            return self._active_txns.get(sandbox_id)

    def release_txn(self, sandbox_id: SandboxId) -> None:
        """Teardown hook (sandbox kill with an open txn): drop staged
        observations and disarm — no restore, the sandbox is dying."""
        with self._txn_lock:
            active = self._active_txns.pop(sandbox_id, None)
        if active is None:
            return
        if active.isolation == "fork":
            self._release_fork_txn(sandbox_id, active)
            return
        # The sandbox is dying with an open txn: its queued writes are
        # discarded exactly as an abort would (design §3.3), so the ledger
        # never keeps a pending row that can no longer fire.
        self._drop_deferred_effects(sandbox_id, active.txn_id)
        self._release_effect_session(sandbox_id)
        try:
            registry = self.response_gate_registry
            if registry is not None:
                registry.release(sandbox_id)
                registry.discard_staged(sandbox_id)
                registry.end_staging(sandbox_id)
        except Exception:
            logger.exception("Txn teardown staging cleanup failed sandbox=%s", sandbox_id)
        journal = self.journal
        if journal is not None:
            try:
                journal.set_active_txn(sandbox_id, None)
            except Exception:
                logger.debug("Failed to clear journal txn on teardown", exc_info=True)
        logger.info("Released open txn %s during sandbox teardown %s", active.txn_id, sandbox_id)

    def _require_txn(self, sandbox_id: SandboxId, txn_id: str) -> TxnDescription:
        with self._txn_lock:
            active = self._active_txns.get(sandbox_id)
        if active is None:
            raise TxnMismatchError(f"no active transaction for {sandbox_id}")
        if active.txn_id != str(txn_id):
            raise TxnMismatchError(
                f"txn mismatch for {sandbox_id}: active={active.txn_id} given={txn_id}"
            )
        return active

    def _clear_txn(self, sandbox_id: SandboxId) -> None:
        journal = self.journal
        if journal is not None:
            try:
                journal.set_active_txn(sandbox_id, None)
            except Exception:
                logger.debug("Failed to clear journal txn", exc_info=True)
        with self._txn_lock:
            self._active_txns.pop(sandbox_id, None)

    # ----- fork-backed transactions (B3) --------------------------------
    # begin = fork (actions run there; the source stays clean and
    # serving); commit = promote the fork's whole state back onto the
    # source's own identity (dump fork -> materialized manifest copy ->
    # clone fs over the source dataset -> CRIU restore under the source
    # id -> zfs promote); abort = destroy the fork, never restore the
    # source. The primary sandbox_id never changes, so leases, journals,
    # storage paths and SDK handles all stay valid.

    def configure_fork_txn_hooks(
        self,
        *,
        fork: Callable[[SandboxId], SandboxId],
        destroy: Callable[[SandboxId], None],
        lease_repair: Callable[[SandboxId], None] | None = None,
        lease_transfer: "_LeaseTransfer | None" = None,
    ) -> None:
        """Engine-owned hooks: forking needs lease allocation + bundle
        replication + restore (Engine.fork_sandbox) and fork teardown
        mirrors the SDK kill path — both live above the system.

        ``lease_transfer(fork_id, source_id)`` moves the fork's network
        identity onto the source before a promotion restore and returns
        whether it happened; the system cannot reach the network manager
        itself. An unwired hook (or a ``False`` return) means the promotion
        takes the pre-existing `lease_repair` path.
        """
        self._fork_txn_fork = fork
        self._fork_txn_destroy = destroy
        self._fork_txn_lease_repair = lease_repair
        self._fork_txn_lease_transfer = lease_transfer

    def _begin_fork_txn(
        self,
        sandbox_id: SandboxId,
        txn_id: str,
        label: str | None,
        *,
        effect_policy: str = "reject",
    ) -> TxnDescription:
        if self._fork_txn_fork is None or self._fork_txn_destroy is None:
            raise TxnError(
                "fork-backed transactions need an engine (fork hooks are not configured)"
            )
        fork_sandbox_id: SandboxId | None = None
        try:
            fork_sandbox_id = SandboxId(str(self._fork_txn_fork(sandbox_id)))
            base_checkpoint_id = self._fork_point_checkpoint_id(fork_sandbox_id)
            # Gated LLM responses inside the txn happen in the fork.
            self.begin_observation_staging(fork_sandbox_id)
            journal = self.journal
            if journal is not None:
                for tagged in (sandbox_id, fork_sandbox_id):
                    try:
                        journal.set_active_txn(tagged, txn_id)
                    except Exception:
                        logger.exception(
                            "Failed to set active txn on journal sandbox=%s", tagged
                        )
            description = TxnDescription(
                txn_id=txn_id,
                sandbox_id=str(sandbox_id),
                base_checkpoint_id=None if base_checkpoint_id is None else str(base_checkpoint_id),
                base_was_fresh=False,
                started_at=utc_now().isoformat(),
                label=label,
                isolation="fork",
                fork_sandbox_id=str(fork_sandbox_id),
                effects=effect_policy,
            )
            with self._txn_lock:
                # Both ids map to the description: auto-checkpoints stay
                # suppressed and merges locked out on either side.
                self._active_txns[sandbox_id] = description
                self._active_txns[fork_sandbox_id] = description
            # The fork is what executes inside a fork-backed txn, so its
            # egress is what the policy must gate.
            self._arm_effect_session(
                fork_sandbox_id, policy=effect_policy, txn_id=txn_id, isolation="fork"
            )
            self._journal_lifecycle(
                sandbox_id,
                "txn_begin",
                metadata={
                    "txn_id": txn_id,
                    "isolation": "fork",
                    "fork_sandbox_id": str(fork_sandbox_id),
                    "base_checkpoint_id": description.base_checkpoint_id,
                    **({"label": label} if label else {}),
                },
            )
            self.telemetry.emit_event(
                "txn.begin",
                self._telemetry_attrs(
                    sandbox_id,
                    component="system",
                    extra={
                        "txn_id": txn_id,
                        "isolation": "fork",
                        "fork_sandbox_id": str(fork_sandbox_id),
                    },
                ),
            )
            logger.info(
                "Began fork-backed txn %s for sandbox %s fork=%s base=%s",
                txn_id,
                sandbox_id,
                fork_sandbox_id,
                description.base_checkpoint_id,
            )
            return description
        except Exception:
            if fork_sandbox_id is not None:
                try:
                    self.end_observation_staging(fork_sandbox_id)
                except Exception:
                    logger.debug("Fork txn staging cleanup failed", exc_info=True)
                try:
                    self._fork_txn_destroy(fork_sandbox_id)
                except Exception:
                    logger.exception(
                        "Failed to destroy fork after txn begin failure fork=%s",
                        fork_sandbox_id,
                    )
                with self._txn_lock:
                    self._active_txns.pop(fork_sandbox_id, None)
            raise

    def _commit_fork_txn(
        self, active: TxnDescription, *, force: bool, observations: str = "append"
    ) -> TxnCommitResult:
        if observations not in OBSERVATION_POLICIES:
            raise ValueError(
                f"unknown observation policy: {observations!r} "
                f"(expected one of {OBSERVATION_POLICIES})"
            )
        source_id = SandboxId(active.sandbox_id)
        assert active.fork_sandbox_id is not None
        fork_id = SandboxId(active.fork_sandbox_id)
        operation = start_operation(
            self.telemetry,
            "txn.fork_commit",
            self._telemetry_attrs(
                source_id,
                component="system",
                extra={"txn_id": active.txn_id, "fork_sandbox_id": str(fork_id), "forced": force},
            ),
        )
        try:
            # 1. Dirty-source gate (C1): promoting over source-side writes
            # would silently discard them.
            if not force and active.base_checkpoint_id is not None:
                source_changes = None
                try:
                    source_changes = self.changeset_since(
                        source_id, CheckpointId(active.base_checkpoint_id)
                    )
                except FileNotFoundError:
                    # The fork-point snapshot is gone from the source's
                    # dataset: a previous commit attempt already swapped
                    # the filesystem (mid-commit retry) — skip the check.
                    logger.warning(
                        "Fork-txn commit retry detected (fork-point snapshot missing); "
                        "skipping dirty check sandbox=%s",
                        source_id,
                    )
                if source_changes is not None and source_changes.entries:
                    raise TxnCommitConflict(
                        f"source {source_id} changed {len(source_changes.entries)} path(s) "
                        "since the fork point; commit with force=True to discard them "
                        "or abort the transaction"
                    )
            # 2-8. Promote the fork's whole state onto the source's own
            # identity — shared with C4's promotion-based process merge.
            # The fork stays alive until the swap is proven, so failures
            # leave the txn open and commit retryable.
            try:
                commit_checkpoint_id, fork_retained = self._promote_fork_onto_source(
                    source_id, fork_id
                )
            except ForkPromotionError as exc:
                raise TxnError(f"{exc}; the txn stays open — retry commit") from exc
            # The committed observations flow now.
            registry = self.response_gate_registry
            if registry is not None:
                registry.release(fork_id)
            released = self.release_staged_observations(fork_id)
            self.end_observation_staging(fork_id)
            # Adopt the fork's action history (C3): the commit's meaning
            # is "take the fork's work", and the fork is about to die.
            # Consolidation failures never unwind the completed swap.
            observations_consolidated: int | None = None
            if observations != "none":
                try:
                    observation_report = self.consolidate_observations(
                        source_id,
                        fork_id,
                        policy=observations,
                        reason="txn_commit",
                    )
                    observations_consolidated = observation_report.consolidated
                except Exception:
                    logger.exception(
                        "Observation consolidation after fork-txn commit failed source=%s fork=%s",
                        source_id,
                        fork_id,
                    )
            self._journal_lifecycle(
                source_id,
                "txn_commit",
                metadata={
                    "txn_id": active.txn_id,
                    "isolation": "fork",
                    "fork_sandbox_id": str(fork_id),
                    "promoted_checkpoint_id": str(commit_checkpoint_id),
                    "released": released,
                    "forced": force,
                    "observations_consolidated": observations_consolidated,
                },
            )
            self._journal_lifecycle(
                fork_id,
                "txn_fork_committed",
                metadata={"txn_id": active.txn_id, "source_sandbox_id": str(source_id)},
            )
            # The fork carried this txn's effect window and is about to be
            # destroyed; `defer` is refused for fork txns so there is never
            # a queue to hand over (D3 decision 12).
            self._release_effect_session(fork_id)
            self._clear_fork_txn(source_id, fork_id)
            if not fork_retained:
                try:
                    self._fork_txn_destroy(fork_id)
                except Exception:
                    logger.exception("Failed to destroy fork after commit fork=%s", fork_id)
            self.telemetry.emit_event(
                "txn.commit",
                self._telemetry_attrs(
                    source_id,
                    component="system",
                    checkpoint_id=commit_checkpoint_id,
                    extra={
                        "txn_id": active.txn_id,
                        "isolation": "fork",
                        "released": released,
                        "forced": force,
                    },
                ),
            )
            operation.finish(
                status="succeeded",
                attributes={"checkpoint_id": str(commit_checkpoint_id)},
            )
            logger.info(
                "Committed fork-backed txn %s: promoted fork %s onto %s at %s released=%d",
                active.txn_id,
                fork_id,
                source_id,
                commit_checkpoint_id,
                released,
            )
            return TxnCommitResult(
                txn_id=active.txn_id,
                released_observations=released,
                base_dropped=False,
                promoted_checkpoint_id=str(commit_checkpoint_id),
                observations_consolidated=observations_consolidated,
            )
        except Exception:
            operation.finish(status="failed")
            raise

    def _promote_fork_onto_source(
        self,
        source_id: SandboxId,
        fork_id: SandboxId,
        *,
        lazy_pages: bool = False,
    ) -> tuple[CheckpointId, bool]:
        """Shared B3/C4 swap: promote the fork's whole state (filesystem
        + processes) onto the source's unchanged identity. Returns
        ``(promoted_checkpoint_id, fork_retained)``.

        Fork lifecycle differs by path (see the dump below):
        - **no-transfer path** (unnetworked, or a fork without its own
          lease): the fork is dumped ``leave_running=True`` and stays
          alive, so a failed swap is retryable against the live fork;
        - **transfer path** (the fork owns a netns the source must adopt):
          the fork is dumped-and-stopped, so its *processes* are gone once
          the dump succeeds. Its filesystem and checkpoint chain survive,
          so a failed restore is recoverable by restoring the replicated
          ``commit_checkpoint_id`` — not by re-running the dead fork.

        Failures raise ``ForkPromotionError``; ``fork_retained=True`` means
        the final detach promote failed and the fork's dataset must NOT be
        destroyed (it still backs the source)."""
        # Pre-flight, before anything irreversible: does this promotion move
        # the fork's network identity onto the source? The probe mutates
        # nothing, and the answer decides how the fork is dumped below.
        transfer_netns = False
        if self._fork_txn_lease_transfer is not None:
            transfer_netns = bool(
                self._fork_txn_lease_transfer(fork_id, source_id, probe=True)
            )
        if transfer_netns:
            bundle_path = getattr(self.runtime, "bundle_path_for", None)
            if bundle_path is not None and not (Path(bundle_path(source_id)) / "config.json").is_file():
                raise ForkPromotionError(
                    f"promotion cannot retarget the netns of {source_id}: its bundle "
                    "config.json is missing; nothing was changed"
                )
        # Dump the fork. On the transfer path the fork's processes must LEAVE
        # the netns the source is about to run in — two live copies of a
        # listener in one namespace collide (CRIU: "Can't bind inet socket:
        # Address already in use") and killing them after an unlocked dump
        # would also RST the very connections promotion exists to carry.
        # Dump-and-stop is the standard CRIU migration shape. Elsewhere the
        # fork keeps running, as before, so a failed swap is retryable
        # against it.
        checkpoint_result = self.checkpoint_once(fork_id, leave_running=not transfer_netns)
        if checkpoint_result.status.value != "succeeded" or checkpoint_result.checkpoint_id is None:
            raise ForkPromotionError(
                f"promotion checkpoint failed for {fork_id}: "
                f"status={checkpoint_result.status.value}"
            )
        commit_checkpoint_id = checkpoint_result.checkpoint_id
        # Replicate the checkpoint chain onto the source id, fully
        # materialized (no symlinks into the fork: it dies afterwards).
        filesystem_checkpoint_id = self._replicate_fork_checkpoint(
            fork_id, source_id, commit_checkpoint_id
        )
        # ---- point of no return: the source container goes down and
        # its dataset is replaced by the fork's committed snapshot.
        try:
            self.runtime.stop(source_id)
        except Exception:
            logger.debug("Source stop before promotion swap failed", exc_info=True)
        try:
            self.runtime.delete_runtime(source_id, force=True, ignore_missing=True)
        except Exception:
            logger.debug("Source runtime delete before promotion swap failed", exc_info=True)
        # The fork is a clone of source@fork-point: promote it first so
        # destroying the source dataset is not blocked by the dependency
        # (zfs; btrfs promote is a no-op). destroy_filesystem_dataset
        # also unbinds the source's old rootfs mount so the clone lands
        # on a clean mountpoint.
        self.runtime.promote_filesystem_dataset(fork_id)
        self.runtime.destroy_filesystem_dataset(source_id)
        source_rootfs = Path(self.runtime.rootfs_path_for(source_id))
        self.runtime.clone_filesystem_snapshot(
            fork_id,
            filesystem_checkpoint_id,
            source_id,
            target_rootfs_path=source_rootfs,
        )
        # The fork was dumped inside its own netns, so its sockets are bound
        # to the fork's guest IP. Restoring that image against the source's
        # netns fails in CRIU's soccr with EADDRNOTAVAIL ("Can't bind inet
        # socket back"), so the address moves with the processes: the source
        # identity adopts the fork's lease, bundle netns and metadata.
        netns_transferred = False
        if transfer_netns and self._fork_txn_lease_transfer is not None:
            try:
                netns_transferred = bool(self._fork_txn_lease_transfer(fork_id, source_id))
            except Exception as exc:
                # The fork is already dumped-and-stopped on this path; its
                # filesystem and image survive, so recovery is a restore of
                # commit_checkpoint_id, not a retry against a live fork.
                raise ForkPromotionError(
                    f"promotion could not transfer the network identity of {fork_id} "
                    f"onto {source_id}: {exc}; the fork's processes are stopped but its "
                    "filesystem is intact"
                ) from exc
        restore = self.restore_once(
            source_id,
            commit_checkpoint_id,
            restore_metadata={"lazy_pages": True} if lazy_pages else None,
        )
        if restore.status.value != "succeeded":
            # On the transfer path the fork is stopped; "intact" means its
            # filesystem, not a live process tree.
            fork_state = (
                "the fork's processes are stopped but its filesystem is intact"
                if netns_transferred
                else "the fork is intact"
            )
            raise ForkPromotionError(
                f"promotion restore failed for {source_id}: "
                f"status={restore.status.value} message={restore.message}; {fork_state}"
            )
        # A transferred netns is already plumbed and was never torn down;
        # repair only applies when the source kept its own lease.
        if not netns_transferred and self._fork_txn_lease_repair is not None:
            try:
                self._fork_txn_lease_repair(source_id)
            except Exception:
                logger.exception("Lease repair after promotion failed sandbox=%s", source_id)
        # Detach the source's new dataset from the fork's snapshot so
        # the fork can die (zfs; btrfs promote is a no-op).
        fork_retained = False
        try:
            self.runtime.promote_filesystem_dataset(source_id)
        except Exception:
            fork_retained = True
            logger.exception(
                "Detach promote after promotion failed; keeping fork %s alive "
                "(the source dataset still depends on it)",
                fork_id,
            )
        return commit_checkpoint_id, fork_retained

    def _replicate_fork_checkpoint(
        self,
        fork_id: SandboxId,
        source_id: SandboxId,
        commit_checkpoint_id: CheckpointId,
    ) -> CheckpointId:
        """Copy the fork's committed checkpoint chain onto the source id,
        fully materialized (no symlinks into the fork — it dies after the
        commit). Returns the filesystem checkpoint id to clone."""
        paths = getattr(self.runtime, "paths", None)
        bundle_root = None if paths is None else paths.bundle_root
        checkpoint_root = None if paths is None else paths.checkpoint_root
        if bundle_root is None or checkpoint_root is None:
            raise TxnError("fork-txn commit requires a runtime exposing bundle/checkpoint roots")
        manifests = {
            cid: self.storage.get_manifest(fork_id, cid)
            for cid in self.storage.list_checkpoints(fork_id)
        }
        copy_plan = forking.resolve_checkpoint_copy_plan(
            list(manifests.keys()), manifests, commit_checkpoint_id
        )
        filesystem_checkpoint_id = next(
            copy_id for copy_id, _, copy_filesystem in reversed(copy_plan) if copy_filesystem
        )
        self._copy_checkpoint_artifacts(
            fork_id,
            source_id,
            manifests=manifests,
            copy_plan=copy_plan,
            leaf_checkpoint_id=commit_checkpoint_id,
            bundle_root=bundle_root,
            checkpoint_root=checkpoint_root,
            chain_sharing_active=False,
        )
        return filesystem_checkpoint_id

    def _abort_fork_txn(self, active: TxnDescription) -> TxnAbortResult:
        source_id = SandboxId(active.sandbox_id)
        assert active.fork_sandbox_id is not None
        fork_id = SandboxId(active.fork_sandbox_id)
        registry = self.response_gate_registry
        if registry is not None:
            registry.release(fork_id)
        discarded = self.discard_staged_observations(fork_id)
        self.end_observation_staging(fork_id)
        self._journal_lifecycle(
            source_id,
            "txn_abort",
            metadata={
                "txn_id": active.txn_id,
                "isolation": "fork",
                "fork_sandbox_id": str(fork_id),
                "discarded": discarded,
                "restored_checkpoint_id": None,
            },
        )
        self._journal_lifecycle(
            fork_id,
            "txn_fork_discarded",
            metadata={"txn_id": active.txn_id, "source_sandbox_id": str(source_id)},
        )
        self._clear_fork_txn(source_id, fork_id)
        # Best-effort teardown: the source is untouched either way; a
        # leaked fork is an operational cleanup, not an open txn.
        if self._fork_txn_destroy is not None:
            try:
                self._fork_txn_destroy(fork_id)
            except Exception:
                logger.exception("Failed to destroy fork on txn abort fork=%s", fork_id)
        # The fork's own flows died with it, but flows the fork fired at
        # the world did not — report them like the snapshot path does.
        mutating_egress = self._mutating_egress_in_txn(fork_id, active.txn_id)
        # A fork-backed txn's effect window lives on the fork.
        fork_dropped = self._drop_deferred_effects(fork_id, active.txn_id)
        self._release_effect_session(fork_id)
        self.telemetry.emit_event(
            "txn.abort",
            self._telemetry_attrs(
                source_id,
                component="system",
                extra={
                    "txn_id": active.txn_id,
                    "isolation": "fork",
                    "discarded": discarded,
                    "mutating_egress": mutating_egress,
                },
            ),
        )
        logger.info(
            "Aborted fork-backed txn %s for sandbox %s (fork %s destroyed, source untouched) "
            "mutating_egress=%d",
            active.txn_id,
            source_id,
            fork_id,
            mutating_egress,
        )
        return TxnAbortResult(
            txn_id=active.txn_id,
            discarded_observations=discarded,
            restored_checkpoint_id=None,
            mutating_egress=mutating_egress,
            deferred_dropped=fork_dropped,
        )

    def _release_fork_txn(self, sandbox_id: SandboxId, active: TxnDescription) -> None:
        """Teardown for a dying sandbox holding a fork-backed txn: clear
        both registrations; when the *source* dies the orphaned fork is
        destroyed too (a dying fork's own kill path finishes itself)."""
        source_id = SandboxId(active.sandbox_id)
        fork_id = None if active.fork_sandbox_id is None else SandboxId(active.fork_sandbox_id)
        if fork_id is not None:
            # The fork owned this txn's effect window; dying discards its
            # queue like an abort would (design §3.3).
            self._drop_deferred_effects(fork_id, active.txn_id)
            self._release_effect_session(fork_id)
        with self._txn_lock:
            self._active_txns.pop(source_id, None)
            if fork_id is not None:
                self._active_txns.pop(fork_id, None)
        if fork_id is not None:
            try:
                registry = self.response_gate_registry
                if registry is not None:
                    registry.release(fork_id)
                    registry.discard_staged(fork_id)
                    registry.end_staging(fork_id)
            except Exception:
                logger.exception("Fork-txn teardown staging cleanup failed fork=%s", fork_id)
        journal = self.journal
        if journal is not None:
            targets = (source_id,) if fork_id is None else (source_id, fork_id)
            for tagged in targets:
                try:
                    journal.set_active_txn(tagged, None)
                except Exception:
                    logger.debug("Failed to clear journal txn on teardown", exc_info=True)
        if (
            fork_id is not None
            and str(sandbox_id) == str(source_id)
            and self._fork_txn_destroy is not None
        ):
            try:
                self._fork_txn_destroy(fork_id)
            except Exception:
                logger.exception("Failed to destroy orphaned txn fork=%s", fork_id)
        logger.info(
            "Released open fork-backed txn %s during sandbox teardown %s",
            active.txn_id,
            sandbox_id,
        )

    def _clear_fork_txn(self, source_id: SandboxId, fork_id: SandboxId) -> None:
        journal = self.journal
        if journal is not None:
            for tagged in (source_id, fork_id):
                try:
                    journal.set_active_txn(tagged, None)
                except Exception:
                    logger.debug("Failed to clear journal txn", exc_info=True)
        with self._txn_lock:
            self._active_txns.pop(source_id, None)
            self._active_txns.pop(fork_id, None)

    def _latest_full_checkpoint_id(self, sandbox_id: SandboxId) -> CheckpointId | None:
        """Newest checkpoint carrying both process and filesystem
        artifacts — the only safe reuse target for a txn base."""
        try:
            checkpoint_ids = self.storage.list_checkpoints(sandbox_id)
        except Exception:
            return None
        for checkpoint_id in reversed(list(checkpoint_ids)):
            try:
                manifest = self.storage.get_manifest(sandbox_id, checkpoint_id)
            except Exception:
                continue
            if getattr(manifest, "process_artifacts", None) and getattr(
                manifest, "filesystem_artifacts", None
            ):
                return checkpoint_id
        return None

    # ----- filesystem changesets (C1) ----------------------------------
    # The backend diff (zfs diff / btrfs send) is the source of truth;
    # the inspector gate is only a fast path that may skip the diff when
    # it can prove the answer is "nothing changed".

    def _latest_filesystem_checkpoint_id(self, sandbox_id: SandboxId) -> CheckpointId | None:
        """Newest checkpoint carrying filesystem artifacts — the boundary
        where the inspector's filesystem cursor was last reset."""
        try:
            checkpoint_ids = self.storage.list_checkpoints(sandbox_id)
        except Exception:
            return None
        for checkpoint_id in reversed(list(checkpoint_ids)):
            try:
                manifest = self.storage.get_manifest(sandbox_id, checkpoint_id)
            except Exception:
                continue
            if getattr(manifest, "filesystem_artifacts", None):
                return checkpoint_id
        return None

    def _filesystem_restore_source_id(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> CheckpointId:
        try:
            resolved_manifest = self._resolve_restore_manifest(
                sandbox_id, checkpoint_id
            )
        except (FileNotFoundError, KeyError):
            # Backward compatibility for runtime snapshots created before a
            # manifest was recorded (and lightweight test/runtime adapters):
            # historical ids were physical ids.
            return checkpoint_id
        filesystem_source_raw = resolved_manifest.metadata.get(
            _FILESYSTEM_RESTORE_CHECKPOINT_ID
        )
        if filesystem_source_raw is None:
            raise ValueError(
                f"checkpoint {checkpoint_id} has no filesystem restore source"
            )
        return CheckpointId(str(filesystem_source_raw))

    def changeset_since(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
        *,
        use_inspector_gate: bool = True,
    ) -> ChangesetResult:
        """Changed rootfs paths of ``sandbox_id`` relative to
        ``checkpoint_id``'s filesystem snapshot (C1). The inspector gate
        may skip the backend diff only when it proves nothing touched
        the filesystem since the last filesystem checkpoint AND that
        checkpoint is the requested base; anything less falls through to
        the authoritative diff."""
        logical_checkpoint_id = CheckpointId(str(checkpoint_id))
        filesystem_checkpoint_id = self._filesystem_restore_source_id(
            sandbox_id, logical_checkpoint_id
        )

        skipped_by_gate = False
        if use_inspector_gate:
            filesystem_changed = True
            try:
                snapshot = self.inspector.inspect(sandbox_id)
                filesystem_changed = bool(snapshot.filesystem_changed)
            except Exception:
                filesystem_changed = True
            if (
                not filesystem_changed
                and self._latest_filesystem_checkpoint_id(sandbox_id)
                == filesystem_checkpoint_id
            ):
                skipped_by_gate = True
        if skipped_by_gate:
            entries: tuple[ChangesetEntry, ...] = ()
        else:
            entries = tuple(
                self.runtime.changeset_since(
                    sandbox_id, filesystem_checkpoint_id
                )
            )
        result = ChangesetResult(
            sandbox_id=sandbox_id,
            base_checkpoint_id=logical_checkpoint_id,
            entries=entries,
            skipped_by_gate=skipped_by_gate,
        )
        self._journal_lifecycle(
            sandbox_id,
            "changeset",
            metadata={
                "base_checkpoint_id": str(logical_checkpoint_id),
                "filesystem_checkpoint_id": str(filesystem_checkpoint_id),
                "entry_count": len(entries),
                "skipped_by_gate": skipped_by_gate,
            },
        )
        self.telemetry.emit_event(
            "changeset.computed",
            self._telemetry_attrs(
                sandbox_id,
                component="system",
                checkpoint_id=logical_checkpoint_id,
                extra={
                    "entry_count": len(entries),
                    "skipped_by_gate": skipped_by_gate,
                    "filesystem_checkpoint_id": str(filesystem_checkpoint_id),
                },
            ),
        )
        logger.info(
            "Computed changeset sandbox=%s base=%s entries=%d skipped_by_gate=%s",
            sandbox_id,
            logical_checkpoint_id,
            len(entries),
            skipped_by_gate,
        )
        return result

    def fork_changeset(self, target_sandbox_id: SandboxId, *, force: bool = False) -> ChangesetResult:
        """Changeset of a fork relative to its fork point (the source
        checkpoint snapshot materialized on the fork's own dataset at
        clone time). ``force=True`` bypasses the inspector gate so the
        backend diff always runs."""
        checkpoint_id = self._fork_point_checkpoint_id(target_sandbox_id)
        if checkpoint_id is None:
            raise ValueError(
                f"no fork_created journal marker for {target_sandbox_id}; "
                "fork_changeset only works for sandboxes created by fork_once "
                "with the action journal enabled"
            )
        return self.changeset_since(
            target_sandbox_id, checkpoint_id, use_inspector_gate=not force
        )

    def _fork_point_checkpoint_id(self, target_sandbox_id: SandboxId) -> CheckpointId | None:
        origin = self._fork_origin(target_sandbox_id)
        return None if origin is None else origin[1]

    def _fork_origin(
        self, target_sandbox_id: SandboxId
    ) -> tuple[SandboxId | None, CheckpointId] | None:
        """``(source_sandbox_id, checkpoint_id)`` from the newest
        ``fork_created`` marker; the source may be None on markers
        written before it was recorded."""
        journal = self.journal
        if journal is None:
            return None
        try:
            records = journal.entries(target_sandbox_id, kind="lifecycle")
        except Exception:
            logger.exception("Failed to read journal for fork point sandbox=%s", target_sandbox_id)
            return None
        for record in reversed(records):
            if record.payload.get("event") != "fork_created":
                continue
            metadata = record.payload.get("metadata") or {}
            raw = metadata.get("checkpoint_id")
            if not raw:
                continue
            raw_source = metadata.get("source_sandbox_id")
            source = SandboxId(str(raw_source)) if raw_source else None
            return source, CheckpointId(str(raw))
        return None

    # ----- filesystem merge (C2) ---------------------------------------
    # Plan-then-apply three-way merge of a fork's changes back into its
    # source. Classification/application live in crab/merging.py; this
    # method owns guards, quiesce, the transient rollback snapshot, and
    # journal/telemetry bookkeeping.

    def _merge_active(self, sandbox_id: SandboxId) -> bool:
        with self._merge_lock:
            return sandbox_id in self._active_merges

    def _pause_for_merge(self, sandbox_id: SandboxId) -> bool:
        try:
            status = str(self.runtime.describe(sandbox_id).status).lower()
        except Exception:
            return False
        if status != "running":
            return False
        self.runtime.pause(sandbox_id)
        return True

    def merge_from_fork(
        self,
        source_sandbox_id: SandboxId,
        fork_sandbox_id: SandboxId,
        *,
        policy: str = "fail_fast",
        ignore_prefixes: tuple[str, ...] | None = None,
        merger: MergerHook | None = None,
        observations: str = "none",
        observation_summarizer: "SummarizerHook | None" = None,
    ) -> MergeReport:
        """Three-way merge of ``fork_sandbox_id``'s filesystem changes
        since its fork point back into ``source_sandbox_id`` (C2). Both
        sandboxes are paused for the window when running; apply-phase
        failures roll back path-by-path from a transient pre-merge
        snapshot and surface as ``MergeError`` carrying the report.
        ``observations`` (C3: none/append/dedupe) adopts the fork's
        journal history into the source after a successful merge."""
        if policy not in MERGE_POLICIES:
            raise ValueError(
                f"unknown merge policy: {policy!r} (expected one of {MERGE_POLICIES})"
            )
        if observations not in OBSERVATION_POLICIES:
            raise ValueError(
                f"unknown observation policy: {observations!r} "
                f"(expected one of {OBSERVATION_POLICIES})"
            )
        prefixes = (
            DEFAULT_MERGE_IGNORE_PREFIXES if ignore_prefixes is None else tuple(ignore_prefixes)
        )
        origin = self._fork_origin(fork_sandbox_id)
        if origin is None:
            raise ValueError(
                f"no fork_created journal marker for {fork_sandbox_id}; "
                "merge_from_fork only works for sandboxes created by fork_once "
                "with the action journal enabled"
            )
        marker_source, base_checkpoint_id = origin
        if marker_source != source_sandbox_id:
            raise ValueError(
                f"{fork_sandbox_id} is not a fork of {source_sandbox_id} "
                f"(fork_created marker names {marker_source})"
            )
        with self._merge_lock:
            if self._active_merges & {source_sandbox_id, fork_sandbox_id}:
                raise MergeError(
                    f"merge already in progress involving {source_sandbox_id} or {fork_sandbox_id}"
                )
            self._active_merges.update((source_sandbox_id, fork_sandbox_id))
        operation = start_operation(
            self.telemetry,
            "merge.flow",
            self._telemetry_attrs(
                source_sandbox_id,
                component="system",
                checkpoint_id=base_checkpoint_id,
                extra={"fork_sandbox_id": str(fork_sandbox_id), "policy": policy},
            ),
        )
        paused: list[SandboxId] = []
        merge_snapshot_id: CheckpointId | None = None
        try:
            if self._txn_active(source_sandbox_id) or self._txn_active(fork_sandbox_id):
                raise MergeError(
                    "transaction active on source or fork; commit or abort it before merging"
                )
            # Quiesce: no writer may race the diffs or the apply window.
            for sandbox_id in (fork_sandbox_id, source_sandbox_id):
                if self._pause_for_merge(sandbox_id):
                    paused.append(sandbox_id)
            fork_result = self.changeset_since(fork_sandbox_id, base_checkpoint_id)
            try:
                source_result = self.changeset_since(source_sandbox_id, base_checkpoint_id)
            except FileNotFoundError as exc:
                raise MergeError(
                    f"source lost its fork-point snapshot ({exc}); "
                    "retention pruned the merge base"
                ) from exc
            source_root = Path(self.runtime.rootfs_path_for(source_sandbox_id))
            fork_root = Path(self.runtime.rootfs_path_for(fork_sandbox_id))
            fork_base_filesystem_id = self._filesystem_restore_source_id(
                fork_sandbox_id, base_checkpoint_id
            )
            base_root = Path(
                self.runtime.snapshot_content_root(
                    fork_sandbox_id, fork_base_filesystem_id
                )
            )
            plan = plan_merge(
                fork_entries=fork_result.entries,
                source_entries=source_result.entries,
                policy=policy,
                fork_root=fork_root,
                source_root=source_root,
                base_root=base_root,
                ignore_prefixes=prefixes,
                merger=merger,
            )
            if plan.ops:
                merge_snapshot_id = CheckpointId(f"merge-{uuid.uuid4().hex[:8]}")
                status = self.runtime.checkpoint_filesystem(source_sandbox_id, merge_snapshot_id)
                if getattr(status, "executed", True) is False:
                    raise MergeError(
                        f"pre-merge snapshot failed: {getattr(status, 'reason', None)}"
                    )
                undo_root = Path(
                    self.runtime.snapshot_content_root(source_sandbox_id, merge_snapshot_id)
                )
                try:
                    apply_plan(
                        plan,
                        source_root=source_root,
                        fork_root=fork_root,
                        undo_root=undo_root,
                    )
                except MergeApplyError as exc:
                    report = build_report(
                        source_sandbox_id=source_sandbox_id,
                        fork_sandbox_id=fork_sandbox_id,
                        base_checkpoint_id=base_checkpoint_id,
                        policy=policy,
                        plan=plan,
                        applied=False,
                        rolled_back=exc.rolled_back,
                    )
                    self._record_merge(report, succeeded=False)
                    raise MergeError(str(exc), report=report) from exc
            report = build_report(
                source_sandbox_id=source_sandbox_id,
                fork_sandbox_id=fork_sandbox_id,
                base_checkpoint_id=base_checkpoint_id,
                policy=policy,
                plan=plan,
                applied=True,
            )
            if observations != "none" and not plan.aborted:
                # A merge that landed adopts the fork's history on
                # request; consolidation failures never unwind the
                # already-applied filesystem merge.
                try:
                    observation_report = self.consolidate_observations(
                        source_sandbox_id,
                        fork_sandbox_id,
                        policy=observations,
                        summarizer=observation_summarizer,
                        reason="merge",
                    )
                    report = replace(report, observations=observation_report)
                except Exception:
                    logger.exception(
                        "Observation consolidation after merge failed source=%s fork=%s",
                        source_sandbox_id,
                        fork_sandbox_id,
                    )
            self._record_merge(report, succeeded=True)
            operation.finish(
                status="succeeded",
                attributes={
                    "applied": len(report.applied),
                    "conflicted": len(report.conflicted),
                    "skipped": len(report.skipped),
                    "aborted": plan.aborted,
                },
            )
            return report
        except Exception:
            operation.finish(status="failed")
            raise
        finally:
            if merge_snapshot_id is not None:
                try:
                    self.runtime.discard_partial_checkpoint(source_sandbox_id, merge_snapshot_id)
                except Exception:
                    logger.exception(
                        "Failed to discard transient merge snapshot sandbox=%s snapshot=%s",
                        source_sandbox_id,
                        merge_snapshot_id,
                    )
            for sandbox_id in paused:
                try:
                    self.runtime.resume(sandbox_id)
                except Exception:
                    logger.exception("Failed to resume after merge sandbox=%s", sandbox_id)
            with self._merge_lock:
                self._active_merges.difference_update((source_sandbox_id, fork_sandbox_id))

    def _record_merge(self, report: MergeReport, *, succeeded: bool) -> None:
        self._journal_lifecycle(
            report.source_sandbox_id,
            "merge",
            metadata={
                "fork_sandbox_id": str(report.fork_sandbox_id),
                "base_checkpoint_id": str(report.base_checkpoint_id),
                "policy": report.policy,
                "applied": len(report.applied),
                "conflicted": len(report.conflicted),
                "skipped": len(report.skipped),
                "rolled_back": report.rolled_back,
                "succeeded": succeeded,
            },
        )
        self._journal_lifecycle(
            report.fork_sandbox_id,
            "merged_into",
            metadata={
                "source_sandbox_id": str(report.source_sandbox_id),
                "base_checkpoint_id": str(report.base_checkpoint_id),
                "policy": report.policy,
                "succeeded": succeeded,
            },
        )
        self.telemetry.emit_event(
            "merge.completed",
            self._telemetry_attrs(
                report.source_sandbox_id,
                component="system",
                checkpoint_id=report.base_checkpoint_id,
                extra={
                    "fork_sandbox_id": str(report.fork_sandbox_id),
                    "policy": report.policy,
                    "applied": len(report.applied),
                    "conflicted": len(report.conflicted),
                    "skipped": len(report.skipped),
                    "rolled_back": report.rolled_back,
                    "succeeded": succeeded,
                },
            ),
        )
        logger.info(
            "Merge %s source=%s fork=%s policy=%s applied=%d conflicted=%d skipped=%d rolled_back=%s",
            "succeeded" if succeeded else "failed",
            report.source_sandbox_id,
            report.fork_sandbox_id,
            report.policy,
            len(report.applied),
            len(report.conflicted),
            len(report.skipped),
            report.rolled_back,
        )

    # ----- observation consolidation (C3) ------------------------------
    # "Observations" are journal records — the only durable, ordered,
    # transportable account of what a fork did (staged LLM responses are
    # deliver/drop-only metadata for live waiters and their bodies are
    # never persisted). Consolidation copies a fork's qualifying records
    # into the source journal as kind="observation" rows with provenance.

    _OBSERVATION_LIFECYCLE_EVENTS = frozenset(
        {"fork_created", "checkpoint", "restore", "merge", "merged_into"}
    )

    def consolidate_observations(
        self,
        source_sandbox_id: SandboxId,
        fork_sandbox_id: SandboxId,
        *,
        policy: str = "append",
        summarizer: SummarizerHook | None = None,
        reason: str = "manual",
    ) -> ObservationReport:
        """Adopt ``fork_sandbox_id``'s journal history into
        ``source_sandbox_id``'s journal (C3). ``append`` copies every
        qualifying record in fork-seq order; ``dedupe`` skips exec
        records the source produced identically itself since the fork
        point; ``none`` copies nothing (useful with ``summarizer``,
        which digests the qualifying records into one summary row).
        Merge/commit-triggered runs are idempotent per (source, fork);
        manual re-runs append again."""
        if policy not in OBSERVATION_POLICIES:
            raise ValueError(
                f"unknown observation policy: {policy!r} "
                f"(expected one of {OBSERVATION_POLICIES})"
            )
        journal = self.journal
        if journal is None:
            raise RuntimeError("observation consolidation requires the action journal")
        if reason != "manual" and self._observations_already_consolidated(
            source_sandbox_id, fork_sandbox_id
        ):
            logger.info(
                "Observations already consolidated; skipping source=%s fork=%s reason=%s",
                source_sandbox_id,
                fork_sandbox_id,
                reason,
            )
            return ObservationReport(
                source_sandbox_id=source_sandbox_id,
                fork_sandbox_id=fork_sandbox_id,
                policy=policy,
                consolidated=0,
                skipped_duplicates=0,
                already_consolidated=True,
                reason=reason,
            )
        fork_records = journal.entries(fork_sandbox_id)
        qualifying = [record for record in fork_records if self._qualifies_as_observation(record)]
        dedupe_keys: set[tuple] | None = None
        if policy == "dedupe":
            dedupe_keys = self._source_exec_keys_since_fork(source_sandbox_id, fork_sandbox_id)
        consolidated = 0
        skipped_duplicates = 0
        if policy != "none":
            for record in qualifying:
                if (
                    dedupe_keys is not None
                    and record.kind == "exec"
                    and self._exec_identity(record.payload) in dedupe_keys
                ):
                    skipped_duplicates += 1
                    continue
                journal.record_observation(
                    source_sandbox_id,
                    payload={
                        "fork_sandbox_id": str(fork_sandbox_id),
                        "origin_seq": record.seq,
                        "origin_kind": record.kind,
                        "origin_txn_id": record.txn_id,
                        "origin_started_at": record.started_at,
                        "origin_finished_at": record.finished_at,
                        "origin_payload": dict(record.payload),
                        "reason": reason,
                    },
                )
                consolidated += 1
        summary_written = False
        if summarizer is not None:
            digest = summarizer([record.to_json() for record in qualifying])
            if digest is not None:
                journal.record_observation(
                    source_sandbox_id,
                    payload={
                        "fork_sandbox_id": str(fork_sandbox_id),
                        "origin_kind": "summary",
                        "reason": reason,
                        "summary": digest,
                    },
                )
                summary_written = True
        report = ObservationReport(
            source_sandbox_id=source_sandbox_id,
            fork_sandbox_id=fork_sandbox_id,
            policy=policy,
            consolidated=consolidated,
            skipped_duplicates=skipped_duplicates,
            summary_written=summary_written,
            reason=reason,
        )
        self._journal_lifecycle(
            source_sandbox_id,
            "observations_consolidated",
            metadata={
                "fork_sandbox_id": str(fork_sandbox_id),
                "policy": policy,
                "consolidated": consolidated,
                "skipped_duplicates": skipped_duplicates,
                "summary_written": summary_written,
                "reason": reason,
            },
        )
        self.telemetry.emit_event(
            "observations.consolidated",
            self._telemetry_attrs(
                source_sandbox_id,
                component="system",
                extra={
                    "fork_sandbox_id": str(fork_sandbox_id),
                    "policy": policy,
                    "consolidated": consolidated,
                    "skipped_duplicates": skipped_duplicates,
                    "reason": reason,
                },
            ),
        )
        logger.info(
            "Consolidated observations source=%s fork=%s policy=%s consolidated=%d skipped=%d reason=%s",
            source_sandbox_id,
            fork_sandbox_id,
            policy,
            consolidated,
            skipped_duplicates,
            reason,
        )
        return report

    def _qualifies_as_observation(self, record) -> bool:
        """exec records always qualify; lifecycle records only for the
        allowlist (mechanics markers like staging_*/txn_*/changeset are
        plumbing, not observations); adopted observation rows never
        re-qualify (no recursive adoption)."""
        if record.kind == "exec":
            return True
        if record.kind == "lifecycle":
            return str(record.payload.get("event")) in self._OBSERVATION_LIFECYCLE_EVENTS
        return False

    @staticmethod
    def _exec_identity(payload: dict) -> tuple:
        return (
            tuple(str(item) for item in (payload.get("argv") or ())),
            payload.get("cwd"),
            payload.get("returncode"),
            payload.get("stdout_sha256"),
            payload.get("stderr_sha256"),
        )

    def _source_exec_keys_since_fork(
        self, source_sandbox_id: SandboxId, fork_sandbox_id: SandboxId
    ) -> set[tuple]:
        """Identity keys of the source's own exec records after the
        matching fork_source marker (whole journal when the marker is
        missing) — "the source already did this itself"."""
        journal = self.journal
        assert journal is not None
        records = journal.entries(source_sandbox_id)
        boundary = -1
        for record in records:
            if record.kind != "lifecycle":
                continue
            if record.payload.get("event") != "fork_source":
                continue
            metadata = record.payload.get("metadata") or {}
            if str(metadata.get("target_sandbox_id")) == str(fork_sandbox_id):
                boundary = record.seq
                break
        return {
            self._exec_identity(record.payload)
            for record in records
            if record.kind == "exec" and record.seq > boundary
        }

    def _observations_already_consolidated(
        self, source_sandbox_id: SandboxId, fork_sandbox_id: SandboxId
    ) -> bool:
        journal = self.journal
        if journal is None:
            return False
        try:
            records = journal.entries(source_sandbox_id, kind="lifecycle")
        except Exception:
            return False
        for record in records:
            if record.payload.get("event") != "observations_consolidated":
                continue
            metadata = record.payload.get("metadata") or {}
            if str(metadata.get("fork_sandbox_id")) == str(fork_sandbox_id):
                return True
        return False

    # ----- effect ledger (D1) ------------------------------------------

    def egress_ledger(
        self,
        sandbox_id: SandboxId,
        *,
        txn_id: str | None = None,
        since_seq: int | None = None,
    ) -> EgressLedger:
        """The sandbox's recorded egress flows (journal ``kind="egress"``
        rows), optionally scoped to one transaction. Requires the action
        journal — it is the ledger's only store."""
        journal = self.journal
        if journal is None:
            raise RuntimeError("the effect ledger requires the action journal")
        flows: list[EgressFlow] = []
        for record in journal.entries(sandbox_id, kind="egress", since_seq=since_seq):
            if txn_id is not None and record.txn_id != txn_id:
                continue
            payload = dict(record.payload)
            payload["seq"] = record.seq
            payload["txn_id"] = record.txn_id
            payload["recorded_at"] = record.finished_at or record.started_at
            # Re-derive rather than trust the stored class: classification
            # is a pure function of the row, so rule changes (and rows
            # written before classification existed) are reflected here.
            payload["classification"] = classify_flow(payload, self.egress_rules)
            flows.append(EgressFlow.from_json(payload))
        return EgressLedger(sandbox_id=sandbox_id, flows=tuple(flows), txn_id=txn_id)

    def begin_egress_replay(
        self,
        sandbox_id: SandboxId,
        *,
        policy: str = "cassette_first",
        cassette_source: SandboxId | str | None = None,
    ) -> None:
        """Serve this sandbox's recorded reads from cassettes instead of
        the network (D2). ``cassette_source`` selects whose bucket to read
        — C4 replays a fork's commands on the source, so the lookup must
        follow the fork that made the reads. Writes and encrypted flows
        always pass through regardless."""
        replayer = self.cassette_replayer
        if replayer is None:
            raise RuntimeError(
                "egress replay requires the egress proxy with recording enabled"
            )
        replayer.begin(
            sandbox_id,
            policy=policy,
            cassette_source=None if cassette_source is None else str(cassette_source),
        )
        logger.info(
            "Egress replay armed sandbox=%s policy=%s cassette_source=%s",
            sandbox_id,
            policy,
            cassette_source or sandbox_id,
        )

    def end_egress_replay(self, sandbox_id: SandboxId) -> EgressReplayReport | None:
        replayer = self.cassette_replayer
        if replayer is None:
            return None
        session = replayer.end(sandbox_id)
        if session is None:
            return None
        # Read the tallies under the replayer's lock: in-flight connections
        # may still be finishing when the window closes.
        served, missed, passed_through, hosts = replayer.snapshot(session)
        report = EgressReplayReport(
            sandbox_id=sandbox_id,
            policy=session.policy,
            cassette_source=session.cassette_source,
            served=served,
            missed=missed,
            passed_through=passed_through,
            hosts=hosts,
        )
        self.telemetry.emit_event(
            "egress_replay.completed",
            self._telemetry_attrs(
                sandbox_id,
                component="system",
                extra={
                    "policy": report.policy,
                    "served": report.served,
                    "missed": report.missed,
                    "cassette_source": report.cassette_source,
                },
            ),
        )
        logger.info(
            "Egress replay finished sandbox=%s served=%d missed=%d passed_through=%d",
            sandbox_id,
            report.served,
            report.missed,
            report.passed_through,
        )
        return report

    def _mutating_egress_in_txn(self, sandbox_id: SandboxId, txn_id: str) -> int:
        """Mutating flows this txn already fired at the world. Rolling back
        the filesystem cannot undo them, so aborts report the count
        instead of pretending the abort was total (blocking/deferring is
        D3's charter)."""
        if self.journal is None:
            return 0
        try:
            # Only flows that actually left the host count: a write the
            # effect gate held or refused (D3) never reached the world.
            return self.egress_ledger(sandbox_id, txn_id=txn_id).mutating_sent
        except Exception:
            logger.debug("Failed to read the effect ledger for %s", sandbox_id, exc_info=True)
            return 0

    # ----- effect gate wiring (D3) --------------------------------------

    def _effect_defaults(self) -> dict:
        return dict(self.effect_policy_defaults or {})

    def _resolve_effect_policy(self, isolation: str, requested: str | None) -> str:
        """Resolve a transaction's effect policy. ``isolation`` is the txn's
        (``"fork"`` or snapshot); bare forks go through
        `_resolve_standalone_fork_policy` instead — they are a third kind,
        not a third isolation."""
        defaults = self._effect_defaults()
        if requested is not None:
            return requested
        key = "fork_policy" if isolation == "fork" else "default_policy"
        return str(defaults.get(key) or ("reject" if isolation == "fork" else "allow"))

    def _resolve_standalone_fork_policy(self, requested: str | None) -> str:
        """Policy for a fork taken outside any transaction (F1).

        Deliberately separate from `_resolve_effect_policy`: the config key
        is `standalone_fork_policy` (default ``allow``, so today's forks
        keep writing), not `fork_policy`, which means fork-*backed
        transactions* and defaults to ``reject``. Only ``allow``/``reject``
        are honorable here — see `validate_standalone_fork_policy`.
        """
        if requested is not None:
            return requested
        defaults = self._effect_defaults()
        return str(defaults.get("standalone_fork_policy") or "allow")

    def validate_standalone_fork_policy(self, effects: str | None) -> str:
        """Resolve and check a bare fork's effect policy, raising before any
        fork is created.

        ``defer`` and ``seal`` are refused rather than silently degraded
        (F1 decision 5): a bare fork has no commit to flush a queue into and
        no abort for a seal to block, so honoring either name would be a
        guarantee this cannot keep.

        A gating policy also needs somewhere to gate. The effect gate only
        exists when the egress proxy is running, so an explicitly requested
        `reject` on an engine without it would pass validation and then
        gate nothing — the same hollow guarantee the refusals above avoid.
        That combination cannot raise (it would break callers who ask for
        `reject` on an unnetworked engine, e.g. in tests), so it warns
        loudly instead. The config-default path stays quiet: a deployment
        that never asked per call has not been promised anything.
        """
        policy = self._resolve_standalone_fork_policy(effects)
        if policy not in EFFECT_POLICIES:
            raise ValueError(
                f"unknown effect policy: {policy!r} (expected one of {EFFECT_POLICIES})"
            )
        if policy == "defer":
            raise ValueError(
                "effects='defer' is not supported for a bare fork: there is no "
                "commit to flush the queue into (the promotion hand-off is "
                "future work). Use 'reject' or 'allow'."
            )
        if policy == "seal":
            raise ValueError(
                "effects='seal' is not supported for a bare fork: seal makes a "
                "transaction non-abortable, and a bare fork has no abort to "
                "block. Use 'reject' or 'allow'."
            )
        if effects is not None and policy != "allow" and self.effect_gate is None:
            logger.warning(
                "fork(effects=%r) cannot be enforced: this engine has no effect "
                "gate, which requires the egress proxy "
                "(enable_egress_proxy=True with enable_sandbox_network=True). "
                "The fork's writes will NOT be gated.",
                policy,
            )
        return policy

    def arm_fork_effect_session(self, fork_id: SandboxId, policy: str) -> None:
        """Install a bare fork's effect session (F1).

        Called per fork right after it is created, so a gated fork is
        covered before anything inside it can run. ``allow`` still opens a
        session, matching the txn paths: counters stay uniform and the gate
        passes writes through.
        """
        self._arm_effect_session(
            fork_id, policy=policy, txn_id=None, isolation="standalone_fork"
        )

    def _arm_effect_session(
        self,
        sandbox_id: SandboxId,
        *,
        policy: str,
        txn_id: str | None,
        isolation: str,
    ) -> None:
        """Open the effect window for a txn, or for a bare fork
        (``txn_id=None``, F1). ``allow`` still opens a session so counters
        and `seal`-style bookkeeping stay uniform; the gate simply passes
        writes through."""
        gate = self.effect_gate
        if gate is None:
            return
        defaults = self._effect_defaults()
        rules = tuple(
            rule if isinstance(rule, EffectRule) else EffectRule.from_json(dict(rule))
            for rule in (defaults.get("rules") or ())
        )
        gate.begin(
            sandbox_id,
            policy=policy,
            on_unlisted=str(defaults.get("on_unlisted") or "reject"),
            opaque_effects=str(defaults.get("opaque_effects") or "allow"),
            rules=rules,
            txn_id=txn_id,
            max_queue_bytes=int(
                defaults.get("max_queue_bytes") or DEFAULT_MAX_QUEUE_BYTES
            ),
            max_queue_entries=int(
                defaults.get("max_queue_entries") or DEFAULT_MAX_QUEUE_ENTRIES
            ),
        )
        logger.info(
            "Effect gate armed sandbox=%s txn=%s policy=%s isolation=%s",
            sandbox_id,
            txn_id,
            policy,
            isolation,
        )

    def _flush_deferred_effects(
        self, sandbox_id: SandboxId, txn_id: str
    ) -> EffectFlushReport | None:
        """Send the queued writes, in enqueue order, one at a time.

        Runs **after** the filesystem commit and goes straight out from the
        host (not through the proxy: no REDIRECT applies, the flush is not
        re-classified, and it cannot be caught by its own gate). A failure
        never unwinds the commit — the filesystem is already committed, so
        the honest move is to report it.
        """
        gate = self.effect_gate
        if gate is None:
            return None
        queued = gate.drain(sandbox_id)
        if not queued:
            return None
        entries: list[FlushedEffect] = []
        succeeded = failed = 0
        for request in queued:
            status, error = flush_deferred_request(request)
            entries.append(
                FlushedEffect(
                    method=request.method,
                    host=request.host,
                    path=request.path,
                    status=status,
                    error=error,
                )
            )
            payload = {
                "host": request.host,
                "dst_ip": request.host,
                "dst_port": request.port,
                "scheme": "http",
                "method": request.method,
                "path": request.path,
                "bytes_out": len(request.body),
                "bytes_in": 0,
                "duration_ms": 0.0,
                "classification": "mutating",
                "effect": "flushed" if error is None else "flush_failed",
                "effect_status": status,
                "effect_error": error,
            }
            if self.journal is not None:
                try:
                    self.journal.record_egress(sandbox_id, payload=payload, txn_id=txn_id)
                except Exception:
                    logger.debug("Failed to journal a flushed effect", exc_info=True)
            if error is None:
                succeeded += 1
            else:
                failed += 1
                logger.warning(
                    "Deferred %s %s%s failed to flush after commit: %s",
                    request.method,
                    request.host,
                    request.path,
                    error,
                )
        report = EffectFlushReport(
            attempted=len(queued),
            succeeded=succeeded,
            failed=failed,
            entries=tuple(entries),
        )
        self.telemetry.emit_event(
            "effects.flushed",
            self._telemetry_attrs(
                sandbox_id,
                component="system",
                extra={
                    "txn_id": txn_id,
                    "attempted": report.attempted,
                    "succeeded": report.succeeded,
                    "failed": report.failed,
                },
            ),
        )
        return report

    def _drop_deferred_effects(self, sandbox_id: SandboxId, txn_id: str) -> int:
        """Abort path: the queue is discarded, so those writes never
        happened at all — the property ``effects="defer"`` exists for."""
        gate = self.effect_gate
        if gate is None:
            return 0
        dropped = gate.drain(sandbox_id)
        for request in dropped:
            if self.journal is None:
                break
            try:
                self.journal.record_egress(
                    sandbox_id,
                    payload={
                        "host": request.host,
                        "dst_ip": request.host,
                        "dst_port": request.port,
                        "scheme": "http",
                        "method": request.method,
                        "path": request.path,
                        "bytes_out": 0,
                        "bytes_in": 0,
                        "duration_ms": 0.0,
                        "classification": "mutating",
                        "effect": "dropped",
                    },
                    txn_id=txn_id,
                )
            except Exception:
                logger.debug("Failed to journal a dropped effect", exc_info=True)
        return len(dropped)

    def backfill_lost_effects(self) -> int:
        """Close out deferred writes whose queue died with the process.

        The queue is in-memory only, so a daemon restart loses it while the
        journal keeps the ``deferred`` rows. Left alone they look like
        writes that might still fire, which would mislead anything reading
        the ledger (``crab egress --pending``) forever. Startup therefore
        appends a terminal ``lost`` row for every deferred write that has
        no outcome yet — the journal stays append-only, no row is rewritten.

        Idempotent: a second run finds the outcomes already recorded and
        writes nothing. Returns the number of rows appended.
        """
        journal = self.journal
        if journal is None:
            return 0
        terminal = {"flushed", "flush_failed", "dropped", "lost"}
        appended = 0
        for raw_id in journal.known_sandbox_ids():
            sandbox_id = SandboxId(raw_id)
            try:
                rows = journal.entries(sandbox_id, kind="egress")
            except Exception:
                logger.debug("Could not read journal for %s", raw_id, exc_info=True)
                continue
            # Deferred writes belong to a txn, and the whole queue dies
            # together, so reconcile per txn: any deferred row beyond the
            # recorded outcomes never fired.
            pending: dict[str, list] = {}
            resolved: dict[str, int] = {}
            for record in rows:
                payload = record.payload or {}
                effect = payload.get("effect")
                if effect is None:
                    continue
                key = str(payload.get("txn_id") or record.txn_id or "")
                if effect == "deferred":
                    pending.setdefault(key, []).append(payload)
                elif effect in terminal:
                    resolved[key] = resolved.get(key, 0) + 1
            for key, queued in pending.items():
                orphans = queued[resolved.get(key, 0) :]
                for payload in orphans:
                    try:
                        journal.record_egress(
                            sandbox_id,
                            payload={
                                "host": payload.get("host"),
                                "dst_ip": payload.get("dst_ip"),
                                "dst_port": payload.get("dst_port"),
                                "scheme": payload.get("scheme", "http"),
                                "method": payload.get("method"),
                                "path": payload.get("path"),
                                "bytes_out": 0,
                                "bytes_in": 0,
                                "duration_ms": 0.0,
                                "classification": "mutating",
                                "effect": "lost",
                                "effect_reason": "queue_lost_on_restart",
                            },
                            txn_id=key or None,
                        )
                        appended += 1
                    except Exception:
                        logger.debug(
                            "Failed to mark a lost deferred write for %s",
                            raw_id,
                            exc_info=True,
                        )
        if appended:
            logger.warning(
                "Marked %d deferred write(s) as lost: their queue did not survive "
                "the last shutdown",
                appended,
            )
        return appended

    def _release_effect_session(self, sandbox_id: SandboxId) -> None:
        gate = self.effect_gate
        if gate is not None:
            gate.end(sandbox_id)

    # ----- process merge (C4) ------------------------------------------
    # The process half of consolidation. "replay" re-runs the fork's
    # journaled execs on a source that keeps its own live processes;
    # "promote" applies the source's fs changes onto the fork and then
    # promotes the fork — processes and all — onto the source's
    # identity via the shared B3 swap (lazy-pages by default); "auto"
    # resolves from a container-side process census on the source.

    def _count_source_processes(self, sandbox_id: SandboxId) -> int:
        """Container-side PID census. The probe's own shell counts, so
        the quiescent baseline is PROCESS_PROBE_BASELINE (init+probe)."""
        result = self.runtime.exec(
            sandbox_id,
            list(PROCESS_PROBE_ARGV),
            capture_output=True,
        )
        returncode = getattr(result, "returncode", 1)
        if returncode != 0:
            raise RuntimeError(
                f"process census failed for {sandbox_id}: rc={returncode} "
                f"stderr={getattr(result, 'stderr', '')!r}"
            )
        try:
            return int(str(getattr(result, "stdout", "")).strip())
        except ValueError as exc:
            raise RuntimeError(
                f"process census returned garbage for {sandbox_id}: "
                f"{getattr(result, 'stdout', '')!r}"
            ) from exc

    def merge_processes(
        self,
        source_sandbox_id: SandboxId,
        fork_sandbox_id: SandboxId,
        *,
        strategy: str = "auto",
        policy: str = "fail_fast",
        observations: str = "append",
        stop_on_deviation: bool = False,
        lazy_pages: bool = True,
        force: bool = False,
        egress_replay: str = "cassette_first",
        replay_effects: str = "reject",
    ) -> ProcessMergeReport:
        """Process-half of consolidation (C4). ``auto`` picks ``replay``
        when the source runs background processes (a promotion would
        kill them) and ``promote`` otherwise. Replay re-executes the
        fork's journal exec records on the source verbatim and reports
        deviations against the recorded outcomes. Promote brings the
        fork up to date with the source's fs changes (``policy``:
        fail_fast / prefer_incoming / prefer_existing / text_merge),
        promotes it wholesale onto the source's identity (lazy-pages
        restore unless ``lazy_pages=False``), adopts its history per
        ``observations`` and destroys it; ``force=True`` promotes over
        live source processes (they die)."""
        if strategy not in PROCESS_MERGE_STRATEGIES:
            raise ValueError(
                f"unknown process merge strategy: {strategy!r} "
                f"(expected one of {PROCESS_MERGE_STRATEGIES})"
            )
        _ = (policy, observations, lazy_pages, force)  # promote knobs (PR-C4.2)
        journal = self.journal
        if journal is None:
            raise RuntimeError("process merge requires the action journal")
        origin = self._fork_origin(fork_sandbox_id)
        if origin is None:
            raise ValueError(
                f"no fork_created journal marker for {fork_sandbox_id}; "
                "merge_processes only works for sandboxes created by fork_once "
                "with the action journal enabled"
            )
        marker_source, _base_checkpoint_id = origin
        if marker_source != source_sandbox_id:
            raise ValueError(
                f"{fork_sandbox_id} is not a fork of {source_sandbox_id} "
                f"(fork_created marker names {marker_source})"
            )
        if self._txn_active(source_sandbox_id) or self._txn_active(fork_sandbox_id):
            raise ProcessMergeConflict(
                "transaction active on source or fork; resolve it before merging processes"
            )
        source_processes = self._count_source_processes(source_sandbox_id)
        has_background = source_processes > PROCESS_PROBE_BASELINE
        resolved = strategy
        if strategy == "auto":
            resolved = "replay" if has_background else "promote"
            logger.info(
                "Process merge auto-resolved to %s (source processes=%d) source=%s fork=%s",
                resolved,
                source_processes,
                source_sandbox_id,
                fork_sandbox_id,
            )
        if resolved == "promote":
            if has_background and not force:
                raise ProcessMergeConflict(
                    f"source {source_sandbox_id} runs background processes "
                    f"(census={source_processes}); promotion would kill them — "
                    "use strategy='replay', or force=True to accept the loss"
                )
            return self._promote_processes(
                source_sandbox_id,
                fork_sandbox_id,
                base_checkpoint_id=CheckpointId(str(_base_checkpoint_id)),
                policy=policy,
                observations=observations,
                lazy_pages=lazy_pages,
                source_processes=source_processes,
            )
        operation = start_operation(
            self.telemetry,
            "process_merge.flow",
            self._telemetry_attrs(
                source_sandbox_id,
                component="system",
                extra={
                    "fork_sandbox_id": str(fork_sandbox_id),
                    "strategy": resolved,
                    "source_processes": source_processes,
                },
            ),
        )
        try:
            records = journal.entries(fork_sandbox_id, kind="exec")

            def _source_exec(argv, **kwargs):
                return self.runtime.exec(source_sandbox_id, argv, **kwargs)

            # Serve the fork's recorded reads from its own cassettes (D2):
            # the fork made the requests, the source re-runs the commands,
            # so the lookup has to follow the fork's bucket. Writes still
            # reach the world — replay is a read cache, not an effect gate.
            replay_armed = False
            if egress_replay != "none" and self.cassette_replayer is not None:
                try:
                    self.begin_egress_replay(
                        source_sandbox_id,
                        policy=egress_replay,
                        cassette_source=fork_sandbox_id,
                    )
                    replay_armed = True
                except Exception:
                    logger.exception(
                        "Failed to arm egress replay for %s; replaying live",
                        source_sandbox_id,
                    )
            # Replayed commands must not re-fire the fork's writes: the
            # fork already sent them, so a second POST is the least
            # deterministic thing replay could do (D3 decision 11). The
            # session goes on the SOURCE — it is what re-executes.
            effects_armed = False
            if replay_effects != "none" and self.effect_gate is not None:
                try:
                    self._arm_effect_session(
                        source_sandbox_id,
                        policy=replay_effects,
                        txn_id=f"replay-{fork_sandbox_id}",
                        isolation="replay",
                    )
                    effects_armed = True
                except Exception:
                    logger.exception(
                        "Failed to arm the effect gate for replay on %s; writes will "
                        "re-fire",
                        source_sandbox_id,
                    )
            try:
                entries, stopped_early = replay_fork_execs(
                    _source_exec, records, stop_on_deviation=stop_on_deviation
                )
            finally:
                replay_report = (
                    self.end_egress_replay(source_sandbox_id) if replay_armed else None
                )
                if effects_armed:
                    self._release_effect_session(source_sandbox_id)
            deviations = sum(1 for entry in entries if entry.deviated)
            report = ProcessMergeReport(
                source_sandbox_id=source_sandbox_id,
                fork_sandbox_id=fork_sandbox_id,
                strategy="replay",
                source_processes=source_processes,
                replayed=tuple(entries),
                deviations=deviations,
                stopped_early=stopped_early,
                egress_replay=replay_report,
            )
            self._journal_lifecycle(
                source_sandbox_id,
                "process_replay",
                metadata={
                    "fork_sandbox_id": str(fork_sandbox_id),
                    "replayed": len(entries),
                    "deviations": deviations,
                    "stopped_early": stopped_early,
                },
            )
            self.telemetry.emit_event(
                "process_merge.completed",
                self._telemetry_attrs(
                    source_sandbox_id,
                    component="system",
                    extra={
                        "fork_sandbox_id": str(fork_sandbox_id),
                        "strategy": "replay",
                        "replayed": len(entries),
                        "deviations": deviations,
                        "stopped_early": stopped_early,
                    },
                ),
            )
            logger.info(
                "Replayed fork execs source=%s fork=%s replayed=%d deviations=%d stopped_early=%s",
                source_sandbox_id,
                fork_sandbox_id,
                len(entries),
                deviations,
                stopped_early,
            )
            operation.finish(
                status="succeeded",
                attributes={"replayed": len(entries), "deviations": deviations},
            )
            return report
        except Exception:
            operation.finish(status="failed")
            raise

    def _apply_source_changes_to_fork(
        self,
        source_id: SandboxId,
        fork_id: SandboxId,
        base_checkpoint_id: CheckpointId,
        *,
        policy: str,
    ) -> tuple[int, int]:
        """Promotion prep (C4): bring the fork up to date with the
        source's filesystem changes since the fork point so nothing of
        the source's work is lost when the fork takes over. The C2
        plan/apply engine runs with swapped roots; ``prefer_incoming``
        keeps the source's version on conflicts, ``prefer_existing`` the
        fork's. Returns ``(applied, conflicted)``; ``fail_fast``
        conflicts raise ProcessMergeConflict before any write, apply
        failures roll back path-level from a transient snapshot on the
        fork's dataset."""
        engine_policy = PROMOTION_POLICIES.get(policy)
        if engine_policy is None:
            raise ValueError(
                f"unknown promotion policy: {policy!r} "
                f"(expected one of {tuple(PROMOTION_POLICIES)})"
            )
        try:
            incoming = self.changeset_since(source_id, base_checkpoint_id)
        except FileNotFoundError as exc:
            raise ProcessMergeConflict(
                f"source lost its fork-point snapshot ({exc}); "
                "cannot compute the changes to carry over"
            ) from exc
        if not incoming.entries:
            return 0, 0
        existing = self.changeset_since(fork_id, base_checkpoint_id)
        source_root = Path(self.runtime.rootfs_path_for(source_id))
        fork_root = Path(self.runtime.rootfs_path_for(fork_id))
        source_base_filesystem_id = self._filesystem_restore_source_id(
            source_id, base_checkpoint_id
        )
        base_root = Path(
            self.runtime.snapshot_content_root(
                source_id, source_base_filesystem_id
            )
        )
        # Direction swap: the "fork side" of the engine is whoever's
        # changes are being applied — here the source's, onto the fork.
        plan = plan_merge(
            fork_entries=incoming.entries,
            source_entries=existing.entries,
            policy=engine_policy,
            fork_root=source_root,
            source_root=fork_root,
            base_root=base_root,
        )
        if plan.aborted:
            raise ProcessMergeConflict(
                f"promotion blocked: {len(plan.conflicted)} conflicting path(s) between "
                "the source's and the fork's changes; use policy='prefer_incoming' or "
                "'prefer_existing', or resolve manually"
            )
        applied = len(plan.entries_to_apply)
        conflicted = len(plan.conflicted)
        if plan.ops:
            snapshot_id = CheckpointId(f"merge-{uuid.uuid4().hex[:8]}")
            status = self.runtime.checkpoint_filesystem(fork_id, snapshot_id)
            if getattr(status, "executed", True) is False:
                raise ProcessMergeConflict(
                    f"pre-apply snapshot failed on the fork: {getattr(status, 'reason', None)}"
                )
            try:
                undo_root = Path(self.runtime.snapshot_content_root(fork_id, snapshot_id))
                try:
                    apply_plan(
                        plan,
                        source_root=fork_root,
                        fork_root=source_root,
                        undo_root=undo_root,
                    )
                except MergeApplyError as exc:
                    raise ProcessMergeConflict(
                        f"promotion fs apply failed and was rolled back: {exc}"
                    ) from exc
            finally:
                try:
                    self.runtime.discard_partial_checkpoint(fork_id, snapshot_id)
                except Exception:
                    logger.exception(
                        "Failed to discard promotion prep snapshot fork=%s", fork_id
                    )
        return applied, conflicted

    def _promote_processes(
        self,
        source_id: SandboxId,
        fork_id: SandboxId,
        *,
        base_checkpoint_id: CheckpointId,
        policy: str,
        observations: str,
        lazy_pages: bool,
        source_processes: int,
    ) -> ProcessMergeReport:
        operation = start_operation(
            self.telemetry,
            "process_merge.flow",
            self._telemetry_attrs(
                source_id,
                component="system",
                extra={
                    "fork_sandbox_id": str(fork_id),
                    "strategy": "promote",
                    "source_processes": source_processes,
                    "lazy_pages": lazy_pages,
                },
            ),
        )
        paused: list[SandboxId] = []
        try:
            # Quiesce both sides for the reverse apply window (the swap
            # afterwards dumps the live fork and stops the source itself).
            for sandbox_id in (fork_id, source_id):
                if self._pause_for_merge(sandbox_id):
                    paused.append(sandbox_id)
            fs_applied, fs_conflicted = self._apply_source_changes_to_fork(
                source_id, fork_id, base_checkpoint_id, policy=policy
            )
            for sandbox_id in paused:
                try:
                    self.runtime.resume(sandbox_id)
                except Exception:
                    logger.exception("Failed to resume before promotion sandbox=%s", sandbox_id)
            paused = []
            promoted_checkpoint_id, fork_retained = self._promote_fork_onto_source(
                source_id, fork_id, lazy_pages=lazy_pages
            )
            # Adopt the fork's history (B3 commit semantics: the fork is
            # consumed); failures never unwind the completed swap.
            observation_report = None
            if observations != "none":
                try:
                    observation_report = self.consolidate_observations(
                        source_id, fork_id, policy=observations, reason="process_merge"
                    )
                except Exception:
                    logger.exception(
                        "Observation consolidation after promotion failed source=%s fork=%s",
                        source_id,
                        fork_id,
                    )
            self._journal_lifecycle(
                source_id,
                "process_promote",
                metadata={
                    "fork_sandbox_id": str(fork_id),
                    "promoted_checkpoint_id": str(promoted_checkpoint_id),
                    "fs_applied": fs_applied,
                    "fs_conflicted": fs_conflicted,
                    "lazy_pages": lazy_pages,
                    "fork_retained": fork_retained,
                },
            )
            self.telemetry.emit_event(
                "process_merge.completed",
                self._telemetry_attrs(
                    source_id,
                    component="system",
                    checkpoint_id=promoted_checkpoint_id,
                    extra={
                        "fork_sandbox_id": str(fork_id),
                        "strategy": "promote",
                        "fs_applied": fs_applied,
                        "fs_conflicted": fs_conflicted,
                        "lazy_pages": lazy_pages,
                    },
                ),
            )
            if not fork_retained and self._fork_txn_destroy is not None:
                try:
                    self._fork_txn_destroy(fork_id)
                except Exception:
                    logger.exception("Failed to destroy fork after promotion fork=%s", fork_id)
            logger.info(
                "Promoted fork %s onto %s at %s (fs_applied=%d, lazy_pages=%s)",
                fork_id,
                source_id,
                promoted_checkpoint_id,
                fs_applied,
                lazy_pages,
            )
            report = ProcessMergeReport(
                source_sandbox_id=source_id,
                fork_sandbox_id=fork_id,
                strategy="promote",
                source_processes=source_processes,
                promoted_checkpoint_id=str(promoted_checkpoint_id),
                fs_applied=fs_applied,
                fs_conflicted=fs_conflicted,
                observations=observation_report,
            )
            operation.finish(
                status="succeeded",
                attributes={"checkpoint_id": str(promoted_checkpoint_id)},
            )
            return report
        except Exception:
            operation.finish(status="failed")
            raise
        finally:
            for sandbox_id in paused:
                try:
                    self.runtime.resume(sandbox_id)
                except Exception:
                    logger.exception("Failed to resume after promotion sandbox=%s", sandbox_id)

    # ----- observation staging (B1) -----------------------------------
    # Thin facade over the response-gate registry's staging extension so
    # the B2 transaction API has one system-level surface to drive. Each
    # transition also lands in the action journal for the C3/C4 audit
    # trail.

    def begin_observation_staging(self, sandbox_id: SandboxId) -> None:
        registry = self.response_gate_registry
        if registry is None:
            raise RuntimeError("response gate registry is not configured")
        registry.begin_staging(sandbox_id)
        self._journal_lifecycle(sandbox_id, "staging_begin")

    def release_staged_observations(self, sandbox_id: SandboxId) -> int:
        """Commit path: deliver everything staged."""
        registry = self.response_gate_registry
        if registry is None:
            raise RuntimeError("response gate registry is not configured")
        released = registry.release_staged(sandbox_id)
        self._journal_lifecycle(
            sandbox_id, "staging_commit", metadata={"released": released}
        )
        return released

    def discard_staged_observations(self, sandbox_id: SandboxId) -> int:
        """Abort path: drop everything staged (callers get 409)."""
        registry = self.response_gate_registry
        if registry is None:
            raise RuntimeError("response gate registry is not configured")
        discarded = registry.discard_staged(sandbox_id)
        self._journal_lifecycle(
            sandbox_id, "staging_abort", metadata={"discarded": discarded}
        )
        return discarded

    def end_observation_staging(self, sandbox_id: SandboxId) -> int:
        """Disarm; leftovers are delivered (fail-open)."""
        registry = self.response_gate_registry
        if registry is None:
            raise RuntimeError("response gate registry is not configured")
        leftover = registry.end_staging(sandbox_id)
        self._journal_lifecycle(
            sandbox_id, "staging_end", metadata={"delivered_leftover": leftover}
        )
        return leftover

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
        if self._txn_active(sandbox_id):
            logger.debug("DIAG.coord.check.txn_active sandbox=%s", sandbox_id)
            return False
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

    def _ensure_current_restore_state(
        self,
    ) -> tuple[
        Lock,
        dict[
            SandboxId,
            tuple[CheckpointId, CheckpointId | None, CheckpointId | None],
        ],
    ]:
        """Return restore-cursor state, tolerating lightweight test adapters.

        Production instances always run ``__post_init__``. A few embedders and
        unit fixtures intentionally construct ``CrabSystem`` with ``__new__``
        and attach only the collaborators needed by the method under test, so
        keep this new bookkeeping lazily initializable for compatibility.
        """
        lock = getattr(self, "_current_restore_lock", None)
        sources = getattr(self, "_current_restore_sources", None)
        if lock is None or sources is None:
            lock = Lock()
            sources = {}
            self._current_restore_lock = lock
            self._current_restore_sources = sources
        return lock, sources

    def _remember_current_restore_sources(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
        process_checkpoint_id: CheckpointId,
        filesystem_checkpoint_id: CheckpointId,
    ) -> None:
        lock, sources = self._ensure_current_restore_state()
        with lock:
            sources[sandbox_id] = (
                CheckpointId(str(checkpoint_id)),
                CheckpointId(str(process_checkpoint_id)),
                CheckpointId(str(filesystem_checkpoint_id)),
            )

    def _mark_current_restore_sources_incomplete(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> None:
        lock, sources = self._ensure_current_restore_state()
        with lock:
            sources[sandbox_id] = (
                CheckpointId(str(checkpoint_id)),
                None,
                None,
            )

    def _forget_current_restore_sources(self, sandbox_id: SandboxId) -> None:
        lock, sources = self._ensure_current_restore_state()
        with lock:
            sources.pop(sandbox_id, None)

    def _latest_restore_sources(
        self,
        sandbox_id: SandboxId,
    ) -> tuple[CheckpointManifest, CheckpointId, CheckpointId] | None:
        """Current recovery point and its physical component ids.

        Normally the current point is the newest stored manifest. Restore can
        move a live sandbox backwards while newer manifests remain on disk,
        so an in-memory cursor takes precedence until another successful
        checkpoint advances it. The cursor intentionally is not reconstructed
        after daemon restart; without a trustworthy inspector/scheduler
        baseline the requested flow forces a new full checkpoint.
        """
        lock, sources = self._ensure_current_restore_state()
        with lock:
            current_sources = sources.get(sandbox_id)
        if current_sources is not None:
            current_id, process_id, filesystem_id = current_sources
            if process_id is None or filesystem_id is None:
                return None
            try:
                candidate = self.storage.get_manifest(sandbox_id, current_id)
                process_manifest = self.storage.get_manifest(
                    sandbox_id, process_id
                )
                filesystem_manifest = self.storage.get_manifest(
                    sandbox_id, filesystem_id
                )
                if not process_manifest.process_artifacts:
                    raise ValueError(
                        f"process source {process_id} has no process artifacts"
                    )
                if not filesystem_manifest.filesystem_artifacts:
                    raise ValueError(
                        f"filesystem source {filesystem_id} has no filesystem artifacts"
                    )
            except Exception:
                logger.warning(
                    "Current restore-source cursor is not resolvable; forcing a "
                    "fresh baseline sandbox=%s checkpoint=%s",
                    sandbox_id,
                    current_id,
                    exc_info=True,
                )
                self._mark_current_restore_sources_incomplete(
                    sandbox_id, current_id
                )
                return None
            return candidate, process_id, filesystem_id

        try:
            checkpoint_ids = self.storage.list_checkpoints(sandbox_id)
        except Exception:
            return None
        for candidate_id in reversed(checkpoint_ids):
            try:
                candidate = self.storage.get_manifest(sandbox_id, candidate_id)
            except (FileNotFoundError, KeyError):
                continue
            try:
                resolved = resolve_restore_manifest(self.storage, candidate)
            except Exception:
                logger.warning(
                    "Latest checkpoint is not resolvable; forcing a fresh baseline "
                    "sandbox=%s checkpoint=%s",
                    sandbox_id,
                    candidate_id,
                    exc_info=True,
                )
                return None
            if not resolved.process_artifacts or not resolved.filesystem_artifacts:
                return None
            process_raw = resolved.metadata.get(_PROCESS_RESTORE_CHECKPOINT_ID)
            filesystem_raw = resolved.metadata.get(_FILESYSTEM_RESTORE_CHECKPOINT_ID)
            if process_raw is None or filesystem_raw is None:
                return None
            return (
                candidate,
                CheckpointId(str(process_raw)),
                CheckpointId(str(filesystem_raw)),
            )
        return None

    def _ensure_incremental_parent_available(
        self,
        sandbox_id: SandboxId,
        decision: SchedulerCheckpointDecision,
    ) -> SchedulerCheckpointDecision:
        """Downgrade an unusable incremental parent to a fresh anchor.

        The scheduler checks the runtime pre-dump before returning its
        decision.  Re-check here at the execution boundary and also require a
        stored process manifest: retention may have removed a restored source,
        or external cleanup may race the scheduler check.  Creating an anchor
        preserves the requested recovery point without emitting a manifest
        whose parent chain can never be restored.
        """
        if not decision.is_incremental_process:
            return decision
        parent_id = decision.parent_process_checkpoint_id
        reset_reason: str | None = None
        if parent_id is None:
            reset_reason = "missing_parent_id"
        else:
            try:
                parent_manifest = self.storage.get_manifest(
                    sandbox_id, parent_id
                )
            except Exception:
                reset_reason = "missing_parent_manifest"
            else:
                if not parent_manifest.process_artifacts:
                    reset_reason = "missing_parent_process_artifacts"
            if reset_reason is None:
                try:
                    parent_path_raw = self.runtime.pre_dump_location(
                        sandbox_id, parent_id
                    )
                    if not parent_path_raw or not Path(parent_path_raw).is_dir():
                        reset_reason = "missing_pre_dump"
                except Exception:
                    reset_reason = "missing_pre_dump"

        if reset_reason is None:
            return decision
        metadata = dict(decision.metadata)
        metadata.pop("incremental_process", None)
        metadata["incremental_chain_role"] = "anchor"
        metadata["process_chain_length_after"] = 0
        metadata["incremental_parent_reset_reason"] = reset_reason
        if parent_id is not None:
            metadata["incremental_parent_candidate"] = str(parent_id)
        logger.warning(
            "Incremental parent is unusable at checkpoint execution; forcing "
            "a fresh anchor sandbox=%s checkpoint=%s reason=%s",
            sandbox_id,
            parent_id,
            reset_reason,
        )
        return replace(
            decision,
            is_incremental_process=False,
            parent_process_checkpoint_id=None,
            produce_pre_dump=True,
            metadata=metadata,
        )

    def _stored_process_chain_length(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> int:
        """Return the persisted incremental depth of a physical process source."""
        length = 0
        cursor = checkpoint_id
        visited: set[CheckpointId] = set()
        while True:
            if cursor in visited:
                raise ValueError(
                    f"cycle in process checkpoint chain at {cursor}"
                )
            visited.add(cursor)
            manifest = self.storage.get_manifest(sandbox_id, cursor)
            if manifest.process_kind != "incremental":
                return length
            parent_id = manifest.parent_checkpoint_id
            if parent_id is None:
                raise ValueError(
                    f"incremental process checkpoint {cursor} has no parent"
                )
            length += 1
            cursor = parent_id

    def _execute_requested_checkpoint_flow(
        self,
        sandbox_id: SandboxId,
        *,
        logical_id: CheckpointId,
        leave_running: bool,
        pending_request: PendingSandboxResponse | None = None,
    ) -> CheckpointResult:
        started = utc_now()
        operation = start_operation(
            self.telemetry,
            "checkpoint.flow",
            self._telemetry_attrs(
                sandbox_id,
                component="system",
                checkpoint_id=logical_id,
                request_id=(
                    None if pending_request is None else pending_request.request_id
                ),
                extra={"logical_checkpoint": True, "reason": "requested"},
            ),
        )
        job: CheckpointJob | None = None
        result: CheckpointResult | None = None
        flow_exception: BaseException | None = None
        physical_checkpoint_created = False
        materialization = "unknown"
        try:
            # Never overwrite or silently reuse an existing recovery point:
            # this method runs after an action, so mapping a colliding id to
            # older state would violate the requested turn boundary.
            try:
                existing = self.storage.get_manifest(sandbox_id, logical_id)
            except (FileNotFoundError, KeyError):
                existing = None
            if existing is not None:
                materialization = str(
                    existing.metadata.get(_CHECKPOINT_MATERIALIZATION, "legacy")
                )
                result = CheckpointResult(
                    job_id=JobId.new(),
                    sandbox_id=sandbox_id,
                    checkpoint_id=logical_id,
                    status=JobStatus.FAILED,
                    started_at=started,
                    finished_at=utc_now(),
                    manifest=None,
                    failure_code=FailureCode.VALIDATION_ERROR,
                    message="logical checkpoint id already exists",
                )
                return result

            restore_sources = self._latest_restore_sources(sandbox_id)
            decision = self.scheduler.query_requested_checkpoint(
                sandbox_id,
                baseline_available=restore_sources is not None,
                leave_running=leave_running,
            )
            decision = self._ensure_incremental_parent_available(
                sandbox_id, decision
            )
            checkpoint_metadata = self._build_checkpoint_metadata(
                sandbox_id, pending_request=pending_request
            )

            if not decision.should_checkpoint:
                if decision.reason != "no_change_signal" or restore_sources is None:
                    result = CheckpointResult(
                        job_id=JobId.new(),
                        sandbox_id=sandbox_id,
                        checkpoint_id=logical_id,
                        status=JobStatus.FAILED,
                        started_at=started,
                        finished_at=utc_now(),
                        manifest=None,
                        failure_code=FailureCode.VALIDATION_ERROR,
                        message=decision.reason,
                    )
                    return result

                _, process_checkpoint_id, filesystem_checkpoint_id = restore_sources
                materialization = "reused"
                finished = utc_now()
                manifest = CheckpointManifest(
                    schema_version="v1",
                    checkpoint_id=logical_id,
                    sandbox_id=sandbox_id,
                    created_at=finished,
                    runtime_name=self.runtime.name,
                    runtime_version=self.runtime.version,
                    process_artifacts=[],
                    filesystem_artifacts=[],
                    metadata={
                        "policy": decision.policy_name,
                        **decision.metadata,
                        **checkpoint_metadata,
                        _LOGICAL_CHECKPOINT: True,
                        _LOGICAL_CHECKPOINT_ID: str(logical_id),
                        _CHECKPOINT_MATERIALIZATION: materialization,
                        _PHYSICAL_CHECKPOINT_CREATED: False,
                        _PROCESS_RESTORE_CHECKPOINT_ID: str(process_checkpoint_id),
                        _FILESYSTEM_RESTORE_CHECKPOINT_ID: str(
                            filesystem_checkpoint_id
                        ),
                        "checkpoint_scope": "reused",
                        "reason": decision.reason,
                        "leave_running": bool(leave_running),
                    },
                ).with_integrity()
                self.storage.put_manifest(manifest)
                self.storage.handle_checkpoint_complete(manifest)
                self._remember_current_restore_sources(
                    sandbox_id,
                    logical_id,
                    process_checkpoint_id,
                    filesystem_checkpoint_id,
                )
                result = CheckpointResult(
                    job_id=JobId.new(),
                    sandbox_id=sandbox_id,
                    checkpoint_id=logical_id,
                    status=JobStatus.SUCCEEDED,
                    started_at=started,
                    finished_at=finished,
                    manifest=manifest,
                    message="reused_previous_physical_checkpoint",
                )
                self._journal_lifecycle(
                    sandbox_id,
                    "checkpoint",
                    metadata={
                        "checkpoint_id": str(logical_id),
                        "logical_checkpoint": True,
                        "physical_checkpoint_created": False,
                        "materialization": materialization,
                        "reason": decision.reason,
                    },
                )
                return result

            prior_process_id = None
            prior_filesystem_id = None
            if restore_sources is not None:
                _, prior_process_id, prior_filesystem_id = restore_sources
            process_checkpoint_id = (
                logical_id if decision.checkpoint_process else prior_process_id
            )
            filesystem_checkpoint_id = (
                logical_id if decision.checkpoint_filesystem else prior_filesystem_id
            )
            if process_checkpoint_id is None or filesystem_checkpoint_id is None:
                result = CheckpointResult(
                    job_id=JobId.new(),
                    sandbox_id=sandbox_id,
                    checkpoint_id=logical_id,
                    status=JobStatus.FAILED,
                    started_at=started,
                    finished_at=utc_now(),
                    manifest=None,
                    failure_code=FailureCode.VALIDATION_ERROR,
                    message="requested checkpoint has no complete restore base",
                )
                return result

            if decision.is_incremental_process:
                materialization = "incremental"
            elif decision.checkpoint_process and decision.checkpoint_filesystem:
                materialization = "full"
            elif decision.checkpoint_filesystem:
                materialization = "filesystem_only"
            else:
                materialization = "process_only"
            job = CheckpointJob(
                job_id=JobId.new(),
                sandbox_id=sandbox_id,
                requested_at=started,
                reason=decision.reason,
                checkpoint_process=decision.checkpoint_process,
                checkpoint_filesystem=decision.checkpoint_filesystem,
                leave_running=decision.leave_running,
                is_incremental_process=decision.is_incremental_process,
                parent_process_checkpoint_id=decision.parent_process_checkpoint_id,
                produce_pre_dump=decision.produce_pre_dump,
                metadata={
                    "policy": decision.policy_name,
                    **decision.metadata,
                    **checkpoint_metadata,
                    "checkpoint_id": str(logical_id),
                    _LOGICAL_CHECKPOINT: True,
                    _LOGICAL_CHECKPOINT_ID: str(logical_id),
                    _CHECKPOINT_MATERIALIZATION: materialization,
                    _PHYSICAL_CHECKPOINT_CREATED: True,
                    _PROCESS_RESTORE_CHECKPOINT_ID: str(process_checkpoint_id),
                    _FILESYSTEM_RESTORE_CHECKPOINT_ID: str(
                        filesystem_checkpoint_id
                    ),
                },
            )
            result = self.executor.submit_checkpoint(job).result()
            if result.status == JobStatus.SUCCEEDED:
                if result.manifest is None or result.checkpoint_id != logical_id:
                    result = replace(
                        result,
                        status=JobStatus.FAILED,
                        manifest=None,
                        failure_code=FailureCode.STORAGE_ERROR,
                        message=(
                            "logical checkpoint materialized with an unexpected id "
                            f"(expected={logical_id}, actual={result.checkpoint_id})"
                        ),
                    )
                    return result
                physical_checkpoint_created = True
                self._remember_current_restore_sources(
                    sandbox_id,
                    logical_id,
                    process_checkpoint_id,
                    filesystem_checkpoint_id,
                )
                self.scheduler.mark_checkpoint_complete(
                    sandbox_id,
                    result.finished_at,
                    process_checkpoint_id=(
                        result.checkpoint_id if job.checkpoint_process else None
                    ),
                    is_incremental_process=job.is_incremental_process,
                )
                self._journal_lifecycle(
                    sandbox_id,
                    "checkpoint",
                    metadata={
                        "checkpoint_id": str(logical_id),
                        "logical_checkpoint": True,
                        "physical_checkpoint_created": True,
                        "materialization": materialization,
                        "reason": decision.reason,
                        "leave_running": bool(job.leave_running),
                    },
                )
            return result
        except BaseException as exc:
            flow_exception = exc
            raise
        finally:
            if job is not None and self._should_resume_after_checkpoint(job, result):
                self._resume_sandbox(sandbox_id)
            if (
                job is not None
                and result is not None
                and result.status == JobStatus.SUCCEEDED
            ):
                self.inspector.mark_checkpoint_complete(
                    sandbox_id,
                    process=job.checkpoint_process,
                    filesystem=job.checkpoint_filesystem,
                    at=result.finished_at,
                )
            if result is not None:
                operation.finish(
                    status=result.status.value,
                    attributes={
                        "checkpoint_id": str(logical_id),
                        "job_id": str(result.job_id),
                        "materialization": materialization,
                        "physical_checkpoint_created": physical_checkpoint_created,
                        "failure_code": result.failure_code.value,
                    },
                )
            else:
                operation.finish(
                    status="failed",
                    attributes={
                        "checkpoint_id": str(logical_id),
                        "materialization": materialization,
                        "error": (
                            type(flow_exception).__name__
                            if flow_exception is not None
                            else "unknown"
                        ),
                    },
                )

    def _execute_checkpoint_flow(
        self,
        sandbox_id: SandboxId,
        *,
        pending_request: PendingSandboxResponse | None = None,
    ) -> CheckpointResult | None:
        if self._txn_active(sandbox_id) or self._merge_active(sandbox_id):
            # Auto-checkpoints are suppressed inside a transaction (they
            # would pollute the retention chain with doomed states and
            # stage gated responses prematurely) and inside a merge
            # window (a mid-merge checkpoint would capture half-applied
            # state and reset the inspector's change cursor). Manual
            # checkpoint_once remains allowed (explicit user intent).
            logger.debug("Skipping scheduled checkpoint; txn or merge active sandbox=%s", sandbox_id)
            return None
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
        prior_restore_sources = self._latest_restore_sources(sandbox_id)
        decision = self.scheduler.query_checkpoint(sandbox_id)
        decision = self._ensure_incremental_parent_available(
            sandbox_id, decision
        )
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
                prior_process_id = (
                    None
                    if prior_restore_sources is None
                    else prior_restore_sources[1]
                )
                prior_filesystem_id = (
                    None
                    if prior_restore_sources is None
                    else prior_restore_sources[2]
                )
                materialized_process = bool(
                    result.manifest is not None
                    and result.manifest.process_artifacts
                )
                materialized_filesystem = bool(
                    result.manifest is not None
                    and result.manifest.filesystem_artifacts
                )
                current_process_id = (
                    result.checkpoint_id
                    if materialized_process
                    else prior_process_id
                )
                current_filesystem_id = (
                    result.checkpoint_id
                    if materialized_filesystem
                    else prior_filesystem_id
                )
                if (
                    current_process_id is not None
                    and current_filesystem_id is not None
                ):
                    self._remember_current_restore_sources(
                        sandbox_id,
                        result.checkpoint_id,
                        current_process_id,
                        current_filesystem_id,
                    )
                else:
                    self._mark_current_restore_sources_incomplete(
                        sandbox_id, result.checkpoint_id
                    )
                self.scheduler.mark_checkpoint_complete(
                    sandbox_id,
                    result.finished_at,
                    process_checkpoint_id=(
                        result.checkpoint_id if materialized_process else None
                    ),
                    is_incremental_process=job.is_incremental_process,
                )
                self._journal_lifecycle(
                    sandbox_id,
                    "checkpoint",
                    metadata={
                        "checkpoint_id": str(result.checkpoint_id),
                        "reason": decision.reason,
                        "leave_running": bool(job.leave_running),
                    },
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
            # Re-baseline the inspector AFTER resume so CRIU's --leave-running
            # parasite-teardown residual pages (written on thaw, after the
            # clear) do not latch a false process_changed=True. See
            # checkpoint_once for the full rationale. mark_checkpoint_complete
            # swallows its own errors, so this cannot break the symmetric
            # telemetry finish above.
            if (
                job is not None
                and result is not None
                and result.status.value == "succeeded"
            ):
                self.inspector.mark_checkpoint_complete(
                    sandbox_id,
                    process=job.checkpoint_process,
                    filesystem=job.checkpoint_filesystem,
                    at=result.finished_at,
                )

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
        if self._txn_active(sandbox_id) or self._merge_active(sandbox_id):
            # Inside a txn the coordination loop must not run at all: its
            # finally-release would prematurely move armed responses into
            # the staged buffer (losing per-request release granularity)
            # while the actual checkpoint flow is suppressed anyway. The
            # same holds for a merge window.
            logger.debug("DIAG.coord.any.txn_or_merge_active sandbox=%s", sandbox_id)
            return False
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
        destroy_filesystem_ref=runtime_impl.destroy_filesystem_ref,
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
    fs_ref_setter = getattr(storage, "set_destroy_filesystem_ref", None)
    if callable(fs_ref_setter):
        fs_ref_setter(runtime_impl.destroy_filesystem_ref)
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
