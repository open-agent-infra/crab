from __future__ import annotations

from dataclasses import replace
from datetime import datetime
import logging
from pathlib import Path
from threading import Lock
from typing import Callable, Protocol

from .config import SchedulerConfig
from .contracts import Runtime, SandboxInspector, SchedulerStateStore, TelemetrySink
from .ids import CheckpointId, SandboxId
from .models import SandboxSnapshot, SchedulerCheckpointDecision, utc_now
from .telemetry import NoopTelemetrySink, start_operation

logger = logging.getLogger(__name__)

_PREEMPTION_NOTICE_KEY = "preemption_notice"
_PREEMPTION_GRACE_SECONDS_KEY = "preemption_grace_remaining_seconds"
_TREE_SEARCH_STEP_KEY = "tree_search_step"
_TREE_SEARCH_IS_FORK_KEY = "tree_search_is_fork"


def _evaluate_change_driven_checkpoint(
    snapshot: SandboxSnapshot,
    *,
    config: SchedulerConfig,
    policy_name: str,
    leave_running: bool,
) -> SchedulerCheckpointDecision:
    changed = snapshot.process_changed or snapshot.filesystem_changed
    checkpoint_process = snapshot.process_changed
    checkpoint_filesystem = changed

    if not snapshot.is_running:
        return SchedulerCheckpointDecision(
            should_checkpoint=False,
            checkpoint_process=False,
            checkpoint_filesystem=False,
            leave_running=leave_running,
            reason="sandbox_not_running",
            policy_name=policy_name,
        )

    request_in_flight = bool(snapshot.metadata.get("llm_request_in_flight", False))

    if config.checkpoint_full_baseline_on_first_checkpoint and snapshot.last_checkpoint_at is None:
        return SchedulerCheckpointDecision(
            should_checkpoint=True,
            checkpoint_process=True,
            checkpoint_filesystem=True,
            leave_running=leave_running,
            reason="no_previous_checkpoint",
            policy_name=policy_name,
            metadata={"llm_request_in_flight": request_in_flight},
        )

    if config.require_change_signal and not changed:
        return SchedulerCheckpointDecision(
            should_checkpoint=False,
            checkpoint_process=False,
            checkpoint_filesystem=False,
            leave_running=leave_running,
            reason="no_change_signal",
            policy_name=policy_name,
        )

    if snapshot.last_checkpoint_at is None:
        if config.require_llm_request_for_checkpoint and not request_in_flight:
            return SchedulerCheckpointDecision(
                should_checkpoint=False,
                checkpoint_process=False,
                checkpoint_filesystem=False,
                leave_running=leave_running,
                reason="llm_request_required",
                policy_name=policy_name,
                metadata={"llm_request_in_flight": request_in_flight},
            )

        reason = "change_signal_and_interval_elapsed"
        if config.prefer_checkpoint_during_llm_request and request_in_flight:
            reason = "llm_request_window_available"
        return SchedulerCheckpointDecision(
            should_checkpoint=True,
            checkpoint_process=checkpoint_process,
            checkpoint_filesystem=checkpoint_filesystem,
            leave_running=leave_running,
            reason=reason,
            policy_name=policy_name,
            metadata={"llm_request_in_flight": request_in_flight},
        )

    elapsed = (snapshot.observed_at - snapshot.last_checkpoint_at).total_seconds()
    min_interval = config.min_checkpoint_interval_seconds
    force_after = config.force_checkpoint_after_seconds

    if force_after > 0 and elapsed >= force_after:
        return SchedulerCheckpointDecision(
            should_checkpoint=True,
            checkpoint_process=checkpoint_process,
            checkpoint_filesystem=checkpoint_filesystem,
            leave_running=leave_running,
            reason="force_interval_elapsed",
            policy_name=policy_name,
            metadata={"elapsed_seconds": elapsed, "llm_request_in_flight": request_in_flight},
        )

    if elapsed < min_interval:
        return SchedulerCheckpointDecision(
            should_checkpoint=False,
            checkpoint_process=False,
            checkpoint_filesystem=False,
            leave_running=leave_running,
            reason="minimum_interval_not_elapsed",
            policy_name=policy_name,
            metadata={"elapsed_seconds": elapsed, "llm_request_in_flight": request_in_flight},
        )

    if config.require_llm_request_for_checkpoint and not request_in_flight:
        return SchedulerCheckpointDecision(
            should_checkpoint=False,
            checkpoint_process=False,
            checkpoint_filesystem=False,
            leave_running=leave_running,
            reason="llm_request_required",
            policy_name=policy_name,
            metadata={"elapsed_seconds": elapsed, "llm_request_in_flight": request_in_flight},
        )

    reason = "change_signal_and_interval_elapsed"
    if config.prefer_checkpoint_during_llm_request and request_in_flight:
        reason = "llm_request_window_available"
    return SchedulerCheckpointDecision(
        should_checkpoint=True,
        checkpoint_process=checkpoint_process,
        checkpoint_filesystem=checkpoint_filesystem,
        leave_running=leave_running,
        reason=reason,
        policy_name=policy_name,
        metadata={"elapsed_seconds": elapsed, "llm_request_in_flight": request_in_flight},
    )


class SchedulerPolicy(Protocol):
    @property
    def name(self) -> str:
        ...

    def evaluate(self, snapshot: SandboxSnapshot) -> SchedulerCheckpointDecision:
        ...


def _resolve_incremental_process(
    decision: SchedulerCheckpointDecision,
    *,
    config: SchedulerConfig,
    last_process_checkpoint_id: CheckpointId | None,
    process_chain_length: int,
) -> SchedulerCheckpointDecision:
    """Layer the full-vs-incremental decision on top of any policy's
    base decision. Called by CRScheduler after the policy returns, so
    all policies (default, fault-tolerance, spot, tree) get incremental
    support uniformly when ``incremental_process_enabled`` is set.
    """
    if not config.incremental_process_enabled:
        return decision
    if not decision.should_checkpoint or not decision.checkpoint_process:
        return decision
    # The pre-dump + final-dump pair runs on every chain participant —
    # both anchors (chain root, chain reset) and incremental nodes — so the
    # next checkpoint always has a parent pre_dump dir to chain off of.
    # The chain anchor differs only in that its pre-dump has no --parent-path
    # and therefore writes the full memory snapshot.
    metadata = dict(decision.metadata)
    if last_process_checkpoint_id is None:
        # Chain root: pair with no parent on the pre-dump.
        metadata["incremental_chain_role"] = "anchor"
        metadata["process_chain_length_after"] = 0
        return replace(decision, produce_pre_dump=True, metadata=metadata)
    next_chain_length = process_chain_length + 1
    if (
        next_chain_length >= config.full_process_checkpoint_interval
        or next_chain_length >= config.max_process_chain_length
    ):
        # Chain reset: another anchor (no parent), starts a fresh chain.
        metadata["incremental_chain_role"] = "anchor"
        metadata["process_chain_length_after"] = 0
        return replace(decision, produce_pre_dump=True, metadata=metadata)
    metadata["incremental_chain_role"] = "node"
    metadata["incremental_process"] = True
    metadata["process_chain_length_after"] = next_chain_length
    return replace(
        decision,
        is_incremental_process=True,
        parent_process_checkpoint_id=last_process_checkpoint_id,
        produce_pre_dump=True,
        metadata=metadata,
    )


class InMemorySchedulerStateStore(SchedulerStateStore):
    def __init__(self) -> None:
        self._lock = Lock()
        self._last_checkpoint: dict[SandboxId, datetime] = {}
        self._last_process_checkpoint: dict[SandboxId, CheckpointId] = {}
        self._process_chain_length: dict[SandboxId, int] = {}

    def set_last_checkpoint(self, sandbox_id: SandboxId, checkpoint_time: datetime) -> None:
        with self._lock:
            self._last_checkpoint[sandbox_id] = checkpoint_time
        logger.debug(
            "Recorded last checkpoint time for sandbox %s at %s",
            sandbox_id,
            checkpoint_time.isoformat(),
        )

    def get_last_checkpoint(self, sandbox_id: SandboxId) -> datetime | None:
        with self._lock:
            return self._last_checkpoint.get(sandbox_id)

    def record_process_checkpoint(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
        *,
        is_incremental: bool,
    ) -> None:
        with self._lock:
            self._last_process_checkpoint[sandbox_id] = checkpoint_id
            if is_incremental:
                self._process_chain_length[sandbox_id] = (
                    self._process_chain_length.get(sandbox_id, 0) + 1
                )
            else:
                self._process_chain_length[sandbox_id] = 0
            chain = self._process_chain_length[sandbox_id]
        logger.debug(
            "Recorded process checkpoint sandbox=%s checkpoint=%s incremental=%s chain_length=%d",
            sandbox_id,
            checkpoint_id,
            is_incremental,
            chain,
        )

    def get_last_process_checkpoint(self, sandbox_id: SandboxId) -> CheckpointId | None:
        with self._lock:
            return self._last_process_checkpoint.get(sandbox_id)

    def get_process_chain_length(self, sandbox_id: SandboxId) -> int:
        with self._lock:
            return self._process_chain_length.get(sandbox_id, 0)

    def set_process_checkpoint_base(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
        *,
        chain_length: int,
    ) -> None:
        if chain_length < 0:
            raise ValueError("chain_length must be >= 0")
        with self._lock:
            self._last_process_checkpoint[sandbox_id] = checkpoint_id
            self._process_chain_length[sandbox_id] = chain_length
        logger.debug(
            "Set process checkpoint base sandbox=%s checkpoint=%s chain_length=%d",
            sandbox_id,
            checkpoint_id,
            chain_length,
        )


class CheckpointingPolicy:
    """
    Default checkpointing policy: checkpoint when sandbox is running and either
    - force interval elapsed since last checkpoint, or
    - change signal is present and minimum interval elapsed.
    """

    def __init__(self, config: SchedulerConfig):
        self._config = config

    @property
    def name(self) -> str:
        return "default-checkpointing"

    def evaluate(self, snapshot: SandboxSnapshot) -> SchedulerCheckpointDecision:
        return _evaluate_change_driven_checkpoint(
            snapshot,
            config=self._config,
            policy_name=self.name,
            leave_running=False,
        )


class FaultToleranceCheckpointingPolicy(CheckpointingPolicy):
    @property
    def name(self) -> str:
        return "fault-tolerance"

    def evaluate(self, snapshot: SandboxSnapshot) -> SchedulerCheckpointDecision:
        return _evaluate_change_driven_checkpoint(
            snapshot,
            config=self._config,
            policy_name=self.name,
            leave_running=True,
        )


class SpotPreemptionCheckpointingPolicy:
    def __init__(self, config: SchedulerConfig):
        self._config = config

    @property
    def name(self) -> str:
        return "spot-preemption"

    def evaluate(self, snapshot: SandboxSnapshot) -> SchedulerCheckpointDecision:
        if not snapshot.is_running:
            return SchedulerCheckpointDecision(
                should_checkpoint=False,
                checkpoint_process=False,
                checkpoint_filesystem=False,
                leave_running=False,
                reason="sandbox_not_running",
                policy_name=self.name,
            )

        notice = bool(snapshot.metadata.get(_PREEMPTION_NOTICE_KEY, False))
        if not notice:
            return SchedulerCheckpointDecision(
                should_checkpoint=False,
                checkpoint_process=False,
                checkpoint_filesystem=False,
                leave_running=False,
                reason="awaiting_preemption_notice",
                policy_name=self.name,
            )

        remaining = float(snapshot.metadata.get(_PREEMPTION_GRACE_SECONDS_KEY, 0.0))
        if remaining <= 0:
            return SchedulerCheckpointDecision(
                should_checkpoint=False,
                checkpoint_process=False,
                checkpoint_filesystem=False,
                leave_running=False,
                reason="preemption_budget_expired",
                policy_name=self.name,
                metadata={_PREEMPTION_GRACE_SECONDS_KEY: remaining},
            )

        return SchedulerCheckpointDecision(
            should_checkpoint=True,
            checkpoint_process=True,
            checkpoint_filesystem=True,
            leave_running=False,
            reason="preemption_notice_received",
            policy_name=self.name,
            metadata={_PREEMPTION_GRACE_SECONDS_KEY: remaining},
        )


class TreeSearchCheckpointingPolicy:
    def __init__(
        self,
        config: SchedulerConfig | None = None,
        *,
        skip_if_no_meaningful_delta: bool = False,
        checkpoint_forks: bool = False,
    ) -> None:
        self._config = config or SchedulerConfig()
        self._skip_if_no_meaningful_delta = bool(skip_if_no_meaningful_delta)
        self._checkpoint_forks = bool(checkpoint_forks)

    @property
    def name(self) -> str:
        return "tree-search"

    def evaluate(self, snapshot: SandboxSnapshot) -> SchedulerCheckpointDecision:
        if not snapshot.is_running:
            return SchedulerCheckpointDecision(
                should_checkpoint=False,
                checkpoint_process=False,
                checkpoint_filesystem=False,
                leave_running=False,
                reason="sandbox_not_running",
                policy_name=self.name,
            )
        step = snapshot.metadata.get(_TREE_SEARCH_STEP_KEY)
        if step is None:
            return SchedulerCheckpointDecision(
                should_checkpoint=False,
                checkpoint_process=False,
                checkpoint_filesystem=False,
                leave_running=False,
                reason="tree_search_step_missing",
                policy_name=self.name,
            )
        is_fork = bool(snapshot.metadata.get(_TREE_SEARCH_IS_FORK_KEY, False))
        if is_fork and not self._checkpoint_forks:
            return SchedulerCheckpointDecision(
                should_checkpoint=False,
                checkpoint_process=False,
                checkpoint_filesystem=False,
                leave_running=False,
                reason="tree_search_fork_disabled",
                policy_name=self.name,
                metadata={
                    _TREE_SEARCH_STEP_KEY: step,
                    _TREE_SEARCH_IS_FORK_KEY: True,
                },
            )
        if not self._skip_if_no_meaningful_delta:
            return SchedulerCheckpointDecision(
                should_checkpoint=True,
                checkpoint_process=True,
                checkpoint_filesystem=True,
                leave_running=True,
                reason="tree_search_step",
                policy_name=self.name,
                metadata={
                    _TREE_SEARCH_STEP_KEY: step,
                    _TREE_SEARCH_IS_FORK_KEY: is_fork,
                },
            )
        decision = _evaluate_change_driven_checkpoint(
            snapshot,
            config=self._config,
            policy_name=self.name,
            leave_running=True,
        )
        metadata = dict(decision.metadata)
        metadata[_TREE_SEARCH_STEP_KEY] = step
        metadata[_TREE_SEARCH_IS_FORK_KEY] = is_fork
        return replace(decision, metadata=metadata)


class CRScheduler:
    def __init__(
        self,
        config: SchedulerConfig,
        inspector: SandboxInspector,
        runtime: Runtime,
        state_store: SchedulerStateStore | None = None,
        telemetry: TelemetrySink | None = None,
        policy: SchedulerPolicy | None = None,
    ):
        self._config = config
        self._policy = policy or CheckpointingPolicy(config)
        self._inspector = inspector
        self._runtime = runtime
        self._state = state_store or InMemorySchedulerStateStore()
        self._telemetry = telemetry or NoopTelemetrySink()
        self._deactivated_lock = Lock()
        self._deactivated: set[SandboxId] = set()

    def deactivate_sandbox(self, sandbox_id: SandboxId) -> None:
        # Terminal deactivation (e.g. verification phase). query_checkpoint short-circuits
        # before any runc pause can happen, so a concurrent verify exec doesn't race with
        # scheduler pause/checkpoint.
        with self._deactivated_lock:
            self._deactivated.add(sandbox_id)
        logger.info("Scheduler deactivated sandbox %s; no further checkpoints will be scheduled", sandbox_id)

    def is_sandbox_deactivated(self, sandbox_id: SandboxId) -> bool:
        with self._deactivated_lock:
            return sandbox_id in self._deactivated

    def _resolve_incremental_decision(
        self,
        decision: SchedulerCheckpointDecision,
        sandbox_id: SandboxId,
        *,
        baseline_available: bool = True,
    ) -> SchedulerCheckpointDecision:
        """Apply incremental policy only when the runtime can honor it.

        Scheduler state can legitimately point at a standalone full process
        checkpoint: manual checkpoints, restores of legacy manifests, and a
        daemon that was upgraded from an incremental-disabled configuration
        can all establish such a base.  Those checkpoints have no ``pre_dump``
        directory and therefore cannot be used as CRIU incremental parents.
        Treat them as a request for a fresh chain anchor instead of handing an
        invalid parent to runc.

        ``baseline_available=False`` deliberately ignores any stale in-memory
        parent.  The resulting first requested checkpoint is still passed
        through the incremental resolver so it produces the pre-dump anchor
        needed by the next process-changing turn.
        """
        if (
            not self._config.incremental_process_enabled
            or not decision.should_checkpoint
            or not decision.checkpoint_process
        ):
            return decision
        try:
            supports_incremental = bool(
                self._runtime.capabilities().supports_incremental_process
            )
        except Exception:
            logger.warning(
                "Unable to read runtime incremental capability for sandbox=%s; "
                "using a standalone full checkpoint",
                sandbox_id,
                exc_info=True,
            )
            return decision
        if not supports_incremental:
            return decision

        parent_checkpoint_id = (
            self._state.get_last_process_checkpoint(sandbox_id)
            if baseline_available
            else None
        )
        process_chain_length = (
            self._state.get_process_chain_length(sandbox_id)
            if baseline_available
            else 0
        )
        reset_parent_id: CheckpointId | None = None
        if parent_checkpoint_id is not None:
            try:
                parent_path_raw = self._runtime.pre_dump_location(
                    sandbox_id, parent_checkpoint_id
                )
                parent_available = bool(
                    parent_path_raw and Path(parent_path_raw).is_dir()
                )
            except Exception:
                parent_available = False
                logger.warning(
                    "Unable to inspect incremental parent sandbox=%s checkpoint=%s; "
                    "forcing a fresh chain anchor",
                    sandbox_id,
                    parent_checkpoint_id,
                    exc_info=True,
                )
            if not parent_available:
                reset_parent_id = parent_checkpoint_id
                parent_checkpoint_id = None
                process_chain_length = 0
                logger.info(
                    "Incremental parent has no usable pre-dump; forcing a fresh "
                    "chain anchor sandbox=%s checkpoint=%s",
                    sandbox_id,
                    reset_parent_id,
                )

        resolved = _resolve_incremental_process(
            decision,
            config=self._config,
            last_process_checkpoint_id=parent_checkpoint_id,
            process_chain_length=process_chain_length,
        )
        if reset_parent_id is not None:
            metadata = dict(resolved.metadata)
            metadata["incremental_parent_reset_reason"] = "missing_pre_dump"
            metadata["incremental_parent_candidate"] = str(reset_parent_id)
            resolved = replace(resolved, metadata=metadata)
        return resolved

    def evaluate(self, sandbox: SandboxSnapshot) -> SchedulerCheckpointDecision:
        operation = start_operation(
            self._telemetry,
            "scheduler.evaluate",
            {
                "component": "scheduler",
                "sandbox_id": str(sandbox.sandbox_id),
                "policy": self._policy.name,
            },
        )
        hydrated = sandbox
        if hydrated.last_checkpoint_at is None:
            last = self._state.get_last_checkpoint(hydrated.sandbox_id)
            if last is not None:
                hydrated = replace(hydrated, last_checkpoint_at=last)
                logger.debug(
                    "Hydrated snapshot for sandbox %s with last checkpoint time %s",
                    hydrated.sandbox_id,
                    last.isoformat(),
                )
        decision = self._policy.evaluate(hydrated)
        decision = self._resolve_incremental_decision(
            decision,
            hydrated.sandbox_id,
        )
        log_fn = logger.info if decision.should_checkpoint else logger.debug
        log_fn(
            "Scheduler evaluated sandbox %s with policy=%s should_checkpoint=%s reason=%s "
            "observed_process_changed=%s observed_filesystem_changed=%s "
            "checkpoint_process=%s checkpoint_filesystem=%s leave_running=%s "
            "is_incremental_process=%s parent_process_checkpoint=%s",
            hydrated.sandbox_id,
            self._policy.name,
            decision.should_checkpoint,
            decision.reason,
            hydrated.process_changed,
            hydrated.filesystem_changed,
            decision.checkpoint_process,
            decision.checkpoint_filesystem,
            decision.leave_running,
            decision.is_incremental_process,
            decision.parent_process_checkpoint_id,
        )
        self._telemetry.emit_event(
            "scheduler.evaluate",
            {
                "sandbox_id": str(hydrated.sandbox_id),
                "policy": self._policy.name,
                "should_checkpoint": decision.should_checkpoint,
                "reason": decision.reason,
                "checkpoint_process": decision.checkpoint_process,
                "checkpoint_filesystem": decision.checkpoint_filesystem,
                "leave_running": decision.leave_running,
            },
        )
        operation.finish(
            status="succeeded",
            attributes={
                "sandbox_id": str(hydrated.sandbox_id),
                "should_checkpoint": decision.should_checkpoint,
                "reason": decision.reason,
                "checkpoint_process": decision.checkpoint_process,
                "checkpoint_filesystem": decision.checkpoint_filesystem,
                "leave_running": decision.leave_running,
            },
        )
        return decision

    def evaluate_requested(
        self,
        sandbox: SandboxSnapshot,
        *,
        baseline_available: bool,
        leave_running: bool,
    ) -> SchedulerCheckpointDecision:
        """Choose materialization scope for a user-requested recovery point.

        Requested checkpoints are not rate-limited: if the inspector reports
        a change, the logical recovery point must materialize enough state to
        represent that exact turn.  The scheduler's interval/LLM-window gates
        remain specific to automatic checkpoints.  A clean turn may reuse the
        previous physical restore sources, while the first recovery point is
        always a full baseline.
        """
        hydrated = sandbox
        if hydrated.last_checkpoint_at is None:
            last = self._state.get_last_checkpoint(hydrated.sandbox_id)
            if last is not None:
                hydrated = replace(hydrated, last_checkpoint_at=last)

        policy_name = "requested-change-driven"
        if not hydrated.is_running:
            decision = SchedulerCheckpointDecision(
                should_checkpoint=False,
                checkpoint_process=False,
                checkpoint_filesystem=False,
                leave_running=leave_running,
                reason="sandbox_not_running",
                policy_name=policy_name,
            )
        elif not baseline_available or hydrated.last_checkpoint_at is None:
            decision = SchedulerCheckpointDecision(
                should_checkpoint=True,
                checkpoint_process=True,
                checkpoint_filesystem=True,
                leave_running=leave_running,
                reason="no_previous_checkpoint",
                policy_name=policy_name,
            )
        elif not (hydrated.process_changed or hydrated.filesystem_changed):
            decision = SchedulerCheckpointDecision(
                should_checkpoint=False,
                checkpoint_process=False,
                checkpoint_filesystem=False,
                leave_running=leave_running,
                reason="no_change_signal",
                policy_name=policy_name,
            )
        else:
            # A process snapshot is only meaningful with the filesystem state
            # from the same turn. Filesystem-only changes can stay partial and
            # reuse the previous process image.
            decision = SchedulerCheckpointDecision(
                should_checkpoint=True,
                checkpoint_process=bool(hydrated.process_changed),
                checkpoint_filesystem=True,
                leave_running=leave_running,
                reason="requested_change_signal",
                policy_name=policy_name,
                metadata={
                    "observed_process_changed": bool(hydrated.process_changed),
                    "observed_filesystem_changed": bool(hydrated.filesystem_changed),
                },
            )

        decision = self._resolve_incremental_decision(
            decision,
            hydrated.sandbox_id,
            baseline_available=baseline_available,
        )
        log_fn = logger.info if decision.should_checkpoint else logger.debug
        log_fn(
            "Scheduler evaluated requested checkpoint sandbox=%s "
            "should_checkpoint=%s reason=%s checkpoint_process=%s "
            "checkpoint_filesystem=%s",
            hydrated.sandbox_id,
            decision.should_checkpoint,
            decision.reason,
            decision.checkpoint_process,
            decision.checkpoint_filesystem,
        )
        self._telemetry.emit_event(
            "scheduler.requested_checkpoint_evaluate",
            {
                "sandbox_id": str(hydrated.sandbox_id),
                "should_checkpoint": decision.should_checkpoint,
                "reason": decision.reason,
                "checkpoint_process": decision.checkpoint_process,
                "checkpoint_filesystem": decision.checkpoint_filesystem,
                "baseline_available": bool(baseline_available),
            },
        )
        return decision

    def query_checkpoint(self, sandbox_id: SandboxId) -> SchedulerCheckpointDecision:
        logger.debug("Querying scheduler for sandbox %s", sandbox_id)
        if self.is_sandbox_deactivated(sandbox_id):
            return SchedulerCheckpointDecision(
                should_checkpoint=False,
                checkpoint_process=False,
                checkpoint_filesystem=False,
                leave_running=True,
                reason="sandbox_deactivated",
                policy_name=self._policy.name,
            )
        if self._config.inspect_without_pause:
            return self._query_checkpoint_with_live_inspection(sandbox_id)
        return self._query_checkpoint_with_pause(sandbox_id)

    def query_requested_checkpoint(
        self,
        sandbox_id: SandboxId,
        *,
        baseline_available: bool,
        leave_running: bool = True,
    ) -> SchedulerCheckpointDecision:
        """Inspect and scope an explicit logical checkpoint request."""
        evaluator = lambda snapshot: self.evaluate_requested(  # noqa: E731
            snapshot,
            baseline_available=baseline_available,
            leave_running=leave_running,
        )
        if self._config.inspect_without_pause:
            return self._query_checkpoint_with_live_inspection(
                sandbox_id, evaluator=evaluator
            )
        return self._query_checkpoint_with_pause(sandbox_id, evaluator=evaluator)

    def _query_checkpoint_with_live_inspection(
        self,
        sandbox_id: SandboxId,
        *,
        evaluator: Callable[[SandboxSnapshot], SchedulerCheckpointDecision] | None = None,
    ) -> SchedulerCheckpointDecision:
        evaluate = evaluator or self.evaluate
        snapshot = self._inspector.inspect(sandbox_id)
        decision = evaluate(snapshot)
        if not decision.should_checkpoint:
            logger.debug(
                "Scheduler declined checkpoint for sandbox %s reason=%s without pausing",
                sandbox_id,
                decision.reason,
            )
            return decision
        if decision.leave_running:
            logger.info(
                "Scheduler selected live-inspection checkpoint for sandbox %s reason=%s observed_process_changed=%s "
                "observed_filesystem_changed=%s checkpoint_process=%s checkpoint_filesystem=%s leave_running=%s",
                sandbox_id,
                decision.reason,
                snapshot.process_changed,
                snapshot.filesystem_changed,
                decision.checkpoint_process,
                decision.checkpoint_filesystem,
                decision.leave_running,
            )
            return decision
        try:
            self._runtime.pause(sandbox_id)
        except Exception:
            fallback_snapshot = self._inspector.inspect(sandbox_id)
            if not fallback_snapshot.is_running:
                decision = evaluate(fallback_snapshot)
                logger.info(
                    "Skipping pause for sandbox %s because it is already not running; reason=%s",
                    sandbox_id,
                    decision.reason,
                )
                return decision
            raise
        try:
            paused_snapshot = self._inspect_after_pause(sandbox_id)
            paused_decision = evaluate(paused_snapshot)
            if not paused_decision.should_checkpoint:
                if paused_snapshot.is_running:
                    self._runtime.resume(sandbox_id)
                else:
                    self._runtime.sync_runtime_state(sandbox_id, is_running=False)
                logger.debug(
                    "Scheduler declined checkpoint for sandbox %s after live inspection reason=%s",
                    sandbox_id,
                    paused_decision.reason,
                )
            else:
                logger.info(
                    "Scheduler selected live-inspection checkpoint for sandbox %s reason=%s observed_process_changed=%s "
                    "observed_filesystem_changed=%s checkpoint_process=%s checkpoint_filesystem=%s leave_running=%s",
                    sandbox_id,
                    paused_decision.reason,
                    paused_snapshot.process_changed,
                    paused_snapshot.filesystem_changed,
                    paused_decision.checkpoint_process,
                    paused_decision.checkpoint_filesystem,
                    paused_decision.leave_running,
                )
            return paused_decision
        except Exception:
            try:
                snapshot = self._inspector.inspect(sandbox_id)
            except Exception:
                snapshot = None
            try:
                if snapshot is not None and not snapshot.is_running:
                    self._runtime.sync_runtime_state(sandbox_id, is_running=False)
                else:
                    self._runtime.resume(sandbox_id)
            except Exception:
                logger.exception("Failed to resume sandbox %s after scheduler exception", sandbox_id)
            raise

    def _query_checkpoint_with_pause(
        self,
        sandbox_id: SandboxId,
        *,
        evaluator: Callable[[SandboxSnapshot], SchedulerCheckpointDecision] | None = None,
    ) -> SchedulerCheckpointDecision:
        evaluate = evaluator or self.evaluate
        try:
            self._runtime.pause(sandbox_id)
        except Exception:
            snapshot = self._inspector.inspect(sandbox_id)
            if not snapshot.is_running:
                decision = evaluate(snapshot)
                logger.info(
                    "Skipping pause for sandbox %s because it is already not running; reason=%s",
                    sandbox_id,
                    decision.reason,
                )
                return decision
            raise
        try:
            snapshot = self._inspect_after_pause(sandbox_id)
            decision = evaluate(snapshot)
            if not decision.should_checkpoint:
                if snapshot.is_running:
                    self._runtime.resume(sandbox_id)
                else:
                    self._runtime.sync_runtime_state(sandbox_id, is_running=False)
                logger.debug(
                    "Scheduler declined checkpoint for sandbox %s reason=%s",
                    sandbox_id,
                    decision.reason,
                )
            else:
                logger.info(
                    "Scheduler selected checkpoint for sandbox %s reason=%s observed_process_changed=%s "
                    "observed_filesystem_changed=%s checkpoint_process=%s checkpoint_filesystem=%s leave_running=%s",
                    sandbox_id,
                    decision.reason,
                    snapshot.process_changed,
                    snapshot.filesystem_changed,
                    decision.checkpoint_process,
                    decision.checkpoint_filesystem,
                    decision.leave_running,
                )
            return decision
        except Exception:
            try:
                snapshot = self._inspector.inspect(sandbox_id)
            except Exception:
                snapshot = None
            try:
                if snapshot is not None and not snapshot.is_running:
                    self._runtime.sync_runtime_state(sandbox_id, is_running=False)
                else:
                    self._runtime.resume(sandbox_id)
            except Exception:
                logger.exception("Failed to resume sandbox %s after scheduler exception", sandbox_id)
            raise

    def _inspect_after_pause(self, sandbox_id: SandboxId) -> SandboxSnapshot:
        snapshot = self._inspector.inspect(sandbox_id)
        if not snapshot.is_running:
            try:
                runtime_state = self._runtime.inspect_runtime(sandbox_id)
            except Exception:
                runtime_state = None
            if runtime_state is not None and runtime_state.is_running:
                logger.debug(
                    "Using runtime state to keep sandbox %s marked running after successful pause; "
                    "inspector snapshot was stale",
                    sandbox_id,
                )
                snapshot = replace(snapshot, is_running=True)
        return snapshot

    def mark_checkpoint_complete(
        self,
        sandbox_id: SandboxId,
        at: datetime | None = None,
        *,
        process_checkpoint_id: CheckpointId | None = None,
        is_incremental_process: bool = False,
        process_chain_length: int | None = None,
    ) -> None:
        ts = at or utc_now()
        self._state.set_last_checkpoint(sandbox_id, ts)
        if process_checkpoint_id is not None:
            if process_chain_length is None:
                self._state.record_process_checkpoint(
                    sandbox_id,
                    process_checkpoint_id,
                    is_incremental=is_incremental_process,
                )
            else:
                self._state.set_process_checkpoint_base(
                    sandbox_id,
                    process_checkpoint_id,
                    chain_length=process_chain_length,
                )
        logger.info(
            "Marked checkpoint complete for sandbox %s at %s process=%s "
            "incremental=%s chain_length=%s",
            sandbox_id,
            ts.isoformat(),
            process_checkpoint_id,
            is_incremental_process,
            process_chain_length,
        )
        self._telemetry.emit_event(
            "scheduler.checkpoint_complete",
            {
                "sandbox_id": str(sandbox_id),
                "at": ts.isoformat(),
                "process_checkpoint_id": (None if process_checkpoint_id is None else str(process_checkpoint_id)),
                "is_incremental_process": is_incremental_process,
                "process_chain_length": process_chain_length,
            },
        )
