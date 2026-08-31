from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from .ids import CheckpointId, JobId, SandboxId
from .json_codec import get_json_codec


MANIFEST_SCHEMA_VERSION = "v1"
_MANIFEST_JSON_CODEC = get_json_codec("auto")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _isoformat(ts: datetime) -> str:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc).isoformat()


def _parse_ts(raw: str) -> datetime:
    return datetime.fromisoformat(raw)


class JobType(str, Enum):
    CHECKPOINT = "checkpoint"
    RESTORE = "restore"


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class SandboxState(str, Enum):
    """High-level sandbox lifecycle state surfaced to SDK users.

    ``PAUSED`` and ``STOPPED`` are deliberately distinct:

    - ``PAUSED`` — the container's cgroup is frozen (``runc pause``). The
      process state is still in memory; ``resume()`` continues it almost
      instantly.
    - ``STOPPED`` — the container's processes have been terminated
      (``runc stop``). The filesystem is preserved, but the process tree is
      gone; bring it back with ``start()`` (fresh processes) or ``restore()``
      (from a checkpoint).
    """

    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    CREATED = "created"
    KILLED = "killed"
    LOST = "lost"
    UNKNOWN = "unknown"


def sandbox_state_from_status(status: str | None) -> SandboxState:
    """Map a runtime/descriptor status string to a :class:`SandboxState`."""
    s = (status or "").strip().lower()
    if s == "running":
        return SandboxState.RUNNING
    if s == "paused":
        return SandboxState.PAUSED
    if s in ("stopped", "exited"):
        return SandboxState.STOPPED
    if s == "created":
        return SandboxState.CREATED
    if s == "killed":
        return SandboxState.KILLED
    if s == "lost":
        return SandboxState.LOST
    if s == "missing":
        return SandboxState.KILLED
    return SandboxState.UNKNOWN


@dataclass(frozen=True)
class SandboxInfo:
    """Snapshot of a sandbox's lifecycle state returned by ``Sandbox.describe()``."""

    sandbox_id: str
    state: SandboxState
    status: str | None = None
    runtime_status: str | None = None
    pid: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ArtifactKind(str, Enum):
    PROCESS = "process"
    FILESYSTEM = "filesystem"
    METADATA = "metadata"


class FailureCode(str, Enum):
    NONE = "none"
    NOT_IMPLEMENTED = "not_implemented"
    RUNTIME_ERROR = "runtime_error"
    STORAGE_ERROR = "storage_error"
    VALIDATION_ERROR = "validation_error"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


class EBPFEventKind(str, Enum):
    PROCESS_EXEC = "process_exec"
    PROCESS_EXIT = "process_exit"
    FILE_WRITE = "file_write"
    FILE_DELETE = "file_delete"
    NETWORK_INGRESS = "network_ingress"
    NETWORK_EGRESS = "network_egress"


@dataclass(frozen=True)
class RuntimeCapabilities:
    supports_process_checkpoint: bool
    supports_filesystem_checkpoint: bool
    supports_incremental_filesystem: bool = False
    supports_custom_checkpoint_dir: bool = False
    supports_incremental_process: bool = False
    supports_lazy_restore: bool = False


@dataclass(frozen=True)
class RuntimeOperationStatus:
    executed: bool
    reason: str
    command: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SandboxRuntimeState:
    sandbox_id: SandboxId
    runtime_name: str
    status: str
    pid: int | None = None
    bundle_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_running(self) -> bool:
        return self.status.lower() in {"running", "paused", "created"} and self.pid is not None


@dataclass(frozen=True)
class SandboxExecResult:
    args: tuple[str, ...]
    returncode: int
    stdout: str = ""
    stderr: str = ""


@dataclass(frozen=True)
class ExecEvent:
    """A single streaming exec event: output on a channel."""
    channel: str  # "stdout" or "stderr"
    text: str


@dataclass(frozen=True)
class ExecDone:
    """Terminal event from streaming exec: the process exited."""
    returncode: int


@dataclass(frozen=True)
class PortAllocation:
    """A port exposure allocation (S4): maps guest_port to host_port."""
    host_port: int
    guest_port: int
    url: str


@dataclass(frozen=True)
class EBPFEvent:
    sandbox_id: SandboxId
    kind: EBPFEventKind
    observed_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ArtifactPayload:
    kind: ArtifactKind
    name: str
    data: bytes
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ArtifactReference:
    kind: ArtifactKind
    name: str
    relative_path: str
    size_bytes: int
    sha256: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "name": self.name,
            "relative_path": self.relative_path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ArtifactReference":
        return cls(
            kind=ArtifactKind(raw["kind"]),
            name=str(raw["name"]),
            relative_path=str(raw["relative_path"]),
            size_bytes=int(raw["size_bytes"]),
            sha256=str(raw["sha256"]),
            metadata=dict(raw.get("metadata", {})),
        )


@dataclass(frozen=True)
class WorkerStepResult:
    success: bool
    artifacts: list[ArtifactPayload] = field(default_factory=list)
    operation_status: RuntimeOperationStatus = field(
        default_factory=lambda: RuntimeOperationStatus(executed=False, reason="unknown")
    )
    failure_code: FailureCode = FailureCode.NONE
    message: str | None = None


@dataclass(frozen=True)
class CheckpointManifest:
    schema_version: str
    checkpoint_id: CheckpointId
    sandbox_id: SandboxId
    created_at: datetime
    runtime_name: str
    runtime_version: str | None
    process_artifacts: list[ArtifactReference]
    filesystem_artifacts: list[ArtifactReference]
    metadata: dict[str, Any] = field(default_factory=dict)
    integrity: dict[str, str] = field(default_factory=dict)
    parent_checkpoint_id: CheckpointId | None = None
    process_kind: str = "full"

    def validate_schema(self) -> None:
        if self.schema_version != MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported manifest schema: {self.schema_version} "
                f"(expected {MANIFEST_SCHEMA_VERSION})"
            )

    def with_integrity(self) -> "CheckpointManifest":
        payload_hash = self.compute_manifest_hash()
        return replace(self, integrity={"manifest_sha256": payload_hash})

    def validate_integrity(self) -> None:
        expected = self.integrity.get("manifest_sha256")
        if not expected:
            raise ValueError("manifest integrity missing manifest_sha256")
        actual = self.compute_manifest_hash()
        if expected != actual:
            raise ValueError("manifest integrity hash mismatch")

    def compute_manifest_hash(self) -> str:
        canonical = self.to_canonical_json_bytes()
        return hashlib.sha256(canonical).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return self._payload_dict(include_integrity=True)

    def to_canonical_json_bytes(self) -> bytes:
        return _MANIFEST_JSON_CODEC.dumps_bytes(
            self._payload_dict(include_integrity=False),
            sort_keys=True,
        )

    def to_json_bytes(self) -> bytes:
        return _MANIFEST_JSON_CODEC.dumps_bytes(
            self._payload_dict(include_integrity=True),
            sort_keys=True,
        )

    def _payload_dict(self, *, include_integrity: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "checkpoint_id": str(self.checkpoint_id),
            "sandbox_id": str(self.sandbox_id),
            "created_at": _isoformat(self.created_at),
            "runtime_name": self.runtime_name,
            "runtime_version": self.runtime_version,
            "process_artifacts": [a.to_dict() for a in self.process_artifacts],
            "filesystem_artifacts": [a.to_dict() for a in self.filesystem_artifacts],
            "metadata": self.metadata,
        }
        # Only emit incremental fields when they hold non-default values so
        # legacy (v1, full-checkpoint-only) manifests round-trip with their
        # original integrity hash unchanged.
        if self.parent_checkpoint_id is not None:
            payload["parent_checkpoint_id"] = str(self.parent_checkpoint_id)
        if self.process_kind != "full":
            payload["process_kind"] = self.process_kind
        if include_integrity:
            payload["integrity"] = self.integrity
        return payload

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "CheckpointManifest":
        parent_raw = raw.get("parent_checkpoint_id")
        manifest = cls(
            schema_version=str(raw["schema_version"]),
            checkpoint_id=CheckpointId(str(raw["checkpoint_id"])),
            sandbox_id=SandboxId(str(raw["sandbox_id"])),
            created_at=_parse_ts(str(raw["created_at"])),
            runtime_name=str(raw["runtime_name"]),
            runtime_version=(None if raw.get("runtime_version") is None else str(raw.get("runtime_version"))),
            process_artifacts=[
                ArtifactReference.from_dict(x) for x in raw.get("process_artifacts", [])
            ],
            filesystem_artifacts=[
                ArtifactReference.from_dict(x) for x in raw.get("filesystem_artifacts", [])
            ],
            metadata=dict(raw.get("metadata", {})),
            integrity=dict(raw.get("integrity", {})),
            parent_checkpoint_id=(None if parent_raw is None else CheckpointId(str(parent_raw))),
            process_kind=str(raw.get("process_kind", "full")),
        )
        manifest.validate_schema()
        manifest.validate_integrity()
        return manifest


@dataclass(frozen=True)
class CheckpointJob:
    job_id: JobId
    sandbox_id: SandboxId
    requested_at: datetime
    reason: str = "manual"
    checkpoint_process: bool = True
    checkpoint_filesystem: bool = True
    leave_running: bool = False
    is_incremental_process: bool = False
    parent_process_checkpoint_id: CheckpointId | None = None
    produce_pre_dump: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.checkpoint_process and not self.checkpoint_filesystem:
            raise ValueError("checkpoint job must include at least one checkpoint scope")
        if self.is_incremental_process and not self.checkpoint_process:
            raise ValueError("is_incremental_process requires checkpoint_process=True")
        if self.is_incremental_process and self.parent_process_checkpoint_id is None:
            raise ValueError("is_incremental_process requires parent_process_checkpoint_id")
        if self.is_incremental_process and not self.produce_pre_dump:
            raise ValueError("is_incremental_process requires produce_pre_dump=True")
        if self.produce_pre_dump and not self.checkpoint_process:
            raise ValueError("produce_pre_dump requires checkpoint_process=True")


@dataclass(frozen=True)
class RestoreJob:
    job_id: JobId
    sandbox_id: SandboxId
    checkpoint_id: CheckpointId
    requested_at: datetime
    reason: str = "manual"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CheckpointResult:
    job_id: JobId
    sandbox_id: SandboxId
    checkpoint_id: CheckpointId
    status: JobStatus
    started_at: datetime
    finished_at: datetime
    manifest: CheckpointManifest | None
    failure_code: FailureCode = FailureCode.NONE
    message: str | None = None
    operation_statuses: tuple[RuntimeOperationStatus, ...] = ()


@dataclass(frozen=True)
class RestoreResult:
    job_id: JobId
    sandbox_id: SandboxId
    checkpoint_id: CheckpointId
    status: JobStatus
    started_at: datetime
    finished_at: datetime
    failure_code: FailureCode = FailureCode.NONE
    message: str | None = None
    operation_statuses: tuple[RuntimeOperationStatus, ...] = ()


@dataclass(frozen=True)
class ChangesetEntry:
    """One changed path in a sandbox rootfs relative to a checkpoint
    snapshot (roadmap C1). ``path`` is container-absolute (rootfs-
    relative, ``/``-rooted). ``change`` is one of ``added`` /
    ``modified`` / ``removed`` / ``renamed``; renames carry the old
    container path in ``renamed_from``. Providers report raw truth —
    ignore policies belong to the merge layer (C2)."""

    path: str
    change: str
    renamed_from: str | None = None

    def to_json(self) -> dict[str, object]:
        payload: dict[str, object] = {"path": self.path, "change": self.change}
        if self.renamed_from is not None:
            payload["renamed_from"] = self.renamed_from
        return payload

    @classmethod
    def from_json(cls, payload: dict[str, object]) -> "ChangesetEntry":
        renamed_from = payload.get("renamed_from")
        return cls(
            path=str(payload["path"]),
            change=str(payload["change"]),
            renamed_from=None if renamed_from is None else str(renamed_from),
        )


@dataclass(frozen=True)
class ChangesetResult:
    """Filesystem changeset of a sandbox relative to a base checkpoint
    (``CrabSystem.changeset_since`` / ``fork_changeset``).
    ``skipped_by_gate`` is True when the inspector's
    ``filesystem_changed=False`` fast path proved the diff empty without
    running a backend diff."""

    sandbox_id: SandboxId
    base_checkpoint_id: CheckpointId
    entries: tuple[ChangesetEntry, ...]
    skipped_by_gate: bool = False

    def to_json(self) -> dict[str, object]:
        return {
            "sandbox_id": str(self.sandbox_id),
            "base_checkpoint_id": str(self.base_checkpoint_id),
            "entries": [entry.to_json() for entry in self.entries],
            "skipped_by_gate": self.skipped_by_gate,
        }

    @classmethod
    def from_json(cls, payload: dict[str, object]) -> "ChangesetResult":
        return cls(
            sandbox_id=SandboxId(str(payload["sandbox_id"])),
            base_checkpoint_id=CheckpointId(str(payload["base_checkpoint_id"])),
            entries=tuple(
                ChangesetEntry.from_json(entry) for entry in (payload.get("entries") or [])
            ),
            skipped_by_gate=bool(payload.get("skipped_by_gate", False)),
        )


@dataclass(frozen=True)
class MergeEntry:
    """One fork-changed path's fate in a merge (C2). ``resolution`` is
    ``applied`` / ``conflicted`` / ``skipped``; ``reason`` explains
    non-applied outcomes (``source_changed`` / ``dir_touch`` /
    ``ignored`` / ``unresolved_text`` / ``merge_aborted``). ``merged``
    marks content produced by the three-way text merge or a merger
    hook rather than taken verbatim from the fork."""

    path: str
    change: str
    resolution: str
    reason: str | None = None
    renamed_from: str | None = None
    merged: bool = False

    def to_json(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "path": self.path,
            "change": self.change,
            "resolution": self.resolution,
        }
        if self.reason is not None:
            payload["reason"] = self.reason
        if self.renamed_from is not None:
            payload["renamed_from"] = self.renamed_from
        if self.merged:
            payload["merged"] = True
        return payload

    @classmethod
    def from_json(cls, payload: dict[str, object]) -> "MergeEntry":
        reason = payload.get("reason")
        renamed_from = payload.get("renamed_from")
        return cls(
            path=str(payload["path"]),
            change=str(payload["change"]),
            resolution=str(payload["resolution"]),
            reason=None if reason is None else str(reason),
            renamed_from=None if renamed_from is None else str(renamed_from),
            merged=bool(payload.get("merged", False)),
        )


@dataclass(frozen=True)
class ObservationReport:
    """Outcome of ``CrabSystem.consolidate_observations`` (C3): how many
    fork journal records were adopted into the source's journal as
    ``kind="observation"`` rows. ``already_consolidated`` marks the
    idempotence fast path (a prior run for the same fork exists and the
    caller was not a manual re-run)."""

    source_sandbox_id: SandboxId
    fork_sandbox_id: SandboxId
    policy: str
    consolidated: int
    skipped_duplicates: int
    summary_written: bool = False
    already_consolidated: bool = False
    reason: str = "manual"

    def to_json(self) -> dict[str, object]:
        return {
            "source_sandbox_id": str(self.source_sandbox_id),
            "fork_sandbox_id": str(self.fork_sandbox_id),
            "policy": self.policy,
            "consolidated": self.consolidated,
            "skipped_duplicates": self.skipped_duplicates,
            "summary_written": self.summary_written,
            "already_consolidated": self.already_consolidated,
            "reason": self.reason,
        }

    @classmethod
    def from_json(cls, payload: dict[str, object]) -> "ObservationReport":
        return cls(
            source_sandbox_id=SandboxId(str(payload["source_sandbox_id"])),
            fork_sandbox_id=SandboxId(str(payload["fork_sandbox_id"])),
            policy=str(payload["policy"]),
            consolidated=int(payload.get("consolidated", 0)),
            skipped_duplicates=int(payload.get("skipped_duplicates", 0)),
            summary_written=bool(payload.get("summary_written", False)),
            already_consolidated=bool(payload.get("already_consolidated", False)),
            reason=str(payload.get("reason") or "manual"),
        )


@dataclass(frozen=True)
class MergeReport:
    """Outcome of ``CrabSystem.merge_from_fork`` (C2). ``rolled_back``
    is True when an apply-phase failure was undone from the pre-merge
    snapshot (the report then rides on ``MergeError``). ``observations``
    carries the C3 consolidation report when the merge was asked to
    adopt the fork's history."""

    source_sandbox_id: SandboxId
    fork_sandbox_id: SandboxId
    base_checkpoint_id: CheckpointId
    policy: str
    applied: tuple[MergeEntry, ...]
    conflicted: tuple[MergeEntry, ...]
    skipped: tuple[MergeEntry, ...]
    rolled_back: bool = False
    observations: "ObservationReport | None" = None

    def to_json(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "source_sandbox_id": str(self.source_sandbox_id),
            "fork_sandbox_id": str(self.fork_sandbox_id),
            "base_checkpoint_id": str(self.base_checkpoint_id),
            "policy": self.policy,
            "applied": [entry.to_json() for entry in self.applied],
            "conflicted": [entry.to_json() for entry in self.conflicted],
            "skipped": [entry.to_json() for entry in self.skipped],
            "rolled_back": self.rolled_back,
        }
        if self.observations is not None:
            payload["observations"] = self.observations.to_json()
        return payload

    @classmethod
    def from_json(cls, payload: dict[str, object]) -> "MergeReport":
        def entries(key: str) -> tuple[MergeEntry, ...]:
            return tuple(MergeEntry.from_json(entry) for entry in (payload.get(key) or []))

        raw_observations = payload.get("observations")
        return cls(
            source_sandbox_id=SandboxId(str(payload["source_sandbox_id"])),
            fork_sandbox_id=SandboxId(str(payload["fork_sandbox_id"])),
            base_checkpoint_id=CheckpointId(str(payload["base_checkpoint_id"])),
            policy=str(payload["policy"]),
            applied=entries("applied"),
            conflicted=entries("conflicted"),
            skipped=entries("skipped"),
            rolled_back=bool(payload.get("rolled_back", False)),
            observations=(
                None
                if not isinstance(raw_observations, dict)
                else ObservationReport.from_json(raw_observations)
            ),
        )


@dataclass(frozen=True)
class ReplayEntry:
    """One replayed fork exec (C4): the source re-ran the journaled
    command and its outcome is diffed against the recorded one.
    ``stdout_matched`` is None when the original run captured no
    output; ``deviated`` = returncode mismatch or a stdout digest
    mismatch where a digest exists."""

    origin_seq: int
    argv: tuple[str, ...]
    returncode: int | None
    expected_returncode: int | None
    stdout_matched: bool | None = None
    deviated: bool = False

    def to_json(self) -> dict[str, object]:
        return {
            "origin_seq": self.origin_seq,
            "argv": list(self.argv),
            "returncode": self.returncode,
            "expected_returncode": self.expected_returncode,
            "stdout_matched": self.stdout_matched,
            "deviated": self.deviated,
        }

    @classmethod
    def from_json(cls, payload: dict[str, object]) -> "ReplayEntry":
        returncode = payload.get("returncode")
        expected = payload.get("expected_returncode")
        stdout_matched = payload.get("stdout_matched")
        return cls(
            origin_seq=int(payload["origin_seq"]),
            argv=tuple(str(item) for item in (payload.get("argv") or [])),
            returncode=None if returncode is None else int(returncode),
            expected_returncode=None if expected is None else int(expected),
            stdout_matched=None if stdout_matched is None else bool(stdout_matched),
            deviated=bool(payload.get("deviated", False)),
        )


@dataclass(frozen=True)
class ProcessMergeReport:
    """Outcome of ``CrabSystem.merge_processes`` (C4). ``strategy`` is
    the resolved strategy (``auto`` never appears here);
    ``source_processes`` is the probed PID count on the source.
    Promotion fields stay at their defaults on the replay path and
    vice versa."""

    source_sandbox_id: SandboxId
    fork_sandbox_id: SandboxId
    strategy: str
    source_processes: int
    replayed: tuple[ReplayEntry, ...] = ()
    deviations: int = 0
    stopped_early: bool = False
    promoted_checkpoint_id: str | None = None
    fs_applied: int = 0
    fs_conflicted: int = 0
    observations: "ObservationReport | None" = None
    egress_replay: "EgressReplayReport | None" = None

    def to_json(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "source_sandbox_id": str(self.source_sandbox_id),
            "fork_sandbox_id": str(self.fork_sandbox_id),
            "strategy": self.strategy,
            "source_processes": self.source_processes,
            "replayed": [entry.to_json() for entry in self.replayed],
            "deviations": self.deviations,
            "stopped_early": self.stopped_early,
            "promoted_checkpoint_id": self.promoted_checkpoint_id,
            "fs_applied": self.fs_applied,
            "fs_conflicted": self.fs_conflicted,
        }
        if self.observations is not None:
            payload["observations"] = self.observations.to_json()
        if self.egress_replay is not None:
            payload["egress_replay"] = self.egress_replay.to_json()
        return payload

    @classmethod
    def from_json(cls, payload: dict[str, object]) -> "ProcessMergeReport":
        promoted = payload.get("promoted_checkpoint_id")
        raw_observations = payload.get("observations")
        raw_replay = payload.get("egress_replay")
        return cls(
            source_sandbox_id=SandboxId(str(payload["source_sandbox_id"])),
            fork_sandbox_id=SandboxId(str(payload["fork_sandbox_id"])),
            strategy=str(payload["strategy"]),
            source_processes=int(payload.get("source_processes", 0)),
            replayed=tuple(
                ReplayEntry.from_json(entry) for entry in (payload.get("replayed") or [])
            ),
            deviations=int(payload.get("deviations", 0)),
            stopped_early=bool(payload.get("stopped_early", False)),
            promoted_checkpoint_id=None if promoted is None else str(promoted),
            fs_applied=int(payload.get("fs_applied", 0)),
            fs_conflicted=int(payload.get("fs_conflicted", 0)),
            observations=(
                None
                if not isinstance(raw_observations, dict)
                else ObservationReport.from_json(raw_observations)
            ),
            egress_replay=(
                None
                if not isinstance(raw_replay, dict)
                else EgressReplayReport.from_json(raw_replay)
            ),
        )


@dataclass(frozen=True)
class EgressFlow:
    """One recorded outbound connection (D1). ``method``/``path`` are
    only known for plaintext HTTP; encrypted flows carry the SNI host
    and nothing more (no MITM). ``txn_id`` is the transaction that was
    active when the flow completed."""

    seq: int
    host: str
    dst_ip: str
    dst_port: int
    scheme: str
    classification: str
    method: str | None = None
    path: str | None = None
    bytes_out: int = 0
    bytes_in: int = 0
    duration_ms: float = 0.0
    txn_id: str | None = None
    recorded_at: str | None = None
    recorded: bool = False
    """A cassette holds this exchange's bodies (D2). Truncated exchanges
    stay False — they are visible but never replayable."""
    request_key: str | None = None
    status: int | None = None
    truncated: bool = False
    replayed: bool = False
    """Served from a cassette instead of the network (D2)."""
    replayed_from_seq: int | None = None
    """The recording's journal seq when known. Recording happens before
    its own journal row exists, so this stays None for cassettes written
    by the proxy; ``replayed`` is the authoritative flag."""
    replayed_from: str | None = None
    """Which sandbox's cassette bucket answered (a fork, for C4)."""
    effect: str | None = None
    """Effect-gate outcome for a write (D3): ``deferred`` (queued for
    commit), ``rejected``, ``sealed`` (sent, txn no longer abortable),
    ``flushed``/``flush_failed`` (recorded by the commit flush), or
    ``lost`` (queued but never flushed — the daemon restarted)."""
    effect_status: int | None = None
    """Upstream status of a flushed deferred write."""

    def to_json(self) -> dict[str, object]:
        return {
            "seq": self.seq,
            "host": self.host,
            "dst_ip": self.dst_ip,
            "dst_port": self.dst_port,
            "scheme": self.scheme,
            "classification": self.classification,
            "method": self.method,
            "path": self.path,
            "bytes_out": self.bytes_out,
            "bytes_in": self.bytes_in,
            "duration_ms": self.duration_ms,
            "txn_id": self.txn_id,
            "recorded_at": self.recorded_at,
            "recorded": self.recorded,
            "request_key": self.request_key,
            "status": self.status,
            "truncated": self.truncated,
            "replayed": self.replayed,
            "replayed_from_seq": self.replayed_from_seq,
            "replayed_from": self.replayed_from,
            "effect": self.effect,
            "effect_status": self.effect_status,
        }

    @classmethod
    def from_json(cls, payload: dict[str, object]) -> "EgressFlow":
        method = payload.get("method")
        path = payload.get("path")
        txn_id = payload.get("txn_id")
        recorded_at = payload.get("recorded_at")
        request_key = payload.get("request_key")
        status = payload.get("status")
        replayed_from = payload.get("replayed_from_seq")
        return cls(
            seq=int(payload.get("seq", 0)),
            host=str(payload.get("host", "")),
            dst_ip=str(payload.get("dst_ip", "")),
            dst_port=int(payload.get("dst_port", 0)),
            scheme=str(payload.get("scheme", "tcp")),
            classification=str(payload.get("classification", "opaque")),
            method=None if method is None else str(method),
            path=None if path is None else str(path),
            bytes_out=int(payload.get("bytes_out", 0)),
            bytes_in=int(payload.get("bytes_in", 0)),
            duration_ms=float(payload.get("duration_ms", 0.0)),
            txn_id=None if txn_id is None else str(txn_id),
            recorded_at=None if recorded_at is None else str(recorded_at),
            # Absent from D1-era rows; defaults keep them deserializable.
            recorded=bool(payload.get("recorded", False)),
            request_key=None if request_key is None else str(request_key),
            status=None if status is None else int(status),
            truncated=bool(payload.get("truncated", False)),
            replayed=bool(payload.get("replayed", False)),
            replayed_from_seq=None if replayed_from is None else int(replayed_from),
            replayed_from=(
                None
                if payload.get("replayed_from") is None
                else str(payload["replayed_from"])
            ),
            effect=None if payload.get("effect") is None else str(payload["effect"]),
            effect_status=(
                None
                if payload.get("effect_status") is None
                else int(payload["effect_status"])
            ),
        )


@dataclass(frozen=True)
class EgressLedger:
    """Effect ledger view (D1): the sandbox's recorded egress flows,
    optionally scoped to one transaction, with per-class counts."""

    sandbox_id: SandboxId
    flows: tuple[EgressFlow, ...] = ()
    txn_id: str | None = None

    @property
    def total(self) -> int:
        return len(self.flows)

    @property
    def idempotent_reads(self) -> int:
        return sum(1 for flow in self.flows if flow.classification == "idempotent_read")

    @property
    def mutating(self) -> int:
        return sum(1 for flow in self.flows if flow.classification == "mutating")

    @property
    def mutating_sent(self) -> int:
        """Mutating flows that actually reached the world.

        A write the effect gate deferred, refused or dropped (D3) never
        left the host, so counting it as "already fired" would misreport
        exactly the guarantee ``effects="defer"`` exists to provide.
        ``flush_failed`` counts: the attempt may have arrived before the
        connection broke.
        """
        held = {"deferred", "rejected", "dropped", "lost"}
        return sum(
            1
            for flow in self.flows
            if flow.classification == "mutating" and flow.effect not in held
        )

    @property
    def opaque(self) -> int:
        return sum(1 for flow in self.flows if flow.classification == "opaque")

    @property
    def hosts(self) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for flow in self.flows:
            seen.setdefault(flow.host, None)
        return tuple(seen)

    @property
    def recorded(self) -> int:
        return sum(1 for flow in self.flows if flow.recorded)

    @property
    def replayed(self) -> int:
        return sum(1 for flow in self.flows if flow.replayed)

    @property
    def deferred(self) -> int:
        """Writes queued for commit. Still-pending ones are those whose
        `effect` is exactly ``deferred`` (a later row records the
        flush/drop outcome)."""
        return sum(1 for flow in self.flows if flow.effect == "deferred")

    @property
    def rejected(self) -> int:
        return sum(1 for flow in self.flows if flow.effect == "rejected")

    @property
    def flushed(self) -> int:
        return sum(1 for flow in self.flows if flow.effect == "flushed")

    def to_json(self) -> dict[str, object]:
        return {
            "sandbox_id": str(self.sandbox_id),
            "txn_id": self.txn_id,
            "flows": [flow.to_json() for flow in self.flows],
            "total": self.total,
            "idempotent_reads": self.idempotent_reads,
            "mutating": self.mutating,
            "opaque": self.opaque,
            "recorded": self.recorded,
            "replayed": self.replayed,
            "deferred": self.deferred,
            "rejected": self.rejected,
            "flushed": self.flushed,
        }

    @classmethod
    def from_json(cls, payload: dict[str, object]) -> "EgressLedger":
        txn_id = payload.get("txn_id")
        return cls(
            sandbox_id=SandboxId(str(payload["sandbox_id"])),
            flows=tuple(
                EgressFlow.from_json(row) for row in (payload.get("flows") or [])
            ),
            txn_id=None if txn_id is None else str(txn_id),
        )


@dataclass(frozen=True)
class EgressReplayReport:
    """Outcome of a replay window (D2). ``served`` flows never touched the
    network; ``missed`` ones found no cassette (and under
    ``cassette_only`` got a 504 instead of live traffic);
    ``passed_through`` counts flows replay never claims — writes and
    encrypted/raw traffic always reach the real world."""

    sandbox_id: SandboxId
    policy: str
    cassette_source: str
    served: int = 0
    missed: int = 0
    passed_through: int = 0
    hosts: tuple[str, ...] = ()

    def to_json(self) -> dict[str, object]:
        return {
            "sandbox_id": str(self.sandbox_id),
            "policy": self.policy,
            "cassette_source": self.cassette_source,
            "served": self.served,
            "missed": self.missed,
            "passed_through": self.passed_through,
            "hosts": list(self.hosts),
        }

    @classmethod
    def from_json(cls, payload: dict[str, object]) -> "EgressReplayReport":
        return cls(
            sandbox_id=SandboxId(str(payload["sandbox_id"])),
            policy=str(payload.get("policy", "cassette_first")),
            cassette_source=str(payload.get("cassette_source", "")),
            served=int(payload.get("served", 0)),
            missed=int(payload.get("missed", 0)),
            passed_through=int(payload.get("passed_through", 0)),
            hosts=tuple(str(host) for host in (payload.get("hosts") or [])),
        )


@dataclass(frozen=True)
class JobRecord:
    job_id: JobId
    job_type: JobType
    sandbox_id: SandboxId
    checkpoint_id: CheckpointId | None
    status: JobStatus
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    failure_code: FailureCode = FailureCode.NONE
    message: str | None = None


@dataclass(frozen=True)
class SandboxSnapshot:
    sandbox_id: SandboxId
    runtime_name: str
    is_running: bool
    process_changed: bool
    filesystem_changed: bool
    observed_at: datetime
    last_checkpoint_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class SchedulerCheckpointDecision:
    should_checkpoint: bool
    checkpoint_process: bool
    checkpoint_filesystem: bool
    leave_running: bool
    reason: str
    policy_name: str
    metadata: dict[str, Any] = field(default_factory=dict)
    is_incremental_process: bool = False
    parent_process_checkpoint_id: CheckpointId | None = None
    produce_pre_dump: bool = False


@dataclass(frozen=True)
class RequestContext:
    request_id: str
    sandbox_id: SandboxId
    started_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RequestState:
    sandbox_id: SandboxId
    active_llm_requests: int = 0
    total_llm_requests: int = 0
    completed_llm_requests: int = 0
    last_request_id: str | None = None
    last_llm_provider: str | None = None
    last_llm_request_started_at: datetime | None = None
    last_llm_request_ended_at: datetime | None = None

    @property
    def llm_request_in_flight(self) -> bool:
        return self.active_llm_requests > 0

    def to_metadata(self) -> dict[str, Any]:
        return {
            "llm_request_in_flight": self.llm_request_in_flight,
            "active_llm_requests": self.active_llm_requests,
            "total_llm_requests": self.total_llm_requests,
            "completed_llm_requests": self.completed_llm_requests,
            "last_llm_provider": self.last_llm_provider,
            "last_request_id": self.last_request_id,
            "last_llm_request_started_at": (
                None
                if self.last_llm_request_started_at is None
                else _isoformat(self.last_llm_request_started_at)
            ),
            "last_llm_request_ended_at": (
                None
                if self.last_llm_request_ended_at is None
                else _isoformat(self.last_llm_request_ended_at)
            ),
        }


@dataclass(frozen=True)
class RequestStateChange:
    sandbox_id: SandboxId
    event_type: str
    request_id: str | None = None
    observed_at: datetime = field(default_factory=utc_now)


@dataclass(frozen=True)
class RecoveryEvent:
    sandbox_id: SandboxId
    event_type: str
    observed_at: datetime
    received_at: datetime = field(default_factory=utc_now)
    reason: str = ""
    grace_remaining_seconds: float | None = None


@dataclass(frozen=True)
class RecoveryRecord:
    sandbox_id: SandboxId
    event_type: str
    started_at: datetime
    finished_at: datetime
    status: str
    checkpoint_id: CheckpointId | None = None
    message: str | None = None


@dataclass(frozen=True)
class SandboxDescription:
    sandbox_id: SandboxId
    runtime_name: str
    status: str
    metadata: dict[str, Any] = field(default_factory=dict)
