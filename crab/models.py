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
