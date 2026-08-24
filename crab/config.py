from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_TELEMETRY_WRITER_MODES = {"sync", "async"}
_TELEMETRY_OVERFLOW_POLICIES = {"drop_new", "block"}
_TELEMETRY_SERIALIZERS = {"auto", "stdlib", "orjson"}
_CHECKPOINT_SCHEDULING_POLICIES = {"fifo", "reactive"}


@dataclass(frozen=True)
class SchedulerConfig:
    min_checkpoint_interval_seconds: float = 30.0
    force_checkpoint_after_seconds: float = 600.0
    require_change_signal: bool = True
    checkpoint_full_baseline_on_first_checkpoint: bool = True
    prefer_checkpoint_during_llm_request: bool = True
    require_llm_request_for_checkpoint: bool = False
    inspect_without_pause: bool = False
    incremental_process_enabled: bool = False
    full_process_checkpoint_interval: int = 8
    max_process_chain_length: int = 16

    def __post_init__(self) -> None:
        if self.min_checkpoint_interval_seconds < 0:
            raise ValueError("min_checkpoint_interval_seconds must be >= 0")
        if self.force_checkpoint_after_seconds < 0:
            raise ValueError("force_checkpoint_after_seconds must be >= 0")
        if self.full_process_checkpoint_interval < 1:
            raise ValueError("full_process_checkpoint_interval must be >= 1")
        if self.max_process_chain_length < 1:
            raise ValueError("max_process_chain_length must be >= 1")


@dataclass(frozen=True)
class ExecutorConfig:
    max_workers: int = 4
    checkpoint_workers: int | None = None
    restore_workers: int | None = None
    coordination_workers: int | None = None
    composite_step_workers: int | None = None
    max_checkpoint_queue_size: int = 10000
    checkpoint_scheduling_policy: str = "fifo"
    reactive_checkpoint_urgent_quota: int = 4
    max_retries: int = 0
    retry_backoff_seconds: float = 0.05

    def __post_init__(self) -> None:
        if self.max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        if self.checkpoint_workers is not None and self.checkpoint_workers < 1:
            raise ValueError("checkpoint_workers must be >= 1 when provided")
        if self.restore_workers is not None and self.restore_workers < 1:
            raise ValueError("restore_workers must be >= 1 when provided")
        if self.coordination_workers is not None and self.coordination_workers < 1:
            raise ValueError("coordination_workers must be >= 1 when provided")
        if self.composite_step_workers is not None and self.composite_step_workers < 1:
            raise ValueError("composite_step_workers must be >= 1 when provided")
        if self.max_checkpoint_queue_size < 1:
            raise ValueError("max_checkpoint_queue_size must be >= 1")
        if self.checkpoint_scheduling_policy not in _CHECKPOINT_SCHEDULING_POLICIES:
            raise ValueError(
                "checkpoint_scheduling_policy must be one of "
                f"{sorted(_CHECKPOINT_SCHEDULING_POLICIES)}, got {self.checkpoint_scheduling_policy!r}"
            )
        if self.reactive_checkpoint_urgent_quota < 1:
            raise ValueError("reactive_checkpoint_urgent_quota must be >= 1")
        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if self.retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds must be >= 0")

    @property
    def resolved_checkpoint_workers(self) -> int:
        return self.max_workers if self.checkpoint_workers is None else self.checkpoint_workers

    @property
    def resolved_restore_workers(self) -> int:
        return self.max_workers if self.restore_workers is None else self.restore_workers

    @property
    def resolved_coordination_workers(self) -> int:
        if self.coordination_workers is not None:
            return self.coordination_workers
        return min(8, self.resolved_checkpoint_workers)

    @property
    def resolved_composite_step_workers(self) -> int:
        if self.composite_step_workers is not None:
            return self.composite_step_workers
        return max(2, min(2 * self.resolved_checkpoint_workers, 16))


@dataclass(frozen=True)
class StorageConfig:
    root_dir: Path
    manifests_dirname: str = "manifests"
    artifacts_dirname: str = "artifacts"
    journal_dirname: str = "journal"
    cassettes_dirname: str = "cassettes"

    def __post_init__(self) -> None:
        if not self.manifests_dirname:
            raise ValueError("manifests_dirname must be non-empty")
        if not self.artifacts_dirname:
            raise ValueError("artifacts_dirname must be non-empty")
        if not self.journal_dirname:
            raise ValueError("journal_dirname must be non-empty")
        if not self.cassettes_dirname:
            raise ValueError("cassettes_dirname must be non-empty")


@dataclass(frozen=True)
class TelemetryConfig:
    enabled: bool = True
    jsonl_path: Path | None = None
    keep_in_memory_copy: bool | None = None
    detail_level: str = "basic"
    capture_command_output: bool = False
    max_text_attribute_bytes: int = 2048
    writer_mode: str = "async"
    queue_capacity: int = 16384
    batch_max_records: int = 256
    flush_interval_ms: int = 50
    overflow_policy: str = "drop_new"
    serializer: str = "auto"

    def __post_init__(self) -> None:
        if self.max_text_attribute_bytes < 32:
            raise ValueError("max_text_attribute_bytes must be >= 32")
        if self.writer_mode not in _TELEMETRY_WRITER_MODES:
            raise ValueError(
                f"writer_mode must be one of {sorted(_TELEMETRY_WRITER_MODES)}, got {self.writer_mode!r}"
            )
        if self.queue_capacity < 1:
            raise ValueError("queue_capacity must be >= 1")
        if self.batch_max_records < 1:
            raise ValueError("batch_max_records must be >= 1")
        if self.flush_interval_ms < 1:
            raise ValueError("flush_interval_ms must be >= 1")
        if self.overflow_policy not in _TELEMETRY_OVERFLOW_POLICIES:
            raise ValueError(
                f"overflow_policy must be one of {sorted(_TELEMETRY_OVERFLOW_POLICIES)}, got {self.overflow_policy!r}"
            )
        if self.serializer not in _TELEMETRY_SERIALIZERS:
            raise ValueError(
                f"serializer must be one of {sorted(_TELEMETRY_SERIALIZERS)}, got {self.serializer!r}"
            )

    def resolve_keep_in_memory_copy(self, *, fallback: bool) -> bool:
        if self.keep_in_memory_copy is not None:
            return bool(self.keep_in_memory_copy)
        if self.jsonl_path is not None:
            return False
        return bool(fallback)
