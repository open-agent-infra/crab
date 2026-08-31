from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import logging
from pathlib import Path
import time
from typing import Callable

from ..contracts import (
    CheckpointManager,
    CompositeCheckpointWorker,
    CompositeRestoreWorker,
    FileSystemCWorker,
    FileSystemRWorker,
    ProcessCWorker,
    ProcessRWorker,
    Runtime,
    TelemetrySink,
)
from ..ids import CheckpointId
from ..models import (
    ArtifactKind,
    CheckpointJob,
    CheckpointManifest,
    CheckpointResult,
    FailureCode,
    JobStatus,
    RestoreJob,
    RestoreResult,
    utc_now,
)
from ..telemetry import NoopTelemetrySink, start_operation

logger = logging.getLogger(__name__)

_PROCESS_RESTORE_CHECKPOINT_ID = "process_restore_checkpoint_id"
_PROCESS_RESTORE_TRACE_CURSOR = "process_restore_trace_cursor"
_FILESYSTEM_RESTORE_CHECKPOINT_ID = "filesystem_restore_checkpoint_id"
_FILESYSTEM_RESTORE_TRACE_CURSOR = "filesystem_restore_trace_cursor"
_PROCESS_RESTORE_CREATED_AT = "process_restore_created_at"
_FILESYSTEM_RESTORE_CREATED_AT = "filesystem_restore_created_at"
_RESTORE_SOURCE_GAP_TURNS = "restore_source_gap_turns"
_RESTORE_SOURCE_GAP_MS = "restore_source_gap_ms"
_RESTORE_ESTIMATED_IO_BYTES = "restore_estimated_io_bytes"
_RESTORE_MIXED_SOURCES = "restore_mixed_sources"
_CAPTURES_INFLIGHT_LLM = "captures_inflight_llm"
_CAPTURED_REQUEST_ID = "captured_request_id"
_CAPTURED_REQUEST_PROVIDER = "captured_request_provider"
_CAPTURED_REQUEST_STARTED_AT = "captured_request_started_at"
_LOGICAL_CHECKPOINT = "logical_checkpoint"


def resolve_restore_manifest(
    checkpoint_manager: CheckpointManager,
    manifest: CheckpointManifest,
    *,
    candidates: list[CheckpointManifest] | None = None,
) -> CheckpointManifest:
    if candidates is None:
        checkpoints = checkpoint_manager.list_checkpoints(manifest.sandbox_id)
        try:
            current_index = checkpoints.index(manifest.checkpoint_id)
            candidate_ids = checkpoints[: current_index + 1]
        except ValueError:
            candidate_ids = checkpoints

        candidates = [checkpoint_manager.get_manifest(manifest.sandbox_id, checkpoint_id) for checkpoint_id in candidate_ids]
    else:
        try:
            current_index = next(
                index for index, candidate in enumerate(candidates) if candidate.checkpoint_id == manifest.checkpoint_id
            )
        except StopIteration:
            candidates = list(candidates)
        else:
            candidates = candidates[: current_index + 1]

    current_candidate = candidates[-1] if candidates else manifest

    process_manifest = None
    filesystem_manifest = None
    if bool(current_candidate.metadata.get(_LOGICAL_CHECKPOINT, False)):
        process_source_raw = current_candidate.metadata.get(
            _PROCESS_RESTORE_CHECKPOINT_ID
        )
        filesystem_source_raw = current_candidate.metadata.get(
            _FILESYSTEM_RESTORE_CHECKPOINT_ID
        )
        if process_source_raw is None or filesystem_source_raw is None:
            raise ValueError(
                f"logical checkpoint {current_candidate.checkpoint_id} is missing "
                "explicit process/filesystem restore sources"
            )
        if process_source_raw is not None:
            process_manifest = _restore_source_manifest(
                checkpoint_manager,
                current_candidate,
                CheckpointId(str(process_source_raw)),
                candidates,
                component="process",
            )
        if filesystem_source_raw is not None:
            filesystem_manifest = _restore_source_manifest(
                checkpoint_manager,
                current_candidate,
                CheckpointId(str(filesystem_source_raw)),
                candidates,
                component="filesystem",
            )

    if process_manifest is None and filesystem_manifest is None:
        filesystem_manifest = _latest_manifest_with_artifacts(
            candidates,
            include=lambda candidate: bool(candidate.filesystem_artifacts),
        )
        process_manifest = _latest_process_manifest_for_restore(
            candidates, filesystem_manifest
        )

    process_artifacts = [] if process_manifest is None else list(process_manifest.process_artifacts)
    filesystem_artifacts = [] if filesystem_manifest is None else list(filesystem_manifest.filesystem_artifacts)
    metadata = dict(current_candidate.metadata)
    if process_manifest is not None:
        metadata[_PROCESS_RESTORE_CHECKPOINT_ID] = str(process_manifest.checkpoint_id)
        metadata[_PROCESS_RESTORE_TRACE_CURSOR] = _committed_trace_cursor(
            process_manifest.metadata
        )
        metadata[_PROCESS_RESTORE_CREATED_AT] = process_manifest.created_at.isoformat()
    if filesystem_manifest is not None:
        metadata[_FILESYSTEM_RESTORE_CHECKPOINT_ID] = str(filesystem_manifest.checkpoint_id)
        metadata[_FILESYSTEM_RESTORE_TRACE_CURSOR] = _committed_trace_cursor(
            filesystem_manifest.metadata
        )
        metadata[_FILESYSTEM_RESTORE_CREATED_AT] = filesystem_manifest.created_at.isoformat()
    if process_manifest is not None and filesystem_manifest is not None:
        metadata[_RESTORE_SOURCE_GAP_TURNS] = abs(
            _committed_trace_cursor(process_manifest.metadata) - _committed_trace_cursor(filesystem_manifest.metadata)
        )
        metadata[_RESTORE_SOURCE_GAP_MS] = abs(
            (filesystem_manifest.created_at - process_manifest.created_at).total_seconds() * 1000.0
        )
        metadata[_RESTORE_MIXED_SOURCES] = str(process_manifest.checkpoint_id) != str(filesystem_manifest.checkpoint_id)
        estimated_io_bytes = 0
        has_estimate = False
        for candidate in (process_manifest, filesystem_manifest):
            raw_process_size = candidate.metadata.get("process_checkpoint_size_bytes")
            raw_fs_written = candidate.metadata.get("filesystem_checkpoint_written_bytes")
            for raw_value in (raw_process_size, raw_fs_written):
                try:
                    estimated_io_bytes += int(raw_value)
                    has_estimate = True
                except (TypeError, ValueError):
                    continue
        if has_estimate:
            metadata[_RESTORE_ESTIMATED_IO_BYTES] = estimated_io_bytes
    _copy_process_restore_metadata(metadata, {} if process_manifest is None else process_manifest.metadata)

    return replace(
        current_candidate,
        process_artifacts=process_artifacts,
        filesystem_artifacts=filesystem_artifacts,
        metadata=metadata,
        parent_checkpoint_id=(
            current_candidate.parent_checkpoint_id
            if process_manifest is None
            else process_manifest.parent_checkpoint_id
        ),
        process_kind=(
            current_candidate.process_kind
            if process_manifest is None
            else process_manifest.process_kind
        ),
    ).with_integrity()


def _restore_source_manifest(
    checkpoint_manager: CheckpointManager,
    logical_manifest: CheckpointManifest,
    source_id: CheckpointId,
    candidates: list[CheckpointManifest],
    *,
    component: str,
) -> CheckpointManifest:
    source = next(
        (candidate for candidate in candidates if candidate.checkpoint_id == source_id),
        None,
    )
    if source is None:
        source = checkpoint_manager.get_manifest(logical_manifest.sandbox_id, source_id)
    artifacts = (
        source.process_artifacts
        if component == "process"
        else source.filesystem_artifacts
    )
    if not artifacts:
        raise ValueError(
            f"logical checkpoint {logical_manifest.checkpoint_id} references "
            f"{component} source {source_id} without {component} artifacts"
        )
    return source


def _latest_manifest_with_artifacts(
    manifests: list[CheckpointManifest],
    *,
    include: Callable[[CheckpointManifest], bool],
) -> CheckpointManifest | None:
    for candidate in reversed(manifests):
        if include(candidate):
            return candidate
    return None


def _latest_process_manifest_for_restore(
    manifests: list[CheckpointManifest],
    filesystem_manifest: CheckpointManifest | None,
) -> CheckpointManifest | None:
    filesystem_trace_cursor = None if filesystem_manifest is None else _committed_trace_cursor(filesystem_manifest.metadata)
    latest_with_artifacts: CheckpointManifest | None = None
    for candidate in reversed(manifests):
        if not candidate.process_artifacts:
            continue
        if latest_with_artifacts is None:
            latest_with_artifacts = candidate
        if filesystem_trace_cursor is None:
            return candidate
        if _committed_trace_cursor(candidate.metadata) <= filesystem_trace_cursor:
            return candidate
    return latest_with_artifacts


def _trace_cursor_from_metadata(metadata: dict[str, object]) -> int:
    raw_value = metadata.get("benchmark_trace_cursor", 0)
    try:
        return max(0, int(raw_value))
    except (TypeError, ValueError):
        return 0


def _committed_trace_cursor(metadata: dict[str, object]) -> int:
    # benchmark_trace_cursor already records the last replay turn that has been
    # served. Capturing the next in-flight LLM request should not rewind that
    # committed replay position during restore selection.
    return _trace_cursor_from_metadata(metadata)


def _validate_incremental_chain(
    runtime: Runtime,
    checkpoint_manager: CheckpointManager,
    manifest: CheckpointManifest,
) -> None:
    """Walk ``manifest``'s parent chain and verify every ancestor's pre-dump
    and final-dump image directories still exist on disk. CRIU follows the
    `parent` symlinks each pre-dump dropped, so a missing ancestor means
    restore would silently corrupt or hang. Raises FileNotFoundError naming
    the missing checkpoint id when any ancestor is unreachable.
    """
    if manifest.process_kind != "incremental":
        return
    if not runtime.capabilities().supports_incremental_process:
        # The runtime doesn't model incremental chains; nothing to validate.
        return

    visited: set[CheckpointId] = set()
    cursor = manifest
    while cursor.parent_checkpoint_id is not None:
        parent_id = cursor.parent_checkpoint_id
        if parent_id in visited:
            raise RuntimeError(
                f"detected cycle in incremental checkpoint chain at {parent_id}"
            )
        visited.add(parent_id)

        pre_dump_path_str = runtime.pre_dump_location(manifest.sandbox_id, parent_id)
        if pre_dump_path_str is None:
            raise FileNotFoundError(
                f"runtime did not provide pre-dump location for parent="
                f"{parent_id} sandbox={manifest.sandbox_id}"
            )
        pre_dump_path = Path(pre_dump_path_str)
        if not pre_dump_path.exists():
            raise FileNotFoundError(
                f"missing pre-dump directory in incremental chain: "
                f"sandbox={manifest.sandbox_id} parent={parent_id} path={pre_dump_path}"
            )

        try:
            parent_manifest = checkpoint_manager.get_manifest(manifest.sandbox_id, parent_id)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"missing manifest for incremental parent: sandbox="
                f"{manifest.sandbox_id} parent={parent_id}"
            ) from exc

        if parent_manifest.process_kind == "full":
            return  # chain root reached
        cursor = parent_manifest


def _copy_process_restore_metadata(target: dict[str, object], source: dict[str, object]) -> None:
    target[_CAPTURES_INFLIGHT_LLM] = bool(source.get(_CAPTURES_INFLIGHT_LLM, False))
    for key in (_CAPTURED_REQUEST_ID, _CAPTURED_REQUEST_PROVIDER, _CAPTURED_REQUEST_STARTED_AT):
        value = source.get(key)
        if value is None:
            target.pop(key, None)
        else:
            target[key] = value


class DefaultCWorker(CompositeCheckpointWorker):
    def __init__(
        self,
        process_worker: ProcessCWorker,
        filesystem_worker: FileSystemCWorker,
        checkpoint_manager: CheckpointManager,
        runtime: Runtime,
        checkpoint_guard: Callable[[CheckpointJob], tuple[bool, str | None]] | None = None,
        telemetry: TelemetrySink | None = None,
        step_workers: int | None = None,
    ):
        self._process_worker = process_worker
        self._filesystem_worker = filesystem_worker
        self._checkpoint_manager = checkpoint_manager
        self._runtime = runtime
        self._checkpoint_guard = checkpoint_guard
        self._telemetry = telemetry or NoopTelemetrySink()
        self._step_pool = ThreadPoolExecutor(
            max_workers=max(2, 2 if step_workers is None else int(step_workers)),
            thread_name_prefix="crab-ckpt-step",
        )

    def close(self) -> None:
        self._step_pool.shutdown(wait=True, cancel_futures=True)

    def checkpoint(self, job: CheckpointJob) -> CheckpointResult:
        job = self._ensure_restorable_checkpoint(job)
        started = utc_now()
        checkpoint_id = CheckpointId(str(job.metadata.get("checkpoint_id", CheckpointId.new())))
        checkpoint_scope = _checkpoint_scope(job)
        if self._checkpoint_guard is not None:
            allowed, message = self._checkpoint_guard(job)
            if not allowed:
                logger.info(
                    "Skipping composite checkpoint for job %s sandbox=%s checkpoint=%s reason=%s",
                    job.job_id,
                    job.sandbox_id,
                    checkpoint_id,
                    "" if message is None else message,
                )
                return CheckpointResult(
                    job_id=job.job_id,
                    sandbox_id=job.sandbox_id,
                    checkpoint_id=checkpoint_id,
                    status=JobStatus.FAILED,
                    started_at=started,
                    finished_at=utc_now(),
                    manifest=None,
                    failure_code=FailureCode.VALIDATION_ERROR,
                    message=message or "checkpoint_rejected",
                )
        logger.info(
            "Starting composite checkpoint for job %s sandbox=%s checkpoint=%s",
            job.job_id,
            job.sandbox_id,
            checkpoint_id,
        )

        process_step = None
        fs_step = None
        process_duration_ms = 0.0
        filesystem_duration_ms = 0.0
        if job.checkpoint_process and job.checkpoint_filesystem:
            process_future = self._step_pool.submit(self._timed_checkpoint_process, job, checkpoint_id)
            fs_future = self._step_pool.submit(self._timed_checkpoint_filesystem, job, checkpoint_id)
            process_step, process_duration_ms = process_future.result()
            fs_step, filesystem_duration_ms = fs_future.result()
        elif job.checkpoint_process:
            process_step, process_duration_ms = self._timed_checkpoint_process(job, checkpoint_id)
        elif job.checkpoint_filesystem:
            fs_step, filesystem_duration_ms = self._timed_checkpoint_filesystem(job, checkpoint_id)

        failed_step = None
        if process_step is not None and not process_step.success:
            failed_step = process_step
            logger.warning(
                "Process checkpoint step failed for job %s sandbox=%s code=%s message=%s",
                job.job_id,
                job.sandbox_id,
                process_step.failure_code.value,
                process_step.message,
            )
        elif fs_step is not None and not fs_step.success:
            failed_step = fs_step
            logger.warning(
                "Filesystem checkpoint step failed for job %s sandbox=%s code=%s message=%s",
                job.job_id,
                job.sandbox_id,
                fs_step.failure_code.value,
                fs_step.message,
            )

        operation_statuses = tuple(
            step.operation_status for step in (process_step, fs_step) if step is not None
        )
        process_size_bytes = _safe_int(None if process_step is None else process_step.operation_status.metadata.get("process_checkpoint_size_bytes"))
        process_file_count = _safe_int(None if process_step is None else process_step.operation_status.metadata.get("process_checkpoint_file_count"))
        filesystem_written_bytes = _safe_int(None if fs_step is None else fs_step.operation_status.metadata.get("filesystem_checkpoint_written_bytes"))
        filesystem_used_bytes = _safe_int(None if fs_step is None else fs_step.operation_status.metadata.get("filesystem_checkpoint_used_bytes"))
        estimated_io_bytes = _checkpoint_estimated_io_bytes(
            process_size_bytes=process_size_bytes,
            filesystem_written_bytes=filesystem_written_bytes,
        )
        if failed_step is not None:
            try:
                self._runtime.discard_partial_checkpoint(job.sandbox_id, checkpoint_id)
            except Exception:
                logger.exception(
                    "Failed to discard partial checkpoint artifacts for job %s sandbox=%s checkpoint=%s",
                    job.job_id,
                    job.sandbox_id,
                    checkpoint_id,
                )
            return CheckpointResult(
                job_id=job.job_id,
                sandbox_id=job.sandbox_id,
                checkpoint_id=checkpoint_id,
                status=JobStatus.FAILED,
                started_at=started,
                finished_at=utc_now(),
                manifest=None,
                failure_code=failed_step.failure_code,
                message=failed_step.message,
                operation_statuses=operation_statuses,
            )

        artifact_started = time.perf_counter()
        persist_artifacts = start_operation(
            self._telemetry,
            "checkpoint.persist_artifacts",
            {"component": "checkpoint", "sandbox_id": str(job.sandbox_id), "checkpoint_id": str(checkpoint_id)},
        )
        refs = []
        for artifact in [
            *(process_step.artifacts if process_step is not None else []),
            *(fs_step.artifacts if fs_step is not None else []),
        ]:
            refs.append(
                self._checkpoint_manager.put_artifact(
                    sandbox_id=job.sandbox_id,
                    checkpoint_id=checkpoint_id,
                    artifact=artifact,
                )
            )
        self._telemetry.emit_metric(
            "checkpoint.persist_artifacts_ms",
            (time.perf_counter() - artifact_started) * 1000.0,
            {"sandbox_id": str(job.sandbox_id), "checkpoint_id": str(checkpoint_id), "artifact_count": len(refs)},
        )
        persist_artifacts.finish(status="succeeded", attributes={"artifact_count": len(refs)})
        logger.debug(
            "Stored %d artifacts for job %s sandbox=%s checkpoint=%s",
            len(refs),
            job.job_id,
            job.sandbox_id,
            checkpoint_id,
        )

        process_refs = [x for x in refs if x.kind == ArtifactKind.PROCESS]
        fs_refs = [x for x in refs if x.kind == ArtifactKind.FILESYSTEM]

        # A checkpoint is "incremental" only if it chains off a parent's
        # pre_dump. Anchors (chain root, chain reset) produce a pre_dump but
        # are themselves "full" — they are the root the chain restores from.
        is_chain_node = (
            job.is_incremental_process
            and job.parent_process_checkpoint_id is not None
            and self._runtime.capabilities().supports_incremental_process
        )
        process_kind = "incremental" if is_chain_node else "full"
        parent_checkpoint_id = (
            job.parent_process_checkpoint_id if is_chain_node else None
        )
        manifest = CheckpointManifest(
            schema_version="v1",
            checkpoint_id=checkpoint_id,
            sandbox_id=job.sandbox_id,
            created_at=utc_now(),
            runtime_name=self._runtime.name,
            runtime_version=self._runtime.version,
            process_artifacts=process_refs,
            filesystem_artifacts=fs_refs,
            metadata={
                **job.metadata,
                "job_id": str(job.job_id),
                "reason": job.reason,
                "leave_running": job.leave_running,
                "checkpoint_scope": checkpoint_scope,
                "process_checkpoint_size_bytes": process_size_bytes,
                "process_checkpoint_file_count": process_file_count,
                "filesystem_checkpoint_written_bytes": filesystem_written_bytes,
                "filesystem_checkpoint_used_bytes": filesystem_used_bytes,
                "checkpoint_estimated_io_bytes": estimated_io_bytes,
            },
            parent_checkpoint_id=parent_checkpoint_id,
            process_kind=process_kind,
        ).with_integrity()
        manifest_started = time.perf_counter()
        persist_manifest = start_operation(
            self._telemetry,
            "checkpoint.persist_manifest",
            {"component": "checkpoint", "sandbox_id": str(job.sandbox_id), "checkpoint_id": str(checkpoint_id)},
        )
        write_manifest_operation = start_operation(
            self._telemetry,
            "checkpoint.write_manifest",
            {"component": "checkpoint", "sandbox_id": str(job.sandbox_id), "checkpoint_id": str(checkpoint_id)},
        )
        retention_operation = start_operation(
            self._telemetry,
            "checkpoint.retention_policy",
            {"component": "checkpoint", "sandbox_id": str(job.sandbox_id), "checkpoint_id": str(checkpoint_id)},
        )
        write_manifest_finished = False
        retention_finished = False
        try:
            write_manifest_started = time.perf_counter()
            self._checkpoint_manager.put_manifest(manifest)
            self._telemetry.emit_metric(
                "checkpoint.write_manifest_ms",
                (time.perf_counter() - write_manifest_started) * 1000.0,
                {"sandbox_id": str(job.sandbox_id), "checkpoint_id": str(checkpoint_id)},
            )
            write_manifest_operation.finish(status="succeeded")
            write_manifest_finished = True

            retention_started = time.perf_counter()
            self._checkpoint_manager.handle_checkpoint_complete(manifest)
            self._telemetry.emit_metric(
                "checkpoint.retention_policy_ms",
                (time.perf_counter() - retention_started) * 1000.0,
                {"sandbox_id": str(job.sandbox_id), "checkpoint_id": str(checkpoint_id)},
            )
            retention_operation.finish(status="succeeded")
            retention_finished = True
        except Exception as exc:
            if not retention_finished:
                retention_operation.finish(status="failed")
            if not write_manifest_finished:
                write_manifest_operation.finish(status="failed")
            persist_manifest.finish(status="failed")
            logger.exception(
                "Failed to persist manifest for job %s sandbox=%s checkpoint=%s",
                job.job_id,
                job.sandbox_id,
                checkpoint_id,
            )
            if not write_manifest_finished:
                # Manifest never landed → on-disk artifacts are unreachable
                # via storage.delete_checkpoint. Clean them up directly so
                # the failed checkpoint doesn't leave a multi-GB orphan.
                try:
                    self._checkpoint_manager.delete_checkpoint(job.sandbox_id, checkpoint_id)
                except Exception:
                    logger.exception(
                        "Failed to delete partial artifacts for job %s sandbox=%s checkpoint=%s",
                        job.job_id,
                        job.sandbox_id,
                        checkpoint_id,
                    )
                try:
                    self._runtime.discard_partial_checkpoint(job.sandbox_id, checkpoint_id)
                except Exception:
                    logger.exception(
                        "Failed to discard partial checkpoint runtime artifacts for job %s sandbox=%s checkpoint=%s",
                        job.job_id,
                        job.sandbox_id,
                        checkpoint_id,
                    )
            return CheckpointResult(
                job_id=job.job_id,
                sandbox_id=job.sandbox_id,
                checkpoint_id=checkpoint_id,
                status=JobStatus.FAILED,
                started_at=started,
                finished_at=utc_now(),
                manifest=None,
                failure_code=FailureCode.STORAGE_ERROR,
                message=str(exc),
                operation_statuses=operation_statuses,
            )
        self._telemetry.emit_metric(
            "checkpoint.persist_manifest_ms",
            (time.perf_counter() - manifest_started) * 1000.0,
            {"sandbox_id": str(job.sandbox_id), "checkpoint_id": str(checkpoint_id)},
        )
        persist_manifest.finish(status="succeeded")
        if process_step is not None:
            self._telemetry.emit_metric(
                "checkpoint.process_ms",
                process_duration_ms,
                {
                    "sandbox_id": str(job.sandbox_id),
                    "checkpoint_id": str(checkpoint_id),
                    "job_id": str(job.job_id),
                    "checkpoint_scope": checkpoint_scope,
                },
            )
        if fs_step is not None:
            self._telemetry.emit_metric(
                "checkpoint.filesystem_ms",
                filesystem_duration_ms,
                {
                    "sandbox_id": str(job.sandbox_id),
                    "checkpoint_id": str(checkpoint_id),
                    "job_id": str(job.job_id),
                    "checkpoint_scope": checkpoint_scope,
                },
            )
        if process_size_bytes is not None:
            self._telemetry.emit_metric(
                "checkpoint.process.size_bytes",
                process_size_bytes,
                {
                    "sandbox_id": str(job.sandbox_id),
                    "checkpoint_id": str(checkpoint_id),
                    "job_id": str(job.job_id),
                    "checkpoint_scope": checkpoint_scope,
                },
            )
        if process_file_count is not None:
            self._telemetry.emit_metric(
                "checkpoint.process.file_count",
                process_file_count,
                {
                    "sandbox_id": str(job.sandbox_id),
                    "checkpoint_id": str(checkpoint_id),
                    "job_id": str(job.job_id),
                    "checkpoint_scope": checkpoint_scope,
                },
            )
        if filesystem_written_bytes is not None:
            self._telemetry.emit_metric(
                "checkpoint.filesystem.written_bytes",
                filesystem_written_bytes,
                {
                    "sandbox_id": str(job.sandbox_id),
                    "checkpoint_id": str(checkpoint_id),
                    "job_id": str(job.job_id),
                    "checkpoint_scope": checkpoint_scope,
                },
            )
        if filesystem_used_bytes is not None:
            self._telemetry.emit_metric(
                "checkpoint.filesystem.used_bytes",
                filesystem_used_bytes,
                {
                    "sandbox_id": str(job.sandbox_id),
                    "checkpoint_id": str(checkpoint_id),
                    "job_id": str(job.job_id),
                    "checkpoint_scope": checkpoint_scope,
                },
            )
        if estimated_io_bytes is not None:
            self._telemetry.emit_metric(
                "checkpoint.estimated_io_bytes",
                estimated_io_bytes,
                {
                    "sandbox_id": str(job.sandbox_id),
                    "checkpoint_id": str(checkpoint_id),
                    "job_id": str(job.job_id),
                    "checkpoint_scope": checkpoint_scope,
                },
            )
        self._telemetry.emit_metric(
            "checkpoint.total_ms",
            (utc_now() - started).total_seconds() * 1000.0,
            {
                "sandbox_id": str(job.sandbox_id),
                "checkpoint_id": str(checkpoint_id),
                "job_id": str(job.job_id),
                "checkpoint_scope": checkpoint_scope,
            },
        )

        logger.info(
            "Composite checkpoint succeeded for job %s sandbox=%s checkpoint=%s",
            job.job_id,
            job.sandbox_id,
            checkpoint_id,
        )
        return CheckpointResult(
            job_id=job.job_id,
            sandbox_id=job.sandbox_id,
            checkpoint_id=checkpoint_id,
            status=JobStatus.SUCCEEDED,
            started_at=started,
            finished_at=utc_now(),
            manifest=manifest,
            operation_statuses=operation_statuses,
        )

    def _ensure_restorable_checkpoint(self, job: CheckpointJob) -> CheckpointJob:
        if job.checkpoint_process or not job.checkpoint_filesystem:
            return job
        if self._has_process_ancestor(job.sandbox_id):
            return job
        logger.info(
            "Promoting checkpoint scope to include process for sandbox %s because no prior process checkpoint exists",
            job.sandbox_id,
        )
        metadata = dict(job.metadata)
        metadata.setdefault("promoted_process_checkpoint", True)
        metadata.setdefault("promoted_process_checkpoint_reason", "missing_process_ancestor")
        return replace(job, checkpoint_process=True, metadata=metadata)

    def _has_process_ancestor(self, sandbox_id) -> bool:
        try:
            checkpoint_ids = self._checkpoint_manager.list_checkpoints(sandbox_id)
        except Exception:
            logger.warning(
                "Falling back to process checkpoint for sandbox %s because existing checkpoints could not be listed",
                sandbox_id,
            )
            return False

        for checkpoint_id in reversed(checkpoint_ids):
            try:
                manifest = self._checkpoint_manager.get_manifest(sandbox_id, checkpoint_id)
            except FileNotFoundError:
                continue
            if manifest.process_artifacts:
                return True
        return False

    def _timed_checkpoint_process(
        self,
        job: CheckpointJob,
        checkpoint_id: CheckpointId,
    ) -> tuple[WorkerStepResult, float]:
        started = time.perf_counter()
        operation = start_operation(
            self._telemetry,
            "checkpoint.process",
            {
                "component": "checkpoint",
                "job_id": str(job.job_id),
                "sandbox_id": str(job.sandbox_id),
                "checkpoint_id": str(checkpoint_id),
            },
        )
        result = self._process_worker.checkpoint(job, checkpoint_id)
        duration_ms = (time.perf_counter() - started) * 1000.0
        operation.finish(status="succeeded" if result.success else "failed")
        return result, duration_ms

    def _timed_checkpoint_filesystem(
        self,
        job: CheckpointJob,
        checkpoint_id: CheckpointId,
    ) -> tuple[WorkerStepResult, float]:
        started = time.perf_counter()
        operation = start_operation(
            self._telemetry,
            "checkpoint.filesystem",
            {
                "component": "checkpoint",
                "job_id": str(job.job_id),
                "sandbox_id": str(job.sandbox_id),
                "checkpoint_id": str(checkpoint_id),
            },
        )
        result = self._filesystem_worker.checkpoint(job, checkpoint_id)
        duration_ms = (time.perf_counter() - started) * 1000.0
        operation.finish(status="succeeded" if result.success else "failed")
        return result, duration_ms


class DefaultRWorker(CompositeRestoreWorker):
    def __init__(
        self,
        process_worker: ProcessRWorker,
        filesystem_worker: FileSystemRWorker,
        checkpoint_manager: CheckpointManager,
        telemetry: TelemetrySink | None = None,
        runtime: Runtime | None = None,
    ):
        self._process_worker = process_worker
        self._filesystem_worker = filesystem_worker
        self._checkpoint_manager = checkpoint_manager
        self._telemetry = telemetry or NoopTelemetrySink()
        self._runtime = runtime

    def restore(self, job: RestoreJob) -> RestoreResult:
        started = utc_now()
        logger.info(
            "Starting composite restore for job %s sandbox=%s checkpoint=%s",
            job.job_id,
            job.sandbox_id,
            job.checkpoint_id,
        )
        manifest_started = time.perf_counter()
        resolve_manifest_operation = start_operation(
            self._telemetry,
            "restore.resolve_manifest",
            {"component": "restore", "sandbox_id": str(job.sandbox_id), "checkpoint_id": str(job.checkpoint_id)},
        )
        try:
            manifest = self._checkpoint_manager.get_manifest(
                sandbox_id=job.sandbox_id,
                checkpoint_id=job.checkpoint_id,
            )
            manifest = resolve_restore_manifest(self._checkpoint_manager, manifest)
        except Exception as exc:
            resolve_manifest_operation.finish(status="failed")
            logger.exception(
                "Failed to load manifest for restore job %s sandbox=%s checkpoint=%s",
                job.job_id,
                job.sandbox_id,
                job.checkpoint_id,
            )
            return RestoreResult(
                job_id=job.job_id,
                sandbox_id=job.sandbox_id,
                checkpoint_id=job.checkpoint_id,
                status=JobStatus.FAILED,
                started_at=started,
                finished_at=utc_now(),
                failure_code=FailureCode.STORAGE_ERROR,
                message=str(exc),
            )
        self._telemetry.emit_metric(
            "restore.resolve_manifest_ms",
            (time.perf_counter() - manifest_started) * 1000.0,
            {"sandbox_id": str(job.sandbox_id), "checkpoint_id": str(job.checkpoint_id)},
        )
        resolve_manifest_operation.finish(status="succeeded")

        if self._runtime is not None:
            try:
                _validate_incremental_chain(
                    self._runtime, self._checkpoint_manager, manifest
                )
            except (FileNotFoundError, RuntimeError) as exc:
                logger.error(
                    "Incremental chain validation failed for restore job %s sandbox=%s checkpoint=%s: %s",
                    job.job_id,
                    job.sandbox_id,
                    job.checkpoint_id,
                    exc,
                )
                return RestoreResult(
                    job_id=job.job_id,
                    sandbox_id=job.sandbox_id,
                    checkpoint_id=job.checkpoint_id,
                    status=JobStatus.FAILED,
                    started_at=started,
                    finished_at=utc_now(),
                    failure_code=FailureCode.STORAGE_ERROR,
                    message=str(exc),
                )

        restore_metric_attributes = _restore_metric_attributes(job, manifest)
        restore_source_gap_turns = _safe_int(manifest.metadata.get(_RESTORE_SOURCE_GAP_TURNS))
        restore_source_gap_ms = _safe_float(manifest.metadata.get(_RESTORE_SOURCE_GAP_MS))
        restore_estimated_io_bytes = _safe_int(manifest.metadata.get(_RESTORE_ESTIMATED_IO_BYTES))

        operation_statuses = []

        fs_step = None
        filesystem_duration_ms = 0.0
        if manifest.filesystem_artifacts:
            restore_started = time.perf_counter()
            fs_operation = start_operation(
                self._telemetry,
                "restore.filesystem",
                {"component": "restore", "sandbox_id": str(job.sandbox_id), "checkpoint_id": str(job.checkpoint_id)},
            )
            fs_step = self._filesystem_worker.restore(job, manifest)
            filesystem_duration_ms = (time.perf_counter() - restore_started) * 1000.0
            fs_operation.finish(status="succeeded" if fs_step.success else "failed")
        if fs_step is not None and not fs_step.success:
            logger.warning(
                "Filesystem restore step failed for job %s sandbox=%s code=%s message=%s",
                job.job_id,
                job.sandbox_id,
                fs_step.failure_code.value,
                fs_step.message,
            )
            return RestoreResult(
                job_id=job.job_id,
                sandbox_id=job.sandbox_id,
                checkpoint_id=job.checkpoint_id,
                status=JobStatus.FAILED,
                started_at=started,
                finished_at=utc_now(),
                failure_code=fs_step.failure_code,
                message=fs_step.message,
                operation_statuses=(fs_step.operation_status,),
            )
        if fs_step is not None:
            operation_statuses.append(fs_step.operation_status)
            self._telemetry.emit_metric(
                "restore.filesystem_ms",
                filesystem_duration_ms,
                restore_metric_attributes,
            )

        process_step = None
        process_duration_ms = 0.0
        if manifest.process_artifacts:
            restore_started = time.perf_counter()
            process_operation = start_operation(
                self._telemetry,
                "restore.process",
                {"component": "restore", "sandbox_id": str(job.sandbox_id), "checkpoint_id": str(job.checkpoint_id)},
            )
            process_step = self._process_worker.restore(job, manifest)
            process_duration_ms = (time.perf_counter() - restore_started) * 1000.0
            process_operation.finish(status="succeeded" if process_step.success else "failed")
        if process_step is not None and not process_step.success:
            logger.warning(
                "Process restore step failed for job %s sandbox=%s code=%s message=%s",
                job.job_id,
                job.sandbox_id,
                process_step.failure_code.value,
                process_step.message,
            )
            return RestoreResult(
                job_id=job.job_id,
                sandbox_id=job.sandbox_id,
                checkpoint_id=job.checkpoint_id,
                status=JobStatus.FAILED,
                started_at=started,
                finished_at=utc_now(),
                failure_code=process_step.failure_code,
                message=process_step.message,
                operation_statuses=tuple(operation_statuses + [process_step.operation_status]),
            )
        if process_step is not None:
            operation_statuses.append(process_step.operation_status)
            self._telemetry.emit_metric(
                "restore.process_ms",
                process_duration_ms,
                restore_metric_attributes,
            )
        if restore_source_gap_turns is not None:
            self._telemetry.emit_metric(
                "restore.source_gap.turns",
                restore_source_gap_turns,
                restore_metric_attributes,
            )
        if restore_source_gap_ms is not None:
            self._telemetry.emit_metric(
                "restore.source_gap.ms",
                restore_source_gap_ms,
                restore_metric_attributes,
            )
        if restore_estimated_io_bytes is not None:
            self._telemetry.emit_metric(
                "restore.estimated_io_bytes",
                restore_estimated_io_bytes,
                restore_metric_attributes,
            )

        logger.info(
            "Composite restore succeeded for job %s sandbox=%s checkpoint=%s",
            job.job_id,
            job.sandbox_id,
            job.checkpoint_id,
        )
        self._telemetry.emit_metric(
            "restore.total_ms",
            (utc_now() - started).total_seconds() * 1000.0,
            restore_metric_attributes,
        )
        return RestoreResult(
            job_id=job.job_id,
            sandbox_id=job.sandbox_id,
            checkpoint_id=job.checkpoint_id,
            status=JobStatus.SUCCEEDED,
            started_at=started,
            finished_at=utc_now(),
            operation_statuses=tuple(operation_statuses),
        )


def _checkpoint_scope(job: CheckpointJob) -> str:
    if job.checkpoint_process and job.checkpoint_filesystem:
        return "full"
    if job.checkpoint_process:
        return "process_only"
    if job.checkpoint_filesystem:
        return "filesystem_only"
    return "none"


def _safe_int(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _checkpoint_estimated_io_bytes(
    *,
    process_size_bytes: int | None,
    filesystem_written_bytes: int | None,
) -> int | None:
    values = [value for value in (process_size_bytes, filesystem_written_bytes) if value is not None]
    if not values:
        return None
    return sum(values)


def _restore_metric_attributes(job: RestoreJob, manifest: CheckpointManifest) -> dict[str, object]:
    attributes: dict[str, object] = {
        "sandbox_id": str(job.sandbox_id),
        "checkpoint_id": str(job.checkpoint_id),
        "job_id": str(job.job_id),
        "restore.process_source_checkpoint_id": str(manifest.metadata.get(_PROCESS_RESTORE_CHECKPOINT_ID, "")),
        "restore.filesystem_source_checkpoint_id": str(manifest.metadata.get(_FILESYSTEM_RESTORE_CHECKPOINT_ID, "")),
        "restore.process_source_created_at": str(manifest.metadata.get(_PROCESS_RESTORE_CREATED_AT, "")),
        "restore.filesystem_source_created_at": str(manifest.metadata.get(_FILESYSTEM_RESTORE_CREATED_AT, "")),
        "restore.process_source_trace_cursor": manifest.metadata.get(_PROCESS_RESTORE_TRACE_CURSOR, ""),
        "restore.filesystem_source_trace_cursor": manifest.metadata.get(_FILESYSTEM_RESTORE_TRACE_CURSOR, ""),
        "restore.mixed_sources": bool(manifest.metadata.get(_RESTORE_MIXED_SOURCES, False)),
    }
    return attributes
