from __future__ import annotations

import json
import logging
from pathlib import Path

from ..contracts import ProcessCWorker, ProcessRWorker, Runtime
from ..ids import CheckpointId
from ..models import ArtifactKind, ArtifactPayload, CheckpointJob, CheckpointManifest, RestoreJob, WorkerStepResult

logger = logging.getLogger(__name__)


def _metadata_artifact(name: str, payload: dict[str, object], *, runtime_name: str) -> ArtifactPayload:
    return ArtifactPayload(
        kind=ArtifactKind.PROCESS,
        name=name,
        data=json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"),
        metadata={"runtime": runtime_name},
    )


class AdapterProcessCWorker(ProcessCWorker):
    def __init__(self, runtime: Runtime):
        self._runtime = runtime

    def checkpoint(self, job: CheckpointJob, checkpoint_id: CheckpointId) -> WorkerStepResult:
        logger.debug(
            "Running process checkpoint worker for sandbox=%s checkpoint=%s runtime=%s",
            job.sandbox_id,
            checkpoint_id,
            self._runtime.name,
        )
        runtime = self._runtime
        runtime_supports_incremental = runtime.capabilities().supports_incremental_process
        # do_pair: run pre-dump + final-dump pair so this checkpoint produces a
        # pre_dump dir that subsequent checkpoints can chain off of. Anchors
        # (parent=None) and chain nodes (parent=set) both take this path when
        # the chain feature is active and the runtime supports it.
        do_pair = job.produce_pre_dump and runtime_supports_incremental
        if job.produce_pre_dump and not runtime_supports_incremental:
            logger.warning(
                "Runtime %s does not support incremental process checkpoints; "
                "falling back to single full dump for sandbox=%s checkpoint=%s",
                runtime.name,
                job.sandbox_id,
                checkpoint_id,
            )

        pre_dump_status = None
        pre_dump_image_path = None
        if do_pair:
            pre_dump_status = runtime.pre_dump_process(
                job.sandbox_id,
                checkpoint_id,
                parent_checkpoint_id=job.parent_process_checkpoint_id,
            )
            pre_dump_image_path = runtime.pre_dump_location(job.sandbox_id, checkpoint_id)
            logger.debug(
                "Pre-dump phase finished sandbox=%s checkpoint=%s executed=%s parent=%s",
                job.sandbox_id,
                checkpoint_id,
                pre_dump_status.executed,
                job.parent_process_checkpoint_id,
            )

        # When the pair is active, the final dump's parent is the pre-dump we
        # just produced (sibling under the same checkpoint id). Without the
        # pair, no parent — produces a standalone restorable dump as before.
        final_parent = checkpoint_id if do_pair else None
        status = runtime.checkpoint_process(
            job.sandbox_id,
            checkpoint_id,
            leave_running=job.leave_running,
            parent_checkpoint_id=final_parent,
        )
        image_path = runtime.process_checkpoint_location(job.sandbox_id, checkpoint_id)
        # process_kind reflects whether THIS checkpoint chains off a parent.
        # An anchor produces a pre_dump but is not itself "incremental" —
        # it is the chain root.
        is_chain_node = do_pair and job.parent_process_checkpoint_id is not None
        process_kind = "incremental" if is_chain_node else "full"
        payload: dict[str, object] = {
            "sandbox_id": str(job.sandbox_id),
            "checkpoint_id": str(checkpoint_id),
            "leave_running": job.leave_running,
            "process_kind": process_kind,
            "process_storage_mode": "runtime_reference" if image_path else "adapter_default",
            "process_checkpoint_location": image_path,
            "status": {
                "executed": status.executed,
                "reason": status.reason,
                "command": list(status.command),
                "metadata": status.metadata,
            },
        }
        if do_pair:
            payload["pre_dump_location"] = pre_dump_image_path
            assert pre_dump_status is not None
            payload["pre_dump_status"] = {
                "executed": pre_dump_status.executed,
                "reason": pre_dump_status.reason,
                "command": list(pre_dump_status.command),
                "metadata": pre_dump_status.metadata,
            }
            if job.parent_process_checkpoint_id is not None:
                payload["parent_checkpoint_id"] = str(job.parent_process_checkpoint_id)
        artifacts = [_metadata_artifact("process_checkpoint.json", payload, runtime_name=runtime.name)]
        logger.debug(
            "Process checkpoint worker finished for sandbox=%s checkpoint=%s executed=%s "
            "process_kind=%s artifacts=%d",
            job.sandbox_id,
            checkpoint_id,
            status.executed,
            process_kind,
            len(artifacts),
        )
        return WorkerStepResult(success=True, artifacts=artifacts, operation_status=status)


class AdapterProcessRWorker(ProcessRWorker):
    def __init__(self, runtime: Runtime):
        self._runtime = runtime

    def restore(self, job: RestoreJob, manifest: CheckpointManifest) -> WorkerStepResult:
        restore_checkpoint_id = _restore_checkpoint_id(
            manifest,
            metadata_key="process_restore_checkpoint_id",
            default=job.checkpoint_id,
        )
        if self._runtime.capabilities().supports_custom_checkpoint_dir:
            image_path = self._runtime.process_checkpoint_location(job.sandbox_id, restore_checkpoint_id)
            if image_path is None:
                raise ValueError("runtime did not provide process checkpoint location for restore")
            checkpoint_dir = Path(str(image_path))
            if not checkpoint_dir.exists():
                raise FileNotFoundError(f"process checkpoint directory not found: {checkpoint_dir}")
        lazy_pages = job.metadata.get("lazy_pages")
        if lazy_pages is not None and self._runtime.capabilities().supports_lazy_restore:
            # Per-call override threaded from RestoreJob metadata (fork
            # restores set it; see Engine.fork_sandbox). Runtimes without
            # lazy-restore support ignore the request.
            status = self._runtime.restore_process(
                job.sandbox_id,
                restore_checkpoint_id,
                lazy_pages=bool(lazy_pages),
            )
        else:
            status = self._runtime.restore_process(job.sandbox_id, restore_checkpoint_id)
        logger.debug(
            "Process restore worker finished for sandbox=%s checkpoint=%s restore_checkpoint=%s executed=%s",
            job.sandbox_id,
            job.checkpoint_id,
            restore_checkpoint_id,
            status.executed,
        )
        return WorkerStepResult(success=True, artifacts=[], operation_status=status)


def _restore_checkpoint_id(
    manifest: CheckpointManifest,
    *,
    metadata_key: str,
    default: CheckpointId,
) -> CheckpointId:
    raw = manifest.metadata.get(metadata_key)
    if raw is None:
        return default
    return CheckpointId(str(raw))
