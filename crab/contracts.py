from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Iterable

from .ids import CheckpointId, JobId, SandboxId
from .models import (
    ArtifactPayload,
    ArtifactReference,
    CheckpointJob,
    CheckpointManifest,
    CheckpointResult,
    RequestContext,
    RequestStateChange,
    RestoreJob,
    RestoreResult,
    EBPFEvent,
    RuntimeOperationStatus,
    RuntimeCapabilities,
    SandboxDescription,
    SandboxExecResult,
    SandboxRuntimeState,
    SandboxSnapshot,
    SchedulerCheckpointDecision,
    WorkerStepResult,
)


class Runtime(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @property
    def version(self) -> str | None:
        return None

    @abstractmethod
    def capabilities(self) -> RuntimeCapabilities:
        raise NotImplementedError

    @abstractmethod
    def launch(self, runtime_name: str, metadata: dict[str, object] | None = None) -> SandboxId:
        raise NotImplementedError

    @abstractmethod
    def stop(self, sandbox_id: SandboxId) -> None:
        raise NotImplementedError

    @abstractmethod
    def pause(self, sandbox_id: SandboxId) -> None:
        raise NotImplementedError

    @abstractmethod
    def resume(self, sandbox_id: SandboxId) -> None:
        raise NotImplementedError

    @abstractmethod
    def sync_runtime_state(self, sandbox_id: SandboxId, *, is_running: bool) -> None:
        raise NotImplementedError

    @abstractmethod
    def prepare_for_restore(self, sandbox_id: SandboxId) -> None:
        raise NotImplementedError

    @abstractmethod
    def mark_restored(self, sandbox_id: SandboxId) -> None:
        raise NotImplementedError

    @abstractmethod
    def delete(self, sandbox_id: SandboxId) -> None:
        raise NotImplementedError

    @abstractmethod
    def describe(self, sandbox_id: SandboxId) -> SandboxDescription:
        raise NotImplementedError

    @abstractmethod
    def write_bundle_spec(self, bundle_dir: Path) -> None:
        raise NotImplementedError

    @abstractmethod
    def inspect_runtime(self, sandbox_id: SandboxId) -> SandboxRuntimeState:
        raise NotImplementedError

    @abstractmethod
    def exec(
        self,
        sandbox_id: SandboxId,
        argv: list[str],
        *,
        cwd: str | None = None,
        env: dict[str, object] | None = None,
        user: str | None = None,
        timeout_s: float | None = None,
        capture_output: bool = True,
    ) -> SandboxExecResult:
        raise NotImplementedError

    @abstractmethod
    def resilient_exec(
        self,
        sandbox_id: SandboxId,
        argv: list[str],
        *,
        cwd: str | None = None,
        env: dict[str, object] | None = None,
        user: str | None = None,
        timeout_s: float | None = None,
        capture_output: bool = True,
    ) -> SandboxExecResult:
        raise NotImplementedError

    @abstractmethod
    def checkpoint_process(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
        *,
        leave_running: bool,
        parent_checkpoint_id: CheckpointId | None = None,
    ) -> RuntimeOperationStatus:
        raise NotImplementedError

    def pre_dump_process(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
        *,
        parent_checkpoint_id: CheckpointId | None = None,
    ) -> RuntimeOperationStatus:
        """Memory-only pre-dump that establishes a parent for subsequent
        incremental checkpoints. The container keeps running. Subclasses that
        do not support incremental process checkpointing leave this as the
        default no-op; the worker layer only invokes it when the runtime's
        capability advertises support.
        """
        _ = parent_checkpoint_id
        return RuntimeOperationStatus(
            executed=False,
            reason=f"{self.name}_pre_dump_not_implemented",
            metadata={
                "phase": "process_pre_dump",
                "runtime": self.name,
                "sandbox_id": str(sandbox_id),
                "checkpoint_id": str(checkpoint_id),
            },
        )

    @abstractmethod
    def restore_process(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> RuntimeOperationStatus:
        raise NotImplementedError

    @abstractmethod
    def process_checkpoint_location(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> str | None:
        raise NotImplementedError

    def pre_dump_location(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> str | None:
        """On-disk path of the pre-dump image directory, when this runtime
        supports incremental process checkpointing. Default: None.
        """
        _ = (sandbox_id, checkpoint_id)
        return None

    def link_ancestor_pre_dump(
        self,
        source_sandbox_id: SandboxId,
        target_sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> bool:
        """Make a chain ancestor's process checkpoint image directory visible
        under ``target_sandbox_id``'s runtime checkpoint root by symlinking
        instead of byte-copying. Used during fork prep on incremental chains:
        without this, ``clone_checkpoint_to_fork`` would ``shutil.copytree``
        every ancestor's gigabyte-scale CRIU image into the fork's tree.

        CRIU walks the ``parent`` symlink chain inside the image set during
        ``runc restore``; that chain is internal to CRIU's images, so the
        per-checkpoint directory symlinks set up here are transparent to
        restore. Returns True when the link was created (or already in
        place); False when the runtime does not support pre-dump linking.
        Default: False (no-op for runtimes without incremental process
        support).
        """
        _ = (source_sandbox_id, target_sandbox_id, checkpoint_id)
        return False

    def materialize_linked_pre_dumps(self, sandbox_id: SandboxId) -> int:
        """Replace any pre-dump dir symlinks under ``sandbox_id``'s runtime
        checkpoint root with byte copies of their targets. Inverse of
        ``link_ancestor_pre_dump``: used before destroying a source sandbox
        whose ancestor images are still referenced by a fork that has been
        promoted into the active role. Returns the count of dirs
        materialized. Default: 0 (no-op for runtimes that never link).
        """
        _ = sandbox_id
        return 0

    def runtime_image_path_in_use(self, path: Path) -> bool:
        """Storage retention safety predicate. Returns True when ``path``
        (or a descendant) is the on-disk image source for a runtime
        operation whose mid-flight failure would manifest as a fatal
        signal rather than a clean error — currently this means any
        active ``criu lazy-pages`` daemon serving userfaultfd page
        faults from that tree. Pruning the runtime tree from under such
        a daemon SIGBUSes the restored process; storage layers that
        consult this predicate before ``shutil.rmtree``-ing a runtime
        checkpoint dir defer the prune (with a logged warning) until
        the daemon exits.

        Default: False (no-op for runtimes that don't issue daemon-
        backed restores). The runc runtime overrides this to consult
        its lazy-pages daemon registry.
        """
        _ = path
        return False

    @abstractmethod
    def checkpoint_filesystem(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> RuntimeOperationStatus:
        raise NotImplementedError

    @abstractmethod
    def restore_filesystem(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> RuntimeOperationStatus:
        raise NotImplementedError

    @abstractmethod
    def filesystem_checkpoint_metadata(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> dict[str, object]:
        raise NotImplementedError

    def discard_partial_checkpoint(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> None:
        """Remove any artifacts a checkpoint step left on disk before the
        composite checkpoint failed. Best-effort; safe to call when nothing
        was written. Subclasses should override to clean up runtime-specific
        scratch (CRIU image dir, ZFS @snapshot) so a failure of one composite
        step does not orphan the other step's output.
        """
        return None

    @abstractmethod
    def delete_runtime(
        self,
        sandbox_id: SandboxId,
        *,
        force: bool = True,
        ignore_missing: bool = True,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def destroy_filesystem_dataset(self, sandbox_id: SandboxId) -> None:
        raise NotImplementedError

    @abstractmethod
    def clone_filesystem_snapshot(
        self,
        source_sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
        target_sandbox_id: SandboxId,
        *,
        target_rootfs_path: Path,
    ) -> str:
        raise NotImplementedError


class ProcessCWorker(ABC):
    @abstractmethod
    def checkpoint(
        self,
        job: CheckpointJob,
        checkpoint_id: CheckpointId,
    ) -> WorkerStepResult:
        raise NotImplementedError


class ProcessRWorker(ABC):
    @abstractmethod
    def restore(
        self,
        job: RestoreJob,
        manifest: CheckpointManifest,
    ) -> WorkerStepResult:
        raise NotImplementedError


class FileSystemCWorker(ABC):
    @abstractmethod
    def checkpoint(
        self,
        job: CheckpointJob,
        checkpoint_id: CheckpointId,
    ) -> WorkerStepResult:
        raise NotImplementedError


class FileSystemRWorker(ABC):
    @abstractmethod
    def restore(
        self,
        job: RestoreJob,
        manifest: CheckpointManifest,
    ) -> WorkerStepResult:
        raise NotImplementedError


class CheckpointManager(ABC):
    @abstractmethod
    def put_manifest(self, manifest: CheckpointManifest) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_manifest(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> CheckpointManifest:
        raise NotImplementedError

    @abstractmethod
    def put_artifact(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
        artifact: ArtifactPayload,
    ) -> ArtifactReference:
        raise NotImplementedError

    @abstractmethod
    def get_artifact(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
        reference: ArtifactReference,
    ) -> bytes:
        raise NotImplementedError

    @abstractmethod
    def list_checkpoints(self, sandbox_id: SandboxId) -> list[CheckpointId]:
        raise NotImplementedError

    @abstractmethod
    def delete_checkpoint(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
        *,
        cascade: bool = False,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def delete_all_checkpoints(self, sandbox_id: SandboxId) -> None:
        raise NotImplementedError

    def descendants(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> list[CheckpointId]:
        """Transitive incremental descendants of ``checkpoint_id``.
        Default implementation returns an empty list; managers that
        track ``parent_checkpoint_id`` should override.
        """
        _ = (sandbox_id, checkpoint_id)
        return []

    @abstractmethod
    def handle_checkpoint_complete(self, manifest: CheckpointManifest) -> None:
        raise NotImplementedError

    @abstractmethod
    def handle_restore_complete(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> None:
        raise NotImplementedError


class RemoteCheckpointBackend(ABC):
    @abstractmethod
    def upload_manifest(self, manifest: CheckpointManifest) -> None:
        raise NotImplementedError

    @abstractmethod
    def download_manifest(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> CheckpointManifest:
        raise NotImplementedError

    @abstractmethod
    def upload_artifact(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
        reference: ArtifactReference,
        data: bytes,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def download_artifact(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
        reference: ArtifactReference,
    ) -> bytes:
        raise NotImplementedError


class SandboxInspector(ABC):
    @abstractmethod
    def inspect(self, sandbox_id: SandboxId) -> SandboxSnapshot:
        raise NotImplementedError

    @abstractmethod
    def mark_checkpoint_complete(
        self,
        sandbox_id: SandboxId,
        *,
        process: bool,
        filesystem: bool,
        at: datetime,
    ) -> None:
        raise NotImplementedError


class EBPFEventCollector(ABC):
    @abstractmethod
    def poll(self, sandbox_id: SandboxId, since: datetime | None = None) -> list[EBPFEvent]:
        raise NotImplementedError


class RequestInterceptorHook(ABC):
    @abstractmethod
    def on_request_start(self, context: RequestContext) -> None:
        raise NotImplementedError

    @abstractmethod
    def on_request_end(self, context: RequestContext) -> None:
        raise NotImplementedError


class TelemetrySink(ABC):
    @abstractmethod
    def emit_event(self, name: str, attributes: dict[str, object]) -> None:
        raise NotImplementedError

    @abstractmethod
    def emit_metric(
        self,
        name: str,
        value: float,
        attributes: dict[str, object] | None = None,
    ) -> None:
        raise NotImplementedError

    def flush(self) -> None:
        return

    def close(self) -> None:
        return


class CompositeCheckpointWorker(ABC):
    @abstractmethod
    def checkpoint(self, job: CheckpointJob) -> CheckpointResult:
        raise NotImplementedError


class CompositeRestoreWorker(ABC):
    @abstractmethod
    def restore(self, job: RestoreJob) -> RestoreResult:
        raise NotImplementedError


class SchedulerStateStore(ABC):
    @abstractmethod
    def set_last_checkpoint(self, sandbox_id: SandboxId, checkpoint_time: datetime) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_last_checkpoint(self, sandbox_id: SandboxId) -> datetime | None:
        raise NotImplementedError

    def record_process_checkpoint(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
        *,
        is_incremental: bool,
    ) -> None:
        """Record that a process checkpoint just succeeded. Updates the
        per-sandbox chain bookkeeping the scheduler consults to decide
        full vs incremental for the next checkpoint. Default no-op for
        stores that don't track chain state.
        """
        _ = (sandbox_id, checkpoint_id, is_incremental)

    def get_last_process_checkpoint(self, sandbox_id: SandboxId) -> CheckpointId | None:
        """Return the most recent successful process checkpoint id, or
        None if there isn't one yet.
        """
        _ = sandbox_id
        return None

    def get_process_chain_length(self, sandbox_id: SandboxId) -> int:
        """Number of incremental process checkpoints since the last full
        process checkpoint. 0 means the chain is anchored at a full.
        """
        _ = sandbox_id
        return 0
