from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Protocol

from ..ids import CheckpointId, SandboxId
from ..models import ChangesetEntry, RuntimeOperationStatus
from .base import CommandResult

# When a backend reports several change kinds for one path, the highest
# precedence wins (shared by every provider's parser).
CHANGE_PRECEDENCE = {"removed": 3, "renamed": 2, "added": 1, "modified": 0}


class FsCommandExecutor(Protocol):
    """Command execution pipeline lent to the provider by the owning
    runtime (``RuncRuntime._run_command``). Routing provider commands
    through the runtime keeps telemetry operations, attributes, and
    error handling identical to the pre-extraction behavior."""

    def __call__(
        self,
        command: list[str],
        *,
        operation: str,
        sandbox_id: SandboxId | None = None,
        checkpoint_id: CheckpointId | None = None,
        cwd: Path | None = None,
        check: bool = True,
        expected_error_substrings: tuple[str, ...] = (),
        metadata: dict[str, object] | None = None,
        timeout_seconds: float | None = None,
    ) -> CommandResult:
        ...


class FsStatusExecutor(Protocol):
    """Status-producing variant (``RuncRuntime._run_status``)."""

    def __call__(
        self,
        command: list[str],
        *,
        operation: str,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId | None = None,
        metadata: dict[str, object] | None = None,
    ) -> RuntimeOperationStatus:
        ...


class DatasetResolver(Protocol):
    """Maps a sandbox to its live filesystem dataset name. Owned by the
    runtime because the mapping consults per-sandbox descriptions
    (``zfs_dataset`` metadata overrides)."""

    def __call__(self, sandbox_id: SandboxId) -> str:
        ...


class RootfsResolver(Protocol):
    def __call__(self, sandbox_id: SandboxId) -> Path:
        ...


class FilesystemProvider(ABC):
    """CoW filesystem backend for sandbox rootfs checkpoint/clone.

    The provider owns backend-specific naming, command construction, and
    retry/error semantics (today: ZFS snapshot/rollback/clone/promote).
    Callers identify state by ``(sandbox_id, checkpoint_id)`` or by
    dataset/snapshot names the provider itself produced. Sandbox-state
    dependent resolution (per-sandbox dataset overrides) stays in the
    runtime and reaches the provider through resolver callables.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend identifier, e.g. ``zfs``."""

    # ------------------------------------------------------------------
    # Naming / layout
    # ------------------------------------------------------------------

    @abstractmethod
    def default_dataset_name(self, sandbox_id: SandboxId) -> str:
        """Dataset name for a sandbox absent any per-sandbox override."""

    @abstractmethod
    def shared_rootfs_details(self, key: str, *, persist_across_runs: bool) -> tuple[str, Path]:
        """(dataset, mountpoint) for the shared rootfs image cache."""

    @abstractmethod
    def shared_rootfs_lock_path(self, key: str, *, persist_across_runs: bool) -> Path:
        ...

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @abstractmethod
    def object_exists(self, name: str) -> bool:
        """Dataset-or-snapshot existence check by exact name."""

    @abstractmethod
    def dataset_exists(self, dataset: str) -> bool:
        ...

    @abstractmethod
    def snapshot_exists(self, snapshot: str) -> bool:
        ...

    # ------------------------------------------------------------------
    # Launch / shared-rootfs mutations
    # ------------------------------------------------------------------

    @abstractmethod
    def create_dataset(
        self,
        dataset: str,
        mountpoint: Path,
        *,
        operation: str,
        sandbox_id: SandboxId | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        """Create a writable dataset mounted at ``mountpoint``."""

    @abstractmethod
    def create_snapshot(self, dataset: str, snapshot: str, *, operation: str) -> None:
        ...

    @abstractmethod
    def clone_shared_base(
        self,
        shared_dataset: str,
        shared_snapshot: str,
        dataset: str,
        rootfs_path: Path,
        *,
        sandbox_id: SandboxId,
    ) -> None:
        """Materialize a sandbox rootfs as a writable clone of the shared
        image snapshot."""

    @abstractmethod
    def destroy_dataset(
        self,
        dataset: str,
        *,
        operation: str,
        sandbox_id: SandboxId | None = None,
    ) -> None:
        """Best-effort recursive destroy (missing dataset is not an error)."""

    # ------------------------------------------------------------------
    # Checkpoint / restore / fork flows (Runtime contract surface)
    # ------------------------------------------------------------------

    @abstractmethod
    def checkpoint_filesystem(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> RuntimeOperationStatus:
        ...

    @abstractmethod
    def restore_filesystem(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> RuntimeOperationStatus:
        ...

    @abstractmethod
    def filesystem_checkpoint_metadata(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> dict[str, object]:
        ...

    @abstractmethod
    def destroy_filesystem_dataset(self, sandbox_id: SandboxId, dataset: str) -> None:
        """Destroy a sandbox's dataset, tolerating transient busy states."""

    @abstractmethod
    def promote_filesystem_dataset(self, sandbox_id: SandboxId) -> None:
        """Detach a clone from its origin so the source can be destroyed.

        Backends without clone-origin dependencies (btrfs) implement this
        as a no-op."""

    @abstractmethod
    def clone_filesystem_snapshot(
        self,
        source_sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
        target_sandbox_id: SandboxId,
        *,
        target_rootfs_path: Path,
    ) -> str:
        """Materialize a fork's rootfs from a source checkpoint snapshot.
        Returns the fork's dataset name."""

    @abstractmethod
    def discard_partial_checkpoint(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> None:
        """Best-effort removal of a checkpoint's filesystem snapshot after
        a partially failed composite checkpoint."""

    @abstractmethod
    def changeset_since(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> list[ChangesetEntry]:
        """Authoritative changed-path set of the live dataset relative to
        the local base snapshot ``{dataset}@{checkpoint_id}``, sorted by
        path. Raises ``FileNotFoundError`` when the base snapshot does
        not exist on this sandbox's dataset."""

    @abstractmethod
    def snapshot_content_root(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> Path:
        """Host-side directory exposing the read-only file content of the
        snapshot ``{dataset}@{checkpoint_id}`` (C2 merge reads three-way
        bases and rollback content from it). Raises ``FileNotFoundError``
        when the snapshot does not exist."""

    @abstractmethod
    def destroy_snapshot_ref(self, fs_ref: str) -> None:
        """Best-effort destroy of a filesystem checkpoint by its opaque
        ``fs_ref`` (recorded in the checkpoint's artifact payload by
        ``filesystem_checkpoint_metadata``). Storage retention routes
        snapshot deletion here instead of shelling out to a backend
        binary itself. Refs the provider does not recognize and
        already-deleted snapshots must be tolerated (log, don't raise):
        retention is best-effort and must never wedge on cleanup."""
