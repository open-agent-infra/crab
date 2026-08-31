from __future__ import annotations

import logging
import os
import re
import subprocess
import time
from dataclasses import replace
from pathlib import Path

from ..ids import CheckpointId, SandboxId
from ..models import RuntimeOperationStatus
from .base import CommandResult
from .fs_provider import (
    CHANGE_PRECEDENCE,
    ChangesetEntry,
    DatasetResolver,
    FilesystemProvider,
    FsCommandExecutor,
    FsStatusExecutor,
    RootfsResolver,
)

logger = logging.getLogger(__name__)

_DATASET_DESTROY_BUSY_RETRIES = 10
_DATASET_DESTROY_BUSY_RETRY_DELAY_S = 0.5
_DATASET_PROMOTE_CONFLICT_RETRIES = 8
_PROMOTE_CONFLICTING_SNAPSHOT_RE = re.compile(r"conflicting snapshot '([^']+)'")
_FS_REF_PREFIX = "zfs:"
_ZFS_DIFF_CHANGE_KINDS = {"+": "added", "M": "modified", "-": "removed", "R": "renamed"}
_ZFS_OCTAL_ESCAPE_RE = re.compile(r"\\(\d{4})")


def _decode_zfs_diff_path(raw: str) -> str:
    r"""zfs diff escapes non-printable path bytes as 4-digit octal
    (space => ``\0040``)."""
    return _ZFS_OCTAL_ESCAPE_RE.sub(lambda match: chr(int(match.group(1), 8)), raw)


def _container_path(raw: str, mountpoint: str) -> str | None:
    """Strip the live mountpoint prefix, yielding a ``/``-rooted
    container path; None when the path is outside the rootfs."""
    trimmed = mountpoint.rstrip("/")
    if raw.rstrip("/") == trimmed:
        return "/"
    if raw.startswith(trimmed + "/"):
        return raw[len(trimmed):]
    return None


def parse_zfs_diff(stdout: str, *, mountpoint: str) -> list[ChangesetEntry]:
    """Parse ``zfs diff -FH`` output (tab-separated: change, type, path
    [, rename-target]). Format pinned from real VM output — rename rows
    carry old and new as the 3rd and 4th tab fields. The rootfs root's
    own mtime churn ("/") is dropped as noise."""
    entries: dict[str, ChangesetEntry] = {}
    for line in stdout.splitlines():
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) < 3:
            continue
        change = _ZFS_DIFF_CHANGE_KINDS.get(fields[0])
        if change is None:
            continue
        path = _container_path(_decode_zfs_diff_path(fields[2]), mountpoint)
        if path is None or path == "/":
            continue
        renamed_from: str | None = None
        if change == "renamed":
            if len(fields) < 4:
                continue
            target = _container_path(_decode_zfs_diff_path(fields[3]), mountpoint)
            if target is None:
                continue
            renamed_from = path
            path = target
        existing = entries.get(path)
        if existing is not None:
            kinds = {existing.change, change}
            if kinds == {"added", "removed"}:
                # A write-temp-then-rename save (sed -i, editors) or a
                # delete-then-recreate shows up as the old inode removed
                # plus a new inode added at the same path: the path
                # survived with new content. btrfs's parser already
                # folds this shape to modified; keep parity.
                entries[path] = ChangesetEntry(path=path, change="modified")
                continue
            if kinds == {"renamed", "removed"}:
                # A rename onto an existing path replaces its inode;
                # keep the rename identity (the content moved in).
                if existing.change != "renamed":
                    existing = ChangesetEntry(path=path, change="renamed", renamed_from=renamed_from)
                entries[path] = existing
                continue
        if existing is None or CHANGE_PRECEDENCE[change] >= CHANGE_PRECEDENCE[existing.change]:
            entries[path] = ChangesetEntry(path=path, change=change, renamed_from=renamed_from)
    return [entries[key] for key in sorted(entries)]


class ZfsProvider(FilesystemProvider):
    """ZFS-backed filesystem provider.

    Extracted verbatim from ``RuncRuntime``: command lines, telemetry
    operation names, retry behavior, and error messages are unchanged.
    Commands run through the owning runtime's executor callables so the
    telemetry pipeline stays identical.
    """

    def __init__(
        self,
        *,
        dataset_prefix: str,
        runtime_name: str,
        run_command: FsCommandExecutor,
        run_status: FsStatusExecutor,
        dataset_resolver: DatasetResolver,
        rootfs_resolver: RootfsResolver,
        zfs_bin: str = "zfs",
    ) -> None:
        self._dataset_prefix = dataset_prefix
        self._runtime_name = runtime_name
        self._run_command = run_command
        self._run_status = run_status
        self._dataset_resolver = dataset_resolver
        self._rootfs_resolver = rootfs_resolver
        self._zfs_bin = zfs_bin

    @property
    def name(self) -> str:
        return "zfs"

    # ------------------------------------------------------------------
    # Naming / layout
    # ------------------------------------------------------------------

    def default_dataset_name(self, sandbox_id: SandboxId) -> str:
        return f"{self._dataset_prefix}/{sandbox_id}"

    def _safe_prefix(self) -> str:
        return "".join(
            char if char.isalnum() or char in {"-", "_", "."} else "_"
            for char in self._dataset_prefix
        )

    def shared_rootfs_details(self, key: str, *, persist_across_runs: bool) -> tuple[str, Path]:
        if persist_across_runs:
            dataset = f"{self._dataset_prefix}-cache-{key}"
        else:
            dataset = f"{self._dataset_prefix}/_shared_rootfs_{key}"
        scope = "persistent" if persist_across_runs else "run"
        mountpoint = Path("/tmp/crab-rootfs-cache") / self._safe_prefix() / scope / key
        return dataset, mountpoint

    def shared_rootfs_lock_path(self, key: str, *, persist_across_runs: bool) -> Path:
        scope = "persistent" if persist_across_runs else "run"
        return Path("/tmp/crab-rootfs-cache-locks") / self._safe_prefix() / scope / f"{key}.lock"

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def object_exists(self, name: str) -> bool:
        result = self._run_command(
            [self._zfs_bin, "list", "-H", "-o", "name", name],
            operation="sandbox.zfs_list",
            check=False,
            metadata={"name": name},
        )
        return result.returncode == 0

    def dataset_exists(self, dataset: str) -> bool:
        result = self._run_command(
            [self._zfs_bin, "list", "-H", "-o", "name", dataset],
            operation="sandbox.zfs_exists",
            check=False,
            metadata={"dataset": dataset},
        )
        return result.returncode == 0

    def snapshot_exists(self, snapshot: str) -> bool:
        result = self._run_command(
            [self._zfs_bin, "list", "-H", "-o", "name", "-t", "snapshot", snapshot],
            operation="sandbox.zfs_snapshot_exists",
            check=False,
            metadata={"snapshot": snapshot},
        )
        return result.returncode == 0

    def _query_origin(self, dataset: str) -> str | None:
        result = self._run_command(
            [self._zfs_bin, "get", "-H", "-o", "value", "origin", dataset],
            operation="sandbox.zfs_get_origin",
            check=False,
            metadata={"dataset": dataset},
        )
        if result.returncode != 0:
            return None
        value = result.stdout.strip()
        if not value or value == "-":
            return None
        return value

    def _query_snapshot_sizes(
        self,
        snapshot: str,
        *,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> tuple[int | None, int | None]:
        result = self._run_command(
            [self._zfs_bin, "get", "-Hp", "-o", "property,value", "written,used", snapshot],
            operation="sandbox.zfs_snapshot_stats",
            sandbox_id=sandbox_id,
            checkpoint_id=checkpoint_id,
            check=False,
            metadata={"snapshot": snapshot},
        )
        if result.returncode != 0:
            return None, None
        values: dict[str, int] = {}
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) < 2:
                continue
            property_name = parts[0].strip()
            raw_value = parts[1].strip()
            try:
                values[property_name] = int(raw_value)
            except ValueError:
                continue
        return values.get("written"), values.get("used")

    # ------------------------------------------------------------------
    # Launch / shared-rootfs mutations
    # ------------------------------------------------------------------

    def create_dataset(
        self,
        dataset: str,
        mountpoint: Path,
        *,
        operation: str,
        sandbox_id: SandboxId | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        self._run_command(
            [self._zfs_bin, "create", "-o", f"mountpoint={mountpoint}", dataset],
            operation=operation,
            sandbox_id=sandbox_id,
            metadata={"dataset": dataset, "mountpoint": str(mountpoint)},
            timeout_seconds=timeout_seconds,
        )

    def create_snapshot(self, dataset: str, snapshot: str, *, operation: str) -> None:
        self._run_command(
            [self._zfs_bin, "snapshot", snapshot],
            operation=operation,
            metadata={"dataset": dataset, "snapshot": snapshot},
        )

    def clone_shared_base(
        self,
        shared_dataset: str,
        shared_snapshot: str,
        dataset: str,
        rootfs_path: Path,
        *,
        sandbox_id: SandboxId,
    ) -> None:
        self._run_command(
            [self._zfs_bin, "clone", "-o", f"mountpoint={rootfs_path}", shared_snapshot, dataset],
            operation="sandbox.zfs_clone_launch_rootfs",
            sandbox_id=sandbox_id,
            metadata={
                "source_dataset": shared_dataset,
                "target_dataset": dataset,
                "snapshot": shared_snapshot,
                "mountpoint": str(rootfs_path),
            },
        )

    def destroy_dataset(
        self,
        dataset: str,
        *,
        operation: str,
        sandbox_id: SandboxId | None = None,
    ) -> None:
        self._run_command(
            [self._zfs_bin, "destroy", "-r", dataset],
            operation=operation,
            sandbox_id=sandbox_id,
            check=False,
            metadata={"dataset": dataset},
        )

    # ------------------------------------------------------------------
    # Checkpoint / restore / fork flows
    # ------------------------------------------------------------------

    def checkpoint_filesystem(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> RuntimeOperationStatus:
        dataset = self._dataset_resolver(sandbox_id)
        snapshot = f"{dataset}@{checkpoint_id}"
        status = self._run_status(
            [self._zfs_bin, "snapshot", snapshot],
            operation="sandbox.checkpoint_filesystem",
            sandbox_id=sandbox_id,
            checkpoint_id=checkpoint_id,
            metadata=self.filesystem_checkpoint_metadata(sandbox_id, checkpoint_id),
        )
        written_bytes, used_bytes = self._query_snapshot_sizes(snapshot, sandbox_id=sandbox_id, checkpoint_id=checkpoint_id)
        return replace(
            status,
            metadata={
                **status.metadata,
                "checkpoint_scope": "filesystem_only",
                "filesystem_checkpoint_written_bytes": written_bytes,
                "filesystem_checkpoint_used_bytes": used_bytes,
            },
        )

    def restore_filesystem(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> RuntimeOperationStatus:
        dataset = self._dataset_resolver(sandbox_id)
        snapshot = f"{dataset}@{checkpoint_id}"
        metadata: dict[str, object] = {
            "phase": "filesystem_restore",
            "runtime": self._runtime_name,
            "dataset": dataset,
            "snapshot": snapshot,
            "mountpoint": str(self._rootfs_resolver(sandbox_id)),
        }
        origin = self._query_origin(dataset)
        if origin is not None and origin.endswith(f"@{checkpoint_id}"):
            # Dataset is a clone of the requested snapshot; its live state
            # already matches, so rollback would be a no-op. Skipping avoids a
            # same-name snapshot on the clone which would later collide with
            # `zfs promote`.
            return RuntimeOperationStatus(
                executed=False,
                reason="clone_origin_matches_checkpoint",
                command=(self._zfs_bin, "rollback", "-r", snapshot),
                metadata={**metadata, "origin": origin},
            )
        return self._run_status(
            [self._zfs_bin, "rollback", "-r", snapshot],
            operation="sandbox.restore_filesystem",
            sandbox_id=sandbox_id,
            checkpoint_id=checkpoint_id,
            metadata=metadata,
        )

    def filesystem_checkpoint_metadata(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> dict[str, object]:
        dataset = self._dataset_resolver(sandbox_id)
        snapshot = f"{dataset}@{checkpoint_id}"
        return {
            "phase": "filesystem_checkpoint",
            "runtime": self._runtime_name,
            "dataset": dataset,
            "snapshot": snapshot,
            "mountpoint": str(self._rootfs_resolver(sandbox_id)),
            # Backend-neutral handle for storage retention. The legacy
            # `snapshot` key above stays for one release so pre-fs_ref
            # manifests and readers keep working.
            "fs_ref": f"{_FS_REF_PREFIX}{snapshot}",
        }

    def destroy_snapshot_ref(self, fs_ref: str) -> None:
        # Legacy artifact payloads carry a bare snapshot name instead of a
        # prefixed fs_ref; accept both so retention of old checkpoints
        # keeps working after the cutover.
        snapshot = fs_ref[len(_FS_REF_PREFIX):] if fs_ref.startswith(_FS_REF_PREFIX) else fs_ref
        if not snapshot:
            return
        result = self._run_command(
            [self._zfs_bin, "destroy", snapshot],
            operation="storage.zfs_destroy_snapshot_ref",
            check=False,
            metadata={"snapshot": snapshot, "fs_ref": fs_ref},
        )
        if result.returncode != 0:
            stderr = result.stderr.strip()
            if "dataset does not exist" in stderr or "snapshot does not exist" in stderr:
                return
            logger.warning(
                "Failed to destroy filesystem snapshot %s rc=%d stderr=%s",
                snapshot,
                result.returncode,
                stderr,
            )

    def destroy_filesystem_dataset(self, sandbox_id: SandboxId, dataset: str) -> None:
        result: CommandResult | None = None
        stderr = ""
        for attempt in range(1, _DATASET_DESTROY_BUSY_RETRIES + 1):
            result = self._run_command(
                [self._zfs_bin, "destroy", "-r", dataset],
                operation="sandbox.zfs_destroy",
                sandbox_id=sandbox_id,
                check=False,
                metadata={"dataset": dataset, "attempt": attempt},
            )
            stderr = result.stderr.strip()
            if result.returncode == 0 or "does not exist" in stderr:
                return
            if "dataset is busy" not in stderr:
                break
            if attempt < _DATASET_DESTROY_BUSY_RETRIES:
                logger.warning(
                    "ZFS dataset destroy reported busy; retrying sandbox=%s dataset=%s attempt=%d/%d stderr=%s",
                    sandbox_id,
                    dataset,
                    attempt,
                    _DATASET_DESTROY_BUSY_RETRIES,
                    stderr,
                )
                time.sleep(_DATASET_DESTROY_BUSY_RETRY_DELAY_S)
        assert result is not None
        if result.returncode != 0 and "does not exist" not in stderr:
            raise RuntimeError(
                f"command failed ({result.returncode}): {self._zfs_bin} destroy -r {dataset}"
                f"\nstdout: {result.stdout.strip()}"
                f"\nstderr: {stderr}"
            )

    def promote_filesystem_dataset(self, sandbox_id: SandboxId) -> None:
        dataset = self._dataset_resolver(sandbox_id)
        last_result: CommandResult | None = None
        last_stderr = ""
        for attempt in range(1, _DATASET_PROMOTE_CONFLICT_RETRIES + 1):
            result = self._run_command(
                [self._zfs_bin, "promote", dataset],
                operation="sandbox.zfs_promote",
                sandbox_id=sandbox_id,
                check=False,
                metadata={"dataset": dataset, "attempt": attempt},
            )
            stderr = result.stderr.strip()
            if result.returncode == 0:
                return
            if "not a cloned filesystem" in stderr or "does not exist" in stderr:
                return
            conflict_match = _PROMOTE_CONFLICTING_SNAPSHOT_RE.search(stderr)
            if conflict_match is None:
                last_result = result
                last_stderr = stderr
                break
            snapshot_name = conflict_match.group(1)
            logger.warning(
                "ZFS dataset promote reported conflicting snapshot; deleting child snapshot and retrying "
                "sandbox=%s dataset=%s snapshot=%s attempt=%d/%d",
                sandbox_id,
                dataset,
                snapshot_name,
                attempt,
                _DATASET_PROMOTE_CONFLICT_RETRIES,
            )
            self._destroy_dataset_snapshot(
                dataset,
                snapshot_name,
                sandbox_id=sandbox_id,
            )
            last_result = result
            last_stderr = stderr
        assert last_result is not None
        raise RuntimeError(
            f"command failed ({last_result.returncode}): {self._zfs_bin} promote {dataset}"
            f"\nstdout: {last_result.stdout.strip()}"
            f"\nstderr: {last_stderr}"
        )

    def _destroy_dataset_snapshot(
        self,
        dataset: str,
        snapshot_name: str,
        *,
        sandbox_id: SandboxId,
    ) -> None:
        snapshot = f"{dataset}@{snapshot_name}"
        result = self._run_command(
            [self._zfs_bin, "destroy", snapshot],
            operation="sandbox.zfs_destroy_snapshot",
            sandbox_id=sandbox_id,
            check=False,
            metadata={"dataset": dataset, "snapshot": snapshot},
        )
        stderr = result.stderr.strip()
        if result.returncode == 0 or "does not exist" in stderr or "snapshot does not exist" in stderr:
            return
        raise RuntimeError(
            f"command failed ({result.returncode}): {self._zfs_bin} destroy {snapshot}"
            f"\nstdout: {result.stdout.strip()}"
            f"\nstderr: {stderr}"
        )

    def clone_filesystem_snapshot(
        self,
        source_sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
        target_sandbox_id: SandboxId,
        *,
        target_rootfs_path: Path,
    ) -> str:
        source_dataset = self._dataset_resolver(source_sandbox_id)
        target_dataset = f"{self._dataset_prefix}/{target_sandbox_id}"
        self._run_command(
            [self._zfs_bin, "destroy", "-r", target_dataset],
            operation="sandbox.zfs_destroy_clone_target",
            sandbox_id=target_sandbox_id,
            check=False,
            metadata={"dataset": target_dataset},
        )
        snapshot = f"{source_dataset}@{checkpoint_id}"
        self._run_command(
            [self._zfs_bin, "clone", "-o", f"mountpoint={target_rootfs_path}", snapshot, target_dataset],
            operation="sandbox.zfs_clone",
            sandbox_id=target_sandbox_id,
            checkpoint_id=checkpoint_id,
            metadata={"source_dataset": source_dataset, "target_dataset": target_dataset, "mountpoint": str(target_rootfs_path)},
        )
        # Materialize the fork-point snapshot on the fork's own dataset
        # (C1): gives changeset_since a local diff base, makes the fork
        # manifests' stamped fs_ref real so retention can reclaim it, and
        # decouples fork-point restores from the source's lineage.
        self._run_command(
            [self._zfs_bin, "snapshot", f"{target_dataset}@{checkpoint_id}"],
            operation="sandbox.zfs_fork_point_snapshot",
            sandbox_id=target_sandbox_id,
            checkpoint_id=checkpoint_id,
            metadata={"dataset": target_dataset},
        )
        return target_dataset

    def changeset_since(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> list[ChangesetEntry]:
        dataset = self._dataset_resolver(sandbox_id)
        snapshot = f"{dataset}@{checkpoint_id}"
        if not self.snapshot_exists(snapshot):
            raise FileNotFoundError(f"changeset base snapshot missing: {snapshot}")
        mountpoint = str(self._rootfs_resolver(sandbox_id))
        result = self._run_command(
            [self._zfs_bin, "diff", "-FH", snapshot, dataset],
            operation="sandbox.changeset_since",
            sandbox_id=sandbox_id,
            checkpoint_id=checkpoint_id,
            metadata={"dataset": dataset, "snapshot": snapshot, "mountpoint": mountpoint},
        )
        return parse_zfs_diff(result.stdout, mountpoint=mountpoint)

    def snapshot_content_root(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> Path:
        dataset = self._dataset_resolver(sandbox_id)
        snapshot = f"{dataset}@{checkpoint_id}"
        if not self.snapshot_exists(snapshot):
            raise FileNotFoundError(f"snapshot missing: {snapshot}")
        # The .zfs control directory stays path-accessible under the
        # default snapdir=hidden; the snapshot automounts on access.
        return Path(str(self._rootfs_resolver(sandbox_id))) / ".zfs" / "snapshot" / str(checkpoint_id)

    def discard_partial_checkpoint(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> None:
        dataset = self._dataset_resolver(sandbox_id)
        snapshot = f"{dataset}@{checkpoint_id}"
        if self.snapshot_exists(snapshot):
            try:
                self._run_command(
                    [self._zfs_bin, "destroy", snapshot],
                    operation="sandbox.discard_partial_checkpoint",
                    check=False,
                    metadata={"snapshot": snapshot},
                )
            except Exception:
                logger.exception(
                    "Failed to destroy partial zfs snapshot sandbox=%s checkpoint=%s snapshot=%s",
                    sandbox_id,
                    checkpoint_id,
                    snapshot,
                )

    # ------------------------------------------------------------------
    # Host preparation (used by the engine before any runtime exists)
    # ------------------------------------------------------------------

    @staticmethod
    def resolve_dataset_prefix(configured: str | None) -> str:
        """Resolve the dataset prefix for the SDK engine. Precedence:
        explicit config > $CRAB_ZFS_DATASET_PREFIX > $CRAB_ZPOOL_NAME >
        an installed zpool whose name starts with `crab`. Moved verbatim
        from Engine._resolve_zfs_dataset_prefix so pool discovery lives
        with the backend that understands it."""
        if configured:
            return configured.rstrip("/")
        env_prefix = os.environ.get("CRAB_ZFS_DATASET_PREFIX", "").strip()
        if env_prefix:
            return env_prefix.rstrip("/")
        env_pool = os.environ.get("CRAB_ZPOOL_NAME", "").strip()
        if env_pool:
            return f"{env_pool}/crab-sdk"
        try:
            result = subprocess.run(
                ["zpool", "list", "-H", "-o", "name"],
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError:
            return "crab/crab-sdk"
        pools = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        for pool in pools:
            if pool.startswith("crab"):
                return f"{pool}/crab-sdk"
        return f"{pools[0]}/crab-sdk" if pools else "crab/crab-sdk"

    @staticmethod
    def ensure_parent_dataset(dataset_prefix: str) -> None:
        """Create the prefix's parent datasets if missing (e.g.
        `crab/crab-sdk` under pool `crab`). Moved verbatim from
        Engine._ensure_zfs_parent_dataset."""
        parts = [part for part in dataset_prefix.strip("/").split("/") if part]
        if len(parts) < 2:
            return
        current = parts[0]
        for part in parts[1:]:
            current = f"{current}/{part}"
            exists = subprocess.run(
                ["zfs", "list", "-H", "-o", "name", current],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode == 0
            if exists:
                continue
            subprocess.run(["zfs", "create", current], check=True)
