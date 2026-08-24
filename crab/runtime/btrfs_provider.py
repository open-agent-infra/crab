from __future__ import annotations

import logging
import shlex
import shutil
import subprocess
import uuid
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from ..ids import CheckpointId, SandboxId
from ..models import RuntimeOperationStatus
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

_TRASH_INFIX = ".trash-"
_CHANGESET_INFIX = "@changeset-"
_BTRFS_DUMP_CREATE_COMMANDS = {"mkfile", "mkdir", "mknod", "mkfifo", "mksock", "symlink"}
_BTRFS_DUMP_MODIFY_COMMANDS = {
    "write",
    "update_extent",
    "clone",
    "truncate",
    "fallocate",
    "fileattr",
    "chmod",
    "chown",
    "set_xattr",
    "remove_xattr",
}
# `utimes` is handled separately (_utimes_is_read_noise): it also fires for
# inodes whose only change is an atime bump from being read or executed.


def _tokenize_dump_line(line: str) -> list[str]:
    r"""Split a ``btrfs receive --dump`` line on unescaped spaces,
    unescaping ``\ `` / ``\\`` sequences inside tokens as it goes."""
    tokens: list[str] = []
    current: list[str] = []
    escaped = False
    for char in line:
        if escaped:
            current.append(char)
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == " ":
            if current:
                tokens.append("".join(current))
                current = []
        else:
            current.append(char)
    if current:
        tokens.append("".join(current))
    return tokens


def _dump_container_path(raw: str, snapshot_name: str) -> str | None:
    """Strip the dump's ``./<snapshot name>`` prefix, yielding a
    ``/``-rooted container path; None for paths outside the snapshot."""
    prefix = f"./{snapshot_name}"
    if not raw.startswith(prefix):
        return None
    remainder = raw[len(prefix):]
    if remainder in ("", "/"):
        return "/"
    if not remainder.startswith("/"):
        return None
    trimmed = remainder.rstrip("/")
    return trimmed if trimmed else "/"


def _parse_dump_time(raw: str | None) -> datetime | None:
    """Parse a `btrfs receive --dump` timestamp (``2026-08-20T10:41:33+0000``).
    Returns None for anything unparseable, so callers stay conservative."""
    if not raw:
        return None
    try:
        return datetime.strptime(raw, "%Y-%m-%dT%H:%M:%S%z")
    except ValueError:
        return None


def _utimes_is_read_noise(args: dict[str, str]) -> bool:
    r"""True when the newest event on this inode was a *read*.

    ``btrfs send`` emits ``utimes`` for every inode it touches, including
    ones whose only change since the base is an atime bump from being read
    or executed. Left unfiltered that floods the changeset with image
    binaries the sandbox merely ran (``/usr/bin/mv``, ``libacl.so`` …),
    which then look like fork-side modifications to the C2 merge and
    collide with the same paths on the source side.

    Any real change pushes ctime to at least atime — a write or chmod sets
    ctime to now while atime stays older, and ``touch`` sets all three to
    now. Only a read leaves ``atime > ctime``. Verified against real dump
    output: read → ``atime=33 mtime=32 ctime=32``; touch → ``33/33/33``;
    write → ``32/33/33`` (plus its own ``update_extent`` line, so content
    changes are never dropped by this rule).

    Missing or malformed timestamps keep the entry.
    """
    atime = _parse_dump_time(args.get("atime"))
    ctime = _parse_dump_time(args.get("ctime"))
    if atime is None or ctime is None:
        return False
    return atime > ctime


def parse_btrfs_receive_dump(stdout: str, *, snapshot_name: str) -> list[ChangesetEntry]:
    r"""Parse ``btrfs receive --dump`` output of a ``send --no-data``
    stream between the base checkpoint snapshot and a transient live
    snapshot. Semantics pinned from real VM output:

    - new files/dirs arrive as ``o<inode>-<gen>-0`` placeholders
      (``mkfile``/``mkdir``) that a later ``rename`` materializes onto
      the real path;
    - a move of a pre-existing file is encoded as ``link <new>
      dest=<old>`` (dest is subvolume-root-relative) plus ``unlink
      <old>`` — the ``rename`` command is not used for it;
    - spaces in paths are escaped as ``\ ``; every positional path is
      prefixed with ``./<snapshot name>/``; the snapshot root's own
      ``utimes`` churn is dropped as noise (matches the zfs parser).
    """
    created: set[str] = set()
    modified: set[str] = set()
    removed: set[str] = set()
    renamed: dict[str, str] = {}
    pending_links: dict[str, str] = {}

    for line in stdout.splitlines():
        tokens = _tokenize_dump_line(line)
        if len(tokens) < 2:
            continue
        command = tokens[0]
        path = _dump_container_path(tokens[1], snapshot_name)
        if path is None or path == "/":
            continue
        args: dict[str, str] = {}
        for token in tokens[2:]:
            if "=" in token:
                key, value = token.split("=", 1)
                args[key] = value
        if command in _BTRFS_DUMP_CREATE_COMMANDS:
            created.add(path)
        elif command == "rename":
            dest_raw = args.get("dest")
            dest = _dump_container_path(dest_raw, snapshot_name) if dest_raw is not None else None
            if dest is None or dest == "/":
                continue
            if path in created:
                # Placeholder (or in-window path) materialization.
                created.discard(path)
                if dest in removed:
                    # Replaced a pre-existing path: net effect is a modify.
                    removed.discard(dest)
                    modified.add(dest)
                else:
                    created.add(dest)
            else:
                renamed[dest] = renamed.pop(path, path)
                modified.discard(path)
        elif command == "link":
            target = args.get("dest")
            if target is not None:
                pending_links[path] = "/" + target
        elif command == "unlink":
            new_path = next((new for new, old in pending_links.items() if old == path), None)
            if new_path is not None:
                # link+unlink pair = move of a pre-existing file.
                del pending_links[new_path]
                if path in created:
                    created.discard(path)
                    created.add(new_path)
                else:
                    renamed[new_path] = renamed.pop(path, path)
                    modified.discard(path)
            elif path in created:
                created.discard(path)  # transient: born and gone in-window
            else:
                removed.add(path)
        elif command == "rmdir":
            if path in created:
                created.discard(path)
            else:
                removed.add(path)
        elif command == "utimes":
            # A bare atime bump is not a modification (see
            # _utimes_is_read_noise); a genuine `touch` still is.
            if _utimes_is_read_noise(args):
                continue
            if path not in created and path not in renamed:
                modified.add(path)
        elif command in _BTRFS_DUMP_MODIFY_COMMANDS:
            if path not in created and path not in renamed:
                modified.add(path)

    # Unconsumed links are genuinely new hard-link names.
    created.update(pending_links)

    entries: dict[str, ChangesetEntry] = {}

    def _fold(path: str, change: str, renamed_from: str | None = None) -> None:
        existing = entries.get(path)
        if existing is None or CHANGE_PRECEDENCE[change] >= CHANGE_PRECEDENCE[existing.change]:
            entries[path] = ChangesetEntry(path=path, change=change, renamed_from=renamed_from)

    for path in modified:
        _fold(path, "modified")
    for path in created:
        _fold(path, "added")
    for path, old_path in renamed.items():
        _fold(path, "renamed", old_path)
    for path in removed:
        _fold(path, "removed")
    return [entries[key] for key in sorted(entries)]


class BtrfsProvider(FilesystemProvider):
    """Btrfs-backed filesystem provider.

    Semantic mapping from the ZFS reference implementation:

    - A sandbox "dataset" is a btrfs subvolume living at its dataset path
      under ``btrfs_root`` (``<root>/sandboxes/<sandbox_id>``). Btrfs has
      no ``mountpoint=`` property, so the rootfs the runtime sees is a
      bind mount of the subvolume — uniform for every sandbox, mirroring
      ZFS mountpoint semantics 1:1.
    - Checkpoints are read-only subvolume snapshots stored flat next to
      the live subvolume as ``<dataset>@<checkpoint_id>`` (same naming
      shape as ZFS snapshots).
    - ``restore_filesystem`` is emulated (btrfs has no in-place
      rollback): unmount the rootfs bind, rename the live subvolume to a
      trash name, take a writable snapshot of the checkpoint back onto
      the live path, re-bind, and best-effort delete the trash. Restore
      only runs while the container is stopped, so the swap is safe.
      Unlike ``zfs rollback -r``, restoring does NOT destroy snapshots
      taken after the restore point: btrfs snapshots are independent
      subvolumes, so later checkpoints stay restorable and retention
      keeps governing their lifetime (accepted design decision; the
      divergence is disk-usage-only because restore selection is
      manifest-driven).
    - ``promote`` is a no-op: btrfs snapshots share extents without a
      clone-origin dependency, so destroying a fork's source never
      invalidates the fork.
    - Per-snapshot byte stats need qgroups, which carry real overhead;
      they are off by default and stats degrade to unknown (telemetry
      tolerates missing byte counts).
    """

    # Opaque fs_ref prefix stamped into checkpoint metadata and stripped
    # by destroy_snapshot_ref. Class attribute so subclasses that reuse
    # the btrfs mechanics under another backend name (OverlayProvider)
    # mint and consume their own refs.
    _fs_ref_prefix = "btrfs:"

    def __init__(
        self,
        *,
        btrfs_root: Path,
        runtime_name: str,
        run_command: FsCommandExecutor,
        run_status: FsStatusExecutor,
        dataset_resolver: DatasetResolver,
        rootfs_resolver: RootfsResolver,
        btrfs_bin: str = "btrfs",
        qgroups_enabled: bool = False,
    ) -> None:
        self._root = Path(btrfs_root)
        self._runtime_name = runtime_name
        self._run_command = run_command
        self._run_status = run_status
        self._dataset_resolver = dataset_resolver
        self._rootfs_resolver = rootfs_resolver
        self._btrfs_bin = btrfs_bin
        self._qgroups_enabled = qgroups_enabled

    @property
    def name(self) -> str:
        return "btrfs"

    # ------------------------------------------------------------------
    # Host preparation (used by the engine before any runtime exists)
    # ------------------------------------------------------------------

    @staticmethod
    def ensure_root(btrfs_root: Path) -> None:
        """Verify ``btrfs_root`` sits on a btrfs filesystem. Unlike the
        ZFS pool flow the engine never creates the filesystem itself
        (mkfs needs a device decision an SDK must not make); the
        installer or operator prepares the mount."""
        probe = subprocess.run(
            ["stat", "-f", "-c", "%T", str(btrfs_root)],
            check=False,
            capture_output=True,
            text=True,
        )
        fs_type = probe.stdout.strip()
        if probe.returncode != 0 or fs_type != "btrfs":
            raise RuntimeError(
                f"filesystem_backend=btrfs requires {btrfs_root} to be on a btrfs "
                f"filesystem (found {fs_type or 'nothing'}); prepare it with "
                "scripts/install-ubuntu.sh --fs-backend btrfs or mount one manually"
            )

    # ------------------------------------------------------------------
    # Naming / layout
    # ------------------------------------------------------------------

    def default_dataset_name(self, sandbox_id: SandboxId) -> str:
        return str(self._root / "sandboxes" / str(sandbox_id))

    def _safe_prefix(self) -> str:
        return "".join(
            char if char.isalnum() or char in {"-", "_", "."} else "_"
            for char in str(self._root)
        )

    def shared_rootfs_details(self, key: str, *, persist_across_runs: bool) -> tuple[str, Path]:
        scope = "persistent" if persist_across_runs else "run"
        dataset = str(self._root / "shared" / scope / key)
        # The subvolume path doubles as the materialization mountpoint:
        # unlike ZFS there is no separate mount to arrange for the cache.
        return dataset, Path(dataset)

    def shared_rootfs_lock_path(self, key: str, *, persist_across_runs: bool) -> Path:
        scope = "persistent" if persist_across_runs else "run"
        return Path("/tmp/crab-rootfs-cache-locks") / self._safe_prefix() / scope / f"{key}.lock"

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def _subvolume_exists(self, path: str, *, operation: str) -> bool:
        result = self._run_command(
            [self._btrfs_bin, "subvolume", "show", path],
            operation=operation,
            check=False,
            metadata={"subvolume": path},
        )
        return result.returncode == 0

    def object_exists(self, name: str) -> bool:
        return self._subvolume_exists(name, operation="sandbox.btrfs_show")

    def dataset_exists(self, dataset: str) -> bool:
        return self._subvolume_exists(dataset, operation="sandbox.btrfs_exists")

    def snapshot_exists(self, snapshot: str) -> bool:
        return self._subvolume_exists(snapshot, operation="sandbox.btrfs_snapshot_exists")

    def _query_snapshot_sizes(
        self,
        snapshot: str,
        *,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> tuple[int | None, int | None]:
        if not self._qgroups_enabled:
            return None, None
        result = self._run_command(
            [self._btrfs_bin, "qgroup", "show", "--raw", "-f", snapshot],
            operation="sandbox.btrfs_snapshot_stats",
            sandbox_id=sandbox_id,
            checkpoint_id=checkpoint_id,
            check=False,
            metadata={"snapshot": snapshot},
        )
        if result.returncode != 0:
            return None, None
        # Last data row: `<qgroupid> <referenced> <exclusive> [path]`.
        # `exclusive` approximates ZFS `written`, `referenced` maps to `used`.
        for line in reversed(result.stdout.splitlines()):
            parts = line.split()
            if len(parts) >= 3 and "/" in parts[0]:
                try:
                    referenced = int(parts[1])
                    exclusive = int(parts[2])
                except ValueError:
                    continue
                return exclusive, referenced
        return None, None

    # ------------------------------------------------------------------
    # Bind-mount helpers
    # ------------------------------------------------------------------

    def _bind_rootfs(self, dataset: str, mountpoint: Path, *, sandbox_id: SandboxId | None) -> None:
        if str(mountpoint) == dataset:
            return
        mountpoint.mkdir(parents=True, exist_ok=True)
        self._run_command(
            ["mount", "--bind", dataset, str(mountpoint)],
            operation="sandbox.btrfs_bind_rootfs",
            sandbox_id=sandbox_id,
            metadata={"subvolume": dataset, "mountpoint": str(mountpoint)},
        )

    def _unbind_rootfs(self, mountpoint: Path, *, sandbox_id: SandboxId | None) -> None:
        # Best-effort: not-mounted is the common case for fresh launches
        # and for datasets that were never bind-mounted.
        self._run_command(
            ["umount", str(mountpoint)],
            operation="sandbox.btrfs_unbind_rootfs",
            sandbox_id=sandbox_id,
            check=False,
            metadata={"mountpoint": str(mountpoint)},
        )

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
        dataset_path = Path(dataset)
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        if dataset_path.exists() and not self._subvolume_exists(dataset, operation=operation):
            # Self-heal: a plain directory at the subvolume path is
            # wreckage from an interrupted materialization (the path
            # doubles as the mountpoint) and blocks `subvolume create`.
            shutil.rmtree(dataset_path)
        self._run_command(
            [self._btrfs_bin, "subvolume", "create", dataset],
            operation=operation,
            sandbox_id=sandbox_id,
            metadata={"subvolume": dataset, "mountpoint": str(mountpoint)},
            timeout_seconds=timeout_seconds,
        )
        self._bind_rootfs(dataset, mountpoint, sandbox_id=sandbox_id)

    def create_snapshot(self, dataset: str, snapshot: str, *, operation: str) -> None:
        self._run_command(
            [self._btrfs_bin, "subvolume", "snapshot", "-r", dataset, snapshot],
            operation=operation,
            metadata={"subvolume": dataset, "snapshot": snapshot},
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
        Path(dataset).parent.mkdir(parents=True, exist_ok=True)
        self._run_command(
            [self._btrfs_bin, "subvolume", "snapshot", shared_snapshot, dataset],
            operation="sandbox.btrfs_clone_launch_rootfs",
            sandbox_id=sandbox_id,
            metadata={
                "source_subvolume": shared_dataset,
                "target_subvolume": dataset,
                "snapshot": shared_snapshot,
                "mountpoint": str(rootfs_path),
            },
        )
        self._bind_rootfs(dataset, rootfs_path, sandbox_id=sandbox_id)

    def destroy_dataset(
        self,
        dataset: str,
        *,
        operation: str,
        sandbox_id: SandboxId | None = None,
    ) -> None:
        # ZFS `destroy -r` also removes the dataset's snapshots; mirror
        # that by deleting the flat `<dataset>@*` siblings first.
        for snapshot in self._sibling_snapshots(dataset):
            self._run_command(
                [self._btrfs_bin, "subvolume", "delete", snapshot],
                operation=operation,
                sandbox_id=sandbox_id,
                check=False,
                metadata={"subvolume": dataset, "snapshot": snapshot},
            )
        self._run_command(
            [self._btrfs_bin, "subvolume", "delete", dataset],
            operation=operation,
            sandbox_id=sandbox_id,
            check=False,
            metadata={"subvolume": dataset},
        )

    def _sibling_snapshots(self, dataset: str) -> list[str]:
        base = Path(dataset)
        try:
            return sorted(str(path) for path in base.parent.glob(f"{base.name}@*"))
        except OSError:
            return []

    def _trash_paths(self, dataset: str) -> list[str]:
        base = Path(dataset)
        try:
            return sorted(str(path) for path in base.parent.glob(f"{base.name}{_TRASH_INFIX}*"))
        except OSError:
            return []

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
            [self._btrfs_bin, "subvolume", "snapshot", "-r", dataset, snapshot],
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
        mountpoint = self._rootfs_resolver(sandbox_id)
        metadata: dict[str, object] = {
            "phase": "filesystem_restore",
            "runtime": self._runtime_name,
            "dataset": dataset,
            "snapshot": snapshot,
            "mountpoint": str(mountpoint),
        }
        # Reclaim trash left by an earlier crashed restore before making
        # more; best-effort, never blocks the current restore.
        for stale in self._trash_paths(dataset):
            self._run_command(
                [self._btrfs_bin, "subvolume", "delete", stale],
                operation="sandbox.btrfs_reclaim_restore_trash",
                sandbox_id=sandbox_id,
                check=False,
                metadata={"subvolume": dataset, "trash": stale},
            )
        self._unbind_rootfs(mountpoint, sandbox_id=sandbox_id)
        # Crash recovery: if a previous restore died between the rename
        # and the snapshot-back, the live path is already vacant — skip
        # straight to rematerializing from the checkpoint snapshot.
        if self.dataset_exists(dataset):
            trash = f"{dataset}{_TRASH_INFIX}{uuid.uuid4().hex}"
            self._run_command(
                ["mv", dataset, trash],
                operation="sandbox.btrfs_rollback_swap_out",
                sandbox_id=sandbox_id,
                checkpoint_id=checkpoint_id,
                metadata={**metadata, "trash": trash},
            )
        status = self._run_status(
            [self._btrfs_bin, "subvolume", "snapshot", snapshot, dataset],
            operation="sandbox.restore_filesystem",
            sandbox_id=sandbox_id,
            checkpoint_id=checkpoint_id,
            metadata=metadata,
        )
        self._bind_rootfs(dataset, mountpoint, sandbox_id=sandbox_id)
        for trash_path in self._trash_paths(dataset):
            self._run_command(
                [self._btrfs_bin, "subvolume", "delete", trash_path],
                operation="sandbox.btrfs_rollback_delete_trash",
                sandbox_id=sandbox_id,
                check=False,
                metadata={"subvolume": dataset, "trash": trash_path},
            )
        return status

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
            "fs_ref": f"{self._fs_ref_prefix}{snapshot}",
        }

    def destroy_filesystem_dataset(self, sandbox_id: SandboxId, dataset: str) -> None:
        self._unbind_rootfs(self._rootfs_resolver(sandbox_id), sandbox_id=sandbox_id)
        self.destroy_dataset(dataset, operation="sandbox.btrfs_destroy", sandbox_id=sandbox_id)

    def promote_filesystem_dataset(self, sandbox_id: SandboxId) -> None:
        # Btrfs snapshots share extents without a clone-origin dependency;
        # a fork never needs promotion to outlive its source.
        logger.debug(
            "Btrfs promote is a no-op (no clone-origin dependency) sandbox=%s",
            sandbox_id,
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
        target_dataset = str(self._root / "sandboxes" / str(target_sandbox_id))
        self.destroy_dataset(
            target_dataset,
            operation="sandbox.btrfs_destroy_clone_target",
            sandbox_id=target_sandbox_id,
        )
        snapshot = f"{source_dataset}@{checkpoint_id}"
        Path(target_dataset).parent.mkdir(parents=True, exist_ok=True)
        self._run_command(
            [self._btrfs_bin, "subvolume", "snapshot", snapshot, target_dataset],
            operation="sandbox.btrfs_clone",
            sandbox_id=target_sandbox_id,
            checkpoint_id=checkpoint_id,
            metadata={
                "source_subvolume": source_dataset,
                "target_subvolume": target_dataset,
                "mountpoint": str(target_rootfs_path),
            },
        )
        # Materialize the fork-point snapshot on the fork's own subvolume
        # (C1): gives changeset_since a local diff base, makes the fork
        # manifests' stamped fs_ref real so retention can reclaim it, and
        # keeps fork-point restores independent of the source's snapshots.
        self._run_command(
            [self._btrfs_bin, "subvolume", "snapshot", "-r", target_dataset, f"{target_dataset}@{checkpoint_id}"],
            operation="sandbox.btrfs_fork_point_snapshot",
            sandbox_id=target_sandbox_id,
            checkpoint_id=checkpoint_id,
            metadata={"subvolume": target_dataset, "snapshot": f"{target_dataset}@{checkpoint_id}"},
        )
        self._bind_rootfs(target_dataset, target_rootfs_path, sandbox_id=target_sandbox_id)
        return target_dataset

    def _changeset_snapshot_paths(self, dataset: str) -> list[str]:
        base = Path(dataset)
        try:
            return sorted(str(path) for path in base.parent.glob(f"{base.name}{_CHANGESET_INFIX}*"))
        except OSError:
            return []

    def changeset_since(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> list[ChangesetEntry]:
        dataset = self._dataset_resolver(sandbox_id)
        base_snapshot = f"{dataset}@{checkpoint_id}"
        if not self.snapshot_exists(base_snapshot):
            raise FileNotFoundError(f"changeset base snapshot missing: {base_snapshot}")
        # Reclaim transient snapshots leaked by earlier crashed runs
        # before making another; best-effort.
        for stale in self._changeset_snapshot_paths(dataset):
            self._run_command(
                [self._btrfs_bin, "subvolume", "delete", stale],
                operation="sandbox.btrfs_changeset_sweep",
                sandbox_id=sandbox_id,
                check=False,
                metadata={"subvolume": dataset, "snapshot": stale},
            )
        tmp_snapshot = f"{dataset}{_CHANGESET_INFIX}{uuid.uuid4().hex[:8]}"
        self._run_command(
            [self._btrfs_bin, "subvolume", "snapshot", "-r", dataset, tmp_snapshot],
            operation="sandbox.btrfs_changeset_snapshot",
            sandbox_id=sandbox_id,
            checkpoint_id=checkpoint_id,
            metadata={"subvolume": dataset, "snapshot": tmp_snapshot},
        )
        try:
            # find-new cannot see deletions; diff via a metadata-only send
            # stream instead. A broken pipeline cannot silently yield an
            # empty changeset: receive errors on empty/truncated input.
            result = self._run_command(
                [
                    "sh",
                    "-c",
                    f"{shlex.quote(self._btrfs_bin)} send --no-data -p {shlex.quote(base_snapshot)} "
                    f"{shlex.quote(tmp_snapshot)} | {shlex.quote(self._btrfs_bin)} receive --dump",
                ],
                operation="sandbox.changeset_since",
                sandbox_id=sandbox_id,
                checkpoint_id=checkpoint_id,
                metadata={"dataset": dataset, "snapshot": base_snapshot, "tmp_snapshot": tmp_snapshot},
            )
            return parse_btrfs_receive_dump(result.stdout, snapshot_name=Path(tmp_snapshot).name)
        finally:
            self._run_command(
                [self._btrfs_bin, "subvolume", "delete", tmp_snapshot],
                operation="sandbox.btrfs_changeset_cleanup",
                sandbox_id=sandbox_id,
                check=False,
                metadata={"subvolume": dataset, "snapshot": tmp_snapshot},
            )

    def snapshot_content_root(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> Path:
        dataset = self._dataset_resolver(sandbox_id)
        snapshot = f"{dataset}@{checkpoint_id}"
        if not self.snapshot_exists(snapshot):
            raise FileNotFoundError(f"snapshot missing: {snapshot}")
        # Snapshots are read-only subvolumes stored flat next to the live
        # one; the path itself is the content root.
        return Path(snapshot)

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
                    [self._btrfs_bin, "subvolume", "delete", snapshot],
                    operation="sandbox.discard_partial_checkpoint",
                    check=False,
                    metadata={"snapshot": snapshot},
                )
            except Exception:
                logger.exception(
                    "Failed to destroy partial btrfs snapshot sandbox=%s checkpoint=%s snapshot=%s",
                    sandbox_id,
                    checkpoint_id,
                    snapshot,
                )

    def destroy_snapshot_ref(self, fs_ref: str) -> None:
        snapshot = fs_ref[len(self._fs_ref_prefix):] if fs_ref.startswith(self._fs_ref_prefix) else fs_ref
        if not snapshot:
            return
        result = self._run_command(
            [self._btrfs_bin, "subvolume", "delete", snapshot],
            operation="storage.btrfs_destroy_snapshot_ref",
            check=False,
            metadata={"snapshot": snapshot, "fs_ref": fs_ref},
        )
        if result.returncode != 0:
            stderr = result.stderr.strip()
            if "No such file or directory" in stderr or "not a subvolume" in stderr:
                return
            logger.warning(
                "Failed to destroy filesystem snapshot %s rc=%d stderr=%s",
                snapshot,
                result.returncode,
                stderr,
            )
