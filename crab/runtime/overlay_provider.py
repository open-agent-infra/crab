from __future__ import annotations

import json
import logging
import os
import shutil
import stat
import subprocess
from pathlib import Path
from typing import Callable

from ..ids import CheckpointId, SandboxId
from .btrfs_provider import BtrfsProvider
from .fs_provider import (
    CHANGE_PRECEDENCE,
    ChangesetEntry,
    DatasetResolver,
    FsCommandExecutor,
    FsStatusExecutor,
    RootfsResolver,
)

logger = logging.getLogger(__name__)

_MARKER_NAME = ".crab-overlay.json"
_EMPTY_LOWER_DIRNAME = "empty-lower"
_SNAPMOUNTS_DIRNAME = "snapmounts"
_UPPER_PREFIX = "/upper"
# Pinned on every rw mount: both features would replace physical
# copy-up/whiteout/rename with xattr redirection and break the changeset
# translation below, so changeset semantics must not depend on kernel
# build defaults (design D4).
_RW_MOUNT_OPTIONS = "redirect_dir=off,metacopy=off"


def _is_whiteout(path: Path) -> bool:
    """Overlay whiteout: a character device 0:0 in an upperdir."""
    try:
        st = os.lstat(path)
    except OSError:
        return False
    return stat.S_ISCHR(st.st_mode) and os.major(st.st_rdev) == 0 and os.minor(st.st_rdev) == 0


def _is_opaque_dir(path: Path) -> bool:
    """Opaque upperdir directory: masks the lower's content entirely.
    The kernel stamps ``trusted.overlay.opaque=y`` (root-only xattr
    namespace; callers of the real probe run as root)."""
    try:
        if not stat.S_ISDIR(os.lstat(path).st_mode):
            return False
        value = os.getxattr(path, "trusted.overlay.opaque", follow_symlinks=False)
    except OSError:
        return False
    return value == b"y"


def _container_path(vol_path: str | None) -> str | None:
    """Map a vol-relative path from the base send-diff (``/upper/...``)
    onto the container namespace; None for overlay-internal paths
    (``/work/**``, the marker, the upper root's own churn)."""
    if vol_path is None:
        return None
    if vol_path.startswith(_UPPER_PREFIX + "/"):
        return vol_path[len(_UPPER_PREFIX):]
    return None


def translate_overlay_changeset(
    entries: list[ChangesetEntry],
    *,
    live_whiteout: Callable[[str], bool],
    base_whiteout: Callable[[str], bool],
    base_exists: Callable[[str], bool],
    live_exists: Callable[[str], bool],
    newly_opaque: Callable[[str], bool],
    base_children: Callable[[str], list[str]],
    live_children: Callable[[str], list[str]],
) -> list[ChangesetEntry]:
    """Decode overlay artifacts out of a vol-level send-diff (design §5).

    ``entries`` is the parsed ``btrfs send --no-data`` diff of the
    sandbox vol between the base checkpoint snapshot and a transient
    live snapshot — i.e. raw *upperdir* deltas. The probes answer
    questions the diff cannot: whether an upper path is a whiteout
    (live/base), whether a path existed in the base *merged* view
    (added vs modified: a copy-up of a lower file arrives as ``added``
    in the upper but is ``modified`` in container semantics), whether a
    directory turned opaque since base, and directory listings of both
    merged views (opaque dirs mask lower children that the upper diff
    never enumerates).

    Documented divergence (design D6): a container-level rename of a
    not-yet-copied-up lower file has no physical rename to detect — it
    surfaces as added/modified(new) + removed(old). In-upper renames
    keep their ``renamed_from`` attribution.
    """
    folded: dict[str, ChangesetEntry] = {}

    def _fold(path: str, change: str, renamed_from: str | None = None) -> None:
        existing = folded.get(path)
        if existing is None or CHANGE_PRECEDENCE[change] >= CHANGE_PRECEDENCE[existing.change]:
            folded[path] = ChangesetEntry(path=path, change=change, renamed_from=renamed_from)

    def _remove_tree(cpath: str) -> None:
        # Match the zfs/btrfs shape for rm -rf: one entry per masked
        # path, descending through the base merged view.
        _fold(cpath, "removed")
        for name in base_children(cpath):
            _remove_tree(f"{cpath.rstrip('/')}/{name}")

    for entry in entries:
        cpath = _container_path(entry.path)
        if cpath is None:
            continue
        if entry.change == "removed":
            if base_whiteout(cpath):
                # The deletion marker vanished without an in-window
                # replacement: the lower content shows through again.
                if live_exists(cpath):
                    _fold(cpath, "added")
                continue
            _fold(cpath, "removed")
            continue
        # added / modified / renamed upper entries.
        if live_whiteout(cpath):
            # A whiteout materialized at this path: the container
            # deleted a lower-visible file.
            _fold(cpath, "removed")
            continue
        renamed_from = _container_path(entry.renamed_from) if entry.renamed_from else None
        if entry.change == "renamed" and renamed_from is not None:
            _fold(cpath, "renamed", renamed_from)
        elif base_exists(cpath):
            _fold(cpath, "modified")
        else:
            _fold(cpath, "added")
        if newly_opaque(cpath):
            live = set(live_children(cpath))
            for name in base_children(cpath):
                if name not in live:
                    _remove_tree(f"{cpath.rstrip('/')}/{name}")
    return [folded[key] for key in sorted(folded)]


class OverlayProvider(BtrfsProvider):
    """Overlayfs-on-btrfs filesystem provider (design doc
    ``.cache/tasks/overlay-provider.md``, A1 spike-validated).

    The sandbox rootfs is an overlay mount: ``lowerdir`` = shared
    read-only image content, ``upperdir``+``workdir`` = a per-sandbox
    btrfs subvolume (the provider "dataset", ``{vol}``). Both live in
    the *same* subvolume — splitting them across subvolume boundaries
    mounts fine but every whiteout/copy-up rename fails EXDEV at
    runtime (A1's one real trap).

    Semantics on top of the inherited btrfs mechanics:

    - ``{vol}/.crab-overlay.json`` records the lowerdir. It travels
      with every snapshot/clone, so restore/fork/promotion remounts are
      self-describing and a fork shares its source's lower (page-cache
      sharing across fork fleets). The lower chain never grows: the
      upper always sits exactly one level above the image snapshot.
    - Shared-cache launches use the cache's ``@base`` snapshot as
      lower; non-shared launches get the static ``empty-lower`` dir
      (degenerate but uniform — everything lands in the upper).
    - Snapshot content roots are lazy *read-only* overlay mounts of
      (snapshot upper : lower) under ``snapmounts/`` — the kernel, not
      Python, resolves whiteout/opaque semantics for merge base reads.
    - ``changeset_since`` = the inherited vol-level send-diff plus
      ``translate_overlay_changeset`` (see its docstring for the one
      documented divergence).
    - Unlike btrfs clones, a running overlay sandbox *actively
      references* its lower at runtime: deleting a shared cache while
      sandboxes run on it breaks them (documented constraint, design
      D11). Nested container engines (docker-in-sandbox) are not
      supported on overlay roots (kernel rejects overlay-upon-overlay
      uppers) — use the zfs/btrfs providers for those workloads.
    - qgroups byte stats account the upper subvolume only ("written
      since launch"), not the full rootfs as on btrfs.
    """

    _fs_ref_prefix = "overlay:"

    def __init__(
        self,
        *,
        overlay_root: Path,
        runtime_name: str,
        run_command: FsCommandExecutor,
        run_status: FsStatusExecutor,
        dataset_resolver: DatasetResolver,
        rootfs_resolver: RootfsResolver,
        btrfs_bin: str = "btrfs",
        qgroups_enabled: bool = False,
    ) -> None:
        super().__init__(
            btrfs_root=Path(overlay_root),
            runtime_name=runtime_name,
            run_command=run_command,
            run_status=run_status,
            dataset_resolver=dataset_resolver,
            rootfs_resolver=rootfs_resolver,
            btrfs_bin=btrfs_bin,
            qgroups_enabled=qgroups_enabled,
        )

    @property
    def name(self) -> str:
        return "overlay"

    # ------------------------------------------------------------------
    # Host preparation (used by the engine before any runtime exists)
    # ------------------------------------------------------------------

    @staticmethod
    def ensure_root(overlay_root: Path) -> None:
        """Verify ``overlay_root`` sits on btrfs and the kernel supports
        overlayfs; create the static empty lower. The root is typically
        a subdirectory of an existing btrfs mount (default:
        ``<btrfs_root>/overlay``), so it is created here rather than by
        the installer; on a misconfigured host the mkdir leaves only
        empty plain directories behind the raised error."""
        root = Path(overlay_root)
        root.mkdir(parents=True, exist_ok=True)
        probe = subprocess.run(
            ["stat", "-f", "-c", "%T", str(root)],
            check=False,
            capture_output=True,
            text=True,
        )
        fs_type = probe.stdout.strip()
        if probe.returncode != 0 or fs_type != "btrfs":
            raise RuntimeError(
                f"filesystem_backend=overlay requires {root} to be on a btrfs "
                f"filesystem (found {fs_type or 'nothing'}); prepare it with "
                "scripts/install-ubuntu.sh --fs-backend overlay or mount one manually"
            )
        try:
            filesystems = Path("/proc/filesystems").read_text(encoding="utf-8")
        except OSError:
            filesystems = ""
        supported = {line.split()[-1] for line in filesystems.splitlines() if line.split()}
        if "overlay" not in supported:
            raise RuntimeError(
                "filesystem_backend=overlay requires kernel overlayfs support "
                "(no 'overlay' in /proc/filesystems); modprobe overlay or use "
                "the zfs/btrfs backends"
            )
        (root / _EMPTY_LOWER_DIRNAME).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Vol scaffold + marker
    # ------------------------------------------------------------------

    def _empty_lower_path(self) -> Path:
        return self._root / _EMPTY_LOWER_DIRNAME

    @staticmethod
    def _marker_path(dataset: str | Path) -> Path:
        return Path(dataset) / _MARKER_NAME

    @staticmethod
    def _validate_lowerdir(lowerdir: str) -> None:
        # ':' and ',' are overlay mount-option metacharacters; fail at
        # launch instead of producing a silently misparsed mount. A
        # *missing* lower surfaces through the mount command itself.
        if ":" in lowerdir or "," in lowerdir:
            raise ValueError(
                f"overlay lowerdir contains mount-option metacharacters (':' or ','): {lowerdir}"
            )

    def _scaffold_overlay_vol(self, dataset: str, lowerdir: str) -> None:
        """Populate a freshly created vol: ``upper/`` + ``work/`` (same
        subvolume — the EXDEV convention) + the lowerdir marker. Plain
        Python fs ops: they must happen for real even under fake-runner
        tests, where the subvolume-create command is only recorded."""
        self._validate_lowerdir(lowerdir)
        vol = Path(dataset)
        (vol / "upper").mkdir(parents=True, exist_ok=True)
        (vol / "work").mkdir(parents=True, exist_ok=True)
        self._marker_path(dataset).write_text(
            json.dumps({"version": 1, "lowerdir": lowerdir}, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _read_lowerdir(self, dataset: str | Path) -> str:
        marker = self._marker_path(dataset)
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except OSError as exc:
            raise FileNotFoundError(f"overlay marker missing in {dataset}: {exc}") from exc
        except ValueError as exc:
            raise RuntimeError(f"overlay marker corrupt in {dataset}: {exc}") from exc
        lowerdir = str(payload.get("lowerdir") or "")
        if not lowerdir:
            raise RuntimeError(f"overlay marker in {dataset} lacks a lowerdir")
        return lowerdir

    # ------------------------------------------------------------------
    # Mount management (replaces the btrfs bind-mount)
    # ------------------------------------------------------------------

    def _bind_rootfs(self, dataset: str, mountpoint: Path, *, sandbox_id: SandboxId | None) -> None:
        if str(mountpoint) == dataset:
            # Shared image cache: a plain subvolume — it *is* the future
            # lowerdir content, never overlaid itself.
            return
        if os.path.ismount(mountpoint):
            # Remount for idempotence: restore/clone flows land here
            # after a vol swap; a stale overlay of the pre-swap vol must
            # not shadow the new one.
            self._unbind_rootfs(mountpoint, sandbox_id=sandbox_id)
        mountpoint.mkdir(parents=True, exist_ok=True)
        lowerdir = self._read_lowerdir(dataset)
        options = (
            f"lowerdir={lowerdir},upperdir={Path(dataset) / 'upper'},"
            f"workdir={Path(dataset) / 'work'},{_RW_MOUNT_OPTIONS}"
        )
        self._run_command(
            ["mount", "-t", "overlay", "overlay", "-o", options, str(mountpoint)],
            operation="sandbox.overlay_mount_rootfs",
            sandbox_id=sandbox_id,
            metadata={"subvolume": dataset, "mountpoint": str(mountpoint), "lowerdir": lowerdir},
        )

    def _snapmount_path(self, snapshot: str) -> Path:
        # `<vol>@<ckpt>` basenames are `<sandbox_id>@<ckpt>` — unique.
        return self._root / _SNAPMOUNTS_DIRNAME / Path(snapshot).name

    def _umount_snapmount(self, snapshot: str, *, sandbox_id: SandboxId | None = None) -> None:
        snapmount = self._snapmount_path(snapshot)
        if os.path.ismount(snapmount):
            self._run_command(
                ["umount", str(snapmount)],
                operation="sandbox.overlay_umount_snapshot",
                sandbox_id=sandbox_id,
                check=False,
                metadata={"snapshot": snapshot, "snapmount": str(snapmount)},
            )
        try:
            snapmount.rmdir()
        except OSError:
            pass

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
        if str(mountpoint) == dataset:
            # Shared image cache: plain subvolume, no scaffold, no mount.
            super().create_dataset(
                dataset,
                mountpoint,
                operation=operation,
                sandbox_id=sandbox_id,
                timeout_seconds=timeout_seconds,
            )
            return
        dataset_path = Path(dataset)
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        if dataset_path.exists() and not self._subvolume_exists(dataset, operation=operation):
            # Same self-heal as the btrfs base: plain-dir wreckage from
            # an interrupted scaffold blocks `subvolume create`.
            shutil.rmtree(dataset_path)
        self._run_command(
            [self._btrfs_bin, "subvolume", "create", dataset],
            operation=operation,
            sandbox_id=sandbox_id,
            metadata={"subvolume": dataset, "mountpoint": str(mountpoint)},
            timeout_seconds=timeout_seconds,
        )
        self._scaffold_overlay_vol(dataset, str(self._empty_lower_path()))
        self._bind_rootfs(dataset, mountpoint, sandbox_id=sandbox_id)

    def clone_shared_base(
        self,
        shared_dataset: str,
        shared_snapshot: str,
        dataset: str,
        rootfs_path: Path,
        *,
        sandbox_id: SandboxId,
    ) -> None:
        # Overlay inverts the btrfs semantics: the image is *referenced*
        # as the immutable lowerdir (the cache's @base snapshot), never
        # copied into the sandbox vol.
        dataset_path = Path(dataset)
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        if dataset_path.exists() and not self._subvolume_exists(
            dataset, operation="sandbox.overlay_create_launch_vol"
        ):
            shutil.rmtree(dataset_path)
        self._run_command(
            [self._btrfs_bin, "subvolume", "create", dataset],
            operation="sandbox.overlay_create_launch_vol",
            sandbox_id=sandbox_id,
            metadata={
                "source_subvolume": shared_dataset,
                "target_subvolume": dataset,
                "lowerdir": shared_snapshot,
                "mountpoint": str(rootfs_path),
            },
        )
        self._scaffold_overlay_vol(dataset, shared_snapshot)
        self._bind_rootfs(dataset, rootfs_path, sandbox_id=sandbox_id)

    def destroy_dataset(
        self,
        dataset: str,
        *,
        operation: str,
        sandbox_id: SandboxId | None = None,
    ) -> None:
        # Snapshot content mounts pin their snapshots; drop them before
        # the inherited snapshot+vol deletion loop.
        for snapshot in self._sibling_snapshots(dataset):
            self._umount_snapmount(snapshot, sandbox_id=sandbox_id)
        super().destroy_dataset(dataset, operation=operation, sandbox_id=sandbox_id)

    # ------------------------------------------------------------------
    # Checkpoint / restore / fork flows
    # ------------------------------------------------------------------

    def filesystem_checkpoint_metadata(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> dict[str, object]:
        metadata = super().filesystem_checkpoint_metadata(sandbox_id, checkpoint_id)
        dataset = self._dataset_resolver(sandbox_id)
        try:
            metadata["lowerdir"] = self._read_lowerdir(dataset)
        except (FileNotFoundError, RuntimeError):
            logger.debug(
                "overlay marker unreadable for checkpoint metadata sandbox=%s dataset=%s",
                sandbox_id,
                dataset,
            )
        return metadata

    def discard_partial_checkpoint(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> None:
        dataset = self._dataset_resolver(sandbox_id)
        self._umount_snapmount(f"{dataset}@{checkpoint_id}", sandbox_id=sandbox_id)
        super().discard_partial_checkpoint(sandbox_id, checkpoint_id)

    def destroy_snapshot_ref(self, fs_ref: str) -> None:
        snapshot = fs_ref[len(self._fs_ref_prefix):] if fs_ref.startswith(self._fs_ref_prefix) else fs_ref
        if snapshot:
            self._umount_snapmount(snapshot)
        super().destroy_snapshot_ref(fs_ref)

    def snapshot_content_root(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> Path:
        dataset = self._dataset_resolver(sandbox_id)
        snapshot = f"{dataset}@{checkpoint_id}"
        if not self.snapshot_exists(snapshot):
            raise FileNotFoundError(f"snapshot missing: {snapshot}")
        snapmount = self._snapmount_path(snapshot)
        if os.path.ismount(snapmount):
            return snapmount
        # Lowerdir-only mount (snapshot upper stacked above the image
        # lower, read from the marker that traveled with the snapshot):
        # kernel-enforced read-only merged view; the rw pin options are
        # upper-specific and stay off this mount (design D5).
        lowerdir = self._read_lowerdir(snapshot)
        snapmount.mkdir(parents=True, exist_ok=True)
        self._run_command(
            [
                "mount",
                "-t",
                "overlay",
                "overlay",
                "-o",
                f"lowerdir={Path(snapshot) / 'upper'}:{lowerdir}",
                str(snapmount),
            ],
            operation="sandbox.overlay_mount_snapshot",
            sandbox_id=sandbox_id,
            checkpoint_id=checkpoint_id,
            metadata={"snapshot": snapshot, "snapmount": str(snapmount), "lowerdir": lowerdir},
        )
        return snapmount

    def changeset_since(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> list[ChangesetEntry]:
        raw = super().changeset_since(sandbox_id, checkpoint_id)
        dataset = self._dataset_resolver(sandbox_id)
        base_snapshot = f"{dataset}@{checkpoint_id}"
        live_upper = Path(dataset) / "upper"
        base_upper = Path(base_snapshot) / "upper"
        rootfs = Path(self._rootfs_resolver(sandbox_id))
        base_root_cache: list[Path | None] = [None]

        def _base_root() -> Path:
            # Lazy: the base merged view needs a snapmount; only pay for
            # it when an entry actually consults the base.
            if base_root_cache[0] is None:
                base_root_cache[0] = self.snapshot_content_root(sandbox_id, checkpoint_id)
            return base_root_cache[0]

        def _rel(root: Path, cpath: str) -> Path:
            return root / cpath.lstrip("/")

        def _children(root: Path, cpath: str) -> list[str]:
            target = _rel(root, cpath)
            try:
                if not stat.S_ISDIR(os.lstat(target).st_mode):
                    return []
                return sorted(os.listdir(target))
            except OSError:
                return []

        return translate_overlay_changeset(
            raw,
            live_whiteout=lambda cpath: _is_whiteout(_rel(live_upper, cpath)),
            base_whiteout=lambda cpath: _is_whiteout(_rel(base_upper, cpath)),
            base_exists=lambda cpath: os.path.lexists(_rel(_base_root(), cpath)),
            live_exists=lambda cpath: os.path.lexists(_rel(rootfs, cpath)),
            newly_opaque=lambda cpath: (
                _is_opaque_dir(_rel(live_upper, cpath))
                and not _is_opaque_dir(_rel(base_upper, cpath))
            ),
            base_children=lambda cpath: _children(_base_root(), cpath),
            live_children=lambda cpath: _children(rootfs, cpath),
        )
