"""Filesystem three-way merge engine (roadmap C2).

Pure planning/application helpers driven by ``CrabSystem.merge_from_fork``:
the system owns orchestration (guards, quiesce, transient snapshot,
journal, telemetry); this module owns classification, policy resolution,
host-side application onto the source rootfs, and path-level rollback
from the pre-merge snapshot. Everything operates on plain directories so
unit tests run without any CoW backend.

Plan-then-apply: every policy decision is taken before the first write.
``fail_fast`` (and ``text_merge`` with unresolved conflicts) abort with
zero side effects. The apply phase only executes a fully resolved plan;
any error there triggers a path-level undo of the touched paths from the
transient ``@merge-*`` snapshot content.

Fidelity notes (documented v1 limits): mode/uid/gid/mtime are preserved;
xattrs/ACLs/hardlink identity are not. Directory-``modified`` entries are
mtime churn and are dropped from both sides (a directory *permission*
change therefore does not merge). Sockets and device nodes are logged
and skipped during application.
"""

from __future__ import annotations

import logging
import os
import shutil
import stat as stat_module
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Sequence

from .models import ChangesetEntry, MergeEntry, MergeReport

logger = logging.getLogger(__name__)

# Default merge-layer noise filter. C1 providers report raw truth; the
# CRIU leave-running dump scratches container /tmp after the filesystem
# snapshot, so every dump-window changeset carries a modified /tmp.
DEFAULT_MERGE_IGNORE_PREFIXES: tuple[str, ...] = ("/tmp", "/var/tmp", "/run")

MERGE_POLICIES: tuple[str, ...] = ("fail_fast", "prefer_fork", "prefer_source", "text_merge")

# (path, base, source, fork) -> merged content, or None when the hook
# cannot resolve the conflict. Only consulted where every present side
# is a regular file. Local engine only — never crosses the daemon RPC.
MergerHook = Callable[[str, "bytes | None", "bytes | None", "bytes | None"], "bytes | None"]


class MergeError(RuntimeError):
    """Merge refused or failed. ``report`` carries the classification
    computed so far (``rolled_back=True`` when the apply phase was
    undone from the pre-merge snapshot)."""

    def __init__(self, message: str, *, report: MergeReport | None = None) -> None:
        super().__init__(message)
        self.report = report


class MergeApplyError(RuntimeError):
    """Internal: the apply phase failed on ``path``; ``rolled_back``
    reports whether the path-level undo completed cleanly."""

    def __init__(self, message: str, *, path: str, rolled_back: bool) -> None:
        super().__init__(message)
        self.path = path
        self.rolled_back = rolled_back


@dataclass(frozen=True)
class _PlannedOp:
    kind: str                     # "remove" | "copy" | "copy_tree" | "write"
    path: str                     # container path mutated on the source
    entry: MergeEntry             # report row this op realizes
    content: bytes | None = None  # kind == "write" only


@dataclass(frozen=True)
class MergePlan:
    """Resolved merge plan. ``aborted`` means conflicts force a no-op
    merge (fail_fast semantics); ``ops`` is empty in that case."""

    ops: tuple[_PlannedOp, ...]
    entries_to_apply: tuple[MergeEntry, ...]
    conflicted: tuple[MergeEntry, ...]
    skipped: tuple[MergeEntry, ...]
    aborted: bool


def build_report(
    *,
    source_sandbox_id,
    fork_sandbox_id,
    base_checkpoint_id,
    policy: str,
    plan: MergePlan,
    applied: bool,
    rolled_back: bool = False,
) -> MergeReport:
    """Single place that turns a plan + outcome into the report shape:
    planned entries demote to skipped when the merge aborted or was
    rolled back."""
    skipped = plan.skipped
    if plan.aborted:
        skipped += tuple(
            replace(entry, resolution="skipped", reason="merge_aborted")
            for entry in plan.entries_to_apply
        )
        applied_entries: tuple[MergeEntry, ...] = ()
    elif not applied:
        skipped += tuple(
            replace(entry, resolution="skipped", reason="rolled_back")
            for entry in plan.entries_to_apply
        )
        applied_entries = ()
    else:
        applied_entries = plan.entries_to_apply
    return MergeReport(
        source_sandbox_id=source_sandbox_id,
        fork_sandbox_id=fork_sandbox_id,
        base_checkpoint_id=base_checkpoint_id,
        policy=policy,
        applied=applied_entries,
        conflicted=plan.conflicted,
        skipped=skipped,
        rolled_back=rolled_back,
    )


# ----------------------------------------------------------------------
# Path helpers
# ----------------------------------------------------------------------

def _ignored(path: str, prefixes: Sequence[str]) -> bool:
    return any(path == prefix or path.startswith(prefix.rstrip("/") + "/") for prefix in prefixes)


def _host_path(root: Path, container_path: str) -> Path:
    """Map a container-absolute path onto ``root`` without ever
    resolving through a symlinked parent and rejecting ``..`` hops."""
    parts = [part for part in container_path.split("/") if part]
    if not parts:
        raise ValueError("refusing to operate on the rootfs root")
    current = root
    for part in parts[:-1]:
        if part == "..":
            raise ValueError(f"refusing '..' in container path: {container_path}")
        current = current / part
        if os.path.islink(current):
            raise ValueError(f"refusing to traverse symlinked parent: {container_path}")
    if parts[-1] == "..":
        raise ValueError(f"refusing '..' in container path: {container_path}")
    return current / parts[-1]


def _lstat_or_none(path: Path) -> os.stat_result | None:
    try:
        return os.lstat(path)
    except OSError:
        return None


def _is_dir_no_follow(root: Path, container_path: str) -> bool:
    if container_path == "/":
        return True
    try:
        st = _lstat_or_none(_host_path(root, container_path))
    except ValueError:
        return False
    return st is not None and stat_module.S_ISDIR(st.st_mode)


def _read_regular_bytes(root: Path, container_path: str) -> bytes | None:
    """Content of a regular file, or None when missing / not regular."""
    try:
        host = _host_path(root, container_path)
    except ValueError:
        return None
    st = _lstat_or_none(host)
    if st is None or not stat_module.S_ISREG(st.st_mode):
        return None
    try:
        return host.read_bytes()
    except OSError:
        return None


def _depth(path: str) -> int:
    return path.count("/")


# ----------------------------------------------------------------------
# Planning
# ----------------------------------------------------------------------

def plan_merge(
    *,
    fork_entries: Sequence[ChangesetEntry],
    source_entries: Sequence[ChangesetEntry],
    policy: str,
    fork_root: Path,
    source_root: Path,
    base_root: Path,
    ignore_prefixes: Sequence[str] = DEFAULT_MERGE_IGNORE_PREFIXES,
    merger: MergerHook | None = None,
) -> MergePlan:
    if policy not in MERGE_POLICIES:
        raise ValueError(f"unknown merge policy: {policy!r} (expected one of {MERGE_POLICIES})")

    skipped: list[MergeEntry] = []
    conflicted: list[MergeEntry] = []
    apply_entries: list[MergeEntry] = []
    remove_ops: list[_PlannedOp] = []
    add_ops: list[_PlannedOp] = []

    # Source side: what changed since the base, after the same noise
    # filtering (directory mtime churn would otherwise make sibling
    # file adds in one directory conflict forever).
    source_changed: dict[str, str] = {}
    source_removed: list[str] = []
    for entry in source_entries:
        if _ignored(entry.path, ignore_prefixes):
            continue
        if entry.change == "modified" and _is_dir_no_follow(source_root, entry.path):
            continue
        source_changed[entry.path] = entry.change
        if entry.change == "removed":
            source_removed.append(entry.path)
        elif entry.change == "renamed" and entry.renamed_from:
            source_changed[entry.renamed_from] = "removed"
            source_removed.append(entry.renamed_from)

    def source_touched(path: str) -> bool:
        if path in source_changed:
            return True
        # A source-side removal of an ancestor swallows the fork path.
        return any(path.startswith(removed + "/") for removed in source_removed)

    def source_changed_under(path: str) -> bool:
        prefix = path + "/"
        return any(candidate.startswith(prefix) for candidate in source_changed)

    def add_apply_ops(entry: ChangesetEntry, report_entry: MergeEntry) -> None:
        if entry.change == "removed":
            remove_ops.append(_PlannedOp(kind="remove", path=entry.path, entry=report_entry))
            return
        if entry.change == "renamed":
            if entry.renamed_from:
                remove_ops.append(
                    _PlannedOp(kind="remove", path=entry.renamed_from, entry=report_entry)
                )
            # A renamed directory arrives as a single row: its children
            # are not re-listed, so the copy must take the whole tree.
            kind = "copy_tree" if _is_dir_no_follow(fork_root, entry.path) else "copy"
            add_ops.append(_PlannedOp(kind=kind, path=entry.path, entry=report_entry))
            return
        add_ops.append(_PlannedOp(kind="copy", path=entry.path, entry=report_entry))

    for entry in sorted(fork_entries, key=lambda item: item.path):
        if _ignored(entry.path, ignore_prefixes):
            skipped.append(
                MergeEntry(
                    path=entry.path,
                    change=entry.change,
                    resolution="skipped",
                    reason="ignored",
                    renamed_from=entry.renamed_from,
                )
            )
            continue
        if entry.change == "modified" and _is_dir_no_follow(fork_root, entry.path):
            skipped.append(
                MergeEntry(path=entry.path, change=entry.change, resolution="skipped", reason="dir_touch")
            )
            continue

        involved = [entry.path]
        if entry.change == "renamed" and entry.renamed_from:
            involved.append(entry.renamed_from)
        conflict = any(source_touched(path) for path in involved)
        if not conflict and entry.change == "removed":
            conflict = source_changed_under(entry.path)
        if not conflict and entry.change == "renamed" and entry.renamed_from:
            conflict = source_changed_under(entry.renamed_from)

        if not conflict:
            report_entry = MergeEntry(
                path=entry.path,
                change=entry.change,
                resolution="applied",
                renamed_from=entry.renamed_from,
            )
            apply_entries.append(report_entry)
            add_apply_ops(entry, report_entry)
            continue

        # --- conflict resolution -------------------------------------
        base_bytes = _read_regular_bytes(base_root, entry.path)
        source_bytes = _read_regular_bytes(source_root, entry.path)
        fork_bytes = _read_regular_bytes(fork_root, entry.path)

        if merger is not None:
            merged = merger(entry.path, base_bytes, source_bytes, fork_bytes)
            if merged is not None:
                report_entry = MergeEntry(
                    path=entry.path,
                    change=entry.change,
                    resolution="applied",
                    renamed_from=entry.renamed_from,
                    merged=True,
                )
                apply_entries.append(report_entry)
                add_ops.append(
                    _PlannedOp(kind="write", path=entry.path, entry=report_entry, content=merged)
                )
                continue

        if policy == "prefer_fork":
            report_entry = MergeEntry(
                path=entry.path,
                change=entry.change,
                resolution="applied",
                reason="source_changed",
                renamed_from=entry.renamed_from,
            )
            apply_entries.append(report_entry)
            add_apply_ops(entry, report_entry)
            continue
        if policy == "prefer_source":
            skipped.append(
                MergeEntry(
                    path=entry.path,
                    change=entry.change,
                    resolution="skipped",
                    reason="source_changed",
                    renamed_from=entry.renamed_from,
                )
            )
            continue
        if policy == "text_merge" and (
            entry.change == "modified"
            and source_changed.get(entry.path) == "modified"
            and base_bytes is not None
            and source_bytes is not None
            and fork_bytes is not None
        ):
            merged_text = _try_text_merge(base_bytes, source_bytes, fork_bytes)
            if merged_text is not None:
                report_entry = MergeEntry(
                    path=entry.path,
                    change=entry.change,
                    resolution="applied",
                    merged=True,
                )
                apply_entries.append(report_entry)
                add_ops.append(
                    _PlannedOp(kind="write", path=entry.path, entry=report_entry, content=merged_text)
                )
                continue
            conflicted.append(
                MergeEntry(
                    path=entry.path,
                    change=entry.change,
                    resolution="conflicted",
                    reason="unresolved_text",
                )
            )
            continue
        conflicted.append(
            MergeEntry(
                path=entry.path,
                change=entry.change,
                resolution="conflicted",
                reason="source_changed",
                renamed_from=entry.renamed_from,
            )
        )

    # fail_fast semantics: any surviving conflict aborts the whole
    # merge before a single write (text_merge degrades to fail_fast on
    # unresolved conflicts — a partial text merge is worse than none).
    aborted = bool(conflicted) and policy in ("fail_fast", "text_merge")

    remove_ops.sort(key=lambda op: (_depth(op.path), op.path), reverse=True)
    add_ops.sort(key=lambda op: (_depth(op.path), op.path))
    ops: tuple[_PlannedOp, ...] = () if aborted else tuple(remove_ops + add_ops)
    return MergePlan(
        ops=ops,
        entries_to_apply=tuple(apply_entries),
        conflicted=tuple(conflicted),
        skipped=tuple(skipped),
        aborted=aborted,
    )


def _try_text_merge(base: bytes, ours: bytes, theirs: bytes) -> bytes | None:
    from .textmerge import merge3

    try:
        merged = merge3(base.decode("utf-8"), ours.decode("utf-8"), theirs.decode("utf-8"))
    except UnicodeDecodeError:
        return None
    return None if merged is None else merged.encode("utf-8")


# ----------------------------------------------------------------------
# Application + path-level rollback
# ----------------------------------------------------------------------

def apply_plan(plan: MergePlan, *, source_root: Path, fork_root: Path, undo_root: Path) -> None:
    """Execute a resolved plan onto ``source_root``. On any failure the
    touched paths are restored from ``undo_root`` (the pre-merge
    snapshot content) in reverse order; raises ``MergeApplyError``."""
    touched: list[str] = []
    for op in plan.ops:
        touched.append(op.path)
        try:
            _apply_op(op, source_root=source_root, fork_root=fork_root)
        except Exception as exc:
            logger.exception(
                "Merge apply failed path=%s kind=%s; rolling back %d touched path(s)",
                op.path,
                op.kind,
                len(touched),
            )
            undo_errors = _undo(touched, source_root=source_root, undo_root=undo_root)
            raise MergeApplyError(
                f"merge apply failed at {op.path}: {exc}",
                path=op.path,
                rolled_back=not undo_errors,
            ) from exc


def _apply_op(op: _PlannedOp, *, source_root: Path, fork_root: Path) -> None:
    dest = _host_path(source_root, op.path)
    if op.kind == "remove":
        _remove_path(dest)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    if op.kind == "write":
        assert op.content is not None
        st = _lstat_or_none(dest)
        if st is not None and not stat_module.S_ISREG(st.st_mode):
            _remove_path(dest)
        dest.write_bytes(op.content)
        donor = _host_path(fork_root, op.path)
        if _lstat_or_none(donor) is not None:
            _copy_stat_owner(donor, dest)
        return
    src = _host_path(fork_root, op.path)
    if op.kind == "copy_tree":
        _remove_path(dest)
        shutil.copytree(src, dest, symlinks=True)
        _chown_tree(src, dest)
        return
    _copy_node(src, dest)


def _remove_path(path: Path) -> None:
    st = _lstat_or_none(path)
    if st is None:
        return
    if stat_module.S_ISDIR(st.st_mode):
        try:
            path.rmdir()
        except OSError:
            # Children the changeset did not list individually (or that
            # the policy resolved fork-wins over): the subtree goes.
            shutil.rmtree(path)
        return
    path.unlink()


def _copy_node(src: Path, dest: Path) -> None:
    st = os.lstat(src)
    if stat_module.S_ISDIR(st.st_mode):
        existing = _lstat_or_none(dest)
        if existing is not None and not stat_module.S_ISDIR(existing.st_mode):
            _remove_path(dest)
        dest.mkdir(exist_ok=True)
        _copy_stat_owner(src, dest)
        return
    existing = _lstat_or_none(dest)
    keep_in_place = (
        existing is not None
        and stat_module.S_ISREG(existing.st_mode)
        and stat_module.S_ISREG(st.st_mode)
    )
    if existing is not None and not keep_in_place:
        # Type change (dir/symlink/fifo/...): clear before recreating.
        _remove_path(dest)
    if stat_module.S_ISLNK(st.st_mode):
        os.symlink(os.readlink(src), dest)
        os.lchown(dest, st.st_uid, st.st_gid)
        return
    if stat_module.S_ISREG(st.st_mode):
        shutil.copyfile(src, dest)
        _copy_stat_owner(src, dest)
        return
    if stat_module.S_ISFIFO(st.st_mode):
        os.mkfifo(dest)
        _copy_stat_owner(src, dest)
        return
    logger.warning("Merge skips unsupported node type: %s (mode=%o)", src, st.st_mode)


def _copy_stat_owner(src: Path, dest: Path) -> None:
    st = os.lstat(src)
    shutil.copystat(src, dest, follow_symlinks=False)
    try:
        os.chown(dest, st.st_uid, st.st_gid, follow_symlinks=False)
    except (PermissionError, NotImplementedError):
        logger.debug("Merge could not chown %s", dest)


def _chown_tree(src_root: Path, dest_root: Path) -> None:
    """copytree preserves mode/times but not ownership; walk it over."""
    _copy_stat_owner(src_root, dest_root)
    for dirpath, dirnames, filenames in os.walk(src_root):
        rel = os.path.relpath(dirpath, src_root)
        dest_dir = dest_root if rel == "." else dest_root / rel
        for name in dirnames + filenames:
            src_item = Path(dirpath) / name
            dest_item = dest_dir / name
            if _lstat_or_none(dest_item) is not None:
                _copy_stat_owner(src_item, dest_item)


def _undo(touched: Sequence[str], *, source_root: Path, undo_root: Path) -> list[str]:
    """Restore every touched container path from the pre-merge snapshot
    content, newest mutation first. Returns paths whose restore failed."""
    errors: list[str] = []
    for container_path in reversed(list(touched)):
        try:
            dest = _host_path(source_root, container_path)
            snap = _host_path(undo_root, container_path)
        except ValueError:
            # The apply refused this path for the same reason (traversal
            # or symlinked parent) — nothing was written, nothing to undo.
            logger.debug("Merge rollback skips unmappable path=%s", container_path)
            continue
        try:
            _remove_path(dest)
            if _lstat_or_none(snap) is not None:
                _restore_node(snap, dest)
        except Exception:
            logger.exception("Merge rollback failed for path=%s", container_path)
            errors.append(container_path)
    return errors


def _restore_node(snap: Path, dest: Path) -> None:
    st = os.lstat(snap)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if stat_module.S_ISDIR(st.st_mode):
        shutil.copytree(snap, dest, symlinks=True)
        _chown_tree(snap, dest)
        return
    if stat_module.S_ISLNK(st.st_mode):
        os.symlink(os.readlink(snap), dest)
        os.lchown(dest, st.st_uid, st.st_gid)
        return
    if stat_module.S_ISREG(st.st_mode):
        shutil.copyfile(snap, dest)
        _copy_stat_owner(snap, dest)
        return
    if stat_module.S_ISFIFO(st.st_mode):
        os.mkfifo(dest)
        _copy_stat_owner(snap, dest)
        return
    logger.warning("Merge rollback skips unsupported node type: %s (mode=%o)", snap, st.st_mode)
