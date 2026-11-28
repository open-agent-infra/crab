"""Unit tests for the C2 fs merge engine: plan classification and policy
resolution, host-side application with path-level rollback (plain
directories stand in for datasets/snapshots), CrabSystem orchestration
against a duck-typed runtime (quiesce, transient snapshot, guards,
journal/telemetry), and the SDK plumbing. Host-runnable — no runc."""
from __future__ import annotations

import os
import shutil
import stat
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from crab import (
    AdapterFileSystemCWorker,
    AdapterFileSystemRWorker,
    AdapterProcessCWorker,
    AdapterProcessRWorker,
    CRExecutor,
    CRScheduler,
    CrabSystem,
    DefaultCWorker,
    DefaultRWorker,
    EBPFSandboxInspector,
    ExecutorConfig,
    InMemorySchedulerStateStore,
    InMemoryTelemetrySink,
    LocalCheckpointManager,
    SandboxId,
    SchedulerConfig,
    StorageConfig,
)
from crab.ids import CheckpointId
from crab.journal import ActionJournal
from crab.merging import (
    MergeApplyError,
    MergeError,
    apply_plan,
    build_report,
    plan_merge,
)
from crab.models import ChangesetEntry, MergeReport
from crab.sandbox import Sandbox
from crab.scheduler import FaultToleranceCheckpointingPolicy
from crab.txn import TxnError


def _entry(path: str, change: str, renamed_from: str | None = None) -> ChangesetEntry:
    return ChangesetEntry(path=path, change=change, renamed_from=renamed_from)


def _write(root: Path, container_path: str, content: str = "x\n", *, mode: int | None = None) -> Path:
    target = root / container_path.lstrip("/")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    if mode is not None:
        os.chmod(target, mode)
    return target


def _tree_signature(root: Path) -> dict[str, object]:
    """Path -> content/type map for whole-tree equality asserts."""
    signature: dict[str, object] = {}
    for path in sorted(root.rglob("*")):
        rel = str(path.relative_to(root))
        st = os.lstat(path)
        if stat.S_ISLNK(st.st_mode):
            signature[rel] = ("link", os.readlink(path))
        elif stat.S_ISDIR(st.st_mode):
            signature[rel] = ("dir",)
        elif stat.S_ISREG(st.st_mode):
            signature[rel] = ("file", path.read_bytes())
        else:
            signature[rel] = ("other", st.st_mode)
    return signature


# ----------------------------------------------------------------------
# Planning: classification + policy resolution
# ----------------------------------------------------------------------


class PlanMergeTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="crab_merge_plan_")
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        self.fork_root = root / "fork"
        self.source_root = root / "source"
        self.base_root = root / "base"
        for directory in (self.fork_root, self.source_root, self.base_root):
            directory.mkdir()

    def _plan(self, fork_entries, source_entries=(), policy="fail_fast", **kwargs):
        return plan_merge(
            fork_entries=fork_entries,
            source_entries=source_entries,
            policy=policy,
            fork_root=self.fork_root,
            source_root=self.source_root,
            base_root=self.base_root,
            **kwargs,
        )

    def test_disjoint_changes_apply_in_order(self) -> None:
        _write(self.fork_root, "/new.txt")
        _write(self.fork_root, "/mod.txt")
        plan = self._plan(
            [_entry("/new.txt", "added"), _entry("/mod.txt", "modified")],
            [_entry("/other.txt", "modified")],
        )
        self.assertFalse(plan.aborted)
        self.assertEqual(plan.conflicted, ())
        self.assertEqual([op.kind for op in plan.ops], ["copy", "copy"])
        self.assertEqual([op.path for op in plan.ops], ["/mod.txt", "/new.txt"])
        self.assertEqual(
            [(entry.path, entry.resolution) for entry in plan.entries_to_apply],
            [("/mod.txt", "applied"), ("/new.txt", "applied")],
        )

    def test_same_path_conflict_fail_fast_aborts_before_any_op(self) -> None:
        _write(self.fork_root, "/shared.txt", "fork\n")
        _write(self.source_root, "/shared.txt", "source\n")
        plan = self._plan([_entry("/shared.txt", "modified")], [_entry("/shared.txt", "modified")])
        self.assertTrue(plan.aborted)
        self.assertEqual(plan.ops, ())
        self.assertEqual(len(plan.conflicted), 1)
        self.assertEqual(plan.conflicted[0].reason, "source_changed")

    def test_build_report_demotes_planned_entries_on_abort(self) -> None:
        _write(self.fork_root, "/shared.txt", "fork\n")
        _write(self.source_root, "/shared.txt", "source\n")
        _write(self.fork_root, "/free.txt")
        plan = self._plan(
            [_entry("/shared.txt", "modified"), _entry("/free.txt", "added")],
            [_entry("/shared.txt", "modified")],
        )
        self.assertTrue(plan.aborted)
        report = build_report(
            source_sandbox_id=SandboxId("s"),
            fork_sandbox_id=SandboxId("f"),
            base_checkpoint_id=CheckpointId("b"),
            policy="fail_fast",
            plan=plan,
            applied=True,
        )
        self.assertEqual(report.applied, ())
        reasons = {(entry.path, entry.reason) for entry in report.skipped}
        self.assertIn(("/free.txt", "merge_aborted"), reasons)

    def test_prefer_fork_applies_conflicts(self) -> None:
        _write(self.fork_root, "/shared.txt", "fork\n")
        _write(self.source_root, "/shared.txt", "source\n")
        plan = self._plan(
            [_entry("/shared.txt", "modified")],
            [_entry("/shared.txt", "modified")],
            policy="prefer_fork",
        )
        self.assertFalse(plan.aborted)
        self.assertEqual(plan.conflicted, ())
        self.assertEqual([op.kind for op in plan.ops], ["copy"])
        self.assertEqual(plan.entries_to_apply[0].reason, "source_changed")

    def test_prefer_source_skips_conflicts(self) -> None:
        _write(self.fork_root, "/shared.txt", "fork\n")
        _write(self.source_root, "/shared.txt", "source\n")
        plan = self._plan(
            [_entry("/shared.txt", "modified")],
            [_entry("/shared.txt", "modified")],
            policy="prefer_source",
        )
        self.assertFalse(plan.aborted)
        self.assertEqual(plan.ops, ())
        self.assertEqual(plan.skipped[0].reason, "source_changed")

    def test_default_ignore_prefixes_filter_dump_noise(self) -> None:
        _write(self.fork_root, "/keep.txt")
        plan = self._plan(
            [_entry("/tmp", "modified"), _entry("/run/lock", "added"), _entry("/keep.txt", "added")],
        )
        self.assertEqual([op.path for op in plan.ops], ["/keep.txt"])
        self.assertEqual(
            sorted(entry.path for entry in plan.skipped if entry.reason == "ignored"),
            ["/run/lock", "/tmp"],
        )

    def test_ignore_prefix_does_not_swallow_siblings(self) -> None:
        _write(self.fork_root, "/keep.txt")
        _write(self.fork_root, "/data/z.txt")
        plan = self._plan(
            [_entry("/keep.txt", "added"), _entry("/data/z.txt", "added")],
            ignore_prefixes=("/keep",),
        )
        # "/keep" must not swallow "/keep.txt"; "/data/z.txt" untouched.
        self.assertEqual([op.path for op in plan.ops], ["/keep.txt", "/data/z.txt"])

    def test_directory_modified_entries_drop_on_both_sides(self) -> None:
        (self.fork_root / "shared").mkdir()
        (self.source_root / "shared").mkdir()
        _write(self.fork_root, "/shared/fork.txt")
        plan = self._plan(
            [_entry("/shared", "modified"), _entry("/shared/fork.txt", "added")],
            [_entry("/shared", "modified"), _entry("/shared/source.txt", "added")],
        )
        self.assertFalse(plan.aborted)
        self.assertEqual(plan.conflicted, ())
        self.assertEqual([op.path for op in plan.ops], ["/shared/fork.txt"])
        self.assertEqual(plan.skipped[0].reason, "dir_touch")

    def test_source_removed_ancestor_conflicts(self) -> None:
        _write(self.fork_root, "/gone/child.txt")
        plan = self._plan([_entry("/gone/child.txt", "added")], [_entry("/gone", "removed")])
        self.assertTrue(plan.aborted)
        self.assertEqual(plan.conflicted[0].path, "/gone/child.txt")

    def test_fork_removal_over_source_child_change_conflicts(self) -> None:
        plan = self._plan([_entry("/dir", "removed")], [_entry("/dir/new.txt", "added")])
        self.assertTrue(plan.aborted)
        self.assertEqual(plan.conflicted[0].path, "/dir")

    def test_rename_clean_expands_to_remove_and_copy(self) -> None:
        _write(self.fork_root, "/new.txt", "moved\n")
        plan = self._plan([_entry("/new.txt", "renamed", renamed_from="/old.txt")])
        self.assertEqual(
            [(op.kind, op.path) for op in plan.ops],
            [("remove", "/old.txt"), ("copy", "/new.txt")],
        )
        entry = plan.entries_to_apply[0]
        self.assertEqual((entry.change, entry.renamed_from), ("renamed", "/old.txt"))

    def test_rename_conflicts_when_source_touched_old_path(self) -> None:
        _write(self.fork_root, "/new.txt", "moved\n")
        plan = self._plan(
            [_entry("/new.txt", "renamed", renamed_from="/old.txt")],
            [_entry("/old.txt", "modified")],
        )
        self.assertTrue(plan.aborted)

    def test_renamed_directory_copies_whole_tree(self) -> None:
        _write(self.fork_root, "/d2/child.txt")
        plan = self._plan([_entry("/d2", "renamed", renamed_from="/d1")])
        self.assertEqual(
            [(op.kind, op.path) for op in plan.ops],
            [("remove", "/d1"), ("copy_tree", "/d2")],
        )

    def test_text_merge_resolves_disjoint_line_edits(self) -> None:
        _write(self.base_root, "/doc.txt", "one\ntwo\nthree\nfour\nfive\n")
        _write(self.source_root, "/doc.txt", "ONE\ntwo\nthree\nfour\nfive\n")
        _write(self.fork_root, "/doc.txt", "one\ntwo\nthree\nfour\nFIVE\n")
        plan = self._plan(
            [_entry("/doc.txt", "modified")],
            [_entry("/doc.txt", "modified")],
            policy="text_merge",
        )
        self.assertFalse(plan.aborted)
        self.assertEqual(plan.ops[0].kind, "write")
        self.assertEqual(plan.ops[0].content, b"ONE\ntwo\nthree\nfour\nFIVE\n")
        self.assertTrue(plan.entries_to_apply[0].merged)

    def test_text_merge_unresolved_overlap_aborts(self) -> None:
        _write(self.base_root, "/doc.txt", "one\ntwo\n")
        _write(self.source_root, "/doc.txt", "UNO\ntwo\n")
        _write(self.fork_root, "/doc.txt", "EINS\ntwo\n")
        plan = self._plan(
            [_entry("/doc.txt", "modified")],
            [_entry("/doc.txt", "modified")],
            policy="text_merge",
        )
        self.assertTrue(plan.aborted)
        self.assertEqual(plan.conflicted[0].reason, "unresolved_text")

    def test_text_merge_binary_content_is_unresolved(self) -> None:
        _write(self.base_root, "/blob.bin", "base\n")
        _write(self.source_root, "/blob.bin", "source\n")
        (self.fork_root / "blob.bin").write_bytes(b"\xff\xfe\x00\x01")
        plan = self._plan(
            [_entry("/blob.bin", "modified")],
            [_entry("/blob.bin", "modified")],
            policy="text_merge",
        )
        self.assertTrue(plan.aborted)
        self.assertEqual(plan.conflicted[0].reason, "unresolved_text")

    def test_merger_hook_gets_first_shot_under_any_policy(self) -> None:
        _write(self.base_root, "/doc.txt", "base\n")
        _write(self.source_root, "/doc.txt", "source\n")
        _write(self.fork_root, "/doc.txt", "fork\n")
        seen: list[tuple] = []

        def hook(path, base, source, fork):
            seen.append((path, base, source, fork))
            return b"hooked\n"

        plan = self._plan(
            [_entry("/doc.txt", "modified")],
            [_entry("/doc.txt", "modified")],
            policy="fail_fast",
            merger=hook,
        )
        self.assertFalse(plan.aborted)
        self.assertEqual(plan.ops[0].kind, "write")
        self.assertEqual(plan.ops[0].content, b"hooked\n")
        self.assertTrue(plan.entries_to_apply[0].merged)
        self.assertEqual(seen, [("/doc.txt", b"base\n", b"source\n", b"fork\n")])

    def test_merger_hook_declining_falls_back_to_policy(self) -> None:
        _write(self.source_root, "/doc.txt", "source\n")
        _write(self.fork_root, "/doc.txt", "fork\n")
        plan = self._plan(
            [_entry("/doc.txt", "modified")],
            [_entry("/doc.txt", "modified")],
            policy="fail_fast",
            merger=lambda path, base, source, fork: None,
        )
        self.assertTrue(plan.aborted)

    def test_removals_deepest_first_and_adds_shallowest_first(self) -> None:
        _write(self.fork_root, "/x/y.txt")
        plan = self._plan(
            [
                _entry("/a/b/c.txt", "removed"),
                _entry("/a/b", "removed"),
                _entry("/x", "added"),
                _entry("/x/y.txt", "added"),
            ]
        )
        self.assertEqual(
            [(op.kind, op.path) for op in plan.ops],
            [
                ("remove", "/a/b/c.txt"),
                ("remove", "/a/b"),
                ("copy", "/x"),
                ("copy", "/x/y.txt"),
            ],
        )

    def test_unknown_policy_raises(self) -> None:
        with self.assertRaises(ValueError):
            self._plan([], policy="theirs")


# ----------------------------------------------------------------------
# Application + rollback on real directories
# ----------------------------------------------------------------------


class ApplyPlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="crab_merge_apply_")
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        self.fork_root = root / "fork"
        self.source_root = root / "source"
        self.base_root = root / "base"
        self.undo_root = root / "undo"
        for directory in (self.fork_root, self.source_root, self.base_root):
            directory.mkdir()

    def _snapshot_source(self) -> None:
        """Emulate the pre-merge snapshot: freeze the source tree."""
        shutil.copytree(self.source_root, self.undo_root, symlinks=True)

    def _plan(self, fork_entries, source_entries=(), policy="fail_fast", **kwargs):
        return plan_merge(
            fork_entries=fork_entries,
            source_entries=source_entries,
            policy=policy,
            fork_root=self.fork_root,
            source_root=self.source_root,
            base_root=self.base_root,
            **kwargs,
        )

    def _apply(self, plan) -> None:
        apply_plan(
            plan,
            source_root=self.source_root,
            fork_root=self.fork_root,
            undo_root=self.undo_root,
        )

    def test_apply_full_matrix(self) -> None:
        _write(self.source_root, "/mod.txt", "v1\n")
        _write(self.source_root, "/del.txt")
        _write(self.source_root, "/old.txt", "moving\n")
        self._snapshot_source()
        _write(self.fork_root, "/mod.txt", "v2\n", mode=0o640)
        _write(self.fork_root, "/new.txt", "fresh\n")
        _write(self.fork_root, "/td/a.txt", "nested\n")
        os.symlink("mod.txt", self.fork_root / "ln")
        _write(self.fork_root, "/new-name.txt", "moving\n")

        plan = self._plan(
            [
                _entry("/mod.txt", "modified"),
                _entry("/new.txt", "added"),
                _entry("/td", "added"),
                _entry("/td/a.txt", "added"),
                _entry("/ln", "added"),
                _entry("/del.txt", "removed"),
                _entry("/new-name.txt", "renamed", renamed_from="/old.txt"),
            ]
        )
        self._apply(plan)

        self.assertEqual((self.source_root / "mod.txt").read_text(), "v2\n")
        self.assertEqual(stat.S_IMODE(os.lstat(self.source_root / "mod.txt").st_mode), 0o640)
        self.assertEqual((self.source_root / "new.txt").read_text(), "fresh\n")
        self.assertEqual((self.source_root / "td" / "a.txt").read_text(), "nested\n")
        self.assertEqual(os.readlink(self.source_root / "ln"), "mod.txt")
        self.assertFalse((self.source_root / "del.txt").exists())
        self.assertFalse((self.source_root / "old.txt").exists())
        self.assertEqual((self.source_root / "new-name.txt").read_text(), "moving\n")

    def test_apply_failure_rolls_back_everything(self) -> None:
        _write(self.source_root, "/keep.txt", "original\n")
        self._snapshot_source()
        before = _tree_signature(self.source_root)
        _write(self.fork_root, "/a_ok.txt", "applied-then-undone\n")
        _write(self.fork_root, "/keep.txt", "fork-version\n")
        # /zz_missing.txt is planned but absent from the fork tree — the
        # copy fails after the first two ops already landed.
        plan = self._plan(
            [
                _entry("/a_ok.txt", "added"),
                _entry("/keep.txt", "modified"),
                _entry("/zz_missing.txt", "added"),
            ]
        )
        with self.assertRaises(MergeApplyError) as caught:
            self._apply(plan)
        self.assertTrue(caught.exception.rolled_back)
        self.assertEqual(caught.exception.path, "/zz_missing.txt")
        self.assertEqual(_tree_signature(self.source_root), before)

    def test_rollback_restores_removed_directory_tree(self) -> None:
        _write(self.source_root, "/data/f1.txt", "one\n")
        _write(self.source_root, "/data/f2.txt", "two\n")
        self._snapshot_source()
        before = _tree_signature(self.source_root)
        plan = self._plan([_entry("/data", "removed"), _entry("/zz_missing.txt", "added")])
        with self.assertRaises(MergeApplyError) as caught:
            self._apply(plan)
        self.assertTrue(caught.exception.rolled_back)
        self.assertEqual(_tree_signature(self.source_root), before)
        self.assertEqual((self.source_root / "data" / "f1.txt").read_text(), "one\n")

    def test_traversal_rejected_without_touching_source(self) -> None:
        _write(self.source_root, "/keep.txt", "original\n")
        self._snapshot_source()
        before = _tree_signature(self.source_root)
        plan = self._plan([_entry("/../evil.txt", "added")])
        with self.assertRaises(MergeApplyError) as caught:
            self._apply(plan)
        self.assertTrue(caught.exception.rolled_back)
        self.assertEqual(_tree_signature(self.source_root), before)
        self.assertFalse((Path(self._tmp.name) / "evil.txt").exists())

    def test_symlinked_parent_rejected(self) -> None:
        (self.source_root / "victim").mkdir()
        _write(self.source_root, "/victim/precious.txt", "safe\n")
        os.symlink("victim", self.source_root / "sl")
        self._snapshot_source()
        _write(self.fork_root, "/sl/x.txt", "attack\n")
        plan = self._plan([_entry("/sl/x.txt", "added")])
        with self.assertRaises(MergeApplyError):
            self._apply(plan)
        self.assertEqual((self.source_root / "victim" / "precious.txt").read_text(), "safe\n")
        self.assertFalse((self.source_root / "victim" / "x.txt").exists())

    def test_fifo_nodes_are_recreated(self) -> None:
        self._snapshot_source()
        os.mkfifo(self.fork_root / "pipe")
        plan = self._plan([_entry("/pipe", "added")])
        self._apply(plan)
        self.assertTrue(stat.S_ISFIFO(os.lstat(self.source_root / "pipe").st_mode))

    def test_type_change_file_to_symlink(self) -> None:
        _write(self.source_root, "/thing", "regular\n")
        self._snapshot_source()
        os.symlink("elsewhere", self.fork_root / "thing")
        plan = self._plan([_entry("/thing", "modified")])
        self._apply(plan)
        self.assertEqual(os.readlink(self.source_root / "thing"), "elsewhere")


# ----------------------------------------------------------------------
# CrabSystem orchestration with a duck-typed runtime
# ----------------------------------------------------------------------


class FakeMergeRuntime:
    """Just enough runtime surface for merge_from_fork: rootfs paths are
    plain directories, filesystem snapshots are copytrees."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.paused: list[str] = []
        self.resumed: list[str] = []
        self.snapshots: list[tuple[str, str]] = []
        self.discarded: list[tuple[str, str]] = []
        self.changesets: dict[str, list[ChangesetEntry]] = {}
        self.statuses: dict[str, str] = {}

    def describe(self, sandbox_id):
        return SimpleNamespace(status=self.statuses.get(str(sandbox_id), "running"))

    def pause(self, sandbox_id) -> None:
        self.paused.append(str(sandbox_id))

    def resume(self, sandbox_id) -> None:
        self.resumed.append(str(sandbox_id))

    def rootfs_path_for(self, sandbox_id) -> Path:
        return self.root / str(sandbox_id) / "rootfs"

    def dataset_name_for(self, sandbox_id) -> str:
        return f"pool/{sandbox_id}"

    def changeset_since(self, sandbox_id, checkpoint_id):
        entries = self.changesets.get(str(sandbox_id))
        if entries is None:
            raise FileNotFoundError(f"changeset base snapshot missing: {checkpoint_id}")
        return list(entries)

    def checkpoint_filesystem(self, sandbox_id, checkpoint_id):
        self.snapshots.append((str(sandbox_id), str(checkpoint_id)))
        snapshot_dir = self._snapshot_dir(sandbox_id, checkpoint_id)
        shutil.copytree(self.rootfs_path_for(sandbox_id), snapshot_dir, symlinks=True)
        return SimpleNamespace(executed=True, reason=None)

    def snapshot_content_root(self, sandbox_id, checkpoint_id) -> Path:
        return self._snapshot_dir(sandbox_id, checkpoint_id)

    def discard_partial_checkpoint(self, sandbox_id, checkpoint_id) -> None:
        self.discarded.append((str(sandbox_id), str(checkpoint_id)))

    def _snapshot_dir(self, sandbox_id, checkpoint_id) -> Path:
        return self.root / str(sandbox_id) / "snapshots" / str(checkpoint_id)


class SystemMergeTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="crab_merge_system_")
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.fake = FakeMergeRuntime(self.root)
        self.telemetry = InMemoryTelemetrySink()
        self.journal = ActionJournal(self.root / "storage" / "journal")
        storage = LocalCheckpointManager(
            StorageConfig(root_dir=self.root / "storage"),
            destroy_filesystem_ref=lambda fs_ref: None,
        )
        executor = CRExecutor(
            ExecutorConfig(max_workers=1),
            DefaultCWorker(
                AdapterProcessCWorker(self.fake),
                AdapterFileSystemCWorker(self.fake),
                storage,
                self.fake,
            ),
            DefaultRWorker(
                AdapterProcessRWorker(self.fake),
                AdapterFileSystemRWorker(self.fake),
                storage,
            ),
            self.telemetry,
        )
        self.addCleanup(executor.shutdown)
        scheduler_cfg = SchedulerConfig(
            min_checkpoint_interval_seconds=0.0,
            force_checkpoint_after_seconds=0.0,
            require_change_signal=True,
        )
        scheduler = CRScheduler(
            scheduler_cfg,
            EBPFSandboxInspector(),
            self.fake,
            InMemorySchedulerStateStore(),
            self.telemetry,
            FaultToleranceCheckpointingPolicy(scheduler_cfg),
        )
        self.system = CrabSystem(
            scheduler=scheduler,
            executor=executor,
            storage=storage,
            inspector=EBPFSandboxInspector(),
            runtime=self.fake,
            telemetry=self.telemetry,
            journal=self.journal,
        )
        self.source = SandboxId("sbx-src")
        self.fork = SandboxId("sbx-fork")
        self.source_root = self.fake.rootfs_path_for(self.source)
        self.fork_root = self.fake.rootfs_path_for(self.fork)
        self.source_root.mkdir(parents=True)
        self.fork_root.mkdir(parents=True)
        # Fork-point snapshot content (base for text merges).
        self.fake._snapshot_dir(self.fork, CheckpointId("base-1")).mkdir(parents=True)
        self.journal.record_lifecycle(
            self.fork,
            "fork_created",
            metadata={"source_sandbox_id": str(self.source), "checkpoint_id": "base-1"},
        )

    def _merge_markers(self, sandbox_id, event: str) -> list[dict]:
        return [
            record.payload
            for record in self.journal.entries(sandbox_id, kind="lifecycle")
            if record.payload.get("event") == event
        ]

    def test_disjoint_merge_applies_and_reports(self) -> None:
        self.fake.changesets = {str(self.fork): [_entry("/new.txt", "added")], str(self.source): []}
        _write(self.fork_root, "/new.txt", "fork\n")

        report = self.system.merge_from_fork(self.source, self.fork)

        self.assertEqual([entry.path for entry in report.applied], ["/new.txt"])
        self.assertEqual(report.conflicted, ())
        self.assertFalse(report.rolled_back)
        self.assertEqual((self.source_root / "new.txt").read_text(), "fork\n")
        self.assertEqual(self.fake.paused, [str(self.fork), str(self.source)])
        self.assertEqual(sorted(self.fake.resumed), sorted([str(self.fork), str(self.source)]))
        self.assertEqual(len(self.fake.snapshots), 1)
        self.assertEqual(self.fake.snapshots, self.fake.discarded)
        marker = self._merge_markers(self.source, "merge")[-1]["metadata"]
        self.assertTrue(marker["succeeded"])
        self.assertEqual(marker["applied"], 1)
        self.assertEqual(marker["policy"], "fail_fast")
        fork_marker = self._merge_markers(self.fork, "merged_into")[-1]["metadata"]
        self.assertEqual(fork_marker["source_sandbox_id"], str(self.source))
        events = [attrs for name, attrs in self.telemetry.events if name == "merge.completed"]
        self.assertTrue(events and events[-1]["succeeded"])
        self.assertFalse(self.system._merge_active(self.source))
        self.assertFalse(self.system._merge_active(self.fork))

    def test_conflict_fail_fast_takes_no_snapshot(self) -> None:
        self.fake.changesets = {
            str(self.fork): [_entry("/shared.txt", "modified")],
            str(self.source): [_entry("/shared.txt", "modified")],
        }
        _write(self.fork_root, "/shared.txt", "fork\n")
        _write(self.source_root, "/shared.txt", "source\n")

        report = self.system.merge_from_fork(self.source, self.fork)

        self.assertEqual(report.applied, ())
        self.assertEqual(len(report.conflicted), 1)
        self.assertEqual(self.fake.snapshots, [])
        self.assertEqual((self.source_root / "shared.txt").read_text(), "source\n")
        marker = self._merge_markers(self.source, "merge")[-1]["metadata"]
        self.assertTrue(marker["succeeded"])
        self.assertEqual(marker["conflicted"], 1)

    def test_prefer_fork_overwrites_source(self) -> None:
        self.fake.changesets = {
            str(self.fork): [_entry("/shared.txt", "modified")],
            str(self.source): [_entry("/shared.txt", "modified")],
        }
        _write(self.fork_root, "/shared.txt", "fork\n")
        _write(self.source_root, "/shared.txt", "source\n")

        report = self.system.merge_from_fork(self.source, self.fork, policy="prefer_fork")

        self.assertEqual((self.source_root / "shared.txt").read_text(), "fork\n")
        self.assertEqual(len(report.applied), 1)
        self.assertEqual(len(self.fake.snapshots), 1)
        self.assertEqual(self.fake.snapshots, self.fake.discarded)

    def test_rejects_sandbox_without_fork_marker(self) -> None:
        with self.assertRaises(ValueError):
            self.system.merge_from_fork(self.source, self.source)

    def test_rejects_fork_of_another_source(self) -> None:
        stranger = SandboxId("sbx-stranger")
        self.journal.record_lifecycle(
            stranger,
            "fork_created",
            metadata={"source_sandbox_id": "sbx-other", "checkpoint_id": "base-9"},
        )
        with self.assertRaises(ValueError):
            self.system.merge_from_fork(self.source, stranger)

    def test_refuses_during_active_txn(self) -> None:
        with self.system._txn_lock:
            self.system._active_txns[self.source] = None
        self.addCleanup(lambda: self.system._active_txns.pop(self.source, None))
        with self.assertRaises(MergeError):
            self.system.merge_from_fork(self.source, self.fork)
        self.assertEqual(self.fake.paused, [])
        self.assertFalse(self.system._merge_active(self.source))

    def test_txn_begin_refuses_during_merge(self) -> None:
        with self.system._merge_lock:
            self.system._active_merges.add(self.source)
        self.addCleanup(lambda: self.system._active_merges.discard(self.source))
        with self.assertRaises(TxnError):
            self.system.begin_txn(self.source)
        self.assertFalse(self.system._txn_active(self.source))

    def test_concurrent_merge_reservation(self) -> None:
        with self.system._merge_lock:
            self.system._active_merges.add(self.fork)
        self.addCleanup(lambda: self.system._active_merges.discard(self.fork))
        with self.assertRaises(MergeError):
            self.system.merge_from_fork(self.source, self.fork)

    def test_apply_failure_rolls_back_and_reports(self) -> None:
        self.fake.changesets = {
            str(self.fork): [_entry("/a_ok.txt", "added"), _entry("/zz_boom.txt", "added")],
            str(self.source): [],
        }
        _write(self.fork_root, "/a_ok.txt", "fork\n")
        # /zz_boom.txt intentionally missing from the fork rootfs.

        with self.assertRaises(MergeError) as caught:
            self.system.merge_from_fork(self.source, self.fork)

        report = caught.exception.report
        self.assertIsNotNone(report)
        self.assertTrue(report.rolled_back)
        self.assertEqual(report.applied, ())
        self.assertIn(
            ("/a_ok.txt", "rolled_back"),
            {(entry.path, entry.reason) for entry in report.skipped},
        )
        self.assertFalse((self.source_root / "a_ok.txt").exists())
        self.assertEqual(len(self.fake.discarded), 1)
        self.assertEqual(sorted(self.fake.resumed), sorted([str(self.fork), str(self.source)]))
        marker = self._merge_markers(self.source, "merge")[-1]["metadata"]
        self.assertFalse(marker["succeeded"])
        self.assertTrue(marker["rolled_back"])
        self.assertFalse(self.system._merge_active(self.source))

    def test_pause_skips_non_running_sandboxes(self) -> None:
        self.fake.statuses[str(self.fork)] = "stopped"
        self.fake.changesets = {str(self.fork): [_entry("/new.txt", "added")], str(self.source): []}
        _write(self.fork_root, "/new.txt", "fork\n")

        self.system.merge_from_fork(self.source, self.fork)

        self.assertEqual(self.fake.paused, [str(self.source)])
        self.assertEqual(self.fake.resumed, [str(self.source)])

    def test_scheduled_checkpoints_suppressed_during_merge(self) -> None:
        with self.system._merge_lock:
            self.system._active_merges.add(self.source)
        self.addCleanup(lambda: self.system._active_merges.discard(self.source))
        self.assertIsNone(self.system._execute_checkpoint_flow(self.source))

    def test_report_json_shape(self) -> None:
        self.fake.changesets = {str(self.fork): [_entry("/new.txt", "added")], str(self.source): []}
        _write(self.fork_root, "/new.txt", "fork\n")
        report = self.system.merge_from_fork(self.source, self.fork)
        payload = report.to_json()
        self.assertEqual(payload["source_sandbox_id"], str(self.source))
        self.assertEqual(payload["fork_sandbox_id"], str(self.fork))
        self.assertEqual(payload["base_checkpoint_id"], "base-1")
        self.assertEqual(payload["policy"], "fail_fast")
        self.assertEqual(payload["applied"], [{"path": "/new.txt", "change": "added", "resolution": "applied"}])
        self.assertFalse(payload["rolled_back"])


# ----------------------------------------------------------------------
# SDK plumbing
# ----------------------------------------------------------------------


class SandboxMergePlumbingTests(unittest.TestCase):
    class _FakeSystem:
        def __init__(self) -> None:
            self.calls: list = []

        def merge_from_fork(self, source_sandbox_id, fork_sandbox_id, *, policy, ignore_prefixes, merger):
            self.calls.append((source_sandbox_id, fork_sandbox_id, policy, ignore_prefixes, merger))
            return MergeReport(
                source_sandbox_id=source_sandbox_id,
                fork_sandbox_id=fork_sandbox_id,
                base_checkpoint_id=CheckpointId("base-1"),
                policy=policy,
                applied=(),
                conflicted=(),
                skipped=(),
            )

    class _FakeEngine:
        def __init__(self, system) -> None:
            self.system = system

        def _register_sandbox(self, sandbox) -> None:
            pass

    def test_merge_plumbs_arguments_and_accepts_sandbox_instance(self) -> None:
        system = self._FakeSystem()
        engine = self._FakeEngine(system)
        source = Sandbox.connect(SandboxId("sbx-src"), engine=engine)
        fork = Sandbox.connect(SandboxId("sbx-fork"), engine=engine)

        report = source.merge(fork, policy="prefer_fork", ignore_prefixes=("/scratch",))

        self.assertEqual(
            system.calls,
            [(SandboxId("sbx-src"), SandboxId("sbx-fork"), "prefer_fork", ("/scratch",), None)],
        )
        self.assertEqual(report.policy, "prefer_fork")

    def test_merge_accepts_plain_id(self) -> None:
        system = self._FakeSystem()
        source = Sandbox.connect(SandboxId("sbx-src"), engine=self._FakeEngine(system))
        source.merge("sbx-fork-2")
        self.assertEqual(system.calls[0][1], SandboxId("sbx-fork-2"))

    def test_engine_without_merge_support_raises(self) -> None:
        class _BareEngine:
            system = object()

            def _register_sandbox(self, sandbox) -> None:
                pass

        sandbox = Sandbox.connect(SandboxId("sbx-bare"), engine=_BareEngine())
        with self.assertRaises(NotImplementedError):
            sandbox.merge("sbx-fork")


if __name__ == "__main__":
    unittest.main()
