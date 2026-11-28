"""Unit tests for C1 changeset extraction: diff-output parsers (fixtures
pinned from real VM output), provider command contracts via capturing
runners, fork-point snapshot planting, CrabSystem gate semantics and
fork-point resolution, and the SDK plumbing. Host-runnable — no runc."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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
    RuncRuntime,
    RuncRuntimePaths,
    SandboxId,
    SandboxSnapshot,
    SchedulerConfig,
    StorageConfig,
)
from crab.ids import CheckpointId
from crab.journal import ActionJournal
from crab.models import ChangesetEntry, ChangesetResult, utc_now
from crab.runtime import BtrfsProvider, CommandRunner, ZfsProvider
from crab.runtime.base import CommandResult
from crab.runtime.btrfs_provider import parse_btrfs_receive_dump
from crab.runtime.overlay_provider import translate_overlay_changeset
from crab.runtime.zfs_provider import parse_zfs_diff
from crab.sandbox import Sandbox
from crab.scheduler import FaultToleranceCheckpointingPolicy


# ----------------------------------------------------------------------
# zfs diff -FH parser (fixture pinned from real VM output)
# ----------------------------------------------------------------------

_ZFS_FIXTURE = (
    "M\t/\t/tmp/probe-zfs/\n"
    "M\tF\t/tmp/probe-zfs/modify.txt\n"
    "-\tF\t/tmp/probe-zfs/delete.txt\n"
    "R\tF\t/tmp/probe-zfs/rename-old.txt\t/tmp/probe-zfs/rename-new.txt\n"
    "M\tF\t/tmp/probe-zfs/sp\\0040ace.txt\n"
    "+\tF\t/tmp/probe-zfs/added.txt\n"
    "+\tF\t/tmp/probe-zfs/newdir/nested.txt\n"
    "+\t/\t/tmp/probe-zfs/newdir\n"
)


class ZfsDiffParserTests(unittest.TestCase):
    def test_pinned_vm_fixture(self) -> None:
        entries = parse_zfs_diff(_ZFS_FIXTURE, mountpoint="/tmp/probe-zfs")
        self.assertEqual(
            [(e.change, e.path) for e in entries],
            [
                ("added", "/added.txt"),
                ("removed", "/delete.txt"),
                ("modified", "/modify.txt"),
                ("added", "/newdir"),
                ("added", "/newdir/nested.txt"),
                ("renamed", "/rename-new.txt"),
                ("modified", "/sp ace.txt"),
            ],
        )
        renamed = entries[5]
        self.assertEqual(renamed.renamed_from, "/rename-old.txt")
        self.assertIsNone(entries[0].renamed_from)

    def test_root_mtime_churn_and_foreign_paths_are_dropped(self) -> None:
        stdout = (
            "M\t/\t/tmp/probe-zfs/\n"
            "M\tF\t/somewhere/else/file.txt\n"
        )
        self.assertEqual(parse_zfs_diff(stdout, mountpoint="/tmp/probe-zfs"), [])

    def test_precedence_collapses_duplicate_paths(self) -> None:
        stdout = (
            "M\tF\t/mnt/a.txt\n"
            "-\tF\t/mnt/a.txt\n"
            "M\tF\t/mnt/b.txt\n"
        )
        entries = parse_zfs_diff(stdout, mountpoint="/mnt")
        self.assertEqual([(e.change, e.path) for e in entries], [("removed", "/a.txt"), ("modified", "/b.txt")])

    def test_malformed_and_blank_lines_are_skipped(self) -> None:
        stdout = "\nnot-a-diff-line\nM\tF\n?\tF\t/mnt/x\n"
        self.assertEqual(parse_zfs_diff(stdout, mountpoint="/mnt"), [])

    def test_to_json_round_trip_shape(self) -> None:
        entries = parse_zfs_diff("R\tF\t/mnt/old\t/mnt/new\n", mountpoint="/mnt")
        self.assertEqual(
            entries[0].to_json(),
            {"path": "/new", "change": "renamed", "renamed_from": "/old"},
        )

    def test_inode_replacement_folds_to_modified(self) -> None:
        # sed -i / editor saves write a temp file and rename it over the
        # target: zfs diff reports the old inode removed plus a new one
        # added at the same path (raw output pinned from the VM). The
        # path survived with new content — that is a modification, and
        # the btrfs parser already folds the same shape to modified.
        stdout = (
            "M\t/\t/tmp/zsed/probe\n"
            "+\tF\t/tmp/zsed/probe/doc.txt\n"
            "-\tF\t/tmp/zsed/probe/doc.txt\n"
        )
        entries = parse_zfs_diff(stdout, mountpoint="/tmp/zsed")
        self.assertEqual(
            [entry.to_json() for entry in entries],
            [
                {"path": "/probe", "change": "modified"},
                {"path": "/probe/doc.txt", "change": "modified"},
            ],
        )

    def test_rename_onto_existing_path_keeps_rename_identity(self) -> None:
        for stdout in (
            "R\tF\t/mnt/a.txt\t/mnt/b.txt\n-\tF\t/mnt/b.txt\n",
            "-\tF\t/mnt/b.txt\nR\tF\t/mnt/a.txt\t/mnt/b.txt\n",
        ):
            entries = parse_zfs_diff(stdout, mountpoint="/mnt")
            self.assertEqual(
                [entry.to_json() for entry in entries],
                [{"path": "/b.txt", "change": "renamed", "renamed_from": "/a.txt"}],
                msg=f"order variant failed: {stdout!r}",
            )


# ----------------------------------------------------------------------
# btrfs receive --dump parser (fixture pinned from real VM output)
# ----------------------------------------------------------------------

_BTRFS_SNAPSHOT_NAME = "sbx-1@changeset-deadbeef"
_BTRFS_PREFIX = f"./{_BTRFS_SNAPSHOT_NAME}"
_BTRFS_FIXTURE = (
    f"snapshot {_BTRFS_PREFIX} uuid=8b7f transid=160 parent_uuid=6c5e parent_transid=151\n"
    f"update_extent {_BTRFS_PREFIX}/modify.txt offset=0 len=3\n"
    f"utimes {_BTRFS_PREFIX}/modify.txt atime=2026 mtime=2026 ctime=2026\n"
    f"unlink {_BTRFS_PREFIX}/delete.txt\n"
    f"link {_BTRFS_PREFIX}/rename-new.txt dest=rename-old.txt\n"
    f"unlink {_BTRFS_PREFIX}/rename-old.txt\n"
    f"utimes {_BTRFS_PREFIX}/ atime=2026 mtime=2026 ctime=2026\n"
    f"update_extent {_BTRFS_PREFIX}/sp\\ ace.txt offset=0 len=6\n"
    f"truncate {_BTRFS_PREFIX}/sp\\ ace.txt size=6\n"
    f"mkfile {_BTRFS_PREFIX}/o262-151-0\n"
    f"rename {_BTRFS_PREFIX}/o262-151-0 dest={_BTRFS_PREFIX}/added.txt\n"
    f"chown {_BTRFS_PREFIX}/added.txt gid=0 uid=0\n"
    f"chmod {_BTRFS_PREFIX}/added.txt mode=644\n"
    f"mkdir {_BTRFS_PREFIX}/o263-151-0\n"
    f"rename {_BTRFS_PREFIX}/o263-151-0 dest={_BTRFS_PREFIX}/newdir\n"
    f"mkfile {_BTRFS_PREFIX}/o264-151-0\n"
    f"rename {_BTRFS_PREFIX}/o264-151-0 dest={_BTRFS_PREFIX}/newdir/nested.txt\n"
    f"utimes {_BTRFS_PREFIX}/newdir atime=2026 mtime=2026 ctime=2026\n"
    f"utimes {_BTRFS_PREFIX}/ atime=2026 mtime=2026 ctime=2026\n"
)


class BtrfsReceiveDumpParserTests(unittest.TestCase):
    def test_pinned_vm_fixture(self) -> None:
        entries = parse_btrfs_receive_dump(_BTRFS_FIXTURE, snapshot_name=_BTRFS_SNAPSHOT_NAME)
        self.assertEqual(
            [(e.change, e.path) for e in entries],
            [
                ("added", "/added.txt"),
                ("removed", "/delete.txt"),
                ("modified", "/modify.txt"),
                ("added", "/newdir"),
                ("added", "/newdir/nested.txt"),
                ("renamed", "/rename-new.txt"),
                ("modified", "/sp ace.txt"),
            ],
        )
        self.assertEqual(entries[5].renamed_from, "/rename-old.txt")

    def test_transient_file_created_then_deleted_is_absent(self) -> None:
        stdout = (
            "snapshot ./s@changeset-x uuid=1 transid=2 parent_uuid=3 parent_transid=4\n"
            "mkfile ./s@changeset-x/o1-1-0\n"
            "rename ./s@changeset-x/o1-1-0 dest=./s@changeset-x/tmp.txt\n"
            "unlink ./s@changeset-x/tmp.txt\n"
        )
        self.assertEqual(parse_btrfs_receive_dump(stdout, snapshot_name="s@changeset-x"), [])

    def test_atime_only_utimes_is_not_a_modification(self) -> None:
        """`btrfs send` emits utimes for inodes the sandbox merely read or
        executed. Those are not changes, and left in they collide with the
        same image paths on the other side of a merge. Timestamps pinned
        from real VM dump output."""
        stdout = (
            "utimes ./s@changeset-x/usr/bin/mv atime=2026-08-20T10:41:33+0000 "
            "mtime=2026-08-20T10:41:32+0000 ctime=2026-08-20T10:41:32+0000\n"
        )
        self.assertEqual(parse_btrfs_receive_dump(stdout, snapshot_name="s@changeset-x"), [])

    def test_touch_is_still_a_modification(self) -> None:
        # `touch` sets atime and mtime to now, so ctime is not older than
        # atime — the read-noise rule must not swallow it.
        stdout = (
            "utimes ./s@changeset-x/probe/g.txt atime=2026-08-20T10:41:33+0000 "
            "mtime=2026-08-20T10:41:33+0000 ctime=2026-08-20T10:41:33+0000\n"
        )
        entries = parse_btrfs_receive_dump(stdout, snapshot_name="s@changeset-x")
        self.assertEqual([(e.change, e.path) for e in entries], [("modified", "/probe/g.txt")])

    def test_written_then_read_file_survives_via_its_extent_line(self) -> None:
        # Content change plus a later read: the utimes line looks like
        # read-noise on its own, but update_extent already reported it.
        stdout = (
            "update_extent ./s@changeset-x/probe/h.txt offset=0 len=4\n"
            "utimes ./s@changeset-x/probe/h.txt atime=2026-08-20T10:41:40+0000 "
            "mtime=2026-08-20T10:41:33+0000 ctime=2026-08-20T10:41:33+0000\n"
        )
        entries = parse_btrfs_receive_dump(stdout, snapshot_name="s@changeset-x")
        self.assertEqual([(e.change, e.path) for e in entries], [("modified", "/probe/h.txt")])

    def test_unparseable_utimes_stays_a_modification(self) -> None:
        # Conservative fallback: an unexpected timestamp format must not
        # silently drop a path from the changeset.
        stdout = "utimes ./s@changeset-x/probe/x.txt atime=2026 mtime=2026 ctime=2026\n"
        entries = parse_btrfs_receive_dump(stdout, snapshot_name="s@changeset-x")
        self.assertEqual([(e.change, e.path) for e in entries], [("modified", "/probe/x.txt")])

    def test_rmdir_of_preexisting_directory_is_removed(self) -> None:
        stdout = "rmdir ./s@changeset-x/olddir\n"
        entries = parse_btrfs_receive_dump(stdout, snapshot_name="s@changeset-x")
        self.assertEqual([(e.change, e.path) for e in entries], [("removed", "/olddir")])

    def test_hard_link_without_unlink_is_added(self) -> None:
        stdout = "link ./s@changeset-x/extra-name.txt dest=kept.txt\n"
        entries = parse_btrfs_receive_dump(stdout, snapshot_name="s@changeset-x")
        self.assertEqual([(e.change, e.path) for e in entries], [("added", "/extra-name.txt")])

    def test_replacing_a_deleted_path_folds_to_modified(self) -> None:
        stdout = (
            "unlink ./s@changeset-x/config.txt\n"
            "mkfile ./s@changeset-x/o9-9-0\n"
            "rename ./s@changeset-x/o9-9-0 dest=./s@changeset-x/config.txt\n"
        )
        entries = parse_btrfs_receive_dump(stdout, snapshot_name="s@changeset-x")
        self.assertEqual([(e.change, e.path) for e in entries], [("modified", "/config.txt")])

    def test_modified_then_moved_reports_single_rename(self) -> None:
        stdout = (
            "update_extent ./s@changeset-x/a.txt offset=0 len=3\n"
            "link ./s@changeset-x/b.txt dest=a.txt\n"
            "unlink ./s@changeset-x/a.txt\n"
        )
        entries = parse_btrfs_receive_dump(stdout, snapshot_name="s@changeset-x")
        self.assertEqual([(e.change, e.path) for e in entries], [("renamed", "/b.txt")])
        self.assertEqual(entries[0].renamed_from, "/a.txt")

    def test_metadata_commands_mark_preexisting_paths_modified(self) -> None:
        stdout = (
            "chmod ./s@changeset-x/script.sh mode=755\n"
            "set_xattr ./s@changeset-x/tagged.txt name=user.k data=v len=1\n"
        )
        entries = parse_btrfs_receive_dump(stdout, snapshot_name="s@changeset-x")
        self.assertEqual(
            [(e.change, e.path) for e in entries],
            [("modified", "/script.sh"), ("modified", "/tagged.txt")],
        )


# ----------------------------------------------------------------------
# Provider command contracts (capturing executor, PR-3 discipline)
# ----------------------------------------------------------------------


class _RecordingExecutor:
    """Stand-in for RuncRuntime._run_command / _run_status that records
    command lines, serves primed responses, and honors check=True like
    the real pipeline (raises on non-zero)."""

    def __init__(self) -> None:
        self.commands: list[tuple[str, ...]] = []
        # command tuple -> (returncode, stdout, stderr)
        self.responses: dict[tuple[str, ...], tuple[int, str, str]] = {}

    def run_command(
        self,
        command,
        *,
        operation,
        sandbox_id=None,
        checkpoint_id=None,
        cwd=None,
        check=True,
        expected_error_substrings=(),
        metadata=None,
        timeout_seconds=None,
    ) -> CommandResult:
        _ = (operation, sandbox_id, checkpoint_id, cwd, expected_error_substrings, metadata, timeout_seconds)
        key = tuple(command)
        self.commands.append(key)
        returncode, stdout, stderr = self.responses.get(key, (0, "", ""))
        if check and returncode != 0:
            raise RuntimeError(f"command failed ({returncode}): {' '.join(key)}")
        return CommandResult(command=key, returncode=returncode, stdout=stdout, stderr=stderr)

    def run_status(self, command, *, operation, sandbox_id, checkpoint_id=None, metadata=None):
        raise AssertionError("changeset flows must not use the status pipeline")


class ZfsProviderChangesetTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="crab_changeset_zfs_")
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.executor = _RecordingExecutor()
        self.provider = ZfsProvider(
            dataset_prefix="pool/crab",
            runtime_name="runc",
            run_command=self.executor.run_command,
            run_status=self.executor.run_status,
            dataset_resolver=lambda sid: f"pool/crab/{sid}",
            rootfs_resolver=lambda sid: self.root / "bundles" / str(sid) / "rootfs",
        )
        self.sid = SandboxId("sbx-1")
        self.ckpt = CheckpointId("ckpt-1")
        self.snapshot = "pool/crab/sbx-1@ckpt-1"
        self.exists_cmd = ("zfs", "list", "-H", "-o", "name", "-t", "snapshot", self.snapshot)
        self.diff_cmd = ("zfs", "diff", "-FH", self.snapshot, "pool/crab/sbx-1")

    def test_missing_base_snapshot_raises_file_not_found(self) -> None:
        self.executor.responses[self.exists_cmd] = (1, "", "does not exist")
        with self.assertRaises(FileNotFoundError):
            self.provider.changeset_since(self.sid, self.ckpt)
        self.assertNotIn(self.diff_cmd, self.executor.commands)

    def test_diff_command_line_and_parsed_entries(self) -> None:
        mountpoint = str(self.root / "bundles" / "sbx-1" / "rootfs")
        self.executor.responses[self.diff_cmd] = (
            0,
            f"M\t/\t{mountpoint}/\n+\tF\t{mountpoint}/hello.txt\n-\tF\t{mountpoint}/gone.txt\n",
            "",
        )
        entries = self.provider.changeset_since(self.sid, self.ckpt)
        self.assertEqual(self.executor.commands, [self.exists_cmd, self.diff_cmd])
        self.assertEqual(
            entries,
            [
                ChangesetEntry(path="/gone.txt", change="removed"),
                ChangesetEntry(path="/hello.txt", change="added"),
            ],
        )

    def test_clone_plants_fork_point_snapshot_on_fork_dataset(self) -> None:
        target_rootfs = self.root / "bundles" / "sbx-fork" / "rootfs"
        self.provider.clone_filesystem_snapshot(
            self.sid,
            self.ckpt,
            SandboxId("sbx-fork"),
            target_rootfs_path=target_rootfs,
        )
        clone_cmd = (
            "zfs", "clone", "-o", f"mountpoint={target_rootfs}",
            self.snapshot, "pool/crab/sbx-fork",
        )
        fork_point_cmd = ("zfs", "snapshot", "pool/crab/sbx-fork@ckpt-1")
        self.assertIn(clone_cmd, self.executor.commands)
        self.assertIn(fork_point_cmd, self.executor.commands)
        self.assertGreater(
            self.executor.commands.index(fork_point_cmd),
            self.executor.commands.index(clone_cmd),
        )


class BtrfsProviderChangesetTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="crab_changeset_btrfs_")
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.executor = _RecordingExecutor()
        self.provider = BtrfsProvider(
            btrfs_root=self.root,
            runtime_name="runc",
            run_command=self.executor.run_command,
            run_status=self.executor.run_status,
            dataset_resolver=lambda sid: str(self.root / "sandboxes" / str(sid)),
            rootfs_resolver=lambda sid: self.root / "bundles" / str(sid) / "rootfs",
        )
        self.sid = SandboxId("sbx-1")
        self.ckpt = CheckpointId("ckpt-1")
        self.dataset = str(self.root / "sandboxes" / "sbx-1")
        self.base_snapshot = f"{self.dataset}@ckpt-1"
        self.show_cmd = ("btrfs", "subvolume", "show", self.base_snapshot)
        self.tmp_snapshot = f"{self.dataset}@changeset-deadbeef"
        self.pipeline_cmd = (
            "sh",
            "-c",
            f"btrfs send --no-data -p {self.base_snapshot} {self.tmp_snapshot} | btrfs receive --dump",
        )

    def _pin_uuid(self):
        return mock.patch(
            "crab.runtime.btrfs_provider.uuid.uuid4",
            return_value=mock.Mock(hex="deadbeefcafebabe"),
        )

    def test_missing_base_snapshot_raises_file_not_found(self) -> None:
        self.executor.responses[self.show_cmd] = (1, "", "ERROR: not a subvolume")
        with self.assertRaises(FileNotFoundError):
            self.provider.changeset_since(self.sid, self.ckpt)
        self.assertEqual(self.executor.commands, [self.show_cmd])

    def test_diff_pipeline_lifecycle_and_parsed_entries(self) -> None:
        prefix = f"./{Path(self.tmp_snapshot).name}"
        self.executor.responses[self.pipeline_cmd] = (
            0,
            (
                f"snapshot {prefix} uuid=1 transid=2 parent_uuid=3 parent_transid=4\n"
                f"mkfile {prefix}/o1-1-0\n"
                f"rename {prefix}/o1-1-0 dest={prefix}/hello.txt\n"
                f"unlink {prefix}/gone.txt\n"
            ),
            "",
        )
        with self._pin_uuid():
            entries = self.provider.changeset_since(self.sid, self.ckpt)
        self.assertEqual(
            self.executor.commands,
            [
                self.show_cmd,
                ("btrfs", "subvolume", "snapshot", "-r", self.dataset, self.tmp_snapshot),
                self.pipeline_cmd,
                ("btrfs", "subvolume", "delete", self.tmp_snapshot),
            ],
        )
        self.assertEqual(
            entries,
            [
                ChangesetEntry(path="/gone.txt", change="removed"),
                ChangesetEntry(path="/hello.txt", change="added"),
            ],
        )

    def test_transient_snapshot_deleted_even_when_send_fails(self) -> None:
        self.executor.responses[self.pipeline_cmd] = (1, "", "ERROR: empty stream")
        with self._pin_uuid():
            with self.assertRaises(RuntimeError):
                self.provider.changeset_since(self.sid, self.ckpt)
        self.assertEqual(
            self.executor.commands[-1],
            ("btrfs", "subvolume", "delete", self.tmp_snapshot),
        )

    def test_stale_transient_snapshots_are_swept_first(self) -> None:
        stale = Path(f"{self.dataset}@changeset-00000000")
        stale.mkdir(parents=True)
        with self._pin_uuid():
            self.provider.changeset_since(self.sid, self.ckpt)
        self.assertEqual(
            self.executor.commands[1],
            ("btrfs", "subvolume", "delete", str(stale)),
        )
        # Sweep failures are tolerated (check=False): rc!=0 must not abort.
        self.executor.commands.clear()
        self.executor.responses[("btrfs", "subvolume", "delete", str(stale))] = (1, "", "busy")
        with self._pin_uuid():
            self.provider.changeset_since(self.sid, self.ckpt)
        self.assertIn(self.pipeline_cmd, self.executor.commands)

    def test_clone_plants_readonly_fork_point_snapshot_before_bind(self) -> None:
        target_rootfs = self.root / "bundles" / "sbx-fork" / "rootfs"
        target_dataset = str(self.root / "sandboxes" / "sbx-fork")
        self.provider.clone_filesystem_snapshot(
            self.sid,
            self.ckpt,
            SandboxId("sbx-fork"),
            target_rootfs_path=target_rootfs,
        )
        commands = self.executor.commands
        clone_cmd = ("btrfs", "subvolume", "snapshot", self.base_snapshot, target_dataset)
        fork_point_cmd = ("btrfs", "subvolume", "snapshot", "-r", target_dataset, f"{target_dataset}@ckpt-1")
        self.assertIn(clone_cmd, commands)
        self.assertIn(fork_point_cmd, commands)
        self.assertGreater(commands.index(fork_point_cmd), commands.index(clone_cmd))
        self.assertEqual(commands[-1], ("mount", "--bind", target_dataset, str(target_rootfs)))


# ----------------------------------------------------------------------
# CrabSystem: gate semantics, fork-point resolution, journal marker
# ----------------------------------------------------------------------


class FakeCommandRunner(CommandRunner):
    def __init__(self) -> None:
        self.commands: list[tuple[str, ...]] = []
        self.responses: dict[tuple[str, ...], tuple[int, str, str]] = {}

    def run(self, command: list[str], *, cwd: Path | None = None, timeout_seconds: float | None = None):
        _ = (cwd, timeout_seconds)
        key = tuple(command)
        self.commands.append(key)
        returncode, stdout, stderr = self.responses.get(key, (0, "", ""))
        return type(
            "Result",
            (),
            {"command": key, "returncode": returncode, "stdout": stdout, "stderr": stderr},
        )()


class SystemChangesetTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="crab_changeset_system_")
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.runner = FakeCommandRunner()
        self.telemetry = InMemoryTelemetrySink()
        self.inspector = EBPFSandboxInspector()
        self.journal = ActionJournal(self.root / "storage" / "journal")
        self.runtime = RuncRuntime(
            command_runner=self.runner,
            paths=RuncRuntimePaths(
                state_root=self.root / "runtime-state",
                bundle_root=self.root / "bundles",
                checkpoint_root=self.root / "checkpoints",
                metadata_root=self.root / "sandbox-metadata",
                zfs_dataset_prefix="pool/crab",
            ),
            action_recorder=self.journal,
        )
        storage = LocalCheckpointManager(
            StorageConfig(root_dir=self.root / "storage"),
            destroy_filesystem_ref=self.runtime.destroy_filesystem_ref,
        )
        self.executor = CRExecutor(
            ExecutorConfig(max_workers=1),
            DefaultCWorker(
                AdapterProcessCWorker(self.runtime),
                AdapterFileSystemCWorker(self.runtime),
                storage,
                self.runtime,
            ),
            DefaultRWorker(
                AdapterProcessRWorker(self.runtime),
                AdapterFileSystemRWorker(self.runtime),
                storage,
            ),
            self.telemetry,
        )
        self.addCleanup(self.executor.shutdown)
        scheduler_cfg = SchedulerConfig(
            min_checkpoint_interval_seconds=0.0,
            force_checkpoint_after_seconds=0.0,
            require_change_signal=True,
        )
        scheduler = CRScheduler(
            scheduler_cfg,
            self.inspector,
            self.runtime,
            InMemorySchedulerStateStore(),
            self.telemetry,
            FaultToleranceCheckpointingPolicy(scheduler_cfg),
        )
        self.system = CrabSystem(
            scheduler=scheduler,
            executor=self.executor,
            storage=storage,
            inspector=self.inspector,
            runtime=self.runtime,
            telemetry=self.telemetry,
            journal=self.journal,
        )
        self.sid = SandboxId("sbx-cs")
        self._launch(self.sid)
        self._mark(self.sid, filesystem_changed=True)

    def _launch(self, sandbox_id: SandboxId) -> None:
        bundle_dir = self.root / "bundles" / str(sandbox_id)
        bundle_dir.mkdir(parents=True)
        (bundle_dir / "config.json").write_text(json.dumps({"process": {}}), encoding="utf-8")
        self.runtime.launch("runc", {"sandbox_id": str(sandbox_id), "bundle_path": str(bundle_dir)})

    def _mark(self, sandbox_id: SandboxId, *, filesystem_changed: bool) -> None:
        self.inspector.upsert_snapshot(
            SandboxSnapshot(
                sandbox_id=sandbox_id,
                runtime_name="runc",
                is_running=True,
                process_changed=True,
                filesystem_changed=filesystem_changed,
                observed_at=utc_now(),
            )
        )

    def _zfs_diff_commands(self) -> list[tuple[str, ...]]:
        return [cmd for cmd in self.runner.commands if cmd[:2] == ("zfs", "diff")]

    def _changeset_journal_markers(self) -> list[dict]:
        return [
            record.payload
            for record in self.journal.entries(self.sid, kind="lifecycle")
            if record.payload.get("event") == "changeset"
        ]

    def test_gate_skips_diff_when_clean_and_base_is_latest(self) -> None:
        checkpoint = self.system.checkpoint_once(self.sid, leave_running=True)
        self.assertEqual(checkpoint.status.value, "succeeded")
        self.runner.commands.clear()

        result = self.system.changeset_since(self.sid, checkpoint.checkpoint_id)

        self.assertIsInstance(result, ChangesetResult)
        self.assertTrue(result.skipped_by_gate)
        self.assertEqual(result.entries, ())
        self.assertEqual(result.base_checkpoint_id, checkpoint.checkpoint_id)
        self.assertEqual(self._zfs_diff_commands(), [])
        marker = self._changeset_journal_markers()[-1]["metadata"]
        self.assertTrue(marker["skipped_by_gate"])
        self.assertEqual(marker["entry_count"], 0)
        events = [attrs for name, attrs in self.telemetry.events if name == "changeset.computed"]
        self.assertTrue(events and events[-1]["skipped_by_gate"])

    def test_gate_misses_when_filesystem_changed(self) -> None:
        checkpoint = self.system.checkpoint_once(self.sid, leave_running=True)
        self._mark(self.sid, filesystem_changed=True)
        self.runner.commands.clear()

        result = self.system.changeset_since(self.sid, checkpoint.checkpoint_id)

        self.assertFalse(result.skipped_by_gate)
        expected_diff = (
            "zfs", "diff", "-FH",
            f"pool/crab/{self.sid}@{checkpoint.checkpoint_id}", f"pool/crab/{self.sid}",
        )
        self.assertEqual(self._zfs_diff_commands(), [expected_diff])

    def test_gate_misses_when_base_is_not_latest_filesystem_checkpoint(self) -> None:
        older = self.system.checkpoint_once(self.sid, leave_running=True)
        self._mark(self.sid, filesystem_changed=True)
        newer = self.system.checkpoint_once(self.sid, leave_running=True)
        self.assertEqual(newer.status.value, "succeeded")
        # Inspector is clean now, but the requested base is not the latest
        # filesystem boundary: the gate must not vouch for it.
        self.runner.commands.clear()

        result = self.system.changeset_since(self.sid, older.checkpoint_id)

        self.assertFalse(result.skipped_by_gate)
        self.assertEqual(len(self._zfs_diff_commands()), 1)

    def test_gate_can_be_disabled(self) -> None:
        checkpoint = self.system.checkpoint_once(self.sid, leave_running=True)
        self.runner.commands.clear()

        result = self.system.changeset_since(
            self.sid, checkpoint.checkpoint_id, use_inspector_gate=False
        )

        self.assertFalse(result.skipped_by_gate)
        self.assertEqual(len(self._zfs_diff_commands()), 1)

    def test_diff_entries_flow_through_result(self) -> None:
        checkpoint = self.system.checkpoint_once(self.sid, leave_running=True)
        self._mark(self.sid, filesystem_changed=True)
        mountpoint = self.runtime.rootfs_path_for(self.sid)
        diff_cmd = (
            "zfs", "diff", "-FH",
            f"pool/crab/{self.sid}@{checkpoint.checkpoint_id}", f"pool/crab/{self.sid}",
        )
        self.runner.responses[diff_cmd] = (0, f"+\tF\t{mountpoint}/new.txt\n", "")

        result = self.system.changeset_since(self.sid, checkpoint.checkpoint_id)

        self.assertEqual(result.entries, (ChangesetEntry(path="/new.txt", change="added"),))
        marker = self._changeset_journal_markers()[-1]["metadata"]
        self.assertEqual(marker["entry_count"], 1)
        self.assertFalse(marker["skipped_by_gate"])

    def test_fork_changeset_resolves_fork_point_marker(self) -> None:
        fork_id = SandboxId("sbx-cs-fork")
        self._launch(fork_id)
        self.journal.record_lifecycle(
            fork_id,
            "fork_created",
            metadata={"source_sandbox_id": str(self.sid), "checkpoint_id": "ckpt-forkpoint"},
        )

        result = self.system.fork_changeset(fork_id)

        self.assertEqual(result.base_checkpoint_id, CheckpointId("ckpt-forkpoint"))
        diff_cmd = (
            "zfs", "diff", "-FH",
            f"pool/crab/{fork_id}@ckpt-forkpoint", f"pool/crab/{fork_id}",
        )
        self.assertIn(diff_cmd, self.runner.commands)

    def test_fork_changeset_without_marker_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.system.fork_changeset(self.sid)

    def test_missing_base_snapshot_propagates_file_not_found(self) -> None:
        self.runner.responses[
            ("zfs", "list", "-H", "-o", "name", "-t", "snapshot", f"pool/crab/{self.sid}@ckpt-nope")
        ] = (1, "", "does not exist")
        with self.assertRaises(FileNotFoundError):
            self.system.changeset_since(self.sid, CheckpointId("ckpt-nope"))


# ----------------------------------------------------------------------
# SDK plumbing
# ----------------------------------------------------------------------


class SandboxChangesetTests(unittest.TestCase):
    class _FakeSystem:
        def __init__(self) -> None:
            self.changeset_calls: list = []
            self.fork_calls: list = []

        def changeset_since(self, sandbox_id, checkpoint_id, *, use_inspector_gate=True):
            self.changeset_calls.append((sandbox_id, checkpoint_id, use_inspector_gate))
            return ChangesetResult(
                sandbox_id=sandbox_id,
                base_checkpoint_id=checkpoint_id,
                entries=(ChangesetEntry(path="/new.txt", change="added"),),
            )

        def fork_changeset(self, sandbox_id, *, force=False):
            self.fork_calls.append((sandbox_id, force))
            return ChangesetResult(
                sandbox_id=sandbox_id,
                base_checkpoint_id=CheckpointId("ckpt-forkpoint"),
                entries=(
                    ChangesetEntry(path="/renamed.txt", change="renamed", renamed_from="/old.txt"),
                ),
            )

    class _FakeEngine:
        def __init__(self, system) -> None:
            self.system = system

        def _register_sandbox(self, sandbox) -> None:
            pass

    def _sandbox(self):
        system = self._FakeSystem()
        sandbox = Sandbox.connect(SandboxId("sbx-sdk"), engine=self._FakeEngine(system))
        return sandbox, system

    def test_explicit_since_calls_changeset_since(self) -> None:
        sandbox, system = self._sandbox()
        entries = sandbox.changeset(since="ckpt-9")
        self.assertEqual(system.changeset_calls, [(SandboxId("sbx-sdk"), CheckpointId("ckpt-9"), True)])
        self.assertEqual(entries, [{"path": "/new.txt", "change": "added"}])

    def test_default_since_resolves_fork_point(self) -> None:
        sandbox, system = self._sandbox()
        entries = sandbox.changeset()
        self.assertEqual(system.fork_calls, [(SandboxId("sbx-sdk"), False)])
        self.assertEqual(
            entries,
            [{"path": "/renamed.txt", "change": "renamed", "renamed_from": "/old.txt"}],
        )

    def test_engine_without_changeset_support_raises(self) -> None:
        class _BareEngine:
            system = object()

            def _register_sandbox(self, sandbox) -> None:
                pass

        sandbox = Sandbox.connect(SandboxId("sbx-bare"), engine=_BareEngine())
        with self.assertRaises(NotImplementedError):
            sandbox.changeset()
        with self.assertRaises(NotImplementedError):
            sandbox.changeset(since="ckpt-1")


class OverlayChangesetTranslationTests(unittest.TestCase):
    """A2 §5 translation pass: vol-level send-diff entries (raw upperdir
    deltas) decoded into container semantics via injectable probes.
    Host-runnable — probes are fakes; the real-probe wiring is covered
    by OverlayRealChangesetTests in the VM."""

    @staticmethod
    def _translate(entries, **overrides):
        probes = {
            "live_whiteout": lambda path: False,
            "base_whiteout": lambda path: False,
            "base_exists": lambda path: False,
            "live_exists": lambda path: False,
            "newly_opaque": lambda path: False,
            "base_children": lambda path: [],
            "live_children": lambda path: [],
        }
        probes.update(overrides)
        return translate_overlay_changeset(entries, **probes)

    @staticmethod
    def _shapes(entries):
        return [(entry.path, entry.change, entry.renamed_from) for entry in entries]

    def test_upper_prefix_strip_and_internal_paths_dropped(self) -> None:
        entries = [
            ChangesetEntry(path="/upper/a.txt", change="added"),
            ChangesetEntry(path="/work/#123", change="modified"),
            ChangesetEntry(path="/.crab-overlay.json", change="modified"),
            ChangesetEntry(path="/upper", change="modified"),
        ]
        self.assertEqual(
            self._shapes(self._translate(entries)),
            [("/a.txt", "added", None)],
        )

    def test_copy_up_of_lower_file_becomes_modified(self) -> None:
        # The diff reports a copy-up as `added` (the file is new in the
        # upper); container semantics say `modified` because the base
        # merged view already had it.
        entries = [ChangesetEntry(path="/upper/etc/issue", change="added")]
        self.assertEqual(
            self._shapes(self._translate(entries, base_exists=lambda p: p == "/etc/issue")),
            [("/etc/issue", "modified", None)],
        )

    def test_sed_i_composite_folds_to_single_modified(self) -> None:
        # sed -i on a lower file = copy-up + in-upper rename of the temp
        # file onto the target: the parser's placeholder folding already
        # yields one `added` entry for the destination; base existence
        # turns it into exactly one `modified` (the C1 pitfall, pinned).
        entries = [ChangesetEntry(path="/upper/etc/target.conf", change="added")]
        self.assertEqual(
            self._shapes(self._translate(entries, base_exists=lambda p: p == "/etc/target.conf")),
            [("/etc/target.conf", "modified", None)],
        )

    def test_new_upper_file_stays_added(self) -> None:
        entries = [ChangesetEntry(path="/upper/fresh.txt", change="added")]
        self.assertEqual(
            self._shapes(self._translate(entries)),
            [("/fresh.txt", "added", None)],
        )

    def test_whiteout_means_removed(self) -> None:
        # Deleting a lower-visible file plants a char 0:0 device in the
        # upper; the diff calls that `added`.
        entries = [ChangesetEntry(path="/upper/gone.txt", change="added")]
        self.assertEqual(
            self._shapes(self._translate(entries, live_whiteout=lambda p: p == "/gone.txt")),
            [("/gone.txt", "removed", None)],
        )

    def test_whiteout_replace_recreation_classified_by_base_view(self) -> None:
        # A file deleted before the base checkpoint (whiteout in the
        # base upper) and recreated since: the raw kind is added or
        # modified depending on stream folding, but the base *merged*
        # view (where the whiteout hides the file) decides `added`.
        entries = [ChangesetEntry(path="/upper/reborn.txt", change="modified")]
        self.assertEqual(
            self._shapes(self._translate(entries)),
            [("/reborn.txt", "added", None)],
        )

    def test_vanished_whiteout_reappearance(self) -> None:
        # The deletion marker disappeared without replacement: the lower
        # shows through again — `added` when the live merged view has
        # the path, dropped otherwise.
        entries = [ChangesetEntry(path="/upper/back.txt", change="removed")]
        self.assertEqual(
            self._shapes(
                self._translate(
                    entries,
                    base_whiteout=lambda p: p == "/back.txt",
                    live_exists=lambda p: p == "/back.txt",
                )
            ),
            [("/back.txt", "added", None)],
        )
        self.assertEqual(
            self._shapes(self._translate(entries, base_whiteout=lambda p: p == "/back.txt")),
            [],
        )

    def test_upper_only_delete_stays_removed(self) -> None:
        entries = [ChangesetEntry(path="/upper/scratch.txt", change="removed")]
        self.assertEqual(
            self._shapes(self._translate(entries)),
            [("/scratch.txt", "removed", None)],
        )

    def test_in_upper_rename_keeps_attribution(self) -> None:
        entries = [
            ChangesetEntry(path="/upper/new.txt", change="renamed", renamed_from="/upper/old.txt"),
        ]
        self.assertEqual(
            self._shapes(self._translate(entries)),
            [("/new.txt", "renamed", "/old.txt")],
        )

    def test_rename_from_outside_upper_degrades(self) -> None:
        entries = [
            ChangesetEntry(path="/upper/new.txt", change="renamed", renamed_from="/work/#42"),
        ]
        self.assertEqual(
            self._shapes(self._translate(entries)),
            [("/new.txt", "added", None)],
        )

    def test_opaque_dir_emits_masked_children_removed_recursively(self) -> None:
        # rm -rf /data && mkdir /data && touch /data/z.txt: the upper
        # diff only shows the (opaque) dir and the recreated child; the
        # masked lower children come from the base merged view, minus
        # what the live merged view re-created, descending subtrees.
        entries = [
            ChangesetEntry(path="/upper/data", change="added"),
            ChangesetEntry(path="/upper/data/z.txt", change="added"),
        ]
        base_tree = {
            "/data": ["a.txt", "sub"],
            "/data/sub": ["x.txt"],
        }
        result = self._translate(
            entries,
            base_exists=lambda p: p in {"/data", "/data/a.txt", "/data/sub", "/data/sub/x.txt"},
            newly_opaque=lambda p: p == "/data",
            base_children=lambda p: base_tree.get(p, []),
            live_children=lambda p: ["z.txt"] if p == "/data" else [],
        )
        self.assertEqual(
            self._shapes(result),
            [
                ("/data", "modified", None),
                ("/data/a.txt", "removed", None),
                ("/data/sub", "removed", None),
                ("/data/sub/x.txt", "removed", None),
                ("/data/z.txt", "added", None),
            ],
        )

    def test_precedence_folds_duplicate_paths(self) -> None:
        # A whiteouted path reported through two raw kinds folds to one
        # `removed` entry.
        entries = [
            ChangesetEntry(path="/upper/x", change="modified"),
            ChangesetEntry(path="/upper/x", change="added"),
        ]
        self.assertEqual(
            self._shapes(self._translate(entries, live_whiteout=lambda p: p == "/x")),
            [("/x", "removed", None)],
        )


if __name__ == "__main__":
    unittest.main()
