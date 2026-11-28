from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from crab.ids import CheckpointId, SandboxId
from crab.models import RuntimeOperationStatus
from crab.runtime import OverlayProvider
from crab.runtime.base import CommandResult

_RW_OPTIONS = "redirect_dir=off,metacopy=off"


class _RecordingExecutor:
    """Fake command pipeline standing in for RuncRuntime._run_command /
    _run_status (same shape as the btrfs provider tests). Providers are
    units: they only need these callables and the resolver pair."""

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
        _ = (operation, sandbox_id, checkpoint_id, cwd, check, expected_error_substrings, metadata, timeout_seconds)
        key = tuple(command)
        self.commands.append(key)
        returncode, stdout, stderr = self.responses.get(key, (0, "", ""))
        return CommandResult(command=key, returncode=returncode, stdout=stdout, stderr=stderr)

    def run_status(
        self,
        command,
        *,
        operation,
        sandbox_id,
        checkpoint_id=None,
        metadata=None,
    ) -> RuntimeOperationStatus:
        result = self.run_command(
            command,
            operation=operation,
            sandbox_id=sandbox_id,
            checkpoint_id=checkpoint_id,
            metadata=metadata,
        )
        merged = dict(metadata or {})
        merged["stdout"] = result.stdout.strip()
        merged["stderr"] = result.stderr.strip()
        return RuntimeOperationStatus(executed=True, reason="command_executed", command=result.command, metadata=merged)


def _make_provider(root: Path, executor: _RecordingExecutor, **kwargs) -> OverlayProvider:
    return OverlayProvider(
        overlay_root=root,
        runtime_name="runc",
        run_command=executor.run_command,
        run_status=executor.run_status,
        dataset_resolver=lambda sid: str(root / "sandboxes" / str(sid)),
        rootfs_resolver=lambda sid: root / "bundles" / str(sid) / "rootfs",
        **kwargs,
    )


def _write_marker(dataset: str | Path, lowerdir: str) -> None:
    vol = Path(dataset)
    vol.mkdir(parents=True, exist_ok=True)
    (vol / ".crab-overlay.json").write_text(
        json.dumps({"lowerdir": lowerdir, "version": 1}, sort_keys=True) + "\n",
        encoding="utf-8",
    )


class OverlayProviderCommandMappingTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="crab_overlay_provider_")
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.executor = _RecordingExecutor()
        self.provider = _make_provider(self.root, self.executor)
        self.sid = SandboxId("sbx-1")
        self.dataset = str(self.root / "sandboxes" / "sbx-1")
        self.rootfs = self.root / "bundles" / "sbx-1" / "rootfs"
        self.empty_lower = str(self.root / "empty-lower")

    def _rw_options(self, dataset: str, lowerdir: str) -> str:
        return (
            f"lowerdir={lowerdir},upperdir={dataset}/upper,"
            f"workdir={dataset}/work,{_RW_OPTIONS}"
        )

    def test_create_dataset_scaffolds_vol_and_mounts_overlay(self) -> None:
        self.provider.create_dataset(self.dataset, self.rootfs, operation="sandbox.zfs_create", sandbox_id=self.sid)

        self.assertEqual(
            self.executor.commands,
            [
                ("btrfs", "subvolume", "create", self.dataset),
                (
                    "mount",
                    "-t",
                    "overlay",
                    "overlay",
                    "-o",
                    self._rw_options(self.dataset, self.empty_lower),
                    str(self.rootfs),
                ),
            ],
        )
        vol = Path(self.dataset)
        self.assertTrue((vol / "upper").is_dir())
        self.assertTrue((vol / "work").is_dir())
        marker = json.loads((vol / ".crab-overlay.json").read_text(encoding="utf-8"))
        self.assertEqual(marker, {"lowerdir": self.empty_lower, "version": 1})

    def test_create_dataset_shared_cache_stays_plain(self) -> None:
        shared_dataset, mountpoint = self.provider.shared_rootfs_details("img", persist_across_runs=False)

        self.provider.create_dataset(shared_dataset, mountpoint, operation="sandbox.zfs_create_shared_rootfs")

        # A plain subvolume: it *is* the future lowerdir content. No
        # scaffold, no overlay mount.
        self.assertEqual(self.executor.commands, [("btrfs", "subvolume", "create", shared_dataset)])
        self.assertFalse((Path(shared_dataset) / ".crab-overlay.json").exists())

    def test_clone_shared_base_references_lower_and_mounts(self) -> None:
        shared_dataset = str(self.root / "shared" / "run" / "img")
        shared_snapshot = f"{shared_dataset}@base"

        self.provider.clone_shared_base(shared_dataset, shared_snapshot, self.dataset, self.rootfs, sandbox_id=self.sid)

        self.assertEqual(
            self.executor.commands,
            [
                ("btrfs", "subvolume", "create", self.dataset),
                (
                    "mount",
                    "-t",
                    "overlay",
                    "overlay",
                    "-o",
                    self._rw_options(self.dataset, shared_snapshot),
                    str(self.rootfs),
                ),
            ],
        )
        marker = json.loads((Path(self.dataset) / ".crab-overlay.json").read_text(encoding="utf-8"))
        self.assertEqual(marker["lowerdir"], shared_snapshot)

    def test_lowerdir_metachar_validation_fails_launch(self) -> None:
        with self.assertRaisesRegex(ValueError, "metacharacters"):
            self.provider.clone_shared_base(
                "shared", "bad:snapshot", self.dataset, self.rootfs, sandbox_id=self.sid
            )

    def test_checkpoint_filesystem_records_overlay_fs_ref_and_lowerdir(self) -> None:
        self.provider.create_dataset(self.dataset, self.rootfs, operation="sandbox.zfs_create", sandbox_id=self.sid)
        self.executor.commands.clear()

        status = self.provider.checkpoint_filesystem(self.sid, CheckpointId("ckpt-1"))

        snapshot = f"{self.dataset}@ckpt-1"
        self.assertEqual(
            self.executor.commands[0],
            ("btrfs", "subvolume", "snapshot", "-r", self.dataset, snapshot),
        )
        self.assertEqual(status.metadata["fs_ref"], f"overlay:{snapshot}")
        self.assertEqual(status.metadata["snapshot"], snapshot)
        self.assertEqual(status.metadata["lowerdir"], self.empty_lower)
        self.assertEqual(status.metadata["checkpoint_scope"], "filesystem_only")
        # qgroups off by default: no stats command, unknown byte counts.
        self.assertIsNone(status.metadata["filesystem_checkpoint_written_bytes"])
        self.assertNotIn(("btrfs", "qgroup", "show", "--raw", "-f", snapshot), self.executor.commands)

    def test_restore_filesystem_trash_swaps_and_remounts_overlay(self) -> None:
        self.provider.create_dataset(self.dataset, self.rootfs, operation="sandbox.zfs_create", sandbox_id=self.sid)
        self.executor.commands.clear()

        status = self.provider.restore_filesystem(self.sid, CheckpointId("ckpt-1"))

        snapshot = f"{self.dataset}@ckpt-1"
        commands = self.executor.commands
        self.assertEqual(commands[0], ("umount", str(self.rootfs)))
        self.assertEqual(commands[1], ("btrfs", "subvolume", "show", self.dataset))
        mv = commands[2]
        self.assertEqual(mv[0], "mv")
        self.assertEqual(mv[1], self.dataset)
        self.assertTrue(mv[2].startswith(f"{self.dataset}.trash-"))
        self.assertEqual(commands[3], ("btrfs", "subvolume", "snapshot", snapshot, self.dataset))
        # The remount is a fresh overlay of the swapped-in vol, not a
        # bind mount.
        self.assertEqual(
            commands[4],
            (
                "mount",
                "-t",
                "overlay",
                "overlay",
                "-o",
                self._rw_options(self.dataset, self.empty_lower),
                str(self.rootfs),
            ),
        )
        self.assertTrue(status.executed)

    def test_clone_filesystem_snapshot_marker_travels_and_fork_mounts(self) -> None:
        shared_snapshot = str(self.root / "shared" / "run" / "img@base")
        _write_marker(self.dataset, shared_snapshot)
        fork_sid = SandboxId("sbx-2")
        fork_dataset = str(self.root / "sandboxes" / "sbx-2")
        fork_rootfs = self.root / "bundles" / "sbx-2" / "rootfs"
        # The faked `subvolume snapshot` command would materialize the
        # fork vol (marker included) on a real host; pre-create what it
        # would produce so the overlay remount can read the marker.
        _write_marker(fork_dataset, shared_snapshot)

        returned = self.provider.clone_filesystem_snapshot(
            self.sid, CheckpointId("ckpt-1"), fork_sid, target_rootfs_path=fork_rootfs
        )

        self.assertEqual(returned, fork_dataset)
        snapshot = f"{self.dataset}@ckpt-1"
        self.assertEqual(
            self.executor.commands,
            [
                ("btrfs", "subvolume", "delete", fork_dataset),
                ("btrfs", "subvolume", "snapshot", snapshot, fork_dataset),
                ("btrfs", "subvolume", "snapshot", "-r", fork_dataset, f"{fork_dataset}@ckpt-1"),
                (
                    "mount",
                    "-t",
                    "overlay",
                    "overlay",
                    "-o",
                    self._rw_options(fork_dataset, shared_snapshot),
                    str(fork_rootfs),
                ),
            ],
        )

    def test_snapshot_content_root_mounts_ro_merged_view(self) -> None:
        snapshot = f"{self.dataset}@ckpt-1"
        _write_marker(snapshot, self.empty_lower)

        content_root = self.provider.snapshot_content_root(self.sid, CheckpointId("ckpt-1"))

        snapmount = self.root / "snapmounts" / "sbx-1@ckpt-1"
        self.assertEqual(content_root, snapmount)
        self.assertEqual(
            self.executor.commands,
            [
                ("btrfs", "subvolume", "show", snapshot),
                (
                    "mount",
                    "-t",
                    "overlay",
                    "overlay",
                    "-o",
                    f"lowerdir={snapshot}/upper:{self.empty_lower}",
                    str(snapmount),
                ),
            ],
        )

        # Idempotence: an already-mounted snapmount is reused, no second
        # mount command.
        self.executor.commands.clear()
        with mock.patch("crab.runtime.overlay_provider.os.path.ismount", return_value=True):
            again = self.provider.snapshot_content_root(self.sid, CheckpointId("ckpt-1"))
        self.assertEqual(again, snapmount)
        self.assertEqual(
            self.executor.commands,
            [("btrfs", "subvolume", "show", snapshot)],
        )

    def test_snapshot_content_root_missing_snapshot_raises(self) -> None:
        snapshot = f"{self.dataset}@ckpt-x"
        self.executor.responses[("btrfs", "subvolume", "show", snapshot)] = (1, "", "not found")

        with self.assertRaises(FileNotFoundError):
            self.provider.snapshot_content_root(self.sid, CheckpointId("ckpt-x"))

    def test_destroy_snapshot_ref_strips_prefix_and_sweeps_snapmount(self) -> None:
        snapshot = f"{self.dataset}@ckpt-1"
        snapmount = self.root / "snapmounts" / "sbx-1@ckpt-1"
        snapmount.mkdir(parents=True)

        self.provider.destroy_snapshot_ref(f"overlay:{snapshot}")

        self.assertEqual(self.executor.commands, [("btrfs", "subvolume", "delete", snapshot)])
        # The (unmounted) snapmount directory is reclaimed.
        self.assertFalse(snapmount.exists())

    def test_destroy_filesystem_dataset_unmounts_rootfs_then_deletes(self) -> None:
        Path(self.dataset).mkdir(parents=True)
        Path(f"{self.dataset}@ckpt-1").mkdir()
        snapmount = self.root / "snapmounts" / "sbx-1@ckpt-1"
        snapmount.mkdir(parents=True)

        self.provider.destroy_filesystem_dataset(self.sid, self.dataset)

        self.assertEqual(
            self.executor.commands,
            [
                ("umount", str(self.rootfs)),
                ("btrfs", "subvolume", "delete", f"{self.dataset}@ckpt-1"),
                ("btrfs", "subvolume", "delete", self.dataset),
            ],
        )
        self.assertFalse(snapmount.exists())

    def test_discard_partial_checkpoint_sweeps_snapmount(self) -> None:
        snapshot = f"{self.dataset}@ckpt-1"
        snapmount = self.root / "snapmounts" / "sbx-1@ckpt-1"
        snapmount.mkdir(parents=True)

        self.provider.discard_partial_checkpoint(self.sid, CheckpointId("ckpt-1"))

        self.assertEqual(
            self.executor.commands,
            [
                ("btrfs", "subvolume", "show", snapshot),
                ("btrfs", "subvolume", "delete", snapshot),
            ],
        )
        self.assertFalse(snapmount.exists())

    def test_promote_is_noop_and_fs_ref_round_trips(self) -> None:
        self.provider.promote_filesystem_dataset(self.sid)
        self.assertEqual(self.executor.commands, [])

        metadata = self.provider.filesystem_checkpoint_metadata(self.sid, CheckpointId("ckpt-p"))
        fs_ref = str(metadata["fs_ref"])
        self.assertTrue(fs_ref.startswith("overlay:"))
        self.assertEqual(fs_ref[len("overlay:"):], str(metadata["snapshot"]))
        # No marker scaffolded: the lowerdir key degrades to absent
        # instead of failing metadata collection.
        self.assertNotIn("lowerdir", metadata)

        self.provider.destroy_snapshot_ref(fs_ref)
        destroy_commands = [cmd for cmd in self.executor.commands if "delete" in cmd]
        self.assertEqual(len(destroy_commands), 1)
        self.assertIn(str(metadata["snapshot"]), destroy_commands[0])


class OverlayBackendSelectionTests(unittest.TestCase):
    """RuncRuntime builds the OverlayProvider for filesystem_backend ==
    "overlay" and derives the overlay root from btrfs_root when no
    explicit overlay_root is configured."""

    def _paths(self, base: Path, **kwargs):
        from crab.runtime import RuncRuntimePaths

        return RuncRuntimePaths(
            state_root=base / "state",
            bundle_root=base / "bundles",
            checkpoint_root=base / "checkpoints",
            metadata_root=base / "metadata",
            zfs_dataset_prefix="pool/crab",
            btrfs_root=base / "btrfs",
            **kwargs,
        )

    def test_runtime_backend_selection_overlay(self) -> None:
        from crab.runtime import RuncRuntime, RuncRuntimeOptions

        with tempfile.TemporaryDirectory(prefix="crab_overlay_sel_") as tmp:
            base = Path(tmp)
            runtime = RuncRuntime(
                paths=self._paths(base),
                options=RuncRuntimeOptions(filesystem_backend="overlay"),
            )
            self.assertEqual(runtime._fs.name, "overlay")
            # Default root derives from btrfs_root so a btrfs-prepared
            # host runs overlay with zero extra setup.
            self.assertEqual(
                runtime._fs.default_dataset_name(SandboxId("sbx-d")),
                str(base / "btrfs" / "overlay" / "sandboxes" / "sbx-d"),
            )

            explicit = RuncRuntime(
                paths=self._paths(base, overlay_root=base / "elsewhere"),
                options=RuncRuntimeOptions(filesystem_backend="overlay"),
            )
            self.assertEqual(
                explicit._fs.default_dataset_name(SandboxId("sbx-d")),
                str(base / "elsewhere" / "sandboxes" / "sbx-d"),
            )

            with self.assertRaisesRegex(ValueError, "unsupported filesystem_backend"):
                RuncRuntime(
                    paths=self._paths(base),
                    options=RuncRuntimeOptions(filesystem_backend="ext4"),
                )

    def test_engine_config_parses_overlay_block_and_flat_keys(self) -> None:
        from crab.engine import EngineConfig

        nested = EngineConfig.from_mapping(
            {
                "filesystem": {
                    "backend": "overlay",
                    "btrfs": {"root": "/mnt/pool"},
                    "overlay": {"root": "/mnt/pool/ovl"},
                }
            }
        )
        self.assertEqual(nested.filesystem_backend, "overlay")
        self.assertEqual(nested.btrfs_root, Path("/mnt/pool"))
        self.assertEqual(nested.overlay_root, Path("/mnt/pool/ovl"))

        flat = EngineConfig.from_mapping(
            {"filesystem_backend": "overlay", "overlay_root": "/mnt/flat-ovl"}
        )
        self.assertEqual(flat.filesystem_backend, "overlay")
        self.assertEqual(flat.overlay_root, Path("/mnt/flat-ovl"))

        # No explicit root: stays None in the config; the engine derives
        # `<btrfs_root>/overlay` at system build time.
        default = EngineConfig.from_mapping({"filesystem_backend": "overlay"})
        self.assertIsNone(default.overlay_root)


def _overlay_playground_available() -> bool:
    if shutil.which("btrfs") is None or os.geteuid() != 0:
        return False
    probe = subprocess.run(
        ["stat", "-f", "-c", "%T", "/var/lib/crab/btrfs"],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0 or probe.stdout.strip() != "btrfs":
        return False
    try:
        filesystems = Path("/proc/filesystems").read_text(encoding="utf-8")
    except OSError:
        return False
    return any(line.split() and line.split()[-1] == "overlay" for line in filesystems.splitlines())


def _is_whiteout_at(path: Path) -> bool:
    try:
        st = os.lstat(path)
    except OSError:
        return False
    return stat.S_ISCHR(st.st_mode) and os.major(st.st_rdev) == 0 and os.minor(st.st_rdev) == 0


class _OverlayRealBase(unittest.TestCase):
    """Real overlay+btrfs semantics against the loop-backed playground
    the VM setup mounts at /var/lib/crab/btrfs. Skipped wherever that
    mount (or root, or kernel overlay support) is absent."""

    def setUp(self) -> None:
        if not _overlay_playground_available():
            self.skipTest("overlay playground not available (needs root, btrfs mount, overlay support)")
        self.playground = Path("/var/lib/crab/btrfs") / f"overlay-test-{SandboxId.new()}"
        OverlayProvider.ensure_root(self.playground)
        self.addCleanup(self._cleanup_playground)
        self.provider = OverlayProvider(
            overlay_root=self.playground,
            runtime_name="runc",
            run_command=self._real_run_command,
            run_status=self._real_run_status,
            dataset_resolver=lambda sid: str(self.playground / "sandboxes" / str(sid)),
            rootfs_resolver=lambda sid: self.playground / "bundles" / str(sid) / "rootfs",
        )

    def _cleanup_playground(self) -> None:
        for mount_dir in ("snapmounts", "bundles"):
            base = self.playground / mount_dir
            if base.exists():
                for path in sorted(base.glob("**/")):
                    subprocess.run(["umount", str(path)], check=False, capture_output=True)
        for subvol_dir in ("sandboxes", "shared"):
            base = self.playground / subvol_dir
            if not base.exists():
                continue
            for path in sorted(base.rglob("*"), reverse=True):
                subprocess.run(["btrfs", "subvolume", "delete", str(path)], check=False, capture_output=True)
        shutil.rmtree(self.playground, ignore_errors=True)

    def _real_run_command(self, command, *, operation, sandbox_id=None, checkpoint_id=None, cwd=None, check=True, expected_error_substrings=(), metadata=None, timeout_seconds=None) -> CommandResult:
        _ = (operation, sandbox_id, checkpoint_id, expected_error_substrings, metadata)
        completed = subprocess.run(
            list(command),
            cwd=None if cwd is None else str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout_seconds or 60,
            check=False,
        )
        if check and completed.returncode != 0:
            raise RuntimeError(f"command failed ({completed.returncode}): {' '.join(command)}\nstderr: {completed.stderr}")
        return CommandResult(command=tuple(command), returncode=completed.returncode, stdout=completed.stdout, stderr=completed.stderr)

    def _real_run_status(self, command, *, operation, sandbox_id, checkpoint_id=None, metadata=None) -> RuntimeOperationStatus:
        result = self._real_run_command(command, operation=operation, sandbox_id=sandbox_id, checkpoint_id=checkpoint_id, metadata=metadata)
        merged = dict(metadata or {})
        merged["stdout"] = result.stdout.strip()
        merged["stderr"] = result.stderr.strip()
        return RuntimeOperationStatus(executed=True, reason="command_executed", command=result.command, metadata=merged)

    def _build_shared_lower(self, files: dict[str, str]) -> tuple[str, str]:
        """Materialize an image-like shared cache subvolume + its @base
        snapshot (the lowerdir), the way prepare_launch does."""
        shared_dataset, mountpoint = self.provider.shared_rootfs_details("img", persist_across_runs=False)
        self.provider.create_dataset(shared_dataset, mountpoint, operation="sandbox.zfs_create_shared_rootfs")
        for rel, content in files.items():
            target = Path(shared_dataset) / rel.lstrip("/")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        shared_snapshot = f"{shared_dataset}@base"
        self.provider.create_snapshot(shared_dataset, shared_snapshot, operation="sandbox.zfs_snapshot_shared_rootfs")
        return shared_dataset, shared_snapshot


class OverlayRealIntegrationTests(_OverlayRealBase):
    def test_lifecycle_checkpoint_restore_fork_on_real_overlay(self) -> None:
        sid = SandboxId("sbx-real")
        dataset = self.provider.default_dataset_name(sid)
        rootfs = self.playground / "bundles" / str(sid) / "rootfs"
        shared_dataset, shared_snapshot = self._build_shared_lower(
            {"/etc/issue": "lower-issue\n", "/doomed.txt": "doomed\n"}
        )

        self.provider.clone_shared_base(shared_dataset, shared_snapshot, dataset, rootfs, sandbox_id=sid)
        # The merged view exposes the lower.
        self.assertEqual((rootfs / "etc" / "issue").read_text(encoding="utf-8"), "lower-issue\n")

        # EXDEV regression (design D2): the very first whiteout rename
        # must succeed — this is exactly what dies when upper and work
        # live in different subvolumes.
        (rootfs / "doomed.txt").unlink()
        self.assertTrue(_is_whiteout_at(Path(dataset) / "upper" / "doomed.txt"))

        (rootfs / "state.txt").write_text("v1\n", encoding="utf-8")
        status = self.provider.checkpoint_filesystem(sid, CheckpointId("ckpt-1"))
        self.assertTrue(status.executed)
        snapshot = f"{dataset}@ckpt-1"
        # The snapshot carries marker and whiteout.
        self.assertEqual(
            json.loads((Path(snapshot) / ".crab-overlay.json").read_text(encoding="utf-8"))["lowerdir"],
            shared_snapshot,
        )
        self.assertTrue(_is_whiteout_at(Path(snapshot) / "upper" / "doomed.txt"))

        # The snapshot content root is a kernel-merged read-only view:
        # base content resolves through upper *and* lower, whiteouts
        # apply.
        content_root = self.provider.snapshot_content_root(sid, CheckpointId("ckpt-1"))
        self.assertEqual((content_root / "state.txt").read_text(encoding="utf-8"), "v1\n")
        self.assertEqual((content_root / "etc" / "issue").read_text(encoding="utf-8"), "lower-issue\n")
        self.assertFalse((content_root / "doomed.txt").exists())

        # Mutate, then roll back through the trash-swap emulation; the
        # remount reuses the snapshot-carried upper+work (A1 exp-3/4).
        (rootfs / "state.txt").write_text("v2\n", encoding="utf-8")
        (rootfs / "extra.txt").write_text("junk\n", encoding="utf-8")
        restore_status = self.provider.restore_filesystem(sid, CheckpointId("ckpt-1"))
        self.assertTrue(restore_status.executed)
        self.assertEqual((rootfs / "state.txt").read_text(encoding="utf-8"), "v1\n")
        self.assertFalse((rootfs / "extra.txt").exists())
        self.assertFalse((rootfs / "doomed.txt").exists())
        self.assertEqual(
            [p for p in (self.playground / "sandboxes").iterdir() if ".trash-" in p.name],
            [],
        )

        # Fork from the checkpoint: marker travels, the fork shares the
        # same lower, writes stay isolated.
        fork_sid = SandboxId("sbx-real-fork")
        fork_rootfs = self.playground / "bundles" / str(fork_sid) / "rootfs"
        self.provider.clone_filesystem_snapshot(sid, CheckpointId("ckpt-1"), fork_sid, target_rootfs_path=fork_rootfs)
        self.assertEqual((fork_rootfs / "state.txt").read_text(encoding="utf-8"), "v1\n")
        (fork_rootfs / "state.txt").write_text("fork\n", encoding="utf-8")
        (fork_rootfs / "fork-marker.txt").write_text("fork\n", encoding="utf-8")
        self.assertEqual((rootfs / "state.txt").read_text(encoding="utf-8"), "v1\n")
        self.assertFalse((rootfs / "fork-marker.txt").exists())

        # promote is a no-op; destroying the source leaves the fork
        # alive (the shared lower is untouched).
        self.provider.promote_filesystem_dataset(fork_sid)
        self.provider.destroy_filesystem_dataset(sid, dataset)
        self.assertEqual((fork_rootfs / "state.txt").read_text(encoding="utf-8"), "fork\n")
        self.assertEqual((fork_rootfs / "etc" / "issue").read_text(encoding="utf-8"), "lower-issue\n")

        # Retention path: destroy the fork's checkpoint by ref; the
        # snapmount (if any) is swept first.
        fork_meta = self.provider.filesystem_checkpoint_metadata(fork_sid, CheckpointId("ckpt-f"))
        self.provider.checkpoint_filesystem(fork_sid, CheckpointId("ckpt-f"))
        self.provider.snapshot_content_root(fork_sid, CheckpointId("ckpt-f"))
        self.provider.destroy_snapshot_ref(str(fork_meta["fs_ref"]))
        self.assertFalse(Path(str(fork_meta["snapshot"])).exists())
        self.assertFalse((self.playground / "snapmounts" / f"{fork_sid}@ckpt-f").exists())


class OverlayRealChangesetTests(_OverlayRealBase):
    def test_changeset_translation_against_scripted_mutations(self) -> None:
        sid = SandboxId("sbx-cs")
        dataset = self.provider.default_dataset_name(sid)
        rootfs = self.playground / "bundles" / str(sid) / "rootfs"
        shared_dataset, shared_snapshot = self._build_shared_lower(
            {
                "/etc/issue": "lower-issue\n",
                "/bin/tool": "tool-v1\n",
                "/data/a.txt": "a\n",
                "/data/b.txt": "b\n",
                "/sedfile.txt": "alpha beta\n",
                "/meta.txt": "meta\n",
            }
        )
        self.provider.clone_shared_base(shared_dataset, shared_snapshot, dataset, rootfs, sandbox_id=sid)
        # Pre-existing upper file: the base checkpoint must see it in
        # the upper so its later rename is a *physical* in-upper rename.
        (rootfs / "pre-upper.txt").write_text("pre\n", encoding="utf-8")

        self.provider.checkpoint_filesystem(sid, CheckpointId("ckpt-1"))

        # Scripted mutations (design §8.4), all through the merged view:
        (rootfs / "newfile.txt").write_text("new\n", encoding="utf-8")
        # sed -i on a *lower* file: write-temp-then-rename (the C1
        # zfs-misfold pitfall) compounds copy-up + in-upper rename and
        # must fold to a single `modified` entry (§5 rules 3+5).
        subprocess.run(
            ["sed", "-i", "s/alpha/gamma/", str(rootfs / "sedfile.txt")],
            check=True,
            capture_output=True,
        )
        (rootfs / "bin" / "tool").unlink()
        shutil.rmtree(rootfs / "data")
        (rootfs / "data").mkdir()
        (rootfs / "data" / "z.txt").write_text("z\n", encoding="utf-8")
        os.rename(rootfs / "etc" / "issue", rootfs / "etc" / "issue.bak")
        os.rename(rootfs / "pre-upper.txt", rootfs / "pre-upper-moved.txt")
        os.chmod(rootfs / "meta.txt", 0o600)

        # A1 open point, re-verified here: the kernel stamps
        # trusted.overlay.opaque=y on the recreated dir.
        self.assertEqual(
            os.getxattr(Path(dataset) / "upper" / "data", "trusted.overlay.opaque"),
            b"y",
        )
        # rm of a lower file plants a 0:0 whiteout in the upper.
        self.assertTrue(_is_whiteout_at(Path(dataset) / "upper" / "bin" / "tool"))

        entries = self.provider.changeset_since(sid, CheckpointId("ckpt-1"))
        observed = [(entry.path, entry.change, entry.renamed_from) for entry in entries]
        self.assertEqual(
            observed,
            [
                # Parent directories whose entries changed are raw truth
                # on zfs/btrfs (mtime churn) and equally on overlay: the
                # whiteout/copy-up materializes them in the upper.
                ("/bin", "modified", None),
                ("/bin/tool", "removed", None),
                ("/data", "modified", None),
                ("/data/a.txt", "removed", None),
                ("/data/b.txt", "removed", None),
                ("/data/z.txt", "added", None),
                # Documented D6 divergence: renaming a not-yet-copied-up
                # lower file degrades to added(new) + removed(old).
                ("/etc", "modified", None),
                ("/etc/issue", "removed", None),
                ("/etc/issue.bak", "added", None),
                ("/meta.txt", "modified", None),
                ("/newfile.txt", "added", None),
                # In-upper renames keep their attribution.
                ("/pre-upper-moved.txt", "renamed", "/pre-upper.txt"),
                ("/sedfile.txt", "modified", None),
            ],
        )
        # The transient diff snapshot is cleaned up.
        self.assertEqual(
            [p for p in (self.playground / "sandboxes").iterdir() if "@changeset-" in p.name],
            [],
        )
        self.provider.destroy_filesystem_dataset(sid, dataset)


class OverlayRealContainerTests(unittest.TestCase):
    """Full runc container lifecycle on an overlay rootfs backed by a
    shared image lower: launch, exec, filesystem checkpoint, CRIU dump,
    filesystem + process restore, and the fork-restore leg the roadmap
    exit criterion demands (checkpoint --leave-running, clone the vol,
    CRIU-restore the fork in its own bundle — the A1 exp-3 shape). Runs
    only inside the crab-dev VM (needs docker/runc/criu/btrfs + root)."""

    _IMAGE = "python:3.11-slim"

    def setUp(self) -> None:
        for tool in ("docker", "runc", "criu", "btrfs"):
            if shutil.which(tool) is None:
                self.skipTest(f"{tool} not installed")
        if not _overlay_playground_available():
            self.skipTest("overlay playground not available")
        probe = subprocess.run(["docker", "image", "inspect", self._IMAGE], capture_output=True, check=False)
        if probe.returncode != 0:
            pull = subprocess.run(["docker", "pull", self._IMAGE], capture_output=True, check=False)
            if pull.returncode != 0:
                self.skipTest(f"cannot pull {self._IMAGE}")

    def _write_bundle_config(self, bundle_dir: Path) -> None:
        subprocess.run(["runc", "spec"], cwd=bundle_dir, check=True, capture_output=True)
        config_path = bundle_dir / "config.json"
        cfg = json.loads(config_path.read_text())
        linux_cfg = cfg.get("linux", {})
        linux_cfg["namespaces"] = [
            ns for ns in linux_cfg.get("namespaces", []) if ns.get("type") not in {"network", "cgroup"}
        ]
        linux_cfg.pop("seccomp", None)
        cfg["linux"] = linux_cfg
        cfg["process"]["terminal"] = False
        cfg["process"]["args"] = ["/bin/sh", "-c", "while :; do sleep 0.5; done"]
        cfg["root"]["path"] = "rootfs"
        cfg["root"]["readonly"] = False
        config_path.write_text(json.dumps(cfg, indent=2))

    def test_container_lifecycle_and_fork_restore_on_overlay_rootfs(self) -> None:
        from crab.runtime import RuncRuntime, RuncRuntimeOptions, RuncRuntimePaths
        from integrations.sandboxes.runtime.image import export_image_rootfs

        playground = Path("/var/lib/crab/btrfs") / f"overlay-container-{SandboxId.new()}"
        OverlayProvider.ensure_root(playground)
        tmpdir = tempfile.TemporaryDirectory(prefix="crab_overlay_e2e_")
        self.addCleanup(tmpdir.cleanup)
        base = Path(tmpdir.name)
        sid = SandboxId("sbx-ovl-e2e")
        fork_name = "sbx-ovl-e2e-fork"
        bundle_dir = base / "bundles" / str(sid)
        fork_bundle_dir = base / "bundles" / fork_name
        state_root = base / "state"

        runtime = RuncRuntime(
            paths=RuncRuntimePaths(
                state_root=state_root,
                bundle_root=base / "bundles",
                checkpoint_root=base / "checkpoints",
                metadata_root=base / "metadata",
                overlay_root=playground,
            ),
            options=RuncRuntimeOptions(filesystem_backend="overlay"),
        )

        def _cleanup() -> None:
            subprocess.run(["runc", "--root", str(state_root), "delete", "-f", fork_name], check=False, capture_output=True)
            runtime.delete_runtime(sid, force=True, ignore_missing=True)
            try:
                runtime.destroy_filesystem_dataset(sid)
            except Exception:
                pass
            for rootfs_dir in (bundle_dir / "rootfs", fork_bundle_dir / "rootfs"):
                subprocess.run(["umount", str(rootfs_dir)], check=False, capture_output=True)
            snapmounts = playground / "snapmounts"
            if snapmounts.exists():
                for path in sorted(snapmounts.iterdir()):
                    subprocess.run(["umount", str(path)], check=False, capture_output=True)
            for subvol_dir in ("sandboxes", "shared"):
                subvol_base = playground / subvol_dir
                if not subvol_base.exists():
                    continue
                for path in sorted(subvol_base.rglob("*"), reverse=True):
                    subprocess.run(["btrfs", "subvolume", "delete", str(path)], check=False, capture_output=True)
            shutil.rmtree(playground, ignore_errors=True)

        self.addCleanup(_cleanup)

        exported_rootfs = export_image_rootfs(tag=self._IMAGE, output_dir=base / "image")
        bundle_dir.mkdir(parents=True, exist_ok=True)
        self._write_bundle_config(bundle_dir)

        # Production shape: the image lands once in the shared cache and
        # every sandbox references its @base snapshot as lowerdir.
        runtime.launch(
            "runc",
            {
                "sandbox_id": str(sid),
                "bundle_path": str(bundle_dir),
                "shared_rootfs_key": "overlay-e2e",
                "rootfs_init_dirs": ["proc", "dev", "dev/pts", "dev/shm", "dev/mqueue", "sys", "run", "tmp"],
                "rootfs_copy_paths": [{"source": str(exported_rootfs), "destination": "/"}],
            },
        )

        # The rootfs really is an overlay mount of the sandbox vol.
        dataset = runtime.dataset_name_for(sid)
        mounts = Path("/proc/mounts").read_text(encoding="utf-8")
        overlay_lines = [
            line for line in mounts.splitlines()
            if line.split()[1] == str(bundle_dir / "rootfs") and line.split()[2] == "overlay"
        ]
        self.assertTrue(overlay_lines, f"rootfs is not an overlay mount:\n{mounts}")
        self.assertIn(f"upperdir={dataset}/upper", overlay_lines[-1])

        write_v1 = runtime.exec(sid, ["/bin/sh", "-c", "echo v1 > /state.txt && cat /state.txt"])
        self.assertEqual(write_v1.returncode, 0)
        self.assertEqual(write_v1.stdout.strip(), "v1")

        fs_status = runtime.checkpoint_filesystem(sid, CheckpointId("ckpt-1"))
        self.assertTrue(fs_status.executed)

        runtime.exec(sid, ["/bin/sh", "-c", "echo v2 > /state.txt"])

        # CRIU dump on the overlay rootfs; zero extra flags (A1 exp-2).
        proc_status = runtime.checkpoint_process(sid, CheckpointId("ckpt-1"), leave_running=False)
        self.assertTrue(proc_status.executed)
        runtime.delete_runtime(sid, force=True, ignore_missing=True)

        restore_fs = runtime.restore_filesystem(sid, CheckpointId("ckpt-1"))
        self.assertTrue(restore_fs.executed)

        # CRIU restore back onto the remounted overlay.
        restore_proc = runtime.restore_process(sid, CheckpointId("ckpt-1"))
        self.assertTrue(restore_proc.executed)

        read_back = runtime.exec(sid, ["/bin/sh", "-c", "cat /state.txt"])
        self.assertEqual(read_back.returncode, 0)
        self.assertEqual(read_back.stdout.strip(), "v1")

        # Fork-restore leg (A1 exp-3 shape): dump the running source
        # with --leave-running, clone the vol at that checkpoint, and
        # CRIU-restore the fork in its own bundle on its own overlay.
        fork_fs = runtime.checkpoint_filesystem(sid, CheckpointId("ckpt-2"))
        self.assertTrue(fork_fs.executed)
        fork_dump = runtime.checkpoint_process(sid, CheckpointId("ckpt-2"), leave_running=True)
        self.assertTrue(fork_dump.executed)

        fork_bundle_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(bundle_dir / "config.json", fork_bundle_dir / "config.json")
        runtime.clone_filesystem_snapshot(
            sid,
            CheckpointId("ckpt-2"),
            SandboxId(fork_name),
            target_rootfs_path=fork_bundle_dir / "rootfs",
        )
        image_path = Path(runtime.process_checkpoint_location(sid, CheckpointId("ckpt-2")) or "")
        fork_work = base / "checkpoints" / fork_name / "ckpt-2" / "work"
        fork_work.mkdir(parents=True, exist_ok=True)
        restore = subprocess.run(
            [
                "runc",
                "--root",
                str(state_root),
                "restore",
                "-d",
                "--bundle",
                str(fork_bundle_dir),
                "--image-path",
                str(image_path),
                "--work-path",
                str(fork_work),
                "--tcp-established",
                "--shell-job",
                "--ext-unix-sk",
                fork_name,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(restore.returncode, 0, f"fork restore failed: {restore.stderr}")

        def _raw_exec(name: str, script: str) -> str:
            result = subprocess.run(
                ["runc", "--root", str(state_root), "exec", name, "/bin/sh", "-c", script],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, f"exec in {name} failed: {result.stderr}")
            return result.stdout.strip()

        # Source and fork run concurrently with isolated writes.
        self.assertEqual(_raw_exec(fork_name, "cat /state.txt"), "v1")
        _raw_exec(fork_name, "echo fork > /state.txt")
        self.assertEqual(_raw_exec(fork_name, "cat /state.txt"), "fork")
        source_read = runtime.exec(sid, ["/bin/sh", "-c", "cat /state.txt"])
        self.assertEqual(source_read.stdout.strip(), "v1")

        # Destroying the source leaves the fork alive: the fork owns its
        # vol and the shared lower is untouched.
        runtime.delete_runtime(sid, force=True, ignore_missing=True)
        runtime.destroy_filesystem_dataset(sid)
        self.assertEqual(_raw_exec(fork_name, "cat /state.txt"), "fork")


if __name__ == "__main__":
    unittest.main()
