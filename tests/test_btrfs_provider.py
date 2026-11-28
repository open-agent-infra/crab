from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from crab.ids import CheckpointId, SandboxId
from crab.models import RuntimeOperationStatus
from crab.runtime import BtrfsProvider, FilesystemProvider, ZfsProvider
from crab.runtime.base import CommandResult


class _RecordingExecutor:
    """Fake command pipeline standing in for RuncRuntime._run_command /
    _run_status. Providers are units: they only need these callables and
    the resolver pair, so they can be tested without a runtime."""

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


def _make_provider(root: Path, executor: _RecordingExecutor, **kwargs) -> BtrfsProvider:
    return BtrfsProvider(
        btrfs_root=root,
        runtime_name="runc",
        run_command=executor.run_command,
        run_status=executor.run_status,
        dataset_resolver=lambda sid: str(root / "sandboxes" / str(sid)),
        rootfs_resolver=lambda sid: root / "bundles" / str(sid) / "rootfs",
        **kwargs,
    )


class BtrfsProviderCommandMappingTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="crab_btrfs_provider_")
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.executor = _RecordingExecutor()
        self.provider = _make_provider(self.root, self.executor)
        self.dataset = str(self.root / "sandboxes" / "sbx-1")
        self.rootfs = self.root / "bundles" / "sbx-1" / "rootfs"

    def test_checkpoint_filesystem_snapshots_readonly_and_records_fs_ref(self) -> None:
        status = self.provider.checkpoint_filesystem(SandboxId("sbx-1"), CheckpointId("ckpt-1"))

        snapshot = f"{self.dataset}@ckpt-1"
        self.assertEqual(
            self.executor.commands[0],
            ("btrfs", "subvolume", "snapshot", "-r", self.dataset, snapshot),
        )
        self.assertEqual(status.metadata["fs_ref"], f"btrfs:{snapshot}")
        self.assertEqual(status.metadata["snapshot"], snapshot)
        self.assertEqual(status.metadata["checkpoint_scope"], "filesystem_only")
        # qgroups are off by default: stats degrade to unknown without
        # issuing a qgroup command.
        self.assertIsNone(status.metadata["filesystem_checkpoint_written_bytes"])
        self.assertNotIn(("btrfs", "qgroup", "show", "--raw", "-f", snapshot), self.executor.commands)

    def test_restore_filesystem_swaps_live_subvolume_through_trash(self) -> None:
        snapshot = f"{self.dataset}@ckpt-1"
        status = self.provider.restore_filesystem(SandboxId("sbx-1"), CheckpointId("ckpt-1"))

        commands = self.executor.commands
        self.assertEqual(commands[0], ("umount", str(self.rootfs)))
        self.assertEqual(commands[1], ("btrfs", "subvolume", "show", self.dataset))
        mv = commands[2]
        self.assertEqual(mv[0], "mv")
        self.assertEqual(mv[1], self.dataset)
        self.assertTrue(mv[2].startswith(f"{self.dataset}.trash-"))
        self.assertEqual(commands[3], ("btrfs", "subvolume", "snapshot", snapshot, self.dataset))
        self.assertEqual(commands[4], ("mount", "--bind", self.dataset, str(self.rootfs)))
        self.assertTrue(status.executed)

    def test_restore_filesystem_recovers_when_previous_restore_crashed_mid_swap(self) -> None:
        # Crash window: previous restore renamed the live subvolume to
        # trash and died before snapshotting back. The live path is gone
        # but a trash sibling exists on disk.
        (self.root / "sandboxes").mkdir(parents=True, exist_ok=True)
        stale_trash = Path(f"{self.dataset}.trash-deadbeef")
        stale_trash.mkdir(parents=True)
        show = ("btrfs", "subvolume", "show", self.dataset)
        self.executor.responses[show] = (1, "", "ERROR: not a subvolume")

        self.provider.restore_filesystem(SandboxId("sbx-1"), CheckpointId("ckpt-1"))

        commands = self.executor.commands
        # Stale trash is reclaimed, no mv is attempted for the missing
        # live subvolume, and the checkpoint is materialized in place.
        self.assertIn(("btrfs", "subvolume", "delete", str(stale_trash)), commands)
        self.assertFalse(any(cmd[0] == "mv" for cmd in commands))
        self.assertIn(
            ("btrfs", "subvolume", "snapshot", f"{self.dataset}@ckpt-1", self.dataset),
            commands,
        )

    def test_restore_filesystem_does_not_destroy_later_snapshots(self) -> None:
        # Divergence from `zfs rollback -r`, by design: later checkpoints
        # stay restorable and retention governs their lifetime.
        (self.root / "sandboxes").mkdir(parents=True, exist_ok=True)
        later = Path(f"{self.dataset}@ckpt-9")
        later.mkdir(parents=True)

        self.provider.restore_filesystem(SandboxId("sbx-1"), CheckpointId("ckpt-1"))

        self.assertNotIn(("btrfs", "subvolume", "delete", str(later)), self.executor.commands)

    def test_promote_is_a_noop(self) -> None:
        self.provider.promote_filesystem_dataset(SandboxId("sbx-1"))
        self.assertEqual(self.executor.commands, [])

    def test_clone_filesystem_snapshot_takes_writable_snapshot_and_binds(self) -> None:
        target_rootfs = self.root / "bundles" / "sbx-fork" / "rootfs"
        target_dataset = str(self.root / "sandboxes" / "sbx-fork")

        returned = self.provider.clone_filesystem_snapshot(
            SandboxId("sbx-1"),
            CheckpointId("ckpt-1"),
            SandboxId("sbx-fork"),
            target_rootfs_path=target_rootfs,
        )

        self.assertEqual(returned, target_dataset)
        commands = self.executor.commands
        self.assertIn(("btrfs", "subvolume", "delete", target_dataset), commands)
        self.assertIn(
            ("btrfs", "subvolume", "snapshot", f"{self.dataset}@ckpt-1", target_dataset),
            commands,
        )
        self.assertEqual(commands[-1], ("mount", "--bind", target_dataset, str(target_rootfs)))

    def test_create_dataset_creates_subvolume_and_bind_mounts(self) -> None:
        self.provider.create_dataset(
            self.dataset,
            self.rootfs,
            operation="sandbox.zfs_create",
            sandbox_id=SandboxId("sbx-1"),
        )
        self.assertEqual(
            self.executor.commands,
            [
                ("btrfs", "subvolume", "create", self.dataset),
                ("mount", "--bind", self.dataset, str(self.rootfs)),
            ],
        )

    def test_create_dataset_reclaims_plain_directory_wreckage(self) -> None:
        # An interrupted shared-base materialization can leave a plain
        # directory at the subvolume path (it doubles as the mountpoint);
        # create_dataset must reclaim it or `subvolume create` fails.
        dataset_path = Path(self.dataset)
        (dataset_path / "bin").mkdir(parents=True)
        show = ("btrfs", "subvolume", "show", self.dataset)
        self.executor.responses[show] = (1, "", "ERROR: Not a Btrfs subvolume")

        self.provider.create_dataset(
            self.dataset,
            self.rootfs,
            operation="sandbox.zfs_create",
            sandbox_id=SandboxId("sbx-1"),
        )

        self.assertFalse(dataset_path.exists())
        self.assertEqual(
            self.executor.commands,
            [
                show,
                ("btrfs", "subvolume", "create", self.dataset),
                ("mount", "--bind", self.dataset, str(self.rootfs)),
            ],
        )

    def test_create_dataset_leaves_existing_subvolume_alone(self) -> None:
        # A real subvolume at the path is not wreckage: never rmtree it.
        dataset_path = Path(self.dataset)
        (dataset_path / "bin").mkdir(parents=True)
        # `subvolume show` succeeds (default response), so the contents
        # survive and creation proceeds against the existing path.
        self.provider.create_dataset(
            self.dataset,
            self.rootfs,
            operation="sandbox.zfs_create",
            sandbox_id=SandboxId("sbx-1"),
        )

        self.assertTrue((dataset_path / "bin").exists())
        self.assertEqual(
            self.executor.commands,
            [
                ("btrfs", "subvolume", "show", self.dataset),
                ("btrfs", "subvolume", "create", self.dataset),
                ("mount", "--bind", self.dataset, str(self.rootfs)),
            ],
        )

    def test_shared_rootfs_details_uses_subvolume_path_as_mountpoint(self) -> None:
        dataset, mountpoint = self.provider.shared_rootfs_details("img-key", persist_across_runs=False)
        self.assertEqual(Path(dataset), mountpoint)
        self.assertEqual(mountpoint, self.root / "shared" / "run" / "img-key")
        # No bind mount needed when the mountpoint IS the subvolume path.
        self.provider.create_dataset(dataset, mountpoint, operation="sandbox.zfs_create_shared_rootfs")
        self.assertEqual(self.executor.commands, [("btrfs", "subvolume", "create", dataset)])

    def test_destroy_dataset_removes_sibling_snapshots_first(self) -> None:
        (self.root / "sandboxes").mkdir(parents=True, exist_ok=True)
        snap_a = Path(f"{self.dataset}@ckpt-a")
        snap_b = Path(f"{self.dataset}@ckpt-b")
        snap_a.mkdir(parents=True)
        snap_b.mkdir(parents=True)

        self.provider.destroy_dataset(self.dataset, operation="sandbox.btrfs_destroy")

        self.assertEqual(
            self.executor.commands,
            [
                ("btrfs", "subvolume", "delete", str(snap_a)),
                ("btrfs", "subvolume", "delete", str(snap_b)),
                ("btrfs", "subvolume", "delete", self.dataset),
            ],
        )

    def test_destroy_snapshot_ref_accepts_prefixed_and_bare_refs(self) -> None:
        snapshot = f"{self.dataset}@ckpt-1"
        self.provider.destroy_snapshot_ref(f"btrfs:{snapshot}")
        self.provider.destroy_snapshot_ref(snapshot)
        self.assertEqual(
            self.executor.commands,
            [
                ("btrfs", "subvolume", "delete", snapshot),
                ("btrfs", "subvolume", "delete", snapshot),
            ],
        )

    def test_snapshot_stats_uses_qgroups_only_when_enabled(self) -> None:
        provider = _make_provider(self.root, self.executor, qgroups_enabled=True)
        snapshot = f"{self.dataset}@ckpt-1"
        self.executor.responses[("btrfs", "qgroup", "show", "--raw", "-f", snapshot)] = (
            0,
            "qgroupid         rfer         excl\n--------         ----         ----\n0/257        1048576       131072\n",
            "",
        )

        status = provider.checkpoint_filesystem(SandboxId("sbx-1"), CheckpointId("ckpt-1"))

        self.assertEqual(status.metadata["filesystem_checkpoint_written_bytes"], 131072)
        self.assertEqual(status.metadata["filesystem_checkpoint_used_bytes"], 1048576)


class ProviderParityTests(unittest.TestCase):
    """Both backends satisfy the same contract shape: everything the
    composite workers and storage retention rely on."""

    def _zfs(self, executor: _RecordingExecutor) -> ZfsProvider:
        return ZfsProvider(
            dataset_prefix="pool/crab",
            runtime_name="runc",
            run_command=executor.run_command,
            run_status=executor.run_status,
            dataset_resolver=lambda sid: f"pool/crab/{sid}",
            rootfs_resolver=lambda sid: Path("/bundles") / str(sid) / "rootfs",
        )

    def test_both_providers_are_complete_filesystem_providers(self) -> None:
        executor = _RecordingExecutor()
        with tempfile.TemporaryDirectory(prefix="crab_parity_") as tmp:
            providers: list[FilesystemProvider] = [
                self._zfs(executor),
                _make_provider(Path(tmp), executor),
            ]
            for provider in providers:
                # Instantiation already proves ABC completeness; verify the
                # naming surface agrees on shape.
                self.assertTrue(provider.name)
                dataset = provider.default_dataset_name(SandboxId("sbx-p"))
                self.assertIn("sbx-p", dataset)

    def test_fs_ref_round_trips_through_destroy_for_both_backends(self) -> None:
        with tempfile.TemporaryDirectory(prefix="crab_parity_") as tmp:
            for make in (self._zfs, lambda ex: _make_provider(Path(tmp), ex)):
                executor = _RecordingExecutor()
                provider = make(executor)
                metadata = provider.filesystem_checkpoint_metadata(SandboxId("sbx-p"), CheckpointId("ckpt-p"))
                fs_ref = str(metadata["fs_ref"])
                self.assertTrue(fs_ref.startswith(f"{provider.name}:"))
                self.assertEqual(fs_ref[len(provider.name) + 1 :], str(metadata["snapshot"]))

                provider.destroy_snapshot_ref(fs_ref)
                destroy_commands = [cmd for cmd in executor.commands if "delete" in cmd or "destroy" in cmd]
                self.assertEqual(len(destroy_commands), 1)
                self.assertIn(str(metadata["snapshot"]), destroy_commands[0])


def _btrfs_playground_available() -> bool:
    if shutil.which("btrfs") is None:
        return False
    probe = subprocess.run(
        ["stat", "-f", "-c", "%T", "/var/lib/crab/btrfs"],
        capture_output=True,
        text=True,
        check=False,
    )
    return probe.returncode == 0 and probe.stdout.strip() == "btrfs"


class BtrfsRealIntegrationTests(unittest.TestCase):
    """Real semantics against the loop-backed playground the VM setup
    mounts at /var/lib/crab/btrfs (see tools/vm/vm-setup.sh). Skipped
    wherever that mount is absent."""

    def setUp(self) -> None:
        if not _btrfs_playground_available():
            self.skipTest("btrfs playground not available")
        self.playground = Path("/var/lib/crab/btrfs") / f"provider-test-{SandboxId.new()}"
        self.playground.mkdir(parents=True)
        self.addCleanup(self._cleanup_playground)
        self.provider = BtrfsProvider(
            btrfs_root=self.playground,
            runtime_name="runc",
            run_command=self._real_run_command,
            run_status=self._real_run_status,
            dataset_resolver=lambda sid: str(self.playground / "sandboxes" / str(sid)),
            rootfs_resolver=lambda sid: self.playground / "bundles" / str(sid) / "rootfs",
        )

    def _cleanup_playground(self) -> None:
        for bundle in sorted((self.playground / "bundles").glob("*/rootfs")):
            subprocess.run(["umount", str(bundle)], check=False, capture_output=True)
        for subvol_dir in ("sandboxes", "shared"):
            base = self.playground / subvol_dir
            if not base.exists():
                continue
            for path in sorted(base.iterdir(), reverse=True):
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

    def test_checkpoint_restore_clone_lifecycle_on_real_btrfs(self) -> None:
        sid = SandboxId("sbx-real")
        dataset = self.provider.default_dataset_name(sid)
        rootfs = self.playground / "bundles" / str(sid) / "rootfs"

        self.provider.create_dataset(dataset, rootfs, operation="sandbox.zfs_create", sandbox_id=sid)
        (rootfs / "state.txt").write_text("v1\n", encoding="utf-8")

        status = self.provider.checkpoint_filesystem(sid, CheckpointId("ckpt-1"))
        self.assertTrue(status.executed)

        # Mutate, then roll back through the trash-swap emulation.
        (rootfs / "state.txt").write_text("v2\n", encoding="utf-8")
        (rootfs / "extra.txt").write_text("junk\n", encoding="utf-8")
        restore_status = self.provider.restore_filesystem(sid, CheckpointId("ckpt-1"))
        self.assertTrue(restore_status.executed)
        self.assertEqual((rootfs / "state.txt").read_text(encoding="utf-8"), "v1\n")
        self.assertFalse((rootfs / "extra.txt").exists())
        # Trash from the swap is reclaimed.
        self.assertEqual(
            [p for p in (self.playground / "sandboxes").iterdir() if ".trash-" in p.name],
            [],
        )

        # Fork from the checkpoint; the fork is independent of the source.
        fork_sid = SandboxId("sbx-real-fork")
        fork_rootfs = self.playground / "bundles" / str(fork_sid) / "rootfs"
        self.provider.clone_filesystem_snapshot(sid, CheckpointId("ckpt-1"), fork_sid, target_rootfs_path=fork_rootfs)
        (fork_rootfs / "state.txt").write_text("fork\n", encoding="utf-8")
        self.assertEqual((rootfs / "state.txt").read_text(encoding="utf-8"), "v1\n")

        # promote is a no-op, and destroying the source must not affect
        # the fork (no clone-origin dependency on btrfs).
        self.provider.promote_filesystem_dataset(fork_sid)
        self.provider.destroy_filesystem_dataset(sid, dataset)
        self.assertEqual((fork_rootfs / "state.txt").read_text(encoding="utf-8"), "fork\n")

        # Retention path: destroy the fork's checkpoint by ref.
        fork_meta = self.provider.filesystem_checkpoint_metadata(fork_sid, CheckpointId("ckpt-f"))
        self.provider.checkpoint_filesystem(fork_sid, CheckpointId("ckpt-f"))
        self.provider.destroy_snapshot_ref(str(fork_meta["fs_ref"]))
        self.assertFalse(Path(str(fork_meta["snapshot"])).exists())


class FilesystemBackendSelectionTests(unittest.TestCase):
    """RuncRuntime builds the provider matching options.filesystem_backend,
    and EngineConfig parses both the nested `filesystem` block and the
    flat legacy keys."""

    def _paths(self, base: Path):
        from crab.runtime import RuncRuntimePaths

        return RuncRuntimePaths(
            state_root=base / "state",
            bundle_root=base / "bundles",
            checkpoint_root=base / "checkpoints",
            metadata_root=base / "metadata",
            zfs_dataset_prefix="pool/crab",
            btrfs_root=base / "btrfs",
        )

    def test_runtime_backend_selection(self) -> None:
        from crab.runtime import RuncRuntime, RuncRuntimeOptions

        with tempfile.TemporaryDirectory(prefix="crab_backend_sel_") as tmp:
            base = Path(tmp)
            default_runtime = RuncRuntime(paths=self._paths(base))
            self.assertEqual(default_runtime._fs.name, "zfs")

            btrfs_runtime = RuncRuntime(
                paths=self._paths(base),
                options=RuncRuntimeOptions(filesystem_backend="btrfs"),
            )
            self.assertEqual(btrfs_runtime._fs.name, "btrfs")

            with self.assertRaisesRegex(ValueError, "unsupported filesystem_backend"):
                RuncRuntime(
                    paths=self._paths(base),
                    options=RuncRuntimeOptions(filesystem_backend="ext4"),
                )

    def test_engine_config_parses_filesystem_block_and_flat_keys(self) -> None:
        from crab.engine import EngineConfig

        nested = EngineConfig.from_mapping(
            {
                "filesystem": {
                    "backend": "btrfs",
                    "btrfs": {"root": "/mnt/pool", "qgroups_enabled": True},
                }
            }
        )
        self.assertEqual(nested.filesystem_backend, "btrfs")
        self.assertEqual(nested.btrfs_root, Path("/mnt/pool"))
        self.assertTrue(nested.btrfs_qgroups_enabled)

        flat = EngineConfig.from_mapping(
            {"filesystem_backend": "btrfs", "btrfs_root": "/mnt/flat"}
        )
        self.assertEqual(flat.filesystem_backend, "btrfs")
        self.assertEqual(flat.btrfs_root, Path("/mnt/flat"))
        self.assertFalse(flat.btrfs_qgroups_enabled)

        default = EngineConfig.from_mapping({})
        self.assertEqual(default.filesystem_backend, "zfs")
        self.assertIsNone(default.btrfs_root)


class BtrfsRealContainerTests(unittest.TestCase):
    """Full runc container lifecycle on a btrfs-backed rootfs: launch,
    exec, filesystem checkpoint, CRIU process checkpoint, filesystem +
    process restore. This is the end-to-end proof that CRIU tolerates
    the bind-mounted subvolume rootfs. Runs only inside the crab-dev VM
    (needs docker/runc/criu/btrfs and root)."""

    _IMAGE = "python:3.11-slim"

    def setUp(self) -> None:
        for tool in ("docker", "runc", "criu", "btrfs"):
            if shutil.which(tool) is None:
                self.skipTest(f"{tool} not installed")
        if not _btrfs_playground_available():
            self.skipTest("btrfs playground not available")
        probe = subprocess.run(["docker", "image", "inspect", self._IMAGE], capture_output=True, check=False)
        if probe.returncode != 0:
            pull = subprocess.run(["docker", "pull", self._IMAGE], capture_output=True, check=False)
            if pull.returncode != 0:
                self.skipTest(f"cannot pull {self._IMAGE}")

    def test_container_lifecycle_with_criu_on_btrfs_rootfs(self) -> None:
        import json as json_module

        from crab.runtime import RuncRuntime, RuncRuntimeOptions, RuncRuntimePaths
        from integrations.sandboxes.runtime.image import export_image_rootfs

        playground = Path("/var/lib/crab/btrfs") / f"container-test-{SandboxId.new()}"
        playground.mkdir(parents=True)
        tmpdir = tempfile.TemporaryDirectory(prefix="crab_btrfs_e2e_")
        self.addCleanup(tmpdir.cleanup)
        base = Path(tmpdir.name)
        sid = SandboxId("sbx-btrfs-e2e")
        bundle_dir = base / "bundles" / str(sid)

        runtime = RuncRuntime(
            paths=RuncRuntimePaths(
                state_root=base / "state",
                bundle_root=base / "bundles",
                checkpoint_root=base / "checkpoints",
                metadata_root=base / "metadata",
                btrfs_root=playground,
            ),
            options=RuncRuntimeOptions(filesystem_backend="btrfs"),
        )

        def _cleanup() -> None:
            runtime.delete_runtime(sid, force=True, ignore_missing=True)
            try:
                runtime.destroy_filesystem_dataset(sid)
            except Exception:
                pass
            subprocess.run(["umount", str(bundle_dir / "rootfs")], check=False, capture_output=True)
            for sub in sorted((playground / "sandboxes").glob("*"), reverse=True):
                subprocess.run(["btrfs", "subvolume", "delete", str(sub)], check=False, capture_output=True)
            shutil.rmtree(playground, ignore_errors=True)

        self.addCleanup(_cleanup)

        exported_rootfs = export_image_rootfs(tag=self._IMAGE, output_dir=base / "image")

        bundle_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(["runc", "spec"], cwd=bundle_dir, check=True, capture_output=True)
        config_path = bundle_dir / "config.json"
        cfg = json_module.loads(config_path.read_text())
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
        config_path.write_text(json_module.dumps(cfg, indent=2))

        runtime.launch(
            "runc",
            {
                "sandbox_id": str(sid),
                "bundle_path": str(bundle_dir),
                "rootfs_init_dirs": ["proc", "dev", "dev/pts", "dev/shm", "dev/mqueue", "sys", "run", "tmp"],
                "rootfs_copy_paths": [{"source": str(exported_rootfs), "destination": "/"}],
            },
        )

        write_v1 = runtime.exec(sid, ["/bin/sh", "-c", "echo v1 > /state.txt && cat /state.txt"])
        self.assertEqual(write_v1.returncode, 0)
        self.assertEqual(write_v1.stdout.strip(), "v1")

        fs_status = runtime.checkpoint_filesystem(sid, CheckpointId("ckpt-1"))
        self.assertTrue(fs_status.executed)

        runtime.exec(sid, ["/bin/sh", "-c", "echo v2 > /state.txt"])

        # CRIU dump on the bind-mounted subvolume rootfs; container stops.
        proc_status = runtime.checkpoint_process(sid, CheckpointId("ckpt-1"), leave_running=False)
        self.assertTrue(proc_status.executed)
        runtime.delete_runtime(sid, force=True, ignore_missing=True)

        restore_fs = runtime.restore_filesystem(sid, CheckpointId("ckpt-1"))
        self.assertTrue(restore_fs.executed)

        # CRIU restore back onto the swapped-in subvolume.
        restore_proc = runtime.restore_process(sid, CheckpointId("ckpt-1"))
        self.assertTrue(restore_proc.executed)

        read_back = runtime.exec(sid, ["/bin/sh", "-c", "cat /state.txt"])
        self.assertEqual(read_back.returncode, 0)
        self.assertEqual(read_back.stdout.strip(), "v1")



if __name__ == "__main__":
    unittest.main()
