"""Unit tests for fast destructive sandbox kill and asynchronous ZFS trash reaping."""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from crab import CheckpointId, EngineConfig, RuncRuntime, RuncRuntimeOptions, RuncRuntimePaths, SandboxId
from crab.daemon.server import _Routes
from crab.models import RuntimeOperationStatus
from crab.runtime import ZfsProvider
from crab.runtime.base import CommandResult, CommandRunner
from crab.sandbox import Sandbox


class _FakeZfsExecutor:
    """Thread-safe in-memory ZFS command simulator for unit tests."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.commands: list[tuple[str, ...]] = []
        self.datasets: set[str] = set()
        self.mounted: set[str] = set()
        self.snapshots: set[str] = set()
        # snapshot -> set of clone datasets
        self.clones_by_snapshot: dict[str, set[str]] = {}
        # clone dataset -> origin snapshot
        self.origin_by_dataset: dict[str, str] = {}
        self.fail_rename: bool = False
        self.destroy_gate: threading.Event | None = None
        self.destroy_entered = threading.Event()

    def add_dataset(self, dataset: str, *, mounted: bool = True) -> None:
        with self.lock:
            self.datasets.add(dataset)
            if mounted:
                self.mounted.add(dataset)

    def add_snapshot(self, dataset: str, snap_name: str) -> str:
        snap = f"{dataset}@{snap_name}"
        with self.lock:
            self.datasets.add(dataset)
            self.snapshots.add(snap)
        return snap

    def add_clone(self, snapshot: str, clone_dataset: str, *, mounted: bool = True) -> None:
        with self.lock:
            self.datasets.add(clone_dataset)
            if mounted:
                self.mounted.add(clone_dataset)
            self.origin_by_dataset[clone_dataset] = snapshot
            self.clones_by_snapshot.setdefault(snapshot, set()).add(clone_dataset)

    def run_command(
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
        _ = (operation, sandbox_id, checkpoint_id, cwd, expected_error_substrings, metadata, timeout_seconds)
        cmd = tuple(command)
        with self.lock:
            self.commands.append(cmd)

        rc, stdout, stderr = self._dispatch(list(command))
        if check and rc != 0:
            raise RuntimeError(f"command failed ({rc}): {' '.join(command)}\nstderr: {stderr}")
        return CommandResult(command=cmd, returncode=rc, stdout=stdout, stderr=stderr)

    def run_status(
        self,
        command: list[str],
        *,
        operation: str,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId | None = None,
        metadata: dict[str, object] | None = None,
    ) -> RuntimeOperationStatus:
        res = self.run_command(
            command,
            operation=operation,
            sandbox_id=sandbox_id,
            checkpoint_id=checkpoint_id,
            check=True,
            metadata=metadata,
        )
        return RuntimeOperationStatus(
            executed=True,
            reason="command_executed",
            command=res.command,
            metadata=dict(metadata or {}),
        )

    def _dispatch(self, cmd: list[str]) -> tuple[int, str, str]:
        if cmd[:6] == ["zfs", "list", "-H", "-o", "clones", "-t"] and cmd[6:8] == ["snapshot", "-r"]:
            target = cmd[8]
            with self.lock:
                if target not in self.datasets:
                    return 1, "", f"cannot open '{target}': dataset does not exist"
                lines: list[str] = []
                for snap in sorted(self.snapshots):
                    ds = snap.split("@", 1)[0]
                    if ds == target or ds.startswith(f"{target}/"):
                        clones = sorted(self.clones_by_snapshot.get(snap, set()))
                        lines.append(",".join(clones) if clones else "-")
                return 0, ("\n".join(lines) + "\n") if lines else "", ""

        if cmd[:7] == ["zfs", "list", "-H", "-d", "1", "-o", "name,origin"]:
            prefix = cmd[7]
            with self.lock:
                if prefix not in self.datasets:
                    return 1, "", f"cannot open '{prefix}': dataset does not exist"
                rows: list[str] = [f"{prefix}\t-"]
                for ds in sorted(self.datasets):
                    if ds.startswith(f"{prefix}/") and "/" not in ds[len(prefix) + 1 :]:
                        origin = self.origin_by_dataset.get(ds, "-")
                        rows.append(f"{ds}\t{origin}")
                return 0, "\n".join(rows) + "\n", ""

        if cmd[:5] == ["zfs", "list", "-H", "-o", "name"]:
            name = cmd[-1]
            with self.lock:
                exists = name in self.datasets or name in self.snapshots
                return (0, f"{name}\n", "") if exists else (1, "", "dataset does not exist")

        if cmd[:3] == ["zfs", "unmount", "-f"]:
            ds = cmd[3]
            with self.lock:
                if ds not in self.datasets:
                    return 1, "", f"cannot open '{ds}': dataset does not exist"
                if ds not in self.mounted:
                    return 1, "", f"cannot unmount '{ds}': not currently mounted"
                self.mounted.discard(ds)
                return 0, "", ""

        if cmd[:2] == ["zfs", "create"]:
            ds = cmd[-1]
            mount_none = "mountpoint=none" in cmd
            with self.lock:
                if "-p" in cmd:
                    parts = ds.split("/")
                    for idx in range(2, len(parts) + 1):
                        self.datasets.add("/".join(parts[:idx]))
                else:
                    if ds in self.datasets:
                        return 1, "", f"cannot create '{ds}': dataset already exists"
                    self.datasets.add(ds)
                if not mount_none:
                    self.mounted.add(ds)
                return 0, "", ""

        if cmd[:3] == ["zfs", "rename", "-u"]:
            src, dst = cmd[3], cmd[4]
            with self.lock:
                if self.fail_rename:
                    return 1, "", "simulated rename failure"
                if src not in self.datasets:
                    return 1, "", f"cannot open '{src}': dataset does not exist"
                parent = dst.rsplit("/", 1)[0]
                if parent not in self.datasets:
                    return 1, "", f"cannot create '{dst}': parent does not exist"
                self.datasets.remove(src)
                self.datasets.add(dst)
                if src in self.mounted:
                    self.mounted.remove(src)
                    self.mounted.add(dst)
                if src in self.origin_by_dataset:
                    origin_snap = self.origin_by_dataset.pop(src)
                    self.origin_by_dataset[dst] = origin_snap
                    clones = self.clones_by_snapshot.get(origin_snap)
                    if clones is not None and src in clones:
                        clones.remove(src)
                        clones.add(dst)
                moved_snaps = [s for s in self.snapshots if s.startswith(f"{src}@")]
                for old_snap in moved_snaps:
                    suffix = old_snap.split("@", 1)[1]
                    new_snap = f"{dst}@{suffix}"
                    self.snapshots.remove(old_snap)
                    self.snapshots.add(new_snap)
                    if old_snap in self.clones_by_snapshot:
                        clones = self.clones_by_snapshot.pop(old_snap)
                        self.clones_by_snapshot[new_snap] = clones
                        for clone_ds in clones:
                            self.origin_by_dataset[clone_ds] = new_snap
                return 0, "", ""

        if cmd[:3] == ["zfs", "destroy", "-r"]:
            target = cmd[3]
            gate = self.destroy_gate
            if gate is not None:
                self.destroy_entered.set()
                gate.wait(timeout=5.0)
            with self.lock:
                matching_ds = {
                    d for d in self.datasets if d == target or d.startswith(f"{target}/")
                }
                matching_snaps = {
                    s
                    for s in self.snapshots
                    if s.split("@", 1)[0] in matching_ds
                }
                if not matching_ds and not matching_snaps:
                    return 1, "", f"cannot open '{target}': dataset does not exist"
                for snap in matching_snaps:
                    ext_clones = {
                        c for c in self.clones_by_snapshot.get(snap, set()) if c not in matching_ds
                    }
                    if ext_clones:
                        return (
                            1,
                            "",
                            f"cannot destroy '{snap}': snapshot has dependent clones\n"
                            + "\n".join(sorted(ext_clones)),
                        )
                for ds in matching_ds:
                    origin_snap = self.origin_by_dataset.pop(ds, None)
                    if origin_snap and origin_snap in self.clones_by_snapshot:
                        self.clones_by_snapshot[origin_snap].discard(ds)
                for snap in matching_snaps:
                    self.clones_by_snapshot.pop(snap, None)
                self.datasets.difference_update(matching_ds)
                self.mounted.difference_update(matching_ds)
                self.snapshots.difference_update(matching_snaps)
                return 0, "", ""

        if cmd[:2] == ["zfs", "destroy"] and "@" in cmd[2]:
            snap = cmd[2]
            with self.lock:
                if snap not in self.snapshots:
                    return 1, "", f"cannot open '{snap}': snapshot does not exist"
                clones = self.clones_by_snapshot.get(snap, set())
                if clones:
                    return (
                        1,
                        "",
                        f"cannot destroy '{snap}': snapshot has dependent clones\n"
                        + "\n".join(sorted(clones)),
                    )
                self.snapshots.remove(snap)
                self.clones_by_snapshot.pop(snap, None)
                return 0, "", ""

        return 0, "", ""


class ZfsAsyncDestroyTests(unittest.TestCase):
    def _make_provider(self, executor: _FakeZfsExecutor, *, async_destroy: bool = True) -> ZfsProvider:
        executor.datasets.add("pool")
        executor.datasets.add("pool/crab")
        return ZfsProvider(
            dataset_prefix="pool/crab",
            runtime_name="runc",
            run_command=executor.run_command,
            run_status=executor.run_status,
            dataset_resolver=lambda sid: f"pool/crab/{sid}",
            rootfs_resolver=lambda sid: Path("/bundles") / str(sid) / "rootfs",
            async_destroy=async_destroy,
        )

    def test_fast_detach_vacates_name_immediately_and_reaps_in_background(self) -> None:
        executor = _FakeZfsExecutor()
        executor.destroy_gate = threading.Event()
        provider = self._make_provider(executor)
        self.addCleanup(provider.close)

        sid = SandboxId("sbx-fast")
        dataset = "pool/crab/sbx-fast"
        executor.add_dataset(dataset, mounted=True)
        executor.add_snapshot(dataset, "ckpt-1")

        t0 = time.perf_counter()
        provider.destroy_filesystem_dataset(sid, dataset)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        self.assertLess(elapsed_ms, 250.0)
        self.assertTrue(executor.destroy_entered.wait(timeout=2.0))
        # Original dataset name and mount are already vacated while background
        # `zfs destroy -r` is still blocked on destroy_gate.
        with executor.lock:
            self.assertNotIn(dataset, executor.datasets)
            self.assertNotIn(dataset, executor.mounted)
            trash_children = [d for d in executor.datasets if d.startswith("pool/crab/.trash/")]
            self.assertEqual(len(trash_children), 1)
            self.assertNotIn(trash_children[0], executor.mounted)

        # Immediate re-creation with the same sandbox_id succeeds without collision.
        provider.create_dataset(dataset, Path("/bundles/sbx-fast/rootfs"), operation="sandbox.zfs_create", sandbox_id=sid)
        self.assertIn(dataset, executor.datasets)

        # Release background reaper and wait deterministically for drain.
        executor.destroy_gate.set()
        self.assertTrue(provider.wait_for_pending_destroys(timeout=5.0))
        with executor.lock:
            trash_children = [d for d in executor.datasets if d.startswith("pool/crab/.trash/")]
            self.assertEqual(trash_children, [])
            self.assertIn(dataset, executor.datasets)

    def test_nonexistent_dataset_returns_without_trash_or_destroy(self) -> None:
        executor = _FakeZfsExecutor()
        provider = self._make_provider(executor)
        self.addCleanup(provider.close)

        provider.destroy_filesystem_dataset(SandboxId("sbx-missing"), "pool/crab/sbx-missing")
        self.assertTrue(provider.wait_for_pending_destroys(timeout=2.0))
        self.assertEqual(
            executor.commands,
            [("zfs", "list", "-H", "-o", "clones", "-t", "snapshot", "-r", "pool/crab/sbx-missing")],
        )

    def test_live_dependent_clone_falls_back_to_sync_destroy_and_raises(self) -> None:
        executor = _FakeZfsExecutor()
        provider = self._make_provider(executor)
        self.addCleanup(provider.close)

        snap = executor.add_snapshot("pool/crab/sbx-src", "ckpt-fork")
        executor.add_clone(snap, "pool/crab/sbx-live-fork")

        with self.assertRaises(RuntimeError) as ctx:
            provider.destroy_filesystem_dataset(SandboxId("sbx-src"), "pool/crab/sbx-src")
        self.assertIn("dependent clones", str(ctx.exception))
        self.assertIn("pool/crab/sbx-src", executor.datasets)

    def test_back_to_back_fork_then_source_destroy_reaps_cleanly_in_fifo_order(self) -> None:
        executor = _FakeZfsExecutor()
        executor.destroy_gate = threading.Event()
        provider = self._make_provider(executor)
        self.addCleanup(provider.close)

        snap = executor.add_snapshot("pool/crab/sbx-src", "ckpt-fork")
        executor.add_clone(snap, "pool/crab/sbx-fork")

        provider.destroy_filesystem_dataset(SandboxId("sbx-fork"), "pool/crab/sbx-fork")
        # Even while sbx-fork's trash dataset is still pending background reap,
        # destroying sbx-src detaches cleanly to .trash behind sbx-fork.
        provider.destroy_filesystem_dataset(SandboxId("sbx-src"), "pool/crab/sbx-src")

        executor.destroy_gate.set()
        self.assertTrue(provider.wait_for_pending_destroys(timeout=5.0))
        with executor.lock:
            self.assertNotIn("pool/crab/sbx-fork", executor.datasets)
            self.assertNotIn("pool/crab/sbx-src", executor.datasets)
            self.assertEqual(
                [d for d in executor.datasets if d.startswith("pool/crab/.trash/")],
                [],
            )

    def test_rename_failure_falls_back_to_synchronous_destroy(self) -> None:
        executor = _FakeZfsExecutor()
        executor.fail_rename = True
        provider = self._make_provider(executor)
        self.addCleanup(provider.close)

        executor.add_dataset("pool/crab/sbx-fallback", mounted=True)
        provider.destroy_filesystem_dataset(SandboxId("sbx-fallback"), "pool/crab/sbx-fallback")

        self.assertNotIn("pool/crab/sbx-fallback", executor.datasets)
        self.assertIn(("zfs", "destroy", "-r", "pool/crab/sbx-fallback"), executor.commands)

    def test_startup_sweep_reaps_leftover_trash_in_clone_before_origin_order(self) -> None:
        executor = _FakeZfsExecutor()
        executor.datasets.add("pool")
        executor.datasets.add("pool/crab")
        executor.datasets.add("pool/crab/.trash")
        # Simulate crash leftovers where "a-parent" sorts alphabetically BEFORE
        # its dependent clone "z-clone".
        snap = executor.add_snapshot("pool/crab/.trash/a-parent", "ckpt-1")
        executor.add_clone(snap, "pool/crab/.trash/z-clone", mounted=False)

        provider = ZfsProvider(
            dataset_prefix="pool/crab",
            runtime_name="runc",
            run_command=executor.run_command,
            run_status=executor.run_status,
            dataset_resolver=lambda sid: f"pool/crab/{sid}",
            rootfs_resolver=lambda sid: Path("/bundles") / str(sid) / "rootfs",
            async_destroy=True,
        )
        self.addCleanup(provider.close)

        provider.sweep_trash_async()
        self.assertTrue(provider.wait_for_pending_destroys(timeout=5.0))
        with executor.lock:
            self.assertIn("pool/crab/.trash", executor.datasets)
            self.assertEqual(
                [d for d in executor.datasets if d.startswith("pool/crab/.trash/")],
                [],
            )


class _DummyRunner(CommandRunner):
    def run(self, command: list[str], *, cwd: Path | None = None, timeout_seconds: float | None = None) -> CommandResult:
        _ = (cwd, timeout_seconds)
        return CommandResult(command=tuple(command), returncode=0, stdout="", stderr="")


class FastKillAndRuntimeConfigTests(unittest.TestCase):
    def test_daemon_kill_skips_graceful_stop_while_stop_route_retains_it(self) -> None:
        calls: list[tuple[str, str]] = []
        fake_engine = SimpleNamespace(
            system=SimpleNamespace(
                release_txn=lambda sid: calls.append(("release_txn", str(sid))),
                prepare_source_destroy=lambda sid: calls.append(("prepare_source_destroy", str(sid))),
                release_fork=lambda sid: calls.append(("release_fork", str(sid))),
            ),
            runtime=SimpleNamespace(
                stop=lambda sid: calls.append(("stop", str(sid))),
                delete=lambda sid: calls.append(("delete", str(sid))),
            ),
            unregister_upstream=lambda sid: calls.append(("unregister_upstream", str(sid))),
            release_network_lease=lambda sid: calls.append(("release_network_lease", str(sid))),
        )
        fake_daemon = SimpleNamespace(
            require_engine=lambda: fake_engine,
            unregister_sandbox=lambda sid: calls.append(("unregister_sandbox", str(sid))),
        )
        routes = _Routes(fake_daemon)

        res = routes.kill_sandbox({}, sandbox_id="sbx-1")
        self.assertTrue(res["ok"])
        self.assertNotIn(("stop", "sbx-1"), calls)
        self.assertIn(("delete", "sbx-1"), calls)

        calls.clear()
        stop_res = routes.stop_sandbox({}, sandbox_id="sbx-1")
        self.assertTrue(stop_res["ok"])
        self.assertEqual(calls, [("stop", "sbx-1")])

    def test_runc_runtime_async_destroy_defaults_and_config_override(self) -> None:
        with tempfile.TemporaryDirectory(prefix="crab_zfs_cfg_") as tmp:
            base = Path(tmp)
            paths = RuncRuntimePaths(
                state_root=base / "state",
                bundle_root=base / "bundles",
                checkpoint_root=base / "checkpoints",
                metadata_root=base / "metadata",
                zfs_dataset_prefix="pool/crab",
            )
            real_runner_rt = RuncRuntime(paths=paths)
            self.assertTrue(real_runner_rt._fs.async_destroy)  # type: ignore[attr-defined]

            injected_runner_rt = RuncRuntime(paths=paths, command_runner=_DummyRunner())
            self.assertFalse(injected_runner_rt._fs.async_destroy)  # type: ignore[attr-defined]

            forced_rt = RuncRuntime(
                paths=paths,
                command_runner=_DummyRunner(),
                options=RuncRuntimeOptions(zfs_async_destroy=True),
            )
            self.assertTrue(forced_rt._fs.async_destroy)  # type: ignore[attr-defined]
            forced_rt.close()

        cfg_default = EngineConfig.from_mapping({})
        self.assertIsNone(cfg_default.runc_options)
        cfg_disabled = EngineConfig.from_mapping({"runc": {"zfs_async_destroy": False}})
        assert cfg_disabled.runc_options is not None
        self.assertIs(cfg_disabled.runc_options.zfs_async_destroy, False)

    def test_remote_sandbox_kill_skips_redundant_post_delete_round_trips(self) -> None:
        calls: list[str] = []
        sbx = object.__new__(Sandbox)
        sbx._sandbox_id = SandboxId("sbx-remote")
        sbx._closed = False
        sbx._engine = SimpleNamespace(
            system=None,
            runtime=SimpleNamespace(
                _client=object(),
                delete=lambda sid: calls.append(f"delete:{sid}"),
            ),
            unregister_upstream=lambda sid: calls.append(f"unregister_upstream:{sid}"),
            release_network_lease=lambda sid: calls.append(f"release_network_lease:{sid}"),
            _unregister_sandbox=lambda _self: calls.append("unregister_sandbox"),
        )
        sbx.kill()
        self.assertEqual(calls, ["delete:sbx-remote", "unregister_sandbox"])

        local_calls: list[str] = []
        local_sbx = object.__new__(Sandbox)
        local_sbx._sandbox_id = SandboxId("sbx-local")
        local_sbx._closed = False
        local_sbx._engine = SimpleNamespace(
            system=None,
            runtime=SimpleNamespace(
                delete=lambda sid: local_calls.append(f"delete:{sid}"),
            ),
            unregister_upstream=lambda sid: local_calls.append(f"unregister_upstream:{sid}"),
            release_network_lease=lambda sid: local_calls.append(f"release_network_lease:{sid}"),
            _unregister_sandbox=lambda _self: local_calls.append("unregister_sandbox"),
        )
        local_sbx.kill()
        self.assertEqual(
            local_calls,
            [
                "delete:sbx-local",
                "unregister_upstream:sbx-local",
                "release_network_lease:sbx-local",
                "unregister_sandbox",
            ],
        )


def _detect_real_zfs_pool() -> str | None:
    if os.geteuid() != 0 or shutil.which("zfs") is None or shutil.which("zpool") is None:
        return None
    env_pool = os.environ.get("CRAB_ZPOOL_NAME", "").strip()
    if env_pool:
        probe = subprocess.run(
            ["zpool", "list", "-H", "-o", "name", env_pool],
            capture_output=True,
            text=True,
            check=False,
        )
        return env_pool if probe.returncode == 0 else None
    probe = subprocess.run(
        ["zpool", "list", "-H", "-o", "name"],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        return None
    pools = [line.strip() for line in probe.stdout.splitlines() if line.strip()]
    for pool in pools:
        if pool.startswith("crab"):
            return pool
    return pools[0] if pools else None


class ZfsRealAsyncDestroyIntegrationTests(unittest.TestCase):
    """End-to-end verification against a real OpenZFS pool when available."""

    def setUp(self) -> None:
        pool = _detect_real_zfs_pool()
        if pool is None:
            self.skipTest("real ZFS pool / root privileges not available")
        self._tmp = tempfile.TemporaryDirectory(prefix="crab_zfs_real_")
        self.addCleanup(self._tmp.cleanup)
        self.bundle_root = Path(self._tmp.name) / "bundles"
        self.prefix = f"{pool}/_crab_async_test_{SandboxId.new()}"
        subprocess.run(
            ["zfs", "create", "-p", "-o", "mountpoint=none", self.prefix],
            check=True,
            capture_output=True,
            text=True,
        )
        self.addCleanup(self._cleanup_zfs_prefix)
        self.provider = ZfsProvider(
            dataset_prefix=self.prefix,
            runtime_name="runc",
            run_command=self._real_run_command,
            run_status=self._real_run_status,
            dataset_resolver=lambda sid: f"{self.prefix}/{sid}",
            rootfs_resolver=lambda sid: self.bundle_root / str(sid) / "rootfs",
            async_destroy=True,
        )
        self.addCleanup(self.provider.close)

    def _cleanup_zfs_prefix(self) -> None:
        self.provider.close(timeout=10.0)
        subprocess.run(
            ["zfs", "destroy", "-r", self.prefix],
            check=False,
            capture_output=True,
            text=True,
        )

    def _real_run_command(
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
        _ = (operation, sandbox_id, checkpoint_id, expected_error_substrings, metadata)
        completed = subprocess.run(
            list(command),
            cwd=None if cwd is None else str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout_seconds or 60.0,
            check=False,
        )
        if check and completed.returncode != 0:
            raise RuntimeError(
                f"command failed ({completed.returncode}): {' '.join(command)}\nstderr: {completed.stderr}"
            )
        return CommandResult(
            command=tuple(command),
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )

    def _real_run_status(
        self,
        command: list[str],
        *,
        operation: str,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId | None = None,
        metadata: dict[str, object] | None = None,
    ) -> RuntimeOperationStatus:
        res = self._real_run_command(
            command,
            operation=operation,
            sandbox_id=sandbox_id,
            checkpoint_id=checkpoint_id,
            check=True,
            metadata=metadata,
        )
        return RuntimeOperationStatus(
            executed=True,
            reason="command_executed",
            command=res.command,
            metadata=dict(metadata or {}),
        )

    def _list_trash_children(self) -> list[str]:
        res = subprocess.run(
            ["zfs", "list", "-H", "-d", "1", "-o", "name", self.provider.trash_prefix],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode != 0:
            return []
        prefix = f"{self.provider.trash_prefix}/"
        return [
            line.strip()
            for line in res.stdout.splitlines()
            if line.strip().startswith(prefix)
        ]

    def test_real_zfs_fast_detach_immediate_recreate_and_clone_safety(self) -> None:
        sid = SandboxId("sbx-real-1")
        ds = self.provider.default_dataset_name(sid)
        rootfs = self.bundle_root / str(sid) / "rootfs"

        self.provider.create_dataset(ds, rootfs, operation="sandbox.zfs_create", sandbox_id=sid)
        self.assertTrue(os.path.ismount(rootfs))
        (rootfs / "payload.bin").write_bytes(b"x" * (1024 * 1024))
        self.provider.checkpoint_filesystem(sid, CheckpointId("ckpt-1"))

        # Destroy detaches mount + dataset name synchronously, then reaps in background.
        self.provider.destroy_filesystem_dataset(sid, ds)
        self.assertFalse(os.path.ismount(rootfs))
        self.assertFalse(self.provider.dataset_exists(ds))
        shutil.rmtree(rootfs.parent)

        # Immediate recreate of same sandbox_id and rootfs path succeeds cleanly.
        self.provider.create_dataset(ds, rootfs, operation="sandbox.zfs_create", sandbox_id=sid)
        self.assertTrue(os.path.ismount(rootfs))
        self.provider.checkpoint_filesystem(sid, CheckpointId("ckpt-fork"))

        fork_sid = SandboxId("sbx-real-fork")
        fork_ds = self.provider.default_dataset_name(fork_sid)
        fork_rootfs = self.bundle_root / str(fork_sid) / "rootfs"
        self.provider.clone_filesystem_snapshot(
            sid,
            CheckpointId("ckpt-fork"),
            fork_sid,
            target_rootfs_path=fork_rootfs,
        )

        # Destroying source while live fork still depends on it raises synchronously.
        with self.assertRaises(RuntimeError):
            self.provider.destroy_filesystem_dataset(sid, ds)

        # Back-to-back fork destroy + source destroy succeeds and drains to 0 trash datasets.
        self.provider.destroy_filesystem_dataset(fork_sid, fork_ds)
        self.provider.destroy_filesystem_dataset(sid, ds)
        self.assertTrue(self.provider.wait_for_pending_destroys(timeout=15.0))
        self.assertEqual(self._list_trash_children(), [])

    def test_real_zfs_startup_sweep_reaps_crash_leftovers(self) -> None:
        trash = self.provider.trash_prefix
        subprocess.run(
            ["zfs", "create", "-p", "-o", "mountpoint=none", f"{trash}/a-stale-parent"],
            check=True,
        )
        subprocess.run(
            ["zfs", "snapshot", f"{trash}/a-stale-parent@c1"],
            check=True,
        )
        subprocess.run(
            [
                "zfs",
                "clone",
                "-o",
                "mountpoint=none",
                f"{trash}/a-stale-parent@c1",
                f"{trash}/z-stale-clone",
            ],
            check=True,
        )

        fresh_provider = ZfsProvider(
            dataset_prefix=self.prefix,
            runtime_name="runc",
            run_command=self._real_run_command,
            run_status=self._real_run_status,
            dataset_resolver=lambda sid: f"{self.prefix}/{sid}",
            rootfs_resolver=lambda sid: self.bundle_root / str(sid) / "rootfs",
            async_destroy=True,
        )
        self.addCleanup(fresh_provider.close)
        fresh_provider.sweep_trash_async()
        self.assertTrue(fresh_provider.wait_for_pending_destroys(timeout=15.0))
        self.assertEqual(self._list_trash_children(), [])


if __name__ == "__main__":
    unittest.main()
