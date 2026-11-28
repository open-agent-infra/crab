"""Real-host end-to-end for fork-backed transactions (B3): the roadmap
exit criteria — concurrent reads on the source see no dirt during the
txn, commit atomically promotes the fork's filesystem *and processes*
onto the source's unchanged identity, abort leaves the source untouched
— plus the dirty-source commit gate and the daemon/CLI passthrough.
zfs primary, btrfs repeat of the core round trip. Self-skipping outside
the crab-dev VM."""
from __future__ import annotations

import contextlib
import io
import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

from crab import Engine, EngineConfig, Sandbox
from crab.cli import commands as cli_commands
from crab.daemon.server import DaemonServer
from crab.txn import TxnCommitConflict

_BTRFS_ROOT = Path("/var/lib/crab/btrfs")


def _real_stack_available() -> bool:
    import os

    if os.geteuid() != 0:
        return False
    return all(shutil.which(tool) is not None for tool in ("docker", "runc", "criu", "zfs"))


def _btrfs_root_available() -> bool:
    if shutil.which("btrfs") is None or not _BTRFS_ROOT.is_dir():
        return False
    result = subprocess.run(
        ["stat", "-f", "-c", "%T", str(_BTRFS_ROOT)],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0 and result.stdout.strip() == "btrfs"


def _overlay_available() -> bool:
    if not _btrfs_root_available():
        return False
    try:
        filesystems = Path("/proc/filesystems").read_text(encoding="utf-8")
    except OSError:
        return False
    return any(line.split() and line.split()[-1] == "overlay" for line in filesystems.splitlines())


_SETUP = "mkdir -p /probe && echo v1 > /probe/state.txt"


class _ForkTxnRealMixin:
    """Shared scenarios; concrete classes bind the backend."""

    _IMAGE = "python:3.11-slim"
    _BACKEND = "zfs"

    def setUp(self) -> None:
        if not _real_stack_available():
            self.skipTest("docker/runc/criu/zfs/root not available")
        self._tmp = tempfile.TemporaryDirectory(prefix="crab_fork_txn_e2e_")
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def _engine(self) -> Engine:
        engine = Engine.start(
            EngineConfig(
                runtime="runc",
                enable_sandbox_network=False,
                enable_interceptor=False,
                storage_root=self.root / "storage",
                runtime_root=self.root / "runtime",
                filesystem_backend=self._BACKEND,
                host_inspector_launch_mode="thread",
            )
        )
        self.addCleanup(engine.stop)
        return engine

    def _run(self, sandbox: Sandbox, script: str) -> str:
        result = sandbox.commands.run(script)
        self.assertEqual(result.returncode, 0, msg=f"command failed: {script!r}: {result.stderr}")
        return result.stdout.strip()

    def _settle(self, engine: Engine, sandbox_id, *, timeout: float = 30.0) -> None:
        """Wait until the real inspector observed the writes — the
        dirty-source commit gate rides the C1 changeset gate."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            snapshot = engine.system.inspector.inspect(sandbox_id)
            if snapshot.filesystem_changed:
                return
            time.sleep(0.5)
        self.fail(f"inspector never saw writes on {sandbox_id}")

    def _assert_fork_gone(self, engine: Engine, fork_id: str) -> None:
        dataset = engine.system.runtime.dataset_name_for(fork_id)
        if self._BACKEND == "zfs":
            result = subprocess.run(
                ["zfs", "list", "-H", "-o", "name", dataset],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0, f"fork dataset leaked: {dataset}")
        else:
            self.assertFalse(Path(dataset).exists(), f"fork subvolume leaked: {dataset}")

    def test_commit_promotes_fs_and_processes_onto_source(self) -> None:
        engine = self._engine()
        sandbox = Sandbox(image=self._IMAGE, engine=engine)
        self.addCleanup(sandbox.kill)
        self._run(sandbox, _SETUP)

        txn = sandbox.begin(isolation="fork")
        self.assertEqual(txn.isolation, "fork")
        self.assertTrue(txn.fork_sandbox_id)
        txn.exec("echo fork-v2 > /probe/state.txt && echo new > /probe/fork-new.txt")
        # Isolation: the source keeps serving the pre-txn state.
        self.assertEqual(self._run(sandbox, "cat /probe/state.txt"), "v1")
        self.assertEqual(
            self._run(sandbox, "test -e /probe/fork-new.txt && echo yes || echo no"), "no"
        )
        # A background process started inside the txn must survive the
        # promotion (this is the capability C4 builds on). CRIU restores
        # keep PIDs, so a pidfile probe works across the swap (the slim
        # image has no pgrep).
        txn.exec("sh -c 'nohup sleep 300 > /dev/null 2>&1 & echo $! > /probe/bg.pid'")

        result = txn.commit()

        self.assertTrue(result.promoted_checkpoint_id)
        self.assertEqual(txn.resolved, "committed")
        self.assertEqual(self._run(sandbox, "cat /probe/state.txt"), "fork-v2")
        self.assertEqual(self._run(sandbox, "cat /probe/fork-new.txt"), "new")
        self.assertEqual(
            self._run(sandbox, "test -d /proc/$(cat /probe/bg.pid) && echo alive"), "alive"
        )
        self._assert_fork_gone(engine, txn.fork_sandbox_id)
        # The identity never changed: same sandbox handle keeps working.
        self.assertEqual(self._run(sandbox, "echo still-me"), "still-me")
        # C3: the commit adopted the fork's action history — the txn's
        # exec records are readable on the source with provenance.
        observations = sandbox.actions(kind="observation")
        self.assertTrue(observations, "commit did not consolidate the fork's history")
        self.assertEqual(
            {row["payload"]["fork_sandbox_id"] for row in observations},
            {txn.fork_sandbox_id},
        )
        adopted_cmds = [
            " ".join(row["payload"]["origin_payload"].get("argv") or [])
            for row in observations
            if row["payload"].get("origin_kind") == "exec"
        ]
        self.assertTrue(
            any("fork-v2" in cmd for cmd in adopted_cmds),
            f"txn exec not adopted: {adopted_cmds}",
        )

    def test_abort_leaves_source_untouched(self) -> None:
        engine = self._engine()
        sandbox = Sandbox(image=self._IMAGE, engine=engine)
        self.addCleanup(sandbox.kill)
        self._run(sandbox, _SETUP)

        txn = sandbox.begin(isolation="fork")
        txn.exec("echo doomed > /probe/state.txt && echo junk > /probe/junk.txt")
        result = txn.abort()

        self.assertIsNone(result.restored_checkpoint_id)
        self.assertEqual(self._run(sandbox, "cat /probe/state.txt"), "v1")
        self.assertEqual(
            self._run(sandbox, "test -e /probe/junk.txt && echo yes || echo no"), "no"
        )
        self._assert_fork_gone(engine, txn.fork_sandbox_id)


class ZfsForkTxnRealTests(_ForkTxnRealMixin, unittest.TestCase):
    _BACKEND = "zfs"

    def test_dirty_source_commit_gate_and_force(self) -> None:
        engine = self._engine()
        sandbox = Sandbox(image=self._IMAGE, engine=engine)
        self.addCleanup(sandbox.kill)
        self._run(sandbox, _SETUP)

        txn = sandbox.begin(isolation="fork")
        txn.exec("echo fork-side > /probe/state.txt")
        # The source goes dirty behind the txn's back.
        self._run(sandbox, "echo dirty > /probe/dirty.txt")
        self._settle(engine, sandbox.sandbox_id)

        with self.assertRaises(TxnCommitConflict):
            txn.commit()
        # The txn stays open and the source still serves its own state.
        self.assertEqual(self._run(sandbox, "cat /probe/dirty.txt"), "dirty")

        result = txn.commit(force=True)
        self.assertTrue(result.promoted_checkpoint_id)
        self.assertEqual(self._run(sandbox, "cat /probe/state.txt"), "fork-side")
        # Forced promotion discards the source-side write.
        self.assertEqual(
            self._run(sandbox, "test -e /probe/dirty.txt && echo yes || echo no"), "no"
        )


class BtrfsForkTxnRealTests(_ForkTxnRealMixin, unittest.TestCase):
    _BACKEND = "btrfs"

    def setUp(self) -> None:
        super().setUp()
        if not _btrfs_root_available():
            self.skipTest(f"btrfs root {_BTRFS_ROOT} not available")


class OverlayForkTxnRealTests(_ForkTxnRealMixin, unittest.TestCase):
    """A2 engine-level leg: B3's commit-by-promotion destroys the source
    overlay (umount + vol delete) and re-clones the fork's vol onto the
    source identity — the remount rides the provider's mount hook."""

    _BACKEND = "overlay"

    def setUp(self) -> None:
        super().setUp()
        if not _overlay_available():
            self.skipTest(f"overlay backend prerequisites missing at {_BTRFS_ROOT}")


class ForkTxnDaemonRealTests(unittest.TestCase):
    _IMAGE = "python:3.11-slim"

    def setUp(self) -> None:
        if not _real_stack_available():
            self.skipTest("docker/runc/criu/zfs/root not available")
        self._tmp = tempfile.TemporaryDirectory(prefix="crab_fork_txnd_e2e_")
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        self.socket_path = root / "crab.sock"
        self.daemon = DaemonServer(
            engine_config=EngineConfig(
                runtime="runc",
                enable_sandbox_network=False,
                enable_interceptor=False,
                storage_root=root / "storage",
                runtime_root=root / "runtime",
                host_inspector_launch_mode="thread",
            ),
            socket_path=self.socket_path,
        )
        self.daemon.start()
        self.addCleanup(self.daemon.stop)
        self.engine = Engine.connect(self.socket_path)

    def _run(self, sandbox: Sandbox, script: str) -> str:
        result = sandbox.commands.run(script)
        self.assertEqual(result.returncode, 0, msg=f"command failed: {script!r}: {result.stderr}")
        return result.stdout.strip()

    def test_remote_fork_txn_round_trip_and_cli(self) -> None:
        sandbox = Sandbox(image=self._IMAGE, engine=self.engine)
        self.addCleanup(sandbox.kill)
        self._run(sandbox, _SETUP)

        # SDK over RPC: begin fork txn, exec routes to the fork.
        txn = sandbox.begin(isolation="fork")
        self.assertTrue(txn.fork_sandbox_id)
        txn.exec("echo remote-fork > /probe/state.txt")
        self.assertEqual(self._run(sandbox, "cat /probe/state.txt"), "v1")
        result = txn.commit()
        self.assertTrue(result.promoted_checkpoint_id)
        self.assertEqual(self._run(sandbox, "cat /probe/state.txt"), "remote-fork")

        # CLI: begin --isolation fork, then commit promotes.
        def _cli(argv: list[str]) -> str:
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                rc = cli_commands.main(["--socket", str(self.socket_path), *argv])
            self.assertEqual(rc, 0, msg=f"cli failed: {argv}: {stdout.getvalue()}")
            return stdout.getvalue().strip()

        txn_id = _cli(
            ["txn", "begin", str(sandbox.sandbox_id), "--isolation", "fork", "--label", "cli-b3"]
        )
        self.assertTrue(txn_id.startswith("txn-"))
        reattached = sandbox.current_txn()
        self.assertIsNotNone(reattached)
        self.assertEqual(reattached.isolation, "fork")
        reattached.exec("echo cli-fork > /probe/state.txt")
        commit_line = _cli(["txn", "commit", str(sandbox.sandbox_id), txn_id])
        self.assertIn("promoted=", commit_line)
        self.assertEqual(self._run(sandbox, "cat /probe/state.txt"), "cli-fork")


if __name__ == "__main__":
    unittest.main()
