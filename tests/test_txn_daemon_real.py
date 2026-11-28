"""Real-host end-to-end for daemon-mode transactions (PR-B2.2): a real
DaemonServer over the Unix socket, the SDK connected via Engine.connect,
plus the `crab txn` CLI. Self-skipping outside the crab-dev VM."""
from __future__ import annotations

import contextlib
import io
import shutil
import tempfile
import unittest
from pathlib import Path

from crab import Engine, EngineConfig, Sandbox, TxnActiveError
from crab.cli import commands as cli_commands
from crab.daemon.server import DaemonServer


def _real_stack_available() -> bool:
    import os

    if os.geteuid() != 0:
        return False
    return all(shutil.which(tool) is not None for tool in ("docker", "runc", "criu", "zfs"))


class TxnDaemonRealTests(unittest.TestCase):
    _IMAGE = "python:3.11-slim"

    def setUp(self) -> None:
        if not _real_stack_available():
            self.skipTest("docker/runc/criu/zfs/root not available")
        self._tmp = tempfile.TemporaryDirectory(prefix="crab_txnd_e2e_")
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

    def test_remote_txn_abort_restores_state(self) -> None:
        sandbox = Sandbox(image=self._IMAGE, engine=self.engine)
        self.addCleanup(sandbox.kill)
        self._run(sandbox, "echo base > /state.txt")

        txn = sandbox.begin(label="remote-abort")
        self.assertTrue(txn.base_checkpoint_id)
        # Nested begin surfaces the typed error across the wire.
        with self.assertRaises(TxnActiveError):
            sandbox.begin()
        txn.exec("echo dirty > /state.txt")
        self.assertEqual(self._run(sandbox, "cat /state.txt"), "dirty")
        result = txn.abort()
        self.assertEqual(result.restored_checkpoint_id, txn.base_checkpoint_id)
        self.assertEqual(self._run(sandbox, "cat /state.txt"), "base")
        self.assertIsNone(sandbox.current_txn())

    def test_remote_txn_commit_persists_and_reattaches(self) -> None:
        sandbox = Sandbox(image=self._IMAGE, engine=self.engine)
        self.addCleanup(sandbox.kill)
        self._run(sandbox, "echo before > /state.txt")

        txn = sandbox.begin()
        # A second client can reattach to the open txn.
        reattached = sandbox.current_txn()
        self.assertIsNotNone(reattached)
        self.assertEqual(reattached.txn_id, txn.txn_id)
        txn.exec("echo committed > /state.txt")
        commit = txn.commit()
        self.assertEqual(commit.txn_id, txn.txn_id)
        self.assertEqual(self._run(sandbox, "cat /state.txt"), "committed")
        self.assertIsNone(sandbox.current_txn())

    def test_cli_txn_flow(self) -> None:
        sandbox = Sandbox(image=self._IMAGE, engine=self.engine)
        self.addCleanup(sandbox.kill)
        self._run(sandbox, "echo cli-base > /state.txt")

        def _cli(argv: list[str]) -> str:
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                rc = cli_commands.main(["--socket", str(self.socket_path), *argv])
            self.assertEqual(rc, 0, msg=f"cli failed: {argv}: {stdout.getvalue()}")
            return stdout.getvalue().strip()

        txn_id = _cli(["txn", "begin", str(sandbox.sandbox_id), "--label", "cli-e2e"])
        self.assertTrue(txn_id.startswith("txn-"))
        status = _cli(["txn", "status", str(sandbox.sandbox_id)])
        self.assertIn(txn_id, status)
        self.assertIn("label=cli-e2e", status)

        self._run(sandbox, "echo cli-dirty > /state.txt")
        abort_line = _cli(["txn", "abort", str(sandbox.sandbox_id), txn_id])
        self.assertIn(f"aborted {txn_id}", abort_line)
        self.assertEqual(self._run(sandbox, "cat /state.txt"), "cli-base")
        self.assertIn("no active transaction", _cli(["txn", "status", str(sandbox.sandbox_id)]))


if __name__ == "__main__":
    unittest.main()
