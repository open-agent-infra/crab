"""Real-host end-to-end for fork in daemon mode (PR-A3.2): a real
DaemonServer (runc + CRIU + the configured filesystem backend) with the
SDK connected over the Unix socket, plus the `crab sandbox fork` CLI.
Self-skipping outside the crab-dev VM."""
from __future__ import annotations

import contextlib
import io
import shutil
import tempfile
import unittest
from pathlib import Path

from crab import Engine, EngineConfig, Sandbox
from crab.cli import commands as cli_commands
from crab.daemon.server import DaemonServer
from crab.daemon.transport import DaemonClient


def _real_stack_available() -> bool:
    import os

    if os.geteuid() != 0:
        return False
    return all(shutil.which(tool) is not None for tool in ("docker", "runc", "criu", "zfs"))


class ForkDaemonRealTests(unittest.TestCase):
    _IMAGE = "python:3.11-slim"

    def setUp(self) -> None:
        if not _real_stack_available():
            self.skipTest("docker/runc/criu/zfs/root not available")
        self._tmp = tempfile.TemporaryDirectory(prefix="crab_forkd_e2e_")
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

    def test_remote_fork_creates_independent_sandboxes(self) -> None:
        source = Sandbox(image=self._IMAGE, engine=self.engine)
        self.addCleanup(source.kill)
        self._run(source, "echo v1 > /state.txt && cat /state.txt")

        forks = source.fork(2)
        for fork in forks:
            self.addCleanup(fork.kill)
        self.assertEqual(len(forks), 2)

        # Inherited state, then isolation in both directions.
        for fork in forks:
            self.assertEqual(self._run(fork, "cat /state.txt"), "v1")
        self._run(forks[0], "echo fork-a > /state.txt")
        self.assertEqual(self._run(forks[1], "cat /state.txt"), "v1")
        self.assertEqual(self._run(source, "cat /state.txt"), "v1")

        # The daemon registry lists the forks (registered server-side).
        client = DaemonClient(self.socket_path, timeout_seconds=30.0)
        listed = {
            row["sandbox_id"]
            for row in client.get_json("/sandboxes")["sandboxes"]
        }
        for fork in forks:
            self.assertIn(str(fork.sandbox_id), listed)

    def test_forks_survive_source_kill_through_daemon(self) -> None:
        # The kill goes through DELETE /sandboxes/{id}; the daemon-side
        # fork bookkeeping (promote clone, materialize chain bytes) must
        # keep the forks alive.
        source = Sandbox(image=self._IMAGE, engine=self.engine)
        self._run(source, "echo keepsake > /state.txt")
        forks = source.fork(1)
        for fork in forks:
            self.addCleanup(fork.kill)

        source.kill()

        self.assertEqual(self._run(forks[0], "cat /state.txt"), "keepsake")

    def test_cli_fork_prints_running_fork_id(self) -> None:
        source = Sandbox(image=self._IMAGE, engine=self.engine)
        self.addCleanup(source.kill)
        self._run(source, "echo cli-state > /state.txt")

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            rc = cli_commands.main(
                ["--socket", str(self.socket_path), "sandbox", "fork", str(source.sandbox_id)]
            )
        self.assertEqual(rc, 0)
        fork_ids = stdout.getvalue().split()
        self.assertEqual(len(fork_ids), 1)

        fork = Sandbox.connect(fork_ids[0], engine=self.engine)
        self.addCleanup(fork.kill)
        self.assertEqual(self._run(fork, "cat /state.txt"), "cli-state")

    def test_cli_fork_from_a_checkpoint_branches_from_the_past(self) -> None:
        source = Sandbox(image=self._IMAGE, engine=self.engine)
        self.addCleanup(source.kill)
        self._run(source, "echo v1 > /state.txt")
        checkpoint_id = source.checkpoint()
        self._run(source, "echo v2 > /state.txt")

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            rc = cli_commands.main(
                [
                    "--socket",
                    str(self.socket_path),
                    "sandbox",
                    "fork",
                    str(source.sandbox_id),
                    "--checkpoint",
                    str(checkpoint_id),
                ]
            )
        self.assertEqual(rc, 0)
        fork_ids = stdout.getvalue().split()
        self.assertEqual(len(fork_ids), 1)

        fork = Sandbox.connect(fork_ids[0], engine=self.engine)
        self.addCleanup(fork.kill)
        self.assertEqual(self._run(fork, "cat /state.txt"), "v1")
        # The source is not restored by a fork off one of its old checkpoints.
        self.assertEqual(self._run(source, "cat /state.txt"), "v2")


if __name__ == "__main__":
    unittest.main()
