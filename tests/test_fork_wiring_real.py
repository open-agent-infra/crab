"""Real-host end-to-end for Sandbox.fork(): full SDK stack (in-process
Engine, runc + CRIU + the configured filesystem backend). Self-skipping
outside the crab-dev VM."""
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from crab import Engine, EngineConfig, Sandbox


def _real_stack_available() -> bool:
    import os

    if os.geteuid() != 0:
        return False
    return all(shutil.which(tool) is not None for tool in ("docker", "runc", "criu", "zfs"))


class ForkWiringRealTests(unittest.TestCase):
    _IMAGE = "python:3.11-slim"

    def setUp(self) -> None:
        if not _real_stack_available():
            self.skipTest("docker/runc/criu/zfs/root not available")
        self._tmp = tempfile.TemporaryDirectory(prefix="crab_fork_e2e_")
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        self.engine = Engine.start(
            EngineConfig(
                runtime="runc",
                enable_sandbox_network=False,
                enable_interceptor=False,
                storage_root=root / "storage",
                runtime_root=root / "runtime",
            )
        )
        self.addCleanup(self.engine.stop)

    def _run(self, sandbox: Sandbox, script: str) -> str:
        result = sandbox.commands.run(script)
        self.assertEqual(result.returncode, 0, msg=f"command failed: {script!r}: {result.stderr}")
        return result.stdout.strip()

    def test_fork_creates_independent_running_sandboxes(self) -> None:
        source = Sandbox(image=self._IMAGE, engine=self.engine)
        self.addCleanup(source.kill)
        self._run(source, "echo v1 > /state.txt && cat /state.txt")

        forks = source.fork(2)
        for fork in forks:
            self.addCleanup(fork.kill)
        self.assertEqual(len(forks), 2)
        self.assertNotEqual(forks[0].sandbox_id, forks[1].sandbox_id)

        # Inherited state is visible in both forks.
        for fork in forks:
            self.assertEqual(self._run(fork, "cat /state.txt"), "v1")

        # Mutating fork A affects neither the source nor fork B.
        self._run(forks[0], "echo fork-a > /state.txt")
        self.assertEqual(self._run(forks[0], "cat /state.txt"), "fork-a")
        self.assertEqual(self._run(forks[1], "cat /state.txt"), "v1")
        self.assertEqual(self._run(source, "cat /state.txt"), "v1")

        # Mutating the source affects no fork.
        self._run(source, "echo source-v2 > /state.txt")
        self.assertEqual(self._run(forks[1], "cat /state.txt"), "v1")

    def test_forks_survive_source_destruction(self) -> None:
        source = Sandbox(image=self._IMAGE, engine=self.engine)
        self._run(source, "echo keepsake > /state.txt")
        forks = source.fork(1)
        for fork in forks:
            self.addCleanup(fork.kill)

        source.kill()

        self.assertEqual(self._run(forks[0], "cat /state.txt"), "keepsake")
        # Fork can still checkpoint after the source is gone.
        checkpoint_id = forks[0].checkpoint()
        self.assertIsNotNone(checkpoint_id)

    def test_lazy_fork_returns_and_serves_pages(self) -> None:
        source = Sandbox(image=self._IMAGE, engine=self.engine)
        self.addCleanup(source.kill)
        # Give the fork some memory to fault in lazily.
        self._run(source, "python3 -c \"open('/blob','wb').write(b'x'*(8<<20))\" && echo done")

        forks = source.fork(1, lazy=True)
        for fork in forks:
            self.addCleanup(fork.kill)

        # Exec forces page faults through the lazy-pages daemon.
        self.assertEqual(self._run(forks[0], "wc -c < /blob"), str(8 << 20))


if __name__ == "__main__":
    unittest.main()
