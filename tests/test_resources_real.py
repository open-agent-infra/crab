"""Real-host end-to-end for S3 resource enforcement: `Sandbox(resources=...)`
limits actually bite via runc/cgroups (allocation beyond `memory` is
OOM-killed, forks inherit the source's limits). Full SDK stack (in-process
Engine, runc + CRIU + the configured filesystem backend).

Double-gated: self-skipping outside the crab-dev VM AND unless
`CRAB_REAL_HOST_TESTS=1` is set — enforcement runs are opt-in even where
the stack exists."""
from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path

from crab import Engine, EngineConfig, Sandbox

MiB = 1024 * 1024


def _real_stack_available() -> bool:
    if os.geteuid() != 0:
        return False
    return all(shutil.which(tool) is not None for tool in ("docker", "runc", "criu", "zfs"))


def _opted_in() -> bool:
    return os.environ.get("CRAB_REAL_HOST_TESTS", "") not in ("", "0")


class ResourceEnforcementRealTests(unittest.TestCase):
    _IMAGE = "python:3.11-slim"

    def setUp(self) -> None:
        if not _opted_in():
            self.skipTest("CRAB_REAL_HOST_TESTS not set")
        if not _real_stack_available():
            self.skipTest("docker/runc/criu/zfs/root not available")
        self._tmp = tempfile.TemporaryDirectory(prefix="crab_resources_e2e_")
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

    # Allocate ~192MiB inside a 128MiB memory limit; the kernel OOM-killer
    # must terminate the process (non-zero return, no MemoryError escape).
    _OOM_SCRIPT = (
        "python3 -c \"x = bytearray(192 * 1024 * 1024); print('survived')\""
    )
    # Small allocation well under the limit sanity-checks the sandbox works.
    _OK_SCRIPT = "python3 -c \"x = bytearray(16 * 1024 * 1024); print('ok')\""

    def test_memory_limit_oom_kills_over_allocation(self) -> None:
        sbx = Sandbox(
            image=self._IMAGE,
            engine=self.engine,
            resources={"memory": "128M", "pids": 128},
        )
        self.addCleanup(sbx.kill)

        result = sbx.commands.run(self._OK_SCRIPT)
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertEqual(result.stdout.strip(), "ok")

        result = sbx.commands.run(self._OOM_SCRIPT)
        self.assertNotEqual(result.returncode, 0, msg="allocation over the limit survived")
        self.assertNotIn("survived", result.stdout)

    def test_fork_inherits_memory_limit(self) -> None:
        source = Sandbox(
            image=self._IMAGE,
            engine=self.engine,
            resources={"memory": "128M", "pids": 128},
        )
        self.addCleanup(source.kill)
        # Warm the source so the checkpoint has state to carry.
        self.assertEqual(source.commands.run("echo warm").returncode, 0)

        forks = source.fork(1)
        fork = forks[0]
        self.addCleanup(fork.kill)

        result = fork.commands.run(self._OK_SCRIPT)
        self.assertEqual(result.returncode, 0, msg=result.stderr)

        result = fork.commands.run(self._OOM_SCRIPT)
        self.assertNotEqual(
            result.returncode, 0, msg="fork escaped the inherited memory limit"
        )
        self.assertNotIn("survived", result.stdout)

    def test_unlimited_sandbox_is_unaffected(self) -> None:
        # Zero-breakage: without `resources` the same allocation succeeds.
        sbx = Sandbox(image=self._IMAGE, engine=self.engine)
        self.addCleanup(sbx.kill)
        result = sbx.commands.run(self._OOM_SCRIPT)
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertEqual(result.stdout.strip(), "survived")


if __name__ == "__main__":
    unittest.main()
