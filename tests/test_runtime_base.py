from __future__ import annotations

import subprocess
import unittest
from unittest.mock import patch

from crab.ids import SandboxId
from crab.runtime.base import SubprocessCommandRunner
from crab.runtime.in_memory import InMemoryRuntime


class RuntimeBaseTests(unittest.TestCase):
    def test_detached_commands_use_devnull_for_all_stdio(self) -> None:
        runner = SubprocessCommandRunner(timeout_seconds=1.0)

        with patch("crab.runtime.base.subprocess.run") as run_mock:
            run_mock.return_value = subprocess.CompletedProcess(
                args=["runc", "run", "-d", "sbx-test"],
                returncode=0,
                stdout="",
                stderr="",
            )

            result = runner.run(["runc", "run", "-d", "sbx-test"])

        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertEqual(result.stderr, "")
        kwargs = run_mock.call_args.kwargs
        self.assertIs(kwargs["stdin"], subprocess.DEVNULL)
        self.assertIs(kwargs["stdout"], subprocess.DEVNULL)
        self.assertIs(kwargs["stderr"], subprocess.DEVNULL)

    def test_attached_commands_capture_stdout_and_stderr(self) -> None:
        runner = SubprocessCommandRunner(timeout_seconds=1.0)

        with patch("crab.runtime.base.subprocess.run") as run_mock:
            run_mock.return_value = subprocess.CompletedProcess(
                args=["runc", "state", "sbx-test"],
                returncode=0,
                stdout="ok\n",
                stderr="",
            )

            result = runner.run(["runc", "state", "sbx-test"])

        self.assertEqual(result.stdout, "ok\n")
        kwargs = run_mock.call_args.kwargs
        self.assertIsNone(kwargs["stdin"])
        self.assertIs(kwargs["stdout"], subprocess.PIPE)
        self.assertIs(kwargs["stderr"], subprocess.PIPE)

    def test_run_accepts_timeout_override(self) -> None:
        runner = SubprocessCommandRunner(timeout_seconds=1.0)

        with patch("crab.runtime.base.subprocess.run") as run_mock:
            run_mock.return_value = subprocess.CompletedProcess(
                args=["zfs", "create", "pool/test"],
                returncode=0,
                stdout="",
                stderr="",
            )

            runner.run(["zfs", "create", "pool/test"], timeout_seconds=123.0)

        self.assertEqual(run_mock.call_args.kwargs["timeout"], 123.0)


class InMemoryRuntimeLifecycleTests(unittest.TestCase):
    def _launched(self) -> tuple[InMemoryRuntime, SandboxId]:
        runtime = InMemoryRuntime()
        sandbox_id = runtime.launch("in-memory")
        return runtime, sandbox_id

    def test_start_relaunches_stopped_sandbox(self) -> None:
        runtime, sid = self._launched()
        runtime.stop(sid)
        self.assertEqual(runtime.describe(sid).status, "stopped")
        runtime.start(sid)
        self.assertEqual(runtime.describe(sid).status, "running")

    def test_restart_returns_sandbox_to_running(self) -> None:
        runtime, sid = self._launched()
        runtime.restart(sid)
        self.assertEqual(runtime.describe(sid).status, "running")


if __name__ == "__main__":
    unittest.main()
