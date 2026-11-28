from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from crab import RuncRuntime, RuncRuntimePaths, SandboxExecResult, SandboxId, SandboxRuntimeState


class RuncRuntimeResilientExecTests(unittest.TestCase):
    def _runtime(self) -> RuncRuntime:
        tmp = tempfile.TemporaryDirectory(prefix="crab_runc_resilient_")
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        return RuncRuntime(
            paths=RuncRuntimePaths(
                state_root=root / "state",
                bundle_root=root / "bundles",
                checkpoint_root=root / "checkpoints",
                metadata_root=root / "metadata",
                zfs_dataset_prefix="pool/crab",
            )
        )

    def test_resilient_exec_retries_after_recovery_with_original_timeout(self) -> None:
        runtime = self._runtime()
        sandbox_id = SandboxId("sbx-retry")
        failure = SandboxExecResult(args=("runc", "exec"), returncode=255, stdout="", stderr="container not running")
        success = SandboxExecResult(args=("runc", "exec"), returncode=0, stdout="ok\n", stderr="")

        with (
            patch.object(runtime, "exec", side_effect=[failure, success]) as exec_mock,
            patch.object(
                runtime,
                "inspect_runtime",
                side_effect=[
                    SandboxRuntimeState(sandbox_id=sandbox_id, runtime_name="runc", status="missing", pid=None),
                    SandboxRuntimeState(sandbox_id=sandbox_id, runtime_name="runc", status="running", pid=123),
                ],
            ),
            patch("crab.runtime.runc.time.sleep", return_value=None),
        ):
            result = runtime.resilient_exec(
                sandbox_id,
                ["bash", "-c", "echo hi"],
                timeout_s=12.0,
                capture_output=True,
            )

        self.assertEqual(result.stdout, "ok\n")
        self.assertEqual(exec_mock.call_count, 2)
        self.assertEqual(exec_mock.call_args_list[0].kwargs["timeout_s"], 12.0)
        self.assertEqual(exec_mock.call_args_list[1].kwargs["timeout_s"], 12.0)
        self.assertEqual(exec_mock.call_args_list[0].args[1], ["bash", "-c", "echo hi"])
        self.assertEqual(exec_mock.call_args_list[1].args[1], ["bash", "-c", "echo hi"])

    def test_resilient_exec_returns_nonretriable_failures_immediately(self) -> None:
        runtime = self._runtime()
        sandbox_id = SandboxId("sbx-fail")
        failure = SandboxExecResult(args=("runc", "exec"), returncode=17, stdout="", stderr="ordinary failure")

        with (
            patch.object(runtime, "exec", return_value=failure) as exec_mock,
            patch.object(
                runtime,
                "inspect_runtime",
                return_value=SandboxRuntimeState(sandbox_id=sandbox_id, runtime_name="runc", status="running", pid=123),
            ),
        ):
            result = runtime.resilient_exec(
                sandbox_id,
                ["bash", "-c", "exit 17"],
                timeout_s=30.0,
                capture_output=True,
            )

        self.assertEqual(result.returncode, 17)
        self.assertEqual(result.stderr, "ordinary failure")
        self.assertEqual(exec_mock.call_count, 1)

    def test_wait_for_runtime_running_enforces_recovery_timeout(self) -> None:
        runtime = self._runtime()
        sandbox_id = SandboxId("sbx-timeout")

        with (
            patch.object(
                runtime,
                "inspect_runtime",
                return_value=SandboxRuntimeState(sandbox_id=sandbox_id, runtime_name="runc", status="missing", pid=None),
            ),
            patch("crab.runtime.runc.time.monotonic", side_effect=[0.0, 1.1]),
            patch("crab.runtime.runc.time.sleep", return_value=None),
        ):
            with self.assertRaisesRegex(RuntimeError, "timed out waiting for sandbox"):
                runtime._wait_for_runtime_running(sandbox_id, timeout_s=1.0)

    def test_exec_decodes_non_utf8_stdout_and_stderr_without_crashing(self) -> None:
        import os
        import stat
        import sys

        tmp = tempfile.TemporaryDirectory(prefix="crab_runc_utf8_")
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        fake_runc = root / "fake-runc"
        fake_runc.write_text(
            f"#!{sys.executable}\n"
            "import sys\n"
            "sys.stdout.buffer.write(b'head\\xe3\\xff\\xfetail\\n')\n"
            "sys.stderr.buffer.write(b'err\\x80\\xc0done\\n')\n"
            "sys.stdout.buffer.flush()\n"
            "sys.stderr.buffer.flush()\n",
            encoding="utf-8",
        )
        os.chmod(fake_runc, stat.S_IRWXU)
        runtime = RuncRuntime(
            runtime_bin=str(fake_runc),
            paths=RuncRuntimePaths(
                state_root=root / "state",
                bundle_root=root / "bundles",
                checkpoint_root=root / "checkpoints",
                metadata_root=root / "metadata",
                zfs_dataset_prefix="pool/crab",
            ),
        )
        sandbox_id = SandboxId("sbx-binary-out")
        (root / "bundles" / str(sandbox_id)).mkdir(parents=True, exist_ok=True)
        with patch.object(runtime, "_mark_isolated_exec_filesystem_changed", return_value=None):
            res = runtime.exec(sandbox_id, ["sh", "-c", "true"], timeout_s=None, capture_output=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("head", res.stdout)
        self.assertIn("tail\n", res.stdout)
        self.assertIn("\ufffd", res.stdout)
        self.assertIn("err", res.stderr)
        self.assertIn("\ufffd", res.stderr)


if __name__ == "__main__":
    unittest.main()
