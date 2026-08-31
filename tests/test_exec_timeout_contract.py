from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from crab.daemon import DaemonRequestError
from crab.errors import (
    ImageNotFoundError,
    SandboxExecCleanupError,
    SandboxExecTimeout,
)
from crab.ids import SandboxId
from crab.remote_engine import _map_create_error, _map_exec_error
from crab.runtime.runc import RuncRuntime, RuncRuntimePaths, _ExecScope


class RuntimeExecTimeoutTests(unittest.TestCase):
    def _runtime(self) -> RuncRuntime:
        temp_dir = tempfile.TemporaryDirectory(prefix="crab_exec_timeout_")
        self.addCleanup(temp_dir.cleanup)
        root = Path(temp_dir.name)
        return RuncRuntime(
            paths=RuncRuntimePaths(
                state_root=root / "state",
                bundle_root=root / "bundles",
                checkpoint_root=root / "checkpoints",
                metadata_root=root / "metadata",
                zfs_dataset_prefix="pool/crab",
            )
        )

    def test_untimed_exec_stays_in_registered_sandbox_cgroup(self) -> None:
        runtime = self._runtime()
        with patch.object(runtime, "_container_cgroup_path") as resolve:
            scope = runtime._prepare_exec_scope(
                SandboxId("sbx-untimed"), timeout_s=None
            )

        resolve.assert_not_called()
        self.assertIsNone(scope.cgroup_name)
        self.assertIsNone(scope.cgroup_path)

    def test_timeout_is_returned_only_after_payload_cleanup(self) -> None:
        runtime = self._runtime()
        scope = _ExecScope(
            token="token",
            pid_file=Path("/tmp/not-used.pid"),
            cgroup_name="exec-token",
            cgroup_path=Path("/sys/fs/cgroup/crab/exec-token"),
            parent_cgroup_path=Path("/sys/fs/cgroup/crab"),
        )
        proc = Mock()
        proc.communicate.side_effect = subprocess.TimeoutExpired(["runc"], 1.0)
        proc.poll.return_value = None
        proc.returncode = None

        with (
            patch.object(runtime, "_prepare_exec_scope", return_value=scope),
            patch(
                "crab.runtime.runc.subprocess.Popen", return_value=proc
            ) as popen,
            patch.object(
                runtime,
                "_abort_exec_process",
                return_value=("partial out", "partial err", None),
            ) as abort,
            patch.object(
                runtime, "_mark_isolated_exec_filesystem_changed"
            ) as mark_dirty,
        ):
            with self.assertRaises(SandboxExecTimeout) as caught:
                runtime.exec(
                    SandboxId("sbx-timeout"),
                    ["sh", "-c", "sleep 30 & wait"],
                    timeout_s=1.0,
                )

        abort.assert_called_once()
        mark_dirty.assert_called_once_with(SandboxId("sbx-timeout"), scope)
        self.assertEqual(caught.exception.stdout, "partial out")
        self.assertEqual(caught.exception.stderr, "partial err")
        command = popen.call_args.args[0]
        self.assertIn("--pid-file", command)
        self.assertEqual(command[command.index("--cgroup") + 1], "exec-token")

    def test_cleanup_failure_is_distinct_from_completed_timeout(self) -> None:
        runtime = self._runtime()
        scope = _ExecScope(token="token", pid_file=Path("/tmp/not-used.pid"))
        proc = Mock()
        proc.communicate.side_effect = subprocess.TimeoutExpired(["runc"], 1.0)
        proc.poll.return_value = None
        proc.returncode = None
        cleanup = SandboxExecCleanupError(
            "payload survived",
            cmd=["sleep", "30"],
            timeout=1.0,
            payload_pid=123,
        )

        with (
            patch.object(runtime, "_prepare_exec_scope", return_value=scope),
            patch("crab.runtime.runc.subprocess.Popen", return_value=proc),
            patch.object(
                runtime,
                "_abort_exec_process",
                return_value=("", "", cleanup),
            ),
        ):
            with self.assertRaises(SandboxExecCleanupError) as caught:
                runtime.exec(
                    SandboxId("sbx-timeout"), ["sleep", "30"], timeout_s=1.0
                )

        self.assertEqual(caught.exception.payload_pid, 123)


class RemoteTypedErrorTests(unittest.TestCase):
    @staticmethod
    def _error(payload: dict[str, object], *, status: int = 500) -> DaemonRequestError:
        return DaemonRequestError(status, "/sandboxes/sbx/exec", json.dumps(payload).encode())

    def test_exec_timeout_rehydrates_with_partial_output(self) -> None:
        mapped = _map_exec_error(
            self._error(
                {
                    "error_type": "exec_timeout",
                    "timeout_s": 2,
                    "stdout": "before\n",
                    "stderr": "partial\n",
                },
                status=408,
            ),
            cmd=["sleep", "30"],
            timeout_s=2,
        )
        self.assertIsInstance(mapped, SandboxExecTimeout)
        self.assertEqual(mapped.stdout, "before\n")
        self.assertEqual(mapped.stderr, "partial\n")

    def test_image_not_found_rehydrates_across_gateway_404(self) -> None:
        mapped = _map_create_error(
            self._error(
                {
                    "error_type": "image_not_found",
                    "image": "missing:latest",
                    "error": "public image was not found",
                },
                status=404,
            )
        )
        self.assertIsInstance(mapped, ImageNotFoundError)
        self.assertEqual(mapped.reference, "missing:latest")


if __name__ == "__main__":
    unittest.main()
