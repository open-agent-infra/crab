from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from crab import RuncRuntime, RuncRuntimePaths, SandboxId
from crab.runtime.base import CommandResult, CommandRunner


class _RecordingRunner(CommandRunner):
    """Records every command; anything not seeded returns success."""

    def __init__(
        self,
        *,
        results_by_key: dict[tuple[str, ...], list[CommandResult]] | None = None,
    ) -> None:
        self.commands: list[tuple[str, ...]] = []
        self._results = {k: list(v) for k, v in (results_by_key or {}).items()}

    def run(
        self,
        command: list[str],
        *,
        cwd: Path | None = None,
        timeout_seconds: float | None = None,
    ) -> CommandResult:
        _ = cwd
        _ = timeout_seconds
        key = tuple(command)
        self.commands.append(key)
        queued = self._results.get(key)
        if queued:
            return queued.pop(0)
        return CommandResult(command=key, returncode=0, stdout="", stderr="")


class RuncRuntimeStartTests(unittest.TestCase):
    def _make_runtime(self, runner: CommandRunner, root: Path) -> RuncRuntime:
        runtime = RuncRuntime(
            command_runner=runner,
            paths=RuncRuntimePaths(
                state_root=root / "state",
                bundle_root=root / "bundles",
                metadata_root=root / "metadata",
                zfs_dataset_prefix="pool/crab",
            ),
        )
        runtime.launch(
            "runc",
            {"sandbox_id": "sbx-test", "bundle_path": str(root / "bundles" / "sbx-test")},
        )
        # Zero the stop grace window so stop's TERM->KILL escalation does not
        # sleep during unit tests.
        runtime._STOP_GRACE_S = 0.0
        return runtime

    def _cmds(self, root: Path) -> dict[str, tuple[str, ...]]:
        state = str(root / "state")
        bundle = str(root / "bundles" / "sbx-test")
        return {
            "create": ("runc", "--root", state, "create", "--bundle", bundle, "sbx-test"),
            "start": ("runc", "--root", state, "start", "sbx-test"),
            "delete": ("runc", "--root", state, "delete", "-f", "sbx-test"),
            "stop": ("runc", "--root", state, "kill", "sbx-test", "TERM"),
            "kill": ("runc", "--root", state, "kill", "sbx-test", "KILL"),
            "state": ("runc", "--root", state, "state", "sbx-test"),
            "resume": ("runc", "--root", state, "resume", "sbx-test"),
        }

    def test_start_relaunches_from_existing_bundle(self) -> None:
        with tempfile.TemporaryDirectory(prefix="crab_runc_start_") as tmp:
            root = Path(tmp)
            runner = _RecordingRunner()
            runtime = self._make_runtime(runner, root)
            c = self._cmds(root)
            runner.commands.clear()  # ignore launch's create/start

            runtime.stop(SandboxId("sbx-test"))
            runtime.start(SandboxId("sbx-test"))

            # start clears stale container state then re-creates and starts
            # from the same bundle.
            self.assertIn(c["delete"], runner.commands)
            self.assertIn(c["create"], runner.commands)
            self.assertIn(c["start"], runner.commands)
            self.assertLess(
                runner.commands.index(c["delete"]), runner.commands.index(c["create"])
            )
            self.assertLess(
                runner.commands.index(c["create"]), runner.commands.index(c["start"])
            )
            self.assertEqual(runtime.describe(SandboxId("sbx-test")).status, "running")

    def test_restart_is_resume_stop_then_start(self) -> None:
        with tempfile.TemporaryDirectory(prefix="crab_runc_start_") as tmp:
            root = Path(tmp)
            runner = _RecordingRunner()
            runtime = self._make_runtime(runner, root)
            c = self._cmds(root)
            runner.commands.clear()

            runtime.restart(SandboxId("sbx-test"))

            # resume (thaw, no-op if running) -> stop (TERM) -> start.
            self.assertIn(c["resume"], runner.commands)
            self.assertIn(c["stop"], runner.commands)
            self.assertIn(c["create"], runner.commands)
            self.assertIn(c["start"], runner.commands)
            self.assertLess(runner.commands.index(c["resume"]), runner.commands.index(c["stop"]))
            self.assertLess(runner.commands.index(c["stop"]), runner.commands.index(c["start"]))
            self.assertEqual(runtime.describe(SandboxId("sbx-test")).status, "running")

    def test_restart_tolerates_already_stopped_sandbox(self) -> None:
        """Restarting a sandbox that is already stopped must still bring it
        back: the stop step raises (nothing to signal) and restart proceeds to
        start."""
        with tempfile.TemporaryDirectory(prefix="crab_runc_start_") as tmp:
            root = Path(tmp)
            c = self._cmds(root)
            runner = _RecordingRunner(
                results_by_key={
                    c["stop"]: [
                        CommandResult(
                            command=c["stop"],
                            returncode=1,
                            stdout="",
                            stderr="container not running",
                        )
                    ]
                },
            )
            runtime = self._make_runtime(runner, root)
            runner.commands.clear()

            runtime.restart(SandboxId("sbx-test"))

            # Despite the failed stop, start ran and the sandbox is up.
            self.assertIn(c["create"], runner.commands)
            self.assertIn(c["start"], runner.commands)
            self.assertEqual(runtime.describe(SandboxId("sbx-test")).status, "running")

    def test_stop_escalates_to_sigkill_when_term_ignored(self) -> None:
        """The sandbox init runs as PID 1 (sleep infinity) which never handles
        SIGTERM, so the container keeps running after the graceful signal. stop
        must then escalate to SIGKILL so the sandbox actually stops."""
        with tempfile.TemporaryDirectory(prefix="crab_runc_start_") as tmp:
            root = Path(tmp)
            c = self._cmds(root)
            runner = _RecordingRunner(
                results_by_key={
                    c["state"]: [
                        CommandResult(
                            command=c["state"],
                            returncode=0,
                            stdout='{"status": "running", "pid": 1}',
                            stderr="",
                        )
                    ]
                },
            )
            runtime = self._make_runtime(runner, root)
            runner.commands.clear()

            runtime.stop(SandboxId("sbx-test"))

            self.assertIn(c["stop"], runner.commands)
            self.assertIn(c["kill"], runner.commands)
            self.assertLess(runner.commands.index(c["stop"]), runner.commands.index(c["kill"]))
            self.assertEqual(runtime.describe(SandboxId("sbx-test")).status, "stopped")

    def test_stop_skips_sigkill_when_container_exits_on_term(self) -> None:
        """When the container does exit promptly on SIGTERM, stop must not send
        the SIGKILL escalation."""
        with tempfile.TemporaryDirectory(prefix="crab_runc_start_") as tmp:
            root = Path(tmp)
            c = self._cmds(root)
            runner = _RecordingRunner(
                results_by_key={
                    c["state"]: [
                        CommandResult(
                            command=c["state"],
                            returncode=0,
                            stdout='{"status": "stopped", "pid": 0}',
                            stderr="",
                        )
                    ]
                },
            )
            runtime = self._make_runtime(runner, root)
            runner.commands.clear()

            runtime.stop(SandboxId("sbx-test"))

            self.assertIn(c["stop"], runner.commands)
            self.assertNotIn(c["kill"], runner.commands)
            self.assertEqual(runtime.describe(SandboxId("sbx-test")).status, "stopped")


if __name__ == "__main__":
    unittest.main()
