"""Real-host end-to-end for the daemon-mode merge/changeset surface
(PR-C2.2): a real DaemonServer over the Unix socket, the SDK connected
via Engine.connect (merge + changeset transported over RPC, typed
MergeError rehydration), plus the `crab sandbox merge` / `crab sandbox
changeset` CLI. Self-skipping outside the crab-dev VM."""
from __future__ import annotations

import contextlib
import io
import shutil
import tempfile
import time
import unittest
from pathlib import Path

from crab import Engine, EngineConfig, Sandbox
from crab.cli import commands as cli_commands
from crab.daemon.server import DaemonServer
from crab.merging import MergeError
from crab.models import MergeReport


def _real_stack_available() -> bool:
    import os

    if os.geteuid() != 0:
        return False
    return all(shutil.which(tool) is not None for tool in ("docker", "runc", "criu", "zfs"))


class MergeDaemonRealTests(unittest.TestCase):
    _IMAGE = "python:3.11-slim"

    def setUp(self) -> None:
        if not _real_stack_available():
            self.skipTest("docker/runc/criu/zfs/root not available")
        self._tmp = tempfile.TemporaryDirectory(prefix="crab_merged_e2e_")
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
                # Real inspector: the daemon-side changeset gate must see
                # genuine write events, or it may legitimately prove a
                # fresh fork clean and merge an empty changeset.
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

    def _settle(self, *sandboxes: Sandbox, timeout: float = 30.0) -> None:
        """Wait until the daemon engine's inspector observed each
        sandbox's mutations (the SDK side only holds a no-op shim)."""
        engine = self.daemon.require_engine()
        for sandbox in sandboxes:
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                snapshot = engine.system.inspector.inspect(sandbox.sandbox_id)
                if snapshot.filesystem_changed:
                    break
                time.sleep(0.5)
            else:
                self.fail(f"inspector never saw writes on {sandbox.sandbox_id}")

    def _forked_pair(self) -> tuple[Sandbox, Sandbox]:
        parent = Sandbox(image=self._IMAGE, engine=self.engine)
        self.addCleanup(parent.kill)
        self._run(parent, "mkdir -p /probe && echo keep-v1 > /probe/keep.txt")
        [fork] = parent.fork()
        self.addCleanup(fork.kill)
        return parent, fork

    def test_remote_merge_applies_and_changeset_rpc(self) -> None:
        parent, fork = self._forked_pair()
        self._run(fork, "echo fork-new > /probe/fork-new.txt")
        self._settle(fork)

        # Changeset over RPC (fork point resolved daemon-side).
        entries = fork.changeset()
        paths = {entry["path"] for entry in entries}
        self.assertIn("/probe/fork-new.txt", paths)

        report = parent.merge(fork, observations="append")
        self.assertIsInstance(report, MergeReport)
        self.assertIn("/probe/fork-new.txt", {entry.path for entry in report.applied})
        self.assertEqual(report.conflicted, ())
        self.assertEqual(self._run(parent, "cat /probe/fork-new.txt"), "fork-new")
        # C3 over RPC: the merge adopted the fork's history and the
        # journal read RPC serves it back.
        self.assertIsNotNone(report.observations)
        self.assertGreaterEqual(report.observations.consolidated, 1)
        rows = parent.actions(kind="observation")
        self.assertTrue(rows)
        self.assertEqual(
            {row["payload"]["fork_sandbox_id"] for row in rows},
            {str(fork.sandbox_id)},
        )

    def test_remote_merge_conflict_policy_and_typed_error(self) -> None:
        parent, fork = self._forked_pair()
        self._run(fork, "echo fork-side > /probe/keep.txt")
        self._run(parent, "echo source-side > /probe/keep.txt")
        self._settle(fork, parent)

        # fail_fast reports the conflict without writing.
        report = parent.merge(fork)
        self.assertEqual(
            [(entry.path, entry.reason) for entry in report.conflicted],
            [("/probe/keep.txt", "source_changed")],
        )
        self.assertEqual(self._run(parent, "cat /probe/keep.txt"), "source-side")

        # A merge refused daemon-side rehydrates as a typed MergeError.
        txn = parent.begin(label="block-merge")
        try:
            with self.assertRaises(MergeError):
                parent.merge(fork, policy="prefer_fork")
        finally:
            txn.abort()

        # prefer_fork lands the fork bytes once the txn is gone.
        report = parent.merge(fork, policy="prefer_fork")
        self.assertEqual(report.conflicted, ())
        self.assertEqual(self._run(parent, "cat /probe/keep.txt"), "fork-side")

    def test_cli_merge_and_changeset_flow(self) -> None:
        parent, fork = self._forked_pair()
        self._run(fork, "echo cli-new > /probe/cli-new.txt")
        self._settle(fork)

        def _cli(argv: list[str], *, expect_rc: int = 0) -> str:
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                rc = cli_commands.main(["--socket", str(self.socket_path), *argv])
            self.assertEqual(rc, expect_rc, msg=f"cli failed: {argv}: {stdout.getvalue()}")
            return stdout.getvalue()

        changeset_out = _cli(["sandbox", "changeset", str(fork.sandbox_id)])
        self.assertIn("added\t/probe/cli-new.txt", changeset_out)

        merge_out = _cli(
            ["sandbox", "merge", str(parent.sandbox_id), str(fork.sandbox_id)]
        )
        self.assertIn("conflicted=0", merge_out)
        self.assertEqual(self._run(parent, "cat /probe/cli-new.txt"), "cli-new")

        # C3 CLI: consolidate the fork's history, then read it back.
        consolidate_out = _cli(
            ["sandbox", "consolidate", str(parent.sandbox_id), str(fork.sandbox_id)]
        )
        self.assertIn("consolidated=", consolidate_out)
        actions_out = _cli(
            ["sandbox", "actions", str(parent.sandbox_id), "--kind", "observation"]
        )
        self.assertIn("observation", actions_out)
        self.assertIn(str(fork.sandbox_id), actions_out)


if __name__ == "__main__":
    unittest.main()
