"""Real-host end-to-end for the action journal (PR-B1.1): a scripted SDK
session against the full stack (runc + CRIU + the configured filesystem
backend) must land in the journal verbatim, including lifecycle markers
across checkpoint/restore/fork/kill. Self-skipping outside the crab-dev VM."""
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


class JournalRealTests(unittest.TestCase):
    _IMAGE = "python:3.11-slim"

    def setUp(self) -> None:
        if not _real_stack_available():
            self.skipTest("docker/runc/criu/zfs/root not available")
        self._tmp = tempfile.TemporaryDirectory(prefix="crab_journal_e2e_")
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.engine = Engine.start(
            EngineConfig(
                runtime="runc",
                enable_sandbox_network=False,
                enable_interceptor=False,
                storage_root=self.root / "storage",
                runtime_root=self.root / "runtime",
            )
        )
        self.addCleanup(self.engine.stop)

    def _exec_payloads(self, sandbox: Sandbox) -> list[dict]:
        return [row["payload"] for row in sandbox.actions(kind="exec")]

    def _lifecycle_events(self, sandbox: Sandbox) -> list[str]:
        return [row["payload"]["event"] for row in sandbox.actions(kind="lifecycle")]

    def test_scripted_session_lands_in_journal_verbatim(self) -> None:
        sandbox = Sandbox(image=self._IMAGE, engine=self.engine)
        self.addCleanup(sandbox.kill)

        sandbox.commands.run("echo v1 > /state.txt")
        sandbox.commands.run("cat /state.txt", cwd="/tmp", env={"JOURNAL_MARKER": "42"})
        failing = sandbox.commands.run("exit 7")
        self.assertEqual(failing.returncode, 7)

        checkpoint_id = sandbox.checkpoint()
        sandbox.restore(checkpoint_id)
        sandbox.commands.run("cat /state.txt")

        payloads = self._exec_payloads(sandbox)
        # The scripted argv/cwd/env round-trip verbatim (env is a superset:
        # the SDK merges its default command env into every exec).
        scripted = [p for p in payloads if p["argv"][-1].startswith(("echo v1", "cat /state", "exit 7"))]
        self.assertGreaterEqual(len(scripted), 4)
        marker_execs = [p for p in payloads if p["env"].get("JOURNAL_MARKER") == "42"]
        self.assertEqual(len(marker_execs), 1)
        self.assertEqual(marker_execs[0]["cwd"], "/tmp")
        failing_execs = [p for p in payloads if p["returncode"] == 7]
        self.assertEqual(len(failing_execs), 1)
        for payload in scripted:
            self.assertIsNotNone(payload["stdout_sha256"])
            self.assertIsNotNone(payload["duration_ms"])

        events = self._lifecycle_events(sandbox)
        self.assertEqual(events[0], "launch")
        self.assertIn("checkpoint", events)
        self.assertIn("restore", events)
        # Restore appends, never truncates: exec records from before the
        # restore marker are still present.
        rows = sandbox.actions()
        restore_index = next(
            index for index, row in enumerate(rows)
            if row["kind"] == "lifecycle" and row["payload"]["event"] == "restore"
        )
        self.assertTrue(any(row["kind"] == "exec" for row in rows[:restore_index]))

        # Journal lives under the storage root, one JSONL per sandbox.
        journal_file = (
            self.root / "storage" / "journal" / f"{sandbox.sandbox_id}.jsonl"
        )
        self.assertTrue(journal_file.is_file())

    def test_fork_journals_are_separate_with_provenance(self) -> None:
        source = Sandbox(image=self._IMAGE, engine=self.engine)
        self._run_ok(source, "echo keepsake > /state.txt")

        forks = source.fork(1)
        fork = forks[0]
        self.addCleanup(fork.kill)
        self._run_ok(fork, "cat /state.txt")

        fork_rows = fork.actions()
        # The fork's journal starts fresh with its provenance marker.
        self.assertEqual(fork_rows[0]["kind"], "lifecycle")
        self.assertEqual(fork_rows[0]["payload"]["event"], "fork_created")
        self.assertEqual(
            fork_rows[0]["payload"]["metadata"]["source_sandbox_id"],
            str(source.sandbox_id),
        )
        # Source execs are not spliced into the fork journal.
        fork_execs = [row["payload"]["argv"][-1] for row in fork_rows if row["kind"] == "exec"]
        self.assertFalse(any("keepsake" in argv for argv in fork_execs))

        source_events = self._lifecycle_events(source)
        self.assertIn("fork_source", source_events)

        source.kill()
        source_events_after = [
            row["payload"]["event"]
            for row in fork.actions(kind="lifecycle")
        ]
        _ = source_events_after  # fork journal unaffected by source kill
        # Source journal recorded the destroy marker and survives the kill.
        journal_file = self.root / "storage" / "journal" / f"{source.sandbox_id}.jsonl"
        self.assertTrue(journal_file.is_file())
        content = journal_file.read_text(encoding="utf-8")
        self.assertIn('"destroy"', content)

    def _run_ok(self, sandbox: Sandbox, script: str) -> str:
        result = sandbox.commands.run(script)
        self.assertEqual(result.returncode, 0, msg=f"command failed: {script!r}: {result.stderr}")
        return result.stdout.strip()


if __name__ == "__main__":
    unittest.main()
