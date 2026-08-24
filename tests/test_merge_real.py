"""Real-host end-to-end for the C2 fs merge engine: the roadmap exit
criteria — disjoint edits merge cleanly, overlapping edits hit the
policy, and a mid-apply failure rolls the source back to its pre-merge
state — on both zfs and btrfs. Exercises the real fork-point snapshot
content roots (zfs .zfs/snapshot access, btrfs snapshot subvolumes) and
the default ignore prefixes against genuine CRIU dump noise.
Self-skipping outside the crab-dev VM."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from crab import Engine, EngineConfig, Sandbox
from crab import merging
from crab.merging import MergeError

_BTRFS_ROOT = Path("/var/lib/crab/btrfs")


def _real_stack_available() -> bool:
    import os

    if os.geteuid() != 0:
        return False
    return all(shutil.which(tool) is not None for tool in ("docker", "runc", "criu", "zfs"))


def _btrfs_root_available() -> bool:
    if shutil.which("btrfs") is None or not _BTRFS_ROOT.is_dir():
        return False
    result = subprocess.run(
        ["stat", "-f", "-c", "%T", str(_BTRFS_ROOT)],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0 and result.stdout.strip() == "btrfs"


def _overlay_available() -> bool:
    if not _btrfs_root_available():
        return False
    try:
        filesystems = Path("/proc/filesystems").read_text(encoding="utf-8")
    except OSError:
        return False
    return any(line.split() and line.split()[-1] == "overlay" for line in filesystems.splitlines())


_SETUP = (
    "mkdir -p /probe && printf 'one\\ntwo\\nthree\\nfour\\nfive\\n' > /probe/doc.txt "
    "&& echo keep-v1 > /probe/keep.txt && echo shared-v1 > /probe/shared.txt"
)


class _MergeRealMixin:
    """Shared scenarios; concrete classes bind the backend. Not a
    TestCase itself so discovery does not run the shared tests twice."""

    _IMAGE = "python:3.11-slim"
    _BACKEND = "zfs"

    def setUp(self) -> None:
        if not _real_stack_available():
            self.skipTest("docker/runc/criu/zfs/root not available")
        self._tmp = tempfile.TemporaryDirectory(prefix="crab_merge_e2e_")
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def _engine(self) -> Engine:
        engine = Engine.start(
            EngineConfig(
                runtime="runc",
                enable_sandbox_network=False,
                enable_interceptor=False,
                storage_root=self.root / "storage",
                runtime_root=self.root / "runtime",
                filesystem_backend=self._BACKEND,
                host_inspector_launch_mode="thread",
            )
        )
        self.addCleanup(engine.stop)
        return engine

    def _run(self, sandbox: Sandbox, script: str) -> str:
        result = sandbox.commands.run(script)
        self.assertEqual(result.returncode, 0, msg=f"command failed: {script!r}: {result.stderr}")
        return result.stdout.strip()

    def _forked_pair(self, engine: Engine) -> tuple[Sandbox, Sandbox]:
        parent = Sandbox(image=self._IMAGE, engine=engine)
        self.addCleanup(parent.kill)
        self._run(parent, _SETUP)
        [fork] = parent.fork()
        self.addCleanup(fork.kill)
        return parent, fork

    def _settle(self, engine: Engine, *sandboxes: Sandbox, timeout: float = 30.0) -> None:
        """Wait until the real inspector has observed each sandbox's
        mutations. Without this the gate may legitimately prove a
        just-forked sandbox clean (its base IS the latest filesystem
        checkpoint) and merge an empty changeset."""
        for sandbox in sandboxes:
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                snapshot = engine.system.inspector.inspect(sandbox.sandbox_id)
                if snapshot.filesystem_changed:
                    break
                time.sleep(0.5)
            else:
                self.fail(f"inspector never saw writes on {sandbox.sandbox_id}")

    def test_disjoint_edits_merge_cleanly(self) -> None:
        engine = self._engine()
        parent, fork = self._forked_pair(engine)
        self._run(fork, "echo fork-new > /probe/fork-new.txt && echo keep-v2 > /probe/keep.txt")
        self._run(parent, "echo src-only > /probe/src-only.txt")
        self._settle(engine, fork, parent)

        report = parent.merge(fork)

        applied = {entry.path for entry in report.applied}
        self.assertIn("/probe/fork-new.txt", applied)
        self.assertIn("/probe/keep.txt", applied)
        self.assertEqual(report.conflicted, ())
        self.assertFalse(report.rolled_back)
        # The parent-directory mtime churn is classified, not merged.
        self.assertIn(
            ("/probe", "dir_touch"),
            {(entry.path, entry.reason) for entry in report.skipped},
        )
        # Fork content landed, the source-only edit survived — verified
        # through the container itself (the sandbox was paused/resumed
        # around the host-side apply window).
        self.assertEqual(self._run(parent, "cat /probe/fork-new.txt"), "fork-new")
        self.assertEqual(self._run(parent, "cat /probe/keep.txt"), "keep-v2")
        self.assertEqual(self._run(parent, "cat /probe/src-only.txt"), "src-only")
        # The fork stays alive and untouched.
        self.assertEqual(self._run(fork, "cat /probe/keep.txt"), "keep-v2")
        marker = [
            record.payload
            for record in engine.system.journal.entries(parent.sandbox_id, kind="lifecycle")
            if record.payload.get("event") == "merge"
        ][-1]
        self.assertTrue(marker["metadata"]["succeeded"])
        self.assertEqual(marker["metadata"]["policy"], "fail_fast")

    def test_text_merge_composes_disjoint_line_edits(self) -> None:
        engine = self._engine()
        parent, fork = self._forked_pair(engine)
        self._run(fork, "sed -i 's/five/FIVE/' /probe/doc.txt")
        self._run(parent, "sed -i 's/one/ONE/' /probe/doc.txt")
        self._settle(engine, fork, parent)

        report = parent.merge(fork, policy="text_merge")

        doc_entries = [entry for entry in report.applied if entry.path == "/probe/doc.txt"]
        self.assertEqual(len(doc_entries), 1)
        self.assertTrue(doc_entries[0].merged)
        self.assertEqual(
            self._run(parent, "cat /probe/doc.txt"),
            "ONE\ntwo\nthree\nfour\nFIVE",
        )

    def test_mid_apply_failure_rolls_back_to_pre_merge_state(self) -> None:
        engine = self._engine()
        parent, fork = self._forked_pair(engine)
        self._run(fork, "echo ok > /probe/aa-ok.txt && echo boom > /probe/zz-boom.txt")
        self._settle(engine, fork)
        before = self._run(parent, "ls /probe && cat /probe/keep.txt")

        original_copy_node = merging._copy_node
        copies: list[str] = []

        def flaky_copy_node(src, dest):
            copies.append(str(dest))
            if len(copies) == 2:
                raise RuntimeError("injected merge failure")
            return original_copy_node(src, dest)

        with mock.patch.object(merging, "_copy_node", side_effect=flaky_copy_node):
            with self.assertRaises(MergeError) as caught:
                parent.merge(fork)

        report = caught.exception.report
        self.assertIsNotNone(report)
        self.assertTrue(report.rolled_back)
        self.assertEqual(report.applied, ())
        # /probe/aa-ok.txt landed before the injected failure and must
        # have been undone from the transient snapshot.
        self.assertEqual(self._run(parent, "ls /probe && cat /probe/keep.txt"), before)
        self._assert_no_transient_merge_snapshot(engine, parent)

    def _assert_no_transient_merge_snapshot(self, engine: Engine, sandbox: Sandbox) -> None:
        dataset = engine.system.runtime.dataset_name_for(sandbox.sandbox_id)
        if self._BACKEND == "zfs":
            result = subprocess.run(
                ["zfs", "list", "-H", "-t", "snapshot", "-o", "name", "-r", dataset],
                capture_output=True,
                text=True,
                check=False,
            )
            leftovers = [line for line in result.stdout.splitlines() if "@merge-" in line]
        else:
            base = Path(dataset)
            leftovers = [str(path) for path in base.parent.glob(f"{base.name}@merge-*")]
        self.assertEqual(leftovers, [], "transient @merge-* snapshot leaked")


class ZfsMergeRealTests(_MergeRealMixin, unittest.TestCase):
    _BACKEND = "zfs"

    def test_conflict_hits_policy_fail_fast_then_prefer_fork(self) -> None:
        engine = self._engine()
        parent, fork = self._forked_pair(engine)
        self._run(fork, "echo fork-side > /probe/shared.txt")
        self._run(parent, "echo source-side > /probe/shared.txt")
        self._settle(engine, fork, parent)

        report = parent.merge(fork)
        self.assertEqual(
            [(entry.path, entry.reason) for entry in report.conflicted],
            [("/probe/shared.txt", "source_changed")],
        )
        self.assertEqual(report.applied, ())
        self.assertEqual(self._run(parent, "cat /probe/shared.txt"), "source-side")

        report = parent.merge(fork, policy="prefer_fork")
        self.assertEqual(report.conflicted, ())
        self.assertEqual(self._run(parent, "cat /probe/shared.txt"), "fork-side")


class BtrfsMergeRealTests(_MergeRealMixin, unittest.TestCase):
    _BACKEND = "btrfs"

    def setUp(self) -> None:
        super().setUp()
        if not _btrfs_root_available():
            self.skipTest(f"btrfs root {_BTRFS_ROOT} not available")


class OverlayMergeRealTests(_MergeRealMixin, unittest.TestCase):
    """A2 engine-level leg: C2's three-way merge consumes the overlay
    provider's changeset (whiteout decoding) and snapshot content roots
    (kernel-merged ro views) through the unchanged interface."""

    _BACKEND = "overlay"

    def setUp(self) -> None:
        super().setUp()
        if not _overlay_available():
            self.skipTest(f"overlay backend prerequisites missing at {_BTRFS_ROOT}")


if __name__ == "__main__":
    unittest.main()
