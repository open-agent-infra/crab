"""Real-host end-to-end for fork changeset extraction (PR-C1): the roadmap
exit criteria — the changeset matches a scripted mutation list exactly on
zfs and btrfs, and the inspector gate skips the diff when nothing was
written since the base checkpoint. Engines run the real (thread-mode)
eBPF host inspector so the gate sees genuine filesystem signals.
Self-skipping outside the crab-dev VM."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

from crab import Engine, EngineConfig, Sandbox
from crab.ids import CheckpointId

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


# The scripted mutation list from the roadmap exit criteria: modify,
# delete, rename, create, mkdir tree, nested create — all under /probe.
_SETUP = (
    "mkdir -p /probe && echo v1 > /probe/mod.txt && echo v1 > /probe/del.txt "
    "&& echo v1 > /probe/old-name.txt"
)
_MUTATIONS = (
    "echo v2 > /probe/mod.txt",                    # modify file
    "rm /probe/del.txt",                           # delete file
    "mv /probe/old-name.txt /probe/new-name.txt",  # rename file
    "echo new > /probe/created.txt",               # create file
    "mkdir -p /probe/tree/a",                      # mkdir tree
    "echo nested > /probe/tree/a/deep.txt",        # nested create
)
# Raw truth from the backend diff: the mutated paths plus the parent
# directory whose entries changed. Identical on zfs and btrfs.
_EXPECTED = [
    {"path": "/probe", "change": "modified"},
    {"path": "/probe/created.txt", "change": "added"},
    {"path": "/probe/del.txt", "change": "removed"},
    {"path": "/probe/mod.txt", "change": "modified"},
    {"path": "/probe/new-name.txt", "change": "renamed", "renamed_from": "/probe/old-name.txt"},
    {"path": "/probe/tree", "change": "added"},
    {"path": "/probe/tree/a", "change": "added"},
    {"path": "/probe/tree/a/deep.txt", "change": "added"},
]
# A leave-running CRIU dump scratches transient state through the
# container's /tmp after the filesystem snapshot is taken, so windows
# that start at a checkpoint of a running sandbox carry this one extra
# mtime-churn entry (raw truth by design; C2's ignore policies filter
# it). Fork windows start at the clone and stay clean.
_DUMP_NOISE = {"path": "/tmp", "change": "modified"}
_EXPECTED_AFTER_DUMP = _EXPECTED + [_DUMP_NOISE]


class _ChangesetRealBase(unittest.TestCase):
    _IMAGE = "python:3.11-slim"

    def setUp(self) -> None:
        if not _real_stack_available():
            self.skipTest("docker/runc/criu/zfs/root not available")
        self._tmp = tempfile.TemporaryDirectory(prefix="crab_changeset_e2e_")
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def _engine(self, *, filesystem_backend: str = "zfs") -> Engine:
        engine = Engine.start(
            EngineConfig(
                runtime="runc",
                enable_sandbox_network=False,
                enable_interceptor=False,
                storage_root=self.root / "storage",
                runtime_root=self.root / "runtime",
                filesystem_backend=filesystem_backend,
                host_inspector_launch_mode="thread",
            )
        )
        self.addCleanup(engine.stop)
        return engine

    def _run(self, sandbox: Sandbox, script: str) -> str:
        result = sandbox.commands.run(script)
        self.assertEqual(result.returncode, 0, msg=f"command failed: {script!r}: {result.stderr}")
        return result.stdout.strip()

    def _wait_filesystem_changed(self, engine: Engine, sandbox: Sandbox, *, timeout: float = 30.0) -> None:
        """Wait for the real inspector to surface the mutation events —
        the gate is an optimization and must observe the writes before a
        changeset call can trust a skip decision."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            snapshot = engine.system.inspector.inspect(sandbox.sandbox_id)
            if snapshot.filesystem_changed:
                return
            time.sleep(0.5)
        self.fail("inspector never reported filesystem_changed after mutations")


class ZfsChangesetRealTests(_ChangesetRealBase):
    def test_txn_base_changeset_matches_mutations(self) -> None:
        engine = self._engine()
        sandbox = Sandbox(image=self._IMAGE, engine=engine)
        self.addCleanup(sandbox.kill)
        self._run(sandbox, _SETUP)

        txn = sandbox.begin(label="changeset-e2e")
        for mutation in _MUTATIONS:
            result = txn.exec(mutation)
            self.assertEqual(result.returncode, 0, msg=f"mutation failed: {mutation!r}: {result.stderr}")
        self._wait_filesystem_changed(engine, sandbox)

        self.assertEqual(sandbox.changeset(since=txn.base_checkpoint_id), _EXPECTED_AFTER_DUMP)

        # The journal audit marker rode along, tagged to the open txn.
        records = engine.system.journal.entries(sandbox.sandbox_id, kind="lifecycle")
        marker = [r for r in records if r.payload.get("event") == "changeset"][-1]
        self.assertEqual(marker.payload["metadata"]["entry_count"], len(_EXPECTED_AFTER_DUMP))
        self.assertFalse(marker.payload["metadata"]["skipped_by_gate"])
        self.assertEqual(marker.txn_id, txn.txn_id)
        txn.commit()

    def test_fork_changeset_matches_mutations(self) -> None:
        engine = self._engine()
        parent = Sandbox(image=self._IMAGE, engine=engine)
        self.addCleanup(parent.kill)
        self._run(parent, _SETUP)

        [fork] = parent.fork()
        self.addCleanup(fork.kill)
        for mutation in _MUTATIONS:
            self._run(fork, mutation)
        self._wait_filesystem_changed(engine, fork)

        # since=None resolves the fork point from the fork_created marker
        # and diffs against the snapshot planted on the fork's dataset.
        self.assertEqual(fork.changeset(), _EXPECTED)

    def test_fast_path_skips_diff_then_misses_after_write(self) -> None:
        engine = self._engine()
        sandbox = Sandbox(image=self._IMAGE, engine=engine)
        self.addCleanup(sandbox.kill)
        self._run(sandbox, _SETUP)
        ckpt = CheckpointId(sandbox.checkpoint())

        # No writes since the checkpoint: the gate proves the diff empty.
        deadline = time.monotonic() + 15.0
        while True:
            result = engine.system.changeset_since(sandbox.sandbox_id, ckpt)
            if result.skipped_by_gate or time.monotonic() >= deadline:
                break
            time.sleep(0.5)
        self.assertTrue(result.skipped_by_gate, "gate never proved the sandbox clean after checkpoint")
        self.assertEqual(result.entries, ())

        # A single write flips the signal and the authoritative diff runs.
        self._run(sandbox, "echo late > /probe/late.txt")
        self._wait_filesystem_changed(engine, sandbox)
        result = engine.system.changeset_since(sandbox.sandbox_id, ckpt)
        self.assertFalse(result.skipped_by_gate)
        self.assertEqual(
            [entry.to_json() for entry in result.entries],
            [
                {"path": "/probe", "change": "modified"},
                {"path": "/probe/late.txt", "change": "added"},
                _DUMP_NOISE,
            ],
        )


class BtrfsChangesetRealTests(_ChangesetRealBase):
    def setUp(self) -> None:
        super().setUp()
        if not _btrfs_root_available():
            self.skipTest(f"btrfs root {_BTRFS_ROOT} not available")

    def test_checkpoint_base_changeset_matches_mutations(self) -> None:
        engine = self._engine(filesystem_backend="btrfs")
        sandbox = Sandbox(image=self._IMAGE, engine=engine)
        self.addCleanup(sandbox.kill)
        self._run(sandbox, _SETUP)
        ckpt = sandbox.checkpoint()

        for mutation in _MUTATIONS:
            self._run(sandbox, mutation)
        self._wait_filesystem_changed(engine, sandbox)

        self.assertEqual(sandbox.changeset(since=ckpt), _EXPECTED_AFTER_DUMP)

    def test_fork_changeset_matches_mutations(self) -> None:
        engine = self._engine(filesystem_backend="btrfs")
        parent = Sandbox(image=self._IMAGE, engine=engine)
        self.addCleanup(parent.kill)
        self._run(parent, _SETUP)

        [fork] = parent.fork()
        self.addCleanup(fork.kill)
        for mutation in _MUTATIONS:
            self._run(fork, mutation)
        self._wait_filesystem_changed(engine, fork)

        self.assertEqual(fork.changeset(), _EXPECTED)


class OverlayChangesetRealTests(_ChangesetRealBase):
    """A2 engine-level leg: the scripted mutation list produces the SAME
    raw-truth changeset on overlay — _SETUP runs after launch so its
    files live in the upper (the mv is a physical in-upper rename with
    attribution), and the dump's /tmp scratch arrives as a copy-up that
    the translation pass reports `modified` like zfs/btrfs mtime churn."""

    def setUp(self) -> None:
        super().setUp()
        if not _overlay_available():
            self.skipTest(f"overlay backend prerequisites missing at {_BTRFS_ROOT}")

    def test_checkpoint_base_changeset_matches_mutations(self) -> None:
        engine = self._engine(filesystem_backend="overlay")
        sandbox = Sandbox(image=self._IMAGE, engine=engine)
        self.addCleanup(sandbox.kill)
        self._run(sandbox, _SETUP)
        ckpt = sandbox.checkpoint()

        for mutation in _MUTATIONS:
            self._run(sandbox, mutation)
        self._wait_filesystem_changed(engine, sandbox)

        self.assertEqual(sandbox.changeset(since=ckpt), _EXPECTED_AFTER_DUMP)

    def test_fork_changeset_matches_mutations(self) -> None:
        engine = self._engine(filesystem_backend="overlay")
        parent = Sandbox(image=self._IMAGE, engine=engine)
        self.addCleanup(parent.kill)
        self._run(parent, _SETUP)

        [fork] = parent.fork()
        self.addCleanup(fork.kill)
        for mutation in _MUTATIONS:
            self._run(fork, mutation)
        self._wait_filesystem_changed(engine, fork)

        self.assertEqual(fork.changeset(), _EXPECTED)


if __name__ == "__main__":
    unittest.main()
