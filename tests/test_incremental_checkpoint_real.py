"""Real runc/CRIU/ZFS coverage for requested incremental checkpoints.

Normal discovery skips this module. Run it in crab-vm with::

    CRAB_REAL_HOST_TESTS=1 python3 -m unittest -v \
        tests.test_incremental_checkpoint_real
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path

from crab import Engine, EngineConfig, Sandbox, SandboxSnapshot, SchedulerConfig
from crab.ids import CheckpointId
from crab.models import JobStatus, utc_now


def _available() -> bool:
    return bool(
        os.environ.get("CRAB_REAL_HOST_TESTS")
        and os.geteuid() == 0
        and all(
            shutil.which(tool) is not None
            for tool in ("docker", "runc", "criu", "zfs")
        )
    )


def _directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for candidate in path.rglob("*"):
        try:
            if candidate.is_file():
                total += candidate.stat().st_size
        except FileNotFoundError:
            continue
    return total


@unittest.skipUnless(_available(), "requires CRAB_REAL_HOST_TESTS=1 and runc/ZFS")
class IncrementalCheckpointRealTests(unittest.TestCase):
    _IMAGE = "python:3.11-slim"

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="crab_incremental_e2e_")
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        self.engine = Engine.start(
            EngineConfig(
                runtime="runc",
                enable_sandbox_network=False,
                enable_interceptor=False,
                storage_root=root / "storage",
                runtime_root=root / "runtime",
                scheduler_config=SchedulerConfig(
                    min_checkpoint_interval_seconds=0.0,
                    force_checkpoint_after_seconds=0.0,
                    require_change_signal=True,
                    incremental_process_enabled=True,
                    full_process_checkpoint_interval=8,
                    max_process_chain_length=16,
                ),
            )
        )
        self.addCleanup(self.engine.stop)

    def _set_changed(
        self,
        sandbox: Sandbox,
        *,
        last_checkpoint_at=None,
    ) -> None:
        upsert = getattr(self.engine.system.inspector, "upsert_snapshot")
        upsert(
            SandboxSnapshot(
                sandbox_id=sandbox.sandbox_id,
                runtime_name="runc",
                is_running=True,
                process_changed=True,
                filesystem_changed=True,
                observed_at=utc_now(),
                last_checkpoint_at=last_checkpoint_at,
            )
        )

    def _wait_for_generation(self, sandbox: Sandbox, generation: int) -> None:
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            result = sandbox.commands.run(
                [
                    "sh",
                    "-lc",
                    "test -f /tmp/worker.generation && "
                    "cat /tmp/worker.generation || true",
                ],
                check=True,
            )
            if result.stdout.strip() == str(generation):
                return
            time.sleep(0.1)
        self.fail(f"worker did not reach generation {generation}")

    def _dirty_worker(
        self,
        sandbox: Sandbox,
        *,
        generation: int,
        page_count: int,
    ) -> None:
        sandbox.files.write("/tmp/worker.dirty-pages", f"{page_count}\n")
        sandbox.commands.run(
            ["sh", "-lc", "kill -USR1 $(cat /tmp/worker.pid)"],
            check=True,
        )
        self._wait_for_generation(sandbox, generation)

    def test_logical_chain_restores_and_standalone_full_resets_to_anchor(self) -> None:
        sandbox = Sandbox(image=self._IMAGE, engine=self.engine, network=False)
        self.addCleanup(sandbox.kill)
        sandbox.files.write(
            "/tmp/incremental-worker.py",
            """\
import os
import signal
import time

pages = bytearray(64 * 1024 * 1024)
generation = 0

def dirty(_signum, _frame):
    global generation
    generation += 1
    try:
        with open('/tmp/worker.dirty-pages', encoding='utf-8') as stream:
            page_count = int(stream.read().strip())
    except (OSError, ValueError):
        page_count = len(pages) // 4096
    dirty_bytes = min(len(pages), max(0, page_count) * 4096)
    for offset in range(0, dirty_bytes, 4096):
        pages[offset] = (pages[offset] + generation) % 256
    with open('/tmp/worker.generation', 'w', encoding='utf-8') as stream:
        stream.write(str(generation))

signal.signal(signal.SIGUSR1, dirty)
with open('/tmp/worker.pid', 'w', encoding='utf-8') as stream:
    stream.write(str(os.getpid()))
with open('/tmp/worker.generation', 'w', encoding='utf-8') as stream:
    stream.write('0')
while True:
    time.sleep(1)
""",
        )
        sandbox.commands.run(
            [
                "sh",
                "-lc",
                "nohup python /tmp/incremental-worker.py "
                ">/tmp/incremental-worker.log 2>&1 &",
            ],
            check=True,
        )
        self._wait_for_generation(sandbox, 0)
        sandbox.files.write("/state.txt", "baseline\n")
        self._set_changed(sandbox)

        baseline_started = time.perf_counter()
        baseline = self.engine.system.checkpoint_requested(
            sandbox.sandbox_id,
            checkpoint_id="ckpt-real-anchor",
        )
        baseline_seconds = time.perf_counter() - baseline_started
        self.assertEqual(baseline.status, JobStatus.SUCCEEDED)
        assert baseline.manifest is not None
        self.assertEqual(baseline.manifest.process_kind, "full")
        self.assertEqual(
            baseline.manifest.metadata["incremental_chain_role"],
            "anchor",
        )
        baseline_pre_dump = Path(
            str(
                self.engine.runtime.pre_dump_location(
                    sandbox.sandbox_id, baseline.checkpoint_id
                )
            )
        )
        baseline_process = Path(
            str(
                self.engine.runtime.process_checkpoint_location(
                    sandbox.sandbox_id, baseline.checkpoint_id
                )
            )
        )
        self.assertTrue(baseline_pre_dump.is_dir())

        incremental_metrics: list[tuple[int, float, int]] = []
        parent = baseline
        for generation, page_count in ((1, 164), (2, 1638), (3, 16384)):
            self._dirty_worker(
                sandbox,
                generation=generation,
                page_count=page_count,
            )
            sandbox.files.write("/state.txt", f"incremental-{generation}\n")
            self._set_changed(sandbox, last_checkpoint_at=parent.finished_at)

            incremental_started = time.perf_counter()
            incremental = self.engine.system.checkpoint_requested(
                sandbox.sandbox_id,
                checkpoint_id=f"ckpt-real-incremental-{generation}",
            )
            incremental_seconds = time.perf_counter() - incremental_started
            self.assertEqual(incremental.status, JobStatus.SUCCEEDED)
            assert incremental.manifest is not None
            self.assertEqual(incremental.manifest.process_kind, "incremental")
            self.assertEqual(
                incremental.manifest.parent_checkpoint_id,
                parent.checkpoint_id,
            )
            self.assertEqual(
                incremental.manifest.metadata["checkpoint_materialization"],
                "incremental",
            )
            incremental_pre_dump = Path(
                str(
                    self.engine.runtime.pre_dump_location(
                        sandbox.sandbox_id, incremental.checkpoint_id
                    )
                )
            )
            incremental_process = Path(
                str(
                    self.engine.runtime.process_checkpoint_location(
                        sandbox.sandbox_id, incremental.checkpoint_id
                    )
                )
            )
            self.assertTrue(incremental_pre_dump.is_dir())
            incremental_metrics.append(
                (
                    page_count,
                    incremental_seconds,
                    _directory_size(incremental_pre_dump)
                    + _directory_size(incremental_process),
                )
            )
            parent = incremental

        self.assertEqual(
            self.engine.system._stored_process_chain_length(
                sandbox.sandbox_id,
                incremental.checkpoint_id,
            ),
            3,
        )

        sandbox.files.write("/state.txt", "after-checkpoint\n")
        self._dirty_worker(
            sandbox,
            generation=4,
            page_count=164,
        )
        incremental_restore_started = time.perf_counter()
        sandbox.restore(incremental.checkpoint_id)
        incremental_restore_seconds = (
            time.perf_counter() - incremental_restore_started
        )
        self.assertEqual(sandbox.files.read("/state.txt"), "incremental-3\n")
        self.assertEqual(sandbox.files.read("/tmp/worker.generation"), "3")
        alive = sandbox.commands.run(
            ["sh", "-lc", "kill -0 $(cat /tmp/worker.pid)"],
            check=True,
        )
        self.assertEqual(alive.returncode, 0)

        manual_started = time.perf_counter()
        manual_id = CheckpointId(
            sandbox.checkpoint(label="standalone", leave_running=True)
        )
        manual_seconds = time.perf_counter() - manual_started
        manual_manifest = self.engine.system.storage.get_manifest(
            sandbox.sandbox_id, manual_id
        )
        manual_pre_dump = Path(
            str(
                self.engine.runtime.pre_dump_location(
                    sandbox.sandbox_id, manual_id
                )
            )
        )
        self.assertFalse(manual_pre_dump.exists())
        manual_process = Path(
            str(
                self.engine.runtime.process_checkpoint_location(
                    sandbox.sandbox_id, manual_id
                )
            )
        )

        self._dirty_worker(
            sandbox,
            generation=4,
            page_count=164,
        )
        sandbox.files.write("/state.txt", "reset-anchor\n")
        self._set_changed(
            sandbox,
            last_checkpoint_at=manual_manifest.created_at,
        )
        reset = self.engine.system.checkpoint_requested(
            sandbox.sandbox_id,
            checkpoint_id="ckpt-real-reset-anchor",
        )
        self.assertEqual(reset.status, JobStatus.SUCCEEDED)
        assert reset.manifest is not None
        self.assertEqual(reset.manifest.process_kind, "full")
        self.assertEqual(
            reset.manifest.metadata["incremental_parent_reset_reason"],
            "missing_pre_dump",
        )
        self.assertEqual(
            reset.manifest.metadata["incremental_parent_candidate"],
            str(manual_id),
        )

        standalone_restore_started = time.perf_counter()
        sandbox.restore(manual_id)
        standalone_restore_seconds = (
            time.perf_counter() - standalone_restore_started
        )
        self.assertEqual(sandbox.files.read("/tmp/worker.generation"), "3")

        metric_text = " ".join(
            f"node-{page_count}p={seconds:.3f}s/{size_bytes}B"
            for page_count, seconds, size_bytes in incremental_metrics
        )
        print(
            "incremental checkpoint probe: "
            f"anchor={baseline_seconds:.3f}s/"
            f"{_directory_size(baseline_pre_dump) + _directory_size(baseline_process)}B "
            f"{metric_text} "
            f"standalone={manual_seconds:.3f}s/"
            f"{_directory_size(manual_process)}B "
            f"restore-chain3={incremental_restore_seconds:.3f}s "
            f"restore-standalone={standalone_restore_seconds:.3f}s"
        )


if __name__ == "__main__":
    unittest.main()
