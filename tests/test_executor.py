from __future__ import annotations

import time
import threading
import unittest

from crab import CRExecutor, ExecutorConfig, InMemoryTelemetrySink
from crab.ids import CheckpointId, JobId, SandboxId
from crab.models import (
    CheckpointJob,
    CheckpointManifest,
    CheckpointResult,
    JobStatus,
    RestoreJob,
    RestoreResult,
    utc_now,
)


class SlowCheckpointWorker:
    def __init__(self, sleep_s: float = 0.05):
        self.sleep_s = sleep_s
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.completed_job_ids: list[str] = []

    def checkpoint(self, job: CheckpointJob) -> CheckpointResult:
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(self.sleep_s)
        now = utc_now()
        ckpt_id = CheckpointId(f"ckpt-{job.job_id.value}")
        with self._lock:
            self.completed_job_ids.append(job.job_id.value)
            self.active -= 1
        manifest = CheckpointManifest(
            schema_version="v1",
            checkpoint_id=ckpt_id,
            sandbox_id=job.sandbox_id,
            created_at=now,
            runtime_name="test",
            runtime_version="1",
            process_artifacts=[],
            filesystem_artifacts=[],
            metadata={},
        ).with_integrity()
        return CheckpointResult(
            job_id=job.job_id,
            sandbox_id=job.sandbox_id,
            checkpoint_id=ckpt_id,
            status=JobStatus.SUCCEEDED,
            started_at=now,
            finished_at=utc_now(),
            manifest=manifest,
        )


class FlakyCheckpointWorker:
    def __init__(self):
        self.attempts = 0

    def checkpoint(self, job: CheckpointJob) -> CheckpointResult:
        self.attempts += 1
        now = utc_now()
        ckpt_id = CheckpointId("ckpt-flaky")
        if self.attempts == 1:
            return CheckpointResult(
                job_id=job.job_id,
                sandbox_id=job.sandbox_id,
                checkpoint_id=ckpt_id,
                status=JobStatus.FAILED,
                started_at=now,
                finished_at=utc_now(),
                manifest=None,
                message="first attempt failed",
            )
        manifest = CheckpointManifest(
            schema_version="v1",
            checkpoint_id=ckpt_id,
            sandbox_id=job.sandbox_id,
            created_at=now,
            runtime_name="test",
            runtime_version="1",
            process_artifacts=[],
            filesystem_artifacts=[],
            metadata={},
        ).with_integrity()
        return CheckpointResult(
            job_id=job.job_id,
            sandbox_id=job.sandbox_id,
            checkpoint_id=ckpt_id,
            status=JobStatus.SUCCEEDED,
            started_at=now,
            finished_at=utc_now(),
            manifest=manifest,
        )


class BlockingCheckpointWorker:
    def __init__(self):
        self._lock = threading.Lock()
        self._release_events: dict[str, threading.Event] = {}
        self._started_events: dict[str, threading.Event] = {}
        self.started_job_ids: list[str] = []

    def checkpoint(self, job: CheckpointJob) -> CheckpointResult:
        with self._lock:
            started_event = self._started_events.setdefault(job.job_id.value, threading.Event())
            release_event = self._release_events.setdefault(job.job_id.value, threading.Event())
            self.started_job_ids.append(job.job_id.value)
            started_event.set()
        if not release_event.wait(timeout=5.0):
            raise TimeoutError(f"timed out waiting to release checkpoint job {job.job_id.value}")
        now = utc_now()
        ckpt_id = CheckpointId(f"ckpt-{job.job_id.value}")
        manifest = CheckpointManifest(
            schema_version="v1",
            checkpoint_id=ckpt_id,
            sandbox_id=job.sandbox_id,
            created_at=now,
            runtime_name="test",
            runtime_version="1",
            process_artifacts=[],
            filesystem_artifacts=[],
            metadata={},
        ).with_integrity()
        return CheckpointResult(
            job_id=job.job_id,
            sandbox_id=job.sandbox_id,
            checkpoint_id=ckpt_id,
            status=JobStatus.SUCCEEDED,
            started_at=now,
            finished_at=utc_now(),
            manifest=manifest,
        )

    def wait_started(self, job_id: str, timeout: float = 2.0) -> bool:
        with self._lock:
            started_event = self._started_events.setdefault(job_id, threading.Event())
        return started_event.wait(timeout=timeout)

    def release(self, job_id: str) -> None:
        with self._lock:
            release_event = self._release_events.setdefault(job_id, threading.Event())
        release_event.set()


class PassThroughRestoreWorker:
    def restore(self, job: RestoreJob) -> RestoreResult:
        now = utc_now()
        return RestoreResult(
            job_id=job.job_id,
            sandbox_id=job.sandbox_id,
            checkpoint_id=job.checkpoint_id,
            status=JobStatus.SUCCEEDED,
            started_at=now,
            finished_at=utc_now(),
        )


class ExecutorTests(unittest.TestCase):
    def _live_checkpoint_job(self, job_id: str, *, request_id: str) -> CheckpointJob:
        return CheckpointJob(
            job_id=JobId(job_id),
            sandbox_id=SandboxId("sbx-live"),
            requested_at=utc_now(),
            metadata={
                "captures_inflight_llm": True,
                "captured_request_id": request_id,
            },
        )

    def test_executor_runs_checkpoints_in_parallel_and_emits_queue_wait(self) -> None:
        worker = SlowCheckpointWorker(sleep_s=0.02)
        telemetry = InMemoryTelemetrySink()
        executor = CRExecutor(
            ExecutorConfig(max_workers=3, checkpoint_workers=3, max_retries=0),
            checkpoint_worker=worker,
            restore_worker=PassThroughRestoreWorker(),
            telemetry=telemetry,
        )
        jobs = [
            CheckpointJob(
                job_id=JobId(f"job-{index}"),
                sandbox_id=SandboxId("sbx-1"),
                requested_at=utc_now(),
            )
            for index in range(4)
        ]

        t0 = time.perf_counter()
        results = executor.run_checkpoints(jobs)
        elapsed = time.perf_counter() - t0
        executor.shutdown()

        self.assertEqual(len(results), 4)
        self.assertTrue(all(r.status == JobStatus.SUCCEEDED for r in results))
        self.assertEqual(worker.max_active, 3)
        # Completion order across parallel workers is nondeterministic
        # (jobs in the same batch race); only membership is guaranteed.
        self.assertEqual(sorted(worker.completed_job_ids), [job.job_id.value for job in jobs])
        self.assertGreater(elapsed, 0.02)
        self.assertLess(elapsed, 0.08)
        queue_waits = [
            value
            for name, value, attributes in telemetry.metrics
            if name == "executor.job_queue_wait_ms" and attributes.get("job_type") == "checkpoint"
        ]
        self.assertEqual(len(queue_waits), 4)
        self.assertTrue(any(value > 0.0 for value in queue_waits))

    def test_executor_retry_on_failed_checkpoint_result(self) -> None:
        worker = FlakyCheckpointWorker()
        executor = CRExecutor(
            ExecutorConfig(max_workers=1, max_retries=1, retry_backoff_seconds=0.001),
            checkpoint_worker=worker,
            restore_worker=PassThroughRestoreWorker(),
        )
        job = CheckpointJob(
            job_id=JobId("job-1"),
            sandbox_id=SandboxId("sbx-1"),
            requested_at=utc_now(),
        )

        result = executor.run_checkpoint(job)
        record = executor.get_job_record(job.job_id)
        executor.shutdown()

        self.assertEqual(result.status, JobStatus.SUCCEEDED)
        self.assertEqual(worker.attempts, 2)
        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record.status, JobStatus.SUCCEEDED)

    def test_reactive_checkpoint_scheduler_promotes_exposed_job(self) -> None:
        worker = BlockingCheckpointWorker()
        executor = CRExecutor(
            ExecutorConfig(
                max_workers=1,
                checkpoint_workers=1,
                checkpoint_scheduling_policy="reactive",
            ),
            checkpoint_worker=worker,
            restore_worker=PassThroughRestoreWorker(),
        )
        job_a = self._live_checkpoint_job("job-a", request_id="req-a")
        job_b = self._live_checkpoint_job("job-b", request_id="req-b")
        job_c = self._live_checkpoint_job("job-c", request_id="req-c")

        future_a = executor.submit_checkpoint(job_a)
        future_b = executor.submit_checkpoint(job_b)
        future_c = executor.submit_checkpoint(job_c)

        self.assertTrue(worker.wait_started("job-a"))
        self.assertTrue(
            executor.notify_live_response_ready(job_c.sandbox_id, "req-c"),
        )

        worker.release("job-a")
        future_a.result(timeout=2.0)
        self.assertTrue(worker.wait_started("job-c"))
        worker.release("job-c")
        future_c.result(timeout=2.0)
        self.assertTrue(worker.wait_started("job-b"))
        worker.release("job-b")
        future_b.result(timeout=2.0)
        executor.shutdown()

        self.assertEqual(worker.started_job_ids, ["job-a", "job-c", "job-b"])

    def test_reactive_checkpoint_scheduler_serves_normal_job_after_urgent_quota(self) -> None:
        worker = BlockingCheckpointWorker()
        executor = CRExecutor(
            ExecutorConfig(
                max_workers=1,
                checkpoint_workers=1,
                checkpoint_scheduling_policy="reactive",
                reactive_checkpoint_urgent_quota=1,
            ),
            checkpoint_worker=worker,
            restore_worker=PassThroughRestoreWorker(),
        )
        job_a = self._live_checkpoint_job("job-a", request_id="req-a")
        job_b = self._live_checkpoint_job("job-b", request_id="req-b")
        job_c = self._live_checkpoint_job("job-c", request_id="req-c")
        job_d = self._live_checkpoint_job("job-d", request_id="req-d")

        futures = [
            executor.submit_checkpoint(job)
            for job in (job_a, job_b, job_c, job_d)
        ]

        self.assertTrue(worker.wait_started("job-a"))
        self.assertTrue(executor.notify_live_response_ready(job_c.sandbox_id, "req-c"))
        self.assertTrue(executor.notify_live_response_ready(job_d.sandbox_id, "req-d"))

        worker.release("job-a")
        futures[0].result(timeout=2.0)
        self.assertTrue(worker.wait_started("job-c"))
        worker.release("job-c")
        futures[2].result(timeout=2.0)
        self.assertTrue(worker.wait_started("job-b"))
        worker.release("job-b")
        futures[1].result(timeout=2.0)
        self.assertTrue(worker.wait_started("job-d"))
        worker.release("job-d")
        futures[3].result(timeout=2.0)
        executor.shutdown()

        self.assertEqual(worker.started_job_ids, ["job-a", "job-c", "job-b", "job-d"])

    def test_reactive_checkpoint_scheduler_handles_response_ready_before_job_submission(self) -> None:
        worker = BlockingCheckpointWorker()
        executor = CRExecutor(
            ExecutorConfig(
                max_workers=1,
                checkpoint_workers=1,
                checkpoint_scheduling_policy="reactive",
            ),
            checkpoint_worker=worker,
            restore_worker=PassThroughRestoreWorker(),
        )
        job_a = self._live_checkpoint_job("job-a", request_id="req-a")
        job_b = self._live_checkpoint_job("job-b", request_id="req-b")
        job_c = self._live_checkpoint_job("job-c", request_id="req-c")

        future_a = executor.submit_checkpoint(job_a)
        self.assertTrue(worker.wait_started("job-a"))
        self.assertFalse(executor.notify_live_response_ready(job_b.sandbox_id, "req-b"))
        future_b = executor.submit_checkpoint(job_b)
        future_c = executor.submit_checkpoint(job_c)

        worker.release("job-a")
        future_a.result(timeout=2.0)
        self.assertTrue(worker.wait_started("job-b"))
        worker.release("job-b")
        future_b.result(timeout=2.0)
        self.assertTrue(worker.wait_started("job-c"))
        worker.release("job-c")
        future_c.result(timeout=2.0)
        executor.shutdown()

        self.assertEqual(worker.started_job_ids, ["job-a", "job-b", "job-c"])

    def test_has_active_job_covers_running_checkpoint_and_restore(self) -> None:
        worker = BlockingCheckpointWorker()
        executor = CRExecutor(
            ExecutorConfig(max_workers=2, checkpoint_workers=1, max_retries=0),
            checkpoint_worker=worker,
            restore_worker=PassThroughRestoreWorker(),
        )
        sandbox_id = SandboxId("sbx-active")
        other_sandbox = SandboxId("sbx-other")
        ckpt_job = CheckpointJob(
            job_id=JobId("job-active-ckpt"),
            sandbox_id=sandbox_id,
            requested_at=utc_now(),
        )

        self.assertFalse(executor.has_active_job(sandbox_id))

        future = executor.submit_checkpoint(ckpt_job)
        self.assertTrue(worker.wait_started("job-active-ckpt"))

        self.assertTrue(executor.has_active_job(sandbox_id))
        self.assertFalse(executor.has_active_job(other_sandbox))

        worker.release("job-active-ckpt")
        future.result(timeout=2.0)

        self.assertFalse(executor.has_active_job(sandbox_id))
        executor.shutdown()


if __name__ == "__main__":
    unittest.main()
