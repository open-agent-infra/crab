from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from crab import (
    AdapterFileSystemCWorker,
    AdapterFileSystemRWorker,
    AdapterProcessCWorker,
    AdapterProcessRWorker,
    ArtifactKind,
    ArtifactPayload,
    ArtifactReference,
    CheckpointId,
    CheckpointJob,
    CheckpointManager,
    DefaultCWorker,
    DefaultRWorker,
    EBPFSandboxInspector,
    EBPFEvent,
    EBPFEventKind,
    FailureCode,
    InMemoryRuntime,
    InMemoryEBPFEventCollector,
    InMemoryTelemetrySink,
    JobId,
    LocalCheckpointManager,
    RestoreJob,
    RuncCheckpointOptions,
    RuncRuntime,
    RuncRuntimeOptions,
    RuncRuntimePaths,
    RuncRestoreOptions,
    Runtime,
    SandboxId,
    SandboxSnapshot,
    StorageConfig,
)
from crab.models import CheckpointManifest, RuntimeOperationStatus, WorkerStepResult, utc_now
from crab.runtime import CommandRunner

DockerRuntimeAdapter = InMemoryRuntime
RuncRuntimeAdapter = RuncRuntime
SandboxRuntimeAdapter = Runtime


class FakeCommandRunner(CommandRunner):
    def __init__(self) -> None:
        self.commands: list[tuple[str, ...]] = []

    def run(self, command: list[str], *, cwd: Path | None = None):
        _ = cwd
        self.commands.append(tuple(command))
        return type(
            "Result",
            (),
            {"command": tuple(command), "returncode": 0, "stdout": "", "stderr": ""},
        )()


class MappingCommandRunner(CommandRunner):
    def __init__(self, responses: dict[tuple[str, ...], tuple[int, str, str]] | None = None) -> None:
        self.responses = responses or {}
        self.commands: list[tuple[str, ...]] = []

    def run(self, command: list[str], *, cwd: Path | None = None):
        _ = cwd
        key = tuple(command)
        self.commands.append(key)
        returncode, stdout, stderr = self.responses.get(key, (0, "", ""))
        return type(
            "Result",
            (),
            {"command": key, "returncode": returncode, "stdout": stdout, "stderr": stderr},
        )()


class RecordingRestoreWorker:
    def __init__(self) -> None:
        self.jobs = []
        self.manifests: list[CheckpointManifest] = []

    def restore(self, job: RestoreJob, manifest: CheckpointManifest) -> WorkerStepResult:
        self.jobs.append(job)
        self.manifests.append(manifest)
        return WorkerStepResult(
            success=True,
            operation_status=RuntimeOperationStatus(executed=False, reason="recorded"),
        )


class NoArtifactCheckpointManager:
    def __init__(self, manifest: CheckpointManifest) -> None:
        self.manifest = manifest

    def get_manifest(self, sandbox_id: SandboxId, checkpoint_id: CheckpointId) -> CheckpointManifest:
        _ = (sandbox_id, checkpoint_id)
        return self.manifest

    def get_artifact(self, sandbox_id: SandboxId, checkpoint_id: CheckpointId, reference) -> bytes:
        _ = (sandbox_id, checkpoint_id, reference)
        raise AssertionError("process restore should not fetch process artifacts from checkpoint storage")

    def list_checkpoints(self, sandbox_id: SandboxId) -> list[CheckpointId]:
        _ = sandbox_id
        return [self.manifest.checkpoint_id]

    def delete_checkpoint(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
        *,
        cascade: bool = False,
    ) -> None:
        _ = (sandbox_id, checkpoint_id, cascade)

    def delete_all_checkpoints(self, sandbox_id: SandboxId) -> None:
        _ = sandbox_id

    def handle_checkpoint_complete(self, manifest: CheckpointManifest) -> None:
        _ = manifest

    def handle_restore_complete(self, sandbox_id: SandboxId, checkpoint_id: CheckpointId) -> None:
        _ = (sandbox_id, checkpoint_id)


class ManifestCheckpointManager:
    def __init__(self, manifests: list[CheckpointManifest]) -> None:
        self._manifests = {(manifest.sandbox_id, manifest.checkpoint_id): manifest for manifest in manifests}
        self._ordered: dict[SandboxId, list[CheckpointId]] = {}
        for manifest in manifests:
            self._ordered.setdefault(manifest.sandbox_id, []).append(manifest.checkpoint_id)

    def get_manifest(self, sandbox_id: SandboxId, checkpoint_id: CheckpointId) -> CheckpointManifest:
        return self._manifests[(sandbox_id, checkpoint_id)]

    def list_checkpoints(self, sandbox_id: SandboxId) -> list[CheckpointId]:
        return list(self._ordered.get(sandbox_id, []))

    def put_manifest(self, manifest: CheckpointManifest) -> None:
        self._manifests[(manifest.sandbox_id, manifest.checkpoint_id)] = manifest

    def put_artifact(self, sandbox_id: SandboxId, checkpoint_id: CheckpointId, artifact):
        raise NotImplementedError

    def get_artifact(self, sandbox_id: SandboxId, checkpoint_id: CheckpointId, reference) -> bytes:
        raise NotImplementedError

    def delete_checkpoint(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
        *,
        cascade: bool = False,
    ) -> None:
        _ = cascade
        self._manifests.pop((sandbox_id, checkpoint_id), None)

    def delete_all_checkpoints(self, sandbox_id: SandboxId) -> None:
        for checkpoint_id in list(self._ordered.get(sandbox_id, [])):
            self.delete_checkpoint(sandbox_id, checkpoint_id)

    def handle_checkpoint_complete(self, manifest: CheckpointManifest) -> None:
        _ = manifest

    def handle_restore_complete(self, sandbox_id: SandboxId, checkpoint_id: CheckpointId) -> None:
        _ = (sandbox_id, checkpoint_id)


class RecordingCheckpointWorker:
    def __init__(self, artifact_kind: str) -> None:
        self.calls: list[tuple[CheckpointJob, CheckpointId]] = []
        self.artifact_kind = artifact_kind

    def checkpoint(self, job: CheckpointJob, checkpoint_id: CheckpointId) -> WorkerStepResult:
        self.calls.append((job, checkpoint_id))
        return WorkerStepResult(
            success=True,
            artifacts=[],
            operation_status=RuntimeOperationStatus(executed=False, reason=self.artifact_kind),
        )


class RecordingCheckpointManager:
    def __init__(self, existing_manifests: list[CheckpointManifest] | None = None) -> None:
        self.manifest: CheckpointManifest | None = None
        self.completed: list[CheckpointManifest] = []
        self._manifests: dict[tuple[SandboxId, CheckpointId], CheckpointManifest] = {}
        self._ordered: dict[SandboxId, list[CheckpointId]] = {}
        for manifest in existing_manifests or []:
            self._manifests[(manifest.sandbox_id, manifest.checkpoint_id)] = manifest
            self._ordered.setdefault(manifest.sandbox_id, []).append(manifest.checkpoint_id)

    def put_manifest(self, manifest: CheckpointManifest) -> None:
        self.manifest = manifest
        self._manifests[(manifest.sandbox_id, manifest.checkpoint_id)] = manifest
        self._ordered.setdefault(manifest.sandbox_id, []).append(manifest.checkpoint_id)

    def get_manifest(self, sandbox_id: SandboxId, checkpoint_id: CheckpointId) -> CheckpointManifest:
        manifest = self._manifests.get((sandbox_id, checkpoint_id))
        if manifest is not None:
            return manifest
        assert self.manifest is not None
        return self.manifest

    def put_artifact(self, sandbox_id: SandboxId, checkpoint_id: CheckpointId, artifact):
        _ = artifact
        return type(
            "ArtifactReference",
            (),
            {
                "kind": None,
                "name": "noop",
            },
        )()

    def get_artifact(self, sandbox_id: SandboxId, checkpoint_id: CheckpointId, reference) -> bytes:
        _ = (sandbox_id, checkpoint_id, reference)
        return b""

    def list_checkpoints(self, sandbox_id: SandboxId) -> list[CheckpointId]:
        return list(self._ordered.get(sandbox_id, []))

    def delete_checkpoint(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
        *,
        cascade: bool = False,
    ) -> None:
        _ = (sandbox_id, checkpoint_id, cascade)

    def delete_all_checkpoints(self, sandbox_id: SandboxId) -> None:
        _ = sandbox_id

    def handle_checkpoint_complete(self, manifest: CheckpointManifest) -> None:
        self.completed.append(manifest)

    def handle_restore_complete(self, sandbox_id: SandboxId, checkpoint_id: CheckpointId) -> None:
        _ = (sandbox_id, checkpoint_id)


class ContractTests(unittest.TestCase):
    def test_runtime_adapters_are_contract_compatible(self) -> None:
        docker = DockerRuntimeAdapter()
        self.assertIsInstance(docker, SandboxRuntimeAdapter)
        self.assertTrue(
            len(docker.checkpoint_process(SandboxId("sbx-1"), CheckpointId("ckpt-1"), leave_running=False).command)
            > 0
        )

        with tempfile.TemporaryDirectory(prefix="crab_runtime_contract_") as tmp:
            adapter = RuncRuntimeAdapter(
                command_runner=FakeCommandRunner(),
                paths=RuncRuntimePaths(
                    state_root=Path(tmp) / "state",
                    bundle_root=Path(tmp) / "bundles",
                    checkpoint_root=Path(tmp) / "checkpoints",
                    zfs_dataset_prefix="pool/crab",
                ),
            )
            self.assertIsInstance(adapter, SandboxRuntimeAdapter)
            self.assertTrue(
                len(
                    adapter.checkpoint_process(
                        SandboxId("sbx-1"),
                        CheckpointId("ckpt-1"),
                        leave_running=False,
                    ).command
                )
                > 0
            )

    def test_runc_runtime_uses_default_optional_checkpoint_and_restore_args(self) -> None:
        with tempfile.TemporaryDirectory(prefix="crab_runtime_contract_") as tmp:
            base = Path(tmp)
            runner = FakeCommandRunner()
            adapter = RuncRuntimeAdapter(
                command_runner=runner,
                paths=RuncRuntimePaths(
                    state_root=base / "state",
                    bundle_root=base / "bundles",
                    checkpoint_root=base / "checkpoints",
                    zfs_dataset_prefix="pool/crab",
                ),
            )

            adapter.checkpoint_process(SandboxId("sbx-1"), CheckpointId("ckpt-1"), leave_running=False)
            adapter.restore_process(SandboxId("sbx-1"), CheckpointId("ckpt-1"))

            self.assertEqual(
                runner.commands[0],
                (
                    "runc",
                    "--root",
                    str(base / "state"),
                    "checkpoint",
                    "--image-path",
                    str(base / "checkpoints" / "sbx-1" / "ckpt-1" / "process"),
                    "--work-path",
                    str(base / "checkpoints" / "sbx-1" / "ckpt-1" / "work"),
                    "--leave-running=false",
                    "--tcp-established",
                    "--shell-job",
                    "--tcp-skip-in-flight",
                    "--ext-unix-sk",
                    "sbx-1",
                ),
            )
            self.assertEqual(
                runner.commands[1],
                (
                    "runc",
                    "--root",
                    str(base / "state"),
                    "restore",
                    "-d",
                    "--bundle",
                    str(base / "bundles" / "sbx-1"),
                    "--image-path",
                    str(base / "checkpoints" / "sbx-1" / "ckpt-1" / "process"),
                    "--work-path",
                    str(base / "checkpoints" / "sbx-1" / "ckpt-1" / "work"),
                    "--tcp-established",
                    "--shell-job",
                    "--ext-unix-sk",
                    "sbx-1",
                ),
            )

    def test_runc_runtime_options_allow_overriding_optional_args(self) -> None:
        with tempfile.TemporaryDirectory(prefix="crab_runtime_contract_") as tmp:
            base = Path(tmp)
            runner = FakeCommandRunner()
            adapter = RuncRuntimeAdapter(
                command_runner=runner,
                paths=RuncRuntimePaths(
                    state_root=base / "state",
                    bundle_root=base / "bundles",
                    checkpoint_root=base / "checkpoints",
                    zfs_dataset_prefix="pool/crab",
                ),
                options=RuncRuntimeOptions(
                    checkpoint=RuncCheckpointOptions(
                        shell_job=False,
                        tcp_skip_in_flight=False,
                        extra_args=("--manage-cgroups-mode=soft",),
                    ),
                    restore=RuncRestoreOptions(
                        detach=False,
                        tcp_established=False,
                        extra_args=("--manage-cgroups-mode=soft",),
                    ),
                ),
            )

            adapter.checkpoint_process(SandboxId("sbx-1"), CheckpointId("ckpt-1"), leave_running=True)
            adapter.restore_process(SandboxId("sbx-1"), CheckpointId("ckpt-1"))

            self.assertEqual(
                runner.commands[0],
                (
                    "runc",
                    "--root",
                    str(base / "state"),
                    "checkpoint",
                    "--image-path",
                    str(base / "checkpoints" / "sbx-1" / "ckpt-1" / "process"),
                    "--work-path",
                    str(base / "checkpoints" / "sbx-1" / "ckpt-1" / "work"),
                    "--leave-running=true",
                    "--tcp-established",
                    "--ext-unix-sk",
                    "--manage-cgroups-mode=soft",
                    "sbx-1",
                ),
            )
            self.assertEqual(
                runner.commands[1],
                (
                    "runc",
                    "--root",
                    str(base / "state"),
                    "restore",
                    "--bundle",
                    str(base / "bundles" / "sbx-1"),
                    "--image-path",
                    str(base / "checkpoints" / "sbx-1" / "ckpt-1" / "process"),
                    "--work-path",
                    str(base / "checkpoints" / "sbx-1" / "ckpt-1" / "work"),
                    "--shell-job",
                    "--ext-unix-sk",
                    "--manage-cgroups-mode=soft",
                    "sbx-1",
                ),
            )

    def test_workers_return_typed_dry_run_results(self) -> None:
        adapter = DockerRuntimeAdapter()
        process_c = AdapterProcessCWorker(adapter)
        process_r = AdapterProcessRWorker(adapter)
        fs_c = AdapterFileSystemCWorker(adapter)
        fs_r = AdapterFileSystemRWorker(adapter)

        cjob = CheckpointJob(
            job_id=JobId("job-1"),
            sandbox_id=SandboxId("sbx-1"),
            requested_at=utc_now(),
        )
        ckpt_id = CheckpointId("ckpt-1")

        c_process_result = process_c.checkpoint(cjob, ckpt_id)
        c_fs_result = fs_c.checkpoint(cjob, ckpt_id)
        self.assertTrue(c_process_result.success)
        self.assertTrue(c_fs_result.success)
        self.assertFalse(c_process_result.operation_status.executed)
        self.assertFalse(c_fs_result.operation_status.executed)

        manifest = CheckpointManifest(
            schema_version="v1",
            checkpoint_id=ckpt_id,
            sandbox_id=cjob.sandbox_id,
            created_at=utc_now(),
            runtime_name="docker",
            runtime_version=None,
            process_artifacts=[],
            filesystem_artifacts=[],
            metadata={},
        ).with_integrity()
        rjob = RestoreJob(
            job_id=JobId("job-2"),
            sandbox_id=SandboxId("sbx-1"),
            checkpoint_id=ckpt_id,
            requested_at=utc_now(),
        )
        r_process_result = process_r.restore(rjob, manifest)
        r_fs_result = fs_r.restore(rjob, manifest)
        self.assertTrue(r_process_result.success)
        self.assertTrue(r_fs_result.success)
        self.assertFalse(r_process_result.operation_status.executed)
        self.assertFalse(r_fs_result.operation_status.executed)

    def test_runc_process_checkpoint_emits_only_metadata_artifact(self) -> None:
        with tempfile.TemporaryDirectory(prefix="crab_process_contract_") as tmp:
            base = Path(tmp)
            adapter = RuncRuntimeAdapter(
                command_runner=FakeCommandRunner(),
                paths=RuncRuntimePaths(
                    state_root=base / "state",
                    bundle_root=base / "bundles",
                    checkpoint_root=base / "checkpoints",
                    zfs_dataset_prefix="pool/crab",
                ),
            )
            process_c = AdapterProcessCWorker(adapter)
            sandbox_id = SandboxId("sbx-1")
            checkpoint_id = CheckpointId("ckpt-1")

            result = process_c.checkpoint(
                CheckpointJob(
                    job_id=JobId("job-1"),
                    sandbox_id=sandbox_id,
                    requested_at=utc_now(),
                ),
                checkpoint_id,
            )

            self.assertTrue(result.success)
            self.assertEqual(len(result.artifacts), 1)
            self.assertEqual(result.artifacts[0].name, "process_checkpoint.json")
            payload = json.loads(result.artifacts[0].data.decode("utf-8"))
            self.assertEqual(payload["process_storage_mode"], "runtime_reference")
            self.assertEqual(
                payload["process_checkpoint_location"],
                str(base / "checkpoints" / str(sandbox_id) / str(checkpoint_id) / "process"),
            )

    def test_runc_process_restore_requires_existing_checkpoint_directory(self) -> None:
        with tempfile.TemporaryDirectory(prefix="crab_process_restore_") as tmp:
            base = Path(tmp)
            adapter = RuncRuntimeAdapter(
                command_runner=FakeCommandRunner(),
                paths=RuncRuntimePaths(
                    state_root=base / "state",
                    bundle_root=base / "bundles",
                    checkpoint_root=base / "checkpoints",
                    zfs_dataset_prefix="pool/crab",
                ),
            )
            process_r = AdapterProcessRWorker(adapter)
            sandbox_id = SandboxId("sbx-1")
            checkpoint_id = CheckpointId("ckpt-1")
            manifest = CheckpointManifest(
                schema_version="v1",
                checkpoint_id=checkpoint_id,
                sandbox_id=sandbox_id,
                created_at=utc_now(),
                runtime_name="runc",
                runtime_version=None,
                process_artifacts=[],
                filesystem_artifacts=[],
                metadata={},
            ).with_integrity()
            job = RestoreJob(
                job_id=JobId("job-2"),
                sandbox_id=sandbox_id,
                checkpoint_id=checkpoint_id,
                requested_at=utc_now(),
            )

            with self.assertRaisesRegex(FileNotFoundError, "process checkpoint directory not found"):
                process_r.restore(job, manifest)

            checkpoint_dir = base / "checkpoints" / str(sandbox_id) / str(checkpoint_id) / "process"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            result = process_r.restore(job, manifest)
            self.assertTrue(result.success)
            self.assertTrue(result.operation_status.executed)

    def test_default_restore_worker_skips_process_artifact_downloads(self) -> None:
        manifest = CheckpointManifest(
            schema_version="v1",
            checkpoint_id=CheckpointId("ckpt-1"),
            sandbox_id=SandboxId("sbx-1"),
            created_at=utc_now(),
            runtime_name="runc",
            runtime_version=None,
            process_artifacts=[],
            filesystem_artifacts=[],
            metadata={},
        ).with_integrity()
        fs_worker = RecordingRestoreWorker()
        process_worker = RecordingRestoreWorker()
        restore_worker = DefaultRWorker(
            process_worker=process_worker,
            filesystem_worker=fs_worker,
            checkpoint_manager=NoArtifactCheckpointManager(manifest),
        )
        job = RestoreJob(
            job_id=JobId("job-restore"),
            sandbox_id=SandboxId("sbx-1"),
            checkpoint_id=CheckpointId("ckpt-1"),
            requested_at=utc_now(),
            metadata={"keep": "me"},
        )

        result = restore_worker.restore(job)

        self.assertEqual(result.status.value, "succeeded")
        self.assertEqual(fs_worker.jobs, [])
        self.assertEqual(process_worker.jobs, [])

    def test_default_checkpoint_worker_honors_scoped_checkpoint_flags(self) -> None:
        manager = RecordingCheckpointManager()
        process_worker = RecordingCheckpointWorker("process")
        filesystem_worker = RecordingCheckpointWorker("filesystem")
        worker = DefaultCWorker(
            process_worker=process_worker,
            filesystem_worker=filesystem_worker,
            checkpoint_manager=manager,
            runtime=DockerRuntimeAdapter(),
        )
        job = CheckpointJob(
            job_id=JobId("job-scoped"),
            sandbox_id=SandboxId("sbx-1"),
            requested_at=utc_now(),
            checkpoint_process=True,
            checkpoint_filesystem=False,
        )

        result = worker.checkpoint(job)

        self.assertEqual(result.status.value, "succeeded")
        self.assertEqual(len(process_worker.calls), 1)
        self.assertEqual(len(filesystem_worker.calls), 0)
        self.assertEqual(len(manager.completed), 1)
        assert result.manifest is not None
        self.assertEqual(result.manifest.filesystem_artifacts, [])

    def test_default_checkpoint_worker_rejects_guarded_job_before_workers_run(self) -> None:
        manager = RecordingCheckpointManager()
        process_worker = RecordingCheckpointWorker("process")
        filesystem_worker = RecordingCheckpointWorker("filesystem")
        worker = DefaultCWorker(
            process_worker=process_worker,
            filesystem_worker=filesystem_worker,
            checkpoint_manager=manager,
            runtime=DockerRuntimeAdapter(),
            checkpoint_guard=lambda job: (False, f"{job.sandbox_id}:sandbox_not_running"),
        )
        job = CheckpointJob(
            job_id=JobId("job-guarded"),
            sandbox_id=SandboxId("sbx-1"),
            requested_at=utc_now(),
            checkpoint_process=True,
            checkpoint_filesystem=True,
        )

        result = worker.checkpoint(job)

        self.assertEqual(result.status.value, "failed")
        self.assertEqual(result.failure_code, FailureCode.VALIDATION_ERROR)
        self.assertEqual(result.message, "sbx-1:sandbox_not_running")
        self.assertEqual(process_worker.calls, [])
        self.assertEqual(filesystem_worker.calls, [])
        self.assertEqual(manager.completed, [])

    def test_default_checkpoint_worker_promotes_first_filesystem_only_checkpoint(self) -> None:
        manager = RecordingCheckpointManager()
        process_worker = RecordingCheckpointWorker("process")
        filesystem_worker = RecordingCheckpointWorker("filesystem")
        worker = DefaultCWorker(
            process_worker=process_worker,
            filesystem_worker=filesystem_worker,
            checkpoint_manager=manager,
            runtime=DockerRuntimeAdapter(),
        )
        job = CheckpointJob(
            job_id=JobId("job-fs-first"),
            sandbox_id=SandboxId("sbx-1"),
            requested_at=utc_now(),
            checkpoint_process=False,
            checkpoint_filesystem=True,
        )

        result = worker.checkpoint(job)

        self.assertEqual(result.status.value, "succeeded")
        self.assertEqual(len(process_worker.calls), 1)
        self.assertEqual(len(filesystem_worker.calls), 1)
        assert result.manifest is not None
        self.assertTrue(result.manifest.metadata["promoted_process_checkpoint"])
        self.assertEqual(result.manifest.metadata["promoted_process_checkpoint_reason"], "missing_process_ancestor")

    def test_default_checkpoint_worker_keeps_filesystem_only_scope_with_process_ancestor(self) -> None:
        sid = SandboxId("sbx-1")
        prior_process = CheckpointManifest(
            schema_version="v1",
            checkpoint_id=CheckpointId("ckpt-1"),
            sandbox_id=sid,
            created_at=utc_now(),
            runtime_name="runc",
            runtime_version=None,
            process_artifacts=[ArtifactReference(kind=ArtifactKind.PROCESS, name="process.json", relative_path="p", size_bytes=1, sha256="0" * 64, metadata={})],
            filesystem_artifacts=[],
            metadata={},
        ).with_integrity()
        manager = RecordingCheckpointManager(existing_manifests=[prior_process])
        process_worker = RecordingCheckpointWorker("process")
        filesystem_worker = RecordingCheckpointWorker("filesystem")
        worker = DefaultCWorker(
            process_worker=process_worker,
            filesystem_worker=filesystem_worker,
            checkpoint_manager=manager,
            runtime=DockerRuntimeAdapter(),
        )
        job = CheckpointJob(
            job_id=JobId("job-fs-next"),
            sandbox_id=sid,
            requested_at=utc_now(),
            checkpoint_process=False,
            checkpoint_filesystem=True,
        )

        result = worker.checkpoint(job)

        self.assertEqual(result.status.value, "succeeded")
        self.assertEqual(len(process_worker.calls), 0)
        self.assertEqual(len(filesystem_worker.calls), 1)
        assert result.manifest is not None
        self.assertNotIn("promoted_process_checkpoint", result.manifest.metadata)

    def test_default_restore_worker_backfills_missing_process_from_previous_checkpoint(self) -> None:
        sid = SandboxId("sbx-1")
        process_ref = LocalCheckpointManager(StorageConfig(root_dir=Path(tempfile.mkdtemp()))).put_artifact(
            sid,
            CheckpointId("ckpt-bootstrap"),
            ArtifactPayload(kind=ArtifactKind.PROCESS, name="process.json", data=b"{}"),
        )
        previous = CheckpointManifest(
            schema_version="v1",
            checkpoint_id=CheckpointId("ckpt-1"),
            sandbox_id=sid,
            created_at=utc_now(),
            runtime_name="runc",
            runtime_version=None,
            process_artifacts=[process_ref],
            filesystem_artifacts=[],
            metadata={},
        ).with_integrity()
        current = CheckpointManifest(
            schema_version="v1",
            checkpoint_id=CheckpointId("ckpt-2"),
            sandbox_id=sid,
            created_at=utc_now(),
            runtime_name="runc",
            runtime_version=None,
            process_artifacts=[],
            filesystem_artifacts=[],
            metadata={},
        ).with_integrity()
        fs_worker = RecordingRestoreWorker()
        process_worker = RecordingRestoreWorker()
        restore_worker = DefaultRWorker(
            process_worker=process_worker,
            filesystem_worker=fs_worker,
            checkpoint_manager=ManifestCheckpointManager([previous, current]),
        )

        result = restore_worker.restore(
            RestoreJob(
                job_id=JobId("job-restore-process"),
                sandbox_id=sid,
                checkpoint_id=CheckpointId("ckpt-2"),
                requested_at=utc_now(),
            )
        )

        self.assertEqual(result.status.value, "succeeded")
        self.assertEqual(len(process_worker.jobs), 1)
        self.assertEqual(len(fs_worker.jobs), 0)
        self.assertEqual(process_worker.manifests[0].metadata["process_restore_checkpoint_id"], "ckpt-1")
        self.assertEqual(len(process_worker.manifests[0].process_artifacts), 1)

    def test_default_restore_worker_backfills_missing_filesystem_from_previous_checkpoint(self) -> None:
        sid = SandboxId("sbx-1")
        fs_ref = LocalCheckpointManager(StorageConfig(root_dir=Path(tempfile.mkdtemp()))).put_artifact(
            sid,
            CheckpointId("ckpt-bootstrap"),
            ArtifactPayload(kind=ArtifactKind.FILESYSTEM, name="filesystem.json", data=b"{}"),
        )
        previous = CheckpointManifest(
            schema_version="v1",
            checkpoint_id=CheckpointId("ckpt-1"),
            sandbox_id=sid,
            created_at=utc_now(),
            runtime_name="runc",
            runtime_version=None,
            process_artifacts=[],
            filesystem_artifacts=[fs_ref],
            metadata={},
        ).with_integrity()
        current = CheckpointManifest(
            schema_version="v1",
            checkpoint_id=CheckpointId("ckpt-2"),
            sandbox_id=sid,
            created_at=utc_now(),
            runtime_name="runc",
            runtime_version=None,
            process_artifacts=[],
            filesystem_artifacts=[],
            metadata={},
        ).with_integrity()
        fs_worker = RecordingRestoreWorker()
        process_worker = RecordingRestoreWorker()
        restore_worker = DefaultRWorker(
            process_worker=process_worker,
            filesystem_worker=fs_worker,
            checkpoint_manager=ManifestCheckpointManager([previous, current]),
        )

        result = restore_worker.restore(
            RestoreJob(
                job_id=JobId("job-restore-fs"),
                sandbox_id=sid,
                checkpoint_id=CheckpointId("ckpt-2"),
                requested_at=utc_now(),
            )
        )

        self.assertEqual(result.status.value, "succeeded")
        self.assertEqual(len(fs_worker.jobs), 1)
        self.assertEqual(len(process_worker.jobs), 0)
        self.assertEqual(fs_worker.manifests[0].metadata["filesystem_restore_checkpoint_id"], "ckpt-1")
        self.assertEqual(len(fs_worker.manifests[0].filesystem_artifacts), 1)

    def test_default_restore_worker_keeps_later_process_when_only_inflight_mutation_is_missing(self) -> None:
        sid = SandboxId("sbx-1")
        manager = LocalCheckpointManager(StorageConfig(root_dir=Path(tempfile.mkdtemp())))
        fs_ref = manager.put_artifact(sid, CheckpointId("ckpt-bootstrap-fs"), ArtifactPayload(kind=ArtifactKind.FILESYSTEM, name="filesystem.json", data=b"{}"))
        process_ref = manager.put_artifact(sid, CheckpointId("ckpt-bootstrap-process"), ArtifactPayload(kind=ArtifactKind.PROCESS, name="process.json", data=b"{}"))
        filesystem_manifest = CheckpointManifest(
            schema_version="v1",
            checkpoint_id=CheckpointId("ckpt-1"),
            sandbox_id=sid,
            created_at=utc_now(),
            runtime_name="runc",
            runtime_version=None,
            process_artifacts=[],
            filesystem_artifacts=[fs_ref],
            metadata={"benchmark_trace_cursor": 3},
        ).with_integrity()
        process_manifest = CheckpointManifest(
            schema_version="v1",
            checkpoint_id=CheckpointId("ckpt-2"),
            sandbox_id=sid,
            created_at=utc_now(),
            runtime_name="runc",
            runtime_version=None,
            process_artifacts=[process_ref],
            filesystem_artifacts=[],
            metadata={
                "benchmark_trace_cursor": 6,
                "benchmark_latest_mutating_response_count": 6,
                "benchmark_previous_mutating_response_count": 0,
                "captures_inflight_llm": True,
            },
        ).with_integrity()
        fs_worker = RecordingRestoreWorker()
        process_worker = RecordingRestoreWorker()
        restore_worker = DefaultRWorker(
            process_worker=process_worker,
            filesystem_worker=fs_worker,
            checkpoint_manager=ManifestCheckpointManager([filesystem_manifest, process_manifest]),
        )

        result = restore_worker.restore(
            RestoreJob(
                job_id=JobId("job-restore-safe-process"),
                sandbox_id=sid,
                checkpoint_id=CheckpointId("ckpt-2"),
                requested_at=utc_now(),
            )
        )

        self.assertEqual(result.status.value, "succeeded")
        self.assertEqual(process_worker.manifests[0].metadata["process_restore_checkpoint_id"], "ckpt-2")
        self.assertEqual(process_worker.manifests[0].metadata["filesystem_restore_checkpoint_id"], "ckpt-1")
        self.assertEqual(process_worker.manifests[0].metadata["filesystem_restore_trace_cursor"], 3)

    def test_default_restore_worker_rewinds_unsafe_newer_process_checkpoint(self) -> None:
        sid = SandboxId("sbx-1")
        manager = LocalCheckpointManager(StorageConfig(root_dir=Path(tempfile.mkdtemp())))
        early_process_ref = manager.put_artifact(sid, CheckpointId("ckpt-bootstrap-process-1"), ArtifactPayload(kind=ArtifactKind.PROCESS, name="process.json", data=b"{}"))
        fs_ref = manager.put_artifact(sid, CheckpointId("ckpt-bootstrap-fs"), ArtifactPayload(kind=ArtifactKind.FILESYSTEM, name="filesystem.json", data=b"{}"))
        late_process_ref = manager.put_artifact(sid, CheckpointId("ckpt-bootstrap-process-2"), ArtifactPayload(kind=ArtifactKind.PROCESS, name="process.json", data=b"{}"))
        early_process_manifest = CheckpointManifest(
            schema_version="v1",
            checkpoint_id=CheckpointId("ckpt-1"),
            sandbox_id=sid,
            created_at=utc_now(),
            runtime_name="runc",
            runtime_version=None,
            process_artifacts=[early_process_ref],
            filesystem_artifacts=[],
            metadata={"benchmark_trace_cursor": 6},
        ).with_integrity()
        filesystem_manifest = CheckpointManifest(
            schema_version="v1",
            checkpoint_id=CheckpointId("ckpt-2"),
            sandbox_id=sid,
            created_at=utc_now(),
            runtime_name="runc",
            runtime_version=None,
            process_artifacts=[],
            filesystem_artifacts=[fs_ref],
            metadata={"benchmark_trace_cursor": 6},
        ).with_integrity()
        late_process_manifest = CheckpointManifest(
            schema_version="v1",
            checkpoint_id=CheckpointId("ckpt-3"),
            sandbox_id=sid,
            created_at=utc_now(),
            runtime_name="runc",
            runtime_version=None,
            process_artifacts=[late_process_ref],
            filesystem_artifacts=[],
            metadata={
                "benchmark_trace_cursor": 12,
                "benchmark_latest_mutating_response_count": 12,
                "benchmark_previous_mutating_response_count": 10,
                "captures_inflight_llm": True,
            },
        ).with_integrity()
        fs_worker = RecordingRestoreWorker()
        process_worker = RecordingRestoreWorker()
        restore_worker = DefaultRWorker(
            process_worker=process_worker,
            filesystem_worker=fs_worker,
            checkpoint_manager=ManifestCheckpointManager(
                [early_process_manifest, filesystem_manifest, late_process_manifest]
            ),
        )

        result = restore_worker.restore(
            RestoreJob(
                job_id=JobId("job-restore-unsafe-process"),
                sandbox_id=sid,
                checkpoint_id=CheckpointId("ckpt-3"),
                requested_at=utc_now(),
            )
        )

        self.assertEqual(result.status.value, "succeeded")
        self.assertEqual(process_worker.manifests[0].metadata["process_restore_checkpoint_id"], "ckpt-1")
        self.assertEqual(process_worker.manifests[0].metadata["filesystem_restore_checkpoint_id"], "ckpt-2")
        self.assertEqual(process_worker.manifests[0].metadata["filesystem_restore_trace_cursor"], 6)

    def test_default_restore_worker_rewinds_unsafe_newer_process_checkpoint_without_mutating_metadata(self) -> None:
        sid = SandboxId("sbx-1")
        manager = LocalCheckpointManager(StorageConfig(root_dir=Path(tempfile.mkdtemp())))
        early_process_ref = manager.put_artifact(
            sid,
            CheckpointId("ckpt-bootstrap-process-1"),
            ArtifactPayload(kind=ArtifactKind.PROCESS, name="process.json", data=b"{}"),
        )
        fs_ref = manager.put_artifact(
            sid,
            CheckpointId("ckpt-bootstrap-fs"),
            ArtifactPayload(kind=ArtifactKind.FILESYSTEM, name="filesystem.json", data=b"{}"),
        )
        late_process_ref = manager.put_artifact(
            sid,
            CheckpointId("ckpt-bootstrap-process-2"),
            ArtifactPayload(kind=ArtifactKind.PROCESS, name="process.json", data=b"{}"),
        )
        early_process_manifest = CheckpointManifest(
            schema_version="v1",
            checkpoint_id=CheckpointId("ckpt-1"),
            sandbox_id=sid,
            created_at=utc_now(),
            runtime_name="runc",
            runtime_version=None,
            process_artifacts=[early_process_ref],
            filesystem_artifacts=[],
            metadata={"benchmark_trace_cursor": 6},
        ).with_integrity()
        filesystem_manifest = CheckpointManifest(
            schema_version="v1",
            checkpoint_id=CheckpointId("ckpt-2"),
            sandbox_id=sid,
            created_at=utc_now(),
            runtime_name="runc",
            runtime_version=None,
            process_artifacts=[],
            filesystem_artifacts=[fs_ref],
            metadata={"benchmark_trace_cursor": 6},
        ).with_integrity()
        late_process_manifest = CheckpointManifest(
            schema_version="v1",
            checkpoint_id=CheckpointId("ckpt-3"),
            sandbox_id=sid,
            created_at=utc_now(),
            runtime_name="runc",
            runtime_version=None,
            process_artifacts=[late_process_ref],
            filesystem_artifacts=[],
            metadata={
                "benchmark_trace_cursor": 12,
                "captures_inflight_llm": True,
            },
        ).with_integrity()
        fs_worker = RecordingRestoreWorker()
        process_worker = RecordingRestoreWorker()
        restore_worker = DefaultRWorker(
            process_worker=process_worker,
            filesystem_worker=fs_worker,
            checkpoint_manager=ManifestCheckpointManager(
                [early_process_manifest, filesystem_manifest, late_process_manifest]
            ),
        )

        result = restore_worker.restore(
            RestoreJob(
                job_id=JobId("job-restore-unsafe-process-no-mutating-metadata"),
                sandbox_id=sid,
                checkpoint_id=CheckpointId("ckpt-3"),
                requested_at=utc_now(),
            )
        )

        self.assertEqual(result.status.value, "succeeded")
        self.assertEqual(process_worker.manifests[0].metadata["process_restore_checkpoint_id"], "ckpt-1")
        self.assertEqual(process_worker.manifests[0].metadata["filesystem_restore_checkpoint_id"], "ckpt-2")
        self.assertEqual(process_worker.manifests[0].metadata["filesystem_restore_trace_cursor"], 6)

    def test_default_restore_worker_preserves_replay_count_for_inflight_filesystem_checkpoint(self) -> None:
        sid = SandboxId("sbx-1")
        manager = LocalCheckpointManager(StorageConfig(root_dir=Path(tempfile.mkdtemp())))
        fs_ref = manager.put_artifact(
            sid,
            CheckpointId("ckpt-bootstrap-fs"),
            ArtifactPayload(kind=ArtifactKind.FILESYSTEM, name="filesystem.json", data=b"{}"),
        )
        filesystem_manifest = CheckpointManifest(
            schema_version="v1",
            checkpoint_id=CheckpointId("ckpt-1"),
            sandbox_id=sid,
            created_at=utc_now(),
            runtime_name="runc",
            runtime_version=None,
            process_artifacts=[],
            filesystem_artifacts=[fs_ref],
            metadata={
                "benchmark_trace_cursor": 3,
                "benchmark_latest_mutating_response_count": 3,
                "benchmark_previous_mutating_response_count": 0,
                "captures_inflight_llm": True,
            },
        ).with_integrity()
        fs_worker = RecordingRestoreWorker()
        process_worker = RecordingRestoreWorker()
        restore_worker = DefaultRWorker(
            process_worker=process_worker,
            filesystem_worker=fs_worker,
            checkpoint_manager=ManifestCheckpointManager([filesystem_manifest]),
        )

        result = restore_worker.restore(
            RestoreJob(
                job_id=JobId("job-restore-inflight-fs"),
                sandbox_id=sid,
                checkpoint_id=CheckpointId("ckpt-1"),
                requested_at=utc_now(),
            )
        )

        self.assertEqual(result.status.value, "succeeded")
        self.assertEqual(fs_worker.manifests[0].metadata["filesystem_restore_checkpoint_id"], "ckpt-1")
        self.assertEqual(fs_worker.manifests[0].metadata["filesystem_restore_trace_cursor"], 3)

    def test_default_restore_worker_rewinds_process_when_filesystem_lacks_latest_mutation(self) -> None:
        sid = SandboxId("sbx-1")
        manager = LocalCheckpointManager(StorageConfig(root_dir=Path(tempfile.mkdtemp())))
        early_process_ref = manager.put_artifact(
            sid,
            CheckpointId("ckpt-bootstrap-process-1"),
            ArtifactPayload(kind=ArtifactKind.PROCESS, name="process.json", data=b"{}"),
        )
        fs_ref = manager.put_artifact(
            sid,
            CheckpointId("ckpt-bootstrap-fs"),
            ArtifactPayload(kind=ArtifactKind.FILESYSTEM, name="filesystem.json", data=b"{}"),
        )
        late_process_ref = manager.put_artifact(
            sid,
            CheckpointId("ckpt-bootstrap-process-2"),
            ArtifactPayload(kind=ArtifactKind.PROCESS, name="process.json", data=b"{}"),
        )
        early_process_manifest = CheckpointManifest(
            schema_version="v1",
            checkpoint_id=CheckpointId("ckpt-1"),
            sandbox_id=sid,
            created_at=utc_now(),
            runtime_name="runc",
            runtime_version=None,
            process_artifacts=[early_process_ref],
            filesystem_artifacts=[],
            metadata={"benchmark_trace_cursor": 4, "benchmark_latest_mutating_response_count": 4},
        ).with_integrity()
        filesystem_manifest = CheckpointManifest(
            schema_version="v1",
            checkpoint_id=CheckpointId("ckpt-2"),
            sandbox_id=sid,
            created_at=utc_now(),
            runtime_name="runc",
            runtime_version=None,
            process_artifacts=[],
            filesystem_artifacts=[fs_ref],
            metadata={
                "benchmark_trace_cursor": 10,
                "benchmark_latest_mutating_response_count": 4,
                "captures_inflight_llm": False,
            },
        ).with_integrity()
        late_process_manifest = CheckpointManifest(
            schema_version="v1",
            checkpoint_id=CheckpointId("ckpt-3"),
            sandbox_id=sid,
            created_at=utc_now(),
            runtime_name="runc",
            runtime_version=None,
            process_artifacts=[late_process_ref],
            filesystem_artifacts=[],
            metadata={
                "benchmark_trace_cursor": 12,
                "benchmark_latest_mutating_response_count": 8,
                "benchmark_previous_mutating_response_count": 8,
                "captures_inflight_llm": True,
            },
        ).with_integrity()
        fs_worker = RecordingRestoreWorker()
        process_worker = RecordingRestoreWorker()
        restore_worker = DefaultRWorker(
            process_worker=process_worker,
            filesystem_worker=fs_worker,
            checkpoint_manager=ManifestCheckpointManager(
                [early_process_manifest, filesystem_manifest, late_process_manifest]
            ),
        )

        result = restore_worker.restore(
            RestoreJob(
                job_id=JobId("job-restore-missing-fs-mutation"),
                sandbox_id=sid,
                checkpoint_id=CheckpointId("ckpt-3"),
                requested_at=utc_now(),
            )
        )

        self.assertEqual(result.status.value, "succeeded")
        self.assertEqual(process_worker.manifests[0].metadata["process_restore_checkpoint_id"], "ckpt-1")
        self.assertEqual(process_worker.manifests[0].metadata["filesystem_restore_checkpoint_id"], "ckpt-2")

    def test_runc_runtime_executes_real_commands_via_runner(self) -> None:
        with tempfile.TemporaryDirectory(prefix="crab_runc_runtime_") as tmp:
            runner = FakeCommandRunner()
            base = Path(tmp)
            adapter = RuncRuntimeAdapter(
                command_runner=runner,
                paths=RuncRuntimePaths(
                    state_root=base / "state",
                    bundle_root=base / "bundles",
                    checkpoint_root=base / "checkpoints",
                    zfs_dataset_prefix="pool/crab",
                ),
            )

            process_status = adapter.checkpoint_process(
                SandboxId("sbx-1"),
                CheckpointId("ckpt-1"),
                leave_running=True,
            )
            fs_status = adapter.checkpoint_filesystem(SandboxId("sbx-1"), CheckpointId("ckpt-1"))

            self.assertTrue(process_status.executed)
            self.assertTrue(fs_status.executed)
            self.assertEqual(runner.commands[0][0:3], ("runc", "--root", str(base / "state")))
            self.assertIn("--leave-running=true", runner.commands[0])
            self.assertEqual(runner.commands[1], ("zfs", "snapshot", "pool/crab/sbx-1@ckpt-1"))

    def test_runc_runtime_checkpoint_reports_process_size_and_snapshot_stats(self) -> None:
        with tempfile.TemporaryDirectory(prefix="crab_runc_runtime_stats_") as tmp:
            base = Path(tmp)
            process_dir = base / "checkpoints" / "sbx-1" / "ckpt-1" / "process"
            process_dir.mkdir(parents=True, exist_ok=True)
            (process_dir / "pages-1.img").write_bytes(b"a" * 32)
            (process_dir / "inventory.img").write_bytes(b"b" * 16)
            runner = MappingCommandRunner(
                responses={
                    ("zfs", "get", "-Hp", "-o", "property,value", "written,used", "pool/crab/sbx-1@ckpt-1"): (
                        0,
                        "written\t128\nused\t512\n",
                        "",
                    )
                }
            )
            adapter = RuncRuntimeAdapter(
                command_runner=runner,
                paths=RuncRuntimePaths(
                    state_root=base / "state",
                    bundle_root=base / "bundles",
                    checkpoint_root=base / "checkpoints",
                    zfs_dataset_prefix="pool/crab",
                ),
            )

            process_status = adapter.checkpoint_process(SandboxId("sbx-1"), CheckpointId("ckpt-1"), leave_running=False)
            fs_status = adapter.checkpoint_filesystem(SandboxId("sbx-1"), CheckpointId("ckpt-1"))

            self.assertEqual(process_status.metadata["process_checkpoint_size_bytes"], 48)
            self.assertEqual(process_status.metadata["process_checkpoint_file_count"], 2)
            self.assertEqual(fs_status.metadata["filesystem_checkpoint_written_bytes"], 128)
            self.assertEqual(fs_status.metadata["filesystem_checkpoint_used_bytes"], 512)

    def test_runc_runtime_exec_emits_telemetry_without_deleted_helper(self) -> None:
        with tempfile.TemporaryDirectory(prefix="crab_runc_exec_") as tmp:
            base = Path(tmp)
            telemetry = InMemoryTelemetrySink()
            adapter = RuncRuntimeAdapter(
                telemetry=telemetry,
                paths=RuncRuntimePaths(
                    state_root=base / "state",
                    bundle_root=base / "bundles",
                    checkpoint_root=base / "checkpoints",
                    metadata_root=base / "metadata",
                    zfs_dataset_prefix="pool/crab",
                ),
            )

            completed = subprocess.CompletedProcess(
                args=["runc", "exec"],
                returncode=0,
                stdout="ok\n",
                stderr="",
            )
            with patch("crab.runtime.runc.subprocess.run", return_value=completed):
                result = adapter.exec(
                    SandboxId("sbx-1"),
                    ["/bin/true"],
                    cwd="/app",
                    capture_output=True,
                )

            self.assertEqual(result.returncode, 0)
            event_names = [name for name, _ in telemetry.events]
            self.assertIn("sandbox.runtime_exec.start", event_names)
            self.assertIn("sandbox.runtime_exec.finish", event_names)
            self.assertIn("sandbox.command", event_names)

    def test_restore_worker_emits_restore_gap_and_mixed_source_metrics(self) -> None:
        sid = SandboxId("sbx-1")
        process_manifest = CheckpointManifest(
            schema_version="v1",
            checkpoint_id=CheckpointId("ckpt-1"),
            sandbox_id=sid,
            created_at=utc_now(),
            runtime_name="runc",
            runtime_version=None,
            process_artifacts=[ArtifactReference(kind=ArtifactKind.PROCESS, name="proc", relative_path="p", size_bytes=1, sha256="b")],
            filesystem_artifacts=[],
            metadata={
                "benchmark_trace_cursor": 3,
                "process_checkpoint_size_bytes": 64,
            },
        ).with_integrity()
        filesystem_manifest = CheckpointManifest(
            schema_version="v1",
            checkpoint_id=CheckpointId("ckpt-2"),
            sandbox_id=sid,
            created_at=utc_now(),
            runtime_name="runc",
            runtime_version=None,
            process_artifacts=[],
            filesystem_artifacts=[ArtifactReference(kind=ArtifactKind.FILESYSTEM, name="fs", relative_path="x", size_bytes=1, sha256="a")],
            metadata={
                "benchmark_trace_cursor": 5,
                "filesystem_checkpoint_written_bytes": 128,
            },
        ).with_integrity()
        restore_manifest = CheckpointManifest(
            schema_version="v1",
            checkpoint_id=CheckpointId("ckpt-3"),
            sandbox_id=sid,
            created_at=utc_now(),
            runtime_name="runc",
            runtime_version=None,
            process_artifacts=[],
            filesystem_artifacts=[],
            metadata={},
        ).with_integrity()
        telemetry = InMemoryTelemetrySink()
        fs_worker = RecordingRestoreWorker()
        process_worker = RecordingRestoreWorker()
        restore_worker = DefaultRWorker(
            process_worker=process_worker,
            filesystem_worker=fs_worker,
            checkpoint_manager=ManifestCheckpointManager([process_manifest, filesystem_manifest, restore_manifest]),
            telemetry=telemetry,
        )

        restore_worker.restore(
            RestoreJob(
                job_id=JobId("job-restore-gap"),
                sandbox_id=sid,
                checkpoint_id=CheckpointId("ckpt-3"),
                requested_at=utc_now(),
            )
        )

        metric_by_name = {name: value for name, value, _ in telemetry.metrics}
        self.assertIn("restore.source_gap.turns", metric_by_name)
        self.assertIn("restore.source_gap.ms", metric_by_name)
        self.assertIn("restore.estimated_io_bytes", metric_by_name)
        total_metric = next(item for item in telemetry.metrics if item[0] == "restore.total_ms")
        self.assertTrue(total_metric[2]["restore.mixed_sources"])

    def test_ebpf_inspector_uses_recorded_events(self) -> None:
        collector = InMemoryEBPFEventCollector()
        inspector = EBPFSandboxInspector(collector)
        sandbox_id = SandboxId("sbx-1")
        inspector.upsert_snapshot(
            SandboxSnapshot(
                sandbox_id=sandbox_id,
                runtime_name="runc",
                is_running=True,
                process_changed=False,
                filesystem_changed=False,
                observed_at=utc_now(),
            )
        )
        collector.record(
            EBPFEvent(
                sandbox_id=sandbox_id,
                kind=EBPFEventKind.FILE_WRITE,
                observed_at=utc_now(),
                metadata={"path": "/tmp/x"},
            )
        )

        snapshot = inspector.inspect(sandbox_id)
        self.assertFalse(snapshot.process_changed)
        self.assertTrue(snapshot.filesystem_changed)
        self.assertEqual(snapshot.metadata["ebpf_event_count"], 1)

    def test_ebpf_inspector_clears_only_checkpointed_dimension(self) -> None:
        collector = InMemoryEBPFEventCollector()
        inspector = EBPFSandboxInspector(collector)
        sandbox_id = SandboxId("sbx-1")
        base_time = utc_now()
        inspector.upsert_snapshot(
            SandboxSnapshot(
                sandbox_id=sandbox_id,
                runtime_name="runc",
                is_running=True,
                process_changed=True,
                filesystem_changed=True,
                observed_at=base_time,
            )
        )

        snapshot = inspector.inspect(sandbox_id)
        self.assertTrue(snapshot.process_changed)
        self.assertTrue(snapshot.filesystem_changed)

        checkpoint_time = utc_now()
        inspector.mark_checkpoint_complete(
            sandbox_id,
            process=True,
            filesystem=False,
            at=checkpoint_time,
        )

        updated = inspector.inspect(sandbox_id)
        self.assertFalse(updated.process_changed)
        self.assertTrue(updated.filesystem_changed)

    def test_local_storage_implements_checkpoint_manager_contract(self) -> None:
        with tempfile.TemporaryDirectory(prefix="crab_contract_") as tmp:
            mgr = LocalCheckpointManager(StorageConfig(root_dir=Path(tmp)))
            self.assertIsInstance(mgr, CheckpointManager)


if __name__ == "__main__":
    unittest.main()
