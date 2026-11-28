from __future__ import annotations

import os
import tempfile
import time
import unittest
from datetime import timedelta
from pathlib import Path
from unittest import mock

from crab import (
    ArtifactKind,
    ArtifactPayload,
    ArtifactReference,
    CheckpointId,
    CheckpointManifest,
    DeleteAfterRestoreCheckpointManager,
    KeepAllCheckpointManager,
    LatestOnlyCheckpointManager,
    LocalCheckpointManager,
    SandboxId,
    StorageConfig,
)
from crab.models import utc_now
from crab.workers.composite import resolve_restore_manifest


class StorageTests(unittest.TestCase):
    def _reference(self, kind: ArtifactKind, name: str) -> ArtifactReference:
        return ArtifactReference(
            kind=kind,
            name=name,
            relative_path=f"artifacts/sbx/{name}",
            size_bytes=1,
            sha256="0" * 64,
            metadata={},
        )

    def _flush_manager(self, manager) -> None:
        flush = getattr(manager, "flush", None)
        if callable(flush):
            flush()

    def test_local_checkpoint_manager_artifacts_and_manifests(self) -> None:
        with tempfile.TemporaryDirectory(prefix="crab_storage_") as tmp:
            mgr = LocalCheckpointManager(StorageConfig(root_dir=Path(tmp)))
            sid = SandboxId("sbx-1")
            ckpt = CheckpointId("ckpt-1")

            artifact_ref = mgr.put_artifact(
                sid,
                ckpt,
                ArtifactPayload(
                    kind=ArtifactKind.PROCESS,
                    name="proc.bin",
                    data=b"hello",
                    metadata={"m": 1},
                ),
            )
            loaded = mgr.get_artifact(sid, ckpt, artifact_ref)
            self.assertEqual(loaded, b"hello")

            manifest = CheckpointManifest(
                schema_version="v1",
                checkpoint_id=ckpt,
                sandbox_id=sid,
                created_at=utc_now(),
                runtime_name="docker",
                runtime_version="stub",
                process_artifacts=[artifact_ref],
                filesystem_artifacts=[],
                metadata={"x": "y"},
            ).with_integrity()
            mgr.put_manifest(manifest)

            loaded_manifest = mgr.get_manifest(sid, ckpt)
            self.assertEqual(loaded_manifest.runtime_name, "docker")
            self.assertEqual(len(loaded_manifest.process_artifacts), 1)

    def test_list_checkpoints_sorted_by_created_at(self) -> None:
        with tempfile.TemporaryDirectory(prefix="crab_storage_") as tmp:
            mgr = LocalCheckpointManager(StorageConfig(root_dir=Path(tmp)))
            sid = SandboxId("sbx-1")

            c1 = CheckpointId("ckpt-1")
            c2 = CheckpointId("ckpt-2")
            base = utc_now()
            m1 = CheckpointManifest(
                schema_version="v1",
                checkpoint_id=c1,
                sandbox_id=sid,
                created_at=base,
                runtime_name="docker",
                runtime_version=None,
                process_artifacts=[],
                filesystem_artifacts=[],
                metadata={},
            ).with_integrity()
            m2 = CheckpointManifest(
                schema_version="v1",
                checkpoint_id=c2,
                sandbox_id=sid,
                created_at=base + timedelta(seconds=1),
                runtime_name="docker",
                runtime_version=None,
                process_artifacts=[],
                filesystem_artifacts=[],
                metadata={},
            ).with_integrity()
            mgr.put_manifest(m1)
            mgr.put_manifest(m2)

            found = mgr.list_checkpoints(sid)
            self.assertEqual(found, [c1, c2])

    def test_list_checkpoints_ignores_manifest_removed_during_scan(self) -> None:
        with tempfile.TemporaryDirectory(prefix="crab_storage_") as tmp:
            mgr = LocalCheckpointManager(StorageConfig(root_dir=Path(tmp)))
            sid = SandboxId("sbx-1")
            c1 = CheckpointId("ckpt-1")
            c2 = CheckpointId("ckpt-2")
            base = utc_now()
            for checkpoint_id, created_at in (
                (c1, base),
                (c2, base + timedelta(seconds=1)),
            ):
                mgr.put_manifest(
                    CheckpointManifest(
                        schema_version="v1",
                        checkpoint_id=checkpoint_id,
                        sandbox_id=sid,
                        created_at=created_at,
                        runtime_name="docker",
                        runtime_version=None,
                        process_artifacts=[],
                        filesystem_artifacts=[],
                        metadata={},
                    ).with_integrity()
                )

            disappearing_path = Path(tmp) / "manifests" / str(sid) / f"{c2}.json"
            original_read_bytes = Path.read_bytes

            def flaky_read_bytes(path_self: Path, *args, **kwargs) -> bytes:
                if path_self == disappearing_path:
                    path_self.unlink()
                    raise FileNotFoundError(path_self)
                return original_read_bytes(path_self, *args, **kwargs)

            with mock.patch("pathlib.Path.read_bytes", autospec=True, side_effect=flaky_read_bytes):
                self.assertEqual(mgr.list_checkpoints(sid), [c1])

    def test_latest_only_manager_prune_tolerates_manifest_removed_between_list_and_load(self) -> None:
        with tempfile.TemporaryDirectory(prefix="crab_storage_") as tmp:
            sid = SandboxId("sbx-1")
            base = LocalCheckpointManager(StorageConfig(root_dir=Path(tmp)))
            mgr = LatestOnlyCheckpointManager(base)
            m1 = CheckpointManifest(
                schema_version="v1",
                checkpoint_id=CheckpointId("ckpt-1"),
                sandbox_id=sid,
                created_at=utc_now(),
                runtime_name="runc",
                runtime_version=None,
                process_artifacts=[],
                filesystem_artifacts=[],
                metadata={},
            ).with_integrity()
            m2 = CheckpointManifest(
                schema_version="v1",
                checkpoint_id=CheckpointId("ckpt-2"),
                sandbox_id=sid,
                created_at=utc_now() + timedelta(seconds=1),
                runtime_name="runc",
                runtime_version=None,
                process_artifacts=[],
                filesystem_artifacts=[],
                metadata={},
            ).with_integrity()
            mgr.put_manifest(m1)
            mgr.put_manifest(m2)

            original_get_manifest = base.get_manifest

            def flaky_get_manifest(sandbox_id: SandboxId, checkpoint_id: CheckpointId) -> CheckpointManifest:
                if checkpoint_id == m1.checkpoint_id:
                    raise FileNotFoundError("manifest disappeared")
                return original_get_manifest(sandbox_id, checkpoint_id)

            with mock.patch.object(base, "get_manifest", side_effect=flaky_get_manifest):
                self.assertEqual(mgr._protected_checkpoint_ids(sid), {m2.checkpoint_id})

    def test_delete_checkpoint_removes_manifest_artifacts_and_runtime_refs(self) -> None:
        with tempfile.TemporaryDirectory(prefix="crab_storage_") as tmp:
            root = Path(tmp)
            destroyed_refs: list[str] = []
            mgr = LocalCheckpointManager(
                StorageConfig(root_dir=root),
                destroy_filesystem_ref=destroyed_refs.append,
            )
            sid = SandboxId("sbx-1")
            ckpt = CheckpointId("ckpt-1")
            process_dir = root / "runtime" / str(sid) / str(ckpt) / "process"
            process_dir.mkdir(parents=True, exist_ok=True)
            (process_dir / "dump.img").write_text("data")

            process_ref = mgr.put_artifact(
                sid,
                ckpt,
                ArtifactPayload(
                    kind=ArtifactKind.PROCESS,
                    name="process_checkpoint.json",
                    data=(
                        '{"process_checkpoint_location": "%s", "status": {"metadata": {"image_path": "%s"}}}'
                        % (process_dir, process_dir)
                    ).encode("utf-8"),
                ),
            )
            fs_ref = mgr.put_artifact(
                sid,
                ckpt,
                ArtifactPayload(
                    kind=ArtifactKind.FILESYSTEM,
                    name="filesystem_checkpoint.json",
                    data=b'{"filesystem": {"snapshot": "pool/crab/sbx-1@ckpt-1"}}',
                ),
            )
            mgr.put_manifest(
                CheckpointManifest(
                    schema_version="v1",
                    checkpoint_id=ckpt,
                    sandbox_id=sid,
                    created_at=utc_now(),
                    runtime_name="runc",
                    runtime_version=None,
                    process_artifacts=[process_ref],
                    filesystem_artifacts=[fs_ref],
                    metadata={},
                ).with_integrity()
            )

            mgr.delete_checkpoint(sid, ckpt)

            self.assertFalse((root / "manifests" / str(sid) / f"{ckpt}.json").exists())
            self.assertFalse((root / "artifacts" / str(sid) / str(ckpt)).exists())
            self.assertFalse(process_dir.parent.exists())
            # Legacy payload (pre-fs_ref) carries only the bare `snapshot`
            # key; retention forwards it verbatim to the runtime hook, and
            # the zfs provider accepts the unprefixed spelling.
            self.assertEqual(destroyed_refs, ["pool/crab/sbx-1@ckpt-1"])

    def test_delete_checkpoint_prefers_fs_ref_over_legacy_snapshot_key(self) -> None:
        with tempfile.TemporaryDirectory(prefix="crab_storage_") as tmp:
            root = Path(tmp)
            destroyed_refs: list[str] = []
            mgr = LocalCheckpointManager(
                StorageConfig(root_dir=root),
                destroy_filesystem_ref=destroyed_refs.append,
            )
            sid = SandboxId("sbx-1")
            ckpt = CheckpointId("ckpt-1")
            fs_ref = mgr.put_artifact(
                sid,
                ckpt,
                ArtifactPayload(
                    kind=ArtifactKind.FILESYSTEM,
                    name="filesystem_checkpoint.json",
                    data=b'{"filesystem": {"fs_ref": "zfs:pool/crab/sbx-1@ckpt-1", "snapshot": "pool/crab/sbx-1@ckpt-1"}}',
                ),
            )
            mgr.put_manifest(
                CheckpointManifest(
                    schema_version="v1",
                    checkpoint_id=ckpt,
                    sandbox_id=sid,
                    created_at=utc_now(),
                    runtime_name="runc",
                    runtime_version=None,
                    process_artifacts=[],
                    filesystem_artifacts=[fs_ref],
                    metadata={},
                ).with_integrity()
            )

            mgr.delete_checkpoint(sid, ckpt)

            self.assertEqual(destroyed_refs, ["zfs:pool/crab/sbx-1@ckpt-1"])

    def test_delete_checkpoint_without_destroy_hook_still_removes_manifest(self) -> None:
        # In-memory setups (and runtimes without filesystem checkpoints)
        # install no destroy hook; retention must still remove the manifest
        # and artifacts instead of wedging on snapshot cleanup.
        with tempfile.TemporaryDirectory(prefix="crab_storage_") as tmp:
            root = Path(tmp)
            mgr = LocalCheckpointManager(StorageConfig(root_dir=root))
            sid = SandboxId("sbx-1")
            ckpt = CheckpointId("ckpt-1")
            fs_ref = mgr.put_artifact(
                sid,
                ckpt,
                ArtifactPayload(
                    kind=ArtifactKind.FILESYSTEM,
                    name="filesystem_checkpoint.json",
                    data=b'{"filesystem": {"fs_ref": "zfs:pool/crab/sbx-1@ckpt-1"}}',
                ),
            )
            mgr.put_manifest(
                CheckpointManifest(
                    schema_version="v1",
                    checkpoint_id=ckpt,
                    sandbox_id=sid,
                    created_at=utc_now(),
                    runtime_name="runc",
                    runtime_version=None,
                    process_artifacts=[],
                    filesystem_artifacts=[fs_ref],
                    metadata={},
                ).with_integrity()
            )

            mgr.delete_checkpoint(sid, ckpt)

            self.assertFalse((root / "manifests" / str(sid) / f"{ckpt}.json").exists())

    def test_latest_only_manager_prunes_older_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory(prefix="crab_storage_") as tmp:
            sid = SandboxId("sbx-1")
            base = LocalCheckpointManager(StorageConfig(root_dir=Path(tmp)))
            mgr = LatestOnlyCheckpointManager(base)
            m1 = CheckpointManifest(
                schema_version="v1",
                checkpoint_id=CheckpointId("ckpt-1"),
                sandbox_id=sid,
                created_at=utc_now(),
                runtime_name="runc",
                runtime_version=None,
                process_artifacts=[],
                filesystem_artifacts=[],
                metadata={},
            ).with_integrity()
            m2 = CheckpointManifest(
                schema_version="v1",
                checkpoint_id=CheckpointId("ckpt-2"),
                sandbox_id=sid,
                created_at=utc_now() + timedelta(seconds=1),
                runtime_name="runc",
                runtime_version=None,
                process_artifacts=[],
                filesystem_artifacts=[],
                metadata={},
            ).with_integrity()
            for manifest in (m1, m2):
                mgr.put_manifest(manifest)
                mgr.handle_checkpoint_complete(manifest)
            self._flush_manager(mgr)

            self.assertEqual(mgr.list_checkpoints(sid), [CheckpointId("ckpt-2")])

    def test_latest_only_manager_keeps_latest_filesystem_ancestor_for_process_only_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory(prefix="crab_storage_") as tmp:
            sid = SandboxId("sbx-1")
            base = LocalCheckpointManager(StorageConfig(root_dir=Path(tmp)))
            mgr = LatestOnlyCheckpointManager(base)
            m1 = CheckpointManifest(
                schema_version="v1",
                checkpoint_id=CheckpointId("ckpt-1"),
                sandbox_id=sid,
                created_at=utc_now(),
                runtime_name="runc",
                runtime_version=None,
                process_artifacts=[self._reference(ArtifactKind.PROCESS, "p1")],
                filesystem_artifacts=[self._reference(ArtifactKind.FILESYSTEM, "f1")],
                metadata={},
            ).with_integrity()
            m2 = CheckpointManifest(
                schema_version="v1",
                checkpoint_id=CheckpointId("ckpt-2"),
                sandbox_id=sid,
                created_at=utc_now() + timedelta(seconds=1),
                runtime_name="runc",
                runtime_version=None,
                process_artifacts=[self._reference(ArtifactKind.PROCESS, "p2")],
                filesystem_artifacts=[],
                metadata={},
            ).with_integrity()
            for manifest in (m1, m2):
                mgr.put_manifest(manifest)
                mgr.handle_checkpoint_complete(manifest)
            self._flush_manager(mgr)

            self.assertEqual(mgr.list_checkpoints(sid), [CheckpointId("ckpt-1"), CheckpointId("ckpt-2")])

    def test_latest_only_manager_keeps_latest_process_ancestor_for_filesystem_only_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory(prefix="crab_storage_") as tmp:
            sid = SandboxId("sbx-1")
            base = LocalCheckpointManager(StorageConfig(root_dir=Path(tmp)))
            mgr = LatestOnlyCheckpointManager(base)
            m1 = CheckpointManifest(
                schema_version="v1",
                checkpoint_id=CheckpointId("ckpt-1"),
                sandbox_id=sid,
                created_at=utc_now(),
                runtime_name="runc",
                runtime_version=None,
                process_artifacts=[self._reference(ArtifactKind.PROCESS, "p1")],
                filesystem_artifacts=[self._reference(ArtifactKind.FILESYSTEM, "f1")],
                metadata={},
            ).with_integrity()
            m2 = CheckpointManifest(
                schema_version="v1",
                checkpoint_id=CheckpointId("ckpt-2"),
                sandbox_id=sid,
                created_at=utc_now() + timedelta(seconds=1),
                runtime_name="runc",
                runtime_version=None,
                process_artifacts=[],
                filesystem_artifacts=[self._reference(ArtifactKind.FILESYSTEM, "f2")],
                metadata={},
            ).with_integrity()
            for manifest in (m1, m2):
                mgr.put_manifest(manifest)
                mgr.handle_checkpoint_complete(manifest)
            self._flush_manager(mgr)

            self.assertEqual(mgr.list_checkpoints(sid), [CheckpointId("ckpt-1"), CheckpointId("ckpt-2")])

    def test_latest_only_manager_can_keep_all_filesystem_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory(prefix="crab_storage_") as tmp:
            sid = SandboxId("sbx-1")
            base = LocalCheckpointManager(StorageConfig(root_dir=Path(tmp)))
            mgr = LatestOnlyCheckpointManager(base, delete_filesystem_checkpoints=False)
            manifests = [
                CheckpointManifest(
                    schema_version="v1",
                    checkpoint_id=CheckpointId("ckpt-1"),
                    sandbox_id=sid,
                    created_at=utc_now(),
                    runtime_name="runc",
                    runtime_version=None,
                    process_artifacts=[self._reference(ArtifactKind.PROCESS, "p1")],
                    filesystem_artifacts=[self._reference(ArtifactKind.FILESYSTEM, "f1")],
                    metadata={},
                ).with_integrity(),
                CheckpointManifest(
                    schema_version="v1",
                    checkpoint_id=CheckpointId("ckpt-2"),
                    sandbox_id=sid,
                    created_at=utc_now() + timedelta(seconds=1),
                    runtime_name="runc",
                    runtime_version=None,
                    process_artifacts=[self._reference(ArtifactKind.PROCESS, "p2")],
                    filesystem_artifacts=[],
                    metadata={},
                ).with_integrity(),
                CheckpointManifest(
                    schema_version="v1",
                    checkpoint_id=CheckpointId("ckpt-3"),
                    sandbox_id=sid,
                    created_at=utc_now() + timedelta(seconds=2),
                    runtime_name="runc",
                    runtime_version=None,
                    process_artifacts=[self._reference(ArtifactKind.PROCESS, "p3")],
                    filesystem_artifacts=[self._reference(ArtifactKind.FILESYSTEM, "f3")],
                    metadata={},
                ).with_integrity(),
                CheckpointManifest(
                    schema_version="v1",
                    checkpoint_id=CheckpointId("ckpt-4"),
                    sandbox_id=sid,
                    created_at=utc_now() + timedelta(seconds=3),
                    runtime_name="runc",
                    runtime_version=None,
                    process_artifacts=[self._reference(ArtifactKind.PROCESS, "p4")],
                    filesystem_artifacts=[],
                    metadata={},
                ).with_integrity(),
            ]
            for manifest in manifests:
                mgr.put_manifest(manifest)
                mgr.handle_checkpoint_complete(manifest)
            self._flush_manager(mgr)

            self.assertEqual(
                mgr.list_checkpoints(sid),
                [CheckpointId("ckpt-1"), CheckpointId("ckpt-3"), CheckpointId("ckpt-4")],
            )

    def test_latest_only_manager_prunes_in_background(self) -> None:
        with tempfile.TemporaryDirectory(prefix="crab_storage_") as tmp:
            sid = SandboxId("sbx-1")
            base = LocalCheckpointManager(StorageConfig(root_dir=Path(tmp)))
            mgr = LatestOnlyCheckpointManager(base)
            m1 = CheckpointManifest(
                schema_version="v1",
                checkpoint_id=CheckpointId("ckpt-1"),
                sandbox_id=sid,
                created_at=utc_now(),
                runtime_name="runc",
                runtime_version=None,
                process_artifacts=[],
                filesystem_artifacts=[],
                metadata={},
            ).with_integrity()
            m2 = CheckpointManifest(
                schema_version="v1",
                checkpoint_id=CheckpointId("ckpt-2"),
                sandbox_id=sid,
                created_at=utc_now() + timedelta(seconds=1),
                runtime_name="runc",
                runtime_version=None,
                process_artifacts=[],
                filesystem_artifacts=[],
                metadata={},
            ).with_integrity()
            mgr.put_manifest(m1)
            mgr.handle_checkpoint_complete(m1)
            self._flush_manager(mgr)

            original_delete = mgr.delete_checkpoint

            def slow_delete(sandbox_id: SandboxId, checkpoint_id: CheckpointId) -> None:
                time.sleep(0.1)
                original_delete(sandbox_id, checkpoint_id)

            with mock.patch.object(mgr, "delete_checkpoint", side_effect=slow_delete):
                mgr.put_manifest(m2)
                started = time.perf_counter()
                mgr.handle_checkpoint_complete(m2)
                duration = time.perf_counter() - started
                self.assertLess(duration, 0.05)
                self._flush_manager(mgr)

            self.assertEqual(mgr.list_checkpoints(sid), [CheckpointId("ckpt-2")])

    def test_latest_only_manager_keeps_safe_process_restore_dependency_for_latest_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory(prefix="crab_storage_") as tmp:
            sid = SandboxId("sbx-1")
            base = LocalCheckpointManager(StorageConfig(root_dir=Path(tmp)))
            mgr = LatestOnlyCheckpointManager(base)
            m1 = CheckpointManifest(
                schema_version="v1",
                checkpoint_id=CheckpointId("ckpt-1"),
                sandbox_id=sid,
                created_at=utc_now(),
                runtime_name="runc",
                runtime_version=None,
                process_artifacts=[self._reference(ArtifactKind.PROCESS, "p1")],
                filesystem_artifacts=[],
                metadata={"benchmark_trace_cursor": 4, "benchmark_latest_mutating_response_count": 4},
            ).with_integrity()
            m2 = CheckpointManifest(
                schema_version="v1",
                checkpoint_id=CheckpointId("ckpt-2"),
                sandbox_id=sid,
                created_at=utc_now() + timedelta(seconds=1),
                runtime_name="runc",
                runtime_version=None,
                process_artifacts=[],
                filesystem_artifacts=[self._reference(ArtifactKind.FILESYSTEM, "f2")],
                metadata={
                    "benchmark_trace_cursor": 10,
                    "benchmark_latest_mutating_response_count": 4,
                },
            ).with_integrity()
            m3 = CheckpointManifest(
                schema_version="v1",
                checkpoint_id=CheckpointId("ckpt-3"),
                sandbox_id=sid,
                created_at=utc_now() + timedelta(seconds=2),
                runtime_name="runc",
                runtime_version=None,
                process_artifacts=[self._reference(ArtifactKind.PROCESS, "p3")],
                filesystem_artifacts=[],
                metadata={
                    "benchmark_trace_cursor": 12,
                    "benchmark_latest_mutating_response_count": 8,
                    "benchmark_previous_mutating_response_count": 8,
                    "captures_inflight_llm": True,
                },
            ).with_integrity()
            for manifest in (m1, m2, m3):
                mgr.put_manifest(manifest)
                mgr.handle_checkpoint_complete(manifest)
            self._flush_manager(mgr)

            self.assertEqual(
                mgr.list_checkpoints(sid),
                [CheckpointId("ckpt-1"), CheckpointId("ckpt-2"), CheckpointId("ckpt-3")],
            )

    def test_latest_only_manager_keeps_pinned_checkpoint_during_prune(self) -> None:
        with tempfile.TemporaryDirectory(prefix="crab_storage_") as tmp:
            sid = SandboxId("sbx-1")
            base = LocalCheckpointManager(StorageConfig(root_dir=Path(tmp)))
            mgr = LatestOnlyCheckpointManager(base)
            m1 = CheckpointManifest(
                schema_version="v1",
                checkpoint_id=CheckpointId("ckpt-1"),
                sandbox_id=sid,
                created_at=utc_now(),
                runtime_name="runc",
                runtime_version=None,
                process_artifacts=[],
                filesystem_artifacts=[],
                metadata={},
            ).with_integrity()
            m2 = CheckpointManifest(
                schema_version="v1",
                checkpoint_id=CheckpointId("ckpt-2"),
                sandbox_id=sid,
                created_at=utc_now() + timedelta(seconds=1),
                runtime_name="runc",
                runtime_version=None,
                process_artifacts=[],
                filesystem_artifacts=[],
                metadata={},
            ).with_integrity()
            m3 = CheckpointManifest(
                schema_version="v1",
                checkpoint_id=CheckpointId("ckpt-3"),
                sandbox_id=sid,
                created_at=utc_now() + timedelta(seconds=2),
                runtime_name="runc",
                runtime_version=None,
                process_artifacts=[],
                filesystem_artifacts=[],
                metadata={},
            ).with_integrity()
            for manifest in (m1, m2):
                mgr.put_manifest(manifest)
                mgr.handle_checkpoint_complete(manifest)
            self._flush_manager(mgr)

            self.assertTrue(mgr.pin_checkpoint(sid, CheckpointId("ckpt-2")))
            mgr.put_manifest(m3)
            mgr.handle_checkpoint_complete(m3)
            self._flush_manager(mgr)
            self.assertEqual(mgr.list_checkpoints(sid), [CheckpointId("ckpt-2"), CheckpointId("ckpt-3")])

            mgr.unpin_checkpoint(sid, CheckpointId("ckpt-2"))
            mgr.handle_checkpoint_complete(m3)
            self._flush_manager(mgr)
            self.assertEqual(mgr.list_checkpoints(sid), [CheckpointId("ckpt-3")])

    def test_resolve_restore_manifest_can_reuse_preloaded_candidates(self) -> None:
        with tempfile.TemporaryDirectory(prefix="crab_storage_") as tmp:
            sid = SandboxId("sbx-1")
            base = LocalCheckpointManager(StorageConfig(root_dir=Path(tmp)))
            manifests = [
                CheckpointManifest(
                    schema_version="v1",
                    checkpoint_id=CheckpointId("ckpt-1"),
                    sandbox_id=sid,
                    created_at=utc_now(),
                    runtime_name="runc",
                    runtime_version=None,
                    process_artifacts=[self._reference(ArtifactKind.PROCESS, "p1")],
                    filesystem_artifacts=[self._reference(ArtifactKind.FILESYSTEM, "f1")],
                    metadata={"benchmark_trace_cursor": 1, "benchmark_latest_mutating_response_count": 1},
                ).with_integrity(),
                CheckpointManifest(
                    schema_version="v1",
                    checkpoint_id=CheckpointId("ckpt-2"),
                    sandbox_id=sid,
                    created_at=utc_now() + timedelta(seconds=1),
                    runtime_name="runc",
                    runtime_version=None,
                    process_artifacts=[],
                    filesystem_artifacts=[self._reference(ArtifactKind.FILESYSTEM, "f2")],
                    metadata={"benchmark_trace_cursor": 2, "benchmark_latest_mutating_response_count": 1},
                ).with_integrity(),
            ]
            for manifest in manifests:
                base.put_manifest(manifest)

            with mock.patch.object(base, "list_checkpoints", side_effect=AssertionError("unexpected list")):
                with mock.patch.object(base, "get_manifest", side_effect=AssertionError("unexpected load")):
                    resolved = resolve_restore_manifest(base, manifests[-1], candidates=manifests)

            self.assertEqual(resolved.checkpoint_id, CheckpointId("ckpt-2"))
            self.assertEqual(
                resolved.metadata["process_restore_checkpoint_id"],
                "ckpt-1",
            )
            self.assertEqual(
                resolved.metadata["filesystem_restore_checkpoint_id"],
                "ckpt-2",
            )

    def test_logical_manifest_resolves_explicit_physical_sources(self) -> None:
        with tempfile.TemporaryDirectory(prefix="crab_storage_") as tmp:
            sid = SandboxId("sbx-logical")
            mgr = LocalCheckpointManager(StorageConfig(root_dir=Path(tmp)))
            physical = CheckpointManifest(
                schema_version="v1",
                checkpoint_id=CheckpointId("ckpt-physical"),
                sandbox_id=sid,
                created_at=utc_now(),
                runtime_name="runc",
                runtime_version=None,
                process_artifacts=[self._reference(ArtifactKind.PROCESS, "p1")],
                filesystem_artifacts=[
                    self._reference(ArtifactKind.FILESYSTEM, "f1")
                ],
                metadata={},
                parent_checkpoint_id=CheckpointId("ckpt-parent"),
                process_kind="incremental",
            ).with_integrity()
            logical = CheckpointManifest(
                schema_version="v1",
                checkpoint_id=CheckpointId("ckpt-logical"),
                sandbox_id=sid,
                created_at=utc_now() + timedelta(seconds=1),
                runtime_name="runc",
                runtime_version=None,
                process_artifacts=[],
                filesystem_artifacts=[],
                metadata={
                    "logical_checkpoint": True,
                    "checkpoint_materialization": "reused",
                    "process_restore_checkpoint_id": "ckpt-physical",
                    "filesystem_restore_checkpoint_id": "ckpt-physical",
                },
            ).with_integrity()
            mgr.put_manifest(physical)
            mgr.put_manifest(logical)

            resolved = resolve_restore_manifest(mgr, logical)

            self.assertEqual(resolved.checkpoint_id, CheckpointId("ckpt-logical"))
            self.assertEqual(resolved.process_artifacts, physical.process_artifacts)
            self.assertEqual(
                resolved.filesystem_artifacts, physical.filesystem_artifacts
            )
            self.assertEqual(resolved.process_kind, "incremental")
            self.assertEqual(
                resolved.parent_checkpoint_id, CheckpointId("ckpt-parent")
            )

    def test_physical_checkpoint_delete_protects_logical_descendant(self) -> None:
        with tempfile.TemporaryDirectory(prefix="crab_storage_") as tmp:
            sid = SandboxId("sbx-logical")
            mgr = LocalCheckpointManager(StorageConfig(root_dir=Path(tmp)))
            physical_id = CheckpointId("ckpt-physical")
            logical_id = CheckpointId("ckpt-logical")
            for manifest in (
                CheckpointManifest(
                    schema_version="v1",
                    checkpoint_id=physical_id,
                    sandbox_id=sid,
                    created_at=utc_now(),
                    runtime_name="runc",
                    runtime_version=None,
                    process_artifacts=[],
                    filesystem_artifacts=[],
                    metadata={},
                ).with_integrity(),
                CheckpointManifest(
                    schema_version="v1",
                    checkpoint_id=logical_id,
                    sandbox_id=sid,
                    created_at=utc_now() + timedelta(seconds=1),
                    runtime_name="runc",
                    runtime_version=None,
                    process_artifacts=[],
                    filesystem_artifacts=[],
                    metadata={
                        "logical_checkpoint": True,
                        "process_restore_checkpoint_id": str(physical_id),
                        "filesystem_restore_checkpoint_id": str(physical_id),
                    },
                ).with_integrity(),
            ):
                mgr.put_manifest(manifest)

            with self.assertRaisesRegex(RuntimeError, "logical restore dependents"):
                mgr.delete_checkpoint(sid, physical_id)

            mgr.delete_checkpoint(sid, physical_id, cascade=True)
            self.assertEqual(mgr.list_checkpoints(sid), [])

    def test_delete_after_restore_manager_removes_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory(prefix="crab_storage_") as tmp:
            sid = SandboxId("sbx-1")
            ckpt = CheckpointId("ckpt-1")
            base = LocalCheckpointManager(StorageConfig(root_dir=Path(tmp)))
            mgr = DeleteAfterRestoreCheckpointManager(base)
            manifest = CheckpointManifest(
                schema_version="v1",
                checkpoint_id=ckpt,
                sandbox_id=sid,
                created_at=utc_now(),
                runtime_name="runc",
                runtime_version=None,
                process_artifacts=[],
                filesystem_artifacts=[],
                metadata={},
            ).with_integrity()
            mgr.put_manifest(manifest)

            mgr.handle_restore_complete(sid, ckpt)

            self.assertEqual(mgr.list_checkpoints(sid), [])

    def _real_artifact(
        self,
        mgr: LocalCheckpointManager,
        sid: SandboxId,
        ckpt: CheckpointId,
        kind: ArtifactKind,
        name: str,
        data: bytes,
    ) -> ArtifactReference:
        return mgr.put_artifact(
            sid,
            ckpt,
            ArtifactPayload(kind=kind, name=name, data=data, metadata={}),
        )

    def test_link_ancestor_artifact_creates_relative_symlink(self) -> None:
        with tempfile.TemporaryDirectory(prefix="crab_storage_") as tmp:
            mgr = LocalCheckpointManager(StorageConfig(root_dir=Path(tmp)))
            src = SandboxId("source")
            tgt = SandboxId("target")
            ckpt = CheckpointId("ckpt-anc")
            src_ref = self._real_artifact(mgr, src, ckpt, ArtifactKind.PROCESS, "proc.bin", b"PAYLOAD")

            linked_ref = mgr.link_ancestor_artifact(src, tgt, ckpt, src_ref)

            self.assertEqual(linked_ref.size_bytes, src_ref.size_bytes)
            self.assertEqual(linked_ref.sha256, src_ref.sha256)
            self.assertNotEqual(linked_ref.relative_path, src_ref.relative_path)
            target_path = Path(tmp) / linked_ref.relative_path
            self.assertTrue(target_path.is_symlink())
            # Symlink target is relative for portability across moves.
            self.assertFalse(Path(os.readlink(target_path)).is_absolute())
            self.assertTrue(mgr.is_linked_artifact(tgt, ckpt, linked_ref))
            # Reading through the link returns the source bytes; SHA256 still
            # validates inside get_artifact.
            self.assertEqual(mgr.get_artifact(tgt, ckpt, linked_ref), b"PAYLOAD")

    def test_link_ancestor_artifact_idempotent(self) -> None:
        with tempfile.TemporaryDirectory(prefix="crab_storage_") as tmp:
            mgr = LocalCheckpointManager(StorageConfig(root_dir=Path(tmp)))
            src = SandboxId("source")
            tgt = SandboxId("target")
            ckpt = CheckpointId("ckpt-anc")
            src_ref = self._real_artifact(mgr, src, ckpt, ArtifactKind.PROCESS, "proc.bin", b"X")
            mgr.link_ancestor_artifact(src, tgt, ckpt, src_ref)
            # Second call replaces the symlink in place rather than erroring.
            relinked = mgr.link_ancestor_artifact(src, tgt, ckpt, src_ref)
            self.assertTrue(mgr.is_linked_artifact(tgt, ckpt, relinked))

    def test_materialize_linked_artifacts_replaces_with_real_files(self) -> None:
        with tempfile.TemporaryDirectory(prefix="crab_storage_") as tmp:
            mgr = LocalCheckpointManager(StorageConfig(root_dir=Path(tmp)))
            src = SandboxId("source")
            tgt = SandboxId("target")
            ckpt = CheckpointId("ckpt-anc")
            src_ref = self._real_artifact(mgr, src, ckpt, ArtifactKind.PROCESS, "proc.bin", b"INLINE")
            linked_ref = mgr.link_ancestor_artifact(src, tgt, ckpt, src_ref)

            count = mgr.materialize_linked_artifacts(tgt)
            self.assertEqual(count, 1)
            target_path = Path(tmp) / linked_ref.relative_path
            self.assertFalse(target_path.is_symlink())
            self.assertEqual(target_path.read_bytes(), b"INLINE")
            # Idempotent: a second call finds nothing to materialize.
            self.assertEqual(mgr.materialize_linked_artifacts(tgt), 0)

    def test_pin_chain_protects_ancestors_from_pruning(self) -> None:
        with tempfile.TemporaryDirectory(prefix="crab_storage_") as tmp:
            sid = SandboxId("chain-sbx")
            base = LocalCheckpointManager(StorageConfig(root_dir=Path(tmp)))
            mgr = LatestOnlyCheckpointManager(base)
            anchor = CheckpointId("ckpt-anchor")
            mid = CheckpointId("ckpt-mid")
            leaf = CheckpointId("ckpt-leaf")
            new = CheckpointId("ckpt-new")
            anchor_m = CheckpointManifest(
                schema_version="v1",
                checkpoint_id=anchor,
                sandbox_id=sid,
                created_at=utc_now(),
                runtime_name="runc",
                runtime_version=None,
                process_artifacts=[self._reference(ArtifactKind.PROCESS, "p")],
                filesystem_artifacts=[self._reference(ArtifactKind.FILESYSTEM, "f")],
                metadata={},
                process_kind="full",
            ).with_integrity()
            mid_m = CheckpointManifest(
                schema_version="v1",
                checkpoint_id=mid,
                sandbox_id=sid,
                created_at=utc_now() + timedelta(seconds=1),
                runtime_name="runc",
                runtime_version=None,
                process_artifacts=[self._reference(ArtifactKind.PROCESS, "p")],
                filesystem_artifacts=[self._reference(ArtifactKind.FILESYSTEM, "f")],
                metadata={},
                process_kind="incremental",
                parent_checkpoint_id=anchor,
            ).with_integrity()
            leaf_m = CheckpointManifest(
                schema_version="v1",
                checkpoint_id=leaf,
                sandbox_id=sid,
                created_at=utc_now() + timedelta(seconds=2),
                runtime_name="runc",
                runtime_version=None,
                process_artifacts=[self._reference(ArtifactKind.PROCESS, "p")],
                filesystem_artifacts=[self._reference(ArtifactKind.FILESYSTEM, "f")],
                metadata={},
                process_kind="incremental",
                parent_checkpoint_id=mid,
            ).with_integrity()
            new_m = CheckpointManifest(
                schema_version="v1",
                checkpoint_id=new,
                sandbox_id=sid,
                created_at=utc_now() + timedelta(seconds=3),
                runtime_name="runc",
                runtime_version=None,
                process_artifacts=[self._reference(ArtifactKind.PROCESS, "p")],
                filesystem_artifacts=[self._reference(ArtifactKind.FILESYSTEM, "f")],
                metadata={},
                process_kind="full",
            ).with_integrity()
            for manifest in (anchor_m, mid_m, leaf_m):
                mgr.put_manifest(manifest)
                mgr.handle_checkpoint_complete(manifest)
            self._flush_manager(mgr)

            self.assertTrue(mgr.pin_chain(sid, leaf))
            mgr.put_manifest(new_m)
            mgr.handle_checkpoint_complete(new_m)
            self._flush_manager(mgr)
            # All ancestors of leaf must survive even though `new` is newer
            # and the LatestOnly policy would otherwise drop them.
            survived = set(mgr.list_checkpoints(sid))
            self.assertIn(anchor, survived)
            self.assertIn(mid, survived)
            self.assertIn(leaf, survived)

            mgr.unpin_chain(sid, leaf)
            mgr.handle_checkpoint_complete(new_m)
            self._flush_manager(mgr)
            self.assertEqual(set(mgr.list_checkpoints(sid)), {new})

    def test_pin_chain_refcounts_overlap_for_two_forks(self) -> None:
        with tempfile.TemporaryDirectory(prefix="crab_storage_") as tmp:
            sid = SandboxId("chain-sbx")
            base = LocalCheckpointManager(StorageConfig(root_dir=Path(tmp)))
            mgr = LatestOnlyCheckpointManager(base)
            anchor = CheckpointId("ckpt-anchor")
            mid = CheckpointId("ckpt-mid")
            leaf_a = CheckpointId("ckpt-a")
            leaf_b = CheckpointId("ckpt-b")
            anchor_m = CheckpointManifest(
                schema_version="v1",
                checkpoint_id=anchor,
                sandbox_id=sid,
                created_at=utc_now(),
                runtime_name="runc",
                runtime_version=None,
                process_artifacts=[self._reference(ArtifactKind.PROCESS, "p")],
                filesystem_artifacts=[self._reference(ArtifactKind.FILESYSTEM, "f")],
                metadata={},
                process_kind="full",
            ).with_integrity()
            mid_m = CheckpointManifest(
                schema_version="v1",
                checkpoint_id=mid,
                sandbox_id=sid,
                created_at=utc_now() + timedelta(seconds=1),
                runtime_name="runc",
                runtime_version=None,
                process_artifacts=[self._reference(ArtifactKind.PROCESS, "p")],
                filesystem_artifacts=[self._reference(ArtifactKind.FILESYSTEM, "f")],
                metadata={},
                process_kind="incremental",
                parent_checkpoint_id=anchor,
            ).with_integrity()
            leaf_a_m = CheckpointManifest(
                schema_version="v1",
                checkpoint_id=leaf_a,
                sandbox_id=sid,
                created_at=utc_now() + timedelta(seconds=2),
                runtime_name="runc",
                runtime_version=None,
                process_artifacts=[self._reference(ArtifactKind.PROCESS, "p")],
                filesystem_artifacts=[self._reference(ArtifactKind.FILESYSTEM, "f")],
                metadata={},
                process_kind="incremental",
                parent_checkpoint_id=mid,
            ).with_integrity()
            leaf_b_m = CheckpointManifest(
                schema_version="v1",
                checkpoint_id=leaf_b,
                sandbox_id=sid,
                created_at=utc_now() + timedelta(seconds=3),
                runtime_name="runc",
                runtime_version=None,
                process_artifacts=[self._reference(ArtifactKind.PROCESS, "p")],
                filesystem_artifacts=[self._reference(ArtifactKind.FILESYSTEM, "f")],
                metadata={},
                process_kind="incremental",
                parent_checkpoint_id=mid,
            ).with_integrity()
            for manifest in (anchor_m, mid_m, leaf_a_m, leaf_b_m):
                mgr.put_manifest(manifest)

            # Two forks share `anchor` and `mid` ancestors. Unpinning one
            # fork's chain must not drop pins still held by the other.
            self.assertTrue(mgr.pin_chain(sid, leaf_a))
            self.assertTrue(mgr.pin_chain(sid, leaf_b))
            mgr.unpin_chain(sid, leaf_a)
            self._flush_manager(mgr)
            mgr.handle_checkpoint_complete(leaf_b_m)
            self._flush_manager(mgr)
            survived = set(mgr.list_checkpoints(sid))
            # leaf_b's chain (anchor, mid, leaf_b) still pinned; leaf_a is
            # not protected anymore (LatestOnly default behavior would also
            # protect latest checkpoints, but we're checking ancestors).
            self.assertIn(anchor, survived)
            self.assertIn(mid, survived)
            self.assertIn(leaf_b, survived)

    def test_runtime_image_path_in_use_defers_prune(self) -> None:
        """``LocalCheckpointManager._delete_process_runtime_paths`` must
        consult the runtime safety predicate before pruning a runtime
        checkpoint tree. When the runtime says the path is in use (an
        active lazy-pages daemon is reading from it), the dir stays on
        disk; the next retention pass after the daemon exits cleans it
        up. Without this gate, the kernel would SIGBUS the restored
        process the next time it raised a userfault for an unloaded
        page."""
        with tempfile.TemporaryDirectory(prefix="crab_storage_") as tmp:
            checkpoint_root = Path(tmp) / "runtime-state"
            (checkpoint_root / "process").mkdir(parents=True)
            (checkpoint_root / "process" / "pages-1.img").write_bytes(b"PAGES")

            calls: list[Path] = []

            def in_use(path: Path) -> bool:
                calls.append(path)
                # First call (with daemon alive): defer.
                return len(calls) == 1

            mgr = LocalCheckpointManager(
                StorageConfig(root_dir=Path(tmp)),
                runtime_image_path_in_use=in_use,
            )
            payload = {
                "process_checkpoint_location": str(checkpoint_root / "process"),
            }
            # First prune attempt: predicate says in-use → directory stays.
            mgr._delete_process_runtime_paths(payload)
            self.assertTrue(checkpoint_root.exists())
            self.assertTrue((checkpoint_root / "process" / "pages-1.img").exists())
            # Second pass (predicate now says free): directory is removed.
            mgr._delete_process_runtime_paths(payload)
            self.assertFalse(checkpoint_root.exists())
            self.assertEqual(len(calls), 2)

    def test_runtime_image_path_in_use_predicate_exception_does_not_wedge_retention(self) -> None:
        with tempfile.TemporaryDirectory(prefix="crab_storage_") as tmp:
            checkpoint_root = Path(tmp) / "runtime-state"
            (checkpoint_root / "process").mkdir(parents=True)

            def boom(path: Path) -> bool:
                raise RuntimeError("predicate is broken")

            mgr = LocalCheckpointManager(
                StorageConfig(root_dir=Path(tmp)),
                runtime_image_path_in_use=boom,
            )
            payload = {"process_checkpoint_location": str(checkpoint_root / "process")}
            # Exception is swallowed (logged); prune proceeds rather than
            # leaking storage indefinitely.
            mgr._delete_process_runtime_paths(payload)
            self.assertFalse(checkpoint_root.exists())

    def test_runtime_image_path_in_use_late_bind(self) -> None:
        with tempfile.TemporaryDirectory(prefix="crab_storage_") as tmp:
            mgr = LocalCheckpointManager(StorageConfig(root_dir=Path(tmp)))
            self.assertIsNone(mgr._runtime_image_path_in_use)
            mgr.set_runtime_image_path_in_use(lambda _p: True)
            self.assertTrue(callable(mgr._runtime_image_path_in_use))

    def test_keep_all_manager_retains_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory(prefix="crab_storage_") as tmp:
            sid = SandboxId("sbx-1")
            base = LocalCheckpointManager(StorageConfig(root_dir=Path(tmp)))
            mgr = KeepAllCheckpointManager(base)
            manifest = CheckpointManifest(
                schema_version="v1",
                checkpoint_id=CheckpointId("ckpt-1"),
                sandbox_id=sid,
                created_at=utc_now(),
                runtime_name="runc",
                runtime_version=None,
                process_artifacts=[],
                filesystem_artifacts=[],
                metadata={},
            ).with_integrity()
            mgr.put_manifest(manifest)
            mgr.handle_checkpoint_complete(manifest)
            self._flush_manager(mgr)

            self.assertEqual(mgr.list_checkpoints(sid), [CheckpointId("ckpt-1")])


if __name__ == "__main__":
    unittest.main()
