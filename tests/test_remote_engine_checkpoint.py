from __future__ import annotations

import unittest

from crab.ids import CheckpointId, SandboxId
from crab.models import JobStatus
from crab.remote_engine import RuntimeProxy, _SystemShim


class _FakeClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, object]] = []

    def get_json(self, path: str):
        self.calls.append(("GET", path, None))
        return {
            "checkpoints": [
                {
                    "checkpoint_id": "ckpt-one",
                    "created_at": "2026-08-02T12:00:00+00:00",
                    "label": "safe-point",
                    "has_process": True,
                    "has_filesystem": True,
                }
            ]
        }

    def post_json(self, path: str, payload: object, *, timeout_seconds: float):
        self.calls.append(("POST", path, payload))
        if path.endswith("/restore"):
            return {"status": "succeeded"}
        return {"checkpoint_id": "ckpt-two", "status": "succeeded"}

    def _request_json(self, method: str, path: str, *, body: bytes | None):
        self.calls.append((method, path, body))
        return {"ok": True}


class RemoteCheckpointShimTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = _FakeClient()
        self.system = _SystemShim(self.client)  # type: ignore[arg-type]
        self.sandbox_id = SandboxId("sandbox-one")

    def test_checkpoint_and_restore_are_routed_to_daemon(self) -> None:
        created = self.system.checkpoint_once(self.sandbox_id)
        self.assertEqual(created.status, JobStatus.SUCCEEDED)
        self.assertEqual(str(created.manifest.checkpoint_id), "ckpt-two")

        restored = self.system.restore_once(self.sandbox_id, CheckpointId("ckpt-two"))
        self.assertEqual(restored.status, JobStatus.SUCCEEDED)

    def test_restore_failure_surfaces_daemon_message(self) -> None:
        zfs_error = (
            "command failed (1): zfs rollback -r crab/sandboxes/sbx@ckpt"
            "\nstdout: \nstderr: cannot rollback to 'crab/sandboxes/sbx@ckpt':"
            " clones of previous snapshots exist"
        )

        class _FailingClient(_FakeClient):
            def post_json(self, path: str, payload: object, *, timeout_seconds: float):
                if path.endswith("/restore"):
                    return {"status": "failed", "message": zfs_error}
                return super().post_json(path, payload, timeout_seconds=timeout_seconds)

        system = _SystemShim(_FailingClient())  # type: ignore[arg-type]
        restored = system.restore_once(self.sandbox_id, CheckpointId("ckpt-two"))
        self.assertEqual(restored.status, JobStatus.FAILED)
        self.assertIn("clones of previous snapshots exist", restored.message)

    def test_checkpoint_failure_keeps_id_without_success_manifest(self) -> None:
        class _FailingClient(_FakeClient):
            def post_json(self, path: str, payload: object, *, timeout_seconds: float):
                if path.endswith("/checkpoints"):
                    return {
                        "ok": False,
                        "checkpoint_id": "ckpt-failed",
                        "status": "failed",
                        "message": "dump failed",
                    }
                return super().post_json(path, payload, timeout_seconds=timeout_seconds)

        system = _SystemShim(_FailingClient())  # type: ignore[arg-type]
        created = system.checkpoint_once(self.sandbox_id)
        self.assertEqual(created.status, JobStatus.FAILED)
        self.assertEqual(str(created.checkpoint_id), "ckpt-failed")
        self.assertIsNone(created.manifest)
        self.assertEqual(created.message, "dump failed")

    def test_checkpoint_storage_lists_metadata_and_deletes(self) -> None:
        checkpoint_ids = self.system.storage.list_checkpoints(self.sandbox_id)
        self.assertEqual([str(item) for item in checkpoint_ids], ["ckpt-one"])

        manifest = self.system.storage.get_manifest(self.sandbox_id, checkpoint_ids[0])
        self.assertEqual(manifest.metadata["label"], "safe-point")
        self.assertTrue(manifest.process_artifacts)
        self.assertTrue(manifest.filesystem_artifacts)

        self.system.storage.delete_checkpoint(self.sandbox_id, checkpoint_ids[0], cascade=True)
        self.assertEqual(self.client.calls[-1][0], "DELETE")
        self.assertIn(b'"cascade": true', self.client.calls[-1][2])


class _RecordingClient:
    def __init__(self) -> None:
        self.posts: list[str] = []
        self.deletes: list[str] = []

    def post_json(self, path: str, payload: object, *, timeout_seconds: float | None = None):
        _ = payload
        _ = timeout_seconds
        self.posts.append(path)
        return {"ok": True}

    def delete(self, path: str, *, timeout_seconds: float | None = None):
        _ = timeout_seconds
        self.deletes.append(path)
        return {"ok": True}


class RemoteRuntimeLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = _RecordingClient()
        self.runtime = RuntimeProxy(self.client, name="runc")  # type: ignore[arg-type]
        self.sandbox_id = SandboxId("sbx-one")

    def test_lifecycle_ops_route_to_matching_endpoints(self) -> None:
        self.runtime.stop(self.sandbox_id)
        self.runtime.pause(self.sandbox_id)
        self.runtime.resume(self.sandbox_id)
        self.runtime.start(self.sandbox_id)
        self.runtime.restart(self.sandbox_id)

        self.assertEqual(
            self.client.posts,
            [
                "/sandboxes/sbx-one/stop",
                "/sandboxes/sbx-one/pause",
                "/sandboxes/sbx-one/resume",
                "/sandboxes/sbx-one/start",
                "/sandboxes/sbx-one/restart",
            ],
        )

    def test_stop_is_graceful_not_delete(self) -> None:
        # stop keeps the sandbox (POST /stop); only kill->delete destroys it.
        self.runtime.stop(self.sandbox_id)
        self.assertEqual(self.client.deletes, [])
        self.assertEqual(self.client.posts, ["/sandboxes/sbx-one/stop"])


if __name__ == "__main__":
    unittest.main()
