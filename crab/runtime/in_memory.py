from __future__ import annotations

from dataclasses import replace
from threading import Lock

from ..contracts import Runtime
from ..ids import CheckpointId, SandboxId
from ..models import RuntimeCapabilities, RuntimeOperationStatus, SandboxDescription, SandboxExecResult, SandboxRuntimeState
from ..remote_inspector import HostInspectorServiceClient


class InMemoryRuntime(Runtime):
    def __init__(
        self,
        *,
        name: str = "docker",
        version: str | None = None,
        host_inspector_client: HostInspectorServiceClient | None = None,
    ) -> None:
        self._name = name
        self._version = version
        self._lock = Lock()
        self._items: dict[SandboxId, SandboxDescription] = {}
        self._host_inspector_client = host_inspector_client

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str | None:
        return self._version

    def capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities(
            supports_process_checkpoint=True,
            supports_filesystem_checkpoint=True,
            supports_incremental_filesystem=False,
            supports_custom_checkpoint_dir=False,
        )

    def launch(self, runtime_name: str, metadata: dict[str, object] | None = None) -> SandboxId:
        with self._lock:
            sandbox_id = SandboxId.new()
            self._items[sandbox_id] = SandboxDescription(
                sandbox_id=sandbox_id,
                runtime_name=runtime_name,
                status="running",
                metadata=dict(metadata or {}),
            )
        return sandbox_id

    def stop(self, sandbox_id: SandboxId) -> None:
        with self._lock:
            cur = self._items[sandbox_id]
            self._items[sandbox_id] = replace(cur, status="stopped")

    def pause(self, sandbox_id: SandboxId) -> None:
        with self._lock:
            cur = self._items[sandbox_id]
            self._items[sandbox_id] = replace(cur, status="paused")

    def resume(self, sandbox_id: SandboxId) -> None:
        with self._lock:
            cur = self._items[sandbox_id]
            self._items[sandbox_id] = replace(cur, status="running")

    def start(self, sandbox_id: SandboxId) -> None:
        with self._lock:
            cur = self._items[sandbox_id]
            self._items[sandbox_id] = replace(cur, status="running")

    def restart(self, sandbox_id: SandboxId) -> None:
        self.stop(sandbox_id)
        self.start(sandbox_id)

    def sync_runtime_state(self, sandbox_id: SandboxId, *, is_running: bool) -> None:
        with self._lock:
            cur = self._items[sandbox_id]
            self._items[sandbox_id] = replace(cur, status="running" if is_running else "stopped")

    def prepare_for_restore(self, sandbox_id: SandboxId) -> None:
        _ = sandbox_id

    def mark_restored(self, sandbox_id: SandboxId) -> None:
        with self._lock:
            cur = self._items[sandbox_id]
            self._items[sandbox_id] = replace(cur, status="running")

    def delete(self, sandbox_id: SandboxId) -> None:
        with self._lock:
            self._items.pop(sandbox_id)

    def describe(self, sandbox_id: SandboxId) -> SandboxDescription:
        with self._lock:
            return self._items[sandbox_id]

    def write_bundle_spec(self, bundle_dir) -> None:
        _ = bundle_dir
        raise NotImplementedError("bundle spec generation is only supported for runc sandboxes")

    def inspect_runtime(self, sandbox_id: SandboxId) -> SandboxRuntimeState:
        description = self.describe(sandbox_id)
        return SandboxRuntimeState(
            sandbox_id=sandbox_id,
            runtime_name=description.runtime_name,
            status=description.status,
            pid=None,
            bundle_path=None,
            metadata=dict(description.metadata),
        )

    def exec(
        self,
        sandbox_id: SandboxId,
        argv: list[str],
        *,
        cwd: str | None = None,
        env: dict[str, object] | None = None,
        user: str | None = None,
        timeout_s: float | None = None,
        capture_output: bool = True,
    ) -> SandboxExecResult:
        _ = (sandbox_id, argv, cwd, env, user, timeout_s, capture_output)
        raise NotImplementedError("sandbox exec is only supported for runc sandboxes")

    def resilient_exec(
        self,
        sandbox_id: SandboxId,
        argv: list[str],
        *,
        cwd: str | None = None,
        env: dict[str, object] | None = None,
        user: str | None = None,
        timeout_s: float | None = None,
        capture_output: bool = True,
    ) -> SandboxExecResult:
        _ = (sandbox_id, argv, cwd, env, user, timeout_s, capture_output)
        raise NotImplementedError("resilient sandbox exec is only supported for runc sandboxes")

    def checkpoint_process(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
        *,
        leave_running: bool,
        parent_checkpoint_id: CheckpointId | None = None,
    ) -> RuntimeOperationStatus:
        metadata: dict[str, object] = {
            "phase": "process_checkpoint",
            "runtime": self._name,
            "sandbox_id": str(sandbox_id),
            "checkpoint_id": str(checkpoint_id),
            "leave_running": leave_running,
        }
        command = [
            "docker",
            "checkpoint",
            "create",
            f"--leave-running={'true' if leave_running else 'false'}",
        ]
        if parent_checkpoint_id is not None:
            metadata["parent_checkpoint_id"] = str(parent_checkpoint_id)
            metadata["process_kind"] = "incremental"
            command.append(f"--parent={parent_checkpoint_id}")
        command.extend([str(sandbox_id), str(checkpoint_id)])
        return RuntimeOperationStatus(
            executed=False,
            reason=f"{self._name}_runtime_not_implemented",
            command=tuple(command),
            metadata=metadata,
        )

    def pre_dump_process(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
        *,
        parent_checkpoint_id: CheckpointId | None = None,
    ) -> RuntimeOperationStatus:
        metadata: dict[str, object] = {
            "phase": "process_pre_dump",
            "runtime": self._name,
            "sandbox_id": str(sandbox_id),
            "checkpoint_id": str(checkpoint_id),
        }
        command = ["docker", "checkpoint", "pre-dump"]
        if parent_checkpoint_id is not None:
            metadata["parent_checkpoint_id"] = str(parent_checkpoint_id)
            command.append(f"--parent={parent_checkpoint_id}")
        command.extend([str(sandbox_id), str(checkpoint_id)])
        return RuntimeOperationStatus(
            executed=False,
            reason=f"{self._name}_runtime_not_implemented",
            command=tuple(command),
            metadata=metadata,
        )

    def restore_process(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> RuntimeOperationStatus:
        metadata = {
            "phase": "process_restore",
            "runtime": self._name,
            "sandbox_id": str(sandbox_id),
            "checkpoint_id": str(checkpoint_id),
        }
        return RuntimeOperationStatus(
            executed=False,
            reason=f"{self._name}_runtime_not_implemented",
            command=tuple(["docker", "start", "--checkpoint", str(checkpoint_id), str(sandbox_id)]),
            metadata=metadata,
        )

    def process_checkpoint_location(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> str | None:
        _ = (sandbox_id, checkpoint_id)
        return None

    def checkpoint_filesystem(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> RuntimeOperationStatus:
        metadata = self.filesystem_checkpoint_metadata(sandbox_id, checkpoint_id)
        return RuntimeOperationStatus(
            executed=False,
            reason=f"{self._name}_runtime_not_implemented",
            command=tuple(["docker", "export", str(sandbox_id), "#=>", f"{checkpoint_id}.tar"]),
            metadata=metadata,
        )

    def restore_filesystem(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> RuntimeOperationStatus:
        metadata = {
            "phase": "filesystem_restore",
            "runtime": self._name,
            "sandbox_id": str(sandbox_id),
            "checkpoint_id": str(checkpoint_id),
        }
        return RuntimeOperationStatus(
            executed=False,
            reason=f"{self._name}_runtime_not_implemented",
            command=tuple(["docker", "import", f"{checkpoint_id}.tar", str(sandbox_id)]),
            metadata=metadata,
        )

    def filesystem_checkpoint_metadata(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> dict[str, object]:
        return {
            "phase": "filesystem_checkpoint",
            "runtime": self._name,
            "sandbox_id": str(sandbox_id),
            "checkpoint_id": str(checkpoint_id),
        }

    def delete_runtime(
        self,
        sandbox_id: SandboxId,
        *,
        force: bool = True,
        ignore_missing: bool = True,
    ) -> None:
        _ = (sandbox_id, force, ignore_missing)
        raise NotImplementedError("runtime deletion is only supported for runc sandboxes")

    def destroy_filesystem_dataset(self, sandbox_id: SandboxId) -> None:
        _ = sandbox_id
        raise NotImplementedError("filesystem dataset management is only supported for runc sandboxes")

    def clone_filesystem_snapshot(
        self,
        source_sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
        target_sandbox_id: SandboxId,
        *,
        target_rootfs_path,
    ) -> str:
        _ = (source_sandbox_id, checkpoint_id, target_sandbox_id, target_rootfs_path)
        raise NotImplementedError("filesystem cloning is only supported for runc sandboxes")
