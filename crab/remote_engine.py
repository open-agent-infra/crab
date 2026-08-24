"""SDK-side proxy that makes a remote Crab daemon look like a local Engine.

`Engine.connect(socket=...)` returns one of these. The proxy exposes the
same attribute surface `Sandbox`/`Agent` already consume from a local
`Engine` — `.runtime`, `.config`, path roots, network helpers, upstream
registration — and translates each call into one of the daemon's
HTTP-over-Unix-socket endpoints.

The proxy itself is intentionally thin:
  - `RemoteEngine` caches the daemon's `/info` payload at connect time
    so callers can read paths without round-tripping every access.
  - `RuntimeProxy` implements the subset of the `Runtime` contract the
    SDK actually uses (`name`, `launch`, `exec`, `write_bundle_spec`,
    `pause`, `resume`, `delete`, `inspect_runtime`, `describe`,
    `update_host_inspector_filters`). Anything else raises a clear
    `NotImplementedError` so we don't silently route checkpoint internals
    over the wire.
  - A no-op `_LocalInspectorShim` keeps `Sandbox._mark_inspector_running`
    happy; the real inspector lives in the daemon and is reached via the
    runtime's `_register_with_host_inspector` path during launch.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Mapping

from .contracts import Runtime
from .daemon import DaemonClient
from .ids import CheckpointId, SandboxId
from .models import JobStatus, SandboxDescription, SandboxExecResult, SandboxRuntimeState
from .telemetry import NoopTelemetrySink

if TYPE_CHECKING:
    from .sandbox import Sandbox

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _NetworkLease:
    """Lightweight stand-in for `runtime.network.NetworkLease`.

    The daemon owns the real bridge + lease registry; we only carry the
    fields the SDK reads when building runc bundle metadata."""

    namespace_path: str
    guest_ip: str


class _LocalInspectorShim:
    """Inspector stand-in used on the SDK side.

    The daemon's host inspector is the source of truth in daemon mode;
    the local `Sandbox._mark_inspector_running` call is best-effort
    bookkeeping for the in-process world. Swallow the call so we don't
    have to refactor every existing call site that pokes the inspector
    pre-launch."""

    def upsert_snapshot(self, *_, **__) -> None:
        return None

    def mark_changed(self, *_, **__) -> None:
        return None


class _RemoteStorageShim:
    """Checkpoint storage facade backed by the daemon's public routes."""

    def __init__(self, client: DaemonClient) -> None:
        self._client = client
        self._entries: dict[tuple[str, str], dict[str, Any]] = {}

    def list_checkpoints(self, sandbox_id: SandboxId) -> list[CheckpointId]:
        response = self._client.get_json(f"/sandboxes/{sandbox_id}/checkpoints")
        entries = list(response.get("checkpoints") or [])
        checkpoint_ids: list[CheckpointId] = []
        for entry in entries:
            checkpoint_id = CheckpointId(str(entry["checkpoint_id"]))
            checkpoint_ids.append(checkpoint_id)
            self._entries[(str(sandbox_id), str(checkpoint_id))] = dict(entry)
        return checkpoint_ids

    def get_manifest(self, sandbox_id: SandboxId, checkpoint_id: CheckpointId):
        key = (str(sandbox_id), str(checkpoint_id))
        if key not in self._entries:
            self.list_checkpoints(sandbox_id)
        entry = self._entries.get(key)
        if entry is None:
            raise KeyError(f"checkpoint not found: {checkpoint_id}")
        created_at = entry.get("created_at")
        return SimpleNamespace(
            checkpoint_id=checkpoint_id,
            created_at=(datetime.fromisoformat(str(created_at)) if created_at else None),
            process_artifacts=([True] if entry.get("has_process") else []),
            filesystem_artifacts=([True] if entry.get("has_filesystem") else []),
            metadata={"label": entry.get("label")} if entry.get("label") else {},
        )

    def delete_checkpoint(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
        *,
        cascade: bool = False,
    ) -> None:
        payload = {"cascade": True} if cascade else None
        self._client._request_json(
            "DELETE",
            f"/sandboxes/{sandbox_id}/checkpoints/{checkpoint_id}",
            body=None if payload is None else json.dumps(payload).encode("utf-8"),
        )
        self._entries.pop((str(sandbox_id), str(checkpoint_id)), None)


class _SystemShim:
    """`Engine.system`-shaped facade for SDK checkpoint operations."""

    def __init__(self, client: DaemonClient) -> None:
        self._client = client
        self.telemetry = NoopTelemetrySink()
        self.inspector = _LocalInspectorShim()
        self.storage = _RemoteStorageShim(client)

    def checkpoint_once(self, sandbox_id: SandboxId, *, leave_running: bool = True):
        response = self._client.post_json(
            f"/sandboxes/{sandbox_id}/checkpoints",
            {"leave_running": bool(leave_running)},
            timeout_seconds=300.0,
        )
        checkpoint_id_raw = response.get("checkpoint_id")
        checkpoint_id = CheckpointId(str(checkpoint_id_raw)) if checkpoint_id_raw else None
        status = JobStatus(str(response.get("status") or JobStatus.SUCCEEDED.value))
        manifest = None if checkpoint_id is None else SimpleNamespace(checkpoint_id=checkpoint_id)
        return SimpleNamespace(
            checkpoint_id=checkpoint_id,
            manifest=manifest,
            status=status,
            message="",
        )

    def restore_once(self, sandbox_id: SandboxId, checkpoint_id: CheckpointId):
        response = self._client.post_json(
            f"/sandboxes/{sandbox_id}/checkpoints/{checkpoint_id}/restore",
            {},
            timeout_seconds=300.0,
        )
        status = JobStatus(str(response.get("status") or JobStatus.SUCCEEDED.value))
        return SimpleNamespace(status=status, message="")


class _ConfigShim:
    """`Engine.config`-shaped object exposing only what the SDK reads."""

    def __init__(self, info: Mapping[str, Any]) -> None:
        self.default_image = info.get("default_image") or "ubuntu:22.04"
        self.runtime = str(info.get("runtime") or "runc")
        self.enable_sandbox_network = info.get("network_bridge_ip") is not None
        self.enable_interceptor = bool(info.get("interceptor_base_url"))


class RuntimeProxy(Runtime):
    """Implements the `Runtime` contract by RPCing into the daemon.

    Only the operations the SDK actually calls are wired up; the rest
    (checkpoint primitives, ZFS clone helpers, pre-dump linkage) belong
    to the daemon's own `RuncRuntime` and are not exposed over the wire."""

    def __init__(self, client: DaemonClient, *, name: str) -> None:
        self._client = client
        self._name = name

    @property
    def name(self) -> str:  # type: ignore[override]
        return self._name

    @property
    def version(self) -> str | None:  # type: ignore[override]
        return None

    @property
    def capabilities(self):  # type: ignore[override]
        # The SDK never reads this in current code paths; returning the
        # daemon's view requires a route we haven't added yet, so raise
        # if someone starts to depend on it.
        raise NotImplementedError("RuntimeProxy.capabilities is not exposed in daemon mode v1")

    def launch(
        self,
        runtime_name: str,
        metadata: dict[str, object] | None = None,
    ) -> SandboxId:  # type: ignore[override]
        payload: dict[str, Any] = {"runtime_name": runtime_name}
        if metadata is not None:
            payload["metadata"] = _make_jsonable(metadata)
        result = self._client.post_json("/sandboxes", payload)
        return SandboxId(str(result["sandbox_id"]))

    def stop(self, sandbox_id: SandboxId) -> None:  # type: ignore[override]
        # `stop` and `delete` are folded into the daemon's `DELETE /sandboxes/{id}`
        # endpoint, which calls runtime.stop + runtime.delete + lease/upstream
        # cleanup. The SDK's `Sandbox.kill()` already calls `delete` directly,
        # so this entry point is rarely hit; expose it to keep the contract
        # complete.
        self._client.delete(f"/sandboxes/{sandbox_id}")

    def pause(self, sandbox_id: SandboxId) -> None:  # type: ignore[override]
        # Pause/resume aren't routed in v1 — the daemon's scheduler drives
        # them internally. Explicit SDK pause is rare; raise if used.
        raise NotImplementedError("Sandbox.pause from the SDK is not exposed in daemon mode v1")

    def resume(self, sandbox_id: SandboxId) -> None:  # type: ignore[override]
        raise NotImplementedError("Sandbox.resume from the SDK is not exposed in daemon mode v1")

    def sync_runtime_state(self, sandbox_id, *, is_running: bool) -> None:  # type: ignore[override]
        return None  # daemon-side bookkeeping

    def prepare_for_restore(self, sandbox_id) -> None:  # type: ignore[override]
        return None

    def mark_restored(self, sandbox_id) -> None:  # type: ignore[override]
        return None

    def delete(self, sandbox_id: SandboxId) -> None:  # type: ignore[override]
        self._client.delete(f"/sandboxes/{sandbox_id}")

    def describe(self, sandbox_id: SandboxId) -> SandboxDescription:  # type: ignore[override]
        result = self._client.get_json(f"/sandboxes/{sandbox_id}")
        return _deserialize_description(result["description"])

    def write_bundle_spec(self, bundle_dir: Path) -> None:  # type: ignore[override]
        self._client.post_json(
            "/runtime/write_bundle_spec",
            {"bundle_dir": str(bundle_dir)},
        )

    def inspect_runtime(self, sandbox_id: SandboxId) -> SandboxRuntimeState:  # type: ignore[override]
        result = self._client.get_json(f"/sandboxes/{sandbox_id}")
        runtime_state = result.get("runtime_state")
        if runtime_state is None:
            return SandboxRuntimeState(
                sandbox_id=sandbox_id,
                runtime_name=self._name,
                status="unknown",
                metadata={},
            )
        return _deserialize_runtime_state(runtime_state)

    def exec(  # type: ignore[override]
        self,
        sandbox_id: SandboxId,
        argv: list[str],
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        user: str | None = None,
        timeout_s: float | None = None,
        capture_output: bool = True,
    ) -> SandboxExecResult:
        payload: dict[str, Any] = {
            "argv": list(argv),
            "capture_output": bool(capture_output),
        }
        if cwd is not None:
            payload["cwd"] = cwd
        if env is not None:
            payload["env"] = dict(env)
        if user is not None:
            payload["user"] = user
        if timeout_s is not None:
            payload["timeout_s"] = float(timeout_s)
        # The daemon's `runtime.exec` can block for the full task timeout,
        # so the HTTP timeout must comfortably exceed it — otherwise the
        # client gives up before the daemon returns. 60s of headroom is
        # plenty for the response read after the exec completes.
        if timeout_s is not None:
            http_timeout = float(timeout_s) + 60.0
        else:
            http_timeout = None  # use client default
        response = self._client.post_json(
            f"/sandboxes/{sandbox_id}/exec",
            payload,
            timeout_seconds=http_timeout,
        )
        raw = response["result"]
        return SandboxExecResult(
            args=tuple(raw.get("args") or []),
            returncode=int(raw["returncode"]),
            stdout=str(raw.get("stdout") or ""),
            stderr=str(raw.get("stderr") or ""),
        )

    # ----- daemon-only API surface the SDK needs in addition to Runtime -----

    def update_host_inspector_filters(
        self,
        sandbox_id: SandboxId,
        *,
        ignore_process_rules=None,
        ignored_path_prefixes=None,
    ) -> None:
        payload: dict[str, Any] = {}
        if ignore_process_rules is not None:
            payload["ignore_process_rules"] = _make_jsonable(ignore_process_rules)
        if ignored_path_prefixes is not None:
            payload["ignored_path_prefixes"] = list(ignored_path_prefixes)
        self._client.post_json(
            f"/sandboxes/{sandbox_id}/host_inspector/filters",
            payload,
        )

    # The remaining Runtime contract members are checkpoint internals
    # that the SDK does not call directly; raise loudly if some code
    # path starts depending on them.

    def resilient_exec(self, *_, **__):  # type: ignore[override]
        raise NotImplementedError("RuntimeProxy.resilient_exec is daemon-side only")

    def checkpoint_process(self, *_, **__):  # type: ignore[override]
        raise NotImplementedError("RuntimeProxy.checkpoint_process is daemon-side only")

    def pre_dump_process(self, *_, **__):  # type: ignore[override]
        raise NotImplementedError("RuntimeProxy.pre_dump_process is daemon-side only")

    def restore_process(self, *_, **__):  # type: ignore[override]
        raise NotImplementedError("RuntimeProxy.restore_process is daemon-side only")

    def process_checkpoint_location(self, *_, **__):  # type: ignore[override]
        raise NotImplementedError("RuntimeProxy.process_checkpoint_location is daemon-side only")

    def pre_dump_location(self, *_, **__):  # type: ignore[override]
        raise NotImplementedError("RuntimeProxy.pre_dump_location is daemon-side only")

    def link_ancestor_pre_dump(self, *_, **__):  # type: ignore[override]
        raise NotImplementedError("RuntimeProxy.link_ancestor_pre_dump is daemon-side only")

    def materialize_linked_pre_dumps(self, *_, **__):  # type: ignore[override]
        raise NotImplementedError("RuntimeProxy.materialize_linked_pre_dumps is daemon-side only")

    def runtime_image_path_in_use(self, *_, **__):  # type: ignore[override]
        raise NotImplementedError("RuntimeProxy.runtime_image_path_in_use is daemon-side only")

    def checkpoint_filesystem(self, *_, **__):  # type: ignore[override]
        raise NotImplementedError("RuntimeProxy.checkpoint_filesystem is daemon-side only")

    def restore_filesystem(self, *_, **__):  # type: ignore[override]
        raise NotImplementedError("RuntimeProxy.restore_filesystem is daemon-side only")

    def filesystem_checkpoint_metadata(self, *_, **__):  # type: ignore[override]
        raise NotImplementedError("RuntimeProxy.filesystem_checkpoint_metadata is daemon-side only")

    def discard_partial_checkpoint(self, *_, **__):  # type: ignore[override]
        raise NotImplementedError("RuntimeProxy.discard_partial_checkpoint is daemon-side only")

    def delete_runtime(self, *_, **__):  # type: ignore[override]
        raise NotImplementedError("RuntimeProxy.delete_runtime is daemon-side only")

    def destroy_filesystem_dataset(self, *_, **__):  # type: ignore[override]
        raise NotImplementedError("RuntimeProxy.destroy_filesystem_dataset is daemon-side only")

    def clone_filesystem_snapshot(self, *_, **__):  # type: ignore[override]
        raise NotImplementedError("RuntimeProxy.clone_filesystem_snapshot is daemon-side only")


class RemoteEngine:
    """SDK-side facade over the daemon. Returned from `Engine.connect(...)`.

    Holds a `DaemonClient` and a snapshot of the daemon's `/info` payload
    so attribute reads (`engine.runtime_root` etc.) are local. The
    snapshot is refreshed lazily — if the daemon has changed its paths
    between connects, the SDK process must reconnect."""

    def __init__(self, client: DaemonClient, *, info: Mapping[str, Any]) -> None:
        self._client = client
        self._info = dict(info)
        self._runtime = RuntimeProxy(client, name=str(info.get("runtime") or "runc"))
        self._system = _SystemShim(client)
        self._config = _ConfigShim(info)
        self._sandboxes_lock = threading.Lock()
        self._sandboxes: "dict[SandboxId, Sandbox]" = {}

    # ----- accessors used by Sandbox / Agent -----

    @property
    def started(self) -> bool:
        return True

    @property
    def config(self) -> _ConfigShim:
        return self._config

    @property
    def runtime(self) -> RuntimeProxy:
        return self._runtime

    @property
    def system(self) -> _SystemShim:
        return self._system

    def _path(self, key: str) -> Path:
        value = self._info.get(key)
        if not value:
            raise RuntimeError(
                f"daemon /info did not return `{key}`; cannot resolve sandbox host path"
            )
        return Path(str(value))

    @property
    def storage_root(self) -> Path:
        return self._path("storage_root")

    @property
    def runtime_root(self) -> Path:
        return self._path("runtime_root")

    @property
    def image_cache_root(self) -> Path:
        return self._path("image_cache_root")

    @property
    def work_dir_host_root(self) -> Path:
        return self._path("work_dir_host_root")

    @property
    def agent_state_root(self) -> Path:
        return self._path("agent_state_root")

    @property
    def interceptor_base_url(self) -> str | None:
        return self._info.get("interceptor_base_url")

    @property
    def forwarder_base_url(self) -> str | None:
        return self._info.get("forwarder_base_url")

    @property
    def network_bridge_ip(self) -> str | None:
        return self._info.get("network_bridge_ip")

    # ----- daemon-routed operations -----

    def register_upstream(self, sandbox_id: SandboxId, url: str) -> None:
        if not url:
            return
        self._client.post_json(
            f"/sandboxes/{sandbox_id}/upstream",
            {"url": url},
        )

    def unregister_upstream(self, sandbox_id: SandboxId) -> None:
        try:
            self._client.delete(f"/sandboxes/{sandbox_id}/upstream")
        except Exception:
            logger.debug("unregister_upstream failed for %s", sandbox_id, exc_info=True)

    def allocate_network_lease(self, sandbox_id: SandboxId):
        result = self._client.post_json(
            f"/sandboxes/{sandbox_id}/network/lease",
            {},
        )
        lease = result["lease"]
        return _NetworkLease(
            namespace_path=str(lease["namespace_path"]),
            guest_ip=str(lease["guest_ip"]),
        )

    def release_network_lease(self, sandbox_id: SandboxId) -> None:
        try:
            self._client.delete(f"/sandboxes/{sandbox_id}/network/lease")
        except Exception:
            logger.debug("release_network_lease failed for %s", sandbox_id, exc_info=True)

    def repair_network_lease(self, sandbox_id: SandboxId) -> bool:
        # No corresponding endpoint yet; surface a clean False so the
        # restore path falls back gracefully.
        return False

    # ----- local sandbox registry (mirrors the in-process Engine) -----

    def _register_sandbox(self, sandbox: "Sandbox") -> None:
        with self._sandboxes_lock:
            self._sandboxes[sandbox.sandbox_id] = sandbox

    def _unregister_sandbox(self, sandbox: "Sandbox") -> None:
        with self._sandboxes_lock:
            self._sandboxes.pop(sandbox.sandbox_id, None)

    # ----- lifecycle (no-op on the client side; the daemon owns it) -----

    def stop(self) -> None:
        """SDK clients never own the daemon lifecycle, so `stop()` only
        clears the local sandbox registry. Use `crab daemon stop`
        (or POST /shutdown via DaemonClient) if you actually want the
        daemon to exit."""
        with self._sandboxes_lock:
            self._sandboxes.clear()

    def __enter__(self) -> "RemoteEngine":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    @property
    def client(self) -> DaemonClient:
        """Underlying daemon client. Useful for tests and the `crab` CLI."""
        return self._client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_jsonable(value: Any) -> Any:
    """Coerce metadata dicts containing Path objects (etc.) into JSON-safe
    primitives before sending to the daemon."""
    if isinstance(value, dict):
        return {str(k): _make_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_make_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        import json

        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


def _deserialize_description(payload: Mapping[str, Any]) -> SandboxDescription:
    return SandboxDescription(
        sandbox_id=SandboxId(str(payload["sandbox_id"])),
        runtime_name=str(payload.get("runtime_name") or "runc"),
        status=str(payload.get("status") or "unknown"),
        metadata=dict(payload.get("metadata") or {}),
    )


def _deserialize_runtime_state(payload: Mapping[str, Any]) -> SandboxRuntimeState:
    return SandboxRuntimeState(
        sandbox_id=SandboxId(str(payload["sandbox_id"])),
        runtime_name=str(payload.get("runtime_name") or "runc"),
        status=str(payload.get("status") or "unknown"),
        pid=payload.get("pid"),
        bundle_path=payload.get("bundle_path"),
        metadata=dict(payload.get("metadata") or {}),
    )


__all__ = [
    "RemoteEngine",
    "RuntimeProxy",
]
