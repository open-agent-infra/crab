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
from typing import TYPE_CHECKING, Any, Iterator, Mapping

from .contracts import Runtime
from .daemon import DaemonClient, DaemonRequestError
from .errors import (
    ImageAuthenticationError,
    ImageCompatibilityError,
    ImageInsufficientDiskError,
    ImageNotFoundError,
    ImagePlatformError,
    ImagePolicyError,
    ImagePullError,
    ImagePullTimeoutError,
    ImageRateLimitError,
    ImageReferenceError,
    ImageTooLargeError,
    SandboxCreateCleanupError,
    SandboxExecCleanupError,
    SandboxExecTimeout,
)
from .ids import CheckpointId, SandboxId
from .models import ChangesetResult, EgressLedger, EgressReplayReport, ExecDone, ExecEvent, JobStatus, MergeReport, ObservationReport, ProcessMergeReport, SandboxDescription, SandboxExecResult, SandboxRuntimeState, SandboxSnapshot, utc_now
from .journal import ActionRecord
from .merging import MergeError, MergerHook
from .process_merge import ProcessMergeConflict
from .telemetry import NoopTelemetrySink
from .txn import (
    EffectFlushReport,
    TxnAbortError,
    TxnAbortResult,
    TxnActiveError,
    TxnCommitConflict,
    TxnCommitResult,
    TxnDescription,
    TxnMismatchError,
    TxnNotAbortable,
)

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
    pre-launch.

    ``inspect()`` proxies to the daemon's read-only inspector peek route
    (``GET /sandboxes/{id}/inspector``) so the SDK can render the observer
    flags without resetting them."""

    def __init__(self, client: "DaemonClient | None" = None) -> None:
        self._client = client

    def upsert_snapshot(self, *_, **__) -> None:
        return None

    def mark_changed(self, *_, **__) -> None:
        return None

    def inspect(self, sandbox_id: SandboxId) -> SandboxSnapshot:
        """Read-only peek at the daemon's inspector state. Does not reset."""
        if self._client is None:
            return SandboxSnapshot(
                sandbox_id=sandbox_id,
                runtime_name="",
                is_running=True,
                process_changed=False,
                filesystem_changed=False,
                observed_at=utc_now(),
            )
        response = self._client.get_json(f"/sandboxes/{sandbox_id}/inspector")
        return SandboxSnapshot(
            sandbox_id=sandbox_id,
            runtime_name=str(response.get("runtime_name") or ""),
            is_running=bool(response.get("is_running", True)),
            filesystem_changed=bool(response.get("filesystem_changed")),
            process_changed=bool(response.get("process_changed")),
            observed_at=utc_now(),
        )


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
        metadata = {
            key: entry.get(source_key)
            for key, source_key in (
                ("label", "label"),
                ("logical_checkpoint", "logical"),
                ("checkpoint_materialization", "materialization"),
                ("process_restore_checkpoint_id", "process_checkpoint_id"),
                ("filesystem_restore_checkpoint_id", "filesystem_checkpoint_id"),
            )
            if entry.get(source_key) is not None
        }
        return SimpleNamespace(
            checkpoint_id=checkpoint_id,
            created_at=(datetime.fromisoformat(str(created_at)) if created_at else None),
            process_artifacts=(
                [True]
                if entry.get("materialized_process", entry.get("has_process"))
                else []
            ),
            filesystem_artifacts=(
                [True]
                if entry.get("materialized_filesystem", entry.get("has_filesystem"))
                else []
            ),
            restore_process_artifacts=([True] if entry.get("has_process") else []),
            restore_filesystem_artifacts=(
                [True] if entry.get("has_filesystem") else []
            ),
            metadata=metadata,
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


class _JournalShim:
    """`ActionJournal.entries()`-shaped reads over the daemon RPC so
    `Sandbox.actions()` stays transport-agnostic (C3)."""

    def __init__(self, client: DaemonClient) -> None:
        self._client = client

    def entries(
        self,
        sandbox_id: SandboxId | str,
        *,
        kind: str | None = None,
        since_seq: int | None = None,
    ) -> list[ActionRecord]:
        payload: dict[str, Any] = {}
        if kind is not None:
            payload["kind"] = kind
        if since_seq is not None:
            payload["since_seq"] = int(since_seq)
        response = self._client.post_json(
            f"/sandboxes/{sandbox_id}/actions",
            payload,
            timeout_seconds=60.0,
        )
        return [ActionRecord.from_json(row) for row in (response.get("records") or [])]


class _SystemShim:
    """`Engine.system`-shaped facade for SDK checkpoint operations."""

    def __init__(self, client: DaemonClient) -> None:
        self._client = client
        self.telemetry = NoopTelemetrySink()
        self.inspector = _LocalInspectorShim(client)
        self.storage = _RemoteStorageShim(client)
        self.journal = _JournalShim(client)

    def checkpoint_once(
        self,
        sandbox_id: SandboxId,
        *,
        leave_running: bool = True,
        checkpoint_id: str | None = None,
    ):
        payload: dict[str, Any] = {"leave_running": bool(leave_running)}
        if checkpoint_id is not None:
            payload["checkpoint_id"] = str(checkpoint_id)
        response = self._client.post_json(
            f"/sandboxes/{sandbox_id}/checkpoints",
            payload,
            timeout_seconds=300.0,
        )
        checkpoint_id_raw = response.get("checkpoint_id")
        checkpoint_id = CheckpointId(str(checkpoint_id_raw)) if checkpoint_id_raw else None
        status = JobStatus(str(response.get("status") or JobStatus.SUCCEEDED.value))
        manifest = (
            None
            if checkpoint_id is None or status != JobStatus.SUCCEEDED
            else SimpleNamespace(checkpoint_id=checkpoint_id)
        )
        return SimpleNamespace(
            checkpoint_id=checkpoint_id,
            manifest=manifest,
            status=status,
            message=str(response.get("message") or ""),
        )

    def restore_once(self, sandbox_id: SandboxId, checkpoint_id: CheckpointId):
        response = self._client.post_json(
            f"/sandboxes/{sandbox_id}/checkpoints/{checkpoint_id}/restore",
            {},
            timeout_seconds=300.0,
        )
        status = JobStatus(str(response.get("status") or JobStatus.SUCCEEDED.value))
        message = str(response.get("message") or "")
        return SimpleNamespace(status=status, message=message)

    # ----- transactions (see crab/txn.py) ------------------------------
    # Proxies mirror CrabSystem's txn surface so Sandbox.begin() stays
    # transport-agnostic. 409 responses carry an `error_type` the daemon
    # set from the original Txn* exception; map it back so SDK callers
    # get identical exceptions in both modes.

    def begin_txn(
        self,
        sandbox_id: SandboxId,
        *,
        label: str | None = None,
        isolation: str = "snapshot",
        effects: str | None = None,
    ) -> TxnDescription:
        payload: dict[str, Any] = {} if label is None else {"label": label}
        if isolation != "snapshot":
            payload["isolation"] = isolation
        if effects is not None:
            payload["effects"] = effects
        try:
            response = self._client.post_json(
                f"/sandboxes/{sandbox_id}/txn",
                payload,
                timeout_seconds=600.0 if isolation == "fork" else 300.0,  # fork begin = checkpoint + clone + restore
            )
        except DaemonRequestError as exc:
            raise _map_txn_error(exc) from exc
        return _deserialize_txn(response["txn"])

    def commit_txn(
        self, sandbox_id: SandboxId, txn_id: str, *, force: bool = False
    ) -> TxnCommitResult:
        try:
            response = self._client.post_json(
                f"/sandboxes/{sandbox_id}/txn/{txn_id}/commit",
                {"force": True} if force else {},
                timeout_seconds=600.0,  # fork-backed commit swaps fs + processes
            )
        except DaemonRequestError as exc:
            raise _map_txn_error(exc) from exc
        raw = response["result"]
        promoted = raw.get("promoted_checkpoint_id")
        consolidated = raw.get("observations_consolidated")
        raw_effects = raw.get("effects")
        return TxnCommitResult(
            txn_id=str(raw["txn_id"]),
            released_observations=int(raw["released_observations"]),
            base_dropped=bool(raw["base_dropped"]),
            promoted_checkpoint_id=None if promoted is None else str(promoted),
            observations_consolidated=None if consolidated is None else int(consolidated),
            # Absent from pre-D3 daemons.
            effects=(
                None
                if not isinstance(raw_effects, dict)
                else EffectFlushReport.from_json(raw_effects)
            ),
        )

    def abort_txn(
        self, sandbox_id: SandboxId, txn_id: str, *, force: bool = False
    ) -> TxnAbortResult:
        try:
            response = self._client.post_json(
                f"/sandboxes/{sandbox_id}/txn/{txn_id}/abort",
                {"force": True} if force else {},
                timeout_seconds=300.0,  # abort restores the base checkpoint
            )
        except DaemonRequestError as exc:
            raise _map_txn_error(exc) from exc
        raw = response["result"]
        restored = raw.get("restored_checkpoint_id")
        return TxnAbortResult(
            txn_id=str(raw["txn_id"]),
            discarded_observations=int(raw["discarded_observations"]),
            restored_checkpoint_id=None if restored is None else str(restored),
            # Absent from pre-D1 daemons; 0 is the honest default there.
            mutating_egress=int(raw.get("mutating_egress") or 0),
            # Absent from pre-D3 daemons.
            deferred_dropped=int(raw.get("deferred_dropped") or 0),
        )

    def current_txn(self, sandbox_id: SandboxId) -> TxnDescription | None:
        response = self._client.get_json(f"/sandboxes/{sandbox_id}/txn")
        raw = response.get("txn")
        return None if raw is None else _deserialize_txn(raw)

    # ----- filesystem merge / changesets (C2) --------------------------
    # Same transport-agnostic story as the txn proxies: Sandbox.merge and
    # Sandbox.changeset duck-type onto these, and 409 merge_error bodies
    # rehydrate into MergeError with the serialized report reattached.

    def merge_from_fork(
        self,
        source_sandbox_id: SandboxId,
        fork_sandbox_id: SandboxId,
        *,
        policy: str = "fail_fast",
        ignore_prefixes: tuple[str, ...] | None = None,
        merger: MergerHook | None = None,
        observations: str = "none",
        observation_summarizer=None,
    ) -> MergeReport:
        if merger is not None:
            raise NotImplementedError(
                "custom merger hooks cannot cross the daemon RPC; "
                "use a policy or run a local in-process engine"
            )
        if observation_summarizer is not None:
            raise NotImplementedError(
                "summarizer hooks cannot cross the daemon RPC; "
                "run a local in-process engine"
            )
        payload: dict[str, Any] = {
            "fork_sandbox_id": str(fork_sandbox_id),
            "policy": policy,
        }
        if observations != "none":
            payload["observations"] = observations
        if ignore_prefixes is not None:
            payload["ignore_prefixes"] = [str(prefix) for prefix in ignore_prefixes]
        try:
            response = self._client.post_json(
                f"/sandboxes/{source_sandbox_id}/merge",
                payload,
                timeout_seconds=600.0,  # quiesce + two backend diffs + apply
            )
        except DaemonRequestError as exc:
            raise _map_merge_error(exc) from exc
        return MergeReport.from_json(response["report"])

    def changeset_since(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
        *,
        use_inspector_gate: bool = True,
    ) -> ChangesetResult:
        payload: dict[str, Any] = {"since": str(checkpoint_id)}
        # force=True on the daemon side bypasses the inspector gate.
        if not use_inspector_gate:
            payload["force"] = True
        response = self._client.post_json(
            f"/sandboxes/{sandbox_id}/changeset",
            payload,
            timeout_seconds=300.0,
        )
        return ChangesetResult.from_json(response["changeset"])

    def fork_changeset(
        self, sandbox_id: SandboxId, *, force: bool = False
    ) -> ChangesetResult:
        payload: dict[str, Any] = {}
        if force:
            payload["force"] = True
        response = self._client.post_json(
            f"/sandboxes/{sandbox_id}/changeset",
            payload,
            timeout_seconds=300.0,
        )
        return ChangesetResult.from_json(response["changeset"])

    def consolidate_observations(
        self,
        source_sandbox_id: SandboxId,
        fork_sandbox_id: SandboxId,
        *,
        policy: str = "append",
        summarizer=None,
        reason: str = "manual",
    ) -> ObservationReport:
        if summarizer is not None:
            raise NotImplementedError(
                "summarizer hooks cannot cross the daemon RPC; "
                "run a local in-process engine"
            )
        response = self._client.post_json(
            f"/sandboxes/{source_sandbox_id}/observations/consolidate",
            {
                "fork_sandbox_id": str(fork_sandbox_id),
                "policy": policy,
                "reason": reason,
            },
            timeout_seconds=60.0,
        )
        return ObservationReport.from_json(response["report"])

    def egress_ledger(
        self,
        sandbox_id: SandboxId,
        *,
        txn_id: str | None = None,
        since_seq: int | None = None,
    ) -> EgressLedger:
        payload: dict[str, Any] = {}
        if txn_id:
            payload["txn_id"] = str(txn_id)
        if since_seq is not None:
            payload["since_seq"] = int(since_seq)
        response = self._client.post_json(
            f"/sandboxes/{sandbox_id}/egress", payload, timeout_seconds=60.0
        )
        return EgressLedger.from_json(response["ledger"])

    def begin_egress_replay(
        self,
        sandbox_id: SandboxId,
        *,
        policy: str = "cassette_first",
        cassette_source: object | None = None,
    ) -> None:
        payload: dict[str, Any] = {"mode": "begin", "policy": policy}
        if cassette_source is not None:
            payload["cassette_source"] = str(cassette_source)
        self._client.post_json(
            f"/sandboxes/{sandbox_id}/egress/replay", payload, timeout_seconds=60.0
        )

    def end_egress_replay(self, sandbox_id: SandboxId) -> EgressReplayReport | None:
        response = self._client.post_json(
            f"/sandboxes/{sandbox_id}/egress/replay",
            {"mode": "end"},
            timeout_seconds=60.0,
        )
        raw = response.get("report")
        return None if not isinstance(raw, dict) else EgressReplayReport.from_json(raw)

    def merge_processes(
        self,
        source_sandbox_id: SandboxId,
        fork_sandbox_id: SandboxId,
        *,
        strategy: str = "auto",
        policy: str = "fail_fast",
        observations: str = "append",
        stop_on_deviation: bool = False,
        lazy_pages: bool = True,
        force: bool = False,
        egress_replay: str = "cassette_first",
        replay_effects: str = "reject",
    ) -> ProcessMergeReport:
        payload: dict[str, Any] = {
            "fork_sandbox_id": str(fork_sandbox_id),
            "strategy": strategy,
            "policy": policy,
            "observations": observations,
            "stop_on_deviation": bool(stop_on_deviation),
            "lazy_pages": bool(lazy_pages),
            "force": bool(force),
            "egress_replay": egress_replay,
            "replay_effects": replay_effects,
        }
        try:
            response = self._client.post_json(
                f"/sandboxes/{source_sandbox_id}/processes/merge",
                payload,
                timeout_seconds=600.0,  # replay runs the fork's whole exec history
            )
        except DaemonRequestError as exc:
            raise _map_process_merge_error(exc) from exc
        return ProcessMergeReport.from_json(response["report"])


class _ConfigShim:
    """`Engine.config`-shaped object exposing only what the SDK reads."""

    def __init__(self, info: Mapping[str, Any]) -> None:
        self.default_image = info.get("default_image") or "ubuntu:22.04"
        self.runtime = str(info.get("runtime") or "runc")
        self.enable_sandbox_network = bool(
            info.get("sandbox_network_default")
            or info.get("network_bridge_ip") is not None
        )
        self.enable_interceptor = bool(info.get("interceptor_base_url"))
        self.enable_egress_proxy = False


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
        try:
            result = self._client.post_json(
                "/sandboxes", payload, timeout_seconds=1200.0
            )
        except DaemonRequestError as exc:
            mapped = _map_create_error(exc)
            if mapped is exc:
                raise
            raise mapped from exc
        return SandboxId(str(result["sandbox_id"]))

    def stop(self, sandbox_id: SandboxId) -> None:  # type: ignore[override]
        # Graceful stop that keeps the bundle/filesystem so the sandbox can be
        # started again. Distinct from delete(), which `Sandbox.kill()` uses to
        # destroy the sandbox via DELETE /sandboxes/{id}.
        self._client.post_json(f"/sandboxes/{sandbox_id}/stop", {}, timeout_seconds=600.0)

    def pause(self, sandbox_id: SandboxId) -> None:  # type: ignore[override]
        self._client.post_json(f"/sandboxes/{sandbox_id}/pause", {}, timeout_seconds=600.0)

    def resume(self, sandbox_id: SandboxId) -> None:  # type: ignore[override]
        self._client.post_json(f"/sandboxes/{sandbox_id}/resume", {}, timeout_seconds=60.0)

    def start(self, sandbox_id: SandboxId) -> None:  # type: ignore[override]
        self._client.post_json(f"/sandboxes/{sandbox_id}/start", {}, timeout_seconds=60.0)

    def restart(self, sandbox_id: SandboxId) -> None:  # type: ignore[override]
        self._client.post_json(f"/sandboxes/{sandbox_id}/restart", {}, timeout_seconds=60.0)

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
        try:
            response = self._client.post_json(
                f"/sandboxes/{sandbox_id}/exec",
                payload,
                timeout_seconds=http_timeout,
            )
        except DaemonRequestError as exc:
            mapped = _map_exec_error(exc, cmd=argv, timeout_s=timeout_s)
            if mapped is exc:
                raise
            raise mapped from exc
        raw = response["result"]
        return SandboxExecResult(
            args=tuple(raw.get("args") or []),
            returncode=int(raw["returncode"]),
            stdout=str(raw.get("stdout") or ""),
            stderr=str(raw.get("stderr") or ""),
        )

    def stream_exec(
        self,
        sandbox_id: SandboxId,
        argv: list[str],
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        user: str | None = None,
        timeout_s: float | None = None,
        capture_output: bool = True,
    ) -> Iterator[ExecEvent | ExecDone]:
        """Streaming exec: yields ExecEvent/ExecDone as output arrives."""
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
        http_timeout = (float(timeout_s) + 60.0) if timeout_s else None
        stream = self._client.stream_post(
            f"/sandboxes/{sandbox_id}/exec?stream=1",
            payload,
            timeout_seconds=http_timeout,
        )
        try:
            for event in stream:
                if event.get("done"):
                    error_type = event.get("error_type")
                    if error_type == SandboxExecTimeout.error_type:
                        raise SandboxExecTimeout(
                            argv,
                            float(event.get("timeout_s") or timeout_s or 0.0),
                            stdout=str(event.get("stdout") or ""),
                            stderr=str(event.get("stderr") or ""),
                        )
                    if error_type == SandboxExecCleanupError.error_type:
                        raise SandboxExecCleanupError(
                            str(event.get("error") or "sandbox exec cleanup failed"),
                            cmd=argv,
                            timeout=event.get("timeout_s", timeout_s),
                            stdout=str(event.get("stdout") or ""),
                            stderr=str(event.get("stderr") or ""),
                            payload_pid=event.get("payload_pid"),
                            cgroup_path=event.get("cgroup_path"),
                        )
                    yield ExecDone(returncode=int(event.get("rc", -1)))
                    return
                ch = event.get("ch", "stdout")
                text = event.get("t", "")
                yield ExecEvent(channel=ch, text=text)
        finally:
            stream.close()

    # ----- daemon-only API surface the SDK needs in addition to Runtime -----

    def batch_action(
        self,
        sandbox_id: SandboxId,
        *,
        argv: list[str],
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        user: str | None = None,
        timeout_s: float | None = None,
        capture_output: bool = True,
        checkpoint: bool = False,
        checkpoint_id: str | None = None,
        changeset: bool = False,
        changeset_since: str | None = None,
        observe: bool = False,
    ) -> dict[str, Any]:
        """Single-round-trip batch action: exec + observe + checkpoint + changeset.

        Returns the raw response dict from the daemon's
        ``POST /sandboxes/{id}/action`` endpoint.  The SDK interprets
        this into an ``ActionResult`` without further network calls."""
        exec_spec: dict[str, Any] = {
            "argv": list(argv),
            "capture_output": bool(capture_output),
        }
        if cwd is not None:
            exec_spec["cwd"] = cwd
        if env is not None:
            exec_spec["env"] = dict(env)
        if user is not None:
            exec_spec["user"] = user
        if timeout_s is not None:
            exec_spec["timeout_s"] = float(timeout_s)

        payload: dict[str, Any] = {"exec": exec_spec}
        if observe:
            payload["observe"] = True
        if checkpoint:
            payload["checkpoint"] = True
            if checkpoint_id is not None:
                payload["checkpoint_id"] = str(checkpoint_id)
        if changeset:
            payload["changeset"] = True
            if changeset_since is not None:
                payload["changeset_since"] = str(changeset_since)

        # Timeout: exec may block for the full task timeout; add headroom.
        http_timeout = (float(timeout_s) + 120.0) if timeout_s else 600.0
        try:
            return self._client.post_json(
                f"/sandboxes/{sandbox_id}/action",
                payload,
                timeout_seconds=http_timeout,
            )
        except DaemonRequestError as exc:
            mapped = _map_exec_error(exc, cmd=argv, timeout_s=timeout_s)
            if mapped is exc:
                raise
            raise mapped from exc

    def poll_job(self, sandbox_id: SandboxId, job_id: str) -> dict[str, Any]:
        """Poll a daemon background job (checkpoint/changeset) status.

        Returns a dict with at least ``status`` ('pending'|'completed'|'failed')
        and on completion a ``result`` sub-dict carrying checkpoint_id / changeset."""
        return self._client.get_json(f"/sandboxes/{sandbox_id}/jobs/{job_id}")

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

    def list_sandboxes(self) -> list[dict[str, Any]]:
        """Return the caller-visible sandboxes as raw dicts.

        Hits ``GET /sandboxes`` on the daemon (or the gateway's tenant-
        scoped equivalent). Each entry carries at least ``sandbox_id``,
        ``runtime_name``, ``status`` and ``metadata``."""
        response = self._client.get_json("/sandboxes")
        return list(response.get("sandboxes") or [])

    def set_idle_timeout(self, sandbox_id: SandboxId, idle_timeout: float | None) -> None:
        """Reset/extend the sandbox's idle-reclaim window (E2B-style
        ``set_timeout``). ``None`` disables idle reclaim."""
        self._client.post_json(
            f"/sandboxes/{sandbox_id}/idle",
            {"idle_timeout": idle_timeout},
        )

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

    def fork_sandbox(
        self,
        source_sandbox_id: SandboxId,
        *,
        count: int = 1,
        lazy: bool = False,
        effects: str | None = None,
        gate_effects: bool = True,
        checkpoint_id: CheckpointId | None = None,
    ) -> list[SandboxId]:
        """Fork via the daemon. Mirrors `Engine.fork_sandbox`; the daemon
        runs the whole checkpoint + clone + restore pipeline and returns
        the new sandbox ids. Timeout scales with count the same way
        checkpoint/restore calls are budgeted (300s each).

        ``checkpoint_id`` rides the request as the explicit fork point, so
        the daemon clones that checkpoint instead of taking a fresh one.

        ``effects`` rides the request so a gated fork (F1) is validated and
        armed daemon-side, before the fork's processes are restored.
        ``gate_effects=False`` has no remote counterpart: the daemon owns
        the fork-txn path itself, so it is only accepted here to keep the
        in-process signature substitutable."""
        if count < 1:
            raise ValueError("fork count must be >= 1")
        if not gate_effects and effects is not None:
            raise ValueError(
                "effects= is not accepted when the caller owns the effect window"
            )
        payload: dict = {"count": int(count), "lazy": bool(lazy)}
        if effects is not None:
            payload["effects"] = str(effects)
        if checkpoint_id is not None:
            payload["checkpoint_id"] = str(checkpoint_id)
        response = self._client.post_json(
            f"/sandboxes/{source_sandbox_id}/fork",
            payload,
            timeout_seconds=300.0 * count,
        )
        return [
            SandboxId(str(entry["sandbox_id"]))
            for entry in (response.get("forks") or [])
        ]

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


def _deserialize_txn(payload: Mapping[str, Any]) -> TxnDescription:
    base = payload.get("base_checkpoint_id")
    label = payload.get("label")
    fork_sandbox_id = payload.get("fork_sandbox_id")
    return TxnDescription(
        txn_id=str(payload["txn_id"]),
        sandbox_id=str(payload["sandbox_id"]),
        base_checkpoint_id=None if base is None else str(base),
        base_was_fresh=bool(payload.get("base_was_fresh")),
        started_at=str(payload.get("started_at") or ""),
        label=None if label is None else str(label),
        isolation=str(payload.get("isolation") or "snapshot"),
        fork_sandbox_id=None if fork_sandbox_id is None else str(fork_sandbox_id),
        effects=str(payload.get("effects") or "allow"),
    )


def _map_txn_error(exc: DaemonRequestError) -> Exception:
    """Rehydrate the daemon's 409 error_type into the matching Txn*
    exception; anything unrecognized re-raises the transport error."""
    try:
        payload = json.loads(exc.body.decode("utf-8", errors="replace"))
    except Exception:
        payload = {}
    error_type = payload.get("error_type") if isinstance(payload, dict) else None
    message = str(payload.get("error")) if isinstance(payload, dict) and payload.get("error") else str(exc)
    if error_type == "txn_active":
        return TxnActiveError(message)
    if error_type == "txn_not_abortable":
        return TxnNotAbortable(message)
    if error_type == "txn_mismatch":
        return TxnMismatchError(message)
    if error_type == "txn_commit_conflict":
        return TxnCommitConflict(message)
    if error_type == "txn_abort_failed":
        return TxnAbortError(message)
    return exc


def _map_exec_error(
    exc: DaemonRequestError,
    *,
    cmd: list[str],
    timeout_s: float | None,
) -> Exception:
    """Rehydrate daemon/gateway exec failures into stable SDK types."""

    try:
        payload = json.loads(exc.body.decode("utf-8", errors="replace"))
    except Exception:
        payload = {}
    if not isinstance(payload, dict):
        return exc
    error_type = payload.get("error_type")
    if error_type == SandboxExecTimeout.error_type:
        return SandboxExecTimeout(
            cmd,
            float(payload.get("timeout_s") or timeout_s or 0.0),
            stdout=str(payload.get("stdout") or ""),
            stderr=str(payload.get("stderr") or ""),
        )
    if error_type == SandboxExecCleanupError.error_type:
        return SandboxExecCleanupError(
            str(payload.get("error") or "sandbox exec cleanup failed"),
            cmd=cmd,
            timeout=payload.get("timeout_s", timeout_s),
            stdout=str(payload.get("stdout") or ""),
            stderr=str(payload.get("stderr") or ""),
            payload_pid=payload.get("payload_pid"),
            cgroup_path=payload.get("cgroup_path"),
        )
    return exc


def _map_create_error(exc: DaemonRequestError) -> Exception:
    """Rehydrate typed image/rollback failures across daemon and gateway."""

    try:
        payload = json.loads(exc.body.decode("utf-8", errors="replace"))
    except Exception:
        payload = {}
    if not isinstance(payload, dict):
        return exc
    error_type = payload.get("error_type")
    message = str(payload.get("error") or exc)
    reference = str(payload.get("image") or "")
    image_types = {
        ImageNotFoundError.error_type: ImageNotFoundError,
        ImageAuthenticationError.error_type: ImageAuthenticationError,
        ImageRateLimitError.error_type: ImageRateLimitError,
        ImagePullTimeoutError.error_type: ImagePullTimeoutError,
        ImagePlatformError.error_type: ImagePlatformError,
        ImageCompatibilityError.error_type: ImageCompatibilityError,
        ImageReferenceError.error_type: ImageReferenceError,
        ImagePolicyError.error_type: ImagePolicyError,
        ImageInsufficientDiskError.error_type: ImageInsufficientDiskError,
        ImageTooLargeError.error_type: ImageTooLargeError,
        ImagePullError.error_type: ImagePullError,
    }
    image_error_type = image_types.get(error_type)
    if image_error_type is not None:
        return image_error_type(reference, message)
    if error_type == SandboxCreateCleanupError.error_type:
        sandbox_id = str(payload.get("sandbox_id") or "unknown")
        cleanup_errors = payload.get("cleanup_errors")
        resources = payload.get("leaked_resources")
        return SandboxCreateCleanupError(
            sandbox_id,
            RuntimeError(message),
            (
                [str(item) for item in cleanup_errors]
                if isinstance(cleanup_errors, list)
                else [message]
            ),
            resources=(
                [str(item) for item in resources]
                if isinstance(resources, list)
                else ()
            ),
        )
    return exc


def _map_merge_error(exc: DaemonRequestError) -> Exception:
    """Rehydrate the daemon's 409 merge_error into MergeError, restoring
    the serialized report when the failure produced one; anything
    unrecognized re-raises the transport error."""
    try:
        payload = json.loads(exc.body.decode("utf-8", errors="replace"))
    except Exception:
        payload = {}
    if not isinstance(payload, dict) or payload.get("error_type") != "merge_error":
        return exc
    message = str(payload.get("error")) if payload.get("error") else str(exc)
    report = None
    raw_report = payload.get("report")
    if isinstance(raw_report, dict):
        try:
            report = MergeReport.from_json(raw_report)
        except Exception:
            logger.exception("Failed to rehydrate merge report from daemon 409")
    return MergeError(message, report=report)


def _map_process_merge_error(exc: DaemonRequestError) -> Exception:
    """Rehydrate the daemon's 409 process_merge_conflict into the typed
    exception; anything unrecognized re-raises the transport error."""
    try:
        payload = json.loads(exc.body.decode("utf-8", errors="replace"))
    except Exception:
        payload = {}
    if isinstance(payload, dict) and payload.get("error_type") == "process_merge_conflict":
        message = str(payload.get("error")) if payload.get("error") else str(exc)
        return ProcessMergeConflict(message)
    return exc


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
