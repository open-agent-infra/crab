from __future__ import annotations

from collections import deque
import threading
import time
import uuid
from dataclasses import dataclass
from dataclasses import replace
from http.server import BaseHTTPRequestHandler
from typing import Callable
import logging

from .contracts import RequestInterceptorHook, SandboxInspector, TelemetrySink
from .http_utils import PooledHTTPServer, ThreadLocalHttpClient
from .ids import SandboxId
from .json_codec import get_json_codec
from .models import RequestContext, RequestState, RequestStateChange, SandboxSnapshot, utc_now
from .telemetry import start_operation
from integrations.llm_services.claude_code_trace_replay.request_classification import (
    REQUEST_KIND_MAIN_LOOP,
    infer_live_request_kind,
)


logger = logging.getLogger(__name__)
_JSON_CODEC = get_json_codec("auto")


def _header_value(headers: dict[str, str], name: str, default: str = "") -> str:
    value = headers.get(name)
    if value is not None:
        return value
    lowered = name.lower()
    for key, candidate in headers.items():
        if key.lower() == lowered:
            return candidate
    return default


def _maybe_requested_model(path: str, body: bytes) -> str | None:
    if path != "/v1/messages":
        return None
    try:
        payload = _JSON_CODEC.loads(body)
    except Exception:
        logger.debug("Failed to decode interceptor request body for model classification", exc_info=True)
        return None
    if not isinstance(payload, dict):
        return None
    model = payload.get("model")
    if not isinstance(model, str):
        return None
    stripped = model.strip()
    return stripped or None


def _claude_request_gate_policy(path: str, body: bytes) -> tuple[str | None, bool]:
    if path not in {"/v1/messages", "/v1/messages/count_tokens"}:
        return None, True
    requested_model = _maybe_requested_model(path, body)
    request_kind = infer_live_request_kind(path=path, requested_model=requested_model)
    return request_kind, request_kind == REQUEST_KIND_MAIN_LOOP

class CompositeRequestInterceptorHook(RequestInterceptorHook):
    def __init__(self, hooks: list[RequestInterceptorHook] | None = None):
        self._hooks = list(hooks or [])

    def add_hook(self, hook: RequestInterceptorHook) -> None:
        self._hooks.append(hook)

    def on_request_start(self, context: RequestContext) -> None:
        for hook in self._hooks:
            hook.on_request_start(context)

    def on_request_end(self, context: RequestContext) -> None:
        for hook in self._hooks:
            hook.on_request_end(context)


class TelemetryRequestInterceptorHook(RequestInterceptorHook):
    def __init__(self, telemetry: TelemetrySink):
        self._telemetry = telemetry

    def on_request_start(self, context: RequestContext) -> None:
        self._telemetry.emit_event(
            "request.start",
            {
                "request_id": context.request_id,
                "sandbox_id": str(context.sandbox_id),
                "provider": context.metadata.get("provider", ""),
                "path": context.metadata.get("path", ""),
            },
        )

    def on_request_end(self, context: RequestContext) -> None:
        self._telemetry.emit_event(
            "request.finish",
            {
                "request_id": context.request_id,
                "sandbox_id": str(context.sandbox_id),
                "provider": context.metadata.get("provider", ""),
                "path": context.metadata.get("path", ""),
            },
        )


class InMemoryRequestStateStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._states: dict[SandboxId, RequestState] = {}
        self._changes: deque[RequestStateChange] = deque()
        self._active_contexts: dict[SandboxId, dict[str, RequestContext]] = {}

    def mark_request_start(self, context: RequestContext) -> RequestState:
        with self._lock:
            current = self._states.get(context.sandbox_id, RequestState(sandbox_id=context.sandbox_id))
            sandbox_contexts = dict(self._active_contexts.get(context.sandbox_id, {}))
            sandbox_contexts[context.request_id] = replace(context, metadata=dict(context.metadata))
            self._active_contexts[context.sandbox_id] = sandbox_contexts
            provider = None if context.metadata.get("provider") is None else str(context.metadata["provider"])
            updated = replace(
                current,
                active_llm_requests=current.active_llm_requests + 1,
                total_llm_requests=current.total_llm_requests + 1,
                last_request_id=context.request_id,
                last_llm_provider=provider,
                last_llm_request_started_at=context.started_at,
            )
            self._states[context.sandbox_id] = updated
            self._changes.append(
                RequestStateChange(
                    sandbox_id=context.sandbox_id,
                    event_type="request_start",
                    request_id=context.request_id,
                    observed_at=context.started_at,
                )
            )
            self._condition.notify_all()
            logger.debug(
                "Marked request start: sandbox_id=%s request_id=%s provider=%s active=%s total=%s",
                context.sandbox_id,
                context.request_id,
                provider,
                updated.active_llm_requests,
                updated.total_llm_requests,
            )
            return updated

    def mark_request_end(self, context: RequestContext) -> RequestState:
        with self._lock:
            current = self._states.get(context.sandbox_id, RequestState(sandbox_id=context.sandbox_id))
            sandbox_contexts = dict(self._active_contexts.get(context.sandbox_id, {}))
            active_context = sandbox_contexts.pop(context.request_id, None)
            if sandbox_contexts:
                self._active_contexts[context.sandbox_id] = sandbox_contexts
            else:
                self._active_contexts.pop(context.sandbox_id, None)
            ended_at = utc_now()
            metadata = context.metadata if active_context is None else active_context.metadata
            provider = None if metadata.get("provider") is None else str(metadata["provider"])
            updated = replace(
                current,
                active_llm_requests=max(0, current.active_llm_requests - 1),
                completed_llm_requests=current.completed_llm_requests + 1,
                last_request_id=context.request_id,
                last_llm_provider=provider,
                last_llm_request_ended_at=ended_at,
            )
            self._states[context.sandbox_id] = updated
            self._changes.append(
                RequestStateChange(
                    sandbox_id=context.sandbox_id,
                    event_type="request_end",
                    request_id=context.request_id,
                    observed_at=ended_at,
                )
            )
            self._condition.notify_all()
            logger.debug(
                "Marked request end: sandbox_id=%s request_id=%s provider=%s active=%s completed=%s",
                context.sandbox_id,
                context.request_id,
                provider,
                updated.active_llm_requests,
                updated.completed_llm_requests,
            )
            return updated

    def get(self, sandbox_id: SandboxId) -> RequestState:
        with self._lock:
            return self._states.get(sandbox_id, RequestState(sandbox_id=sandbox_id))

    def get_request_context(self, sandbox_id: SandboxId, request_id: str) -> RequestContext | None:
        with self._lock:
            context = self._active_contexts.get(sandbox_id, {}).get(request_id)
            if context is None:
                return None
            return replace(context, metadata=dict(context.metadata))

    def get_request_contexts_for_group(
        self,
        sandbox_id: SandboxId,
        *,
        request_group_kind: str,
        request_group_id: str,
    ) -> list[RequestContext]:
        with self._lock:
            contexts = list(self._active_contexts.get(sandbox_id, {}).values())
        matches: list[RequestContext] = []
        for context in contexts:
            metadata = dict(context.metadata)
            if metadata.get("request_group_kind") != request_group_kind:
                continue
            if metadata.get("request_group_id") != request_group_id:
                continue
            matches.append(replace(context, metadata=metadata))
        return matches

    def wait_for_change(self, timeout: float | None = None) -> RequestStateChange | None:
        with self._condition:
            if not self._changes:
                self._condition.wait(timeout=timeout)
            if not self._changes:
                return None
            return self._changes.popleft()

    def notify_waiters(self) -> None:
        with self._condition:
            self._condition.notify_all()


@dataclass(frozen=True)
class PendingSandboxResponse:
    sandbox_id: SandboxId
    request_id: str
    generation: int
    request_ids: tuple[str, ...] = ()
    request_group_kind: str | None = None
    request_group_id: str | None = None


@dataclass
class _PendingResponseEntry:
    generation: int
    primary_request_id: str
    request_ids: list[str]
    request_group_kind: str | None = None
    request_group_id: str | None = None

    def as_pending(self, sandbox_id: SandboxId) -> PendingSandboxResponse:
        return PendingSandboxResponse(
            sandbox_id=sandbox_id,
            request_id=self.primary_request_id,
            generation=self.generation,
            request_ids=tuple(self.request_ids),
            request_group_kind=self.request_group_kind,
            request_group_id=self.request_group_id,
        )


class SandboxResponseGateRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._enabled = False
        self._states: dict[
            SandboxId,
            dict[
                str,
                int
                | dict[int, _PendingResponseEntry]
                | dict[tuple[str, str], int]
                | set[tuple[str, str]]
                | threading.Condition,
            ],
        ] = {}

    def enable(self) -> None:
        with self._lock:
            self._enabled = True
            logger.debug("Enabled sandbox response gate registry")

    def disable(self) -> None:
        with self._lock:
            self._enabled = False
            conditions = [state["condition"] for state in self._states.values()]
            for state in self._states.values():
                pending = state["pending"]
                assert isinstance(pending, dict)
                pending.clear()
            logger.debug("Disabled sandbox response gate registry and released %s sandbox states", len(conditions))
        for condition in conditions:
            assert isinstance(condition, threading.Condition)
            with condition:
                condition.notify_all()

    def arm(
        self,
        sandbox_id: SandboxId,
        request_id: str,
        *,
        request_group_kind: str | None = None,
        request_group_id: str | None = None,
    ) -> int | None:
        with self._lock:
            if not self._enabled:
                logger.debug(
                    "Skipped arming response gate because registry is disabled: sandbox_id=%s request_id=%s",
                    sandbox_id,
                    request_id,
                )
                return None
            state = self._states.get(sandbox_id)
            if state is None:
                condition = threading.Condition()
                state = {
                    "generation": 0,
                    "pending": {},
                    "pending_groups": {},
                    "satisfied_groups": set(),
                    "condition": condition,
                }
                self._states[sandbox_id] = state
            group_key = None
            if request_group_kind and request_group_id:
                group_key = (request_group_kind, request_group_id)
                satisfied_groups = state["satisfied_groups"]
                assert isinstance(satisfied_groups, set)
                if group_key in satisfied_groups:
                    logger.debug(
                        "Skipped arming response gate for already-satisfied request group: sandbox_id=%s request_id=%s group=%s",
                        sandbox_id,
                        request_id,
                        group_key,
                    )
                    return None
                pending_groups = state["pending_groups"]
                assert isinstance(pending_groups, dict)
                existing_generation = pending_groups.get(group_key)
                if existing_generation is not None:
                    pending = state["pending"]
                    assert isinstance(pending, dict)
                    entry = pending.get(existing_generation)
                    if entry is not None:
                        if request_id not in entry.request_ids:
                            entry.request_ids.append(request_id)
                        logger.debug(
                            "Joined existing response gate group: sandbox_id=%s request_id=%s generation=%s group=%s",
                            sandbox_id,
                            request_id,
                            existing_generation,
                            group_key,
                        )
                        return existing_generation
            state["generation"] = int(state["generation"]) + 1
            pending = state["pending"]
            assert isinstance(pending, dict)
            generation = int(state["generation"])
            entry = _PendingResponseEntry(
                generation=generation,
                primary_request_id=request_id,
                request_ids=[request_id],
                request_group_kind=request_group_kind,
                request_group_id=request_group_id,
            )
            pending[generation] = entry
            if group_key is not None:
                pending_groups = state["pending_groups"]
                assert isinstance(pending_groups, dict)
                pending_groups[group_key] = generation
            logger.debug(
                "Armed response gate: sandbox_id=%s request_id=%s generation=%s",
                sandbox_id,
                request_id,
                generation,
            )
            return generation

    def wait_for_release(self, sandbox_id: SandboxId, generation: int | None, timeout: float | None = None) -> None:
        if generation is None:
            return
        with self._lock:
            state = self._states.get(sandbox_id)
            if state is None:
                return
            condition = state["condition"]
            assert isinstance(condition, threading.Condition)
        logger.debug(
            "Waiting for response gate release: sandbox_id=%s generation=%s timeout=%s",
            sandbox_id,
            generation,
            timeout,
        )
        with condition:
            released = condition.wait_for(
                lambda: self._is_generation_released(sandbox_id, generation),
                timeout=timeout,
            )
        logger.debug(
            "Finished waiting for response gate: sandbox_id=%s generation=%s released=%s",
            sandbox_id,
            generation,
            released,
        )

    def get_pending(self, sandbox_id: SandboxId) -> PendingSandboxResponse | None:
        with self._lock:
            state = self._states.get(sandbox_id)
            if state is None:
                return None
            pending = state["pending"]
            assert isinstance(pending, dict)
            if not pending:
                return None
            generation = max(pending)
            return pending[generation].as_pending(sandbox_id)

    def get_oldest_pending(self, sandbox_id: SandboxId) -> PendingSandboxResponse | None:
        with self._lock:
            state = self._states.get(sandbox_id)
            if state is None:
                return None
            pending = state["pending"]
            assert isinstance(pending, dict)
            if not pending:
                return None
            generation = min(pending)
            return pending[generation].as_pending(sandbox_id)

    def get_pending_generation(self, sandbox_id: SandboxId, generation: int) -> PendingSandboxResponse | None:
        with self._lock:
            state = self._states.get(sandbox_id)
            if state is None:
                return None
            pending = state["pending"]
            assert isinstance(pending, dict)
            entry = pending.get(generation)
            if entry is None:
                return None
            return entry.as_pending(sandbox_id)

    def find_pending_request(self, sandbox_id: SandboxId, request_id: str) -> PendingSandboxResponse | None:
        with self._lock:
            state = self._states.get(sandbox_id)
            if state is None:
                return None
            pending = state["pending"]
            assert isinstance(pending, dict)
            for generation in sorted(pending):
                entry = pending[generation]
                if request_id in entry.request_ids:
                    return entry.as_pending(sandbox_id)
            return None

    def find_pending_group(
        self,
        sandbox_id: SandboxId,
        *,
        request_group_kind: str,
        request_group_id: str,
    ) -> PendingSandboxResponse | None:
        with self._lock:
            state = self._states.get(sandbox_id)
            if state is None:
                return None
            pending_groups = state["pending_groups"]
            assert isinstance(pending_groups, dict)
            generation = pending_groups.get((request_group_kind, request_group_id))
            if generation is None:
                return None
            pending = state["pending"]
            assert isinstance(pending, dict)
            entry = pending.get(generation)
            if entry is None:
                return None
            return entry.as_pending(sandbox_id)

    def release_pending(self, sandbox_id: SandboxId, *, request_id: str, generation: int | None = None) -> bool:
        with self._lock:
            state = self._states.get(sandbox_id)
            if state is None:
                logger.debug(
                    "No pending response gate to release: sandbox_id=%s request_id=%s generation=%s",
                    sandbox_id,
                    request_id,
                    generation,
                )
                return False
            pending = state["pending"]
            assert isinstance(pending, dict)
            target_generation = generation
            if target_generation is None:
                for candidate_generation in sorted(pending):
                    if request_id in pending[candidate_generation].request_ids:
                        target_generation = candidate_generation
                        break
            if target_generation is None:
                logger.debug(
                    "Skipped releasing response gate due to missing generation: sandbox_id=%s request_id=%s",
                    sandbox_id,
                    request_id,
                )
                return False
            entry = pending.get(target_generation)
            if entry is None or request_id not in entry.request_ids:
                logger.debug(
                    "Skipped releasing response gate due to request_id mismatch: sandbox_id=%s actual=%s generation=%s",
                    sandbox_id,
                    request_id,
                    target_generation,
                )
                return False
            pending.pop(target_generation, None)
            group_key = None
            if entry.request_group_kind and entry.request_group_id:
                group_key = (entry.request_group_kind, entry.request_group_id)
                pending_groups = state["pending_groups"]
                assert isinstance(pending_groups, dict)
                pending_groups.pop(group_key, None)
                satisfied_groups = state["satisfied_groups"]
                assert isinstance(satisfied_groups, set)
                satisfied_groups.add(group_key)
            condition = state["condition"]
            assert isinstance(condition, threading.Condition)
        with condition:
            condition.notify_all()
        logger.debug(
            "Released pending response gate: sandbox_id=%s request_id=%s generation=%s",
            sandbox_id,
            request_id,
            target_generation,
        )
        return True

    def release(self, sandbox_id: SandboxId, *, generation: int | None = None) -> None:
        with self._lock:
            state = self._states.get(sandbox_id)
            if state is None:
                logger.debug("No response gate state found to release: sandbox_id=%s", sandbox_id)
                return
            pending = state["pending"]
            assert isinstance(pending, dict)
            if generation is None:
                pending_groups = state["pending_groups"]
                assert isinstance(pending_groups, dict)
                satisfied_groups = state["satisfied_groups"]
                assert isinstance(satisfied_groups, set)
                for entry in pending.values():
                    if entry.request_group_kind and entry.request_group_id:
                        satisfied_groups.add((entry.request_group_kind, entry.request_group_id))
                pending.clear()
                pending_groups.clear()
            else:
                entry = pending.pop(generation, None)
                if entry is not None and entry.request_group_kind and entry.request_group_id:
                    pending_groups = state["pending_groups"]
                    assert isinstance(pending_groups, dict)
                    pending_groups.pop((entry.request_group_kind, entry.request_group_id), None)
                    satisfied_groups = state["satisfied_groups"]
                    assert isinstance(satisfied_groups, set)
                    satisfied_groups.add((entry.request_group_kind, entry.request_group_id))
            condition = state["condition"]
            assert isinstance(condition, threading.Condition)
        with condition:
            condition.notify_all()
        logger.debug("Released response gate: sandbox_id=%s generation=%s", sandbox_id, generation)

    def _is_generation_released(self, sandbox_id: SandboxId, generation: int) -> bool:
        with self._lock:
            state = self._states.get(sandbox_id)
            if state is None:
                return True
            if not self._enabled:
                return True
            pending = state["pending"]
            assert isinstance(pending, dict)
            return generation not in pending


class RequestAwareSandboxInspector(SandboxInspector):
    def __init__(self, base: SandboxInspector, request_state_store: InMemoryRequestStateStore) -> None:
        self._base = base
        self._request_state_store = request_state_store

    def __getattr__(self, name: str):
        return getattr(self._base, name)

    def inspect(self, sandbox_id: SandboxId) -> SandboxSnapshot:
        base_snapshot = self._base.inspect(sandbox_id)
        request_state = self._request_state_store.get(sandbox_id)
        observed_at = base_snapshot.observed_at
        if request_state.last_llm_request_started_at is not None:
            observed_at = max(observed_at, request_state.last_llm_request_started_at)
        if request_state.last_llm_request_ended_at is not None:
            observed_at = max(observed_at, request_state.last_llm_request_ended_at)
        return replace(
            base_snapshot,
            observed_at=observed_at,
            metadata={**base_snapshot.metadata, **request_state.to_metadata()},
        )

    def mark_checkpoint_complete(
        self,
        sandbox_id: SandboxId,
        *,
        process: bool,
        filesystem: bool,
        at,
    ) -> None:
        self._base.mark_checkpoint_complete(
            sandbox_id,
            process=process,
            filesystem=filesystem,
            at=at,
        )


class CrabRequestInterceptor:
    def __init__(
        self,
        *,
        upstream_transport: Callable[[str, dict[str, str], bytes], tuple[int, list[tuple[str, str]], bytes]],
        request_state_store: InMemoryRequestStateStore,
        hook: RequestInterceptorHook | None = None,
        telemetry: TelemetrySink | None = None,
        on_state_change: Callable[[SandboxId], None] | None = None,
        on_response_ready: Callable[[SandboxId, str, int | None], None] | None = None,
        response_gate_registry: SandboxResponseGateRegistry | None = None,
        sandbox_id_resolver: Callable[[str | None, dict[str, str], bytes], str | None] | None = None,
    ) -> None:
        self._upstream_transport = upstream_transport
        self._request_state_store = request_state_store
        self._hook = hook or CompositeRequestInterceptorHook()
        self._telemetry = telemetry
        self._on_state_change = on_state_change
        self._on_response_ready = on_response_ready
        self._response_gate_registry = response_gate_registry
        self._sandbox_id_resolver = sandbox_id_resolver

    def intercept(
        self,
        *,
        path: str,
        headers: dict[str, str],
        body: bytes,
        client_host: str | None = None,
    ) -> tuple[int, list[tuple[str, str]], bytes]:
        sandbox_id_raw = headers.get("X-Agent-Sandbox-Id", "").strip()
        if self._sandbox_id_resolver is not None:
            resolved = self._sandbox_id_resolver(client_host, headers, body)
            if resolved is not None and resolved.strip():
                sandbox_id_raw = resolved.strip()
        if not sandbox_id_raw:
            logger.debug("Rejecting request without sandbox identity: path=%s client_host=%s", path, client_host)
            raise ValueError("missing sandbox identity")
        upstream_headers = dict(headers)
        upstream_headers["X-Agent-Sandbox-Id"] = sandbox_id_raw
        provider = "openai" if path == "/v1/chat/completions" else "anthropic"
        request_kind, response_gate_enabled = _claude_request_gate_policy(path, body)
        request_group_kind = None
        request_group_id = None
        request_group_role = None
        spec_pair_id = _header_value(headers, "X-Crab-Spec-Pair-Id").strip()
        if spec_pair_id:
            request_group_kind = "spec_pair"
            request_group_id = spec_pair_id
        metadata = {
            "provider": provider,
            "path": path,
            "response_gate_enabled": response_gate_enabled,
        }
        if request_kind is not None:
            metadata["request_kind"] = request_kind
        if request_group_kind is not None and request_group_id is not None:
            metadata["request_group_kind"] = request_group_kind
            metadata["request_group_id"] = request_group_id
            spec_role = _header_value(headers, "X-Crab-Spec-Role").strip().lower()
            if spec_role:
                request_group_role = spec_role
                metadata["request_group_role"] = spec_role
        context = RequestContext(
            request_id=upstream_headers.get("X-Request-Id", "").strip() or str(uuid.uuid4()),
            sandbox_id=SandboxId(sandbox_id_raw),
            started_at=utc_now(),
            metadata=metadata,
        )
        upstream_headers["X-Request-Id"] = context.request_id
        upstream_headers["X-Crab-Request-Id"] = context.request_id
        request_attributes = {
            "component": "interceptor",
            "request_id": context.request_id,
            "sandbox_id": str(context.sandbox_id),
            "provider": provider,
            "path": path,
            "response_gate_enabled": response_gate_enabled,
        }
        if request_kind is not None:
            request_attributes["request_kind"] = request_kind
        if request_group_kind is not None and request_group_id is not None:
            request_attributes["request_group_kind"] = request_group_kind
            request_attributes["request_group_id"] = request_group_id
            if request_group_role is not None:
                request_attributes["request_group_role"] = request_group_role
        if self._telemetry is not None:
            self._telemetry.emit_event("interceptor.request.received", request_attributes)
        logger.debug(
            "Intercepting request start: request_id=%s sandbox_id=%s provider=%s path=%s client_host=%s body_bytes=%s",
            context.request_id,
            context.sandbox_id,
            provider,
            path,
            client_host,
            len(body),
        )
        self._hook.on_request_start(context)
        self._request_state_store.mark_request_start(context)
        gate_generation = None
        request_started = time.perf_counter()
        crab_delay_started_at: float | None = None
        if self._response_gate_registry is not None and response_gate_enabled:
            gate_generation = self._response_gate_registry.arm(
                context.sandbox_id,
                context.request_id,
                request_group_kind=request_group_kind,
                request_group_id=request_group_id,
            )
        elif self._response_gate_registry is not None and request_kind is not None:
            logger.debug(
                "Skipped response gate for auxiliary Claude request: sandbox_id=%s request_id=%s kind=%s path=%s",
                context.sandbox_id,
                context.request_id,
                request_kind,
                path,
            )
        self._notify(context.sandbox_id)
        try:
            forward_operation = None if self._telemetry is None else start_operation(
                self._telemetry,
                "interceptor.request.forward",
                request_attributes,
            )
            try:
                response = self._upstream_transport(path, upstream_headers, body)
            finally:
                if forward_operation is not None:
                    forward_operation.finish()
            if self._telemetry is not None:
                self._telemetry.emit_event(
                    "interceptor.request.upstream_response_received",
                    request_attributes,
                )
            crab_delay_started_at = time.perf_counter()
            if self._on_response_ready is not None and gate_generation is not None:
                try:
                    self._on_response_ready(context.sandbox_id, context.request_id, gate_generation)
                except Exception:
                    logger.exception(
                        "Response-ready callback failed: sandbox_id=%s request_id=%s generation=%s",
                        context.sandbox_id,
                        context.request_id,
                        gate_generation,
                    )
            logger.debug(
                "Intercepting request complete: request_id=%s sandbox_id=%s status_code=%s response_bytes=%s",
                context.request_id,
                context.sandbox_id,
                response[0],
                len(response[2]),
            )
            logger.debug(f"Intercepted request response of request_id={context.request_id}: {response}")
            return response
        finally:
            gate_wait_ms = 0.0
            released_at = time.perf_counter()
            if self._response_gate_registry is not None and gate_generation is not None:
                gate_operation = None if self._telemetry is None else start_operation(
                    self._telemetry,
                    "interceptor.response_gate.wait",
                    request_attributes,
                )
                wait_started = time.perf_counter()
                self._response_gate_registry.wait_for_release(context.sandbox_id, gate_generation)
                released_at = time.perf_counter()
                gate_wait_ms = (released_at - wait_started) * 1000.0
                crab_delay_ms = None
                if crab_delay_started_at is not None:
                    crab_delay_ms = max(0.0, (released_at - crab_delay_started_at) * 1000.0)
                total_ms = max(0.0, (released_at - request_started) * 1000.0)
                if gate_operation is not None:
                    gate_operation.finish()
            else:
                crab_delay_ms = None if crab_delay_started_at is None else 0.0
                total_ms = max(0.0, (released_at - request_started) * 1000.0)
            if self._telemetry is not None:
                if crab_delay_started_at is not None:
                    self._telemetry.emit_metric(
                        "llm.gate_wait_ms",
                        gate_wait_ms,
                        request_attributes,
                    )
                    self._telemetry.emit_metric(
                        "llm.crab_delay_ms",
                        0.0 if crab_delay_ms is None else crab_delay_ms,
                        request_attributes,
                    )
                self._telemetry.emit_event("interceptor.response.released", request_attributes)
            if self._telemetry is not None:
                self._telemetry.emit_metric(
                    "llm.request_total_ms",
                    total_ms,
                    request_attributes,
                )
                self._telemetry.emit_metric(
                    "llm.interceptor_total_ms",
                    total_ms,
                    request_attributes,
                )
            self._hook.on_request_end(context) # record telemetry, etc
            self._notify(context.sandbox_id)
            self._request_state_store.mark_request_end(context)

    def _notify(self, sandbox_id: SandboxId) -> None:
        if self._on_state_change is not None:
            try:
                self._on_state_change(sandbox_id)
            except Exception:
                logger.exception("State change callback failed: sandbox_id=%s", sandbox_id)
                return


class CrabRequestInterceptorServer:
    def __init__(
        self,
        *,
        upstream_url: str,
        request_state_store: InMemoryRequestStateStore,
        hook: RequestInterceptorHook | None = None,
        telemetry: TelemetrySink | None = None,
        on_state_change: Callable[[SandboxId], None] | None = None,
        on_response_ready: Callable[[SandboxId, str, int | None], None] | None = None,
        response_gate_registry: SandboxResponseGateRegistry | None = None,
        sandbox_id_resolver: Callable[[str | None, dict[str, str], bytes], str | None] | None = None,
        host: str = "127.0.0.1",
        port: int = 0,
        upstream_timeout_seconds: float = 3600.0,
        max_workers: int | None = None,
    ) -> None:
        self._upstream_url = upstream_url.rstrip("/")
        self._upstream_timeout_seconds = upstream_timeout_seconds
        self._upstream_client = ThreadLocalHttpClient(
            self._upstream_url,
            timeout_seconds=self._upstream_timeout_seconds,
        )
        self._interceptor = CrabRequestInterceptor(
            upstream_transport=self._forward,
            request_state_store=request_state_store,
            hook=hook,
            telemetry=telemetry,
            on_state_change=on_state_change,
            on_response_ready=on_response_ready,
            response_gate_registry=response_gate_registry,
            sandbox_id_resolver=sandbox_id_resolver,
        )
        self._telemetry = telemetry
        self._server = PooledHTTPServer((host, port), self._build_handler(), max_workers=max_workers)
        self._thread: threading.Thread | None = None

    @property
    def host(self) -> str:
        return str(self._server.server_address[0])

    @property
    def port(self) -> int:
        return int(self._server.server_address[1])

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> None:
        if self._thread is not None:
            logger.debug("Interceptor server already running: base_url=%s", self.base_url)
            return
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        logger.debug("Started interceptor server: base_url=%s upstream_url=%s", self.base_url, self._upstream_url)

    def stop(self) -> None:
        self._server.shutdown()
        self._upstream_client.close()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.debug("Stopped interceptor server: base_url=%s", self.base_url)

    def _build_handler(self):
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def end_headers(self) -> None:
                self.send_header("Connection", "close")
                self.close_connection = True
                super().end_headers()

            def do_GET(self) -> None:
                if self.path != "/healthz":
                    self.send_error(404)
                    return
                body = _JSON_CODEC.dumps_bytes(
                    {
                        "ok": True,
                        "upstream_url": outer._upstream_url,
                    }
                )
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:
                request_path = self.path.split("?")[0]
                if request_path not in {"/v1/chat/completions", "/v1/messages", "/v1/messages/count_tokens"}:
                    self.send_error(404)
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    body = self.rfile.read(length) if length else b"{}"
                    status_code, headers, body = outer._interceptor.intercept(
                        path=request_path,
                        headers=dict(self.headers.items()),
                        body=body,
                        client_host=str(self.client_address[0]),
                    )
                    self.send_response(status_code)
                    for key, value in headers:
                        if key.lower() in {"transfer-encoding", "connection"}:
                            continue
                        self.send_header(key, value)
                    self.end_headers()
                    try:
                        self.wfile.write(body)
                    except BrokenPipeError:
                        logger.debug("Client disconnected while writing response body: path=%s", self.path)
                        return
                except ValueError as exc:
                    logger.debug("Rejecting invalid intercepted request: path=%s error=%s", self.path, exc)
                    self.send_error(400, str(exc))

            def log_message(self, format: str, *args) -> None:
                _ = (format, args)
                return

        return Handler

    def _forward(
        self,
        path: str,
        headers: dict[str, str],
        payload: bytes,
    ) -> tuple[int, list[tuple[str, str]], bytes]:
        headers = {
            key: value
            for key, value in headers.items()
            if key.lower() not in {"host", "content-length"}
        }
        request_id = headers.get("X-Crab-Request-Id", "").strip()
        request_attributes = {
            "component": "interceptor",
            "request_id": request_id,
            "path": path,
        }
        logger.debug(
            "Forwarding request upstream: path=%s header_count=%s payload_bytes=%s upstream_url=%s",
            path,
            len(headers),
            len(payload),
            self._upstream_url,
        )
        started = time.perf_counter()
        status_code, response_headers, response_body = self._upstream_client.request(
            "POST",
            path,
            body=payload,
            headers=headers,
        )
        if self._telemetry is not None:
            self._telemetry.emit_metric(
                "llm.upstream_latency_ms",
                (time.perf_counter() - started) * 1000.0,
                {**request_attributes, "status_code": int(status_code)},
            )
        logger.debug(
            "Received upstream response: path=%s status_code=%s response_bytes=%s",
            path,
            status_code,
            len(response_body),
        )
        return int(status_code), response_headers, response_body
