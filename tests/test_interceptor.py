from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from crab import (
    CrabRequestInterceptor,
    CrabRequestInterceptorServer,
    CompositeRequestInterceptorHook,
    InMemoryRequestStateStore,
    InMemoryTelemetrySink,
    SandboxId,
    SandboxSnapshot,
    SandboxResponseGateRegistry,
    StorageConfig,
    TelemetryRequestInterceptorHook,
    build_default_system,
)
from crab.models import utc_now
from integrations.llm_services.simulated.service import SimulatedLLMState, handle_request


class SlowRecordingTelemetrySink(InMemoryTelemetrySink):
    def emit_event(self, name: str, attributes: dict[str, object]) -> None:
        time.sleep(0.02)
        super().emit_event(name, attributes)

    def emit_metric(
        self,
        name: str,
        value: float,
        attributes: dict[str, object] | None = None,
    ) -> None:
        time.sleep(0.02)
        super().emit_metric(name, value, attributes)


class InterceptorTests(unittest.TestCase):
    def test_build_default_system_disables_restore_validation_by_default_and_allows_opt_in(self) -> None:
        with tempfile.TemporaryDirectory(prefix="crab_interceptor_builder_") as tmp:
            default_system = build_default_system(
                storage_root=tmp,
                runtime="docker",
                storage_config=StorageConfig(root_dir=Path(tmp)),
            )
            enabled_system = build_default_system(
                storage_root=tmp,
                runtime="docker",
                storage_config=StorageConfig(root_dir=Path(tmp)),
                enforce_restore_checkpoint_validation=True,
            )

            self.assertFalse(default_system.enforce_restore_checkpoint_validation)
            self.assertTrue(enabled_system.enforce_restore_checkpoint_validation)
            default_system.executor.shutdown()
            enabled_system.executor.shutdown()

    def test_response_gate_registry_tracks_request_identity(self) -> None:
        registry = SandboxResponseGateRegistry()
        sandbox_id = SandboxId("sbx-gate")
        registry.enable()

        generation = registry.arm(sandbox_id, "req-1")

        self.assertEqual(generation, 1)
        pending = registry.get_pending(sandbox_id)
        self.assertIsNotNone(pending)
        assert pending is not None
        self.assertEqual(pending.request_id, "req-1")
        self.assertEqual(pending.generation, 1)

    def test_response_gate_registry_targeted_release_releases_older_request_without_releasing_newer_request(self) -> None:
        registry = SandboxResponseGateRegistry()
        sandbox_id = SandboxId("sbx-gate")
        registry.enable()

        first_generation = registry.arm(sandbox_id, "req-1")
        second_generation = registry.arm(sandbox_id, "req-2")

        self.assertTrue(
            registry.release_pending(sandbox_id, request_id="req-1", generation=first_generation),
        )
        pending = registry.get_pending(sandbox_id)
        self.assertIsNotNone(pending)
        assert pending is not None
        self.assertEqual(pending.request_id, "req-2")
        self.assertEqual(pending.generation, second_generation)
        self.assertTrue(
            registry.release_pending(sandbox_id, request_id="req-2", generation=second_generation),
        )
        self.assertIsNone(registry.get_pending(sandbox_id))

    def test_response_gate_registry_groups_spec_pair_requests_into_one_generation(self) -> None:
        registry = SandboxResponseGateRegistry()
        sandbox_id = SandboxId("sbx-spec-group")
        registry.enable()

        first_generation = registry.arm(
            sandbox_id,
            "req-draft",
            request_group_kind="spec_pair",
            request_group_id="pair-1",
        )
        second_generation = registry.arm(
            sandbox_id,
            "req-oracle",
            request_group_kind="spec_pair",
            request_group_id="pair-1",
        )

        self.assertEqual(first_generation, second_generation)
        pending = registry.get_oldest_pending(sandbox_id)
        self.assertIsNotNone(pending)
        assert pending is not None
        self.assertEqual(pending.request_group_kind, "spec_pair")
        self.assertEqual(pending.request_group_id, "pair-1")
        self.assertEqual(set(pending.request_ids), {"req-draft", "req-oracle"})

    def test_response_gate_registry_does_not_rearm_late_sibling_after_group_release(self) -> None:
        registry = SandboxResponseGateRegistry()
        sandbox_id = SandboxId("sbx-spec-group")
        registry.enable()

        generation = registry.arm(
            sandbox_id,
            "req-draft",
            request_group_kind="spec_pair",
            request_group_id="pair-1",
        )
        self.assertIsNotNone(generation)
        assert generation is not None
        self.assertTrue(
            registry.release_pending(
                sandbox_id,
                request_id="req-draft",
                generation=generation,
            )
        )

        self.assertIsNone(
            registry.arm(
                sandbox_id,
                "req-oracle",
                request_group_kind="spec_pair",
                request_group_id="pair-1",
            )
        )

    def test_interceptor_tracks_request_state_and_emits_telemetry(self) -> None:
        request_state_store = InMemoryRequestStateStore()
        telemetry = InMemoryTelemetrySink()
        hook = CompositeRequestInterceptorHook([TelemetryRequestInterceptorHook(telemetry)])
        llm_state = SimulatedLLMState()
        interceptor = CrabRequestInterceptor(
            upstream_transport=lambda path, headers, body: (
                200,
                [("Content-Type", "application/json")],
                json.dumps(
                    handle_request(
                        path=path,
                        headers=headers,
                        payload=json.loads(body.decode("utf-8")),
                        state=llm_state,
                    ),
                    sort_keys=True,
                ).encode("utf-8"),
            ),
            request_state_store=request_state_store,
            hook=hook,
        )
        _, _, body = interceptor.intercept(
            path="/v1/chat/completions",
            headers={"Content-Type": "application/json", "X-Agent-Sandbox-Id": "sbx-1", "X-Request-Id": "req-1"},
            body=json.dumps(
                {
                    "model": "simulated-openai",
                    "messages": [{"role": "user", "content": "continue"}],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "show_pwd",
                                "description": "x",
                                "parameters": {"type": "object", "properties": {}},
                            },
                        }
                    ],
                }
            ).encode("utf-8"),
        )
        payload = json.loads(body.decode("utf-8"))

        self.assertEqual(payload["choices"][0]["message"]["tool_calls"][0]["function"]["name"], "show_pwd")
        state = request_state_store.get(SandboxId("sbx-1"))
        self.assertEqual(state.total_llm_requests, 1)
        self.assertEqual(state.completed_llm_requests, 1)
        self.assertEqual(state.active_llm_requests, 0)
        self.assertEqual(state.last_llm_provider, "openai")
        event_names = [name for name, _ in telemetry.events]
        self.assertEqual(event_names.count("request.start"), 1)
        self.assertEqual(event_names.count("request.finish"), 1)

    def test_interceptor_calls_response_ready_callback_before_gate_release(self) -> None:
        request_state_store = InMemoryRequestStateStore()
        response_gate_registry = SandboxResponseGateRegistry()
        response_gate_registry.enable()
        callback_event = threading.Event()
        callback_calls: list[tuple[SandboxId, str, int | None]] = []
        interceptor = CrabRequestInterceptor(
            upstream_transport=lambda path, headers, body: (
                200,
                [("Content-Type", "application/json")],
                json.dumps(
                    handle_request(
                        path=path,
                        headers=headers,
                        payload=json.loads(body.decode("utf-8")),
                        state=SimulatedLLMState(),
                    ),
                    sort_keys=True,
                ).encode("utf-8"),
            ),
            request_state_store=request_state_store,
            on_response_ready=lambda sandbox_id, request_id, generation: (
                callback_calls.append((sandbox_id, request_id, generation)),
                callback_event.set(),
            ),
            response_gate_registry=response_gate_registry,
        )
        response_holder: dict[str, object] = {}

        def _run_request() -> None:
            response_holder["response"] = interceptor.intercept(
                path="/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "X-Agent-Sandbox-Id": "sbx-callback",
                    "X-Request-Id": "req-callback",
                },
                body=json.dumps(
                    {
                        "model": "simulated-openai",
                        "messages": [{"role": "user", "content": "continue"}],
                    }
                ).encode("utf-8"),
            )

        thread = threading.Thread(target=_run_request)
        thread.start()
        self.assertTrue(callback_event.wait(timeout=2.0))
        pending = response_gate_registry.find_pending_request(SandboxId("sbx-callback"), "req-callback")
        self.assertIsNotNone(pending)
        assert pending is not None
        self.assertEqual(callback_calls, [(SandboxId("sbx-callback"), "req-callback", pending.generation)])
        self.assertTrue(
            response_gate_registry.release_pending(
                SandboxId("sbx-callback"),
                request_id="req-callback",
                generation=pending.generation,
            )
        )
        thread.join(timeout=2.0)
        self.assertFalse(thread.is_alive())
        self.assertIn("response", response_holder)

    def test_interceptor_resolves_sandbox_id_from_client_host_and_overrides_forwarded_header(self) -> None:
        request_state_store = InMemoryRequestStateStore()
        forwarded_headers: dict[str, str] = {}

        def _upstream_transport(path, headers, body):
            forwarded_headers.update(headers)
            return (
                200,
                [("Content-Type", "application/json")],
                json.dumps(
                    handle_request(
                        path=path,
                        headers=headers,
                        payload=json.loads(body.decode("utf-8")),
                        state=SimulatedLLMState(),
                    ),
                    sort_keys=True,
                ).encode("utf-8"),
            )

        interceptor = CrabRequestInterceptor(
            upstream_transport=_upstream_transport,
            request_state_store=request_state_store,
            sandbox_id_resolver=lambda client_host, headers, body: "fork-1" if client_host == "10.250.0.8" else None,
        )

        interceptor.intercept(
            path="/v1/chat/completions",
            headers={"Content-Type": "application/json", "X-Agent-Sandbox-Id": "source-1", "X-Request-Id": "req-1"},
            body=json.dumps({"model": "simulated-openai", "messages": [{"role": "user", "content": "continue"}]}).encode(
                "utf-8"
            ),
            client_host="10.250.0.8",
        )

        self.assertEqual(request_state_store.get(SandboxId("fork-1")).total_llm_requests, 1)
        self.assertEqual(request_state_store.get(SandboxId("source-1")).total_llm_requests, 0)
        self.assertEqual(forwarded_headers["X-Agent-Sandbox-Id"], "fork-1")
        self.assertEqual(forwarded_headers["X-Request-Id"], "req-1")
        self.assertEqual(forwarded_headers["X-Crab-Request-Id"], "req-1")

    def test_interceptor_notifies_system_scheduler(self) -> None:
        with tempfile.TemporaryDirectory(prefix="crab_interceptor_") as tmp:
            system = build_default_system(
                storage_root=tmp,
                runtime="docker",
                storage_config=StorageConfig(root_dir=Path(tmp)),
            )

            interceptor = CrabRequestInterceptor(
                upstream_transport=lambda path, headers, body: (
                    200,
                    [("Content-Type", "application/json")],
                    json.dumps(
                        handle_request(
                            path=path,
                            headers=headers,
                            payload=json.loads(body.decode("utf-8")),
                            state=SimulatedLLMState(),
                            response_delay_ms=50,
                        ),
                        sort_keys=True,
                    ).encode("utf-8"),
                ),
                request_state_store=system.request_state_store or InMemoryRequestStateStore(),
                on_state_change=system.notify_interceptor_state_change,
            )
            interceptor.intercept(
                path="/v1/messages",
                headers={"Content-Type": "application/json", "X-Agent-Sandbox-Id": "sbx-2"},
                body=json.dumps(
                    {
                        "model": "simulated-anthropic",
                        "messages": [{"role": "user", "content": "continue"}],
                        "tools": [
                            {
                                "name": "read_workdir",
                                "description": "x",
                                "input_schema": {"type": "object", "properties": {}},
                            }
                        ],
                    }
                ).encode("utf-8"),
            )

            # notify_interceptor_state_change reconciles against the actual
            # response-gate state instead of unconditionally marking the
            # sandbox pending (an unconditional add would leak the sandbox
            # forever in manual mode). This request completed synchronously
            # with no gated response, so no pending signal remains — but the
            # notification path itself must have fired.
            self.assertFalse(system.has_pending_interceptor_signal(SandboxId("sbx-2")))
            event_names = [name for name, _ in system.telemetry.events]
            self.assertIn("interceptor.state_changed", event_names)
            system.executor.shutdown()

    def test_interceptor_rejects_request_when_sandbox_identity_is_missing(self) -> None:
        request_state_store = InMemoryRequestStateStore()
        interceptor = CrabRequestInterceptor(
            upstream_transport=lambda path, headers, body: (
                200,
                [("Content-Type", "application/json")],
                json.dumps(
                    handle_request(
                        path=path,
                        headers=headers,
                        payload=json.loads(body.decode("utf-8")),
                        state=SimulatedLLMState(),
                    ),
                    sort_keys=True,
                ).encode("utf-8"),
            ),
            request_state_store=request_state_store,
        )

        with self.assertRaisesRegex(ValueError, "missing sandbox identity"):
            interceptor.intercept(
                path="/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                body=json.dumps(
                    {
                        "model": "simulated-openai",
                        "messages": [{"role": "user", "content": "continue"}],
                    }
                ).encode("utf-8"),
            )

    def test_interceptor_resolver_maps_distinct_client_hosts_to_distinct_sandboxes(self) -> None:
        request_state_store = InMemoryRequestStateStore()
        interceptor = CrabRequestInterceptor(
            upstream_transport=lambda path, headers, body: (
                200,
                [("Content-Type", "application/json")],
                json.dumps(
                    handle_request(
                        path=path,
                        headers=headers,
                        payload=json.loads(body.decode("utf-8")),
                        state=SimulatedLLMState(),
                    ),
                    sort_keys=True,
                ).encode("utf-8"),
            ),
            request_state_store=request_state_store,
            sandbox_id_resolver=lambda client_host, headers, body: {
                "172.17.0.240": "sbx-iflow-a",
                "172.17.0.241": "sbx-iflow-b",
            }.get("" if client_host is None else client_host),
        )

        interceptor.intercept(
            path="/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            body=json.dumps({"model": "simulated-openai", "messages": [{"role": "user", "content": "continue"}]}).encode(
                "utf-8"
            ),
            client_host="172.17.0.240",
        )
        interceptor.intercept(
            path="/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            body=json.dumps({"model": "simulated-openai", "messages": [{"role": "user", "content": "continue"}]}).encode(
                "utf-8"
            ),
            client_host="172.17.0.241",
        )

        self.assertEqual(request_state_store.get(SandboxId("sbx-iflow-a")).total_llm_requests, 1)
        self.assertEqual(request_state_store.get(SandboxId("sbx-iflow-b")).total_llm_requests, 1)

    def test_interceptor_waits_for_system_checkpoint_flow(self) -> None:
        with tempfile.TemporaryDirectory(prefix="crab_interceptor_system_") as tmp:
            system = build_default_system(
                storage_root=tmp,
                runtime="docker",
                storage_config=StorageConfig(root_dir=Path(tmp)),
            )
            sandbox_id = system.sandbox_manager.launch("docker")
            system.inspector.upsert_snapshot(
                SandboxSnapshot(
                    sandbox_id=sandbox_id,
                    runtime_name="docker",
                    is_running=True,
                    process_changed=True,
                    filesystem_changed=False,
                    observed_at=utc_now(),
                )
            )
            system.start()

            interceptor = CrabRequestInterceptor(
                upstream_transport=lambda path, headers, body: (
                    200,
                    [("Content-Type", "application/json")],
                    json.dumps(
                        handle_request(
                            path=path,
                            headers=headers,
                            payload=json.loads(body.decode("utf-8")),
                            state=SimulatedLLMState(),
                        ),
                        sort_keys=True,
                    ).encode("utf-8"),
                ),
                request_state_store=system.request_state_store or InMemoryRequestStateStore(),
                response_gate_registry=system.response_gate_registry,
            )
            _, _, body = interceptor.intercept(
                path="/v1/chat/completions",
                headers={"Content-Type": "application/json", "X-Agent-Sandbox-Id": str(sandbox_id)},
                body=json.dumps(
                    {
                        "model": "simulated-openai",
                        "messages": [{"role": "user", "content": "continue"}],
                    }
                ).encode("utf-8"),
            )

            payload = json.loads(body.decode("utf-8"))
            self.assertIn("choices", payload)
            state = system.request_state_store.get(sandbox_id) if system.request_state_store is not None else None
            self.assertIsNotNone(state)
            assert state is not None
            self.assertEqual(state.active_llm_requests, 0)
            event_names = [name for name, _ in system.telemetry.events]
            self.assertIn("scheduler.evaluate", event_names)
            self.assertIn("executor.job_finished", event_names)
            self.assertEqual(system.sandbox_manager.describe(sandbox_id).status, "running")
            system.stop()
            system.executor.shutdown()

    def test_interceptor_releases_buffered_response_for_matching_restored_request(self) -> None:
        request_state_store = InMemoryRequestStateStore()
        response_gate_registry = SandboxResponseGateRegistry()
        response_gate_registry.enable()
        interceptor = CrabRequestInterceptor(
            upstream_transport=lambda path, headers, body: (
                200,
                [("Content-Type", "application/json")],
                json.dumps(
                    handle_request(
                        path=path,
                        headers=headers,
                        payload=json.loads(body.decode("utf-8")),
                        state=SimulatedLLMState(),
                    ),
                    sort_keys=True,
                ).encode("utf-8"),
            ),
            request_state_store=request_state_store,
            response_gate_registry=response_gate_registry,
        )

        response_body: dict[str, object] = {}
        finished = threading.Event()

        def _run_request() -> None:
            _, _, body = interceptor.intercept(
                path="/v1/chat/completions",
                headers={"Content-Type": "application/json", "X-Agent-Sandbox-Id": "sbx-restore", "X-Request-Id": "req-live"},
                body=json.dumps(
                    {
                        "model": "simulated-openai",
                        "messages": [{"role": "user", "content": "continue"}],
                    }
                ).encode("utf-8"),
            )
            response_body["payload"] = json.loads(body.decode("utf-8"))
            finished.set()

        thread = threading.Thread(target=_run_request)
        thread.start()
        pending = None
        for _ in range(50):
            pending = response_gate_registry.get_pending(SandboxId("sbx-restore"))
            if pending is not None:
                break
            finished.wait(0.01)
        self.assertIsNotNone(pending)
        assert pending is not None
        self.assertTrue(
            response_gate_registry.release_pending(
                SandboxId("sbx-restore"),
                request_id="req-live",
                generation=pending.generation,
            )
        )
        thread.join(timeout=2.0)

        self.assertTrue(finished.is_set())
        self.assertIn("payload", response_body)
        state = request_state_store.get(SandboxId("sbx-restore"))
        self.assertEqual(state.completed_llm_requests, 1)

    def test_interceptor_keeps_newer_overlapping_request_gated_until_its_generation_releases(self) -> None:
        request_state_store = InMemoryRequestStateStore()
        response_gate_registry = SandboxResponseGateRegistry()
        response_gate_registry.enable()
        interceptor = CrabRequestInterceptor(
            upstream_transport=lambda path, headers, body: (200, [("Content-Type", "application/json")], b"{}"),
            request_state_store=request_state_store,
            response_gate_registry=response_gate_registry,
        )

        finished: list[str] = []

        def _run_request(request_id: str) -> None:
            interceptor.intercept(
                path="/v1/chat/completions",
                headers={"Content-Type": "application/json", "X-Agent-Sandbox-Id": "sbx-overlap", "X-Request-Id": request_id},
                body=b"{}",
            )
            finished.append(request_id)

        first = threading.Thread(target=_run_request, args=("req-1",))
        second = threading.Thread(target=_run_request, args=("req-2",))
        first.start()
        pending_first = None
        for _ in range(50):
            pending_first = response_gate_registry.find_pending_request(SandboxId("sbx-overlap"), "req-1")
            if pending_first is not None:
                break
            time.sleep(0.01)
        self.assertIsNotNone(pending_first)
        second.start()
        pending_second = None
        for _ in range(50):
            pending_second = response_gate_registry.find_pending_request(SandboxId("sbx-overlap"), "req-2")
            if pending_second is not None:
                break
            time.sleep(0.01)
        self.assertIsNotNone(pending_second)
        assert pending_first is not None
        assert pending_second is not None

        self.assertTrue(
            response_gate_registry.release_pending(
                SandboxId("sbx-overlap"),
                request_id="req-1",
                generation=pending_first.generation,
            )
        )
        first.join(timeout=2.0)
        self.assertFalse(first.is_alive())
        second.join(timeout=0.05)
        self.assertTrue(second.is_alive())
        self.assertEqual(finished, ["req-1"])
        self.assertTrue(
            response_gate_registry.release_pending(
                SandboxId("sbx-overlap"),
                request_id="req-2",
                generation=pending_second.generation,
            )
        )
        second.join(timeout=2.0)
        self.assertFalse(second.is_alive())
        self.assertEqual(finished, ["req-1", "req-2"])

    def test_interceptor_gate_metrics_are_sampled_before_telemetry_writes(self) -> None:
        request_state_store = InMemoryRequestStateStore()
        response_gate_registry = SandboxResponseGateRegistry()
        response_gate_registry.enable()
        telemetry = SlowRecordingTelemetrySink()
        interceptor = CrabRequestInterceptor(
            upstream_transport=lambda path, headers, body: (200, [("Content-Type", "application/json")], b"{}"),
            request_state_store=request_state_store,
            response_gate_registry=response_gate_registry,
            telemetry=telemetry,
        )

        def _release_gate() -> None:
            pending = None
            for _ in range(100):
                pending = response_gate_registry.find_pending_request(SandboxId("sbx-metrics"), "req-metrics")
                if pending is not None:
                    break
                time.sleep(0.005)
            assert pending is not None
            time.sleep(0.03)
            response_gate_registry.release_pending(
                SandboxId("sbx-metrics"),
                request_id="req-metrics",
                generation=pending.generation,
            )

        releaser = threading.Thread(target=_release_gate)
        releaser.start()
        interceptor.intercept(
            path="/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "X-Agent-Sandbox-Id": "sbx-metrics",
                "X-Request-Id": "req-metrics",
                "X-Crab-Spec-Pair-Id": "pair-metrics",
                "X-Crab-Spec-Role": "oracle",
            },
            body=b"{}",
        )
        releaser.join(timeout=2.0)

        metric_map = {
            name: value
            for name, value, attributes in telemetry.metrics
            if attributes.get("request_id") == "req-metrics"
        }
        gate_wait = metric_map["llm.gate_wait_ms"]
        gate_operation = metric_map["interceptor.response_gate.wait.duration_ms"]
        crab_delay = metric_map["llm.crab_delay_ms"]

        self.assertLess(abs(gate_operation - gate_wait), 25.0)
        self.assertLess(abs(crab_delay - gate_wait), 25.0)
        metric_attrs = {
            name: attributes
            for name, _value, attributes in telemetry.metrics
            if attributes.get("request_id") == "req-metrics"
        }
        self.assertEqual(metric_attrs["llm.gate_wait_ms"]["request_group_role"], "oracle")
        self.assertEqual(metric_attrs["llm.crab_delay_ms"]["request_group_role"], "oracle")
        self.assertEqual(metric_attrs["interceptor.response_gate.wait.duration_ms"]["request_group_role"], "oracle")

    def test_anthropic_count_tokens_request_bypasses_response_gate(self) -> None:
        request_state_store = InMemoryRequestStateStore()
        response_gate_registry = SandboxResponseGateRegistry()
        response_gate_registry.enable()
        telemetry = InMemoryTelemetrySink()
        interceptor = CrabRequestInterceptor(
            upstream_transport=lambda path, headers, body: (
                200,
                [("Content-Type", "application/json")],
                json.dumps({"input_tokens": 7}).encode("utf-8"),
            ),
            request_state_store=request_state_store,
            response_gate_registry=response_gate_registry,
            telemetry=telemetry,
        )

        interceptor.intercept(
            path="/v1/messages/count_tokens",
            headers={
                "Content-Type": "application/json",
                "X-Agent-Sandbox-Id": "sbx-count",
                "X-Request-Id": "req-count",
            },
            body=json.dumps({"messages": [{"role": "user", "content": "task"}]}).encode("utf-8"),
        )

        self.assertIsNone(response_gate_registry.get_pending(SandboxId("sbx-count")))
        metric_map = {
            name: value
            for name, value, attributes in telemetry.metrics
            if attributes.get("request_id") == "req-count"
        }
        self.assertEqual(metric_map["llm.gate_wait_ms"], 0.0)
        request_events = [
            (name, attributes)
            for name, attributes in telemetry.events
            if attributes.get("request_id") == "req-count"
        ]
        received_attrs = next(attrs for name, attrs in request_events if name == "interceptor.request.received")
        self.assertEqual(received_attrs["request_kind"], "count_tokens")
        self.assertEqual(received_attrs["response_gate_enabled"], False)

    def test_anthropic_helper_request_bypasses_response_gate(self) -> None:
        request_state_store = InMemoryRequestStateStore()
        response_gate_registry = SandboxResponseGateRegistry()
        response_gate_registry.enable()
        telemetry = InMemoryTelemetrySink()
        interceptor = CrabRequestInterceptor(
            upstream_transport=lambda path, headers, body: (
                200,
                [("Content-Type", "text/event-stream")],
                (
                    b'event: message_start\n'
                    b'data: {"type":"message_start","message":{"id":"msg_helper","type":"message","role":"assistant","content":[],"model":"claude-haiku-4-5-20251001","stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":1,"cache_creation_input_tokens":0,"cache_read_input_tokens":0}}}\n\n'
                    b'event: message_stop\n'
                    b'data: {"type":"message_stop"}\n\n'
                ),
            ),
            request_state_store=request_state_store,
            response_gate_registry=response_gate_registry,
            telemetry=telemetry,
        )

        interceptor.intercept(
            path="/v1/messages",
            headers={
                "Content-Type": "application/json",
                "X-Agent-Sandbox-Id": "sbx-helper",
                "X-Request-Id": "req-helper",
            },
            body=json.dumps(
                {
                    "model": "claude-haiku-4-5-20251001",
                    "messages": [{"role": "user", "content": "helper"}],
                }
            ).encode("utf-8"),
        )

        self.assertIsNone(response_gate_registry.get_pending(SandboxId("sbx-helper")))
        metric_map = {
            name: value
            for name, value, attributes in telemetry.metrics
            if attributes.get("request_id") == "req-helper"
        }
        self.assertEqual(metric_map["llm.gate_wait_ms"], 0.0)
        received_attrs = next(
            attributes
            for name, attributes in telemetry.events
            if name == "interceptor.request.received" and attributes.get("request_id") == "req-helper"
        )
        self.assertEqual(received_attrs["request_kind"], "helper")
        self.assertEqual(received_attrs["response_gate_enabled"], False)

    def test_anthropic_main_loop_request_still_uses_response_gate(self) -> None:
        request_state_store = InMemoryRequestStateStore()
        response_gate_registry = SandboxResponseGateRegistry()
        response_gate_registry.enable()
        telemetry = InMemoryTelemetrySink()
        interceptor = CrabRequestInterceptor(
            upstream_transport=lambda path, headers, body: (
                200,
                [("Content-Type", "text/event-stream")],
                (
                    b'event: message_start\n'
                    b'data: {"type":"message_start","message":{"id":"msg_main","type":"message","role":"assistant","content":[],"model":"claude-opus-4-6","stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":1,"cache_creation_input_tokens":0,"cache_read_input_tokens":0}}}\n\n'
                    b'event: message_stop\n'
                    b'data: {"type":"message_stop"}\n\n'
                ),
            ),
            request_state_store=request_state_store,
            response_gate_registry=response_gate_registry,
            telemetry=telemetry,
        )

        finished = threading.Event()

        def _run_request() -> None:
            interceptor.intercept(
                path="/v1/messages",
                headers={
                    "Content-Type": "application/json",
                    "X-Agent-Sandbox-Id": "sbx-main",
                    "X-Request-Id": "req-main",
                },
                body=json.dumps(
                    {
                        "model": "claude-opus-4-6",
                        "messages": [{"role": "user", "content": "continue"}],
                    }
                ).encode("utf-8"),
            )
            finished.set()

        thread = threading.Thread(target=_run_request)
        thread.start()
        pending = None
        for _ in range(100):
            pending = response_gate_registry.find_pending_request(SandboxId("sbx-main"), "req-main")
            if pending is not None:
                break
            time.sleep(0.005)
        self.assertIsNotNone(pending)
        assert pending is not None
        active_context = request_state_store.get_request_context(SandboxId("sbx-main"), "req-main")
        self.assertIsNotNone(active_context)
        assert active_context is not None
        self.assertEqual(active_context.metadata["request_kind"], "main_loop")
        self.assertEqual(active_context.metadata["response_gate_enabled"], True)

        time.sleep(0.02)
        response_gate_registry.release_pending(
            SandboxId("sbx-main"),
            request_id="req-main",
            generation=pending.generation,
        )
        thread.join(timeout=2.0)

        self.assertTrue(finished.is_set())
        metric_map = {
            name: value
            for name, value, attributes in telemetry.metrics
            if attributes.get("request_id") == "req-main"
        }
        self.assertGreater(metric_map["llm.gate_wait_ms"], 0.0)

    def test_interceptor_server_healthz_returns_json(self) -> None:
        server = CrabRequestInterceptorServer(
            upstream_url="http://127.0.0.1:9999",
            request_state_store=InMemoryRequestStateStore(),
            port=0,
        )
        server.start()
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{server.port}/healthz", timeout=2.0) as response:
                self.assertEqual(response.status, 200)
                payload = json.loads(response.read().decode("utf-8"))
        finally:
            server.stop()

        self.assertEqual(payload["ok"], True)
        self.assertEqual(payload["upstream_url"], "http://127.0.0.1:9999")

    def test_interceptor_server_forwards_anthropic_count_tokens_requests(self) -> None:
        captured_paths: list[str] = []

        class _UpstreamHandler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                captured_paths.append(self.path)
                length = int(self.headers.get("Content-Length", "0"))
                _ = self.rfile.read(length) if length else b"{}"
                body = json.dumps({"input_tokens": 7}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args) -> None:
                _ = (format, args)
                return

        upstream = HTTPServer(("127.0.0.1", 0), _UpstreamHandler)
        upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
        upstream_thread.start()
        self.addCleanup(upstream.shutdown)
        self.addCleanup(upstream.server_close)
        self.addCleanup(upstream_thread.join, 5.0)

        server = CrabRequestInterceptorServer(
            upstream_url=f"http://127.0.0.1:{upstream.server_address[1]}",
            request_state_store=InMemoryRequestStateStore(),
            port=0,
        )
        server.start()
        try:
            request = urllib.request.Request(
                f"http://127.0.0.1:{server.port}/v1/messages/count_tokens",
                data=json.dumps({"messages": [{"role": "user", "content": "task"}]}).encode("utf-8"),
                headers={"Content-Type": "application/json", "X-Agent-Sandbox-Id": "sbx-count"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=2.0) as response:
                payload = json.loads(response.read().decode("utf-8"))
        finally:
            server.stop()

        self.assertEqual(payload["input_tokens"], 7)
        self.assertEqual(captured_paths, ["/v1/messages/count_tokens"])
    def test_interceptor_server_stop_closes_upstream_connections_before_server_close(self) -> None:
        server = CrabRequestInterceptorServer(
            upstream_url="http://127.0.0.1:9999",
            request_state_store=InMemoryRequestStateStore(),
            port=0,
        )
        call_order: list[str] = []
        server._server.server_close()

        class _FakeServer:
            server_address = ("127.0.0.1", 12345)

            def shutdown(self) -> None:
                call_order.append("shutdown")

            def server_close(self) -> None:
                call_order.append("server_close")

        class _FakeUpstreamClient:
            def close(self) -> None:
                call_order.append("upstream_close")

        server._server = _FakeServer()  # type: ignore[assignment]
        server._upstream_client = _FakeUpstreamClient()  # type: ignore[assignment]
        server._thread = None

        server.stop()

        self.assertEqual(call_order, ["shutdown", "upstream_close", "server_close"])


if __name__ == "__main__":
    unittest.main()
