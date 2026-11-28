"""Unit + simulated-E2E tests for observation staging (PR-B1.2): the
response-gate registry's staging extension, the interceptor's DROP → 409
handling (in-process and over real HTTP), and the CrabSystem staging
facade with its journal markers. Host-runnable — no runc."""
from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from crab import (
    AdapterFileSystemCWorker,
    AdapterFileSystemRWorker,
    AdapterProcessCWorker,
    AdapterProcessRWorker,
    CRExecutor,
    CRScheduler,
    CrabSystem,
    DefaultCWorker,
    DefaultRWorker,
    EBPFSandboxInspector,
    ExecutorConfig,
    InMemorySchedulerStateStore,
    InMemoryTelemetrySink,
    LocalCheckpointManager,
    RuncRuntime,
    RuncRuntimePaths,
    SandboxId,
    SchedulerConfig,
    StorageConfig,
)
from crab.interceptor import (
    CrabRequestInterceptor,
    CrabRequestInterceptorServer,
    InMemoryRequestStateStore,
    ReleaseDisposition,
    SandboxResponseGateRegistry,
)
from crab.journal import ActionJournal
from crab.runtime import CommandRunner
from crab.scheduler import FaultToleranceCheckpointingPolicy


class FakeCommandRunner(CommandRunner):
    def __init__(self) -> None:
        self.commands: list[tuple[str, ...]] = []

    def run(self, command: list[str], *, cwd: Path | None = None, timeout_seconds: float | None = None):
        _ = (cwd, timeout_seconds)
        self.commands.append(tuple(command))
        return type(
            "Result",
            (),
            {"command": tuple(command), "returncode": 0, "stdout": "", "stderr": ""},
        )()


def _wait_in_thread(registry: SandboxResponseGateRegistry, sandbox_id: SandboxId, generation: int):
    """Run wait_for_release in a thread; returns (thread, holder)."""
    holder: dict[str, ReleaseDisposition] = {}

    def _run() -> None:
        holder["disposition"] = registry.wait_for_release(sandbox_id, generation, timeout=10.0)

    thread = threading.Thread(target=_run)
    thread.start()
    return thread, holder


class StagingRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = SandboxResponseGateRegistry()
        self.registry.enable()
        self.sid = SandboxId("sbx-stage")

    def test_non_staging_release_delivers_immediately(self) -> None:
        generation = self.registry.arm(self.sid, "req-1")
        assert generation is not None
        thread, holder = _wait_in_thread(self.registry, self.sid, generation)
        self.registry.release(self.sid)
        thread.join(timeout=2.0)
        self.assertFalse(thread.is_alive())
        self.assertIs(holder["disposition"], ReleaseDisposition.DELIVER)

    def test_staged_release_keeps_waiter_blocked_then_commit_delivers(self) -> None:
        self.registry.begin_staging(self.sid)
        generation = self.registry.arm(self.sid, "req-1")
        assert generation is not None
        thread, holder = _wait_in_thread(self.registry, self.sid, generation)
        # Checkpoint-completion style release stages instead of waking.
        self.registry.release(self.sid)
        time.sleep(0.1)
        self.assertTrue(thread.is_alive(), "waiter must stay parked while staged")
        released = self.registry.release_staged(self.sid)
        thread.join(timeout=2.0)
        self.assertFalse(thread.is_alive())
        self.assertEqual(released, 1)
        self.assertIs(holder["disposition"], ReleaseDisposition.DELIVER)

    def test_discard_drops_waiter(self) -> None:
        self.registry.begin_staging(self.sid)
        generation = self.registry.arm(self.sid, "req-1")
        assert generation is not None
        thread, holder = _wait_in_thread(self.registry, self.sid, generation)
        self.registry.release(self.sid)
        discarded = self.registry.discard_staged(self.sid)
        thread.join(timeout=2.0)
        self.assertFalse(thread.is_alive())
        self.assertEqual(discarded, 1)
        self.assertIs(holder["disposition"], ReleaseDisposition.DROP)

    def test_end_staging_fail_open_delivers(self) -> None:
        self.registry.begin_staging(self.sid)
        generation = self.registry.arm(self.sid, "req-1")
        assert generation is not None
        thread, holder = _wait_in_thread(self.registry, self.sid, generation)
        self.registry.release(self.sid)
        time.sleep(0.05)
        self.assertTrue(thread.is_alive())
        leftover = self.registry.end_staging(self.sid)
        thread.join(timeout=2.0)
        self.assertFalse(thread.is_alive())
        self.assertEqual(leftover, 1)
        self.assertIs(holder["disposition"], ReleaseDisposition.DELIVER)
        self.assertFalse(self.registry.staging_active(self.sid))

    def test_release_pending_targeted_stages_when_staging(self) -> None:
        self.registry.begin_staging(self.sid)
        generation = self.registry.arm(self.sid, "req-1")
        assert generation is not None
        thread, holder = _wait_in_thread(self.registry, self.sid, generation)
        self.assertTrue(
            self.registry.release_pending(self.sid, request_id="req-1", generation=generation)
        )
        time.sleep(0.1)
        self.assertTrue(thread.is_alive(), "targeted release must stage too")
        self.registry.release_staged(self.sid)
        thread.join(timeout=2.0)
        self.assertIs(holder["disposition"], ReleaseDisposition.DELIVER)

    def test_group_satisfied_while_staged_blocks_late_sibling_rearm(self) -> None:
        self.registry.begin_staging(self.sid)
        generation = self.registry.arm(
            self.sid, "req-a", request_group_kind="spec_pair", request_group_id="pair-1"
        )
        assert generation is not None
        self.registry.release(self.sid)  # stages; marks group satisfied
        late = self.registry.arm(
            self.sid, "req-b", request_group_kind="spec_pair", request_group_id="pair-1"
        )
        self.assertIsNone(late, "satisfied group must not re-arm while staged")

    def test_begin_staging_clears_previous_drop_markers(self) -> None:
        self.registry.begin_staging(self.sid)
        first = self.registry.arm(self.sid, "req-1")
        assert first is not None
        self.registry.release(self.sid)
        self.registry.discard_staged(self.sid)
        self.assertIs(
            self.registry.wait_for_release(self.sid, first, timeout=1.0),
            ReleaseDisposition.DROP,
        )
        # New scope: fresh generation must not inherit the drop marker.
        self.registry.begin_staging(self.sid)
        second = self.registry.arm(self.sid, "req-2")
        assert second is not None
        self.registry.release(self.sid)
        self.registry.release_staged(self.sid)
        self.assertIs(
            self.registry.wait_for_release(self.sid, second, timeout=1.0),
            ReleaseDisposition.DELIVER,
        )

    def test_disabled_registry_still_returns_deliver(self) -> None:
        generation = self.registry.arm(self.sid, "req-1")
        assert generation is not None
        self.registry.disable()
        self.assertIs(
            self.registry.wait_for_release(self.sid, generation, timeout=1.0),
            ReleaseDisposition.DELIVER,
        )


# ---------------------------------------------------------------------------
# Interceptor DROP handling (in-process)
# ---------------------------------------------------------------------------


def _upstream_ok(path, headers, body):
    _ = (path, headers, body)
    payload = json.dumps({"choices": [{"message": {"content": "upstream-answer"}}]})
    return 200, [("Content-Type", "application/json")], payload.encode("utf-8")


def _chat_body() -> bytes:
    return json.dumps(
        {
            "model": "simulated-openai",
            "messages": [{"role": "user", "content": "continue"}],
        }
    ).encode("utf-8")


class InterceptorDropTests(unittest.TestCase):
    def _intercept_in_thread(self, interceptor: CrabRequestInterceptor, sandbox: str, request: str):
        holder: dict[str, tuple] = {}

        def _run() -> None:
            holder["response"] = interceptor.intercept(
                path="/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "X-Agent-Sandbox-Id": sandbox,
                    "X-Request-Id": request,
                },
                body=_chat_body(),
            )

        thread = threading.Thread(target=_run)
        thread.start()
        return thread, holder

    def _build(self):
        registry = SandboxResponseGateRegistry()
        registry.enable()
        ready = threading.Event()
        interceptor = CrabRequestInterceptor(
            upstream_transport=_upstream_ok,
            request_state_store=InMemoryRequestStateStore(),
            on_response_ready=lambda *_: ready.set(),
            response_gate_registry=registry,
        )
        return registry, interceptor, ready

    def test_discard_returns_409_crab_txn_aborted(self) -> None:
        registry, interceptor, ready = self._build()
        sid = SandboxId("sbx-drop")
        registry.begin_staging(sid)
        thread, holder = self._intercept_in_thread(interceptor, "sbx-drop", "req-drop")
        self.assertTrue(ready.wait(timeout=2.0))
        registry.release(sid)  # checkpoint-completion path: stages
        time.sleep(0.1)
        self.assertTrue(thread.is_alive(), "caller must stay gated while staged")
        registry.discard_staged(sid)
        thread.join(timeout=2.0)
        self.assertFalse(thread.is_alive())
        status, headers, body = holder["response"]
        self.assertEqual(status, 409)
        payload = json.loads(body.decode("utf-8"))
        self.assertEqual(payload["error"]["type"], "crab_txn_aborted")
        self.assertEqual(payload["error"]["request_id"], "req-drop")
        self.assertIn(("Content-Type", "application/json"), headers)

    def test_commit_delivers_upstream_response_unchanged(self) -> None:
        registry, interceptor, ready = self._build()
        sid = SandboxId("sbx-keep")
        registry.begin_staging(sid)
        thread, holder = self._intercept_in_thread(interceptor, "sbx-keep", "req-keep")
        self.assertTrue(ready.wait(timeout=2.0))
        registry.release(sid)
        registry.release_staged(sid)
        thread.join(timeout=2.0)
        self.assertFalse(thread.is_alive())
        status, _, body = holder["response"]
        self.assertEqual(status, 200)
        self.assertIn(b"upstream-answer", body)


# ---------------------------------------------------------------------------
# CrabSystem facade + journal markers
# ---------------------------------------------------------------------------


class SystemStagingFacadeTests(unittest.TestCase):
    def _build(self, root: Path):
        runner = FakeCommandRunner()
        telemetry = InMemoryTelemetrySink()
        inspector = EBPFSandboxInspector()
        journal = ActionJournal(root / "storage" / "journal")
        runtime = RuncRuntime(
            command_runner=runner,
            paths=RuncRuntimePaths(
                state_root=root / "runtime-state",
                bundle_root=root / "bundles",
                checkpoint_root=root / "checkpoints",
                metadata_root=root / "sandbox-metadata",
                zfs_dataset_prefix="pool/crab",
            ),
            action_recorder=journal,
        )
        storage = LocalCheckpointManager(
            StorageConfig(root_dir=root / "storage"),
            destroy_filesystem_ref=runtime.destroy_filesystem_ref,
        )
        executor = CRExecutor(
            ExecutorConfig(max_workers=1),
            DefaultCWorker(
                AdapterProcessCWorker(runtime),
                AdapterFileSystemCWorker(runtime),
                storage,
                runtime,
            ),
            DefaultRWorker(
                AdapterProcessRWorker(runtime),
                AdapterFileSystemRWorker(runtime),
                storage,
            ),
            telemetry,
        )
        scheduler_cfg = SchedulerConfig(
            min_checkpoint_interval_seconds=0.0,
            force_checkpoint_after_seconds=0.0,
            require_change_signal=True,
        )
        scheduler = CRScheduler(
            scheduler_cfg,
            inspector,
            runtime,
            InMemorySchedulerStateStore(),
            telemetry,
            FaultToleranceCheckpointingPolicy(scheduler_cfg),
        )
        system = CrabSystem(
            scheduler=scheduler,
            executor=executor,
            storage=storage,
            inspector=inspector,
            runtime=runtime,
            telemetry=telemetry,
            journal=journal,
        )
        return system, journal

    def test_facade_drives_registry_and_journals_transitions(self) -> None:
        with tempfile.TemporaryDirectory(prefix="crab_staging_sys_") as tmp:
            root = Path(tmp)
            system, journal = self._build(root)
            self.addCleanup(system.executor.shutdown)
            registry = system.response_gate_registry
            assert registry is not None
            registry.enable()
            sid = SandboxId("sbx-txn")

            system.begin_observation_staging(sid)
            self.assertTrue(registry.staging_active(sid))
            generation = registry.arm(sid, "req-1")
            assert generation is not None
            registry.release(sid)  # stages
            released = system.release_staged_observations(sid)
            self.assertEqual(released, 1)

            generation = registry.arm(sid, "req-2")
            assert generation is not None
            registry.release(sid)
            discarded = system.discard_staged_observations(sid)
            self.assertEqual(discarded, 1)

            leftover = system.end_observation_staging(sid)
            self.assertEqual(leftover, 0)
            self.assertFalse(registry.staging_active(sid))

            events = [
                (record.payload["event"], record.payload.get("metadata", {}))
                for record in journal.entries(sid, kind="lifecycle")
            ]
            self.assertEqual(
                [name for name, _ in events],
                ["staging_begin", "staging_commit", "staging_abort", "staging_end"],
            )
            metadata = dict(events)
            self.assertEqual(metadata["staging_commit"], {"released": 1})
            self.assertEqual(metadata["staging_abort"], {"discarded": 1})
            self.assertEqual(metadata["staging_end"], {"delivered_leftover": 0})


# ---------------------------------------------------------------------------
# Simulated E2E over real HTTP (interceptor server + fake LLM upstream)
# ---------------------------------------------------------------------------


class _FakeUpstreamHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        _ = self.rfile.read(length) if length else b""
        body = json.dumps({"choices": [{"message": {"content": "live-upstream"}}]}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        _ = (format, args)


class StagingHttpE2ETests(unittest.TestCase):
    """The full wire path: HTTP client → interceptor server → fake LLM
    upstream, with staging driven system-side. Simulated E2E per the B1
    design's exit criteria; the sandboxed variant arrives with B2's txn
    E2E."""

    def setUp(self) -> None:
        self.upstream = HTTPServer(("127.0.0.1", 0), _FakeUpstreamHandler)
        threading.Thread(target=self.upstream.serve_forever, daemon=True).start()
        self.addCleanup(self.upstream.server_close)
        self.addCleanup(self.upstream.shutdown)
        self.registry = SandboxResponseGateRegistry()
        self.registry.enable()
        self.ready = threading.Event()
        upstream_host, upstream_port = self.upstream.server_address
        self.server = CrabRequestInterceptorServer(
            upstream_url=f"http://{upstream_host}:{upstream_port}",
            request_state_store=InMemoryRequestStateStore(),
            response_gate_registry=self.registry,
            on_response_ready=lambda *_: self.ready.set(),
        )
        self.server.start()
        self.addCleanup(self.server.stop)

    def _post_in_thread(self, sandbox: str, request_id: str):
        holder: dict[str, object] = {}

        def _run() -> None:
            req = urllib.request.Request(
                f"{self.server.base_url}/v1/chat/completions",
                data=_chat_body(),
                headers={
                    "Content-Type": "application/json",
                    "X-Agent-Sandbox-Id": sandbox,
                    "X-Request-Id": request_id,
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=15.0) as response:
                    holder["status"] = response.status
                    holder["body"] = response.read()
            except urllib.error.HTTPError as exc:
                holder["status"] = exc.code
                holder["body"] = exc.read()

        thread = threading.Thread(target=_run)
        thread.start()
        return thread, holder

    def test_staged_response_dropped_over_the_wire(self) -> None:
        sid = SandboxId("sbx-wire")
        self.registry.begin_staging(sid)
        thread, holder = self._post_in_thread("sbx-wire", "req-wire-1")
        self.assertTrue(self.ready.wait(timeout=5.0))
        self.registry.release(sid)
        time.sleep(0.1)
        self.assertTrue(thread.is_alive(), "HTTP caller must stay gated while staged")
        self.registry.discard_staged(sid)
        thread.join(timeout=5.0)
        self.assertFalse(thread.is_alive())
        self.assertEqual(holder["status"], 409)
        payload = json.loads(bytes(holder["body"]).decode("utf-8"))
        self.assertEqual(payload["error"]["type"], "crab_txn_aborted")

    def test_staged_response_delivered_exactly_once_on_commit(self) -> None:
        sid = SandboxId("sbx-wire-2")
        self.registry.begin_staging(sid)
        self.ready.clear()
        thread, holder = self._post_in_thread("sbx-wire-2", "req-wire-2")
        self.assertTrue(self.ready.wait(timeout=5.0))
        self.registry.release(sid)
        released = self.registry.release_staged(sid)
        thread.join(timeout=5.0)
        self.assertFalse(thread.is_alive())
        self.assertEqual(released, 1)
        self.assertEqual(holder["status"], 200)
        self.assertIn(b"live-upstream", bytes(holder["body"]))
        # Nothing left staged: a second commit is a no-op.
        self.assertEqual(self.registry.release_staged(sid), 0)


if __name__ == "__main__":
    unittest.main()
