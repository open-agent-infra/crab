"""Unit tests for the crab-gateway HTTP facade (track S, S1): bearer
auth and tenant isolation, the quota gate, two-phase create with intent
metadata, route proxying and verbatim daemon-error passthrough, the
startup reconciliation pass with the /info boot-identity check, and the
transport socket perms/group override. Host-runnable — the daemon is a
scripted stub over a real Unix socket; no runc/CRIU/zfs, no root."""
from __future__ import annotations

import contextlib
import http.client
import io
import json
import stat
import tempfile
import threading
import time
import unittest
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any
from unittest import mock

from crab.daemon.transport import DEFAULT_SOCKET_PERMS, DaemonClient, serve_unix_socket
from crab.gateway import server as gateway_server
from crab.gateway.cli import main as gateway_cli_main
from crab.gateway.registry import PENDING_ID_PREFIX, GatewayRegistry
from crab.gateway.server import (
    GATEWAY_INTENT_METADATA_KEY,
    GatewayServer,
    _DaemonUnreachable,
)


# ---------------------------------------------------------------------------
# Stub daemon — the real wire protocol (HTTP over a Unix socket) with a
# scripted in-memory sandbox table, so the gateway's DaemonClient path is
# exercised end to end.
# ---------------------------------------------------------------------------


class _StubDaemonState:
    def __init__(self, pid: int = 1000) -> None:
        self.pid = pid
        self.lock = threading.Lock()
        self.sandboxes: dict[str, dict[str, Any]] = {}
        self.counter = 0
        self.requests: list[tuple[str, str, dict[str, Any]]] = []
        self.stop_delay_s = 0.0
        # Runtime name reported by /info. Cloud-mode SDK tests set this to
        # "docker" so `Sandbox._launch` takes the metadata-only path instead
        # of the host-coupled runc bundle prep.
        self.runtime = "runc"

    def add_sandbox(self, sandbox_id: str, metadata: dict[str, Any] | None = None) -> None:
        with self.lock:
            self.sandboxes[sandbox_id] = dict(metadata or {})

    def create_bodies(self) -> list[dict[str, Any]]:
        with self.lock:
            return [body for method, path, body in self.requests if (method, path) == ("POST", "/sandboxes")]


def _build_stub_daemon_handler(state: _StubDaemonState):
    class Handler(BaseHTTPRequestHandler):
        server_version = "stub-crab-daemon/1"
        sys_version = ""

        def log_message(self, fmt: str, *args: Any) -> None:
            pass

        def do_GET(self) -> None:  # noqa: N802
            self._dispatch("GET")

        def do_POST(self) -> None:  # noqa: N802
            self._dispatch("POST")

        def do_DELETE(self) -> None:  # noqa: N802
            self._dispatch("DELETE")

        def _dispatch(self, method: str) -> None:
            path = self.path.split("?", 1)[0]
            body = self._read_body()
            with state.lock:
                state.requests.append((method, path, body))
            status, payload = self._route(method, path, body)
            self._send_json(status, payload)

        def _route(self, method: str, path: str, body: dict[str, Any]):
            parts = [p for p in path.strip("/").split("/") if p]
            if (method, path) == ("GET", "/healthz"):
                return HTTPStatus.OK, {"ok": True, "started": True}
            if (method, path) == ("GET", "/info"):
                with state.lock:
                    count = len(state.sandboxes)
                return HTTPStatus.OK, {
                    "ok": True,
                    "version": 1,
                    "pid": state.pid,
                    "runtime": state.runtime,
                    "default_image": "ubuntu:22.04",
                    "storage_root": "/secret/storage",
                    "runtime_root": "/secret/runtime",
                    "network_bridge_ip": "10.100.0.1",
                    "sandbox_count": count,
                }
            if (method, path) == ("GET", "/sandboxes"):
                with state.lock:
                    rows = [
                        {
                            "sandbox_id": sid,
                            "runtime_name": "runc",
                            "status": "running",
                            "metadata": dict(meta),
                        }
                        for sid, meta in state.sandboxes.items()
                    ]
                return HTTPStatus.OK, {"ok": True, "sandboxes": rows}
            if (method, path) == ("POST", "/sandboxes"):
                with state.lock:
                    state.counter += 1
                    sandbox_id = f"sb-{state.counter}"
                    state.sandboxes[sandbox_id] = dict(body.get("metadata") or {})
                return HTTPStatus.OK, {"ok": True, "sandbox_id": sandbox_id}
            if (method, path) == ("POST", "/runtime/write_bundle_spec"):
                return HTTPStatus.OK, {"ok": True}
            if len(parts) >= 2 and parts[0] == "sandboxes":
                sandbox_id = parts[1]
                with state.lock:
                    known = sandbox_id in state.sandboxes
                if not known:
                    return HTTPStatus.NOT_FOUND, {
                        "ok": False,
                        "error": f"unknown sandbox: {sandbox_id}",
                    }
                return self._sandbox_route(method, parts, sandbox_id, body)
            return HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"}

        def _sandbox_route(
            self, method: str, parts: list[str], sandbox_id: str, body: dict[str, Any]
        ):
            sub = parts[2:]
            if method == "GET" and not sub:
                with state.lock:
                    metadata = dict(state.sandboxes[sandbox_id])
                return HTTPStatus.OK, {
                    "ok": True,
                    "description": {
                        "sandbox_id": sandbox_id,
                        "runtime_name": "runc",
                        "status": "running",
                        "metadata": metadata,
                    },
                    "runtime_state": None,
                }
            if method == "DELETE" and not sub:
                with state.lock:
                    state.sandboxes.pop(sandbox_id, None)
                return HTTPStatus.OK, {"ok": True, "sandbox_id": sandbox_id}
            if method == "POST" and sub == ["exec"]:
                argv = list(body.get("argv") or [])
                if not argv:
                    return HTTPStatus.BAD_REQUEST, {
                        "ok": False,
                        "error": "exec requires non-empty argv",
                    }
                return HTTPStatus.OK, {
                    "ok": True,
                    "result": {
                        "args": argv,
                        "returncode": 0,
                        "stdout": f"ran {argv[0]}",
                        "stderr": "",
                    },
                }
            if method == "POST" and sub == ["action"]:
                exec_spec = body.get("exec") or {}
                argv = list(exec_spec.get("argv") or [])
                if not argv:
                    return HTTPStatus.BAD_REQUEST, {
                        "ok": False,
                        "error": "action exec requires non-empty argv",
                    }
                payload: dict[str, Any] = {
                    "ok": True,
                    "exec": {
                        "returncode": 0,
                        "stdout": f"ran {argv[0]}",
                        "stderr": "",
                    },
                }
                if body.get("checkpoint"):
                    payload["checkpoint_id"] = (
                        body.get("checkpoint_id") or "ckpt-action"
                    )
                if body.get("observe"):
                    payload["filesystem_changed"] = True
                    payload["process_changed"] = False
                return HTTPStatus.OK, payload
            if method == "POST" and sub == ["stop"]:
                # Optionally outlast the gateway's per-call timeout (the
                # 504 regression test dials the route timeout down).
                if state.stop_delay_s:
                    time.sleep(state.stop_delay_s)
                return HTTPStatus.OK, {"ok": True, "sandbox_id": sandbox_id, "stopped": True}
            if method == "POST" and sub == ["fork"]:
                count = int(body.get("count", 1))
                forks = []
                with state.lock:
                    for index in range(count):
                        fork_id = f"{sandbox_id}-fork-{index}"
                        state.sandboxes[fork_id] = {}
                        forks.append({"sandbox_id": fork_id})
                return HTTPStatus.OK, {"ok": True, "sandbox_id": sandbox_id, "forks": forks}
            if method == "GET" and sub == ["inspector"]:
                # Read-only inspector peek (feat/sdk-improvements). Does not
                # reset flags; the real daemon serves the same shape.
                return HTTPStatus.OK, {
                    "ok": True,
                    "sandbox_id": sandbox_id,
                    "filesystem_changed": True,
                    "process_changed": False,
                }
            if method == "GET" and sub == ["checkpoints"]:
                return HTTPStatus.OK, {"ok": True, "checkpoints": []}
            if method == "POST" and sub == ["checkpoints"]:
                # Honour client-preallocated checkpoint_id when present.
                client_id = body.get("checkpoint_id") or "ck-1"
                return HTTPStatus.OK, {
                    "ok": True,
                    "checkpoint_id": client_id,
                    "checkpoint": {"checkpoint_id": client_id},
                }
            if (
                method == "POST"
                and len(sub) == 3
                and sub[0] == "checkpoints"
                and sub[2] == "restore"
            ):
                return HTTPStatus.OK, {"ok": True, "status": "succeeded"}
            if method == "DELETE" and len(sub) == 2 and sub[0] == "checkpoints":
                return HTTPStatus.OK, {"ok": True, "deleted": [sub[1]]}
            if method == "POST" and sub == ["processes", "merge"]:
                return HTTPStatus.OK, {"ok": True, "report": {
                    "source_sandbox_id": sandbox_id,
                    "fork_sandbox_id": body.get("fork_sandbox_id", "fork-1"),
                    "strategy": "promote",
                    "source_processes": 1,
                    "replayed": [],
                    "deviations": 0,
                    "stopped_early": False,
                }}
            if method == "POST" and sub == ["upstream"]:
                return HTTPStatus.OK, {"ok": True}
            if method == "DELETE" and sub == ["upstream"]:
                return HTTPStatus.OK, {"ok": True}
            if method == "POST" and sub == ["network", "lease"]:
                return HTTPStatus.OK, {"ok": True, "lease": {
                    "namespace_path": "/run/netns/test",
                    "guest_ip": "10.100.0.2",
                    "bridge_ip": "10.100.0.1",
                }}
            if method == "DELETE" and sub == ["network", "lease"]:
                return HTTPStatus.OK, {"ok": True}
            if method == "POST" and sub == ["host_inspector", "filters"]:
                return HTTPStatus.OK, {"ok": True}
            return HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"}

        def _read_body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0:
                return {}
            raw = self.rfile.read(length)
            return json.loads(raw) if raw else {}

        def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            try:
                self.send_response(int(status))
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                # The gateway gave up on us (per-call timeout) — fine.
                pass

    return Handler


# ---------------------------------------------------------------------------
# Test base — one stub daemon + one gateway per test.
# ---------------------------------------------------------------------------


class GatewayTestBase(unittest.TestCase):
    daemon_pid = 1000

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name)
        self.state = _StubDaemonState(pid=self.daemon_pid)
        self.daemon_socket = self.base / "daemon.sock"
        self._start_stub_daemon()

    def _start_stub_daemon(self) -> None:
        self.daemon_server = serve_unix_socket(
            self.daemon_socket, _build_stub_daemon_handler(self.state)
        )
        thread = threading.Thread(target=self.daemon_server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(self._stop_stub_daemon)

    def _stop_stub_daemon(self) -> None:
        if self.daemon_server is not None:
            self.daemon_server.shutdown()
            self.daemon_server.server_close()
            self.daemon_server = None
        if self.daemon_socket.exists():
            self.daemon_socket.unlink()

    def start_gateway(self) -> None:
        self.gateway = GatewayServer(
            data_dir=self.base / "gw",
            daemon_socket=self.daemon_socket,
            host="127.0.0.1",
            port=0,
            admin_socket_path=self.base / "admin.sock",
        )
        self.gateway.start()
        self.addCleanup(self.gateway.stop)
        self.admin = DaemonClient(self.base / "admin.sock")

    def make_tenant(
        self,
        name: str,
        max_sandboxes: int | None = None,
        quotas: dict[str, Any] | None = None,
    ):
        body: dict[str, Any] = {"name": name}
        merged: dict[str, Any] = dict(quotas or {})
        if max_sandboxes is not None:
            merged["max_sandboxes"] = max_sandboxes
        if merged:
            body["quotas"] = merged
        tenant = self.admin.post_json("/admin/tenants", body)["tenant"]
        key = self.admin.post_json("/admin/keys", {"tenant_id": tenant["id"]})["api_key"]
        return tenant, key

    def request(
        self,
        method: str,
        path: str,
        api_key: str | None = None,
        body: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        conn = http.client.HTTPConnection("127.0.0.1", self.gateway.port, timeout=10)
        headers: dict[str, str] = {}
        if api_key is not None:
            headers["Authorization"] = f"Bearer {api_key}"
        payload = None
        if body is not None:
            payload = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        try:
            conn.request(method, path, body=payload, headers=headers)
            response = conn.getresponse()
            raw = response.read()
            status = response.status
        finally:
            conn.close()
        return status, (json.loads(raw) if raw else {})


class GatewayLiveTestBase(GatewayTestBase):
    def setUp(self) -> None:
        super().setUp()
        self.start_gateway()


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


class AuthTests(GatewayLiveTestBase):
    def test_healthz_is_open(self) -> None:
        for path in ("/healthz", "/v1/healthz"):
            status, payload = self.request("GET", path)
            self.assertEqual(status, 200)
            self.assertTrue(payload["ok"])

    def test_missing_key_is_401(self) -> None:
        status, payload = self.request("GET", "/v1/sandboxes")
        self.assertEqual(status, 401)
        self.assertEqual(payload["error_type"], "unauthorized")

    def test_unknown_key_is_401(self) -> None:
        status, _ = self.request("GET", "/v1/sandboxes", api_key="crab_sk_" + "0" * 48)
        self.assertEqual(status, 401)

    def test_revoked_key_is_401(self) -> None:
        _tenant, key = self.make_tenant("acme")
        status, _ = self.request("GET", "/v1/sandboxes", api_key=key)
        self.assertEqual(status, 200)
        self.admin.post_json("/admin/keys/revoke", {"key": key})
        status, payload = self.request("GET", "/v1/sandboxes", api_key=key)
        self.assertEqual(status, 401)
        self.assertEqual(payload["error_type"], "unauthorized")


# ---------------------------------------------------------------------------
# Tenancy and ownership
# ---------------------------------------------------------------------------


class TenancyTests(GatewayLiveTestBase):
    def setUp(self) -> None:
        super().setUp()
        self.tenant_a, self.key_a = self.make_tenant("tenant-a")
        self.tenant_b, self.key_b = self.make_tenant("tenant-b")

    def _create(self, key: str) -> str:
        status, payload = self.request("POST", "/v1/sandboxes", api_key=key, body={})
        self.assertEqual(status, 200)
        return payload["sandbox_id"]

    def test_cross_tenant_is_404_not_403(self) -> None:
        sandbox_id = self._create(self.key_a)
        for method, path, body in (
            ("GET", f"/v1/sandboxes/{sandbox_id}", None),
            ("DELETE", f"/v1/sandboxes/{sandbox_id}", None),
            ("POST", f"/v1/sandboxes/{sandbox_id}/exec", {"argv": ["true"]}),
            ("POST", f"/v1/sandboxes/{sandbox_id}/fork", {}),
        ):
            status, payload = self.request(method, path, api_key=self.key_b, body=body)
            self.assertEqual(status, 404, f"{method} {path}")
            self.assertIn("unknown sandbox", payload["error"])
        # The owner still sees it.
        status, _ = self.request("GET", f"/v1/sandboxes/{sandbox_id}", api_key=self.key_a)
        self.assertEqual(status, 200)

    def test_list_is_tenant_filtered(self) -> None:
        sandbox_id = self._create(self.key_a)
        status, payload = self.request("GET", "/v1/sandboxes", api_key=self.key_a)
        self.assertEqual(status, 200)
        self.assertEqual([row["sandbox_id"] for row in payload["sandboxes"]], [sandbox_id])
        status, payload = self.request("GET", "/v1/sandboxes", api_key=self.key_b)
        self.assertEqual(status, 200)
        self.assertEqual(payload["sandboxes"], [])

    def test_fork_children_inherit_tenant(self) -> None:
        sandbox_id = self._create(self.key_a)
        status, payload = self.request(
            "POST", f"/v1/sandboxes/{sandbox_id}/fork", api_key=self.key_a, body={"count": 2}
        )
        self.assertEqual(status, 200)
        fork_ids = [fork["sandbox_id"] for fork in payload["forks"]]
        self.assertEqual(len(fork_ids), 2)
        status, listing = self.request("GET", "/v1/sandboxes", api_key=self.key_a)
        listed = {row["sandbox_id"] for row in listing["sandboxes"]}
        self.assertEqual(listed, {sandbox_id, *fork_ids})
        # And they are invisible to the other tenant.
        status, _ = self.request(
            "GET", f"/v1/sandboxes/{fork_ids[0]}", api_key=self.key_b
        )
        self.assertEqual(status, 404)

    def test_kill_flips_status_then_404(self) -> None:
        sandbox_id = self._create(self.key_a)
        status, payload = self.request(
            "DELETE", f"/v1/sandboxes/{sandbox_id}", api_key=self.key_a
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["sandbox_id"], sandbox_id)
        self.assertEqual(self.gateway.registry.get_sandbox(sandbox_id)["status"], "killed")
        status, _ = self.request("GET", f"/v1/sandboxes/{sandbox_id}", api_key=self.key_a)
        self.assertEqual(status, 404)


# ---------------------------------------------------------------------------
# Quota gate
# ---------------------------------------------------------------------------


class QuotaTests(GatewayLiveTestBase):
    def setUp(self) -> None:
        super().setUp()
        self.tenant, self.key = self.make_tenant("capped", max_sandboxes=2)

    def test_create_quota_409_with_arithmetic(self) -> None:
        for _ in range(2):
            status, _ = self.request("POST", "/v1/sandboxes", api_key=self.key, body={})
            self.assertEqual(status, 200)
        status, payload = self.request("POST", "/v1/sandboxes", api_key=self.key, body={})
        self.assertEqual(status, 409)
        self.assertEqual(payload["error_type"], "quota_exceeded")
        self.assertEqual(payload["quota"]["max_sandboxes"], 2)
        self.assertEqual(payload["quota"]["live_sandboxes"], 2)

    def test_kill_frees_quota(self) -> None:
        status, payload = self.request("POST", "/v1/sandboxes", api_key=self.key, body={})
        sandbox_id = payload["sandbox_id"]
        status, _ = self.request("POST", "/v1/sandboxes", api_key=self.key, body={})
        self.assertEqual(status, 200)
        self.request("DELETE", f"/v1/sandboxes/{sandbox_id}", api_key=self.key)
        status, _ = self.request("POST", "/v1/sandboxes", api_key=self.key, body={})
        self.assertEqual(status, 200)

    def test_fork_hits_the_same_gate(self) -> None:
        status, payload = self.request("POST", "/v1/sandboxes", api_key=self.key, body={})
        sandbox_id = payload["sandbox_id"]
        status, payload = self.request(
            "POST", f"/v1/sandboxes/{sandbox_id}/fork", api_key=self.key, body={"count": 2}
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload["error_type"], "quota_exceeded")
        status, _ = self.request(
            "POST", f"/v1/sandboxes/{sandbox_id}/fork", api_key=self.key, body={"count": 1}
        )
        self.assertEqual(status, 200)

    def test_concurrent_creates_respect_quota(self) -> None:
        tenant, key = self.make_tenant("burst", max_sandboxes=3)
        statuses: list[int] = []
        lock = threading.Lock()

        def worker() -> None:
            status, _ = self.request("POST", "/v1/sandboxes", api_key=key, body={})
            with lock:
                statuses.append(status)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(statuses.count(200), 3)
        self.assertEqual(statuses.count(409), 5)
        self.assertEqual(self.gateway.registry.live_count(tenant["id"]), 3)


# ---------------------------------------------------------------------------
# Aggregate resource quota gate (S3) — claims ride create metadata
# ---------------------------------------------------------------------------


MIB = 1024 * 1024


class AggregateQuotaHttpTests(GatewayLiveTestBase):
    def setUp(self) -> None:
        super().setUp()
        self.tenant, self.key = self.make_tenant(
            "metered", quotas={"max_memory_bytes": 1024 * MIB, "max_cpu": 4}
        )

    def _create(self, claim: dict[str, Any] | None) -> tuple[int, dict[str, Any]]:
        body: dict[str, Any] = {}
        if claim is not None:
            body["metadata"] = {"resources": claim}
        return self.request("POST", "/v1/sandboxes", api_key=self.key, body=body)

    def test_over_cap_create_is_409_with_arithmetic(self) -> None:
        status, _ = self._create({"memory_bytes": 768 * MIB, "cpus": 2})
        self.assertEqual(status, 200)
        status, payload = self._create({"memory_bytes": 512 * MIB, "cpus": 1})
        self.assertEqual(status, 409)
        self.assertEqual(payload["error_type"], "quota_exceeded")
        self.assertEqual(
            payload["quota"],
            {
                "max_memory_bytes": 1024 * MIB,
                "live_memory_bytes": 768 * MIB,
                "requested_memory_bytes": 512 * MIB,
            },
        )

    def test_undeclared_limits_on_capped_tenant_are_409(self) -> None:
        status, payload = self._create(None)
        self.assertEqual(status, 409)
        self.assertEqual(payload["error_type"], "quota_exceeded")
        self.assertIsNone(payload["quota"]["requested_memory_bytes"])

    def test_malformed_resources_are_400_not_500(self) -> None:
        for bad in ({"memory": "512M"}, {"cpus": "two"}, ["cpus"]):
            status, payload = self._create(bad)  # type: ignore[arg-type]
            self.assertEqual(status, 400, repr(bad))
            self.assertIn("invalid resources", payload["error"])

    def test_kill_releases_aggregate_quota(self) -> None:
        claim = {"memory_bytes": 768 * MIB, "cpus": 2}
        status, payload = self._create(claim)
        self.assertEqual(status, 200)
        sandbox_id = payload["sandbox_id"]
        status, _ = self._create(claim)
        self.assertEqual(status, 409)
        self.request("DELETE", f"/v1/sandboxes/{sandbox_id}", api_key=self.key)
        status, _ = self._create(claim)
        self.assertEqual(status, 200)

    def test_fork_children_inherit_claim_and_count(self) -> None:
        claim = {"memory_bytes": 256 * MIB, "cpus": 1}
        status, payload = self._create(claim)
        self.assertEqual(status, 200)
        sandbox_id = payload["sandbox_id"]
        # Three children fit exactly (256M source + 3*256M == 1024M cap);
        # four do not (256M + 4*256M > 1024M).
        status, payload = self.request(
            "POST", f"/v1/sandboxes/{sandbox_id}/fork", api_key=self.key, body={"count": 4}
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload["quota"]["requested_memory_bytes"], 1024 * MIB)
        status, payload = self.request(
            "POST", f"/v1/sandboxes/{sandbox_id}/fork", api_key=self.key, body={"count": 2}
        )
        self.assertEqual(status, 200)
        for fork in payload["forks"]:
            row = self.gateway.registry.get_sandbox(fork["sandbox_id"])
            self.assertEqual(row["resources"], claim)

    def test_uncapped_tenant_is_unaffected(self) -> None:
        # Zero-breakage: without aggregate caps the gate never engages.
        _tenant, key = self.make_tenant("free")
        for body in ({}, {"metadata": {"resources": {"memory_bytes": 1 << 60}}}):
            status, _ = self.request("POST", "/v1/sandboxes", api_key=key, body=body)
            self.assertEqual(status, 200)


# ---------------------------------------------------------------------------
# Facade behavior — passthrough, redaction, blocked routes, error relay
# ---------------------------------------------------------------------------


class FacadeTests(GatewayLiveTestBase):
    def setUp(self) -> None:
        super().setUp()
        self.tenant, self.key = self.make_tenant("acme")

    def test_create_injects_intent_metadata(self) -> None:
        status, payload = self.request(
            "POST", "/v1/sandboxes", api_key=self.key, body={"metadata": {"label": "x"}}
        )
        self.assertEqual(status, 200)
        bodies = self.state.create_bodies()
        self.assertEqual(len(bodies), 1)
        metadata = bodies[0]["metadata"]
        self.assertEqual(metadata["label"], "x")
        self.assertTrue(metadata[GATEWAY_INTENT_METADATA_KEY].startswith(PENDING_ID_PREFIX))
        # Two-phase: the pending row was rekeyed to the daemon-assigned id.
        row = self.gateway.registry.get_sandbox(payload["sandbox_id"])
        self.assertEqual(row["status"], "active")
        self.assertEqual(self.gateway.registry.pending_rows(), [])

    def test_create_rejects_invalid_idle_action_without_calling_daemon(self) -> None:
        status, payload = self.request(
            "POST",
            "/v1/sandboxes",
            api_key=self.key,
            body={
                "metadata": {
                    "idle_timeout": 60,
                    "idle_action": "archive",
                }
            },
        )

        self.assertEqual(status, 400)
        self.assertIn("idle_action", payload["error"])
        self.assertEqual(self.state.create_bodies(), [])
        self.assertEqual(self.gateway.registry.pending_rows(), [])

    def test_exec_passthrough_result(self) -> None:
        _, payload = self.request("POST", "/v1/sandboxes", api_key=self.key, body={})
        sandbox_id = payload["sandbox_id"]
        status, payload = self.request(
            "POST",
            f"/v1/sandboxes/{sandbox_id}/exec",
            api_key=self.key,
            body={"argv": ["echo", "hi"]},
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["result"]["stdout"], "ran echo")

    def test_fork_forwards_the_checkpoint_fork_point(self) -> None:
        # The gateway only reads `count` (for quota); the rest of the fork
        # body has to reach the daemon untouched.
        _, payload = self.request("POST", "/v1/sandboxes", api_key=self.key, body={})
        sandbox_id = payload["sandbox_id"]
        status, _ = self.request(
            "POST",
            f"/v1/sandboxes/{sandbox_id}/fork",
            api_key=self.key,
            body={"count": 1, "checkpoint_id": "ckpt-7"},
        )
        self.assertEqual(status, 200)
        forwarded = [
            body
            for method, path, body in self.state.requests
            if (method, path) == ("POST", f"/sandboxes/{sandbox_id}/fork")
        ]
        self.assertEqual(forwarded, [{"count": 1, "checkpoint_id": "ckpt-7"}])

    def test_daemon_error_relayed_verbatim(self) -> None:
        _, payload = self.request("POST", "/v1/sandboxes", api_key=self.key, body={})
        sandbox_id = payload["sandbox_id"]
        status, payload = self.request(
            "POST", f"/v1/sandboxes/{sandbox_id}/exec", api_key=self.key, body={"argv": []}
        )
        self.assertEqual(status, 400)
        # The daemon's exact error body, not a gateway rewrite.
        self.assertEqual(payload, {"ok": False, "error": "exec requires non-empty argv"})

    def test_delete_body_is_forwarded(self) -> None:
        # `delete_checkpoint(cascade=True)` rides a DELETE with a JSON
        # body; the passthrough must relay it, not silently drop it.
        _, payload = self.request("POST", "/v1/sandboxes", api_key=self.key, body={})
        sandbox_id = payload["sandbox_id"]
        status, _ = self.request(
            "DELETE",
            f"/v1/sandboxes/{sandbox_id}/checkpoints/ck-1",
            api_key=self.key,
            body={"cascade": True},
        )
        self.assertEqual(status, 200)
        with self.state.lock:
            seen = [
                b
                for m, p, b in self.state.requests
                if (m, p) == ("DELETE", f"/sandboxes/{sandbox_id}/checkpoints/ck-1")
            ]
        self.assertEqual(seen, [{"cascade": True}])

    def test_info_is_redacted_and_tenant_scoped(self) -> None:
        self.request("POST", "/v1/sandboxes", api_key=self.key, body={})
        status, payload = self.request("GET", "/v1/info", api_key=self.key)
        self.assertEqual(status, 200)
        for leaked in ("storage_root", "runtime_root", "network_bridge_ip", "pid"):
            self.assertNotIn(leaked, payload)
        self.assertEqual(payload["runtime"], "runc")
        self.assertEqual(payload["sandbox_count"], 1)
        # A second tenant sees its own count, not the host's.
        _tenant_b, key_b = self.make_tenant("other")
        _, payload = self.request("GET", "/v1/info", api_key=key_b)
        self.assertEqual(payload["sandbox_count"], 0)

    def test_shutdown_route_is_not_exposed(self) -> None:
        # Only /shutdown remains blocked (operator-only).
        status, _ = self.request("POST", "/v1/shutdown", api_key=self.key, body={})
        self.assertEqual(status, 404, "POST /v1/shutdown must not be exposed")

    def test_previously_hidden_routes_are_now_exposed(self) -> None:
        """S5 full-access: all per-sandbox host-coupled routes + runtime
        are now proxied through the gateway."""
        _, payload = self.request("POST", "/v1/sandboxes", api_key=self.key, body={})
        sandbox_id = payload["sandbox_id"]
        for method, path in (
            ("POST", "/v1/runtime/write_bundle_spec"),
            ("POST", f"/v1/sandboxes/{sandbox_id}/processes/merge"),
            ("POST", f"/v1/sandboxes/{sandbox_id}/upstream"),
            ("POST", f"/v1/sandboxes/{sandbox_id}/network/lease"),
            ("POST", f"/v1/sandboxes/{sandbox_id}/host_inspector/filters"),
        ):
            status, _ = self.request(method, path, api_key=self.key, body={})
            self.assertEqual(status, 200, f"{method} {path} should now be exposed")

    def test_daemon_unreachable_is_502(self) -> None:
        self._stop_stub_daemon()
        status, payload = self.request("GET", "/v1/sandboxes", api_key=self.key)
        self.assertEqual(status, 502)
        self.assertEqual(payload["error_type"], "daemon_unreachable")


# ---------------------------------------------------------------------------
# Timeout mapping (PR #29 review regression) — a daemon call that exceeds
# its per-route timeout must surface as 504 daemon_timeout, not fall into
# the generic 500 handler. On Python >= 3.10 `socket.timeout` *is*
# `TimeoutError` (bpo-42413), which `GatewayServer.proxy` catches; this
# test pins that empirically instead of by argument.
# ---------------------------------------------------------------------------


class TimeoutTests(GatewayTestBase):
    def test_daemon_timeout_maps_to_504(self) -> None:
        # Dial the /stop route's per-call timeout down so the test does
        # not sit through the real 30s fast tier. The route table bakes
        # timeouts in at start(), so patch before starting the gateway.
        short_routes = [
            (method, subpath, 0.3 if subpath == "/stop" else timeout)
            for method, subpath, timeout in gateway_server._PASSTHROUGH_SANDBOX_ROUTES
        ]
        with mock.patch.object(
            gateway_server, "_PASSTHROUGH_SANDBOX_ROUTES", short_routes
        ):
            self.start_gateway()
        _tenant, key = self.make_tenant("acme")
        status, payload = self.request("POST", "/v1/sandboxes", api_key=key, body={})
        self.assertEqual(status, 200)
        sandbox_id = payload["sandbox_id"]

        self.state.stop_delay_s = 1.5  # well past the 0.3s route timeout
        status, payload = self.request(
            "POST", f"/v1/sandboxes/{sandbox_id}/stop", api_key=key, body={}
        )
        self.assertEqual(status, 504)
        self.assertEqual(payload["error_type"], "daemon_timeout")
        self.assertIn("timed out", payload["error"])

        # The gateway is still healthy afterwards — the timeout was
        # per-call, not a wedged worker thread.
        self.state.stop_delay_s = 0.0
        status, _ = self.request("GET", f"/v1/sandboxes/{sandbox_id}", api_key=key)
        self.assertEqual(status, 200)


# ---------------------------------------------------------------------------
# Startup reconciliation + boot identity
# ---------------------------------------------------------------------------


class ReconciliationTests(GatewayTestBase):
    """These tests seed the registry file before the gateway starts, so
    setUp does not start a gateway."""

    def _seed_registry(self):
        registry = GatewayRegistry(self.base / "gw" / "gateway.sqlite3")
        tenant = registry.create_tenant("acme")
        key = registry.create_api_key(tenant["id"])["api_key"]
        return registry, tenant, key

    def test_pending_row_matched_by_intent_flips_active(self) -> None:
        registry, tenant, key = self._seed_registry()
        intent = registry.begin_create(tenant["id"])
        registry.set_meta("daemon_boot_id", str(self.daemon_pid))
        registry.close()
        self.state.add_sandbox("sb-recovered", {GATEWAY_INTENT_METADATA_KEY: intent})

        self.start_gateway()
        row = self.gateway.registry.get_sandbox("sb-recovered")
        self.assertIsNotNone(row)
        self.assertEqual(row["status"], "active")
        self.assertEqual(self.gateway.registry.pending_rows(), [])
        status, _ = self.request("GET", "/v1/sandboxes/sb-recovered", api_key=key)
        self.assertEqual(status, 200)

    def test_unmatched_pending_row_is_reaped(self) -> None:
        registry, tenant, _key = self._seed_registry()
        intent = registry.begin_create(tenant["id"])
        registry.close()

        self.start_gateway()
        self.assertEqual(self.gateway.registry.pending_rows(), [])
        self.assertIsNone(self.gateway.registry.get_sandbox(intent))
        self.assertEqual(self.gateway.registry.live_count(tenant["id"]), 0)

    def test_boot_identity_mismatch_marks_rows_lost_410(self) -> None:
        registry, tenant, key = self._seed_registry()
        registry.register_sandbox(tenant["id"], "sb-old")
        registry.set_meta("daemon_boot_id", str(self.daemon_pid + 1))  # older boot
        registry.close()

        self.start_gateway()
        self.assertEqual(self.gateway.registry.get_sandbox("sb-old")["status"], "lost")
        status, payload = self.request("GET", "/v1/sandboxes/sb-old", api_key=key)
        self.assertEqual(status, 410)
        self.assertEqual(payload["error_type"], "sandbox_lost")
        # Lost sandboxes don't count against quota; creates still work.
        status, _ = self.request("POST", "/v1/sandboxes", api_key=key, body={})
        self.assertEqual(status, 200)

    def test_active_row_missing_from_daemon_marked_lost(self) -> None:
        registry, tenant, key = self._seed_registry()
        registry.register_sandbox(tenant["id"], "sb-vanished")
        registry.set_meta("daemon_boot_id", str(self.daemon_pid))  # same boot
        registry.close()

        self.start_gateway()
        status, payload = self.request("GET", "/v1/sandboxes/sb-vanished", api_key=key)
        self.assertEqual(status, 410)
        self.assertEqual(payload["error_type"], "sandbox_lost")

    def test_start_fails_fast_when_daemon_unreachable(self) -> None:
        self._stop_stub_daemon()
        gateway = GatewayServer(
            data_dir=self.base / "gw",
            daemon_socket=self.daemon_socket,
            host="127.0.0.1",
            port=0,
            admin_socket_path=self.base / "admin.sock",
        )
        with self.assertRaises(_DaemonUnreachable):
            gateway.start()


# ---------------------------------------------------------------------------
# Admin CLI (over the gateway's own Unix socket)
# ---------------------------------------------------------------------------


class AdminCliTests(GatewayLiveTestBase):
    def _run_cli(self, *argv: str) -> tuple[int, dict[str, Any]]:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = gateway_cli_main([*argv, "--admin-socket", str(self.base / "admin.sock")])
        raw = stdout.getvalue()
        return code, (json.loads(raw) if raw else {})

    def test_tenants_keys_quotas_lifecycle(self) -> None:
        code, created = self._run_cli("tenants", "create", "acme", "--max-sandboxes", "1")
        self.assertEqual(code, 0)
        tenant_id = created["tenant"]["id"]
        self.assertEqual(created["tenant"]["quotas"], {"max_sandboxes": 1})

        code, listing = self._run_cli("tenants", "list")
        self.assertEqual(code, 0)
        self.assertEqual([t["name"] for t in listing["tenants"]], ["acme"])

        code, key_payload = self._run_cli("keys", "create", "--tenant", tenant_id)
        self.assertEqual(code, 0)
        key = key_payload["api_key"]
        status, _ = self.request("GET", "/v1/sandboxes", api_key=key)
        self.assertEqual(status, 200)

        code, quota_payload = self._run_cli(
            "quotas", "set", "--tenant", tenant_id, "--max-sandboxes", "5"
        )
        self.assertEqual(code, 0)
        self.assertEqual(quota_payload["tenant"]["quotas"], {"max_sandboxes": 5})

        code, revoke_payload = self._run_cli("keys", "revoke", key)
        self.assertEqual(code, 0)
        self.assertTrue(revoke_payload["revoked"])
        status, _ = self.request("GET", "/v1/sandboxes", api_key=key)
        self.assertEqual(status, 401)


# ---------------------------------------------------------------------------
# Transport perms/group override (the daemon-side S1 change)
# ---------------------------------------------------------------------------


class TransportSocketPermsTests(unittest.TestCase):
    def _spawn(self, socket_path: Path, **kwargs):
        handler = _build_stub_daemon_handler(_StubDaemonState())
        server = serve_unix_socket(socket_path, handler, **kwargs)
        self.addCleanup(server.server_close)
        return server

    def test_default_perms_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            socket_path = Path(tmp) / "crab.sock"
            self._spawn(socket_path)
            mode = stat.S_IMODE(socket_path.stat().st_mode)
            self.assertEqual(mode, DEFAULT_SOCKET_PERMS)
            self.assertEqual(DEFAULT_SOCKET_PERMS, 0o600)

    def test_perms_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            socket_path = Path(tmp) / "crab.sock"
            self._spawn(socket_path, socket_perms=0o660)
            mode = stat.S_IMODE(socket_path.stat().st_mode)
            self.assertEqual(mode, 0o660)

    def test_group_override(self) -> None:
        import grp
        import os

        # chgrp to a group the test user already belongs to — no root needed.
        gid = os.getgid()
        group_name = grp.getgrgid(gid).gr_name
        with tempfile.TemporaryDirectory() as tmp:
            socket_path = Path(tmp) / "crab.sock"
            self._spawn(socket_path, socket_perms=0o660, socket_group=group_name)
            self.assertEqual(socket_path.stat().st_gid, gid)
            self.assertEqual(stat.S_IMODE(socket_path.stat().st_mode), 0o660)


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# Sandboxes adopt (admin plane)
# ---------------------------------------------------------------------------


class AdoptSandboxTests(GatewayLiveTestBase):
    """Tests for the `POST /admin/sandboxes/adopt` admin route."""

    def test_adopt_single(self) -> None:
        tenant, _key = self.make_tenant("acme")
        # The daemon has a sandbox the gateway doesn't know about
        self.state.add_sandbox("sbx-orphan1")
        result = self.admin.post_json(
            "/admin/sandboxes/adopt",
            {"tenant": "acme", "sandbox_ids": ["sbx-orphan1"]},
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["adopted"], ["sbx-orphan1"])
        self.assertEqual(result["skipped"], [])
        # Verify it's now in the registry
        sb = self.gateway.registry.get_sandbox("sbx-orphan1")
        self.assertIsNotNone(sb)
        self.assertEqual(sb["tenant_id"], tenant["id"])
        self.assertEqual(sb["status"], "active")

    def test_adopt_multiple(self) -> None:
        tenant, _key = self.make_tenant("multi")
        self.state.add_sandbox("sbx-a")
        self.state.add_sandbox("sbx-b")
        result = self.admin.post_json(
            "/admin/sandboxes/adopt",
            {"tenant": tenant["id"], "sandbox_ids": ["sbx-a", "sbx-b"]},
        )
        self.assertTrue(result["ok"])
        self.assertCountEqual(result["adopted"], ["sbx-a", "sbx-b"])
        self.assertEqual(result["skipped"], [])

    def test_adopt_already_registered_is_skipped(self) -> None:
        tenant, _key = self.make_tenant("skip")
        self.state.add_sandbox("sbx-dup")
        # First adopt
        self.admin.post_json(
            "/admin/sandboxes/adopt",
            {"tenant": "skip", "sandbox_ids": ["sbx-dup"]},
        )
        # Second adopt of same id → skipped
        result = self.admin.post_json(
            "/admin/sandboxes/adopt",
            {"tenant": "skip", "sandbox_ids": ["sbx-dup", "sbx-new"]},
        )
        self.assertTrue(result["ok"])
        self.assertIn("sbx-dup", result["skipped"])
        self.assertIn("sbx-new", result["adopted"])

    def test_adopt_unknown_tenant_is_404(self) -> None:
        from crab.daemon.transport import DaemonRequestError
        with self.assertRaises(DaemonRequestError) as ctx:
            self.admin.post_json(
                "/admin/sandboxes/adopt",
                {"tenant": "nonexistent", "sandbox_ids": ["sbx-x"]},
            )
        self.assertIn("unknown tenant", str(ctx.exception))

    def test_adopt_by_tenant_name(self) -> None:
        tenant, _key = self.make_tenant("byname")
        result = self.admin.post_json(
            "/admin/sandboxes/adopt",
            {"tenant": "byname", "sandbox_ids": ["sbx-named"]},
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["adopted"], ["sbx-named"])

    def test_adopt_sandbox_not_in_daemon_still_works(self) -> None:
        """Gateway adopt is registry-only; daemon presence is not required."""
        self.make_tenant("lax")
        # sbx-ghost is NOT added to self.state (daemon doesn't know it)
        result = self.admin.post_json(
            "/admin/sandboxes/adopt",
            {"tenant": "lax", "sandbox_ids": ["sbx-ghost"]},
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["adopted"], ["sbx-ghost"])

    def test_adopt_with_resources(self) -> None:
        tenant, _key = self.make_tenant("res")
        result = self.admin.post_json(
            "/admin/sandboxes/adopt",
            {
                "tenant": "res",
                "sandbox_ids": ["sbx-r1"],
                "resources": {"memory_bytes": 536870912},
            },
        )
        self.assertTrue(result["ok"])
        sb = self.gateway.registry.get_sandbox("sbx-r1")
        self.assertEqual(sb["resources"], {"memory_bytes": 536870912})

    def test_adopt_empty_ids_is_bad_request(self) -> None:
        self.make_tenant("empty")
        from crab.daemon.transport import DaemonRequestError
        with self.assertRaises(DaemonRequestError):
            self.admin.post_json(
                "/admin/sandboxes/adopt",
                {"tenant": "empty", "sandbox_ids": []},
            )


# ---------------------------------------------------------------------------
# Periodic reconciliation (S5)
# ---------------------------------------------------------------------------


class PeriodicReconciliationTests(GatewayTestBase):
    """Tests for _periodic_reconcile(): lightweight daemon→registry sync."""

    def test_periodic_marks_missing_sandbox_lost(self) -> None:
        # Seed daemon with sbx-1 and sbx-2; registry has sbx-1, sbx-2, sbx-3.
        self.state.add_sandbox("sbx-1")
        self.state.add_sandbox("sbx-2")
        self.start_gateway()
        tenant, key = self.make_tenant("acme")
        self.gateway.registry.register_sandbox(tenant["id"], "sbx-1")
        self.gateway.registry.register_sandbox(tenant["id"], "sbx-2")
        self.gateway.registry.register_sandbox(tenant["id"], "sbx-3")

        self.gateway._periodic_reconcile()

        self.assertEqual(self.gateway.registry.get_sandbox("sbx-1")["status"], "active")
        self.assertEqual(self.gateway.registry.get_sandbox("sbx-2")["status"], "active")
        self.assertEqual(self.gateway.registry.get_sandbox("sbx-3")["status"], "lost")

    def test_periodic_releases_ports_of_lost_sandbox(self) -> None:
        self.state.add_sandbox("sbx-1")
        self.start_gateway()
        tenant, key = self.make_tenant("acme")
        self.gateway.registry.register_sandbox(tenant["id"], "sbx-1")
        self.gateway.registry.register_sandbox(tenant["id"], "sbx-gone")

        # Allocate a port for sbx-gone
        import socket
        echo_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        echo_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        echo_sock.bind(("127.0.0.1", 0))
        echo_sock.listen(1)
        echo_port = echo_sock.getsockname()[1]
        self.addCleanup(echo_sock.close)

        host_port = self.gateway.port_manager.allocate("sbx-gone", "127.0.0.1", echo_port)
        self.gateway.registry.allocate_port(
            "sbx-gone",
            tenant["id"],
            echo_port,
            host_port,
            guest_ip="127.0.0.1",
        )

        self.gateway._periodic_reconcile()

        # sbx-gone marked lost
        self.assertEqual(self.gateway.registry.get_sandbox("sbx-gone")["status"], "lost")
        # Port released from registry
        self.assertEqual(self.gateway.registry.list_ports("sbx-gone"), [])
        # Forwarder stopped — connecting should fail
        time.sleep(0.2)
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.settimeout(1.0)
        with self.assertRaises(OSError):
            client.connect(("127.0.0.1", host_port))
        client.close()

    def test_daemon_unreachable_skips_safely(self) -> None:
        self.start_gateway()
        tenant, key = self.make_tenant("acme")
        self.gateway.registry.register_sandbox(tenant["id"], "sbx-1")

        # Kill the daemon
        self._stop_stub_daemon()

        # Should not raise and should not mark anything lost
        self.gateway._periodic_reconcile()

        self.assertEqual(self.gateway.registry.get_sandbox("sbx-1")["status"], "active")

    def test_healthy_sandboxes_unaffected(self) -> None:
        self.state.add_sandbox("sbx-1")
        self.state.add_sandbox("sbx-2")
        self.start_gateway()
        tenant, key = self.make_tenant("acme")
        self.gateway.registry.register_sandbox(tenant["id"], "sbx-1")
        self.gateway.registry.register_sandbox(tenant["id"], "sbx-2")

        self.gateway._periodic_reconcile()

        self.assertEqual(self.gateway.registry.get_sandbox("sbx-1")["status"], "active")
        self.assertEqual(self.gateway.registry.get_sandbox("sbx-2")["status"], "active")

    def test_serve_forever_triggers_at_interval(self) -> None:
        self.state.add_sandbox("sbx-1")
        self.start_gateway()
        tenant, key = self.make_tenant("acme")
        self.gateway.registry.register_sandbox(tenant["id"], "sbx-1")
        self.gateway.registry.register_sandbox(tenant["id"], "sbx-vanish")

        # Set very short reconcile interval
        self.gateway._reconcile_interval = 1.0

        # Patch _periodic_reconcile to count calls
        call_count = [0]
        orig = self.gateway._periodic_reconcile

        def counting_reconcile():
            orig()
            call_count[0] += 1

        self.gateway._periodic_reconcile = counting_reconcile  # type: ignore[method-assign]

        # Run serve_forever in a background thread
        t = threading.Thread(target=self.gateway.serve_forever, daemon=True)
        t.start()
        # Wait enough for at least one reconcile tick
        time.sleep(2.5)

        # Verify reconciliation happened and had effect (before shutdown)
        self.assertGreaterEqual(call_count[0], 1)
        row = self.gateway.registry.get_sandbox("sbx-vanish")
        self.assertEqual(row["status"], "lost")
        row = self.gateway.registry.get_sandbox("sbx-1")
        self.assertEqual(row["status"], "active")

        self.gateway.request_shutdown()
        t.join(timeout=5.0)
