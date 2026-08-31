"""Tests for the S4 streaming exec chain.

Validates the full path: stub daemon -> DaemonClient.stream_post ->
gateway relay -> CloudClient.stream_post -> RemoteEngine.stream_exec ->
commands.stream(). Uses the same scripted-stub-daemon pattern as
test_gateway_server.py, extended with a handler that speaks chunked
NDJSON for ?stream=1 requests.
"""
from __future__ import annotations

import http.client
import json
import os
import tempfile
import threading
import time
import unittest
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

from crab.daemon.transport import DaemonClient, StreamIterator, serve_unix_socket
from crab.gateway.server import GatewayServer
from crab.models import ExecDone, ExecEvent


# ---------------------------------------------------------------------------
# Stub daemon that supports ?stream=1 on exec routes.
# ---------------------------------------------------------------------------

_STREAM_EVENTS: list[dict[str, Any]] = [
    {"ch": "stdout", "t": "hello\n"},
    {"ch": "stderr", "t": "warn\n"},
    {"ch": "stdout", "t": "world\n"},
    {"done": True, "rc": 0},
]


def _build_streaming_stub_handler(
    events: list[dict[str, Any]] | None = None,
    sandbox_ids: set[str] | None = None,
):
    response_events = events if events is not None else list(_STREAM_EVENTS)
    known_sandboxes = sandbox_ids if sandbox_ids is not None else {"sbx-1"}

    class Handler(BaseHTTPRequestHandler):
        server_version = "stub-stream-daemon/1"
        sys_version = ""

        def log_message(self, fmt: str, *args: Any) -> None:
            pass

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path == "/healthz":
                self._send_json({"ok": True, "started": True})
                return
            if path == "/info":
                self._send_json({
                    "ok": True, "version": 1, "pid": 1000,
                    "runtime": "runc", "default_image": "ubuntu:22.04",
                    "storage_root": "/tmp/s", "runtime_root": "/tmp/r",
                    "network_bridge_ip": "10.0.0.1",
                    "sandbox_count": len(known_sandboxes),
                })
                return
            if path == "/sandboxes":
                rows = [
                    {"sandbox_id": sid, "runtime_name": "runc",
                     "status": "running", "metadata": {}}
                    for sid in known_sandboxes
                ]
                self._send_json({"ok": True, "sandboxes": rows})
                return
            parts = [p for p in path.strip("/").split("/") if p]
            if len(parts) == 2 and parts[0] == "sandboxes":
                sid = parts[1]
                if sid in known_sandboxes:
                    self._send_json({
                        "ok": True,
                        "description": {
                            "sandbox_id": sid, "runtime_name": "runc",
                            "status": "running", "metadata": {},
                        },
                        "runtime_state": None,
                    })
                    return
            self._send_error(404, "not found")

        def do_POST(self) -> None:
            path_full = self.path
            path = path_full.split("?", 1)[0]
            query = path_full.split("?", 1)[1] if "?" in path_full else ""
            body = self._read_body()

            if path == "/sandboxes":
                sid = f"sbx-new-{len(known_sandboxes)+1}"
                known_sandboxes.add(sid)
                self._send_json({"ok": True, "sandbox_id": sid})
                return

            parts = [p for p in path.strip("/").split("/") if p]
            if (len(parts) == 3 and parts[0] == "sandboxes"
                    and parts[2] == "exec"):
                sid = parts[1]
                if sid not in known_sandboxes:
                    self._send_error(404, f"unknown sandbox: {sid}")
                    return
                if "stream=1" in query:
                    self._stream_exec(body)
                    return
                argv = body.get("argv", [])
                self._send_json({
                    "ok": True,
                    "result": {
                        "args": argv, "returncode": 0,
                        "stdout": "ok\n", "stderr": "",
                    },
                })
                return

            if (len(parts) == 3 and parts[0] == "sandboxes"
                    and parts[2] == "stop"):
                self._send_json({
                    "ok": True, "sandbox_id": parts[1], "stopped": True,
                })
                return

            self._send_error(404, "not found")

        def do_DELETE(self) -> None:
            path = self.path.split("?", 1)[0]
            parts = [p for p in path.strip("/").split("/") if p]
            if len(parts) == 2 and parts[0] == "sandboxes":
                sid = parts[1]
                known_sandboxes.discard(sid)
                self._send_json({"ok": True, "sandbox_id": sid})
                return
            self._send_error(404, "not found")

        def _stream_exec(self, body: dict[str, Any]) -> None:
            self.send_response(200)
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                for event in response_events:
                    line = (json.dumps(event) + "\n").encode("utf-8")
                    chunk_header = f"{len(line):x}\r\n".encode("ascii")
                    self.wfile.write(chunk_header)
                    self.wfile.write(line)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _read_body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0:
                return {}
            raw = self.rfile.read(length)
            return json.loads(raw) if raw else {}

        def _send_json(self, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _send_error(self, code: int, msg: str) -> None:
            body = json.dumps({"ok": False, "error": msg}).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

    return Handler


# ---------------------------------------------------------------------------
# Test base
# ---------------------------------------------------------------------------


class StreamingTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name)
        self.daemon_socket = self.base / "daemon.sock"
        self._start_stub_daemon()
        self._start_gateway()

    def _start_stub_daemon(self) -> None:
        handler = _build_streaming_stub_handler()
        self.daemon_server = serve_unix_socket(self.daemon_socket, handler)
        thread = threading.Thread(
            target=self.daemon_server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(self._stop_daemon)

    def _stop_daemon(self) -> None:
        self.daemon_server.shutdown()
        self.daemon_server.server_close()
        if self.daemon_socket.exists():
            self.daemon_socket.unlink()

    def _start_gateway(self) -> None:
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
        tenant = self.admin.post_json(
            "/admin/tenants", {"name": "test"})["tenant"]
        self.tenant_id = tenant["id"]
        self.api_key = self.admin.post_json(
            "/admin/keys", {"tenant_id": self.tenant_id}
        )["api_key"]
        self.admin.post_json(
            "/admin/sandboxes/adopt",
            {"tenant": "test", "sandbox_ids": ["sbx-1"]},
        )


# ---------------------------------------------------------------------------
# DaemonClient.stream_post tests
# ---------------------------------------------------------------------------


class DaemonClientStreamTests(StreamingTestBase):
    def test_stream_post_yields_events(self) -> None:
        client = DaemonClient(self.daemon_socket)
        with client.stream_post(
            "/sandboxes/sbx-1/exec?stream=1", {"argv": ["/bin/echo", "hi"]}
        ) as stream:
            events = list(stream)
        self.assertEqual(len(events), 4)
        self.assertEqual(events[0], {"ch": "stdout", "t": "hello\n"})
        self.assertEqual(events[1], {"ch": "stderr", "t": "warn\n"})
        self.assertEqual(events[2], {"ch": "stdout", "t": "world\n"})
        self.assertEqual(events[3], {"done": True, "rc": 0})

    def test_stream_post_early_close(self) -> None:
        client = DaemonClient(self.daemon_socket)
        stream = client.stream_post(
            "/sandboxes/sbx-1/exec?stream=1", {"argv": ["/bin/echo", "hi"]}
        )
        first = next(stream)
        self.assertEqual(first["ch"], "stdout")
        stream.close()
        with self.assertRaises(StopIteration):
            next(stream)

    def test_non_stream_exec_unchanged(self) -> None:
        client = DaemonClient(self.daemon_socket)
        result = client.post_json(
            "/sandboxes/sbx-1/exec", {"argv": ["echo", "hi"]})
        self.assertTrue(result["ok"])
        self.assertEqual(result["result"]["returncode"], 0)


# ---------------------------------------------------------------------------
# Gateway streaming relay tests
# ---------------------------------------------------------------------------


class GatewayStreamRelayTests(StreamingTestBase):
    def _stream_request(
        self, path: str, body: dict[str, Any]
    ) -> list[dict[str, Any]]:
        conn = http.client.HTTPConnection(
            "127.0.0.1", self.gateway.port, timeout=10)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = json.dumps(body).encode("utf-8")
        conn.request("POST", path, body=payload, headers=headers)
        response = conn.getresponse()
        self.assertEqual(response.status, 200)
        events: list[dict[str, Any]] = []
        while True:
            line = response.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))
        conn.close()
        return events

    def test_stream_exec_through_gateway(self) -> None:
        events = self._stream_request(
            "/v1/sandboxes/sbx-1/exec?stream=1",
            {"argv": ["/bin/echo", "hello"]},
        )
        self.assertEqual(len(events), 4)
        self.assertEqual(events[0], {"ch": "stdout", "t": "hello\n"})
        self.assertEqual(events[-1], {"done": True, "rc": 0})

    def test_stream_exec_unknown_sandbox_404(self) -> None:
        conn = http.client.HTTPConnection(
            "127.0.0.1", self.gateway.port, timeout=10)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = json.dumps({"argv": ["echo"]}).encode("utf-8")
        conn.request(
            "POST", "/v1/sandboxes/sbx-unknown/exec?stream=1",
            body=payload, headers=headers)
        response = conn.getresponse()
        self.assertEqual(response.status, 404)
        conn.close()

    def test_stream_exec_no_auth_401(self) -> None:
        conn = http.client.HTTPConnection(
            "127.0.0.1", self.gateway.port, timeout=10)
        payload = json.dumps({"argv": ["echo"]}).encode("utf-8")
        conn.request(
            "POST", "/v1/sandboxes/sbx-1/exec?stream=1",
            body=payload, headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        self.assertEqual(response.status, 401)
        conn.close()

    def test_non_stream_exec_regression(self) -> None:
        conn = http.client.HTTPConnection(
            "127.0.0.1", self.gateway.port, timeout=10)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = json.dumps({"argv": ["echo", "test"]}).encode("utf-8")
        conn.request(
            "POST", "/v1/sandboxes/sbx-1/exec",
            body=payload, headers=headers)
        response = conn.getresponse()
        self.assertEqual(response.status, 200)
        result = json.loads(response.read())
        self.assertTrue(result["ok"])
        self.assertEqual(result["result"]["returncode"], 0)
        conn.close()


# ---------------------------------------------------------------------------
# CloudClient.stream_post tests
# ---------------------------------------------------------------------------


class CloudClientStreamTests(StreamingTestBase):
    def _make_cloud_client(self):
        from crab.cloud_client import CloudClient
        return CloudClient(
            f"http://127.0.0.1:{self.gateway.port}",
            api_key=self.api_key,
            timeout_seconds=10,
        )

    def test_cloud_stream_post_yields_events(self) -> None:
        client = self._make_cloud_client()
        with client.stream_post(
            "/sandboxes/sbx-1/exec?stream=1", {"argv": ["echo", "hi"]}
        ) as stream:
            events = list(stream)
        self.assertEqual(len(events), 4)
        self.assertEqual(events[0], {"ch": "stdout", "t": "hello\n"})
        self.assertEqual(events[-1], {"done": True, "rc": 0})

    def test_cloud_stream_post_error_raises(self) -> None:
        from crab.cloud_client import SandboxNotFound
        client = self._make_cloud_client()
        with self.assertRaises(SandboxNotFound):
            client.stream_post(
                "/sandboxes/sbx-unknown/exec?stream=1", {"argv": ["echo"]})


# ---------------------------------------------------------------------------
# RemoteEngine.stream_exec tests
# ---------------------------------------------------------------------------


class RemoteEngineStreamTests(StreamingTestBase):
    def _make_runtime_proxy(self):
        from crab.cloud_client import CloudClient
        from crab.remote_engine import RuntimeProxy
        client = CloudClient(
            f"http://127.0.0.1:{self.gateway.port}",
            api_key=self.api_key,
            timeout_seconds=10,
        )
        return RuntimeProxy(client, name="runc")

    def test_stream_exec_yields_typed_events(self) -> None:
        from crab.ids import SandboxId
        proxy = self._make_runtime_proxy()
        events = list(
            proxy.stream_exec(SandboxId("sbx-1"), ["/bin/echo", "hi"]))
        self.assertEqual(len(events), 4)
        self.assertIsInstance(events[0], ExecEvent)
        self.assertEqual(events[0].channel, "stdout")
        self.assertEqual(events[0].text, "hello\n")
        self.assertIsInstance(events[1], ExecEvent)
        self.assertEqual(events[1].channel, "stderr")
        self.assertIsInstance(events[-1], ExecDone)
        self.assertEqual(events[-1].returncode, 0)

    def test_non_stream_exec_still_works(self) -> None:
        from crab.ids import SandboxId
        proxy = self._make_runtime_proxy()
        result = proxy.exec(SandboxId("sbx-1"), ["echo", "test"])
        self.assertEqual(result.returncode, 0)


class RemoteEngineTimeoutStreamTests(StreamingTestBase):
    def _start_stub_daemon(self) -> None:
        events = [
            {"ch": "stdout", "t": "started\n"},
            {
                "done": True,
                "rc": None,
                "error": "command timed out",
                "error_type": "exec_timeout",
                "timeout_s": 1.0,
                "stdout": "started\n",
                "stderr": "",
                "cleanup_completed": True,
            },
        ]
        handler = _build_streaming_stub_handler(events=events)
        self.daemon_server = serve_unix_socket(self.daemon_socket, handler)
        thread = threading.Thread(
            target=self.daemon_server.serve_forever, daemon=True
        )
        thread.start()
        self.addCleanup(self._stop_daemon)

    def test_stream_timeout_rehydrates_stable_type(self) -> None:
        from crab.cloud_client import CloudClient
        from crab.errors import SandboxExecTimeout
        from crab.ids import SandboxId
        from crab.remote_engine import RuntimeProxy

        proxy = RuntimeProxy(
            CloudClient(
                f"http://127.0.0.1:{self.gateway.port}",
                api_key=self.api_key,
                timeout_seconds=10,
            ),
            name="runc",
        )
        stream = proxy.stream_exec(
            SandboxId("sbx-1"), ["sh", "-c", "sleep 30 & wait"], timeout_s=1.0
        )
        first = next(stream)
        self.assertIsInstance(first, ExecEvent)
        with self.assertRaises(SandboxExecTimeout) as caught:
            list(stream)
        self.assertEqual(caught.exception.stdout, "started\n")


# ---------------------------------------------------------------------------
# StreamIterator unit tests
# ---------------------------------------------------------------------------


class StreamIteratorTests(unittest.TestCase):
    def test_double_close_is_safe(self) -> None:
        si = StreamIterator.__new__(StreamIterator)
        si._closed = True
        si._response = None
        si._conn = None
        si.close()

    def test_iteration_stops_on_empty_read(self) -> None:
        class FakeResponse:
            def readline(self):
                return b""
            def close(self):
                pass

        class FakeConn:
            def close(self):
                pass

        si = StreamIterator(FakeResponse(), FakeConn())
        events = list(si)
        self.assertEqual(events, [])


# ---------------------------------------------------------------------------
# Real daemon E2E test (gate-controlled)
# ---------------------------------------------------------------------------


@unittest.skipUnless(
    os.environ.get("CRAB_REAL_HOST_TESTS"), "requires real daemon"
)
class RealDaemonStreamTests(unittest.TestCase):
    """End-to-end streaming exec against a real running daemon."""

    def test_real_stream_exec_incremental(self) -> None:
        import crab

        sandbox = crab.Sandbox()
        try:
            events: list = []
            for event in sandbox.commands.stream(
                "echo hello && sleep 0.5 && echo world"
            ):
                events.append(event)

            # Should have at least 2 stdout events (hello, world) + ExecDone
            stdout_events = [
                e for e in events if isinstance(e, ExecEvent) and e.channel == "stdout"
            ]
            self.assertGreaterEqual(
                len(stdout_events), 2,
                f"expected >=2 stdout events, got {stdout_events}",
            )
            # Verify content
            combined = "".join(e.text for e in stdout_events)
            self.assertIn("hello", combined)
            self.assertIn("world", combined)

            # Last event should be ExecDone with rc=0
            self.assertIsInstance(events[-1], ExecDone)
            self.assertEqual(events[-1].returncode, 0)
        finally:
            sandbox.kill()


if __name__ == "__main__":
    unittest.main()
