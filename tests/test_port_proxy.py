"""Tests for S4 port exposure: PortForwarder, PortManager, gateway routes.

Validates:
- PortForwarder bidirectional TCP relay via a local echo server
- PortManager allocation, release, and port-range exhaustion
- Gateway HTTP routes: POST/GET/DELETE /v1/sandboxes/{id}/ports
- Kill cascade releases ports
- Tenant quota enforcement (max 10 ports)
"""
from __future__ import annotations

import http.client
import json
import os
import socket
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any

from crab.daemon.transport import DaemonClient, serve_unix_socket
from crab.gateway.ports import PortForwarder, PortManager
from crab.gateway.server import GatewayServer
from crab.models import PortAllocation


# ---------------------------------------------------------------------------
# Echo server — simulates a service running inside the sandbox.
# ---------------------------------------------------------------------------


def _start_echo_server(host: str = "127.0.0.1", port: int = 0) -> tuple[socket.socket, int]:
    """Start a TCP echo server; returns (server_socket, bound_port)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.listen(5)
    sock.settimeout(5.0)
    bound_port = sock.getsockname()[1]

    def _serve():
        while True:
            try:
                client, _ = sock.accept()
            except (socket.timeout, OSError):
                break
            threading.Thread(target=_echo_client, args=(client,), daemon=True).start()

    def _echo_client(client: socket.socket):
        try:
            while True:
                data = client.recv(4096)
                if not data:
                    break
                client.sendall(data)
        except OSError:
            pass
        finally:
            client.close()

    threading.Thread(target=_serve, daemon=True).start()
    return sock, bound_port


# ---------------------------------------------------------------------------
# PortForwarder tests
# ---------------------------------------------------------------------------


class PortForwarderTests(unittest.TestCase):
    def test_bidirectional_relay(self) -> None:
        echo_sock, echo_port = _start_echo_server()
        self.addCleanup(echo_sock.close)

        fwd = PortForwarder(0, "127.0.0.1", echo_port)
        # Bind to a random port
        fwd.host_port = _find_free_port()
        fwd.start()
        self.addCleanup(fwd.stop)
        time.sleep(0.1)

        # Connect through the forwarder
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.settimeout(5.0)
        client.connect(("127.0.0.1", fwd.host_port))
        client.sendall(b"hello world")
        reply = client.recv(1024)
        client.close()
        self.assertEqual(reply, b"hello world")

    def test_multiple_connections(self) -> None:
        echo_sock, echo_port = _start_echo_server()
        self.addCleanup(echo_sock.close)

        fwd = PortForwarder(_find_free_port(), "127.0.0.1", echo_port)
        fwd.start()
        self.addCleanup(fwd.stop)
        time.sleep(0.1)

        results = []
        for i in range(3):
            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client.settimeout(5.0)
            client.connect(("127.0.0.1", fwd.host_port))
            msg = f"msg-{i}".encode()
            client.sendall(msg)
            results.append(client.recv(1024))
            client.close()

        for i, r in enumerate(results):
            self.assertEqual(r, f"msg-{i}".encode())

    def test_stop_closes_connections(self) -> None:
        echo_sock, echo_port = _start_echo_server()
        self.addCleanup(echo_sock.close)

        fwd = PortForwarder(_find_free_port(), "127.0.0.1", echo_port)
        fwd.start()
        time.sleep(0.1)
        fwd.stop()

        # After stop, connecting should fail
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.settimeout(1.0)
        with self.assertRaises(OSError):
            client.connect(("127.0.0.1", fwd.host_port))
        client.close()


# ---------------------------------------------------------------------------
# PortManager tests
# ---------------------------------------------------------------------------


class PortManagerTests(unittest.TestCase):
    def test_allocate_and_release(self) -> None:
        echo_sock, echo_port = _start_echo_server()
        self.addCleanup(echo_sock.close)

        mgr = PortManager(port_range=(31000, 31020))
        self.addCleanup(mgr.shutdown)

        host_port = mgr.allocate("sbx-1", "127.0.0.1", echo_port)
        self.assertGreaterEqual(host_port, 31000)
        self.assertLessEqual(host_port, 31020)

        # Verify forwarding works
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.settimeout(5.0)
        client.connect(("127.0.0.1", host_port))
        client.sendall(b"test")
        self.assertEqual(client.recv(1024), b"test")
        client.close()

        mgr.release(host_port)

    def test_port_range_exhaustion(self) -> None:
        echo_sock, echo_port = _start_echo_server()
        self.addCleanup(echo_sock.close)

        # Very small range
        mgr = PortManager(port_range=(31050, 31052))
        self.addCleanup(mgr.shutdown)

        ports = []
        for _ in range(3):
            ports.append(mgr.allocate("sbx-1", "127.0.0.1", echo_port))

        with self.assertRaises(RuntimeError):
            mgr.allocate("sbx-1", "127.0.0.1", echo_port)

        # Release one and allocate again
        mgr.release(ports[0])
        new_port = mgr.allocate("sbx-1", "127.0.0.1", echo_port)
        self.assertEqual(new_port, ports[0])


# ---------------------------------------------------------------------------
# Stub daemon for gateway tests
# ---------------------------------------------------------------------------


def _build_port_test_handler(state: dict[str, Any]):
    """Minimal daemon stub supporting sandbox describe (with guest_ip)."""

    class Handler:
        pass

    from http.server import BaseHTTPRequestHandler

    class StubHandler(BaseHTTPRequestHandler):
        def log_message(self, *a: Any) -> None:
            pass

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path == "/healthz":
                self._json(200, {"ok": True})
            elif path == "/info":
                self._json(200, {"ok": True, "pid": state.get("pid", 9999)})
            elif path == "/sandboxes":
                self._json(200, {"ok": True, "sandboxes": state.get("sandboxes", [])})
            elif path.startswith("/sandboxes/") and path.count("/") == 2:
                sid = path.split("/")[2]
                self._json(200, {
                    "ok": True,
                    "description": {
                        "sandbox_id": sid,
                        "runtime_name": "fake",
                        "status": "running",
                        "metadata": {
                            "guest_ip": state.get("guest_ip", "10.250.0.2"),
                            "network_mode": state.get("network_mode", "isolated"),
                            "network_namespace_path": state.get(
                                "network_namespace_path", "/var/run/netns/ts-test"
                            ),
                        },
                    },
                    "runtime_state": None,
                })
            else:
                self._json(404, {"ok": False, "error": "not found"})

        def do_POST(self) -> None:
            self._json(200, {"ok": True})

        def do_DELETE(self) -> None:
            path = self.path.split("?", 1)[0]
            if "/sandboxes/" in path:
                self._json(200, {"ok": True})
            else:
                self._json(404, {"ok": False})

        def _json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return StubHandler


class GatewayPortTestBase(unittest.TestCase):
    """Base: stub daemon + gateway with port management."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name)
        self.daemon_socket = self.base / "daemon.sock"
        self.state: dict[str, Any] = {
            "pid": 1000,
            "guest_ip": "127.0.0.1",
            "network_mode": "isolated",
            "network_namespace_path": "/var/run/netns/ts-test",
            "sandboxes": [
                {
                    "sandbox_id": "sbx-1",
                    "metadata": {
                        "guest_ip": "127.0.0.1",
                        "network_mode": "isolated",
                        "network_namespace_path": "/var/run/netns/ts-test",
                    },
                }
            ],
        }
        self._start_stub_daemon()
        self._start_gateway()

    def _start_stub_daemon(self) -> None:
        handler = _build_port_test_handler(self.state)
        self.daemon_server = serve_unix_socket(self.daemon_socket, handler)
        thread = threading.Thread(target=self.daemon_server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(self._stop_daemon)

    def _stop_daemon(self) -> None:
        if self.daemon_server:
            self.daemon_server.shutdown()
            self.daemon_server.server_close()

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
        # Register the sandbox in the registry
        self.admin = DaemonClient(self.base / "admin.sock")
        tenant = self.admin.post_json("/admin/tenants", {"name": "test"})["tenant"]
        self.tenant_id = tenant["id"]
        key_resp = self.admin.post_json("/admin/keys", {"tenant_id": self.tenant_id})
        self.api_key = key_resp["api_key"]
        # Register sandbox sbx-1
        self.gateway.registry.register_sandbox(self.tenant_id, "sbx-1")

    def _start_echo_for_guest(self) -> int:
        """Start echo server to act as guest service; returns port."""
        sock, port = _start_echo_server()
        self.addCleanup(sock.close)
        self.state["guest_ip"] = "127.0.0.1"
        return port

    def request(
        self, method: str, path: str, body: dict | None = None
    ) -> tuple[int, dict]:
        conn = http.client.HTTPConnection("127.0.0.1", self.gateway.port, timeout=10)
        headers: dict[str, str] = {"Authorization": f"Bearer {self.api_key}"}
        payload = None
        if body is not None:
            payload = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        conn.request(method, path, body=payload, headers=headers)
        resp = conn.getresponse()
        raw = resp.read()
        status = resp.status
        conn.close()
        return status, json.loads(raw) if raw else {}


# ---------------------------------------------------------------------------
# Gateway route tests
# ---------------------------------------------------------------------------


class GatewayPortRouteTests(GatewayPortTestBase):
    def test_expose_port_uses_guest_ip_from_daemon_description(self) -> None:
        self.state["guest_ip"] = "10.250.99.7"
        captured: dict[str, object] = {}
        original_allocate = self.gateway.port_manager.allocate

        def capture_allocate(
            sandbox_id: str, guest_ip: str, guest_port: int, **_: Any
        ) -> int:
            captured.update(
                sandbox_id=sandbox_id,
                guest_ip=guest_ip,
                guest_port=guest_port,
            )
            return _find_free_port()

        self.gateway.port_manager.allocate = capture_allocate
        self.addCleanup(
            setattr, self.gateway.port_manager, "allocate", original_allocate
        )

        status, _ = self.request(
            "POST", "/v1/sandboxes/sbx-1/ports", {"port": 8080}
        )

        self.assertEqual(status, 200)
        self.assertEqual(captured["sandbox_id"], "sbx-1")
        self.assertEqual(captured["guest_ip"], "10.250.99.7")
        self.assertEqual(captured["guest_port"], 8080)

    def test_expose_port_rejects_host_network_without_allocating(self) -> None:
        self.state["network_mode"] = "host"
        self.state["network_namespace_path"] = None
        original_allocate = self.gateway.port_manager.allocate
        called = False

        def capture_allocate(*args: Any, **kwargs: Any) -> int:
            nonlocal called
            called = True
            return original_allocate(*args, **kwargs)

        self.gateway.port_manager.allocate = capture_allocate
        self.addCleanup(
            setattr, self.gateway.port_manager, "allocate", original_allocate
        )

        status, body = self.request(
            "POST", "/v1/sandboxes/sbx-1/ports", {"port": 8080}
        )

        self.assertEqual(status, 400)
        self.assertIn("isolated network namespace", body["error"])
        self.assertFalse(called)
        self.assertEqual(self.gateway.registry.list_ports("sbx-1"), [])

    def test_expose_port_rejects_isolated_metadata_without_guest_ip(self) -> None:
        self.state["guest_ip"] = None

        status, body = self.request(
            "POST", "/v1/sandboxes/sbx-1/ports", {"port": 8080}
        )

        self.assertEqual(status, 400)
        self.assertIn("has no guest_ip", body["error"])
        self.assertEqual(self.gateway.registry.list_ports("sbx-1"), [])

    def test_expose_port(self) -> None:
        guest_port = self._start_echo_for_guest()
        status, body = self.request("POST", "/v1/sandboxes/sbx-1/ports", {"port": guest_port})
        self.assertEqual(status, 200)
        self.assertTrue(body["ok"])
        self.assertIn("host_port", body)
        self.assertEqual(body["guest_port"], guest_port)
        self.assertIn("tcp://", body["url"])

        # Verify actual forwarding works
        hp = body["host_port"]
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.settimeout(5.0)
        client.connect(("127.0.0.1", hp))
        client.sendall(b"via gateway")
        reply = client.recv(1024)
        client.close()
        self.assertEqual(reply, b"via gateway")

    def test_list_ports(self) -> None:
        guest_port = self._start_echo_for_guest()
        self.request("POST", "/v1/sandboxes/sbx-1/ports", {"port": guest_port})
        status, body = self.request("GET", "/v1/sandboxes/sbx-1/ports")
        self.assertEqual(status, 200)
        self.assertEqual(len(body["ports"]), 1)
        self.assertEqual(body["ports"][0]["guest_port"], guest_port)

    def test_release_port(self) -> None:
        guest_port = self._start_echo_for_guest()
        _, alloc = self.request("POST", "/v1/sandboxes/sbx-1/ports", {"port": guest_port})
        host_port = alloc["host_port"]

        status, body = self.request("DELETE", f"/v1/sandboxes/sbx-1/ports/{guest_port}")
        self.assertEqual(status, 200)
        self.assertTrue(body["ok"])

        # Port should be freed — connection should fail
        time.sleep(0.2)
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.settimeout(1.0)
        with self.assertRaises(OSError):
            client.connect(("127.0.0.1", host_port))
        client.close()

    def test_kill_cascade_releases_ports(self) -> None:
        guest_port = self._start_echo_for_guest()
        _, alloc = self.request("POST", "/v1/sandboxes/sbx-1/ports", {"port": guest_port})
        host_port = alloc["host_port"]

        # Kill the sandbox
        status, _ = self.request("DELETE", "/v1/sandboxes/sbx-1")
        self.assertEqual(status, 200)

        # Port should be released
        time.sleep(0.2)
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.settimeout(1.0)
        with self.assertRaises(OSError):
            client.connect(("127.0.0.1", host_port))
        client.close()

    def test_unknown_sandbox_404(self) -> None:
        status, body = self.request("POST", "/v1/sandboxes/sbx-unknown/ports", {"port": 8080})
        self.assertEqual(status, 404)

    def test_no_auth_401(self) -> None:
        conn = http.client.HTTPConnection("127.0.0.1", self.gateway.port, timeout=5)
        conn.request("POST", "/v1/sandboxes/sbx-1/ports",
                     json.dumps({"port": 8080}).encode(),
                     {"Content-Type": "application/json"})
        resp = conn.getresponse()
        self.assertEqual(resp.status, 401)
        conn.close()

    def test_quota_exceeded(self) -> None:
        guest_port = self._start_echo_for_guest()
        # Allocate 10 ports (the max)
        for i in range(10):
            status, _ = self.request("POST", "/v1/sandboxes/sbx-1/ports", {"port": guest_port + i})
            self.assertEqual(status, 200, f"allocation {i} failed")
        # 11th should fail
        status, body = self.request("POST", "/v1/sandboxes/sbx-1/ports", {"port": guest_port + 10})
        self.assertEqual(status, 409)
        self.assertEqual(body["error_type"], "quota_exceeded")

    def test_missing_port_field_400(self) -> None:
        status, body = self.request("POST", "/v1/sandboxes/sbx-1/ports", {})
        self.assertEqual(status, 400)


# ---------------------------------------------------------------------------
# Port rehydration tests (S5)
# ---------------------------------------------------------------------------


class PortRehydrationTests(unittest.TestCase):
    """Tests for S5 port rehydration at gateway restart."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name)
        self.daemon_socket = self.base / "daemon.sock"
        self.state: dict[str, Any] = {
            "pid": 1000,
            "guest_ip": "127.0.0.1",
            "network_mode": "isolated",
            "network_namespace_path": "/var/run/netns/ts-test",
            "sandboxes": [
                {
                    "sandbox_id": "sbx-1",
                    "metadata": {
                        "guest_ip": "127.0.0.1",
                        "network_mode": "isolated",
                        "network_namespace_path": "/var/run/netns/ts-test",
                    },
                }
            ],
        }
        self._start_stub_daemon()
        # Start echo server that persists across gateway restarts
        self.echo_sock, self.echo_port = _start_echo_server()
        self.addCleanup(self.echo_sock.close)

    def _start_stub_daemon(self) -> None:
        handler = _build_port_test_handler(self.state)
        self.daemon_server = serve_unix_socket(self.daemon_socket, handler)
        thread = threading.Thread(target=self.daemon_server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(self._stop_daemon)

    def _stop_daemon(self) -> None:
        if self.daemon_server:
            self.daemon_server.shutdown()
            self.daemon_server.server_close()

    def _make_gateway(self) -> GatewayServer:
        gw = GatewayServer(
            data_dir=self.base / "gw",
            daemon_socket=self.daemon_socket,
            host="127.0.0.1",
            port=0,
            admin_socket_path=self.base / "admin.sock",
        )
        gw.start()
        return gw

    def test_rehydrate_rebuilds_forwarder_after_restart(self) -> None:
        # First gateway: allocate a port
        gw1 = self._make_gateway()
        admin = DaemonClient(self.base / "admin.sock")
        tenant = admin.post_json("/admin/tenants", {"name": "t1"})["tenant"]
        gw1.registry.register_sandbox(tenant["id"], "sbx-1")
        host_port = gw1.port_manager.allocate("sbx-1", "127.0.0.1", self.echo_port)
        gw1.registry.allocate_port(
            "sbx-1", tenant["id"], self.echo_port, host_port, guest_ip="127.0.0.1"
        )
        # Verify forwarding works
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.settimeout(5.0)
        client.connect(("127.0.0.1", host_port))
        client.sendall(b"before-restart")
        self.assertEqual(client.recv(1024), b"before-restart")
        client.close()
        # Stop gateway (simulates restart)
        gw1.stop()
        time.sleep(0.3)

        # Verify port is dead after stop
        dead = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        dead.settimeout(1.0)
        with self.assertRaises(OSError):
            dead.connect(("127.0.0.1", host_port))
        dead.close()

        # Second gateway: should rehydrate
        # Need to remove old admin socket
        admin_sock_path = self.base / "admin.sock"
        if admin_sock_path.exists():
            admin_sock_path.unlink()
        gw2 = self._make_gateway()
        self.addCleanup(gw2.stop)
        # Forwarder should be rebuilt — TCP should work
        time.sleep(0.2)
        client2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client2.settimeout(5.0)
        client2.connect(("127.0.0.1", host_port))
        client2.sendall(b"after-restart")
        self.assertEqual(client2.recv(1024), b"after-restart")
        client2.close()

    def test_lost_sandbox_allocation_cleaned_not_rebuilt(self) -> None:
        # First gateway: allocate a port, then mark sandbox lost
        gw1 = self._make_gateway()
        admin = DaemonClient(self.base / "admin.sock")
        tenant = admin.post_json("/admin/tenants", {"name": "t1"})["tenant"]
        gw1.registry.register_sandbox(tenant["id"], "sbx-1")
        host_port = gw1.port_manager.allocate("sbx-1", "127.0.0.1", self.echo_port)
        gw1.registry.allocate_port(
            "sbx-1", tenant["id"], self.echo_port, host_port, guest_ip="127.0.0.1"
        )
        # Mark sandbox lost (simulates daemon restart)
        gw1.registry.mark_missing_lost(set())  # all become lost
        gw1.stop()

        # Second gateway: should clean the DB row, not rebuild
        admin_sock_path = self.base / "admin.sock"
        if admin_sock_path.exists():
            admin_sock_path.unlink()
        # Remove sbx-1 from daemon listing so reconcile marks it lost again
        self.state["sandboxes"] = []
        gw2 = self._make_gateway()
        self.addCleanup(gw2.stop)
        # No port allocations should remain
        self.assertEqual(gw2.registry.list_ports("sbx-1"), [])
        # Forwarder should NOT be running
        dead = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        dead.settimeout(1.0)
        with self.assertRaises(OSError):
            dead.connect(("127.0.0.1", host_port))
        dead.close()

    def test_host_network_allocation_cleaned_not_rebuilt(self) -> None:
        gw1 = self._make_gateway()
        admin = DaemonClient(self.base / "admin.sock")
        tenant = admin.post_json("/admin/tenants", {"name": "t1"})["tenant"]
        gw1.registry.register_sandbox(tenant["id"], "sbx-1")
        host_port = gw1.port_manager.allocate(
            "sbx-1", "127.0.0.1", self.echo_port
        )
        gw1.registry.allocate_port(
            "sbx-1",
            tenant["id"],
            self.echo_port,
            host_port,
            guest_ip="127.0.0.1",
        )
        gw1.stop()
        self.state["network_mode"] = "host"
        self.state["network_namespace_path"] = None
        self.state["sandboxes"][0]["metadata"].update(
            {"network_mode": "host", "network_namespace_path": None}
        )

        admin_sock_path = self.base / "admin.sock"
        if admin_sock_path.exists():
            admin_sock_path.unlink()
        gw2 = self._make_gateway()
        self.addCleanup(gw2.stop)

        self.assertEqual(gw2.registry.list_ports("sbx-1"), [])
        self.assertNotIn(host_port, gw2.port_manager._forwarders)

    def test_rehydrate_refreshes_guest_ip_from_daemon(self) -> None:
        gw1 = self._make_gateway()
        admin = DaemonClient(self.base / "admin.sock")
        tenant = admin.post_json("/admin/tenants", {"name": "t1"})["tenant"]
        gw1.registry.register_sandbox(tenant["id"], "sbx-1")
        host_port = _find_free_port()
        gw1.registry.allocate_port(
            "sbx-1",
            tenant["id"],
            self.echo_port,
            host_port,
            guest_ip="10.250.0.2",
        )
        gw1.stop()
        self.state["sandboxes"][0]["metadata"]["guest_ip"] = "10.250.0.9"

        admin_sock_path = self.base / "admin.sock"
        if admin_sock_path.exists():
            admin_sock_path.unlink()
        gw2 = self._make_gateway()
        self.addCleanup(gw2.stop)

        self.assertEqual(
            gw2.port_manager._forwarders[host_port].guest_ip, "10.250.0.9"
        )
        allocations = gw2.registry.list_all_port_allocations()
        self.assertEqual(allocations[0]["guest_ip"], "10.250.0.9")

    def test_occupied_port_releases_db_row(self) -> None:
        # First gateway: allocate a port
        gw1 = self._make_gateway()
        admin = DaemonClient(self.base / "admin.sock")
        tenant = admin.post_json("/admin/tenants", {"name": "t1"})["tenant"]
        gw1.registry.register_sandbox(tenant["id"], "sbx-1")
        host_port = gw1.port_manager.allocate("sbx-1", "127.0.0.1", self.echo_port)
        gw1.registry.allocate_port(
            "sbx-1", tenant["id"], self.echo_port, host_port, guest_ip="127.0.0.1"
        )
        gw1.stop()

        # Occupy the port so rehydration fails
        blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        blocker.bind(("0.0.0.0", host_port))
        blocker.listen(1)
        self.addCleanup(blocker.close)

        # Second gateway: rehydration should fail and clean DB row
        admin_sock_path = self.base / "admin.sock"
        if admin_sock_path.exists():
            admin_sock_path.unlink()
        gw2 = self._make_gateway()
        self.addCleanup(gw2.stop)
        # DB row should be gone
        self.assertEqual(gw2.registry.list_ports("sbx-1"), [])

    def test_guest_ip_migration_idempotent(self) -> None:
        from crab.gateway.registry import GatewayRegistry
        # Create a registry without guest_ip column (simulate pre-S5)
        db_path = self.base / "gw" / "gateway.sqlite3"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        reg1 = GatewayRegistry(db_path)
        # Check guest_ip column exists after first init
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        cols = {row[1] for row in conn.execute("PRAGMA table_info(port_allocations)").fetchall()}
        self.assertIn("guest_ip", cols)
        conn.close()
        reg1.close()
        # Open again — migration should be idempotent
        reg2 = GatewayRegistry(db_path)
        conn2 = sqlite3.connect(str(db_path))
        cols2 = {row[1] for row in conn2.execute("PRAGMA table_info(port_allocations)").fetchall()}
        self.assertIn("guest_ip", cols2)
        conn2.close()
        reg2.close()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _find_free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


if __name__ == "__main__":
    unittest.main()
