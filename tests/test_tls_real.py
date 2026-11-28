"""Real-host end-to-end tests for TLS interception (PR-T1.3).

These tests require the full crab-dev VM stack (root, runc, ZFS,
iptables, bridge netns, cryptography). They self-skip on any
environment that lacks the necessary tools or privileges.

Test matrix (§5 of the TLS interception design doc):
- Interception off: flow is opaque, ledger has SNI only.
- Interception on + CA injected: HTTPS GET → idempotent_read, recorded, replayable.
- fork(effects="reject") HTTPS POST → 503, never reaches server.
- Sandbox does not trust CA → handshake fails, passthrough → opaque.
- on_handshake_failure=refuse → flow fails instead of tunnelling.
- Init-process daemon sees CA env vars.
"""
from __future__ import annotations

import http.server
import os
import shutil
import socket
import ssl
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path

from crab import Engine, EngineConfig, Sandbox


def _real_stack_available() -> bool:
    """Check if the full crab-dev VM stack is available."""
    if os.geteuid() != 0:
        return False
    tools = ("docker", "runc", "criu", "zfs", "iptables", "ip")
    return all(shutil.which(tool) is not None for tool in tools)


def _cryptography_available() -> bool:
    """Check if the cryptography package is installed."""
    try:
        import cryptography  # noqa: F401
        return True
    except ImportError:
        return False


def _host_lan_ip() -> str | None:
    """A routable host address reachable from the sandbox bridge."""
    result = subprocess.run(
        ["ip", "-4", "-o", "addr", "show", "scope", "global"],
        capture_output=True, text=True, check=False,
    )
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) < 4 or fields[1].startswith(("acb", "lo", "vh")):
            continue
        return fields[3].split("/")[0]
    return None


_SKIP_REASON = (
    "TLS real E2E requires: root, docker, runc, criu, zfs, iptables, ip, "
    "cryptography — only available in the crab-dev VM"
)

_IMAGE = "python:3.11-slim"


def _inject_ca_into_sandbox(engine, sandbox):
    """Write the engine's CA cert into the sandbox at the expected path.

    Works around shared-rootfs caching: the ZFS snapshot may have been
    built by a previous sandbox that didn't have TLS interception on, so
    the CA cert file is absent.  We write it in-band via commands.run.
    """
    ca_path = engine.tls_ca_cert_path
    if ca_path is None:
        return
    ca_pem = ca_path.read_text()
    # Use heredoc to avoid shell escaping issues with PEM content
    sandbox.commands.run(
        "sh -c 'mkdir -p /usr/local/share/ca-certificates && "
        "cat > /usr/local/share/ca-certificates/crab-ca.crt << \"CRAB_CA_EOF\"\n"
        f"{ca_pem}"
        "CRAB_CA_EOF\n'"
    )


def _patch_upstream_trust(engine, tls_server):
    """Monkey-patch the TLS interceptor's upstream context to trust the test CA.

    In production the proxy verifies upstream using system CAs. In tests,
    the upstream server uses a throwaway CA that is not in the system
    store, so we add it explicitly.
    """
    ti = engine._tls_interceptor_ref
    if ti is None:
        return
    original = ti.build_upstream_context

    def _patched(sni):
        ctx = original(sni)
        ctx.load_verify_locations(cafile=str(tls_server.ca_cert_path))
        return ctx

    ti.build_upstream_context = _patched


class _TLSServer:
    """An HTTPS server on 0.0.0.0 using a throwaway CA.

    The CA cert and key are generated on demand via crab.tls_ca.CAStore.
    The server certificate is minted for the given hostname.

    Binds to port 443 by default so the egress proxy's TLS interceptor
    (which only intercepts on ports in _WEB_PORTS={443,8443}) will
    actually perform interception.  Using 443 (not 8443) ensures the
    Host header omits the port number, which the proxy passes through
    as server_hostname for upstream verification.
    """

    def __init__(self, hostname: str = "tls-test.example.com", port: int = 443):
        from crab.tls_ca import CAStore, LeafMinter

        self._tmp = tempfile.TemporaryDirectory(prefix="tls_server_")
        ca_dir = Path(self._tmp.name) / "ca"
        self.ca_store = CAStore(ca_dir)
        self.ca_cert_path = self.ca_store.cert_path
        minter = LeafMinter(self.ca_store)
        leaf_cert, leaf_key = minter.get_or_mint(hostname)
        self.hostname = hostname

        # Write leaf cert + key to temp files for ssl context
        self._leaf_cert_path = Path(self._tmp.name) / "leaf.crt"
        self._leaf_key_path = Path(self._tmp.name) / "leaf.key"

        from cryptography.hazmat.primitives import serialization
        self._leaf_cert_path.write_bytes(
            leaf_cert.public_bytes(serialization.Encoding.PEM)
        )
        self._leaf_key_path.write_bytes(
            leaf_key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.TraditionalOpenSSL,
                serialization.NoEncryption(),
            )
        )

        self.requests: list[str] = []
        self._setup_server(port)

    def _setup_server(self, port: int):
        requests = self.requests

        class _Handler(http.server.BaseHTTPRequestHandler):
            def _respond(self_h):
                requests.append(f"{self_h.command} {self_h.path}")
                length = int(self_h.headers.get("Content-Length") or 0)
                if length:
                    self_h.rfile.read(length)
                body = b"tls-ok"
                self_h.send_response(200)
                self_h.send_header("Content-Length", str(len(body)))
                self_h.end_headers()
                self_h.wfile.write(body)

            do_GET = _respond
            do_POST = _respond

            def log_message(self_h, *args):
                pass

        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(str(self._leaf_cert_path), str(self._leaf_key_path))

        self._server = http.server.ThreadingHTTPServer(("0.0.0.0", port), _Handler)
        self._server.socket = ctx.wrap_socket(
            self._server.socket, server_side=True
        )
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def close(self):
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5.0)
        self._tmp.cleanup()


class TestTLSInterceptionOff(unittest.TestCase):
    """With interception disabled, HTTPS flows are opaque (today's behavior)."""

    def setUp(self):
        if not _real_stack_available() or not _cryptography_available():
            self.skipTest(_SKIP_REASON)
        self.host_ip = _host_lan_ip()
        if self.host_ip is None:
            self.skipTest("no global-scope host address")
        self._tmp = tempfile.TemporaryDirectory(prefix="crab_tls_off_")
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        self.tls_server = _TLSServer()
        self.addCleanup(self.tls_server.close)
        self.engine = Engine.start(
            EngineConfig(
                runtime="runc",
                enable_sandbox_network=True,
                enable_interceptor=False,
                enable_egress_proxy=True,
                egress_tls_interception_enabled=False,
                storage_root=root / "storage",
                runtime_root=root / "runtime",
            )
        )
        self.addCleanup(self.engine.stop)

    def _flows(self, sandbox, *, expected, timeout=20.0):
        deadline = time.monotonic() + timeout
        rows = []
        while time.monotonic() < deadline:
            rows = [r["payload"] for r in sandbox.actions(kind="egress")]
            if len(rows) >= expected:
                return rows
            time.sleep(0.25)
        self.fail(f"expected >={expected} egress flows, saw {len(rows)}: {rows}")

    def test_opaque_flow_sni_only(self):
        """HTTPS flow with interception off → scheme=tls, host=SNI, no method."""
        sandbox = Sandbox(image=_IMAGE, engine=self.engine)
        self.addCleanup(sandbox.kill)

        # The sandbox sends an HTTPS request. TLS handshake will fail (no CA
        # trust) but the proxy still records the SNI from the ClientHello.
        sandbox.commands.run(
            "python3 -c \""
            "import socket, ssl\n"
            "ctx = ssl.create_default_context()\n"
            "ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE\n"
            f"s = socket.create_connection(('{self.host_ip}', {self.tls_server.port}), timeout=10)\n"
            f"wrapped = ctx.wrap_socket(s, server_hostname='{self.tls_server.hostname}')\n"
            "wrapped.sendall(b'GET / HTTP/1.1\\r\\nHost: {host}\\r\\n\\r\\n'.format(host='"
            f"{self.tls_server.hostname}'))\n"
            "wrapped.recv(1024)\n"
            "wrapped.close()\""
        )
        flows = self._flows(sandbox, expected=1)
        tls_flows = [f for f in flows if f.get("scheme") == "tls"]
        self.assertTrue(tls_flows, f"no TLS flow: {flows}")
        flow = tls_flows[0]
        self.assertEqual(flow["host"], self.tls_server.hostname)
        self.assertIsNone(flow.get("method"))
        self.assertIsNone(flow.get("path"))
        self.assertEqual(flow["classification"], "opaque")


class TestTLSInterceptionOnWithCA(unittest.TestCase):
    """With interception on and CA injected, HTTPS is fully classifiable."""

    def setUp(self):
        if not _real_stack_available() or not _cryptography_available():
            self.skipTest(_SKIP_REASON)
        self.host_ip = _host_lan_ip()
        if self.host_ip is None:
            self.skipTest("no global-scope host address")
        self._tmp = tempfile.TemporaryDirectory(prefix="crab_tls_on_")
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        self.tls_server = _TLSServer()
        self.addCleanup(self.tls_server.close)
        self.engine = Engine.start(
            EngineConfig(
                runtime="runc",
                enable_sandbox_network=True,
                enable_interceptor=False,
                enable_egress_proxy=True,
                enable_egress_recording=True,
                egress_tls_interception_enabled=True,
                storage_root=root / "storage",
                runtime_root=root / "runtime",
            )
        )
        self.addCleanup(self.engine.stop)
        _patch_upstream_trust(self.engine, self.tls_server)

    def _flows(self, sandbox, *, expected, timeout=20.0):
        deadline = time.monotonic() + timeout
        rows = []
        while time.monotonic() < deadline:
            rows = [r["payload"] for r in sandbox.actions(kind="egress")]
            if len(rows) >= expected:
                return rows
            time.sleep(0.25)
        self.fail(f"expected >={expected} egress flows, saw {len(rows)}: {rows}")

    def _https_get(self, path: str) -> str:
        """Python script to do HTTPS GET using the injected CA trust (SSL_CERT_FILE)."""
        # Port 443 is default for HTTPS so omit from URL to keep Host header clean.
        port_suffix = "" if self.tls_server.port == 443 else f":{self.tls_server.port}"
        return (
            "python3 -c \""
            "import urllib.request, ssl, os\n"
            "ctx = ssl.create_default_context(cafile=os.environ.get('SSL_CERT_FILE'))\n"
            f"url = 'https://{self.tls_server.hostname}{port_suffix}{path}'\n"
            "print(urllib.request.urlopen(url, timeout=10, context=ctx).read().decode())\""
        )

    def _https_post(self, path: str) -> str:
        """Python script to do HTTPS POST using the injected CA."""
        port_suffix = "" if self.tls_server.port == 443 else f":{self.tls_server.port}"
        return (
            "python3 -c \""
            "import urllib.request, ssl, os\n"
            "ctx = ssl.create_default_context(cafile=os.environ.get('SSL_CERT_FILE'))\n"
            f"url = 'https://{self.tls_server.hostname}{port_suffix}{path}'\n"
            "req = urllib.request.Request(url, data=b'payload', method='POST')\n"
            "print(urllib.request.urlopen(req, timeout=10, context=ctx).status)\""
        )

    def test_https_get_idempotent_read(self):
        """HTTPS GET → classified idempotent_read, recorded, replayable."""
        sandbox = Sandbox(image=_IMAGE, engine=self.engine)
        self.addCleanup(sandbox.kill)

        # Inject CA cert + hosts entry into sandbox.
        _inject_ca_into_sandbox(self.engine, sandbox)
        sandbox.commands.run(
            f"sh -c 'echo \"{self.host_ip} {self.tls_server.hostname}\" >> /etc/hosts'"
        )

        result = sandbox.commands.run(self._https_get("/read"))
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("tls-ok", result.stdout)

        flows = self._flows(sandbox, expected=1)
        flow = flows[0]
        self.assertEqual(flow["scheme"], "https")
        self.assertEqual(flow["method"], "GET")
        self.assertEqual(flow["path"], "/read")
        self.assertEqual(flow["classification"], "idempotent_read")
        self.assertEqual(flow["host"], self.tls_server.hostname)

    def test_https_get_recorded_and_replayable(self):
        """Recorded HTTPS GET can be replayed from cassette."""
        sandbox = Sandbox(image=_IMAGE, engine=self.engine)
        self.addCleanup(sandbox.kill)

        _inject_ca_into_sandbox(self.engine, sandbox)
        sandbox.commands.run(
            f"sh -c 'echo \"{self.host_ip} {self.tls_server.hostname}\" >> /etc/hosts'"
        )

        # First request: recorded
        result = sandbox.commands.run(self._https_get("/replay-test"))
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("tls-ok", result.stdout)
        self._flows(sandbox, expected=1)

        # Replay: should serve from cassette
        with sandbox.replay_egress(policy="cassette_only") as window:
            result2 = sandbox.commands.run(self._https_get("/replay-test"))
            self.assertEqual(result2.returncode, 0, msg=result2.stderr)
            self.assertIn("tls-ok", result2.stdout)
        self.assertIsNotNone(window.report)
        self.assertEqual(window.report.served, 1)
        self.assertEqual(window.report.missed, 0)


class TestTLSForkEffectsReject(unittest.TestCase):
    """fork(effects='reject') blocks HTTPS POST with 503."""

    def setUp(self):
        if not _real_stack_available() or not _cryptography_available():
            self.skipTest(_SKIP_REASON)
        self.host_ip = _host_lan_ip()
        if self.host_ip is None:
            self.skipTest("no global-scope host address")
        self._tmp = tempfile.TemporaryDirectory(prefix="crab_tls_reject_")
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        self.tls_server = _TLSServer()
        self.addCleanup(self.tls_server.close)
        self.engine = Engine.start(
            EngineConfig(
                runtime="runc",
                enable_sandbox_network=True,
                enable_interceptor=False,
                enable_egress_proxy=True,
                egress_tls_interception_enabled=True,
                storage_root=root / "storage",
                runtime_root=root / "runtime",
            )
        )
        self.addCleanup(self.engine.stop)
        _patch_upstream_trust(self.engine, self.tls_server)

    def _https_post(self, path: str) -> str:
        port_suffix = "" if self.tls_server.port == 443 else f":{self.tls_server.port}"
        return (
            "python3 -c \""
            "import urllib.request, ssl, os\n"
            "ctx = ssl.create_default_context(cafile=os.environ.get('SSL_CERT_FILE'))\n"
            f"url = 'https://{self.tls_server.hostname}{port_suffix}{path}'\n"
            "req = urllib.request.Request(url, data=b'payload', method='POST')\n"
            "try:\n"
            "    resp = urllib.request.urlopen(req, timeout=10, context=ctx)\n"
            "    print(resp.status)\n"
            "except urllib.error.HTTPError as e:\n"
            "    print(e.code)\n"
            "    import sys; sys.exit(1)\""
        )

    def test_https_post_rejected_never_reaches_server(self):
        """HTTPS POST from reject-fork → 503, server never sees it."""
        sandbox = Sandbox(image=_IMAGE, engine=self.engine)
        self.addCleanup(sandbox.kill)
        _inject_ca_into_sandbox(self.engine, sandbox)
        sandbox.commands.run(
            f"sh -c 'echo \"{self.host_ip} {self.tls_server.hostname}\" >> /etc/hosts'"
        )

        [fork] = sandbox.fork(effects="reject")
        self.addCleanup(fork.kill)

        result = fork.commands.run(self._https_post("/fork-write"))
        self.assertNotEqual(result.returncode, 0, msg="the fork's write was not refused")
        self.assertIn("503", result.stdout + result.stderr)
        self.assertEqual(
            self.tls_server.requests, [],
            "a gated fork's HTTPS write reached the server"
        )


class TestTLSHandshakeFailurePassthrough(unittest.TestCase):
    """Sandbox that doesn't trust CA → handshake fails, passthrough → opaque."""

    def setUp(self):
        if not _real_stack_available() or not _cryptography_available():
            self.skipTest(_SKIP_REASON)
        self.host_ip = _host_lan_ip()
        if self.host_ip is None:
            self.skipTest("no global-scope host address")
        self._tmp = tempfile.TemporaryDirectory(prefix="crab_tls_passthrough_")
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        self.tls_server = _TLSServer()
        self.addCleanup(self.tls_server.close)
        self.engine = Engine.start(
            EngineConfig(
                runtime="runc",
                enable_sandbox_network=True,
                enable_interceptor=False,
                enable_egress_proxy=True,
                egress_tls_interception_enabled=True,
                egress_tls_on_handshake_failure="passthrough",
                storage_root=root / "storage",
                runtime_root=root / "runtime",
            )
        )
        self.addCleanup(self.engine.stop)
        _patch_upstream_trust(self.engine, self.tls_server)

    def _flows(self, sandbox, *, expected, timeout=20.0):
        deadline = time.monotonic() + timeout
        rows = []
        while time.monotonic() < deadline:
            rows = [r["payload"] for r in sandbox.actions(kind="egress")]
            if len(rows) >= expected:
                return rows
            time.sleep(0.25)
        self.fail(f"expected >={expected} egress flows, saw {len(rows)}: {rows}")

    def test_untrusted_ca_passthrough_opaque(self):
        """Sandbox rejecting minted leaf → handshake fails → retry goes opaque."""
        sandbox = Sandbox(image=_IMAGE, engine=self.engine)
        self.addCleanup(sandbox.kill)
        sandbox.commands.run(
            f"sh -c 'echo \"{self.host_ip} {self.tls_server.hostname}\" >> /etc/hosts'"
        )

        # First attempt: sandbox uses system CA bundle (not the crab CA).
        # The proxy terminates with minted leaf, sandbox rejects it → handshake
        # failure. Under passthrough, the host is added to runtime bypass.
        # Second attempt (retry): goes opaque directly.
        # NOTE: SSL_CERT_FILE is unset so the sandbox uses only system CAs
        # (not the injected crab CA).  This simulates a client that does not
        # trust the interception proxy.
        sandbox.commands.run(
            "python3 -c \""
            "import os, socket, ssl\n"
            "os.environ.pop('SSL_CERT_FILE', None)\n"
            "ctx = ssl.create_default_context()\n"  # system CA only
            f"s = socket.create_connection(('{self.tls_server.hostname}', {self.tls_server.port}), timeout=10)\n"
            "try:\n"
            f"    ctx.wrap_socket(s, server_hostname='{self.tls_server.hostname}')\n"
            "except Exception as e:\n"
            "    print(f'First attempt failed as expected: {{e}}')\n"
            "    s.close()\""
        )
        time.sleep(1)  # let the runtime bypass register

        # Second attempt: should go opaque (bypassed), reaching the real server
        result = sandbox.commands.run(
            "python3 -c \""
            "import socket, ssl\n"
            "ctx = ssl.create_default_context()\n"
            "ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE\n"
            f"s = socket.create_connection(('{self.tls_server.hostname}', {self.tls_server.port}), timeout=10)\n"
            f"wrapped = ctx.wrap_socket(s, server_hostname='{self.tls_server.hostname}')\n"
            f"wrapped.sendall(b'GET / HTTP/1.1\\r\\nHost: {self.tls_server.hostname}\\r\\n\\r\\n')\n"
            "data = wrapped.recv(4096)\n"
            "print(data.decode(errors='replace'))\n"
            "wrapped.close()\""
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)

        # We should have at least one opaque (tls) flow in the journal
        flows = self._flows(sandbox, expected=1)
        tls_flows = [f for f in flows if f.get("scheme") == "tls"]
        self.assertTrue(tls_flows, f"no opaque TLS flow recorded: {flows}")
        self.assertEqual(tls_flows[-1]["classification"], "opaque")


class TestTLSHandshakeFailureRefuse(unittest.TestCase):
    """on_handshake_failure=refuse → flow fails instead of tunnelling."""

    def setUp(self):
        if not _real_stack_available() or not _cryptography_available():
            self.skipTest(_SKIP_REASON)
        self.host_ip = _host_lan_ip()
        if self.host_ip is None:
            self.skipTest("no global-scope host address")
        self._tmp = tempfile.TemporaryDirectory(prefix="crab_tls_refuse_")
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        self.tls_server = _TLSServer()
        self.addCleanup(self.tls_server.close)
        self.engine = Engine.start(
            EngineConfig(
                runtime="runc",
                enable_sandbox_network=True,
                enable_interceptor=False,
                enable_egress_proxy=True,
                egress_tls_interception_enabled=True,
                egress_tls_on_handshake_failure="refuse",
                storage_root=root / "storage",
                runtime_root=root / "runtime",
            )
        )
        self.addCleanup(self.engine.stop)

    def test_refuse_mode_no_tunnel(self):
        """Handshake failure with refuse → connection closed, flow doesn't reach server."""
        sandbox = Sandbox(image=_IMAGE, engine=self.engine)
        self.addCleanup(sandbox.kill)
        sandbox.commands.run(
            f"sh -c 'echo \"{self.host_ip} {self.tls_server.hostname}\" >> /etc/hosts'"
        )

        # Sandbox uses system CA (not the crab CA) — handshake will fail.
        # Under refuse mode: no bypass, just close. Server should NOT see the request.
        # NOTE: SSL_CERT_FILE is unset so the sandbox only trusts system CAs,
        # not the injected crab CA — simulating a client that refuses the proxy cert.
        result = sandbox.commands.run(
            "python3 -c \""
            "import os, socket, ssl\n"
            "os.environ.pop('SSL_CERT_FILE', None)\n"
            "ctx = ssl.create_default_context()\n"  # system CA only
            f"s = socket.create_connection(('{self.tls_server.hostname}', {self.tls_server.port}), timeout=10)\n"
            "try:\n"
            f"    ctx.wrap_socket(s, server_hostname='{self.tls_server.hostname}')\n"
            "    print('SHOULD_NOT_SUCCEED')\n"
            "except Exception as e:\n"
            "    print(f'Failed: {{e}}')\n"
            "    s.close()\""
        )
        # The handshake should fail
        self.assertNotIn("SHOULD_NOT_SUCCEED", result.stdout)
        # Server should not have received any request
        self.assertEqual(self.tls_server.requests, [])


class TestTLSInitProcessCAEnv(unittest.TestCase):
    """Init-process daemon started by sandbox sees the CA env vars."""

    def setUp(self):
        if not _real_stack_available() or not _cryptography_available():
            self.skipTest(_SKIP_REASON)
        self.host_ip = _host_lan_ip()
        if self.host_ip is None:
            self.skipTest("no global-scope host address")
        self._tmp = tempfile.TemporaryDirectory(prefix="crab_tls_env_")
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        self.engine = Engine.start(
            EngineConfig(
                runtime="runc",
                enable_sandbox_network=True,
                enable_interceptor=False,
                enable_egress_proxy=True,
                egress_tls_interception_enabled=True,
                storage_root=root / "storage",
                runtime_root=root / "runtime",
            )
        )
        self.addCleanup(self.engine.stop)

    def test_init_daemon_has_ca_env(self):
        """A command run via commands.run has SSL_CERT_FILE etc."""
        sandbox = Sandbox(image=_IMAGE, engine=self.engine)
        self.addCleanup(sandbox.kill)

        result = sandbox.commands.run("python3 -c 'import os; print(os.environ.get(\"SSL_CERT_FILE\", \"MISSING\"))'")
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertNotEqual(result.stdout.strip(), "MISSING")
        self.assertIn("crab-ca.crt", result.stdout)

        result2 = sandbox.commands.run("python3 -c 'import os; print(os.environ.get(\"REQUESTS_CA_BUNDLE\", \"MISSING\"))'")
        self.assertEqual(result2.returncode, 0, msg=result2.stderr)
        self.assertNotEqual(result2.stdout.strip(), "MISSING")

        result3 = sandbox.commands.run("python3 -c 'import os; print(os.environ.get(\"NODE_EXTRA_CA_CERTS\", \"MISSING\"))'")
        self.assertEqual(result3.returncode, 0, msg=result3.stderr)
        self.assertNotEqual(result3.stdout.strip(), "MISSING")
