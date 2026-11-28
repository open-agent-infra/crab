"""Real-host end-to-end for D1.1 egress interception: the repository's
first ``enable_sandbox_network=True`` test. A sandbox on the bridge
network reaches a host-side "external" service; every TCP flow is
redirected into the egress proxy and lands in the sandbox's journal
with the host/method/scheme the proxy could observe. Self-skipping
outside the crab-dev VM."""
from __future__ import annotations

import base64
import http.server
import json
import os
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path

from crab import Engine, EngineConfig, Sandbox
from crab.txn import TxnNotAbortable


def _real_stack_available() -> bool:
    if os.geteuid() != 0:
        return False
    tools = ("docker", "runc", "criu", "zfs", "iptables", "ip")
    return all(shutil.which(tool) is not None for tool in tools)


def _host_lan_ip() -> str | None:
    """An address of this host that a sandbox on the bridge can reach
    which is NOT the bridge IP itself (host-bound traffic is excluded
    from the redirect on purpose)."""
    result = subprocess.run(
        ["ip", "-4", "-o", "addr", "show", "scope", "global"],
        capture_output=True,
        text=True,
        check=False,
    )
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) < 4 or fields[1].startswith(("acb", "lo", "vh")):
            continue
        return fields[3].split("/")[0]
    return None


class _ExternalHTTPServer:
    """Stands in for the outside world: bound on all interfaces so the
    sandbox reaches it through the bridge (not via the host address the
    redirect rule excludes)."""

    class _Handler(http.server.BaseHTTPRequestHandler):
        def _respond(self) -> None:
            body = b"external-ok"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        do_GET = _respond
        do_POST = _respond

        def log_message(self, *args) -> None:  # silence
            pass

    def __init__(self) -> None:
        self._server = http.server.ThreadingHTTPServer(("0.0.0.0", 0), self._Handler)
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5.0)


class _RawTCPServer:
    """Opaque (non-HTTP, non-TLS) listener for the raw-flow case."""

    def __init__(self) -> None:
        self._sock = socket.socket()
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("0.0.0.0", 0))
        self._sock.listen(4)
        self.port = self._sock.getsockname()[1]
        self.received: list[bytes] = []
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        while True:
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return
            with conn:
                try:
                    self.received.append(conn.recv(256))
                    conn.sendall(b"raw-ok\n")
                except OSError:
                    pass

    def close(self) -> None:
        self._sock.close()
        self._thread.join(timeout=2.0)


class _CountingHTTPServer:
    """Like _ExternalHTTPServer but records what it was asked for, so a
    test can assert a write never arrived."""

    def __init__(self) -> None:
        requests: list[str] = []

        class _Handler(http.server.BaseHTTPRequestHandler):
            def _respond(self) -> None:
                requests.append(f"{self.command} {self.path}")
                length = int(self.headers.get("Content-Length") or 0)
                if length:
                    self.rfile.read(length)
                body = b"external-ok"
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            do_GET = _respond
            do_POST = _respond

            def log_message(self, *args) -> None:
                pass

        self.requests = requests
        self._server = http.server.ThreadingHTTPServer(("0.0.0.0", 0), _Handler)
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5.0)


class EgressRealTests(unittest.TestCase):
    _IMAGE = "python:3.11-slim"

    def setUp(self) -> None:
        if not _real_stack_available():
            self.skipTest("docker/runc/criu/zfs/iptables/root not available")
        self.host_ip = _host_lan_ip()
        if self.host_ip is None:
            self.skipTest("no global-scope host address for the external service")
        self._tmp = tempfile.TemporaryDirectory(prefix="crab_egress_e2e_")
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        self.external = _ExternalHTTPServer()
        self.addCleanup(self.external.close)
        self.engine = Engine.start(
            EngineConfig(
                runtime="runc",
                enable_sandbox_network=True,
                enable_interceptor=False,
                enable_egress_proxy=True,
                storage_root=root / "storage",
                runtime_root=root / "runtime",
            )
        )
        self.addCleanup(self.engine.stop)

    def _run(self, sandbox: Sandbox, script: str) -> str:
        result = sandbox.commands.run(script)
        self.assertEqual(result.returncode, 0, msg=f"command failed: {script!r}: {result.stderr}")
        return result.stdout.strip()

    def _flows(self, sandbox: Sandbox, *, expected: int, timeout: float = 20.0) -> list[dict]:
        deadline = time.monotonic() + timeout
        rows: list[dict] = []
        while time.monotonic() < deadline:
            rows = [row["payload"] for row in sandbox.actions(kind="egress")]
            if len(rows) >= expected:
                return rows
            time.sleep(0.25)
        self.fail(f"expected >={expected} egress flows, saw {len(rows)}: {rows}")

    def test_all_sandbox_egress_lands_in_the_ledger(self) -> None:
        sandbox = Sandbox(image=self._IMAGE, engine=self.engine)
        self.addCleanup(sandbox.kill)
        # Egress interception only works when the sandbox sits in the
        # bridge netns (the REDIRECT hook point) — assert it up front so a
        # host-networked sandbox can never masquerade as a passing test.
        self.assertIsNotNone(
            self.engine._network_manager.lease_for(sandbox.sandbox_id),
            "sandbox has no bridge network lease; egress would bypass the proxy",
        )
        base = f"http://{self.host_ip}:{self.external.port}"

        # 1. plain HTTP GET + POST through the redirect
        self.assertIn(
            "external-ok",
            self._run(sandbox, f"python3 -c \"import urllib.request;print(urllib.request.urlopen('{base}/read').read().decode())\""),
        )
        self._run(
            sandbox,
            f"python3 -c \"import urllib.request;urllib.request.urlopen(urllib.request.Request('{base}/write', data=b'x', method='POST')).read()\"",
        )
        flows = self._flows(sandbox, expected=2)
        by_path = {flow.get("path"): flow for flow in flows if flow.get("path")}
        self.assertIn("/read", by_path, msg=str(flows))
        self.assertIn("/write", by_path, msg=str(flows))
        self.assertEqual(by_path["/read"]["method"], "GET")
        self.assertEqual(by_path["/write"]["method"], "POST")
        for path in ("/read", "/write"):
            flow = by_path[path]
            self.assertEqual(flow["scheme"], "http")
            self.assertEqual(flow["dst_ip"], self.host_ip)
            self.assertEqual(flow["dst_port"], self.external.port)
            self.assertIn(self.host_ip, flow["host"])  # Host header carries ip:port
            self.assertGreater(flow["bytes_out"], 0)
            self.assertGreater(flow["bytes_in"], 0)
        self.assertEqual(by_path["/read"]["classification"], "idempotent_read")
        self.assertEqual(by_path["/write"]["classification"], "mutating")

        # The ledger view (D1.2) reads the same rows with counts.
        ledger = sandbox.egress()
        self.assertGreaterEqual(ledger.total, 2)
        self.assertGreaterEqual(ledger.idempotent_reads, 1)
        self.assertGreaterEqual(ledger.mutating, 1)
        self.assertIn(self.host_ip, " ".join(ledger.hosts))

    def test_tls_sni_and_raw_tcp_are_recorded(self) -> None:
        sandbox = Sandbox(image=self._IMAGE, engine=self.engine)
        self.addCleanup(sandbox.kill)
        raw = _RawTCPServer()
        self.addCleanup(raw.close)

        # TLS: the handshake fails (no server), but the ClientHello's SNI
        # is what the ledger needs — no decryption involved.
        sandbox.commands.run(
            "python3 -c \""
            "import socket,ssl\n"
            "ctx=ssl.create_default_context()\n"
            "ctx.check_hostname=False; ctx.verify_mode=ssl.CERT_NONE\n"
            f"s=socket.create_connection(('{self.host_ip}',{self.external.port}),timeout=5)\n"
            "try:\n"
            "    ctx.wrap_socket(s,server_hostname='secure.example.com').close()\n"
            "except Exception: pass\""
        )
        self._run(
            sandbox,
            f"python3 -c \"import socket;s=socket.create_connection(('{self.host_ip}',{raw.port}),timeout=5);s.sendall(b'\\x01\\x02opaque');print(s.recv(16).decode().strip());s.close()\"",
        )

        flows = self._flows(sandbox, expected=2)
        tls = [flow for flow in flows if flow["scheme"] == "tls"]
        opaque = [flow for flow in flows if flow["dst_port"] == raw.port]
        self.assertTrue(tls, msg=f"no TLS flow recorded: {flows}")
        self.assertEqual(tls[0]["host"], "secure.example.com")
        self.assertIsNone(tls[0]["method"])
        self.assertTrue(opaque, msg=f"no raw TCP flow recorded: {flows}")
        self.assertEqual(opaque[0]["scheme"], "tcp")
        self.assertEqual(opaque[0]["host"], self.host_ip)
        self.assertTrue(raw.received and raw.received[0].endswith(b"opaque"))

    def test_txn_correlation(self) -> None:
        sandbox = Sandbox(image=self._IMAGE, engine=self.engine)
        self.addCleanup(sandbox.kill)
        base = f"http://{self.host_ip}:{self.external.port}"
        txn = sandbox.begin("egress-txn")
        txn.exec(
            f"python3 -c \"import urllib.request;urllib.request.urlopen(urllib.request.Request('{base}/in-txn', data=b'x', method='POST')).read()\""
        )
        self._flows(sandbox, expected=1)
        # Scoped ledger view sees exactly this txn's mutating flow.
        scoped = sandbox.egress(txn_id=txn.txn_id)
        self.assertEqual(scoped.mutating, 1)
        self.assertEqual(scoped.flows[0].path, "/in-txn")
        self.assertEqual(scoped.flows[0].txn_id, txn.txn_id)
        txn.commit()
        rows = [row for row in sandbox.actions(kind="egress") if row["payload"].get("path") == "/in-txn"]
        self.assertTrue(rows, "in-txn flow missing from the ledger")
        self.assertEqual(rows[0]["txn_id"], txn.txn_id)

    def test_abort_reports_mutating_egress_it_cannot_undo(self) -> None:
        sandbox = Sandbox(image=self._IMAGE, engine=self.engine)
        self.addCleanup(sandbox.kill)
        base = f"http://{self.host_ip}:{self.external.port}"
        txn = sandbox.begin("egress-abort")
        txn.exec(
            f"python3 -c \"import urllib.request;urllib.request.urlopen(urllib.request.Request('{base}/fired', data=b'x', method='POST')).read()\""
        )
        self._flows(sandbox, expected=1)
        result = txn.abort()
        # The filesystem rolled back; the POST already left the machine.
        self.assertEqual(result.mutating_egress, 1)

    def test_redirect_rule_is_scoped_and_removed(self) -> None:
        # The proxy must listen on the bridge address only: REDIRECT
        # rewrites destinations to it, and a wildcard bind would expose
        # the proxy to anything that can reach this host.
        bridge_ip = self.engine._network_manager.bridge_ip
        listen_addr = self.engine._egress_proxy.server_address
        self.assertEqual(listen_addr[0], bridge_ip)
        listeners = subprocess.run(
            ["ss", "-ltn"], capture_output=True, text=True, check=False
        ).stdout
        self.assertNotIn(
            f"0.0.0.0:{listen_addr[1]}",
            listeners,
            "egress proxy is bound on every interface",
        )
        rules = subprocess.run(
            ["iptables", "-t", "nat", "-S", "PREROUTING"],
            capture_output=True, text=True, check=True,
        ).stdout
        redirects = [line for line in rules.splitlines() if "REDIRECT" in line]
        self.assertTrue(redirects, "no REDIRECT rule installed")
        # Host-bound traffic must be excluded, or the LLM interceptor path
        # would be swallowed by the proxy.
        self.assertTrue(
            all("! -d " in line for line in redirects),
            f"redirect rule is not host-scoped: {redirects}",
        )
        self.engine.stop()
        after = subprocess.run(
            ["iptables", "-t", "nat", "-S", "PREROUTING"],
            capture_output=True, text=True, check=True,
        ).stdout
        self.assertNotIn("REDIRECT", after, "redirect rule leaked after engine stop")


class EgressLlmUnchangedRealTests(unittest.TestCase):
    """Exit criterion: with the egress proxy on, the LLM interception
    path behaves exactly as before (host-bound traffic is excluded from
    the redirect)."""

    _IMAGE = "python:3.11-slim"

    def setUp(self) -> None:
        if not _real_stack_available():
            self.skipTest("docker/runc/criu/zfs/iptables/root not available")
        self._tmp = tempfile.TemporaryDirectory(prefix="crab_egress_llm_")
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        self.upstream_hits: list[str] = []
        self.upstream = self._start_upstream()
        self.addCleanup(self.upstream.close)
        self.engine = Engine.start(
            EngineConfig(
                runtime="runc",
                enable_sandbox_network=True,
                enable_interceptor=True,
                enable_egress_proxy=True,
                storage_root=root / "storage",
                runtime_root=root / "runtime",
            )
        )
        self.addCleanup(self.engine.stop)

    def _start_upstream(self):
        hits = self.upstream_hits

        class _Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                hits.append(self.path)
                length = int(self.headers.get("Content-Length") or 0)
                self.rfile.read(length)
                body = b'{"id":"cmpl-1","choices":[{"message":{"content":"pong"}}]}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args) -> None:
                pass

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()

        class _Holder:
            def __init__(self, srv):
                self._srv = srv
                self.port = srv.server_address[1]

            def close(self):
                self._srv.shutdown()
                self._srv.server_close()

        return _Holder(server)

    def test_llm_request_still_reaches_the_interceptor(self) -> None:
        sandbox = Sandbox(image=self._IMAGE, engine=self.engine)
        self.addCleanup(sandbox.kill)
        self.assertIsNotNone(
            self.engine._network_manager.lease_for(sandbox.sandbox_id),
            "sandbox has no bridge lease; the ledger check would be vacuous",
        )
        self.engine.register_upstream(
            sandbox.sandbox_id, f"http://127.0.0.1:{self.upstream.port}"
        )
        interceptor_url = self.engine.interceptor_base_url
        self.assertIsNotNone(interceptor_url)
        result = sandbox.commands.run(
            "python3 -c \"import json,urllib.request;"
            f"req=urllib.request.Request('{interceptor_url}/v1/chat/completions',"
            "data=json.dumps({'model':'m','messages':[{'role':'user','content':'ping'}]}).encode(),"
            "headers={'Content-Type':'application/json'});"
            "print(urllib.request.urlopen(req, timeout=30).read().decode())\""
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("pong", result.stdout)
        self.assertEqual(self.upstream_hits, ["/v1/chat/completions"])
        # The interceptor path is host-bound, so it must NOT be in the
        # egress ledger (the redirect rule excludes it).
        time.sleep(1.0)
        ledger_hosts = [row["payload"]["host"] for row in sandbox.actions(kind="egress")]
        self.assertFalse(
            any("127.0.0.1" in host for host in ledger_hosts),
            f"interceptor traffic was redirected into the proxy: {ledger_hosts}",
        )


class EgressRecordingRealTests(unittest.TestCase):
    """D2.1 E2E: an idempotent read's bodies land in a cassette on disk,
    credentials do not, and mutating flows are never recorded."""

    _IMAGE = "python:3.11-slim"

    def setUp(self) -> None:
        if not _real_stack_available():
            self.skipTest("docker/runc/criu/zfs/iptables/root not available")
        self.host_ip = _host_lan_ip()
        if self.host_ip is None:
            self.skipTest("no global-scope host address for the external service")
        self._tmp = tempfile.TemporaryDirectory(prefix="crab_rec_e2e_")
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.external = _ExternalHTTPServer()
        self.addCleanup(self.external.close)
        self.engine = Engine.start(
            EngineConfig(
                runtime="runc",
                enable_sandbox_network=True,
                enable_interceptor=False,
                enable_egress_proxy=True,
                enable_egress_recording=True,
                storage_root=self.root / "storage",
                runtime_root=self.root / "runtime",
            )
        )
        self.addCleanup(self.engine.stop)
        self.cassettes = self.root / "storage" / "cassettes"

    def _run(self, sandbox: Sandbox, script: str) -> str:
        result = sandbox.commands.run(script)
        self.assertEqual(result.returncode, 0, msg=f"command failed: {script!r}: {result.stderr}")
        return result.stdout.strip()

    def _recorded_flows(self, sandbox: Sandbox, *, timeout: float = 20.0) -> list:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            flows = [flow for flow in sandbox.egress().flows if flow.recorded]
            if flows:
                return flows
            time.sleep(0.25)
        self.fail(f"no recorded flows: {[f.to_json() for f in sandbox.egress().flows]}")

    def test_read_bodies_are_recorded_writes_are_not(self) -> None:
        sandbox = Sandbox(image=self._IMAGE, engine=self.engine)
        self.addCleanup(sandbox.kill)
        base = f"http://{self.host_ip}:{self.external.port}"

        # An idempotent read carrying a credential header.
        self.assertIn(
            "external-ok",
            self._run(
                sandbox,
                "python3 -c \"import urllib.request;"
                f"req=urllib.request.Request('{base}/read', headers={{'Authorization':'Bearer super-secret'}});"
                "print(urllib.request.urlopen(req).read().decode())\"",
            ),
        )
        # A write, which must never be recorded.
        self._run(
            sandbox,
            f"python3 -c \"import urllib.request;urllib.request.urlopen(urllib.request.Request('{base}/write', data=b'x', method='POST')).read()\"",
        )

        [recorded] = self._recorded_flows(sandbox)
        self.assertEqual(recorded.path, "/read")
        self.assertEqual(recorded.status, 200)
        self.assertFalse(recorded.truncated)
        ledger = sandbox.egress()
        self.assertEqual(ledger.recorded, 1)      # the POST is absent
        self.assertGreaterEqual(ledger.mutating, 1)
        self.assertEqual(ledger.replayed, 0)      # nothing served yet (D2.2)

        # The body is on disk (base64-encoded), the credential is not.
        entries = list(self.cassettes.rglob("*.json"))
        self.assertTrue(entries, "no cassette written")
        blob = "\n".join(path.read_text(encoding="utf-8") for path in entries)
        self.assertNotIn("super-secret", blob)
        self.assertNotIn("/write", blob)
        stored = [json.loads(path.read_text(encoding="utf-8")) for path in entries]
        bodies = [base64.b64decode(entry["body_b64"]) for entry in stored]
        self.assertIn(b"external-ok", bodies)
        header_names = {
            name.lower() for entry in stored for name, _ in entry["response_headers"]
        }
        self.assertNotIn("authorization", header_names)
        self.assertNotIn("set-cookie", header_names)

    def test_cassettes_are_pruned_when_the_sandbox_dies(self) -> None:
        sandbox = Sandbox(image=self._IMAGE, engine=self.engine)
        base = f"http://{self.host_ip}:{self.external.port}"
        self._run(
            sandbox,
            f"python3 -c \"import urllib.request;urllib.request.urlopen('{base}/read').read()\"",
        )
        self._recorded_flows(sandbox)
        bucket = self.cassettes / str(sandbox.sandbox_id)
        self.assertTrue(bucket.is_dir())
        sandbox.kill()
        # Replay must therefore happen before the recording sandbox dies.
        self.assertFalse(bucket.exists())


class EgressReplayRealTests(unittest.TestCase):
    """D2.2 E2E: the decisive proof is stopping the origin server and
    still getting the body — it can only have come from a cassette."""

    _IMAGE = "python:3.11-slim"

    def setUp(self) -> None:
        if not _real_stack_available():
            self.skipTest("docker/runc/criu/zfs/iptables/root not available")
        self.host_ip = _host_lan_ip()
        if self.host_ip is None:
            self.skipTest("no global-scope host address for the external service")
        self._tmp = tempfile.TemporaryDirectory(prefix="crab_replay_e2e_")
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.external = _ExternalHTTPServer()
        self.port = self.external.port
        self.engine = Engine.start(
            EngineConfig(
                runtime="runc",
                enable_sandbox_network=True,
                enable_interceptor=False,
                enable_egress_proxy=True,
                enable_egress_recording=True,
                storage_root=self.root / "storage",
                runtime_root=self.root / "runtime",
            )
        )
        self.addCleanup(self.engine.stop)

    def _run(self, sandbox: Sandbox, script: str):
        return sandbox.commands.run(script)

    def _ok(self, sandbox: Sandbox, script: str) -> str:
        result = self._run(sandbox, script)
        self.assertEqual(result.returncode, 0, msg=f"command failed: {script!r}: {result.stderr}")
        return result.stdout.strip()

    def _fetch(self, path: str) -> str:
        return (
            "python3 -c \"import urllib.request;"
            f"print(urllib.request.urlopen('http://{self.host_ip}:{self.port}{path}', timeout=10).read().decode())\""
        )

    def _wait_recorded(self, sandbox: Sandbox, *, timeout: float = 20.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if any(flow.recorded for flow in sandbox.egress().flows):
                return
            time.sleep(0.25)
        self.fail("nothing was recorded")

    def test_replay_serves_the_cassette_with_the_origin_stopped(self) -> None:
        sandbox = Sandbox(image=self._IMAGE, engine=self.engine)
        self.addCleanup(sandbox.kill)
        self.assertIn("external-ok", self._ok(sandbox, self._fetch("/read")))
        self._wait_recorded(sandbox)

        # Kill the origin: any live request now fails.
        self.external.close()
        failed = self._run(sandbox, self._fetch("/read"))
        self.assertNotEqual(failed.returncode, 0, msg="origin still reachable")

        with sandbox.replay_egress(policy="cassette_only") as window:
            self.assertIn("external-ok", self._ok(sandbox, self._fetch("/read")))
        self.assertIsNotNone(window.report)
        self.assertEqual(window.report.served, 1)
        self.assertEqual(window.report.missed, 0)
        ledger = sandbox.egress()
        self.assertGreaterEqual(ledger.replayed, 1)

    def test_c4_replay_uses_the_forks_cassettes(self) -> None:
        sandbox = Sandbox(image=self._IMAGE, engine=self.engine)
        self.addCleanup(sandbox.kill)
        self._ok(sandbox, "mkdir -p /probe && sh -c 'nohup sleep 300 >/dev/null 2>&1 & echo $! > /probe/bg.pid'")

        [fork] = sandbox.fork()
        self.addCleanup(fork.kill)
        # The fork does the reading and writes what it read.
        self._ok(
            fork,
            "python3 -c \"import urllib.request;"
            f"open('/probe/fetched.txt','w').write(urllib.request.urlopen('http://{self.host_ip}:{self.port}/read', timeout=10).read().decode())\"",
        )
        self._wait_recorded(fork)

        # Origin down: a live replay would fail, so zero deviations can
        # only mean the fork's cassettes answered on the source.
        self.external.close()
        report = sandbox.merge_processes(fork, strategy="replay")

        self.assertEqual(report.strategy, "replay")
        self.assertEqual(report.deviations, 0, msg=str([e.argv for e in report.replayed]))
        self.assertIsNotNone(report.egress_replay)
        self.assertGreaterEqual(report.egress_replay.served, 1)
        self.assertEqual(report.egress_replay.cassette_source, str(fork.sandbox_id))
        self.assertEqual(self._ok(sandbox, "cat /probe/fetched.txt"), "external-ok")
        # The source's background process survived the whole thing.
        self.assertEqual(
            self._ok(sandbox, "test -d /proc/$(cat /probe/bg.pid) && echo alive"), "alive"
        )

    def test_writes_are_never_swallowed_by_replay(self) -> None:
        sandbox = Sandbox(image=self._IMAGE, engine=self.engine)
        self.addCleanup(sandbox.kill)
        with sandbox.replay_egress(policy="cassette_first") as window:
            self._ok(
                sandbox,
                "python3 -c \"import urllib.request;"
                f"urllib.request.urlopen(urllib.request.Request('http://{self.host_ip}:{self.port}/write', data=b'x', method='POST'), timeout=10).read()\"",
            )
        self.assertEqual(window.report.served, 0)
        self.assertGreaterEqual(window.report.passed_through, 1)


class EffectGateRealTests(unittest.TestCase):
    """D3.1 E2E: the decisive property is the external server's request
    log — a refused or deferred write must not appear in it."""

    _IMAGE = "python:3.11-slim"

    def setUp(self) -> None:
        if not _real_stack_available():
            self.skipTest("docker/runc/criu/zfs/iptables/root not available")
        self.host_ip = _host_lan_ip()
        if self.host_ip is None:
            self.skipTest("no global-scope host address for the external service")
        self._tmp = tempfile.TemporaryDirectory(prefix="crab_effects_e2e_")
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        self.external = _CountingHTTPServer()
        self.addCleanup(self.external.close)
        self.port = self.external.port
        self.engine = Engine.start(
            EngineConfig(
                runtime="runc",
                enable_sandbox_network=True,
                enable_interceptor=False,
                enable_egress_proxy=True,
                storage_root=root / "storage",
                runtime_root=root / "runtime",
            )
        )
        self.addCleanup(self.engine.stop)
        self.gate = self.engine._system.effect_gate
        self.assertIsNotNone(self.gate, "engine did not wire the effect gate")

    def _post(self, path: str) -> str:
        return (
            "python3 -c \"import urllib.request;"
            f"req=urllib.request.Request('http://{self.host_ip}:{self.port}{path}', data=b'payload', method='POST');"
            "print(urllib.request.urlopen(req, timeout=10).status)\""
        )

    def _get(self, path: str) -> str:
        return (
            "python3 -c \"import urllib.request;"
            f"print(urllib.request.urlopen('http://{self.host_ip}:{self.port}{path}', timeout=10).read().decode())\""
        )

    def _wait_flows(self, sandbox: Sandbox, *, expected: int, timeout: float = 20.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            flows = sandbox.egress().flows
            if len(flows) >= expected:
                return flows
            time.sleep(0.25)
        self.fail(f"expected {expected} flows, saw {[f.to_json() for f in sandbox.egress().flows]}")

    def test_reject_refuses_the_write_and_the_server_never_sees_it(self) -> None:
        sandbox = Sandbox(image=self._IMAGE, engine=self.engine)
        self.addCleanup(sandbox.kill)
        self.gate.begin(sandbox.sandbox_id, policy="reject", txn_id="txn-e2e")

        result = sandbox.commands.run(self._post("/write"))
        self.assertNotEqual(result.returncode, 0, msg="the write was not refused")
        self.assertIn("503", result.stderr + result.stdout)
        self.assertEqual(self.external.requests, [], "a refused write reached the server")

        # Reads in the same window still work.
        read = sandbox.commands.run(self._get("/read"))
        self.assertEqual(read.returncode, 0, msg=read.stderr)
        self.assertIn("external-ok", read.stdout)
        self.assertEqual(self.external.requests, ["GET /read"])

        flows = self._wait_flows(sandbox, expected=2)
        effects = [flow.effect for flow in flows]
        self.assertIn("rejected", effects)
        self.assertEqual(sandbox.egress().rejected, 1)

    def test_defer_answers_202_and_holds_the_write(self) -> None:
        sandbox = Sandbox(image=self._IMAGE, engine=self.engine)
        self.addCleanup(sandbox.kill)
        from crab.effects import EffectRule

        self.gate.begin(
            sandbox.sandbox_id,
            policy="defer",
            rules=(EffectRule(host_glob="*", method="POST"),),
            txn_id="txn-defer",
        )
        result = sandbox.commands.run(self._post("/write"))
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("202", result.stdout)  # the sandbox saw Accepted
        self.assertEqual(self.external.requests, [], "a deferred write was sent anyway")

        [flow] = [f for f in self._wait_flows(sandbox, expected=1) if f.effect]
        self.assertEqual(flow.effect, "deferred")
        self.assertEqual(sandbox.egress().deferred, 1)
        # The queue holds the real request, body and all (D3.2 flushes it).
        [queued] = self.gate.drain(sandbox.sandbox_id)
        self.assertEqual((queued.method, queued.path), ("POST", "/write"))
        self.assertEqual(queued.body, b"payload")
        self.assertEqual(queued.txn_id, "txn-defer")

    def test_without_a_session_writes_flow_as_before(self) -> None:
        sandbox = Sandbox(image=self._IMAGE, engine=self.engine)
        self.addCleanup(sandbox.kill)
        result = sandbox.commands.run(self._post("/write"))
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertEqual(self.external.requests, ["POST /write"])


class StandaloneForkEffectRealTests(unittest.TestCase):
    """F1 E2E: the external server's request log is the verdict. A fork
    created with ``effects="reject"`` must not reach it; a fork created
    without the argument still must (that default is the pinned narrowing of
    D3 decision 10, and gating it would break RL-rollout style callers)."""

    _IMAGE = "python:3.11-slim"

    def setUp(self) -> None:
        if not _real_stack_available():
            self.skipTest("docker/runc/criu/zfs/iptables/root not available")
        self.host_ip = _host_lan_ip()
        if self.host_ip is None:
            self.skipTest("no global-scope host address for the external service")
        self._tmp = tempfile.TemporaryDirectory(prefix="crab_fork_effects_")
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        self.external = _CountingHTTPServer()
        self.addCleanup(self.external.close)
        self.port = self.external.port
        self.engine = Engine.start(
            EngineConfig(
                runtime="runc",
                enable_sandbox_network=True,
                enable_interceptor=False,
                enable_egress_proxy=True,
                storage_root=root / "storage",
                runtime_root=root / "runtime",
            )
        )
        self.addCleanup(self.engine.stop)

    def _post(self, path: str) -> str:
        return (
            "python3 -c \"import urllib.request;"
            f"req=urllib.request.Request('http://{self.host_ip}:{self.port}{path}', data=b'payload', method='POST');"
            "print(urllib.request.urlopen(req, timeout=10).status)\""
        )

    def _get(self, path: str) -> str:
        return (
            "python3 -c \"import urllib.request;"
            f"print(urllib.request.urlopen('http://{self.host_ip}:{self.port}{path}', timeout=10).read().decode())\""
        )

    def test_a_gated_fork_cannot_write_while_the_source_still_can(self) -> None:
        sandbox = Sandbox(image=self._IMAGE, engine=self.engine)
        self.addCleanup(sandbox.kill)
        [fork] = sandbox.fork(effects="reject")
        self.addCleanup(fork.kill)

        refused = fork.commands.run(self._post("/fork-write"))
        self.assertNotEqual(refused.returncode, 0, msg="the fork's write was not refused")
        self.assertIn("503", refused.stderr + refused.stdout)
        self.assertEqual(
            self.external.requests, [], "a gated fork's write reached the server"
        )

        # Reads from the gated fork still work (D3 decision 9).
        read = fork.commands.run(self._get("/fork-read"))
        self.assertEqual(read.returncode, 0, msg=read.stderr)
        self.assertEqual(self.external.requests, ["GET /fork-read"])

        # The session is per sandbox: the source was never gated.
        source_write = sandbox.commands.run(self._post("/source-write"))
        self.assertEqual(source_write.returncode, 0, msg=source_write.stderr)
        self.assertIn("POST /source-write", self.external.requests)

        self.assertEqual(fork.egress().rejected, 1)

    def test_an_ungated_fork_still_writes(self) -> None:
        # The regression guard for F1 decision 4: omitting `effects` keeps
        # today's behavior, so independent branches are unaffected.
        sandbox = Sandbox(image=self._IMAGE, engine=self.engine)
        self.addCleanup(sandbox.kill)
        [fork] = sandbox.fork()
        self.addCleanup(fork.kill)

        result = fork.commands.run(self._post("/ungated"))
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("POST /ungated", self.external.requests)

    def test_defer_and_seal_are_refused_before_forking(self) -> None:
        sandbox = Sandbox(image=self._IMAGE, engine=self.engine)
        self.addCleanup(sandbox.kill)
        for policy in ("defer", "seal"):
            with self.assertRaises(ValueError):
                sandbox.fork(effects=policy)


class EffectTxnRealTests(unittest.TestCase):
    """D3.2 E2E: the external server's request log decides everything — a
    deferred write appears there only after commit, never after abort."""

    _IMAGE = "python:3.11-slim"

    def setUp(self) -> None:
        if not _real_stack_available():
            self.skipTest("docker/runc/criu/zfs/iptables/root not available")
        self.host_ip = _host_lan_ip()
        if self.host_ip is None:
            self.skipTest("no global-scope host address for the external service")
        self._tmp = tempfile.TemporaryDirectory(prefix="crab_effects_txn_")
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        self.external = _CountingHTTPServer()
        self.addCleanup(self.external.close)
        self.port = self.external.port
        self.engine = Engine.start(
            EngineConfig(
                runtime="runc",
                enable_sandbox_network=True,
                enable_interceptor=False,
                enable_egress_proxy=True,
                effects_rules=({"host_glob": "*", "method": "POST"},),
                storage_root=root / "storage",
                runtime_root=root / "runtime",
            )
        )
        self.addCleanup(self.engine.stop)

    def _post(self, path: str) -> str:
        return (
            "python3 -c \"import urllib.request;"
            f"req=urllib.request.Request('http://{self.host_ip}:{self.port}{path}', data=b'payload', method='POST');"
            "print(urllib.request.urlopen(req, timeout=10).status)\""
        )

    def _ok(self, sandbox: Sandbox, script: str) -> str:
        result = sandbox.commands.run(script)
        self.assertEqual(result.returncode, 0, msg=f"{script!r}: {result.stderr}")
        return result.stdout.strip()

    def test_deferred_write_fires_on_commit_only(self) -> None:
        sandbox = Sandbox(image=self._IMAGE, engine=self.engine)
        self.addCleanup(sandbox.kill)
        txn = sandbox.begin("defer-commit", effects="defer")
        self.assertEqual(sandbox.current_txn().effects, "defer")

        self.assertIn("202", self._ok(sandbox, self._post("/write")))
        self.assertEqual(self.external.requests, [], "a deferred write escaped early")

        result = txn.commit()
        self.assertIsNotNone(result.effects)
        self.assertEqual(
            (result.effects.attempted, result.effects.succeeded, result.effects.failed),
            (1, 1, 0),
        )
        # Exactly once, and only after the commit.
        self.assertEqual(self.external.requests, ["POST /write"])
        ledger = sandbox.egress()
        self.assertGreaterEqual(ledger.deferred, 1)
        self.assertGreaterEqual(ledger.flushed, 1)

    def test_abort_drops_the_write_entirely(self) -> None:
        sandbox = Sandbox(image=self._IMAGE, engine=self.engine)
        self.addCleanup(sandbox.kill)
        txn = sandbox.begin("defer-abort", effects="defer")
        self.assertIn("202", self._ok(sandbox, self._post("/write")))

        result = txn.abort()
        self.assertEqual(result.deferred_dropped, 1)
        # The whole point of defer: an aborted txn leaves the world alone.
        self.assertEqual(self.external.requests, [])
        self.assertEqual(result.mutating_egress, 0)

    def test_flush_failure_is_reported_and_commit_stands(self) -> None:
        sandbox = Sandbox(image=self._IMAGE, engine=self.engine)
        self.addCleanup(sandbox.kill)
        self._ok(sandbox, "mkdir -p /probe && echo committed > /probe/state.txt")
        txn = sandbox.begin("defer-flush-fail", effects="defer")
        self.assertIn("202", self._ok(sandbox, self._post("/write")))

        # Real disconnect: the origin is gone before the flush runs.
        self.external.close()
        result = txn.commit()

        self.assertIsNotNone(result.effects)
        self.assertEqual(result.effects.failed, 1)
        self.assertIsNotNone(result.effects.entries[0].error)
        # The filesystem commit stands: a flush failure never unwinds it.
        self.assertEqual(self._ok(sandbox, "cat /probe/state.txt"), "committed")
        self.assertIsNone(sandbox.current_txn())
        rows = [row.effect for row in sandbox.egress().flows if row.effect]
        self.assertIn("flush_failed", rows)

    def test_seal_blocks_abort_until_forced(self) -> None:
        sandbox = Sandbox(image=self._IMAGE, engine=self.engine)
        self.addCleanup(sandbox.kill)
        txn = sandbox.begin("seal", effects="seal")
        self.assertIn("200", self._ok(sandbox, self._post("/write")))
        self.assertEqual(self.external.requests, ["POST /write"])  # seal lets it out

        with self.assertRaises(TxnNotAbortable):
            txn.abort()
        # Forcing accepts that the external write stands.
        result = txn.abort(force=True)
        self.assertEqual(result.txn_id, txn.txn_id)

    def test_fork_txn_refuses_writes_by_default(self) -> None:
        sandbox = Sandbox(image=self._IMAGE, engine=self.engine)
        self.addCleanup(sandbox.kill)
        txn = sandbox.begin("fork-default", isolation="fork")
        self.assertEqual(sandbox.current_txn().effects, "reject")
        result = txn.exec(self._post("/write"))
        self.assertNotEqual(result.returncode, 0, msg="a fork write was allowed")
        self.assertEqual(self.external.requests, [])
        txn.abort()

    def test_fork_with_defer_is_refused_at_begin(self) -> None:
        sandbox = Sandbox(image=self._IMAGE, engine=self.engine)
        self.addCleanup(sandbox.kill)
        with self.assertRaises(ValueError):
            sandbox.begin("nope", isolation="fork", effects="defer")

    def test_c4_replay_does_not_refire_the_forks_write(self) -> None:
        sandbox = Sandbox(image=self._IMAGE, engine=self.engine)
        self.addCleanup(sandbox.kill)
        self._ok(sandbox, "mkdir -p /probe && sh -c 'nohup sleep 300 >/dev/null 2>&1 & echo $! > /probe/bg.pid'")
        [fork] = sandbox.fork()
        self.addCleanup(fork.kill)
        self._ok(fork, self._post("/write"))
        self.assertEqual(self.external.requests, ["POST /write"])  # the fork's write

        report = sandbox.merge_processes(fork, strategy="replay")

        # Replay refuses the write instead of firing it a second time, and
        # says so as a deviation rather than hiding it.
        self.assertEqual(self.external.requests, ["POST /write"])
        self.assertGreaterEqual(report.deviations, 1)


if __name__ == "__main__":
    unittest.main()
