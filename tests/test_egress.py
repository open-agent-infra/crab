"""Unit tests for D1.1 egress interception: sniffers, SO_ORIGINAL_DST
decoding, the proxy's splice/attribution/journal behavior (over
loopback sockets, no netfilter), the bridge redirect rule shape, and
the engine config gates. Host-runnable."""
from __future__ import annotations

import contextlib
import io
import socket
import struct
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from crab.egress import (
    EgressFlowRecorder,
    EgressProxyServer,
    _strip_port_for_sni,
    parse_original_dst,
    sniff_http_head,
    sniff_tls_sni,
)
from crab.ids import SandboxId
from crab.journal import ActionJournal


def _client_hello(server_name: bytes | None) -> bytes:
    """A minimal but structurally valid TLS 1.2 ClientHello."""
    extensions = b""
    if server_name is not None:
        name_entry = b"\x00" + struct.pack("!H", len(server_name)) + server_name
        sni_body = struct.pack("!H", len(name_entry)) + name_entry
        extensions = struct.pack("!HH", 0x0000, len(sni_body)) + sni_body
    body = (
        b"\x03\x03"  # client_version
        + b"\x11" * 32  # random
        + b"\x00"  # session id length
        + struct.pack("!H", 2) + b"\x13\x01"  # cipher suites
        + b"\x01\x00"  # compression methods
        + struct.pack("!H", len(extensions)) + extensions
    )
    handshake = b"\x01" + len(body).to_bytes(3, "big") + body
    return b"\x16\x03\x01" + struct.pack("!H", len(handshake)) + handshake


class SnifferTests(unittest.TestCase):
    def test_http_head(self) -> None:
        head = b"POST /v1/things?a=b HTTP/1.1\r\nHost: api.example.com\r\nX: y\r\n\r\n{}"
        self.assertEqual(
            sniff_http_head(head), ("POST", "/v1/things?a=b", "api.example.com")
        )

    def test_http_head_without_host(self) -> None:
        self.assertEqual(
            sniff_http_head(b"GET / HTTP/1.0\r\n\r\n"), ("GET", "/", "")
        )

    def test_http_head_rejects_non_http(self) -> None:
        self.assertIsNone(sniff_http_head(b"\x16\x03\x01\x00\x50 garbage\r\n"))
        self.assertIsNone(sniff_http_head(b"HELLO world\r\n"))
        self.assertIsNone(sniff_http_head(b"no newline yet"))
        # A valid-looking verb with a bogus protocol token is not HTTP.
        self.assertIsNone(sniff_http_head(b"GET / SSH-2.0\r\n\r\n"))

    def test_tls_sni(self) -> None:
        self.assertEqual(
            sniff_tls_sni(_client_hello(b"secure.example.com")), "secure.example.com"
        )

    def test_tls_without_sni_and_non_tls(self) -> None:
        self.assertIsNone(sniff_tls_sni(_client_hello(None)))
        self.assertIsNone(sniff_tls_sni(b"GET / HTTP/1.1\r\nHost: x\r\n\r\n"))
        self.assertIsNone(sniff_tls_sni(b"\x16\x03\x01"))  # truncated

    def test_original_dst_decoding(self) -> None:
        packed = struct.pack("!HH", socket.AF_INET, 8443) + socket.inet_aton("10.250.0.9") + b"\x00" * 8
        self.assertEqual(parse_original_dst(packed), ("10.250.0.9", 8443))
        with self.assertRaises(ValueError):
            parse_original_dst(b"\x02\x00")


class _EchoUpstream:
    """Trivial TCP echo server standing in for the "external world"."""

    def __init__(self, response: bytes = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nhi") -> None:
        self.response = response
        self.received: list[bytes] = []
        self._sock = socket.socket()
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(8)
        self.port = self._sock.getsockname()[1]
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
                    self.received.append(conn.recv(4096))
                    conn.sendall(self.response)
                except OSError:
                    pass

    def close(self) -> None:
        self._sock.close()
        self._thread.join(timeout=2.0)


class _ProxyHarness:
    """Drives the proxy over loopback with SO_ORIGINAL_DST patched (no
    netfilter needed to prove the flow/record behavior). Mixin, not a
    TestCase, so subclasses do not re-collect its tests."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="crab_egress_")
        self.addCleanup(self._tmp.cleanup)
        self.journal = ActionJournal(Path(self._tmp.name) / "journal")
        self.sandbox_id = SandboxId("sbx-egress")
        self.upstream = _EchoUpstream()
        self.addCleanup(self.upstream.close)
        self.proxy = EgressProxyServer(
            journal=self.journal,
            sandbox_id_resolver=lambda peer: self.sandbox_id if peer == "127.0.0.1" else None,
            host="127.0.0.1",
            port=0,
            head_timeout_seconds=1.0,
        )
        self.proxy.start()
        self.addCleanup(self.proxy.stop)
        patcher = mock.patch(
            "crab.egress.original_destination",
            return_value=("127.0.0.1", self.upstream.port),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _talk(self, payload: bytes) -> bytes:
        with socket.create_connection(("127.0.0.1", self.proxy.port), timeout=5.0) as sock:
            sock.sendall(payload)
            chunks = []
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                chunks.append(chunk)
        return b"".join(chunks)

    def _flows(self, *, expected: int = 1, timeout: float = 5.0) -> list[dict]:
        """Flows are recorded when the connection completes, which is just
        after the client sees EOF — poll instead of racing it."""
        deadline = time.monotonic() + timeout
        records: list = []
        while time.monotonic() < deadline:
            records = self.journal.entries(self.sandbox_id, kind="egress")
            if len(records) >= expected:
                break
            time.sleep(0.05)
        return [record.payload for record in records]

    def _no_flows(self, *, settle: float = 1.0) -> list[dict]:
        time.sleep(settle)
        return [
            record.payload
            for record in self.journal.entries(self.sandbox_id, kind="egress")
        ]

    def test_http_flow_recorded_and_spliced(self) -> None:
        response = self._talk(
            b"POST /v1/orders HTTP/1.1\r\nHost: api.example.com\r\nContent-Length: 2\r\n\r\nhi"
        )
        self.assertIn(b"200 OK", response)
        self.assertIn(b"/v1/orders", self.upstream.received[0])
        [flow] = self._flows()
        self.assertEqual(flow["scheme"], "http")
        self.assertEqual(flow["method"], "POST")
        self.assertEqual(flow["path"], "/v1/orders")
        self.assertEqual(flow["host"], "api.example.com")
        self.assertEqual(flow["dst_port"], self.upstream.port)
        self.assertEqual(flow["classification"], "mutating")  # POST
        self.assertGreater(flow["bytes_out"], 0)
        self.assertGreater(flow["bytes_in"], 0)

    def test_tls_flow_records_sni_without_decrypting(self) -> None:
        hello = _client_hello(b"secure.example.com")
        self._talk(hello)
        [flow] = self._flows()
        self.assertEqual(flow["scheme"], "tls")
        self.assertEqual(flow["host"], "secure.example.com")
        self.assertIsNone(flow["method"])
        # The ClientHello reached the upstream byte-identical.
        self.assertEqual(self.upstream.received[0][: len(hello)], hello)

    def test_opaque_flow_falls_back_to_ip(self) -> None:
        self._talk(b"\x00\x01\x02binary-protocol\n")
        [flow] = self._flows()
        self.assertEqual(flow["scheme"], "tcp")
        self.assertEqual(flow["host"], "127.0.0.1")
        self.assertIsNone(flow["method"])

    def test_unattributed_flow_is_not_recorded(self) -> None:
        self.proxy._sandbox_id_resolver = lambda peer: None
        self._talk(b"GET / HTTP/1.1\r\nHost: api.example.com\r\n\r\n")
        self.assertEqual(self._no_flows(), [])

    def test_txn_stamp_rides_along(self) -> None:
        self.journal.set_active_txn(self.sandbox_id, "txn-1")
        self.addCleanup(self.journal.set_active_txn, self.sandbox_id, None)
        self._talk(b"GET /health HTTP/1.1\r\nHost: api.example.com\r\n\r\n")
        self._flows()
        [record] = self.journal.entries(self.sandbox_id, kind="egress")
        self.assertEqual(record.txn_id, "txn-1")

    def test_non_redirected_connection_is_dropped(self) -> None:
        with mock.patch("crab.egress.original_destination", return_value=None):
            self._talk(b"GET / HTTP/1.1\r\nHost: api.example.com\r\n\r\n")
        self.assertEqual(self._no_flows(), [])
        self.assertEqual(self.upstream.received, [])

    def test_recorder_swallows_journal_failures(self) -> None:
        broken = mock.Mock()
        broken.record_egress.side_effect = RuntimeError("disk on fire")
        EgressFlowRecorder(broken).record(self.sandbox_id, {"host": "x"})
        EgressFlowRecorder(None).record(self.sandbox_id, {"host": "x"})


class ProxyFlowTests(_ProxyHarness, unittest.TestCase):
    pass


class RedirectRuleTests(unittest.TestCase):
    """The rule shape IS the "LLM interception unchanged" guarantee, so
    it is asserted literally."""

    def _manager(self):
        from integrations.sandboxes.runtime.network import BenchmarkNetworkManager

        manager = BenchmarkNetworkManager()
        manager._bridge_name = "acbdeadbeef"
        return manager

    def test_enable_adds_bridge_rule_excluding_the_host(self) -> None:
        manager = self._manager()
        with mock.patch(
            "integrations.sandboxes.runtime.network.subprocess.run"
        ) as run:
            run.return_value = mock.Mock(returncode=1)  # -C says "absent"
            manager.enable_egress_redirect(19999)
        appended = [
            call.args[0] for call in run.call_args_list if "-A" in call.args[0]
        ]
        self.assertEqual(
            appended,
            [
                [
                    "iptables", "-t", "nat", "-A", "PREROUTING",
                    "-i", "acbdeadbeef", "-p", "tcp",
                    "!", "-d", manager.bridge_ip,
                    "-j", "REDIRECT", "--to-ports", "19999",
                ]
            ],
        )

    def test_enable_is_idempotent_and_checks_first(self) -> None:
        manager = self._manager()
        with mock.patch(
            "integrations.sandboxes.runtime.network.subprocess.run"
        ) as run:
            run.return_value = mock.Mock(returncode=0)  # -C says "present"
            manager.enable_egress_redirect(19999)
            manager.enable_egress_redirect(19999)  # cached: no further calls
        self.assertEqual(
            [call.args[0][3] for call in run.call_args_list], ["-C"]
        )

    def test_disable_deletes_the_same_rule(self) -> None:
        manager = self._manager()
        with mock.patch(
            "integrations.sandboxes.runtime.network.subprocess.run"
        ) as run:
            run.return_value = mock.Mock(returncode=1)
            manager.enable_egress_redirect(18888)
            run.reset_mock()
            manager.disable_egress_redirect()
        [deleted] = [call.args[0] for call in run.call_args_list]
        self.assertEqual(deleted[:5], ["iptables", "-t", "nat", "-D", "PREROUTING"])
        self.assertIn("18888", deleted)
        # Second disable is a no-op (nothing left to remove).
        with mock.patch(
            "integrations.sandboxes.runtime.network.subprocess.run"
        ) as run:
            manager.disable_egress_redirect()
        run.assert_not_called()

    def test_enable_requires_the_bridge(self) -> None:
        from integrations.sandboxes.runtime.network import BenchmarkNetworkManager

        with self.assertRaises(RuntimeError):
            BenchmarkNetworkManager().enable_egress_redirect(1234)

    def test_cleanup_removes_the_redirect(self) -> None:
        manager = self._manager()
        with mock.patch(
            "integrations.sandboxes.runtime.network.subprocess.run"
        ) as run:
            run.return_value = mock.Mock(returncode=1)
            manager.enable_egress_redirect(17777)
            run.reset_mock()
            manager.cleanup()
        self.assertTrue(
            any(
                "REDIRECT" in call.args[0] and "-D" in call.args[0]
                for call in run.call_args_list
            ),
            "cleanup did not drop the redirect rule",
        )


class LeaseTransferTests(unittest.TestCase):
    """PR-N1: promotion moves a fork's network identity onto the source.
    The release ordering inside the manager is the whole risk —
    `release_lease` pops by sandbox id and then destroys whatever it
    popped, so re-keying before releasing would tear down the netns just
    transferred in."""

    def _manager(self):
        from integrations.sandboxes.runtime.network import BenchmarkNetworkManager

        manager = BenchmarkNetworkManager()
        manager._bridge_name = "acbdeadbeef"
        return manager

    def _seed(self, manager, sandbox_id: str, guest_ip: str):
        from integrations.sandboxes.runtime.network import BenchmarkNetworkLease
        from crab.ids import SandboxId

        sid = SandboxId(sandbox_id)
        suffix = guest_ip.replace(".", "")
        lease = BenchmarkNetworkLease(
            sandbox_id=sid,
            namespace_name=f"ts-{suffix}",
            namespace_path=Path(f"/var/run/netns/ts-{suffix}"),
            host_veth_name=f"vh{suffix}",
            guest_veth_name=f"vg{suffix}",
            guest_ip=guest_ip,
        )
        manager._leases[sid] = lease
        manager._ip_to_sandbox[guest_ip] = sid
        return sid, lease

    def test_transfer_rekeys_and_destroys_only_the_outgoing_netns(self) -> None:
        from crab.ids import SandboxId

        manager = self._manager()
        source, old = self._seed(manager, "sbx-src", "10.250.0.2")
        fork, incoming = self._seed(manager, "sbx-src-fork-1", "10.250.0.3")

        with mock.patch(
            "integrations.sandboxes.runtime.network.subprocess.run"
        ) as run:
            transferred = manager.transfer_lease(fork, source)

        # The fork's plumbing survives untouched; only the source's old
        # netns/veth are destroyed.
        destroyed = [call.args[0] for call in run.call_args_list]
        self.assertEqual(
            destroyed,
            [
                ["ip", "netns", "del", old.namespace_name],
                ["ip", "link", "delete", old.host_veth_name],
            ],
        )
        # The lease now belongs to the source, address and namespace intact.
        self.assertEqual(transferred.guest_ip, incoming.guest_ip)
        self.assertEqual(transferred.namespace_name, incoming.namespace_name)
        self.assertEqual(transferred.sandbox_id, source)
        self.assertEqual(manager.lease_for(source), transferred)
        self.assertIsNone(manager.lease_for(fork))
        # Attribution follows the address.
        self.assertEqual(manager.resolve_sandbox_id(incoming.guest_ip), source)
        self.assertIsNone(manager.resolve_sandbox_id(old.guest_ip))
        # The freed address returns to the pool.
        self.assertEqual(manager._free_ip_indices, [2])

    def test_releasing_the_old_id_after_a_transfer_is_a_no_op(self) -> None:
        """Fork teardown runs `release_network_lease(fork_id)` after a
        promotion. Post-transfer the fork id owns nothing, so the netns the
        source now runs in must survive."""
        manager = self._manager()
        source, _ = self._seed(manager, "sbx-src", "10.250.0.2")
        fork, incoming = self._seed(manager, "sbx-src-fork-1", "10.250.0.3")
        with mock.patch("integrations.sandboxes.runtime.network.subprocess.run"):
            manager.transfer_lease(fork, source)

        with mock.patch(
            "integrations.sandboxes.runtime.network.subprocess.run"
        ) as run:
            manager.release_lease(fork)

        run.assert_not_called()
        self.assertEqual(manager.lease_for(source).namespace_name, incoming.namespace_name)
        self.assertEqual(manager.resolve_sandbox_id(incoming.guest_ip), source)

    def test_transfer_without_a_source_lease_only_rekeys(self) -> None:
        manager = self._manager()
        from crab.ids import SandboxId

        source = SandboxId("sbx-src")
        fork, incoming = self._seed(manager, "sbx-src-fork-1", "10.250.0.3")

        with mock.patch(
            "integrations.sandboxes.runtime.network.subprocess.run"
        ) as run:
            transferred = manager.transfer_lease(fork, source)

        run.assert_not_called()
        self.assertEqual(transferred.guest_ip, incoming.guest_ip)
        self.assertEqual(manager.resolve_sandbox_id(incoming.guest_ip), source)

    def test_repeated_transfer_is_a_no_op(self) -> None:
        manager = self._manager()
        source, _ = self._seed(manager, "sbx-src", "10.250.0.2")
        fork, incoming = self._seed(manager, "sbx-src-fork-1", "10.250.0.3")
        with mock.patch("integrations.sandboxes.runtime.network.subprocess.run"):
            self.assertIsNotNone(manager.transfer_lease(fork, source))
            # Nothing left under the fork id: a retry must not disturb the
            # lease the source now holds.
            self.assertIsNone(manager.transfer_lease(fork, source))
        self.assertEqual(manager.lease_for(source).namespace_name, incoming.namespace_name)


class _RecordingRuntime:
    """Minimal runtime for Engine.transfer_network_lease: records the
    metadata refresh and hands out a bundle dir the retarget can rewrite."""

    def __init__(self, bundle_root: Path) -> None:
        self._bundle_root = bundle_root
        self.metadata_updates: list[tuple] = []

    def bundle_path_for(self, sandbox_id):
        return self._bundle_root / str(sandbox_id)

    def update_network_metadata(self, sandbox_id, *, guest_ip, network_namespace_path):
        self.metadata_updates.append((str(sandbox_id), guest_ip, network_namespace_path))


class EngineTransferNetworkLeaseTests(unittest.TestCase):
    """PR-N1 decision 9: the Engine call is the three-in-one — lease re-key,
    bundle netns retarget, and runtime metadata refresh. A transfer that
    only moves the lease passes the E2E socket assertions while silently
    breaking attribution, so the composition is pinned directly here."""

    def _engine(self, manager, runtime):
        from crab.engine import Engine

        engine = Engine.__new__(Engine)  # bypass full construction
        engine._network_manager = manager
        engine._runtime = runtime
        return engine

    def _write_bundle(self, bundle_root: Path, sandbox_id: str, netns_path: str) -> Path:
        import json

        bundle = bundle_root / sandbox_id
        bundle.mkdir(parents=True)
        spec = {"linux": {"namespaces": [{"type": "network", "path": netns_path}]}}
        (bundle / "config.json").write_text(json.dumps(spec), encoding="utf-8")
        return bundle

    def _manager(self):
        from integrations.sandboxes.runtime.network import BenchmarkNetworkManager

        manager = BenchmarkNetworkManager()
        manager._bridge_name = "acbdeadbeef"
        return manager

    def _seed(self, manager, sandbox_id: str, guest_ip: str):
        from integrations.sandboxes.runtime.network import BenchmarkNetworkLease
        from crab.ids import SandboxId

        sid = SandboxId(sandbox_id)
        suffix = guest_ip.replace(".", "")
        lease = BenchmarkNetworkLease(
            sandbox_id=sid,
            namespace_name=f"ts-{suffix}",
            namespace_path=Path(f"/var/run/netns/ts-{suffix}"),
            host_veth_name=f"vh{suffix}",
            guest_veth_name=f"vg{suffix}",
            guest_ip=guest_ip,
        )
        manager._leases[sid] = lease
        manager._ip_to_sandbox[guest_ip] = sid
        return sid

    def test_transfer_moves_lease_bundle_netns_and_metadata_together(self) -> None:
        from crab.ids import SandboxId

        with tempfile.TemporaryDirectory(prefix="crab_engine_xfer_") as tmp:
            bundle_root = Path(tmp) / "bundles"
            manager = self._manager()
            source = self._seed(manager, "sbx-src", "10.250.0.2")
            fork = self._seed(manager, "sbx-src-fork-1", "10.250.0.3")
            fork_lease = manager.lease_for(fork)
            self._write_bundle(bundle_root, "sbx-src", "/var/run/netns/ts-old")
            runtime = _RecordingRuntime(bundle_root)
            engine = self._engine(manager, runtime)

            with mock.patch("integrations.sandboxes.runtime.network.subprocess.run"):
                result = engine.transfer_network_lease(fork, source)

            self.assertTrue(result)
            # 1. lease re-keyed onto the source.
            self.assertEqual(manager.resolve_sandbox_id("10.250.0.3"), source)
            # 2. the source bundle now names the fork's netns.
            import json

            spec = json.loads((bundle_root / "sbx-src" / "config.json").read_text())
            netns = [ns for ns in spec["linux"]["namespaces"] if ns["type"] == "network"][0]
            self.assertEqual(netns["path"], str(fork_lease.namespace_path))
            # 3. runtime metadata refreshed to the new address + netns.
            self.assertEqual(
                runtime.metadata_updates,
                [("sbx-src", "10.250.0.3", str(fork_lease.namespace_path))],
            )

    def test_probe_reports_without_mutating(self) -> None:
        manager = self._manager()
        source = self._seed(manager, "sbx-src", "10.250.0.2")
        fork = self._seed(manager, "sbx-src-fork-1", "10.250.0.3")
        runtime = _RecordingRuntime(Path("/nonexistent"))
        engine = self._engine(manager, runtime)

        self.assertTrue(engine.transfer_network_lease(fork, source, probe=True))
        # No lease moved, no metadata touched.
        self.assertEqual(manager.resolve_sandbox_id("10.250.0.3"), fork)
        self.assertEqual(runtime.metadata_updates, [])

    def test_vanished_lease_after_probe_raises_rather_than_downgrading(self) -> None:
        # The fork was already dumped-and-stopped by the time the real
        # transfer runs; a silent False would send the caller down the
        # repair path with an image bound to the fork's address (decision
        # 13). No fork lease seeded → transfer_lease returns None.
        from crab.ids import SandboxId

        manager = self._manager()
        self._seed(manager, "sbx-src", "10.250.0.2")
        runtime = _RecordingRuntime(Path("/nonexistent"))
        engine = self._engine(manager, runtime)

        with self.assertRaises(RuntimeError):
            engine.transfer_network_lease(SandboxId("sbx-src-fork-1"), SandboxId("sbx-src"))


class ConfigGateTests(unittest.TestCase):
    def test_defaults_off(self) -> None:
        from crab.engine import EngineConfig

        cfg = EngineConfig()
        self.assertFalse(cfg.enable_egress_proxy)
        self.assertEqual(cfg.egress_proxy_port, 0)

    def test_from_mapping_nested_and_flat(self) -> None:
        from crab.engine import EngineConfig

        nested = EngineConfig.from_mapping({"egress": {"enabled": True, "port": 5555}})
        self.assertTrue(nested.enable_egress_proxy)
        self.assertEqual(nested.egress_proxy_port, 5555)
        flat = EngineConfig.from_mapping(
            {"enable_egress_proxy": True, "egress_proxy_port": 6666}
        )
        self.assertTrue(flat.enable_egress_proxy)
        self.assertEqual(flat.egress_proxy_port, 6666)

    def test_requires_sandbox_network(self) -> None:
        from crab.engine import Engine, EngineConfig

        with tempfile.TemporaryDirectory(prefix="crab_egress_cfg_") as tmp:
            cfg = EngineConfig(
                runtime="in-memory",
                enable_interceptor=False,
                enable_sandbox_network=False,
                enable_egress_proxy=True,
                storage_root=Path(tmp) / "storage",
            )
            with self.assertRaises(RuntimeError) as ctx:
                Engine.start(cfg)
            self.assertIn("enable_egress_proxy requires", str(ctx.exception))

    def test_requires_the_action_journal(self) -> None:
        from crab.engine import Engine, EngineConfig

        with tempfile.TemporaryDirectory(prefix="crab_egress_cfg_") as tmp:
            cfg = EngineConfig(
                runtime="in-memory",
                enable_interceptor=False,
                enable_sandbox_network=True,
                enable_egress_proxy=True,
                enable_action_journal=False,
                storage_root=Path(tmp) / "storage",
            )
            with self.assertRaises(RuntimeError) as ctx:
                Engine.start(cfg)
            # Without the ledger's store the proxy would forward everything
            # and record nothing — indistinguishable from working.
            self.assertIn("enable_action_journal", str(ctx.exception))

    def test_default_bind_is_loopback_not_wildcard(self) -> None:
        from crab.egress import EgressProxyServer

        proxy = EgressProxyServer()
        self.addCleanup(proxy.stop)
        self.assertEqual(proxy.server_address[0], "127.0.0.1")


# ---------------------------------------------------------------------------
# D1.2: classification + effect ledger
# ---------------------------------------------------------------------------

from crab.egress import EgressRule, classify_flow
from crab.ids import SandboxId as _SandboxId
from crab.models import EgressFlow, EgressLedger


def _flow_payload(**overrides) -> dict:
    payload = {
        "host": "api.example.com",
        "dst_ip": "203.0.113.7",
        "dst_port": 443,
        "scheme": "http",
        "method": "GET",
        "path": "/things",
        "bytes_out": 120,
        "bytes_in": 340,
        "duration_ms": 4.2,
    }
    payload.update(overrides)
    return payload


class _CliHarness:
    """CLI driver over a stubbed daemon client; mixin so subclasses do
    not re-collect these tests."""

    def _run_cli(self, argv: list[str], responses: dict) -> tuple[int, str, list]:
        requests: list[dict] = []

        class _CliClient:
            def __init__(self, socket_path, *, timeout_seconds):
                requests.append({"socket": str(socket_path), "timeout": timeout_seconds})

            def post_json(self, path, payload=None, *, timeout_seconds=None):
                requests.append({"path": path, "payload": payload})
                return responses[path]

            def get_json(self, path, *, timeout_seconds=None):
                return responses[path]

        stdout = io.StringIO()
        from crab.cli import commands

        with mock.patch.object(commands, "DaemonClient", _CliClient):
            with contextlib.redirect_stdout(stdout):
                rc = commands.main(argv)
        return rc, stdout.getvalue(), requests


class _FakeDaemonClient:
    """Records requests and replays canned responses (or raises)."""

    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.responses: dict = {}

    def post_json(self, path, payload=None, *, timeout_seconds=None):
        self.requests.append(
            {"path": path, "payload": payload, "timeout": timeout_seconds}
        )
        response = self.responses.get(path)
        if isinstance(response, Exception):
            raise response
        return response or {"ok": True}

    def get_json(self, path, *, timeout_seconds=None):
        return self.responses.get(path) or {"ok": True}


class ClassificationTests(unittest.TestCase):
    def test_http_method_table(self) -> None:
        for method in ("GET", "HEAD", "OPTIONS", "TRACE"):
            self.assertEqual(
                classify_flow(_flow_payload(method=method)), "idempotent_read"
            )
        for method in ("POST", "PUT", "PATCH", "DELETE"):
            self.assertEqual(classify_flow(_flow_payload(method=method)), "mutating")
        # Lowercase from a sloppy client still classifies.
        self.assertEqual(classify_flow(_flow_payload(method="post")), "mutating")

    def test_encrypted_and_raw_flows_are_opaque(self) -> None:
        self.assertEqual(
            classify_flow(_flow_payload(scheme="tls", method=None, path=None)), "opaque"
        )
        self.assertEqual(
            classify_flow(_flow_payload(scheme="tcp", method=None, path=None)), "opaque"
        )
        # CONNECT tunnels reveal nothing about the tunnelled request.
        self.assertEqual(classify_flow(_flow_payload(method="CONNECT")), "opaque")

    def test_host_rules_override_the_protocol_default(self) -> None:
        rules = (
            EgressRule(host_glob="*.internal.example", classify="idempotent_read"),
            EgressRule(host_glob="payments.example.com", classify="mutating"),
        )
        # TLS to a known read-only internal API: rule refines what the
        # protocol cannot reveal.
        self.assertEqual(
            classify_flow(
                _flow_payload(host="reports.internal.example", scheme="tls", method=None),
                rules,
            ),
            "idempotent_read",
        )
        # A rule also overrides an explicit GET.
        self.assertEqual(
            classify_flow(_flow_payload(host="payments.example.com", method="GET"), rules),
            "mutating",
        )
        # Non-matching hosts keep the protocol default.
        self.assertEqual(
            classify_flow(_flow_payload(host="api.example.com", method="POST"), rules),
            "mutating",
        )

    def test_first_matching_rule_wins_and_matching_is_case_insensitive(self) -> None:
        rules = (
            EgressRule(host_glob="*.EXAMPLE.com", classify="opaque"),
            EgressRule(host_glob="api.example.com", classify="mutating"),
        )
        self.assertEqual(classify_flow(_flow_payload(method="GET"), rules), "opaque")

    def test_rule_validation_and_json(self) -> None:
        with self.assertRaises(ValueError):
            EgressRule(host_glob="*", classify="write")
        rule = EgressRule.from_json({"host_glob": "*.x", "classify": "mutating"})
        self.assertEqual((rule.host_glob, rule.classify), ("*.x", "mutating"))

    def test_classification_is_rederivable_from_stored_rows(self) -> None:
        stored = _flow_payload(method="DELETE", classification="idempotent_read")
        # Reclassifying a stored row ignores the old verdict.
        self.assertEqual(classify_flow(stored), "mutating")


class ProxyClassificationTests(_ProxyHarness, unittest.TestCase):
    """The proxy stamps the class at record time, rules included."""

    def test_recorded_flow_carries_its_class(self) -> None:
        self._talk(b"DELETE /things/1 HTTP/1.1\r\nHost: api.example.com\r\n\r\n")
        [flow] = self._flows()
        self.assertEqual(flow["classification"], "mutating")

    def test_proxy_applies_host_rules(self) -> None:
        self.proxy.rules = (
            EgressRule(host_glob="api.example.com", classify="idempotent_read"),
        )
        self._talk(b"POST /things HTTP/1.1\r\nHost: api.example.com\r\n\r\n")
        [flow] = self._flows()
        self.assertEqual(flow["classification"], "idempotent_read")


class LedgerModelTests(unittest.TestCase):
    def _ledger(self) -> EgressLedger:
        return EgressLedger(
            sandbox_id=_SandboxId("sbx-1"),
            flows=(
                EgressFlow(
                    seq=1, host="a.example", dst_ip="10.0.0.1", dst_port=80,
                    scheme="http", classification="idempotent_read", method="GET",
                    path="/", bytes_out=10, bytes_in=20,
                ),
                EgressFlow(
                    seq=2, host="a.example", dst_ip="10.0.0.1", dst_port=80,
                    scheme="http", classification="mutating", method="POST",
                    path="/w", txn_id="txn-1",
                ),
                EgressFlow(
                    seq=3, host="b.example", dst_ip="10.0.0.2", dst_port=443,
                    scheme="tls", classification="opaque",
                ),
            ),
        )

    def test_counts_and_hosts(self) -> None:
        ledger = self._ledger()
        self.assertEqual(ledger.total, 3)
        self.assertEqual(ledger.idempotent_reads, 1)
        self.assertEqual(ledger.mutating, 1)
        self.assertEqual(ledger.opaque, 1)
        self.assertEqual(ledger.hosts, ("a.example", "b.example"))

    def test_round_trip(self) -> None:
        ledger = self._ledger()
        restored = EgressLedger.from_json(ledger.to_json())
        self.assertEqual(restored, ledger)
        payload = ledger.to_json()
        self.assertEqual(payload["mutating"], 1)  # counts are serialized too


class SystemLedgerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="crab_ledger_")
        self.addCleanup(self._tmp.cleanup)
        self.journal = ActionJournal(Path(self._tmp.name) / "journal")
        self.sandbox_id = SandboxId("sbx-ledger")
        self.system = _minimal_system(self.journal)

    def _record(self, **overrides) -> None:
        payload = _flow_payload(**overrides)
        payload.setdefault("classification", classify_flow(payload))
        self.journal.record_egress(self.sandbox_id, payload=payload)

    def test_reads_flows_with_counts_and_provenance(self) -> None:
        self._record(method="GET", path="/a")
        self._record(method="POST", path="/b")
        ledger = self.system.egress_ledger(self.sandbox_id)
        self.assertEqual((ledger.total, ledger.idempotent_reads, ledger.mutating), (2, 1, 1))
        self.assertEqual([flow.seq for flow in ledger.flows], [0, 1])  # journal seq starts at 0
        self.assertTrue(all(flow.recorded_at for flow in ledger.flows))
        self.assertIsNone(ledger.txn_id)

    def test_txn_scoping_and_since_seq(self) -> None:
        self._record(method="GET", path="/before")
        self.journal.set_active_txn(self.sandbox_id, "txn-7")
        self._record(method="POST", path="/in-txn")
        self.journal.set_active_txn(self.sandbox_id, None)
        self._record(method="GET", path="/after")

        scoped = self.system.egress_ledger(self.sandbox_id, txn_id="txn-7")
        self.assertEqual(scoped.total, 1)
        self.assertEqual(scoped.mutating, 1)
        self.assertEqual(scoped.flows[0].path, "/in-txn")
        self.assertEqual(scoped.flows[0].txn_id, "txn-7")
        self.assertEqual(scoped.txn_id, "txn-7")

        # since_seq is exclusive; rows are seq 0/1/2 here.
        tail = self.system.egress_ledger(self.sandbox_id, since_seq=1)
        self.assertEqual([flow.path for flow in tail.flows], ["/after"])

    def test_requires_the_journal(self) -> None:
        self.system.journal = None
        with self.assertRaises(RuntimeError):
            self.system.egress_ledger(self.sandbox_id)

    def test_mutating_egress_helper_is_failure_tolerant(self) -> None:
        self.journal.set_active_txn(self.sandbox_id, "txn-9")
        self._record(method="PUT", path="/x")
        self.journal.set_active_txn(self.sandbox_id, None)
        self.assertEqual(self.system._mutating_egress_in_txn(self.sandbox_id, "txn-9"), 1)
        self.assertEqual(self.system._mutating_egress_in_txn(self.sandbox_id, "txn-none"), 0)
        with mock.patch.object(
            self.system, "egress_ledger", side_effect=RuntimeError("boom")
        ):
            self.assertEqual(self.system._mutating_egress_in_txn(self.sandbox_id, "txn-9"), 0)
        self.system.journal = None
        self.assertEqual(self.system._mutating_egress_in_txn(self.sandbox_id, "txn-9"), 0)


def _minimal_system(journal):
    """Real CrabSystem wired to a real journal with stub collaborators —
    the ledger only touches the journal."""
    from types import SimpleNamespace

    from crab import (
        CRExecutor,
        CRScheduler,
        CrabSystem,
        EBPFSandboxInspector,
        ExecutorConfig,
        InMemorySchedulerStateStore,
        InMemoryTelemetrySink,
        LocalCheckpointManager,
        SchedulerConfig,
        StorageConfig,
    )
    from crab.interceptor import SandboxResponseGateRegistry
    from crab.scheduler import FaultToleranceCheckpointingPolicy

    root = journal.root.parent
    runtime = SimpleNamespace(name="runc")
    telemetry = InMemoryTelemetrySink()
    storage = LocalCheckpointManager(
        StorageConfig(root_dir=root / "storage"), destroy_filesystem_ref=lambda ref: None
    )
    executor = CRExecutor(ExecutorConfig(max_workers=1), None, None, telemetry)
    cfg = SchedulerConfig(require_change_signal=True)
    scheduler = CRScheduler(
        cfg,
        EBPFSandboxInspector(),
        runtime,
        InMemorySchedulerStateStore(),
        telemetry,
        FaultToleranceCheckpointingPolicy(cfg),
    )
    return CrabSystem(
        scheduler=scheduler,
        executor=executor,
        storage=storage,
        inspector=EBPFSandboxInspector(),
        runtime=runtime,
        telemetry=telemetry,
        response_gate_registry=SandboxResponseGateRegistry(),
        journal=journal,
    )


class SandboxEgressPlumbingTests(unittest.TestCase):
    def test_delegates_with_scoping(self) -> None:
        from crab.sandbox import Sandbox

        calls: list = []

        class _System:
            def egress_ledger(self, sandbox_id, *, txn_id=None, since_seq=None):
                calls.append((str(sandbox_id), txn_id, since_seq))
                return EgressLedger(sandbox_id=sandbox_id)

        class _Engine:
            system = _System()

            def _register_sandbox(self, sandbox) -> None:
                pass

        sandbox = Sandbox.connect("sbx-1", engine=_Engine())
        ledger = sandbox.egress(txn_id="txn-3", since_seq=5)
        self.assertEqual(ledger.total, 0)
        self.assertEqual(calls, [("sbx-1", "txn-3", 5)])

    def test_bare_engine_raises(self) -> None:
        from crab.sandbox import Sandbox

        class _Bare:
            system = object()

            def _register_sandbox(self, sandbox) -> None:
                pass

        with self.assertRaises(NotImplementedError):
            Sandbox.connect("sbx-1", engine=_Bare()).egress()


class _LedgerDaemonSystem:
    def __init__(self) -> None:
        self.calls: list = []
        self.error: Exception | None = None

    def egress_ledger(self, sandbox_id, *, txn_id=None, since_seq=None):
        self.calls.append((str(sandbox_id), txn_id, since_seq))
        if self.error is not None:
            raise self.error
        return EgressLedger(
            sandbox_id=SandboxId(str(sandbox_id)),
            flows=(
                EgressFlow(
                    seq=1, host="api.example.com", dst_ip="10.0.0.9", dst_port=443,
                    scheme="http", classification="mutating", method="POST", path="/w",
                    txn_id=txn_id,
                ),
            ),
            txn_id=txn_id,
        )


class LedgerDaemonTests(unittest.TestCase):
    def setUp(self) -> None:
        from types import SimpleNamespace

        from crab.daemon.server import _Routes

        self.engine = SimpleNamespace(system=_LedgerDaemonSystem())

        class _Daemon:
            def __init__(self, engine):
                self._engine = engine

            def require_engine(self):
                return self._engine

            def register_sandbox(self, sandbox_id) -> None:
                pass

            def unregister_sandbox(self, sandbox_id) -> None:
                pass

        self.daemon_cls = _Daemon
        self.routes = _Routes(_Daemon(self.engine))

    def test_route_serializes_and_scopes(self) -> None:
        response = self.routes.sandbox_egress(
            {"txn_id": "txn-2", "since_seq": 4}, sandbox_id="sbx-1"
        )
        self.assertTrue(response["ok"])
        self.assertEqual(response["ledger"]["mutating"], 1)
        self.assertEqual(self.engine.system.calls, [("sbx-1", "txn-2", 4)])

    def test_missing_journal_is_a_bad_request(self) -> None:
        from crab.daemon.server import _BadRequest

        self.engine.system.error = RuntimeError("the effect ledger requires the journal")
        with self.assertRaises(_BadRequest):
            self.routes.sandbox_egress({}, sandbox_id="sbx-1")

    def test_dispatch_over_socket(self) -> None:
        from crab.daemon.server import _build_handler
        from crab.daemon.transport import DaemonClient, serve_unix_socket

        tmp = tempfile.TemporaryDirectory(prefix="crab_ledgerd_")
        self.addCleanup(tmp.cleanup)
        socket_path = Path(tmp.name) / "crab.sock"
        server = serve_unix_socket(socket_path, _build_handler(self.daemon_cls(self.engine)))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        client = DaemonClient(socket_path, timeout_seconds=10.0)
        response = client.post_json("/sandboxes/sbx-1/egress", {})
        self.assertEqual(response["ledger"]["total"], 1)


class LedgerShimTests(unittest.TestCase):
    def test_payload_and_rehydration(self) -> None:
        from crab.remote_engine import RemoteEngine

        client = _FakeDaemonClient()
        ledger = EgressLedger(
            sandbox_id=SandboxId("sbx-1"),
            flows=(
                EgressFlow(
                    seq=2, host="a.example", dst_ip="10.0.0.1", dst_port=80,
                    scheme="http", classification="idempotent_read", method="GET", path="/",
                ),
            ),
        )
        client.responses["/sandboxes/sbx-1/egress"] = {"ok": True, "ledger": ledger.to_json()}
        engine = RemoteEngine(client, info={"runtime": "runc", "default_image": "ubuntu:22.04"})
        restored = engine.system.egress_ledger(SandboxId("sbx-1"), txn_id="txn-1", since_seq=2)
        self.assertIsInstance(restored.flows[0], EgressFlow)
        self.assertEqual(restored.idempotent_reads, 1)
        self.assertEqual(
            client.requests[0]["payload"], {"txn_id": "txn-1", "since_seq": 2}
        )
        # No scoping -> empty body (the daemon defaults to everything).
        engine.system.egress_ledger(SandboxId("sbx-1"))
        self.assertEqual(client.requests[1]["payload"], {})


class LedgerCliTests(_CliHarness, unittest.TestCase):
    def test_egress_summary_and_rows(self) -> None:
        ledger = EgressLedger(
            sandbox_id=SandboxId("sbx-1"),
            flows=(
                EgressFlow(
                    seq=1, host="api.example.com", dst_ip="10.0.0.1", dst_port=443,
                    scheme="http", classification="mutating", method="POST", path="/orders",
                    txn_id="txn-4",
                ),
                EgressFlow(
                    seq=2, host="secure.example.com", dst_ip="10.0.0.2", dst_port=443,
                    scheme="tls", classification="opaque",
                ),
            ),
        )
        rc, out, requests = self._run_cli(
            ["sandbox", "egress", "sbx-1", "--txn", "txn-4"],
            {"/sandboxes/sbx-1/egress": {"ok": True, "ledger": ledger.to_json()}},
        )
        self.assertEqual(rc, 0)
        self.assertIn("total=2 reads=0 mutating=1 opaque=1", out)
        self.assertIn("1\tmutating\tPOST\tapi.example.com:443\t/orders\ttxn=txn-4", out)
        self.assertIn("2\topaque\ttls\tsecure.example.com:443", out)
        self.assertEqual(requests[-1]["payload"], {"txn_id": "txn-4"})


class LedgerReclassificationTests(unittest.TestCase):
    """Classification is re-derived when the ledger is read, so rule
    changes and rows written before D1.2 are reflected in the view."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="crab_reclass_")
        self.addCleanup(self._tmp.cleanup)
        self.journal = ActionJournal(Path(self._tmp.name) / "journal")
        self.sandbox_id = SandboxId("sbx-reclass")
        self.system = _minimal_system(self.journal)

    def test_rows_recorded_before_classification_are_classified_on_read(self) -> None:
        # A D1.1-era row: the field exists but carries the placeholder.
        self.journal.record_egress(
            self.sandbox_id,
            payload=_flow_payload(method="POST", classification="unclassified"),
        )
        ledger = self.system.egress_ledger(self.sandbox_id)
        self.assertEqual(ledger.flows[0].classification, "mutating")
        self.assertEqual(ledger.mutating, 1)

    def test_rule_changes_apply_to_history(self) -> None:
        self.journal.record_egress(
            self.sandbox_id,
            payload=_flow_payload(
                host="reports.internal.example", scheme="tls", method=None, path=None,
                classification="opaque",
            ),
        )
        self.assertEqual(self.system.egress_ledger(self.sandbox_id).opaque, 1)

        # The deployment learns this host is read-only; history follows.
        self.system.egress_rules = (
            EgressRule(host_glob="*.internal.example", classify="idempotent_read"),
        )
        ledger = self.system.egress_ledger(self.sandbox_id)
        self.assertEqual(ledger.idempotent_reads, 1)
        self.assertEqual(ledger.opaque, 0)

    def test_stored_row_is_left_untouched(self) -> None:
        self.journal.record_egress(
            self.sandbox_id,
            payload=_flow_payload(method="DELETE", classification="unclassified"),
        )
        self.system.egress_ledger(self.sandbox_id)
        [record] = self.journal.entries(self.sandbox_id, kind="egress")
        self.assertEqual(record.payload["classification"], "unclassified")


class AbortMutatingEgressTransportTests(unittest.TestCase):
    """The count must survive the daemon hop: computing it locally and
    dropping it on the wire would make remote aborts silently report 0."""

    def test_daemon_serializes_the_count(self) -> None:
        from types import SimpleNamespace

        from crab.daemon.server import _Routes
        from crab.txn import TxnAbortResult

        class _System:
            def abort_txn(self, sandbox_id, txn_id):
                return TxnAbortResult(
                    txn_id=txn_id,
                    discarded_observations=2,
                    restored_checkpoint_id="ckpt-1",
                    mutating_egress=3,
                )

        class _Daemon:
            def require_engine(self):
                return SimpleNamespace(system=_System())

            def register_sandbox(self, sandbox_id) -> None:
                pass

            def unregister_sandbox(self, sandbox_id) -> None:
                pass

        response = _Routes(_Daemon()).abort_txn({}, sandbox_id="sbx-1", txn_id="txn-1")
        self.assertEqual(response["result"]["mutating_egress"], 3)

    def test_shim_reads_it_and_tolerates_older_daemons(self) -> None:
        from crab.remote_engine import RemoteEngine

        client = _FakeDaemonClient()
        engine = RemoteEngine(client, info={"runtime": "runc", "default_image": "ubuntu:22.04"})
        path = "/sandboxes/sbx-1/txn/txn-1/abort"
        client.responses[path] = {
            "ok": True,
            "result": {
                "txn_id": "txn-1",
                "discarded_observations": 0,
                "restored_checkpoint_id": None,
                "mutating_egress": 4,
            },
        }
        self.assertEqual(
            engine.system.abort_txn(SandboxId("sbx-1"), "txn-1").mutating_egress, 4
        )
        # A pre-D1 daemon omits the field entirely.
        client.responses[path] = {
            "ok": True,
            "result": {
                "txn_id": "txn-1",
                "discarded_observations": 0,
                "restored_checkpoint_id": None,
            },
        }
        self.assertEqual(
            engine.system.abort_txn(SandboxId("sbx-1"), "txn-1").mutating_egress, 0
        )


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# D2.1: recording through the proxy
# ---------------------------------------------------------------------------

from crab.cassettes import CassetteStore
from crab.egress import CassetteRecorder


class ProxyRecordingTests(_ProxyHarness, unittest.TestCase):
    """The tee path: a real (loopback) exchange must land in a cassette
    and leave its index fields on the journal row."""

    def setUp(self) -> None:
        super().setUp()
        self.store = CassetteStore(Path(self._tmp.name) / "cassettes")
        self.proxy.cassette_recorder = CassetteRecorder(self.store)

    def test_idempotent_read_is_recorded_end_to_end(self) -> None:
        self.upstream.response = (
            b"HTTP/1.1 200 OK\r\nContent-Length: 11\r\n\r\nhello world"
        )
        body = self._talk(b"GET /things HTTP/1.1\r\nHost: api.example.com\r\n\r\n")
        self.assertIn(b"hello world", body)  # client still got the real bytes

        [flow] = self._flows()
        self.assertTrue(flow["recorded"])
        self.assertEqual(flow["status"], 200)
        self.assertFalse(flow["truncated"])
        entry = self.store.get(self.sandbox_id, flow["request_key"])
        self.assertEqual(entry.body, b"hello world")
        self.assertEqual(entry.method, "GET")
        self.assertEqual(entry.path, "/things")

    def test_mutating_flow_is_never_recorded(self) -> None:
        self._talk(b"POST /things HTTP/1.1\r\nHost: api.example.com\r\n\r\n")
        [flow] = self._flows()
        self.assertEqual(flow["classification"], "mutating")
        self.assertNotIn("request_key", flow)
        self.assertEqual(self.store.count(self.sandbox_id), 0)

    def test_host_rules_can_make_a_read_recordable(self) -> None:
        # TLS-like opaque case: a rule declares the host read-only, so the
        # same gate that classifies also enables recording.
        self.proxy.rules = (
            EgressRule(host_glob="api.example.com", classify="idempotent_read"),
        )
        self.upstream.response = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nhi"
        self._talk(b"POST /things HTTP/1.1\r\nHost: api.example.com\r\n\r\n")
        [flow] = self._flows()
        self.assertTrue(flow["recorded"])

    def test_oversized_response_is_marked_truncated_and_not_stored(self) -> None:
        self.proxy.cassette_recorder = CassetteRecorder(self.store, max_body_bytes=32)
        self.upstream.response = (
            b"HTTP/1.1 200 OK\r\nContent-Length: 200\r\n\r\n" + b"x" * 200
        )
        self._talk(b"GET /big HTTP/1.1\r\nHost: api.example.com\r\n\r\n")
        [flow] = self._flows()
        self.assertFalse(flow["recorded"])
        self.assertTrue(flow["truncated"])
        self.assertEqual(self.store.count(self.sandbox_id), 0)

    def test_recording_off_leaves_flows_untouched(self) -> None:
        self.proxy.cassette_recorder = None
        self._talk(b"GET /things HTTP/1.1\r\nHost: api.example.com\r\n\r\n")
        [flow] = self._flows()
        self.assertNotIn("recorded", flow)
        self.assertEqual(self.store.count(self.sandbox_id), 0)


class RecordingConfigTests(unittest.TestCase):
    def test_defaults_off_and_nested_parsing(self) -> None:
        from crab.engine import EngineConfig

        self.assertFalse(EngineConfig().enable_egress_recording)
        cfg = EngineConfig.from_mapping(
            {
                "egress": {
                    "enabled": True,
                    "recording": {
                        "enabled": True,
                        "max_body_bytes": 2048,
                        "record_errors": True,
                        "varying_headers": ["accept", "range"],
                        "record_partial": True,
                    },
                }
            }
        )
        self.assertTrue(cfg.enable_egress_recording)
        self.assertEqual(cfg.egress_recording_max_body_bytes, 2048)
        self.assertTrue(cfg.egress_recording_record_errors)
        self.assertEqual(cfg.egress_recording_varying_headers, ("accept", "range"))
        self.assertTrue(cfg.egress_recording_record_partial)

    def test_recording_requires_the_proxy(self) -> None:
        from crab.engine import Engine, EngineConfig

        with tempfile.TemporaryDirectory(prefix="crab_rec_cfg_") as tmp:
            cfg = EngineConfig(
                runtime="in-memory",
                enable_interceptor=False,
                enable_sandbox_network=False,
                enable_egress_proxy=False,
                enable_egress_recording=True,
                storage_root=Path(tmp) / "storage",
            )
            with self.assertRaises(RuntimeError) as ctx:
                Engine.start(cfg)
            self.assertIn("enable_egress_recording requires", str(ctx.exception))

    def test_cassettes_dirname_is_validated(self) -> None:
        from crab.config import StorageConfig

        self.assertEqual(StorageConfig(root_dir=Path("/tmp")).cassettes_dirname, "cassettes")
        with self.assertRaises(ValueError):
            StorageConfig(root_dir=Path("/tmp"), cassettes_dirname="")


class LedgerRecordingFieldsTests(unittest.TestCase):
    def test_counters_and_defaults(self) -> None:
        recorded = EgressFlow(
            seq=1, host="a", dst_ip="10.0.0.1", dst_port=80, scheme="http",
            classification="idempotent_read", method="GET", path="/",
            recorded=True, request_key="k", status=200,
        )
        replayed = EgressFlow(
            seq=2, host="a", dst_ip="10.0.0.1", dst_port=80, scheme="http",
            classification="idempotent_read", method="GET", path="/",
            recorded=True, request_key="k", status=200, replayed=True,
            replayed_from="sbx-fork",
        )
        plain = EgressFlow(
            seq=3, host="a", dst_ip="10.0.0.1", dst_port=80, scheme="http",
            classification="mutating", method="POST", path="/w",
        )
        ledger = EgressLedger(
            sandbox_id=SandboxId("sbx-1"), flows=(recorded, replayed, plain)
        )
        self.assertEqual(ledger.recorded, 2)
        self.assertEqual(ledger.replayed, 1)
        payload = ledger.to_json()
        self.assertEqual((payload["recorded"], payload["replayed"]), (2, 1))
        self.assertEqual(EgressLedger.from_json(payload), ledger)

    def test_d1_era_rows_deserialize_with_defaults(self) -> None:
        legacy = {
            "seq": 7, "host": "a", "dst_ip": "10.0.0.1", "dst_port": 80,
            "scheme": "http", "classification": "idempotent_read",
            "method": "GET", "path": "/", "bytes_out": 1, "bytes_in": 2,
            "duration_ms": 1.0, "txn_id": None, "recorded_at": None,
        }
        flow = EgressFlow.from_json(legacy)
        self.assertFalse(flow.recorded)
        self.assertIsNone(flow.request_key)
        self.assertIsNone(flow.status)
        self.assertFalse(flow.truncated)
        self.assertFalse(flow.replayed)
        self.assertIsNone(flow.replayed_from_seq)
        self.assertIsNone(flow.replayed_from)


class RecordedCliTests(_CliHarness, unittest.TestCase):
    def _ledger(self) -> EgressLedger:
        return EgressLedger(
            sandbox_id=SandboxId("sbx-1"),
            flows=(
                EgressFlow(
                    seq=1, host="api.example.com", dst_ip="10.0.0.1", dst_port=80,
                    scheme="http", classification="idempotent_read", method="GET",
                    path="/things", recorded=True, request_key="k1", status=200,
                ),
                EgressFlow(
                    seq=2, host="api.example.com", dst_ip="10.0.0.1", dst_port=80,
                    scheme="http", classification="mutating", method="POST", path="/w",
                ),
            ),
        )

    def test_summary_reports_recorded_counts(self) -> None:
        rc, out, _ = self._run_cli(
            ["sandbox", "egress", "sbx-1"],
            {"/sandboxes/sbx-1/egress": {"ok": True, "ledger": self._ledger().to_json()}},
        )
        self.assertEqual(rc, 0)
        self.assertIn("recorded=1 replayed=0", out)
        self.assertIn("rec:200", out)

    def test_recorded_filter_hides_unrecorded_flows(self) -> None:
        rc, out, _ = self._run_cli(
            ["sandbox", "egress", "sbx-1", "--recorded"],
            {"/sandboxes/sbx-1/egress": {"ok": True, "ledger": self._ledger().to_json()}},
        )
        self.assertEqual(rc, 0)
        self.assertIn("/things", out)
        self.assertNotIn("POST", out)


# ---------------------------------------------------------------------------
# D2.2: replay
# ---------------------------------------------------------------------------

from crab.cassettes import CassetteEntry, sha256_hex
from crab.egress import CassetteReplayer
from crab.models import EgressReplayReport


class _ReplayHarness(_ProxyHarness):
    """Proxy with both a recorder and a replayer, plus a counter on the
    upstream so "served from cassette" can be proven by absence."""

    def setUp(self) -> None:
        super().setUp()
        self.store = CassetteStore(Path(self._tmp.name) / "cassettes")
        self.proxy.cassette_recorder = CassetteRecorder(self.store)
        self.replayer = CassetteReplayer(self.store)
        self.proxy.cassette_replayer = self.replayer

    def _upstream_hits(self) -> int:
        return len(self.upstream.received)

    def _seed(self, *, sandbox_id=None, path="/things", body=b"recorded body", status=200):
        """Record one exchange the honest way (through the proxy)."""
        self.upstream.response = (
            f"HTTP/1.1 {status} OK\r\nContent-Length: {len(body)}\r\n\r\n".encode() + body
        )
        if sandbox_id is not None:
            original = self.sandbox_id
            self.sandbox_id = sandbox_id
            try:
                self._talk(f"GET {path} HTTP/1.1\r\nHost: api.example.com\r\n\r\n".encode())
            finally:
                self.sandbox_id = original
        else:
            self._talk(f"GET {path} HTTP/1.1\r\nHost: api.example.com\r\n\r\n".encode())
        self._flows()


class ReplayServingTests(_ReplayHarness, unittest.TestCase):
    def test_hit_serves_the_cassette_without_touching_upstream(self) -> None:
        self._seed(body=b"recorded body")
        hits_after_record = self._upstream_hits()

        self.replayer.begin(self.sandbox_id, policy="cassette_first")
        response = self._talk(b"GET /things HTTP/1.1\r\nHost: api.example.com\r\n\r\n")
        session = self.replayer.end(self.sandbox_id)

        self.assertIn(b"recorded body", response)
        # The decisive assertion: the origin never saw a second request.
        self.assertEqual(self._upstream_hits(), hits_after_record)
        self.assertEqual((session.served, session.missed), (1, 0))
        rows = self._flows(expected=2)
        self.assertEqual(rows[-1]["replayed_from"], str(self.sandbox_id))
        self.assertTrue(rows[-1]["recorded"])

    def test_miss_falls_through_under_cassette_first(self) -> None:
        self.upstream.response = b"HTTP/1.1 200 OK\r\nContent-Length: 4\r\n\r\nlive"
        self.replayer.begin(self.sandbox_id, policy="cassette_first")
        response = self._talk(b"GET /never-recorded HTTP/1.1\r\nHost: api.example.com\r\n\r\n")
        session = self.replayer.end(self.sandbox_id)
        self.assertIn(b"live", response)
        self.assertEqual((session.served, session.missed), (0, 1))

    def test_miss_returns_504_under_cassette_only(self) -> None:
        self.replayer.begin(self.sandbox_id, policy="cassette_only")
        hits_before = self._upstream_hits()
        response = self._talk(b"GET /never-recorded HTTP/1.1\r\nHost: api.example.com\r\n\r\n")
        session = self.replayer.end(self.sandbox_id)
        self.assertIn(b"504", response)
        self.assertIn(b"X-Crab-Replay: miss", response)
        self.assertEqual(self._upstream_hits(), hits_before)  # hermetic
        self.assertEqual((session.served, session.missed), (0, 1))

    def test_writes_always_reach_the_world_in_both_policies(self) -> None:
        for policy in ("cassette_first", "cassette_only"):
            with self.subTest(policy=policy):
                self.upstream.response = b"HTTP/1.1 201 Created\r\nContent-Length: 2\r\n\r\nok"
                hits_before = self._upstream_hits()
                self.replayer.begin(self.sandbox_id, policy=policy)
                response = self._talk(
                    b"POST /orders HTTP/1.1\r\nHost: api.example.com\r\n\r\n"
                )
                session = self.replayer.end(self.sandbox_id)
                self.assertIn(b"201", response)
                self.assertEqual(self._upstream_hits(), hits_before + 1)
                self.assertEqual(session.passed_through, 1)
                self.assertEqual(session.served, 0)

    def test_reclassified_host_stops_being_served(self) -> None:
        self._seed(body=b"recorded body")
        hits_after_record = self._upstream_hits()
        # The deployment now declares this host opaque: the stored row
        # still says recorded=True, but the gate is recomputed on replay.
        self.proxy.rules = (EgressRule(host_glob="api.example.com", classify="opaque"),)
        self.upstream.response = b"HTTP/1.1 200 OK\r\nContent-Length: 4\r\n\r\nlive"
        self.replayer.begin(self.sandbox_id, policy="cassette_first")
        response = self._talk(b"GET /things HTTP/1.1\r\nHost: api.example.com\r\n\r\n")
        session = self.replayer.end(self.sandbox_id)
        self.assertIn(b"live", response)
        self.assertGreater(self._upstream_hits(), hits_after_record)
        self.assertEqual(session.served, 0)
        self.assertEqual(session.passed_through, 1)

    def test_truncated_cassette_is_never_served(self) -> None:
        entry = CassetteEntry(
            request_key="k" * 64, method="GET", host="api.example.com", port=80,
            path="/things", status=200, body=b"partial",
            body_sha256=sha256_hex(b"partial"), truncated=True,
        )
        self.store.put(self.sandbox_id, entry)
        session = self.replayer.begin(self.sandbox_id, policy="cassette_first")
        found = self.replayer.lookup(
            session,
            request_head=b"GET /things HTTP/1.1\r\nHost: api.example.com\r\n\r\n",
            host="api.example.com", port=80, method="GET", path="/things",
            varying_headers=("accept", "accept-encoding"),
        )
        self.assertIsNone(found)

    def test_replay_off_by_default(self) -> None:
        self._seed(body=b"recorded body")
        hits_after_record = self._upstream_hits()
        self._talk(b"GET /things HTTP/1.1\r\nHost: api.example.com\r\n\r\n")
        # No session armed: the request goes out for real.
        self.assertEqual(self._upstream_hits(), hits_after_record + 1)


class ReplayCrossBucketTests(_ReplayHarness, unittest.TestCase):
    """cassette_source is what makes C4 work: the fork recorded, the
    source replays."""

    def test_source_replays_the_forks_cassettes(self) -> None:
        fork_id = SandboxId("sbx-egress-fork")
        self._seed(sandbox_id=fork_id, body=b"fork body")
        hits_after_record = self._upstream_hits()

        # Without the redirect the source's own (empty) bucket is used.
        self.replayer.begin(self.sandbox_id, policy="cassette_only")
        response = self._talk(b"GET /things HTTP/1.1\r\nHost: api.example.com\r\n\r\n")
        miss_session = self.replayer.end(self.sandbox_id)
        self.assertIn(b"504", response)
        self.assertEqual((miss_session.served, miss_session.missed), (0, 1))

        # With it, the fork's recording answers the source's request.
        self.replayer.begin(
            self.sandbox_id, policy="cassette_only", cassette_source=fork_id
        )
        response = self._talk(b"GET /things HTTP/1.1\r\nHost: api.example.com\r\n\r\n")
        hit_session = self.replayer.end(self.sandbox_id)
        self.assertIn(b"fork body", response)
        self.assertEqual((hit_session.served, hit_session.missed), (1, 0))
        self.assertEqual(self._upstream_hits(), hits_after_record)
        self.assertEqual(hit_session.cassette_source, str(fork_id))


class ReplaySessionTests(unittest.TestCase):
    def test_policy_validation_and_session_lifecycle(self) -> None:
        replayer = CassetteReplayer(store=None)
        with self.assertRaises(ValueError):
            replayer.begin("sbx-1", policy="none")
        with self.assertRaises(ValueError):
            replayer.begin("sbx-1", policy="always")
        session = replayer.begin("sbx-1", policy="cassette_first")
        self.assertEqual(session.cassette_source, "sbx-1")  # defaults to itself
        self.assertIs(replayer.session_for("sbx-1"), session)
        self.assertIs(replayer.end("sbx-1"), session)
        self.assertIsNone(replayer.session_for("sbx-1"))
        self.assertIsNone(replayer.end("sbx-1"))  # idempotent

    def test_report_round_trip(self) -> None:
        report = EgressReplayReport(
            sandbox_id=SandboxId("sbx-1"), policy="cassette_first",
            cassette_source="sbx-fork", served=3, missed=1, passed_through=2,
            hosts=("a.example", "b.example"),
        )
        self.assertEqual(EgressReplayReport.from_json(report.to_json()), report)


class SystemReplayFacadeTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="crab_replay_sys_")
        self.addCleanup(self._tmp.cleanup)
        self.journal = ActionJournal(Path(self._tmp.name) / "journal")
        self.system = _minimal_system(self.journal)
        self.store = CassetteStore(Path(self._tmp.name) / "cassettes")
        self.system.cassette_replayer = CassetteReplayer(self.store)

    def test_begin_end_round_trip(self) -> None:
        self.system.begin_egress_replay(
            SandboxId("sbx-1"), policy="cassette_only", cassette_source=SandboxId("sbx-fork")
        )
        session = self.system.cassette_replayer.session_for("sbx-1")
        session.served = 2
        session.missed = 1
        session.hosts.add("api.example.com")
        report = self.system.end_egress_replay(SandboxId("sbx-1"))
        self.assertEqual(report.policy, "cassette_only")
        self.assertEqual(report.cassette_source, "sbx-fork")
        self.assertEqual((report.served, report.missed), (2, 1))
        self.assertEqual(report.hosts, ("api.example.com",))
        events = [name for name, _ in self.system.telemetry.events]
        self.assertIn("egress_replay.completed", events)
        # Ending twice is a no-op rather than an error.
        self.assertIsNone(self.system.end_egress_replay(SandboxId("sbx-1")))

    def test_requires_a_replayer(self) -> None:
        self.system.cassette_replayer = None
        with self.assertRaises(RuntimeError):
            self.system.begin_egress_replay(SandboxId("sbx-1"))
        self.assertIsNone(self.system.end_egress_replay(SandboxId("sbx-1")))


class _KeepAliveUpstream:
    """Serves two requests on one connection, so the recorder's
    one-exchange-per-connection assumption can be tested rather than
    assumed (a D2.1 review follow-up)."""

    def __init__(self, responses: list[bytes]) -> None:
        self.responses = responses
        self.requests: list[bytes] = []
        self._sock = socket.socket()
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(8)
        self.port = self._sock.getsockname()[1]
        self.connections = 0
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        while True:
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return
            self.connections += 1
            with conn:
                try:
                    for response in self.responses:
                        chunk = conn.recv(4096)
                        if not chunk:
                            break
                        self.requests.append(chunk)
                        conn.sendall(response)
                except OSError:
                    pass

    def close(self) -> None:
        self._sock.close()
        self._thread.join(timeout=2.0)


class KeepAliveTests(unittest.TestCase):
    """Pins what a keep-alive connection does to recording and replay.
    v1 records one exchange per connection; the point of these tests is
    that the second exchange must not corrupt the first one's cassette."""

    _FIRST = b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nfirst"
    _SECOND = b"HTTP/1.1 200 OK\r\nContent-Length: 6\r\n\r\nsecond"

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="crab_keepalive_")
        self.addCleanup(self._tmp.cleanup)
        self.journal = ActionJournal(Path(self._tmp.name) / "journal")
        self.store = CassetteStore(Path(self._tmp.name) / "cassettes")
        self.sandbox_id = SandboxId("sbx-keepalive")
        self.upstream = _KeepAliveUpstream([self._FIRST, self._SECOND])
        self.addCleanup(self.upstream.close)
        self.proxy = EgressProxyServer(
            journal=self.journal,
            sandbox_id_resolver=lambda peer: self.sandbox_id,
            host="127.0.0.1",
            port=0,
            head_timeout_seconds=1.0,
            cassette_recorder=CassetteRecorder(self.store),
        )
        self.replayer = CassetteReplayer(self.store)
        self.proxy.cassette_replayer = self.replayer
        self.proxy.start()
        self.addCleanup(self.proxy.stop)
        patcher = mock.patch(
            "crab.egress.original_destination",
            return_value=("127.0.0.1", self.upstream.port),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _pipeline_two_requests(self) -> bytes:
        """One connection, two sequential requests (classic keep-alive)."""
        received = b""
        with socket.create_connection(("127.0.0.1", self.proxy.port), timeout=5.0) as sock:
            sock.sendall(b"GET /one HTTP/1.1\r\nHost: api.example.com\r\n\r\n")
            received += sock.recv(4096)
            sock.sendall(b"GET /two HTTP/1.1\r\nHost: api.example.com\r\n\r\n")
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                received += chunk
        return received

    def _flows(self, *, expected: int = 1, timeout: float = 5.0) -> list[dict]:
        deadline = time.monotonic() + timeout
        records: list = []
        while time.monotonic() < deadline:
            records = self.journal.entries(self.sandbox_id, kind="egress")
            if len(records) >= expected:
                break
            time.sleep(0.05)
        return [record.payload for record in records]

    def test_second_exchange_does_not_corrupt_the_first_cassette(self) -> None:
        body = self._pipeline_two_requests()
        self.assertIn(b"first", body)
        self.assertIn(b"second", body)  # both answers reached the client
        self.assertEqual(self.upstream.connections, 1)  # genuinely keep-alive

        [flow] = self._flows()
        self.assertTrue(flow["recorded"])
        entry = self.store.get(self.sandbox_id, flow["request_key"])
        # One exchange per connection: the cassette holds the FIRST
        # request's response and nothing of the second one.
        self.assertEqual(entry.path, "/one")
        self.assertEqual(entry.body, b"first")
        self.assertNotIn(b"second", entry.body)
        self.assertEqual(self.store.count(self.sandbox_id), 1)

    def test_replay_hit_serves_one_exchange_and_closes(self) -> None:
        self._pipeline_two_requests()
        self._flows()
        connections_after_record = self.upstream.connections

        self.replayer.begin(self.sandbox_id, policy="cassette_first")
        with socket.create_connection(("127.0.0.1", self.proxy.port), timeout=5.0) as sock:
            sock.sendall(b"GET /one HTTP/1.1\r\nHost: api.example.com\r\n\r\n")
            served = b""
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                served += chunk
        session = self.replayer.end(self.sandbox_id)

        self.assertIn(b"first", served)
        # Replay answers with Connection: close, so a keep-alive client
        # knows not to reuse the socket for a second request.
        self.assertIn(b"Connection: close", served)
        self.assertEqual(session.served, 1)
        self.assertEqual(self.upstream.connections, connections_after_record)


class ReplayCounterConcurrencyTests(unittest.TestCase):
    """The proxy is one thread per connection, so the session tallies are
    written concurrently: a plain ``+=`` loses updates and the report
    (C4's determinism input) under-counts."""

    def test_concurrent_bumps_are_not_lost(self) -> None:
        replayer = CassetteReplayer(store=None)
        session = replayer.begin("sbx-1", policy="cassette_first")
        bumps = 200
        workers = 8

        def hammer() -> None:
            for _ in range(bumps):
                replayer.record_hit(session, "api.example.com")
                replayer.record_miss(session)
                replayer.record_pass_through(session)

        threads = [threading.Thread(target=hammer) for _ in range(workers)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        served, missed, passed, hosts = replayer.snapshot(session)
        self.assertEqual(served, bumps * workers)
        self.assertEqual(missed, bumps * workers)
        self.assertEqual(passed, bumps * workers)
        self.assertEqual(hosts, ("api.example.com",))


class StripPortForSniTests(unittest.TestCase):
    """Unit tests for _strip_port_for_sni helper."""

    def test_domain_with_port(self):
        self.assertEqual(_strip_port_for_sni("api.example.com:8443"), "api.example.com")

    def test_domain_without_port(self):
        self.assertEqual(_strip_port_for_sni("api.example.com"), "api.example.com")

    def test_domain_default_https_port(self):
        self.assertEqual(_strip_port_for_sni("example.com:443"), "example.com")

    def test_ipv6_bracketed_with_port(self):
        self.assertEqual(_strip_port_for_sni("[::1]:8443"), "::1")

    def test_ipv6_bracketed_without_port(self):
        self.assertEqual(_strip_port_for_sni("[::1]"), "::1")

    def test_ipv6_bare(self):
        self.assertEqual(_strip_port_for_sni("::1"), "::1")

    def test_ipv6_full_bracketed_with_port(self):
        self.assertEqual(
            _strip_port_for_sni("[2001:db8::1]:443"), "2001:db8::1"
        )

    def test_empty_string(self):
        self.assertEqual(_strip_port_for_sni(""), "")

    def test_localhost_with_port(self):
        self.assertEqual(_strip_port_for_sni("localhost:9090"), "localhost")

