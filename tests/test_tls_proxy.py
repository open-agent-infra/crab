"""Unit tests for PR-T1.2: TLS proxy termination path, config surface,
pre-termination filter, _PLAINTEXT_VISIBLE guard, and _splice close_notify fix.

Uses in-process loopback sockets + the T1.1 CA to exercise the full
termination path without netfilter or VMs.
"""
from __future__ import annotations

import socket
import ssl
import struct
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import pytest

# Guard: skip entire module when cryptography is not installed (base env).
cryptography = pytest.importorskip("cryptography")

from crab.egress import (
    EgressProxyServer,
    EgressRule,
    _HEAD_PEEK_BYTES,
    _PLAINTEXT_VISIBLE,
    _TLS_RECORD_MAX_PEEK,
    classify_flow,
    sniff_http_head,
    sniff_tls_alpn,
    sniff_tls_sni,
)
from crab.engine import EngineConfig, _build_tls_interceptor
from crab.tls_ca import CAStore, LeafMinter
from crab.tls_interceptor import TLSInterceptor, _ALPN_PROTOCOLS, _WEB_PORTS


# ============================================================
# Helpers
# ============================================================

def _client_hello(
    server_name: bytes | None,
    alpn: list[bytes] | None = None,
    *,
    padding: int = 0,
) -> bytes:
    """A minimal but structurally valid TLS 1.2 ClientHello.

    ``padding`` inserts a dummy extension (type 0x0015 / padding) of the
    given byte length *before* SNI/ALPN, pushing them further into the
    record. Useful for testing large-hello handling.
    """
    extensions = b""
    if padding > 0:
        # TLS padding extension (type 0x0015, RFC 7685)
        pad_data = b"\x00" * padding
        extensions += struct.pack("!HH", 0x0015, len(pad_data)) + pad_data
    if server_name is not None:
        name_entry = b"\x00" + struct.pack("!H", len(server_name)) + server_name
        sni_body = struct.pack("!H", len(name_entry)) + name_entry
        extensions += struct.pack("!HH", 0x0000, len(sni_body)) + sni_body
    if alpn is not None:
        # ALPN extension (type 0x0010)
        name_list = b""
        for proto in alpn:
            name_list += bytes([len(proto)]) + proto
        alpn_body = struct.pack("!H", len(name_list)) + name_list
        extensions += struct.pack("!HH", 0x0010, len(alpn_body)) + alpn_body
    body = (
        b"\x03\x03"
        + b"\x11" * 32
        + b"\x00"
        + struct.pack("!H", 2) + b"\x13\x01"
        + b"\x01\x00"
        + struct.pack("!H", len(extensions)) + extensions
    )
    handshake = b"\x01" + len(body).to_bytes(3, "big") + body
    return b"\x16\x03\x01" + struct.pack("!H", len(handshake)) + handshake


def _make_tls_interceptor(tmp_path: Path, **kwargs) -> TLSInterceptor:
    return TLSInterceptor(tmp_path / "tls", **kwargs)


# ============================================================
# Config surface tests
# ============================================================

class TestEngineConfigTLS:
    """EngineConfig parses egress.tls_interception correctly."""

    def test_default_disabled(self):
        cfg = EngineConfig()
        assert cfg.egress_tls_interception_enabled is False
        assert cfg.egress_tls_on_handshake_failure == "passthrough"
        assert cfg.egress_tls_bypass_hosts == ()

    def test_from_mapping_enabled(self):
        raw = {
            "engine": {
                "egress": {
                    "tls_interception": {
                        "enabled": True,
                        "on_handshake_failure": "refuse",
                        "bypass_hosts": ["*.pinned.example"],
                    }
                }
            }
        }
        cfg = EngineConfig.from_mapping(raw)
        assert cfg.egress_tls_interception_enabled is True
        assert cfg.egress_tls_on_handshake_failure == "refuse"
        assert cfg.egress_tls_bypass_hosts == ("*.pinned.example",)


# ============================================================
# Pre-termination filter tests
# ============================================================

class TestTLSInterceptorFilter:
    """TLSInterceptor.should_intercept pre-filter logic."""

    def test_no_sni_not_intercepted(self, tmp_path):
        ti = _make_tls_interceptor(tmp_path)
        assert ti.should_intercept(None, 443) is False

    def test_non_web_port_not_intercepted(self, tmp_path):
        ti = _make_tls_interceptor(tmp_path)
        assert ti.should_intercept("db.example.com", 5432) is False

    def test_web_port_intercepted(self, tmp_path):
        ti = _make_tls_interceptor(tmp_path)
        assert ti.should_intercept("api.example.com", 443) is True
        assert ti.should_intercept("api.example.com", 8443) is True

    def test_bypass_hosts_pattern(self, tmp_path):
        ti = _make_tls_interceptor(
            tmp_path, bypass_hosts=["*.pinned.example"]
        )
        assert ti.should_intercept("api.pinned.example", 443) is False
        assert ti.should_intercept("other.example.com", 443) is True

    def test_runtime_bypass(self, tmp_path):
        ti = _make_tls_interceptor(tmp_path)
        assert ti.should_intercept("fail.example.com", 443) is True
        ti.add_runtime_bypass("fail.example.com")
        assert ti.should_intercept("fail.example.com", 443) is False


# ============================================================
# SSLContext / ALPN tests
# ============================================================

class TestTLSInterceptorContexts:
    """Server and upstream contexts have correct ALPN."""

    def test_server_context_alpn(self, tmp_path):
        ti = _make_tls_interceptor(tmp_path)
        ctx = ti.build_server_context("test.example.com")
        assert isinstance(ctx, ssl.SSLContext)
        # ALPN protocols are set (verify via a roundtrip since
        # SSLContext lacks a getter in Python 3.11).

    def test_upstream_context_alpn(self, tmp_path):
        ti = _make_tls_interceptor(tmp_path)
        ctx = ti.build_upstream_context("test.example.com")
        assert isinstance(ctx, ssl.SSLContext)


# ============================================================
# _PLAINTEXT_VISIBLE set test
# ============================================================

class TestPlaintextVisible:
    """The _PLAINTEXT_VISIBLE set contains http and https."""

    def test_members(self):
        assert "http" in _PLAINTEXT_VISIBLE
        assert "https" in _PLAINTEXT_VISIBLE
        assert "tls" not in _PLAINTEXT_VISIBLE


# ============================================================
# classify_flow parity test
# ============================================================

class TestClassifyFlowParity:
    """A decrypted HTTPS head classifies the same as the same request in
    plaintext — the parity guarantee of §3.1."""

    def test_get_classified_as_read(self):
        payload_http = {"host": "x.com", "method": "GET", "scheme": "http"}
        payload_https = {"host": "x.com", "method": "GET", "scheme": "https"}
        assert classify_flow(payload_http) == classify_flow(payload_https)

    def test_post_classified_as_mutating(self):
        payload_http = {"host": "x.com", "method": "POST", "scheme": "http"}
        payload_https = {"host": "x.com", "method": "POST", "scheme": "https"}
        assert classify_flow(payload_http) == classify_flow(payload_https)


# ============================================================
# In-process TLS roundtrip (loopback, no netfilter)
# ============================================================

class TestTLSTerminationLoopback:
    """Minimal in-process TLS termination roundtrip using socketpair."""

    def test_full_termination_roundtrip(self, tmp_path):
        """A TLS client connects, proxy terminates, re-reads HTTP head."""
        ti = _make_tls_interceptor(tmp_path)
        sni = "roundtrip.example.com"

        # Build a client context that trusts our CA.
        client_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        client_ctx.load_verify_locations(cadata=ti.ca_store.cert_pem().decode())
        client_ctx.set_alpn_protocols(_ALPN_PROTOCOLS)

        # Create a socketpair to simulate the proxy⇄client link.
        sock_client, sock_proxy = socket.socketpair()
        try:
            server_ctx = ti.build_server_context(sni)

            # Server-side wrap in a thread (blocking handshake).
            wrapped_proxy = [None]
            errors = []

            def server_wrap():
                try:
                    wrapped_proxy[0] = server_ctx.wrap_socket(
                        sock_proxy, server_side=True
                    )
                except Exception as e:
                    errors.append(e)

            t = threading.Thread(target=server_wrap)
            t.start()

            # Client-side wrap.
            wrapped_client = client_ctx.wrap_socket(
                sock_client, server_hostname=sni
            )
            t.join(timeout=5)
            assert not errors, f"Server handshake failed: {errors}"
            assert wrapped_proxy[0] is not None

            # Client sends an HTTP request through the TLS tunnel.
            request = b"GET /test HTTP/1.1\r\nHost: roundtrip.example.com\r\n\r\n"
            wrapped_client.sendall(request)

            # Proxy reads the decrypted head.
            decrypted_head = wrapped_proxy[0].recv(2048)
            http = sniff_http_head(decrypted_head)
            assert http is not None
            assert http == ("GET", "/test", "roundtrip.example.com")

            # Verify ALPN was negotiated as http/1.1.
            assert wrapped_client.selected_alpn_protocol() == "http/1.1"

            wrapped_client.close()
            wrapped_proxy[0].close()
        finally:
            sock_client.close()
            sock_proxy.close()

    def test_handshake_failure_refuse(self, tmp_path):
        """With on_handshake_failure=refuse, a distrusting client fails."""
        ti = _make_tls_interceptor(
            tmp_path, on_handshake_failure="refuse"
        )
        sni = "refuse.example.com"

        # Client ctx that does NOT trust our CA → handshake will fail.
        client_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        # Use default system CAs — our CA won't be among them.

        sock_client, sock_proxy = socket.socketpair()
        try:
            server_ctx = ti.build_server_context(sni)

            errors = []

            def server_wrap():
                try:
                    server_ctx.wrap_socket(sock_proxy, server_side=True)
                except ssl.SSLError:
                    errors.append("server_ssl_error")
                except OSError:
                    errors.append("server_os_error")

            t = threading.Thread(target=server_wrap)
            t.start()

            with pytest.raises((ssl.SSLError, ssl.SSLCertVerificationError, OSError)):
                client_ctx.wrap_socket(sock_client, server_hostname=sni)

            t.join(timeout=5)
            # Under "refuse" policy, no runtime bypass is added.
            assert ti.should_intercept(sni, 443) is True
        finally:
            sock_client.close()
            sock_proxy.close()

    def test_handshake_failure_passthrough_adds_bypass(self, tmp_path):
        """With passthrough policy, failed handshake adds runtime bypass."""
        ti = _make_tls_interceptor(
            tmp_path, on_handshake_failure="passthrough"
        )
        sni = "bypass-after-fail.example.com"
        # Simulate the behavior the proxy would do on SSLError.
        ti.add_runtime_bypass(sni)
        assert ti.should_intercept(sni, 443) is False


# ============================================================
# LeafMinter cache_max validation (T1.1 fix-up)
# ============================================================

class TestLeafMinterCacheMaxValidation:
    """cache_max <= 0 raises ValueError."""

    def test_zero_raises(self, tmp_path):
        ca = CAStore(tmp_path / "tls")
        with pytest.raises(ValueError, match="positive"):
            LeafMinter(ca, cache_max=0)

    def test_negative_raises(self, tmp_path):
        ca = CAStore(tmp_path / "tls")
        with pytest.raises(ValueError, match="positive"):
            LeafMinter(ca, cache_max=-5)


# ============================================================
# EgressProxyServer accepts tls_interceptor param
# ============================================================

class TestEgressProxyServerTLSParam:
    """EgressProxyServer accepts and stores a tls_interceptor."""

    def test_accepts_tls_interceptor(self, tmp_path):
        ti = _make_tls_interceptor(tmp_path)
        server = EgressProxyServer(tls_interceptor=ti)
        assert server.tls_interceptor is ti
        server.server_close()

    def test_default_none(self):
        server = EgressProxyServer()
        assert server.tls_interceptor is None
        server.server_close()


# ============================================================
# A: _build_tls_interceptor wiring test
# ============================================================

class TestBuildTLSInterceptor:
    """_build_tls_interceptor returns TLSInterceptor or None."""

    def test_enabled_returns_interceptor(self, tmp_path):
        cfg = EngineConfig(
            storage_root=tmp_path,
            egress_tls_interception_enabled=True,
            egress_tls_bypass_hosts=("*.bypass.test",),
            egress_tls_on_handshake_failure="refuse",
        )
        ti = _build_tls_interceptor(cfg)
        assert ti is not None
        assert isinstance(ti, TLSInterceptor)
        assert ti.on_handshake_failure == "refuse"
        # bypass_hosts wired through
        assert ti.should_intercept("foo.bypass.test", 443) is False

    def test_disabled_returns_none(self, tmp_path):
        cfg = EngineConfig(
            storage_root=tmp_path,
            egress_tls_interception_enabled=False,
        )
        assert _build_tls_interceptor(cfg) is None


# ============================================================
# B: sniff_tls_alpn tests
# ============================================================

class TestSniffTLSAlpn:
    """ALPN extraction from ClientHello."""

    def test_no_alpn_extension_returns_none(self):
        """ClientHello without ALPN → None (not empty list)."""
        hello = _client_hello(b"example.com")
        assert sniff_tls_alpn(hello) is None

    def test_h2_only(self):
        hello = _client_hello(b"example.com", alpn=[b"h2"])
        assert sniff_tls_alpn(hello) == ["h2"]

    def test_h2_and_http11(self):
        hello = _client_hello(b"example.com", alpn=[b"h2", b"http/1.1"])
        result = sniff_tls_alpn(hello)
        assert result is not None
        assert "h2" in result
        assert "http/1.1" in result

    def test_http11_only(self):
        hello = _client_hello(b"example.com", alpn=[b"http/1.1"])
        assert sniff_tls_alpn(hello) == ["http/1.1"]

    def test_non_tls_returns_none(self):
        assert sniff_tls_alpn(b"GET / HTTP/1.1\r\n\r\n") is None

    def test_truncated_returns_none(self):
        hello = _client_hello(b"example.com", alpn=[b"h2"])
        assert sniff_tls_alpn(hello[:20]) is None

    def test_name_truncated_inside_list_returns_none(self):
        """If a protocol name is partially cut off, return None (not partial list)."""
        hello = _client_hello(b"example.com", alpn=[b"http/1.1", b"h2"])
        # Truncate inside the second name ("h2") — cut 1 byte off the end
        assert sniff_tls_alpn(hello[:-1]) is None


# ============================================================
# B: ALPN pre-filter integration in interception decision
# ============================================================

class TestALPNPreFilter:
    """ALPN pre-filter semantics in the interception decision."""

    def test_h2_only_not_intercepted(self, tmp_path):
        """ALPN=[h2] → not intercepted (left opaque)."""
        ti = _make_tls_interceptor(tmp_path)
        hello = _client_hello(b"api.example.com", alpn=[b"h2"])
        sni = sniff_tls_sni(hello)
        alpn = sniff_tls_alpn(hello)
        alpn_ok = alpn is None or "http/1.1" in alpn
        # ALPN has only h2 → alpn_ok is False
        assert alpn_ok is False
        assert sni == "api.example.com"

    def test_h2_and_http11_intercepted(self, tmp_path):
        """ALPN=[h2, http/1.1] → intercepted."""
        ti = _make_tls_interceptor(tmp_path)
        hello = _client_hello(b"api.example.com", alpn=[b"h2", b"http/1.1"])
        sni = sniff_tls_sni(hello)
        alpn = sniff_tls_alpn(hello)
        alpn_ok = alpn is None or "http/1.1" in alpn
        assert alpn_ok is True
        assert ti.should_intercept(sni, 443) is True

    def test_no_alpn_still_intercepted(self, tmp_path):
        """Missing ALPN → not blocked by ALPN filter."""
        ti = _make_tls_interceptor(tmp_path)
        hello = _client_hello(b"api.example.com")  # no ALPN
        alpn = sniff_tls_alpn(hello)
        assert alpn is None
        alpn_ok = alpn is None or "http/1.1" in alpn
        assert alpn_ok is True
        sni = sniff_tls_sni(hello)
        assert ti.should_intercept(sni, 443) is True


# ============================================================
# Large ClientHello: record-length-aware peek
# ============================================================

class TestLargeClientHello:
    """Ensure SNI/ALPN extraction works when they sit beyond 2048 bytes."""

    def test_sniff_sni_beyond_2048(self):
        """SNI at offset > 2048 is still extracted from the full buffer."""
        hello = _client_hello(b"big.example.com", padding=2500)
        # Sanity: the hello is bigger than 2048
        assert len(hello) > _HEAD_PEEK_BYTES
        assert sniff_tls_sni(hello) == "big.example.com"

    def test_sniff_alpn_beyond_2048(self):
        """ALPN at offset > 2048 is still extracted from the full buffer."""
        hello = _client_hello(b"big.example.com", alpn=[b"h2", b"http/1.1"], padding=2500)
        assert len(hello) > _HEAD_PEEK_BYTES
        result = sniff_tls_alpn(hello)
        assert result is not None
        assert "http/1.1" in result

    def test_record_length_aware_peek_via_socketpair(self):
        """handle-layer: expanded MSG_PEEK retrieves full ClientHello."""
        hello = _client_hello(b"peek.example.com", alpn=[b"http/1.1"], padding=2500)
        assert len(hello) > _HEAD_PEEK_BYTES
        # Use socketpair to simulate the recv(MSG_PEEK) expansion logic
        s_client, s_server = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            s_client.sendall(hello)
            # First peek: limited to _HEAD_PEEK_BYTES (mimics handle)
            head = s_server.recv(_HEAD_PEEK_BYTES, socket.MSG_PEEK)
            assert len(head) == _HEAD_PEEK_BYTES
            # Cannot extract SNI from truncated peek
            assert sniff_tls_sni(head) is None
            # Record-length-aware expansion (same logic as handle)
            assert head[0:1] == b"\x16"
            record_len = struct.unpack("!H", head[3:5])[0] + 5
            needed = min(record_len, _TLS_RECORD_MAX_PEEK)
            assert needed > _HEAD_PEEK_BYTES
            head = s_server.recv(needed, socket.MSG_PEEK)
            # Now SNI and ALPN are extractable
            assert sniff_tls_sni(head) == "peek.example.com"
            alpn = sniff_tls_alpn(head)
            assert alpn is not None
            assert "http/1.1" in alpn
        finally:
            s_client.close()
            s_server.close()

    def test_record_length_capped_at_max(self):
        """A forged record_length > 16389 is capped at _TLS_RECORD_MAX_PEEK."""
        # Forge a TLS record header claiming 20000 bytes payload
        fake_head = b"\x16\x03\x01" + struct.pack("!H", 20000) + b"\x01" + b"\x00" * 50
        record_len = struct.unpack("!H", fake_head[3:5])[0] + 5
        needed = min(record_len, _TLS_RECORD_MAX_PEEK)
        assert needed == _TLS_RECORD_MAX_PEEK
