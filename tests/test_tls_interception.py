"""TLS certificate and egress interception tests.

Covers: CA generation idempotency, key file permissions, leaf SAN correctness,
per-host caching, cache expiry, EKU, clock-skew backdate, capacity eviction,
host validation, CA→leaf chain verification, and rejected HTTPS responses.
"""

from __future__ import annotations

import datetime
import http.client
import ipaddress
import os
import socket
import ssl
import stat
from unittest import mock

import pytest

# Guard: skip entire module when cryptography is not installed (base env).
cryptography = pytest.importorskip("cryptography")

from cryptography import x509  # noqa: E402
from cryptography.hazmat.primitives import hashes, serialization  # noqa: E402
from cryptography.hazmat.primitives.asymmetric import padding, rsa  # noqa: E402

from crab.egress import EgressProxyServer  # noqa: E402
from crab.effects import EffectGate  # noqa: E402
from crab.ids import SandboxId  # noqa: E402
from crab.journal import ActionJournal  # noqa: E402
from crab.tls_ca import CAStore, LeafMinter, _LEAF_CACHE_MAX  # noqa: E402
from crab.tls_interceptor import TLSInterceptor, _ALPN_PROTOCOLS  # noqa: E402


class TestCAStore:
    """CA generation and persistence."""

    def test_generates_ca_on_first_use(self, tmp_path):
        """First instantiation creates ca.crt and ca.key."""
        store = CAStore(tmp_path / "tls")
        assert (tmp_path / "tls" / "ca.crt").exists()
        assert (tmp_path / "tls" / "ca.key").exists()
        assert store.cert is not None
        assert store.key is not None

    def test_idempotent_reload(self, tmp_path):
        """Second instantiation loads the same CA, not a new one."""
        d = tmp_path / "tls"
        store1 = CAStore(d)
        serial1 = store1.cert.serial_number

        store2 = CAStore(d)
        serial2 = store2.cert.serial_number

        assert serial1 == serial2, "CA must be idempotent: same serial on reload"

    def test_key_file_permission_0600(self, tmp_path):
        """Private key file must be chmod 0600."""
        store = CAStore(tmp_path / "tls")
        key_stat = os.stat(store.key_path)
        mode = stat.S_IMODE(key_stat.st_mode)
        assert mode == 0o600, f"Expected 0600, got {oct(mode)}"

    def test_key_permission_enforced_on_reload(self, tmp_path):
        """Even if someone loosens the permission, reload re-enforces 0600."""
        d = tmp_path / "tls"
        CAStore(d)
        # Loosen permission.
        os.chmod(d / "ca.key", 0o644)
        # Re-load.
        CAStore(d)
        mode = stat.S_IMODE(os.stat(d / "ca.key").st_mode)
        assert mode == 0o600

    def test_ca_is_self_signed(self, tmp_path):
        """CA cert issuer == subject (self-signed)."""
        store = CAStore(tmp_path / "tls")
        assert store.cert.issuer == store.cert.subject

    def test_ca_has_basic_constraints_ca_true(self, tmp_path):
        """CA cert must have BasicConstraints(ca=True)."""
        store = CAStore(tmp_path / "tls")
        bc = store.cert.extensions.get_extension_for_class(
            x509.BasicConstraints
        ).value
        assert bc.ca is True

    def test_cert_pem_returns_bytes(self, tmp_path):
        """cert_pem() returns valid PEM bytes."""
        store = CAStore(tmp_path / "tls")
        pem = store.cert_pem()
        assert pem.startswith(b"-----BEGIN CERTIFICATE-----")

    def test_ca_not_valid_before_is_backdated(self, tmp_path):
        """CA not_valid_before should be <= now (clock-skew backdate)."""
        now = datetime.datetime.now(datetime.timezone.utc)
        store = CAStore(tmp_path / "tls")
        assert store.cert.not_valid_before <= now.replace(tzinfo=None)


class TestLeafMinter:
    """Leaf certificate minting and caching."""

    @pytest.fixture()
    def ca(self, tmp_path):
        return CAStore(tmp_path / "tls")

    @pytest.fixture()
    def minter(self, ca):
        return LeafMinter(ca)

    def test_leaf_has_dns_san_for_hostname(self, minter):
        """Hostname leaf gets a DNSName SAN."""
        cert, _key = minter.get_or_mint("example.com")
        san = cert.extensions.get_extension_for_class(
            x509.SubjectAlternativeName
        ).value
        dns_names = san.get_values_for_type(x509.DNSName)
        assert "example.com" in dns_names

    def test_leaf_has_ip_san_for_ip_literal(self, minter):
        """IP-literal leaf gets an IPAddress SAN."""
        cert, _key = minter.get_or_mint("192.168.1.1")
        san = cert.extensions.get_extension_for_class(
            x509.SubjectAlternativeName
        ).value
        ips = san.get_values_for_type(x509.IPAddress)
        assert ipaddress.IPv4Address("192.168.1.1") in ips

    def test_leaf_has_ip_san_for_ipv6(self, minter):
        """IPv6 literal leaf gets an IPAddress SAN."""
        cert, _key = minter.get_or_mint("::1")
        san = cert.extensions.get_extension_for_class(
            x509.SubjectAlternativeName
        ).value
        ips = san.get_values_for_type(x509.IPAddress)
        assert ipaddress.IPv6Address("::1") in ips

    def test_leaf_cached_per_host(self, minter):
        """Same host returns cached (same object) on second call."""
        cert1, key1 = minter.get_or_mint("cached.test")
        cert2, key2 = minter.get_or_mint("cached.test")
        assert cert1 is cert2
        assert key1 is key2

    def test_different_hosts_get_different_certs(self, minter):
        """Different hosts produce different certificates."""
        cert_a, _ = minter.get_or_mint("a.test")
        cert_b, _ = minter.get_or_mint("b.test")
        assert cert_a.serial_number != cert_b.serial_number

    def test_leaf_signed_by_ca(self, ca, minter):
        """Leaf certificate is verifiable against the CA public key."""
        cert, _key = minter.get_or_mint("verify.test")
        # Verify signature: the CA public key should verify the leaf cert.
        ca_public_key = ca.key.public_key()
        # This will raise InvalidSignature if verification fails.
        ca_public_key.verify(
            cert.signature,
            cert.tbs_certificate_bytes,
            padding.PKCS1v15(),
            cert.signature_hash_algorithm,
        )

    def test_leaf_not_ca(self, minter):
        """Leaf cert must have BasicConstraints(ca=False)."""
        cert, _ = minter.get_or_mint("notca.test")
        bc = cert.extensions.get_extension_for_class(
            x509.BasicConstraints
        ).value
        assert bc.ca is False

    def test_leaf_validity_24h(self, minter):
        """Leaf cert validity period is ~24h + 5min skew."""
        cert, _ = minter.get_or_mint("ttl.test")
        delta = cert.not_valid_after - cert.not_valid_before
        # 24h + 5min backdate = ~24h5m total window.
        total_secs = delta.total_seconds()
        assert 24 * 3600 <= total_secs <= 25 * 3600

    def test_get_cert_and_key_pem(self, minter):
        """get_cert_and_key_pem returns valid PEM bytes."""
        cert_pem, key_pem = minter.get_cert_and_key_pem("pem.test")
        assert cert_pem.startswith(b"-----BEGIN CERTIFICATE-----")
        assert key_pem.startswith(b"-----BEGIN RSA PRIVATE KEY-----")

    def test_leaf_issuer_matches_ca_subject(self, ca, minter):
        """Leaf issuer field matches CA subject."""
        cert, _ = minter.get_or_mint("issuer.test")
        assert cert.issuer == ca.cert.subject

    # --- New tests from review ---

    def test_leaf_has_server_auth_eku(self, minter):
        """Leaf cert must include serverAuth EKU."""
        cert, _ = minter.get_or_mint("eku.test")
        eku = cert.extensions.get_extension_for_class(
            x509.ExtendedKeyUsage
        ).value
        assert x509.oid.ExtendedKeyUsageOID.SERVER_AUTH in eku

    def test_leaf_not_valid_before_is_backdated(self, minter):
        """Leaf not_valid_before should be <= now (clock-skew backdate)."""
        now = datetime.datetime.now(datetime.timezone.utc)
        cert, _ = minter.get_or_mint("skew.test")
        assert cert.not_valid_before <= now.replace(tzinfo=None)

    def test_cache_evicts_expired_leaf(self, ca):
        """Expired cached leaf is re-minted on next access."""
        minter = LeafMinter(ca)
        # Mint a valid cert first.
        cert1, key1 = minter.get_or_mint("expire.test")
        serial1 = cert1.serial_number

        # Inject a synthetic expired cert into the cache.
        expired_key = rsa.generate_private_key(
            public_exponent=65537, key_size=2048
        )
        now = datetime.datetime.now(datetime.timezone.utc)
        expired_cert = (
            x509.CertificateBuilder()
            .subject_name(x509.Name([
                x509.NameAttribute(x509.oid.NameOID.COMMON_NAME, "expire.test")
            ]))
            .issuer_name(ca.cert.subject)
            .public_key(expired_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(hours=48))
            .not_valid_after(now - datetime.timedelta(hours=1))
            .add_extension(
                x509.SubjectAlternativeName([x509.DNSName("expire.test")]),
                critical=False,
            )
            .add_extension(
                x509.BasicConstraints(ca=False, path_length=None),
                critical=True,
            )
            .sign(ca.key, hashes.SHA256())
        )
        # Force inject expired entry.
        with minter._lock:
            minter._cache["expire.test"] = (expired_cert, expired_key)

        # Now get_or_mint should detect expiry and re-mint.
        cert2, key2 = minter.get_or_mint("expire.test")
        assert cert2.serial_number != expired_cert.serial_number
        assert cert2.not_valid_after > now.replace(tzinfo=None)

    def test_cache_capacity_limit(self, ca):
        """Cache respects capacity limit and evicts oldest entries."""
        small_max = 5
        minter = LeafMinter(ca, cache_max=small_max)
        # Fill beyond capacity.
        for i in range(small_max + 3):
            minter.get_or_mint(f"host-{i}.test")
        assert len(minter._cache) == small_max

    def test_mint_rejects_empty_host(self, minter):
        """Empty host raises ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            minter.get_or_mint("")

    def test_mint_rejects_host_with_port(self, minter):
        """Non-IP host containing ':' (port) raises ValueError."""
        with pytest.raises(ValueError, match="port separator"):
            minter.get_or_mint("example.com:443")


class TestTLSEffectRejectLoopback:
    """A rejected HTTPS write must receive its 503 before TLS closes."""

    def test_nonempty_post_receives_503(self, tmp_path):
        interceptor = TLSInterceptor(tmp_path / "tls")
        sni = "reject.example.com"
        sandbox_id = SandboxId("sbx-tls-reject")
        gate = EffectGate()
        gate.begin(sandbox_id, policy="reject")
        proxy = EgressProxyServer(
            journal=ActionJournal(tmp_path / "journal"),
            sandbox_id_resolver=lambda _peer: sandbox_id,
            host="127.0.0.1",
            port=0,
            head_timeout_seconds=1.0,
            effect_gate=gate,
            tls_interceptor=interceptor,
        )

        client_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        client_ctx.load_verify_locations(
            cadata=interceptor.ca_store.cert_pem().decode()
        )
        client_ctx.set_alpn_protocols(_ALPN_PROTOCOLS)
        request_head = (
            b"POST /write HTTP/1.1\r\n"
            b"Host: reject.example.com\r\n"
            b"Content-Length: 7\r\n\r\n"
        )

        with mock.patch(
            "crab.egress.original_destination",
            return_value=("127.0.0.1", 443),
        ):
            proxy.start()
            try:
                with socket.create_connection(
                    ("127.0.0.1", proxy.port), timeout=5.0
                ) as raw_client:
                    with client_ctx.wrap_socket(
                        raw_client, server_hostname=sni
                    ) as tls_client:
                        # Separate writes become separate TLS records. The
                        # proxy's first recv sees the head while the payload
                        # remains unread, matching urllib's real-host flow.
                        tls_client.sendall(request_head)
                        tls_client.sendall(b"payload")

                        response = http.client.HTTPResponse(tls_client)
                        response.begin()
                        assert response.status == 503
                        assert response.getheader("X-Crab-Effect") == "rejected"
                        assert response.read() == (
                            b"crab: mutating egress refused by txn policy\n"
                        )
            finally:
                proxy.stop()

        assert gate.counters(sandbox_id)[1] == 1
