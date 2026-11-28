"""TLS interception helpers for the egress proxy (roadmap T1.2).

This module is **lazily imported** only when TLS interception is enabled at
runtime.  It depends on ``crab.tls_ca`` (which itself requires ``cryptography``
from the ``crab[tls]`` extra).  No resident code path imports this module.

Responsibilities:
- Build server-side and upstream-client SSLContexts for interception.
- Pre-termination filter (bypass_hosts, port filter, ALPN check).
- Runtime bypass set (hosts that failed handshake get auto-bypassed).
"""
from __future__ import annotations

import fnmatch
import logging
import os
import ssl
import tempfile
import threading
from pathlib import Path
from typing import Sequence

from .tls_ca import CAStore, LeafMinter

logger = logging.getLogger(__name__)

# Standard HTTPS ports considered for interception.  Connections to
# other ports (e.g. 5432/postgres, 3306/mysql) are left opaque.
_WEB_PORTS: frozenset[int] = frozenset({443, 8443})

# ALPN we advertise — only HTTP/1.1 (decision 7).
_ALPN_PROTOCOLS: list[str] = ["http/1.1"]


def _matches_bypass(host: str, patterns: Sequence[str]) -> bool:
    """Return True if *host* matches any bypass glob pattern."""
    lower = host.lower()
    return any(fnmatch.fnmatch(lower, p.lower()) for p in patterns)


class TLSInterceptor:
    """Manages CA/leaf state and SSLContext construction for egress TLS
    interception.  One instance per ``EgressProxyServer``.

    Parameters
    ----------
    storage_dir : Path | str
        Directory for CA material (``<storage_root>/tls``).
    bypass_hosts : sequence of str
        Glob patterns for hosts that are never intercepted.
    on_handshake_failure : str
        ``"passthrough"`` or ``"refuse"``.
    """

    def __init__(
        self,
        storage_dir: Path | str,
        *,
        bypass_hosts: Sequence[str] = (),
        on_handshake_failure: str = "passthrough",
    ) -> None:
        self._ca = CAStore(storage_dir)
        self._minter = LeafMinter(self._ca)
        self._bypass_patterns = tuple(bypass_hosts)
        self.on_handshake_failure = on_handshake_failure
        # Runtime bypass: hosts whose handshake failed under "passthrough"
        # are added here so subsequent connections go opaque immediately.
        self._runtime_bypass: set[str] = set()
        self._bypass_lock = threading.Lock()

    @property
    def ca_store(self) -> CAStore:
        return self._ca

    def should_intercept(self, sni: str | None, dst_port: int) -> bool:
        """Pre-termination decision: should this flow be intercepted?

        Returns False (leave opaque) when:
        - SNI is absent (no-SNI / IP-literal — decision: don't intercept)
        - Host matches bypass_hosts or runtime bypass
        - Destination port is not a standard web port
        """
        if not sni:
            return False
        if dst_port not in _WEB_PORTS:
            return False
        if _matches_bypass(sni, self._bypass_patterns):
            return False
        with self._bypass_lock:
            if sni in self._runtime_bypass:
                return False
        return True

    def add_runtime_bypass(self, host: str) -> None:
        """Record a host whose handshake failed — future connections skip."""
        with self._bypass_lock:
            self._runtime_bypass.add(host.lower())
        logger.info("TLS interception: added runtime bypass for %s", host)

    def build_server_context(self, sni: str) -> ssl.SSLContext:
        """SSLContext for the server side (proxy ↔ sandbox client).

        Mints a leaf cert for *sni* on the fly, loads it, and sets
        ALPN to ``["http/1.1"]``.
        """
        cert_pem, key_pem = self._minter.get_cert_and_key_pem(sni)
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.set_alpn_protocols(_ALPN_PROTOCOLS)
        # Write cert+key to temp files for load_cert_chain (no in-memory API
        # until Python 3.12 for SSLContext).
        cert_file = tempfile.NamedTemporaryFile(
            suffix=".pem", delete=False, mode="wb"
        )
        key_file = tempfile.NamedTemporaryFile(
            suffix=".pem", delete=False, mode="wb"
        )
        try:
            cert_file.write(cert_pem)
            cert_file.close()
            key_file.write(key_pem)
            key_file.close()
            ctx.load_cert_chain(cert_file.name, key_file.name)
        finally:
            os.unlink(cert_file.name)
            os.unlink(key_file.name)
        return ctx

    def build_upstream_context(self, sni: str) -> ssl.SSLContext:
        """SSLContext for the upstream leg (proxy → real server).

        Verifies the real server's cert normally, and advertises only
        ``http/1.1`` in ALPN.
        """
        ctx = ssl.create_default_context()
        ctx.set_alpn_protocols(_ALPN_PROTOCOLS)
        return ctx


__all__ = ["TLSInterceptor"]
