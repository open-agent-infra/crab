"""Transparent egress interception (roadmap D1).

All sandbox TCP egress is redirected (iptables REDIRECT on the bridge)
into ``EgressProxyServer``, which recovers the original destination via
``SO_ORIGINAL_DST``, sniffs just enough of the head to identify the
flow (HTTP request line + Host, or a TLS ClientHello's SNI), splices
the bytes both ways untouched, and appends one ``kind="egress"``
journal record per connection.

With TLS interception enabled (roadmap T1), a ClientHello on a standard
HTTPS port with a valid SNI is terminated in-proxy: the proxy mints a
leaf cert for the SNI, completes the handshake with the sandbox client,
re-reads the now-decrypted head, then opens an upstream TLS connection
to the real server. The decrypted flow enters the same replay → gate →
record pipeline as plaintext HTTP, with ``scheme="https"``.
"""
from __future__ import annotations

import fnmatch
import logging
import socket
import socketserver
import ssl
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

from .cassettes import (
    DEFAULT_MAX_BODY_BYTES,
    DEFAULT_RECORDABLE_STATUSES,
    DEFAULT_VARYING_HEADERS,
    PARTIAL_STATUS,
    REDIRECT_STATUSES,
    RESPONSE_HEADER_DENY_LIST,
    CassetteEntry,
    filter_headers,
    request_key,
    sha256_hex,
)
from .effects import (
    DECISION_DEFER,
    DECISION_REJECT,
    DECISION_SEAL,
    DEFERRED_RESPONSE,
    REJECTED_RESPONSE,
    build_deferred_request,
    read_remaining_body,
)
from .http_wire import ResponseAssembler, parse_head, serialize_response
from .ids import SandboxId
from .models import utc_now

logger = logging.getLogger(__name__)

# Linux: getsockopt(SOL_IP, SO_ORIGINAL_DST) yields the pre-REDIRECT
# destination as a sockaddr_in.
SO_ORIGINAL_DST = 80

_HEAD_PEEK_BYTES = 2048
_SPLICE_CHUNK = 65536
# TLS record: 5-byte header + up to 16384 payload (RFC 8446 §5.1).
# Used to expand the MSG_PEEK for ClientHello sniffing so that SNI/ALPN
# in large hellos (PQ key_share, many extensions) are not missed.
# Multi-record fragmented ClientHellos are out-of-scope (extremely rare).
_TLS_RECORD_MAX_PEEK = 5 + 16384
_HTTP_METHODS = frozenset(
    {"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "TRACE", "CONNECT"}
)

# Schemes whose decrypted head is visible to replay/gate/recording.
_PLAINTEXT_VISIBLE = frozenset({"http", "https"})

# Sent under cassette_only when nothing matches: hermetic replay must
# fail loudly instead of quietly reaching the network.
_REPLAY_MISS_RESPONSE = (
    b"HTTP/1.1 504 Gateway Timeout\r\n"
    b"X-Crab-Replay: miss\r\n"
    b"Content-Length: 24\r\n"
    b"Connection: close\r\n\r\n"
    b"crab: no cassette match\n"
)

SandboxIdResolver = Callable[[str], "SandboxId | None"]

# Effect classes (D1.2). Without TLS interception the method of an
# encrypted request is invisible, so TLS/raw flows are "opaque" — the
# roadmap's explicit degraded mode. D2 (record/replay) owns bodies.
CLASSIFICATION_IDEMPOTENT_READ = "idempotent_read"
CLASSIFICATION_MUTATING = "mutating"
CLASSIFICATION_OPAQUE = "opaque"
EGRESS_CLASSIFICATIONS: tuple[str, ...] = (
    CLASSIFICATION_IDEMPOTENT_READ,
    CLASSIFICATION_MUTATING,
    CLASSIFICATION_OPAQUE,
)

_READ_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "TRACE"})
_MUTATING_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


def _strip_port_for_sni(host: str) -> str:
    """Strip the port suffix from *host* for use as TLS server_hostname.

    TLS certificates carry bare domain names (no port), so the value passed
    to ``wrap_socket(server_hostname=...)`` must never include ``:port``.
    IPv6 addresses wrapped in brackets are also unwrapped:

    * ``"example.com:8443"`` → ``"example.com"``
    * ``"[::1]:8443"`` → ``"::1"``
    * ``"example.com"`` → ``"example.com"`` (unchanged)
    * ``"::1"`` → ``"::1"`` (unchanged)
    """
    if host.startswith("["):
        closing = host.find("]")
        if closing == -1:
            return host  # malformed, return as-is
        # Strip brackets and any trailing :port
        return host[1:closing]
    # A single colon with digits after → host:port; multiple colons → bare IPv6.
    if host.count(":") == 1:
        name, _, port = host.rpartition(":")
        if port.isdigit():
            return name
    return host


@dataclass(frozen=True)
class EgressRule:
    """Host-scoped classification override, e.g. mark a known read-only
    internal API's TLS traffic as ``idempotent_read`` even though the
    proxy cannot see the method."""

    host_glob: str
    classify: str

    def __post_init__(self) -> None:
        if self.classify not in EGRESS_CLASSIFICATIONS:
            raise ValueError(
                f"unknown egress classification: {self.classify!r} "
                f"(expected one of {EGRESS_CLASSIFICATIONS})"
            )

    def matches(self, host: str) -> bool:
        return fnmatch.fnmatch(host.lower(), self.host_glob.lower())

    @classmethod
    def from_json(cls, payload: dict) -> "EgressRule":
        return cls(
            host_glob=str(payload.get("host_glob") or payload.get("host") or "*"),
            classify=str(payload["classify"]),
        )


def classify_flow(payload: dict, rules: Sequence[EgressRule] = ()) -> str:
    """Effect class for one flow payload. Protocol-level default first
    (HTTP method table; everything encrypted or unrecognized is opaque),
    then the first matching host rule wins — rules exist precisely to
    refine what the protocol cannot reveal.

    Pure and re-derivable: stored rows can be reclassified later without
    replaying traffic.
    """
    method = payload.get("method")
    if isinstance(method, str):
        upper = method.upper()
        if upper in _MUTATING_METHODS:
            default = CLASSIFICATION_MUTATING
        elif upper in _READ_METHODS:
            default = CLASSIFICATION_IDEMPOTENT_READ
        else:
            default = CLASSIFICATION_OPAQUE
    else:
        default = CLASSIFICATION_OPAQUE
    host = str(payload.get("host") or "")
    for rule in rules:
        if rule.matches(host):
            return rule.classify
    return default


def parse_original_dst(packed: bytes) -> tuple[str, int]:
    """Decode a ``sockaddr_in`` from SO_ORIGINAL_DST into (ip, port)."""
    if len(packed) < 8:
        raise ValueError(f"short sockaddr_in: {len(packed)} bytes")
    port, = struct.unpack("!H", packed[2:4])
    return socket.inet_ntoa(packed[4:8]), int(port)


def original_destination(sock: socket.socket) -> tuple[str, int] | None:
    """The connection's pre-redirect destination, or None when the
    socket was not redirected (direct connect, or a platform without
    SO_ORIGINAL_DST)."""
    try:
        packed = sock.getsockopt(socket.SOL_IP, SO_ORIGINAL_DST, 16)
    except OSError:
        return None
    try:
        return parse_original_dst(packed)
    except ValueError:
        return None


def sniff_http_head(head: bytes) -> tuple[str, str, str] | None:
    """``(method, path, host)`` for a plaintext HTTP request head, or
    None when the bytes are not HTTP. ``host`` is "" when the request
    carries no Host header (HTTP/1.0)."""
    newline = head.find(b"\r\n")
    if newline < 0:
        return None
    parts = head[:newline].decode("latin-1").split(" ")
    if len(parts) != 3 or not parts[2].startswith("HTTP/"):
        return None
    method, path = parts[0], parts[1]
    if method not in _HTTP_METHODS:
        return None
    host = ""
    for raw_line in head[newline + 2:].split(b"\r\n"):
        if not raw_line:
            break
        name, _, value = raw_line.partition(b":")
        if name.strip().lower() == b"host":
            host = value.strip().decode("latin-1")
            break
    return method, path, host


def sniff_tls_sni(head: bytes) -> str | None:
    """The SNI host from a TLS ClientHello, or None when the bytes are
    not a ClientHello or carry no SNI extension. Parses only — nothing
    is decrypted or rewritten."""
    # TLS record: type(1) version(2) length(2); handshake: type(1) len(3)
    if len(head) < 45 or head[0] != 0x16 or head[5] != 0x01:
        return None
    try:
        pos = 9 + 2 + 32  # handshake header + client_version + random
        session_len = head[pos]
        pos += 1 + session_len
        cipher_len, = struct.unpack("!H", head[pos:pos + 2])
        pos += 2 + cipher_len
        compression_len = head[pos]
        pos += 1 + compression_len
        if pos + 2 > len(head):
            return None
        extensions_len, = struct.unpack("!H", head[pos:pos + 2])
        pos += 2
        end = min(pos + extensions_len, len(head))
        while pos + 4 <= end:
            ext_type, ext_len = struct.unpack("!HH", head[pos:pos + 4])
            pos += 4
            if ext_type == 0x0000:  # server_name
                # list_len(2) name_type(1) name_len(2) name
                if pos + 5 > len(head):
                    return None
                name_len, = struct.unpack("!H", head[pos + 3:pos + 5])
                name = head[pos + 5:pos + 5 + name_len]
                # SNI is an A-label (ASCII); IDN hosts arrive punycoded and
                # stay that way — the ledger records host identity, not a
                # display form (the idna codec rejects error handlers).
                return name.decode("ascii", errors="replace").lower() if name else None
            pos += ext_len
    except (IndexError, struct.error):
        return None
    return None


def sniff_tls_alpn(head: bytes) -> list[str] | None:
    """Extract ALPN protocol list from a TLS ClientHello, or None.

    Returns None when the bytes are not a ClientHello, the extensions
    area is truncated, or no ALPN extension (type 0x0010) is present.
    Returns an empty list only if the extension is present but contains
    no protocol names (malformed).
    """
    if len(head) < 45 or head[0] != 0x16 or head[5] != 0x01:
        return None
    try:
        pos = 9 + 2 + 32
        session_len = head[pos]
        pos += 1 + session_len
        cipher_len, = struct.unpack("!H", head[pos:pos + 2])
        pos += 2 + cipher_len
        compression_len = head[pos]
        pos += 1 + compression_len
        if pos + 2 > len(head):
            return None
        extensions_len, = struct.unpack("!H", head[pos:pos + 2])
        pos += 2
        end = min(pos + extensions_len, len(head))
        while pos + 4 <= end:
            ext_type, ext_len = struct.unpack("!HH", head[pos:pos + 4])
            pos += 4
            if ext_type == 0x0010:  # application_layer_protocol_negotiation
                # ALPN body: list_len(2) then [name_len(1) name]...
                if pos + 2 > end:
                    return None
                list_len, = struct.unpack("!H", head[pos:pos + 2])
                alpn_pos = pos + 2
                alpn_end = min(alpn_pos + list_len, end)
                protocols: list[str] = []
                while alpn_pos < alpn_end:
                    name_len = head[alpn_pos]
                    alpn_pos += 1
                    if alpn_pos + name_len > alpn_end:
                        return None
                    protocols.append(
                        head[alpn_pos:alpn_pos + name_len].decode("ascii", errors="replace")
                    )
                    alpn_pos += name_len
                return protocols
            pos += ext_len
    except (IndexError, struct.error):
        return None
    return None


class EgressFlowRecorder:
    """Journal-backed sink for completed flows (the effect ledger's only
    store — same ruling as C3: the journal is the durable, ordered,
    txn-stamped account). Recording failures never break the flow."""

    def __init__(self, journal) -> None:
        self._journal = journal

    def record(self, sandbox_id: SandboxId, payload: dict) -> None:
        if self._journal is None:
            return
        try:
            self._journal.record_egress(sandbox_id, payload=payload)
        except Exception:
            logger.debug("Failed to record egress flow", exc_info=True)


class CassetteRecorder:
    """Turns a completed plaintext exchange into a cassette (D2).

    Returns the journal index fields for the flow, or None when the
    exchange is not recordable — an unparsable, oversized or
    uninteresting response degrades to "no cassette", never to a broken
    flow. Credentials are stripped before anything is written.
    """

    def __init__(
        self,
        store,
        *,
        max_body_bytes: int = DEFAULT_MAX_BODY_BYTES,
        record_errors: bool = False,
        record_partial: bool = False,
        varying_headers: Sequence[str] = DEFAULT_VARYING_HEADERS,
    ) -> None:
        self._store = store
        self.max_body_bytes = int(max_body_bytes)
        self.record_errors = bool(record_errors)
        self.record_partial = bool(record_partial)
        self.varying_headers = tuple(varying_headers)

    def _recordable_status(self, status: int) -> bool:
        if status in DEFAULT_RECORDABLE_STATUSES or status in REDIRECT_STATUSES:
            return True
        if status == PARTIAL_STATUS:
            # Ranged reads need the Range header in the key, or a second
            # range would be served the first one's bytes.
            return self.record_partial and "range" in {
                name.lower() for name in self.varying_headers
            }
        return self.record_errors and 400 <= status < 600

    def record_exchange(
        self,
        *,
        sandbox_id,
        request_head: bytes,
        host: str,
        port: int,
        method: str,
        path: str,
        assembler: ResponseAssembler,
        bytes_in: int,
    ) -> dict | None:
        request = parse_head(request_head)
        if request is None:
            return None
        request_body = request.rest
        request_body_digest = sha256_hex(request_body)
        key = request_key(
            method=method,
            host=host,
            port=port,
            path=path,
            body_sha256=request_body_digest,
            headers=list(request.headers),
            varying_headers=self.varying_headers,
        )
        if assembler.truncated:
            # Visible in the ledger, never replayable.
            return {
                "recorded": False,
                "request_key": key,
                "truncated": True,
            }
        parsed = assembler.result()
        if parsed is None:
            return None
        _response_head, status, reason, body = parsed
        if not self._recordable_status(status):
            return None
        entry = CassetteEntry(
            request_key=key,
            method=method.upper(),
            host=host,
            port=int(port),
            path=path,
            status=status,
            reason=reason,
            response_headers=tuple(
                filter_headers(list(_response_head.headers), RESPONSE_HEADER_DENY_LIST)
            ),
            body=body,
            body_sha256=sha256_hex(body),
            request_body_sha256=request_body_digest,
            bytes_in=bytes_in,
            recorded_at=utc_now().isoformat(),
            origin_sandbox_id=str(sandbox_id),
        )
        try:
            self._store.put(sandbox_id, entry)
        except Exception:
            logger.debug("Failed to write cassette", exc_info=True)
            return None
        return {
            "recorded": True,
            "request_key": key,
            "response_sha256": entry.body_sha256,
            "status": status,
            "truncated": False,
        }


REPLAY_POLICIES: tuple[str, ...] = ("none", "cassette_first", "cassette_only")


@dataclass
class ReplaySession:
    """One sandbox's replay window (D2). ``cassette_source`` is the bucket
    to read: for C4 the reads happened in the *fork* while the replay runs
    on the *source*, so without redirecting the lookup every request
    misses and hits the network instead."""

    policy: str
    cassette_source: str
    served: int = 0
    missed: int = 0
    passed_through: int = 0
    hosts: set = field(default_factory=set)


class CassetteReplayer:
    """Serves recorded responses in place of live requests.

    The replay gate is re-evaluated here with the *current* rules rather
    than trusting the row's stored class — a host reclassified as
    mutating/opaque must stop being served even though its cassettes are
    still on disk (the same read-time re-derivation D1.2 established).
    Mutating and opaque flows always pass through: swallowing a write
    here would pre-empt D3 and hide a real side effect.
    """

    def __init__(self, store) -> None:
        self._store = store
        self._sessions: dict[str, ReplaySession] = {}
        self._lock = threading.Lock()

    def begin(self, sandbox_id, *, policy: str, cassette_source=None) -> ReplaySession:
        if policy not in REPLAY_POLICIES or policy == "none":
            raise ValueError(
                f"unknown replay policy: {policy!r} "
                f"(expected one of {REPLAY_POLICIES[1:]})"
            )
        session = ReplaySession(
            policy=policy,
            cassette_source=str(cassette_source or sandbox_id),
        )
        with self._lock:
            self._sessions[str(sandbox_id)] = session
        return session

    def end(self, sandbox_id) -> ReplaySession | None:
        with self._lock:
            return self._sessions.pop(str(sandbox_id), None)

    def session_for(self, sandbox_id) -> ReplaySession | None:
        with self._lock:
            return self._sessions.get(str(sandbox_id))

    # The proxy is one thread per connection, so the session's tallies are
    # mutated concurrently; ``+=`` is a lost-update race. Fold every
    # counter bump through the lock so the report (C4's determinism input)
    # stays accurate under real traffic.
    def record_hit(self, session: ReplaySession, host: str) -> None:
        with self._lock:
            session.served += 1
            session.hosts.add(host)

    def record_miss(self, session: ReplaySession) -> None:
        with self._lock:
            session.missed += 1

    def record_pass_through(self, session: ReplaySession) -> None:
        with self._lock:
            session.passed_through += 1

    def snapshot(self, session: ReplaySession) -> tuple[int, int, int, tuple[str, ...]]:
        with self._lock:
            return (
                session.served,
                session.missed,
                session.passed_through,
                tuple(sorted(session.hosts)),
            )

    def lookup(
        self,
        session: ReplaySession,
        *,
        request_head: bytes,
        host: str,
        port: int,
        method: str,
        path: str,
        varying_headers: Sequence[str],
    ) -> CassetteEntry | None:
        request = parse_head(request_head)
        if request is None:
            return None
        key = request_key(
            method=method,
            host=host,
            port=port,
            path=path,
            body_sha256=sha256_hex(request.rest),
            headers=list(request.headers),
            varying_headers=varying_headers,
        )
        entry = self._store.get(session.cassette_source, key)
        if entry is None or entry.truncated:
            # A partial response is worse than a live one.
            return None
        return entry


def _send_rejected_response(
    client: socket.socket,
    request_head: bytes,
    *,
    drain_timeout_seconds: float,
) -> None:
    """Return the synthetic 503 without resetting a TLS connection.

    HTTP clients commonly write headers and a body as separate TLS records.
    The proxy only needs the first record to reject the request, but closing
    the socket while the body remains unread can make Linux send a TCP RST and
    discard the 503 that was just written. Send the response promptly, then
    best-effort drain the declared body before the handler closes the socket.
    An absolute deadline keeps a slow or malicious peer from holding the
    handler thread indefinitely.
    """
    client.sendall(REJECTED_RESPONSE)

    parsed = parse_head(request_head)
    if parsed is None:
        return
    content_length = parsed.content_length()
    if content_length is None:
        return
    remaining = max(0, content_length - len(parsed.rest))
    if remaining == 0:
        return

    deadline = time.monotonic() + max(0.0, drain_timeout_seconds)
    previous_timeout = client.gettimeout()
    try:
        while remaining > 0:
            timeout = deadline - time.monotonic()
            if timeout <= 0:
                break
            client.settimeout(timeout)
            try:
                chunk = client.recv(min(_SPLICE_CHUNK, remaining))
            except (socket.timeout, OSError):
                break
            if not chunk:
                break
            remaining -= len(chunk)
    finally:
        try:
            client.settimeout(previous_timeout)
        except OSError:
            pass


class _EgressHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:  # noqa: C901 - straight-line proxy flow
        server: "EgressProxyServer" = self.server  # type: ignore[assignment]
        client = self.request
        peer_ip = self.client_address[0]
        destination = original_destination(client)
        if destination is None:
            # Not redirected: nothing to proxy to. Reject rather than
            # guess a destination (a direct hit on the proxy port).
            logger.debug("Dropping non-redirected connection from %s", peer_ip)
            return
        dst_ip, dst_port = destination
        sandbox_id = server.resolve_sandbox_id(peer_ip)
        started = time.monotonic()
        head = b""
        scheme = "tcp"
        method: str | None = None
        path: str | None = None
        host = ""
        bytes_out = 0
        bytes_in = 0
        record_meta: dict | None = None
        upstream: socket.socket | None = None
        tls_interceptor = server.tls_interceptor
        try:
            client.settimeout(server.head_timeout_seconds)
            try:
                head = client.recv(_HEAD_PEEK_BYTES, socket.MSG_PEEK)
            except (socket.timeout, OSError):
                head = b""
            client.settimeout(None)

            # --- TLS interception branch (T1.2) ---
            # Detect ClientHello; if interception is enabled and the
            # pre-filter passes, terminate TLS in-proxy.
            # Record-length-aware peek: a modern ClientHello with PQ/hybrid
            # key_share can exceed 2048 bytes, pushing SNI/ALPN past the
            # initial peek.  Re-peek using the record length from the 5-byte
            # TLS record header (capped at _TLS_RECORD_MAX_PEEK).
            if len(head) >= 5 and head[0:1] == b"\x16":
                record_len = struct.unpack("!H", head[3:5])[0] + 5
                needed = min(record_len, _TLS_RECORD_MAX_PEEK)
                if needed > len(head):
                    client.settimeout(server.head_timeout_seconds)
                    try:
                        head = client.recv(needed, socket.MSG_PEEK)
                    except (socket.timeout, OSError):
                        pass  # keep whatever we got from first peek
                    client.settimeout(None)
            sni = sniff_tls_sni(head) if head and head[0:1] == b"\x16" else None
            # ALPN pre-filter (decision 9): if ALPN is present and does
            # NOT include "http/1.1", leave the flow opaque — no
            # termination.  Missing ALPN is fine (many HTTP clients omit it).
            alpn = sniff_tls_alpn(head) if sni is not None else None
            alpn_ok = alpn is None or "http/1.1" in alpn
            if (
                sni is not None
                and alpn_ok
                and tls_interceptor is not None
                and tls_interceptor.should_intercept(sni, dst_port)
            ):
                # Attempt server-side TLS termination with minted leaf.
                try:
                    server_ctx = tls_interceptor.build_server_context(sni)
                    client = server_ctx.wrap_socket(
                        client, server_side=True
                    )
                except ssl.SSLError as exc:
                    logger.debug(
                        "TLS handshake failed for %s: %s", sni, exc
                    )
                    if tls_interceptor.on_handshake_failure == "passthrough":
                        tls_interceptor.add_runtime_bypass(sni)
                    # Close: client must retry (will go opaque/bypass).
                    return
                # Re-read decrypted head from the now-wrapped socket.
                client.settimeout(server.head_timeout_seconds)
                try:
                    head = client.recv(_HEAD_PEEK_BYTES)
                except (socket.timeout, OSError):
                    head = b""
                client.settimeout(None)
                http = sniff_http_head(head)
                if http is not None:
                    scheme = "https"
                    method, path, host = http
                else:
                    # Decrypted head is not HTTP/1.1: downgrade to opaque
                    # byte-tunnel (still terminated, bytes spliced, no
                    # HTTP parsing). Decision 9.
                    scheme = "tls"
                    host = sni
                    upstream = socket.create_connection(
                        (dst_ip, dst_port),
                        timeout=server.connect_timeout_seconds,
                    )
                    upstream.settimeout(None)
                    upstream_ctx = tls_interceptor.build_upstream_context(sni)
                    upstream = upstream_ctx.wrap_socket(
                        upstream, server_hostname=sni
                    )
                    if head:
                        upstream.sendall(head)
                        bytes_out += len(head)
                    out_extra, in_extra = _splice(client, upstream)
                    bytes_out += out_extra
                    bytes_in += in_extra
                    return  # finally writes journal row
            else:
                # Consume head now (non-TLS-interception path).
                # The peek above was MSG_PEEK; consume it for the legacy path.
                if head:
                    client.settimeout(server.head_timeout_seconds)
                    try:
                        head = client.recv(_HEAD_PEEK_BYTES)
                    except (socket.timeout, OSError):
                        head = b""
                    client.settimeout(None)
                http = sniff_http_head(head)
                if http is not None:
                    scheme = "http"
                    method, path, host = http
                else:
                    sni_check = sniff_tls_sni(head)
                    if sni_check is not None:
                        scheme = "tls"
                        host = sni_check
            # ---- replay (D2): a cassette hit never opens an upstream
            # connection. The gate is recomputed here with the current
            # rules, and only reads are eligible — mutating and opaque
            # flows always pass through to the real world.
            replayer = server.cassette_replayer
            session = None if replayer is None or sandbox_id is None else replayer.session_for(sandbox_id)
            if session is not None:
                eligible = scheme in _PLAINTEXT_VISIBLE and classify_flow(
                    {"host": host or dst_ip, "method": method, "scheme": scheme},
                    server.rules,
                ) == CLASSIFICATION_IDEMPOTENT_READ
                if eligible:
                    entry = replayer.lookup(
                        session,
                        request_head=head,
                        host=host or dst_ip,
                        port=dst_port,
                        method=method or "GET",
                        path=path or "/",
                        varying_headers=server.replay_varying_headers,
                    )
                    if entry is not None:
                        wire = serialize_response(
                            status=entry.status,
                            reason=entry.reason,
                            headers=list(entry.response_headers),
                            body=entry.body,
                        )
                        client.sendall(wire)
                        bytes_in = len(wire)
                        replayer.record_hit(session, entry.host)
                        record_meta = {
                            "recorded": True,
                            "request_key": entry.request_key,
                            "response_sha256": entry.body_sha256,
                            "status": entry.status,
                            "replayed": True,
                            "replayed_from_seq": entry.origin_seq,
                            "replayed_from": session.cassette_source,
                        }
                        return  # finally still writes the journal row
                    replayer.record_miss(session)
                    if session.policy == "cassette_only":
                        client.sendall(_REPLAY_MISS_RESPONSE)
                        record_meta = {"replay_miss": True}
                        return
                else:
                    replayer.record_pass_through(session)
            # ---- effect gate (D3): a write may be queued for commit,
            # refused, or sealed. Reads are never touched. Runs after
            # replay so a cassette hit still short-circuits, and before
            # the upstream connection so a held write truly never leaves.
            gate = server.effect_gate
            if gate is not None and sandbox_id is not None:
                decision = None
                gate_session = None
                if scheme in _PLAINTEXT_VISIBLE:
                    verdict = classify_flow(
                        {"host": host or dst_ip, "method": method, "scheme": scheme},
                        server.rules,
                    )
                    if verdict == CLASSIFICATION_MUTATING:
                        outcome = gate.decide_write(
                            sandbox_id,
                            host=host or dst_ip,
                            method=method or "",
                            path=path or "/",
                        )
                        if outcome is not None:
                            decision, gate_session = outcome
                    elif verdict == CLASSIFICATION_OPAQUE:
                        outcome = gate.decide_opaque(sandbox_id)
                        if outcome is not None:
                            decision, gate_session = outcome
                else:
                    # TLS / raw TCP: no method to classify (D3 decision 4).
                    outcome = gate.decide_opaque(sandbox_id)
                    if outcome is not None:
                        decision, gate_session = outcome
                if decision == DECISION_DEFER:
                    parsed = parse_head(head)
                    body, complete = (
                        (b"", False)
                        if parsed is None
                        else read_remaining_body(
                            client, parsed, parsed.rest, limit=server.max_deferred_body_bytes
                        )
                    )
                    if parsed is None or not complete:
                        # Unparsable head, unframed/chunked body, or a body
                        # past the cap: refuse rather than queue a request
                        # that would flush corrupted at commit.
                        _send_rejected_response(
                            client,
                            head,
                            drain_timeout_seconds=server.head_timeout_seconds,
                        )
                        bytes_in = len(REJECTED_RESPONSE)
                        record_meta = {"effect": "rejected", "effect_reason": "unqueueable"}
                        return
                    queued = build_deferred_request(
                        parsed_head=parsed,
                        body=body,
                        host=host or dst_ip,
                        port=dst_port,
                        method=method or "POST",
                        path=path or "/",
                        txn_id=gate_session.txn_id if gate_session else None,
                        enqueued_at=utc_now().isoformat(),
                    )
                    position = gate.enqueue(sandbox_id, queued)
                    if position <= 0:
                        # position == -1: the txn's queue hit its ceiling.
                        # position == 0: the session closed between the
                        # decision and the enqueue (commit/abort raced us).
                        # Either way the write was NOT queued, so answering
                        # 202 would be a lie — refuse instead.
                        _send_rejected_response(
                            client,
                            head,
                            drain_timeout_seconds=server.head_timeout_seconds,
                        )
                        bytes_in = len(REJECTED_RESPONSE)
                        record_meta = {
                            "effect": "rejected",
                            "effect_reason": "queue_full" if position < 0 else "session_closed",
                        }
                        return
                    client.sendall(DEFERRED_RESPONSE)
                    bytes_in = len(DEFERRED_RESPONSE)
                    record_meta = {
                        "effect": "deferred",
                        "effect_queue_position": position,
                        "request_body_sha256": queued.body_sha256,
                    }
                    return  # finally still writes the journal row
                if decision == DECISION_REJECT:
                    _send_rejected_response(
                        client,
                        head,
                        drain_timeout_seconds=server.head_timeout_seconds,
                    )
                    bytes_in = len(REJECTED_RESPONSE)
                    record_meta = {"effect": "rejected"}
                    return
                if decision == DECISION_SEAL:
                    # The write goes out; the txn just lost abortability.
                    record_meta = {"effect": "sealed"}
            upstream = socket.create_connection(
                (dst_ip, dst_port), timeout=server.connect_timeout_seconds
            )
            upstream.settimeout(None)
            # If this is an intercepted HTTPS flow, wrap upstream in TLS.
            if scheme == "https" and tls_interceptor is not None:
                _sni_host = _strip_port_for_sni(host or sni or dst_ip)
                upstream_ctx = tls_interceptor.build_upstream_context(_sni_host)
                upstream = upstream_ctx.wrap_socket(
                    upstream, server_hostname=_sni_host
                )
            if head:
                upstream.sendall(head)
                bytes_out += len(head)
            # Recording gate (D2): plaintext HTTP idempotent reads only.
            # The class is computed once here from the same classify_flow
            # the ledger uses, so "recordable" has one definition.
            recorder = server.cassette_recorder
            assembler = None
            if recorder is not None and scheme in _PLAINTEXT_VISIBLE:
                probe = {"host": host or dst_ip, "method": method, "scheme": scheme}
                if classify_flow(probe, server.rules) == CLASSIFICATION_IDEMPOTENT_READ:
                    assembler = ResponseAssembler(limit=recorder.max_body_bytes)
            out_extra, in_extra = _splice(
                client,
                upstream,
                on_inbound=None if assembler is None else assembler.feed,
            )
            bytes_out += out_extra
            bytes_in += in_extra
            if assembler is not None:
                record_meta = recorder.record_exchange(
                    sandbox_id=sandbox_id,
                    request_head=head,
                    host=host or dst_ip,
                    port=dst_port,
                    method=method or "GET",
                    path=path or "/",
                    assembler=assembler,
                    bytes_in=bytes_in,
                )
        except OSError as exc:
            logger.debug(
                "Egress flow to %s:%s failed: %s", dst_ip, dst_port, exc, exc_info=True
            )
        finally:
            for sock in (upstream, client):
                if sock is not None:
                    try:
                        sock.close()
                    except OSError:
                        pass
            if sandbox_id is None:
                # Unattributable flow (no lease for this peer): counted in
                # the log, not in any sandbox's ledger.
                logger.debug(
                    "Unattributed egress flow peer=%s dst=%s:%s", peer_ip, dst_ip, dst_port
                )
                return
            payload = {
                "host": host or dst_ip,
                "dst_ip": dst_ip,
                "dst_port": dst_port,
                "scheme": scheme,
                "method": method,
                "path": path,
                "bytes_out": bytes_out,
                "bytes_in": bytes_in,
                "duration_ms": round((time.monotonic() - started) * 1000, 3),
            }
            payload["classification"] = classify_flow(payload, server.rules)
            if record_meta:
                # The journal stays the index: key + digest + status only,
                # never the body (that lives in the cassette store).
                payload.update(record_meta)
            server.recorder.record(sandbox_id, payload)


def _splice(
    left: socket.socket,
    right: socket.socket,
    *,
    on_inbound: Callable[[bytes], None] | None = None,
) -> tuple[int, int]:
    """Pump bytes both ways until either side closes. Returns
    ``(left->right, right->left)`` byte counts. ``on_inbound`` tees the
    upstream->client direction for recording; it never alters what the
    client receives."""
    counters = [0, 0]

    def pump(src: socket.socket, dst: socket.socket, slot: int) -> None:
        tee = on_inbound if slot == 1 else None
        try:
            while True:
                chunk = src.recv(_SPLICE_CHUNK)
                if not chunk:
                    break
                dst.sendall(chunk)
                counters[slot] += len(chunk)
                if tee is not None:
                    try:
                        tee(chunk)
                    except Exception:
                        logger.debug("Egress recording tee failed", exc_info=True)
        except OSError:
            pass
        finally:
            try:
                if isinstance(dst, ssl.SSLSocket):
                    dst.unwrap()
                else:
                    dst.shutdown(socket.SHUT_WR)
            except OSError:
                pass

    upstream_pump = threading.Thread(
        target=pump, args=(left, right, 0), daemon=True, name="crab-egress-out"
    )
    upstream_pump.start()
    pump(right, left, 1)
    upstream_pump.join(timeout=5.0)
    return counters[0], counters[1]


class EgressProxyServer(socketserver.ThreadingTCPServer):
    """Host-side transparent proxy for redirected sandbox egress."""

    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        *,
        journal=None,
        sandbox_id_resolver: SandboxIdResolver | None = None,
        host: str = "127.0.0.1",
        port: int = 0,
        head_timeout_seconds: float = 2.0,
        connect_timeout_seconds: float = 10.0,
        rules: Sequence[EgressRule] = (),
        cassette_recorder: "CassetteRecorder | None" = None,
        cassette_replayer: "CassetteReplayer | None" = None,
        replay_varying_headers: Sequence[str] = DEFAULT_VARYING_HEADERS,
        effect_gate: object | None = None,
        max_deferred_body_bytes: int = DEFAULT_MAX_BODY_BYTES,
        tls_interceptor: object | None = None,
    ) -> None:
        """``host`` should be the bridge's own address: REDIRECT rewrites
        each flow's destination to it, so binding there receives all
        redirected traffic without exposing the proxy elsewhere."""
        super().__init__((host, port), _EgressHandler)
        self.recorder = EgressFlowRecorder(journal)
        self.cassette_recorder = cassette_recorder
        self.cassette_replayer = cassette_replayer
        self.effect_gate = effect_gate
        self.max_deferred_body_bytes = int(max_deferred_body_bytes)
        self.replay_varying_headers = tuple(replay_varying_headers)
        self.rules = tuple(rules)
        self.tls_interceptor = tls_interceptor
        self._sandbox_id_resolver = sandbox_id_resolver
        self.head_timeout_seconds = head_timeout_seconds
        self.connect_timeout_seconds = connect_timeout_seconds
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        return int(self.server_address[1])

    def resolve_sandbox_id(self, peer_ip: str) -> SandboxId | None:
        if self._sandbox_id_resolver is None:
            return None
        try:
            return self._sandbox_id_resolver(peer_ip)
        except Exception:
            logger.debug("Egress sandbox attribution failed peer=%s", peer_ip, exc_info=True)
            return None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self.serve_forever, daemon=True, name="crab-egress-proxy"
        )
        self._thread.start()
        logger.info("Egress proxy listening on port %d", self.port)

    def stop(self) -> None:
        # shutdown() waits for the serve_forever loop to acknowledge, so it
        # would hang forever on a server that was never started.
        if self._thread is not None:
            self.shutdown()
        self.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None


__all__ = [
    "SO_ORIGINAL_DST",
    "CLASSIFICATION_IDEMPOTENT_READ",
    "CLASSIFICATION_MUTATING",
    "CLASSIFICATION_OPAQUE",
    "EGRESS_CLASSIFICATIONS",
    "CassetteRecorder",
    "CassetteReplayer",
    "REPLAY_POLICIES",
    "ReplaySession",
    "EgressFlowRecorder",
    "EgressProxyServer",
    "EgressRule",
    "classify_flow",
    "original_destination",
    "parse_original_dst",
    "sniff_http_head",
    "sniff_tls_alpn",
    "sniff_tls_sni",
]
