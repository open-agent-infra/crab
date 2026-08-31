"""Deferred side effects (roadmap D3).

Inside a transaction, a mutating request can be held back instead of
firing immediately, so an abort really does leave the outside world
untouched. The policies and their costs:

- ``allow``   pass through (today's behavior; abort reports what already left)
- ``defer``   queue allow-listed writes, answer ``202``, flush on commit
- ``reject``  refuse with ``503``; nothing is sent
- ``seal``    pass through, but the transaction becomes non-abortable

The load-bearing constraint behind ``defer``: the sandbox process is
blocked on the response while ``commit`` arrives from outside it, so a
held request cannot wait for the commit without deadlocking. Deferral
therefore answers immediately with a synthetic ``202``, which changes
what the application sees — hence the explicit per-endpoint allow-list
(``EffectRule``) rather than a blanket default.

Encrypted and raw flows carry no method, so they cannot be classified as
reads or writes; ``opaque_effects`` decides what happens to them and
defaults to ``allow`` because refusing all TLS would break reads too.
That is the honest limit of D3: a transaction over HTTPS cannot be made
write-free without TLS interception (out of scope since D1).
"""
from __future__ import annotations

import fnmatch
import logging
import threading
from dataclasses import dataclass, field

from .cassettes import REQUEST_HEADER_DENY_LIST, filter_headers, sha256_hex

logger = logging.getLogger(__name__)

EFFECT_POLICIES: tuple[str, ...] = ("allow", "defer", "reject", "seal")
UNLISTED_BEHAVIORS: tuple[str, ...] = ("reject", "allow")
OPAQUE_BEHAVIORS: tuple[str, ...] = ("allow", "reject", "seal")

# Whole-queue ceilings (the per-entry body cap lives on the proxy). Every
# queued entry keeps its body in memory until commit, so a transaction
# must not be able to grow the queue without bound.
DEFAULT_MAX_QUEUE_BYTES = 16 * 1024 * 1024
DEFAULT_MAX_QUEUE_ENTRIES = 256

# What the gate decided for one flow; the proxy turns this into wire
# bytes and the journal records it verbatim.
DECISION_PASS = "pass"
DECISION_DEFER = "defer"
DECISION_REJECT = "reject"
DECISION_SEAL = "seal"

def _synthetic_response(status_line: bytes, effect: bytes, body: bytes) -> bytes:
    """Build a well-formed synthetic response.

    Content-Length is computed, never hand-counted: the first cut of the
    503 declared 41 for a 44-byte body, and lenient clients hid it behind
    ``Connection: close`` while a strict one would have mis-parsed or hung
    waiting for the missing bytes.
    """
    return (
        status_line
        + b"\r\nX-Crab-Effect: "
        + effect
        + b"\r\nContent-Length: "
        + str(len(body)).encode("ascii")
        + b"\r\nConnection: close\r\n\r\n"
        + body
    )


DEFERRED_RESPONSE = _synthetic_response(b"HTTP/1.1 202 Accepted", b"deferred", b"")
REJECTED_RESPONSE = _synthetic_response(
    b"HTTP/1.1 503 Service Unavailable",
    b"rejected",
    b"crab: mutating egress refused by txn policy\n",
)


@dataclass(frozen=True)
class EffectRule:
    """Declares that one endpoint tolerates deferral. Crab cannot infer
    this: only the deployment knows whether a caller can live with
    ``202 Accepted`` instead of the real response."""

    host_glob: str = "*"
    method: str = "*"
    path_glob: str = "*"
    defer: bool = True

    def matches(self, *, host: str, method: str, path: str) -> bool:
        if self.method != "*" and self.method.upper() != method.upper():
            return False
        if not fnmatch.fnmatch(host.lower(), self.host_glob.lower()):
            return False
        return fnmatch.fnmatch(path, self.path_glob)

    @classmethod
    def from_json(cls, payload: dict) -> "EffectRule":
        return cls(
            host_glob=str(payload.get("host_glob") or payload.get("host") or "*"),
            method=str(payload.get("method") or "*"),
            path_glob=str(payload.get("path_glob") or payload.get("path") or "*"),
            defer=bool(payload.get("defer", True)),
        )


@dataclass(frozen=True)
class DeferredRequest:
    """A write held back until commit. Credential headers are stripped
    before it is stored, like cassettes."""

    method: str
    host: str
    port: int
    path: str
    headers: tuple[tuple[str, str], ...]
    body: bytes
    txn_id: str | None = None
    enqueued_at: str | None = None

    @property
    def body_sha256(self) -> str:
        return sha256_hex(self.body)


@dataclass
class EffectSession:
    policy: str
    on_unlisted: str = "reject"
    opaque_effects: str = "allow"
    rules: tuple[EffectRule, ...] = ()
    txn_id: str | None = None
    queue: list = field(default_factory=list)
    # Queue ceilings: a per-entry body cap alone cannot stop a loop that
    # posts a thousand allow-listed writes, and every entry holds its
    # body in the daemon's memory.
    max_queue_bytes: int = DEFAULT_MAX_QUEUE_BYTES
    max_queue_entries: int = DEFAULT_MAX_QUEUE_ENTRIES
    queued_bytes: int = 0
    deferred: int = 0
    rejected: int = 0
    passed: int = 0
    sealed: bool = False


class EffectGate:
    """Per-sandbox effect policy, driven by transaction lifecycle.

    Same shape as ``CassetteReplayer``: the proxy holds no reference to
    ``CrabSystem``, so the system opens and closes sessions. Every
    mutation goes through the lock — the proxy runs one thread per
    connection, so counters and the queue are touched concurrently.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, EffectSession] = {}
        self._lock = threading.Lock()

    def begin(
        self,
        sandbox_id,
        *,
        policy: str,
        on_unlisted: str = "reject",
        opaque_effects: str = "allow",
        rules=(),
        txn_id: str | None = None,
        max_queue_bytes: int = DEFAULT_MAX_QUEUE_BYTES,
        max_queue_entries: int = DEFAULT_MAX_QUEUE_ENTRIES,
    ) -> EffectSession:
        if policy not in EFFECT_POLICIES:
            raise ValueError(
                f"unknown effect policy: {policy!r} (expected one of {EFFECT_POLICIES})"
            )
        if on_unlisted not in UNLISTED_BEHAVIORS:
            raise ValueError(
                f"unknown on_unlisted behavior: {on_unlisted!r} "
                f"(expected one of {UNLISTED_BEHAVIORS})"
            )
        if opaque_effects not in OPAQUE_BEHAVIORS:
            raise ValueError(
                f"unknown opaque_effects behavior: {opaque_effects!r} "
                f"(expected one of {OPAQUE_BEHAVIORS})"
            )
        session = EffectSession(
            policy=policy,
            on_unlisted=on_unlisted,
            opaque_effects=opaque_effects,
            rules=tuple(rules),
            txn_id=txn_id,
            max_queue_bytes=int(max_queue_bytes),
            max_queue_entries=int(max_queue_entries),
        )
        with self._lock:
            self._sessions[str(sandbox_id)] = session
        return session

    def end(self, sandbox_id) -> EffectSession | None:
        with self._lock:
            return self._sessions.pop(str(sandbox_id), None)

    def session_for(self, sandbox_id) -> EffectSession | None:
        with self._lock:
            return self._sessions.get(str(sandbox_id))

    def sealed(self, sandbox_id) -> bool:
        with self._lock:
            session = self._sessions.get(str(sandbox_id))
            return bool(session and session.sealed)

    def drain(self, sandbox_id) -> list:
        """Take the queue (commit flushes it, abort discards it)."""
        with self._lock:
            session = self._sessions.get(str(sandbox_id))
            if session is None:
                return []
            queued, session.queue = session.queue, []
            session.queued_bytes = 0
            return queued

    def counters(self, sandbox_id) -> tuple[int, int, int, bool]:
        with self._lock:
            session = self._sessions.get(str(sandbox_id))
            if session is None:
                return (0, 0, 0, False)
            return (session.deferred, session.rejected, session.passed, session.sealed)

    def _decide_locked(self, session: EffectSession, *, host: str, method: str, path: str) -> str:
        if session.policy == "allow":
            return DECISION_PASS
        if session.policy == "reject":
            return DECISION_REJECT
        if session.policy == "seal":
            return DECISION_SEAL
        # defer: only allow-listed endpoints may be queued; anything else
        # follows on_unlisted (refuse by default, never silently queued).
        for rule in session.rules:
            if rule.matches(host=host, method=method, path=path):
                return DECISION_DEFER if rule.defer else DECISION_PASS
        return DECISION_PASS if session.on_unlisted == "allow" else DECISION_REJECT

    def decide_write(
        self, sandbox_id, *, host: str, method: str, path: str
    ) -> tuple[str, EffectSession] | None:
        """Decision for a plaintext mutating request, or None when no
        session is armed. Counters are updated here, under the lock."""
        with self._lock:
            session = self._sessions.get(str(sandbox_id))
            if session is None:
                return None
            decision = self._decide_locked(session, host=host, method=method, path=path)
            if decision == DECISION_PASS:
                session.passed += 1
            elif decision == DECISION_REJECT:
                session.rejected += 1
            elif decision == DECISION_SEAL:
                # The write goes out, but the txn can no longer be aborted.
                session.sealed = True
                session.passed += 1
            return decision, session

    def decide_opaque(self, sandbox_id) -> tuple[str, EffectSession] | None:
        """Encrypted/raw flows carry no method. ``opaque_effects`` decides;
        ``allow`` (the default) keeps HTTPS working at the cost of not
        being able to guarantee a write-free transaction."""
        with self._lock:
            session = self._sessions.get(str(sandbox_id))
            if session is None:
                return None
            behavior = session.opaque_effects
            if behavior == "reject":
                session.rejected += 1
                return DECISION_REJECT, session
            if behavior == "seal":
                session.sealed = True
                session.passed += 1
                return DECISION_SEAL, session
            session.passed += 1
            return DECISION_PASS, session

    def enqueue(self, sandbox_id, request: DeferredRequest) -> int:
        """Queue a deferred write; returns its 1-based position, 0 when no
        session is armed, or -1 when the queue is full (the caller then
        refuses — dropping the write silently would be worse)."""
        with self._lock:
            session = self._sessions.get(str(sandbox_id))
            if session is None:
                return 0
            size = len(request.body)
            if (
                len(session.queue) >= session.max_queue_entries
                or session.queued_bytes + size > session.max_queue_bytes
            ):
                session.rejected += 1
                return -1
            session.queue.append(request)
            session.queued_bytes += size
            session.deferred += 1
            return len(session.queue)

    def queue_size(self, sandbox_id) -> tuple[int, int]:
        """``(entries, bytes)`` currently queued."""
        with self._lock:
            session = self._sessions.get(str(sandbox_id))
            if session is None:
                return (0, 0)
            return (len(session.queue), session.queued_bytes)


def bare_host(host: str) -> str:
    """Strip a trailing ``:port`` from a Host header value.

    The queued host comes from the request's Host header, which carries
    the port for non-default ports. A flush connects with ``(host, port)``,
    so leaving the port in the hostname makes resolution fail every time —
    the third appearance of this trap after D2.1's request_key and the
    first cut of the flush, which is why IPv6 is handled here too:
    ``[::1]:8080`` → ``[::1]``, while a bare ``fe80::1`` (no brackets, so
    no port can be attached) is returned unchanged.
    """
    if host.startswith("["):
        closing = host.find("]")
        if closing == -1:
            return host
        rest = host[closing + 1 :]
        if rest.startswith(":") and rest[1:].isdigit():
            return host[: closing + 1]
        return host
    if host.count(":") != 1:
        # No colon, or several: a bare IPv6 literal cannot carry a port.
        return host
    name, _, port = host.rpartition(":")
    return name if port.isdigit() else host


def build_deferred_request(
    *,
    parsed_head,
    body: bytes,
    host: str,
    port: int,
    method: str,
    path: str,
    txn_id: str | None,
    enqueued_at: str | None,
) -> DeferredRequest:
    """Turn an already-parsed head plus its complete body into a queue
    entry, dropping credential headers."""
    return DeferredRequest(
        method=method.upper(),
        host=bare_host(host),
        port=int(port),
        path=path,
        headers=tuple(
            filter_headers(list(parsed_head.headers), REQUEST_HEADER_DENY_LIST)
        ),
        body=body,
        txn_id=txn_id,
        enqueued_at=enqueued_at,
    )


def read_remaining_body(sock, parsed_head, already: bytes, *, limit: int) -> tuple[bytes, bool]:
    """Pull the rest of a request body off the socket.

    Only the first peek reached the proxy, so a deferred write with a
    larger body would otherwise be queued truncated and flush a corrupt
    request at commit. Returns ``(body, complete)``; ``complete`` is
    False when the body is unframed, exceeds ``limit``, or the peer
    stopped early — the caller must then refuse rather than queue.
    """
    length = parsed_head.content_length()
    if length is None:
        # No Content-Length: chunked or unframed request bodies are not
        # reconstructable here, so they cannot be deferred.
        return already, not already
    if length > limit:
        return already, False
    body = bytearray(already[:length])
    while len(body) < length:
        try:
            chunk = sock.recv(min(65536, length - len(body)))
        except OSError:
            break
        if not chunk:
            break
        body += chunk
    return bytes(body), len(body) == length


def flush_deferred_request(request: DeferredRequest, *, timeout: float = 30.0) -> tuple:
    """Send one queued write from the host and report what happened.

    Deliberately **not** routed through the egress proxy: no REDIRECT
    applies to host-originated traffic, the flush is not re-classified or
    re-recorded, and it cannot be caught by the very gate that queued it.
    One attempt only — no retries, no idempotency keys (a non-goal); the
    caller journals the outcome either way.

    Returns ``(status, error)`` with exactly one of them set.
    """
    import http.client

    connection = None
    try:
        connection = http.client.HTTPConnection(
            bare_host(request.host), request.port, timeout=timeout
        )
        headers = {name: value for name, value in request.headers}
        # Framing is ours to state: the queued body is complete by
        # construction (unframed bodies are refused at enqueue time).
        headers.pop("Transfer-Encoding", None)
        headers["Content-Length"] = str(len(request.body))
        connection.request(request.method, request.path, body=request.body, headers=headers)
        response = connection.getresponse()
        status = int(response.status)
        response.read()
        return status, None
    except Exception as exc:  # network, DNS, protocol - all reported alike
        return None, f"{type(exc).__name__}: {exc}"
    finally:
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass


__all__ = [
    "DECISION_DEFER",
    "DECISION_PASS",
    "DECISION_REJECT",
    "DECISION_SEAL",
    "DEFAULT_MAX_QUEUE_BYTES",
    "DEFAULT_MAX_QUEUE_ENTRIES",
    "DEFERRED_RESPONSE",
    "EFFECT_POLICIES",
    "OPAQUE_BEHAVIORS",
    "REJECTED_RESPONSE",
    "UNLISTED_BEHAVIORS",
    "DeferredRequest",
    "EffectGate",
    "EffectRule",
    "EffectSession",
    "build_deferred_request",
    "bare_host",
    "flush_deferred_request",
    "read_remaining_body",
]
