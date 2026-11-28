"""Cassette store for egress record/replay (roadmap D2).

Recorded request/response pairs live outside the journal: the journal
keeps one line per flow (its index, with the request key and response
digest), and the bodies land here, content-addressed. Same split the
storage layer already uses for checkpoint artifacts, and the same reason
— JSONL rows must stay small.

Buckets are per sandbox. A replay may *read* another sandbox's bucket
only along a verified fork lineage (``cassette_source``); there is no
shared pool.
"""
from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

logger = logging.getLogger(__name__)

DEFAULT_MAX_BODY_BYTES = 1024 * 1024

# Names which response is current inside a request-key directory.
_LATEST_POINTER = "latest"

# Dropped before anything is written: a cassette must not become a
# convenient credential dump. Note the deliberate limit documented in
# the design: credentials passed in the query string still land in the
# stored path (same exposure the journal already has).
REQUEST_HEADER_DENY_LIST: frozenset[str] = frozenset(
    {"authorization", "cookie", "proxy-authorization", "x-api-key", "x-auth-token"}
)
RESPONSE_HEADER_DENY_LIST: frozenset[str] = frozenset({"set-cookie"})

# Headers that legitimately change a response, and therefore belong in
# the request key. Everything else is ignored so a User-Agent bump does
# not silently miss every cassette.
DEFAULT_VARYING_HEADERS: tuple[str, ...] = ("accept", "accept-encoding")

# Statuses worth replaying. 206 is absent on purpose: partial content
# depends on the Range header, which is not part of the key by default,
# so recording it would eventually serve the wrong byte range.
DEFAULT_RECORDABLE_STATUSES: frozenset[int] = frozenset({200, 203, 204})
PARTIAL_STATUS = 206
REDIRECT_STATUSES: frozenset[int] = frozenset({301, 302, 303, 307, 308})


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def filter_headers(
    headers: Sequence[tuple[str, str]], deny: Iterable[str]
) -> list[tuple[str, str]]:
    denied = {name.lower() for name in deny}
    return [(name, value) for name, value in headers if name.lower() not in denied]


def request_key(
    *,
    method: str,
    host: str,
    port: int,
    path: str,
    body_sha256: str,
    headers: Sequence[tuple[str, str]] = (),
    varying_headers: Sequence[str] = DEFAULT_VARYING_HEADERS,
) -> str:
    """Stable digest identifying a request for cassette lookup.

    The path keeps its query string (it selects the response), and only
    allow-listed headers participate — incidental header churn must not
    look like "replay does not work".
    """
    varying = [name.lower() for name in varying_headers]
    seen = {name.lower(): value.strip() for name, value in headers}
    # The Host header may or may not carry the port (clients omit the
    # default one), and the port is already a key component — normalise
    # so the same request cannot produce two keys.
    bare_host = host.lower().rsplit(":", 1)[0] if ":" in host else host.lower()
    parts = [
        method.upper(),
        bare_host,
        str(int(port)),
        path,
        body_sha256,
    ]
    parts.extend(f"{name}={seen.get(name, '')}" for name in sorted(varying))
    return sha256_hex("\n".join(parts).encode("utf-8"))


@dataclass(frozen=True)
class CassetteEntry:
    """One recorded exchange. ``truncated`` entries are stored for
    visibility but must never be replayed (a partial response is worse
    than a live one)."""

    request_key: str
    method: str
    host: str
    port: int
    path: str
    status: int
    reason: str = ""
    response_headers: tuple[tuple[str, str], ...] = ()
    body: bytes = b""
    body_sha256: str = ""
    request_body_sha256: str = ""
    bytes_in: int = 0
    truncated: bool = False
    recorded_at: str | None = None
    origin_sandbox_id: str | None = None
    origin_seq: int | None = None

    def to_json(self) -> dict[str, object]:
        return {
            "request_key": self.request_key,
            "method": self.method,
            "host": self.host,
            "port": self.port,
            "path": self.path,
            "status": self.status,
            "reason": self.reason,
            "response_headers": [list(pair) for pair in self.response_headers],
            "body_b64": base64.b64encode(self.body).decode("ascii"),
            "body_sha256": self.body_sha256 or sha256_hex(self.body),
            "request_body_sha256": self.request_body_sha256,
            "bytes_in": self.bytes_in,
            "truncated": self.truncated,
            "recorded_at": self.recorded_at,
            "origin_sandbox_id": self.origin_sandbox_id,
            "origin_seq": self.origin_seq,
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, object]) -> "CassetteEntry":
        body = base64.b64decode(str(payload.get("body_b64") or ""))
        origin_seq = payload.get("origin_seq")
        return cls(
            request_key=str(payload["request_key"]),
            method=str(payload.get("method", "GET")),
            host=str(payload.get("host", "")),
            port=int(payload.get("port", 0)),
            path=str(payload.get("path", "/")),
            status=int(payload.get("status", 200)),
            reason=str(payload.get("reason", "")),
            response_headers=tuple(
                (str(pair[0]), str(pair[1]))
                for pair in (payload.get("response_headers") or [])
                if len(pair) == 2
            ),
            body=body,
            body_sha256=str(payload.get("body_sha256") or sha256_hex(body)),
            request_body_sha256=str(payload.get("request_body_sha256", "")),
            bytes_in=int(payload.get("bytes_in", 0)),
            truncated=bool(payload.get("truncated", False)),
            recorded_at=(
                None if payload.get("recorded_at") is None else str(payload["recorded_at"])
            ),
            origin_sandbox_id=(
                None
                if payload.get("origin_sandbox_id") is None
                else str(payload["origin_sandbox_id"])
            ),
            origin_seq=None if origin_seq is None else int(origin_seq),
        )


@dataclass
class CassetteStore:
    """Content-addressed cassettes under ``{root}/{sandbox_id}/``.

    Layout: ``<sandbox_id>/<request_key>/<response_sha256>.json``, so a
    changed response adds a sibling instead of clobbering history and
    lookups stay a single directory listing.
    """

    root: Path
    _lock: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root)

    def bucket(self, sandbox_id: object) -> Path:
        return self.root / str(sandbox_id)

    def put(self, sandbox_id: object, entry: CassetteEntry) -> Path:
        """Write atomically (temp + rename, the storage layer's pattern)
        and return the entry's path. A ``latest`` pointer records which
        response is current — mtime cannot decide that (two writes in the
        same clock tick are indistinguishable)."""
        digest = entry.body_sha256 or sha256_hex(entry.body)
        directory = self.bucket(sandbox_id) / entry.request_key
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"{digest}.json"
        self._write_atomic(target, json.dumps(entry.to_json(), separators=(",", ":")))
        self._write_atomic(directory / _LATEST_POINTER, digest)
        return target

    def _write_atomic(self, target: Path, payload: str) -> None:
        handle, temp_path = tempfile.mkstemp(dir=str(target.parent), suffix=".tmp")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_path, target)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(temp_path)
            raise

    def get(self, sandbox_id: object, request_key: str) -> CassetteEntry | None:
        """The current entry for this key, or None. Unreadable files are
        skipped: a corrupt cassette must degrade to a cache miss, never
        break the flow."""
        directory = self.bucket(sandbox_id) / request_key
        if not directory.is_dir():
            return None
        candidates: list[Path] = []
        pointer = directory / _LATEST_POINTER
        if pointer.is_file():
            try:
                digest = pointer.read_text(encoding="utf-8").strip()
            except OSError:
                digest = ""
            if digest:
                candidates.append(directory / f"{digest}.json")
        # Fall back to whatever is on disk (pointer missing or stale).
        candidates.extend(
            sorted(
                directory.glob("*.json"),
                key=lambda path: path.stat().st_mtime_ns,
                reverse=True,
            )
        )
        for path in candidates:
            if not path.is_file():
                continue
            try:
                with path.open("r", encoding="utf-8") as stream:
                    return CassetteEntry.from_json(json.load(stream))
            except Exception:
                logger.debug("Skipping unreadable cassette %s", path, exc_info=True)
        return None

    def count(self, sandbox_id: object) -> int:
        bucket = self.bucket(sandbox_id)
        if not bucket.is_dir():
            return 0
        return sum(1 for _ in bucket.glob("*/*.json"))

    def prune(self, sandbox_id: object) -> None:
        """Drop a sandbox's bucket (called on destroy). Replay must
        therefore happen before the recording sandbox is killed."""
        bucket = self.bucket(sandbox_id)
        if bucket.exists():
            shutil.rmtree(bucket, ignore_errors=True)


__all__ = [
    "DEFAULT_MAX_BODY_BYTES",
    "DEFAULT_RECORDABLE_STATUSES",
    "DEFAULT_VARYING_HEADERS",
    "PARTIAL_STATUS",
    "REDIRECT_STATUSES",
    "REQUEST_HEADER_DENY_LIST",
    "RESPONSE_HEADER_DENY_LIST",
    "CassetteEntry",
    "CassetteStore",
    "filter_headers",
    "request_key",
    "sha256_hex",
]
