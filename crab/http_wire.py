"""Minimal HTTP/1.x wire parsing for egress recording (roadmap D2).

Just enough to record and replay a single plaintext exchange: parse a
request head, parse a response (Content-Length, chunked, or
read-until-close), and serialise a response back onto the wire. The
proxy stays a byte pump — this module only *reads* what it has already
copied, so a parse failure degrades to "not recorded", never to a
broken flow.

Deliberately not a general HTTP implementation: no keep-alive
multiplexing (one exchange per connection is recorded), no compression
handling (bodies are stored exactly as they arrived, with their
Content-Encoding header intact), no trailers.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_HEADER_TERMINATOR = b"\r\n\r\n"
MAX_HEAD_BYTES = 64 * 1024


@dataclass(frozen=True)
class HttpHead:
    """A parsed request or response head plus the bytes that followed."""

    start_line: str
    headers: tuple[tuple[str, str], ...]
    rest: bytes = b""

    def get(self, name: str, default: str = "") -> str:
        lowered = name.lower()
        for key, value in self.headers:
            if key.lower() == lowered:
                return value
        return default

    def content_length(self) -> int | None:
        raw = self.get("content-length")
        if not raw:
            return None
        try:
            return int(raw.strip())
        except ValueError:
            return None

    def is_chunked(self) -> bool:
        return "chunked" in self.get("transfer-encoding").lower()


def parse_head(data: bytes) -> HttpHead | None:
    """Split ``data`` into start line, headers and remainder, or None
    when the head is absent/malformed/oversized."""
    if len(data) > MAX_HEAD_BYTES:
        data = data[:MAX_HEAD_BYTES]
    terminator = data.find(_HEADER_TERMINATOR)
    if terminator < 0:
        return None
    block = data[:terminator].decode("latin-1")
    rest = data[terminator + len(_HEADER_TERMINATOR):]
    lines = block.split("\r\n")
    if not lines or not lines[0]:
        return None
    headers: list[tuple[str, str]] = []
    for line in lines[1:]:
        name, separator, value = line.partition(":")
        if not separator:
            return None  # continuation lines / garbage: refuse to guess
        headers.append((name.strip(), value.strip()))
    return HttpHead(start_line=lines[0], headers=tuple(headers), rest=rest)


def parse_status_line(start_line: str) -> tuple[int, str] | None:
    """``("HTTP/1.1 200 OK")`` -> ``(200, "OK")``."""
    parts = start_line.split(" ", 2)
    if len(parts) < 2 or not parts[0].upper().startswith("HTTP/"):
        return None
    try:
        status = int(parts[1])
    except ValueError:
        return None
    return status, parts[2] if len(parts) > 2 else ""


def dechunk(data: bytes) -> bytes | None:
    """Decode a chunked body, or None when the stream is incomplete or
    malformed (recording then simply skips the flow)."""
    out = bytearray()
    position = 0
    while True:
        line_end = data.find(b"\r\n", position)
        if line_end < 0:
            return None
        size_field = data[position:line_end].split(b";", 1)[0].strip()
        try:
            size = int(size_field, 16)
        except ValueError:
            return None
        position = line_end + 2
        if size == 0:
            return bytes(out)
        if position + size > len(data):
            return None
        out += data[position:position + size]
        position += size + 2  # skip the chunk's trailing CRLF


@dataclass
class ResponseAssembler:
    """Accumulates response bytes and reports when the body is complete.

    ``limit`` caps how much is buffered: past it the exchange is marked
    truncated (and therefore unreplayable) while the proxy keeps
    streaming the real bytes to the client untouched.
    """

    limit: int
    buffer: bytearray = field(default_factory=bytearray)
    truncated: bool = False
    _total: int = 0

    def feed(self, chunk: bytes) -> None:
        self._total += len(chunk)
        if len(self.buffer) >= self.limit:
            self.truncated = True
            return
        room = self.limit - len(self.buffer)
        if len(chunk) > room:
            self.buffer += chunk[:room]
            self.truncated = True
        else:
            self.buffer += chunk

    @property
    def total_bytes(self) -> int:
        return self._total

    def result(self) -> tuple[HttpHead, int, str, bytes] | None:
        """``(head, status, reason, body)`` for a complete, untruncated
        response; None when it cannot be recorded."""
        if self.truncated:
            return None
        head = parse_head(bytes(self.buffer))
        if head is None:
            return None
        status_line = parse_status_line(head.start_line)
        if status_line is None:
            return None
        status, reason = status_line
        if head.is_chunked():
            body = dechunk(head.rest)
            if body is None:
                return None
        else:
            length = head.content_length()
            if length is None:
                # No framing: the body is whatever arrived before close.
                body = head.rest
            elif len(head.rest) < length:
                return None  # incomplete
            else:
                body = head.rest[:length]
        return head, status, reason, body


def serialize_response(
    *,
    status: int,
    reason: str,
    headers: list[tuple[str, str]],
    body: bytes,
) -> bytes:
    """Render a response for replay. Framing is rewritten to a plain
    Content-Length so a recorded chunked response replays cleanly, and
    hop-by-hop framing headers are dropped."""
    dropped = {"transfer-encoding", "content-length", "connection", "keep-alive"}
    lines = [f"HTTP/1.1 {int(status)} {reason or ''}".rstrip()]
    for name, value in headers:
        if name.lower() in dropped:
            continue
        lines.append(f"{name}: {value}")
    lines.append(f"Content-Length: {len(body)}")
    lines.append("Connection: close")
    head = ("\r\n".join(lines) + "\r\n\r\n").encode("latin-1")
    return head + body


__all__ = [
    "MAX_HEAD_BYTES",
    "HttpHead",
    "ResponseAssembler",
    "dechunk",
    "parse_head",
    "parse_status_line",
    "serialize_response",
]
