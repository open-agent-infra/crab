#!/usr/bin/env python3
"""Dependency-free DNS/HTTP(S) egress measurement for host and sandbox use."""

from __future__ import annotations

import argparse
import http.client
import json
import socket
import ssl
import time
from typing import Any
from urllib.parse import urljoin, urlsplit, urlunsplit


class _TimedHTTPConnection(http.client.HTTPConnection):
    connect_seconds: float
    remote_address: str | None

    def connect(self) -> None:
        started = time.perf_counter()
        super().connect()
        self.connect_seconds = time.perf_counter() - started
        peer = self.sock.getpeername() if self.sock is not None else None
        self.remote_address = None if peer is None else str(peer[0])


class _TimedHTTPSConnection(http.client.HTTPSConnection):
    connect_seconds: float
    remote_address: str | None

    def connect(self) -> None:
        started = time.perf_counter()
        super().connect()
        self.connect_seconds = time.perf_counter() - started
        peer = self.sock.getpeername() if self.sock is not None else None
        self.remote_address = None if peer is None else str(peer[0])


def redact_url(url: str) -> str:
    """Remove credentials and query values from a diagnostic URL.

    The probe still requests the original URL.  Only the incident-friendly
    JSON representation is sanitized so signed query strings, basic-auth
    userinfo, and fragments cannot be copied into logs accidentally.
    """

    try:
        parsed = urlsplit(url)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError:
        return "<redacted-url>"
    if not parsed.scheme or not hostname:
        return "<redacted-url>"
    host = hostname
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    if port is not None:
        host = f"{host}:{port}"
    if parsed.username is not None or parsed.password is not None:
        host = f"<redacted>@{host}"
    query = "<redacted>" if parsed.query else ""
    return urlunsplit((parsed.scheme, host, parsed.path, query, ""))


def probe(url: str, *, timeout: float, max_redirects: int = 5) -> dict[str, Any]:
    original_url = url
    redirects: list[dict[str, object]] = []
    total_started = time.perf_counter()
    for _ in range(max_redirects + 1):
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError(f"unsupported URL: {redact_url(url)!r}")
        port = parsed.port or (443 if parsed.scheme == "https" else 80)

        dns_started = time.perf_counter()
        answers = socket.getaddrinfo(
            parsed.hostname,
            port,
            type=socket.SOCK_STREAM,
        )
        dns_seconds = time.perf_counter() - dns_started
        dns_addresses = list(dict.fromkeys(str(item[4][0]) for item in answers))

        if parsed.scheme == "https":
            connection: _TimedHTTPConnection | _TimedHTTPSConnection = (
                _TimedHTTPSConnection(
                    parsed.hostname,
                    port,
                    timeout=timeout,
                    context=ssl.create_default_context(),
                )
            )
        else:
            connection = _TimedHTTPConnection(
                parsed.hostname,
                port,
                timeout=timeout,
            )
        target = parsed.path or "/"
        if parsed.query:
            target = f"{target}?{parsed.query}"
        request_started = time.perf_counter()
        try:
            connection.request(
                "GET",
                target,
                headers={
                    "Connection": "close",
                    "User-Agent": "crab-egress-diagnostic/1",
                },
            )
            response = connection.getresponse()
            status = int(response.status)
            location = response.getheader("Location")
            if status in {301, 302, 303, 307, 308} and location:
                response.read()
                redirects.append(
                    {
                        "status": status,
                        "from": redact_url(url),
                        "to": redact_url(urljoin(url, location)),
                    }
                )
                url = urljoin(url, location)
                continue

            first_byte_seconds = time.perf_counter() - request_started
            download_started = time.perf_counter()
            byte_count = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                byte_count += len(chunk)
            download_seconds = time.perf_counter() - download_started
            total_seconds = time.perf_counter() - total_started
            return {
                "requested_url": redact_url(original_url),
                "final_url": redact_url(url),
                "status": status,
                "dns_answers": dns_addresses,
                "dns_seconds": dns_seconds,
                "remote_address": connection.remote_address,
                "connect_seconds": connection.connect_seconds,
                "first_byte_seconds": first_byte_seconds,
                "download_seconds": download_seconds,
                "total_seconds": total_seconds,
                "bytes": byte_count,
                "bytes_per_second": (
                    byte_count / download_seconds if download_seconds > 0 else None
                ),
                "redirects": redirects,
            }
        finally:
            connection.close()
    raise RuntimeError(
        f"too many redirects while fetching {redact_url(original_url)!r}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    started_at = time.time()
    try:
        result = probe(args.url, timeout=args.timeout)
        payload = {"ok": True, "started_at_unix": started_at, **result}
    except Exception as exc:
        payload = {
            "ok": False,
            "started_at_unix": started_at,
            "requested_url": redact_url(args.url),
            "error_type": type(exc).__name__,
            "error": str(exc).replace(args.url, redact_url(args.url)),
        }
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
