"""CloudClient — the SDK's TCP transport for talking to a crab-gateway.

`CloudClient` is the cloud twin of `crab.daemon.DaemonClient`: it speaks
the same three JSON verbs (`get_json` / `post_json` / `delete`, plus the
private `_request_json` seam the storage shim uses for DELETE-with-body)
so `crab.remote_engine.RemoteEngine` can consume either client without
change. The differences are confined to the wire:

- transport is plain HTTP(S) over TCP to a gateway, not a Unix socket;
- every request carries `Authorization: Bearer <api key>` and is rooted
  under the gateway's `/v1` prefix;
- gateway error statuses are surfaced as *typed* exceptions
  (`CloudAuthError`, `SandboxNotFound`, `QuotaExceeded`, `SandboxLost`,
  `DaemonUnreachableError`, `GatewayTimeoutError`).

Two deliberate design points:

1. **Every typed exception subclasses `DaemonRequestError`.** RemoteEngine
   rehydrates structured daemon errors (txn conflicts, merge conflicts)
   by catching `DaemonRequestError` and inspecting `.status_code`/`.body`.
   Because the gateway relays daemon errors verbatim, subclassing keeps
   that rehydration working over the cloud path with zero RemoteEngine
   changes.

2. **Host-shim guard.** The gateway does not expose the daemon's
   `/shutdown` route (an operator action, not a tenant API). All other
   daemon routes (per-sandbox lifecycle, upstream, network lease,
   inspector filters, process merge, runtime helpers) are now proxied
   through the gateway for full remote-access parity (S5 full-access
   unlock). See the design doc's §5.1 tension note.

Standard library only, matching the rest of the SDK.
"""
from __future__ import annotations

import http.client
import json
import os
from typing import Any, Mapping
from urllib.parse import urlsplit

from .daemon import DaemonRequestError
from .daemon.transport import StreamIterator

API_KEY_ENV = "CRAB_API_KEY"
"""Environment variable consulted when no explicit `api_key=` is given."""

_API_PREFIX = "/v1"
"""Gateway API version prefix. Frozen as of S2 — additions only."""


# ---------------------------------------------------------------------------
# Typed exceptions (all subclass DaemonRequestError — see module docstring).
# ---------------------------------------------------------------------------


class CloudRequestError(DaemonRequestError):
    """Gateway returned a non-2xx response that has no more specific type."""


class CloudAuthError(CloudRequestError):
    """401/403 — missing, malformed, unknown, or revoked API key."""


class SandboxNotFound(CloudRequestError):
    """404 — no such resource *within this tenant* (cross-tenant IDs
    deliberately look identical to nonexistent ones)."""


class QuotaExceeded(CloudRequestError):
    """409 with `error_type: quota_exceeded` — tenant sandbox cap reached.

    `.quota` carries the gateway's quota snapshot (limit / in_use) when
    the response body includes one.
    """

    quota: dict[str, Any] | None = None


class SandboxLost(CloudRequestError):
    """410 — the sandbox existed but its daemon-side state is gone
    (daemon restarted underneath the gateway registry)."""


class DaemonUnreachableError(CloudRequestError):
    """502 — the gateway could not reach its daemon socket."""


class GatewayTimeoutError(CloudRequestError):
    """504 — the *gateway's* daemon call timed out. A timeout of the
    client's own TCP request raises plain `TimeoutError` instead, same
    as `DaemonClient`."""


class CloudConnectionError(ConnectionError):
    """TCP-level failure reaching the gateway (refused, DNS, reset)."""


class CloudUnsupportedOperation(NotImplementedError):
    """The requested daemon route is not exposed by the gateway.

    Raised client-side, before any wire traffic. These are host-coupled
    helper routes that only make sense when the SDK shares a filesystem
    or network namespace with the daemon; in cloud mode they have no
    meaningful implementation (design doc §5.1 tension note, resolved
    in S2 as an SDK-side guard)."""


# ---------------------------------------------------------------------------
# Unexposed-route guard.
# ---------------------------------------------------------------------------


def _unexposed_route_reason(path: str) -> str | None:
    """Return a human explanation if `path` is a daemon route the gateway
    does not expose, else None.

    As of S5 full-access, only `/shutdown` is blocked (operator-only).
    All per-sandbox and runtime routes are now proxied through the gateway."""
    parts = tuple(p for p in path.split("/") if p)
    if parts == ("shutdown",):
        return "daemon shutdown is an operator action, not a tenant API"
    return None


# ---------------------------------------------------------------------------
# Error mapping.
# ---------------------------------------------------------------------------


def _error_from_response(status_code: int, path: str, body: bytes) -> CloudRequestError:
    """Map a gateway error response onto the most specific typed exception.

    The gateway's own errors carry `error_type` in a JSON body; daemon
    errors are relayed verbatim (arbitrary daemon JSON). Unknown shapes
    fall back to the base `CloudRequestError` — including 429, which has
    no v0 source but is reserved by the doc's error table.
    """
    error_type = None
    decoded: dict[str, Any] | None = None
    try:
        parsed = json.loads(body)
        if isinstance(parsed, dict):
            decoded = parsed
            error_type = parsed.get("error_type")
    except (json.JSONDecodeError, UnicodeDecodeError):
        pass
    if status_code in (401, 403):
        return CloudAuthError(status_code, path, body)
    if status_code == 404:
        return SandboxNotFound(status_code, path, body)
    if status_code == 409 and error_type == "quota_exceeded":
        exc = QuotaExceeded(status_code, path, body)
        if decoded is not None and isinstance(decoded.get("quota"), dict):
            exc.quota = decoded["quota"]
        return exc
    if status_code == 410:
        return SandboxLost(status_code, path, body)
    if status_code == 502:
        return DaemonUnreachableError(status_code, path, body)
    if status_code == 504:
        return GatewayTimeoutError(status_code, path, body)
    return CloudRequestError(status_code, path, body)


# ---------------------------------------------------------------------------
# The client.
# ---------------------------------------------------------------------------


class CloudClient:
    """Synchronous HTTP(S) client speaking to a crab-gateway over TCP.

    Verb-for-verb compatible with `DaemonClient` — `RemoteEngine` accepts
    either. One client per call site is fine; connections are per-request
    (no pooling in v0), mirroring `DaemonClient`.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        *,
        timeout_seconds: float = 30.0,
    ) -> None:
        parsed = urlsplit(base_url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(
                f"cloud base URL must be http:// or https://, got {base_url!r}"
            )
        if not parsed.hostname:
            raise ValueError(f"cloud base URL has no host: {base_url!r}")
        self._scheme = parsed.scheme
        self._host = parsed.hostname
        self._port = parsed.port or (443 if parsed.scheme == "https" else 80)
        # Keep any reverse-proxy path prefix (e.g. https://api.example.com/crab).
        self._base_path = parsed.path.rstrip("/")
        key = api_key if api_key is not None else os.environ.get(API_KEY_ENV)
        if not key:
            raise ValueError(
                "cloud mode requires an API key: pass api_key= or set "
                f"${API_KEY_ENV}"
            )
        self._api_key = key
        self._timeout = float(timeout_seconds)

    @property
    def base_url(self) -> str:
        return f"{self._scheme}://{self._host}:{self._port}{self._base_path}"

    def ping(self) -> bool:
        try:
            self.get_json("/healthz")
            return True
        except (DaemonRequestError, OSError):
            return False

    # -- verbs: signatures mirror DaemonClient exactly -----------------

    def get_json(self, path: str, *, timeout_seconds: float | None = None) -> dict[str, Any]:
        return self._request_json("GET", path, body=None, timeout_seconds=timeout_seconds)

    def post_json(
        self,
        path: str,
        payload: Mapping[str, Any] | None = None,
        *,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        body = b"" if payload is None else json.dumps(payload).encode("utf-8")
        return self._request_json("POST", path, body=body, timeout_seconds=timeout_seconds)

    def delete(self, path: str, *, timeout_seconds: float | None = None) -> dict[str, Any]:
        return self._request_json("DELETE", path, body=None, timeout_seconds=timeout_seconds)

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        body: bytes | None,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        reason = _unexposed_route_reason(path)
        if reason is not None:
            raise CloudUnsupportedOperation(
                f"{method} {path} is not available in cloud mode: {reason}"
            )
        effective = self._timeout if timeout_seconds is None else float(timeout_seconds)
        if self._scheme == "https":
            conn: http.client.HTTPConnection = http.client.HTTPSConnection(
                self._host, self._port, timeout=effective
            )
        else:
            conn = http.client.HTTPConnection(self._host, self._port, timeout=effective)
        headers = {"Authorization": f"Bearer {self._api_key}"}
        if body is not None and body != b"":
            headers["Content-Type"] = "application/json"
        target = f"{self._base_path}{_API_PREFIX}{path}"
        try:
            conn.request(method, target, body=body, headers=headers)
            response = conn.getresponse()
            try:
                payload_bytes = response.read()
                status_code = int(response.status)
            finally:
                response.close()
        except TimeoutError:
            # Client-side timeout: same presentation as DaemonClient
            # (socket.timeout is TimeoutError since 3.10).
            raise
        except OSError as exc:
            raise CloudConnectionError(
                f"crab gateway not reachable at {self.base_url} "
                f"({exc.__class__.__name__}: {exc})"
            ) from exc
        finally:
            conn.close()
        if status_code >= 400:
            raise _error_from_response(status_code, path, payload_bytes)
        if not payload_bytes:
            return {}
        try:
            decoded = json.loads(payload_bytes)
        except json.JSONDecodeError as exc:
            raise CloudRequestError(status_code, path, payload_bytes) from exc
        if not isinstance(decoded, dict):
            raise CloudRequestError(status_code, path, payload_bytes)
        return decoded

    def stream_post(
        self,
        path: str,
        payload: Mapping[str, Any] | None = None,
        *,
        timeout_seconds: float | None = None,
    ) -> StreamIterator:
        """POST that returns a streaming NDJSON iterator.

        The path is checked against the unexposed-route guard and then
        prefixed with the API version (same as `_request_json`). The
        caller must close the iterator when done."""
        reason = _unexposed_route_reason(path)
        if reason is not None:
            raise CloudUnsupportedOperation(
                f"POST {path} is not available in cloud mode: {reason}"
            )
        effective = self._timeout if timeout_seconds is None else float(timeout_seconds)
        body = b"" if payload is None else json.dumps(payload).encode("utf-8")
        if self._scheme == "https":
            conn: http.client.HTTPConnection = http.client.HTTPSConnection(
                self._host, self._port, timeout=effective
            )
        else:
            conn = http.client.HTTPConnection(self._host, self._port, timeout=effective)
        headers: dict[str, str] = {"Authorization": f"Bearer {self._api_key}"}
        if body:
            headers["Content-Type"] = "application/json"
        target = f"{self._base_path}{_API_PREFIX}{path}"
        try:
            conn.request("POST", target, body=body, headers=headers)
            response = conn.getresponse()
        except TimeoutError:
            conn.close()
            raise
        except OSError as exc:
            conn.close()
            raise CloudConnectionError(
                f"crab gateway not reachable at {self.base_url} "
                f"({exc.__class__.__name__}: {exc})"
            ) from exc
        except Exception:
            conn.close()
            raise
        if response.status >= 400:
            try:
                payload_bytes = response.read()
            finally:
                response.close()
                conn.close()
            raise _error_from_response(response.status, path, payload_bytes)
        return StreamIterator(response, conn)
