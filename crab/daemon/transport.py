"""HTTP-over-Unix-socket transport shared by the daemon and its clients.

Both the daemon's HTTP server and the SDK/CLI client speak JSON over a
Unix domain socket — the same model dockerd uses (`unix:///var/run/docker.sock`).
File-permission gating on the socket (0600 by default) is the only auth
in v1; richer schemes are deferred.

The two pieces live together because the request format is symmetric and
small: a single `DaemonClient` covers both the CLI and the SDK proxy
(`crab.remote_engine.RemoteEngine`), and a single `serve_unix_socket`
helper covers the daemon's bind/listen path. Putting them in one module
keeps the wire contract obvious — both ends import the same
`DEFAULT_SOCKET_NAME` and `default_socket_path()`.
"""
from __future__ import annotations

import http.client
import json
import logging
import os
import socket
import socketserver
import stat
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)

DEFAULT_SOCKET_NAME = "crab.sock"
"""Filename component placed under the runtime directory by default."""

DEFAULT_SOCKET_PERMS = 0o600
"""Permissions applied to the daemon socket. Matches dockerd's default:
the local user is sole authority."""


def default_socket_path() -> Path:
    """Return the conventional daemon socket path for this user.

    Honors `$CRAB_DAEMON_SOCKET` first, then falls back to
    `$XDG_RUNTIME_DIR/crab/crab.sock` for non-root users and
    `/run/crab/crab.sock` for root. Mirroring the dockerd convention
    keeps the discovery rule predictable for both the CLI and the SDK
    proxy."""
    explicit = os.environ.get("CRAB_DAEMON_SOCKET")
    if explicit:
        return Path(explicit).expanduser().resolve()
    if os.geteuid() == 0:
        base = Path("/run/crab")
    else:
        xdg = os.environ.get("XDG_RUNTIME_DIR")
        base = Path(xdg).expanduser() if xdg else Path.home() / ".cache" / "crab"
        base = base / "crab"
    return (base / DEFAULT_SOCKET_NAME).resolve()


class DaemonRequestError(RuntimeError):
    """Daemon returned a non-2xx response."""

    def __init__(self, status_code: int, path: str, body: bytes) -> None:
        self.status_code = int(status_code)
        self.path = path
        self.body = body
        try:
            decoded = body.decode("utf-8", errors="replace")
        except Exception:  # pragma: no cover
            decoded = repr(body)
        super().__init__(
            f"crab daemon request failed: status={self.status_code} path={self.path} body={decoded}"
        )


# ---------------------------------------------------------------------------
# Server side: a ThreadingMixIn HTTPServer bound to an AF_UNIX socket.
# ---------------------------------------------------------------------------


class _UnixHTTPServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    """`UnixStreamServer` + `ThreadingMixIn` + a couple of HTTPServer hooks.

    `http.server.HTTPServer` derives from `TCPServer`, so it can't directly
    serve over a Unix domain socket. We swap the base to `UnixStreamServer`
    and add the two attributes `BaseHTTPRequestHandler` needs to format
    its response status line (`server_name`, `server_port`)."""

    allow_reuse_address = True
    daemon_threads = True

    def server_bind(self) -> None:
        # Strip any stale socket file before binding so daemon restarts
        # (after a hard crash, etc.) don't fail with EADDRINUSE.
        path = Path(self.server_address)
        if path.exists():
            try:
                if stat.S_ISSOCK(path.stat().st_mode):
                    path.unlink()
            except OSError:
                pass
        path.parent.mkdir(parents=True, exist_ok=True)
        super().server_bind()
        os.chmod(self.server_address, DEFAULT_SOCKET_PERMS)
        self.server_name = "crab"
        self.server_port = 0


def serve_unix_socket(
    socket_path: Path,
    handler_factory,
) -> _UnixHTTPServer:
    """Bind a Unix-socket HTTP server at `socket_path` using `handler_factory`.

    Returns the server; the caller is responsible for `serve_forever` and
    shutdown. The socket file is created with permissions 0600."""
    server = _UnixHTTPServer(str(socket_path), handler_factory)
    return server


# ---------------------------------------------------------------------------
# Client side: HTTPConnection variant that connects to a Unix socket.
# ---------------------------------------------------------------------------


class _UnixHTTPConnection(http.client.HTTPConnection):
    """`http.client.HTTPConnection` that talks to a Unix socket.

    `http.client` already knows how to drive HTTP over any duplex stream;
    we just have to give it a connected `AF_UNIX` socket and a stable
    host string for the `Host:` header."""

    def __init__(self, socket_path: str, *, timeout: float) -> None:
        # The host string only matters for the HTTP `Host:` header — the
        # actual connection uses the Unix socket below.
        super().__init__("crab", timeout=timeout)
        self._socket_path = socket_path

    def connect(self) -> None:  # type: ignore[override]
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        sock.connect(self._socket_path)
        self.sock = sock


class DaemonClient:
    """Synchronous HTTP client speaking to the daemon over a Unix socket.

    One client per call site is fine — the underlying `HTTPConnection`
    handles keep-alive, and the daemon's handler is thread-safe. We do
    not pool connections in v1; the daemon will run in a single process
    per host so contention isn't a concern."""

    def __init__(
        self,
        socket_path: str | os.PathLike[str] | None = None,
        *,
        timeout_seconds: float = 30.0,
    ) -> None:
        self._socket_path = str(
            Path(socket_path).expanduser().resolve()
            if socket_path is not None
            else default_socket_path()
        )
        self._timeout = float(timeout_seconds)

    @property
    def socket_path(self) -> str:
        return self._socket_path

    def ping(self) -> bool:
        try:
            self.get_json("/healthz")
            return True
        except (DaemonRequestError, OSError, FileNotFoundError):
            return False

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
        # Per-call timeout overrides the client default. Use this when the
        # daemon's handler itself can legitimately block for a long time
        # (e.g. `runtime.exec` with a multi-minute task timeout).
        effective = self._timeout if timeout_seconds is None else float(timeout_seconds)
        conn = _UnixHTTPConnection(self._socket_path, timeout=effective)
        headers = {"Host": "crab"}
        if body is not None and body != b"":
            headers["Content-Type"] = "application/json"
        try:
            conn.request(method, path, body=body, headers=headers)
            response = conn.getresponse()
            try:
                payload_bytes = response.read()
                status_code = int(response.status)
            finally:
                response.close()
        except (FileNotFoundError, ConnectionRefusedError) as exc:
            # The socket is missing (daemon not running) or present but
            # not yet/no-longer listening (mid-startup or just stopped).
            # Both shapes are user-facing "daemon isn't reachable" — the
            # CLI keys off `FileNotFoundError` to pretty-print this, so
            # collapse both onto the same exception type.
            raise FileNotFoundError(
                f"crab daemon not reachable at {self._socket_path} ({exc.__class__.__name__}); "
                "start the daemon with `crab daemon start` (or "
                "`python -m crab.daemon`)."
            ) from exc
        finally:
            conn.close()
        if status_code >= 400:
            raise DaemonRequestError(status_code, path, payload_bytes)
        if not payload_bytes:
            return {}
        try:
            decoded = json.loads(payload_bytes)
        except json.JSONDecodeError as exc:
            raise DaemonRequestError(status_code, path, payload_bytes) from exc
        if not isinstance(decoded, dict):
            raise DaemonRequestError(status_code, path, payload_bytes)
        return decoded
