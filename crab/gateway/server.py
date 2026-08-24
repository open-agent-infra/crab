"""crab-gateway — the service layer in front of a local crab daemon (S1).

The daemon stays the single-user root runtime; this process is the only
component meant to face a network. It authenticates bearer API keys,
resolves tenancy/ownership against its SQLite registry
(`crab.gateway.registry`), enforces per-tenant quotas, and translates
each authorized `/v1/...` request into a `DaemonClient` call — a 1:1
JSON facade over the daemon dispatch table, minus host-internal routes
(`/shutdown`, `/runtime/write_bundle_spec`, upstream/network-lease/
inspector endpoints, and — deferred in v0 — `processes/merge`).

Running model:
  - Threaded HTTP server on loopback plaintext HTTP by default; TLS
    termination is a reverse proxy's job in v0. Internet exposure
    without a proxy is unsupported.
  - Per-call daemon timeouts: slow routes (checkpoint, fork, txn,
    merge, exec, create, kill) get a long budget so a CRIU dump cannot
    be killed mid-flight; fast routes fail fast.
  - Admin plane (tenants/keys/quotas) is local-only, served over the
    gateway's own Unix socket — never over the TCP listener.
  - Startup reconciliation: the daemon's boot identity (`/info` pid —
    the daemon exposes no dedicated boot id in v0) is compared against
    the last seen value; a mismatch marks every active row `lost`
    (410 SandboxLost, the pre-S5 honesty route). Pending rows are
    matched against daemon sandboxes by the `gateway_intent_id`
    metadata the gateway injects at create; unmatched pending rows are
    reaped. There is no periodic loop in v0 (design doc §10).

Auth semantics: missing/unknown/revoked key -> 401; authenticated but
not the owner -> 404, never 403 (don't leak existence). `/healthz` is
open.
"""
from __future__ import annotations

import json
import logging
import os
import signal
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

from ..daemon.transport import (
    DaemonClient,
    DaemonRequestError,
    serve_unix_socket,
)
from ..resources import validate_claim
from .ports import PortManager
from .registry import GatewayRegistry, QuotaExceeded

logger = logging.getLogger(__name__)

API_PREFIX = "/v1"
"""Public API version prefix. `/v1` freezes when S2 ships; breaking
changes open `/v2` (design doc §5.1)."""

GATEWAY_INTENT_METADATA_KEY = "gateway_intent_id"
"""Metadata key the gateway injects into daemon create bodies so the
startup reconciliation pass can match pending registry rows to
daemon-side sandboxes (the daemon assigns sandbox ids, so the id itself
cannot be recorded before the create call)."""

DEFAULT_BIND_HOST = "127.0.0.1"
DEFAULT_BIND_PORT = 8700

ADMIN_SOCKET_NAME = "crab-gateway.sock"

_FAST_TIMEOUT_S = 30.0
"""Budget for control-plane reads and quick lifecycle calls."""

_SLOW_TIMEOUT_S = 600.0
"""Budget for daemon calls that legitimately block: CRIU dump/restore
(checkpoint, fork, txn begin/commit/abort), merges, exec with long task
timeouts, create (first image fetch), kill (chain materialization)."""


def default_data_dir() -> Path:
    """Registry/state directory: `$CRAB_GATEWAY_DATA_DIR` first, then
    `/var/lib/crab/gateway` for root and the XDG data dir otherwise."""
    explicit = os.environ.get("CRAB_GATEWAY_DATA_DIR")
    if explicit:
        return Path(explicit).expanduser().resolve()
    if os.geteuid() == 0:
        return Path("/var/lib/crab/gateway")
    xdg = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".local" / "share"
    return (base / "crab" / "gateway").resolve()


def default_admin_socket_path() -> Path:
    """Admin-plane Unix socket path; mirrors the daemon's
    `default_socket_path()` convention with the gateway's own name."""
    explicit = os.environ.get("CRAB_GATEWAY_SOCKET")
    if explicit:
        return Path(explicit).expanduser().resolve()
    if os.geteuid() == 0:
        base = Path("/run/crab")
    else:
        xdg = os.environ.get("XDG_RUNTIME_DIR")
        base = Path(xdg).expanduser() if xdg else Path.home() / ".cache" / "crab"
        base = base / "crab"
    return (base / ADMIN_SOCKET_NAME).resolve()


# ---------------------------------------------------------------------------
# Error taxonomy — one exception per HTTP outcome, mirroring the daemon's
# handler style.
# ---------------------------------------------------------------------------


class _BadRequest(Exception):
    pass


class _Unauthorized(Exception):
    pass


class _NotFound(Exception):
    pass


class _SandboxLost(Exception):
    """The row is `lost`: the daemon restarted (pre-S5) and the sandbox
    is gone. 410, `error_type: sandbox_lost` — never a phantom 200."""


class _DaemonUnreachable(Exception):
    """The daemon socket is missing or refusing — 502 per request."""


class _GatewayTimeout(Exception):
    """A daemon call outlived its per-route budget — 504. The daemon may
    still complete the operation; the client must reconcile via reads."""


class _DaemonError(Exception):
    """The daemon answered >= 400. Relayed verbatim (status + body) so
    the facade stays a provable passthrough — error shapes reaching the
    SDK are the daemon's own."""

    def __init__(self, cause: DaemonRequestError) -> None:
        super().__init__(str(cause))
        self.status_code = cause.status_code
        self.body = cause.body


# ---------------------------------------------------------------------------
# Tenant-facing routes.
# ---------------------------------------------------------------------------


class _GatewayRoutes:
    """Handlers for the `/v1` facade. Uniform signature
    `fn(tenant_id, body, *, path, **variables)`; open routes receive
    `tenant_id=None`."""

    def __init__(self, gateway: "GatewayServer") -> None:
        self._gateway = gateway

    # ----- open -----------------------------------------------------------

    def healthz(self, tenant_id: str | None, body: dict[str, Any], **_: Any) -> dict[str, Any]:
        return self._gateway.proxy("GET", "/healthz", None, _FAST_TIMEOUT_S)

    # ----- control plane ----------------------------------------------------

    def info(self, tenant_id: str | None, body: dict[str, Any], **_: Any) -> dict[str, Any]:
        assert tenant_id is not None
        raw = self._gateway.proxy("GET", "/info", None, _FAST_TIMEOUT_S)
        # Redaction is a whitelist: host paths, pids, bridge IPs and
        # internal service URLs never cross the tenant boundary.
        return {
            "ok": bool(raw.get("ok", True)),
            "version": raw.get("version"),
            "runtime": raw.get("runtime"),
            "default_image": raw.get("default_image"),
            "sandbox_count": self._gateway.registry.live_count(tenant_id),
        }

    def list_sandboxes(
        self, tenant_id: str | None, body: dict[str, Any], **_: Any
    ) -> dict[str, Any]:
        assert tenant_id is not None
        owned = {
            row["sandbox_id"]
            for row in self._gateway.registry.list_sandboxes(tenant_id, statuses=("active",))
        }
        raw = self._gateway.proxy("GET", "/sandboxes", None, _FAST_TIMEOUT_S)
        rows = [row for row in raw.get("sandboxes") or [] if row.get("sandbox_id") in owned]
        return {"ok": True, "sandboxes": rows}

    # ----- create / kill / fork — the routes that touch ownership ----------

    def launch_sandbox(
        self, tenant_id: str | None, body: dict[str, Any], **_: Any
    ) -> dict[str, Any]:
        assert tenant_id is not None
        name_raw = body.get("name")
        name = name_raw if isinstance(name_raw, str) else None
        metadata = dict(body.get("metadata") or {})
        # S3: the SDK ships the normalized resources claim in the launch
        # metadata; re-validate at the trust boundary (malformed -> 400).
        try:
            claim = validate_claim(metadata.get("resources"))
        except ValueError as exc:
            raise _BadRequest(f"invalid resources: {exc}") from exc
        # Phase one: durable intent + quota gate (409 on exhaustion),
        # including the per-tenant aggregate resource caps.
        intent_id = self._gateway.registry.begin_create(tenant_id, name=name, resources=claim)
        daemon_body = dict(body)
        metadata[GATEWAY_INTENT_METADATA_KEY] = intent_id
        daemon_body["metadata"] = metadata
        try:
            result = self._gateway.proxy("POST", "/sandboxes", daemon_body, _SLOW_TIMEOUT_S)
        except (_DaemonError, _DaemonUnreachable):
            # The daemon answered with an error or was never reached —
            # no sandbox exists, so the intent row can be reaped now.
            self._gateway.registry.abort_create(intent_id)
            raise
        except _GatewayTimeout:
            # The create may still land daemon-side. Keep the pending row;
            # the next startup reconciliation pass resolves it either way.
            raise
        sandbox_id = str(result.get("sandbox_id") or "")
        if not sandbox_id:
            self._gateway.registry.abort_create(intent_id)
            raise _BadRequest("daemon create returned no sandbox_id")
        # Phase two: flip pending -> active under the daemon-assigned id.
        self._gateway.registry.complete_create(intent_id, sandbox_id)
        return result

    def kill_sandbox(
        self, tenant_id: str | None, body: dict[str, Any], *, sandbox_id: str, **_: Any
    ) -> dict[str, Any]:
        assert tenant_id is not None
        self._gateway.require_owned(tenant_id, sandbox_id)
        # S4: release port allocations before killing the sandbox
        host_ports = self._gateway.registry.release_all_ports(sandbox_id)
        if host_ports:
            self._gateway.port_manager.release_all(sandbox_id, host_ports)
        result = self._gateway.proxy(
            "DELETE", f"/sandboxes/{sandbox_id}", None, _SLOW_TIMEOUT_S
        )
        self._gateway.registry.set_status(sandbox_id, "killed")
        return result

    def fork_sandbox(
        self, tenant_id: str | None, body: dict[str, Any], *, sandbox_id: str, **_: Any
    ) -> dict[str, Any]:
        assert tenant_id is not None
        source_row = self._gateway.require_owned(tenant_id, sandbox_id)
        try:
            count = int(body.get("count", 1))
        except (TypeError, ValueError):
            count = 1  # malformed counts are the daemon's 400 to give
        # Forks create sandboxes, so they hit the same quota gate as
        # create. Children inherit the source's resource claim (§4 S3 —
        # forks copy the source's limits), so each child counts it against
        # the aggregate caps. Children are registered post-hoc (their ids
        # are daemon-assigned), not two-phase — see the design doc.
        claim = dict(source_row.get("resources") or {})
        self._gateway.registry.ensure_capacity(
            tenant_id, additional=max(count, 0), resources=claim
        )
        result = self._gateway.proxy(
            "POST", f"/sandboxes/{sandbox_id}/fork", body, _SLOW_TIMEOUT_S
        )
        for fork in result.get("forks") or []:
            fork_id = fork.get("sandbox_id")
            if fork_id:
                self._gateway.registry.register_sandbox(
                    tenant_id, str(fork_id), resources=claim
                )
        return result

    # ----- port exposure (S4) -----------------------------------------------

    _MAX_PORTS_PER_TENANT = 10

    def expose_port(
        self, tenant_id: str | None, body: dict[str, Any], *, sandbox_id: str, **_: Any
    ) -> dict[str, Any]:
        assert tenant_id is not None
        self._gateway.require_owned(tenant_id, sandbox_id)
        guest_port = body.get("port")
        if guest_port is None:
            raise _BadRequest("missing 'port' in body")
        try:
            guest_port = int(guest_port)
        except (TypeError, ValueError):
            raise _BadRequest("'port' must be an integer")
        if not (1 <= guest_port <= 65535):
            raise _BadRequest("'port' must be between 1 and 65535")
        # Quota check
        count = self._gateway.registry.count_tenant_ports(tenant_id)
        if count >= self._MAX_PORTS_PER_TENANT:
            raise QuotaExceeded(
                f"port quota exceeded for tenant {tenant_id}",
                {"max_ports": self._MAX_PORTS_PER_TENANT, "current_ports": count},
            )
        # Get guest IP from daemon
        raw = self._gateway.proxy(
            "GET", f"/sandboxes/{sandbox_id}", None, _FAST_TIMEOUT_S
        )
        metadata = raw.get("metadata") or {}
        guest_ip = metadata.get("guest_ip") or "127.0.0.1"
        # Allocate host port and start forwarder
        host_port = self._gateway.port_manager.allocate(
            sandbox_id, guest_ip, guest_port
        )
        self._gateway.registry.allocate_port(
            sandbox_id, tenant_id, guest_port, host_port, guest_ip=guest_ip
        )
        host = self._gateway._host
        return {
            "ok": True,
            "host_port": host_port,
            "guest_port": guest_port,
            "url": f"tcp://{host}:{host_port}",
        }

    def list_ports(
        self, tenant_id: str | None, body: dict[str, Any], *, sandbox_id: str, **_: Any
    ) -> dict[str, Any]:
        assert tenant_id is not None
        self._gateway.require_owned(tenant_id, sandbox_id)
        ports = self._gateway.registry.list_ports(sandbox_id)
        return {"ok": True, "ports": ports}

    def release_port(
        self, tenant_id: str | None, body: dict[str, Any], *, sandbox_id: str, guest_port: str, **_: Any
    ) -> dict[str, Any]:
        assert tenant_id is not None
        self._gateway.require_owned(tenant_id, sandbox_id)
        try:
            gp = int(guest_port)
        except (TypeError, ValueError):
            raise _BadRequest("invalid guest_port")
        host_port = self._gateway.registry.release_port(sandbox_id, gp)
        if host_port is None:
            raise _NotFound(f"no allocation for port {gp}")
        self._gateway.port_manager.release(host_port)
        return {"ok": True, "released_host_port": host_port}


def _make_passthrough(gateway: "GatewayServer", method: str, timeout: float):
    """Ownership-checked 1:1 proxy: the daemon path is the request path
    minus the `/v1` prefix, the body rides verbatim."""

    def handler(
        tenant_id: str | None, body: dict[str, Any], *, path: str, sandbox_id: str, **_: Any
    ) -> dict[str, Any]:
        assert tenant_id is not None
        gateway.require_owned(tenant_id, sandbox_id)
        # POST and DELETE both forward their body verbatim (the daemon's
        # checkpoint DELETE accepts a `{"cascade": true}` body).
        payload = body if method in ("POST", "DELETE") else None
        return gateway.proxy(method, path[len(API_PREFIX):], payload, timeout)

    return handler


def _make_global_passthrough(gateway: "GatewayServer", method: str, timeout: float):
    """Non-sandbox passthrough: proxy to daemon with no ownership check."""

    def handler(
        tenant_id: str | None, body: dict[str, Any], *, path: str, **_: Any
    ) -> dict[str, Any]:
        payload = body if method in ("POST", "DELETE") else None
        return gateway.proxy(method, path[len(API_PREFIX):], payload, timeout)

    return handler


# Per-sandbox passthrough routes: (method, subpath, per-call timeout).
# All per-sandbox daemon routes are proxied (S5 full-access unlock).
_PASSTHROUGH_SANDBOX_ROUTES: list[tuple[str, str, float]] = [
    ("GET", "", _FAST_TIMEOUT_S),  # describe
    ("GET", "/inspector", _FAST_TIMEOUT_S),  # read-only inspector peek
    ("POST", "/exec", _SLOW_TIMEOUT_S),
    ("POST", "/stop", _FAST_TIMEOUT_S),
    ("POST", "/pause", _FAST_TIMEOUT_S),
    ("POST", "/resume", _FAST_TIMEOUT_S),
    ("GET", "/checkpoints", _FAST_TIMEOUT_S),
    ("POST", "/checkpoints", _SLOW_TIMEOUT_S),
    ("DELETE", "/checkpoints/{checkpoint_id}", _FAST_TIMEOUT_S),
    ("POST", "/checkpoints/{checkpoint_id}/restore", _SLOW_TIMEOUT_S),
    ("GET", "/txn", _FAST_TIMEOUT_S),
    ("POST", "/txn", _SLOW_TIMEOUT_S),
    ("POST", "/txn/{txn_id}/commit", _SLOW_TIMEOUT_S),
    ("POST", "/txn/{txn_id}/abort", _SLOW_TIMEOUT_S),
    ("POST", "/merge", _SLOW_TIMEOUT_S),
    ("POST", "/changeset", _SLOW_TIMEOUT_S),
    ("POST", "/observations/consolidate", _SLOW_TIMEOUT_S),
    ("POST", "/actions", _FAST_TIMEOUT_S),
    ("POST", "/egress", _FAST_TIMEOUT_S),
    ("POST", "/egress/replay", _SLOW_TIMEOUT_S),
    # S5 full-access: previously host-coupled routes now proxied
    ("POST", "/upstream", _FAST_TIMEOUT_S),
    ("DELETE", "/upstream", _FAST_TIMEOUT_S),
    ("POST", "/network/lease", _FAST_TIMEOUT_S),
    ("DELETE", "/network/lease", _FAST_TIMEOUT_S),
    ("POST", "/host_inspector/filters", _FAST_TIMEOUT_S),
    ("POST", "/processes/merge", _SLOW_TIMEOUT_S),
]


def _try_match(pattern: str, path: str) -> dict[str, str] | None:
    pat_parts = [p for p in pattern.strip("/").split("/") if p]
    path_parts = [p for p in path.strip("/").split("/") if p]
    if len(pat_parts) != len(path_parts):
        return None
    variables: dict[str, str] = {}
    for pat, real in zip(pat_parts, path_parts, strict=True):
        if pat.startswith("{") and pat.endswith("}"):
            variables[pat[1:-1]] = real
            continue
        if pat != real:
            return None
    return variables


class _JsonHandler(BaseHTTPRequestHandler):
    """Shared JSON plumbing for the public and admin handlers — the same
    read-body/send-json shapes the daemon handler uses."""

    server_version = "crab-gateway/1"
    sys_version = ""

    def log_message(self, fmt: str, *args: Any) -> None:
        logger.debug("crab-gateway: " + fmt, *args)

    def do_GET(self) -> None:  # noqa: N802
        self._dispatch("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._dispatch("POST")

    def do_DELETE(self) -> None:  # noqa: N802
        self._dispatch("DELETE")

    def _dispatch(self, method: str) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def _read_body(self) -> dict[str, Any]:
        length_header = self.headers.get("Content-Length")
        if not length_header:
            return {}
        try:
            length = int(length_header)
        except ValueError:
            raise _BadRequest("invalid Content-Length") from None
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        if not raw:
            return {}
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise _BadRequest(f"invalid JSON body: {exc}") from exc
        if not isinstance(decoded, dict):
            raise _BadRequest("request body must be a JSON object")
        return decoded

    def _send_json(self, status: HTTPStatus | int, payload: dict[str, Any]) -> None:
        self._send_raw(status, json.dumps(payload).encode("utf-8"))

    def _send_raw(self, status: HTTPStatus | int, body: bytes) -> None:
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass


def _build_handler(gateway: "GatewayServer"):
    routes = _GatewayRoutes(gateway)

    # (method, pattern, requires_auth, handler) — flat and auditable,
    # like the daemon's table.
    table: list[tuple[str, str, bool, Callable[..., dict[str, Any]]]] = [
        ("GET", "/healthz", False, routes.healthz),
        ("GET", f"{API_PREFIX}/healthz", False, routes.healthz),
        ("GET", f"{API_PREFIX}/info", True, routes.info),
        ("GET", f"{API_PREFIX}/sandboxes", True, routes.list_sandboxes),
        ("POST", f"{API_PREFIX}/sandboxes", True, routes.launch_sandbox),
        ("DELETE", f"{API_PREFIX}/sandboxes/{{sandbox_id}}", True, routes.kill_sandbox),
        ("POST", f"{API_PREFIX}/sandboxes/{{sandbox_id}}/fork", True, routes.fork_sandbox),
        # S4: port exposure routes
        ("POST", f"{API_PREFIX}/sandboxes/{{sandbox_id}}/ports", True, routes.expose_port),
        ("GET", f"{API_PREFIX}/sandboxes/{{sandbox_id}}/ports", True, routes.list_ports),
        ("DELETE", f"{API_PREFIX}/sandboxes/{{sandbox_id}}/ports/{{guest_port}}", True, routes.release_port),
        # S5 full-access: runtime helper route
        ("POST", f"{API_PREFIX}/runtime/write_bundle_spec", True,
         _make_global_passthrough(gateway, "POST", _FAST_TIMEOUT_S)),
    ]
    for method, subpath, timeout in _PASSTHROUGH_SANDBOX_ROUTES:
        pattern = f"{API_PREFIX}/sandboxes/{{sandbox_id}}{subpath}"
        table.append((method, pattern, True, _make_passthrough(gateway, method, timeout)))

    def _match(method: str, path: str):
        for route_method, pattern, requires_auth, fn in table:
            if route_method != method:
                continue
            variables = _try_match(pattern, path)
            if variables is not None:
                return fn, variables, requires_auth
        return None, None, False

    class Handler(_JsonHandler):
        def _dispatch(self, method: str) -> None:
            raw_path = self.path
            path = raw_path.split("?", 1)[0]
            query_string = raw_path.split("?", 1)[1] if "?" in raw_path else ""
            try:
                fn, variables, requires_auth = _match(method, path)
                if fn is None:
                    self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})
                    return
                tenant_id = self._authenticate() if requires_auth else None
                body = self._read_body()
                # Streaming exec fork: if exec route + stream=1, relay chunked
                if (
                    "stream=1" in query_string
                    and method == "POST"
                    and variables
                    and "sandbox_id" in variables
                    and path.endswith("/exec")
                ):
                    assert tenant_id is not None
                    self._handle_stream_exec(
                        tenant_id, body, variables["sandbox_id"]
                    )
                    return
                result = fn(tenant_id, body, path=path, **(variables or {}))
                self._send_json(HTTPStatus.OK, result)
            except _Unauthorized as exc:
                self._send_json(
                    HTTPStatus.UNAUTHORIZED,
                    {"ok": False, "error": str(exc), "error_type": "unauthorized"},
                )
            except _BadRequest as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
            except _NotFound as exc:
                self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": str(exc)})
            except _SandboxLost as exc:
                self._send_json(
                    HTTPStatus.GONE,
                    {"ok": False, "error": str(exc), "error_type": "sandbox_lost"},
                )
            except QuotaExceeded as exc:
                self._send_json(
                    HTTPStatus.CONFLICT,
                    {
                        "ok": False,
                        "error": str(exc),
                        "error_type": "quota_exceeded",
                        "quota": exc.quota,
                    },
                )
            except _DaemonError as exc:
                # Verbatim relay: the daemon's status and body are the
                # contract the SDK shims already understand.
                self._send_raw(exc.status_code, exc.body)
            except _DaemonUnreachable as exc:
                self._send_json(
                    HTTPStatus.BAD_GATEWAY,
                    {"ok": False, "error": str(exc), "error_type": "daemon_unreachable"},
                )
            except _GatewayTimeout as exc:
                self._send_json(
                    HTTPStatus.GATEWAY_TIMEOUT,
                    {"ok": False, "error": str(exc), "error_type": "daemon_timeout"},
                )
            except Exception as exc:
                logger.exception("gateway request failed: %s %s", method, path)
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"ok": False, "error": f"{type(exc).__name__}: {exc}"},
                )

        def _authenticate(self) -> str:
            header = self.headers.get("Authorization")
            if not header or not header.startswith("Bearer "):
                raise _Unauthorized("missing bearer API key")
            plaintext = header[len("Bearer "):].strip()
            tenant_id = gateway.registry.resolve_api_key(plaintext)
            if tenant_id is None:
                raise _Unauthorized("invalid or revoked API key")
            return tenant_id

        def _handle_stream_exec(
            self, tenant_id: str, body: dict[str, Any], sandbox_id: str
        ) -> None:
            """Relay chunked NDJSON from daemon's streaming exec to client."""
            try:
                gateway.require_owned(tenant_id, sandbox_id)
            except (_NotFound, _SandboxLost) as exc:
                self._send_json(
                    HTTPStatus.NOT_FOUND if isinstance(exc, _NotFound) else HTTPStatus.GONE,
                    {"ok": False, "error": str(exc)},
                )
                return
            # Open streaming connection to daemon
            daemon_path = f"/sandboxes/{sandbox_id}/exec?stream=1"
            try:
                stream = gateway._client.stream_post(daemon_path, body)
            except FileNotFoundError as exc:
                self._send_json(
                    HTTPStatus.BAD_GATEWAY,
                    {"ok": False, "error": str(exc), "error_type": "daemon_unreachable"},
                )
                return
            except Exception as exc:
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"ok": False, "error": f"{type(exc).__name__}: {exc}"},
                )
                return
            # Send chunked response headers
            self.send_response(200)
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                for event in stream:
                    line = (json.dumps(event) + "\n").encode("utf-8")
                    self.wfile.write(f"{len(line):x}\r\n".encode("ascii"))
                    self.wfile.write(line)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                stream.close()
                try:
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass

    return Handler


# ---------------------------------------------------------------------------
# Admin plane — tenants/keys/quotas over the gateway's own Unix socket.
# Local-only in v0: file permissions (0600 default) are the auth.
# ---------------------------------------------------------------------------


class _AdminRoutes:
    def __init__(self, gateway: "GatewayServer") -> None:
        self._gateway = gateway

    def create_tenant(self, body: dict[str, Any], **_: Any) -> dict[str, Any]:
        name = body.get("name")
        if not isinstance(name, str) or not name:
            raise _BadRequest("tenant create requires name")
        quotas_raw = body.get("quotas")
        if quotas_raw is not None and not isinstance(quotas_raw, dict):
            raise _BadRequest("quotas must be a JSON object")
        tenant = self._gateway.registry.create_tenant(name, quotas_raw)
        return {"ok": True, "tenant": tenant}

    def list_tenants(self, body: dict[str, Any], **_: Any) -> dict[str, Any]:
        return {"ok": True, "tenants": self._gateway.registry.list_tenants()}

    def create_key(self, body: dict[str, Any], **_: Any) -> dict[str, Any]:
        tenant_id = body.get("tenant_id")
        if not isinstance(tenant_id, str) or not tenant_id:
            raise _BadRequest("key create requires tenant_id")
        try:
            created = self._gateway.registry.create_api_key(tenant_id)
        except KeyError as exc:
            raise _NotFound(str(exc.args[0] if exc.args else exc)) from exc
        # Plaintext appears in this response only; the registry stores the hash.
        return {"ok": True, **created}

    def revoke_key(self, body: dict[str, Any], **_: Any) -> dict[str, Any]:
        key = body.get("key")
        if not isinstance(key, str) or not key:
            raise _BadRequest("key revoke requires key (plaintext or sha256)")
        revoked = self._gateway.registry.revoke_api_key(key)
        if not revoked:
            raise _NotFound("unknown API key")
        return {"ok": True, "revoked": True}

    def set_quotas(self, body: dict[str, Any], **_: Any) -> dict[str, Any]:
        tenant_id = body.get("tenant_id")
        quotas = body.get("quotas")
        if not isinstance(tenant_id, str) or not tenant_id:
            raise _BadRequest("quotas set requires tenant_id")
        if not isinstance(quotas, dict):
            raise _BadRequest("quotas set requires a quotas JSON object")
        try:
            tenant = self._gateway.registry.set_quotas(tenant_id, quotas)
        except KeyError as exc:
            raise _NotFound(str(exc.args[0] if exc.args else exc)) from exc
        return {"ok": True, "tenant": tenant}

    def adopt_sandboxes(self, body: dict[str, Any], **_: Any) -> dict[str, Any]:
        """Adopt daemon-side sandboxes into a tenant's registry."""
        tenant_ref = body.get("tenant")
        if not isinstance(tenant_ref, str) or not tenant_ref:
            raise _BadRequest("sandboxes adopt requires 'tenant' (name or id)")
        sandbox_ids = body.get("sandbox_ids")
        if not isinstance(sandbox_ids, list) or not sandbox_ids:
            raise _BadRequest("sandboxes adopt requires non-empty 'sandbox_ids' list")
        resources = body.get("resources")  # optional per-sandbox claim
        if resources is not None and not isinstance(resources, dict):
            raise _BadRequest("resources must be a JSON object or null")

        tenant = self._gateway.registry.resolve_tenant(tenant_ref)
        if tenant is None:
            raise _NotFound(f"unknown tenant: {tenant_ref}")
        tenant_id = tenant["id"]

        adopted = []
        skipped = []
        for sid in sandbox_ids:
            if not isinstance(sid, str) or not sid:
                continue
            ok = self._gateway.registry.adopt_sandbox(tenant_id, sid, resources)
            if ok:
                adopted.append(sid)
            else:
                skipped.append(sid)
        return {"ok": True, "adopted": adopted, "skipped": skipped}


def _build_admin_handler(gateway: "GatewayServer"):
    routes = _AdminRoutes(gateway)
    table: list[tuple[str, str, Callable[..., dict[str, Any]]]] = [
        ("POST", "/admin/tenants", routes.create_tenant),
        ("GET", "/admin/tenants", routes.list_tenants),
        ("POST", "/admin/keys", routes.create_key),
        ("POST", "/admin/keys/revoke", routes.revoke_key),
        ("POST", "/admin/quotas", routes.set_quotas),
        ("POST", "/admin/sandboxes/adopt", routes.adopt_sandboxes),
    ]

    class AdminHandler(_JsonHandler):
        def _dispatch(self, method: str) -> None:
            path = self.path.split("?", 1)[0]
            try:
                for route_method, pattern, fn in table:
                    if route_method == method and _try_match(pattern, path) is not None:
                        result = fn(self._read_body())
                        self._send_json(HTTPStatus.OK, result)
                        return
                self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})
            except _BadRequest as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
            except _NotFound as exc:
                self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": str(exc)})
            except Exception as exc:
                logger.exception("gateway admin request failed: %s %s", method, path)
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"ok": False, "error": f"{type(exc).__name__}: {exc}"},
                )

    return AdminHandler


# ---------------------------------------------------------------------------
# GatewayServer — owns the registry, the daemon client, and both listeners.
# ---------------------------------------------------------------------------


class GatewayServer:
    """The crab-gateway process."""

    _RECONCILE_INTERVAL_S: float = 60.0

    def __init__(
        self,
        *,
        data_dir: Path | None = None,
        daemon_socket: Path | os.PathLike[str] | str | None = None,
        daemon_client: DaemonClient | None = None,
        host: str = DEFAULT_BIND_HOST,
        port: int = DEFAULT_BIND_PORT,
        admin_socket_path: Path | None = None,
    ) -> None:
        self._data_dir = (data_dir or default_data_dir()).expanduser().resolve()
        self._client = daemon_client or DaemonClient(daemon_socket)
        self._host = host
        self._port = int(port)
        self._admin_socket_path = (
            (admin_socket_path or default_admin_socket_path()).expanduser().resolve()
        )
        self._registry: GatewayRegistry | None = None
        self._http_server: ThreadingHTTPServer | None = None
        self._admin_server = None
        self._threads: list[threading.Thread] = []
        self._stop_event = threading.Event()
        self._port_manager = PortManager()
        self._reconcile_interval: float = float(
            os.environ.get("CRAB_GATEWAY_RECONCILE_S", self._RECONCILE_INTERVAL_S)
        )

    @property
    def registry(self) -> GatewayRegistry:
        registry = self._registry
        if registry is None:
            raise RuntimeError("gateway is not started")
        return registry

    @property
    def port_manager(self) -> PortManager:
        return self._port_manager

    @property
    def port(self) -> int:
        """The bound TCP port (useful with port=0 in tests)."""
        return self._port

    @property
    def admin_socket_path(self) -> Path:
        return self._admin_socket_path

    # ----- daemon access ----------------------------------------------------

    def proxy(
        self,
        method: str,
        daemon_path: str,
        body: dict[str, Any] | None,
        timeout: float,
    ) -> dict[str, Any]:
        """One `DaemonClient` call with the route's timeout, translating
        transport failures into the gateway's error taxonomy."""
        try:
            if method == "GET":
                return self._client.get_json(daemon_path, timeout_seconds=timeout)
            if method == "DELETE":
                if body:
                    # `DaemonClient.delete` is body-less; the daemon's
                    # checkpoint DELETE takes an optional JSON body
                    # (cascade), so drop to the private request seam.
                    return self._client._request_json(
                        "DELETE",
                        daemon_path,
                        body=json.dumps(body).encode("utf-8"),
                        timeout_seconds=timeout,
                    )
                return self._client.delete(daemon_path, timeout_seconds=timeout)
            return self._client.post_json(daemon_path, body or None, timeout_seconds=timeout)
        except DaemonRequestError as exc:
            raise _DaemonError(exc) from exc
        except FileNotFoundError as exc:
            raise _DaemonUnreachable(str(exc)) from exc
        except TimeoutError as exc:
            raise _GatewayTimeout(
                f"daemon call timed out after {timeout:.0f}s: {method} {daemon_path}"
            ) from exc

    def require_owned(self, tenant_id: str, sandbox_id: str) -> dict[str, Any]:
        """Ownership gate for every per-sandbox route. Unknown, other
        tenant's, pending, or killed -> 404 (never 403 — don't leak
        existence); lost -> 410."""
        row = self.registry.get_sandbox(sandbox_id)
        if row is None or row["tenant_id"] != tenant_id or row["status"] in ("pending", "killed"):
            raise _NotFound(f"unknown sandbox: {sandbox_id}")
        if row["status"] == "lost":
            raise _SandboxLost(
                f"sandbox {sandbox_id} was lost to a daemon restart (pre-S5 semantics)"
            )
        return row

    # ----- reconciliation -----------------------------------------------------

    def reconcile(self) -> None:
        """Startup pass (the only one in v0 — no periodic loop, §10):

        1. `/info` boot identity (pid) vs the last stored value; on
           mismatch every active row flips `lost`.
        2. Pending rows are matched against daemon sandboxes via the
           injected `gateway_intent_id` metadata: match -> `active`
           under the real id; no match -> reaped.
        3. Active rows whose sandbox the daemon no longer lists flip
           `lost`; daemon sandboxes with no row are logged as orphans
           and stay invisible to every tenant."""
        registry = self.registry
        info = self.proxy("GET", "/info", None, _FAST_TIMEOUT_S)
        boot_id = str(info.get("pid"))
        stored = registry.get_meta("daemon_boot_id")
        if stored is not None and stored != boot_id:
            lost = registry.mark_all_active_lost()
            logger.warning(
                "daemon boot identity changed (%s -> %s); marked %d sandbox rows lost",
                stored,
                boot_id,
                lost,
            )
        listing = self.proxy("GET", "/sandboxes", None, _FAST_TIMEOUT_S)
        present: set[str] = set()
        intents: dict[str, str] = {}
        for row in listing.get("sandboxes") or []:
            sandbox_id = str(row.get("sandbox_id") or "")
            if not sandbox_id:
                continue
            present.add(sandbox_id)
            metadata = row.get("metadata") or {}
            intent = metadata.get(GATEWAY_INTENT_METADATA_KEY)
            if intent:
                intents[str(intent)] = sandbox_id
        for pending in registry.pending_rows():
            intent_id = pending["sandbox_id"]
            matched = intents.get(intent_id)
            if matched and registry.get_sandbox(matched) is None:
                registry.complete_create(intent_id, matched)
                logger.info(
                    "reconciled pending create %s -> active sandbox %s", intent_id, matched
                )
            else:
                registry.abort_create(intent_id)
                logger.info("reaped pending create %s (no matching daemon sandbox)", intent_id)
        missing = registry.mark_missing_lost(present)
        if missing:
            logger.warning(
                "marked %d sandbox rows lost (daemon no longer lists them): %s",
                len(missing),
                ", ".join(missing),
            )
        orphans = present - registry.all_tracked_ids()
        if orphans:
            logger.warning(
                "daemon has %d sandbox(es) with no registry row (invisible to tenants; "
                "operator cleanup, see S5): %s",
                len(orphans),
                ", ".join(sorted(orphans)),
            )
        registry.set_meta("daemon_boot_id", boot_id)

    def _periodic_reconcile(self) -> None:
        """Lightweight periodic reconciliation: compare daemon listing against
        the registry and mark vanished sandboxes as lost. Runs every
        `_reconcile_interval` seconds inside `serve_forever`."""
        try:
            listing = self.proxy("GET", "/sandboxes", None, _FAST_TIMEOUT_S)
        except (_DaemonUnreachable, _GatewayTimeout, FileNotFoundError):
            logger.debug("periodic reconcile skipped: daemon unreachable")
            return
        except Exception as exc:
            logger.warning("periodic reconcile failed: %s", exc)
            return
        present: set[str] = set()
        for row in listing.get("sandboxes") or []:
            sid = str(row.get("sandbox_id") or "")
            if sid:
                present.add(sid)
        lost_ids = self.registry.mark_missing_lost(present)
        if lost_ids:
            for sid in lost_ids:
                host_ports = self.registry.release_all_ports(sid)
                if host_ports:
                    self._port_manager.release_all(sid, host_ports)
            logger.warning("periodic reconcile: marked %d sandboxes lost", len(lost_ids))
        orphans = present - self.registry.all_tracked_ids()
        if orphans:
            logger.info(
                "periodic reconcile: %d daemon orphans (not in registry)", len(orphans)
            )

    def _rehydrate_ports(self) -> None:
        """Restart forwarders for persisted port allocations (S5).

        Called once at startup after reconcile(). Skips allocations whose
        sandbox is no longer active; removes DB rows for ports that cannot
        be re-bound (occupied by another process)."""
        allocations = self.registry.list_all_port_allocations()
        if not allocations:
            return
        # Only rehydrate for sandboxes that are still active
        active_ids = self.registry.active_sandbox_ids()
        rebuilt = 0
        cleaned = 0
        for alloc in allocations:
            sid = alloc["sandbox_id"]
            if sid not in active_ids:
                # Sandbox is lost/killed — clean up stale DB row
                self.registry.release_port(sid, alloc["guest_port"])
                cleaned += 1
                continue
            try:
                self._port_manager.rehydrate(
                    alloc["host_port"], alloc["guest_ip"], alloc["guest_port"]
                )
                rebuilt += 1
            except OSError:
                # Port occupied — release the DB row
                self.registry.release_port(sid, alloc["guest_port"])
                cleaned += 1
                logger.warning(
                    "port rehydration: cannot bind host_port %d for %s:%d — released",
                    alloc["host_port"], sid, alloc["guest_port"],
                )
        if rebuilt or cleaned:
            logger.info(
                "port rehydration: rebuilt %d forwarders, cleaned %d stale rows",
                rebuilt, cleaned,
            )

    # ----- lifecycle ------------------------------------------------------

    def start(self) -> None:
        if self._registry is not None:
            raise RuntimeError("gateway is already started")
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._registry = GatewayRegistry(self._data_dir / "gateway.sqlite3")
        try:
            # Fail fast when the daemon is unreachable at boot — a gateway
            # that cannot see its daemon has nothing truthful to serve.
            self.reconcile()
            self._rehydrate_ports()
            handler = _build_handler(self)
            self._http_server = ThreadingHTTPServer((self._host, self._port), handler)
            self._http_server.daemon_threads = True
            self._port = int(self._http_server.server_address[1])
            admin_handler = _build_admin_handler(self)
            self._admin_server = serve_unix_socket(self._admin_socket_path, admin_handler)
        except Exception:
            self.stop()
            raise
        for name, server in (
            ("crab-gateway-http", self._http_server),
            ("crab-gateway-admin", self._admin_server),
        ):
            thread = threading.Thread(target=server.serve_forever, name=name, daemon=True)
            thread.start()
            self._threads.append(thread)
        logger.info(
            "crab-gateway ready: http=%s:%d admin=%s registry=%s daemon=%s",
            self._host,
            self._port,
            self._admin_socket_path,
            self.registry.path,
            self._client.socket_path,
        )

    def stop(self) -> None:
        for server in (self._http_server, self._admin_server):
            if server is not None:
                try:
                    server.shutdown()
                    server.server_close()
                except Exception:
                    logger.exception("gateway server shutdown failed")
        self._http_server = None
        self._admin_server = None
        for thread in self._threads:
            thread.join(timeout=5.0)
        self._threads = []
        try:
            if self._admin_socket_path.exists():
                self._admin_socket_path.unlink()
        except OSError:
            pass
        if self._registry is not None:
            self._registry.close()
            self._registry = None
        self._port_manager.shutdown()

    def request_shutdown(self) -> None:
        self._stop_event.set()

    def serve_forever(self) -> None:
        """Block until SIGTERM/SIGINT or `request_shutdown()`."""
        self._install_signal_handlers()
        last_reconcile = time.monotonic()
        try:
            while not self._stop_event.wait(timeout=1.0):
                now = time.monotonic()
                if now - last_reconcile >= self._reconcile_interval:
                    last_reconcile = now
                    self._periodic_reconcile()
        finally:
            self.stop()

    def _install_signal_handlers(self) -> None:
        def _on_signal(signum: int, _frame: Any) -> None:
            logger.info("gateway received signal %d; shutting down", signum)
            self._stop_event.set()

        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, _on_signal)
            except ValueError:
                # Not on the main thread; skipped in test embedding.
                pass
