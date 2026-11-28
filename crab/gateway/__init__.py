"""crab-gateway — the multi-tenant service layer in front of a crab daemon.

Track S (S1). Operators run one gateway per deployment
(`python -m crab.gateway` / `crab-gateway serve`); it is the only
internet-facing component. The daemon underneath stays unchanged — the
gateway authenticates bearer API keys, resolves tenancy and ownership
against its own SQLite registry, enforces per-tenant quotas, and
proxies each authorized `/v1/...` request to the daemon over its Unix
socket via `DaemonClient`.

Public surface:
  - `GatewayServer` (server.py): the process — public `/v1` facade,
    local-only admin plane, startup reconciliation.
  - `GatewayRegistry` (registry.py): tenants / API keys / sandbox
    ownership, SQLite in WAL mode, two-phase create.
  - `default_data_dir()` / `default_admin_socket_path()` (server.py):
    the conventional state and admin-socket locations.
"""
from __future__ import annotations

from .registry import GatewayRegistry, QuotaExceeded
from .server import GatewayServer, default_admin_socket_path, default_data_dir

__all__ = [
    "GatewayRegistry",
    "GatewayServer",
    "QuotaExceeded",
    "default_admin_socket_path",
    "default_data_dir",
]
