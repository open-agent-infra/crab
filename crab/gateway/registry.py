"""SQLite-backed gateway registry: tenants, API keys, sandbox ownership.

The registry is the gateway's only durable state — the same "cheap
mirror" role the daemon's in-memory sandbox set plays, made persistent
and tenant-aware. It runs in WAL mode so the threaded HTTP server can
read while a write is in flight, and every multi-statement invariant
(the quota gate + intent insert) runs under `BEGIN IMMEDIATE` so two
concurrent creates cannot both pass the quota check.

Two-phase create (crash-safety, see the track design doc §4 S1): the
daemon assigns sandbox ids at launch, so the gateway cannot know the
final id before calling the daemon. `begin_create` therefore inserts a
`pending` row keyed by a gateway-generated *intent id*
(`pending:<uuid>`), which the gateway also injects into the create
request's metadata. `complete_create` rekeys the row to the daemon's
sandbox id and flips it `active`. A crash between the two steps leaves
a pending row that the startup reconciliation pass resolves by matching
intent ids against daemon-side sandbox metadata.

API keys are stored as SHA-256 digests only; the plaintext
(`crab_sk_...`) is returned exactly once, at creation.
"""
from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

API_KEY_PREFIX = "crab_sk_"
"""Plaintext API keys look like `crab_sk_<48 hex chars>`."""

PENDING_ID_PREFIX = "pending:"
"""Placeholder `sandbox_id` used for rows inserted by `begin_create`
before the daemon has assigned the real id."""

LIVE_STATUSES = ("pending", "active")
"""Statuses that count against the tenant's quotas (`max_sandboxes` and
the S3 aggregate resource caps)."""

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tenants (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    quota_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS api_keys (
    key_sha256 TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL REFERENCES tenants(id),
    created_at TEXT NOT NULL,
    revoked INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS sandboxes (
    sandbox_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL REFERENCES tenants(id),
    name TEXT,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL,
    resources_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def hash_api_key(plaintext: str) -> str:
    return hashlib.sha256(plaintext.encode("utf-8")).hexdigest()


class QuotaExceeded(Exception):
    """A create/fork would push the tenant past `max_sandboxes` or one of
    the S3 aggregate resource caps (`max_memory_bytes`, `max_cpu`).

    Carries the quota arithmetic so the HTTP layer can put it in the
    409 error body, per the design doc."""

    def __init__(self, message: str, quota: dict[str, Any]) -> None:
        super().__init__(message)
        self.quota = dict(quota)


class GatewayRegistry:
    """Thread-safe wrapper over the gateway's SQLite file.

    One connection, shared across request threads: `sqlite3` objects are
    only safe from the creating thread by default, so the connection is
    opened with `check_same_thread=False` and every access is serialized
    under an `RLock`. That is enough at sandbox-control request rates —
    the daemon call, not the registry, dominates request latency."""

    def __init__(self, db_path: Path | str) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        # Autocommit mode (isolation_level=None): single statements commit
        # immediately; multi-statement invariants use explicit
        # BEGIN IMMEDIATE so the quota check and the insert are atomic.
        self._conn = sqlite3.connect(
            str(self._path), check_same_thread=False, isolation_level=None
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)
        self._migrate()
        self._migrate_ports()

    def _migrate(self) -> None:
        """Additive migrations for registries created by older gateways.
        S3 added `sandboxes.resources_json` (per-sandbox claim, feeds the
        aggregate quota gate); pre-S3 rows read as `{}` — no limits."""
        columns = {
            str(row["name"])
            for row in self._conn.execute("PRAGMA table_info(sandboxes)").fetchall()
        }
        if "resources_json" not in columns:
            self._conn.execute(
                "ALTER TABLE sandboxes ADD COLUMN resources_json TEXT NOT NULL DEFAULT '{}'"
            )

    @property
    def path(self) -> Path:
        return self._path

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def journal_mode(self) -> str:
        with self._lock:
            row = self._conn.execute("PRAGMA journal_mode").fetchone()
        return str(row[0])

    # ----- tenants ------------------------------------------------------

    def create_tenant(self, name: str, quotas: dict[str, Any] | None = None) -> dict[str, Any]:
        tenant_id = "tn_" + uuid.uuid4().hex[:12]
        quota_json = json.dumps(quotas or {})
        with self._lock:
            self._conn.execute(
                "INSERT INTO tenants (id, name, quota_json) VALUES (?, ?, ?)",
                (tenant_id, name, quota_json),
            )
        return {"id": tenant_id, "name": name, "quotas": json.loads(quota_json)}

    def get_tenant(self, tenant_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT id, name, quota_json FROM tenants WHERE id = ?", (tenant_id,)
            ).fetchone()
        if row is None:
            return None
        return {"id": row["id"], "name": row["name"], "quotas": json.loads(row["quota_json"])}

    def get_tenant_by_name(self, name: str) -> dict[str, Any] | None:
        """Look up a tenant by its human-readable name."""
        with self._lock:
            row = self._conn.execute(
                "SELECT id, name, quota_json FROM tenants WHERE name = ?", (name,)
            ).fetchone()
        if row is None:
            return None
        return {"id": row["id"], "name": row["name"], "quotas": json.loads(row["quota_json"])}

    def resolve_tenant(self, tenant_ref: str) -> dict[str, Any] | None:
        """Resolve a tenant by id or name (tries id first)."""
        tenant = self.get_tenant(tenant_ref)
        if tenant is None:
            tenant = self.get_tenant_by_name(tenant_ref)
        return tenant

    def list_tenants(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, name, quota_json FROM tenants ORDER BY name"
            ).fetchall()
        return [
            {"id": row["id"], "name": row["name"], "quotas": json.loads(row["quota_json"])}
            for row in rows
        ]

    def set_quotas(self, tenant_id: str, quotas: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE tenants SET quota_json = ? WHERE id = ?",
                (json.dumps(dict(quotas)), tenant_id),
            )
            if cursor.rowcount == 0:
                raise KeyError(f"unknown tenant: {tenant_id}")
        tenant = self.get_tenant(tenant_id)
        assert tenant is not None
        return tenant

    # ----- API keys -----------------------------------------------------

    def create_api_key(self, tenant_id: str) -> dict[str, Any]:
        """Mint a key for `tenant_id`. The plaintext appears only in the
        returned dict — the row stores the SHA-256 digest."""
        plaintext = API_KEY_PREFIX + secrets.token_hex(24)
        digest = hash_api_key(plaintext)
        with self._lock:
            if self.get_tenant(tenant_id) is None:
                raise KeyError(f"unknown tenant: {tenant_id}")
            self._conn.execute(
                "INSERT INTO api_keys (key_sha256, tenant_id, created_at, revoked)"
                " VALUES (?, ?, ?, 0)",
                (digest, tenant_id, _utc_now()),
            )
        return {"api_key": plaintext, "key_sha256": digest, "tenant_id": tenant_id}

    def revoke_api_key(self, key_or_sha256: str) -> bool:
        """Revoke by plaintext key or by its SHA-256 digest (the digest is
        what `keys create` printed alongside the plaintext)."""
        digest = (
            hash_api_key(key_or_sha256)
            if key_or_sha256.startswith(API_KEY_PREFIX)
            else key_or_sha256
        )
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE api_keys SET revoked = 1 WHERE key_sha256 = ?", (digest,)
            )
            return cursor.rowcount > 0

    def resolve_api_key(self, plaintext: str) -> str | None:
        """Plaintext bearer key -> tenant id, or None when unknown/revoked."""
        digest = hash_api_key(plaintext)
        with self._lock:
            row = self._conn.execute(
                "SELECT tenant_id FROM api_keys WHERE key_sha256 = ? AND revoked = 0",
                (digest,),
            ).fetchone()
        return None if row is None else str(row["tenant_id"])

    # ----- sandboxes: two-phase create ----------------------------------

    def begin_create(
        self,
        tenant_id: str,
        name: str | None = None,
        resources: dict[str, Any] | None = None,
    ) -> str:
        """Phase one: durably record the create intent, gated by quota.

        Runs quota check + insert under BEGIN IMMEDIATE so concurrent
        creates serialize on the write lock and cannot both pass the
        check. `resources` is the sandbox's normalized claim; it is
        stored on the pending row so the aggregate gate already counts
        in-flight creates. Returns the intent id the caller must inject
        into the daemon create request's metadata."""
        intent_id = PENDING_ID_PREFIX + uuid.uuid4().hex
        claim = dict(resources or {})
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._check_capacity_locked(tenant_id, additional=1, resources=claim)
                self._conn.execute(
                    "INSERT INTO sandboxes"
                    " (sandbox_id, tenant_id, name, created_at, status, resources_json)"
                    " VALUES (?, ?, ?, ?, 'pending', ?)",
                    (intent_id, tenant_id, name, _utc_now(), json.dumps(claim)),
                )
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
        return intent_id

    def complete_create(self, intent_id: str, sandbox_id: str) -> None:
        """Phase two: rekey the pending row to the daemon-assigned id."""
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE sandboxes SET sandbox_id = ?, status = 'active'"
                " WHERE sandbox_id = ? AND status = 'pending'",
                (sandbox_id, intent_id),
            )
            if cursor.rowcount == 0:
                raise KeyError(f"unknown pending create: {intent_id}")

    def abort_create(self, intent_id: str) -> None:
        """Reap a pending row whose daemon create definitively failed."""
        with self._lock:
            self._conn.execute(
                "DELETE FROM sandboxes WHERE sandbox_id = ? AND status = 'pending'",
                (intent_id,),
            )

    def ensure_capacity(
        self,
        tenant_id: str,
        additional: int,
        resources: dict[str, Any] | None = None,
    ) -> None:
        """Quota pre-check without an insert — used by fork, which learns
        the children's ids only after the daemon call (they are then
        registered via `register_sandbox`; the fork children are not
        two-phase, see the design doc). `resources` is the *per-child*
        claim (forks inherit the source's limits), counted `additional`
        times by the aggregate gate."""
        with self._lock:
            self._check_capacity_locked(
                tenant_id, additional=additional, resources=dict(resources or {})
            )

    def _check_capacity_locked(
        self,
        tenant_id: str,
        *,
        additional: int,
        resources: dict[str, Any] | None = None,
    ) -> None:
        tenant_row = self._conn.execute(
            "SELECT quota_json FROM tenants WHERE id = ?", (tenant_id,)
        ).fetchone()
        if tenant_row is None:
            raise KeyError(f"unknown tenant: {tenant_id}")
        quotas = json.loads(tenant_row["quota_json"])
        max_sandboxes = quotas.get("max_sandboxes")
        live = self._live_count_locked(tenant_id)
        if max_sandboxes is not None and live + additional > int(max_sandboxes):
            raise QuotaExceeded(
                f"sandbox quota exceeded for tenant {tenant_id}",
                {
                    "max_sandboxes": int(max_sandboxes),
                    "live_sandboxes": live,
                    "requested": additional,
                },
            )
        claim = resources or {}
        self._check_aggregate_locked(
            tenant_id,
            quotas,
            additional=additional,
            claim_key="memory_bytes",
            cap_key="max_memory_bytes",
            claim_value=claim.get("memory_bytes"),
            unit="memory",
        )
        self._check_aggregate_locked(
            tenant_id,
            quotas,
            additional=additional,
            claim_key="cpus",
            cap_key="max_cpu",
            claim_value=claim.get("cpus"),
            unit="cpu",
        )

    def _check_aggregate_locked(
        self,
        tenant_id: str,
        quotas: dict[str, Any],
        *,
        additional: int,
        claim_key: str,
        cap_key: str,
        claim_value: Any,
        unit: str,
    ) -> None:
        """One S3 aggregate gate (§4 S3): with `cap_key` configured, the sum
        of live claims plus the request must stay within the cap. A capped
        tenant's sandboxes must declare the corresponding limit — an
        undeclared (unbounded) sandbox would escape the accounting, so it
        is refused outright. Uncapped tenants skip the gate entirely.

        Payload arithmetic keys are per-dimension: `max_memory_bytes` /
        `live_memory_bytes` / `requested_memory_bytes` (and the `_cpu`
        analogs, with `max_cpu` matching the quota_json spelling).
        """
        cap_raw = quotas.get(cap_key)
        if cap_raw is None:
            return
        cap = int(cap_raw)
        live_key = "live_cpu" if unit == "cpu" else "live_memory_bytes"
        requested_key = "requested_cpu" if unit == "cpu" else "requested_memory_bytes"
        live_sum = self._live_resource_sum_locked(tenant_id, claim_key)
        if claim_value is None:
            raise QuotaExceeded(
                f"tenant {tenant_id} has a {cap_key} quota; sandboxes must declare"
                f" a {unit} limit in resources",
                {cap_key: cap, live_key: live_sum, requested_key: None},
            )
        requested = int(claim_value) * max(int(additional), 0)
        if live_sum + requested > cap:
            raise QuotaExceeded(
                f"{unit} quota exceeded for tenant {tenant_id}",
                {cap_key: cap, live_key: live_sum, requested_key: requested},
            )

    # ----- sandboxes: ownership and status -------------------------------

    def register_sandbox(
        self,
        tenant_id: str,
        sandbox_id: str,
        name: str | None = None,
        status: str = "active",
        resources: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO sandboxes"
                " (sandbox_id, tenant_id, name, created_at, status, resources_json)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (sandbox_id, tenant_id, name, _utc_now(), status, json.dumps(dict(resources or {}))),
            )

    def adopt_sandbox(
        self,
        tenant_id: str,
        sandbox_id: str,
        resources: dict[str, Any] | None = None,
    ) -> bool:
        """Register an existing daemon sandbox into a tenant. Idempotent:
        returns True if newly adopted, False if already registered."""
        with self._lock:
            existing = self._conn.execute(
                "SELECT sandbox_id FROM sandboxes WHERE sandbox_id = ?",
                (sandbox_id,),
            ).fetchone()
            if existing is not None:
                return False
            self._conn.execute(
                "INSERT INTO sandboxes"
                " (sandbox_id, tenant_id, name, created_at, status, resources_json)"
                " VALUES (?, ?, ?, ?, 'active', ?)",
                (sandbox_id, tenant_id, None, _utc_now(), json.dumps(dict(resources or {}))),
            )
            return True

    @staticmethod
    def _row_dict(row: sqlite3.Row) -> dict[str, Any]:
        """Row -> dict with `resources_json` parsed into a `resources` claim."""
        out = dict(row)
        out["resources"] = json.loads(out.pop("resources_json", None) or "{}")
        return out

    def get_sandbox(self, sandbox_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT sandbox_id, tenant_id, name, created_at, status, resources_json"
                " FROM sandboxes WHERE sandbox_id = ?",
                (sandbox_id,),
            ).fetchone()
        return None if row is None else self._row_dict(row)

    def set_status(self, sandbox_id: str, status: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE sandboxes SET status = ? WHERE sandbox_id = ?",
                (status, sandbox_id),
            )

    def list_sandboxes(
        self, tenant_id: str, statuses: Iterable[str] | None = None
    ) -> list[dict[str, Any]]:
        wanted = tuple(statuses) if statuses is not None else None
        with self._lock:
            if wanted is None:
                rows = self._conn.execute(
                    "SELECT sandbox_id, tenant_id, name, created_at, status, resources_json"
                    " FROM sandboxes WHERE tenant_id = ? ORDER BY created_at",
                    (tenant_id,),
                ).fetchall()
            else:
                placeholders = ",".join("?" for _ in wanted)
                rows = self._conn.execute(
                    "SELECT sandbox_id, tenant_id, name, created_at, status, resources_json"
                    f" FROM sandboxes WHERE tenant_id = ? AND status IN ({placeholders})"
                    " ORDER BY created_at",
                    (tenant_id, *wanted),
                ).fetchall()
        return [self._row_dict(row) for row in rows]

    def live_count(self, tenant_id: str) -> int:
        with self._lock:
            return self._live_count_locked(tenant_id)

    def _live_count_locked(self, tenant_id: str) -> int:
        placeholders = ",".join("?" for _ in LIVE_STATUSES)
        row = self._conn.execute(
            "SELECT COUNT(*) FROM sandboxes"
            f" WHERE tenant_id = ? AND status IN ({placeholders})",
            (tenant_id, *LIVE_STATUSES),
        ).fetchone()
        return int(row[0])

    def _live_resource_sum_locked(self, tenant_id: str, claim_key: str) -> int:
        """Sum of one claim dimension over the tenant's live rows. Python-side
        integer arithmetic over the JSON column; runs under the same lock
        (and, from `begin_create`, the same BEGIN IMMEDIATE transaction) as
        the gate that consumes it, so concurrent creates stay precise."""
        placeholders = ",".join("?" for _ in LIVE_STATUSES)
        rows = self._conn.execute(
            "SELECT resources_json FROM sandboxes"
            f" WHERE tenant_id = ? AND status IN ({placeholders})",
            (tenant_id, *LIVE_STATUSES),
        ).fetchall()
        total = 0
        for row in rows:
            value = (json.loads(row["resources_json"] or "{}")).get(claim_key)
            if value is not None:
                total += int(value)
        return total

    # ----- reconciliation helpers (startup pass) --------------------------

    def pending_rows(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT sandbox_id, tenant_id, name, created_at, status, resources_json"
                " FROM sandboxes WHERE status = 'pending'"
            ).fetchall()
        return [self._row_dict(row) for row in rows]

    def all_tracked_ids(self) -> set[str]:
        """Every non-pending sandbox id the registry knows (any status) —
        used to spot daemon-side orphans, which stay invisible to tenants."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT sandbox_id FROM sandboxes WHERE status != 'pending'"
            ).fetchall()
        return {str(row["sandbox_id"]) for row in rows}

    def active_sandbox_ids(self) -> set[str]:
        """Only sandbox ids with status='active' (for port rehydration)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT sandbox_id FROM sandboxes WHERE status = 'active'"
            ).fetchall()
        return {str(row["sandbox_id"]) for row in rows}

    def mark_all_active_lost(self) -> int:
        """Boot-identity mismatch: the daemon restarted and (pre-S5) lost
        every sandbox — flip all active rows to `lost` so their routes
        answer 410 instead of pretending state survived."""
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE sandboxes SET status = 'lost' WHERE status = 'active'"
            )
            return cursor.rowcount

    def mark_missing_lost(self, present_ids: set[str]) -> list[str]:
        """Flip active rows whose sandbox the daemon no longer lists to
        `lost`; returns the affected ids."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT sandbox_id FROM sandboxes WHERE status = 'active'"
            ).fetchall()
            missing = [str(row["sandbox_id"]) for row in rows if row["sandbox_id"] not in present_ids]
            for sandbox_id in missing:
                self._conn.execute(
                    "UPDATE sandboxes SET status = 'lost' WHERE sandbox_id = ?",
                    (sandbox_id,),
                )
        return missing

    # ----- port allocations -----------------------------------------------

    def _migrate_ports(self) -> None:
        """S4/S5 migration: port_allocations table + guest_ip column."""
        with self._lock:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS port_allocations ("
                "  id INTEGER PRIMARY KEY,"
                "  sandbox_id TEXT NOT NULL,"
                "  guest_port INTEGER NOT NULL,"
                "  host_port INTEGER NOT NULL UNIQUE,"
                "  tenant_id TEXT NOT NULL,"
                "  created_at TEXT NOT NULL"
                ")"
            )
            # S5: add guest_ip column for port rehydration
            cols = {
                row[1]
                for row in self._conn.execute(
                    "PRAGMA table_info(port_allocations)"
                ).fetchall()
            }
            if "guest_ip" not in cols:
                self._conn.execute(
                    "ALTER TABLE port_allocations"
                    " ADD COLUMN guest_ip TEXT NOT NULL DEFAULT '127.0.0.1'"
                )

    def allocate_port(
        self, sandbox_id: str, tenant_id: str, guest_port: int, host_port: int,
        *, guest_ip: str = "127.0.0.1",
    ) -> int:
        """Record a port allocation; returns the row id."""
        with self._lock:
            cursor = self._conn.execute(
                "INSERT INTO port_allocations"
                " (sandbox_id, tenant_id, guest_port, host_port, guest_ip, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (sandbox_id, tenant_id, int(guest_port), int(host_port), guest_ip, _utc_now()),
            )
            return cursor.lastrowid  # type: ignore[return-value]

    def release_port(self, sandbox_id: str, guest_port: int) -> int | None:
        """Release one allocation; returns the host_port freed (or None)."""
        with self._lock:
            row = self._conn.execute(
                "SELECT host_port FROM port_allocations"
                " WHERE sandbox_id = ? AND guest_port = ?",
                (sandbox_id, int(guest_port)),
            ).fetchone()
            if row is None:
                return None
            host_port = int(row["host_port"])
            self._conn.execute(
                "DELETE FROM port_allocations"
                " WHERE sandbox_id = ? AND guest_port = ?",
                (sandbox_id, int(guest_port)),
            )
            return host_port

    def release_all_ports(self, sandbox_id: str) -> list[int]:
        """Release all port allocations for a sandbox; returns freed host_ports."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT host_port FROM port_allocations WHERE sandbox_id = ?",
                (sandbox_id,),
            ).fetchall()
            host_ports = [int(row["host_port"]) for row in rows]
            if host_ports:
                self._conn.execute(
                    "DELETE FROM port_allocations WHERE sandbox_id = ?",
                    (sandbox_id,),
                )
            return host_ports

    def list_ports(self, sandbox_id: str) -> list[dict[str, Any]]:
        """List all port allocations for a sandbox."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, sandbox_id, tenant_id, guest_port, host_port, created_at"
                " FROM port_allocations WHERE sandbox_id = ? ORDER BY guest_port",
                (sandbox_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def count_tenant_ports(self, tenant_id: str) -> int:
        """Count total port allocations across all sandboxes for a tenant."""
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM port_allocations WHERE tenant_id = ?",
                (tenant_id,),
            ).fetchone()
            return int(row[0])

    def list_all_port_allocations(self) -> list[dict[str, Any]]:
        """All port allocations (for rehydration at startup)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT sandbox_id, guest_port, host_port, guest_ip, tenant_id"
                " FROM port_allocations"
            ).fetchall()
        return [dict(row) for row in rows]

    # ----- meta -----------------------------------------------------------

    def get_meta(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM meta WHERE key = ?", (key,)
            ).fetchone()
        return None if row is None else str(row["value"])

    def set_meta(self, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?)"
                " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )
