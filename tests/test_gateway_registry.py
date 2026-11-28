"""Unit tests for the gateway registry (track S, S1): WAL mode, tenants,
hashed API keys, the two-phase create protocol, quota arithmetic, and
reconciliation helpers. S3 added the aggregate resource caps
(`max_memory_bytes`/`max_cpu`) and the `resources_json` column — covered
below. Host-runnable — no daemon, no root."""
from __future__ import annotations

import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path

from crab.gateway.registry import (
    API_KEY_PREFIX,
    PENDING_ID_PREFIX,
    GatewayRegistry,
    QuotaExceeded,
    hash_api_key,
)


class RegistryTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db_path = Path(self._tmp.name) / "gateway.sqlite3"
        self.registry = GatewayRegistry(self.db_path)
        self.addCleanup(self.registry.close)


class WalAndSchemaTests(RegistryTestCase):
    def test_wal_mode_enabled(self) -> None:
        self.assertEqual(self.registry.journal_mode().lower(), "wal")

    def test_state_survives_reopen(self) -> None:
        tenant = self.registry.create_tenant("acme", {"max_sandboxes": 3})
        created = self.registry.create_api_key(tenant["id"])
        self.registry.register_sandbox(tenant["id"], "sb-1")
        self.registry.set_meta("daemon_boot_id", "1234")
        self.registry.close()

        reopened = GatewayRegistry(self.db_path)
        self.addCleanup(reopened.close)
        self.assertEqual(reopened.get_tenant(tenant["id"])["quotas"], {"max_sandboxes": 3})
        self.assertEqual(reopened.resolve_api_key(created["api_key"]), tenant["id"])
        self.assertEqual(reopened.get_sandbox("sb-1")["status"], "active")
        self.assertEqual(reopened.get_meta("daemon_boot_id"), "1234")


class TenantTests(RegistryTestCase):
    def test_create_and_list(self) -> None:
        a = self.registry.create_tenant("acme")
        b = self.registry.create_tenant("blit", {"max_sandboxes": 1})
        self.assertTrue(a["id"].startswith("tn_"))
        self.assertEqual(a["quotas"], {})
        names = [t["name"] for t in self.registry.list_tenants()]
        self.assertEqual(names, ["acme", "blit"])
        self.assertEqual(self.registry.get_tenant(b["id"])["quotas"], {"max_sandboxes": 1})

    def test_duplicate_name_rejected(self) -> None:
        self.registry.create_tenant("acme")
        with self.assertRaises(Exception):
            self.registry.create_tenant("acme")

    def test_set_quotas(self) -> None:
        tenant = self.registry.create_tenant("acme")
        updated = self.registry.set_quotas(tenant["id"], {"max_sandboxes": 7})
        self.assertEqual(updated["quotas"], {"max_sandboxes": 7})
        with self.assertRaises(KeyError):
            self.registry.set_quotas("tn_missing", {})


class ApiKeyTests(RegistryTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.tenant = self.registry.create_tenant("acme")

    def test_plaintext_shown_once_hash_stored(self) -> None:
        created = self.registry.create_api_key(self.tenant["id"])
        self.assertTrue(created["api_key"].startswith(API_KEY_PREFIX))
        self.assertEqual(created["key_sha256"], hash_api_key(created["api_key"]))
        # The plaintext never appears in the database file.
        raw = self.db_path.read_bytes()
        for sibling in (self.db_path.with_suffix(".sqlite3-wal"),):
            if sibling.exists():
                raw += sibling.read_bytes()
        self.assertNotIn(created["api_key"].encode(), raw)

    def test_resolve_and_revoke(self) -> None:
        created = self.registry.create_api_key(self.tenant["id"])
        self.assertEqual(self.registry.resolve_api_key(created["api_key"]), self.tenant["id"])
        self.assertIsNone(self.registry.resolve_api_key(API_KEY_PREFIX + "0" * 48))
        self.assertTrue(self.registry.revoke_api_key(created["api_key"]))
        self.assertIsNone(self.registry.resolve_api_key(created["api_key"]))

    def test_revoke_by_sha256(self) -> None:
        created = self.registry.create_api_key(self.tenant["id"])
        self.assertTrue(self.registry.revoke_api_key(created["key_sha256"]))
        self.assertIsNone(self.registry.resolve_api_key(created["api_key"]))

    def test_revoke_unknown_returns_false(self) -> None:
        self.assertFalse(self.registry.revoke_api_key("deadbeef"))

    def test_key_for_unknown_tenant_rejected(self) -> None:
        with self.assertRaises(KeyError):
            self.registry.create_api_key("tn_missing")


class TwoPhaseCreateTests(RegistryTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.tenant = self.registry.create_tenant("acme", {"max_sandboxes": 2})

    def test_begin_complete(self) -> None:
        intent = self.registry.begin_create(self.tenant["id"], name="worker")
        self.assertTrue(intent.startswith(PENDING_ID_PREFIX))
        row = self.registry.get_sandbox(intent)
        self.assertEqual(row["status"], "pending")
        self.assertEqual(row["name"], "worker")

        self.registry.complete_create(intent, "sb-1")
        self.assertIsNone(self.registry.get_sandbox(intent))
        row = self.registry.get_sandbox("sb-1")
        self.assertEqual(row["status"], "active")
        self.assertEqual(row["tenant_id"], self.tenant["id"])

    def test_abort_reaps_pending_row(self) -> None:
        intent = self.registry.begin_create(self.tenant["id"])
        self.registry.abort_create(intent)
        self.assertIsNone(self.registry.get_sandbox(intent))
        self.assertEqual(self.registry.live_count(self.tenant["id"]), 0)

    def test_complete_unknown_intent_rejected(self) -> None:
        with self.assertRaises(KeyError):
            self.registry.complete_create(PENDING_ID_PREFIX + "nope", "sb-1")

    def test_pending_rows_count_against_quota(self) -> None:
        self.registry.begin_create(self.tenant["id"])
        self.registry.begin_create(self.tenant["id"])
        with self.assertRaises(QuotaExceeded) as ctx:
            self.registry.begin_create(self.tenant["id"])
        self.assertEqual(ctx.exception.quota["max_sandboxes"], 2)
        self.assertEqual(ctx.exception.quota["live_sandboxes"], 2)

    def test_killed_and_lost_do_not_count(self) -> None:
        self.registry.register_sandbox(self.tenant["id"], "sb-1", status="killed")
        self.registry.register_sandbox(self.tenant["id"], "sb-2", status="lost")
        self.assertEqual(self.registry.live_count(self.tenant["id"]), 0)
        self.registry.begin_create(self.tenant["id"])  # quota 2, still room

    def test_ensure_capacity_for_fork(self) -> None:
        self.registry.register_sandbox(self.tenant["id"], "sb-1")
        self.registry.ensure_capacity(self.tenant["id"], additional=1)
        with self.assertRaises(QuotaExceeded):
            self.registry.ensure_capacity(self.tenant["id"], additional=2)

    def test_unknown_tenant_rejected(self) -> None:
        with self.assertRaises(KeyError):
            self.registry.begin_create("tn_missing")

    def test_concurrent_begin_create_respects_quota(self) -> None:
        tenant = self.registry.create_tenant("burst", {"max_sandboxes": 4})
        outcomes: list[str] = []
        lock = threading.Lock()

        def worker() -> None:
            try:
                self.registry.begin_create(tenant["id"])
                result = "ok"
            except QuotaExceeded:
                result = "quota"
            with lock:
                outcomes.append(result)

        threads = [threading.Thread(target=worker) for _ in range(12)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(outcomes.count("ok"), 4)
        self.assertEqual(outcomes.count("quota"), 8)
        self.assertEqual(self.registry.live_count(tenant["id"]), 4)


class ReconciliationHelperTests(RegistryTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.tenant = self.registry.create_tenant("acme")

    def test_mark_all_active_lost(self) -> None:
        self.registry.register_sandbox(self.tenant["id"], "sb-1")
        self.registry.register_sandbox(self.tenant["id"], "sb-2", status="killed")
        self.assertEqual(self.registry.mark_all_active_lost(), 1)
        self.assertEqual(self.registry.get_sandbox("sb-1")["status"], "lost")
        self.assertEqual(self.registry.get_sandbox("sb-2")["status"], "killed")

    def test_mark_missing_lost(self) -> None:
        self.registry.register_sandbox(self.tenant["id"], "sb-1")
        self.registry.register_sandbox(self.tenant["id"], "sb-2")
        missing = self.registry.mark_missing_lost({"sb-2"})
        self.assertEqual(missing, ["sb-1"])
        self.assertEqual(self.registry.get_sandbox("sb-1")["status"], "lost")
        self.assertEqual(self.registry.get_sandbox("sb-2")["status"], "active")

    def test_pending_rows_and_tracked_ids(self) -> None:
        intent = self.registry.begin_create(self.tenant["id"])
        self.registry.register_sandbox(self.tenant["id"], "sb-1")
        pending = self.registry.pending_rows()
        self.assertEqual([row["sandbox_id"] for row in pending], [intent])
        self.assertEqual(self.registry.all_tracked_ids(), {"sb-1"})

    def test_meta_roundtrip(self) -> None:
        self.assertIsNone(self.registry.get_meta("daemon_boot_id"))
        self.registry.set_meta("daemon_boot_id", "42")
        self.registry.set_meta("daemon_boot_id", "43")
        self.assertEqual(self.registry.get_meta("daemon_boot_id"), "43")


class AggregateQuotaTests(RegistryTestCase):
    """S3 per-tenant aggregate caps: sums over live claims, fork
    multiplication, release on kill, and 409 payload arithmetic."""

    MEM = 512 * 1024 * 1024  # one sandbox's memory claim

    def setUp(self) -> None:
        super().setUp()
        self.tenant = self.registry.create_tenant(
            "metered", {"max_memory_bytes": 2 * self.MEM, "max_cpu": 4}
        )
        self.claim = {"cpus": 2, "memory_bytes": self.MEM}

    def test_within_cap_accepted_and_stored(self) -> None:
        intent = self.registry.begin_create(self.tenant["id"], resources=self.claim)
        self.assertEqual(self.registry.get_sandbox(intent)["resources"], self.claim)
        self.registry.complete_create(intent, "sb-1")
        # complete_create rekeys the row; the claim must survive.
        self.assertEqual(self.registry.get_sandbox("sb-1")["resources"], self.claim)

    def test_memory_cap_blocks_with_arithmetic(self) -> None:
        self.registry.register_sandbox(self.tenant["id"], "sb-1", resources=self.claim)
        self.registry.register_sandbox(
            self.tenant["id"], "sb-2", resources={"memory_bytes": self.MEM}
        )
        with self.assertRaises(QuotaExceeded) as ctx:
            self.registry.begin_create(
                self.tenant["id"], resources={"memory_bytes": 1, "cpus": 1}
            )
        self.assertEqual(
            ctx.exception.quota,
            {
                "max_memory_bytes": 2 * self.MEM,
                "live_memory_bytes": 2 * self.MEM,
                "requested_memory_bytes": 1,
            },
        )

    def test_cpu_cap_blocks_with_arithmetic(self) -> None:
        self.registry.register_sandbox(self.tenant["id"], "sb-1", resources=self.claim)
        with self.assertRaises(QuotaExceeded) as ctx:
            self.registry.begin_create(
                self.tenant["id"], resources={"cpus": 3, "memory_bytes": 1}
            )
        self.assertEqual(
            ctx.exception.quota,
            {"max_cpu": 4, "live_cpu": 2, "requested_cpu": 3},
        )

    def test_capped_tenant_must_declare_limits(self) -> None:
        # An undeclared (unbounded) sandbox would escape the accounting.
        with self.assertRaises(QuotaExceeded) as ctx:
            self.registry.begin_create(self.tenant["id"])
        self.assertEqual(ctx.exception.quota["max_memory_bytes"], 2 * self.MEM)
        self.assertIsNone(ctx.exception.quota["requested_memory_bytes"])
        with self.assertRaises(QuotaExceeded):
            self.registry.begin_create(
                self.tenant["id"], resources={"memory_bytes": self.MEM}
            )  # memory declared, cpus not — the max_cpu gate still refuses

    def test_kill_releases_aggregate_quota(self) -> None:
        self.registry.register_sandbox(self.tenant["id"], "sb-1", resources=self.claim)
        self.registry.register_sandbox(self.tenant["id"], "sb-2", resources=self.claim)
        with self.assertRaises(QuotaExceeded):
            self.registry.begin_create(self.tenant["id"], resources=self.claim)
        self.registry.set_status("sb-1", "killed")
        self.registry.begin_create(self.tenant["id"], resources=self.claim)  # room again

    def test_fork_counts_claim_per_child(self) -> None:
        self.registry.register_sandbox(self.tenant["id"], "sb-src", resources=self.claim)
        # One inherited child fits (memory 2*MEM cap), two do not.
        self.registry.ensure_capacity(self.tenant["id"], additional=1, resources=self.claim)
        with self.assertRaises(QuotaExceeded) as ctx:
            self.registry.ensure_capacity(
                self.tenant["id"], additional=2, resources=self.claim
            )
        self.assertEqual(ctx.exception.quota["requested_memory_bytes"], 2 * self.MEM)

    def test_uncapped_tenant_ignores_resources(self) -> None:
        # Zero-breakage: no aggregate caps -> no gate, declared or not.
        tenant = self.registry.create_tenant("free", {"max_sandboxes": 10})
        self.registry.begin_create(tenant["id"])
        self.registry.begin_create(tenant["id"], resources={"memory_bytes": 1 << 60})
        self.assertEqual(self.registry.live_count(tenant["id"]), 2)

    def test_concurrent_creates_respect_memory_cap(self) -> None:
        # 2*MEM cap, MEM per claim: exactly two of twelve may pass.
        outcomes: list[str] = []
        lock = threading.Lock()

        def worker() -> None:
            try:
                self.registry.begin_create(self.tenant["id"], resources=self.claim)
                result = "ok"
            except QuotaExceeded:
                result = "quota"
            with lock:
                outcomes.append(result)

        threads = [threading.Thread(target=worker) for _ in range(12)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(outcomes.count("ok"), 2)
        self.assertEqual(outcomes.count("quota"), 10)


class ResourcesMigrationTests(unittest.TestCase):
    def test_pre_s3_registry_gains_resources_column(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "gateway.sqlite3"
            # A registry file created by the S1/S2 gateway — no
            # resources_json column.
            conn = sqlite3.connect(str(db_path))
            conn.executescript(
                """
                CREATE TABLE tenants (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    quota_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE TABLE api_keys (
                    key_sha256 TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL REFERENCES tenants(id),
                    created_at TEXT NOT NULL,
                    revoked INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE sandboxes (
                    sandbox_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL REFERENCES tenants(id),
                    name TEXT,
                    created_at TEXT NOT NULL,
                    status TEXT NOT NULL
                );
                CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                INSERT INTO tenants (id, name) VALUES ('tn_old', 'legacy');
                INSERT INTO sandboxes (sandbox_id, tenant_id, name, created_at, status)
                    VALUES ('sb-old', 'tn_old', NULL, '2026-01-01T00:00:00+00:00', 'active');
                """
            )
            conn.commit()
            conn.close()

            registry = GatewayRegistry(db_path)
            try:
                # Pre-S3 rows read as "no limits".
                self.assertEqual(registry.get_sandbox("sb-old")["resources"], {})
                # And the upgraded registry accepts claims.
                registry.register_sandbox(
                    "tn_old", "sb-new", resources={"memory_bytes": 1024}
                )
                self.assertEqual(
                    registry.get_sandbox("sb-new")["resources"], {"memory_bytes": 1024}
                )
            finally:
                registry.close()


if __name__ == "__main__":
    unittest.main()
