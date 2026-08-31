"""Unit tests for the gateway idle auto-reclaim feature (registry idle
columns/methods, timeout parsing, and the sweeper's decision + action
routing). Host-runnable — no daemon, no root."""
from __future__ import annotations

import tempfile
import threading
import types
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from crab.gateway.registry import GatewayRegistry
from crab.gateway.server import (
    DEFAULT_IDLE_ACTION,
    IDLE_ACTIONS,
    GatewayServer,
    _BadRequest,
    _GatewayActivityGate,
    _GatewayRoutes,
    _parse_idle_action,
    _parse_idle_timeout,
)


def _iso(seconds_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)).isoformat(
        timespec="seconds"
    )


class RegistryIdleTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.registry = GatewayRegistry(Path(self._tmp.name) / "gateway.sqlite3")
        self.addCleanup(self.registry.close)
        self.tenant = self.registry.create_tenant("acme", {})

    def test_migration_adds_idle_columns(self) -> None:
        cols = {
            str(row["name"])
            for row in self.registry._conn.execute("PRAGMA table_info(sandboxes)")
        }
        for col in (
            "idle_timeout",
            "idle_action",
            "last_activity",
            "last_idle_action",
            "last_idle_status",
            "last_idle_checkpoint_id",
            "last_idle_reclaim_at",
            "last_idle_error",
        ):
            self.assertIn(col, cols)

    def test_set_idle_policy_and_list(self) -> None:
        self.registry.register_sandbox(self.tenant["id"], "sb-idle")
        self.registry.set_idle_policy("sb-idle", 60.0, "stop")
        rows = self.registry.list_active_with_idle()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["sandbox_id"], "sb-idle")
        self.assertEqual(rows[0]["idle_timeout"], 60.0)
        self.assertEqual(rows[0]["idle_action"], "stop")
        self.assertIsNotNone(rows[0]["last_activity"])

    def test_no_policy_means_not_listed(self) -> None:
        self.registry.register_sandbox(self.tenant["id"], "sb-plain")
        self.assertEqual(self.registry.list_active_with_idle(), [])

    def test_disable_by_setting_none(self) -> None:
        self.registry.register_sandbox(self.tenant["id"], "sb-idle")
        self.registry.set_idle_policy("sb-idle", 60.0, "stop")
        self.registry.set_idle_policy("sb-idle", None)
        self.assertEqual(self.registry.list_active_with_idle(), [])

    def test_set_idle_policy_none_action_preserves_existing(self) -> None:
        self.registry.register_sandbox(self.tenant["id"], "sb-idle")
        self.registry.set_idle_policy("sb-idle", 60.0, "pause")
        # Extend the timer without specifying an action -> keep "pause".
        self.registry.set_idle_policy("sb-idle", 120.0, None)
        rows = self.registry.list_active_with_idle()
        self.assertEqual(rows[0]["idle_timeout"], 120.0)
        self.assertEqual(rows[0]["idle_action"], "pause")

    def test_touch_updates_activity_only_for_idle_sandboxes(self) -> None:
        self.registry.register_sandbox(self.tenant["id"], "sb-idle")
        self.registry.set_idle_policy("sb-idle", 60.0, "stop")
        before = self.registry.list_active_with_idle()[0]["last_activity"]
        self.registry.touch("sb-idle")
        after = self.registry.list_active_with_idle()[0]["last_activity"]
        self.assertGreaterEqual(after, before)
        # A sandbox with no idle policy is left untouched (no error).
        self.registry.register_sandbox(self.tenant["id"], "sb-plain")
        self.registry.touch("sb-plain")

    def test_record_idle_reclaim_is_available_for_api_reads(self) -> None:
        self.registry.register_sandbox(self.tenant["id"], "sb-idle")
        self.registry.set_idle_policy("sb-idle", 60.0, "checkpoint_stop")
        self.registry.record_idle_reclaim(
            "sb-idle",
            action="checkpoint_stop",
            status="succeeded",
            checkpoint_id="ckpt-idle-1",
        )
        row = self.registry.get_idle_policy("sb-idle")
        assert row is not None
        self.assertEqual(row["last_idle_action"], "checkpoint_stop")
        self.assertEqual(row["last_idle_status"], "succeeded")
        self.assertEqual(row["last_idle_checkpoint_id"], "ckpt-idle-1")
        self.assertIsNotNone(row["last_idle_reclaim_at"])
        sweeper_row = self.registry.list_active_with_idle()[0]
        self.assertEqual(sweeper_row["last_idle_action"], "checkpoint_stop")
        self.assertEqual(sweeper_row["last_idle_status"], "succeeded")
        self.assertEqual(sweeper_row["last_idle_checkpoint_id"], "ckpt-idle-1")
        self.assertIsNotNone(sweeper_row["last_idle_reclaim_at"])


class ParseIdleTimeoutTests(unittest.TestCase):
    def test_numeric(self) -> None:
        self.assertEqual(_parse_idle_timeout(60), 60.0)
        self.assertEqual(_parse_idle_timeout(90.5), 90.5)

    def test_numeric_string(self) -> None:
        self.assertEqual(_parse_idle_timeout("120"), 120.0)

    def test_none_and_nonpositive_disable(self) -> None:
        self.assertIsNone(_parse_idle_timeout(None))
        self.assertIsNone(_parse_idle_timeout(0))
        self.assertIsNone(_parse_idle_timeout(-5))

    def test_malformed_is_bad_request(self) -> None:
        with self.assertRaises(_BadRequest):
            _parse_idle_timeout("not-a-number")


class ParseIdleActionTests(unittest.TestCase):
    def test_known_actions_and_none(self) -> None:
        self.assertIsNone(_parse_idle_action(None))
        for action in IDLE_ACTIONS:
            self.assertEqual(_parse_idle_action(action), action)

    def test_unknown_or_non_string_action_is_bad_request(self) -> None:
        for value in ("archive", 1, True):
            with self.subTest(value=value), self.assertRaises(_BadRequest):
                _parse_idle_action(value)


class GatewayActivityGateTests(unittest.TestCase):
    def test_activity_leases_are_shared(self) -> None:
        gate = _GatewayActivityGate()
        first_entered = threading.Event()
        second_entered = threading.Event()
        release_first = threading.Event()

        def first_activity():
            with gate.activity("sb-1"):
                first_entered.set()
                release_first.wait(timeout=2.0)

        def second_activity():
            with gate.activity("sb-1"):
                second_entered.set()

        first_thread = threading.Thread(target=first_activity)
        second_thread = threading.Thread(target=second_activity)
        try:
            first_thread.start()
            self.assertTrue(first_entered.wait(timeout=1.0))
            second_thread.start()
            self.assertTrue(second_entered.wait(timeout=1.0))
        finally:
            release_first.set()
            first_thread.join(timeout=2.0)
            second_thread.join(timeout=2.0)

        self.assertFalse(first_thread.is_alive())
        self.assertFalse(second_thread.is_alive())


class IdleRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.registry = GatewayRegistry(Path(self._tmp.name) / "gateway.sqlite3")
        self.addCleanup(self.registry.close)
        self.tenant = self.registry.create_tenant("acme", {})["id"]
        self.registry.register_sandbox(self.tenant, "source")
        self.registry.set_idle_policy("source", 120.0, "pause")

    def _gateway(self, *, forks=None):
        return types.SimpleNamespace(
            _activity_gate=_GatewayActivityGate(),
            registry=self.registry,
            require_owned=lambda tenant_id, sandbox_id: self.registry.get_sandbox(
                sandbox_id
            ),
            proxy=lambda *args: {
                "ok": True,
                "forks": list(forks or []),
            },
        )

    def test_get_idle_returns_last_checkpoint_id(self) -> None:
        self.registry.record_idle_reclaim(
            "source",
            action="checkpoint_stop",
            status="succeeded",
            checkpoint_id="ckpt-idle-api",
        )
        response = _GatewayRoutes(self._gateway()).get_idle(
            self.tenant, {}, sandbox_id="source"
        )
        self.assertEqual(response["idle_timeout"], 120.0)
        self.assertEqual(response["idle_action"], "pause")
        self.assertEqual(
            response["last_reclaim"]["checkpoint_id"], "ckpt-idle-api"
        )

    def test_fork_inherits_idle_policy_with_fresh_activity_clock(self) -> None:
        source_activity = self.registry.get_idle_policy("source")["last_activity"]
        routes = _GatewayRoutes(
            self._gateway(forks=[{"sandbox_id": "child"}])
        )
        routes.fork_sandbox(self.tenant, {"count": 1}, sandbox_id="source")
        child = self.registry.get_idle_policy("child")
        assert child is not None
        self.assertEqual(child["idle_timeout"], 120.0)
        self.assertEqual(child["idle_action"], "pause")
        self.assertGreaterEqual(child["last_activity"], source_activity)

    def test_fork_refreshes_source_activity_before_and_after_daemon_call(self) -> None:
        old_activity = "2020-01-01T00:00:00+00:00"
        self.registry._conn.execute(
            "UPDATE sandboxes SET last_activity = ? WHERE sandbox_id = ?",
            (old_activity, "source"),
        )
        activity_during_proxy: list[str] = []

        def proxy(*args):
            activity_during_proxy.append(
                self.registry.get_idle_policy("source")["last_activity"]
            )
            return {"ok": True, "forks": [{"sandbox_id": "child"}]}

        gateway = types.SimpleNamespace(
            _activity_gate=_GatewayActivityGate(),
            registry=self.registry,
            require_owned=lambda tenant_id, sandbox_id: self.registry.get_sandbox(
                sandbox_id
            ),
            proxy=proxy,
        )
        _GatewayRoutes(gateway).fork_sandbox(
            self.tenant, {"count": 1}, sandbox_id="source"
        )

        self.assertEqual(len(activity_during_proxy), 1)
        self.assertGreater(activity_during_proxy[0], old_activity)
        self.assertGreaterEqual(
            self.registry.get_idle_policy("source")["last_activity"],
            activity_during_proxy[0],
        )

    def test_reaper_does_not_queue_stop_behind_fork_from_stale_snapshot(self) -> None:
        old_activity = "2020-01-01T00:00:00+00:00"
        self.registry._conn.execute(
            "UPDATE sandboxes SET last_activity = ? WHERE sandbox_id = ?",
            (old_activity, "source"),
        )
        stale_rows = self.registry.list_active_with_idle()
        fork_started = threading.Event()
        release_fork = threading.Event()
        proxy_calls: list[tuple[str, str]] = []
        thread_errors: list[BaseException] = []

        class SnapshotRegistry:
            def list_active_with_idle(inner_self):
                return [dict(row) for row in stale_rows]

            def __getattr__(inner_self, name):
                return getattr(self.registry, name)

        gateway = types.SimpleNamespace(
            _activity_gate=_GatewayActivityGate(),
            registry=SnapshotRegistry(),
            require_owned=lambda tenant_id, sandbox_id: self.registry.get_sandbox(
                sandbox_id
            ),
            _port_manager=_FakePortManager(),
        )

        def proxy(method, path, body, timeout):
            proxy_calls.append((method, path))
            if path.endswith("/fork"):
                fork_started.set()
                if not release_fork.wait(timeout=2.0):
                    raise TimeoutError("test did not release fork")
                return {"ok": True, "forks": []}
            return {"ok": True}

        gateway.proxy = proxy
        gateway._apply_idle_action = GatewayServer._apply_idle_action.__get__(
            gateway, GatewayServer
        )

        def run_fork():
            try:
                _GatewayRoutes(gateway).fork_sandbox(
                    self.tenant, {"count": 1}, sandbox_id="source"
                )
            except BaseException as exc:
                thread_errors.append(exc)

        fork_thread = threading.Thread(target=run_fork)
        try:
            fork_thread.start()
            self.assertTrue(fork_started.wait(timeout=1.0))
            GatewayServer._reap_idle_sandboxes(
                gateway, {"source": "running"}
            )
        finally:
            release_fork.set()
            fork_thread.join(timeout=2.0)

        self.assertFalse(fork_thread.is_alive())
        self.assertEqual(thread_errors, [])
        self.assertNotIn(("POST", "/sandboxes/source/stop"), proxy_calls)


class _FakeRegistry:
    def __init__(self, rows) -> None:
        self._rows = rows
        self._current = {str(row["sandbox_id"]): dict(row) for row in rows}
        self.statuses: dict[str, str] = {}
        self.reclaims: list[dict] = []

    def list_active_with_idle(self):
        return self._rows

    def get_sandbox(self, sandbox_id):
        row = self._current.get(sandbox_id)
        return None if row is None else dict(row)

    def touch(self, sandbox_id):
        row = self._current.get(sandbox_id)
        if row is not None and row.get("idle_timeout") is not None:
            row["last_activity"] = _iso(0)

    def release_all_ports(self, sandbox_id):
        return []

    def set_status(self, sandbox_id, status):
        self.statuses[sandbox_id] = status
        if sandbox_id in self._current:
            self._current[sandbox_id]["status"] = status

    def record_idle_reclaim(self, sandbox_id, **result):
        self.reclaims.append({"sandbox_id": sandbox_id, **result})


class _FakePortManager:
    def release_all(self, sandbox_id, ports):
        pass


def _make_fake_gateway(
    rows, *, checkpoint_result=None, checkpoint_exception: Exception | None = None
):
    gw = types.SimpleNamespace()
    gw._activity_gate = _GatewayActivityGate()
    gw.registry = _FakeRegistry(rows)
    gw.proxy_calls = []
    gw.proxy_bodies = []
    gw.reclaims_at_proxy = []

    def proxy(method, path, body, timeout):
        gw.proxy_calls.append((method, path))
        gw.proxy_bodies.append((method, path, body))
        gw.reclaims_at_proxy.append(
            [dict(entry) for entry in gw.registry.reclaims]
        )
        if path.endswith("/checkpoint-stop"):
            if checkpoint_exception is not None:
                raise checkpoint_exception
            if checkpoint_result is not None:
                result = dict(checkpoint_result)
                result.setdefault("checkpoint_id", body["checkpoint_id"])
                return result
            return {
                "ok": True,
                "status": "succeeded",
                "checkpoint_id": body["checkpoint_id"],
                "stopped": True,
            }
        return {"ok": True}

    gw.proxy = proxy
    gw._port_manager = _FakePortManager()
    gw._apply_idle_action = GatewayServer._apply_idle_action.__get__(gw, GatewayServer)
    return gw


def _row(sid, timeout, action, last_activity):
    return {
        "sandbox_id": sid,
        "tenant_id": "t1",
        "status": "active",
        "created_at": "2020-01-01T00:00:00+00:00",
        "last_activity": last_activity,
        "idle_timeout": timeout,
        "idle_action": action,
        "last_idle_action": None,
        "last_idle_status": None,
        "last_idle_checkpoint_id": None,
        "last_idle_reclaim_at": None,
    }


class IdleSweeperTests(unittest.TestCase):
    def test_reaps_running_idle_sandbox_with_stop(self) -> None:
        gw = _make_fake_gateway([_row("sb-1", 60.0, "stop", _iso(3600))])
        GatewayServer._reap_idle_sandboxes(gw, {"sb-1": "running"})
        self.assertIn(("POST", "/sandboxes/sb-1/stop"), gw.proxy_calls)

    def test_skips_sandbox_within_idle_window(self) -> None:
        gw = _make_fake_gateway([_row("sb-1", 3600.0, "stop", _iso(10))])
        GatewayServer._reap_idle_sandboxes(gw, {"sb-1": "running"})
        self.assertEqual(gw.proxy_calls, [])

    def test_revalidates_stale_idle_snapshot_before_reclaim(self) -> None:
        gw = _make_fake_gateway([_row("sb-1", 60.0, "stop", _iso(3600))])
        gw.registry.touch("sb-1")

        GatewayServer._reap_idle_sandboxes(gw, {"sb-1": "running"})

        self.assertEqual(gw.proxy_calls, [])
        self.assertEqual(gw.registry.reclaims, [])

    def test_busy_activity_is_skipped_without_blocking_reaper(self) -> None:
        gw = _make_fake_gateway([_row("sb-1", 60.0, "stop", _iso(3600))])

        with gw._activity_gate.activity("sb-1"):
            GatewayServer._reap_idle_sandboxes(gw, {"sb-1": "running"})

        self.assertEqual(gw.proxy_calls, [])
        self.assertEqual(gw.registry.reclaims, [])

    def test_skips_sandbox_not_running(self) -> None:
        # Already paused/stopped -> not re-reaped.
        for status in ("paused", "stopped"):
            gw = _make_fake_gateway([_row("sb-1", 60.0, "stop", _iso(3600))])
            GatewayServer._reap_idle_sandboxes(gw, {"sb-1": status})
            self.assertEqual(gw.proxy_calls, [])

    def test_falls_back_to_created_at_without_last_activity(self) -> None:
        row = _row("sb-1", 60.0, "stop", None)
        gw = _make_fake_gateway([row])
        GatewayServer._reap_idle_sandboxes(gw, {"sb-1": "running"})
        self.assertIn(("POST", "/sandboxes/sb-1/stop"), gw.proxy_calls)

    def test_default_action_is_stop(self) -> None:
        self.assertEqual(DEFAULT_IDLE_ACTION, "stop")

    def test_pause_action_routes_pause(self) -> None:
        gw = _make_fake_gateway([_row("sb-1", 60.0, "pause", _iso(3600))])
        GatewayServer._reap_idle_sandboxes(gw, {"sb-1": "running"})
        self.assertIn(("POST", "/sandboxes/sb-1/pause"), gw.proxy_calls)

    def test_kill_action_routes_delete_and_marks_killed(self) -> None:
        gw = _make_fake_gateway([_row("sb-1", 60.0, "kill", _iso(3600))])
        GatewayServer._reap_idle_sandboxes(gw, {"sb-1": "running"})
        self.assertIn(("DELETE", "/sandboxes/sb-1"), gw.proxy_calls)
        self.assertEqual(gw.registry.statuses.get("sb-1"), "killed")

    def test_checkpoint_stop_checkpoints_then_stops(self) -> None:
        gw = _make_fake_gateway([_row("sb-1", 60.0, "checkpoint_stop", _iso(3600))])
        GatewayServer._reap_idle_sandboxes(gw, {"sb-1": "running"})
        self.assertEqual(
            gw.proxy_calls,
            [("POST", "/sandboxes/sb-1/checkpoint-stop")],
        )
        allocated_id = gw.registry.reclaims[0]["checkpoint_id"]
        self.assertEqual(gw.registry.reclaims[0]["status"], "in_progress")
        self.assertTrue(allocated_id.startswith("ckpt-"))
        self.assertEqual(
            gw.reclaims_at_proxy[0][-1],
            {
                "sandbox_id": "sb-1",
                "action": "checkpoint_stop",
                "status": "in_progress",
                "checkpoint_id": allocated_id,
            },
        )
        self.assertEqual(
            gw.proxy_bodies,
            [
                (
                    "POST",
                    "/sandboxes/sb-1/checkpoint-stop",
                    {"checkpoint_id": allocated_id},
                )
            ],
        )
        self.assertEqual(gw.registry.reclaims[-1]["status"], "succeeded")
        self.assertEqual(gw.registry.reclaims[-1]["checkpoint_id"], allocated_id)

    def test_failed_checkpoint_stop_is_recorded_and_not_followed_by_stop(self) -> None:
        gw = _make_fake_gateway(
            [_row("sb-1", 60.0, "checkpoint_stop", _iso(3600))],
            checkpoint_result={
                "ok": False,
                "status": "failed",
                "stopped": False,
                "message": "checkpoint rejected",
            },
        )
        GatewayServer._reap_idle_sandboxes(gw, {"sb-1": "running"})
        self.assertEqual(
            gw.proxy_calls,
            [("POST", "/sandboxes/sb-1/checkpoint-stop")],
        )
        self.assertEqual(gw.registry.reclaims[-1]["status"], "failed")
        self.assertEqual(
            gw.registry.reclaims[-1]["checkpoint_id"],
            gw.registry.reclaims[0]["checkpoint_id"],
        )
        self.assertEqual(
            gw.registry.reclaims[-1]["error"], "checkpoint rejected"
        )

    def test_checkpoint_stop_transport_failure_preserves_preallocated_id(self) -> None:
        gw = _make_fake_gateway(
            [_row("sb-1", 60.0, "checkpoint_stop", _iso(3600))],
            checkpoint_exception=TimeoutError("response lost"),
        )
        GatewayServer._reap_idle_sandboxes(gw, {"sb-1": "running"})

        self.assertEqual(
            [entry["status"] for entry in gw.registry.reclaims],
            ["in_progress", "failed"],
        )
        allocated_id = gw.registry.reclaims[0]["checkpoint_id"]
        self.assertEqual(gw.registry.reclaims[1]["checkpoint_id"], allocated_id)
        self.assertEqual(
            gw.proxy_bodies[-1][2], {"checkpoint_id": allocated_id}
        )

    def test_checkpoint_stop_rejects_unexpected_daemon_id(self) -> None:
        gw = _make_fake_gateway(
            [_row("sb-1", 60.0, "checkpoint_stop", _iso(3600))],
            checkpoint_result={
                "ok": True,
                "status": "succeeded",
                "checkpoint_id": "ckpt-unexpected",
                "stopped": True,
            },
        )
        GatewayServer._reap_idle_sandboxes(gw, {"sb-1": "running"})

        allocated_id = gw.registry.reclaims[0]["checkpoint_id"]
        self.assertEqual(gw.registry.reclaims[-1]["status"], "failed")
        self.assertEqual(gw.registry.reclaims[-1]["checkpoint_id"], allocated_id)

    def test_checkpoint_stop_retry_reuses_durable_id_without_new_activity(self) -> None:
        row = _row("sb-1", 60.0, "checkpoint_stop", _iso(3600))
        row.update(
            last_idle_action="checkpoint_stop",
            last_idle_status="failed",
            last_idle_checkpoint_id="ckpt-durable",
            last_idle_reclaim_at=_iso(120),
        )
        gw = _make_fake_gateway([row])
        GatewayServer._reap_idle_sandboxes(gw, {"sb-1": "running"})

        self.assertEqual(
            gw.proxy_bodies[-1][2], {"checkpoint_id": "ckpt-durable"}
        )
        self.assertTrue(
            all(
                entry["checkpoint_id"] == "ckpt-durable"
                for entry in gw.registry.reclaims
            )
        )

    def test_checkpoint_stop_after_new_activity_allocates_a_new_id(self) -> None:
        row = _row("sb-1", 60.0, "checkpoint_stop", _iso(3600))
        row.update(
            last_idle_action="checkpoint_stop",
            last_idle_status="failed",
            last_idle_checkpoint_id="ckpt-prior-episode",
            last_idle_reclaim_at=_iso(7200),
        )
        gw = _make_fake_gateway([row])
        GatewayServer._reap_idle_sandboxes(gw, {"sb-1": "running"})

        allocated_id = gw.registry.reclaims[0]["checkpoint_id"]
        self.assertNotEqual(allocated_id, "ckpt-prior-episode")
        self.assertEqual(gw.proxy_bodies[-1][2], {"checkpoint_id": allocated_id})

    def test_idle_actions_constant(self) -> None:
        self.assertEqual(
            set(IDLE_ACTIONS), {"pause", "stop", "checkpoint_stop", "kill"}
        )


if __name__ == "__main__":
    unittest.main()
