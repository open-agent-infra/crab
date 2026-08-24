"""Unit tests for the daemon-mode fork surface (PR-A3.2): the daemon
route/handler, the RemoteEngine proxy, remote `Sandbox.fork`, and the
`crab sandbox fork` CLI command. Host-runnable — no runc/CRIU/zfs."""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from crab.daemon.server import _build_handler, _BadRequest, _Routes
from crab.daemon.transport import DaemonClient, DaemonRequestError, serve_unix_socket
from crab.ids import SandboxId
from crab.remote_engine import RemoteEngine
from crab.sandbox import Sandbox


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeInspector:
    def __init__(self, log: list) -> None:
        self._log = log

    def upsert_snapshot(self, snapshot) -> None:
        self._log.append(("upsert_snapshot", str(snapshot.sandbox_id), snapshot.is_running))


class _FakeSystem:
    def __init__(self, log: list) -> None:
        self._log = log
        self.inspector = _FakeInspector(log)

    def release_txn(self, sandbox_id) -> None:
        self._log.append(("release_txn", str(sandbox_id)))

    def prepare_source_destroy(self, sandbox_id) -> None:
        self._log.append(("prepare_source_destroy", str(sandbox_id)))

    def release_fork(self, sandbox_id) -> None:
        self._log.append(("release_fork", str(sandbox_id)))


class _FakeRuntime:
    name = "runc"

    def __init__(self, log: list) -> None:
        self._log = log

    def stop(self, sandbox_id) -> None:
        self._log.append(("stop", str(sandbox_id)))

    def delete(self, sandbox_id) -> None:
        self._log.append(("delete", str(sandbox_id)))


class _FakeEngine:
    def __init__(self) -> None:
        self.log: list = []
        self.system = _FakeSystem(self.log)
        self.runtime = _FakeRuntime(self.log)
        self.fork_calls: list[dict] = []
        self.fork_error: Exception | None = None

    def fork_sandbox(self, source_sandbox_id, *, count=1, lazy=False, effects=None):
        self.fork_calls.append(
            {
                "source": str(source_sandbox_id),
                "count": count,
                "lazy": lazy,
                "effects": effects,
            }
        )
        if self.fork_error is not None:
            raise self.fork_error
        return [
            SandboxId(f"{source_sandbox_id}-fork-{index}") for index in range(count)
        ]

    def unregister_upstream(self, sandbox_id) -> None:
        self.log.append(("unregister_upstream", str(sandbox_id)))

    def release_network_lease(self, sandbox_id) -> None:
        self.log.append(("release_network_lease", str(sandbox_id)))


class _FakeDaemon:
    def __init__(self, engine: _FakeEngine) -> None:
        self.engine = engine
        self.registered: list[str] = []
        self.unregistered: list[str] = []

    def require_engine(self) -> _FakeEngine:
        return self.engine

    def register_sandbox(self, sandbox_id) -> None:
        self.registered.append(str(sandbox_id))

    def unregister_sandbox(self, sandbox_id) -> None:
        self.unregistered.append(str(sandbox_id))


# ---------------------------------------------------------------------------
# Daemon handler
# ---------------------------------------------------------------------------


class ForkRouteHandlerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = _FakeEngine()
        self.daemon = _FakeDaemon(self.engine)
        self.routes = _Routes(self.daemon)

    def test_fork_defaults_to_single_eager_fork(self) -> None:
        response = self.routes.fork_sandbox({}, sandbox_id="src")
        self.assertTrue(response["ok"])
        self.assertEqual(response["forks"], [{"sandbox_id": "src-fork-0"}])
        self.assertEqual(
            self.engine.fork_calls,
            [{"source": "src", "count": 1, "lazy": False, "effects": None}],
        )

    def test_fork_propagates_count_and_lazy_and_registers(self) -> None:
        response = self.routes.fork_sandbox(
            {"count": 2, "lazy": True}, sandbox_id="src"
        )
        self.assertEqual(
            [entry["sandbox_id"] for entry in response["forks"]],
            ["src-fork-0", "src-fork-1"],
        )
        self.assertEqual(
            self.engine.fork_calls,
            [{"source": "src", "count": 2, "lazy": True, "effects": None}],
        )
        # Forks land in the daemon registry so /sandboxes lists them and
        # daemon shutdown tears them down.
        self.assertEqual(self.daemon.registered, ["src-fork-0", "src-fork-1"])
        # The daemon seeds its own engine's inspector for each fork
        # (the SDK-side seeding is a no-op shim in daemon mode).
        self.assertIn(("upsert_snapshot", "src-fork-0", True), self.engine.log)
        self.assertIn(("upsert_snapshot", "src-fork-1", True), self.engine.log)

    def test_fork_propagates_the_effect_policy(self) -> None:
        # F1: the policy has to reach the daemon-side engine, which is where
        # it is validated and armed before the fork's processes restore.
        self.routes.fork_sandbox({"effects": "reject"}, sandbox_id="src")
        self.assertEqual(
            self.engine.fork_calls,
            [{"source": "src", "count": 1, "lazy": False, "effects": "reject"}],
        )

    def test_fork_rejects_a_non_string_effect_policy(self) -> None:
        with self.assertRaises(_BadRequest):
            self.routes.fork_sandbox({"effects": 7}, sandbox_id="src")
        self.assertEqual(self.engine.fork_calls, [])


    def test_fork_rejects_bad_count(self) -> None:
        with self.assertRaises(_BadRequest):
            self.routes.fork_sandbox({"count": 0}, sandbox_id="src")
        with self.assertRaises(_BadRequest):
            self.routes.fork_sandbox({"count": "nope"}, sandbox_id="src")
        self.assertEqual(self.engine.fork_calls, [])

    def test_fork_translates_engine_failure_to_bad_request(self) -> None:
        self.engine.fork_error = RuntimeError("restore failed")
        with self.assertRaises(_BadRequest):
            self.routes.fork_sandbox({}, sandbox_id="src")
        self.assertEqual(self.daemon.registered, [])

    def test_kill_runs_fork_bookkeeping_before_delete(self) -> None:
        self.routes.kill_sandbox({}, sandbox_id="src")
        self.assertEqual(
            self.engine.log[:4],
            [
                ("release_txn", "src"),
                ("prepare_source_destroy", "src"),
                ("release_fork", "src"),
                ("stop", "src"),
            ],
        )
        self.assertIn(("delete", "src"), self.engine.log)
        self.assertLess(
            self.engine.log.index(("release_fork", "src")),
            self.engine.log.index(("delete", "src")),
        )


class ForkRouteDispatchTests(unittest.TestCase):
    """The fork route is reachable over the real Unix-socket HTTP stack."""

    def setUp(self) -> None:
        self.engine = _FakeEngine()
        self.daemon = _FakeDaemon(self.engine)
        self._tmp = tempfile.TemporaryDirectory(prefix="crab_forkd_")
        self.addCleanup(self._tmp.cleanup)
        socket_path = Path(self._tmp.name) / "crab.sock"
        self.server = serve_unix_socket(socket_path, _build_handler(self.daemon))
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.client = DaemonClient(socket_path, timeout_seconds=10.0)

    def test_post_fork_round_trip(self) -> None:
        response = self.client.post_json(
            "/sandboxes/src/fork", {"count": 2, "lazy": False}
        )
        self.assertTrue(response["ok"])
        self.assertEqual(
            [entry["sandbox_id"] for entry in response["forks"]],
            ["src-fork-0", "src-fork-1"],
        )

    def test_post_fork_bad_count_is_400(self) -> None:
        with self.assertRaises(DaemonRequestError) as ctx:
            self.client.post_json("/sandboxes/src/fork", {"count": 0})
        self.assertEqual(ctx.exception.status_code, 400)


# ---------------------------------------------------------------------------
# RemoteEngine proxy + remote Sandbox.fork
# ---------------------------------------------------------------------------


class _FakeDaemonClient:
    """Just enough of DaemonClient for RemoteEngine unit tests."""

    def __init__(self, response: dict) -> None:
        self.response = response
        self.requests: list[dict] = []

    def post_json(self, path, payload=None, *, timeout_seconds=None):
        self.requests.append(
            {"path": path, "payload": payload, "timeout_seconds": timeout_seconds}
        )
        return dict(self.response)


class RemoteEngineForkTests(unittest.TestCase):
    _INFO = {"runtime": "runc", "default_image": "ubuntu:22.04"}

    def test_fork_sandbox_posts_and_parses_ids(self) -> None:
        client = _FakeDaemonClient(
            {
                "ok": True,
                "forks": [
                    {"sandbox_id": "src-fork-a"},
                    {"sandbox_id": "src-fork-b"},
                ],
            }
        )
        engine = RemoteEngine(client, info=self._INFO)
        fork_ids = engine.fork_sandbox(SandboxId("src"), count=2, lazy=True)
        self.assertEqual([str(fid) for fid in fork_ids], ["src-fork-a", "src-fork-b"])
        self.assertEqual(len(client.requests), 1)
        request = client.requests[0]
        self.assertEqual(request["path"], "/sandboxes/src/fork")
        self.assertEqual(request["payload"], {"count": 2, "lazy": True})
        # Budget scales with count, mirroring checkpoint/restore calls.
        self.assertEqual(request["timeout_seconds"], 600.0)

    def test_fork_sandbox_rejects_bad_count_locally(self) -> None:
        client = _FakeDaemonClient({"ok": True, "forks": []})
        engine = RemoteEngine(client, info=self._INFO)
        with self.assertRaises(ValueError):
            engine.fork_sandbox(SandboxId("src"), count=0)
        self.assertEqual(client.requests, [])

    def test_fork_sandbox_sends_the_effect_policy_only_when_given(self) -> None:
        # F1 over RPC. Omitting `effects` must not add the key at all, so a
        # newer SDK keeps working against a daemon that predates it.
        client = _FakeDaemonClient(
            {"ok": True, "forks": [{"sandbox_id": "src-fork-a"}]}
        )
        engine = RemoteEngine(client, info=self._INFO)

        engine.fork_sandbox(SandboxId("src"), count=1)
        self.assertEqual(client.requests[-1]["payload"], {"count": 1, "lazy": False})

        engine.fork_sandbox(SandboxId("src"), count=1, effects="reject")
        self.assertEqual(
            client.requests[-1]["payload"],
            {"count": 1, "lazy": False, "effects": "reject"},
        )

    def test_fork_sandbox_refuses_effects_when_the_caller_owns_the_window(self) -> None:
        # The fork-txn path arms its own session; a standalone policy on top
        # of it would be two windows on one sandbox.
        client = _FakeDaemonClient({"ok": True, "forks": []})
        engine = RemoteEngine(client, info=self._INFO)
        with self.assertRaises(ValueError):
            engine.fork_sandbox(
                SandboxId("src"), count=1, effects="reject", gate_effects=False
            )
        self.assertEqual(client.requests, [])

    def test_sandbox_fork_is_transport_agnostic(self) -> None:
        client = _FakeDaemonClient(
            {"ok": True, "forks": [{"sandbox_id": "src-fork-a"}]}
        )
        engine = RemoteEngine(client, info=self._INFO)
        source = Sandbox.connect("src", engine=engine)
        forks = source.fork(1)
        self.assertEqual(len(forks), 1)
        self.assertIsInstance(forks[0], Sandbox)
        self.assertEqual(str(forks[0].sandbox_id), "src-fork-a")
        self.assertEqual(client.requests[0]["path"], "/sandboxes/src/fork")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class CliForkTests(unittest.TestCase):
    def _run_cli(self, argv: list[str], response: dict) -> tuple[int, str, list]:
        requests: list[dict] = []

        class _CliClient:
            def __init__(self, socket_path, *, timeout_seconds):
                requests.append({"socket": str(socket_path), "timeout": timeout_seconds})

            def post_json(self, path, payload=None, *, timeout_seconds=None):
                requests.append({"path": path, "payload": payload})
                return dict(response)

        stdout = io.StringIO()
        from crab.cli import commands

        with mock.patch.object(commands, "DaemonClient", _CliClient):
            with contextlib.redirect_stdout(stdout):
                rc = commands.main(argv)
        return rc, stdout.getvalue(), requests

    def test_sandbox_fork_prints_one_id_per_line(self) -> None:
        rc, out, requests = self._run_cli(
            ["sandbox", "fork", "sbx-1", "-n", "2", "--lazy"],
            {
                "ok": True,
                "forks": [
                    {"sandbox_id": "sbx-1-fork-a"},
                    {"sandbox_id": "sbx-1-fork-b"},
                ],
            },
        )
        self.assertEqual(rc, 0)
        self.assertEqual(out.splitlines(), ["sbx-1-fork-a", "sbx-1-fork-b"])
        self.assertEqual(requests[-1]["path"], "/sandboxes/sbx-1/fork")
        self.assertEqual(requests[-1]["payload"], {"count": 2, "lazy": True})
        # HTTP budget scales with count.
        self.assertEqual(requests[0]["timeout"], 600.0)

    def test_sandbox_fork_json_output(self) -> None:
        rc, out, _ = self._run_cli(
            ["--json", "sandbox", "fork", "sbx-1"],
            {"ok": True, "forks": [{"sandbox_id": "sbx-1-fork-a"}]},
        )
        self.assertEqual(rc, 0)
        self.assertEqual(json.loads(out), [{"sandbox_id": "sbx-1-fork-a"}])

    def test_sandbox_fork_rejects_bad_count_without_daemon_call(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            rc, _, requests = self._run_cli(
                ["sandbox", "fork", "sbx-1", "-n", "0"],
                {"ok": True, "forks": []},
            )
        self.assertEqual(rc, 2)
        self.assertEqual(requests, [])


if __name__ == "__main__":
    unittest.main()
