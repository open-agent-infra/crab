"""Unit tests for the daemon-mode transaction surface (PR-B2.2): daemon
routes/handlers with error mapping, the _SystemShim proxies, remote
Sandbox.begin, and the `crab txn` CLI. Host-runnable — no runc."""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from crab.daemon.server import _build_handler, _Routes, _TxnConflict
from crab.daemon.transport import DaemonClient, DaemonRequestError, serve_unix_socket
from crab.ids import SandboxId
from crab.models import utc_now
from crab.remote_engine import RemoteEngine
from crab.sandbox import Sandbox
from crab.txn import (
    TxnAbortError,
    TxnAbortResult,
    TxnActiveError,
    TxnCommitResult,
    TxnDescription,
    TxnMismatchError,
)


def _description(sandbox_id: str = "src", txn_id: str = "txn-1") -> TxnDescription:
    return TxnDescription(
        txn_id=txn_id,
        sandbox_id=sandbox_id,
        base_checkpoint_id="ckpt-base",
        base_was_fresh=True,
        started_at=utc_now().isoformat(),
        label="demo",
    )


class _FakeSystem:
    def __init__(self) -> None:
        self.calls: list = []
        self.begin_error: Exception | None = None
        self.commit_error: Exception | None = None
        self.abort_error: Exception | None = None
        self.current: TxnDescription | None = None

    def begin_txn(self, sandbox_id, *, label=None, isolation="snapshot"):
        self.calls.append(("begin_txn", str(sandbox_id), label, isolation))
        if self.begin_error is not None:
            raise self.begin_error
        return _description(str(sandbox_id))

    def commit_txn(self, sandbox_id, txn_id, *, force=False):
        self.calls.append(("commit_txn", str(sandbox_id), txn_id, force))
        if self.commit_error is not None:
            raise self.commit_error
        return TxnCommitResult(txn_id=str(txn_id), released_observations=2, base_dropped=True)

    def abort_txn(self, sandbox_id, txn_id):
        self.calls.append(("abort_txn", str(sandbox_id), txn_id))
        if self.abort_error is not None:
            raise self.abort_error
        return TxnAbortResult(
            txn_id=str(txn_id), discarded_observations=1, restored_checkpoint_id="ckpt-base"
        )

    def current_txn(self, sandbox_id):
        self.calls.append(("current_txn", str(sandbox_id)))
        return self.current

    def release_txn(self, sandbox_id):
        self.calls.append(("release_txn", str(sandbox_id)))

    def prepare_source_destroy(self, sandbox_id):
        self.calls.append(("prepare_source_destroy", str(sandbox_id)))

    def release_fork(self, sandbox_id):
        self.calls.append(("release_fork", str(sandbox_id)))


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
        self.system = _FakeSystem()
        self.runtime = _FakeRuntime(self.system.calls)

    def unregister_upstream(self, sandbox_id) -> None:
        self.system.calls.append(("unregister_upstream", str(sandbox_id)))

    def release_network_lease(self, sandbox_id) -> None:
        self.system.calls.append(("release_network_lease", str(sandbox_id)))


class _FakeDaemon:
    def __init__(self, engine: _FakeEngine) -> None:
        self.engine = engine

    def require_engine(self) -> _FakeEngine:
        return self.engine

    def register_sandbox(self, sandbox_id) -> None:
        pass

    def unregister_sandbox(self, sandbox_id) -> None:
        pass


class TxnRouteHandlerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = _FakeEngine()
        self.routes = _Routes(_FakeDaemon(self.engine))

    def test_begin_serializes_description(self) -> None:
        response = self.routes.begin_txn({"label": "demo"}, sandbox_id="src")
        self.assertTrue(response["ok"])
        txn = response["txn"]
        self.assertEqual(txn["txn_id"], "txn-1")
        self.assertEqual(txn["base_checkpoint_id"], "ckpt-base")
        self.assertTrue(txn["base_was_fresh"])
        self.assertEqual(txn["label"], "demo")
        self.assertEqual(txn["isolation"], "snapshot")
        self.assertIn(("begin_txn", "src", "demo", "snapshot"), self.engine.system.calls)

    def test_begin_active_maps_to_conflict(self) -> None:
        self.engine.system.begin_error = TxnActiveError("already active")
        with self.assertRaises(_TxnConflict) as ctx:
            self.routes.begin_txn({}, sandbox_id="src")
        self.assertEqual(ctx.exception.error_type, "txn_active")

    def test_commit_and_abort_results(self) -> None:
        commit = self.routes.commit_txn({}, sandbox_id="src", txn_id="txn-1")
        self.assertEqual(commit["result"]["released_observations"], 2)
        self.assertTrue(commit["result"]["base_dropped"])
        abort = self.routes.abort_txn({}, sandbox_id="src", txn_id="txn-1")
        self.assertEqual(abort["result"]["discarded_observations"], 1)
        self.assertEqual(abort["result"]["restored_checkpoint_id"], "ckpt-base")

    def test_mismatch_and_abort_failure_map_to_conflict(self) -> None:
        self.engine.system.commit_error = TxnMismatchError("nope")
        with self.assertRaises(_TxnConflict) as ctx:
            self.routes.commit_txn({}, sandbox_id="src", txn_id="txn-x")
        self.assertEqual(ctx.exception.error_type, "txn_mismatch")
        self.engine.system.abort_error = TxnAbortError("restore failed")
        with self.assertRaises(_TxnConflict) as ctx:
            self.routes.abort_txn({}, sandbox_id="src", txn_id="txn-1")
        self.assertEqual(ctx.exception.error_type, "txn_abort_failed")

    def test_current_txn_none_and_present(self) -> None:
        self.assertIsNone(self.routes.current_txn({}, sandbox_id="src")["txn"])
        self.engine.system.current = _description("src", "txn-live")
        self.assertEqual(
            self.routes.current_txn({}, sandbox_id="src")["txn"]["txn_id"], "txn-live"
        )

    def test_kill_releases_txn_before_delete(self) -> None:
        self.routes.kill_sandbox({}, sandbox_id="src")
        calls = self.engine.system.calls
        self.assertLess(
            calls.index(("release_txn", "src")), calls.index(("delete", "src"))
        )


class TxnRouteDispatchTests(unittest.TestCase):
    """Txn routes over the real Unix-socket HTTP stack, incl. 409 mapping."""

    def setUp(self) -> None:
        self.engine = _FakeEngine()
        self._tmp = tempfile.TemporaryDirectory(prefix="crab_txnd_")
        self.addCleanup(self._tmp.cleanup)
        socket_path = Path(self._tmp.name) / "crab.sock"
        self.server = serve_unix_socket(socket_path, _build_handler(_FakeDaemon(self.engine)))
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.client = DaemonClient(socket_path, timeout_seconds=10.0)

    def test_txn_round_trip(self) -> None:
        begin = self.client.post_json("/sandboxes/src/txn", {"label": "x"})
        self.assertEqual(begin["txn"]["txn_id"], "txn-1")
        current = self.client.get_json("/sandboxes/src/txn")
        self.assertIsNone(current["txn"])
        commit = self.client.post_json("/sandboxes/src/txn/txn-1/commit", {})
        self.assertTrue(commit["result"]["base_dropped"])
        abort = self.client.post_json("/sandboxes/src/txn/txn-1/abort", {})
        self.assertEqual(abort["result"]["discarded_observations"], 1)

    def test_conflict_surfaces_409_with_error_type(self) -> None:
        self.engine.system.begin_error = TxnActiveError("busy")
        with self.assertRaises(DaemonRequestError) as ctx:
            self.client.post_json("/sandboxes/src/txn", {})
        self.assertEqual(ctx.exception.status_code, 409)
        payload = json.loads(ctx.exception.body.decode("utf-8"))
        self.assertEqual(payload["error_type"], "txn_active")


# ---------------------------------------------------------------------------
# _SystemShim proxies + remote Sandbox.begin
# ---------------------------------------------------------------------------


class _FakeDaemonClient:
    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.responses: dict[str, object] = {}

    def post_json(self, path, payload=None, *, timeout_seconds=None):
        self.requests.append({"method": "POST", "path": path, "payload": payload, "timeout": timeout_seconds})
        response = self.responses.get(path)
        if isinstance(response, Exception):
            raise response
        return response or {"ok": True}

    def get_json(self, path, *, timeout_seconds=None):
        self.requests.append({"method": "GET", "path": path, "timeout": timeout_seconds})
        response = self.responses.get(path)
        if isinstance(response, Exception):
            raise response
        return response or {"ok": True}


def _conflict(path: str, error_type: str) -> DaemonRequestError:
    body = json.dumps({"ok": False, "error": "boom", "error_type": error_type}).encode("utf-8")
    return DaemonRequestError(409, path, body)


class SystemShimTxnTests(unittest.TestCase):
    _INFO = {"runtime": "runc", "default_image": "ubuntu:22.04"}

    def _engine(self) -> tuple[RemoteEngine, _FakeDaemonClient]:
        client = _FakeDaemonClient()
        return RemoteEngine(client, info=self._INFO), client

    def test_begin_posts_and_deserializes(self) -> None:
        engine, client = self._engine()
        client.responses["/sandboxes/src/txn"] = {
            "ok": True,
            "txn": {
                "txn_id": "txn-9",
                "sandbox_id": "src",
                "base_checkpoint_id": "ckpt-9",
                "base_was_fresh": False,
                "started_at": "2026-01-01T00:00:00+00:00",
                "label": None,
            },
        }
        description = engine.system.begin_txn(SandboxId("src"), label=None)
        self.assertIsInstance(description, TxnDescription)
        self.assertEqual(description.txn_id, "txn-9")
        self.assertFalse(description.base_was_fresh)
        request = client.requests[0]
        self.assertEqual(request["path"], "/sandboxes/src/txn")
        self.assertEqual(request["timeout"], 300.0)

    def test_commit_abort_results_and_timeouts(self) -> None:
        engine, client = self._engine()
        client.responses["/sandboxes/src/txn/txn-9/commit"] = {
            "ok": True,
            "result": {"txn_id": "txn-9", "released_observations": 3, "base_dropped": False},
        }
        client.responses["/sandboxes/src/txn/txn-9/abort"] = {
            "ok": True,
            "result": {
                "txn_id": "txn-9",
                "discarded_observations": 2,
                "restored_checkpoint_id": "ckpt-9",
            },
        }
        commit = engine.system.commit_txn(SandboxId("src"), "txn-9")
        self.assertEqual(commit.released_observations, 3)
        abort = engine.system.abort_txn(SandboxId("src"), "txn-9")
        self.assertEqual(abort.discarded_observations, 2)
        # Commit budgets for the heavier fork-backed swap since B3.
        self.assertEqual(client.requests[0]["timeout"], 600.0)
        self.assertEqual(client.requests[1]["timeout"], 300.0)

    def test_conflict_maps_back_to_typed_errors(self) -> None:
        engine, client = self._engine()
        client.responses["/sandboxes/src/txn"] = _conflict("/sandboxes/src/txn", "txn_active")
        with self.assertRaises(TxnActiveError):
            engine.system.begin_txn(SandboxId("src"))
        client.responses["/sandboxes/src/txn/t/commit"] = _conflict("x", "txn_mismatch")
        with self.assertRaises(TxnMismatchError):
            engine.system.commit_txn(SandboxId("src"), "t")
        client.responses["/sandboxes/src/txn/t/abort"] = _conflict("x", "txn_abort_failed")
        with self.assertRaises(TxnAbortError):
            engine.system.abort_txn(SandboxId("src"), "t")
        # Unrecognized errors pass through as transport errors.
        client.responses["/sandboxes/src/txn"] = DaemonRequestError(500, "/x", b"{}")
        with self.assertRaises(DaemonRequestError):
            engine.system.begin_txn(SandboxId("src"))

    def test_remote_sandbox_begin_is_transport_agnostic(self) -> None:
        engine, client = self._engine()
        client.responses["/sandboxes/src/txn"] = {
            "ok": True,
            "txn": {
                "txn_id": "txn-remote",
                "sandbox_id": "src",
                "base_checkpoint_id": "ckpt-1",
                "base_was_fresh": True,
                "started_at": "2026-01-01T00:00:00+00:00",
                "label": None,
            },
        }
        client.responses["/sandboxes/src/txn/txn-remote/commit"] = {
            "ok": True,
            "result": {"txn_id": "txn-remote", "released_observations": 0, "base_dropped": True},
        }
        sandbox = Sandbox.connect("src", engine=engine)
        with sandbox.begin() as txn:
            self.assertEqual(txn.txn_id, "txn-remote")
        self.assertEqual(txn.resolved, "committed")
        # current_txn deserializes too.
        client.responses["/sandboxes/src/txn"] = {"ok": True, "txn": None}
        self.assertIsNone(sandbox.current_txn())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class CliTxnTests(unittest.TestCase):
    def _run_cli(self, argv: list[str], responses: dict) -> tuple[int, str, list]:
        requests: list[dict] = []

        class _CliClient:
            def __init__(self, socket_path, *, timeout_seconds):
                requests.append({"socket": str(socket_path), "timeout": timeout_seconds})

            def post_json(self, path, payload=None, *, timeout_seconds=None):
                requests.append({"path": path, "payload": payload})
                return responses[path]

            def get_json(self, path, *, timeout_seconds=None):
                requests.append({"path": path})
                return responses[path]

        stdout = io.StringIO()
        from crab.cli import commands

        with mock.patch.object(commands, "DaemonClient", _CliClient):
            with contextlib.redirect_stdout(stdout):
                rc = commands.main(argv)
        return rc, stdout.getvalue(), requests

    def test_begin_prints_txn_id(self) -> None:
        rc, out, requests = self._run_cli(
            ["txn", "begin", "sbx-1", "--label", "demo"],
            {
                "/sandboxes/sbx-1/txn": {
                    "ok": True,
                    "txn": {"txn_id": "txn-cli", "base_checkpoint_id": "c1"},
                }
            },
        )
        self.assertEqual(rc, 0)
        self.assertEqual(out.strip(), "txn-cli")
        self.assertEqual(requests[-1]["payload"], {"label": "demo"})
        self.assertEqual(requests[0]["timeout"], 300.0)

    def test_commit_abort_and_status(self) -> None:
        rc, out, _ = self._run_cli(
            ["txn", "commit", "sbx-1", "txn-cli"],
            {
                "/sandboxes/sbx-1/txn/txn-cli/commit": {
                    "ok": True,
                    "result": {"txn_id": "txn-cli", "released_observations": 1, "base_dropped": True},
                }
            },
        )
        self.assertEqual(rc, 0)
        self.assertIn("committed txn-cli", out)
        rc, out, _ = self._run_cli(
            ["txn", "abort", "sbx-1", "txn-cli"],
            {
                "/sandboxes/sbx-1/txn/txn-cli/abort": {
                    "ok": True,
                    "result": {
                        "txn_id": "txn-cli",
                        "discarded_observations": 2,
                        "restored_checkpoint_id": "c1",
                    },
                }
            },
        )
        self.assertEqual(rc, 0)
        self.assertIn("aborted txn-cli", out)
        self.assertIn("discarded=2", out)
        rc, out, _ = self._run_cli(
            ["txn", "status", "sbx-1"],
            {"/sandboxes/sbx-1/txn": {"ok": True, "txn": None}},
        )
        self.assertEqual(rc, 0)
        self.assertIn("no active transaction", out)


if __name__ == "__main__":
    unittest.main()
