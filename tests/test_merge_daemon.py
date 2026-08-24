"""Unit tests for the daemon-mode merge/changeset surface (PR-C2.2):
daemon routes/handlers with error mapping (409 merge_error carrying the
serialized report), the _SystemShim proxies with MergeReport /
ChangesetResult rehydration, remote Sandbox.merge/changeset, and the
`crab sandbox merge` / `crab sandbox changeset` CLI. Host-runnable — no
runc."""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from crab.daemon.server import _build_handler, _MergeConflict, _Routes
from crab.daemon.transport import DaemonClient, DaemonRequestError, serve_unix_socket
from crab.ids import CheckpointId, SandboxId
from crab.merging import MergeError
from crab.models import ChangesetEntry, ChangesetResult, MergeEntry, MergeReport
from crab.remote_engine import RemoteEngine
from crab.sandbox import Sandbox


def _report(source: str = "src", fork: str = "fork", policy: str = "fail_fast") -> MergeReport:
    return MergeReport(
        source_sandbox_id=SandboxId(source),
        fork_sandbox_id=SandboxId(fork),
        base_checkpoint_id=CheckpointId("ckpt-base"),
        policy=policy,
        applied=(MergeEntry(path="/new.txt", change="added", resolution="applied"),),
        conflicted=(),
        skipped=(
            MergeEntry(path="/tmp", change="modified", resolution="skipped", reason="ignored"),
        ),
    )


def _changeset(sandbox: str = "fork") -> ChangesetResult:
    return ChangesetResult(
        sandbox_id=SandboxId(sandbox),
        base_checkpoint_id=CheckpointId("ckpt-base"),
        entries=(
            ChangesetEntry(path="/new.txt", change="added"),
            ChangesetEntry(path="/moved.txt", change="renamed", renamed_from="/old.txt"),
        ),
    )


class _FakeSystem:
    def __init__(self) -> None:
        self.calls: list = []
        self.merge_error: Exception | None = None
        self.changeset_error: Exception | None = None

    def merge_from_fork(self, source_sandbox_id, fork_sandbox_id, *, policy, **kwargs):
        self.calls.append(
            ("merge_from_fork", str(source_sandbox_id), str(fork_sandbox_id), policy, kwargs)
        )
        if self.merge_error is not None:
            raise self.merge_error
        return _report(str(source_sandbox_id), str(fork_sandbox_id), policy)

    def changeset_since(self, sandbox_id, checkpoint_id, *, use_inspector_gate=True):
        self.calls.append(("changeset_since", str(sandbox_id), str(checkpoint_id)))
        if self.changeset_error is not None:
            raise self.changeset_error
        return _changeset(str(sandbox_id))

    def fork_changeset(self, sandbox_id, *, force=False):
        self.calls.append(("fork_changeset", str(sandbox_id)))
        if self.changeset_error is not None:
            raise self.changeset_error
        return _changeset(str(sandbox_id))


class _FakeEngine:
    def __init__(self) -> None:
        self.system = _FakeSystem()


class _FakeDaemon:
    def __init__(self, engine: _FakeEngine) -> None:
        self.engine = engine

    def require_engine(self) -> _FakeEngine:
        return self.engine

    def register_sandbox(self, sandbox_id) -> None:
        pass

    def unregister_sandbox(self, sandbox_id) -> None:
        pass


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------


class MergeRouteHandlerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = _FakeEngine()
        self.routes = _Routes(_FakeDaemon(self.engine))

    def test_merge_serializes_report_and_plumbs_arguments(self) -> None:
        response = self.routes.merge_sandbox(
            {"fork_sandbox_id": "fork", "policy": "prefer_fork", "ignore_prefixes": ["/scratch"]},
            sandbox_id="src",
        )
        self.assertTrue(response["ok"])
        report = response["report"]
        self.assertEqual(report["policy"], "prefer_fork")
        self.assertEqual(report["applied"][0]["path"], "/new.txt")
        self.assertEqual(
            self.engine.system.calls,
            [
                (
                    "merge_from_fork",
                    "src",
                    "fork",
                    "prefer_fork",
                    {"ignore_prefixes": ("/scratch",)},
                )
            ],
        )

    def test_merge_defaults_policy_and_omits_prefixes(self) -> None:
        self.routes.merge_sandbox({"fork_sandbox_id": "fork"}, sandbox_id="src")
        self.assertEqual(self.engine.system.calls[0][3], "fail_fast")
        self.assertEqual(self.engine.system.calls[0][4], {})

    def test_merge_requires_fork_id(self) -> None:
        from crab.daemon.server import _BadRequest

        with self.assertRaises(_BadRequest):
            self.routes.merge_sandbox({}, sandbox_id="src")

    def test_merge_rejects_malformed_prefixes(self) -> None:
        from crab.daemon.server import _BadRequest

        with self.assertRaises(_BadRequest):
            self.routes.merge_sandbox(
                {"fork_sandbox_id": "fork", "ignore_prefixes": "/tmp"}, sandbox_id="src"
            )

    def test_merge_error_maps_to_conflict_with_report(self) -> None:
        failed = _report()
        self.engine.system.merge_error = MergeError("apply failed", report=failed)
        with self.assertRaises(_MergeConflict) as ctx:
            self.routes.merge_sandbox({"fork_sandbox_id": "fork"}, sandbox_id="src")
        self.assertEqual(ctx.exception.report, failed.to_json())

    def test_merge_error_without_report(self) -> None:
        self.engine.system.merge_error = MergeError("txn active")
        with self.assertRaises(_MergeConflict) as ctx:
            self.routes.merge_sandbox({"fork_sandbox_id": "fork"}, sandbox_id="src")
        self.assertIsNone(ctx.exception.report)

    def test_merge_value_error_maps_to_bad_request(self) -> None:
        from crab.daemon.server import _BadRequest

        self.engine.system.merge_error = ValueError("not a fork")
        with self.assertRaises(_BadRequest):
            self.routes.merge_sandbox({"fork_sandbox_id": "fork"}, sandbox_id="src")

    def test_changeset_with_since_and_fork_point_default(self) -> None:
        response = self.routes.changeset_sandbox({"since": "ckpt-9"}, sandbox_id="fork")
        self.assertEqual(response["changeset"]["entries"][0]["path"], "/new.txt")
        self.assertIn(("changeset_since", "fork", "ckpt-9"), self.engine.system.calls)
        self.routes.changeset_sandbox({}, sandbox_id="fork")
        self.assertIn(("fork_changeset", "fork"), self.engine.system.calls)

    def test_changeset_errors_map_to_bad_request(self) -> None:
        from crab.daemon.server import _BadRequest

        self.engine.system.changeset_error = FileNotFoundError("snapshot missing")
        with self.assertRaises(_BadRequest):
            self.routes.changeset_sandbox({"since": "ckpt-9"}, sandbox_id="fork")
        self.engine.system.changeset_error = ValueError("no fork marker")
        with self.assertRaises(_BadRequest):
            self.routes.changeset_sandbox({}, sandbox_id="fork")


class MergeRouteDispatchTests(unittest.TestCase):
    """Merge/changeset routes over the real Unix-socket HTTP stack."""

    def setUp(self) -> None:
        self.engine = _FakeEngine()
        self._tmp = tempfile.TemporaryDirectory(prefix="crab_merged_")
        self.addCleanup(self._tmp.cleanup)
        socket_path = Path(self._tmp.name) / "crab.sock"
        self.server = serve_unix_socket(socket_path, _build_handler(_FakeDaemon(self.engine)))
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.client = DaemonClient(socket_path, timeout_seconds=10.0)

    def test_merge_round_trip(self) -> None:
        response = self.client.post_json(
            "/sandboxes/src/merge", {"fork_sandbox_id": "fork", "policy": "prefer_source"}
        )
        self.assertTrue(response["ok"])
        self.assertEqual(response["report"]["policy"], "prefer_source")

    def test_merge_conflict_surfaces_409_with_report(self) -> None:
        failed = _report()
        self.engine.system.merge_error = MergeError("rolled back", report=failed)
        with self.assertRaises(DaemonRequestError) as ctx:
            self.client.post_json("/sandboxes/src/merge", {"fork_sandbox_id": "fork"})
        self.assertEqual(ctx.exception.status_code, 409)
        payload = json.loads(ctx.exception.body.decode("utf-8"))
        self.assertEqual(payload["error_type"], "merge_error")
        self.assertEqual(payload["report"], failed.to_json())

    def test_changeset_round_trip(self) -> None:
        response = self.client.post_json("/sandboxes/fork/changeset", {})
        entries = response["changeset"]["entries"]
        self.assertEqual(entries[1]["renamed_from"], "/old.txt")


# ---------------------------------------------------------------------------
# _SystemShim proxies + remote Sandbox surface
# ---------------------------------------------------------------------------


class _FakeDaemonClient:
    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.responses: dict[str, object] = {}

    def post_json(self, path, payload=None, *, timeout_seconds=None):
        self.requests.append(
            {"method": "POST", "path": path, "payload": payload, "timeout": timeout_seconds}
        )
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


class SystemShimMergeTests(unittest.TestCase):
    _INFO = {"runtime": "runc", "default_image": "ubuntu:22.04"}

    def _engine(self) -> tuple[RemoteEngine, _FakeDaemonClient]:
        client = _FakeDaemonClient()
        return RemoteEngine(client, info=self._INFO), client

    def test_merge_posts_and_rehydrates_report(self) -> None:
        engine, client = self._engine()
        client.responses["/sandboxes/src/merge"] = {"ok": True, "report": _report().to_json()}
        report = engine.system.merge_from_fork(
            SandboxId("src"), SandboxId("fork"), policy="prefer_fork", ignore_prefixes=("/x",)
        )
        self.assertIsInstance(report, MergeReport)
        self.assertIsInstance(report.applied[0], MergeEntry)
        self.assertEqual(report.source_sandbox_id, SandboxId("src"))
        self.assertEqual(report.skipped[0].reason, "ignored")
        request = client.requests[0]
        self.assertEqual(request["path"], "/sandboxes/src/merge")
        self.assertEqual(
            request["payload"],
            {"fork_sandbox_id": "fork", "policy": "prefer_fork", "ignore_prefixes": ["/x"]},
        )
        self.assertEqual(request["timeout"], 600.0)

    def test_merger_hook_is_rejected_client_side(self) -> None:
        engine, client = self._engine()
        with self.assertRaises(NotImplementedError):
            engine.system.merge_from_fork(
                SandboxId("src"), SandboxId("fork"), merger=lambda *args: None
            )
        self.assertEqual(client.requests, [])

    def test_conflict_rehydrates_merge_error_with_report(self) -> None:
        engine, client = self._engine()
        failed = _report()
        body = json.dumps(
            {"ok": False, "error": "rolled back", "error_type": "merge_error", "report": failed.to_json()}
        ).encode("utf-8")
        client.responses["/sandboxes/src/merge"] = DaemonRequestError(409, "/x", body)
        with self.assertRaises(MergeError) as ctx:
            engine.system.merge_from_fork(SandboxId("src"), SandboxId("fork"))
        self.assertEqual(str(ctx.exception), "rolled back")
        self.assertIsInstance(ctx.exception.report, MergeReport)
        self.assertEqual(ctx.exception.report.applied[0].path, "/new.txt")

    def test_unrecognized_errors_pass_through(self) -> None:
        engine, client = self._engine()
        client.responses["/sandboxes/src/merge"] = DaemonRequestError(500, "/x", b"{}")
        with self.assertRaises(DaemonRequestError):
            engine.system.merge_from_fork(SandboxId("src"), SandboxId("fork"))

    def test_changeset_since_and_fork_changeset(self) -> None:
        engine, client = self._engine()
        client.responses["/sandboxes/fork/changeset"] = {
            "ok": True,
            "changeset": _changeset().to_json(),
        }
        result = engine.system.changeset_since(SandboxId("fork"), CheckpointId("ckpt-9"))
        self.assertIsInstance(result, ChangesetResult)
        self.assertEqual(result.entries[1].renamed_from, "/old.txt")
        self.assertEqual(client.requests[0]["payload"], {"since": "ckpt-9"})
        result = engine.system.fork_changeset(SandboxId("fork"))
        self.assertEqual(result.base_checkpoint_id, CheckpointId("ckpt-base"))
        self.assertEqual(client.requests[1]["payload"], {})

    def test_remote_sandbox_merge_and_changeset_are_transport_agnostic(self) -> None:
        engine, client = self._engine()
        client.responses["/sandboxes/src/merge"] = {"ok": True, "report": _report().to_json()}
        client.responses["/sandboxes/fork/changeset"] = {
            "ok": True,
            "changeset": _changeset().to_json(),
        }
        source = Sandbox.connect("src", engine=engine)
        fork = Sandbox.connect("fork", engine=engine)
        report = source.merge(fork, policy="fail_fast")
        self.assertIsInstance(report, MergeReport)
        entries = fork.changeset()
        self.assertEqual(entries[0], {"path": "/new.txt", "change": "added"})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class CliMergeTests(unittest.TestCase):
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

    def test_merge_prints_summary_and_plumbs_payload(self) -> None:
        rc, out, requests = self._run_cli(
            [
                "sandbox", "merge", "sbx-src", "sbx-fork",
                "--policy", "prefer_fork", "--ignore-prefix", "/scratch",
            ],
            {"/sandboxes/sbx-src/merge": {"ok": True, "report": _report().to_json()}},
        )
        self.assertEqual(rc, 0)
        self.assertIn("applied=1 conflicted=0 skipped=1", out)
        self.assertEqual(
            requests[-1]["payload"],
            {
                "fork_sandbox_id": "sbx-fork",
                "policy": "prefer_fork",
                "ignore_prefixes": ["/scratch"],
            },
        )
        self.assertEqual(requests[0]["timeout"], 600.0)

    def test_merge_conflicts_exit_nonzero_and_list_paths(self) -> None:
        conflicted = MergeReport(
            source_sandbox_id=SandboxId("sbx-src"),
            fork_sandbox_id=SandboxId("sbx-fork"),
            base_checkpoint_id=CheckpointId("ckpt-base"),
            policy="fail_fast",
            applied=(),
            conflicted=(
                MergeEntry(
                    path="/shared.txt",
                    change="modified",
                    resolution="conflicted",
                    reason="source_changed",
                ),
            ),
            skipped=(),
        )
        rc, out, _ = self._run_cli(
            ["sandbox", "merge", "sbx-src", "sbx-fork"],
            {"/sandboxes/sbx-src/merge": {"ok": True, "report": conflicted.to_json()}},
        )
        self.assertEqual(rc, 1)
        self.assertIn("conflict\t/shared.txt\tsource_changed", out)

    def test_merge_json_output(self) -> None:
        rc, out, _ = self._run_cli(
            ["--json", "sandbox", "merge", "sbx-src", "sbx-fork"],
            {"/sandboxes/sbx-src/merge": {"ok": True, "report": _report().to_json()}},
        )
        self.assertEqual(rc, 0)
        self.assertEqual(json.loads(out)["policy"], "fail_fast")

    def test_changeset_prints_rows_and_since_payload(self) -> None:
        rc, out, requests = self._run_cli(
            ["sandbox", "changeset", "sbx-fork", "--since", "ckpt-9"],
            {"/sandboxes/sbx-fork/changeset": {"ok": True, "changeset": _changeset().to_json()}},
        )
        self.assertEqual(rc, 0)
        self.assertIn("added\t/new.txt", out)
        self.assertIn("renamed\t/moved.txt\t(from /old.txt)", out)
        self.assertEqual(requests[-1]["payload"], {"since": "ckpt-9"})

    def test_changeset_defaults_to_fork_point(self) -> None:
        rc, _, requests = self._run_cli(
            ["sandbox", "changeset", "sbx-fork"],
            {"/sandboxes/sbx-fork/changeset": {"ok": True, "changeset": _changeset().to_json()}},
        )
        self.assertEqual(rc, 0)
        self.assertEqual(requests[-1]["payload"], {})


if __name__ == "__main__":
    unittest.main()
