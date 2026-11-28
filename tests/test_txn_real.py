"""Real-host end-to-end for the transaction API (PR-B2.1): the roadmap
exit criteria — mutate fs in txn → abort → state identical to base;
commit → mutations persist and staged LLM observations deliver exactly
once. Self-skipping outside the crab-dev VM."""
from __future__ import annotations

import json
import shutil
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from crab import Engine, EngineConfig, Sandbox
from crab.ids import CheckpointId


def _real_stack_available() -> bool:
    import os

    if os.geteuid() != 0:
        return False
    return all(shutil.which(tool) is not None for tool in ("docker", "runc", "criu", "zfs"))


class TxnRealTests(unittest.TestCase):
    _IMAGE = "python:3.11-slim"

    def setUp(self) -> None:
        if not _real_stack_available():
            self.skipTest("docker/runc/criu/zfs/root not available")
        self._tmp = tempfile.TemporaryDirectory(prefix="crab_txn_e2e_")
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def _engine(self, *, interceptor: bool = False) -> Engine:
        engine = Engine.start(
            EngineConfig(
                runtime="runc",
                enable_sandbox_network=False,
                enable_interceptor=interceptor,
                storage_root=self.root / "storage",
                runtime_root=self.root / "runtime",
            )
        )
        self.addCleanup(engine.stop)
        return engine

    def _run(self, sandbox: Sandbox, script: str) -> str:
        result = sandbox.commands.run(script)
        self.assertEqual(result.returncode, 0, msg=f"command failed: {script!r}: {result.stderr}")
        return result.stdout.strip()

    def test_abort_restores_base_state(self) -> None:
        engine = self._engine()
        sandbox = Sandbox(image=self._IMAGE, engine=engine)
        self.addCleanup(sandbox.kill)
        self._run(sandbox, "mkdir -p /probe && echo base-1 > /probe/a && echo base-2 > /probe/b")
        self._run(
            sandbox,
            "sh -c 'nohup sleep 600 >/dev/null 2>&1 & echo $! > /probe/worker.pid'",
        )
        base_digest = self._run(sandbox, "cat /probe/a /probe/b | sha256sum")

        txn = sandbox.begin(label="abort-e2e")
        self.assertTrue(txn.base_checkpoint_id)
        txn.exec("echo dirty > /probe/a")        # modify
        txn.exec("rm /probe/b")                  # delete
        txn.exec("echo new > /probe/c")          # create
        self.assertEqual(self._run(sandbox, "cat /probe/a"), "dirty")
        txn.abort()

        self.assertEqual(
            self._run(sandbox, "cat /probe/a /probe/b | sha256sum"), base_digest
        )
        self.assertEqual(self._run(sandbox, "test -e /probe/c && echo yes || echo no"), "no")
        # Background process from before the txn survives the rewind.
        self.assertEqual(
            self._run(sandbox, "sh -c 'kill -0 $(cat /probe/worker.pid) && echo alive'"),
            "alive",
        )
        # Base checkpoint is kept on abort.
        self.assertIn(
            CheckpointId(txn.base_checkpoint_id),
            engine.system.storage.list_checkpoints(sandbox.sandbox_id),
        )

    def test_commit_persists_and_drops_fresh_base(self) -> None:
        engine = self._engine()
        sandbox = Sandbox(image=self._IMAGE, engine=engine)
        self.addCleanup(sandbox.kill)
        self._run(sandbox, "echo before > /state.txt")

        with sandbox.begin() as txn:
            self.assertTrue(txn.base_checkpoint_id)
            fresh_base = CheckpointId(txn.base_checkpoint_id)
            txn.exec("echo committed > /state.txt")
        self.assertEqual(sandbox.current_txn(), None)
        self.assertEqual(self._run(sandbox, "cat /state.txt"), "committed")
        # The freshly-taken base was dropped on commit.
        self.assertNotIn(
            fresh_base, engine.system.storage.list_checkpoints(sandbox.sandbox_id)
        )

    def test_adaptive_base_reuses_manual_checkpoint(self) -> None:
        engine = self._engine()
        sandbox = Sandbox(image=self._IMAGE, engine=engine)
        self.addCleanup(sandbox.kill)
        self._run(sandbox, "echo v1 > /state.txt")
        checkpoint_id = sandbox.checkpoint()

        txn = sandbox.begin()
        self.assertEqual(txn.base_checkpoint_id, str(checkpoint_id))
        txn.commit()
        # Reused base survives commit.
        self.assertIn(
            CheckpointId(str(checkpoint_id)),
            engine.system.storage.list_checkpoints(sandbox.sandbox_id),
        )

    def test_staged_llm_observation_delivered_once_on_commit(self) -> None:
        upstream_hits: list[str] = []

        class _Upstream(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", "0"))
                _ = self.rfile.read(length) if length else b""
                upstream_hits.append(self.path)
                body = json.dumps(
                    {"choices": [{"message": {"content": "txn-answer"}}]}
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args) -> None:  # noqa: A002
                _ = (format, args)

        upstream = HTTPServer(("127.0.0.1", 0), _Upstream)
        threading.Thread(target=upstream.serve_forever, daemon=True).start()
        self.addCleanup(upstream.server_close)
        self.addCleanup(upstream.shutdown)
        upstream_host, upstream_port = upstream.server_address

        engine = self._engine(interceptor=True)
        sandbox = Sandbox(image=self._IMAGE, engine=engine)
        self.addCleanup(sandbox.kill)
        engine.register_upstream(
            sandbox.sandbox_id, f"http://{upstream_host}:{upstream_port}"
        )

        txn = sandbox.begin(label="llm-e2e")

        script = (
            "python3 - <<'PYEOF'\n"
            "import json, urllib.request\n"
            f"req = urllib.request.Request('{engine.interceptor_base_url}/v1/chat/completions',\n"
            "    data=json.dumps({'model': 'simulated-openai', 'messages': [{'role': 'user', 'content': 'hi'}]}).encode(),\n"
            "    headers={'Content-Type': 'application/json',\n"
            f"             'X-Agent-Sandbox-Id': '{sandbox.sandbox_id}',\n"
            "             'X-Request-Id': 'req-txn-e2e'},\n"
            "    method='POST')\n"
            "resp = urllib.request.urlopen(req, timeout=180)\n"
            "print(resp.status, resp.read().decode())\n"
            "PYEOF"
        )
        holder: dict[str, object] = {}

        def _issue() -> None:
            holder["result"] = sandbox.commands.run(script, timeout=180)

        thread = threading.Thread(target=_issue)
        thread.start()

        registry = engine.system.response_gate_registry
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            if registry.get_pending(sandbox.sandbox_id) is not None:
                break
            time.sleep(0.2)
        else:
            self.fail("gated LLM request never armed")

        time.sleep(0.5)
        self.assertTrue(thread.is_alive(), "request must stay gated inside the txn")
        self.assertEqual(len(upstream_hits), 1, "upstream must be hit exactly once")

        txn.commit()
        thread.join(timeout=60.0)
        self.assertFalse(thread.is_alive())
        result = holder["result"]
        self.assertEqual(result.returncode, 0, msg=f"in-sandbox request failed: {result.stderr}")
        self.assertIn("200", result.stdout)
        self.assertIn("txn-answer", result.stdout)
        self.assertEqual(len(upstream_hits), 1, "commit must not replay the upstream call")


if __name__ == "__main__":
    unittest.main()
