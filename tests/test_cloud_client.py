"""Full-stack tests for the SDK's cloud mode (track S, S2+S5): a real
`GatewayServer` over a scripted stub daemon, driven end to end through
`CloudClient` + `Engine.connect(url=...)` + the unchanged
`RemoteEngine`/`Sandbox` surface. Host-runnable — no runc/CRIU/zfs, no
root, no external network (the "cloud" is 127.0.0.1).

Covers the S2 exit surface: connect dispatch on argument shape, the
lifecycle verbs (create/exec/checkpoint/restore/fork/kill), the typed
error taxonomy (401 auth, cross-tenant 404, quota 409, lost 410, daemon
502, gateway 504 vs client-side timeout), the host-shim guard (only
/shutdown remains blocked after S5 full-access), and the three-verb
signature conformance between `DaemonClient` and `CloudClient` (§8).

The stub daemon reports runtime "docker" here so `Sandbox._launch`
takes the remote-mode path (S5) — no client-side bundle prep.
"""
from __future__ import annotations

import inspect
import os
import socket as socket_mod
import unittest
from pathlib import Path
from unittest import mock

from crab.cloud_client import (
    API_KEY_ENV,
    CloudAuthError,
    CloudClient,
    CloudConnectionError,
    CloudUnsupportedOperation,
    DaemonUnreachableError,
    GatewayTimeoutError,
    QuotaExceeded,
    SandboxLost,
    SandboxNotFound,
)
from crab.daemon.transport import DaemonClient, DaemonRequestError
from crab.engine import Engine
from crab.gateway import server as gateway_server
from crab.remote_engine import RemoteEngine
from crab.sandbox import Sandbox
from tests.test_gateway_server import GatewayTestBase


class CloudTestBase(GatewayTestBase):
    """Stub daemon + gateway from the S1 harness, plus SDK-side helpers."""

    def setUp(self) -> None:
        super().setUp()
        # Metadata-only launch path (see module docstring).
        self.state.runtime = "docker"

    @property
    def gateway_url(self) -> str:
        return f"http://127.0.0.1:{self.gateway.port}"

    def connect(self, api_key: str) -> Engine:
        return Engine.connect(url=self.gateway_url, api_key=api_key)

    def cloud_client(self, api_key: str) -> CloudClient:
        return CloudClient(self.gateway_url, api_key)

    def daemon_requests(self, needle: str) -> list[tuple[str, str, dict]]:
        with self.state.lock:
            return [(m, p, b) for m, p, b in self.state.requests if needle in p]


class CloudLiveTestBase(CloudTestBase):
    def setUp(self) -> None:
        super().setUp()
        self.start_gateway()
        self.tenant, self.key = self.make_tenant("acme")


# ---------------------------------------------------------------------------
# Engine.connect dispatch — shape decides socket vs cloud.
# ---------------------------------------------------------------------------


class ConnectDispatchTests(CloudLiveTestBase):
    def test_url_kwarg_returns_remote_engine(self) -> None:
        engine = self.connect(self.key)
        self.assertIsInstance(engine, RemoteEngine)
        # /info flowed through the gateway's redacted whitelist.
        self.assertEqual(engine.runtime.name, "docker")
        self.assertEqual(engine.config.default_image, "ubuntu:22.04")

    def test_positional_url_is_dispatched_to_cloud(self) -> None:
        # An http(s):// string in the socket slot is a gateway URL.
        engine = Engine.connect(self.gateway_url, api_key=self.key)
        self.assertIsInstance(engine, RemoteEngine)
        self.assertEqual(engine.runtime.name, "docker")

    def test_url_and_socket_together_is_an_error(self) -> None:
        with self.assertRaises(ValueError):
            Engine.connect("/tmp/crab.sock", url=self.gateway_url, api_key=self.key)

    def test_api_key_without_url_is_an_error(self) -> None:
        with self.assertRaises(ValueError):
            Engine.connect(api_key=self.key)

    def test_api_key_env_fallback(self) -> None:
        with mock.patch.dict(os.environ, {API_KEY_ENV: self.key}):
            engine = Engine.connect(url=self.gateway_url)
        self.assertIsInstance(engine, RemoteEngine)

    def test_missing_api_key_is_an_error(self) -> None:
        env = {k: v for k, v in os.environ.items() if k != API_KEY_ENV}
        with mock.patch.dict(os.environ, env, clear=True):
            with self.assertRaises(ValueError):
                Engine.connect(url=self.gateway_url)

    def test_bad_key_raises_typed_auth_error_at_connect(self) -> None:
        with self.assertRaises(CloudAuthError) as ctx:
            Engine.connect(url=self.gateway_url, api_key="crab_sk_" + "0" * 48)
        self.assertEqual(ctx.exception.status_code, 401)

    def test_unreachable_gateway_is_connection_error(self) -> None:
        probe = socket_mod.socket()
        probe.bind(("127.0.0.1", 0))
        dead_port = probe.getsockname()[1]
        probe.close()
        with self.assertRaises(CloudConnectionError):
            Engine.connect(url=f"http://127.0.0.1:{dead_port}", api_key=self.key)

    def test_bad_scheme_is_an_error(self) -> None:
        with self.assertRaises(ValueError):
            CloudClient("ftp://gateway.example.com", "k")

    def test_host_paths_are_unresolvable_in_cloud_mode(self) -> None:
        # The gateway's /info whitelist omits storage_root & friends; the
        # RemoteEngine path helpers must fail loudly, not invent paths.
        engine = self.connect(self.key)
        with self.assertRaises(RuntimeError):
            engine.storage_root


# ---------------------------------------------------------------------------
# Lifecycle — the unchanged Sandbox surface over CloudClient.
# ---------------------------------------------------------------------------


class CloudLifecycleTests(CloudLiveTestBase):
    def test_create_preserves_network_tristate_on_the_wire(self) -> None:
        engine = self.connect(self.key)
        explicit_host = Sandbox(image="ubuntu:22.04", network=False, engine=engine)
        automatic = Sandbox(image="ubuntu:22.04", network=None, engine=engine)
        try:
            create_bodies = self.state.create_bodies()
            self.assertIs(create_bodies[-2]["metadata"]["network"], False)
            self.assertNotIn("network", create_bodies[-1]["metadata"])
        finally:
            explicit_host.kill()
            automatic.kill()

    def test_create_rejects_non_boolean_network_before_wire_call(self) -> None:
        engine = self.connect(self.key)
        before = len(self.state.create_bodies())
        with self.assertRaisesRegex(ValueError, "true, false, or null"):
            Sandbox(image="ubuntu:22.04", network="false", engine=engine)  # type: ignore[arg-type]
        self.assertEqual(len(self.state.create_bodies()), before)

    def test_full_lifecycle_over_gateway(self) -> None:
        engine = self.connect(self.key)
        sbx = Sandbox(image="ubuntu:22.04", engine=engine)
        sandbox_id = str(sbx.sandbox_id)

        row = self.gateway.registry.get_sandbox(sandbox_id)
        self.assertEqual(row["status"], "active")
        self.assertEqual(row["tenant_id"], self.tenant["id"])

        result = sbx.commands.run("echo hi")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "ran /bin/sh")

        checkpoint_id = sbx.checkpoint()
        self.assertEqual(checkpoint_id, "ck-1")
        sbx.restore(checkpoint_id)  # cloud repair_network_lease is a no-op

        forks = sbx.fork(2)
        self.assertEqual(len(forks), 2)
        for fork in forks:
            fork_row = self.gateway.registry.get_sandbox(str(fork.sandbox_id))
            self.assertEqual(fork_row["status"], "active")
            self.assertEqual(fork_row["tenant_id"], self.tenant["id"])

        sbx.kill()
        self.assertEqual(
            self.gateway.registry.get_sandbox(sandbox_id)["status"], "killed"
        )
        with self.assertRaises(SandboxNotFound):
            engine.runtime.describe(sbx.sandbox_id)

    def test_delete_checkpoint_cascade_body_reaches_daemon(self) -> None:
        # DELETE-with-body rides the private `_request_json` seam and the
        # gateway forwards it verbatim (S2 fix for the S1 passthrough).
        engine = self.connect(self.key)
        sbx = Sandbox(engine=engine)
        sandbox_id = str(sbx.sandbox_id)
        engine.system.storage.delete_checkpoint(
            sbx.sandbox_id, "ck-1", cascade=True
        )
        seen = self.daemon_requests(f"/sandboxes/{sandbox_id}/checkpoints/ck-1")
        self.assertEqual(seen, [("DELETE", f"/sandboxes/{sandbox_id}/checkpoints/ck-1", {"cascade": True})])

    def test_action_result_over_gateway(self) -> None:
        # feat/sdk-improvements: commands.run() enriched with checkpoint
        # + observe travels the full gateway -> daemon path. The client
        # pre-allocates the checkpoint id and the daemon echoes it back.
        engine = self.connect(self.key)
        sbx = Sandbox(engine=engine)
        sandbox_id = str(sbx.sandbox_id)
        result = sbx.commands.run("echo hi", checkpoint=True, observe=True)
        self.assertEqual(result.returncode, 0)
        self.assertIsNotNone(result.checkpoint)
        assert result.checkpoint is not None
        returned = result.checkpoint.wait(timeout=5.0)
        self.assertTrue(returned.startswith("ckpt-"))
        # Daemon saw the checkpoint_id in the body.
        seen = self.daemon_requests(f"/sandboxes/{sandbox_id}/action")
        post_calls = [(m, p, b) for m, p, b in seen if m == "POST"]
        self.assertEqual(len(post_calls), 1)
        self.assertEqual(post_calls[0][2].get("checkpoint_id"), returned)
        self.assertTrue(post_calls[0][2].get("observe"))
        # Observe flags came back in the same batch response.
        self.assertTrue(result.filesystem_changed)
        self.assertFalse(result.process_changed)


# ---------------------------------------------------------------------------
# Error taxonomy — typed exceptions per the gateway error table.
# ---------------------------------------------------------------------------


class CloudErrorTests(CloudLiveTestBase):
    def test_revoked_key_mid_session(self) -> None:
        client = self.cloud_client(self.key)
        self.assertTrue(client.get_json("/sandboxes")["ok"])
        self.admin.post_json("/admin/keys/revoke", {"key": self.key})
        with self.assertRaises(CloudAuthError):
            client.get_json("/sandboxes")

    def test_cross_tenant_sandbox_is_not_found(self) -> None:
        _, rival_key = self.make_tenant("rival")
        _, payload = self.request("POST", "/v1/sandboxes", api_key=rival_key, body={})
        rival_sandbox = payload["sandbox_id"]
        engine = self.connect(self.key)
        with self.assertRaises(SandboxNotFound):
            engine.runtime.describe(rival_sandbox)

    def test_quota_exceeded_is_typed_with_arithmetic(self) -> None:
        _, capped_key = self.make_tenant("capped", max_sandboxes=1)
        engine = Engine.connect(url=self.gateway_url, api_key=capped_key)
        Sandbox(engine=engine)
        with self.assertRaises(QuotaExceeded) as ctx:
            Sandbox(engine=engine)
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.quota["max_sandboxes"], 1)
        self.assertEqual(ctx.exception.quota["live_sandboxes"], 1)

    def test_aggregate_quota_exceeded_is_typed_with_arithmetic(self) -> None:
        # S3 aggregate caps: the SDK's `resources=` claim rides create
        # metadata and the gateway's sum gate answers with the same typed
        # 409 + arithmetic payload the count gate uses.
        mib = 1024 * 1024
        _, metered_key = self.make_tenant(
            "metered", quotas={"max_memory_bytes": 1024 * mib}
        )
        engine = Engine.connect(url=self.gateway_url, api_key=metered_key)
        first = Sandbox(engine=engine, resources={"memory": "768M"})
        row = self.gateway.registry.get_sandbox(str(first.sandbox_id))
        self.assertEqual(row["resources"], {"memory_bytes": 768 * mib})
        with self.assertRaises(QuotaExceeded) as ctx:
            Sandbox(engine=engine, resources={"memory": "512M"})
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(
            ctx.exception.quota,
            {
                "max_memory_bytes": 1024 * mib,
                "live_memory_bytes": 768 * mib,
                "requested_memory_bytes": 512 * mib,
            },
        )

    def test_lost_sandbox_is_410(self) -> None:
        engine = self.connect(self.key)
        sbx = Sandbox(engine=engine)
        # Daemon restarted underneath the gateway (boot-identity mismatch).
        self.gateway.registry.mark_all_active_lost()
        with self.assertRaises(SandboxLost):
            sbx.commands.run("echo hi")

    def test_daemon_unreachable_is_502(self) -> None:
        client = self.cloud_client(self.key)
        self.assertTrue(client.ping())
        self._stop_stub_daemon()
        with self.assertRaises(DaemonUnreachableError):
            client.get_json("/sandboxes")


class CloudTimeoutTests(CloudTestBase):
    def test_gateway_timeout_surfaces_as_typed_504(self) -> None:
        # Dial the /stop route timeout down (baked in at start()).
        short_routes = [
            (method, subpath, 0.3 if subpath == "/stop" else timeout)
            for method, subpath, timeout in gateway_server._PASSTHROUGH_SANDBOX_ROUTES
        ]
        with mock.patch.object(
            gateway_server, "_PASSTHROUGH_SANDBOX_ROUTES", short_routes
        ):
            self.start_gateway()
        _tenant, key = self.make_tenant("acme")
        engine = self.connect(key)
        sbx = Sandbox(engine=engine)
        self.state.stop_delay_s = 1.5  # past the 0.3s gateway route timeout
        client = self.cloud_client(key)
        with self.assertRaises(GatewayTimeoutError) as ctx:
            client.post_json(f"/sandboxes/{sbx.sandbox_id}/stop", {})
        self.assertEqual(ctx.exception.status_code, 504)

    def test_client_side_timeout_is_plain_timeout_error(self) -> None:
        # When the *client's* own deadline fires first the presentation
        # matches DaemonClient: plain TimeoutError, no gateway status.
        self.start_gateway()
        _tenant, key = self.make_tenant("acme")
        engine = self.connect(key)
        sbx = Sandbox(engine=engine)
        self.state.stop_delay_s = 1.0  # gateway route allows it; we don't
        client = self.cloud_client(key)
        try:
            with self.assertRaises(TimeoutError) as ctx:
                client.post_json(
                    f"/sandboxes/{sbx.sandbox_id}/stop", {}, timeout_seconds=0.2
                )
            self.assertNotIsInstance(ctx.exception, DaemonRequestError)
        finally:
            self.state.stop_delay_s = 0.0


# ---------------------------------------------------------------------------
# Host-shim guard — only /shutdown remains blocked; all other routes are
# now proxied through the gateway (S5 full-access unlock).
# ---------------------------------------------------------------------------


class HostShimGuardTests(CloudLiveTestBase):
    def setUp(self) -> None:
        super().setUp()
        self.engine = self.connect(self.key)
        self.sbx = Sandbox(engine=self.engine)
        self.sandbox_id = self.sbx.sandbox_id

    def _assert_hit_the_wire(self, needle: str) -> None:
        self.assertTrue(
            len(self.daemon_requests(needle)) > 0,
            f"expected daemon request containing '{needle}'",
        )

    def test_write_bundle_spec_goes_through(self) -> None:
        # No longer guarded; the gateway proxies to daemon.
        self.engine.runtime.write_bundle_spec(Path("/tmp/bundle"))
        self._assert_hit_the_wire("/runtime/")

    def test_host_inspector_filters_go_through(self) -> None:
        self.engine.runtime.update_host_inspector_filters(
            self.sandbox_id, ignored_path_prefixes=["/tmp"]
        )
        self._assert_hit_the_wire("host_inspector")

    def test_register_upstream_goes_through(self) -> None:
        self.engine.register_upstream(self.sandbox_id, "http://127.0.0.1:9999")
        self._assert_hit_the_wire("upstream")

    def test_allocate_network_lease_goes_through(self) -> None:
        self.engine.allocate_network_lease(self.sandbox_id)
        self._assert_hit_the_wire("network/lease")

    def test_process_merge_goes_through(self) -> None:
        self.engine.system.merge_processes(self.sandbox_id, "some-fork")
        self._assert_hit_the_wire("processes/merge")

    def test_daemon_shutdown_is_guarded(self) -> None:
        with self.assertRaises(CloudUnsupportedOperation):
            self.cloud_client(self.key).post_json("/shutdown")

    def test_best_effort_cleanup_helpers_go_through(self) -> None:
        # RemoteEngine swallows errors; the routes now hit the daemon.
        self.engine.unregister_upstream(self.sandbox_id)
        self.engine.release_network_lease(self.sandbox_id)
        self._assert_hit_the_wire("upstream")
        self._assert_hit_the_wire("network/lease")

    def test_kill_completes_with_cleanup_going_through(self) -> None:
        # Sandbox.kill() calls upstream/lease cleanup then DELETE.
        self.sbx.kill()
        row = self.gateway.registry.get_sandbox(str(self.sandbox_id))
        self.assertEqual(row["status"], "killed")


# ---------------------------------------------------------------------------
# Transport-protocol conformance (design doc §8) — both clients satisfy
# the same three-verb contract, so RemoteEngine can consume either.
# ---------------------------------------------------------------------------


class VerbConformanceTests(unittest.TestCase):
    def test_verb_signatures_are_identical(self) -> None:
        for verb in ("get_json", "post_json", "delete", "_request_json", "ping"):
            self.assertEqual(
                inspect.signature(getattr(DaemonClient, verb)),
                inspect.signature(getattr(CloudClient, verb)),
                f"CloudClient.{verb} drifted from DaemonClient.{verb}",
            )

    def test_typed_errors_subclass_daemon_request_error(self) -> None:
        # RemoteEngine rehydrates structured daemon errors by catching
        # DaemonRequestError; the cloud taxonomy must stay inside it.
        for exc_type in (
            CloudAuthError,
            SandboxNotFound,
            QuotaExceeded,
            SandboxLost,
            DaemonUnreachableError,
            GatewayTimeoutError,
        ):
            self.assertTrue(issubclass(exc_type, DaemonRequestError), exc_type)


if __name__ == "__main__":
    unittest.main()
