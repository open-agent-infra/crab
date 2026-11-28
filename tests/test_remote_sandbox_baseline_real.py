"""Real runc/ZFS daemon+gateway acceptance for the sandbox baseline.

The matrix intentionally pulls public images and changes host networking, so
normal unit discovery skips it. Run in crab-vm with
`CRAB_REAL_HOST_TESTS=1 python3 -m unittest -v
tests.test_remote_sandbox_baseline_real`.
"""

from __future__ import annotations

import os
import shutil
import socket
import tempfile
import threading
import time
import unittest
import urllib.request
import uuid
from pathlib import Path

from crab import (
    Engine,
    EngineConfig,
    ImageNotFoundError,
    Sandbox,
    SandboxExecTimeout,
)
from crab.cloud_client import CloudRequestError
from crab.daemon import DaemonClient
from crab.daemon.server import DaemonServer
from crab.gateway.server import GatewayServer


def _available() -> bool:
    return bool(
        os.environ.get("CRAB_REAL_HOST_TESTS")
        and os.geteuid() == 0
        and all(
            shutil.which(tool) is not None
            for tool in ("docker", "runc", "criu", "zfs")
        )
    )


@unittest.skipUnless(_available(), "requires CRAB_REAL_HOST_TESTS=1 and runc/ZFS")
class RemoteSandboxBaselineRealTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temp = tempfile.TemporaryDirectory(prefix="crab_remote_baseline_")
        root = Path(cls._temp.name)
        cls.socket_path = root / "crab.sock"
        cls.daemon = DaemonServer(
            engine_config=EngineConfig(
                runtime="runc",
                enable_sandbox_network=True,
                enable_interceptor=False,
                enable_egress_proxy=False,
                # Exercise the production cgroup/eBPF attribution path.  The
                # lightweight in-process inspector is a synthetic test double
                # and cannot observe or accept runtime dirty invalidations.
                host_inspector_launch_mode="thread",
                storage_root=root / "storage",
                runtime_root=root / "runtime",
                image_cache_root=root / "images",
                image_pull_timeout_seconds=600.0,
                image_min_free_bytes=512 * 1024 * 1024,
                image_cache_max_bytes=16 * 1024 * 1024 * 1024,
            ),
            socket_path=cls.socket_path,
        )
        cls.daemon.start()
        cls.local_engine = cls.daemon.require_engine()
        cls.daemon_engine = Engine.connect(
            socket=cls.socket_path,
            timeout_seconds=1200.0,
        )
        cls.gateway = GatewayServer(
            data_dir=root / "gateway",
            daemon_socket=cls.socket_path,
            host="127.0.0.1",
            port=0,
            admin_socket_path=root / "gateway-admin.sock",
        )
        cls.gateway.start()
        admin = DaemonClient(root / "gateway-admin.sock")
        tenant = admin.post_json(
            "/admin/tenants",
            {"name": "baseline-real", "quotas": {"max_sandboxes": 20}},
        )["tenant"]
        cls.tenant_id = tenant["id"]
        cls.api_key = admin.post_json(
            "/admin/keys", {"tenant_id": tenant["id"]}
        )["api_key"]
        cls.engine = Engine.connect(
            url=f"http://127.0.0.1:{cls.gateway.port}",
            api_key=cls.api_key,
            timeout_seconds=1200.0,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.gateway.stop()
        cls.daemon.stop()
        cls._temp.cleanup()

    def _sandbox(
        self,
        image: str,
        *,
        network: bool | None,
        engine=None,
    ) -> Sandbox:
        sandbox = Sandbox(
            image=image,
            network=network,
            engine=self.engine if engine is None else engine,
        )
        self.addCleanup(sandbox.kill)
        return sandbox

    def test_00_concurrent_create_shares_atomic_image_export(self) -> None:
        sandboxes: list[Sandbox] = []
        failures: list[BaseException] = []
        lock = threading.Lock()
        start = threading.Barrier(2)

        def create() -> None:
            try:
                start.wait(timeout=10)
                sandbox = Sandbox(
                    image="debian:12-slim",
                    network=False,
                    engine=self.engine,
                )
                with lock:
                    sandboxes.append(sandbox)
            except BaseException as exc:  # pragma: no cover - asserted below
                with lock:
                    failures.append(exc)

        threads = [threading.Thread(target=create) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=1200)
        try:
            self.assertFalse(any(thread.is_alive() for thread in threads))
            self.assertEqual(failures, [])
            self.assertEqual(len(sandboxes), 2)
            image_ids = set()
            for sandbox in sandboxes:
                result = sandbox.commands.run(
                    ["sh", "-c", "test -s /etc/os-release"], check=True
                )
                self.assertEqual(result.returncode, 0)
                image_ids.add(sandbox.describe().metadata.get("image_id"))
            self.assertEqual(len(image_ids), 1)
        finally:
            for sandbox in sandboxes:
                sandbox.kill()

    def test_local_sdk_path_gets_same_dns_and_capability_baseline(self) -> None:
        sandbox = self._sandbox(
            "ubuntu:22.04",
            network=False,
            engine=self.local_engine,
        )
        dns = sandbox.commands.run(
            ["getent", "hosts", "archive.ubuntu.com"], timeout=60, check=True
        )
        self.assertTrue(dns.stdout.strip())
        apt = sandbox.commands.run(
            ["apt-get", "update"], timeout=600, check=True
        )
        self.assertEqual(apt.returncode, 0)
        capability_hex = sandbox.commands.run(
            ["sh", "-lc", "awk '/^CapEff:/{print $2}' /proc/self/status"],
            check=True,
        ).stdout.strip()
        effective = int(capability_hex, 16)
        self.assertTrue(effective & (1 << 6))
        self.assertTrue(effective & (1 << 7))
        self.assertFalse(effective & (1 << 21))

    def test_direct_daemon_sdk_path_uses_daemon_network_default(self) -> None:
        sandbox = self._sandbox(
            "python:3.12-slim",
            network=None,
            engine=self.daemon_engine,
        )
        self.assertEqual(sandbox.describe().metadata["network_mode"], "isolated")
        result = sandbox.commands.run(
            ["python", "-c", "import socket; print(socket.gethostbyname('example.com'))"],
            timeout=60,
            check=True,
        )
        self.assertTrue(result.stdout.strip())

    def test_dns_capabilities_and_ordinary_apt_through_gateway(self) -> None:
        sandbox = self._sandbox("ubuntu:22.04", network=False)
        dns = sandbox.commands.run(
            ["getent", "hosts", "archive.ubuntu.com"], timeout=60, check=True
        )
        self.assertTrue(dns.stdout.strip())
        apt = sandbox.commands.run(
            [
                "sh",
                "-lc",
                "apt-get update && DEBIAN_FRONTEND=noninteractive "
                "apt-get install -y --no-install-recommends curl",
            ],
            timeout=600,
            check=False,
        )
        if apt.returncode != 0:
            resolver = sandbox.files.read("/etc/resolv.conf")
            dns_diagnostic = sandbox.commands.run(
                [
                    "sh",
                    "-lc",
                    "id; getent passwd _apt; "
                    "stat -c '%A %a %u:%g %n' / /etc /etc/resolv.conf "
                    "/etc/nsswitch.conf; "
                    "getent hosts archive.ubuntu.com; "
                    "echo root_getent_rc=$?; "
                    "if command -v setpriv >/dev/null; then "
                    "setpriv --reuid=_apt --regid=nogroup --clear-groups "
                    "sh -c 'id; cat /etc/resolv.conf >/dev/null; "
                    "echo apt_resolv_read_rc=$?; getent hosts localhost; "
                    "echo apt_hosts_rc=$?; getent hosts archive.ubuntu.com; "
                    "echo apt_dns_rc=$?'; echo setpriv_rc=$?; fi",
                ],
                timeout=60,
                check=False,
            )
            self.fail(
                f"apt failed rc={apt.returncode}\n"
                f"resolver:\n{resolver}\n"
                f"diagnostic stdout:\n{dns_diagnostic.stdout}\n"
                f"diagnostic stderr:\n{dns_diagnostic.stderr}\n"
                f"apt stderr:\n{apt.stderr[-4000:]}"
            )
        capabilities = sandbox.commands.run(
            ["sh", "-lc", "awk '/^CapEff:/{print $2}' /proc/self/status"],
            check=True,
        ).stdout.strip()
        effective = int(capabilities, 16)
        self.assertTrue(effective & (1 << 6), "CAP_SETGID missing")
        self.assertTrue(effective & (1 << 7), "CAP_SETUID missing")
        self.assertFalse(effective & (1 << 21), "CAP_SYS_ADMIN unexpectedly present")
        self.assertEqual(sandbox.describe().metadata["network_mode"], "host")

    def test_explicit_isolated_network_has_dns_and_egress(self) -> None:
        sandbox = self._sandbox("python:3.12-slim", network=True)
        result = sandbox.commands.run(
            [
                "python3",
                "-c",
                "import socket,urllib.request; "
                "assert socket.getaddrinfo('archive.ubuntu.com', 443); "
                "print(urllib.request.urlopen('https://example.com', timeout=30).status)",
            ],
            timeout=60,
            check=True,
        )
        self.assertEqual(result.stdout.strip(), "200")
        metadata = sandbox.describe().metadata
        self.assertEqual(metadata["network_mode"], "isolated")
        self.assertTrue(metadata.get("network_namespace_path"))
        self.assertTrue(metadata.get("guest_ip"))

    def test_required_image_and_network_dns_matrix(self) -> None:
        for image in ("ubuntu:22.04", "python:3.11-slim"):
            for network in (False, True):
                with self.subTest(image=image, network=network):
                    sandbox = self._sandbox(image, network=network)
                    result = sandbox.commands.run(
                        ["getent", "hosts", "archive.ubuntu.com"],
                        timeout=60,
                        check=True,
                    )
                    self.assertTrue(result.stdout.strip())
                    self.assertEqual(
                        sandbox.describe().metadata["network_mode"],
                        "isolated" if network else "host",
                    )

    def test_dns_file_is_sandbox_state_across_lifecycle_and_fork(self) -> None:
        for network in (False, True):
            with self.subTest(network=network):
                sandbox = self._sandbox("ubuntu:22.04", network=network)
                initial = "nameserver 192.0.2.53\n"
                sandbox.files.write("/etc/resolv.conf", initial)
                checkpoint_id = sandbox.checkpoint()
                sandbox.files.write(
                    "/etc/resolv.conf", "nameserver 192.0.2.54\n"
                )
                sandbox.restore(checkpoint_id)
                self.assertEqual(sandbox.files.read("/etc/resolv.conf"), initial)
                forks = sandbox.fork(1, checkpoint_id=checkpoint_id)
                fork = forks[0]
                self.addCleanup(fork.kill)
                self.assertEqual(fork.files.read("/etc/resolv.conf"), initial)
                expected_mode = "isolated" if network else "host"
                fork_metadata = fork.describe().metadata
                self.assertEqual(fork_metadata["network_mode"], expected_mode)
                if network:
                    source_metadata = sandbox.describe().metadata
                    self.assertNotEqual(
                        fork_metadata.get("guest_ip"),
                        source_metadata.get("guest_ip"),
                    )
                    self.assertNotEqual(
                        fork_metadata.get("network_namespace_path"),
                        source_metadata.get("network_namespace_path"),
                    )
                sandbox.stop()
                sandbox.start()
                self.assertEqual(sandbox.files.read("/etc/resolv.conf"), initial)
                self.assertEqual(
                    sandbox.describe().metadata["network_mode"], expected_mode
                )

    def test_direct_runtime_timeout_reaps_complete_payload(self) -> None:
        sandbox = self._sandbox(
            "ubuntu:22.04",
            network=False,
            engine=self.local_engine,
        )
        runtime = self.local_engine.runtime
        with self.assertRaises(SandboxExecTimeout):
            runtime.exec(
                sandbox.sandbox_id,
                [
                    "sh",
                    "-c",
                    "echo $$ >/tmp/direct-shell.pid; sleep 30 & "
                    "echo $! >/tmp/direct-child.pid; wait",
                ],
                timeout_s=1,
            )
        self.assertTrue(
            self.local_engine.system.inspector.inspect(
                sandbox.sandbox_id
            ).filesystem_changed,
            "timeout-isolated exec must conservatively invalidate the "
            "host-inspector clean baseline",
        )
        probe = sandbox.commands.run(
            [
                "sh",
                "-c",
                "test ! -e /proc/$(cat /tmp/direct-shell.pid) && "
                "test ! -e /proc/$(cat /tmp/direct-child.pid)",
            ],
            check=False,
        )
        self.assertEqual(probe.returncode, 0)

    def test_timeout_kills_descendants_but_not_sibling_exec(self) -> None:
        sandbox = self._sandbox("ubuntu:22.04", network=False)
        sibling_result: list[object] = []

        def sibling() -> None:
            sibling_result.append(
                sandbox.commands.run(
                    [
                        "sh",
                        "-c",
                        "echo $$ >/tmp/sibling.pid; exec sleep 30",
                    ],
                    timeout=40,
                    check=False,
                )
            )

        sibling_thread = threading.Thread(target=sibling)
        sibling_thread.start()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if sandbox.files.exists("/tmp/sibling.pid"):
                break
            time.sleep(0.05)
        else:
            self.fail("sibling exec did not start")

        with self.assertRaises(SandboxExecTimeout):
            sandbox.commands.run(
                [
                    "sh",
                    "-c",
                    "echo $$ >/tmp/target-shell.pid; sleep 30 & "
                    "echo $! >/tmp/target-child.pid; wait",
                ],
                timeout=1,
            )

        probes = sandbox.commands.run(
            [
                "sh",
                "-c",
                "for item in target-shell target-child sibling; do "
                "pid=$(cat /tmp/$item.pid); "
                "if test -e /proc/$pid; then "
                "echo \"$item alive pid=$pid\"; "
                "grep '^State:' /proc/$pid/status; "
                "cat /proc/$pid/cgroup; "
                "else echo \"$item gone pid=$pid\"; fi; done; "
                "test ! -e /proc/$(cat /tmp/target-shell.pid) && "
                "test ! -e /proc/$(cat /tmp/target-child.pid) && "
                "test -e /proc/$(cat /tmp/sibling.pid)",
            ],
            check=False,
        )
        self.assertEqual(
            probes.returncode,
            0,
            f"stdout:\n{probes.stdout}\nstderr:\n{probes.stderr}",
        )
        sandbox.commands.run(
            ["sh", "-c", "kill $(cat /tmp/sibling.pid)"], check=False
        )
        sibling_thread.join(timeout=10)
        self.assertFalse(sibling_thread.is_alive())
        self.assertEqual(len(sibling_result), 1)

    def test_stream_and_action_share_timeout_contract(self) -> None:
        sandbox = self._sandbox("ubuntu:22.04", network=False)
        with self.assertRaises(SandboxExecTimeout):
            list(
                sandbox.commands.stream(
                    ["sh", "-c", "echo streaming; sleep 30 & wait"], timeout=1
                )
            )
        previous = sandbox._last_checkpoint_id
        with self.assertRaises(SandboxExecTimeout):
            sandbox.commands.run(
                ["sh", "-c", "sleep 30 & wait"],
                timeout=1,
                checkpoint=True,
            )
        self.assertEqual(sandbox._last_checkpoint_id, previous)

    def test_action_checkpoint_returns_stable_logical_id_and_reuses_clean_state(self) -> None:
        sandbox = self._sandbox("ubuntu:22.04", network=False)

        first = sandbox.commands.run(
            "echo logical-state >/tmp/logical-state",
            checkpoint=True,
            observe=True,
        )
        assert first.checkpoint is not None
        first_id = first.checkpoint.checkpoint_id
        self.assertEqual(first.checkpoint.wait(timeout=120.0), first_id)
        self.assertEqual(first.checkpoint.materialization, "full")
        self.assertTrue(first.checkpoint.physical_checkpoint_created)

        clean = sandbox.commands.run("true", checkpoint=True, observe=True)
        assert clean.checkpoint is not None
        clean_id = clean.checkpoint.checkpoint_id
        self.assertNotEqual(clean_id, first_id)
        self.assertEqual(clean.checkpoint.wait(timeout=120.0), clean_id)
        self.assertEqual(clean.checkpoint.materialization, "reused")
        self.assertFalse(clean.checkpoint.physical_checkpoint_created)

        listed = {
            str(item["checkpoint_id"]): item
            for item in sandbox.checkpoints.list()
        }
        self.assertEqual(listed[clean_id]["materialization"], "reused")
        self.assertFalse(listed[clean_id]["materialized_process"])
        self.assertFalse(listed[clean_id]["materialized_filesystem"])
        self.assertEqual(listed[clean_id]["process_checkpoint_id"], first_id)
        self.assertEqual(listed[clean_id]["filesystem_checkpoint_id"], first_id)

        fork = sandbox.fork(count=1, checkpoint_id=clean_id)[0]
        self.addCleanup(fork.kill)
        forked = fork.commands.run("cat /tmp/logical-state", check=True)
        self.assertEqual(forked.stdout.strip(), "logical-state")
        fork.kill()

        sandbox.restore(clean_id)
        restored = sandbox.commands.run("cat /tmp/logical-state", check=True)
        self.assertEqual(restored.stdout.strip(), "logical-state")

        filesystem_change = sandbox.commands.run(
            "echo filesystem-only >/tmp/filesystem-only",
            checkpoint=True,
            observe=True,
        )
        assert filesystem_change.checkpoint is not None
        filesystem_id = filesystem_change.checkpoint.checkpoint_id
        self.assertEqual(
            filesystem_change.checkpoint.wait(timeout=120.0), filesystem_id
        )
        self.assertEqual(
            filesystem_change.checkpoint.materialization, "filesystem_only"
        )
        self.assertTrue(
            filesystem_change.checkpoint.physical_checkpoint_created
        )
        listed = {
            str(item["checkpoint_id"]): item
            for item in sandbox.checkpoints.list()
        }
        self.assertEqual(
            listed[filesystem_id]["process_checkpoint_id"], first_id
        )
        self.assertEqual(
            listed[filesystem_id]["filesystem_checkpoint_id"], filesystem_id
        )
        self.assertFalse(listed[filesystem_id]["materialized_process"])
        self.assertTrue(listed[filesystem_id]["materialized_filesystem"])

        sandbox.restore(filesystem_id)
        restored = sandbox.commands.run(
            "cat /tmp/logical-state /tmp/filesystem-only", check=True
        )
        self.assertEqual(
            restored.stdout.splitlines(),
            ["logical-state", "filesystem-only"],
        )

    def test_port_exposure_rejects_host_and_works_in_isolated_mode(self) -> None:
        host = self._sandbox("python:3.12-slim", network=False)
        with self.assertRaises(CloudRequestError) as caught:
            host.ports.expose(8080)
        self.assertEqual(caught.exception.status_code, 400)

        for network in (None, True):
            with self.subTest(network=network):
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                    probe.bind(("127.0.0.1", 0))
                    guest_port = int(probe.getsockname()[1])
                sandbox = self._sandbox("python:3.12-slim", network=network)
                sandbox.commands.run(
                    "python -m http.server "
                    f"{guest_port} --bind 0.0.0.0 "
                    ">/tmp/crab-http.log 2>&1 & "
                    "echo $! >/tmp/crab-http.pid",
                    detach=True,
                    timeout=30,
                    check=True,
                )
                readiness = sandbox.commands.run(
                    [
                        "python",
                        "-c",
                        "import socket,time; "
                        "deadline=time.monotonic()+10; error=None; "
                        f"address=('127.0.0.1',{guest_port}); "
                        "\nwhile time.monotonic()<deadline:\n"
                        " try:\n"
                        "  s=socket.create_connection(address,1); s.close(); break\n"
                        " except OSError as exc:\n"
                        "  error=exc; time.sleep(.05)\n"
                        "else: raise error",
                    ],
                    timeout=15,
                    check=False,
                )
                if readiness.returncode != 0:
                    diagnostics = sandbox.commands.run(
                        [
                            "sh",
                            "-c",
                            "pid=$(cat /tmp/crab-http.pid 2>/dev/null || true); "
                            "echo pid=$pid; "
                            "test -n \"$pid\" && "
                            "cat /proc/$pid/status /proc/$pid/cgroup 2>&1 || true; "
                            "echo log:; cat /tmp/crab-http.log 2>&1 || true",
                        ],
                        timeout=15,
                        check=False,
                    )
                    self.fail(
                        "guest HTTP server was not reachable: "
                        f"{readiness.stderr}\n{diagnostics.stdout}\n"
                        f"{diagnostics.stderr}"
                    )
                metadata = sandbox.describe().metadata
                direct_ip = str(metadata.get("guest_ip") or "127.0.0.1")
                direct_error: BaseException | None = None
                direct_deadline = time.monotonic() + 10
                while time.monotonic() < direct_deadline:
                    try:
                        with socket.create_connection(
                            (direct_ip, guest_port), timeout=1
                        ):
                            pass
                        break
                    except OSError as exc:
                        direct_error = exc
                        time.sleep(0.05)
                else:
                    self.fail(
                        "guest port was not reachable from gateway host: "
                        f"address={direct_ip}:{guest_port} error={direct_error} "
                        f"metadata={metadata}"
                    )
                allocation = sandbox.ports.expose(guest_port)
                deadline = time.monotonic() + 30
                last_error: BaseException | None = None
                while time.monotonic() < deadline:
                    try:
                        with urllib.request.urlopen(
                            f"http://127.0.0.1:{allocation.host_port}", timeout=5
                        ) as response:
                            status = response.status
                            body = response.read()
                        self.assertEqual(status, 200)
                        self.assertTrue(body)
                        break
                    except Exception as exc:
                        last_error = exc
                        time.sleep(0.1)
                else:
                    forwarder = self.gateway.port_manager._forwarders.get(
                        allocation.host_port
                    )
                    forward_target = (
                        None
                        if forwarder is None
                        else f"{forwarder.guest_ip}:{forwarder.guest_port}"
                    )
                    self.fail(
                        f"port exposure did not become ready network={network}: "
                        f"{last_error}; direct={direct_ip}:{guest_port}; "
                        f"forwarder={forward_target}; metadata={metadata}"
                    )

    def test_nonexistent_image_is_typed_and_rolls_back_create(self) -> None:
        runtime = self.local_engine.runtime
        bundle_root = runtime.paths.bundle_root
        before_sandboxes = set(self.daemon.sandbox_ids())
        before_bundles = (
            {entry.name for entry in bundle_root.iterdir() if entry.is_dir()}
            if bundle_root.is_dir()
            else set()
        )
        reference = f"crab-image-must-not-exist-{uuid.uuid4().hex}:latest"

        with self.assertRaises(ImageNotFoundError):
            Sandbox(image=reference, network=False, engine=self.engine)

        self.assertEqual(set(self.daemon.sandbox_ids()), before_sandboxes)
        self.assertEqual(
            (
                {entry.name for entry in bundle_root.iterdir() if entry.is_dir()}
                if bundle_root.is_dir()
                else set()
            ),
            before_bundles,
        )
        self.assertEqual(self.gateway.registry.pending_rows(), [])

    def test_public_image_matrix_and_digest_pin(self) -> None:
        commands = {
            "ubuntu:22.04": ["sh", "-c", "cat /etc/os-release | head -1"],
            "ubuntu:24.04": ["sh", "-c", "cat /etc/os-release | head -1"],
            "debian:12-slim": ["sh", "-c", "cat /etc/os-release | head -1"],
            "python:3.11-slim": ["python", "--version"],
            "python:3.12-slim": ["python", "--version"],
            "node:20-bookworm-slim": ["node", "--version"],
        }
        first_metadata = None
        for image, command in commands.items():
            with self.subTest(image=image):
                sandbox = Sandbox(image=image, network=False, engine=self.engine)
                try:
                    result = sandbox.commands.run(command, timeout=60, check=True)
                    self.assertTrue((result.stdout or result.stderr).strip())
                    metadata = sandbox.describe().metadata
                    self.assertTrue(metadata.get("image_id"))
                    self.assertTrue(metadata.get("image_digest"))
                    if image == "python:3.12-slim":
                        first_metadata = metadata
                finally:
                    sandbox.kill()
        assert first_metadata is not None
        digest_ref = f"python@{first_metadata['image_digest']}"
        pinned = self._sandbox(digest_ref, network=False)
        pinned_metadata = pinned.describe().metadata
        self.assertEqual(
            pinned_metadata["image_digest"], first_metadata["image_digest"]
        )
        self.assertEqual(pinned_metadata["image_id"], first_metadata["image_id"])


if __name__ == "__main__":
    unittest.main()
