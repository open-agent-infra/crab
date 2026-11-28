from __future__ import annotations

import json
import os
import shlex
import shutil
import socket
import statistics
import subprocess
import tempfile
import threading
import time
import unittest
import urllib.request
from pathlib import Path

from crab import (
    CrabRequestInterceptorServer,
    CrabSystem,
    AdapterFileSystemCWorker,
    AdapterFileSystemRWorker,
    AdapterProcessCWorker,
    AdapterProcessRWorker,
    CRExecutor,
    CRScheduler,
    CompositeRequestInterceptorHook,
    DefaultCWorker,
    DefaultRWorker,
    ExecutorConfig,
    HostInspectorServiceClient,
    InMemoryRequestStateStore,
    InMemorySchedulerStateStore,
    InMemoryTelemetrySink,
    LocalCheckpointManager,
    RemoteSandboxInspector,
    RequestAwareSandboxInspector,
    RuncRuntime,
    RuncRuntimePaths,
    SandboxId,
    SchedulerConfig,
    StorageConfig,
    TelemetryRequestInterceptorHook,
)
from crab.host_inspector.fs_helper import LibbpfFilesystemMonitor
from crab.host_inspector.runtime_resolver import RuntimeResolver
from crab.host_inspector.server import HostInspectorDaemon, HostInspectorServer
from crab.models import JobStatus, utc_now
from integrations.llm_services.simulated.service import serve
from integrations.sandboxes.runtime.image import build_image, export_image_rootfs
from integrations.sandboxes.simulated import DOCKERFILE_PATH as SIMULATED_DOCKERFILE_PATH

RuncRuntimeAdapter = RuncRuntime
RuncSandboxManager = RuncRuntime
RuncSandboxManagerPaths = RuncRuntimePaths


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_http_json(url: str, *, timeout_s: float = 20.0) -> dict[str, object]:
    deadline = time.time() + timeout_s
    last_exc: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2.0) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # pragma: no cover - real host only
            last_exc = exc
            time.sleep(0.2)
    raise RuntimeError(f"timed out waiting for {url}: {last_exc}")


def _wait_for(predicate, *, timeout_s: float = 20.0, interval_s: float = 0.2) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(interval_s)
    raise RuntimeError("timed out waiting for predicate")


def _helper_binary() -> Path:
    return Path(__file__).resolve().parents[1] / "crab" / "host_inspector" / "bpf" / "fs_monitor"


def _ensure_helper_built() -> Path:
    helper = _helper_binary()
    subprocess.run(["make"], cwd=helper.parent, check=True)
    return helper


def _docker_image_for_tests() -> str | None:
    preferred = "agent-sandbox-bench:latest"
    result = subprocess.run(
        ["docker", "image", "inspect", preferred],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode == 0:
        return preferred
    return None


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    position = (len(ordered) - 1) * pct
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


class HostInspectorRealIntegrationTests(unittest.TestCase):
    def _require_docker_host_inspector_prereqs(self) -> tuple[str, Path]:
        if os.geteuid() != 0 or shutil.which("docker") is None:
            self.skipTest("root and docker required")

        image = _docker_image_for_tests()
        if image is None:
            self.skipTest("agent-sandbox-bench:latest image not available")
        return image, _ensure_helper_built()

    def _safe_unregister(self, client: HostInspectorServiceClient, sandbox_id: SandboxId) -> None:
        try:
            client.unregister_sandbox(sandbox_id)
        except Exception:
            pass

    def _start_docker_host_inspector_fixture(
        self,
        *,
        name_prefix: str,
        sandbox_name: str,
        process_poll_interval_s: float = 0.2,
    ) -> tuple[HostInspectorServer, HostInspectorServiceClient, RemoteSandboxInspector, str, SandboxId]:
        image, helper_path = self._require_docker_host_inspector_prereqs()
        suffix = int(time.time() * 1000)
        container_name = f"{name_prefix}-{suffix}"
        sandbox_id = SandboxId(sandbox_name)
        server = HostInspectorServer(
            host="127.0.0.1",
            port=_find_free_port(),
            daemon=HostInspectorDaemon(
                process_poll_interval_s=process_poll_interval_s,
                fs_monitor=LibbpfFilesystemMonitor(helper_path=str(helper_path)),
            ),
        )
        server.start()
        self.addCleanup(server.stop)
        _wait_for_http_json(f"http://127.0.0.1:{server.port}/healthz")

        subprocess.run(["docker", "run", "-d", "--name", container_name, image, "sleep", "10000000"], check=True)
        self.addCleanup(
            lambda: subprocess.run(
                ["docker", "rm", "-f", container_name],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        )

        client = HostInspectorServiceClient(f"http://127.0.0.1:{server.port}")
        client.register_sandbox(sandbox_id, "docker", container_name)
        self.addCleanup(self._safe_unregister, client, sandbox_id)
        return server, client, RemoteSandboxInspector(client), container_name, sandbox_id

    def _status_for(self, client: HostInspectorServiceClient, sandbox_id: SandboxId) -> dict[str, object]:
        return dict(client.get_proc_and_fs_status(sandbox_id)["status"])

    def _wait_for_status(
        self,
        client: HostInspectorServiceClient,
        sandbox_id: SandboxId,
        *,
        predicate,
        timeout_s: float = 12.0,
        interval_s: float = 0.1,
    ) -> dict[str, object]:
        deadline = time.time() + timeout_s
        last_status: dict[str, object] | None = None
        while time.time() < deadline:
            last_status = self._status_for(client, sandbox_id)
            if predicate(last_status):
                return last_status
            time.sleep(interval_s)
        raise RuntimeError(f"timed out waiting for sandbox status; last_status={last_status}")

    def _reset_and_wait_clear(self, client: HostInspectorServiceClient, sandbox_id: SandboxId) -> dict[str, object]:
        client.reset_sandbox(sandbox_id, utc_now())
        return self._wait_for_status(
            client,
            sandbox_id,
            predicate=lambda status: not bool(status["process_changed"]) and not bool(status["filesystem_changed"]),
            timeout_s=12.0,
            interval_s=0.1,
        )

    def _cleanup_case_paths(self, container_name: str, paths: list[str]) -> None:
        if not paths:
            return
        cleanup_cmd = "rm -rf " + " ".join(shlex.quote(path) for path in paths)
        subprocess.run(
            ["docker", "exec", container_name, "sh", "-lc", cleanup_cmd],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def _run_exec_case(
        self,
        *,
        client: HostInspectorServiceClient,
        sandbox_id: SandboxId,
        container_name: str,
        name: str,
        argv: list[str],
        expected_process_changed: bool,
        expected_filesystem_changed: bool,
        cleanup_paths: list[str] | None = None,
    ) -> dict[str, object]:
        self._cleanup_case_paths(container_name, cleanup_paths or [])
        self._reset_and_wait_clear(client, sandbox_id)

        proc = subprocess.Popen(
            ["docker", "exec", container_name, *argv],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            status = self._wait_for_status(
                client,
                sandbox_id,
                predicate=lambda payload: (
                    bool(payload["process_changed"]) == expected_process_changed
                    and bool(payload["filesystem_changed"]) == expected_filesystem_changed
                ),
                timeout_s=12.0,
                interval_s=0.1,
            )
        except Exception as exc:
            stdout, stderr = proc.communicate(timeout=10.0)
            raise AssertionError(
                f"{name} failed to reach expected status\n"
                f"argv={argv}\n"
                f"stdout={stdout!r}\n"
                f"stderr={stderr!r}\n"
                f"error={exc}"
            ) from exc

        stdout, stderr = proc.communicate(timeout=10.0)
        self.assertEqual(
            proc.returncode,
            0,
            f"{name} command failed\nargv={argv}\nstdout={stdout}\nstderr={stderr}",
        )
        self._cleanup_case_paths(container_name, cleanup_paths or [])
        return status

    def _measure_inspect_latencies(
        self,
        inspector: RemoteSandboxInspector,
        sandbox_id: SandboxId,
        *,
        iterations: int,
    ) -> tuple[list[float], float]:
        latencies_ms: list[float] = []
        started = time.perf_counter()
        for _ in range(iterations):
            call_started = time.perf_counter()
            snapshot = inspector.inspect(sandbox_id)
            latencies_ms.append((time.perf_counter() - call_started) * 1000.0)
            self.assertEqual(snapshot.sandbox_id, sandbox_id)
            self.assertNotIn("inspector_error", snapshot.metadata)
        return latencies_ms, time.perf_counter() - started

    def _latency_summary(self, latencies_ms: list[float]) -> dict[str, float]:
        return {
            "min_ms": min(latencies_ms),
            "mean_ms": statistics.mean(latencies_ms),
            "p50_ms": _percentile(latencies_ms, 0.50),
            "p95_ms": _percentile(latencies_ms, 0.95),
            "max_ms": max(latencies_ms),
        }

    def test_real_container_daemon_remote_inspector_watch(self) -> None:
        if os.geteuid() != 0 or shutil.which("docker") is None:
            self.skipTest("root and docker required")

        image = _docker_image_for_tests()
        if image is None:
            self.skipTest("agent-sandbox-bench:latest image not available")

        helper_path = _ensure_helper_built()
        container_name = f"crab-watch-{int(time.time())}"
        sandbox_id = SandboxId("watch-sbx")
        server = HostInspectorServer(
            host="127.0.0.1",
            port=_find_free_port(),
            daemon=HostInspectorDaemon(
                process_poll_interval_s=0.2,
                fs_monitor=LibbpfFilesystemMonitor(helper_path=str(helper_path)),
            ),
        )
        server.start()
        self.addCleanup(server.stop)

        subprocess.run(["docker", "run", "-d", "--name", container_name, image, "sleep", "10000000"], check=True)
        self.addCleanup(
            lambda: subprocess.run(
                ["docker", "rm", "-f", container_name],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        )

        client = HostInspectorServiceClient(f"http://127.0.0.1:{server.port}")
        client.register_sandbox(sandbox_id, "docker", container_name)
        client.reset_sandbox(sandbox_id, utc_now())

        def mutate() -> None:
            time.sleep(1.4)
            subprocess.run(
                [
                    "docker",
                    "exec",
                    "-d",
                    container_name,
                    "python3",
                    "-c",
                    "import time; buf=bytearray(8*1024*1024); buf[4096]=1; "
                    "open('/tmp/watch-it.txt','w',encoding='utf-8').write('x'); time.sleep(3)",
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        worker = threading.Thread(target=mutate, daemon=True)
        worker.start()

        watch_proc = subprocess.run(
            [
                "python3",
                "-m",
                "crab.host_inspector.watch",
                "--base-url",
                f"http://127.0.0.1:{server.port}",
                "--interval",
                "1",
                "--iterations",
                "5",
                str(sandbox_id),
            ],
            check=True,
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
        )
        worker.join(timeout=5.0)

        lines = [line.strip() for line in watch_proc.stdout.splitlines() if line.strip()]
        self.assertTrue(any("process_changed=False filesystem_changed=False" in line for line in lines), lines)
        self.assertTrue(any("process_changed=True" in line or "filesystem_changed=True" in line for line in lines), lines)

        client.unregister_sandbox(sandbox_id)

    def test_real_docker_command_matrix_for_process_and_filesystem_changes(self) -> None:
        _, client, _, container_name, sandbox_id = self._start_docker_host_inspector_fixture(
            name_prefix="crab-matrix",
            sandbox_name="matrix-sbx",
            process_poll_interval_s=0.1,
        )

        cases = [
            {
                "name": "stdout_only_stdout",
                "argv": ["sh", "-lc", "echo 123; sleep 1"],
                "expected_process_changed": True,
                "expected_filesystem_changed": False,
                "cleanup_paths": [],
            },
            {
                "name": "stdout_only_stderr",
                "argv": ["sh", "-lc", "echo err 1>&2; sleep 1"],
                "expected_process_changed": True,
                "expected_filesystem_changed": False,
                "cleanup_paths": [],
            },
            {
                "name": "read_only_cat",
                "argv": ["sh", "-lc", "cat /etc/hostname; sleep 1"],
                "expected_process_changed": True,
                "expected_filesystem_changed": False,
                "cleanup_paths": [],
            },
            {
                "name": "read_only_ls",
                "argv": ["sh", "-lc", "ls -la /tmp >/dev/null; sleep 1"],
                "expected_process_changed": True,
                "expected_filesystem_changed": False,
                "cleanup_paths": [],
            },
            {
                "name": "memory_only_python",
                "argv": ["python3", "-B", "-c", "import time; buf=bytearray(8*1024*1024); buf[4096]=1; time.sleep(2)"],
                "expected_process_changed": True,
                "expected_filesystem_changed": False,
                "cleanup_paths": [],
            },
            {
                "name": "tmp_file_write",
                "argv": ["sh", "-lc", "echo 123 >/tmp/hi-out.txt; sleep 1"],
                "expected_process_changed": True,
                "expected_filesystem_changed": True,
                "cleanup_paths": ["/tmp/hi-out.txt"],
            },
            {
                "name": "root_file_write",
                "argv": ["sh", "-lc", "echo 123 >/root/hi-root.txt; sleep 1"],
                "expected_process_changed": True,
                "expected_filesystem_changed": True,
                "cleanup_paths": ["/root/hi-root.txt"],
            },
            {
                "name": "tmp_mkdir",
                "argv": ["sh", "-lc", "mkdir -p /tmp/hi-dir; sleep 1"],
                "expected_process_changed": True,
                "expected_filesystem_changed": True,
                "cleanup_paths": ["/tmp/hi-dir"],
            },
            {
                "name": "root_mkdir",
                "argv": ["sh", "-lc", "mkdir -p /root/hi-root-dir; sleep 1"],
                "expected_process_changed": True,
                "expected_filesystem_changed": True,
                "cleanup_paths": ["/root/hi-root-dir"],
            },
            {
                "name": "workspace_file_write",
                "argv": ["sh", "-lc", "mkdir -p /workspace && echo 123 >/workspace/hi-workspace.txt; sleep 1"],
                "expected_process_changed": True,
                "expected_filesystem_changed": True,
                "cleanup_paths": ["/workspace/hi-workspace.txt"],
            },
            {
                "name": "rename_move",
                "argv": ["sh", "-lc", "echo abc >/tmp/hi-a.txt && mv /tmp/hi-a.txt /tmp/hi-b.txt; sleep 1"],
                "expected_process_changed": True,
                "expected_filesystem_changed": True,
                "cleanup_paths": ["/tmp/hi-a.txt", "/tmp/hi-b.txt"],
            },
            {
                "name": "hard_link",
                "argv": ["sh", "-lc", "echo abc >/tmp/hi-link-src.txt && ln /tmp/hi-link-src.txt /tmp/hi-link-hard.txt; sleep 1"],
                "expected_process_changed": True,
                "expected_filesystem_changed": True,
                "cleanup_paths": ["/tmp/hi-link-src.txt", "/tmp/hi-link-hard.txt"],
            },
            {
                "name": "soft_link",
                "argv": ["sh", "-lc", "echo abc >/tmp/hi-symlink-src.txt && ln -s /tmp/hi-symlink-src.txt /tmp/hi-link-soft.txt; sleep 1"],
                "expected_process_changed": True,
                "expected_filesystem_changed": True,
                "cleanup_paths": ["/tmp/hi-symlink-src.txt", "/tmp/hi-link-soft.txt"],
            },
            {
                "name": "tmp_rm",
                "argv": ["sh", "-lc", "echo abc >/tmp/hi-rm.txt && rm /tmp/hi-rm.txt; sleep 1"],
                "expected_process_changed": True,
                "expected_filesystem_changed": False,
                "cleanup_paths": ["/tmp/hi-rm.txt"],
            },
            {
                "name": "workspace_rm",
                "argv": ["sh", "-lc", "mkdir -p /workspace && echo abc >/workspace/hi-rm.txt && rm /workspace/hi-rm.txt; sleep 1"],
                "expected_process_changed": True,
                "expected_filesystem_changed": False,
                "cleanup_paths": ["/workspace/hi-rm.txt"],
            },
            {
                "name": "tmp_rmdir",
                "argv": ["sh", "-lc", "mkdir -p /tmp/hi-rmdir && rmdir /tmp/hi-rmdir; sleep 1"],
                "expected_process_changed": True,
                "expected_filesystem_changed": False,
                "cleanup_paths": ["/tmp/hi-rmdir"],
            },
            {
                "name": "root_rmdir",
                "argv": ["sh", "-lc", "mkdir -p /root/hi-rmdir && rmdir /root/hi-rmdir; sleep 1"],
                "expected_process_changed": True,
                "expected_filesystem_changed": False,
                "cleanup_paths": ["/root/hi-rmdir"],
            },
        ]

        for case in cases:
            with self.subTest(case=case["name"]):
                status = self._run_exec_case(
                    client=client,
                    sandbox_id=sandbox_id,
                    container_name=container_name,
                    name=str(case["name"]),
                    argv=list(case["argv"]),
                    expected_process_changed=bool(case["expected_process_changed"]),
                    expected_filesystem_changed=bool(case["expected_filesystem_changed"]),
                    cleanup_paths=list(case["cleanup_paths"]),
                )
                self.assertTrue(status["is_running"], f"container should still be running after {case['name']}")

    def test_real_docker_remote_inspect_latency(self) -> None:
        if os.environ.get("CRAB_RUN_PERF") != "1":
            self.skipTest("set CRAB_RUN_PERF=1 to run host inspector perf test")

        _, client, inspector, container_name, sandbox_id = self._start_docker_host_inspector_fixture(
            name_prefix="crab-perf",
            sandbox_name="perf-sbx",
            process_poll_interval_s=0.1,
        )
        self._cleanup_case_paths(container_name, ["/workspace/hi-perf"])
        self._reset_and_wait_clear(client, sandbox_id)

        self._measure_inspect_latencies(inspector, sandbox_id, iterations=20)

        idle_latencies_ms, idle_duration_s = self._measure_inspect_latencies(inspector, sandbox_id, iterations=200)
        idle_summary = self._latency_summary(idle_latencies_ms)
        self.assertLess(idle_duration_s, 30.0, f"idle inspect sweep too slow: {idle_duration_s:.3f}s")

        subprocess.run(
            [
                "docker",
                "exec",
                "-d",
                container_name,
                "python3",
                "-c",
                (
                    "import os, time; os.makedirs('/workspace/hi-perf', exist_ok=True); buffers=[]; end=time.time()+8.0; i=0; "
                    "while time.time() < end: "
                    " data=bytearray(2*1024*1024); data[4096]=i % 251; buffers.append(data); "
                    " open(f'/workspace/hi-perf/{i}.txt', 'w', encoding='utf-8').write('x'*32); "
                    " buffers=buffers[-4:]; i += 1; time.sleep(0.05)"
                ),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(0.4)

        active_latencies_ms, active_duration_s = self._measure_inspect_latencies(inspector, sandbox_id, iterations=200)
        active_summary = self._latency_summary(active_latencies_ms)
        self.assertLess(active_duration_s, 30.0, f"active inspect sweep too slow: {active_duration_s:.3f}s")

        print(f"idle inspect latency ms: {idle_summary}")
        print(f"active inspect latency ms: {active_summary}")

    def test_real_docker_process_changed_is_evaluated_at_query_time(self) -> None:
        _, client, _, container_name, sandbox_id = self._start_docker_host_inspector_fixture(
            name_prefix="crab-proc-now",
            sandbox_name="proc-now-sbx",
            process_poll_interval_s=0.1,
        )

        self._reset_and_wait_clear(client, sandbox_id)
        subprocess.run(
            ["docker", "exec", container_name, "sh", "-lc", "sleep 0.5"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        status_after_transient = self._status_for(client, sandbox_id)
        self.assertFalse(status_after_transient["process_changed"], status_after_transient)
        self.assertFalse(status_after_transient["filesystem_changed"], status_after_transient)

        self._reset_and_wait_clear(client, sandbox_id)
        live_proc = subprocess.Popen(
            ["docker", "exec", container_name, "sh", "-lc", "sleep 2"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            live_status = self._wait_for_status(
                client,
                sandbox_id,
                predicate=lambda payload: bool(payload["process_changed"]),
                timeout_s=8.0,
                interval_s=0.1,
            )
        finally:
            stdout, stderr = live_proc.communicate(timeout=10.0)
            self.assertEqual(live_proc.returncode, 0, f"live sleep failed\nstdout={stdout}\nstderr={stderr}")
        self.assertTrue(live_status["process_changed"], live_status)
        self.assertFalse(live_status["filesystem_changed"], live_status)

        status_after_exit = self._wait_for_status(
            client,
            sandbox_id,
            predicate=lambda payload: not bool(payload["process_changed"]),
            timeout_s=8.0,
            interval_s=0.1,
        )
        self.assertFalse(status_after_exit["filesystem_changed"], status_after_exit)

        self._reset_and_wait_clear(client, sandbox_id)
        subprocess.run(
            [
                "docker",
                "exec",
                container_name,
                "python3",
                "-B",
                "-c",
                "buf=bytearray(8*1024*1024); buf[4096]=1",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        status_after_transient_dirty = self._status_for(client, sandbox_id)
        self.assertFalse(status_after_transient_dirty["process_changed"], status_after_transient_dirty)
        self.assertFalse(status_after_transient_dirty["filesystem_changed"], status_after_transient_dirty)

    def test_real_runc_criu_zfs_checkpoint_restore_with_remote_inspector(self) -> None:
        if (
            os.geteuid() != 0
            or shutil.which("docker") is None
            or shutil.which("runc") is None
            or shutil.which("criu") is None
            or shutil.which("zfs") is None
        ):
            self.skipTest("root plus docker/runc/criu/zfs required")

        helper_path = _ensure_helper_built()
        tmpdir = tempfile.TemporaryDirectory(prefix="crab_remote_real_it_")
        self.addCleanup(tmpdir.cleanup)
        root = Path(tmpdir.name)
        sandbox_id = SandboxId("sbx-real-remote")
        pool_name = f"crabremote{int(time.time())}"
        provider = "openai"
        image_tag = f"crab-simulated-agent-remote:{int(time.time())}"
        executor: CRExecutor | None = None

        llm_server = serve(host="127.0.0.1", port=0, response_delay_ms=250)
        llm_thread = threading.Thread(target=llm_server.serve_forever, daemon=True)
        llm_thread.start()
        self.addCleanup(llm_server.shutdown)
        self.addCleanup(llm_server.server_close)
        self.addCleanup(llm_thread.join, 5.0)

        try:
            pool_file = root / "zpool.img"
            bundle_dir = root / "bundles" / str(sandbox_id)
            runtime_state_root = root / "runtime-state"
            checkpoint_root = root / "checkpoints"
            storage_root = root / "storage"
            sandbox_metadata_root = root / "sandbox-meta"
            status_port = _find_free_port()
            image_root = root / "image"

            inspector_server = HostInspectorServer(
                host="127.0.0.1",
                port=_find_free_port(),
                daemon=HostInspectorDaemon(
                    resolver=RuntimeResolver(runc_state_root=runtime_state_root),
                    process_poll_interval_s=0.2,
                    fs_monitor=LibbpfFilesystemMonitor(helper_path=str(helper_path)),
                ),
            )
            inspector_server.start()
            self.addCleanup(inspector_server.stop)
            host_client = HostInspectorServiceClient(f"http://127.0.0.1:{inspector_server.port}")

            build_image(
                tag=image_tag,
                build_context=SIMULATED_DOCKERFILE_PATH.parent,
                dockerfile_path=SIMULATED_DOCKERFILE_PATH,
            )
            self.addCleanup(
                lambda: subprocess.run(
                    ["docker", "rmi", "-f", image_tag],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            )
            exported_rootfs = export_image_rootfs(tag=image_tag, output_dir=image_root)

            subprocess.run(["truncate", "-s", "512M", str(pool_file)], check=True)
            subprocess.run(["zpool", "create", "-f", pool_name, str(pool_file)], check=True)
            self.addCleanup(
                lambda: subprocess.run(
                    ["zpool", "destroy", "-f", pool_name],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            )
            subprocess.run(["zfs", "create", f"{pool_name}/crab"], check=True)

            bundle_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run(["runc", "spec"], cwd=bundle_dir, check=True)

            telemetry = InMemoryTelemetrySink()
            request_state_store = InMemoryRequestStateStore()
            base_inspector = RemoteSandboxInspector(host_client)
            inspector = RequestAwareSandboxInspector(base_inspector, request_state_store)
            runtime = RuncRuntimeAdapter(
                paths=RuncRuntimePaths(
                    state_root=runtime_state_root,
                    bundle_root=root / "bundles",
                    checkpoint_root=checkpoint_root,
                    zfs_dataset_prefix=f"{pool_name}/crab",
                )
            )
            storage = LocalCheckpointManager(StorageConfig(root_dir=storage_root))
            executor = CRExecutor(
                ExecutorConfig(max_workers=1),
                DefaultCWorker(
                    AdapterProcessCWorker(runtime),
                    AdapterFileSystemCWorker(runtime),
                    storage,
                    runtime,
                ),
                DefaultRWorker(
                    AdapterProcessRWorker(runtime),
                    AdapterFileSystemRWorker(runtime),
                    storage,
                ),
                telemetry,
            )
            sandbox_manager = RuncSandboxManager(
                host_inspector_client=host_client,
                paths=RuncSandboxManagerPaths(
                    state_root=runtime_state_root,
                    bundle_root=root / "bundles",
                    metadata_root=sandbox_metadata_root,
                    zfs_dataset_prefix=f"{pool_name}/crab",
                ),
            )
            scheduler = CRScheduler(
                SchedulerConfig(
                    min_checkpoint_interval_seconds=0.0,
                    force_checkpoint_after_seconds=0.0,
                    require_change_signal=True,
                    prefer_checkpoint_during_llm_request=True,
                    require_llm_request_for_checkpoint=True,
                ),
                inspector,
                sandbox_manager,
                InMemorySchedulerStateStore(),
                telemetry,
            )
            system = CrabSystem(
                scheduler=scheduler,
                executor=executor,
                storage=storage,
                inspector=inspector,
                runtime=sandbox_manager,
                telemetry=telemetry,
                request_state_store=request_state_store,
            )
            hook = CompositeRequestInterceptorHook([TelemetryRequestInterceptorHook(telemetry)])
            interceptor = CrabRequestInterceptorServer(
                upstream_url=f"http://127.0.0.1:{llm_server.server_address[1]}",
                request_state_store=request_state_store,
                hook=hook,
                on_state_change=system.notify_interceptor_state_change,
                on_response_ready=system.notify_live_response_ready,
                host="127.0.0.1",
                port=0,
            )
            interceptor.start()
            self.addCleanup(interceptor.stop)
            _wait_for_http_json(f"http://127.0.0.1:{interceptor.port}/healthz")

            self._write_bundle_config(
                bundle_dir=bundle_dir,
                llm_base_url=f"http://127.0.0.1:{interceptor.port}/v1",
                provider=provider,
                status_port=status_port,
                cgroup_path=f"crab-tests/{pool_name}/{sandbox_id}",
                sandbox_id=sandbox_id,
            )

            launched_id = system.sandbox_manager.launch(
                "runc",
                {
                    "sandbox_id": str(sandbox_id),
                    "bundle_path": str(bundle_dir),
                    "rootfs_init_dirs": [
                        "work",
                        "tmp",
                        "proc",
                        "dev",
                        "dev/pts",
                        "dev/shm",
                        "dev/mqueue",
                        "sys",
                        "run",
                        "var",
                    ],
                    "rootfs_copy_paths": [{"source": str(exported_rootfs), "destination": "/"}],
                },
            )
            self.assertEqual(launched_id, sandbox_id)
            self.addCleanup(
                lambda: subprocess.run(
                    ["runc", "--root", str(runtime_state_root), "delete", "-f", str(sandbox_id)],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            )

            work_dir = bundle_dir / "rootfs" / "work"
            status_before = self._wait_for_sandbox_status(
                status_port=status_port,
                runtime_state_root=runtime_state_root,
                sandbox_id=sandbox_id,
                work_dir=work_dir,
            )
            self.assertGreaterEqual(int(status_before["total_actions"]), 0)

            host_client.reset_sandbox(sandbox_id, utc_now())
            _wait_for(lambda: self._enough_agent_progress(status_port), timeout_s=45.0)
            status_before = _wait_for_http_json(f"http://127.0.0.1:{status_port}/status")
            runtime_id_before = str(status_before["runtime_id"])
            total_before = int(status_before["total_actions"])

            _wait_for(
                lambda: self._host_status_changed(host_client, sandbox_id),
                timeout_s=20.0,
            )
            # The simulated LLM answers in ~250ms, so a snapshot taken after
            # a separate wait usually misses the in-flight window. The agent
            # keeps issuing requests, so sample the inspector inside the wait
            # predicate and keep the snapshot that caught the window.
            inflight_snapshot: dict[str, object] = {}

            def _catch_inflight_window() -> bool:
                snap = system.inspector.inspect(sandbox_id)
                if bool(snap.metadata["llm_request_in_flight"]):
                    inflight_snapshot["snapshot"] = snap
                    return True
                return False

            _wait_for(_catch_inflight_window, timeout_s=20.0)
            inspected = inflight_snapshot["snapshot"]
            self.assertTrue(bool(inspected.metadata["llm_request_in_flight"]))
            self.assertTrue(inspected.process_changed or inspected.filesystem_changed)

            checkpoint_result = system.checkpoint_if_due(sandbox_id)
            self.assertIsNotNone(checkpoint_result)
            assert checkpoint_result is not None
            self.assertEqual(checkpoint_result.status, JobStatus.SUCCEEDED)
            scheduler_events = [attrs for name, attrs in telemetry.events if name == "scheduler.evaluate"]
            # checkpoint_full_baseline_on_first_checkpoint defaults to True, so
            # the first checkpoint is an unconditional full baseline; that
            # branch outranks the llm-window reason in the scheduler.
            self.assertTrue(any(x["reason"] == "no_previous_checkpoint" for x in scheduler_events))

            tamper_path = work_dir / "host_tamper.txt"
            tamper_path.write_text("tampered\n", encoding="utf-8")
            self.assertTrue(tamper_path.exists())

            restore_result = system.restore_once(sandbox_id, checkpoint_result.checkpoint_id)
            self.assertEqual(
                restore_result.status,
                JobStatus.SUCCEEDED,
                self._restore_failure_details(
                    checkpoint_root=checkpoint_root,
                    checkpoint_id=str(checkpoint_result.checkpoint_id),
                    sandbox_id=str(sandbox_id),
                    message=restore_result.message,
                ),
            )
            self.assertFalse(tamper_path.exists(), "zfs rollback should remove host-side tamper")

            _wait_for(
                lambda: int(_wait_for_http_json(f"http://127.0.0.1:{status_port}/status")["total_actions"]) > total_before,
                timeout_s=20.0,
                interval_s=0.5,
            )
            status_after = _wait_for_http_json(f"http://127.0.0.1:{status_port}/status")
            self.assertEqual(str(status_after["runtime_id"]), runtime_id_before)
            self.assertGreater(int(status_after["total_actions"]), total_before)
            self.assertGreaterEqual(int(status_after["filesystem_actions"]), 1)
            self.assertGreaterEqual(int(status_after["process_actions"]), 1)
            self.assertGreaterEqual(int(status_after["network_actions"]), 1)
            self.assertGreaterEqual(int(status_after["stateful_actions"]), 1)
            self.assertTrue(bool(status_after["state_file_exists"]))
            self.assertTrue(bool(status_after["tool_artifact_exists"]))
            self.assertTrue(bool(status_after["journal_exists"]))

            inspected_after = system.inspector.inspect(sandbox_id)
            self.assertIn("last_llm_provider", inspected_after.metadata)

            system.sandbox_manager.stop(sandbox_id)
            self.assertEqual(system.sandbox_manager.describe(sandbox_id).status, "stopped")
            system.sandbox_manager.delete(sandbox_id)
        finally:
            if executor is not None:
                executor.shutdown()
            subprocess.run(
                ["runc", "--root", str(root / "runtime-state"), "delete", "-f", str(sandbox_id)],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            subprocess.run(
                ["zfs", "destroy", "-r", f"{pool_name}/crab/{sandbox_id}"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            subprocess.run(
                ["zfs", "destroy", "-r", f"{pool_name}/crab"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            subprocess.run(
                ["zpool", "destroy", "-f", pool_name],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

    def _host_status_changed(self, client: HostInspectorServiceClient, sandbox_id: SandboxId) -> bool:
        payload = client.get_proc_and_fs_status(sandbox_id)
        status = payload["status"]
        return bool(status["process_changed"] or status["filesystem_changed"])

    def _write_bundle_config(
        self,
        *,
        bundle_dir: Path,
        llm_base_url: str,
        provider: str,
        status_port: int,
        cgroup_path: str,
        sandbox_id: SandboxId,
    ) -> None:
        config_path = bundle_dir / "config.json"
        cfg = json.loads(config_path.read_text())
        linux_cfg = cfg.get("linux", {})
        linux_cfg["namespaces"] = [
            ns for ns in linux_cfg.get("namespaces", []) if ns.get("type") not in {"network", "cgroup"}
        ]
        linux_cfg["cgroupsPath"] = cgroup_path
        linux_cfg.pop("seccomp", None)
        cfg["linux"] = linux_cfg
        cfg["process"]["terminal"] = False
        cfg["process"]["cwd"] = "/work"
        cfg["process"]["args"] = [
            "/bin/sh",
            "-lc",
            f"exec /usr/local/bin/agent-cli run --provider {provider} >/dev/null 2>/dev/null",
        ]
        cfg["process"]["env"] = [
            "PATH=/usr/local/bin:/usr/bin:/bin",
            "PYTHONUNBUFFERED=1",
            f"CRAB_LLM_BASE_URL={llm_base_url}",
            f"STATUS_PORT={status_port}",
            "POLL_INTERVAL_S=0.2",
            "AGENT_WORK_DIR=/work",
            f"AGENT_SANDBOX_ID={sandbox_id}",
            f"AGENT_PROVIDER={provider}",
        ]
        cfg["root"]["path"] = "rootfs"
        cfg["root"]["readonly"] = False
        config_path.write_text(json.dumps(cfg, indent=2))

    def _enough_agent_progress(self, status_port: int) -> bool:
        try:
            payload = _wait_for_http_json(f"http://127.0.0.1:{status_port}/status", timeout_s=1.0)
        except Exception:
            return False
        return (
            int(payload["total_actions"]) >= 6
            and int(payload["filesystem_actions"]) >= 1
            and int(payload["process_actions"]) >= 1
            and int(payload["network_actions"]) >= 1
            and int(payload["stateful_actions"]) >= 1
        )

    def _wait_for_sandbox_status(
        self,
        *,
        status_port: int,
        runtime_state_root: Path,
        sandbox_id: SandboxId,
        work_dir: Path,
    ) -> dict[str, object]:
        try:
            return _wait_for_http_json(f"http://127.0.0.1:{status_port}/status")
        except Exception as exc:
            state = subprocess.run(
                ["runc", "--root", str(runtime_state_root), "state", str(sandbox_id)],
                check=False,
                capture_output=True,
                text=True,
            )
            work_listing = sorted(x.name for x in work_dir.glob("*")) if work_dir.exists() else []
            raise RuntimeError(
                "sandbox status endpoint did not come up\n"
                f"runc_state_rc={state.returncode}\n"
                f"runc_state_stdout={state.stdout.strip()}\n"
                f"runc_state_stderr={state.stderr.strip()}\n"
                f"work_dir={work_dir}\n"
                f"work_listing={work_listing}\n"
                f"original_error={exc}"
            ) from exc

    def _restore_failure_details(
        self,
        *,
        checkpoint_root: Path,
        checkpoint_id: str,
        sandbox_id: str,
        message: str | None,
    ) -> str:
        work_dir = checkpoint_root / sandbox_id / checkpoint_id / "work"
        image_dir = checkpoint_root / sandbox_id / checkpoint_id / "process"
        log_parts = ["" if message is None else message]
        for root in (work_dir, image_dir):
            if not root.exists():
                continue
            for path in sorted(root.glob("*.log")):
                try:
                    content = path.read_text(encoding="utf-8", errors="replace")
                except Exception as exc:
                    content = f"<failed to read {path.name}: {exc}>"
                log_parts.append(f"{path.name}:\n{content}")
        return "\n\n".join(part for part in log_parts if part)


if __name__ == "__main__":
    unittest.main()
