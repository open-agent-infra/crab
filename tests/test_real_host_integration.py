from __future__ import annotations

import json
import shutil
import socket
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
    EBPFEvent,
    EBPFEventKind,
    EBPFSandboxInspector,
    ExecutorConfig,
    InMemoryEBPFEventCollector,
    InMemoryRequestStateStore,
    InMemorySchedulerStateStore,
    InMemoryTelemetrySink,
    LocalCheckpointManager,
    RequestAwareSandboxInspector,
    RuncRuntime,
    RuncRuntimePaths,
    SandboxId,
    SandboxSnapshot,
    SchedulerConfig,
    StorageConfig,
    TelemetryRequestInterceptorHook,
)
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


class RealHostIntegrationTests(unittest.TestCase):
    def test_real_runc_criu_zfs_checkpoint_restore(self) -> None:
        if (
            shutil.which("docker") is None
            or shutil.which("runc") is None
            or shutil.which("criu") is None
            or shutil.which("zfs") is None
        ):
            self.skipTest("docker/runc/criu/zfs not installed")

        tmpdir = tempfile.TemporaryDirectory(prefix="crab_real_it_")
        self.addCleanup(tmpdir.cleanup)
        root = Path(tmpdir.name)
        sandbox_id = SandboxId("sbx-real")
        pool_name = f"crabit{int(time.time())}"
        provider = "openai"
        image_tag = f"crab-simulated-agent:{int(time.time())}"
        executor: CRExecutor | None = None

        llm_server = serve(host="127.0.0.1", port=0, response_delay_ms=250)
        llm_thread = threading.Thread(target=llm_server.serve_forever, daemon=True)
        llm_thread.start()
        self.addCleanup(llm_server.shutdown)
        self.addCleanup(llm_server.server_close)
        self.addCleanup(llm_thread.join, 5.0)

        try:
            pool_file = root / "zpool.img"
            bundle_dir = root / "bundles" / "sbx-real"
            runtime_state_root = root / "runtime-state"
            checkpoint_root = root / "checkpoints"
            storage_root = root / "storage"
            sandbox_metadata_root = root / "sandbox-meta"
            status_port = _find_free_port()
            image_root = root / "image"

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
            collector = InMemoryEBPFEventCollector()
            request_state_store = InMemoryRequestStateStore()
            base_inspector = EBPFSandboxInspector(collector)
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
                paths=RuncSandboxManagerPaths(
                    state_root=runtime_state_root,
                    bundle_root=root / "bundles",
                    metadata_root=sandbox_metadata_root,
                    zfs_dataset_prefix=f"{pool_name}/crab",
                )
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
                response_gate_registry=system.response_gate_registry,
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
                cgroup_path=f"crab-tests/{pool_name}/sbx-real",
            )

            inspector.upsert_snapshot(
                SandboxSnapshot(
                    sandbox_id=sandbox_id,
                    runtime_name="runc",
                    is_running=True,
                    process_changed=False,
                    filesystem_changed=False,
                    observed_at=utc_now(),
                )
            )

            sandbox_id = system.sandbox_manager.launch(
                "runc",
                {
                    "sandbox_id": "sbx-real",
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

            _wait_for(lambda: self._enough_agent_progress(status_port), timeout_s=45.0)
            status_before = _wait_for_http_json(f"http://127.0.0.1:{status_port}/status")
            runtime_id_before = str(status_before["runtime_id"])
            total_before = int(status_before["total_actions"])

            _wait_for(lambda: request_state_store.get(sandbox_id).llm_request_in_flight, timeout_s=20.0)
            self._record_activity_events(
                collector=collector,
                sandbox_id=sandbox_id,
                previous={},
                current=status_before,
            )
            inspected = system.inspector.inspect(sandbox_id)
            self.assertTrue(bool(inspected.metadata["llm_request_in_flight"]))
            self.assertGreaterEqual(int(inspected.metadata["active_llm_requests"]), 1)
            self.assertGreaterEqual(int(inspected.metadata["network_event_count"]), 1)

            checkpoint_result = system.checkpoint_if_due(sandbox_id)
            self.assertIsNotNone(checkpoint_result)
            assert checkpoint_result is not None
            self.assertEqual(checkpoint_result.status, JobStatus.SUCCEEDED)
            scheduler_events = [attrs for name, attrs in telemetry.events if name == "scheduler.evaluate"]
            self.assertTrue(any(x["reason"] == "llm_request_window_available" for x in scheduler_events))

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

            self.assertTrue((work_dir / "agent_state.json").exists())
            self.assertTrue((work_dir / "tool_artifact.txt").exists())
            self.assertTrue((work_dir / "journal.log").exists())

            inspected_after = system.inspector.inspect(sandbox_id)
            self.assertIn("last_llm_provider", inspected_after.metadata)
            request_events = [name for name, _ in telemetry.events if name.startswith("request.")]
            self.assertIn("request.start", request_events)
            self.assertIn("request.finish", request_events)

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
                ["zfs", "destroy", "-r", f"{pool_name}/crab/sbx-real"],
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

    def _write_bundle_config(
        self,
        *,
        bundle_dir: Path,
        llm_base_url: str,
        provider: str,
        status_port: int,
        cgroup_path: str,
    ) -> None:
        config_path = bundle_dir / "config.json"
        cfg = json.loads(config_path.read_text())
        linux_cfg = cfg.get("linux", {})
        linux_cfg["namespaces"] = [
            ns
            for ns in linux_cfg.get("namespaces", [])
            if ns.get("type") not in {"network", "cgroup"}
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
            "AGENT_SANDBOX_ID=sbx-real",
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

    def _record_activity_events(
        self,
        *,
        collector: InMemoryEBPFEventCollector,
        sandbox_id: SandboxId,
        previous: dict[str, object],
        current: dict[str, object],
    ) -> None:
        mappings = [
            ("filesystem_actions", EBPFEventKind.FILE_WRITE, {"path": "/work/tool_artifact.txt"}),
            ("process_actions", EBPFEventKind.PROCESS_EXEC, {"command": "/bin/sh"}),
            ("network_actions", EBPFEventKind.NETWORK_EGRESS, {"address": "127.0.0.1"}),
        ]
        for field, kind, metadata in mappings:
            previous_count = int(previous.get(field, 0))
            current_count = int(current.get(field, 0))
            if current_count > previous_count:
                collector.record(
                    EBPFEvent(
                        sandbox_id=sandbox_id,
                        kind=kind,
                        observed_at=utc_now(),
                        metadata=metadata,
                    )
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

    def _restore_failure_details(self, *, checkpoint_root: Path, checkpoint_id: str, message: str | None) -> str:
        work_dir = checkpoint_root / "sbx-real" / checkpoint_id / "work"
        image_dir = checkpoint_root / "sbx-real" / checkpoint_id / "process"
        log_parts = ["" if message is None else message]
        for root in (work_dir, image_dir):
            if not root.exists():
                continue
            for path in sorted(root.glob("*.log")):
                try:
                    content = path.read_text(encoding="utf-8", errors="replace")
                except Exception as exc:  # pragma: no cover - debug path only
                    content = f"<failed to read {path.name}: {exc}>"
                log_parts.append(f"{path.name}:\n{content}")
        return "\n\n".join(part for part in log_parts if part)


if __name__ == "__main__":
    unittest.main()
