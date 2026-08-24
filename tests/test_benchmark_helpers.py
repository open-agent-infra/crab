from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
import io
import ipaddress
import json
import os
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from crab import (
    ArtifactKind,
    ArtifactReference,
    CheckpointId,
    CheckpointManifest,
    EBPFEventKind,
    InMemoryTelemetrySink,
    SandboxId,
    SchedulerConfig,
)
from integrations.agents import BaseAgent, IFlowAgent, SandboxHandle, SimulatedAgent, TaskConfig, TaskDescription
from integrations.agents.iflow import IFLOW_WRAPPER_ARG
from integrations.sandboxes.runtime import bundle as sandbox_bundle
from integrations.sandboxes.runtime import compose as sandbox_compose
from integrations.sandboxes.runtime import image as sandbox_image
from integrations.sandboxes.runtime import launcher as sandbox_launcher
from integrations.sandboxes.runtime import network as sandbox_network
from benchmarks.config import BenchmarkConfig
from benchmarks.core import benchmark_phase_map, benchmark_setup_run_pipeline, emit_benchmark_phase_skipped
from benchmarks.scenarios.common import finalize_replay_row
from benchmarks.scenarios.e2e import run_manual as run_e2e_manual
from crab.models import utc_now
from benchmarks.scenarios.tree import choose_replay_steps
from benchmarks.support import (
    BenchmarkTaskRecord,
    TreeSearchCheckpointRecord,
    bounded_probability,
    build_tree_search_checkpoint_index,
    compute_summary,
    compute_summary_aliases,
    compute_telemetry_summary,
    effective_trace_replay_progress_count,
    resolve_checkpoint_copy_plan,
    resolve_tree_search_replay_checkpoint,
    resolve_work_dir_host_path,
    select_injected_indices,
    total_actions,
    write_rows,
)
from benchmarks.real_host_scenario_base import (
    RealHostScenarioHarness,
    _merge_runtime_launch_metadata,
)
from integrations.sandboxes.runtime.network import (
    benchmark_network_guest_capacity,
    parse_ipv4_route_networks,
    select_benchmark_network,
)

ImageRuntimeDefaults = sandbox_image.ImageRuntimeDefaults


class BenchmarkHelperTests(unittest.TestCase):
    def test_llm_router_max_workers_scales_above_benchmark_workers(self) -> None:
        harness = RealHostScenarioHarness(
            provider="openai",
            transfer_delay_ms=0.0,
            scheduler_config=SchedulerConfig(require_change_signal=False),
            scheduler_policy=object(),
            checkpoint_manager_factory=lambda base: base,
            max_workers=5,
        )

        self.assertEqual(harness._llm_router_max_workers(), 32)

        harness.max_workers = 12

        self.assertEqual(harness._llm_router_max_workers(), 48)

    def test_resolve_benchmark_run_root_uses_configured_home_and_name(self) -> None:
        harness = RealHostScenarioHarness(
            provider="openai",
            transfer_delay_ms=0.0,
            scheduler_config=SchedulerConfig(),
            scheduler_policy=None,
            checkpoint_manager_factory=lambda base: base,
            max_workers=1,
            benchmark_root_home=Path("/bench-home"),
            benchmark_run_name="run-a",
        )

        root, uses_temporary_root = harness._resolve_benchmark_run_root()

        self.assertEqual(root, Path("/bench-home/run-a"))
        self.assertFalse(uses_temporary_root)
        self.assertEqual(harness.configured_benchmark_root, Path("/bench-home"))

    def test_resolve_benchmark_run_root_generates_name_for_home_only(self) -> None:
        harness = RealHostScenarioHarness(
            provider="openai",
            transfer_delay_ms=0.0,
            scheduler_config=SchedulerConfig(),
            scheduler_policy=None,
            checkpoint_manager_factory=lambda base: base,
            max_workers=1,
            benchmark_root_home=Path("/bench-home"),
        )

        with patch.object(
            RealHostScenarioHarness,
            "_generated_benchmark_run_name",
            return_value="20260416_010203",
        ):
            root, uses_temporary_root = harness._resolve_benchmark_run_root()

        self.assertEqual(root, Path("/bench-home/20260416_010203"))
        self.assertFalse(uses_temporary_root)

    def test_resolve_benchmark_run_root_keeps_legacy_root_as_home_alias(self) -> None:
        harness = RealHostScenarioHarness(
            provider="openai",
            transfer_delay_ms=0.0,
            scheduler_config=SchedulerConfig(),
            scheduler_policy=None,
            checkpoint_manager_factory=lambda base: base,
            max_workers=1,
            benchmark_root=Path("/legacy-home"),
            benchmark_run_name="run-a",
        )

        root, uses_temporary_root = harness._resolve_benchmark_run_root()

        self.assertEqual(root, Path("/legacy-home/run-a"))
        self.assertFalse(uses_temporary_root)
        self.assertEqual(harness.configured_benchmark_root_home, Path("/legacy-home"))

    def test_resolve_benchmark_run_root_rejects_name_without_home(self) -> None:
        harness = RealHostScenarioHarness(
            provider="openai",
            transfer_delay_ms=0.0,
            scheduler_config=SchedulerConfig(),
            scheduler_policy=None,
            checkpoint_manager_factory=lambda base: base,
            max_workers=1,
            benchmark_run_name="run-a",
        )

        with patch.dict(os.environ, {"CRAB_BENCH_DIR": "tmp"}):
            with self.assertRaisesRegex(ValueError, "benchmark_run_name requires"):
                harness._resolve_benchmark_run_root()

    def _load_verification_uv_shim_namespace(self) -> dict[str, object]:
        namespace: dict[str, object] = {}
        exec(RealHostScenarioHarness._verification_uv_python_shim_script(), namespace)
        return namespace

    def _tree_search_manifest(
        self,
        checkpoint_id: str,
        *,
        step: int | None,
    ) -> CheckpointManifest:
        metadata: dict[str, object] = {}
        if step is not None:
            metadata["tree_search_step"] = step
        return CheckpointManifest(
            schema_version="v1",
            checkpoint_id=CheckpointId(checkpoint_id),
            sandbox_id=SandboxId("sbx"),
            created_at=utc_now(),
            runtime_name="runc",
            runtime_version=None,
            process_artifacts=[],
            filesystem_artifacts=[],
            metadata=metadata,
        ).with_integrity()

    def test_choose_replay_steps_is_deterministic(self) -> None:
        self.assertEqual(choose_replay_steps(6, 2), [1, 3])
        self.assertEqual(choose_replay_steps(4, 10), [1, 2, 3])

    def test_write_rows_supports_nonuniform_field_sets(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "rows.csv"

            write_rows(
                str(output_path),
                [
                    {"sandbox_id": "fault-0", "success_ratio": 1.0, "verification_status": "passed"},
                    {
                        "sandbox_id": "fault-1",
                        "success_ratio": 0.0,
                        "verification_status": "failed",
                        "verification_stderr": "boom",
                    },
                ],
            )
            self.assertEqual(
                output_path.read_text(encoding="utf-8").splitlines(),
                [
                    "sandbox_id,success_ratio,verification_status,verification_stderr",
                    "fault-0,1.0,passed,",
                    "fault-1,0.0,failed,boom",
                ],
            )

    def test_wait_for_task_completion_short_circuits_when_replay_is_complete(self) -> None:
        harness = RealHostScenarioHarness.__new__(RealHostScenarioHarness)
        harness.system = None
        harness.runtime = None
        future: Future[None] = Future()
        future.set_result(None)
        sandbox = SandboxHandle(
            sandbox_id=SandboxId("sbx-replay-complete"),
            bundle_dir=Path("/tmp/sbx-replay-complete"),
            status_port=8123,
            last_status={
                "state": "finished",
                "total_actions": 12,
                "replay_trace_cursor": 12,
                "replay_is_complete": True,
            },
            agent_type="claude_code",
            llm_service_type="claude_code_trace_replay",
            launch_metadata={"benchmark": {"task_id": "hf-model-inference", "trace_response_count": 12}},
            task_future=future,
        )
        sandbox.task_run = SimpleNamespace(poll_status=lambda: dict(sandbox.last_status))

        RealHostScenarioHarness.wait_for_task_completion(harness, sandbox, timeout_s=0.01)

    def test_wait_for_task_completion_waits_for_pending_interceptor_signal_before_short_circuit(self) -> None:
        harness = RealHostScenarioHarness.__new__(RealHostScenarioHarness)
        harness.system = SimpleNamespace(has_pending_interceptor_signal=Mock(side_effect=[True, False]))
        harness.runtime = SimpleNamespace(resume=Mock())
        future: Future[None] = Future()
        future.set_result(None)
        sandbox = SandboxHandle(
            sandbox_id=SandboxId("sbx-replay-pending"),
            bundle_dir=Path("/tmp/sbx-replay-pending"),
            status_port=8123,
            last_status={
                "state": "running",
                "total_actions": 12,
                "replay_trace_cursor": 12,
                "replay_is_complete": True,
            },
            agent_type="claude_code",
            llm_service_type="claude_code_trace_replay",
            launch_metadata={"benchmark": {"task_id": "mailman", "trace_response_count": 12}},
            task_future=future,
        )
        sandbox.task_run = SimpleNamespace(poll_status=lambda: dict(sandbox.last_status))

        RealHostScenarioHarness.wait_for_task_completion(harness, sandbox, timeout_s=0.2)

        self.assertEqual(harness.system.has_pending_interceptor_signal.call_count, 2)
        harness.runtime.resume.assert_called_once_with(sandbox.sandbox_id)

    def test_wait_for_task_completion_waits_for_replay_future_before_short_circuit(self) -> None:
        harness = RealHostScenarioHarness.__new__(RealHostScenarioHarness)
        harness.system = None
        harness.runtime = SimpleNamespace(resume=Mock())
        future: Future[None] = Future()
        sandbox = SandboxHandle(
            sandbox_id=SandboxId("sbx-replay-future"),
            bundle_dir=Path("/tmp/sbx-replay-future"),
            status_port=8123,
            last_status={
                "state": "running",
                "total_actions": 12,
                "replay_trace_cursor": 12,
                "replay_is_complete": True,
            },
            agent_type="claude_code",
            llm_service_type="claude_code_trace_replay",
            launch_metadata={"benchmark": {"task_id": "build-pov-ray", "trace_response_count": 12}},
            task_future=future,
        )
        sandbox.task_run = SimpleNamespace(poll_status=lambda: dict(sandbox.last_status))

        def _finish_later() -> None:
            time.sleep(0.05)
            future.set_result(None)

        thread = threading.Thread(target=_finish_later)
        thread.start()
        started = time.monotonic()
        try:
            RealHostScenarioHarness.wait_for_task_completion(harness, sandbox, timeout_s=0.5)
        finally:
            thread.join(timeout=1.0)

        self.assertGreaterEqual(time.monotonic() - started, 0.05)
        harness.runtime.resume.assert_called_once_with(sandbox.sandbox_id)

    def test_wait_for_task_completion_requests_stop_for_stale_replay_future_after_grace(self) -> None:
        harness = RealHostScenarioHarness.__new__(RealHostScenarioHarness)
        harness.system = SimpleNamespace(has_pending_interceptor_signal=Mock(return_value=False))
        harness.runtime = SimpleNamespace(resume=Mock())
        harness.REPLAY_COMPLETION_TASK_FUTURE_GRACE_SECONDS = 0.0
        future: Future[None] = Future()
        stop_requests: list[str] = []

        def _request_stop() -> None:
            stop_requests.append("stop")
            future.set_result(None)

        sandbox = SandboxHandle(
            sandbox_id=SandboxId("sbx-replay-stale-future"),
            bundle_dir=Path("/tmp/sbx-replay-stale-future"),
            status_port=8123,
            last_status={
                "state": "running",
                "total_actions": 15,
                "replay_trace_cursor": 15,
                "replay_is_complete": True,
            },
            agent_type="claude_code",
            llm_service_type="claude_code_trace_replay",
            launch_metadata={"benchmark": {"task_id": "bn-fit-modify", "trace_response_count": 15}},
            task_future=future,
        )
        sandbox.task_run = SimpleNamespace(
            poll_status=lambda: dict(sandbox.last_status),
            request_stop=_request_stop,
        )

        RealHostScenarioHarness.wait_for_task_completion(harness, sandbox, timeout_s=0.5)

        self.assertEqual(stop_requests, ["stop"])
        harness.runtime.resume.assert_called_once_with(sandbox.sandbox_id)

    def test_wait_for_task_completion_ignores_replay_exception_after_completion(self) -> None:
        harness = RealHostScenarioHarness.__new__(RealHostScenarioHarness)
        harness.system = None
        harness.runtime = None
        future: Future[None] = Future()
        future.set_exception(RuntimeError("timed out waiting for claude_code replay action count 1"))
        sandbox = SandboxHandle(
            sandbox_id=SandboxId("sbx-replay-complete-exc"),
            bundle_dir=Path("/tmp/sbx-replay-complete-exc"),
            status_port=8123,
            last_status={
                "state": "finished",
                "total_actions": 8,
                "replay_trace_cursor": 8,
                "replay_is_complete": True,
            },
            agent_type="claude_code",
            llm_service_type="claude_code_trace_replay",
            launch_metadata={"benchmark": {"task_id": "kv-store-grpc", "trace_response_count": 8}},
            task_future=future,
        )
        sandbox.task_run = SimpleNamespace(poll_status=lambda: dict(sandbox.last_status))

        RealHostScenarioHarness.wait_for_task_completion(harness, sandbox, timeout_s=0.01)

    def test_finalize_replay_row_preserves_pre_verification_replay_completion(self) -> None:
        config = BenchmarkConfig(
            config_path=Path("/tmp/bench.yaml"),
            scenario="fault",
            mode="auto",
        )
        sandbox = SandboxHandle(
            sandbox_id=SandboxId("sbx-finalize-replay"),
            bundle_dir=Path("/tmp/sbx-finalize-replay"),
            status_port=8123,
            last_status={},
            agent_type="claude_code",
            llm_service_type="claude_code_trace_replay",
            task_config=TaskConfig(options={"task_id": "bn-fit-modify"}),
            launch_metadata={"benchmark": {"trace_response_count": 15}},
        )
        statuses = iter(
            [
                {
                    "state": "running",
                    "replay_trace_cursor": 15,
                    "replay_is_complete": True,
                },
                {
                    "state": "finished",
                    "replay_trace_cursor": 0,
                    "replay_is_complete": False,
                },
            ]
        )
        sandbox.task_run = SimpleNamespace(poll_status=lambda: dict(next(statuses)))
        harness = SimpleNamespace()

        with patch(
            "benchmarks.scenarios.common.verify_task_accuracy",
            return_value=(
                "",
                {
                    "verification_status": "passed",
                    "verification_exit_code": 0,
                    "verification_ms": 12.0,
                },
            ),
        ):
            row = finalize_replay_row(
                config,
                harness,
                sandbox,
                row={"task_id": "bn-fit-modify"},
                verify_task_accuracy_result=True,
            )

        self.assertEqual(row["verification_status"], "passed")
        self.assertEqual(row["success_ratio"], 1.0)
        self.assertEqual(row["replay_final_trace_cursor"], 15)
        self.assertTrue(row["replay_is_complete"])
        self.assertEqual(row["task_error"], "")

    def test_verification_uv_bootstrap_script_installs_working_uv_shim(self) -> None:
        script = RealHostScenarioHarness._verification_uv_bootstrap_script()
        self.assertIn('export PATH="$shim_bin:$PATH"', script)
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir) / "home"
            work = Path(tmpdir) / "work"
            home.mkdir()
            work.mkdir()
            env = {**os.environ, "HOME": str(home)}

            bootstrap = subprocess.run(
                ["bash", "-lc", script],
                cwd=work,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(bootstrap.returncode, 0, bootstrap.stderr)

            verify = subprocess.run(
                [
                    "bash",
                    "-lc",
                    (
                        'source "$HOME/.local/bin/env" && '
                        "uv venv .tbench-testing && "
                        "source .tbench-testing/bin/activate && "
                        "uv pip install --system --help >/dev/null && "
                        'test "$(uv run python -c \'print(123)\')" = "123" && '
                        "uv python pin 3.11"
                    ),
                ],
                cwd=work,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(verify.returncode, 0, verify.stderr)

    def test_verify_task_accuracy_bootstraps_uv_before_running_tests(self) -> None:
        harness = RealHostScenarioHarness.__new__(RealHostScenarioHarness)
        harness.telemetry = None
        harness.sandbox_process_cwd = lambda sandbox: "/app"
        harness.benchmark_telemetry_attributes = lambda sandbox: {}
        harness._agent_requires_benchmark_network = lambda agent_type: False
        calls: list[tuple[list[str], dict[str, object]]] = []

        def fake_exec_in_sandbox(sandbox, command, **kwargs):
            calls.append((list(command), dict(kwargs)))
            return SimpleNamespace(
                returncode=0,
                stdout="verification ok\n",
                stderr="",
                args=list(command),
            )

        harness.exec_in_sandbox = fake_exec_in_sandbox
        sandbox = SandboxHandle(
            sandbox_id=SandboxId("sbx-verify"),
            bundle_dir=Path("/tmp/sbx-verify"),
            status_port=8123,
            last_status={},
            agent_type="claude_code",
            llm_service_type="claude_code_trace_replay",
            task_config=TaskConfig(),
        )

        verification = RealHostScenarioHarness.verify_task_accuracy(harness, sandbox, timeout_s=120.0)

        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][0][:2], ["/bin/bash", "-lc"])
        self.assertIn("$HOME/.local/bin/uv", calls[0][0][2])
        self.assertIn("$HOME/.local/crab-verification/bin", calls[0][0][2])
        self.assertAlmostEqual(float(calls[0][1]["timeout_s"]), 120.0, places=2)
        self.assertEqual(calls[1][0], ["/bin/bash", "-lc", "bash /tests/run-tests.sh"])
        self.assertEqual(calls[1][1]["env"]["TEST_DIR"], "/tests")
        self.assertIn("/root/.local/crab-verification/bin", calls[1][1]["env"]["PATH"])
        self.assertEqual(calls[1][1]["timeout_s"], 120.0)
        self.assertEqual(verification["verification_status"], "passed")

    def test_ensure_verification_uv_retries_transient_bootstrap_failures(self) -> None:
        harness = RealHostScenarioHarness.__new__(RealHostScenarioHarness)
        harness.sandbox_process_cwd = lambda sandbox: "/app"
        responses = [
            SimpleNamespace(
                returncode=100,
                stdout="",
                stderr="Temporary failure resolving 'archive.ubuntu.com'\n",
                args=["/bin/bash", "-lc", "bootstrap"],
            ),
            SimpleNamespace(
                returncode=0,
                stdout="",
                stderr="",
                args=["/bin/bash", "-lc", "bootstrap"],
            ),
        ]
        calls: list[dict[str, object]] = []

        def fake_exec_in_sandbox(sandbox, command, **kwargs):
            calls.append({"command": list(command), **kwargs})
            return responses.pop(0)

        harness.exec_in_sandbox = fake_exec_in_sandbox
        sandbox = SandboxHandle(
            sandbox_id=SandboxId("sbx-verify-retry"),
            bundle_dir=Path("/tmp/sbx-verify-retry"),
            status_port=8123,
            last_status={},
            agent_type="claude_code",
            llm_service_type="claude_code_trace_replay",
        )

        with patch("benchmarks.real_host_scenario_base.time.sleep") as sleep:
            RealHostScenarioHarness._ensure_verification_uv(harness, sandbox, timeout_s=120.0)

        self.assertEqual(len(calls), 2)
        self.assertGreaterEqual(float(calls[0]["timeout_s"]), 1.0)
        sleep.assert_called_once()

    def test_verification_uv_shim_installs_pytest_inside_fake_venv(self) -> None:
        namespace = self._load_verification_uv_shim_namespace()
        with tempfile.TemporaryDirectory() as tmpdir:
            venv_root = Path(tmpdir) / "venv"
            site_packages = venv_root / "lib" / "python3.11" / "site-packages"
            calls: list[list[str]] = []

            with patch.dict(os.environ, {"VIRTUAL_ENV": str(venv_root)}, clear=False):
                with patch.dict(namespace, {"_resolve_python": lambda _requested: "python3"}):
                    with patch.object(namespace["subprocess"], "call") as call_mock:
                        call_mock.side_effect = lambda cmd: calls.append(list(cmd)) or 0
                        with patch.dict(namespace, {"_python_version_tag": lambda _python: "3.11"}):
                            result = namespace["_run_pip"](["install", "pytest==8.4.1"])

            self.assertEqual(result, 0)
            self.assertEqual(
                calls,
                [
                    [
                        "python3",
                        "-m",
                        "pip",
                        "install",
                        "--target",
                        str(site_packages),
                        "pytest==8.4.1",
                    ]
                ],
            )

    def test_verification_uv_pytest_fallback_emits_pytest_style_summary_banner(self) -> None:
        namespace = self._load_verification_uv_shim_namespace()
        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = Path(tmpdir) / "test_sample.py"
            test_path.write_text(
                "def test_passes():\n"
                "    return None\n",
                encoding="utf-8",
            )
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                result = namespace["_run_pytest_fallback"]([str(test_path), "-rA"])

        self.assertEqual(result, 0)
        self.assertEqual(stderr.getvalue(), "")
        self.assertIn(
            "=========================== short test summary info ============================",
            stdout.getvalue(),
        )
        self.assertIn(f"PASSED {test_path.resolve()}::test_passes", stdout.getvalue())

    def test_verification_uv_shim_uses_real_pytest_when_available_in_fake_venv(self) -> None:
        namespace = self._load_verification_uv_shim_namespace()
        calls: list[list[str]] = []
        with tempfile.TemporaryDirectory() as tmpdir:
            venv_root = Path(tmpdir) / "venv"
            (venv_root / ".crab_fake_venv").parent.mkdir(parents=True, exist_ok=True)
            (venv_root / ".crab_fake_venv").write_text("", encoding="utf-8")
            with patch.dict(os.environ, {"VIRTUAL_ENV": str(venv_root)}, clear=False):
                with patch.object(namespace["shutil"], "which", return_value="/tmp/pytest"):
                    with patch.object(namespace["subprocess"], "call") as call_mock:
                        call_mock.side_effect = lambda cmd: calls.append(list(cmd)) or 0
                        result = namespace["_run_command"](
                            ["pytest", "/tests/test_outputs.py", "-rA"]
                        )

        self.assertEqual(result, 0)
        self.assertEqual(calls, [["pytest", "/tests/test_outputs.py", "-rA"]])

    def test_verification_uv_shim_runs_python_module_pytest_inside_venv_without_script(self) -> None:
        namespace = self._load_verification_uv_shim_namespace()
        calls: list[list[str]] = []
        with tempfile.TemporaryDirectory() as tmpdir:
            venv_root = Path(tmpdir) / "venv"
            venv_root.mkdir(parents=True, exist_ok=True)
            with patch.dict(os.environ, {"VIRTUAL_ENV": str(venv_root)}, clear=False):
                with patch.object(namespace["shutil"], "which", return_value=None):
                    with patch.dict(namespace, {"_resolve_python": lambda _requested: "python3"}):
                        with patch.object(namespace["subprocess"], "call") as call_mock:
                            call_mock.side_effect = lambda cmd: calls.append(list(cmd)) or 0
                            result = namespace["_run_command"](
                                ["pytest", "/tests/test_outputs.py", "-rA"]
                            )

        self.assertEqual(result, 0)
        self.assertEqual(calls, [["python3", "-m", "pytest", "/tests/test_outputs.py", "-rA"]])

    def test_verification_uv_failure_classifies_apt_lock_as_transient(self) -> None:
        harness = RealHostScenarioHarness.__new__(RealHostScenarioHarness)

        self.assertTrue(
            RealHostScenarioHarness._verification_uv_failure_is_transient(
                harness,
                "E: Could not get lock /var/lib/apt/lists/lock. It is held by process 10 (apt-get)\n"
                "E: Unable to lock directory /var/lib/apt/lists/\n",
            )
        )

    def test_wait_for_verification_network_retries_until_dns_recovers(self) -> None:
        harness = RealHostScenarioHarness.__new__(RealHostScenarioHarness)
        harness.sandbox_process_cwd = lambda sandbox: "/app"
        harness._agent_requires_benchmark_network = lambda agent_type: True
        harness._refresh_sandbox_resolv_conf = Mock(return_value=False)
        responses = [
            SimpleNamespace(
                returncode=1,
                stdout="",
                stderr="archive.ubuntu.com:80 [Errno -3] Temporary failure in name resolution\n",
                args=["/bin/bash", "-lc", "probe"],
            ),
            SimpleNamespace(
                returncode=0,
                stdout="",
                stderr="",
                args=["/bin/bash", "-lc", "probe"],
            ),
        ]
        calls: list[dict[str, object]] = []

        def fake_exec_in_sandbox(sandbox, command, **kwargs):
            calls.append({"command": list(command), **kwargs})
            return responses.pop(0)

        harness.exec_in_sandbox = fake_exec_in_sandbox
        sandbox = SandboxHandle(
            sandbox_id=SandboxId("sbx-network-ready"),
            bundle_dir=Path("/tmp/sbx-network-ready"),
            status_port=8123,
            last_status={},
            agent_type="claude_code",
            llm_service_type="claude_code_trace_replay",
        )

        with patch("benchmarks.real_host_scenario_base.time.sleep") as sleep:
            RealHostScenarioHarness._wait_for_verification_network(harness, sandbox, timeout_s=120.0)

        self.assertEqual(len(calls), 2)
        sleep.assert_called_once()

    def test_wait_for_verification_network_refreshes_resolv_conf_after_dns_failure(self) -> None:
        harness = RealHostScenarioHarness.__new__(RealHostScenarioHarness)
        harness.sandbox_process_cwd = lambda sandbox: "/app"
        harness._agent_requires_benchmark_network = lambda agent_type: True
        harness._refresh_sandbox_resolv_conf = Mock(return_value=True)
        responses = [
            SimpleNamespace(
                returncode=1,
                stdout="",
                stderr="archive.ubuntu.com:80 [Errno -3] Temporary failure in name resolution\n",
                args=["/bin/bash", "-lc", "probe"],
            ),
            SimpleNamespace(
                returncode=0,
                stdout="",
                stderr="",
                args=["/bin/bash", "-lc", "probe"],
            ),
        ]
        harness.exec_in_sandbox = lambda sandbox, command, **kwargs: responses.pop(0)
        sandbox = SandboxHandle(
            sandbox_id=SandboxId("sbx-network-resolv"),
            bundle_dir=Path("/tmp/sbx-network-resolv"),
            status_port=8123,
            last_status={},
            agent_type="claude_code",
            llm_service_type="claude_code_trace_replay",
        )

        with patch("benchmarks.real_host_scenario_base.time.sleep"):
            RealHostScenarioHarness._wait_for_verification_network(harness, sandbox, timeout_s=120.0)

        harness._refresh_sandbox_resolv_conf.assert_called_once_with(sandbox)

    def test_merge_runtime_launch_metadata_appends_copy_paths_and_ignores_rules(self) -> None:
        merged = _merge_runtime_launch_metadata(
            {
                "rootfs_init_dirs": ["work", "root/.claude"],
                "rootfs_copy_paths": [{"source": "/base", "destination": "/"}],
                "host_inspector_ignore_process_rules": [{"cmdline_contains": ["base"]}],
                "zfs_dataset": "pool/a",
            },
            {
                "rootfs_init_dirs": ["root/.claude", "opt/claude-code-logs"],
                "rootfs_copy_paths": [{"source": "/seed", "destination": "/root/.claude"}],
                "host_inspector_ignore_process_rules": [{"cmdline_contains": ["wrapper"]}],
            },
        )

        self.assertEqual(
            merged["rootfs_init_dirs"],
            ["work", "root/.claude", "opt/claude-code-logs"],
        )
        self.assertEqual(
            merged["rootfs_copy_paths"],
            [
                {"source": "/base", "destination": "/"},
                {"source": "/seed", "destination": "/root/.claude"},
            ],
        )
        self.assertEqual(
            merged["host_inspector_ignore_process_rules"],
            [
                {"cmdline_contains": ["base"]},
                {"cmdline_contains": ["wrapper"]},
            ],
        )
        self.assertEqual(merged["zfs_dataset"], "pool/a")

    def test_ensure_zpool_reuses_existing_dataset_when_destroy_does_not_remove_it(self) -> None:
        harness = RealHostScenarioHarness.__new__(RealHostScenarioHarness)
        harness.root = Path("/tmp/bench-root")
        harness.pool_name = "benchpool"
        harness.configured_zpool_image = Path("/tmp/benchpool.img")
        harness._zpool_image_path = None
        harness.reuse_zpool = True
        harness.zpool_size = "10G"

        with (
            patch("benchmarks.real_host_scenario_base._zpool_exists", return_value=True),
            patch("benchmarks.real_host_scenario_base._zfs_dataset_exists", side_effect=[True, True]),
            patch("benchmarks.real_host_scenario_base.subprocess.run") as run,
        ):
            harness._ensure_zpool()

        self.assertEqual(
            run.call_args_list,
            [
                unittest.mock.call(
                    ["zfs", "destroy", "-r", "benchpool/crab"],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            ],
        )
    def test_resolve_runtime_plane_root_defaults_to_benchmark_run_root(self) -> None:
        harness = RealHostScenarioHarness(
            provider="openai",
            transfer_delay_ms=0.0,
            scheduler_config=SchedulerConfig(),
            scheduler_policy=None,
            checkpoint_manager_factory=lambda base: base,
            max_workers=1,
        )
        harness.root = Path("/bench-root/run")
        resolved = harness._resolve_runtime_plane_root(
            zpool_image_path=Path("/data/benchpool.zpool.img")
        )

        self.assertEqual(resolved, harness.root)

    def test_shared_rootfs_launch_metadata_reuses_key_for_duplicate_compose_recipe(self) -> None:
        harness = RealHostScenarioHarness(
            provider="openai",
            transfer_delay_ms=0.0,
            scheduler_config=SchedulerConfig(),
            scheduler_policy=None,
            checkpoint_manager_factory=lambda base: base,
            max_workers=1,
            reuse_zpool=True,
        )
        compose_file = Path("/tmp/task/docker-compose.yaml")
        runtime_metadata_a = {
            "rootfs_init_dirs": ["work", "tests"],
            "rootfs_copy_paths": [
                {"source": "/tmp/rootfs", "destination": "/"},
                {"source": "/tmp/task/tests", "destination": "/tests"},
            ],
        }
        runtime_metadata_b = {
            "rootfs_init_dirs": ["tests", "work"],
            "rootfs_copy_paths": [
                {"source": "/tmp/task/tests", "destination": "/tests"},
                {"source": "/tmp/rootfs", "destination": "/"},
            ],
        }

        harness._attach_shared_rootfs_launch_metadata(
            runtime_metadata_a,
            agent_type="iflow",
            compose_file=compose_file,
            service_name="client",
        )
        harness._attach_shared_rootfs_launch_metadata(
            runtime_metadata_b,
            agent_type="iflow",
            compose_file=compose_file,
            service_name="client",
        )

        self.assertEqual(runtime_metadata_a["shared_rootfs_key"], runtime_metadata_b["shared_rootfs_key"])
        self.assertTrue(runtime_metadata_a["shared_rootfs_persist"])
        self.assertTrue(runtime_metadata_b["shared_rootfs_persist"])

    def test_shared_rootfs_launch_metadata_logs_reuse_recipe(self) -> None:
        harness = RealHostScenarioHarness(
            provider="openai",
            transfer_delay_ms=0.0,
            scheduler_config=SchedulerConfig(),
            scheduler_policy=None,
            checkpoint_manager_factory=lambda base: base,
            max_workers=1,
            reuse_zpool=True,
        )
        runtime_metadata = {
            "sandbox_id": "sbx-rootfs-log",
            "rootfs_init_dirs": ["work", "tests"],
            "rootfs_copy_paths": [
                {"source": "/tmp/rootfs", "destination": "/"},
                {"source": "/tmp/task/tests", "destination": "/tests"},
            ],
        }

        with self.assertLogs("benchmarks.real_host_scenario_base", level="DEBUG") as captured:
            harness._attach_shared_rootfs_launch_metadata(
                runtime_metadata,
                agent_type="iflow",
                compose_file=Path("/tmp/task/docker-compose.yaml"),
                service_name="client",
            )

        joined = "\n".join(captured.output)
        self.assertIn("Configured shared rootfs reuse sandbox=sbx-rootfs-log", joined)
        self.assertIn("Shared rootfs reuse recipe sandbox=sbx-rootfs-log", joined)

    def test_shared_rootfs_launch_metadata_separates_service_name_and_run_scoped_reuse(self) -> None:
        harness_a = RealHostScenarioHarness(
            provider="openai",
            transfer_delay_ms=0.0,
            scheduler_config=SchedulerConfig(),
            scheduler_policy=None,
            checkpoint_manager_factory=lambda base: base,
            max_workers=1,
            reuse_zpool=True,
            run_id="run-a",
        )
        harness_b = RealHostScenarioHarness(
            provider="openai",
            transfer_delay_ms=0.0,
            scheduler_config=SchedulerConfig(),
            scheduler_policy=None,
            checkpoint_manager_factory=lambda base: base,
            max_workers=1,
            reuse_zpool=True,
            run_id="run-b",
        )
        compose_primary = {
            "rootfs_init_dirs": ["work"],
            "rootfs_copy_paths": [{"source": "/tmp/rootfs", "destination": "/"}],
        }
        compose_secondary = {
            "rootfs_init_dirs": ["work"],
            "rootfs_copy_paths": [{"source": "/tmp/rootfs", "destination": "/"}],
        }
        run_scoped_a = {
            "rootfs_init_dirs": ["work"],
            "rootfs_copy_paths": [{"source": "/tmp/rootfs", "destination": "/"}],
        }
        run_scoped_b = {
            "rootfs_init_dirs": ["work"],
            "rootfs_copy_paths": [{"source": "/tmp/rootfs", "destination": "/"}],
        }

        harness_a._attach_shared_rootfs_launch_metadata(
            compose_primary,
            agent_type="iflow",
            compose_file=Path("/tmp/task/docker-compose.yaml"),
            service_name="client",
        )
        harness_a._attach_shared_rootfs_launch_metadata(
            compose_secondary,
            agent_type="iflow",
            compose_file=Path("/tmp/task/docker-compose.yaml"),
            service_name="worker",
        )
        harness_a._attach_shared_rootfs_launch_metadata(
            run_scoped_a,
            agent_type="simulated",
            compose_file=None,
            service_name=None,
        )
        harness_b._attach_shared_rootfs_launch_metadata(
            run_scoped_b,
            agent_type="simulated",
            compose_file=None,
            service_name=None,
        )

        self.assertNotEqual(compose_primary["shared_rootfs_key"], compose_secondary["shared_rootfs_key"])
        self.assertTrue(compose_primary["shared_rootfs_persist"])
        self.assertFalse(run_scoped_a["shared_rootfs_persist"])
        self.assertFalse(run_scoped_b["shared_rootfs_persist"])
        self.assertNotEqual(run_scoped_a["shared_rootfs_key"], run_scoped_b["shared_rootfs_key"])

    def test_build_tree_search_checkpoint_index_collects_steps(self) -> None:
        manifests = [
            self._tree_search_manifest("ckpt-1", step=1),
            self._tree_search_manifest("ckpt-2", step=2),
            self._tree_search_manifest("ckpt-3", step=3),
        ]

        self.assertEqual(
            build_tree_search_checkpoint_index(manifests, initial_steps=3, require_complete=True),
            {
                1: TreeSearchCheckpointRecord(CheckpointId("ckpt-1"), replay_actions=1),
                2: TreeSearchCheckpointRecord(CheckpointId("ckpt-2"), replay_actions=2),
                3: TreeSearchCheckpointRecord(CheckpointId("ckpt-3"), replay_actions=3),
            },
        )

    def test_build_tree_search_checkpoint_index_rejects_duplicates(self) -> None:
        manifests = [
            self._tree_search_manifest("ckpt-1", step=1),
            self._tree_search_manifest("ckpt-2", step=1),
        ]

        with self.assertRaisesRegex(ValueError, "duplicate tree-search checkpoint for step 1"):
            build_tree_search_checkpoint_index(manifests, initial_steps=2)

    def test_build_tree_search_checkpoint_index_rejects_missing_steps_when_required(self) -> None:
        manifests = [
            self._tree_search_manifest("ckpt-1", step=1),
            self._tree_search_manifest("ckpt-3", step=3),
        ]

        with self.assertRaisesRegex(ValueError, r"missing tree-search checkpoints for steps \[2\]"):
            build_tree_search_checkpoint_index(manifests, initial_steps=3, require_complete=True)

    def test_build_tree_search_checkpoint_index_ignores_extra_trailing_steps(self) -> None:
        manifests = [
            self._tree_search_manifest("ckpt-1", step=1),
            self._tree_search_manifest("ckpt-2", step=2),
            self._tree_search_manifest("ckpt-3", step=3),
            self._tree_search_manifest("ckpt-4", step=4),
        ]

        self.assertEqual(
            build_tree_search_checkpoint_index(manifests, initial_steps=3),
            {
                1: TreeSearchCheckpointRecord(CheckpointId("ckpt-1"), replay_actions=1),
                2: TreeSearchCheckpointRecord(CheckpointId("ckpt-2"), replay_actions=2),
                3: TreeSearchCheckpointRecord(CheckpointId("ckpt-3"), replay_actions=3),
            },
        )

    def test_resolve_tree_search_replay_checkpoint_uses_latest_prior_step(self) -> None:
        indexed = {
            1: TreeSearchCheckpointRecord(CheckpointId("ckpt-1"), replay_actions=1),
            3: TreeSearchCheckpointRecord(CheckpointId("ckpt-3"), replay_actions=3),
        }

        checkpoint_step, checkpoint = resolve_tree_search_replay_checkpoint(indexed, 2)

        self.assertEqual(checkpoint_step, 1)
        self.assertEqual(checkpoint, indexed[1])

    def test_resolve_tree_search_replay_checkpoint_rejects_missing_ancestor(self) -> None:
        indexed = {
            2: TreeSearchCheckpointRecord(CheckpointId("ckpt-2"), replay_actions=2),
        }

        with self.assertRaisesRegex(ValueError, "no tree-search checkpoint available at or before replay step 1"):
            resolve_tree_search_replay_checkpoint(indexed, 1)

    def test_effective_trace_replay_progress_count_falls_back_to_trace_response_count(self) -> None:
        record = BenchmarkTaskRecord(
            agent_type="iflow",
            task_description=TaskDescription(""),
            task_config=TaskConfig(),
            trace_replay_progress_count=None,
            trace_response_count=7,
        )

        self.assertEqual(effective_trace_replay_progress_count(record), 7)

    def test_compute_summary_averages_metrics(self) -> None:
        rows = [
            {"checkpoint_ms": 10.0, "restore_ms": 20.0},
            {"checkpoint_ms": 30.0, "restore_ms": 40.0},
        ]
        self.assertEqual(
            compute_summary(rows, ["checkpoint_ms", "restore_ms"]),
            {"checkpoint_ms": 20.0, "restore_ms": 30.0},
        )

    def test_compute_summary_aliases_maps_row_fields_to_summary_keys(self) -> None:
        rows = [
            {"recovery_ms_avg": 10.0, "success_ratio": 1.0},
            {"recovery_ms_avg": 14.0, "success_ratio": 0.0},
        ]
        self.assertEqual(
            compute_summary_aliases(rows, {"recovery_ms": "recovery_ms_avg", "success_ratio": "success_ratio"}),
            {"recovery_ms": 12.0, "success_ratio": 0.5},
        )

    def test_compute_telemetry_summary_filters_by_run_id_and_attributes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            telemetry_path = Path(tmp) / "telemetry.jsonl"
            telemetry_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "kind": "metric",
                                "name": "benchmark.recovery_ms",
                                "value": 12.0,
                                "attributes": {"run_id": "run-a", "event_injected": 1},
                            }
                        ),
                        json.dumps(
                            {
                                "kind": "metric",
                                "name": "benchmark.recovery_ms",
                                "value": 50.0,
                                "attributes": {"run_id": "run-a", "event_injected": 0},
                            }
                        ),
                        json.dumps(
                            {
                                "kind": "metric",
                                "name": "benchmark.recovery_ms",
                                "value": 99.0,
                                "attributes": {"run_id": "run-b", "event_injected": 1},
                            }
                        ),
                    ]
                ),
                encoding="utf-8",
            )
            summary = compute_telemetry_summary(
                telemetry_path,
                {"recovery_ms": "benchmark.recovery_ms"},
                run_id="run-a",
                attribute_filters={"recovery_ms": {"event_injected": 1}},
            )
        self.assertEqual(summary, {"recovery_ms": 12.0})

    def test_parse_ipv4_route_networks_ignores_default_and_host_routes(self) -> None:
        self.assertEqual(
            parse_ipv4_route_networks(
                "\n".join(
                    [
                        "default via 172.24.95.253 dev eth0",
                        "10.250.0.0/24 dev acb0 proto kernel scope link src 10.250.0.1",
                        "172.24.95.253 dev eth0 scope link src 172.24.82.236",
                    ]
                )
            ),
            [ipaddress.ip_network("10.250.0.0/24")],
        )

    def test_select_benchmark_network_skips_occupied_routes(self) -> None:
        bridge_ip, network_cidr = select_benchmark_network(
            existing_routes="\n".join(
                [
                    "10.250.0.0/24 dev acb0 proto kernel scope link src 10.250.0.1",
                    "10.250.1.0/24 dev acb1 proto kernel scope link src 10.250.1.1",
                ]
            )
        )
        self.assertEqual((bridge_ip, network_cidr), ("10.250.2.1", "10.250.2.0/24"))

    def test_select_benchmark_network_expands_cidr_for_large_runs(self) -> None:
        bridge_ip, network_cidr = select_benchmark_network(
            existing_routes="10.250.0.0/24 dev acb0 proto kernel scope link src 10.250.0.1",
            required_guest_capacity=640,
        )

        self.assertEqual((bridge_ip, network_cidr), ("10.250.4.1", "10.250.4.0/22"))

    def test_total_actions_reads_payload(self) -> None:
        self.assertEqual(total_actions({"total_actions": 7}), 7)

    def test_select_injected_indices_honors_first_forced_iteration(self) -> None:
        import random

        rng = random.Random(0)
        self.assertEqual(
            select_injected_indices(3, iteration=1, rate=1.0, first_forced_iteration=3, rng=rng),
            [],
        )
        self.assertEqual(
            select_injected_indices(3, iteration=2, rate=1.0, first_forced_iteration=3, rng=rng),
            [],
        )
        forced = select_injected_indices(3, iteration=3, rate=0.0, first_forced_iteration=3, rng=rng)
        self.assertEqual(forced, [0])

    def test_resolve_checkpoint_copy_plan_only_includes_required_ancestors(self) -> None:
        sid = SandboxId("sbx")
        checkpoint_order = [CheckpointId("ckpt-1"), CheckpointId("ckpt-2"), CheckpointId("ckpt-3")]
        process_ref = ArtifactReference(
            kind=ArtifactKind.PROCESS,
            name="process_checkpoint.json",
            relative_path="artifacts/sbx/ckpt-1/process/process_checkpoint.json",
            size_bytes=1,
            sha256="0" * 64,
            metadata={},
        )
        filesystem_ref = ArtifactReference(
            kind=ArtifactKind.FILESYSTEM,
            name="filesystem_checkpoint.json",
            relative_path="artifacts/sbx/ckpt-1/filesystem/filesystem_checkpoint.json",
            size_bytes=1,
            sha256="1" * 64,
            metadata={},
        )
        manifests = {
            CheckpointId("ckpt-1"): CheckpointManifest(
                schema_version="v1",
                checkpoint_id=CheckpointId("ckpt-1"),
                sandbox_id=sid,
                created_at=utc_now(),
                runtime_name="runc",
                runtime_version=None,
                process_artifacts=[process_ref],
                filesystem_artifacts=[filesystem_ref],
                metadata={},
            ).with_integrity(),
            CheckpointId("ckpt-2"): CheckpointManifest(
                schema_version="v1",
                checkpoint_id=CheckpointId("ckpt-2"),
                sandbox_id=sid,
                created_at=utc_now(),
                runtime_name="runc",
                runtime_version=None,
                process_artifacts=[],
                filesystem_artifacts=[],
                metadata={},
            ).with_integrity(),
            CheckpointId("ckpt-3"): CheckpointManifest(
                schema_version="v1",
                checkpoint_id=CheckpointId("ckpt-3"),
                sandbox_id=sid,
                created_at=utc_now(),
                runtime_name="runc",
                runtime_version=None,
                process_artifacts=[process_ref],
                filesystem_artifacts=[],
                metadata={},
            ).with_integrity(),
        }

        self.assertEqual(
            resolve_checkpoint_copy_plan(checkpoint_order, manifests, CheckpointId("ckpt-3")),
            [
                (CheckpointId("ckpt-1"), False, True),
                (CheckpointId("ckpt-3"), True, False),
            ],
        )

    def test_bounded_probability_rejects_invalid_values(self) -> None:
        self.assertEqual(bounded_probability("0.3"), 0.3)
        with self.assertRaises(Exception):
            bounded_probability("3.0")

    def test_resolve_work_dir_host_path_uses_per_sandbox_subdirectory(self) -> None:
        root = Path("/tmp/bench-workdirs")
        self.assertEqual(
            resolve_work_dir_host_path(root, "sandbox-1"),
            root / "sandbox-1",
        )
        self.assertIsNone(resolve_work_dir_host_path(None, "sandbox-1"))

    def test_resolve_work_dir_host_path_makes_relative_roots_absolute(self) -> None:
        self.assertEqual(
            resolve_work_dir_host_path(Path("logs/tmp"), "sandbox-1"),
            (Path.cwd() / "logs/tmp").resolve() / "sandbox-1",
        )

    def test_sandbox_build_context_is_narrow_for_iflow(self) -> None:
        harness = RealHostScenarioHarness(
            provider="openai",
            transfer_delay_ms=0.0,
            scheduler_config=SchedulerConfig(require_change_signal=False),
            scheduler_policy=object(),
            checkpoint_manager_factory=lambda base: base,
            max_workers=1,
        )

        self.assertEqual(
            harness._sandbox_build_context_path("iflow"),
            Path("integrations/sandboxes/iflow").resolve(),
        )
        self.assertEqual(
            harness._sandbox_build_context_path("simulated"),
            Path("integrations/sandboxes/simulated").resolve(),
        )
        self.assertEqual(
            harness._sandbox_image_tag("simulated"),
            "crab-simulated-bench:workspace",
        )

    def test_compose_build_tag_is_stable_for_same_definition(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            compose_file = Path(tmp) / "compose.yaml"
            compose_file.write_text("services: {}\n", encoding="utf-8")
            build_spec = {"context": ".", "dockerfile": "Dockerfile", "args": {"A": "1", "B": "2"}}

            first = sandbox_compose.compose_build_tag(
                compose_file=compose_file,
                service_name="web",
                build_spec=build_spec,
            )
            second = sandbox_compose.compose_build_tag(
                compose_file=compose_file,
                service_name="web",
                build_spec=build_spec,
            )
            changed = sandbox_compose.compose_build_tag(
                compose_file=compose_file,
                service_name="worker",
                build_spec=build_spec,
            )

        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)

    def test_compose_process_args_use_image_defaults_when_service_omits_startup(self) -> None:
        result = sandbox_compose.compose_process_args(
            {},
            image_defaults=ImageRuntimeDefaults(
                entrypoint=("/docker-entrypoint.sh",),
                command=("serve", "--port", "8080"),
            ),
        )
        self.assertEqual(result, ["/docker-entrypoint.sh", "serve", "--port", "8080"])

    def test_compose_mounts_resolve_relative_bind_sources_from_compose_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            compose_dir = root / "dataset" / "scenario"
            compose_dir.mkdir(parents=True, exist_ok=True)
            compose_file = compose_dir / "compose.yaml"
            compose_file.write_text("services: {}\n", encoding="utf-8")

            mounts = sandbox_compose.compose_mounts(
                {
                    "volumes": [
                        "./data:/app/data:ro",
                        {"type": "bind", "source": "../shared", "target": "/mnt/shared"},
                    ]
                },
                compose_file=compose_file,
            )

            self.assertEqual(
                mounts,
                [
                    {
                        "destination": "/app/data",
                        "source": str((compose_dir / "data").resolve()),
                        "type": "bind",
                        "options": ["rbind", "ro"],
                    },
                    {
                        "destination": "/mnt/shared",
                        "source": str((compose_dir / "../shared").resolve()),
                        "type": "bind",
                        "options": ["rbind", "rw"],
                    },
                ],
            )

    def test_clone_host_work_dir_copies_source_contents_for_fork(self) -> None:
        harness = RealHostScenarioHarness(
            provider="openai",
            transfer_delay_ms=0.0,
            scheduler_config=SchedulerConfig(require_change_signal=False),
            scheduler_policy=object(),
            checkpoint_manager_factory=lambda base: base,
            max_workers=1,
            work_dir_host_root=Path("/tmp/placeholder"),
        )
        with tempfile.TemporaryDirectory() as tmp:
            harness.work_dir_host_root = Path(tmp)
            source_dir = resolve_work_dir_host_path(harness.work_dir_host_root, "source-box")
            target_dir = resolve_work_dir_host_path(harness.work_dir_host_root, "fork-box")
            assert source_dir is not None
            assert target_dir is not None
            source_dir.mkdir(parents=True, exist_ok=True)
            (source_dir / "agent_cli.log").write_text("from-source\n", encoding="utf-8")
            (source_dir / "nested").mkdir()
            (source_dir / "nested" / "state.json").write_text('{"step": 1}\n', encoding="utf-8")
            target_dir.mkdir(parents=True, exist_ok=True)
            (target_dir / "stale.txt").write_text("stale\n", encoding="utf-8")

            harness._clone_host_work_dir(SandboxId("source-box"), SandboxId("fork-box"))

            self.assertEqual((target_dir / "agent_cli.log").read_text(encoding="utf-8"), "from-source\n")
            self.assertEqual((target_dir / "nested" / "state.json").read_text(encoding="utf-8"), '{"step": 1}\n')
            self.assertFalse((target_dir / "stale.txt").exists())

    def test_write_bundle_config_adds_work_dir_bind_mount(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle_dir = Path(tmp)
            config_path = bundle_dir / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "linux": {"namespaces": [{"type": "network"}, {"type": "pid"}], "seccomp": {"defaultAction": "SCMP_ACT_ERRNO"}},
                        "mounts": [{"destination": "/proc", "source": "proc", "type": "proc"}],
                        "process": {"terminal": True, "cwd": "/", "args": [], "env": []},
                        "root": {"path": "rootfs", "readonly": True},
                    }
                ),
                encoding="utf-8",
            )

            work_dir_host_path = bundle_dir / "host-work" / "sandbox-1"
            sandbox_bundle.write_bundle_config(
                bundle_dir=bundle_dir,
                llm_base_url="http://127.0.0.1:9000/v1",
                provider="openai",
                sandbox_name="sandbox-1",
                status_port=9001,
                cgroup_path="crab/test/sandbox-1",
                work_dir_host_path=work_dir_host_path,
            )

            payload = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertIn("CRAB_LLM_BASE_URL=http://127.0.0.1:9000/v1", payload["process"]["env"])
            work_mounts = [mount for mount in payload["mounts"] if mount["destination"] == "/work"]
            self.assertEqual(
                work_mounts,
                [
                    {
                        "destination": "/work",
                        "source": str(work_dir_host_path),
                        "type": "bind",
                        "options": ["rbind", "rw"],
                    }
                ],
            )
            self.assertTrue(work_dir_host_path.is_dir())
            self.assertEqual(payload["process"]["cwd"], "/work")

    def test_write_bundle_config_inherits_image_env_workdir_and_user(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle_dir = Path(tmp) / "bundle"
            bundle_dir.mkdir(parents=True, exist_ok=True)
            config_path = bundle_dir / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "linux": {"namespaces": [], "seccomp": {"defaultAction": "SCMP_ACT_ERRNO"}},
                        "mounts": [],
                        "process": {"terminal": True, "cwd": "/", "args": [], "env": []},
                        "root": {"path": "rootfs", "readonly": True},
                    }
                ),
                encoding="utf-8",
            )
            rootfs_dir = Path(tmp) / "rootfs"
            (rootfs_dir / "etc").mkdir(parents=True, exist_ok=True)
            (rootfs_dir / "etc" / "passwd").write_text("app:x:1001:1002::/home/app:/bin/sh\n", encoding="utf-8")
            (rootfs_dir / "etc" / "group").write_text("app:x:1002:\n", encoding="utf-8")

            sandbox_bundle.write_bundle_config(
                bundle_dir=bundle_dir,
                llm_base_url="http://127.0.0.1:9000/v1",
                provider="openai",
                sandbox_name="sandbox-1",
                status_port=9001,
                cgroup_path="crab/test/sandbox-1",
                image_defaults=ImageRuntimeDefaults(
                    environment=("IMAGE_ONLY=1", "PATH=/image/bin"),
                    working_dir="/app",
                    user="app",
                ),
                image_rootfs_dir=rootfs_dir,
            )

            payload = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["process"]["cwd"], "/app")
            self.assertIn("IMAGE_ONLY=1", payload["process"]["env"])
            self.assertIn("PATH=/usr/local/bin:/usr/bin:/bin", payload["process"]["env"])
            self.assertEqual(payload["process"]["user"], {"uid": 1001, "gid": 1002})

    def test_write_bundle_config_applies_resource_limits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle_dir = Path(tmp)
            config_path = bundle_dir / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "linux": {"namespaces": [], "seccomp": {"defaultAction": "SCMP_ACT_ERRNO"}},
                        "mounts": [],
                        "process": {"terminal": True, "cwd": "/", "args": [], "env": []},
                        "root": {"path": "rootfs", "readonly": True},
                    }
                ),
                encoding="utf-8",
            )

            sandbox_bundle.write_bundle_config(
                bundle_dir=bundle_dir,
                llm_base_url="http://127.0.0.1:9000/v1",
                provider="openai",
                sandbox_name="sandbox-rl",
                status_port=9001,
                cgroup_path="crab/test/sandbox-rl",
                resource_limits=sandbox_bundle.SandboxResourceLimits(
                    cpus=4,
                    memory_bytes=2 * 1024 * 1024 * 1024,
                    pids_limit=512,
                ),
            )

            payload = json.loads(config_path.read_text(encoding="utf-8"))
            resources = payload["linux"]["resources"]
            self.assertIn("cpus", resources["cpu"])
            host_cpus = os.cpu_count() or 1
            limit_cpus = min(4, host_cpus)
            slot = 0  # "sandbox-rl" has no integer component → slot 0
            start = (slot * limit_cpus) % host_cpus
            end = start + limit_cpus - 1
            expected = f"{start}-{end}" if limit_cpus > 1 else str(start)
            self.assertEqual(resources["cpu"]["cpus"], expected)
            self.assertEqual(resources["cpu"]["period"], 100_000)
            self.assertEqual(resources["cpu"]["quota"], 400_000)
            self.assertEqual(resources["memory"]["limit"], 2 * 1024 * 1024 * 1024)
            self.assertEqual(resources["pids"]["limit"], 512)
            env = payload["process"]["env"]
            self.assertIn("DJANGO_TEST_PROCESSES=4", env)
            self.assertIn("OMP_NUM_THREADS=4", env)
            self.assertIn("OPENBLAS_NUM_THREADS=4", env)
            self.assertIn("MKL_NUM_THREADS=4", env)
            self.assertIn("NUMEXPR_MAX_THREADS=4", env)
            self.assertIn("LOKY_MAX_CPU_COUNT=4", env)
            mounts = payload.get("mounts", [])
            cpu_overlay_dests = {
                "/sys/devices/system/cpu/online",
                "/sys/devices/system/cpu/possible",
                "/sys/devices/system/cpu/present",
            }
            overlay_mounts = [m for m in mounts if m["destination"] in cpu_overlay_dests]
            self.assertEqual(
                {m["destination"] for m in overlay_mounts},
                cpu_overlay_dests,
            )
            for m in overlay_mounts:
                self.assertEqual(m["type"], "bind")
                self.assertIn("ro", m["options"])
                src = Path(str(m["source"]))
                self.assertTrue(src.is_file())
                self.assertEqual(src.read_text(encoding="utf-8"), "0-3\n")

    def test_write_bundle_config_without_resource_limits_leaves_resources_unset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle_dir = Path(tmp)
            config_path = bundle_dir / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "linux": {"namespaces": [], "seccomp": {"defaultAction": "SCMP_ACT_ERRNO"}},
                        "mounts": [],
                        "process": {"terminal": True, "cwd": "/", "args": [], "env": []},
                        "root": {"path": "rootfs", "readonly": True},
                    }
                ),
                encoding="utf-8",
            )

            sandbox_bundle.write_bundle_config(
                bundle_dir=bundle_dir,
                llm_base_url="http://127.0.0.1:9000/v1",
                provider="openai",
                sandbox_name="sandbox-no-rl",
                status_port=9001,
                cgroup_path="crab/test/sandbox-no-rl",
            )

            payload = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertNotIn("resources", payload["linux"])
            env = payload["process"]["env"]
            self.assertFalse(any(item.startswith("DJANGO_TEST_PROCESSES=") for item in env))
            self.assertFalse(any(item.startswith("OMP_NUM_THREADS=") for item in env))

    def test_sandbox_handle_status_url_uses_status_host(self) -> None:
        handle = SandboxHandle(
            sandbox_id=SandboxId("sbx-status"),
            bundle_dir=Path("/tmp/sbx-status"),
            status_port=8123,
            status_host="10.250.0.22",
            last_status={},
        )

        self.assertEqual(handle.status_url, "http://10.250.0.22:8123/status")

    def test_allocate_benchmark_network_lease_assigns_unique_guest_ips(self) -> None:
        manager = sandbox_network.BenchmarkNetworkManager()

        with patch("integrations.sandboxes.runtime.network.subprocess.run") as run:
            first = manager.allocate_lease(SandboxId("sbx-a"))
            second = manager.allocate_lease(SandboxId("sbx-b"))

        self.assertNotEqual(first.guest_ip, second.guest_ip)
        self.assertEqual(first.namespace_path.name, first.namespace_name)
        self.assertEqual(second.namespace_path.name, second.namespace_name)
        self.assertTrue(run.called)

    def test_allocate_benchmark_network_lease_respects_configured_network_capacity(self) -> None:
        manager = sandbox_network.BenchmarkNetworkManager()
        manager._network_cidr = "10.250.0.0/30"
        manager._bridge_ip = "10.250.0.1"

        with patch("integrations.sandboxes.runtime.network.subprocess.run"):
            lease = manager.allocate_lease(SandboxId("sbx-a"))
            self.assertEqual(lease.guest_ip, "10.250.0.2")
            with self.assertRaisesRegex(RuntimeError, "max_guests=1"):
                manager.allocate_lease(SandboxId("sbx-b"))

    def test_benchmark_network_manager_configure_accepts_large_env_override(self) -> None:
        manager = sandbox_network.BenchmarkNetworkManager()

        with patch.dict("os.environ", {"CRAB_BENCHMARK_NETWORK_CIDR": "10.250.8.0/22"}, clear=False):
            manager.configure(expected_sandboxes=640)

        self.assertEqual(manager.network_cidr, "10.250.8.0/22")
        self.assertEqual(manager.bridge_ip, "10.250.8.1")
        self.assertGreaterEqual(
            benchmark_network_guest_capacity(ipaddress.ip_network(manager.network_cidr)),
            640,
        )

    def test_benchmark_network_manager_configure_rejects_small_env_override(self) -> None:
        manager = sandbox_network.BenchmarkNetworkManager()

        with patch.dict("os.environ", {"CRAB_BENCHMARK_NETWORK_CIDR": "10.250.8.0/24"}, clear=False):
            with self.assertRaisesRegex(ValueError, "supports at most 253 guest sandboxes, need 640"):
                manager.configure(expected_sandboxes=640)

    def test_release_benchmark_network_lease_cleans_up_ip_mapping(self) -> None:
        manager = sandbox_network.BenchmarkNetworkManager()

        with patch("integrations.sandboxes.runtime.network.subprocess.run"):
            lease = manager.allocate_lease(SandboxId("sbx-a"))
            manager.register_guest_ip(lease.guest_ip, SandboxId("sbx-a"))
            manager.release_lease(SandboxId("sbx-a"))

        self.assertIsNone(manager.resolve_sandbox_id(lease.guest_ip))
        self.assertIsNone(manager.lease_for(SandboxId("sbx-a")))

    def test_release_benchmark_network_lease_returns_guest_ip_to_pool(self) -> None:
        manager = sandbox_network.BenchmarkNetworkManager()

        with patch("integrations.sandboxes.runtime.network.subprocess.run"):
            first = manager.allocate_lease(SandboxId("sbx-a"))
            second = manager.allocate_lease(SandboxId("sbx-b"))
            manager.release_lease(SandboxId("sbx-a"))
            reused = manager.allocate_lease(SandboxId("sbx-c"))

        self.assertEqual(first.guest_ip, "10.250.0.2")
        self.assertEqual(second.guest_ip, "10.250.0.3")
        self.assertEqual(reused.guest_ip, first.guest_ip)

    def test_repair_benchmark_network_lease_uses_configured_prefix_length(self) -> None:
        manager = sandbox_network.BenchmarkNetworkManager()
        manager._network_cidr = "10.250.8.0/22"
        manager._bridge_ip = "10.250.8.1"

        commands: list[list[str]] = []

        def fake_run(cmd, *args, **kwargs):
            commands.append(list(cmd))
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with patch("integrations.sandboxes.runtime.network.subprocess.run", side_effect=fake_run):
            lease = manager.allocate_lease(SandboxId("sbx-a"))
            self.assertTrue(manager.repair_lease(SandboxId("sbx-a")))

        self.assertIn(
            ["ip", "netns", "exec", lease.namespace_name, "ip", "addr", "replace", f"{lease.guest_ip}/22", "dev", "eth0"],
            commands,
        )

    def test_resolve_interceptor_sandbox_id_uses_registered_guest_ip_for_any_benchmark(self) -> None:
        harness = RealHostScenarioHarness(
            provider="openai",
            transfer_delay_ms=0.0,
            scheduler_config=SchedulerConfig(
                min_checkpoint_interval_seconds=0.0,
                force_checkpoint_after_seconds=0.0,
                require_change_signal=False,
            ),
            scheduler_policy=object(),
            checkpoint_manager_factory=lambda base: base,
            max_workers=1,
        )
        harness.network_manager.register_guest_ip("10.250.0.42", SandboxId("spot-0"))

        self.assertEqual(
            harness.resolve_interceptor_sandbox_id("10.250.0.42", {}, b""),
            "spot-0",
        )
        self.assertIsNone(harness.resolve_interceptor_sandbox_id("10.250.0.43", {}, b""))

    def test_resolve_interceptor_sandbox_id_prefers_explicit_header_over_guest_ip_mapping(self) -> None:
        harness = RealHostScenarioHarness(
            provider="openai",
            transfer_delay_ms=0.0,
            scheduler_config=SchedulerConfig(
                min_checkpoint_interval_seconds=0.0,
                force_checkpoint_after_seconds=0.0,
                require_change_signal=False,
            ),
            scheduler_policy=object(),
            checkpoint_manager_factory=lambda base: base,
            max_workers=1,
        )
        harness.network_manager.register_guest_ip("10.250.0.42", SandboxId("spot-0"))

        self.assertEqual(
            harness.resolve_interceptor_sandbox_id(
                "10.250.0.42",
                {"X-Agent-Sandbox-Id": "fault-1"},
                b"",
            ),
            "fault-1",
        )

    def test_allocate_benchmark_network_lease_creates_bridge_once_under_concurrency(self) -> None:
        manager = sandbox_network.BenchmarkNetworkManager()

        call_lock = threading.Lock()
        bridge_add_commands: list[list[str]] = []

        def fake_run(cmd, *args, **kwargs):
            if cmd[:3] == ["ip", "link", "add"] and len(cmd) >= 6 and cmd[4:6] == ["type", "bridge"]:
                with call_lock:
                    bridge_add_commands.append(list(cmd))
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with patch("integrations.sandboxes.runtime.network.subprocess.run", side_effect=fake_run):
            with ThreadPoolExecutor(max_workers=2) as executor:
                first_future = executor.submit(manager.allocate_lease, SandboxId("sbx-a"))
                second_future = executor.submit(manager.allocate_lease, SandboxId("sbx-b"))
                first = first_future.result()
                second = second_future.result()

        self.assertNotEqual(first.guest_ip, second.guest_ip)
        self.assertEqual(len(bridge_add_commands), 1)

    def test_launch_sandbox_and_task_records_task_metadata(self) -> None:
        harness = RealHostScenarioHarness(
            provider="openai",
            transfer_delay_ms=0.0,
            scheduler_config=SchedulerConfig(require_change_signal=False),
            scheduler_policy=object(),
            checkpoint_manager_factory=lambda base: base,
            max_workers=1,
        )
        handle = SandboxHandle(
            sandbox_id=SandboxId("sbx-launch"),
            bundle_dir=Path("/tmp/sbx-launch"),
            status_port=8123,
            last_status={},
        )
        task_description = TaskDescription("solve a task")
        task_config = TaskConfig()

        with patch.object(harness, "launch_sandbox", return_value=handle) as launch_sandbox:
            with patch.object(harness, "launch_task") as launch_task:
                result = harness.launch_sandbox_and_task(
                    "sbx-launch",
                    agent_type="simulated",
                    llm_service_type=None,
                    task_description=task_description,
                    task_config=task_config,
                )

        self.assertIs(result, handle)
        launch_sandbox.assert_called_once_with(
            "sbx-launch",
            agent_type="simulated",
            llm_service_type=None,
            llm_service_config=None,
            task_description=task_description,
            task_config=task_config,
        )
        launch_task.assert_called_once_with("simulated", task_description, task_config, "sbx-launch")

    def test_launch_task_creates_task_run_and_future(self) -> None:
        harness = RealHostScenarioHarness(
            provider="openai",
            transfer_delay_ms=0.0,
            scheduler_config=SchedulerConfig(require_change_signal=False),
            scheduler_policy=object(),
            checkpoint_manager_factory=lambda base: base,
            max_workers=1,
        )
        handle = SandboxHandle(
            sandbox_id=SandboxId("sbx-agent"),
            bundle_dir=Path("/tmp/sbx-agent"),
            status_port=8123,
            last_status={},
        )
        harness._sandbox_by_id[handle.sandbox_id] = handle
        mock_task_run = Mock()
        mock_future = Mock()
        task_description = TaskDescription("progress")
        task_config = TaskConfig()

        with patch.object(harness, "build_task_run", return_value=mock_task_run) as build_task_run:
            with patch.object(harness._task_executor, "submit", return_value=mock_future) as submit:
                returned = harness.launch_task("simulated", task_description, task_config, "sbx-agent")

        self.assertIs(returned, mock_task_run)
        self.assertEqual(handle.agent_type, "simulated")
        self.assertEqual(handle.task_description, task_description)
        self.assertEqual(handle.task_config, task_config)
        self.assertIs(handle.task_run, mock_task_run)
        self.assertIs(handle.task_future, mock_future)
        build_task_run.assert_called_once_with("simulated", handle, task_description, task_config)
        submit.assert_called_once()
        submitted_callable = submit.call_args.args[0]
        self.assertTrue(callable(submitted_callable))
        self.assertIsNot(submitted_callable, mock_task_run.perform_task)

    def test_launch_task_emits_queue_wait_and_run_duration_metrics(self) -> None:
        harness = RealHostScenarioHarness(
            provider="openai",
            transfer_delay_ms=0.0,
            scheduler_config=SchedulerConfig(require_change_signal=False),
            scheduler_policy=object(),
            checkpoint_manager_factory=lambda base: base,
            max_workers=1,
        )
        harness.telemetry = InMemoryTelemetrySink()
        handle = SandboxHandle(
            sandbox_id=SandboxId("sbx-agent-telemetry"),
            bundle_dir=Path("/tmp/sbx-agent-telemetry"),
            status_port=8123,
            last_status={},
        )
        harness._sandbox_by_id[handle.sandbox_id] = handle

        class _TaskRun:
            def __init__(self) -> None:
                self.perform_task = Mock(side_effect=lambda: time.sleep(0.01))

        task_run = _TaskRun()

        def _submit(fn):
            future: Future[None] = Future()
            try:
                fn()
            except Exception as exc:
                future.set_exception(exc)
            else:
                future.set_result(None)
            return future

        with patch.object(harness, "build_task_run", return_value=task_run), patch.object(
            harness._task_executor,
            "submit",
            side_effect=_submit,
        ):
            returned = harness.launch_task("simulated", TaskDescription("progress"), TaskConfig(), "sbx-agent-telemetry")

        self.assertIs(returned, task_run)
        metric_values = {name: value for name, value, _attributes in harness.telemetry.metrics}
        self.assertIn("benchmark.task.queue_wait_ms", metric_values)
        self.assertIn("benchmark.task.run.duration_ms", metric_values)
        self.assertIn("benchmark.task.duration_ms", metric_values)
        self.assertGreaterEqual(metric_values["benchmark.task.queue_wait_ms"], 0.0)
        self.assertGreater(metric_values["benchmark.task.run.duration_ms"], 0.0)
        self.assertGreaterEqual(
            metric_values["benchmark.task.duration_ms"],
            metric_values["benchmark.task.run.duration_ms"],
        )

    def test_build_task_run_passes_explicit_runtime_inputs(self) -> None:
        class _RecordingAgent(BaseAgent):
            agent_type = "recording"

            def perform_task(self) -> None:
                return None

        harness = RealHostScenarioHarness(
            provider="openai",
            transfer_delay_ms=0.0,
            scheduler_config=SchedulerConfig(require_change_signal=False),
            scheduler_policy=object(),
            checkpoint_manager_factory=lambda base: base,
            max_workers=1,
        )
        harness.root = Path("/tmp/harness-root")
        harness.runtime_state_root = Path("/tmp/runtime-root")
        harness.telemetry = InMemoryTelemetrySink()
        sandbox = SandboxHandle(
            sandbox_id=SandboxId("sbx-explicit"),
            bundle_dir=Path("/tmp/sbx-explicit"),
            status_port=8123,
            last_status={},
            llm_base_url="http://127.0.0.1:43123/v1",
        )

        with patch.object(harness, "get_agent_class", return_value=_RecordingAgent):
            agent = harness.build_task_run("recording", sandbox, TaskDescription("go"), TaskConfig())

        self.assertEqual(agent.runtime_state_root, Path("/tmp/runtime-root"))
        self.assertEqual(agent.agent_host_dir, Path("/tmp/harness-root/recording/sbx-explicit"))
        self.assertEqual(agent.llm_base_url, "http://127.0.0.1:43123/v1")
        self.assertIs(agent.telemetry, harness.telemetry)

    def test_base_agent_post_task_finish_deactivates_sandbox_runtime(self) -> None:
        class _RecordingAgent(BaseAgent):
            agent_type = "recording"

            def perform_task(self) -> None:
                return None

        sandbox = SandboxHandle(
            sandbox_id=SandboxId("sbx-post-finish"),
            bundle_dir=Path("/tmp/sbx-post-finish"),
            status_port=8123,
            last_status={},
        )
        agent = _RecordingAgent(
            sandbox,
            TaskDescription("go"),
            TaskConfig(),
            runtime_state_root=Path("/tmp/runtime-root"),
            sandbox_manager=SimpleNamespace(delete_runtime=Mock()),
        )

        agent.post_task_finish()

        agent.sandbox_manager.delete_runtime.assert_called_once_with(
            SandboxId("sbx-post-finish"),
            force=True,
            ignore_missing=True,
        )

    def test_simulated_agent_calls_post_task_finish_after_task_completion(self) -> None:
        sandbox = SandboxHandle(
            sandbox_id=SandboxId("sbx-simulated-post-finish"),
            bundle_dir=Path("/tmp/sbx-simulated-post-finish"),
            status_port=8123,
            last_status={},
        )
        agent = SimulatedAgent(
            sandbox,
            TaskDescription("go"),
            TaskConfig(),
            runtime_state_root=Path("/tmp/runtime-root"),
        )

        with patch.object(agent, "wait_for_progress") as wait_for_progress, patch.object(
            agent, "wait_for_sandbox_exit"
        ) as wait_for_sandbox_exit, patch.object(agent, "post_task_finish") as post_task_finish:
            agent.perform_task()

        wait_for_progress.assert_called_once()
        wait_for_sandbox_exit.assert_called_once_with()
        post_task_finish.assert_called_once_with()

    def test_launch_task_requests_stop_before_replacing_running_task(self) -> None:
        harness = RealHostScenarioHarness(
            provider="openai",
            transfer_delay_ms=0.0,
            scheduler_config=SchedulerConfig(require_change_signal=False),
            scheduler_policy=object(),
            checkpoint_manager_factory=lambda base: base,
            max_workers=1,
        )
        handle = SandboxHandle(
            sandbox_id=SandboxId("sbx-agent-stop"),
            bundle_dir=Path("/tmp/sbx-agent-stop"),
            status_port=8123,
            last_status={},
        )
        existing_task_run = Mock()
        existing_future = Mock(done=Mock(return_value=False))
        handle.task_run = existing_task_run
        handle.task_future = existing_future
        harness._sandbox_by_id[handle.sandbox_id] = handle
        replacement_task_run = Mock()
        replacement_future = Mock()

        with patch.object(harness, "build_task_run", return_value=replacement_task_run), patch.object(
            harness._task_executor,
            "submit",
            return_value=replacement_future,
        ):
            returned = harness.launch_task("simulated", TaskDescription("progress"), TaskConfig(), "sbx-agent-stop")

        existing_task_run.request_stop.assert_called_once()
        existing_future.cancel.assert_called_once()
        self.assertIs(returned, replacement_task_run)
        self.assertIs(handle.task_run, replacement_task_run)
        self.assertIs(handle.task_future, replacement_future)

    def test_relaunch_sandbox_creates_fresh_task_run(self) -> None:
        harness = RealHostScenarioHarness(
            provider="openai",
            transfer_delay_ms=0.0,
            scheduler_config=SchedulerConfig(require_change_signal=False),
            scheduler_policy=object(),
            checkpoint_manager_factory=lambda base: base,
            max_workers=1,
        )
        sandbox = SandboxHandle(
            sandbox_id=SandboxId("sbx-relaunch"),
            bundle_dir=Path("/tmp/sbx-relaunch"),
            status_port=8123,
            last_status={},
            agent_type="iflow",
            llm_base_url="http://10.250.0.1:43123/v1",
            task_description=TaskDescription("resume"),
            task_config=TaskConfig(),
        )
        harness.sandbox_manager = SimpleNamespace(
            describe=lambda sandbox_id: SimpleNamespace(metadata={"zfs_dataset": "", "bundle_path": "/tmp/bundle"}),
            delete_runtime=Mock(),
            destroy_filesystem_dataset=Mock(),
            launch=Mock(),
        )
        harness.base_inspector = SimpleNamespace(upsert_snapshot=Mock())
        harness.runtime_state_root = Path("/tmp/runtime")

        fake_task_run = SimpleNamespace(
            wait_for_task_ready=Mock(),
            poll_status=Mock(return_value={"total_actions": 0}),
        )
        with patch.object(harness, "launch_task", side_effect=lambda *args, **kwargs: setattr(sandbox, "task_run", fake_task_run)):
            harness.relaunch_sandbox(sandbox)

        self.assertIs(sandbox.task_run, fake_task_run)
        fake_task_run.wait_for_task_ready.assert_called_once()
        fake_task_run.poll_status.assert_called_once()
        self.assertEqual(sandbox.llm_base_url, "http://10.250.0.1:43123/v1")

    def test_relaunch_sandbox_reuses_fault_resilient_task_run(self) -> None:
        harness = RealHostScenarioHarness(
            provider="openai",
            transfer_delay_ms=0.0,
            scheduler_config=SchedulerConfig(require_change_signal=False),
            scheduler_policy=object(),
            checkpoint_manager_factory=lambda base: base,
            max_workers=1,
        )
        sandbox = SandboxHandle(
            sandbox_id=SandboxId("sbx-relaunch-reuse"),
            bundle_dir=Path("/tmp/sbx-relaunch-reuse"),
            status_port=8123,
            last_status={},
            agent_type="iflow",
            llm_base_url="http://10.250.0.1:43123/v1",
            task_description=TaskDescription("resume"),
            task_config=TaskConfig(),
        )
        harness.sandbox_manager = SimpleNamespace(
            describe=lambda sandbox_id: SimpleNamespace(metadata={"zfs_dataset": "", "bundle_path": "/tmp/bundle"}),
            delete_runtime=Mock(),
            destroy_filesystem_dataset=Mock(),
            launch=Mock(),
        )
        harness.base_inspector = SimpleNamespace(upsert_snapshot=Mock())
        harness.runtime_state_root = Path("/tmp/runtime")

        fake_task_run = SimpleNamespace(
            survives_fault_relaunch=Mock(return_value=True),
            wait_for_task_ready=Mock(),
            on_restore_complete=Mock(),
            poll_status=Mock(return_value={"total_actions": 3}),
        )
        sandbox.task_run = fake_task_run
        sandbox.task_future = SimpleNamespace(done=Mock(return_value=False))

        with patch.object(harness, "launch_task") as launch_task:
            harness.relaunch_sandbox(sandbox)

        launch_task.assert_not_called()
        self.assertIs(sandbox.task_run, fake_task_run)
        fake_task_run.wait_for_task_ready.assert_called_once()
        fake_task_run.on_restore_complete.assert_called_once()
        fake_task_run.poll_status.assert_called_once()

    def test_relaunch_sandbox_preserves_restored_filesystem_when_requested(self) -> None:
        harness = RealHostScenarioHarness(
            provider="openai",
            transfer_delay_ms=0.0,
            scheduler_config=SchedulerConfig(require_change_signal=False),
            scheduler_policy=object(),
            checkpoint_manager_factory=lambda base: base,
            max_workers=1,
        )
        sandbox = SandboxHandle(
            sandbox_id=SandboxId("sbx-relaunch-preserve-fs"),
            bundle_dir=Path("/tmp/sbx-relaunch-preserve-fs"),
            status_port=8123,
            last_status={},
            agent_type="claude_code",
            llm_service_type="claude_code_trace_replay",
            llm_base_url="http://10.250.0.1:43123/v1",
            task_description=TaskDescription("resume"),
            task_config=TaskConfig(),
        )
        runtime_launch = Mock()
        harness.sandbox_manager = SimpleNamespace(
            describe=lambda sandbox_id: SimpleNamespace(
                metadata={
                    "sandbox_id": str(sandbox.sandbox_id),
                    "zfs_dataset": "pool/crab/sbx-relaunch-preserve-fs",
                    "bundle_path": "/tmp/bundle",
                }
            ),
            delete_runtime=Mock(),
            destroy_filesystem_dataset=Mock(),
            launch=runtime_launch,
        )
        harness.base_inspector = SimpleNamespace(upsert_snapshot=Mock())
        harness.runtime_state_root = Path("/tmp/runtime")

        fake_task_run = SimpleNamespace(
            survives_fault_relaunch=Mock(return_value=True),
            wait_for_task_ready=Mock(),
            on_restore_complete=Mock(),
            poll_status=Mock(return_value={"total_actions": 3}),
        )
        sandbox.task_run = fake_task_run
        sandbox.task_future = SimpleNamespace(done=Mock(return_value=False))

        with patch.object(harness, "_destroy_filesystem_dataset") as destroy_dataset:
            harness.relaunch_sandbox(sandbox, preserve_filesystem_state=True)

        destroy_dataset.assert_not_called()
        runtime_launch.assert_called_once()
        launch_metadata = runtime_launch.call_args.args[1]
        self.assertTrue(launch_metadata["_crab_runtime_reuse_existing_rootfs"])
        self.assertEqual(launch_metadata["zfs_dataset"], "pool/crab/sbx-relaunch-preserve-fs")

    def test_harness_exit_requests_stop_for_running_tasks(self) -> None:
        harness = RealHostScenarioHarness(
            provider="openai",
            transfer_delay_ms=0.0,
            scheduler_config=SchedulerConfig(require_change_signal=False),
            scheduler_policy=object(),
            checkpoint_manager_factory=lambda base: base,
            max_workers=1,
        )
        sandbox = SandboxHandle(
            sandbox_id=SandboxId("sbx-exit"),
            bundle_dir=Path("/tmp/sbx-exit"),
            status_port=8123,
            last_status={},
            task_run=Mock(),
        )
        harness.sandboxes = [sandbox]
        harness.system = None
        harness.interceptor = None
        harness.executor = None
        harness.runtime_state_root = None
        events: list[str] = []
        harness.runtime = SimpleNamespace(
            delete_runtime=Mock(side_effect=lambda *args, **kwargs: events.append("delete_runtime"))
        )
        harness.pool_name = None
        harness.llm_server = None
        harness.llm_thread = None
        harness.network_manager.cleanup = Mock()
        harness._stop_host_inspector_server = Mock()
        harness._task_executor.shutdown = Mock(side_effect=lambda *args, **kwargs: events.append("shutdown"))

        harness.__exit__(None, None, None)

        sandbox.task_run.request_stop.assert_called_once()
        harness.runtime.delete_runtime.assert_called_once_with(
            SandboxId("sbx-exit"),
            force=True,
            ignore_missing=True,
        )
        harness.network_manager.cleanup.assert_called_once_with()
        harness._task_executor.shutdown.assert_called_once_with(wait=True, cancel_futures=True)
        # task_executor.shutdown must run before runtime deletion: deleting
        # runtimes with in-flight tasks mid-turn makes the next `runc exec`
        # fail and hang shutdown in a restore-wait (see harness.__exit__).
        self.assertEqual(events, ["shutdown", "delete_runtime"])

    def test_launch_sandbox_from_docker_compose_file_translates_supported_service(self) -> None:
        harness = RealHostScenarioHarness(
            provider="openai",
            transfer_delay_ms=0.0,
            scheduler_config=SchedulerConfig(require_change_signal=False),
            scheduler_policy=object(),
            checkpoint_manager_factory=lambda base: base,
            max_workers=1,
        )
        harness.root = Path(tempfile.mkdtemp(prefix="crab_compose_test_"))
        harness.base_inspector = SimpleNamespace(upsert_snapshot=Mock())
        launch_mock = Mock()
        harness.system = SimpleNamespace(sandbox_manager=SimpleNamespace(launch=launch_mock))
        config_dir = harness.root / "bundle"
        config_dir.mkdir(parents=True, exist_ok=True)
        (config_dir / "config.json").write_text(
            json.dumps(
                {
                    "linux": {"namespaces": [], "cgroupsPath": ""},
                    "mounts": [],
                    "process": {"terminal": False, "cwd": "/", "args": [], "env": []},
                    "root": {"path": "rootfs", "readonly": False},
                }
            ),
            encoding="utf-8",
        )
        handle = SandboxHandle(
            sandbox_id=SandboxId("sbx-compose"),
            bundle_dir=config_dir,
            status_port=8123,
            status_host="127.0.0.1",
            last_status={},
        )
        compose_file = harness.root / "compose.yaml"
        compose_file.write_text(
            """
services:
  app:
    image: alpine:3.20
    command: "echo hello"
    working_dir: /work
    environment:
      HELLO: world
""".strip(),
            encoding="utf-8",
        )
        with patch.object(harness.network_manager, "allocate_lease", return_value=SimpleNamespace(guest_ip="10.250.0.2")):
            with patch.object(harness, "_prepare_sandbox_handle", return_value=(handle, None)):
                with patch(
                    "integrations.sandboxes.runtime.compose.inspect_image_runtime_defaults",
                    return_value=ImageRuntimeDefaults(),
                ):
                    with patch("integrations.sandboxes.runtime.compose.export_image_rootfs", return_value=harness.root / "rootfs"):
                        result = harness.launch_sandbox_from_docker_compose_file(
                            compose_file,
                            None,
                            sandbox_name="sbx-compose",
                            service_name="app",
                        )

        self.assertEqual(result.launch_source, "compose")
        launch_mock.assert_called_once()
        metadata = launch_mock.call_args.args[1]
        self.assertEqual(metadata["compose_service_name"], "app")

    def test_launch_sandbox_from_docker_compose_file_materializes_termnius_tests(self) -> None:
        harness = RealHostScenarioHarness(
            provider="openai",
            transfer_delay_ms=0.0,
            scheduler_config=SchedulerConfig(require_change_signal=False),
            scheduler_policy=object(),
            checkpoint_manager_factory=lambda base: base,
            max_workers=1,
        )
        harness.root = Path(tempfile.mkdtemp(prefix="crab_compose_termnius_test_"))
        harness.base_inspector = SimpleNamespace(upsert_snapshot=Mock())
        launch_mock = Mock()
        harness.system = SimpleNamespace(sandbox_manager=SimpleNamespace(launch=launch_mock))
        config_dir = harness.root / "bundle"
        config_dir.mkdir(parents=True, exist_ok=True)
        (config_dir / "config.json").write_text(
            json.dumps(
                {
                    "linux": {"namespaces": [], "cgroupsPath": ""},
                    "mounts": [],
                    "process": {"terminal": False, "cwd": "/", "args": [], "env": []},
                    "root": {"path": "rootfs", "readonly": False},
                }
            ),
            encoding="utf-8",
        )
        handle = SandboxHandle(
            sandbox_id=SandboxId("sbx-compose-tests"),
            bundle_dir=config_dir,
            status_port=8123,
            status_host="127.0.0.1",
            last_status={},
        )
        compose_file = harness.root / "compose.yaml"
        compose_file.write_text(
            "\n".join(
                [
                    "services:",
                    "  client:",
                    "    image: alpine:3.20",
                    "    command: \"echo hello\"",
                ]
            ),
            encoding="utf-8",
        )
        task_root = harness.root / "hello-world"
        (task_root / "tests").mkdir(parents=True, exist_ok=True)
        (task_root / "tests" / "test_outputs.py").write_text("", encoding="utf-8")
        (task_root / "run-tests.sh").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")

        with patch.object(harness.network_manager, "allocate_lease", return_value=SimpleNamespace(guest_ip="10.250.0.2")):
            with patch.object(harness, "_prepare_sandbox_handle", return_value=(handle, None)):
                with patch(
                    "integrations.sandboxes.runtime.compose.inspect_image_runtime_defaults",
                    return_value=ImageRuntimeDefaults(),
                ):
                    with patch("integrations.sandboxes.runtime.compose.export_image_rootfs", return_value=harness.root / "rootfs"):
                        harness.launch_sandbox_from_docker_compose_file(
                            compose_file,
                            None,
                            sandbox_name="sbx-compose-tests",
                            service_name="client",
                            task_root=task_root,
                        )

        metadata = launch_mock.call_args.args[1]
        self.assertIn(
            {"source": str(task_root / "tests"), "destination": "/tests"},
            metadata["rootfs_copy_paths"],
        )
        self.assertIn(
            {"source": str(task_root / "run-tests.sh"), "destination": "/tests/run-tests.sh"},
            metadata["rootfs_copy_paths"],
        )
        self.assertIn("tests", metadata["rootfs_init_dirs"])

    def test_launch_sandbox_from_docker_compose_file_inherits_image_env_workdir_and_user(self) -> None:
        harness = RealHostScenarioHarness(
            provider="openai",
            transfer_delay_ms=0.0,
            scheduler_config=SchedulerConfig(require_change_signal=False),
            scheduler_policy=object(),
            checkpoint_manager_factory=lambda base: base,
            max_workers=1,
        )
        harness.root = Path(tempfile.mkdtemp(prefix="crab_compose_defaults_test_"))
        harness.base_inspector = SimpleNamespace(upsert_snapshot=Mock())
        launch_mock = Mock()
        harness.system = SimpleNamespace(sandbox_manager=SimpleNamespace(launch=launch_mock))
        config_dir = harness.root / "bundle"
        config_dir.mkdir(parents=True, exist_ok=True)
        (config_dir / "config.json").write_text(
            json.dumps(
                {
                    "linux": {"namespaces": [], "cgroupsPath": ""},
                    "mounts": [],
                    "process": {"terminal": False, "cwd": "/", "args": [], "env": []},
                    "root": {"path": "rootfs", "readonly": False},
                }
            ),
            encoding="utf-8",
        )
        handle = SandboxHandle(
            sandbox_id=SandboxId("sbx-compose-defaults"),
            bundle_dir=config_dir,
            status_port=8123,
            status_host="127.0.0.1",
            last_status={},
        )
        compose_file = harness.root / "compose.yaml"
        compose_file.write_text(
            """
services:
  app:
    image: example:latest
""".strip(),
            encoding="utf-8",
        )
        env_file = harness.root / ".env"
        env_file.write_text("", encoding="utf-8")
        rootfs_dir = harness.root / "rootfs"
        (rootfs_dir / "etc").mkdir(parents=True, exist_ok=True)
        (rootfs_dir / "etc" / "passwd").write_text("app:x:1001:1002::/home/app:/bin/sh\n", encoding="utf-8")
        (rootfs_dir / "etc" / "group").write_text("app:x:1002:\n", encoding="utf-8")

        with patch.object(harness.network_manager, "allocate_lease", return_value=SimpleNamespace(guest_ip="10.250.0.2")):
            with patch.object(harness, "_prepare_sandbox_handle", return_value=(handle, None)):
                with patch(
                    "integrations.sandboxes.runtime.compose.inspect_image_runtime_defaults",
                    return_value=ImageRuntimeDefaults(
                        environment=("IMAGE_ONLY=1",),
                        working_dir="/srv/app",
                        user="app",
                        entrypoint=("python",),
                        command=("-m", "http.server"),
                    ),
                ):
                    with patch("integrations.sandboxes.runtime.compose.export_image_rootfs", return_value=rootfs_dir):
                        harness.launch_sandbox_from_docker_compose_file(
                            compose_file,
                            env_file,
                            sandbox_name="sbx-compose-defaults",
                            service_name="app",
                        )

        payload = json.loads((config_dir / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["process"]["cwd"], "/srv/app")
        self.assertEqual(payload["process"]["args"], ["python", "-m", "http.server"])
        self.assertIn("IMAGE_ONLY=1", payload["process"]["env"])
        self.assertEqual(payload["process"]["user"], {"uid": 1001, "gid": 1002})
        launch_mock.assert_called_once()

    def test_launch_sandbox_preserves_image_defaults_when_configure_bundle_is_noop(self) -> None:
        harness = RealHostScenarioHarness(
            provider="openai",
            transfer_delay_ms=0.0,
            scheduler_config=SchedulerConfig(require_change_signal=False),
            scheduler_policy=object(),
            checkpoint_manager_factory=lambda base: base,
            max_workers=1,
        )
        harness.root = Path(tempfile.mkdtemp(prefix="crab_launch_defaults_test_"))
        harness.base_inspector = SimpleNamespace(upsert_snapshot=Mock())
        harness.system = SimpleNamespace(sandbox_manager=SimpleNamespace(launch=Mock()))
        harness.interceptor = SimpleNamespace(port=43123)
        harness.llm_server = SimpleNamespace(benchmark_llm_router=SimpleNamespace(register_sandbox=Mock()))

        config_dir = harness.root / "bundle"
        config_dir.mkdir(parents=True, exist_ok=True)
        (config_dir / "config.json").write_text(
            json.dumps(
                {
                    "linux": {"namespaces": [], "cgroupsPath": ""},
                    "mounts": [],
                    "process": {"terminal": False, "cwd": "/", "args": [], "env": []},
                    "root": {"path": "rootfs", "readonly": False},
                }
            ),
            encoding="utf-8",
        )
        handle = SandboxHandle(
            sandbox_id=SandboxId("sbx-launch-defaults"),
            bundle_dir=config_dir,
            status_port=8123,
            status_host="127.0.0.1",
            last_status={},
        )
        rootfs_dir = harness.root / "rootfs"
        (rootfs_dir / "etc").mkdir(parents=True, exist_ok=True)
        (rootfs_dir / "etc" / "passwd").write_text("app:x:1001:1002::/home/app:/bin/sh\n", encoding="utf-8")
        (rootfs_dir / "etc" / "group").write_text("app:x:1002:\n", encoding="utf-8")
        sandbox_image = SimpleNamespace(
            exported_rootfs=rootfs_dir,
            image_defaults=ImageRuntimeDefaults(
                environment=("IMAGE_ONLY=1", "SHARED=image"),
                working_dir="/app",
                user="app",
            ),
        )

        def prepare_handle(*args, **kwargs):
            sandbox_bundle.write_bundle_config(
                bundle_dir=config_dir,
                llm_base_url="http://127.0.0.1:43123/v1",
                provider="openai",
                sandbox_name="sbx-launch-defaults",
                status_port=8123,
                cgroup_path="crab/test/sbx-launch-defaults",
                image_defaults=kwargs["image_defaults"],
                image_rootfs_dir=kwargs["image_rootfs_dir"],
            )
            return handle, None

        task_run = SimpleNamespace(
            prepare_sandbox=Mock(),
            configure_bundle=Mock(),
            rootfs_init_dirs=Mock(return_value=[]),
            extra_launch_metadata=Mock(return_value={}),
            wait_for_task_ready=Mock(),
        )

        with patch.object(harness, "ensure_sandbox_image", return_value=sandbox_image):
            with patch.object(harness, "_prepare_sandbox_handle", side_effect=prepare_handle):
                with patch.object(harness, "build_task_run", return_value=task_run):
                    result = harness.launch_sandbox("sbx-launch-defaults", agent_type="simulated")

        self.assertIs(result, handle)
        payload = json.loads((config_dir / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["process"]["cwd"], "/app")
        self.assertIn("IMAGE_ONLY=1", payload["process"]["env"])
        self.assertIn("SHARED=image", payload["process"]["env"])
        self.assertEqual(payload["process"]["user"], {"uid": 1001, "gid": 1002})

    def test_launch_sandbox_uses_host_network_for_simulated_agents(self) -> None:
        harness = RealHostScenarioHarness(
            provider="openai",
            transfer_delay_ms=0.0,
            scheduler_config=SchedulerConfig(require_change_signal=False),
            scheduler_policy=object(),
            checkpoint_manager_factory=lambda base: base,
            max_workers=1,
        )
        harness.root = Path(tempfile.mkdtemp(prefix="crab_launch_test_"))
        harness.base_inspector = SimpleNamespace(upsert_snapshot=Mock())
        harness.system = SimpleNamespace(sandbox_manager=SimpleNamespace(launch=Mock()))
        harness.interceptor = SimpleNamespace(port=43123)
        harness.llm_server = SimpleNamespace(benchmark_llm_router=SimpleNamespace(register_sandbox=Mock()))
        sandbox_image = SimpleNamespace(
            exported_rootfs=harness.root / "rootfs",
            image_defaults=ImageRuntimeDefaults(),
        )
        sandbox_image.exported_rootfs.mkdir(parents=True, exist_ok=True)

        handle = SandboxHandle(
            sandbox_id=SandboxId("sbx-sim"),
            bundle_dir=harness.root / "bundle",
            status_port=8123,
            status_host="127.0.0.1",
            last_status={},
        )
        handle.bundle_dir.mkdir(parents=True, exist_ok=True)
        (handle.bundle_dir / "config.json").write_text(
            json.dumps(
                {
                    "linux": {"namespaces": [], "cgroupsPath": ""},
                    "mounts": [],
                    "process": {"terminal": False, "cwd": "/", "args": [], "env": []},
                    "root": {"path": "rootfs", "readonly": False},
                }
            ),
            encoding="utf-8",
        )
        task_run = SimpleNamespace(
            prepare_sandbox=Mock(),
            configure_bundle=Mock(),
            rootfs_init_dirs=Mock(return_value=[]),
            extra_launch_metadata=Mock(return_value={}),
            wait_for_task_ready=Mock(),
        )

        with patch.object(harness.network_manager, "allocate_lease") as allocate_network:
            with patch.object(harness, "_prepare_sandbox_handle", return_value=(handle, None)) as prepare_handle:
                with patch.object(harness, "build_task_run", return_value=task_run), patch.object(
                    harness,
                    "ensure_sandbox_image",
                    return_value=sandbox_image,
                ):
                    result = harness.launch_sandbox("sbx-sim", agent_type="simulated")

        self.assertIs(result, handle)
        allocate_network.assert_not_called()
        prepare_handle.assert_called_once_with(
            "sbx-sim",
            interceptor_host="127.0.0.1",
            network_lease=None,
            agent_type="simulated",
            llm_service_type="simulated",
            llm_service_config=None,
            image_defaults=ImageRuntimeDefaults(),
            image_rootfs_dir=sandbox_image.exported_rootfs,
        )

    def test_ensure_sandbox_image_serializes_parallel_calls(self) -> None:
        harness = RealHostScenarioHarness(
            provider="openai",
            transfer_delay_ms=0.0,
            scheduler_config=SchedulerConfig(require_change_signal=False),
            scheduler_policy=object(),
            checkpoint_manager_factory=lambda base: base,
            max_workers=2,
        )
        call_counts = {"build": 0, "export": 0}
        count_lock = threading.Lock()
        results: list[object] = []
        start_barrier = threading.Barrier(3)

        with tempfile.TemporaryDirectory(prefix="crab_image_lock_test_") as tmp:
            harness.root = Path(tmp)
            exported_rootfs = harness.root / "image" / "simulated" / "rootfs"
            exported_rootfs.mkdir(parents=True, exist_ok=True)

            def fake_build_image(*, tag: str, build_context: Path, dockerfile_path: Path, **kwargs) -> None:
                del tag, build_context, dockerfile_path
                del kwargs
                with count_lock:
                    call_counts["build"] += 1
                time.sleep(0.05)

            def fake_export_image_rootfs(*, tag: str, output_dir: Path, **kwargs) -> Path:
                del tag, output_dir
                del kwargs
                with count_lock:
                    call_counts["export"] += 1
                time.sleep(0.05)
                return exported_rootfs

            def worker() -> None:
                start_barrier.wait()
                results.append(harness.ensure_sandbox_image("simulated"))

            with patch("benchmarks.real_host_scenario_base.sandbox_image.build_image", side_effect=fake_build_image), patch(
                "benchmarks.real_host_scenario_base.sandbox_image.inspect_image_runtime_defaults",
                return_value=ImageRuntimeDefaults(),
            ), patch(
                "benchmarks.real_host_scenario_base.sandbox_image.export_image_rootfs",
                side_effect=fake_export_image_rootfs,
            ):
                threads = [threading.Thread(target=worker) for _ in range(2)]
                for thread in threads:
                    thread.start()
                start_barrier.wait()
                for thread in threads:
                    thread.join()

        self.assertEqual(call_counts, {"build": 1, "export": 1})
        self.assertEqual(len(results), 2)
        self.assertIs(results[0], results[1])

    def test_prepare_sandbox_handle_sets_llm_base_url(self) -> None:
        harness = RealHostScenarioHarness(
            provider="openai",
            transfer_delay_ms=0.0,
            scheduler_config=SchedulerConfig(require_change_signal=False),
            scheduler_policy=object(),
            checkpoint_manager_factory=lambda base: base,
            max_workers=1,
        )
        harness.root = Path(tempfile.mkdtemp(prefix="crab_handle_test_"))
        harness.interceptor = SimpleNamespace(port=43123)
        harness.sandbox_manager = SimpleNamespace(write_bundle_spec=Mock())
        register_sandbox = Mock()
        harness.llm_server = SimpleNamespace(
            server_address=("127.0.0.1", 45678),
            benchmark_llm_router=SimpleNamespace(register_sandbox=register_sandbox),
        )

        with patch(
            "integrations.sandboxes.runtime.launcher.prepare_bundle_launch",
            return_value=sandbox_launcher.PreparedBundleLaunch(
                bundle_dir=harness.root / "bundles" / "sbx-url",
                work_dir_host_path=None,
                status_host="127.0.0.1",
                status_port=8123,
                llm_base_url="http://10.250.0.1:43123/v1",
            ),
        ):
            handle, _ = harness._prepare_sandbox_handle(
                "sbx-url",
                interceptor_host="10.250.0.1",
                network_lease=None,
                agent_type="iflow",
                llm_service_type="simulated_for_iflow",
            )

        self.assertEqual(handle.llm_base_url, "http://10.250.0.1:43123/v1")
        register_sandbox.assert_called_once_with(
            sandbox_id="sbx-url",
            llm_service_type="simulated_for_iflow",
            llm_service_config=None,
        )

    def test_restore_llm_service_state_uses_max_restore_trace_cursor(self) -> None:
        harness = RealHostScenarioHarness(
            provider="openai",
            transfer_delay_ms=0.0,
            scheduler_config=SchedulerConfig(require_change_signal=False),
            scheduler_policy=object(),
            checkpoint_manager_factory=lambda base: base,
            max_workers=1,
        )
        restore_sandbox = Mock()
        harness.llm_server = SimpleNamespace(
            benchmark_llm_router=SimpleNamespace(restore_sandbox=restore_sandbox),
        )

        harness._restore_llm_service_state(
            SandboxId("sbx-replay"),
            CheckpointManifest(
                schema_version="v1",
                checkpoint_id=CheckpointId("ckpt-1"),
                sandbox_id=SandboxId("sbx-replay"),
                created_at=utc_now(),
                runtime_name="runc",
                runtime_version=None,
                process_artifacts=[],
                filesystem_artifacts=[],
                metadata={
                    "process_restore_trace_cursor": 2,
                    "benchmark_trace_cursor": 5,
                    "filesystem_restore_trace_cursor": 4,
                },
            ).with_integrity(),
        )

        restore_sandbox.assert_called_once_with("sbx-replay", consumed_response_count=4)

    def test_restore_llm_service_state_skips_restore_without_process_restore_trace_cursor(self) -> None:
        harness = RealHostScenarioHarness(
            provider="openai",
            transfer_delay_ms=0.0,
            scheduler_config=SchedulerConfig(require_change_signal=False),
            scheduler_policy=object(),
            checkpoint_manager_factory=lambda base: base,
            max_workers=1,
        )
        restore_sandbox = Mock()
        harness.llm_server = SimpleNamespace(
            benchmark_llm_router=SimpleNamespace(restore_sandbox=restore_sandbox),
        )

        harness._restore_llm_service_state(
            SandboxId("sbx-replay"),
            CheckpointManifest(
                schema_version="v1",
                checkpoint_id=CheckpointId("ckpt-1"),
                sandbox_id=SandboxId("sbx-replay"),
                created_at=utc_now(),
                runtime_name="runc",
                runtime_version=None,
                process_artifacts=[],
                filesystem_artifacts=[],
                metadata={
                    "benchmark_trace_cursor": 5,
                    "filesystem_restore_trace_cursor": 4,
                },
            ).with_integrity(),
        )

        restore_sandbox.assert_not_called()

    def test_launch_sandbox_uses_benchmark_network_for_iflow_agents(self) -> None:
        harness = RealHostScenarioHarness(
            provider="openai",
            transfer_delay_ms=0.0,
            scheduler_config=SchedulerConfig(require_change_signal=False),
            scheduler_policy=object(),
            checkpoint_manager_factory=lambda base: base,
            max_workers=1,
        )
        harness.root = Path(tempfile.mkdtemp(prefix="crab_launch_test_"))
        harness.base_inspector = SimpleNamespace(upsert_snapshot=Mock())
        harness.system = SimpleNamespace(sandbox_manager=SimpleNamespace(launch=Mock()))
        harness.interceptor = SimpleNamespace(port=43123)
        harness.llm_server = SimpleNamespace(benchmark_llm_router=SimpleNamespace(register_sandbox=Mock()))
        sandbox_image = SimpleNamespace(
            exported_rootfs=harness.root / "rootfs",
            image_defaults=ImageRuntimeDefaults(),
        )
        sandbox_image.exported_rootfs.mkdir(parents=True, exist_ok=True)

        lease = SimpleNamespace(guest_ip="10.250.0.2", namespace_path=Path("/var/run/netns/test-iflow"))
        handle = SandboxHandle(
            sandbox_id=SandboxId("sbx-iflow"),
            bundle_dir=harness.root / "bundle-iflow",
            status_port=8124,
            status_host=lease.guest_ip,
            last_status={},
        )
        task_run = SimpleNamespace(
            prepare_sandbox=Mock(),
            configure_bundle=Mock(),
            rootfs_init_dirs=Mock(return_value=[]),
            extra_launch_metadata=Mock(return_value={}),
            wait_for_task_ready=Mock(),
        )

        with patch.object(harness.network_manager, "allocate_lease", return_value=lease) as allocate_network:
            with patch.object(harness, "_prepare_sandbox_handle", return_value=(handle, None)) as prepare_handle:
                with patch.object(harness, "build_task_run", return_value=task_run), patch.object(
                    harness,
                    "ensure_sandbox_image",
                    return_value=sandbox_image,
                ):
                    result = harness.launch_sandbox("sbx-iflow", agent_type="iflow")

        self.assertIs(result, handle)
        allocate_network.assert_called_once_with(SandboxId("sbx-iflow"))
        prepare_handle.assert_called_once_with(
            "sbx-iflow",
            interceptor_host=harness.benchmark_bridge_ip,
            network_lease=lease,
            agent_type="iflow",
            llm_service_type="simulated_for_iflow",
            llm_service_config=None,
            image_defaults=ImageRuntimeDefaults(),
            image_rootfs_dir=sandbox_image.exported_rootfs,
        )
        self.assertEqual(harness.network_manager.resolve_sandbox_id(lease.guest_ip), SandboxId("sbx-iflow"))

    def test_launch_sandbox_from_docker_compose_file_rejects_unsupported_features(self) -> None:
        harness = RealHostScenarioHarness(
            provider="openai",
            transfer_delay_ms=0.0,
            scheduler_config=SchedulerConfig(require_change_signal=False),
            scheduler_policy=object(),
            checkpoint_manager_factory=lambda base: base,
            max_workers=1,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            harness.root = root
            compose_file = root / "compose.yaml"
            compose_file.write_text(
                """
services:
  app:
    image: alpine
    depends_on:
      - db
""".strip(),
                encoding="utf-8",
            )
            env_file = root / ".env"
            env_file.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unsupported compose features"):
                harness.launch_sandbox_from_docker_compose_file(
                    compose_file,
                    env_file,
                    sandbox_name="sbx-compose",
                    service_name="app",
                )

    def test_prepare_compose_task_record_preserves_translated_rootfs_copy_paths_when_agent_metadata_is_empty(self) -> None:
        harness = RealHostScenarioHarness(
            provider="openai",
            transfer_delay_ms=0.0,
            scheduler_config=SchedulerConfig(require_change_signal=False),
            scheduler_policy=object(),
            checkpoint_manager_factory=lambda base: base,
            max_workers=1,
            reuse_zpool=True,
        )
        harness.root = Path(tempfile.mkdtemp(prefix="crab_prepare_compose_test_"))
        harness.image_cache_root = harness.root / "images"
        harness.image_cache_root.mkdir(parents=True, exist_ok=True)
        harness._compose_image_tags = set()
        lease = SimpleNamespace(guest_ip="10.250.0.2", namespace_path=Path("/var/run/netns/test-claude"))
        harness.network_manager = SimpleNamespace(
            allocate_lease=Mock(return_value=lease),
            bridge_ip="10.250.0.1",
        )

        compose_file = harness.root / "compose.yaml"
        compose_file.write_text("services:\n  client:\n    image: ignored\n", encoding="utf-8")
        handle = SandboxHandle(
            sandbox_id=SandboxId("sbx-claude-compose"),
            bundle_dir=harness.root / "bundle-claude-compose",
            status_port=8123,
            status_host=lease.guest_ip,
            last_status={},
            agent_type="claude_code",
            llm_service_type="claude_code_trace_replay",
        )
        work_dir_host_path = harness.root / "workdir"
        translated_rootfs = harness.root / "compose-rootfs"
        translated_rootfs.mkdir(parents=True, exist_ok=True)
        translation = sandbox_compose.ComposeSandboxTranslation(
            runtime_launch_metadata={
                "sandbox_id": str(handle.sandbox_id),
                "bundle_path": str(handle.bundle_dir),
                "work_dir_host_path": str(work_dir_host_path),
                "rootfs_init_dirs": ["work", "tmp"],
                "rootfs_copy_paths": [{"source": str(translated_rootfs), "destination": "/"}],
            },
            compose_launch_metadata={"service_name": "client", "image_ref": "example/client:latest", "ports": []},
        )
        prelaunch_task_run = SimpleNamespace(
            prepare_sandbox=Mock(),
            configure_bundle=Mock(),
            rootfs_init_dirs=Mock(return_value=["opt/claude-code-home"]),
            extra_launch_metadata=Mock(return_value={"rootfs_copy_paths": []}),
        )
        task_record = BenchmarkTaskRecord(
            agent_type="claude_code",
            task_description=TaskDescription("solve the task"),
            task_config=TaskConfig(),
            task_id="task-claude-compose",
            llm_service_type="claude_code_trace_replay",
            docker_compose_file=compose_file,
            service_name="client",
            llm_service_config={"trace_path": "/tmp/trace.json"},
        )

        with (
            patch.object(harness, "_build_termnius_compose_env", return_value={}),
            patch("benchmarks.real_host_scenario_base.sandbox_compose.load_compose_service", return_value=("client", {})),
            patch("benchmarks.real_host_scenario_base.sandbox_compose.translate_compose_service", return_value=translation),
            patch.object(harness, "_prepare_sandbox_handle", return_value=(handle, work_dir_host_path)),
            patch.object(harness, "build_task_run", return_value=prelaunch_task_run),
            patch.object(harness, "_configure_termnius_bundle_privileges"),
        ):
            prepared = harness._prepare_compose_task_record(
                sandbox_name="sbx-claude-compose",
                task_record=task_record,
            )

        runtime_metadata = prepared.handle.launch_metadata["runtime"]
        self.assertIn(
            {"source": str(translated_rootfs), "destination": "/"},
            runtime_metadata["rootfs_copy_paths"],
        )
        self.assertTrue(runtime_metadata["shared_rootfs_key"])

    def test_resolve_compose_image_ref_uses_explicit_image_name_for_build_service(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            compose_file = root / "compose.yaml"
            build_dir = root / "context"
            build_dir.mkdir(parents=True, exist_ok=True)
            with patch("integrations.sandboxes.runtime.image.subprocess.run") as run_build:
                run_build.return_value = SimpleNamespace(returncode=1, stdout="", stderr="")
                image_ref = sandbox_compose.resolve_compose_image_ref(
                    compose_file=compose_file,
                    service_name="client",
                    service={"build": {"context": str(build_dir)}, "image": "example/client:latest"},
                    compose_image_tags=set(),
                )

        self.assertEqual(image_ref, "example/client:latest")
        self.assertEqual(run_build.call_args.args[0][:4], ["docker", "build", "-t", "example/client:latest"])

    def test_iflow_agent_completes_from_existing_markers_without_live_sandbox(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_dir = Path(tmp)
            sandbox = SandboxHandle(
                sandbox_id=SandboxId("sbx-iflow"),
                bundle_dir=Path("/tmp/sbx-iflow"),
                status_port=8123,
                last_status={},
                launch_metadata={
                    "iflow": {
                        "entrypoint": "/opt/iflow-runtime/global/lib/node_modules/@iflow-ai/iflow-cli/bundle/entry.js",
                        "logs_dir": str(logs_dir),
                    }
                },
            )
            agent = IFlowAgent(
                sandbox,
                TaskDescription("do work"),
                TaskConfig(),
                runtime_state_root=Path("/tmp/runtime"),
            )
            (logs_dir / "iflow.task.done").write_text("done\n", encoding="utf-8")
            (logs_dir / "iflow.task.exit").write_text("0\n", encoding="utf-8")

            with patch.object(agent, "_sandbox_is_live", return_value=False) as sandbox_is_live:
                agent.perform_task()

        sandbox_is_live.assert_not_called()

    def test_iflow_agent_calls_post_task_finish_after_completion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_dir = Path(tmp)
            sandbox = SandboxHandle(
                sandbox_id=SandboxId("sbx-iflow-post-finish"),
                bundle_dir=Path("/tmp/sbx-iflow-post-finish"),
                status_port=8123,
                last_status={},
                launch_metadata={
                    "iflow": {
                        "entrypoint": "/opt/iflow-runtime/global/lib/node_modules/@iflow-ai/iflow-cli/bundle/entry.js",
                        "logs_dir": str(logs_dir),
                    }
                },
            )
            agent = IFlowAgent(
                sandbox,
                TaskDescription("do work"),
                TaskConfig(),
                runtime_state_root=Path("/tmp/runtime"),
            )
            (logs_dir / "iflow.task.done").write_text("done\n", encoding="utf-8")
            (logs_dir / "iflow.task.exit").write_text("0\n", encoding="utf-8")

            with patch.object(agent, "post_task_finish") as post_task_finish, patch.object(
                agent, "_sandbox_is_live", return_value=False
            ):
                agent.perform_task()

        post_task_finish.assert_called_once_with()

    def test_iflow_configure_bundle_runs_task_at_init(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle_dir = Path(tmp)
            config_path = bundle_dir / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "linux": {"namespaces": [], "cgroupsPath": ""},
                        "mounts": [],
                        "process": {"terminal": False, "cwd": "/", "args": [], "env": ["IMAGE_ONLY=1", "PYTHONUNBUFFERED=0"]},
                        "root": {"path": "rootfs", "readonly": False},
                    }
                ),
                encoding="utf-8",
            )
            sandbox = SandboxHandle(
                sandbox_id=SandboxId("sbx-iflow-config"),
                bundle_dir=bundle_dir,
                status_port=8123,
                last_status={},
                launch_metadata={
                    "iflow": {
                        "runtime_root": "/tmp/runtime-root",
                        "iflow_home": "/tmp/iflow-home",
                        "npm_home": "/tmp/npm-home",
                        "logs_dir": "/tmp/logs",
                        "entrypoint": "/opt/iflow-runtime/global/lib/node_modules/@iflow-ai/iflow-cli/bundle/entry.js",
                    }
                },
            )
            agent = IFlowAgent(sandbox, TaskDescription("do work"), TaskConfig(options={"FOO": "bar"}))

            agent.configure_bundle()

            payload = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["process"]["args"][:2], ["/bin/sh", "-lc"])
            self.assertIn("/opt/iflow-runtime/global/lib/node_modules/@iflow-ai/iflow-cli/bundle/entry.js", payload["process"]["args"][2])
            self.assertIn("/opt/iflow-logs/iflow.task.exit", payload["process"]["args"][2])
            self.assertIn("/opt/iflow-logs/iflow.task.done", payload["process"]["args"][2])
            self.assertNotIn("/data/iflow-task-logs", payload["process"]["args"][2])
            self.assertNotIn("rm -f /dev/null", payload["process"]["args"][2])
            self.assertIn(">/dev/null 2>/dev/null", payload["process"]["args"][2])
            self.assertNotIn("exec >/dev/null 2>&1", payload["process"]["args"][2])
            self.assertIn("IMAGE_ONLY=1", payload["process"]["env"])
            self.assertIn("FOO=bar", payload["process"]["env"])

    def test_iflow_configure_bundle_sets_compose_replay_wrapper_launch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle_dir = Path(tmp)
            config_path = bundle_dir / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "linux": {"namespaces": [], "cgroupsPath": ""},
                        "mounts": [],
                        "process": {
                            "terminal": False,
                            "cwd": "/app",
                            "args": ["sleep", "infinity"],
                            "env": ["PATH=/usr/bin"],
                        },
                        "root": {"path": "rootfs", "readonly": False},
                    }
                ),
                encoding="utf-8",
            )
            sandbox = SandboxHandle(
                sandbox_id=SandboxId("sbx-iflow-compose-replay"),
                bundle_dir=bundle_dir,
                status_port=8123,
                last_status={},
                llm_service_type="iflow_trace_replay",
                launch_source="compose",
                launch_metadata={
                    "iflow": {
                        "runtime_root": "/tmp/runtime-root",
                        "iflow_home": "/tmp/iflow-home",
                        "npm_home": "/tmp/npm-home",
                        "logs_dir": "/tmp/logs",
                        "entrypoint": "/opt/iflow-runtime/global/lib/node_modules/@iflow-ai/iflow-cli/bundle/entry.js",
                    }
                },
            )
            agent = IFlowAgent(sandbox, TaskDescription("do work"), TaskConfig(options={"FOO": "bar"}))

            agent.configure_bundle()

            payload = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["process"]["cwd"], "/app")
            self.assertEqual(payload["process"]["args"][:2], ["/bin/sh", "-lc"])
            self.assertIn("export CRAB_IFLOW_ENTRYPOINT=/opt/iflow-runtime/global/lib/node_modules/@iflow-ai/iflow-cli/bundle/entry.js", payload["process"]["args"][2])
            self.assertIn("export CRAB_IFLOW_CWD=/app", payload["process"]["args"][2])
            self.assertIn("export CRAB_IFLOW_KEEPALIVE_AFTER_TASK=true", payload["process"]["args"][2])
            self.assertIn("cd /app", payload["process"]["args"][2])
            self.assertIn("if [ -f /installed-agent/setup-env.sh ]; then . /installed-agent/setup-env.sh; fi", payload["process"]["args"][2])
            self.assertIn(
                "elif command -v apt-get >/dev/null 2>&1; then export DEBIAN_FRONTEND=noninteractive; apt-get update && apt-get install -y curl git wget xz-utils openssh-client patch xauth build-essential dpkg-dev procps net-tools psmisc; fi",
                payload["process"]["args"][2],
            )
            self.assertIn("exec /opt/iflow-runtime/node/bin/node -e", payload["process"]["args"][2])
            self.assertIn(IFLOW_WRAPPER_ARG, payload["process"]["args"][2])
            self.assertNotIn("/data/iflow-task-logs", payload["process"]["args"][2])
            self.assertNotIn("rm -f /dev/null", payload["process"]["args"][2])
            self.assertIn(">/dev/null 2>/dev/null", payload["process"]["args"][2])
            self.assertIn("FOO=bar", payload["process"]["env"])
            mounted_destinations = {mount["destination"] for mount in payload["mounts"]}
            self.assertIn("/opt/iflow-runtime", mounted_destinations)
            self.assertIn("/root/.iflow", mounted_destinations)
            self.assertIn("/root/.npm", mounted_destinations)
            self.assertIn("/opt/iflow-logs", mounted_destinations)

    def test_iflow_prepare_sandbox_uses_extended_benchmark_session_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sandbox = SandboxHandle(
                sandbox_id=SandboxId("sbx-iflow-prepare"),
                bundle_dir=Path(tmp) / "bundle",
                status_port=8123,
                last_status={},
            )
            runtime = SimpleNamespace(
                root=Path(tmp) / "runtime",
                mounted_entrypoint="/opt/iflow-runtime/entry.js",
                ignore_process_rules=[],
            )
            state = SimpleNamespace(
                iflow_home=Path(tmp) / ".iflow",
                npm_home=Path(tmp) / ".npm",
                logs_dir=Path(tmp) / "logs",
            )
            agent = IFlowAgent(
                sandbox,
                TaskDescription("do work"),
                TaskConfig(),
                agent_host_dir=Path(tmp) / "iflow" / "sbx-iflow-prepare",
                llm_base_url="http://10.250.9.1:4567/v1",
            )

            with patch("integrations.agents.iflow.prepare_iflow_runtime", return_value=runtime), patch(
                "integrations.agents.iflow.prepare_iflow_state",
                return_value=state,
            ) as prepare_state:
                agent.prepare_sandbox()

        self.assertEqual(prepare_state.call_args.kwargs["max_session_turns"], 4096)
        self.assertEqual(prepare_state.call_args.kwargs["base_url"], "http://10.250.9.1:4567/v1")

    def test_iflow_agent_uses_replay_router_state_for_progress(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle_dir = Path(tmp)
            (bundle_dir / "config.json").write_text(
                json.dumps(
                    {
                        "linux": {"namespaces": [], "cgroupsPath": ""},
                        "mounts": [],
                        "process": {"terminal": False, "cwd": "/app", "args": ["sleep", "infinity"], "env": ["PATH=/usr/bin"]},
                        "root": {"path": "rootfs", "readonly": False},
                    }
                ),
                encoding="utf-8",
            )
            sandbox = SandboxHandle(
                sandbox_id=SandboxId("sbx-iflow-replay-progress"),
                bundle_dir=bundle_dir,
                status_port=8123,
                last_status={},
                llm_service_type="iflow_trace_replay",
                llm_control_base_url="http://127.0.0.1:12345",
                launch_source="compose",
                launch_metadata={
                    "iflow": {
                        "runtime_root": "/tmp/runtime-root",
                        "iflow_home": "/tmp/iflow-home",
                        "npm_home": "/tmp/npm-home",
                        "logs_dir": "/tmp/logs",
                        "entrypoint": "/opt/iflow-runtime/entry.js",
                    }
                },
            )
            agent = IFlowAgent(sandbox, TaskDescription("do work"), TaskConfig())
            agent._started_at_monotonic = time.monotonic()

            state_payloads = [
                {"state": {"state": {"trace_cursor": 1, "total_responses": 5, "is_complete": False}}},
                {"state": {"state": {"trace_cursor": 2, "total_responses": 5, "is_complete": False}}},
                {"state": {"state": {"trace_cursor": 2, "total_responses": 5, "is_complete": False}}},
                {"state": {"state": {"trace_cursor": 2, "total_responses": 5, "is_complete": False}}},
                {"state": {"state": {"trace_cursor": 3, "total_responses": 5, "is_complete": False}}},
                {"state": {"state": {"trace_cursor": 3, "total_responses": 5, "is_complete": False}}},
            ]

            with patch.object(agent, "wait_for_http_json", side_effect=state_payloads):
                payload = agent.wait_for_progress(minimum_actions=2)
                delta_payload = agent.wait_for_action_delta(delta=1)

        self.assertEqual(int(payload["total_actions"]), 2)
        self.assertEqual(int(delta_payload["total_actions"]), 3)
        self.assertEqual(sandbox.last_status, delta_payload)
        self.assertEqual(len(agent._recorded_activity_events()), 6)

    def test_iflow_replay_restore_clears_host_markers_before_resuming(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_dir = Path(tmp) / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            (logs_dir / "iflow.task.done").write_text("done\n", encoding="utf-8")
            (logs_dir / "iflow.task.exit").write_text("0\n", encoding="utf-8")
            sandbox = SandboxHandle(
                sandbox_id=SandboxId("sbx-iflow-replay-restore"),
                bundle_dir=Path(tmp) / "bundle",
                status_port=8123,
                last_status={},
                llm_service_type="iflow_trace_replay",
                launch_source="compose",
                launch_metadata={
                    "iflow": {
                        "runtime_root": "/tmp/runtime-root",
                        "iflow_home": "/tmp/iflow-home",
                        "npm_home": "/tmp/npm-home",
                        "logs_dir": str(logs_dir),
                        "entrypoint": "/opt/iflow-runtime/entry.js",
                    }
                },
            )
            agent = IFlowAgent(sandbox, TaskDescription("do work"), TaskConfig())

            agent.on_restore_complete()

        self.assertFalse((logs_dir / "iflow.task.done").exists())
        self.assertFalse((logs_dir / "iflow.task.exit").exists())

    def test_iflow_perform_task_in_compose_replay_mode_only_waits_on_markers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle_dir = Path(tmp)
            (bundle_dir / "config.json").write_text(
                json.dumps(
                    {
                        "linux": {"namespaces": [], "cgroupsPath": ""},
                        "mounts": [],
                        "process": {
                            "terminal": False,
                            "cwd": "/app",
                            "args": ["sleep", "infinity"],
                            "env": ["PATH=/usr/bin", "HOME=/root"],
                            "user": {"uid": 0, "gid": 0},
                        },
                        "root": {"path": "rootfs", "readonly": False},
                    }
                ),
                encoding="utf-8",
            )
            logs_dir = Path(tmp) / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            sandbox = SandboxHandle(
                sandbox_id=SandboxId("sbx-iflow-replay-exec"),
                bundle_dir=bundle_dir,
                status_port=8123,
                last_status={},
                llm_service_type="iflow_trace_replay",
                launch_source="compose",
                launch_metadata={
                    "iflow": {
                        "runtime_root": "/tmp/runtime-root",
                        "iflow_home": "/tmp/iflow-home",
                        "npm_home": "/tmp/npm-home",
                        "logs_dir": str(logs_dir),
                        "entrypoint": "/opt/iflow-runtime/entry.js",
                    }
                },
            )
            agent = IFlowAgent(
                sandbox,
                TaskDescription("do work"),
                TaskConfig(),
                runtime_state_root=Path("/tmp/runtime"),
            )
            marker_paths = agent._task_marker_paths(logs_dir)
            marker_paths["done"].write_text("done\n", encoding="utf-8")
            marker_paths["exit"].write_text("0\n", encoding="utf-8")
            completion_spy = Mock(wraps=agent._wait_for_task_completion)

            with patch.object(agent, "_sandbox_is_live", return_value=True), patch.object(
                agent, "_wait_for_task_completion", completion_spy
            ):
                agent.perform_task()

        completion_spy.assert_called_once()

    def test_iflow_agent_exposes_synthetic_progress_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_dir = Path(tmp)
            sandbox = SandboxHandle(
                sandbox_id=SandboxId("sbx-iflow-progress"),
                bundle_dir=Path("/tmp/sbx-iflow-progress"),
                status_port=8123,
                last_status={},
                launch_metadata={
                    "iflow": {
                        "entrypoint": "/opt/iflow-runtime/global/lib/node_modules/@iflow-ai/iflow-cli/bundle/entry.js",
                        "logs_dir": str(logs_dir),
                    }
                },
            )
            agent = IFlowAgent(
                sandbox,
                TaskDescription("do work"),
                TaskConfig(options={"action_tick_seconds": 0.01}),
                runtime_state_root=Path("/tmp/runtime"),
            )
            allow_finish = threading.Event()

            def _finish_later() -> None:
                allow_finish.wait(timeout=1.0)
                (logs_dir / "iflow.task.done").write_text("done\n", encoding="utf-8")
                (logs_dir / "iflow.task.exit").write_text("0\n", encoding="utf-8")

            threading.Thread(target=_finish_later, daemon=True).start()

            with patch.object(agent, "_sandbox_is_live", return_value=True):
                worker = threading.Thread(target=agent.perform_task)
                worker.start()
                payload = agent.wait_for_progress(minimum_actions=2)
                self.assertEqual(agent.poll_status()["state"], "running")
                delta_payload = agent.wait_for_action_delta(delta=1)
                allow_finish.set()
                worker.join(timeout=1.0)

        self.assertFalse(worker.is_alive())
        self.assertGreaterEqual(int(payload["total_actions"]), 2)
        self.assertGreaterEqual(int(delta_payload["total_actions"]), int(payload["total_actions"]) + 1)
        self.assertEqual(sandbox.last_status, delta_payload)
        final_payload = agent.poll_status()
        self.assertEqual(final_payload["state"], "finished")
        self.assertGreaterEqual(int(final_payload["total_actions"]), int(delta_payload["total_actions"]))
        self.assertEqual(sandbox.last_status, delta_payload)
        self.assertEqual(len(agent._recorded_activity_events()), 6)
        self.assertEqual(
            {event.kind for event in agent._recorded_activity_events()},
            {EBPFEventKind.FILE_WRITE, EBPFEventKind.PROCESS_EXEC, EBPFEventKind.NETWORK_EGRESS},
        )

    def test_iflow_agent_waits_across_fault_and_relaunch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_dir = Path(tmp)
            sandbox = SandboxHandle(
                sandbox_id=SandboxId("sbx-iflow-fault"),
                bundle_dir=Path("/tmp/sbx-iflow-fault"),
                status_port=8123,
                last_status={},
                launch_metadata={
                    "iflow": {
                        "entrypoint": "/opt/iflow-runtime/global/lib/node_modules/@iflow-ai/iflow-cli/bundle/entry.js",
                        "logs_dir": str(logs_dir),
                    }
                },
            )
            agent = IFlowAgent(
                sandbox,
                TaskDescription("do work"),
                TaskConfig(),
                runtime_state_root=Path("/tmp/runtime"),
            )
            sandbox_live = {"value": True}
            allow_finish = threading.Event()
            errors: list[BaseException] = []

            def _run_task() -> None:
                try:
                    agent.perform_task()
                except BaseException as exc:  # pragma: no cover - exercised via assertion below
                    errors.append(exc)

            def _finish_after_restore() -> None:
                allow_finish.wait(timeout=1.0)
                (logs_dir / "iflow.task.done").write_text("done\n", encoding="utf-8")
                (logs_dir / "iflow.task.exit").write_text("0\n", encoding="utf-8")

            threading.Thread(target=_finish_after_restore, daemon=True).start()

            with patch.object(agent, "_sandbox_is_live", side_effect=lambda: sandbox_live["value"]):
                worker = threading.Thread(target=_run_task)
                worker.start()
                time.sleep(0.05)
                self.assertTrue(worker.is_alive())
                sandbox_live["value"] = False
                time.sleep(0.05)
                sandbox.last_status = {"total_actions": 3}
                sandbox_live["value"] = True
                agent.on_restore_complete()
                allow_finish.set()
                worker.join(timeout=1.0)

        self.assertFalse(worker.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(agent.poll_status()["state"], "finished")

    def test_iflow_agent_fails_when_sandbox_is_dead_before_markers_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_dir = Path(tmp)
            sandbox = SandboxHandle(
                sandbox_id=SandboxId("sbx-iflow-dead"),
                bundle_dir=Path("/tmp/sbx-iflow-dead"),
                status_port=8123,
                last_status={},
                launch_metadata={
                    "iflow": {
                        "entrypoint": "/opt/iflow-runtime/global/lib/node_modules/@iflow-ai/iflow-cli/bundle/entry.js",
                        "logs_dir": str(logs_dir),
                    }
                },
            )
            agent = IFlowAgent(
                sandbox,
                TaskDescription("do work"),
                TaskConfig(),
                runtime_state_root=Path("/tmp/runtime"),
            )

            with patch.object(agent, "_sandbox_is_live", return_value=False):
                with self.assertRaisesRegex(RuntimeError, "stopped before writing task completion markers"):
                    agent.perform_task()

    def test_iflow_agent_raises_for_non_zero_exit_marker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_dir = Path(tmp)
            sandbox = SandboxHandle(
                sandbox_id=SandboxId("sbx-iflow-fail"),
                bundle_dir=Path("/tmp/sbx-iflow-fail"),
                status_port=8123,
                last_status={},
                launch_metadata={
                    "iflow": {
                        "entrypoint": "/opt/iflow-runtime/global/lib/node_modules/@iflow-ai/iflow-cli/bundle/entry.js",
                        "logs_dir": str(logs_dir),
                    }
                },
            )
            agent = IFlowAgent(
                sandbox,
                TaskDescription("do work"),
                TaskConfig(),
                runtime_state_root=Path("/tmp/runtime"),
            )

            (logs_dir / "iflow.task.exit").write_text("7\n", encoding="utf-8")
            with patch.object(agent, "_sandbox_is_live", return_value=True):
                with self.assertRaisesRegex(RuntimeError, "exit code 7"):
                    agent.perform_task()

    def test_iflow_agent_treats_zero_exit_without_done_marker_as_completion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_dir = Path(tmp)
            sandbox = SandboxHandle(
                sandbox_id=SandboxId("sbx-iflow-zero-exit"),
                bundle_dir=Path("/tmp/sbx-iflow-zero-exit"),
                status_port=8123,
                last_status={},
                launch_metadata={
                    "iflow": {
                        "entrypoint": "/opt/iflow-runtime/global/lib/node_modules/@iflow-ai/iflow-cli/bundle/entry.js",
                        "logs_dir": str(logs_dir),
                    }
                },
            )
            agent = IFlowAgent(
                sandbox,
                TaskDescription("do work"),
                TaskConfig(),
                runtime_state_root=Path("/tmp/runtime"),
            )

            (logs_dir / "iflow.task.exit").write_text("0\n", encoding="utf-8")
            with patch.object(agent, "_sandbox_is_live", return_value=True):
                agent.perform_task()

        self.assertEqual(agent.poll_status()["state"], "finished")

    def test_iflow_agent_uses_one_second_default_action_tick(self) -> None:
        sandbox = SandboxHandle(
            sandbox_id=SandboxId("sbx-iflow-default-tick"),
            bundle_dir=Path("/tmp/sbx-iflow-default-tick"),
            status_port=8123,
            last_status={},
            launch_metadata={
                "iflow": {
                    "entrypoint": "/opt/iflow-runtime/global/lib/node_modules/@iflow-ai/iflow-cli/bundle/entry.js",
                }
            },
        )
        agent = IFlowAgent(sandbox, TaskDescription("do work"), TaskConfig(), runtime_state_root=Path("/tmp/runtime"))
        self.assertEqual(agent._tick_seconds, 1.0)

    def test_iflow_replay_action_wait_timeout_uses_task_budget(self) -> None:
        sandbox = SandboxHandle(
            sandbox_id=SandboxId("sbx-iflow-replay-timeout"),
            bundle_dir=Path("/tmp/sbx-iflow-replay-timeout"),
            status_port=8123,
            last_status={},
            llm_service_type="iflow_trace_replay",
            launch_source="compose",
            launch_metadata={
                "iflow": {
                    "entrypoint": "/opt/iflow-runtime/global/lib/node_modules/@iflow-ai/iflow-cli/bundle/entry.js",
                }
            },
        )
        agent = IFlowAgent(
            sandbox,
            TaskDescription("do work"),
            TaskConfig(options={"max_agent_timeout_sec": 360.0}),
            runtime_state_root=Path("/tmp/runtime"),
        )

        self.assertEqual(agent._replay_action_wait_timeout_seconds(1), 360.0)
        self.assertEqual(agent._replay_action_wait_timeout_seconds(50), 500.0)

    def test_iflow_agent_on_restore_complete_resumes_synthetic_progress(self) -> None:
        sandbox = SandboxHandle(
            sandbox_id=SandboxId("sbx-iflow-restore"),
            bundle_dir=Path("/tmp/sbx-iflow-restore"),
            status_port=8123,
            last_status={"total_actions": 4},
            launch_metadata={
                "iflow": {
                    "entrypoint": "/opt/iflow-runtime/global/lib/node_modules/@iflow-ai/iflow-cli/bundle/entry.js",
                }
            },
        )
        agent = IFlowAgent(sandbox, TaskDescription("do work"), TaskConfig(), runtime_state_root=Path("/tmp/runtime"))
        agent._started_at_monotonic = time.monotonic() - 20.0
        agent._finished_at_monotonic = None

        baseline = agent._synthetic_action_count()
        agent.on_restore_complete()
        resumed = agent._synthetic_action_count()

        self.assertEqual(agent._task_state(), "running")
        self.assertGreaterEqual(baseline, 18)
        self.assertGreaterEqual(resumed, 4)
        self.assertLess(resumed, baseline)
        self.assertTrue(agent._restore_complete_event.is_set())

    def test_iflow_agent_request_stop_unblocks_restore_wait(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_dir = Path(tmp)
            sandbox = SandboxHandle(
                sandbox_id=SandboxId("sbx-iflow-stop"),
                bundle_dir=Path("/tmp/sbx-iflow-stop"),
                status_port=8123,
                last_status={},
                launch_metadata={
                    "iflow": {
                        "entrypoint": "/opt/iflow-runtime/global/lib/node_modules/@iflow-ai/iflow-cli/bundle/entry.js",
                        "logs_dir": str(logs_dir),
                    }
                },
            )
            agent = IFlowAgent(
                sandbox,
                TaskDescription("do work"),
                TaskConfig(),
                runtime_state_root=Path("/tmp/runtime"),
            )
            sandbox_live = {"value": True}
            errors: list[BaseException] = []

            def _run_task() -> None:
                try:
                    agent.perform_task()
                except BaseException as exc:  # pragma: no cover - exercised via assertion below
                    errors.append(exc)

            def _drop_sandbox() -> None:
                time.sleep(0.05)
                sandbox_live["value"] = False

            with patch.object(agent, "_sandbox_is_live", side_effect=lambda: sandbox_live["value"]):
                worker = threading.Thread(target=_run_task)
                worker.start()
                threading.Thread(target=_drop_sandbox, daemon=True).start()
                time.sleep(0.1)
                agent.request_stop()
                worker.join(timeout=1.0)

        self.assertFalse(worker.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(agent.poll_status()["state"], "finished")

    def test_iflow_restore_wait_keeps_waiting_until_sandbox_is_live(self) -> None:
        sandbox = SandboxHandle(
            sandbox_id=SandboxId("sbx-iflow-restore-live"),
            bundle_dir=Path("/tmp/sbx-iflow-restore-live"),
            status_port=8123,
            last_status={},
            launch_metadata={
                "iflow": {
                    "entrypoint": "/opt/iflow-runtime/global/lib/node_modules/@iflow-ai/iflow-cli/bundle/entry.js",
                }
            },
        )
        agent = IFlowAgent(sandbox, TaskDescription("do work"), TaskConfig(), runtime_state_root=Path("/tmp/runtime"))
        live_states = iter([False, False, True])

        agent.on_restore_complete()

        with patch.object(agent, "_sandbox_is_live", side_effect=lambda: next(live_states)):
            self.assertTrue(agent._wait_for_restore_or_stop())

        self.assertFalse(agent._restore_reactivation_pending.is_set())

    def test_iflow_restore_wait_uses_pending_reactivation_without_new_signal(self) -> None:
        sandbox = SandboxHandle(
            sandbox_id=SandboxId("sbx-iflow-restore-pending"),
            bundle_dir=Path("/tmp/sbx-iflow-restore-pending"),
            status_port=8123,
            last_status={},
            launch_metadata={
                "iflow": {
                    "entrypoint": "/opt/iflow-runtime/global/lib/node_modules/@iflow-ai/iflow-cli/bundle/entry.js",
                }
            },
        )
        agent = IFlowAgent(sandbox, TaskDescription("do work"), TaskConfig(), runtime_state_root=Path("/tmp/runtime"))
        agent._restore_reactivation_pending.set()

        with patch.object(agent, "_sandbox_is_live", side_effect=[False, True]):
            self.assertTrue(agent._wait_for_restore_or_stop())

        self.assertFalse(agent._restore_reactivation_pending.is_set())

    def test_iflow_agent_rate_limits_runtime_liveness_checks_while_waiting_on_markers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_dir = Path(tmp)
            sandbox = SandboxHandle(
                sandbox_id=SandboxId("sbx-iflow-rate-limit"),
                bundle_dir=Path("/tmp/sbx-iflow-rate-limit"),
                status_port=8123,
                last_status={},
                launch_metadata={
                    "iflow": {
                        "entrypoint": "/opt/iflow-runtime/global/lib/node_modules/@iflow-ai/iflow-cli/bundle/entry.js",
                        "logs_dir": str(logs_dir),
                    }
                },
            )
            agent = IFlowAgent(
                sandbox,
                TaskDescription("do work"),
                TaskConfig(),
                runtime_state_root=Path("/tmp/runtime"),
            )

            def _finish_task() -> None:
                time.sleep(0.25)
                (logs_dir / "iflow.task.done").write_text("done\n", encoding="utf-8")
                (logs_dir / "iflow.task.exit").write_text("0\n", encoding="utf-8")

            worker = threading.Thread(target=_finish_task, daemon=True)
            worker.start()

            with patch.object(IFlowAgent, "SANDBOX_LIVENESS_REFRESH_SECONDS", 10.0), patch.object(
                agent, "_sandbox_is_live", return_value=True
            ) as sandbox_is_live:
                agent.perform_task()

        self.assertEqual(sandbox_is_live.call_count, 1)

    def test_iflow_poll_status_is_non_mutating_until_status_helpers_run(self) -> None:
        sandbox = SandboxHandle(
            sandbox_id=SandboxId("sbx-iflow-poll"),
            bundle_dir=Path("/tmp/sbx-iflow-poll"),
            status_port=8123,
            last_status={"total_actions": 1},
            launch_metadata={
                "iflow": {
                    "entrypoint": "/opt/iflow-runtime/global/lib/node_modules/@iflow-ai/iflow-cli/bundle/entry.js",
                }
            },
        )
        agent = IFlowAgent(sandbox, TaskDescription("do work"), TaskConfig())
        agent._started_at_monotonic = time.monotonic() - 0.05

        polled = agent.poll_status()

        self.assertGreaterEqual(int(polled["total_actions"]), 0)
        self.assertEqual(sandbox.last_status, {"total_actions": 1})
        agent._set_status(polled)
        self.assertEqual(sandbox.last_status, polled)

    def test_clone_checkpoint_to_fork_preserves_llm_base_url(self) -> None:
        harness = RealHostScenarioHarness(
            provider="openai",
            transfer_delay_ms=0.0,
            scheduler_config=SchedulerConfig(require_change_signal=False),
            scheduler_policy=object(),
            checkpoint_manager_factory=lambda base: base,
            max_workers=1,
        )
        source = SandboxHandle(
            sandbox_id=SandboxId("sbx-source"),
            bundle_dir=Path("/tmp/sbx-source"),
            status_port=8123,
            last_status={},
            agent_type="iflow",
            llm_service_type="simulated_for_iflow",
            llm_base_url="http://10.250.0.1:43123/v1",
            task_description=TaskDescription("resume"),
            task_config=TaskConfig(),
        )
        target = SandboxHandle(
            sandbox_id=SandboxId("sbx-fork"),
            bundle_dir=Path("/tmp/sbx-fork"),
            status_port=8124,
            last_status={},
        )
        harness.root = Path("/tmp")
        harness.runtime = object()
        harness.sandbox_manager = SimpleNamespace(
            _items={},
            _persist=Mock(),
            dataset_name_for=lambda sandbox_id: f"pool/crab/{sandbox_id}",
            clone_filesystem_snapshot=Mock(return_value="pool/crab/sbx-fork"),
        )
        harness.base_inspector = SimpleNamespace(upsert_snapshot=Mock())
        harness.storage = SimpleNamespace(
            put_manifest=Mock(),
            get_artifact=Mock(),
            put_artifact=Mock(),
        )

        manifest = CheckpointManifest(
            schema_version="v1",
            checkpoint_id=CheckpointId("ckpt-1"),
            sandbox_id=source.sandbox_id,
            created_at=utc_now(),
            runtime_name="runc",
            runtime_version=None,
            process_artifacts=[],
            filesystem_artifacts=[],
            metadata={},
        ).with_integrity()

        mock_task_run = Mock()
        mock_task_run.extra_launch_metadata.return_value = {}
        with patch.object(harness, "_agent_requires_benchmark_network", return_value=False), patch.object(
            harness, "_prepare_sandbox_handle", return_value=(target, None)
        ), patch.object(harness, "list_checkpoint_manifests", return_value=[manifest]), patch.object(
            harness, "build_task_run", return_value=mock_task_run
        ), patch("benchmarks.support.resolve_checkpoint_copy_plan", return_value=[(CheckpointId("ckpt-1"), False, True)]), patch(
            "benchmarks.real_host_scenario_base.subprocess.run"
        ):
            forked = harness.clone_checkpoint_to_fork(source, CheckpointId("ckpt-1"), "sbx-fork")

        self.assertIs(forked, target)
        self.assertEqual(target.llm_base_url, "http://10.250.0.1:43123/v1")
        mock_task_run.prepare_sandbox.assert_called_once_with()

    def test_clone_checkpoint_to_fork_persists_extra_launch_metadata(self) -> None:
        harness = RealHostScenarioHarness(
            provider="openai",
            transfer_delay_ms=0.0,
            scheduler_config=SchedulerConfig(require_change_signal=False),
            scheduler_policy=object(),
            checkpoint_manager_factory=lambda base: base,
            max_workers=1,
        )
        source = SandboxHandle(
            sandbox_id=SandboxId("sbx-source"),
            bundle_dir=Path("/tmp/sbx-source"),
            status_port=8123,
            last_status={},
            agent_type="iflow",
            llm_service_type="simulated_for_iflow",
            llm_base_url="http://10.250.0.1:43123/v1",
            task_description=TaskDescription("resume"),
            task_config=TaskConfig(),
        )
        target = SandboxHandle(
            sandbox_id=SandboxId("sbx-fork"),
            bundle_dir=Path("/tmp/sbx-fork"),
            status_port=8124,
            last_status={},
        )
        harness.root = Path("/tmp")
        persisted_descriptions: list[object] = []
        harness.runtime = object()
        clone_snapshot = Mock(return_value="pool/crab/sbx-fork")
        harness.sandbox_manager = SimpleNamespace(
            _items={},
            _persist=lambda description: persisted_descriptions.append(description),
            dataset_name_for=lambda sandbox_id: f"pool/crab/{sandbox_id}",
            clone_filesystem_snapshot=clone_snapshot,
        )
        harness.base_inspector = SimpleNamespace(upsert_snapshot=Mock())
        harness.storage = SimpleNamespace(
            put_manifest=Mock(),
            get_artifact=Mock(),
            put_artifact=Mock(),
        )

        manifest = CheckpointManifest(
            schema_version="v1",
            checkpoint_id=CheckpointId("ckpt-1"),
            sandbox_id=source.sandbox_id,
            created_at=utc_now(),
            runtime_name="runc",
            runtime_version=None,
            process_artifacts=[],
            filesystem_artifacts=[],
            metadata={},
        ).with_integrity()

        mock_task_run = Mock()
        mock_task_run.extra_launch_metadata.return_value = {
            "host_inspector_ignore_process_rules": [{"executable_basename": "node"}]
        }
        with patch.object(harness, "_agent_requires_benchmark_network", return_value=False), patch.object(
            harness, "_prepare_sandbox_handle", return_value=(target, None)
        ), patch.object(harness, "list_checkpoint_manifests", return_value=[manifest]), patch.object(
            harness, "build_task_run", return_value=mock_task_run
        ), patch("benchmarks.support.resolve_checkpoint_copy_plan", return_value=[(CheckpointId("ckpt-1"), False, True)]), patch(
            "benchmarks.real_host_scenario_base.subprocess.run"
        ):
            harness.clone_checkpoint_to_fork(source, CheckpointId("ckpt-1"), "sbx-fork")

        persisted = harness.sandbox_manager._items[target.sandbox_id]
        self.assertEqual(
            persisted.metadata["host_inspector_ignore_process_rules"],
            [{"executable_basename": "node"}],
        )
        self.assertEqual(
            persisted_descriptions[-1].metadata["host_inspector_ignore_process_rules"],
            [{"executable_basename": "node"}],
        )

    def test_restore_once_notifies_task_run_after_successful_restore(self) -> None:
        harness = RealHostScenarioHarness(
            provider="openai",
            transfer_delay_ms=0.0,
            scheduler_config=SchedulerConfig(require_change_signal=False),
            scheduler_policy=object(),
            checkpoint_manager_factory=lambda base: base,
            max_workers=1,
        )
        sandbox = SandboxHandle(
            sandbox_id=SandboxId("sbx-restore"),
            bundle_dir=Path("/tmp/sbx-restore"),
            status_port=8123,
            last_status={},
            task_run=SimpleNamespace(on_restore_complete=Mock()),
        )
        harness.system = SimpleNamespace(
            restore_once=Mock(
                return_value=SimpleNamespace(
                    status=SimpleNamespace(value="succeeded"),
                    checkpoint_id=CheckpointId("ckpt-1"),
                )
            )
        )

        result = harness.restore_once(sandbox, CheckpointId("ckpt-1"))

        sandbox.task_run.on_restore_complete.assert_called_once()
        self.assertEqual(result.checkpoint_id, CheckpointId("ckpt-1"))

    def test_wait_for_recovery_notifies_task_run_after_successful_restore(self) -> None:
        harness = RealHostScenarioHarness(
            provider="openai",
            transfer_delay_ms=0.0,
            scheduler_config=SchedulerConfig(require_change_signal=False),
            scheduler_policy=object(),
            checkpoint_manager_factory=lambda base: base,
            max_workers=1,
        )
        observed_at = utc_now()
        sandbox = SandboxHandle(
            sandbox_id=SandboxId("sbx-recovery"),
            bundle_dir=Path("/tmp/sbx-recovery"),
            status_port=8123,
            last_status={},
            task_run=SimpleNamespace(on_restore_complete=Mock()),
        )
        harness.system = SimpleNamespace(
            get_last_recovery_record=Mock(
                return_value=SimpleNamespace(
                    event_type="fault",
                    started_at=observed_at,
                    status="restored",
                    checkpoint_id=CheckpointId("ckpt-1"),
                )
            )
        )

        record = harness.wait_for_recovery(
            sandbox,
            event_type="fault",
            observed_after=observed_at,
            timeout_s=0.1,
        )

        sandbox.task_run.on_restore_complete.assert_called_once()
        self.assertEqual(record.checkpoint_id, CheckpointId("ckpt-1"))

    def test_inject_fault_waits_for_quiescent_window(self) -> None:
        harness = RealHostScenarioHarness(
            provider="openai",
            transfer_delay_ms=0.0,
            scheduler_config=SchedulerConfig(require_change_signal=False),
            scheduler_policy=object(),
            checkpoint_manager_factory=lambda base: base,
            max_workers=1,
        )
        sandbox = SandboxHandle(
            sandbox_id=SandboxId("sbx-fault"),
            bundle_dir=Path("/tmp/sbx-fault"),
            status_port=8123,
            last_status={},
        )
        state = {"request_in_flight": True, "checkpoint_active": True}
        harness.request_state_store = SimpleNamespace(
            get=lambda sandbox_id: SimpleNamespace(llm_request_in_flight=state["request_in_flight"])
        )
        harness.executor = SimpleNamespace(has_active_checkpoint=lambda sandbox_id: state["checkpoint_active"])
        harness._delete_runtime = Mock()
        harness._set_sandbox_running_state = Mock()

        def _settle() -> None:
            time.sleep(0.05)
            state["request_in_flight"] = False
            time.sleep(0.05)
            state["checkpoint_active"] = False

        thread = threading.Thread(target=_settle)
        thread.start()
        try:
            harness.inject_fault(sandbox)
        finally:
            thread.join(timeout=1.0)

        harness._delete_runtime.assert_called_once_with(sandbox.sandbox_id)
        harness._set_sandbox_running_state.assert_called_once_with(sandbox.sandbox_id, is_running=False)

    def test_inject_fault_times_out_when_quiescent_window_never_arrives(self) -> None:
        harness = RealHostScenarioHarness(
            provider="openai",
            transfer_delay_ms=0.0,
            scheduler_config=SchedulerConfig(require_change_signal=False),
            scheduler_policy=object(),
            checkpoint_manager_factory=lambda base: base,
            max_workers=1,
        )
        sandbox = SandboxHandle(
            sandbox_id=SandboxId("sbx-fault"),
            bundle_dir=Path("/tmp/sbx-fault"),
            status_port=8123,
            last_status={},
        )
        harness.request_state_store = SimpleNamespace(
            get=lambda sandbox_id: SimpleNamespace(llm_request_in_flight=True)
        )
        harness.executor = SimpleNamespace(has_active_checkpoint=lambda sandbox_id: True)
        harness._delete_runtime = Mock()
        harness._set_sandbox_running_state = Mock()

        result = harness.wait_for_fault_injection_window(sandbox, timeout_s=0.05)
        self.assertFalse(result)

        harness._delete_runtime.assert_not_called()
        harness._set_sandbox_running_state.assert_not_called()

    def test_wait_for_fault_injection_window_skips_finished_task(self) -> None:
        harness = RealHostScenarioHarness(
            provider="openai",
            transfer_delay_ms=0.0,
            scheduler_config=SchedulerConfig(require_change_signal=False),
            scheduler_policy=object(),
            checkpoint_manager_factory=lambda base: base,
            max_workers=1,
        )
        completed = Future()
        completed.set_result(None)
        sandbox = SandboxHandle(
            sandbox_id=SandboxId("sbx-finished"),
            bundle_dir=Path("/tmp/sbx-finished"),
            status_port=8123,
            last_status={"state": "finished"},
            task_future=completed,
        )
        harness.request_state_store = SimpleNamespace(
            get=lambda sandbox_id: SimpleNamespace(llm_request_in_flight=False)
        )
        harness.executor = SimpleNamespace(has_active_checkpoint=lambda sandbox_id: False)

        with self.assertLogs("benchmarks.real_host_scenario_base", level="INFO") as captured:
            result = harness.wait_for_fault_injection_window(sandbox, timeout_s=0.05)

        self.assertFalse(result)
        self.assertIn("Skipping fault injection for sandbox=sbx-finished because task already finished", "\n".join(captured.output))

    def test_load_dataset_normalizes_relative_paths_and_cycles_rows(self) -> None:
        harness = RealHostScenarioHarness(
            provider="openai",
            transfer_delay_ms=0.0,
            scheduler_config=SchedulerConfig(require_change_signal=False),
            scheduler_policy=object(),
            checkpoint_manager_factory=lambda base: base,
            max_workers=1,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_path = root / "tasks.jsonl"
            dataset_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "task_id": "task-a",
                                "agent_type": "simulated",
                                "llm_service_type": "manual",
                                "task_description": "task-a",
                                "task_config": {"minimum_actions": 1},
                                "docker_compose_file": "compose.yaml",
                                "env_file": "task.env",
                                "task_root": "tasks/task-a",
                                "llm_service_config": {"trace_path": "traces/task-a.log"},
                                "trace_response_count": 7,
                                "trace_malformed_line_count": 2,
                            }
                        ),
                        json.dumps(
                            {
                                "agent_type": "iflow",
                                "task_description": {"prompt": "task-b"},
                            }
                        ),
                    ]
                ),
                encoding="utf-8",
            )
            records = harness.load_dataset(dataset_path)
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0].task_id, "task-a")
            self.assertEqual(records[0].llm_service_type, "manual")
            self.assertEqual(records[0].docker_compose_file, (root / "compose.yaml").resolve())
            self.assertEqual(records[0].env_file, (root / "task.env").resolve())
            self.assertEqual(records[0].task_root, (root / "tasks" / "task-a").resolve())
            self.assertEqual(
                records[0].llm_service_config,
                {"trace_path": str((root / "traces" / "task-a.log").resolve())},
            )
            self.assertEqual(records[0].trace_response_count, 7)
            self.assertEqual(records[0].trace_malformed_line_count, 2)
            selected = harness.select_task_record(
                records,
                sandbox_index=3,
                default_agent_type="simulated",
                default_llm_service_type="simulated",
                default_task_description=TaskDescription("ignored"),
                default_task_config=TaskConfig(),
            )
            self.assertEqual(selected.agent_type, "iflow")
            self.assertEqual(selected.task_description.prompt, "task-b")
            self.assertIsNone(selected.llm_service_type)

    def test_resolve_llm_service_type_defaults_from_agent(self) -> None:
        harness = RealHostScenarioHarness(
            provider="openai",
            transfer_delay_ms=0.0,
            scheduler_config=SchedulerConfig(require_change_signal=False),
            scheduler_policy=object(),
            checkpoint_manager_factory=lambda base: base,
            max_workers=1,
        )

        self.assertEqual(
            harness.resolve_llm_service_type(agent_type="simulated", llm_service_type=None),
            "simulated",
        )
        self.assertEqual(
            harness.resolve_llm_service_type(agent_type="iflow", llm_service_type=None),
            "simulated_for_iflow",
        )

    def test_resolve_llm_service_type_rejects_anthropic_for_manual_only_services(self) -> None:
        harness = RealHostScenarioHarness(
            provider="anthropic",
            transfer_delay_ms=0.0,
            scheduler_config=SchedulerConfig(require_change_signal=False),
            scheduler_policy=object(),
            checkpoint_manager_factory=lambda base: base,
            max_workers=1,
        )

        with self.assertRaisesRegex(ValueError, "only supports provider=openai"):
            harness.resolve_llm_service_type(agent_type="simulated", llm_service_type="manual")

    def test_launch_task_record_routes_dataset_compose_rows(self) -> None:
        harness = RealHostScenarioHarness(
            provider="openai",
            transfer_delay_ms=0.0,
            scheduler_config=SchedulerConfig(require_change_signal=False),
            scheduler_policy=object(),
            checkpoint_manager_factory=lambda base: base,
            max_workers=1,
        )
        record = BenchmarkTaskRecord(
            agent_type="simulated",
            task_description=TaskDescription("compose"),
            task_config=TaskConfig(),
            docker_compose_file=Path("/tmp/compose.yaml"),
            env_file=Path("/tmp/task.env"),
        )
        handle = SandboxHandle(
            sandbox_id=SandboxId("sbx-compose"),
            bundle_dir=Path("/tmp/sbx-compose"),
            status_port=8123,
            last_status={},
        )
        with patch.object(harness, "launch_sandbox_from_docker_compose_file", return_value=handle) as launch_compose:
            result = harness.launch_task_record("sbx-compose", record)

        self.assertIs(result, handle)
        launch_compose.assert_called_once()
        self.assertEqual(result.launch_metadata["benchmark"]["task_id"], "sbx-compose")

    def test_launch_task_record_composes_setup_and_run_methods(self) -> None:
        harness = RealHostScenarioHarness(
            provider="openai",
            transfer_delay_ms=0.0,
            scheduler_config=SchedulerConfig(require_change_signal=False),
            scheduler_policy=object(),
            checkpoint_manager_factory=lambda base: base,
            max_workers=1,
        )
        record = BenchmarkTaskRecord(
            agent_type="simulated",
            task_description=TaskDescription("task"),
            task_config=TaskConfig(),
        )
        handle = SandboxHandle(
            sandbox_id=SandboxId("sbx-staged"),
            bundle_dir=Path("/tmp/sbx-staged"),
            status_port=8123,
            last_status={},
        )
        prepared = SimpleNamespace(handle=handle)

        with (
            patch.object(harness, "setup_task_record", return_value=prepared) as setup_task_record,
            patch.object(harness, "run_prepared_task_record", return_value=handle) as run_prepared_task_record,
            patch.object(harness, "_set_benchmark_launch_metadata") as set_benchmark_launch_metadata,
        ):
            result = harness.launch_task_record("sbx-staged", record)

        self.assertIs(result, handle)
        setup_task_record.assert_called_once_with("sbx-staged", record)
        run_prepared_task_record.assert_called_once_with(prepared)
        set_benchmark_launch_metadata.assert_not_called()

    def test_setup_task_record_prepares_runtime_but_defers_launch_until_run_phase(self) -> None:
        harness = RealHostScenarioHarness(
            provider="openai",
            transfer_delay_ms=0.0,
            scheduler_config=SchedulerConfig(require_change_signal=False),
            scheduler_policy=object(),
            checkpoint_manager_factory=lambda base: base,
            max_workers=1,
        )
        record = BenchmarkTaskRecord(
            agent_type="simulated",
            task_description=TaskDescription("task"),
            task_config=TaskConfig(),
        )
        handle = SandboxHandle(
            sandbox_id=SandboxId("sbx-staged"),
            bundle_dir=Path("/tmp/sbx-staged"),
            status_port=8123,
            last_status={},
            agent_type=record.agent_type,
            task_description=record.task_description,
            task_config=record.task_config,
        )
        prepared = SimpleNamespace(
            handle=handle,
            sandbox_name="sbx-staged",
            task_record=record,
            runtime_prepared=False,
            wait_for_ready_before_task_start=False,
            wait_for_ready_after_task_start=False,
            emit_ready_event=True,
            runtime_launched=False,
        )

        with (
            patch.object(harness, "_ensure_task_record_inputs") as ensure_task_record_inputs,
            patch.object(harness, "_prepare_runc_task_record", return_value=prepared) as prepare_runc_task_record,
            patch.object(harness, "_set_benchmark_launch_metadata") as set_benchmark_launch_metadata,
            patch.object(harness, "_prepare_prepared_runtime") as prepare_prepared_runtime,
            patch.object(harness, "_launch_prepared_runtime") as launch_prepared_runtime,
        ):
            result = harness.setup_task_record("sbx-staged", record)

        self.assertIs(result, prepared)
        ensure_task_record_inputs.assert_called_once_with("sbx-staged", record)
        prepare_runc_task_record.assert_called_once_with(sandbox_name="sbx-staged", task_record=record)
        set_benchmark_launch_metadata.assert_called_once_with(
            handle,
            sandbox_name="sbx-staged",
            task_record=record,
        )
        prepare_prepared_runtime.assert_called_once_with(prepared)
        launch_prepared_runtime.assert_not_called()

        with (
            patch.object(harness, "_prepare_prepared_runtime") as prepare_prepared_runtime,
            patch.object(harness, "_launch_prepared_runtime") as launch_prepared_runtime,
            patch.object(harness, "launch_task", return_value=Mock()) as launch_task,
        ):
            launched = harness.run_prepared_task_record(prepared)

        self.assertIs(launched, handle)
        prepare_prepared_runtime.assert_called_once_with(prepared)
        launch_prepared_runtime.assert_called_once_with(prepared)
        launch_task.assert_called_once_with(
            record.agent_type,
            record.task_description,
            record.task_config,
            "sbx-staged",
        )

    def test_benchmark_phase_map_emits_phase_telemetry(self) -> None:
        telemetry = InMemoryTelemetrySink()
        harness = SimpleNamespace(telemetry=telemetry)

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            results = benchmark_phase_map(
                [1, 2],
                lambda value: value * 2,
                phase="setup",
                max_workers=2,
                harness=harness,
                item_attributes=lambda value: {
                    "component": "benchmark",
                    "sandbox_id": f"sbx-{value}",
                    "task_id": f"task-{value}",
                    "phase": "setup",
                    "phase_scope": "sandbox",
                },
            )

        self.assertEqual(results, [2, 4])
        output = buffer.getvalue()
        self.assertIn("benchmark.phase.setup start sandboxes=2 max_workers=2", output)
        self.assertIn("benchmark.phase.setup end sandboxes=2 max_workers=2 duration_s=", output)
        event_names = [name for name, _ in telemetry.events]
        metric_names = [name for name, _, _ in telemetry.metrics]
        self.assertIn("benchmark.phase.setup.start", event_names)
        self.assertIn("benchmark.phase.setup.finish", event_names)
        self.assertIn("benchmark.phase.setup.item.start", event_names)
        self.assertIn("benchmark.phase.setup.item.finish", event_names)
        self.assertIn("benchmark.phase.setup.duration_ms", metric_names)
        self.assertIn("benchmark.phase.setup.item.duration_ms", metric_names)
        self.assertIn("benchmark.phase.setup.configured_max_workers", metric_names)

    def test_benchmark_setup_run_pipeline_keeps_separate_executor_pools_by_default(self) -> None:
        run_started = threading.Event()
        second_setup_started = threading.Event()
        release_first_run = threading.Event()
        errors: list[BaseException] = []

        def setup_fn(item: int) -> int:
            if item == 1:
                second_setup_started.set()
            return item

        def run_fn(item: int, prepared: int) -> int:
            _ = prepared
            if item == 0:
                run_started.set()
                self.assertTrue(second_setup_started.wait(timeout=1.0))
                release_first_run.wait(timeout=1.0)
            return item

        def _run_pipeline() -> None:
            try:
                benchmark_setup_run_pipeline(
                    [0, 1],
                    setup_fn=setup_fn,
                    run_fn=run_fn,
                    setup_max_workers=1,
                    run_max_workers=1,
                    harness=SimpleNamespace(telemetry=None),
                )
            except BaseException as exc:  # pragma: no cover - surfaced via assertion below
                errors.append(exc)

        thread = threading.Thread(
            target=_run_pipeline,
        )
        thread.start()
        try:
            self.assertTrue(run_started.wait(timeout=1.0))
        finally:
            release_first_run.set()
            thread.join(timeout=1.0)

        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])

    def test_benchmark_setup_run_pipeline_can_use_shared_executor_pool(self) -> None:
        second_setup_started = threading.Event()
        order: list[str] = []

        def setup_fn(item: int) -> int:
            order.append(f"setup-{item}")
            if item == 1:
                second_setup_started.set()
            return item

        def run_fn(item: int, prepared: int) -> int:
            _ = prepared
            order.append(f"run-{item}")
            if item == 0:
                self.assertFalse(second_setup_started.wait(timeout=0.1))
            return item

        results = benchmark_setup_run_pipeline(
            [0, 1],
            setup_fn=setup_fn,
            run_fn=run_fn,
            setup_max_workers=1,
            run_max_workers=1,
            harness=SimpleNamespace(telemetry=None),
            executor_pool="shared",
        )

        self.assertEqual(results, [0, 1])
        self.assertEqual(order, ["setup-0", "run-0", "setup-1", "run-1"])

    def test_emit_benchmark_phase_skipped_prints_progress_line(self) -> None:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            emit_benchmark_phase_skipped(
                phase="verification",
                sandbox_count=3,
                configured_max_workers=5,
            )

        self.assertEqual(
            buffer.getvalue(),
            "benchmark.phase.verification skipped sandboxes=3 max_workers=5\n",
        )

    def test_e2e_benchmark_uses_shared_harness_launch_flow(self) -> None:
        task_run = SimpleNamespace(
            wait_for_progress=Mock(return_value={"total_actions": 6}),
        )
        prepared = SimpleNamespace(
            sandbox_name="sandbox-0",
            handle=SandboxHandle(
                sandbox_id=SandboxId("sandbox-0"),
                bundle_dir=Path("/tmp") / "sandbox-0",
                status_port=8123,
                last_status={
                    "total_actions": 6,
                    "filesystem_actions": 1,
                    "process_actions": 1,
                    "network_actions": 1,
                },
                task_run=task_run,
            ),
        )
        harness = SimpleNamespace(
            setup_task_record=Mock(return_value=prepared),
            run_prepared_task_record=Mock(return_value=prepared.handle),
            request_state_store=None,
            checkpoint_if_due=Mock(return_value=None),
            restore_once=Mock(),
            get_sandbox_handle=lambda sandbox_id: SandboxHandle(
                sandbox_id=SandboxId(sandbox_id),
                bundle_dir=Path("/tmp") / sandbox_id,
                status_port=8123,
                last_status={},
            ),
            set_snapshot_metadata=Mock(),
        )
        rows = run_e2e_manual(
            BenchmarkConfig(
                config_path=Path("/tmp/bench.yaml"),
                scenario="e2e",
                mode="manual",
                provider="openai",
                agent="simulated",
                llm_service="simulated",
                task_dataset=None,
                sandboxes=1,
                iterations=1,
                output=None,
                log_level="info",
                transfer_delay_ms=0.0,
                work_dir_host_root=None,
                scenario_options={},
            ),
            harness,
        )
        self.assertEqual(len(rows), 1)
        harness.setup_task_record.assert_called_once()
        harness.run_prepared_task_record.assert_called_once_with(prepared)


if __name__ == "__main__":
    unittest.main()
