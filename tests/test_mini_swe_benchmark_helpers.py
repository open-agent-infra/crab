from __future__ import annotations

import unittest
from concurrent.futures import Future
from datetime import datetime, timezone
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock, call

from crab import CheckpointId, SandboxDescription, SandboxId
from crab.models import CheckpointManifest
from benchmarks.real_host_scenario_base import RealHostScenarioHarness, _SpeculativeSandboxController
from benchmarks.support import is_replay_llm_service_type
from integrations.agents import SandboxHandle
from integrations.llm_services.router import default_llm_service_type_for_agent


class MiniSWEBenchmarkHelperTests(unittest.TestCase):
    def test_replay_detection_includes_mini_swe_trace_replay(self) -> None:
        self.assertTrue(is_replay_llm_service_type("mini_swe_trace_replay"))
        self.assertTrue(is_replay_llm_service_type("mini_swe_spec_trace_replay"))

    def test_default_llm_service_type_for_mini_swe(self) -> None:
        self.assertEqual(default_llm_service_type_for_agent("mini_swe"), "mini_swe_trace_replay")

    def test_llm_service_checkpoint_metadata_prefers_task_status_cursor(self) -> None:
        sandbox_id = SandboxId("sbx-mini-meta")
        sandbox = SandboxHandle(
            sandbox_id=sandbox_id,
            bundle_dir=Path("/tmp/bundle"),
            status_port=8123,
            last_status={},
            llm_service_type="mini_swe_trace_replay",
        )
        sandbox.task_run = Mock()
        sandbox.task_run.poll_status.return_value = {"replay_trace_cursor": 7}

        harness = RealHostScenarioHarness.__new__(RealHostScenarioHarness)
        harness._sandbox_by_id = {sandbox_id: sandbox}
        harness._snapshot_llm_services = Mock()

        metadata = RealHostScenarioHarness._llm_service_checkpoint_metadata(harness, sandbox_id)

        self.assertEqual(metadata, {"benchmark_trace_cursor": 7})
        harness._snapshot_llm_services.assert_not_called()

    def test_llm_service_checkpoint_metadata_accepts_claude_snapshot_cursor(self) -> None:
        sandbox_id = SandboxId("sbx-claude-meta")
        sandbox = SandboxHandle(
            sandbox_id=sandbox_id,
            bundle_dir=Path("/tmp/bundle"),
            status_port=8123,
            last_status={},
            llm_service_type="claude_code_trace_replay",
        )
        sandbox.task_run = Mock()
        sandbox.task_run.poll_status.side_effect = RuntimeError("timed out")

        harness = RealHostScenarioHarness.__new__(RealHostScenarioHarness)
        harness._sandbox_by_id = {sandbox_id: sandbox}
        harness._snapshot_llm_services = Mock(
            return_value={
                str(sandbox_id): {
                    "llm_service_type": "claude_code_trace_replay",
                    "state": {"trace_cursor": 9, "total_responses": 23, "is_complete": False},
                }
            }
        )

        metadata = RealHostScenarioHarness._llm_service_checkpoint_metadata(harness, sandbox_id)

        self.assertEqual(metadata, {"benchmark_trace_cursor": 9})

    def test_llm_service_checkpoint_metadata_ignores_claude_auxiliary_request_counters(self) -> None:
        sandbox_id = SandboxId("sbx-claude-meta-aux")
        sandbox = SandboxHandle(
            sandbox_id=sandbox_id,
            bundle_dir=Path("/tmp/bundle"),
            status_port=8123,
            last_status={},
            llm_service_type="claude_code_trace_replay",
        )
        sandbox.task_run = Mock()
        sandbox.task_run.poll_status.side_effect = RuntimeError("timed out")

        harness = RealHostScenarioHarness.__new__(RealHostScenarioHarness)
        harness._sandbox_by_id = {sandbox_id: sandbox}
        harness._snapshot_llm_services = Mock(
            return_value={
                str(sandbox_id): {
                    "llm_service_type": "claude_code_trace_replay",
                    "state": {
                        "trace_cursor": 4,
                        "total_responses": 23,
                        "is_complete": False,
                        "main_loop_request_count": 4,
                        "helper_request_count": 12,
                        "count_tokens_request_count": 7,
                    },
                }
            }
        )

        metadata = RealHostScenarioHarness._llm_service_checkpoint_metadata(harness, sandbox_id)

        self.assertEqual(metadata, {"benchmark_trace_cursor": 4})

    def test_restore_llm_service_state_records_restore_trace_cursor_for_task_run(self) -> None:
        sandbox_id = SandboxId("sbx-mini-restore-cursor")
        sandbox = SandboxHandle(
            sandbox_id=sandbox_id,
            bundle_dir=Path("/tmp/bundle"),
            status_port=8123,
            last_status={},
            llm_service_type="mini_swe_trace_replay",
        )
        sandbox.task_run = Mock()

        harness = RealHostScenarioHarness.__new__(RealHostScenarioHarness)
        harness._sandbox_by_id = {sandbox_id: sandbox}
        harness.llm_router_client = Mock()
        harness.llm_server = None
        harness.storage = Mock()
        harness.storage.get_manifest.return_value = CheckpointManifest(
            schema_version="v1",
            checkpoint_id=CheckpointId("ckpt-fs"),
            sandbox_id=sandbox_id,
            created_at=datetime.now(timezone.utc),
            runtime_name="runc",
            runtime_version=None,
            process_artifacts=[],
            filesystem_artifacts=[],
            metadata={"benchmark_trace_cursor": 7},
            integrity={},
        )
        manifest = CheckpointManifest(
            schema_version="v1",
            checkpoint_id=CheckpointId("ckpt-current"),
            sandbox_id=sandbox_id,
            created_at=datetime.now(timezone.utc),
            runtime_name="runc",
            runtime_version=None,
            process_artifacts=[],
            filesystem_artifacts=[],
            metadata={
                "filesystem_restore_checkpoint_id": "ckpt-fs",
                "process_restore_trace_cursor": 7,
                "filesystem_restore_trace_cursor": 7,
            },
            integrity={},
        )

        RealHostScenarioHarness._restore_llm_service_state(harness, sandbox_id, manifest)

        sandbox.task_run.record_restore_trace_cursor.assert_called_once_with(7)

    def test_relaunch_sandbox_skips_llm_reset_when_preserving_mini_swe_task(self) -> None:
        sandbox_id = SandboxId("sbx-mini-relaunch")
        sandbox = SandboxHandle(
            sandbox_id=sandbox_id,
            bundle_dir=Path("/tmp/bundle"),
            status_port=8123,
            last_status={"state": "running"},
            llm_service_type="mini_swe_trace_replay",
            launch_source="compose",
        )
        sandbox.task_run = Mock()
        sandbox.task_run.survives_fault_relaunch.return_value = True
        sandbox.task_future = Future()

        harness = RealHostScenarioHarness.__new__(RealHostScenarioHarness)
        harness.base_inspector = Mock()
        harness.runtime = Mock()
        harness.runtime.describe.return_value = SandboxDescription(
            sandbox_id=sandbox_id,
            runtime_name="runc",
            status="running",
            metadata={"bundle_path": "/tmp/bundle"},
        )
        harness.runtime.launch.return_value = sandbox_id
        harness._sandbox_by_id = {sandbox_id: sandbox}
        harness._delete_runtime = Mock()
        harness._destroy_filesystem_dataset = Mock()
        harness._reset_llm_service_state = Mock()

        payload = RealHostScenarioHarness.relaunch_sandbox(harness, sandbox)

        self.assertEqual(payload, {"state": "running"})
        harness._reset_llm_service_state.assert_not_called()
        sandbox.task_run.request_stop.assert_not_called()

    def test_speculative_controller_promotion_copies_llm_state_before_swap(self) -> None:
        active = SandboxHandle(
            sandbox_id=SandboxId("sbx-active"),
            bundle_dir=Path("/tmp/active"),
            status_port=8123,
            last_status={},
        )
        fork = SandboxHandle(
            sandbox_id=SandboxId("sbx-fork"),
            bundle_dir=Path("/tmp/fork"),
            status_port=8124,
            last_status={},
        )
        harness = Mock()
        controller = _SpeculativeSandboxController(harness, active)

        controller.promote_fork(fork)

        harness._copy_llm_router_state.assert_called_once_with(
            source_sandbox_id=SandboxId("sbx-active"),
            target_sandbox_id=SandboxId("sbx-fork"),
        )
        harness._swap_speculative_sandbox_state.assert_called_once_with(active, fork)

    def test_speculative_controller_promotion_promotes_clone_before_swap(self) -> None:
        active = SandboxHandle(
            sandbox_id=SandboxId("sbx-active"),
            bundle_dir=Path("/tmp/active"),
            status_port=8123,
            last_status={},
        )
        fork = SandboxHandle(
            sandbox_id=SandboxId("sbx-fork"),
            bundle_dir=Path("/tmp/fork"),
            status_port=8124,
            last_status={},
        )
        harness = Mock()
        controller = _SpeculativeSandboxController(harness, active)

        controller.promote_fork(fork)

        harness._promote_filesystem_dataset.assert_called_once_with(SandboxId("sbx-fork"))
        self.assertEqual(
            harness.mock_calls[:3],
            [
                call._copy_llm_router_state(
                    source_sandbox_id=SandboxId("sbx-active"),
                    target_sandbox_id=SandboxId("sbx-fork"),
                ),
                call._promote_filesystem_dataset(SandboxId("sbx-fork")),
                call._swap_speculative_sandbox_state(active, fork),
            ],
        )

    def test_speculative_controller_restores_new_fork_before_reuse(self) -> None:
        active = SandboxHandle(
            sandbox_id=SandboxId("sbx-active"),
            bundle_dir=Path("/tmp/active"),
            status_port=8123,
            last_status={},
        )
        fork = SandboxHandle(
            sandbox_id=SandboxId("sbx-fork"),
            bundle_dir=Path("/tmp/fork"),
            status_port=8124,
            last_status={},
        )
        fork.task_run = Mock()
        harness = Mock()
        harness.storage = Mock()
        harness.storage.list_checkpoints.return_value = [CheckpointId("ckpt-1")]
        harness.clone_checkpoint_to_fork.return_value = fork
        harness.restore_once.return_value = SimpleNamespace(
            status=SimpleNamespace(value="succeeded"),
            message="",
        )
        controller = _SpeculativeSandboxController(harness, active)

        prepared = controller.ensure_fork()

        self.assertEqual(prepared, (fork, True, False))
        harness.clone_checkpoint_to_fork.assert_called_once_with(active, CheckpointId("ckpt-1"), "sbx-active-spec-1")
        fork.task_run.configure_bundle.assert_called_once_with()
        harness.restore_once.assert_called_once_with(fork, CheckpointId("ckpt-1"))

    def test_benchmark_telemetry_attributes_include_stable_task_run_id(self) -> None:
        sandbox = SandboxHandle(
            sandbox_id=SandboxId("sbx-active-spec-3"),
            bundle_dir=Path("/tmp/active"),
            status_port=8123,
            last_status={},
            agent_type="mini_swe",
            llm_service_type="mini_swe_spec_trace_replay",
        )
        sandbox.launch_metadata["benchmark"] = {
            "task_id": "django__django-13820",
            "task_run_id": "sbx-active",
        }
        harness = RealHostScenarioHarness.__new__(RealHostScenarioHarness)
        harness._task_attempts = {str(sandbox.sandbox_id): 2}

        attributes = RealHostScenarioHarness.benchmark_telemetry_attributes(harness, sandbox)

        self.assertEqual(attributes["sandbox_id"], "sbx-active-spec-3")
        self.assertEqual(attributes["task_id"], "django__django-13820")
        self.assertEqual(attributes["task_run_id"], "sbx-active")
        self.assertEqual(attributes["task_attempt"], 2)

    def test_speculative_controller_uses_stable_root_name_after_promotion(self) -> None:
        active = SandboxHandle(
            sandbox_id=SandboxId("sbx-active"),
            bundle_dir=Path("/tmp/active"),
            status_port=8123,
            last_status={},
        )
        first_fork = SandboxHandle(
            sandbox_id=SandboxId("sbx-active-spec-1"),
            bundle_dir=Path("/tmp/fork-1"),
            status_port=8124,
            last_status={},
        )
        second_fork = SandboxHandle(
            sandbox_id=SandboxId("sbx-active-spec-2"),
            bundle_dir=Path("/tmp/fork-2"),
            status_port=8125,
            last_status={},
        )
        first_fork.task_run = Mock()
        second_fork.task_run = Mock()
        harness = Mock()
        harness.storage = Mock()
        harness.storage.list_checkpoints.return_value = [CheckpointId("ckpt-1")]
        harness.clone_checkpoint_to_fork.side_effect = [first_fork, second_fork]
        harness.restore_once.return_value = SimpleNamespace(
            status=SimpleNamespace(value="succeeded"),
            message="",
        )
        controller = _SpeculativeSandboxController(harness, active)

        prepared_first = controller.ensure_fork()
        self.assertEqual(prepared_first, (first_fork, True, False))

        active.sandbox_id = SandboxId("sbx-active-spec-1")
        controller.release_fork(first_fork, reusable=False, accepted=False)

        prepared_second = controller.ensure_fork()

        self.assertEqual(prepared_second, (second_fork, True, False))
        self.assertEqual(
            harness.clone_checkpoint_to_fork.call_args_list[1].args,
            (active, CheckpointId("ckpt-1"), "sbx-active-spec-2"),
        )

    def test_speculative_controller_invalidate_fork_clears_cached_handle(self) -> None:
        active = SandboxHandle(
            sandbox_id=SandboxId("sbx-active"),
            bundle_dir=Path("/tmp/active"),
            status_port=8123,
            last_status={},
        )
        fork = SandboxHandle(
            sandbox_id=SandboxId("sbx-active-spec-1"),
            bundle_dir=Path("/tmp/fork-1"),
            status_port=8124,
            last_status={},
        )
        harness = Mock()
        controller = _SpeculativeSandboxController(harness, active)
        controller._cached_fork = fork

        controller.invalidate_fork()

        self.assertIsNone(controller._cached_fork)
        harness.destroy_sandbox_dataset.assert_called_once_with(fork)

    def test_speculative_controller_defaults_eager_cleanup_off(self) -> None:
        active = SandboxHandle(
            sandbox_id=SandboxId("sbx-active"),
            bundle_dir=Path("/tmp/active"),
            status_port=8123,
            last_status={},
        )
        controller = _SpeculativeSandboxController(Mock(), active)

        self.assertFalse(controller.eager_cleanup_on_reject)

    def test_speculative_controller_accepts_eager_cleanup_flag(self) -> None:
        active = SandboxHandle(
            sandbox_id=SandboxId("sbx-active"),
            bundle_dir=Path("/tmp/active"),
            status_port=8123,
            last_status={},
        )
        controller = _SpeculativeSandboxController(
            Mock(), active, eager_cleanup_on_reject=True
        )

        self.assertTrue(controller.eager_cleanup_on_reject)

    def test_harness_build_task_run_propagates_eager_cleanup_flag(self) -> None:
        harness = RealHostScenarioHarness.__new__(RealHostScenarioHarness)
        harness.fork_reuse_enabled = False
        harness.eager_fork_cleanup_on_reject = True
        harness.background_prefork_enabled = False
        harness.prefork_min_interval_seconds = 0.0
        harness.prefork_wait_timeout_seconds = 0.0
        harness.prefork_max_concurrent_global = 0
        harness._speculation_controllers_by_sandbox = {}
        harness.runtime_state_root = Path("/tmp/rt")
        harness.runtime = Mock()
        harness._effective_agent_host_root = lambda: Path("/tmp/agent-host")
        harness.telemetry = None

        sandbox = SandboxHandle(
            sandbox_id=SandboxId("sbx-main"),
            bundle_dir=Path("/tmp/main"),
            status_port=8123,
            last_status={},
            llm_service_type="mini_swe_spec_trace_replay",
        )
        captured_kwargs: dict[str, object] = {}

        class _StubAgent:
            def __init__(self, *args, **kwargs) -> None:
                captured_kwargs.update(kwargs)

        harness.get_agent_class = lambda agent_type: _StubAgent
        RealHostScenarioHarness.build_task_run(
            harness,
            "mini_swe",
            sandbox,
            Mock(),
            Mock(),
        )

        controller = captured_kwargs.get("speculation_controller")
        self.assertIsNotNone(controller)
        self.assertTrue(controller.eager_cleanup_on_reject)
        self.assertFalse(controller._fork_reuse_enabled)

    def test_destroy_sandbox_dataset_removes_handle_from_sandboxes_list(self) -> None:
        main = SandboxHandle(
            sandbox_id=SandboxId("sbx-main"),
            bundle_dir=Path("/tmp/main"),
            status_port=8123,
            last_status={},
        )
        fork = SandboxHandle(
            sandbox_id=SandboxId("sbx-main-spec-1"),
            bundle_dir=Path("/tmp/fork-1"),
            status_port=8124,
            last_status={},
        )
        harness = RealHostScenarioHarness.__new__(RealHostScenarioHarness)
        harness.sandboxes = [main, fork]
        harness._sandbox_by_id = {main.sandbox_id: main, fork.sandbox_id: fork}
        harness.runtime = Mock()
        harness.runtime.describe.side_effect = KeyError(fork.sandbox_id)
        harness._delete_runtime = Mock()
        harness._unregister_host_inspector = Mock()
        harness._unregister_llm_service = Mock()
        harness.network_manager = Mock()
        harness._destroy_filesystem_dataset = Mock()
        harness._fork_chain_pins = {}
        harness._speculation_controllers_by_sandbox = {}

        RealHostScenarioHarness.destroy_sandbox_dataset(harness, fork)

        self.assertEqual(harness.sandboxes, [main])
        self.assertNotIn(fork.sandbox_id, harness._sandbox_by_id)

    def test_sandbox_requires_benchmark_network_detects_bundle_network_namespace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle_dir = Path(tmp) / "bundle"
            bundle_dir.mkdir(parents=True, exist_ok=True)
            (bundle_dir / "config.json").write_text(
                '{"linux":{"namespaces":[{"type":"network","path":"/var/run/netns/test"}]}}',
                encoding="utf-8",
            )
            sandbox = SandboxHandle(
                sandbox_id=SandboxId("sbx-mini-spec"),
                bundle_dir=bundle_dir,
                status_port=8123,
                last_status={},
                agent_type="mini_swe",
            )
            harness = RealHostScenarioHarness.__new__(RealHostScenarioHarness)
            harness.network_manager = Mock()
            harness.network_manager.lease_for.return_value = None
            harness._agent_registry = {"mini_swe": SimpleNamespace(requires_network_namespace=False)}

            self.assertTrue(RealHostScenarioHarness._sandbox_requires_benchmark_network(harness, sandbox))


if __name__ == "__main__":
    unittest.main()
