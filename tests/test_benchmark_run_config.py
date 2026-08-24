from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from benchmarks.config import (
    BenchmarkConfig,
    BenchmarkPhaseMergingConfig,
    BenchmarkRootfsReuseConfig,
    BenchmarkStoragePlanesConfig,
    TelemetryReportConfig,
    load_config,
)
from benchmarks.core import resolve_task_records
from benchmarks.run import _incomplete_replay_lines, run_benchmark_config
from benchmarks.scenarios import HarnessSettings, ScenarioDefinition
from integrations.agents import TaskConfig, TaskDescription


class BenchmarkConfigTests(unittest.TestCase):
    def test_load_config_resolves_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "bench.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "scenario: fault",
                        "mode: auto",
                        "task_dataset: datasets/tasks.jsonl",
                        "sandboxes: 4",
                        "output: results/out.csv",
                        "log_file: results/out.log",
                        "max_workers: 3",
                        "log_file_mode: write",
                        "benchmark_root: benchmark-runs",
                        "zpool_size: 32G",
                        "zpool_name: benchcache",
                        "zpool_image: cache/bench.zpool.img",
                        "reuse_zpool: true",
                        "runtime_command_timeout_seconds: 180",
                        "runtime_zfs_prepare_timeout_seconds: 420",
                        "storage_planes:",
                        "  runtime_root: runtime-plane",
                        "  storage_root: storage-plane",
                        "  agent_host_root: agent-plane",
                        "telemetry_output: results/out.telemetry.jsonl",
                        "image_cache_root: cache/images",
                        "work_dir_host_root: workdirs",
                        "scenario_options:",
                        "  injection_rate: 0.25",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_config(config_path)

            self.assertEqual(config.scenario, "fault")
            self.assertEqual(config.mode, "auto")
            self.assertEqual(config.task_dataset, (root / "datasets" / "tasks.jsonl").resolve())
            self.assertEqual(config.output, (root / "results" / "out.csv").resolve())
            self.assertEqual(config.log_file, (root / "results" / "out.log").resolve())
            self.assertEqual(config.max_workers, 3)
            self.assertEqual(config.effective_max_workers, 3)
            self.assertEqual(config.effective_phase_workers.as_dict(), {"setup": 3, "run": 3, "verification": 3})
            self.assertEqual(config.log_file_mode, "write")
            self.assertEqual(config.benchmark_root, (root / "benchmark-runs").resolve())
            self.assertEqual(config.benchmark_root_home, (root / "benchmark-runs").resolve())
            self.assertIsNone(config.benchmark_run_name)
            self.assertFalse(config.clear_benchmark_root_after_run)
            self.assertEqual(config.zpool_size, "32G")
            self.assertEqual(config.zpool_name, "benchcache")
            self.assertEqual(config.zpool_image, (root / "cache" / "bench.zpool.img").resolve())
            self.assertTrue(config.reuse_zpool)
            self.assertEqual(config.runtime_command_timeout_seconds, 180.0)
            self.assertEqual(config.runtime_zfs_prepare_timeout_seconds, 420.0)
            self.assertEqual(config.storage_planes.runtime_root, (root / "runtime-plane").resolve())
            self.assertEqual(config.storage_planes.storage_root, (root / "storage-plane").resolve())
            self.assertEqual(config.storage_planes.agent_host_root, (root / "agent-plane").resolve())
            self.assertEqual(config.telemetry_output, (root / "results" / "out.telemetry.jsonl").resolve())
            self.assertEqual(config.image_cache_root, (root / "cache" / "images").resolve())
            self.assertEqual(config.work_dir_host_root, (root / "workdirs").resolve())
            self.assertEqual(config.scenario_options["injection_rate"], 0.25)

    def test_load_config_resolves_benchmark_root_home_and_run_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "bench.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "scenario: fault",
                        "mode: auto",
                        "benchmark_root_home: benchmark-runs",
                        "benchmark_run_name: named-run",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_config(config_path)

            self.assertEqual(config.benchmark_root_home, (root / "benchmark-runs").resolve())
            self.assertEqual(config.benchmark_root, (root / "benchmark-runs").resolve())
            self.assertEqual(config.benchmark_run_name, "named-run")

    def test_load_config_rejects_conflicting_benchmark_root_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "bench.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "scenario: fault",
                        "mode: auto",
                        "benchmark_root: old-root",
                        "benchmark_root_home: new-root",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "aliases"):
                load_config(config_path)

    def test_load_config_rejects_invalid_benchmark_run_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "bench.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "scenario: fault",
                        "mode: auto",
                        "benchmark_root_home: benchmark-runs",
                        "benchmark_run_name: nested/run",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "benchmark_run_name"):
                load_config(config_path)

    def test_load_config_supports_nested_telemetry_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "bench.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "scenario: fault",
                        "mode: auto",
                        "telemetry:",
                        "  output: results/nested.telemetry.jsonl",
                        "  detail_level: detailed",
                        "  capture_command_output: true",
                        "  max_text_attribute_bytes: 512",
                        "  keep_in_memory_copy: false",
                        "  writer_mode: sync",
                        "  queue_capacity: 99",
                        "  batch_max_records: 7",
                        "  flush_interval_ms: 11",
                        "  overflow_policy: block",
                        "  serializer: stdlib",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_config(config_path)

            self.assertEqual(
                config.telemetry_output,
                (root / "results" / "nested.telemetry.jsonl").resolve(),
            )
            self.assertEqual(config.telemetry_file_mode, "append")
            self.assertEqual(config.telemetry_detail_level, "detailed")
            self.assertTrue(config.telemetry_capture_command_output)
            self.assertEqual(config.telemetry_max_text_attribute_bytes, 512)
            self.assertFalse(config.telemetry_keep_in_memory_copy)
            self.assertEqual(config.telemetry_writer_mode, "sync")
            self.assertEqual(config.telemetry_queue_capacity, 99)
            self.assertEqual(config.telemetry_batch_max_records, 7)
            self.assertEqual(config.telemetry_flush_interval_ms, 11)
            self.assertEqual(config.telemetry_overflow_policy, "block")
            self.assertEqual(config.telemetry_serializer, "stdlib")
            self.assertTrue(config.telemetry_report.enabled)
            self.assertIsNone(config.telemetry_report.output_dir)
            self.assertEqual(config.monitoring.sample_interval_ms, 1000)

    def test_load_config_allows_zero_iterations_for_fault(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "bench.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "scenario: fault",
                        "mode: auto",
                        "iterations: 0",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_config(config_path)

        self.assertEqual(config.iterations, 0)

    def test_load_config_defaults_spec_iterations_to_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "bench.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "scenario: spec",
                        "mode: auto",
                        "agent: mini_swe",
                        "llm_service: mini_swe_spec_trace_replay",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_config(config_path)

        self.assertEqual(config.iterations, 0)

    def test_load_config_allows_zero_iterations_for_spec(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "bench.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "scenario: spec",
                        "mode: manual",
                        "agent: mini_swe",
                        "llm_service: mini_swe_spec_trace_replay",
                        "iterations: 0",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_config(config_path)

        self.assertEqual(config.iterations, 0)

    def test_load_config_rejects_positive_iterations_for_spec(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "bench.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "scenario: spec",
                        "mode: auto",
                        "agent: mini_swe",
                        "llm_service: mini_swe_spec_trace_replay",
                        "iterations: 1",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "iterations must be exactly 0 for scenario 'spec'"):
                load_config(config_path)

    def test_load_config_rejects_non_mini_swe_agent_for_spec(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "bench.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "scenario: spec",
                        "mode: auto",
                        "agent: iflow",
                        "llm_service: mini_swe_spec_trace_replay",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "scenario='spec' requires agent='mini_swe'"):
                load_config(config_path)

    def test_load_config_requires_spec_llm_service_for_spec(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "bench.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "scenario: spec",
                        "mode: manual",
                        "agent: mini_swe",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                "scenario='spec' requires llm_service='mini_swe_spec_trace_replay'",
            ):
                load_config(config_path)

    def test_load_config_supports_verification_and_telemetry_file_modes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "bench.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "scenario: fault",
                        "mode: auto",
                        "verification:",
                        "  enabled: false",
                        "telemetry:",
                        "  output: results/out.telemetry.jsonl",
                        "  file_mode: write",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_config(config_path)

            self.assertFalse(config.verification_enabled)
            self.assertEqual(config.telemetry_file_mode, "write")

    def test_load_config_rejects_invalid_telemetry_file_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "bench.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "scenario: fault",
                        "mode: auto",
                        "telemetry:",
                        "  file_mode: rotate",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "telemetry.file_mode"):
                load_config(config_path)

    def test_load_config_supports_report_and_monitoring_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "bench.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "scenario: fault",
                        "mode: auto",
                        "telemetry:",
                        "  output: results/out.telemetry.jsonl",
                        "  report:",
                        "    enabled: true",
                        "    output_dir: results/report-dir",
                        "    top_k: 17",
                        "    log_scale_charts: true",
                        "    export_svg: false",
                        "monitoring:",
                        "  enabled: true",
                        "  sample_interval_ms: 250",
                        "  include_host: true",
                        "  include_sandboxes: false",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_config(config_path)

            self.assertTrue(config.telemetry_report.enabled)
            self.assertEqual(config.telemetry_report.output_dir, (root / "results" / "report-dir").resolve())
            self.assertEqual(config.telemetry_report.top_k, 17)
            self.assertTrue(config.telemetry_report.log_scale_charts)
            self.assertFalse(config.telemetry_report.export_svg)
            self.assertTrue(config.monitoring.enabled)
            self.assertEqual(config.monitoring.sample_interval_ms, 250)
            self.assertTrue(config.monitoring.include_host)
            self.assertFalse(config.monitoring.include_sandboxes)

    def test_load_config_supports_rootfs_reuse_and_phase_merging_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "bench.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "scenario: fault",
                        "mode: auto",
                        "rootfs_reuse:",
                        "  enabled: false",
                        "phase_merging:",
                        "  setup_and_run: true",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_config(config_path)

        self.assertFalse(config.rootfs_reuse.enabled)
        self.assertTrue(config.phase_merging.setup_and_run)
        self.assertEqual(config.phase_merging.setup_and_run_executor_pool, "separate")

    def test_load_config_supports_shared_setup_run_executor_pool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "bench.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "scenario: fault",
                        "mode: auto",
                        "phase_merging:",
                        "  setup_and_run: true",
                        "  setup_and_run_executor_pool: shared",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_config(config_path)

        self.assertTrue(config.phase_merging.setup_and_run)
        self.assertEqual(config.phase_merging.setup_and_run_executor_pool, "shared")

    def test_load_config_rejects_unknown_setup_run_executor_pool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "bench.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "scenario: fault",
                        "mode: auto",
                        "phase_merging:",
                        "  setup_and_run_executor_pool: combined",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "setup_and_run_executor_pool"):
                load_config(config_path)

    def test_load_config_nested_telemetry_output_overrides_legacy_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "bench.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "scenario: fault",
                        "mode: auto",
                        "telemetry_output: results/legacy.telemetry.jsonl",
                        "telemetry:",
                        "  output: results/nested.telemetry.jsonl",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_config(config_path)

            self.assertEqual(
                config.telemetry_output,
                (root / "results" / "nested.telemetry.jsonl").resolve(),
            )

    def test_load_config_supports_executor_scheduler_and_llm_server_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "bench.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "scenario: fault",
                        "mode: auto",
                        "executor:",
                        "  checkpoint_workers: 7",
                        "  restore_workers: 9",
                        "  coordination_workers: 5",
                        "  composite_step_workers: 11",
                        "  checkpoint_queue_size: 123",
                        "  checkpoint_scheduling_policy: reactive",
                        "  reactive_checkpoint_urgent_quota: 6",
                        "  max_retries: 2",
                        "  retry_backoff_seconds: 0.25",
                        "scheduler:",
                        "  policy: no_checkpointing",
                        "  min_checkpoint_interval_seconds: 1.5",
                        "  force_checkpoint_after_seconds: 9.0",
                        "  require_change_signal: false",
                        "  checkpoint_full_baseline_on_first_checkpoint: false",
                        "  prefer_checkpoint_during_llm_request: false",
                        "  require_llm_request_for_checkpoint: true",
                        "  inspect_without_pause: true",
                        "llm_server:",
                        "  launch_mode: thread",
                        "host_inspector:",
                        "  launch_mode: thread",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_config(config_path)
            resolved_executor = config.executor.resolve(default_workers=3)

            self.assertEqual(config.executor.checkpoint_workers, 7)
            self.assertEqual(config.executor.restore_workers, 9)
            self.assertEqual(config.executor.coordination_workers, 5)
            self.assertEqual(config.executor.composite_step_workers, 11)
            self.assertEqual(config.executor.checkpoint_queue_size, 123)
            self.assertEqual(config.executor.checkpoint_scheduling_policy, "reactive")
            self.assertEqual(config.executor.reactive_checkpoint_urgent_quota, 6)
            self.assertEqual(config.executor.max_retries, 2)
            self.assertEqual(config.executor.retry_backoff_seconds, 0.25)
            self.assertEqual(resolved_executor.resolved_checkpoint_workers, 7)
            self.assertEqual(resolved_executor.resolved_restore_workers, 9)
            self.assertEqual(resolved_executor.resolved_coordination_workers, 5)
            self.assertEqual(resolved_executor.resolved_composite_step_workers, 11)
            self.assertEqual(resolved_executor.max_checkpoint_queue_size, 123)
            self.assertEqual(config.scheduler.policy, "no_checkpointing")
            self.assertEqual(resolved_executor.checkpoint_scheduling_policy, "reactive")
            self.assertEqual(resolved_executor.reactive_checkpoint_urgent_quota, 6)
            self.assertEqual(config.scheduler.min_checkpoint_interval_seconds, 1.5)
            self.assertEqual(config.scheduler.force_checkpoint_after_seconds, 9.0)
            self.assertFalse(config.scheduler.require_change_signal)
            self.assertFalse(config.scheduler.checkpoint_full_baseline_on_first_checkpoint)
            self.assertFalse(config.scheduler.prefer_checkpoint_during_llm_request)
            self.assertTrue(config.scheduler.require_llm_request_for_checkpoint)
            self.assertTrue(config.scheduler.inspect_without_pause)
            self.assertEqual(config.llm_server.launch_mode, "thread")
            self.assertEqual(config.host_inspector.launch_mode, "thread")

    def test_load_config_rejects_invalid_scheduler_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "bench.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "scenario: fault",
                        "mode: auto",
                        "scheduler:",
                        "  policy: not-a-policy",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "scheduler.policy must be one of"):
                load_config(config_path)

    def test_resolve_task_records_merges_llm_service_options_without_overriding_dataset_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_path = root / "tasks.jsonl"
            dataset_path.write_text(
                "\n".join(
                    [
                        (
                            '{"agent_type":"iflow","llm_service_type":"iflow_trace_replay",'
                            '"task_description":{"prompt":"task-a"},"task_config":{},'
                            '"llm_service_config":{"trace_path":"trace-a.log","response_delay_policy":"fixed"}}'
                        )
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            config_path = root / "bench.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "scenario: fault",
                        "mode: auto",
                        "agent: iflow",
                        "llm_service: iflow_trace_replay",
                        "task_dataset: tasks.jsonl",
                        "llm_service_options:",
                        "  response_delay_policy: trace_replay",
                        "  response_delay_scaling_factor: 0.5",
                        "  response_delay_ms: 77",
                        "  minimal_delay: 12",
                        "  maximal_delay: 345",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_config(config_path)
            records = resolve_task_records(
                config,
                default_task_description=TaskDescription("ignored"),
                default_task_config=TaskConfig(),
            )

        self.assertEqual(
            config.llm_service_options,
            {
                "response_delay_policy": "trace_replay",
                "response_delay_scaling_factor": 0.5,
                "response_delay_ms": 77,
                "minimal_delay": 12,
                "maximal_delay": 345,
            },
        )
        self.assertEqual(
            records[0].llm_service_config,
            {
                "trace_path": str((root / "trace-a.log").resolve()),
                "response_delay_policy": "fixed",
                "response_delay_scaling_factor": 0.5,
                "response_delay_ms": 77,
                "minimal_delay": 12,
                "maximal_delay": 345,
            },
        )

    def test_load_config_parses_timeout_scales(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "bench.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "scenario: fault",
                        "mode: auto",
                        "max_agent_timeout_scale: 2.5",
                        "max_test_timeout_scale: 0.5",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_config(config_path)

        self.assertEqual(config.max_agent_timeout_scale, 2.5)
        self.assertEqual(config.max_test_timeout_scale, 0.5)

    def test_load_config_parses_clear_benchmark_root_after_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "bench.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "scenario: fault",
                        "mode: auto",
                        "clear_benchmark_root_after_run: true",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_config(config_path)

        self.assertTrue(config.clear_benchmark_root_after_run)

    def test_resolve_task_records_scales_task_timeouts_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_path = root / "tasks.jsonl"
            dataset_path.write_text(
                "\n".join(
                    [
                        (
                            '{"agent_type":"iflow","llm_service_type":"iflow_trace_replay",'
                            '"task_description":{"prompt":"task-a"},'
                            '"task_config":{"options":{"max_agent_timeout_sec":120,"max_test_timeout_sec":30}}}'
                        )
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            config_path = root / "bench.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "scenario: fault",
                        "mode: auto",
                        "agent: iflow",
                        "llm_service: iflow_trace_replay",
                        "task_dataset: tasks.jsonl",
                        "max_agent_timeout_scale: 2.0",
                        "max_test_timeout_scale: 0.5",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_config(config_path)
            records = resolve_task_records(
                config,
                default_task_description=TaskDescription("ignored"),
                default_task_config=TaskConfig(),
            )

        self.assertEqual(records[0].task_config.options["max_agent_timeout_sec"], 240.0)
        self.assertEqual(records[0].task_config.options["max_test_timeout_sec"], 15.0)

    def test_resolve_task_records_per_task_scale_overrides_take_precedence(self) -> None:
        """Per-task scale overrides shadow the global scale. Used to bump
        max_test_timeout_sec for tasks whose recorded budget was
        calibrated against an uncapped baseline (e.g. pytorch-model-cli's
        180s budget that consistently exceeds on our 4-CPU cgroup)."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_path = root / "tasks.jsonl"
            dataset_path.write_text(
                "\n".join(
                    [
                        # task-keep-default: not overridden → uses global scale
                        '{"agent_type":"iflow","llm_service_type":"iflow_trace_replay",'
                        '"task_description":{"prompt":"a"},'
                        '"task_config":{"options":{"task_id":"task-keep-default",'
                        '"max_agent_timeout_sec":100,"max_test_timeout_sec":40}}}',
                        # task-bump: agent and test scales both overridden
                        '{"agent_type":"iflow","llm_service_type":"iflow_trace_replay",'
                        '"task_description":{"prompt":"b"},'
                        '"task_config":{"options":{"task_id":"task-bump",'
                        '"max_agent_timeout_sec":100,"max_test_timeout_sec":40}}}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            config_path = root / "bench.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "scenario: fault",
                        "mode: auto",
                        "agent: iflow",
                        "llm_service: iflow_trace_replay",
                        "task_dataset: tasks.jsonl",
                        "sandboxes: 2",
                        "max_agent_timeout_scale: 2.0",
                        "max_test_timeout_scale: 1.0",
                        "max_agent_timeout_scale_overrides:",
                        "  task-bump: 5.0",
                        "max_test_timeout_scale_overrides:",
                        "  task-bump: 4.0",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_config(config_path)
            records = resolve_task_records(
                config,
                default_task_description=TaskDescription("ignored"),
                default_task_config=TaskConfig(),
            )

        # task-keep-default uses the global 2.0 / 1.0
        self.assertEqual(records[0].task_config.options["task_id"], "task-keep-default")
        self.assertEqual(records[0].task_config.options["max_agent_timeout_sec"], 200.0)
        self.assertEqual(records[0].task_config.options["max_test_timeout_sec"], 40.0)
        # task-bump uses the override 5.0 / 4.0
        self.assertEqual(records[1].task_config.options["task_id"], "task-bump")
        self.assertEqual(records[1].task_config.options["max_agent_timeout_sec"], 500.0)
        self.assertEqual(records[1].task_config.options["max_test_timeout_sec"], 160.0)

    def test_resolve_task_records_leaves_missing_timeout_values_unset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_path = root / "tasks.jsonl"
            dataset_path.write_text(
                "\n".join(
                    [
                        (
                            '{"agent_type":"iflow","llm_service_type":"iflow_trace_replay",'
                            '"task_description":{"prompt":"task-a"},'
                            '"task_config":{"options":{"custom_timeout":42}}}'
                        )
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            config_path = root / "bench.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "scenario: fault",
                        "mode: auto",
                        "agent: iflow",
                        "llm_service: iflow_trace_replay",
                        "task_dataset: tasks.jsonl",
                        "max_agent_timeout_scale: 2.0",
                        "max_test_timeout_scale: 0.5",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_config(config_path)
            records = resolve_task_records(
                config,
                default_task_description=TaskDescription("ignored"),
                default_task_config=TaskConfig(),
            )

        self.assertNotIn("max_agent_timeout_sec", records[0].task_config.options)
        self.assertNotIn("max_test_timeout_sec", records[0].task_config.options)
        self.assertEqual(records[0].task_config.options["custom_timeout"], 42)

    def test_load_config_rejects_invalid_log_file_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "bench.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "scenario: fault",
                        "mode: auto",
                        "log_file_mode: rotate",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "log_file_mode"):
                load_config(config_path)

    def test_load_config_rejects_non_positive_max_workers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "bench.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "scenario: fault",
                        "mode: auto",
                        "max_workers: 0",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "max_workers"):
                load_config(config_path)

    def test_load_config_parses_phase_workers_with_fallbacks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "bench.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "scenario: fault",
                        "mode: auto",
                        "sandboxes: 8",
                        "max_workers: 4",
                        "phase_workers:",
                        "  setup: 2",
                        "  run: 3",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_config(config_path)

        self.assertEqual(config.effective_phase_workers.as_dict(), {"setup": 2, "run": 3, "verification": 4})

    def test_load_config_rejects_unknown_phase_worker_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "bench.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "scenario: fault",
                        "mode: auto",
                        "phase_workers:",
                        "  setup: 1",
                        "  deploy: 2",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "unknown phases"):
                load_config(config_path)

    def test_load_config_rejects_non_positive_phase_workers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "bench.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "scenario: fault",
                        "mode: auto",
                        "phase_workers:",
                        "  verification: 0",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "phase_workers.verification"):
                load_config(config_path)

    def test_load_config_rejects_legacy_build_and_prepare_phase_worker_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "bench.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "scenario: fault",
                        "mode: auto",
                        "phase_workers:",
                        "  build: 2",
                        "  prepare: 2",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "unknown phases"):
                load_config(config_path)

    def test_load_config_rejects_invalid_scenario_mode_pair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "bench.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "scenario: e2e",
                        "mode: auto",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "does not support"):
                load_config(config_path)


class BenchmarkRunDispatchTests(unittest.TestCase):
    def _config(self, mode: str) -> BenchmarkConfig:
        return BenchmarkConfig(
            config_path=Path("/tmp/bench.yaml"),
            scenario="fake",
            mode=mode,
            provider="openai",
            agent="simulated",
            llm_service="simulated",
            task_dataset=None,
            sandboxes=1,
            iterations=1,
            output=None,
            telemetry_output=None,
            telemetry_detail_level="basic",
            telemetry_capture_command_output=False,
            telemetry_max_text_attribute_bytes=2048,
            log_file=None,
            log_file_mode="append",
            benchmark_root=None,
            zpool_size="10G",
            zpool_name=None,
            zpool_image=None,
            reuse_zpool=False,
            image_cache_root=None,
            log_level="info",
            transfer_delay_ms=0.0,
            work_dir_host_root=None,
            scenario_options={},
        )

    def test_run_benchmark_config_dispatches_selected_mode(self) -> None:
        calls: list[str] = []

        class _HarnessContext:
            def __init__(self, **kwargs) -> None:
                self.kwargs = kwargs

            def __enter__(self):
                return {"kind": "fake-harness", "kwargs": self.kwargs}

            def __exit__(self, exc_type, exc, tb) -> None:
                _ = (exc_type, exc, tb)

        scenario = ScenarioDefinition(
            name="fake",
            supported_modes=frozenset({"manual", "auto"}),
            build_harness_settings=lambda config: HarnessSettings(
                scheduler_config={"mode": config.mode},
                scheduler_policy=None,
                checkpoint_manager_factory=lambda base: base,
                max_workers=1,
            ),
            run_manual=lambda config, harness: calls.append(f"manual:{config.mode}:{harness['kind']}") or [],
            run_auto=lambda config, harness: calls.append(f"auto:{config.mode}:{harness['kind']}") or [],
            summarize=lambda config, rows: {"success_ratio": float(len(rows))},
        )

        with patch("benchmarks.run.RealHostScenarioHarness", _HarnessContext), patch.dict(
            "benchmarks.run.SCENARIOS",
            {"fake": scenario},
            clear=True,
        ):
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                run_benchmark_config(self._config("manual"))
                run_benchmark_config(self._config("auto"))

        self.assertEqual(calls, ["manual:manual:fake-harness", "auto:auto:fake-harness"])
        self.assertIn("success_ratio_avg: 0.000", buffer.getvalue())

    def test_run_benchmark_config_defaults_telemetry_output_path(self) -> None:
        calls: list[dict[str, object]] = []

        class _HarnessContext:
            def __init__(self, **kwargs) -> None:
                calls.append(kwargs)

            def __enter__(self):
                return {"kind": "fake-harness"}

            def __exit__(self, exc_type, exc, tb) -> None:
                _ = (exc_type, exc, tb)

        scenario = ScenarioDefinition(
            name="fake",
            supported_modes=frozenset({"manual"}),
            build_harness_settings=lambda config: HarnessSettings(
                scheduler_config={"mode": config.mode},
                scheduler_policy=None,
                checkpoint_manager_factory=lambda base: base,
                max_workers=1,
            ),
            run_manual=lambda config, harness: [],
            run_auto=None,
            summarize=lambda config, rows: {},
        )
        config = BenchmarkConfig(
            config_path=Path("/tmp/bench.yaml"),
            scenario="fake",
            mode="manual",
            provider="openai",
            agent="simulated",
            llm_service="simulated",
            output=Path("/tmp/out.csv"),
            telemetry_output=None,
            log_file=Path("/tmp/out.log"),
            log_file_mode="append",
            benchmark_root_home=Path("/tmp/bench-root"),
            benchmark_run_name="named-run",
            zpool_size="10G",
            zpool_name=None,
            zpool_image=None,
            reuse_zpool=False,
            image_cache_root=None,
        )

        with patch("benchmarks.run.RealHostScenarioHarness", _HarnessContext), patch.dict(
            "benchmarks.run.SCENARIOS",
            {"fake": scenario},
            clear=True,
        ):
            run_benchmark_config(config)

        self.assertEqual(calls[0]["telemetry_output"], Path("/tmp/out.telemetry.jsonl"))
        self.assertEqual(calls[0]["telemetry_detail_level"], "basic")
        self.assertFalse(calls[0]["telemetry_capture_command_output"])
        self.assertEqual(calls[0]["telemetry_max_text_attribute_bytes"], 2048)
        self.assertEqual(calls[0]["benchmark_root_home"], Path("/tmp/bench-root"))
        self.assertEqual(calls[0]["benchmark_run_name"], "named-run")

    def test_run_benchmark_config_clears_explicit_benchmark_run_root_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            benchmark_run_root = root / "bench-root" / "20260330_000000"
            benchmark_run_root.mkdir(parents=True, exist_ok=True)

            class _HarnessContext:
                def __init__(self, **kwargs) -> None:
                    self.kwargs = kwargs
                    self.root = benchmark_run_root
                    self.uses_temporary_root = False

                def __enter__(self):
                    return {"kind": "fake-harness"}

                def __exit__(self, exc_type, exc, tb) -> None:
                    _ = (exc_type, exc, tb)

            scenario = ScenarioDefinition(
                name="fake",
                supported_modes=frozenset({"manual"}),
                build_harness_settings=lambda config: HarnessSettings(
                    scheduler_config={"mode": config.mode},
                    scheduler_policy=None,
                    checkpoint_manager_factory=lambda base: base,
                    max_workers=1,
                ),
                run_manual=lambda config, harness: [],
                run_auto=None,
                summarize=lambda config, rows: {},
            )
            config = BenchmarkConfig(
                config_path=root / "bench.yaml",
                scenario="fake",
                mode="manual",
                provider="openai",
                agent="simulated",
                llm_service="simulated",
                benchmark_root=root / "bench-root",
                clear_benchmark_root_after_run=True,
            )

            with patch("benchmarks.run.RealHostScenarioHarness", _HarnessContext), patch.dict(
                "benchmarks.run.SCENARIOS",
                {"fake": scenario},
                clear=True,
            ):
                run_benchmark_config(config)

            self.assertFalse(benchmark_run_root.exists())
            self.assertTrue((root / "bench-root").exists())

    def test_run_benchmark_config_leaves_tmpdir_backed_benchmark_root_alone(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            benchmark_run_root = root / "crab_scenario_bench_tmpdir"
            benchmark_run_root.mkdir(parents=True, exist_ok=True)

            class _HarnessContext:
                def __init__(self, **kwargs) -> None:
                    self.kwargs = kwargs
                    self.root = benchmark_run_root
                    self.uses_temporary_root = True

                def __enter__(self):
                    return {"kind": "fake-harness"}

                def __exit__(self, exc_type, exc, tb) -> None:
                    _ = (exc_type, exc, tb)

            scenario = ScenarioDefinition(
                name="fake",
                supported_modes=frozenset({"manual"}),
                build_harness_settings=lambda config: HarnessSettings(
                    scheduler_config={"mode": config.mode},
                    scheduler_policy=None,
                    checkpoint_manager_factory=lambda base: base,
                    max_workers=1,
                ),
                run_manual=lambda config, harness: [],
                run_auto=None,
                summarize=lambda config, rows: {},
            )
            config = BenchmarkConfig(
                config_path=root / "bench.yaml",
                scenario="fake",
                mode="manual",
                provider="openai",
                agent="simulated",
                llm_service="simulated",
                clear_benchmark_root_after_run=True,
            )

            with patch("benchmarks.run.RealHostScenarioHarness", _HarnessContext), patch.dict(
                "benchmarks.run.SCENARIOS",
                {"fake": scenario},
                clear=True,
            ):
                run_benchmark_config(config)

            self.assertTrue(benchmark_run_root.exists())

    def test_run_benchmark_config_passes_telemetry_yaml_settings_to_harness(self) -> None:
        calls: list[dict[str, object]] = []

        class _HarnessContext:
            def __init__(self, **kwargs) -> None:
                calls.append(kwargs)

            def __enter__(self):
                return {"kind": "fake-harness"}

            def __exit__(self, exc_type, exc, tb) -> None:
                _ = (exc_type, exc, tb)

        scenario = ScenarioDefinition(
            name="fake",
            supported_modes=frozenset({"manual"}),
            build_harness_settings=lambda config: HarnessSettings(
                scheduler_config={"mode": config.mode},
                scheduler_policy=None,
                checkpoint_manager_factory=lambda base: base,
                max_workers=1,
            ),
            run_manual=lambda config, harness: [],
            run_auto=None,
            summarize=lambda config, rows: {},
        )
        config = BenchmarkConfig(
            config_path=Path("/tmp/bench.yaml"),
            scenario="fake",
            mode="manual",
            provider="openai",
            agent="simulated",
            llm_service="simulated",
            output=None,
            telemetry_output=Path("/tmp/out.telemetry.jsonl"),
            telemetry_detail_level="detailed",
            telemetry_capture_command_output=True,
            telemetry_max_text_attribute_bytes=512,
            log_file=None,
            log_file_mode="append",
            benchmark_root=None,
            zpool_size="10G",
            zpool_name=None,
            zpool_image=None,
            reuse_zpool=False,
            image_cache_root=None,
        )

        with patch("benchmarks.run.RealHostScenarioHarness", _HarnessContext), patch.dict(
            "benchmarks.run.SCENARIOS",
            {"fake": scenario},
            clear=True,
        ):
            run_benchmark_config(config)

        self.assertEqual(calls[0]["telemetry_detail_level"], "detailed")
        self.assertTrue(calls[0]["telemetry_capture_command_output"])
        self.assertEqual(calls[0]["telemetry_max_text_attribute_bytes"], 512)

    def test_run_benchmark_config_passes_expected_sandboxes_to_harness(self) -> None:
        calls: list[dict[str, object]] = []

        class _HarnessContext:
            def __init__(self, **kwargs) -> None:
                calls.append(kwargs)

            def __enter__(self):
                return {"kind": "fake-harness"}

            def __exit__(self, exc_type, exc, tb) -> None:
                _ = (exc_type, exc, tb)

        scenario = ScenarioDefinition(
            name="fake",
            supported_modes=frozenset({"manual"}),
            build_harness_settings=lambda config: HarnessSettings(
                scheduler_config={"mode": config.mode},
                scheduler_policy=None,
                checkpoint_manager_factory=lambda base: base,
                max_workers=1,
            ),
            run_manual=lambda config, harness: [],
            run_auto=None,
            summarize=lambda config, rows: {},
        )
        config = BenchmarkConfig(
            config_path=Path("/tmp/bench.yaml"),
            scenario="fake",
            mode="manual",
            provider="openai",
            agent="simulated",
            llm_service="simulated",
            sandboxes=640,
        )

        with patch("benchmarks.run.RealHostScenarioHarness", _HarnessContext), patch.dict(
            "benchmarks.run.SCENARIOS",
            {"fake": scenario},
            clear=True,
        ):
            run_benchmark_config(config)

        self.assertEqual(calls[0]["expected_sandboxes"], 640)

    def test_run_benchmark_config_uses_scenario_expected_sandboxes_override(self) -> None:
        calls: list[dict[str, object]] = []

        class _HarnessContext:
            def __init__(self, **kwargs) -> None:
                calls.append(kwargs)

            def __enter__(self):
                return {"kind": "fake-harness"}

            def __exit__(self, exc_type, exc, tb) -> None:
                _ = (exc_type, exc, tb)

        scenario = ScenarioDefinition(
            name="fake",
            supported_modes=frozenset({"manual"}),
            build_harness_settings=lambda config: HarnessSettings(
                scheduler_config={"mode": config.mode},
                scheduler_policy=None,
                checkpoint_manager_factory=lambda base: base,
                max_workers=1,
                expected_sandboxes=config.sandboxes * 2,
            ),
            run_manual=lambda config, harness: [],
            run_auto=None,
            summarize=lambda config, rows: {},
        )
        config = BenchmarkConfig(
            config_path=Path("/tmp/bench.yaml"),
            scenario="fake",
            mode="manual",
            provider="openai",
            agent="simulated",
            llm_service="simulated",
            sandboxes=128,
        )

        with patch("benchmarks.run.RealHostScenarioHarness", _HarnessContext), patch.dict(
            "benchmarks.run.SCENARIOS",
            {"fake": scenario},
            clear=True,
        ):
            run_benchmark_config(config)

        self.assertEqual(calls[0]["expected_sandboxes"], 256)

    def test_run_benchmark_config_passes_rootfs_reuse_setting_to_harness(self) -> None:
        calls: list[dict[str, object]] = []

        class _HarnessContext:
            def __init__(self, **kwargs) -> None:
                calls.append(kwargs)

            def __enter__(self):
                return {"kind": "fake-harness"}

            def __exit__(self, exc_type, exc, tb) -> None:
                _ = (exc_type, exc, tb)

        scenario = ScenarioDefinition(
            name="fake",
            supported_modes=frozenset({"manual"}),
            build_harness_settings=lambda config: HarnessSettings(
                scheduler_config={"mode": config.mode},
                scheduler_policy=None,
                checkpoint_manager_factory=lambda base: base,
                max_workers=1,
            ),
            run_manual=lambda config, harness: [],
            run_auto=None,
            summarize=lambda config, rows: {},
        )
        config = BenchmarkConfig(
            config_path=Path("/tmp/bench.yaml"),
            scenario="fake",
            mode="manual",
            provider="openai",
            agent="simulated",
            llm_service="simulated",
            rootfs_reuse=BenchmarkRootfsReuseConfig(enabled=False),
            phase_merging=BenchmarkPhaseMergingConfig(setup_and_run=True),
        )

        with patch("benchmarks.run.RealHostScenarioHarness", _HarnessContext), patch.dict(
            "benchmarks.run.SCENARIOS",
            {"fake": scenario},
            clear=True,
        ):
            run_benchmark_config(config)

        self.assertFalse(calls[0]["rootfs_reuse_enabled"])
    def test_run_benchmark_config_write_mode_clears_existing_telemetry_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            telemetry_output = root / "results" / "out.telemetry.jsonl"
            telemetry_output.parent.mkdir(parents=True, exist_ok=True)
            telemetry_output.write_text("stale\n", encoding="utf-8")

            class _HarnessContext:
                def __init__(self, **kwargs) -> None:
                    self.kwargs = kwargs

                def __enter__(self):
                    telemetry_path = self.kwargs["telemetry_output"]
                    assert isinstance(telemetry_path, Path)
                    telemetry_path.write_text("fresh\n", encoding="utf-8")
                    return {"kind": "fake-harness"}

                def __exit__(self, exc_type, exc, tb) -> None:
                    _ = (exc_type, exc, tb)

            scenario = ScenarioDefinition(
                name="fake",
                supported_modes=frozenset({"manual"}),
                build_harness_settings=lambda config: HarnessSettings(
                    scheduler_config={"mode": config.mode},
                    scheduler_policy=None,
                    checkpoint_manager_factory=lambda base: base,
                    max_workers=1,
                ),
                run_manual=lambda config, harness: [],
                run_auto=None,
                summarize=lambda config, rows: {},
            )
            config = BenchmarkConfig(
                config_path=root / "bench.yaml",
                scenario="fake",
                mode="manual",
                provider="openai",
                agent="simulated",
                llm_service="simulated",
                telemetry_output=telemetry_output,
                telemetry_file_mode="write",
                telemetry_report=TelemetryReportConfig(enabled=False),
            )

            with patch("benchmarks.run.RealHostScenarioHarness", _HarnessContext), patch.dict(
                "benchmarks.run.SCENARIOS",
                {"fake": scenario},
                clear=True,
            ):
                run_benchmark_config(config)

            self.assertEqual(telemetry_output.read_text(encoding="utf-8"), "fresh\n")

    def test_run_benchmark_config_passes_storage_plane_roots_to_harness(self) -> None:
        calls: list[dict[str, object]] = []

        class _HarnessContext:
            def __init__(self, **kwargs) -> None:
                calls.append(kwargs)

            def __enter__(self):
                return {"kind": "fake-harness"}

            def __exit__(self, exc_type, exc, tb) -> None:
                _ = (exc_type, exc, tb)

        scenario = ScenarioDefinition(
            name="fake",
            supported_modes=frozenset({"manual"}),
            build_harness_settings=lambda config: HarnessSettings(
                scheduler_config={"mode": config.mode},
                scheduler_policy=None,
                checkpoint_manager_factory=lambda base: base,
                max_workers=1,
            ),
            run_manual=lambda config, harness: [],
            run_auto=None,
            summarize=lambda config, rows: {},
        )
        config = BenchmarkConfig(
            config_path=Path("/tmp/bench.yaml"),
            scenario="fake",
            mode="manual",
            provider="openai",
            agent="simulated",
            llm_service="simulated",
            benchmark_root=Path("/tmp/bench-root"),
            zpool_size="10G",
            storage_planes=BenchmarkStoragePlanesConfig(
                runtime_root=Path("/tmp/runtime-plane"),
                storage_root=Path("/tmp/storage-plane"),
                agent_host_root=Path("/tmp/agent-plane"),
            ),
        )

        with patch("benchmarks.run.RealHostScenarioHarness", _HarnessContext), patch.dict(
            "benchmarks.run.SCENARIOS",
            {"fake": scenario},
            clear=True,
        ):
            run_benchmark_config(config)

        self.assertEqual(calls[0]["runtime_root"], Path("/tmp/runtime-plane"))
        self.assertEqual(calls[0]["storage_root"], Path("/tmp/storage-plane"))
        self.assertEqual(calls[0]["agent_host_root"], Path("/tmp/agent-plane"))

    def test_run_benchmark_config_passes_monitoring_settings_to_harness(self) -> None:
        calls: list[dict[str, object]] = []

        class _HarnessContext:
            def __init__(self, **kwargs) -> None:
                calls.append(kwargs)

            def __enter__(self):
                return {"kind": "fake-harness"}

            def __exit__(self, exc_type, exc, tb) -> None:
                _ = (exc_type, exc, tb)

        scenario = ScenarioDefinition(
            name="fake",
            supported_modes=frozenset({"manual"}),
            build_harness_settings=lambda config: HarnessSettings(
                scheduler_config={"mode": config.mode},
                scheduler_policy=None,
                checkpoint_manager_factory=lambda base: base,
                max_workers=1,
            ),
            run_manual=lambda config, harness: [],
            run_auto=None,
            summarize=lambda config, rows: {},
        )
        config = self._config("manual")
        config = config.__class__(**{**config.__dict__, "scenario": "fake"})

        with patch("benchmarks.run.RealHostScenarioHarness", _HarnessContext), patch.dict(
            "benchmarks.run.SCENARIOS",
            {"fake": scenario},
            clear=True,
        ):
            run_benchmark_config(config)

        self.assertTrue(calls[0]["monitoring_enabled"])
        self.assertEqual(calls[0]["monitoring_sample_interval_ms"], 1000)
        self.assertTrue(calls[0]["monitoring_include_host"])
        self.assertTrue(calls[0]["monitoring_include_sandboxes"])

    def test_run_benchmark_config_auto_generates_report_bundle(self) -> None:
        calls: list[dict[str, object]] = []
        report_calls: list[dict[str, object]] = []

        class _HarnessContext:
            def __init__(self, **kwargs) -> None:
                calls.append(kwargs)

            def __enter__(self):
                telemetry_output = calls[-1]["telemetry_output"]
                assert isinstance(telemetry_output, Path)
                telemetry_output.parent.mkdir(parents=True, exist_ok=True)
                telemetry_output.write_text(
                    json.dumps(
                        {
                            "timestamp": "2026-03-24T00:00:00+08:00",
                            "kind": "metric",
                            "name": "benchmark.task.duration_ms",
                            "value": 10.0,
                            "attributes": {
                                "run_id": "run-x",
                                "sandbox_id": "sbx-1",
                                "task_id": "task-1",
                                "component": "benchmark",
                            },
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                return {"kind": "fake-harness"}

            def __exit__(self, exc_type, exc, tb) -> None:
                _ = (exc_type, exc, tb)

        scenario = ScenarioDefinition(
            name="fake",
            supported_modes=frozenset({"manual"}),
            build_harness_settings=lambda config: HarnessSettings(
                scheduler_config={"mode": config.mode},
                scheduler_policy=None,
                checkpoint_manager_factory=lambda base: base,
                max_workers=1,
            ),
            run_manual=lambda config, harness: [],
            run_auto=None,
            summarize=lambda config, rows: {},
        )
        config = self._config("manual")
        config = config.__class__(**{**config.__dict__, "scenario": "fake", "telemetry_output": Path("/tmp/auto.telemetry.jsonl")})

        def _fake_report_bundle(*args, **kwargs):
            report_calls.append({"args": args, "kwargs": kwargs})
            return None

        with patch("benchmarks.run.RealHostScenarioHarness", _HarnessContext), patch.dict(
            "benchmarks.run.SCENARIOS",
            {"fake": scenario},
            clear=True,
        ), patch("benchmarks.telemetry_analysis.generate_report_bundle", side_effect=_fake_report_bundle):
            run_benchmark_config(config)

        self.assertEqual(len(report_calls), 1)
        self.assertEqual(report_calls[0]["args"][0], Path("/tmp/auto.telemetry.jsonl"))
        self.assertEqual(report_calls[0]["kwargs"]["output_dir"], Path("/tmp/auto.telemetry.report"))

    def test_run_benchmark_config_replicates_configured_artifacts_to_benchmark_run_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "results" / "out.csv"
            log_file = root / "results" / "out.log"
            telemetry_output = root / "results" / "out.telemetry.jsonl"
            report_output_dir = root / "results" / "out.report"
            benchmark_run_root = root / "benchmark-runs" / "run-a"

            class _HarnessContext:
                def __init__(self, **kwargs) -> None:
                    self.kwargs = kwargs
                    self.root = benchmark_run_root
                    self.uses_temporary_root = False

                def __enter__(self):
                    telemetry_path = self.kwargs["telemetry_output"]
                    assert isinstance(telemetry_path, Path)
                    telemetry_path.parent.mkdir(parents=True, exist_ok=True)
                    telemetry_path.write_text(
                        json.dumps(
                            {
                                "timestamp": "2026-03-24T00:00:00+08:00",
                                "kind": "metric",
                                "name": "benchmark.task.duration_ms",
                                "value": 10.0,
                                "attributes": {"run_id": "run-x"},
                            }
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    return {"kind": "fake-harness"}

                def __exit__(self, exc_type, exc, tb) -> None:
                    _ = (exc_type, exc, tb)

            scenario = ScenarioDefinition(
                name="fake",
                supported_modes=frozenset({"manual"}),
                build_harness_settings=lambda config: HarnessSettings(
                    scheduler_config={"mode": config.mode},
                    scheduler_policy=None,
                    checkpoint_manager_factory=lambda base: base,
                    max_workers=1,
                ),
                run_manual=lambda config, harness: [{"success_ratio": 1.0, "task_id": "task-1"}],
                run_auto=None,
                summarize=lambda config, rows: {"success_ratio": 1.0},
            )
            config = BenchmarkConfig(
                config_path=root / "bench.yaml",
                scenario="fake",
                mode="manual",
                provider="openai",
                agent="simulated",
                llm_service="simulated",
                output=output,
                telemetry_output=telemetry_output,
                telemetry_report=TelemetryReportConfig(enabled=True, output_dir=report_output_dir),
                log_file=log_file,
                log_file_mode="write",
                benchmark_root_home=root / "benchmark-runs",
                benchmark_run_name="run-a",
            )

            def _fake_report_bundle(*args, **kwargs):
                _ = args
                report_dir = kwargs["output_dir"]
                assert isinstance(report_dir, Path)
                report_dir.mkdir(parents=True, exist_ok=True)
                (report_dir / "index.html").write_text("report\n", encoding="utf-8")

            with patch("benchmarks.run.RealHostScenarioHarness", _HarnessContext), patch.dict(
                "benchmarks.run.SCENARIOS",
                {"fake": scenario},
                clear=True,
            ), patch("benchmarks.telemetry_analysis.generate_report_bundle", side_effect=_fake_report_bundle):
                run_benchmark_config(config)

            self.assertEqual(
                (benchmark_run_root / "out.csv").read_text(encoding="utf-8"),
                output.read_text(encoding="utf-8"),
            )
            self.assertEqual(
                (benchmark_run_root / "out.telemetry.jsonl").read_text(encoding="utf-8"),
                telemetry_output.read_text(encoding="utf-8"),
            )
            self.assertEqual(
                (benchmark_run_root / "out.report" / "index.html").read_text(encoding="utf-8"),
                "report\n",
            )
            replica_log = (benchmark_run_root / "out.log").read_text(encoding="utf-8")
            self.assertIn("benchmark.run start", replica_log)
            self.assertIn("benchmark.run end status=completed", replica_log)

    def test_run_benchmark_config_logs_run_markers_and_summary_to_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log_file = root / "benchmark.log"
            output = root / "out.csv"
            telemetry_output = root / "out.telemetry.jsonl"

            class _HarnessContext:
                def __init__(self, **kwargs) -> None:
                    self.kwargs = kwargs

                def __enter__(self):
                    return {"kind": "fake-harness"}

                def __exit__(self, exc_type, exc, tb) -> None:
                    _ = (exc_type, exc, tb)

            scenario = ScenarioDefinition(
                name="fake",
                supported_modes=frozenset({"manual"}),
                build_harness_settings=lambda config: HarnessSettings(
                    scheduler_config={"mode": config.mode},
                    scheduler_policy=None,
                    checkpoint_manager_factory=lambda base: base,
                    max_workers=1,
                ),
                run_manual=lambda config, harness: [{"success_ratio": 1.0}],
                run_auto=None,
                summarize=lambda config, rows: {"success_ratio": 1.0},
            )
            config = BenchmarkConfig(
                config_path=root / "bench.yaml",
                scenario="fake",
                mode="manual",
                provider="openai",
                agent="simulated",
                llm_service="simulated",
                output=output,
                telemetry_output=telemetry_output,
                log_file=log_file,
                log_file_mode="write",
                benchmark_root=root / "bench-root",
                zpool_size="10G",
                zpool_name=None,
                zpool_image=None,
                reuse_zpool=False,
                image_cache_root=None,
            )

            with patch("benchmarks.run.RealHostScenarioHarness", _HarnessContext), patch.dict(
                "benchmarks.run.SCENARIOS",
                {"fake": scenario},
                clear=True,
            ):
                buffer = io.StringIO()
                with redirect_stdout(buffer):
                    run_benchmark_config(config)

            log_text = log_file.read_text(encoding="utf-8")
            self.assertIn("benchmark.run start", log_text)
            self.assertIn("max_workers=1", log_text)
            self.assertIn("success_ratio_avg: 1.000", log_text)
            self.assertIn(f"output: {output.resolve()}", log_text)
            self.assertIn("benchmark.run end status=completed", log_text)

    def test_run_benchmark_config_emits_teardown_and_postprocess_progress(self) -> None:
        class _HarnessContext:
            def __init__(self, **kwargs) -> None:
                self.kwargs = kwargs

            def __enter__(self):
                return {"kind": "fake-harness"}

            def __exit__(self, exc_type, exc, tb) -> None:
                _ = (exc_type, exc, tb)

        scenario = ScenarioDefinition(
            name="fake",
            supported_modes=frozenset({"manual"}),
            build_harness_settings=lambda config: HarnessSettings(
                scheduler_config={"mode": config.mode},
                scheduler_policy=None,
                checkpoint_manager_factory=lambda base: base,
                max_workers=1,
            ),
            run_manual=lambda config, harness: [{"success_ratio": 1.0}],
            run_auto=None,
            summarize=lambda config, rows: {"success_ratio": 1.0},
        )
        config = self._config("manual").__class__(**{**self._config("manual").__dict__, "scenario": "fake"})

        with patch("benchmarks.run.RealHostScenarioHarness", _HarnessContext), patch.dict(
            "benchmarks.run.SCENARIOS",
            {"fake": scenario},
            clear=True,
        ):
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                run_benchmark_config(config)

        output = buffer.getvalue()
        self.assertIn("benchmark.phase.teardown start sandboxes=1 max_workers=1", output)
        self.assertIn("benchmark.phase.teardown end sandboxes=1 max_workers=1 duration_s=", output)
        self.assertIn("benchmark.phase.postprocess start sandboxes=1 max_workers=1", output)
        self.assertIn("benchmark.phase.postprocess end sandboxes=1 max_workers=1 duration_s=", output)

    def test_incomplete_replay_skips_spec_fork_reused_successors(self) -> None:
        """Spec fork-reused successors are named `<root>-spec-<n>` and
        inherit parent state via CRIU restore — their per-fork
        `replay_final_trace_cursor` starts at 0 even though the parent
        already drove the trace. They're not real timeouts and must not
        produce INCOMPLETE_REPLAY lines.
        """
        rows = [
            # Real primary timeout — should report.
            {"sandbox_id": "spec-0", "task_id": "t", "trace_response_count": 29, "replay_final_trace_cursor": 5},
            # Fork-reused successors — must be skipped.
            {"sandbox_id": "spec-0-spec-29", "task_id": "t", "trace_response_count": 29, "replay_final_trace_cursor": 0},
            {"sandbox_id": "spec-1-spec-24", "task_id": "u", "trace_response_count": 25, "replay_final_trace_cursor": 0},
            # Non-spec partial replay (e.g. fault-mode timeout) — should report.
            {"sandbox_id": "fault-7", "task_id": "v", "trace_response_count": 12, "replay_final_trace_cursor": 8},
        ]
        lines = _incomplete_replay_lines(rows)
        joined = "\n".join(lines)
        self.assertIn("sandbox=spec-0 ", joined)
        self.assertIn("sandbox=fault-7 ", joined)
        self.assertNotIn("spec-0-spec-29", joined)
        self.assertNotIn("spec-1-spec-24", joined)

    def test_example_yaml_files_load(self) -> None:
        examples_root = Path(__file__).resolve().parents[1] / "benchmarks" / "examples"
        examples = sorted(examples_root.rglob("*.yaml"))
        self.assertTrue(examples)
        for path in examples:
            with self.subTest(path=path.name):
                config = load_config(path)
                self.assertEqual(config.config_path, path.resolve())


if __name__ == "__main__":
    unittest.main()
