from __future__ import annotations

import json
import subprocess
import tempfile
import textwrap
import time
import unittest
from pathlib import Path

from benchmarks.diagnose import (
    diagnose_benchmark_config,
    render_run_diagnosis_html,
    render_run_diagnosis_markdown,
    render_run_diagnosis_text,
)
from benchmarks.diagnose.artifacts import infer_actual_benchmark_root, load_artifacts
from benchmarks.diagnose.csv_report import parse_csv_report
from benchmarks.diagnose.dataset import load_dataset_index
from benchmarks.diagnose.iflow_report import extract_trace_tool_calls, summarize_iflow_session
from benchmarks.diagnose.log_report import parse_log
from benchmarks.diagnose.models import DatasetTaskInfo
from benchmarks.diagnose.telemetry_report import parse_telemetry


class BenchmarkDiagnoseTests(unittest.TestCase):
    def _write(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def _write_jsonl(self, path: Path, rows: list[dict[str, object]]) -> None:
        self._write(path, "\n".join(json.dumps(row) for row in rows) + "\n")

    def _make_trace(self, path: Path) -> None:
        self._write_jsonl(
            path,
            [
                {
                    "type": "request",
                    "timestamp": 1.0,
                    "data": json.dumps(
                        {
                            "messages": [
                                {"role": "user", "content": "Build the thing"},
                            ]
                        }
                    ),
                },
                {
                    "type": "response",
                    "timestamp": 1.2,
                    "data": json.dumps(
                        {
                            "id": "resp-1",
                            "choices": [
                                {
                                    "message": {
                                        "role": "assistant",
                                        "tool_calls": [
                                            {
                                                "id": "call-1",
                                                "type": "function",
                                                "function": {
                                                    "name": "run_shell_command",
                                                    "arguments": json.dumps(
                                                        {
                                                            "command": "make all",
                                                            "description": "Build the compiler",
                                                        }
                                                    ),
                                                },
                                            }
                                        ],
                                    }
                                }
                            ],
                        }
                    ),
                },
                {
                    "type": "request",
                    "timestamp": 3.4,
                    "data": json.dumps(
                        {
                            "messages": [
                                {"role": "user", "content": "Build the thing"},
                                {"role": "assistant", "content": "Running build", "tool_calls": []},
                                {
                                    "role": "tool",
                                    "tool_call_id": "call-1",
                                    "content": (
                                        "Command: make all\n"
                                        "Stdout: build ok\n"
                                        "Stderr: (none)\n"
                                        "Error: (none)\n"
                                        "Exit Code: 0\n"
                                    ),
                                },
                            ]
                        }
                    ),
                },
                {
                    "type": "response",
                    "timestamp": 3.6,
                    "data": json.dumps(
                        {
                            "id": "resp-2",
                            "choices": [
                                {
                                    "message": {
                                        "role": "assistant",
                                        "content": "done",
                                    }
                                }
                            ],
                        }
                    ),
                },
            ],
        )

    def _make_session(
        self,
        path: Path,
        *,
        final_text: str = "Build completed successfully.",
        include_replay_marker: bool = False,
    ) -> None:
        rows = [
            {
                "timestamp": "2026-03-23T13:31:08.421Z",
                "type": "user",
                "message": {"role": "user", "content": "Please compile"},
            },
            {
                "timestamp": "2026-03-23T13:32:08.944Z",
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"text": "Running the build now."},
                        {
                            "functionCall": {
                                "name": "run_shell_command",
                                "args": {
                                    "command": "make all && ./ccomp -version",
                                    "description": "Build the compiler",
                                },
                            }
                        },
                    ],
                },
            },
            {
                "timestamp": "2026-03-23T13:32:10.264Z",
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "content": {
                                "functionResponse": {
                                    "name": "run_shell_command",
                                    "response": {
                                        "output": (
                                            "Command: make all && ./ccomp -version\n"
                                            "Stdout: \n"
                                            "Stderr: bash: warning: setlocale: LC_ALL: cannot change locale (zh_CN.UTF-8)\n"
                                            "/tmp/out missing\n"
                                            "Exit Code: 1\n"
                                        )
                                    },
                                }
                            },
                        }
                    ],
                },
                "toolUseResult": {"status": "success"},
            },
            {
                "timestamp": "2026-03-23T13:33:08.944Z",
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"text": final_text},
                    ],
                },
            },
        ]
        if include_replay_marker:
            rows[1]["trace_cursor"] = 3
            rows[1]["consumed_response_count"] = 2
        self._write_jsonl(path, rows)

    def _make_fixture(self, base: Path) -> dict[str, Path]:
        benchmark_base = base / "results-root"
        actual_root = benchmark_base / "20260323_212146"
        trace_dir = base / "traces"
        trace_a = trace_dir / "task-a.log"
        trace_b = trace_dir / "task-b.log"
        self._make_trace(trace_a)
        self._make_trace(trace_b)
        dataset_path = base / "dataset.jsonl"
        dataset_rows = [
            {
                "agent_type": "iflow",
                "llm_service_type": "iflow_trace_replay",
                "llm_service_config": {"trace_path": str(trace_a)},
                "task_description": {"prompt": "Compile task A"},
                "task_config": {"options": {"task_id": "task-a"}},
                "task_id": "task-a",
                "trace_response_count": 1,
                "trace_malformed_line_count": 0,
            },
            {
                "agent_type": "iflow",
                "llm_service_type": "iflow_trace_replay",
                "llm_service_config": {"trace_path": str(trace_b)},
                "task_description": {"prompt": "Compile task B"},
                "task_config": {"options": {"task_id": "task-b"}},
                "task_id": "task-b",
                "trace_response_count": 1,
                "trace_malformed_line_count": 0,
            },
        ]
        self._write_jsonl(dataset_path, dataset_rows)
        csv_path = base / "logs" / "run.csv"
        self._write(
            csv_path,
            textwrap.dedent(
                """\
                scenario,mode,provider,agent,llm_service,sandbox_id,task_id,iteration,success_ratio,task_error,event_type,verification_status,verification_exit_code,verification_stdout,verification_stderr
                fault,auto,openai,iflow,iflow_trace_replay,fault-54,task-a,1,0.0,,fault,failed,1,,all tests failed
                """
            ),
        )
        log_path = base / "logs" / "run.log"
        self._write(
            log_path,
            textwrap.dedent(
                f"""\
                2026-03-23 21:00:00,000 INFO benchmarks.real_host_scenario_base: runc --root {actual_root}/runtime-state state fault-54
                2026-03-23 21:32:10,000 INFO crab.scheduler: Scheduler selected checkpoint for sandbox fault-54 reason=llm_request_window_available observed_process_changed=False observed_filesystem_changed=False checkpoint_process=True checkpoint_filesystem=True leave_running=True
                2026-03-23 21:32:20,000 INFO crab.executor: Finished checkpoint job job-1 for sandbox fault-54 with status=succeeded checkpoint=ckpt-1
                2026-03-23 21:32:25,000 INFO benchmarks.real_host_scenario_base: Benchmark notifying fault sandbox=fault-54 reason=fault
                2026-03-23 21:32:35,000 INFO crab.system: Recovery restore succeeded sandbox=fault-54 checkpoint=ckpt-1
                2026-03-23 21:32:40,000 INFO benchmarks.real_host_scenario_base: Completed run-tests.sh sandbox=fault-54 exit_code=1 command=/bin/bash -lc 'bash /tests/run-tests.sh'
                2026-03-23 21:32:40,100 WARNING benchmarks.real_host_scenario_base: run-tests stderr sandbox=fault-54
                E: Version '1.2.3' for 'curl' was not found
                2026-03-23 21:32:40,200 WARNING benchmarks.real_host_scenario_base: binary missing for sandbox=fault-54 task_id=task-a
                """
            ),
        )
        telemetry_path = base / "logs" / "run.telemetry.jsonl"
        telemetry_rows = [
            {
                "timestamp": "2026-03-23T13:00:10+08:00",
                "kind": "event",
                "name": "benchmark.fault.injected",
                "attributes": {"sandbox_id": "fault-54", "task_id": "task-a", "event_type": "fault"},
            },
            {
                "timestamp": "2026-03-23T13:00:11+08:00",
                "kind": "event",
                "name": "checkpoint.flow.finish",
                "attributes": {"sandbox_id": "fault-54", "task_id": "task-a", "checkpoint_id": "ckpt-1"},
            },
            {
                "timestamp": "2026-03-23T13:00:12+08:00",
                "kind": "event",
                "name": "restore.flow.finish",
                "attributes": {"sandbox_id": "fault-54", "task_id": "task-a", "checkpoint_id": "ckpt-1"},
            },
            {
                "timestamp": "2026-03-23T13:00:13+08:00",
                "kind": "event",
                "name": "benchmark.task.verify.finish",
                "attributes": {"sandbox_id": "fault-54", "task_id": "task-a", "status": "failed"},
            },
            {
                "timestamp": "2026-03-23T13:00:14+08:00",
                "kind": "metric",
                "name": "sandbox.command_duration_ms",
                "value": 45000.0,
                "attributes": {
                    "sandbox_id": "fault-54",
                    "task_id": "task-a",
                    "component": "runtime",
                    "operation": "sandbox.exec",
                    "command": ["make", "all"],
                    "status": "failed",
                },
            },
            {
                "timestamp": "2026-03-23T13:00:14+08:00",
                "kind": "event",
                "name": "runtime.command.finish",
                "attributes": {
                    "sandbox_id": "fault-54",
                    "task_id": "task-a",
                    "component": "runtime",
                    "operation": "sandbox.exec",
                    "command": "make all && ./ccomp -version",
                    "status": "failed",
                },
            },
        ]
        self._write_jsonl(telemetry_path, telemetry_rows)
        session_dir = actual_root / "iflow" / "fault-54" / "iflow-state" / ".iflow" / "projects" / "-app"
        older_session = session_dir / "session-old.jsonl"
        newer_session = session_dir / "session-new.jsonl"
        self._make_session(older_session, final_text="Intermediate result.")
        time.sleep(0.01)
        self._make_session(newer_session, final_text="Build completed successfully.")
        config_path = base / "bench.yaml"
        self._write(
            config_path,
            textwrap.dedent(
                f"""\
                scenario: fault
                mode: auto
                provider: openai
                agent: iflow
                llm_service: iflow_trace_replay
                task_dataset: {dataset_path}
                sandboxes: 2
                output: {csv_path}
                log_file: {log_path}
                benchmark_root: {benchmark_base}
                telemetry:
                  output: {telemetry_path}
                  detail_level: detailed
                """
            ),
        )
        return {
            "config": config_path,
            "dataset": dataset_path,
            "log": log_path,
            "csv": csv_path,
            "telemetry": telemetry_path,
            "actual_root": actual_root,
            "session_dir": session_dir,
        }

    def test_load_artifacts_honors_nested_telemetry_and_inferrs_root(self) -> None:
        with tempfile.TemporaryDirectory(prefix="diagnose_fixture_") as tmp:
            paths = self._make_fixture(Path(tmp))
            loaded = load_artifacts(paths["config"])
            self.assertEqual(loaded.context.telemetry_path, paths["telemetry"].resolve())
            self.assertEqual(loaded.context.actual_benchmark_root, paths["actual_root"].resolve())
            self.assertIn(str(paths["actual_root"].resolve()), loaded.context.inferred_run_roots)

    def test_infer_actual_benchmark_root_from_log(self) -> None:
        with tempfile.TemporaryDirectory(prefix="diagnose_fixture_") as tmp:
            log_path = Path(tmp) / "run.log"
            self._write(
                log_path,
                "\n".join(
                    [
                        "2026-03-23 10:00:00,000 INFO x: runc --root /tmp/root-a/runtime-state state a",
                        "2026-03-23 10:00:00,100 INFO x: runc --root /tmp/root-a/runtime-state state b",
                        "2026-03-23 10:00:00,200 INFO x: runc --root /tmp/root-b/runtime-state state c",
                    ]
                ),
            )
            actual_root, seen = infer_actual_benchmark_root(log_path)
            self.assertEqual(actual_root, Path("/tmp/root-a"))
            self.assertEqual(seen[0], "/tmp/root-a")

    def test_csv_missing_detection_handles_duplicate_task_ids(self) -> None:
        with tempfile.TemporaryDirectory(prefix="diagnose_fixture_") as tmp:
            base = Path(tmp)
            dataset_path = base / "dataset.jsonl"
            trace = base / "trace.log"
            self._make_trace(trace)
            self._write_jsonl(
                dataset_path,
                [
                    {
                        "agent_type": "iflow",
                        "llm_service_type": "iflow_trace_replay",
                        "llm_service_config": {"trace_path": str(trace)},
                        "task_description": {"prompt": "same"},
                        "task_config": {"options": {"task_id": "same"}},
                        "task_id": "same",
                    },
                    {
                        "agent_type": "iflow",
                        "llm_service_type": "iflow_trace_replay",
                        "llm_service_config": {"trace_path": str(trace)},
                        "task_description": {"prompt": "same"},
                        "task_config": {"options": {"task_id": "same"}},
                        "task_id": "same",
                    },
                ],
            )
            csv_path = base / "out.csv"
            self._write(
                csv_path,
                "sandbox_id,task_id,success_ratio,verification_status\nfault-0,same,1.0,passed\n",
            )
            dataset_index = load_dataset_index(dataset_path)
            report = parse_csv_report(csv_path, dataset_index.tasks)
            self.assertEqual(len(report.missing_tasks), 1)
            self.assertEqual(report.missing_tasks[0].task_id, "same")
            self.assertEqual(report.missing_tasks[0].occurrences_expected, 2)
            self.assertEqual(report.missing_tasks[0].occurrences_observed, 1)

    def test_log_and_telemetry_parsers_extract_timelines(self) -> None:
        with tempfile.TemporaryDirectory(prefix="diagnose_fixture_") as tmp:
            paths = self._make_fixture(Path(tmp))
            parsed_log = parse_log(paths["log"])
            self.assertIn("fault-54", parsed_log.lines_by_sandbox)
            self.assertTrue(any(item.label == "fault injected" for item in parsed_log.timeline_by_sandbox["fault-54"]))
            key_event_labels = [item.label for item in parsed_log.key_events_by_sandbox["fault-54"]]
            self.assertIn("Checkpoint P+F Succeed", key_event_labels)
            self.assertIn("Fault Injection Succeed", key_event_labels)
            self.assertIn("Restore Succeed", key_event_labels)
            parsed_telemetry = parse_telemetry(paths["telemetry"])
            labels = [item.label for item in parsed_telemetry.timeline_by_sandbox["fault-54"]]
            self.assertIn("checkpoint", labels)
            self.assertIn("restore", labels)
            self.assertIn("verification", labels)
            self.assertTrue(parsed_telemetry.tool_calls_by_sandbox["fault-54"])

    def test_extract_trace_tool_calls_uses_trace_tool_result_and_duration(self) -> None:
        with tempfile.TemporaryDirectory(prefix="diagnose_fixture_") as tmp:
            trace_path = Path(tmp) / "trace.jsonl"
            self._make_trace(trace_path)
            tool_calls = extract_trace_tool_calls(
                DatasetTaskInfo(
                    dataset_index=0,
                    task_id="task-a",
                    agent_type="iflow",
                    llm_service_type="iflow_trace_replay",
                    trace_path=trace_path,
                    trace_response_count=2,
                    trace_malformed_line_count=0,
                    task_root=None,
                    service_name=None,
                    prompt_preview="Build the thing",
                    raw={},
                )
            )
            self.assertEqual(len(tool_calls), 1)
            self.assertEqual(tool_calls[0].exit_code, 0)
            self.assertEqual(tool_calls[0].result_summary, "ok")
            self.assertAlmostEqual(float(tool_calls[0].duration_ms or 0.0), 2200.0, delta=1.0)

    def test_iflow_session_summary_uses_newest_session_file(self) -> None:
        with tempfile.TemporaryDirectory(prefix="diagnose_fixture_") as tmp:
            paths = self._make_fixture(Path(tmp))
            session_summary, tool_calls = summarize_iflow_session(
                benchmark_root=paths["actual_root"],
                sandbox_id="fault-54",
            )
            self.assertTrue(session_summary["available"])
            self.assertTrue(session_summary["selected_newest_session"])
            self.assertTrue(str(session_summary["session_file"]).endswith("session-new.jsonl"))
            self.assertEqual(len(tool_calls), 1)
            self.assertEqual(tool_calls[0].exit_code, 1)
            self.assertIsNotNone(tool_calls[0].duration_ms)
            self.assertTrue(tool_calls[0].has_error_indicators)
            self.assertIn("nonzero exit", str(tool_calls[0].result_summary))
            self.assertNotIn("setlocale", str(tool_calls[0].raw_result_preview))
            self.assertEqual(session_summary["replay_marker_count"], 0)

    def test_iflow_session_summary_detects_replay_markers(self) -> None:
        with tempfile.TemporaryDirectory(prefix="diagnose_fixture_") as tmp:
            actual_root = Path(tmp) / "root"
            session_dir = actual_root / "iflow" / "fault-9" / "iflow-state" / ".iflow" / "projects" / "-app"
            self._make_session(session_dir / "session-new.jsonl", include_replay_marker=True)
            session_summary, _tool_calls = summarize_iflow_session(
                benchmark_root=actual_root,
                sandbox_id="fault-9",
            )
            self.assertGreaterEqual(int(session_summary["replay_marker_count"]), 2)
            self.assertTrue(any("trace_cursor" in path for path in session_summary["replay_marker_paths"]))

    def test_end_to_end_diagnosis_includes_failed_and_missing_tasks(self) -> None:
        with tempfile.TemporaryDirectory(prefix="diagnose_fixture_") as tmp:
            paths = self._make_fixture(Path(tmp))
            report = diagnose_benchmark_config(paths["config"])
            self.assertEqual(report.context.actual_benchmark_root, paths["actual_root"].resolve())
            self.assertIn("fault-54", report.sandboxes)
            self.assertIn("missing-task-1", report.sandboxes)
            self.assertEqual(report.sandboxes["fault-54"].task_id, "task-a")
            observed_key_event_labels = [
                event.label for event in report.sandboxes["fault-54"].tool_calls[0].observed_key_events
            ]
            self.assertEqual(
                observed_key_event_labels,
                ["Checkpoint P+F Succeed", "Fault Injection Succeed", "Restore Succeed"],
            )
            finding_titles = [finding.title for finding in report.sandboxes["fault-54"].findings]
            self.assertIn("Verifier setup/package failure", finding_titles)
            self.assertIn("Trace/session tool sequence is misaligned", finding_titles)
            self.assertEqual(report.sandboxes["fault-54"].tool_alignment_summary["trace_tool_call_count"], 1)
            markdown = render_run_diagnosis_markdown(
                report,
                max_visualized_tool_arg_chars=32,
                max_tool_comparison_rows=1,
                max_visualized_tool_rows=1,
            )
            text = render_run_diagnosis_text(
                report,
                max_visualized_tool_arg_chars=32,
                max_tool_comparison_rows=1,
                max_visualized_tool_rows=1,
            )
            html = render_run_diagnosis_html(
                report,
                max_visualized_tool_arg_chars=32,
                max_tool_comparison_rows=1,
                max_visualized_tool_rows=1,
            )
            self.assertIn("fault-54", markdown)
            self.assertIn("missing-task-1", markdown)
            self.assertIn("Dataset coverage is incomplete", text)
            self.assertIn("<table>", html)
            self.assertIn("Tool Alignment", html)
            self.assertIn("Trace ms", html)
            self.assertIn("Observed ms", html)
            self.assertIn("Observed Key Event", html)
            self.assertIn("Checkpoint P+F Succeed", html)
            self.assertIn("Trace Tool Calls</summary>", html)
            self.assertIn("Showing 1 high-signal comparison rows", html)

    def test_cli_writes_json_and_markdown_outputs(self) -> None:
        with tempfile.TemporaryDirectory(prefix="diagnose_fixture_") as tmp:
            paths = self._make_fixture(Path(tmp))
            json_out = Path(tmp) / "diagnosis.json"
            md_out = Path(tmp) / "diagnosis.md"
            html_out = Path(tmp) / "diagnosis.html"
            output_dir = Path(tmp) / "diag-out"
            result = subprocess.run(
                [
                    "python3",
                    "-m",
                    "benchmarks.diagnose",
                    "--config",
                    str(paths["config"]),
                    "--output-dir",
                    str(output_dir),
                    "--output-json",
                    str(json_out),
                    "--output-markdown",
                    str(md_out),
                    "--output-html",
                    str(html_out),
                    "--max-visualized-tool-arg-chars",
                    "32",
                    "--max-tool-comparison-rows",
                    "1",
                    "--max-visualized-tool-rows",
                    "1",
                ],
                check=False,
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertTrue(json_out.exists())
            self.assertTrue(md_out.exists())
            self.assertTrue(html_out.exists())
            self.assertTrue((output_dir / "diagnosis.txt").exists())
            self.assertTrue((output_dir / "diagnosis.md").exists())
            self.assertTrue((output_dir / "diagnosis.html").exists())
            self.assertTrue((output_dir / "diagnosis.json").exists())
            self.assertTrue((output_dir / "fault-54.log").exists())
            payload = json.loads(json_out.read_text(encoding="utf-8"))
            self.assertIn("context", payload)
            self.assertIn("failed_sandboxes", payload)
            self.assertIn("Benchmark Diagnosis", html_out.read_text(encoding="utf-8"))
            self.assertIn("Showing 1 high-signal comparison rows", html_out.read_text(encoding="utf-8"))
            self.assertIn("sandbox=fault-54", (output_dir / "fault-54.log").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
