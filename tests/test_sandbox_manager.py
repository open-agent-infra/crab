from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from crab import InMemoryTelemetrySink, RuncRuntime, RuncRuntimePaths, SandboxId
from crab.runtime import CommandRunner

RuncSandboxManager = RuncRuntime
RuncSandboxManagerPaths = RuncRuntimePaths


class FakeCommandRunner(CommandRunner):
    def __init__(self) -> None:
        self.commands: list[tuple[str, ...]] = []
        self.datasets: set[str] = set()
        self.snapshots: set[str] = set()

    def run(self, command: list[str], *, cwd: Path | None = None, timeout_seconds: float | None = None):
        _ = cwd
        _ = timeout_seconds
        self.commands.append(tuple(command))
        returncode = 0
        stdout = ""
        stderr = ""
        if command[:5] == ["zfs", "list", "-H", "-o", "name"]:
            name = command[-1]
            exists = name in self.datasets or name in self.snapshots
            returncode = 0 if exists else 1
            stderr = "" if exists else "dataset does not exist"
        elif command[:2] == ["zfs", "create"]:
            self.datasets.add(command[-1])
        elif command[:2] == ["zfs", "snapshot"]:
            self.snapshots.add(command[-1])
        elif command[:2] == ["zfs", "clone"]:
            self.datasets.add(command[-1])
        elif command[:3] == ["zfs", "destroy", "-r"]:
            target = command[-1]
            datasets = {name for name in self.datasets if name == target or name.startswith(f"{target}/")}
            snapshots = {
                name
                for name in self.snapshots
                if name.startswith(f"{target}@") or name.startswith(f"{target}/")
            }
            if not datasets and not snapshots:
                returncode = 1
                stderr = "dataset does not exist"
            self.datasets.difference_update(datasets)
            self.snapshots.difference_update(snapshots)
        return type(
            "Result",
            (),
            {"command": tuple(command), "returncode": returncode, "stdout": stdout, "stderr": stderr},
        )()


class ResumeRaceCommandRunner(CommandRunner):
    def __init__(self, *, state_status_after_resume: str) -> None:
        self.commands: list[tuple[str, ...]] = []
        self._state_status_after_resume = state_status_after_resume

    def run(self, command: list[str], *, cwd: Path | None = None, timeout_seconds: float | None = None):
        _ = cwd
        _ = timeout_seconds
        self.commands.append(tuple(command))
        if command[-2:] == ["resume", "sbx-test"]:
            return type(
                "Result",
                (),
                {
                    "command": tuple(command),
                    "returncode": 1,
                    "stdout": "",
                    "stderr": 'time="2026-03-19T22:23:26+08:00" level=error msg="container not paused"',
                },
            )()
        if command[-2:] == ["state", "sbx-test"]:
            payload = {"status": self._state_status_after_resume, "pid": 123 if self._state_status_after_resume == "running" else 0}
            return type(
                "Result",
                (),
                {"command": tuple(command), "returncode": 0, "stdout": json.dumps(payload), "stderr": ""},
            )()
        return type(
            "Result",
            (),
            {"command": tuple(command), "returncode": 0, "stdout": "", "stderr": ""},
        )()


class PauseNotRunningCommandRunner(CommandRunner):
    def __init__(self) -> None:
        self.commands: list[tuple[str, ...]] = []

    def run(self, command: list[str], *, cwd: Path | None = None, timeout_seconds: float | None = None):
        _ = cwd
        _ = timeout_seconds
        self.commands.append(tuple(command))
        if command[-2:] == ["pause", "sbx-test"]:
            return type(
                "Result",
                (),
                {
                    "command": tuple(command),
                    "returncode": 1,
                    "stdout": "",
                    "stderr": 'time="2026-03-28T17:17:43+08:00" level=error msg="container not running"',
                },
            )()
        return type(
            "Result",
            (),
            {"command": tuple(command), "returncode": 0, "stdout": "", "stderr": ""},
        )()


class FakeHostInspectorClient:
    def __init__(self) -> None:
        self.register_calls: list[tuple[SandboxId, str, str, object | None]] = []
        self.unregister_calls: list[SandboxId] = []

    def register_sandbox(
        self,
        sandbox_id: SandboxId,
        runtime: str,
        object_id: str,
        *,
        ignore_process_rules=None,
        ignored_path_prefixes=None,
    ) -> dict[str, object]:
        _ = ignored_path_prefixes
        self.register_calls.append((sandbox_id, runtime, object_id, ignore_process_rules))
        return {"ok": True}

    def unregister_sandbox(self, sandbox_id: SandboxId) -> dict[str, object]:
        self.unregister_calls.append(sandbox_id)
        return {"ok": True}


class FlakyHostInspectorClient(FakeHostInspectorClient):
    def __init__(self, *, failures_before_success: int) -> None:
        super().__init__()
        self._failures_before_success = failures_before_success

    def register_sandbox(
        self,
        sandbox_id: SandboxId,
        runtime: str,
        object_id: str,
        *,
        ignore_process_rules=None,
        ignored_path_prefixes=None,
    ) -> dict[str, object]:
        _ = ignored_path_prefixes
        self.register_calls.append((sandbox_id, runtime, object_id, ignore_process_rules))
        if self._failures_before_success > 0:
            self._failures_before_success -= 1
            raise TimeoutError("timed out")
        return {"ok": True}


class SandboxManagerTests(unittest.TestCase):
    def test_runc_prepare_launch_moves_rootfs_materialization_out_of_launch(self) -> None:
        with tempfile.TemporaryDirectory(prefix="crab_sandbox_mgr_") as tmp:
            root = Path(tmp)
            runner = FakeCommandRunner()
            manager = RuncSandboxManager(
                command_runner=runner,
                paths=RuncSandboxManagerPaths(
                    state_root=root / "state",
                    bundle_root=root / "bundles",
                    metadata_root=root / "metadata",
                    zfs_dataset_prefix="pool/crab",
                ),
            )
            rootfs_source = root / "source-file.txt"
            rootfs_source.write_text("prepared\n", encoding="utf-8")
            metadata = {
                "sandbox_id": "sbx-test",
                "bundle_path": str(root / "bundles" / "sbx-test"),
                "rootfs_init_dirs": ["work"],
                "rootfs_copy_paths": [{"source": str(rootfs_source), "destination": "/work/source-file.txt"}],
            }

            sandbox_id = manager.prepare_launch("runc", metadata)

            self.assertEqual(sandbox_id, SandboxId("sbx-test"))
            self.assertEqual(
                runner.commands,
                [
                    ("zfs", "destroy", "-r", "pool/crab/sbx-test"),
                    ("zfs", "create", "-o", f"mountpoint={root / 'bundles' / 'sbx-test' / 'rootfs'}", "pool/crab/sbx-test"),
                ],
            )
            self.assertTrue(bool(metadata.get("_crab_runtime_prepared")))
            self.assertEqual(metadata["zfs_dataset"], "pool/crab/sbx-test")
            self.assertEqual(metadata["rootfs_path"], str(root / "bundles" / "sbx-test" / "rootfs"))
            self.assertTrue((root / "bundles" / "sbx-test" / "rootfs" / "work").is_dir())
            self.assertEqual(
                (root / "bundles" / "sbx-test" / "rootfs" / "work" / "source-file.txt").read_text(encoding="utf-8"),
                "prepared\n",
            )

            runner.commands.clear()
            launched_id = manager.launch("runc", metadata)

            self.assertEqual(launched_id, SandboxId("sbx-test"))
            self.assertEqual(
                runner.commands,
                [
                    ("runc", "--root", str(root / "state"), "create", "--bundle", str(root / "bundles" / "sbx-test"), "sbx-test"),
                    ("runc", "--root", str(root / "state"), "start", "sbx-test"),
                ],
            )

    def test_runc_prepare_launch_can_reuse_existing_rootfs_dataset(self) -> None:
        with tempfile.TemporaryDirectory(prefix="crab_sandbox_mgr_") as tmp:
            root = Path(tmp)
            runner = FakeCommandRunner()
            manager = RuncSandboxManager(
                command_runner=runner,
                paths=RuncSandboxManagerPaths(
                    state_root=root / "state",
                    bundle_root=root / "bundles",
                    metadata_root=root / "metadata",
                    zfs_dataset_prefix="pool/crab",
                ),
            )
            rootfs_path = root / "bundles" / "sbx-test" / "rootfs"
            rootfs_path.mkdir(parents=True, exist_ok=True)
            runner.datasets.add("pool/crab/sbx-test")
            metadata = {
                "sandbox_id": "sbx-test",
                "bundle_path": str(root / "bundles" / "sbx-test"),
                "_crab_runtime_reuse_existing_rootfs": True,
            }

            sandbox_id = manager.prepare_launch("runc", metadata)

            self.assertEqual(sandbox_id, SandboxId("sbx-test"))
            self.assertEqual(
                runner.commands,
                [
                    ("zfs", "list", "-H", "-o", "name", "pool/crab/sbx-test"),
                ],
            )
            self.assertTrue(bool(metadata.get("_crab_runtime_prepared")))
            self.assertEqual(metadata["zfs_dataset"], "pool/crab/sbx-test")
            self.assertEqual(metadata["rootfs_path"], str(rootfs_path))

    def test_runc_prepare_launch_clones_shared_rootfs_base_when_key_present(self) -> None:
        with tempfile.TemporaryDirectory(prefix="crab_sandbox_mgr_") as tmp:
            root = Path(tmp)
            runner = FakeCommandRunner()
            manager = RuncSandboxManager(
                command_runner=runner,
                paths=RuncSandboxManagerPaths(
                    state_root=root / "state",
                    bundle_root=root / "bundles",
                    metadata_root=root / "metadata",
                    zfs_dataset_prefix="pool/crab",
                ),
            )
            rootfs_source = root / "source-file.txt"
            rootfs_source.write_text("prepared\n", encoding="utf-8")
            metadata = {
                "sandbox_id": "sbx-shared-a",
                "bundle_path": str(root / "bundles" / "sbx-shared-a"),
                "rootfs_init_dirs": ["work"],
                "rootfs_copy_paths": [{"source": str(rootfs_source), "destination": "/work/source-file.txt"}],
                "shared_rootfs_key": "compose-cache-key",
                "shared_rootfs_persist": True,
            }

            manager.prepare_launch("runc", metadata)

            self.assertIn(
                ("zfs", "create", "-o", "mountpoint=/tmp/crab-rootfs-cache/pool_crab/persistent/compose-cache-key", "pool/crab-cache-compose-cache-key"),
                runner.commands,
            )
            self.assertIn(
                ("zfs", "clone", "-o", f"mountpoint={root / 'bundles' / 'sbx-shared-a' / 'rootfs'}", "pool/crab-cache-compose-cache-key@base", "pool/crab/sbx-shared-a"),
                runner.commands,
            )
            self.assertEqual(
                (root / "bundles" / "sbx-shared-a" / "rootfs" / "work" / "source-file.txt").read_text(encoding="utf-8"),
                "prepared\n",
            )

            runner.commands.clear()
            metadata_b = {
                "sandbox_id": "sbx-shared-b",
                "bundle_path": str(root / "bundles" / "sbx-shared-b"),
                "rootfs_init_dirs": ["work"],
                "rootfs_copy_paths": [{"source": str(rootfs_source), "destination": "/work/source-file.txt"}],
                "shared_rootfs_key": "compose-cache-key",
                "shared_rootfs_persist": True,
            }

            manager.prepare_launch("runc", metadata_b)

            self.assertIn(
                ("zfs", "list", "-H", "-o", "name", "pool/crab-cache-compose-cache-key"),
                runner.commands,
            )
            self.assertIn(
                ("zfs", "list", "-H", "-o", "name", "pool/crab-cache-compose-cache-key@base"),
                runner.commands,
            )
            self.assertNotIn(
                ("zfs", "create", "-o", "mountpoint=/tmp/crab-rootfs-cache/pool_crab/persistent/compose-cache-key", "pool/crab-cache-compose-cache-key"),
                runner.commands,
            )
            self.assertEqual(
                (root / "bundles" / "sbx-shared-b" / "rootfs" / "work" / "source-file.txt").read_text(encoding="utf-8"),
                "prepared\n",
            )

    def test_runc_prepare_launch_emits_prepare_phase_telemetry(self) -> None:
        with tempfile.TemporaryDirectory(prefix="crab_sandbox_mgr_") as tmp:
            root = Path(tmp)
            runner = FakeCommandRunner()
            telemetry = InMemoryTelemetrySink()
            manager = RuncSandboxManager(
                command_runner=runner,
                telemetry=telemetry,
                paths=RuncSandboxManagerPaths(
                    state_root=root / "state",
                    bundle_root=root / "bundles",
                    metadata_root=root / "metadata",
                    zfs_dataset_prefix="pool/crab",
                ),
            )
            rootfs_source = root / "source-file.txt"
            rootfs_source.write_text("prepared\n", encoding="utf-8")
            metadata = {
                "sandbox_id": "sbx-telemetry",
                "bundle_path": str(root / "bundles" / "sbx-telemetry"),
                "rootfs_init_dirs": ["work"],
                "rootfs_copy_paths": [{"source": str(rootfs_source), "destination": "/work/source-file.txt"}],
            }

            manager.prepare_launch("runc", metadata)

        event_names = [name for name, _ in telemetry.events]
        metric_names = [name for name, _, _ in telemetry.metrics]
        self.assertIn("sandbox.runtime_prepare_launch.start", event_names)
        self.assertIn("sandbox.runtime_prepare_launch.finish", event_names)
        self.assertIn("sandbox.rootfs_materialize.start", event_names)
        self.assertIn("sandbox.rootfs_materialize.finish", event_names)
        self.assertIn("sandbox.runtime_prepare_launch.duration_ms", metric_names)
        self.assertIn("sandbox.rootfs_materialize.duration_ms", metric_names)

    def test_runc_sandbox_manager_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory(prefix="crab_sandbox_mgr_") as tmp:
            root = Path(tmp)
            runner = FakeCommandRunner()
            host_inspector = FakeHostInspectorClient()
            manager = RuncSandboxManager(
                command_runner=runner,
                host_inspector_client=host_inspector,
                paths=RuncSandboxManagerPaths(
                    state_root=root / "state",
                    bundle_root=root / "bundles",
                    metadata_root=root / "metadata",
                    zfs_dataset_prefix="pool/crab",
                ),
            )

            sandbox_id = manager.launch(
                "runc",
                {
                    "sandbox_id": "sbx-test",
                    "bundle_path": str(root / "bundles" / "sbx-test"),
                },
            )
            self.assertEqual(sandbox_id, SandboxId("sbx-test"))
            self.assertEqual(manager.describe(sandbox_id).status, "running")

            manager.pause(sandbox_id)
            self.assertEqual(manager.describe(sandbox_id).status, "paused")

            manager.resume(sandbox_id)
            self.assertEqual(manager.describe(sandbox_id).status, "running")

            manager.stop(sandbox_id)
            self.assertEqual(manager.describe(sandbox_id).status, "stopped")

            manager.delete(sandbox_id)
            # The runtime always layers its default ignore rules (criu helper
            # writes) on top of any per-sandbox rules before registering.
            self.assertEqual(
                host_inspector.register_calls,
                [(SandboxId("sbx-test"), "runc", "sbx-test", [{"executable_basename": "criu"}])],
            )
            self.assertEqual(host_inspector.unregister_calls, [SandboxId("sbx-test")])
            self.assertEqual(
                runner.commands,
                [
                    ("zfs", "destroy", "-r", "pool/crab/sbx-test"),
                    ("zfs", "create", "-o", f"mountpoint={root / 'bundles' / 'sbx-test' / 'rootfs'}", "pool/crab/sbx-test"),
                    ("runc", "--root", str(root / "state"), "create", "--bundle", str(root / "bundles" / "sbx-test"), "sbx-test"),
                    ("runc", "--root", str(root / "state"), "start", "sbx-test"),
                    ("runc", "--root", str(root / "state"), "pause", "sbx-test"),
                    ("runc", "--root", str(root / "state"), "resume", "sbx-test"),
                    ("runc", "--root", str(root / "state"), "kill", "sbx-test", "TERM"),
                    ("runc", "--root", str(root / "state"), "delete", "-f", "sbx-test"),
                    ("zfs", "destroy", "-r", "pool/crab/sbx-test"),
                ],
            )

    def test_runc_pause_logs_expected_non_running_state_as_benign(self) -> None:
        with tempfile.TemporaryDirectory(prefix="crab_sandbox_mgr_") as tmp:
            root = Path(tmp)
            runner = PauseNotRunningCommandRunner()
            host_inspector = FakeHostInspectorClient()
            manager = RuncSandboxManager(
                command_runner=runner,
                host_inspector_client=host_inspector,
                paths=RuncSandboxManagerPaths(
                    state_root=root / "state",
                    bundle_root=root / "bundles",
                    metadata_root=root / "metadata",
                    zfs_dataset_prefix="pool/crab",
                ),
            )
            sandbox_id = manager.launch(
                "runc",
                {
                    "sandbox_id": "sbx-test",
                    "bundle_path": str(root / "bundles" / "sbx-test"),
                },
            )

            with self.assertLogs("crab.runtime.runc", level="INFO") as captured:
                with self.assertRaises(RuntimeError):
                    manager.pause(sandbox_id)

        joined = "\n".join(captured.output)
        self.assertIn("Runtime command returned expected non-zero", joined)
        self.assertNotIn("Runtime command failed rc=1", joined)

    def test_runc_sandbox_manager_registers_restored_sandbox_with_host_inspector(self) -> None:
        with tempfile.TemporaryDirectory(prefix="crab_sandbox_mgr_") as tmp:
            root = Path(tmp)
            runner = FakeCommandRunner()
            host_inspector = FakeHostInspectorClient()
            manager = RuncSandboxManager(
                command_runner=runner,
                host_inspector_client=host_inspector,
                paths=RuncSandboxManagerPaths(
                    state_root=root / "state",
                    bundle_root=root / "bundles",
                    metadata_root=root / "metadata",
                    zfs_dataset_prefix="pool/crab",
                ),
            )

            sandbox_id = manager.launch(
                "runc",
                {
                    "sandbox_id": "sbx-restore",
                    "bundle_path": str(root / "bundles" / "sbx-restore"),
                    "host_inspector_ignore_process_rules": [{"executable_basename": "node"}],
                },
            )
            host_inspector.register_calls.clear()

            manager.prepare_for_restore(sandbox_id)
            manager.mark_restored(sandbox_id)

            self.assertEqual(manager.describe(sandbox_id).status, "running")
            self.assertEqual(
                host_inspector.register_calls,
                [
                    (
                        SandboxId("sbx-restore"),
                        "runc",
                        "sbx-restore",
                        [{"executable_basename": "criu"}, {"executable_basename": "node"}],
                    )
                ],
            )

    def test_runc_sandbox_manager_retries_host_inspector_registration(self) -> None:
        with tempfile.TemporaryDirectory(prefix="crab_sandbox_mgr_") as tmp:
            root = Path(tmp)
            runner = FakeCommandRunner()
            host_inspector = FlakyHostInspectorClient(failures_before_success=2)
            manager = RuncSandboxManager(
                command_runner=runner,
                host_inspector_client=host_inspector,
                paths=RuncSandboxManagerPaths(
                    state_root=root / "state",
                    bundle_root=root / "bundles",
                    metadata_root=root / "metadata",
                    zfs_dataset_prefix="pool/crab",
                ),
            )

            with patch("crab.runtime.runc.time.sleep") as sleep:
                sandbox_id = manager.launch(
                    "runc",
                    {
                        "sandbox_id": "sbx-retry",
                        "bundle_path": str(root / "bundles" / "sbx-retry"),
                    },
                )

            self.assertEqual(sandbox_id, SandboxId("sbx-retry"))
            self.assertEqual(
                host_inspector.register_calls,
                [
                    (SandboxId("sbx-retry"), "runc", "sbx-retry", [{"executable_basename": "criu"}]),
                    (SandboxId("sbx-retry"), "runc", "sbx-retry", [{"executable_basename": "criu"}]),
                    (SandboxId("sbx-retry"), "runc", "sbx-retry", [{"executable_basename": "criu"}]),
                ],
            )
            self.assertEqual(sleep.call_count, 2)

    def test_runc_resume_treats_not_paused_as_benign_when_container_is_already_running(self) -> None:
        with tempfile.TemporaryDirectory(prefix="crab_sandbox_mgr_") as tmp:
            root = Path(tmp)
            runner = ResumeRaceCommandRunner(state_status_after_resume="running")
            manager = RuncSandboxManager(
                command_runner=runner,
                paths=RuncSandboxManagerPaths(
                    state_root=root / "state",
                    bundle_root=root / "bundles",
                    metadata_root=root / "metadata",
                    zfs_dataset_prefix="pool/crab",
                ),
            )

            sandbox_id = manager.launch(
                "runc",
                {
                    "sandbox_id": "sbx-test",
                    "bundle_path": str(root / "bundles" / "sbx-test"),
                },
            )
            manager.pause(sandbox_id)
            manager.resume(sandbox_id)

            self.assertEqual(manager.describe(sandbox_id).status, "running")
            self.assertIn(
                ("runc", "--root", str(root / "state"), "state", "sbx-test"),
                runner.commands,
            )

    def test_runc_resume_treats_not_paused_as_benign_when_container_is_already_stopped(self) -> None:
        with tempfile.TemporaryDirectory(prefix="crab_sandbox_mgr_") as tmp:
            root = Path(tmp)
            runner = ResumeRaceCommandRunner(state_status_after_resume="stopped")
            manager = RuncSandboxManager(
                command_runner=runner,
                paths=RuncSandboxManagerPaths(
                    state_root=root / "state",
                    bundle_root=root / "bundles",
                    metadata_root=root / "metadata",
                    zfs_dataset_prefix="pool/crab",
                ),
            )

            sandbox_id = manager.launch(
                "runc",
                {
                    "sandbox_id": "sbx-test",
                    "bundle_path": str(root / "bundles" / "sbx-test"),
                },
            )
            manager.pause(sandbox_id)
            manager.resume(sandbox_id)

            self.assertEqual(manager.describe(sandbox_id).status, "stopped")


if __name__ == "__main__":
    unittest.main()
