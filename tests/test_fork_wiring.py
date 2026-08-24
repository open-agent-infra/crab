from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from crab import (
    AdapterFileSystemCWorker,
    AdapterFileSystemRWorker,
    AdapterProcessCWorker,
    AdapterProcessRWorker,
    CRExecutor,
    CRScheduler,
    CrabSystem,
    DefaultCWorker,
    DefaultRWorker,
    EBPFSandboxInspector,
    ExecutorConfig,
    InMemorySchedulerStateStore,
    InMemoryTelemetrySink,
    LocalCheckpointManager,
    RuncRuntime,
    RuncRuntimePaths,
    SandboxId,
    SandboxSnapshot,
    SchedulerConfig,
    StorageConfig,
)
from crab import forking
from crab.ids import CheckpointId
from crab.models import utc_now
from crab.runtime import CommandRunner
from crab.scheduler import FaultToleranceCheckpointingPolicy


class FakeCommandRunner(CommandRunner):
    def __init__(self) -> None:
        self.commands: list[tuple[str, ...]] = []

    def run(self, command: list[str], *, cwd: Path | None = None, timeout_seconds: float | None = None):
        _ = (cwd, timeout_seconds)
        self.commands.append(tuple(command))
        return type(
            "Result",
            (),
            {"command": tuple(command), "returncode": 0, "stdout": "", "stderr": ""},
        )()


class ForkingHelperTests(unittest.TestCase):
    def test_copy_plan_is_re_exported_from_benchmarks_support(self) -> None:
        from benchmarks import support

        self.assertIs(support.resolve_checkpoint_copy_plan, forking.resolve_checkpoint_copy_plan)

    def test_replicate_bundle_config_rewrites_per_sandbox_paths(self) -> None:
        with tempfile.TemporaryDirectory(prefix="crab_fork_bundle_") as tmp:
            base = Path(tmp)
            source_dir = base / "sbx-src"
            target_dir = base / "sbx-src-fork-1"
            source_dir.mkdir()
            target_dir.mkdir()
            (base / "hostdirs" / "sbx-src" / "logs").mkdir(parents=True)
            source_cfg = {
                "mounts": [
                    {"destination": "/logs", "type": "bind", "source": str(base / "hostdirs" / "sbx-src" / "logs")},
                    {"destination": "/proc", "type": "proc", "source": "proc"},
                ],
                "process": {
                    "cwd": "/app",
                    "env": ["PATH=/usr/bin", "MARKER=1"],
                    "noNewPrivileges": False,
                    "capabilities": {"bounding": ["CAP_SETUID", "CAP_SETGID"]},
                    "user": {"uid": 0, "gid": 0},
                },
            }
            (source_dir / "config.json").write_text(json.dumps(source_cfg), encoding="utf-8")
            (target_dir / "config.json").write_text(json.dumps({"process": {}}), encoding="utf-8")

            forking.replicate_bundle_config(source_dir, target_dir, SandboxId("sbx-src"), SandboxId("sbx-src-fork-1"))

            rewritten = json.loads((target_dir / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(
                rewritten["mounts"][0]["source"],
                str(base / "hostdirs" / "sbx-src-fork-1" / "logs"),
            )
            self.assertTrue((base / "hostdirs" / "sbx-src-fork-1" / "logs").is_dir())
            self.assertEqual(rewritten["mounts"][1]["source"], "proc")
            self.assertEqual(rewritten["process"]["cwd"], "/app")
            self.assertEqual(rewritten["process"]["env"], ["PATH=/usr/bin", "MARKER=1"])
            self.assertFalse(rewritten["process"]["noNewPrivileges"])
            self.assertEqual(rewritten["process"]["capabilities"]["bounding"], ["CAP_SETUID", "CAP_SETGID"])

    def test_rewrite_filesystem_artifact_stamps_fork_metadata(self) -> None:
        payload = json.dumps(
            {
                "sandbox_id": "sbx-src",
                "filesystem": {
                    "dataset": "pool/crab/sbx-src",
                    "snapshot": "pool/crab/sbx-src@ckpt-1",
                    "mountpoint": "/bundles/sbx-src/rootfs",
                    "fs_ref": "zfs:pool/crab/sbx-src@ckpt-1",
                    "phase": "filesystem_checkpoint",
                },
                "status": {"metadata": {"snapshot": "pool/crab/sbx-src@ckpt-1", "stdout": "keep-me"}},
            }
        ).encode("utf-8")
        fork_metadata = {
            "dataset": "pool/crab/sbx-fork",
            "snapshot": "pool/crab/sbx-fork@ckpt-1",
            "mountpoint": "/bundles/sbx-fork/rootfs",
            "fs_ref": "zfs:pool/crab/sbx-fork@ckpt-1",
        }

        rewritten = json.loads(
            forking.rewrite_filesystem_artifact(
                payload,
                target_sandbox_id=SandboxId("sbx-fork"),
                checkpoint_id=CheckpointId("ckpt-1"),
                filesystem_metadata=fork_metadata,
            ).decode("utf-8")
        )

        self.assertEqual(rewritten["sandbox_id"], "sbx-fork")
        self.assertEqual(rewritten["filesystem"]["fs_ref"], "zfs:pool/crab/sbx-fork@ckpt-1")
        self.assertEqual(rewritten["filesystem"]["snapshot"], "pool/crab/sbx-fork@ckpt-1")
        self.assertEqual(rewritten["filesystem"]["phase"], "filesystem_checkpoint")
        self.assertEqual(rewritten["status"]["metadata"]["snapshot"], "pool/crab/sbx-fork@ckpt-1")
        self.assertEqual(rewritten["status"]["metadata"]["stdout"], "keep-me")

    def test_rewrite_process_artifact_linked_rewrites_without_copying(self) -> None:
        with tempfile.TemporaryDirectory(prefix="crab_fork_linked_") as tmp:
            base = Path(tmp)
            payload = json.dumps(
                {
                    "sandbox_id": "sbx-src",
                    "process_checkpoint_location": str(base / "ckpts" / "sbx-src" / "ckpt-1" / "process"),
                    "status": {"metadata": {"image_path": "old"}},
                }
            ).encode("utf-8")

            rewritten = json.loads(
                forking.rewrite_process_artifact_linked(
                    payload,
                    target_sandbox_id=SandboxId("sbx-fork"),
                    checkpoint_id=CheckpointId("ckpt-1"),
                    bundle_root=base / "bundles",
                    checkpoint_root=base / "ckpts",
                ).decode("utf-8")
            )

            self.assertEqual(rewritten["sandbox_id"], "sbx-fork")
            self.assertEqual(
                rewritten["process_checkpoint_location"],
                str(base / "ckpts" / "sbx-fork" / "ckpt-1" / "process"),
            )
            # No copytree: the fork's checkpoint dir must NOT have been created.
            self.assertFalse((base / "ckpts" / "sbx-fork").exists())


class ForkOnceTests(unittest.TestCase):
    """CrabSystem.fork_once against the real system wiring with a fake
    command runner: manifests/artifacts are cloned onto the fork id, the
    runtime adopts a description, and release hooks are safe no-ops when
    nothing is pinned."""

    def _build(self, root: Path):
        runner = FakeCommandRunner()
        telemetry = InMemoryTelemetrySink()
        inspector = EBPFSandboxInspector()
        runtime = RuncRuntime(
            command_runner=runner,
            paths=RuncRuntimePaths(
                state_root=root / "runtime-state",
                bundle_root=root / "bundles",
                checkpoint_root=root / "checkpoints",
                metadata_root=root / "sandbox-metadata",
                zfs_dataset_prefix="pool/crab",
            ),
        )
        storage = LocalCheckpointManager(
            StorageConfig(root_dir=root / "storage"),
            destroy_filesystem_ref=runtime.destroy_filesystem_ref,
        )
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
        scheduler_cfg = SchedulerConfig(
            min_checkpoint_interval_seconds=0.0,
            force_checkpoint_after_seconds=0.0,
            require_change_signal=True,
        )
        scheduler = CRScheduler(
            scheduler_cfg,
            inspector,
            runtime,
            InMemorySchedulerStateStore(),
            telemetry,
            FaultToleranceCheckpointingPolicy(scheduler_cfg),
        )
        system = CrabSystem(
            scheduler=scheduler,
            executor=executor,
            storage=storage,
            inspector=inspector,
            runtime=runtime,
            telemetry=telemetry,
        )
        return system, runtime, executor, inspector

    def test_fork_once_clones_checkpoint_state_onto_target(self) -> None:
        with tempfile.TemporaryDirectory(prefix="crab_fork_once_") as tmp:
            root = Path(tmp)
            system, runtime, executor, inspector = self._build(root)
            self.addCleanup(executor.shutdown)
            source = SandboxId("sbx-src")
            bundle_dir = root / "bundles" / str(source)
            bundle_dir.mkdir(parents=True)
            (bundle_dir / "config.json").write_text(json.dumps({"process": {"cwd": "/work"}}), encoding="utf-8")
            runtime.launch("runc", {"sandbox_id": str(source), "bundle_path": str(bundle_dir)})
            inspector.upsert_snapshot(
                SandboxSnapshot(
                    sandbox_id=source,
                    runtime_name="runc",
                    is_running=True,
                    process_changed=True,
                    filesystem_changed=True,
                    observed_at=utc_now(),
                )
            )

            fork_id = SandboxId("sbx-src-fork-1")
            fork_bundle = root / "bundles" / str(fork_id)
            fork_bundle.mkdir(parents=True)
            result = system.fork_once(source, fork_id, target_rootfs_path=fork_bundle / "rootfs")

            # Fresh checkpoint was taken and cloned onto the fork id.
            self.assertEqual(result.source_sandbox_id, source)
            source_checkpoints = system.storage.list_checkpoints(source)
            self.assertIn(result.checkpoint_id, source_checkpoints)
            fork_checkpoints = system.storage.list_checkpoints(fork_id)
            self.assertIn(result.checkpoint_id, fork_checkpoints)
            fork_manifest = system.storage.get_manifest(fork_id, result.checkpoint_id)
            self.assertEqual(fork_manifest.sandbox_id, fork_id)
            # Filesystem artifact payload points at the fork's dataset/ref.
            fs_ref_payloads = [
                json.loads(system.storage.get_artifact(fork_id, result.checkpoint_id, ref).decode("utf-8"))
                for ref in fork_manifest.filesystem_artifacts
            ]
            self.assertTrue(fs_ref_payloads)
            for payload in fs_ref_payloads:
                self.assertEqual(payload["sandbox_id"], str(fork_id))
                self.assertIn(str(fork_id), str(payload["filesystem"]["fs_ref"]))
            # Runtime adopted a stopped description for the fork.
            description = runtime.describe(fork_id)
            self.assertEqual(description.status, "stopped")
            self.assertEqual(description.metadata["forked_from"], str(source))
            # The provider-level clone was issued for the fork's dataset.
            runner = runtime._runner  # test-owned fake
            self.assertIn(
                ("zfs", "clone", "-o", f"mountpoint={fork_bundle / 'rootfs'}",
                 f"pool/crab/{source}@{result.checkpoint_id}", f"pool/crab/{fork_id}"),
                runner.commands,
            )
            # No incremental chain on a fresh full checkpoint => no pin.
            self.assertFalse(result.chain_shared)
            system.release_fork(fork_id)  # safe no-op
            system.prepare_source_destroy(source)  # safe no-op


if __name__ == "__main__":
    unittest.main()


class ForkNetworkNamespaceRetargetTests(unittest.TestCase):
    """A fork must run in its own netns: inheriting the source's spec made
    it share the source's network stack, so its egress was attributed to
    the source and its own lease went unused (found by a D2 E2E)."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="crab_fork_netns_")
        self.addCleanup(self._tmp.cleanup)
        self.bundle = Path(self._tmp.name) / "bundle"
        self.bundle.mkdir()

    def _write(self, namespaces) -> Path:
        config = {"linux": {"namespaces": namespaces}, "process": {"args": ["sh"]}}
        path = self.bundle / "config.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        return path

    def test_rewrites_only_the_network_namespace(self) -> None:
        path = self._write(
            [
                {"type": "pid"},
                {"type": "network", "path": "/var/run/netns/ts-source"},
                {"type": "mount"},
            ]
        )
        changed = forking.retarget_bundle_network_namespace(
            self.bundle, "/var/run/netns/ts-fork"
        )
        self.assertTrue(changed)
        namespaces = json.loads(path.read_text(encoding="utf-8"))["linux"]["namespaces"]
        self.assertEqual(
            namespaces,
            [
                {"type": "pid"},
                {"type": "network", "path": "/var/run/netns/ts-fork"},
                {"type": "mount"},
            ],
        )

    def test_no_change_when_already_correct_or_absent(self) -> None:
        self._write([{"type": "network", "path": "/var/run/netns/ts-fork"}])
        self.assertFalse(
            forking.retarget_bundle_network_namespace(self.bundle, "/var/run/netns/ts-fork")
        )
        # Host networking (no network namespace entry) is left alone.
        self._write([{"type": "pid"}])
        self.assertFalse(
            forking.retarget_bundle_network_namespace(self.bundle, "/var/run/netns/ts-fork")
        )

    def test_missing_or_broken_config_is_tolerated(self) -> None:
        empty = Path(self._tmp.name) / "empty"
        empty.mkdir()
        self.assertFalse(
            forking.retarget_bundle_network_namespace(empty, "/var/run/netns/ts-fork")
        )
        (self.bundle / "config.json").write_text("{not json", encoding="utf-8")
        self.assertFalse(
            forking.retarget_bundle_network_namespace(self.bundle, "/var/run/netns/ts-fork")
        )
