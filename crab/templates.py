"""Sandbox templates for the SDK.

Templates are intentionally small user-facing objects: they describe the
filesystem/process shape a sandbox should start from, while the Engine keeps
owning the heavy runc/ZFS/network details. The first concrete template is a
Docker Compose service translator, reused from the benchmark harness so
Terminal-Bench style task images can run through `Sandbox(...)`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .ids import SandboxId

if TYPE_CHECKING:
    from .engine import Engine

@dataclass(frozen=True)
class TemplateLaunchData:
    runtime_metadata: dict[str, object]
    metadata: dict[str, object] = field(default_factory=dict)
    process_cwd: str | None = None
    image: str | None = None


class SandboxTemplate:
    """Base class for templates that can customize a runc bundle."""

    def configure_runc_bundle(
        self,
        *,
        engine: "Engine",
        sandbox_id: SandboxId,
        bundle_dir: Path,
        work_dir_host_path: Path | None,
    ) -> TemplateLaunchData:
        raise NotImplementedError


@dataclass(frozen=True)
class DockerComposeTemplate(SandboxTemplate):
    """Launch a single Docker Compose service as a runc-backed sandbox.

    This is the SDK counterpart to the benchmark harness's compose
    translation path. It deliberately exposes one clean object instead of
    pushing compose image roots, rootfs copy paths, and network knobs into
    `Sandbox(...)`.
    """

    compose_file: Path | str
    service_name: str | None = None
    env_file: Path | str | None = None
    task_root: Path | str | None = None
    logs_root: Path | str | None = None

    @classmethod
    def from_dataset_row(
        cls,
        dataset_path: Path | str,
        row: dict[str, Any],
    ) -> "DockerComposeTemplate":
        dataset_root = Path(dataset_path).expanduser().resolve().parent
        compose_file = row.get("docker_compose_file")
        if not isinstance(compose_file, str) or not compose_file:
            raise ValueError("dataset row does not define docker_compose_file")
        task_root = row.get("task_root")
        env_file = row.get("env_file")
        return cls(
            compose_file=(dataset_root / compose_file).resolve(),
            service_name=None if row.get("service_name") is None else str(row["service_name"]),
            env_file=None if env_file is None else (dataset_root / str(env_file)).resolve(),
            task_root=None if task_root is None else (dataset_root / str(task_root)).resolve(),
        )

    def configure_runc_bundle(
        self,
        *,
        engine: "Engine",
        sandbox_id: SandboxId,
        bundle_dir: Path,
        work_dir_host_path: Path | None,
    ) -> TemplateLaunchData:
        from integrations.sandboxes.runtime import compose as sandbox_compose

        compose_file = Path(self.compose_file).expanduser().resolve()
        env_file = None if self.env_file is None else Path(self.env_file).expanduser().resolve()
        task_root = None if self.task_root is None else Path(self.task_root).expanduser().resolve()
        service_name, service = sandbox_compose.load_compose_service(
            compose_file=compose_file,
            env_file=env_file,
            extra_env=self._compose_env(engine=engine, sandbox_id=sandbox_id, task_root=task_root),
            service_name=self.service_name,
        )
        translation = sandbox_compose.translate_compose_service(
            compose_file=compose_file,
            service_name=service_name,
            service=service,
            bundle_dir=bundle_dir,
            sandbox_id=str(sandbox_id),
            work_dir_host_path=work_dir_host_path,
            compose_image_root=engine.image_cache_root,
            compose_image_tags=None,
            telemetry=engine.system.telemetry,
        )
        runtime_metadata = dict(translation.runtime_launch_metadata)
        if task_root is not None:
            runtime_metadata["task_root"] = str(task_root)
            self._extend_tests_materialization(runtime_metadata, task_root)
        from integrations.sandboxes.runtime.baseline import add_dns_materialization

        add_dns_materialization(runtime_metadata, bundle_dir=bundle_dir)
        process_cwd = self._bundle_process_cwd(bundle_dir)
        image_ref = translation.compose_launch_metadata.get("image_ref")
        return TemplateLaunchData(
            runtime_metadata=runtime_metadata,
            metadata={
                "template": "docker-compose",
                "compose": dict(translation.compose_launch_metadata),
                "compose_file": str(compose_file),
                "service_name": service_name,
                "task_root": None if task_root is None else str(task_root),
            },
            process_cwd=process_cwd,
            image=image_ref if isinstance(image_ref, str) else None,
        )

    def _compose_env(
        self,
        *,
        engine: "Engine",
        sandbox_id: SandboxId,
        task_root: Path | None,
    ) -> dict[str, str]:
        from integrations.sandboxes.runtime.image import docker_tag_component

        task_id = "task" if task_root is None else task_root.name
        logs_root = (
            Path(self.logs_root).expanduser().resolve()
            if self.logs_root is not None
            else engine.agent_state_root / "termnius-logs" / str(sandbox_id)
        )
        task_logs_path = logs_root / "logs"
        task_agent_logs_path = logs_root / "agent-logs"
        task_logs_path.mkdir(parents=True, exist_ok=True)
        task_agent_logs_path.mkdir(parents=True, exist_ok=True)
        image_component = docker_tag_component(task_id)
        return {
            "T_BENCH_CONTAINER_LOGS_PATH": "/logs",
            "T_BENCH_CONTAINER_AGENT_LOGS_PATH": "/agent-logs",
            "T_BENCH_TASK_LOGS_PATH": str(task_logs_path),
            "T_BENCH_TASK_AGENT_LOGS_PATH": str(task_agent_logs_path),
            "T_BENCH_TASK_DOCKER_CLIENT_CONTAINER_NAME": f"crab-{image_component}",
            "T_BENCH_TASK_DOCKER_CLIENT_IMAGE_NAME": f"crab-termnius-{image_component}",
            "T_BENCH_TEST_DIR": "/tests",
        }

    def _extend_tests_materialization(self, runtime_metadata: dict[str, object], task_root: Path) -> None:
        tests_dir = task_root / "tests"
        run_tests = task_root / "run-tests.sh"
        if not run_tests.is_file():
            raise FileNotFoundError(f"missing task run-tests.sh: {run_tests}")
        copy_paths = list(runtime_metadata.get("rootfs_copy_paths", []))
        if tests_dir.is_dir():
            copy_paths.append({"source": str(tests_dir), "destination": "/tests"})
        copy_paths.append({"source": str(run_tests), "destination": "/tests/run-tests.sh"})
        runtime_metadata["rootfs_copy_paths"] = copy_paths
        init_dirs = {str(item).strip("/") for item in runtime_metadata.get("rootfs_init_dirs", [])}
        init_dirs.add("tests")
        runtime_metadata["rootfs_init_dirs"] = sorted(item for item in init_dirs if item)

    def _bundle_process_cwd(self, bundle_dir: Path) -> str | None:
        config_path = bundle_dir / "config.json"
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        process = payload.get("process")
        if not isinstance(process, dict):
            return None
        cwd = process.get("cwd")
        return cwd if isinstance(cwd, str) and cwd else None

__all__ = [
    "DockerComposeTemplate",
    "SandboxTemplate",
    "TemplateLaunchData",
]
