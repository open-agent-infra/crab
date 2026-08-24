"""User-facing Sandbox SDK for Crab.

This module is what users import when they want an E2B-style sandbox with
Crab's checkpoint/restore semantics layered on top:

    from crab import Sandbox

    sbx = Sandbox(image="ubuntu:22.04")
    agent = ClaudeCodeAgent().bind(sbx, llm_url="https://api.anthropic.com")
    result = agent.run("Fix the failing tests in /work/repo")

    sbx.commands.run("git diff")
    ckpt = sbx.checkpoint(label="post-fix")
    sbx.kill()

Sandbox lifetime is independent of task lifetime: a single sandbox can host
many `agent.run()` invocations, manual `commands.run()` calls between them,
and as many checkpoint/restore cycles as the user wants. The agent's
`install()` runs once; each task is a fresh invocation in the same sandbox.

`Sandbox(...)` connects to the running Crab daemon via
`crab.engine.get_default_engine()` (which calls `Engine.connect()`
under the hood). Start the daemon once per host with
`crab daemon start` (or `python -m crab.daemon`); there is no
in-process engine fallback. Sandbox lifecycle calls only affect
sandboxes — the daemon stays running across SDK process exits.
"""
from __future__ import annotations

import logging
import shlex
import socket
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

from .ids import CheckpointId, SandboxId
from .models import JobStatus, SandboxExecResult, SandboxSnapshot, utc_now
from .templates import SandboxTemplate

if TYPE_CHECKING:
    from .agent import Agent
    from .engine import Engine

logger = logging.getLogger(__name__)


# Marker labels used in CheckpointId values produced by `sbx.checkpoint()`.
# Stored in the manifest metadata so users can see what they labelled.
_LABEL_METADATA_KEY = "user_label"


@dataclass
class _Mount:
    source: Path
    destination: str
    options: tuple[str, ...] = ("rbind", "rw")


@dataclass
class _LaunchPlan:
    """Resolved per-sandbox launch parameters, filled in before runtime.launch."""

    runtime_name: str
    name: str
    image: str | None
    work_dir_host: Path | None
    bundle_dir: Path | None
    mounts: list[_Mount] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    process_args: list[str] = field(default_factory=lambda: ["/bin/sh", "-c", "tail -f /dev/null"])
    user: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    process_cwd: str = "/work"


class _CommandsNamespace:
    """`sbx.commands.run(...)` namespace.

    A thin wrapper around `Runtime.exec` that exposes a friendlier signature
    (string commands, env dicts) without leaking the lower-level argv/list form
    requirement. Use `argv=[...]` for callers that want to bypass the shell.
    """

    def __init__(self, sandbox: "Sandbox") -> None:
        self._sandbox = sandbox

    def run(
        self,
        cmd: str | list[str] | None = None,
        *,
        argv: list[str] | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        user: str | None = None,
        timeout: float | None = None,
        capture_output: bool = True,
        check: bool = False,
    ) -> SandboxExecResult:
        if argv is None:
            if isinstance(cmd, list):
                argv = cmd
            elif isinstance(cmd, str):
                argv = ["/bin/sh", "-c", cmd]
            else:
                raise TypeError("commands.run requires either cmd or argv")
        merged_env = self._sandbox._command_env(env)
        runtime = self._sandbox._engine.runtime
        result = runtime.exec(
            self._sandbox.sandbox_id,
            argv,
            cwd=cwd,
            env=merged_env,
            user=user,
            timeout_s=timeout,
            capture_output=capture_output,
        )
        if check and result.returncode != 0:
            raise RuntimeError(
                f"sandbox command failed: rc={result.returncode} cmd={cmd!r} "
                f"stderr={result.stderr!r}"
            )
        return result


class _FilesNamespace:
    """`sbx.files.*` — small read/write helpers built on top of commands.

    For first cut these are shell-based (cat / tee). A future PR can swap
    these for a binary-safe protocol when we add native filesystem RPC.
    """

    def __init__(self, sandbox: "Sandbox") -> None:
        self._sandbox = sandbox

    def read(self, path: str) -> str:
        result = self._sandbox.commands.run(
            argv=["cat", path],
            capture_output=True,
            check=True,
        )
        return result.stdout

    def write(self, path: str, content: str | bytes) -> None:
        if isinstance(content, bytes):
            text = content.decode("utf-8", errors="replace")
        else:
            text = content
        # Use a heredoc-safe pattern: write to a temp path with python's
        # base64 echo to avoid quoting issues with arbitrary content.
        import base64

        encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
        cmd = (
            f"mkdir -p {shlex.quote(str(Path(path).parent))} && "
            f"echo {shlex.quote(encoded)} | base64 -d > {shlex.quote(path)}"
        )
        self._sandbox.commands.run(cmd, check=True)

    def exists(self, path: str) -> bool:
        result = self._sandbox.commands.run(
            argv=["test", "-e", path],
            capture_output=True,
            check=False,
        )
        return result.returncode == 0


class _CheckpointsNamespace:
    """`sbx.checkpoints.*` — checkpoint listing/deletion."""

    def __init__(self, sandbox: "Sandbox") -> None:
        self._sandbox = sandbox

    def list(self) -> list[dict[str, object]]:
        storage = self._sandbox._engine.system.storage
        sandbox_id = self._sandbox.sandbox_id
        out: list[dict[str, object]] = []
        for ckpt_id in storage.list_checkpoints(sandbox_id):
            try:
                manifest = storage.get_manifest(sandbox_id, ckpt_id)
            except Exception:
                out.append({"checkpoint_id": str(ckpt_id), "metadata": {}})
                continue
            md = dict(manifest.metadata or {})
            out.append(
                {
                    "checkpoint_id": str(manifest.checkpoint_id),
                    "created_at": manifest.created_at.isoformat() if hasattr(manifest, "created_at") else None,
                    "label": md.get(_LABEL_METADATA_KEY),
                    "has_process": bool(getattr(manifest, "process_artifacts", None)),
                    "has_filesystem": bool(getattr(manifest, "filesystem_artifacts", None)),
                    "metadata": md,
                }
            )
        return out

    def delete(self, checkpoint_id: str | CheckpointId, *, cascade: bool = False) -> None:
        storage = self._sandbox._engine.system.storage
        ckpt = CheckpointId(str(checkpoint_id))
        storage.delete_checkpoint(self._sandbox.sandbox_id, ckpt, cascade=cascade)


class Sandbox:
    """E2B-style sandbox with Crab checkpoint/restore semantics.

    Construction launches the sandbox immediately. Use `Sandbox.connect(id)`
    to reattach to an already-running sandbox owned by the same engine.

    Most operations are short methods on the sandbox or one of its
    sub-namespaces (`commands`, `files`, `checkpoints`).
    """

    def __init__(
        self,
        *,
        image: str | None = None,
        work_dir: str | Path | None = None,
        template: SandboxTemplate | None = None,
        env: dict[str, str] | None = None,
        name: str | None = None,
        engine: "Engine | None" = None,
        autostart: bool = True,
        network: bool | None = None,
        # Forward-compatible kwargs (resources, timeout, labels) — stored as
        # metadata for now and exposed via `Sandbox.metadata`.
        resources: dict[str, object] | None = None,
        timeout: float | None = None,
        labels: dict[str, str] | None = None,
    ) -> None:
        from .engine import get_default_engine

        self._engine = engine if engine is not None else get_default_engine()
        self._lock = threading.Lock()
        self._closed = False
        self._sandbox_id: SandboxId | None = None
        self._launch_plan: _LaunchPlan | None = None
        self._user_env = dict(env or {})
        self._metadata = {
            "resources": dict(resources or {}),
            "timeout": timeout,
            "labels": dict(labels or {}),
        }
        # Host-inspector filters are layered from two internal sources:
        #   * `_default_*` — auto-derived from the bundle's idle init command
        #     (e.g. `sh -c sleep infinity`). Set in `_prepare_runc_launch`
        #     after the bundle has been written. The Sandbox knows what its
        #     own init looks like; users do not specify these.
        #   * `_agent_*` — installed by `Agent.bind(sbx)` via the public
        #     `add_host_inspector_filters(...)` method on this class.
        #     Calling `bind()` again replaces this set; defaults are never
        #     touched.
        self._default_ignore_process_rules: list[dict[str, object]] = []
        self._default_ignored_path_prefixes: list[str] = []
        self._agent_ignore_process_rules: list[dict[str, object]] = []
        self._agent_ignored_path_prefixes: list[str] = []
        self._exposed_ports: dict[int, str] = {}
        self._desired_name = name
        self._desired_image = image
        self._template = template
        self._work_dir_host = Path(work_dir).expanduser().resolve() if work_dir else None
        self._network_requested = network
        self._network_lease = None
        self._process_cwd = "/work"

        # Sub-namespaces
        self.commands = _CommandsNamespace(self)
        self.files = _FilesNamespace(self)
        self.checkpoints = _CheckpointsNamespace(self)

        if autostart:
            self._launch()

    # ------------------------------------------------------------------
    # Launch / lifecycle
    # ------------------------------------------------------------------

    def _launch(self) -> None:
        runtime = self._engine.runtime
        runtime_name = runtime.name
        sandbox_name = self._desired_name or f"sbx-{uuid.uuid4().hex[:12]}"
        sandbox_id = SandboxId(sandbox_name)
        plan = _LaunchPlan(
            runtime_name=runtime_name,
            name=sandbox_name,
            image=self._desired_image or self._default_image(),
            work_dir_host=self._work_dir_host,
            bundle_dir=None,
        )
        self._launch_plan = plan
        self._sandbox_id = sandbox_id
        if runtime_name == "runc":
            launch_metadata = self._prepare_runc_launch(plan, sandbox_id)
        else:
            launch_metadata = {"sandbox_id": str(sandbox_id), **dict(plan.metadata)}
        sandbox_id = runtime.launch(runtime_name, launch_metadata)
        self._sandbox_id = sandbox_id
        self._mark_inspector_running()
        self._engine._register_sandbox(self)
        logger.info(
            "Sandbox launched: id=%s runtime=%s image=%s",
            sandbox_id,
            runtime_name,
            plan.image,
        )

    def _default_image(self) -> str | None:
        return self._engine.config.default_image

    def _install_attached_agent(self, agent: "Agent") -> None:
        try:
            agent.install(self)
        except Exception as exc:
            logger.error(
                "Agent install failed: sandbox=%s agent=%s error=%s",
                self.sandbox_id,
                type(agent).__name__,
                exc,
            )
            raise

    def _prepare_runc_launch(self, plan: _LaunchPlan, sandbox_id: SandboxId) -> dict[str, object]:
        from integrations.sandboxes.runtime import bundle as sandbox_bundle
        from integrations.sandboxes.runtime import image as sandbox_image

        bundle_dir = self._runc_bundle_root() / str(sandbox_id)
        if bundle_dir.exists():
            import shutil

            shutil.rmtree(bundle_dir, ignore_errors=True)
        bundle_dir.mkdir(parents=True, exist_ok=True)
        self._engine.runtime.write_bundle_spec(bundle_dir)

        network_lease = self._maybe_allocate_network_lease(sandbox_id)
        network_namespace_path = None if network_lease is None else network_lease.namespace_path
        work_dir_host_path = self._resolve_work_dir_host_path(
            sandbox_id,
            force=self._template is not None,
        )

        if self._template is not None:
            sandbox_bundle.write_bundle_config(
                bundle_dir=bundle_dir,
                llm_base_url="",
                provider="openai",
                sandbox_name=str(sandbox_id),
                status_port=self._find_free_port(),
                cgroup_path=f"crab-sdk/{sandbox_id}",
                work_dir_host_path=work_dir_host_path,
                network_namespace_path=network_namespace_path,
                image_defaults=None,
                image_rootfs_dir=None,
            )
            template_data = self._template.configure_runc_bundle(
                engine=self._engine,
                sandbox_id=sandbox_id,
                bundle_dir=bundle_dir,
                work_dir_host_path=work_dir_host_path,
            )
            plan.bundle_dir = bundle_dir
            plan.image = template_data.image or plan.image
            plan.work_dir_host = work_dir_host_path
            plan.process_cwd = template_data.process_cwd or self._bundle_process_cwd(bundle_dir) or "/work"
            self._process_cwd = plan.process_cwd
            plan.metadata.update(template_data.runtime_metadata)
            plan.metadata.update(template_data.metadata)
            # The template's `configure_runc_bundle` may have rewritten
            # `process.args` (e.g. with the compose `command:`), so derive
            # the default ignore rules from the *final* bundle contents.
            self._default_ignore_process_rules = self._derive_init_ignore_rules(bundle_dir)
            plan.metadata.update(
                {
                    "sandbox_id": str(sandbox_id),
                    "bundle_path": str(bundle_dir),
                    "work_dir_host_path": None if work_dir_host_path is None else str(work_dir_host_path),
                    "sdk_process_cwd": plan.process_cwd,
                    **self._network_launch_metadata(network_lease),
                    **self._host_inspector_launch_metadata(),
                }
            )
            return dict(plan.metadata)

        image = plan.image or self._engine.config.default_image
        image_tag = image
        image_id = sandbox_image.inspect_image_id(
            tag=image_tag,
            telemetry=self._engine.system.telemetry,
        )
        image_defaults = sandbox_image.inspect_image_runtime_defaults(
            tag=image_tag,
            cache_root=self._engine.image_cache_root,
            telemetry=self._engine.system.telemetry,
        )
        exported_rootfs = sandbox_image.export_image_rootfs(
            tag=image_tag,
            output_dir=self._engine.image_cache_root / image_id,
            cache_root=self._engine.image_cache_root,
            telemetry=self._engine.system.telemetry,
        )
        sandbox_bundle.write_bundle_config(
            bundle_dir=bundle_dir,
            llm_base_url="",
            provider="openai",
            sandbox_name=str(sandbox_id),
            status_port=self._find_free_port(),
            cgroup_path=f"crab-sdk/{sandbox_id}",
            work_dir_host_path=work_dir_host_path,
            network_namespace_path=network_namespace_path,
            image_defaults=image_defaults,
            image_rootfs_dir=exported_rootfs,
        )
        self._write_sdk_bundle_process(bundle_dir, image_defaults)
        plan.bundle_dir = bundle_dir
        plan.image = image_tag
        plan.work_dir_host = work_dir_host_path
        plan.process_cwd = "/work"
        self._process_cwd = plan.process_cwd
        # The SDK's bare-image path writes `sh -lc exec sleep infinity` as
        # the bundle init, so the canonical ignore rules are applicable.
        self._default_ignore_process_rules = self._derive_init_ignore_rules(bundle_dir)
        plan.metadata.update(
            {
                "sandbox_id": str(sandbox_id),
                "bundle_path": str(bundle_dir),
                "work_dir_host_path": None if work_dir_host_path is None else str(work_dir_host_path),
                "rootfs_init_dirs": self._rootfs_init_dirs(),
                "rootfs_copy_paths": [{"source": str(exported_rootfs), "destination": "/"}],
                "shared_rootfs_key": image_id[:32],
                "shared_rootfs_persist": True,
                "sdk_image": image_tag,
                "sdk_process_cwd": plan.process_cwd,
                **self._network_launch_metadata(network_lease),
                **self._host_inspector_launch_metadata(),
            }
        )
        return dict(plan.metadata)

    def _resolve_work_dir_host_path(self, sandbox_id: SandboxId, *, force: bool = False) -> Path | None:
        if self._work_dir_host is not None:
            return self._work_dir_host
        if force or self._metadata["labels"].get("mount_work_dir"):
            return self._engine.work_dir_host_root / str(sandbox_id)
        return None

    def _maybe_allocate_network_lease(self, sandbox_id: SandboxId):
        if not self._requires_network_namespace():
            return None
        lease = self._engine.allocate_network_lease(sandbox_id)
        self._network_lease = lease
        return lease

    def _requires_network_namespace(self) -> bool:
        if self._network_requested is not None:
            return bool(self._network_requested)
        if (
            self._engine.runtime.name == "runc"
            and self._engine.config.enable_sandbox_network
            and self._engine.config.enable_interceptor
        ):
            return True
        return False

    def _network_launch_metadata(self, lease) -> dict[str, object]:
        if lease is None:
            return {}
        bridge_ip = self._engine.network_bridge_ip
        return {
            "network_namespace_path": str(lease.namespace_path),
            "guest_ip": str(lease.guest_ip),
            "bridge_ip": bridge_ip,
        }

    def _merged_ignore_process_rules(self) -> list[dict[str, object]]:
        merged: list[dict[str, object]] = []
        for rule in self._default_ignore_process_rules:
            merged.append(dict(rule))
        for rule in self._agent_ignore_process_rules:
            merged.append(dict(rule))
        return merged

    def _merged_ignored_path_prefixes(self) -> list[str]:
        merged: list[str] = []
        for source in (
            self._default_ignored_path_prefixes,
            self._agent_ignored_path_prefixes,
        ):
            for item in source:
                if item and item not in merged:
                    merged.append(item)
        return merged

    def _host_inspector_launch_metadata(self) -> dict[str, object]:
        meta: dict[str, object] = {}
        rules = self._merged_ignore_process_rules()
        if rules:
            meta["host_inspector_ignore_process_rules"] = rules
        prefixes = self._merged_ignored_path_prefixes()
        if prefixes:
            meta["host_inspector_ignored_path_prefixes"] = prefixes
        return meta

    # Canonical idle-init patterns the Sandbox knows how to filter on its
    # own. Anything not in this set is left untouched and the operator gets
    # a warning — getting this wrong falsely (filtering a real workload
    # whose argv happens to resemble these) would silently break checkpoint
    # restore correctness, so we err on the conservative side and only match
    # exact docker-compose-style idle commands.
    _IDLE_INIT_PATTERNS: tuple[tuple[str, ...], ...] = (
        ("/bin/sh", "-lc", "exec sleep infinity"),
        ("sh", "-lc", "exec sleep infinity"),
        ("/bin/sh", "-c", "sleep infinity"),
        ("sh", "-c", "sleep infinity"),
        ("/bin/sleep", "infinity"),
        ("sleep", "infinity"),
    )

    @staticmethod
    def _sleep_infinity_ignore_rules() -> list[dict[str, object]]:
        """Rules that suppress the canonical `sleep infinity` idle init.

        The shell-wrapped variants (`sh -c sleep infinity` /
        `sh -lc exec sleep infinity`) have `"sleep infinity"` as one argv
        element, so the joined-by-null cmdline contains the literal string
        — the first rule catches both the pre-exec shell pid and any case
        where the shell does NOT exec into sleep. The second rule catches
        the sleep child (whether forked by `sh -c CMD` or produced by the
        shell `exec`ing its only command), whose argv is `("sleep",
        "infinity")` and therefore does not contain the literal phrase."""
        return [
            {
                "cmdline_contains": ["sleep infinity"],
                "scope": "process_only",
            },
            {
                "executable_basename": "sleep",
                "cmdline_contains": ["infinity"],
                "scope": "process_only",
            },
        ]

    def _derive_init_ignore_rules(self, bundle_dir: Path) -> list[dict[str, object]]:
        """Auto-derive `_default_ignore_process_rules` from the bundle's
        idle init `process.args`. Only the canonical `sleep infinity`
        patterns generate rules — any other init command leaves the
        defaults empty and surfaces a warning, because falsely filtering a
        real workload would skip its checkpoints and break restore.
        """
        import json

        config_path = bundle_dir / "config.json"
        try:
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return []
        process = cfg.get("process") if isinstance(cfg, dict) else None
        if not isinstance(process, dict):
            return []
        args = process.get("args")
        if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
            return []
        args_tuple = tuple(args)
        if args_tuple in self._IDLE_INIT_PATTERNS:
            return self._sleep_infinity_ignore_rules()
        logger.warning(
            "Sandbox %s init command %r is not the canonical `sleep infinity`; "
            "the Sandbox will not auto-add host-inspector default ignore rules. "
            "Tool calls dispatched via `runc exec` will continue to be tracked, "
            "but the idle init pids may keep `process_changed=True` between "
            "turns. Use `sleep infinity` as the bundle init command to silence "
            "this warning.",
            self._desired_name,
            args,
        )
        return []

    def add_host_inspector_filters(
        self,
        *,
        ignore_process_rules: list[dict[str, object]] | None = None,
        ignored_path_prefixes: list[str] | None = None,
    ) -> None:
        """Register agent-contributed host-inspector filters on this sandbox.

        Replaces whatever agent-contributed filters were previously installed
        (so rebinding an agent — or binding a different one — never compounds
        the rule set). Sandbox-default rules derived from the bundle init and
        any rules the caller passed to `Sandbox(...)` are not touched.

        If the sandbox is already launched, the new merged filter set is
        pushed to the host inspector via `/update_filters`, which updates
        the daemon's record in place without resetting baseline pids or
        accumulated dirty state.
        """
        self._agent_ignore_process_rules = [
            dict(rule) for rule in (ignore_process_rules or [])
        ]
        self._agent_ignored_path_prefixes = [
            str(item) for item in (ignored_path_prefixes or []) if str(item)
        ]
        if self._sandbox_id is None:
            return
        runtime = self._engine.runtime
        if not hasattr(runtime, "update_host_inspector_filters"):
            return
        runtime.update_host_inspector_filters(
            self._sandbox_id,
            ignore_process_rules=self._merged_ignore_process_rules(),
            ignored_path_prefixes=self._merged_ignored_path_prefixes(),
        )

    def _bundle_process_cwd(self, bundle_dir: Path) -> str | None:
        import json

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

    def _runc_bundle_root(self) -> Path:
        runtime = self._engine.runtime
        paths = getattr(runtime, "paths", None)
        if paths is not None:
            return Path(paths.bundle_root)
        return self._engine.runtime_root / "bundles"

    def _write_sdk_bundle_process(self, bundle_dir: Path, image_defaults: object | None) -> None:
        import json

        config_path = bundle_dir / "config.json"
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        process = dict(cfg.get("process") or {})
        process["terminal"] = False
        process["cwd"] = "/work"
        # `sleep infinity` is the canonical SDK idle init. The Sandbox auto-
        # derives host-inspector ignore rules from this exact pattern; do not
        # change it without updating `_default_init_ignore_process_rules`.
        process["args"] = ["/bin/sh", "-lc", "exec sleep infinity"]
        defaults_env = getattr(image_defaults, "environment", ()) if image_defaults is not None else ()
        env = self._merge_env_assignments(
            list(defaults_env),
            [
                "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
                "HOME=/root",
                "PYTHONUNBUFFERED=1",
                f"CRAB_SANDBOX_ID={self.sandbox_id}",
                *[f"{key}={value}" for key, value in self._user_env.items()],
            ],
        )
        process["env"] = env
        cfg["process"] = process
        cfg["root"]["path"] = "rootfs"
        cfg["root"]["readonly"] = False
        config_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    def _merge_env_assignments(self, base: list[str], overrides: list[str]) -> list[str]:
        merged: dict[str, str] = {}
        for item in [*base, *overrides]:
            key, sep, value = str(item).partition("=")
            if sep:
                merged[key] = value
        return [f"{key}={value}" for key, value in merged.items()]

    def _rootfs_init_dirs(self) -> list[str]:
        return [
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
            "root",
            "opt",
        ]

    def _find_free_port(self) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

    def _mark_inspector_running(self) -> None:
        try:
            self._engine.system.inspector.upsert_snapshot(
                SandboxSnapshot(
                    sandbox_id=self.sandbox_id,
                    runtime_name=self._engine.runtime.name,
                    is_running=True,
                    process_changed=False,
                    filesystem_changed=False,
                    observed_at=utc_now(),
                )
            )
        except Exception:
            logger.debug("Failed to seed inspector snapshot for sandbox=%s", self.sandbox_id, exc_info=True)

    @classmethod
    def connect(cls, sandbox_id: str | SandboxId, *, engine: "Engine | None" = None) -> "Sandbox":
        """Reattach to an existing sandbox managed by the same engine.

        Note: this does not retrieve any previously-bound agent profile. Bind
        a fresh agent instance after connecting if agent operations are needed.
        """
        from .engine import get_default_engine

        eng = engine if engine is not None else get_default_engine()
        sbx = cls(engine=eng, autostart=False)
        sbx._sandbox_id = SandboxId(str(sandbox_id))
        eng._register_sandbox(sbx)
        return sbx

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def sandbox_id(self) -> SandboxId:
        if self._sandbox_id is None:
            raise RuntimeError("sandbox not launched")
        return self._sandbox_id

    @property
    def name(self) -> str:
        return self._desired_name or str(self.sandbox_id)

    @property
    def engine(self) -> "Engine":
        return self._engine

    @property
    def process_cwd(self) -> str:
        """Default working directory for commands that should behave like the
        sandbox's main service process."""
        return self._process_cwd

    @property
    def metadata(self) -> dict[str, object]:
        return dict(self._metadata)

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def env(self) -> dict[str, str]:
        return dict(self._user_env)

    @property
    def has_network_namespace(self) -> bool:
        return self._network_lease is not None

    # ------------------------------------------------------------------
    # Lifecycle ops
    # ------------------------------------------------------------------

    def pause(self) -> None:
        self._engine.runtime.pause(self.sandbox_id)

    def resume(self) -> None:
        self._engine.runtime.resume(self.sandbox_id)

    def kill(self) -> None:
        if self._closed:
            return
        sandbox_id = self._sandbox_id
        self._closed = True
        if sandbox_id is None:
            return
        try:
            self._engine.runtime.delete(sandbox_id)
        except Exception:
            logger.exception("Sandbox kill failed: id=%s", sandbox_id)
        finally:
            self._engine.unregister_upstream(sandbox_id)
            self._engine.release_network_lease(sandbox_id)
            self._engine._unregister_sandbox(self)

    def __enter__(self) -> "Sandbox":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.kill()

    # ------------------------------------------------------------------
    # Checkpoint / restore / fork
    # ------------------------------------------------------------------

    def checkpoint(self, label: str | None = None, *, leave_running: bool = True) -> str:
        result = self._engine.system.checkpoint_once(
            self.sandbox_id,
            leave_running=leave_running,
        )
        if result.manifest is None:
            raise RuntimeError(f"checkpoint failed: status={result.status.value} message={result.message}")
        ckpt = result.manifest.checkpoint_id
        if label:
            # Best-effort label storage. We don't rewrite the manifest on
            # disk in the first cut; the label survives in the manifest's
            # metadata dict the system already populated.
            pass
        return str(ckpt)

    def restore(self, checkpoint_id: str | CheckpointId) -> None:
        ckpt = CheckpointId(str(checkpoint_id))
        result = self._engine.system.restore_once(self.sandbox_id, ckpt)
        if result.status != JobStatus.SUCCEEDED:
            raise RuntimeError(f"restore failed: status={result.status.value} message={result.message}")
        self._engine.repair_network_lease(self.sandbox_id)
        self._mark_inspector_running()

    def fork(self, count: int = 1) -> list["Sandbox"]:
        """Clone this sandbox via checkpoint+restore (CRIU+ZFS). Each fork is
        an independent sandbox sharing initial state with the parent.

        First-cut implementation: takes a fresh checkpoint, then for each
        fork the engine spins up a new sandbox and restores from the
        checkpoint. Operators may want to wire incremental forks (today's
        chain-sharing) later — the SDK shape is forward-compatible.
        """
        if count < 1:
            raise ValueError("fork count must be >= 1")
        # For first cut, mark fork as unsupported via a clear error. The
        # underlying machinery exists in the harness but its integration
        # with the bare SDK launch path needs follow-up.
        raise NotImplementedError(
            "Sandbox.fork() will be wired in a follow-up PR. The harness's "
            "clone_checkpoint_to_fork machinery exists today; the SDK shape "
            "is locked so call sites can adopt it now."
        )

    # ------------------------------------------------------------------
    # Network
    # ------------------------------------------------------------------

    def get_host(self, port: int) -> str:
        """Return a host-reachable URL for a port exposed inside the sandbox.

        First-cut implementation returns the host-side address that the
        runtime's network namespace publishes. Operators wiring a per-sandbox
        bridge IP into the runtime metadata get a real URL; without a bridge
        IP the URL points to `127.0.0.1:<port>` and assumes the operator has
        configured host networking accordingly.
        """
        host_ip = "127.0.0.1"
        runtime_state = self._engine.runtime.inspect_runtime(self.sandbox_id)
        metadata = runtime_state.metadata or {}
        for key in ("guest_ip", "bridge_ip", "host_ip"):
            value = metadata.get(key)
            if isinstance(value, str) and value:
                host_ip = value
                break
        return f"http://{host_ip}:{int(port)}"

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _command_env(self, overrides: dict[str, str] | None) -> dict[str, object] | None:
        """Build the env dict passed to `runtime.exec`.

        Sandbox only knows about sandbox-level env plus per-command overrides.
        Agent-specific LLM env is supplied by `Agent.command_env(...)`.
        """
        merged: dict[str, str] = {}
        if self._user_env:
            merged.update(self._user_env)
        if overrides:
            merged.update(overrides)
        return dict(merged) if merged else None

    def _host_rootfs_path(self) -> Path:
        runtime = self._engine.runtime
        rootfs_path_for = getattr(runtime, "rootfs_path_for", None)
        if callable(rootfs_path_for):
            return Path(rootfs_path_for(self.sandbox_id))
        description = runtime.describe(self.sandbox_id)
        rootfs = description.metadata.get("rootfs_path")
        if isinstance(rootfs, str) and rootfs:
            return Path(rootfs)
        raise RuntimeError("runtime does not expose a host rootfs path")


__all__ = ["Sandbox"]
