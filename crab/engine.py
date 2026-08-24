"""Crab Engine — the host-side runtime manager.

The Engine is the equivalent of `dockerd`: a long-running host service
that owns runc state, ZFS datasets, the host inspector, the LLM
interceptor and forwarder, and the network bridge. It is **always** run
as a daemon process; the SDK never instantiates one in-process.

Two entry points:
  - `Engine.start(config)` — daemon-internal. The Crab daemon
    (`crab.daemon.server`) calls this once at startup to bring the
    Engine up inside its own process. SDK callers should not invoke
    this directly.
  - `Engine.connect(socket=...)` — what the SDK uses. Returns a
    `RemoteEngine` proxy that translates each call into a request to
    the running daemon. `get_default_engine()` is the most common
    entry point.

A single Engine owns:
  - the underlying `Runtime` (in-memory for testing, `RuncRuntime` for prod)
  - the `CrabSystem` (scheduler, executor, checkpoint manager, retention)
  - the `CrabRequestInterceptorServer` (semantic-aware C/R via LLM traffic)
  - per-sandbox upstream URL bookkeeping

Running two daemons (or mixing a daemon with another in-process Engine)
on the same host is not supported — they would race on the same runtime
paths, the host-inspector port, and the network bridge.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import uuid
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from dataclasses import dataclass, field, replace, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from .config import ExecutorConfig, SchedulerConfig, StorageConfig, TelemetryConfig
from .contracts import Runtime
from .executor import CRExecutor
from .ids import SandboxId
from .inspector import EBPFSandboxInspector
from .journal import ActionJournal
from .interceptor import (
    CrabRequestInterceptorServer,
    CompositeRequestInterceptorHook,
    InMemoryRequestStateStore,
    RequestAwareSandboxInspector,
    SandboxResponseGateRegistry,
)
from .remote_inspector import HostInspectorServiceClient, RemoteSandboxInspector
from . import forking
from .runtime import (
    BtrfsProvider,
    OverlayProvider,
    RuncCheckpointOptions,
    RuncRestoreOptions,
    RuncRuntime,
    RuncRuntimeOptions,
    RuncRuntimePaths,
    ZfsProvider,
)
from .scheduler import CRScheduler, FaultToleranceCheckpointingPolicy, InMemorySchedulerStateStore
from .sdk_llm_forwarder import SdkLLMForwarder, serve_sdk_llm_forwarder
from .storage import LocalCheckpointManager
from .models import SandboxSnapshot, utc_now
from .system import CrabSystem, build_default_system
from .telemetry import build_configured_telemetry_sink
from .workers import (
    AdapterFileSystemCWorker,
    AdapterFileSystemRWorker,
    AdapterProcessCWorker,
    AdapterProcessRWorker,
    DefaultCWorker,
    DefaultRWorker,
)

if TYPE_CHECKING:
    from .sandbox import Sandbox

logger = logging.getLogger(__name__)


def _require_mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _optional_mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    return _require_mapping(value, label=label)


def _resolve_config_path(value: object, *, base_dir: Path) -> Path | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _as_bool(value: object, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"cannot parse boolean value {value!r}")


def _as_int(value: object, *, default: int | None = None) -> int | None:
    if value is None:
        return default
    return int(value)


def _as_float(value: object, *, default: float) -> float:
    if value is None:
        return default
    return float(value)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_http_json(url: str, *, timeout_s: float = 30.0) -> dict[str, object]:
    deadline = time.time() + timeout_s
    last_exc: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2.0) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            last_exc = exc
            time.sleep(0.2)
    raise RuntimeError(f"timed out waiting for {url}: {last_exc}")


def _read_log_tail(path: Path | None, *, max_bytes: int = 8192) -> str:
    if path is None:
        return ""
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - max_bytes))
            return fh.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _load_yaml_mapping(path: Path) -> Mapping[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - PyYAML is part of the benchmark env.
        raise RuntimeError("EngineConfig.from_file() requires PyYAML to load YAML config files") from exc
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return _require_mapping(payload, label=f"engine config {path}")


def _telemetry_config_from_mapping(data: Mapping[str, Any], *, base_dir: Path) -> TelemetryConfig:
    base = TelemetryConfig()
    jsonl_path = _resolve_config_path(
        data.get("jsonl_path", data.get("output")),
        base_dir=base_dir,
    )
    return TelemetryConfig(
        enabled=_as_bool(data.get("enabled"), default=base.enabled),
        jsonl_path=jsonl_path,
        keep_in_memory_copy=(
            None if data.get("keep_in_memory_copy") is None else _as_bool(data.get("keep_in_memory_copy"), default=False)
        ),
        detail_level=str(data.get("detail_level", base.detail_level)),
        capture_command_output=_as_bool(
            data.get("capture_command_output"),
            default=base.capture_command_output,
        ),
        max_text_attribute_bytes=int(data.get("max_text_attribute_bytes", base.max_text_attribute_bytes)),
        writer_mode=str(data.get("writer_mode", base.writer_mode)),
        queue_capacity=int(data.get("queue_capacity", base.queue_capacity)),
        batch_max_records=int(data.get("batch_max_records", base.batch_max_records)),
        flush_interval_ms=int(data.get("flush_interval_ms", base.flush_interval_ms)),
        overflow_policy=str(data.get("overflow_policy", base.overflow_policy)),
        serializer=str(data.get("serializer", base.serializer)),
    )


def _executor_config_from_mapping(data: Mapping[str, Any], *, default_workers: int | None) -> ExecutorConfig:
    base = ExecutorConfig(max_workers=max(1, int(default_workers or ExecutorConfig().max_workers)))
    return ExecutorConfig(
        max_workers=int(data.get("max_workers", base.max_workers)),
        checkpoint_workers=_as_int(data.get("checkpoint_workers")),
        restore_workers=_as_int(data.get("restore_workers")),
        coordination_workers=_as_int(data.get("coordination_workers")),
        composite_step_workers=_as_int(data.get("composite_step_workers")),
        max_checkpoint_queue_size=int(
            data.get("max_checkpoint_queue_size", data.get("checkpoint_queue_size", base.max_checkpoint_queue_size))
        ),
        checkpoint_scheduling_policy=str(data.get("checkpoint_scheduling_policy", base.checkpoint_scheduling_policy)),
        reactive_checkpoint_urgent_quota=int(
            data.get("reactive_checkpoint_urgent_quota", base.reactive_checkpoint_urgent_quota)
        ),
        max_retries=int(data.get("max_retries", base.max_retries)),
        retry_backoff_seconds=float(data.get("retry_backoff_seconds", base.retry_backoff_seconds)),
    )


def _scheduler_config_from_mapping(data: Mapping[str, Any]) -> SchedulerConfig:
    base = SchedulerConfig()
    return SchedulerConfig(
        min_checkpoint_interval_seconds=_as_float(
            data.get("min_checkpoint_interval_seconds"),
            default=base.min_checkpoint_interval_seconds,
        ),
        force_checkpoint_after_seconds=_as_float(
            data.get("force_checkpoint_after_seconds"),
            default=base.force_checkpoint_after_seconds,
        ),
        require_change_signal=_as_bool(data.get("require_change_signal"), default=base.require_change_signal),
        checkpoint_full_baseline_on_first_checkpoint=_as_bool(
            data.get("checkpoint_full_baseline_on_first_checkpoint"),
            default=base.checkpoint_full_baseline_on_first_checkpoint,
        ),
        prefer_checkpoint_during_llm_request=_as_bool(
            data.get("prefer_checkpoint_during_llm_request"),
            default=base.prefer_checkpoint_during_llm_request,
        ),
        require_llm_request_for_checkpoint=_as_bool(
            data.get("require_llm_request_for_checkpoint"),
            default=base.require_llm_request_for_checkpoint,
        ),
        inspect_without_pause=_as_bool(data.get("inspect_without_pause"), default=base.inspect_without_pause),
        incremental_process_enabled=_as_bool(
            data.get("incremental_process_enabled"),
            default=base.incremental_process_enabled,
        ),
        full_process_checkpoint_interval=int(
            data.get("full_process_checkpoint_interval", base.full_process_checkpoint_interval)
        ),
        max_process_chain_length=int(data.get("max_process_chain_length", base.max_process_chain_length)),
    )


def _runc_options_from_mapping(data: Mapping[str, Any]) -> RuncRuntimeOptions:
    base = RuncRuntimeOptions()
    checkpoint_data = _optional_mapping(data.get("checkpoint"), label="runc.checkpoint")
    restore_data = _optional_mapping(data.get("restore"), label="runc.restore")
    checkpoint = RuncCheckpointOptions(
        tcp_established=_as_bool(
            checkpoint_data.get("tcp_established"),
            default=base.checkpoint.tcp_established,
        ),
        shell_job=_as_bool(checkpoint_data.get("shell_job"), default=base.checkpoint.shell_job),
        tcp_skip_in_flight=_as_bool(
            checkpoint_data.get("tcp_skip_in_flight"),
            default=base.checkpoint.tcp_skip_in_flight,
        ),
        ext_unix_sk=_as_bool(checkpoint_data.get("ext_unix_sk"), default=base.checkpoint.ext_unix_sk),
        extra_args=tuple(str(item) for item in checkpoint_data.get("extra_args", base.checkpoint.extra_args)),
    )
    restore = RuncRestoreOptions(
        detach=_as_bool(restore_data.get("detach"), default=base.restore.detach),
        tcp_established=_as_bool(restore_data.get("tcp_established"), default=base.restore.tcp_established),
        shell_job=_as_bool(restore_data.get("shell_job"), default=base.restore.shell_job),
        ext_unix_sk=_as_bool(restore_data.get("ext_unix_sk"), default=base.restore.ext_unix_sk),
        lazy_pages=_as_bool(restore_data.get("lazy_pages"), default=base.restore.lazy_pages),
        extra_args=tuple(str(item) for item in restore_data.get("extra_args", base.restore.extra_args)),
    )
    return RuncRuntimeOptions(
        checkpoint=checkpoint,
        restore=restore,
        command_timeout_seconds=float(data.get("command_timeout_seconds", base.command_timeout_seconds)),
        zfs_prepare_timeout_seconds=float(data.get("zfs_prepare_timeout_seconds", base.zfs_prepare_timeout_seconds)),
        filesystem_backend=str(data.get("filesystem_backend", base.filesystem_backend)).strip().lower(),
        btrfs_qgroups_enabled=_as_bool(
            data.get("btrfs_qgroups_enabled"),
            default=base.btrfs_qgroups_enabled,
        ),
    )


def _runc_paths_from_mapping(data: Mapping[str, Any], *, base_dir: Path) -> RuncRuntimePaths:
    base = RuncRuntimePaths()
    return RuncRuntimePaths(
        state_root=_resolve_config_path(data.get("state_root"), base_dir=base_dir) or base.state_root,
        bundle_root=_resolve_config_path(data.get("bundle_root"), base_dir=base_dir) or base.bundle_root,
        checkpoint_root=_resolve_config_path(data.get("checkpoint_root"), base_dir=base_dir) or base.checkpoint_root,
        metadata_root=_resolve_config_path(data.get("metadata_root"), base_dir=base_dir) or base.metadata_root,
        zfs_dataset_prefix=str(data.get("zfs_dataset_prefix", base.zfs_dataset_prefix)).rstrip("/"),
        btrfs_root=_resolve_config_path(data.get("btrfs_root"), base_dir=base_dir) or base.btrfs_root,
    )


def _coerce_engine_config(config: object | None) -> "EngineConfig":
    if config is None:
        return EngineConfig()
    if isinstance(config, EngineConfig):
        return config
    if isinstance(config, (str, os.PathLike)):
        return EngineConfig.from_file(config)
    if isinstance(config, Mapping):
        return EngineConfig.from_mapping(config)
    raise TypeError(f"unsupported engine config type: {type(config).__name__}")


def _build_tls_interceptor(cfg: "EngineConfig") -> object | None:
    """Construct a TLSInterceptor when TLS interception is enabled.

    The import of ``crab.tls_interceptor`` (and transitively ``cryptography``)
    is deferred to this function body so that deployments which never enable
    interception do not need the ``crab[tls]`` extra installed.
    """
    if not cfg.egress_tls_interception_enabled:
        return None
    from .tls_interceptor import TLSInterceptor  # lazy: requires crab[tls]

    storage_dir = Path(cfg.storage_root) / "tls" if cfg.storage_root else Path("/tmp/crab-tls")
    return TLSInterceptor(
        storage_dir,
        bypass_hosts=cfg.egress_tls_bypass_hosts,
        on_handshake_failure=cfg.egress_tls_on_handshake_failure,
    )


@dataclass
class EngineConfig:
    """Engine configuration.

    Operators tune this in `/etc/crab/config.yaml` (planned). For the SDK,
    `EngineConfig()` produces sensible defaults for in-process use with the
    real runc backend. Unit tests can still pass `runtime="docker"` to use
    the in-memory runtime.

    Telemetry is intentionally not exposed here for user-facing use — see
    `crab.telemetry` for sysadmin-side configuration.
    """

    runtime: str = "runc"
    """`runc` selects the real Crab runtime. `docker` selects the
    in-memory runtime used by lightweight unit tests."""

    run_id: str | None = None
    """Optional run identifier stamped onto engine telemetry records."""

    storage_root: Path | None = None
    """Root directory for checkpoint manifests. Defaults to a temp dir under
    `/tmp/crab-<random>` for the in-process case."""

    interceptor_host: str = "127.0.0.1"
    interceptor_port: int = 0
    """0 means pick a free port automatically."""

    forwarder_host: str = "127.0.0.1"
    forwarder_port: int = 0
    """Host + port for the per-sandbox LLM forwarder. The interceptor's
    upstream is set to this forwarder's URL; the forwarder dispatches each
    sandbox's traffic to the real LLM URL the sandbox registered. Same
    architectural pattern as the benchmark harness, which points the
    interceptor at a single BenchmarkLLMRouter URL."""

    enable_interceptor: bool = True
    """Disable the LLM interceptor when the host does not need outbound
    LLM tagging (e.g. bare-image experiments). Disables the forwarder too."""

    enable_sandbox_network: bool = True
    """For runc, create a small bridge/veth network when in-sandbox agents
    need deterministic LLM request attribution across multiple sandboxes."""

    enable_action_journal: bool = True
    """Record every sandbox exec + lifecycle marker into the per-sandbox
    action journal under the storage root (roadmap B1). Cheap JSONL
    appends; disable for runs that must avoid any extra I/O."""

    enable_egress_proxy: bool = False
    """Route all sandbox TCP egress through the host-side transparent
    proxy and record every flow in the effect ledger (roadmap D1).
    Off by default: it rewires the sandbox's outbound path. Requires
    runtime="runc", enable_sandbox_network=True and the action journal
    (the ledger's store). Host-bound traffic is never redirected, so the
    LLM interceptor path is unaffected."""

    egress_proxy_port: int = 0
    """Port for the egress proxy; 0 picks a free one."""

    egress_rules: tuple = ()
    """Host-scoped classification overrides for the effect ledger, e.g.
    ``({"host_glob": "*.internal.example", "classify": "idempotent_read"},)``.
    Encrypted flows are ``opaque`` by default because the proxy cannot
    see the method; rules are how a deployment states what it knows."""

    egress_tls_interception_enabled: bool = False
    """Terminate TLS in the egress proxy to classify HTTPS flows the
    same way as plaintext HTTP (roadmap T1). Off by default: today's
    opaque-tunnel behavior is preserved unless explicitly enabled.
    Requires `crab[tls]` (the cryptography package)."""

    egress_tls_on_handshake_failure: str = "passthrough"
    """What happens when the sandbox rejects our minted leaf cert
    (pinning, missing trust). `passthrough` adds the host to a runtime
    bypass set and closes the connection (client retries go opaque);
    `refuse` closes the connection with no fallback."""

    egress_tls_bypass_hosts: tuple = ()
    """Host globs that are never intercepted (matched on SNI before any
    TLS termination). Uses fnmatch style, e.g. `("*.pinned.example",)`."""

    enable_egress_recording: bool = False
    """Record request/response bodies for plaintext HTTP idempotent reads
    into per-sandbox cassettes (roadmap D2), so replays can serve them
    back. Requires the egress proxy. Off by default: it persists response
    bodies to disk. With TLS interception, HTTPS reads become recordable."""

    egress_recording_max_body_bytes: int = 1024 * 1024
    """Per-body cap; larger exchanges are marked truncated and are never
    replayable."""

    egress_recording_record_errors: bool = False
    """Also record 4xx/5xx responses."""

    egress_recording_varying_headers: tuple = ("accept", "accept-encoding")
    """Request headers that participate in the cassette key. Add ``range``
    (together with ``egress_recording_record_partial``) for ranged reads."""

    egress_recording_record_partial: bool = False
    """Record 206 responses. Only meaningful with ``range`` among the
    varying headers, or different ranges would collide on one cassette."""

    effects_default_policy: str = "allow"
    """Effect policy for snapshot transactions (roadmap D3):
    ``allow`` (writes pass through, aborts report what already left),
    ``defer`` (allow-listed writes queue until commit, answered 202),
    ``reject`` (writes refused with 503), ``seal`` (writes pass but the
    txn becomes non-abortable). Default keeps today's behavior."""

    effects_fork_policy: str = "reject"
    """Policy for fork-backed transactions: a speculative fork should not
    write to the world (roadmap: "multiple forks must not double-fire
    external writes")."""

    effects_standalone_fork_policy: str = "allow"
    """Policy for a `sandbox.fork()` taken **outside** any transaction
    (roadmap F1). Distinct from ``effects_fork_policy``, which governs
    fork-*backed transactions*: an independent branch (RL rollout, tree
    search) is a first-class timeline whose external effects are intended,
    so the default stays ``allow`` and today's forks keep writing. A
    speculative fork — one that serves a main line and must produce its
    effect exactly once — opts in per call via ``fork(effects="reject")``
    or by flipping this default. Only ``allow``/``reject`` apply; ``defer``
    and ``seal`` are refused for a bare fork (no commit to flush a queue
    into, no abort for a seal to block)."""

    effects_rules: tuple = ()
    """Endpoints that tolerate deferral, e.g.
    ``({"host_glob": "*.internal", "method": "POST", "path_glob": "/events*"},)``.
    Empty by default: under ``defer`` an unlisted write is refused rather
    than silently queued, since only the deployment knows whether a caller
    can accept ``202`` instead of the real response."""

    effects_on_unlisted: str = "reject"
    """What ``defer`` does with writes that match no rule."""

    effects_opaque_effects: str = "allow"
    """Encrypted/raw flows carry no method, so they cannot be classified:
    ``allow`` (default) lets them through — refusing would break HTTPS
    reads too; ``reject`` gives a hard seal; ``seal`` marks the txn
    non-abortable on the first opaque flow. Without TLS interception a
    transaction using HTTPS cannot be guaranteed write-free."""

    effects_max_queue_bytes: int = 16 * 1024 * 1024
    """Whole-queue byte ceiling per transaction. Deferred bodies live in
    the daemon's memory until commit, so the per-request cap alone cannot
    bound a loop that posts many allow-listed writes; past this the write
    is refused (``effect_reason="queue_full"``)."""

    effects_max_queue_entries: int = 256
    """Whole-queue entry ceiling per transaction (same reasoning)."""

    network_expected_sandboxes: int | None = None
    """Optional capacity hint for the bridge network allocator."""

    runc_options: RuncRuntimeOptions | None = None
    """Override knobs for `RuncRuntime`. Only used when `runtime == "runc"`."""

    runc_paths: RuncRuntimePaths | None = None
    """Override runc state/bundle/checkpoint/metadata paths."""

    zfs_dataset_prefix: str | None = None
    """ZFS dataset prefix for sandbox rootfs datasets, e.g.
    `crab-300/crab-sdk`. If unset, the engine uses
    `$CRAB_ZFS_DATASET_PREFIX`, `$CRAB_ZPOOL_NAME/crab-sdk`, or
    an installed zpool whose name starts with `crab`."""

    filesystem_backend: str = "zfs"
    """CoW backend for sandbox rootfs checkpoints: `zfs` (default),
    `btrfs`, or `overlay` (overlayfs rootfs with upper/work subvolumes
    on btrfs). With btrfs/overlay, the engine verifies the btrfs area
    instead of resolving/creating a zpool dataset prefix."""

    btrfs_root: Path | None = None
    """Root of the btrfs filesystem holding sandbox subvolumes. Only
    used when `filesystem_backend == "btrfs"`. Defaults to
    `/var/lib/crab/btrfs` (the installer's `--fs-backend btrfs` mount).
    Mount it `noatime` (the installer does): with atime enabled, mere
    reads leak into `btrfs send`-based changesets as utimes-only noise."""

    overlay_root: Path | None = None
    """Root of the overlay backend's btrfs area (per-sandbox upper/work
    subvolumes, shared lowers, snapshot mounts). Only used when
    `filesystem_backend == "overlay"`; defaults to `<btrfs_root>/overlay`
    so a host prepared with `--fs-backend btrfs`/`overlay` runs overlay
    with zero extra setup and the two backends' namespaces never
    collide. Must sit on a btrfs mount (same noatime advice applies)."""

    btrfs_qgroups_enabled: bool = False
    """Enable btrfs qgroups-backed per-snapshot byte stats (real
    overhead; off by default, stats degrade to unknown)."""

    runtime_root: Path | None = None
    """Host root for runc state, bundles, metadata, image cache, work dirs,
    and per-agent state. Defaults under `storage_root`."""

    image_cache_root: Path | None = None
    work_dir_host_root: Path | None = None
    agent_state_root: Path | None = None
    default_image: str = "ubuntu:22.04"

    scheduler_config: SchedulerConfig | None = None
    executor_config: ExecutorConfig | None = None
    storage_config: StorageConfig | None = None
    telemetry_config: TelemetryConfig | None = None

    log_file: Path | None = None
    """Optional SDK engine log file. If unset, the engine does not install
    a file logger and regular Python logging configuration applies."""

    log_level: str = "INFO"
    log_file_mode: str = "append"
    """`append` or `write` when `log_file` is set."""

    host_inspector_launch_mode: str = "in_process"
    """`in_process` uses the lightweight `EBPFSandboxInspector` (no real
    eBPF events — useful for tests). `process` launches the real
    `crab.host_inspector.server` as a subprocess and routes inspection
    through a `RemoteSandboxInspector`. `thread` runs the same daemon
    in-thread."""

    host_inspector_host: str = "127.0.0.1"
    host_inspector_port: int = 0
    """0 means pick a free port automatically when launching the real
    host inspector."""

    host_inspector_log_level: str = "INFO"
    host_inspector_log_file: Path | None = None
    """When the real host inspector is launched, write its stdout/stderr
    to this file. The engine also writes a `host-inspector.stderr.log`
    sibling so the subprocess output is recoverable when --log-file is
    unset."""

    extra: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: str | os.PathLike[str]) -> "EngineConfig":
        config_path = Path(path).expanduser().resolve()
        return cls.from_mapping(_load_yaml_mapping(config_path), base_dir=config_path.parent)

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        base_dir: Path | None = None,
    ) -> "EngineConfig":
        data = _require_mapping(payload.get("engine", payload), label="engine config")
        config_base_dir = Path.cwd() if base_dir is None else Path(base_dir).expanduser().resolve()
        storage_planes = _optional_mapping(data.get("storage_planes"), label="storage_planes")
        network = _optional_mapping(data.get("network"), label="network")
        interceptor = _optional_mapping(data.get("interceptor"), label="interceptor")
        forwarder = _optional_mapping(data.get("forwarder"), label="forwarder")
        logging_config = _optional_mapping(data.get("logging"), label="logging")
        host_inspector = _optional_mapping(data.get("host_inspector"), label="host_inspector")
        journal_data = _optional_mapping(data.get("journal"), label="journal")
        egress = _optional_mapping(data.get("egress"), label="egress")
        tls_interception = _optional_mapping(
            egress.get("tls_interception"), label="egress.tls_interception"
        )
        recording = _optional_mapping(egress.get("recording"), label="egress.recording")
        effects = _optional_mapping(data.get("effects"), label="effects")

        default_workers = _as_int(data.get("max_workers"), default=None)
        executor_data = _optional_mapping(data.get("executor"), label="executor")
        scheduler_data = _optional_mapping(data.get("scheduler"), label="scheduler")
        telemetry_data = _optional_mapping(data.get("telemetry"), label="telemetry")
        runc_data = _optional_mapping(data.get("runc"), label="runc")
        runc_paths_data = _optional_mapping(data.get("runc_paths"), label="runc_paths")
        filesystem_data = _optional_mapping(data.get("filesystem"), label="filesystem")
        btrfs_data = _optional_mapping(filesystem_data.get("btrfs"), label="filesystem.btrfs")
        zfs_data = _optional_mapping(filesystem_data.get("zfs"), label="filesystem.zfs")
        overlay_data = _optional_mapping(filesystem_data.get("overlay"), label="filesystem.overlay")

        storage_root = _resolve_config_path(
            data.get("storage_root", storage_planes.get("storage_root")),
            base_dir=config_base_dir,
        )
        runtime_root = _resolve_config_path(
            data.get("runtime_root", storage_planes.get("runtime_root")),
            base_dir=config_base_dir,
        )
        image_cache_root = _resolve_config_path(
            data.get("image_cache_root", storage_planes.get("image_cache_root")),
            base_dir=config_base_dir,
        )
        work_dir_host_root = _resolve_config_path(
            data.get("work_dir_host_root", storage_planes.get("work_dir_host_root")),
            base_dir=config_base_dir,
        )
        agent_state_root = _resolve_config_path(
            data.get(
                "agent_state_root",
                storage_planes.get("agent_state_root", storage_planes.get("agent_host_root")),
            ),
            base_dir=config_base_dir,
        )

        telemetry_config = _telemetry_config_from_mapping(telemetry_data, base_dir=config_base_dir) if telemetry_data else None
        executor_config = _executor_config_from_mapping(executor_data, default_workers=default_workers) if executor_data or default_workers else None
        scheduler_config = _scheduler_config_from_mapping(scheduler_data) if scheduler_data else None
        runc_options = _runc_options_from_mapping(runc_data) if runc_data else None
        runc_paths = _runc_paths_from_mapping(runc_paths_data, base_dir=config_base_dir) if runc_paths_data else None

        log_file = _resolve_config_path(
            data.get("log_file", logging_config.get("file", logging_config.get("log_file"))),
            base_dir=config_base_dir,
        )
        log_level = str(data.get("log_level", logging_config.get("level", logging_config.get("log_level", "INFO"))))
        log_file_mode = str(
            data.get("log_file_mode", logging_config.get("file_mode", logging_config.get("log_file_mode", "append")))
        ).strip().lower()

        host_inspector_launch_mode = str(
            host_inspector.get("launch_mode", "in_process")
        ).strip().lower()
        host_inspector_host = str(host_inspector.get("host", "127.0.0.1"))
        host_inspector_port = int(host_inspector.get("port", 0))
        host_inspector_log_level = str(host_inspector.get("log_level", "INFO")).upper()
        host_inspector_log_file_raw = host_inspector.get("log_file")
        if isinstance(host_inspector_log_file_raw, bool):
            host_inspector_log_file = None
        else:
            host_inspector_log_file = _resolve_config_path(
                host_inspector_log_file_raw,
                base_dir=config_base_dir,
            )

        return cls(
            runtime=str(data.get("runtime", "runc")),
            run_id=None if data.get("run_id") is None else str(data.get("run_id")),
            storage_root=storage_root,
            interceptor_host=str(interceptor.get("host", data.get("interceptor_host", "127.0.0.1"))),
            interceptor_port=int(interceptor.get("port", data.get("interceptor_port", 0))),
            forwarder_host=str(forwarder.get("host", data.get("forwarder_host", "127.0.0.1"))),
            forwarder_port=int(forwarder.get("port", data.get("forwarder_port", 0))),
            enable_interceptor=_as_bool(
                interceptor.get("enabled", data.get("enable_interceptor")),
                default=True,
            ),
            enable_sandbox_network=_as_bool(
                network.get("enable_sandbox_network", data.get("enable_sandbox_network")),
                default=True,
            ),
            enable_action_journal=_as_bool(
                journal_data.get("enabled", data.get("enable_action_journal")),
                default=True,
            ),
            enable_egress_proxy=_as_bool(
                egress.get("enabled", data.get("enable_egress_proxy")),
                default=False,
            ),
            egress_proxy_port=_as_int(
                egress.get("port", data.get("egress_proxy_port")),
                default=0,
            )
            or 0,
            egress_rules=tuple(egress.get("rules", data.get("egress_rules")) or ()),
            egress_tls_interception_enabled=_as_bool(
                tls_interception.get(
                    "enabled", data.get("egress_tls_interception_enabled")
                ),
                default=False,
            ),
            egress_tls_on_handshake_failure=str(
                tls_interception.get(
                    "on_handshake_failure",
                    data.get("egress_tls_on_handshake_failure"),
                )
                or "passthrough"
            ),
            egress_tls_bypass_hosts=tuple(
                tls_interception.get(
                    "bypass_hosts", data.get("egress_tls_bypass_hosts")
                )
                or ()
            ),
            enable_egress_recording=_as_bool(
                recording.get("enabled", data.get("enable_egress_recording")),
                default=False,
            ),
            egress_recording_max_body_bytes=_as_int(
                recording.get("max_body_bytes", data.get("egress_recording_max_body_bytes")),
                default=1024 * 1024,
            )
            or 1024 * 1024,
            egress_recording_record_errors=_as_bool(
                recording.get("record_errors", data.get("egress_recording_record_errors")),
                default=False,
            ),
            egress_recording_varying_headers=tuple(
                recording.get(
                    "varying_headers", data.get("egress_recording_varying_headers")
                )
                or ("accept", "accept-encoding")
            ),
            egress_recording_record_partial=_as_bool(
                recording.get("record_partial", data.get("egress_recording_record_partial")),
                default=False,
            ),
            effects_default_policy=str(
                effects.get("default_policy", data.get("effects_default_policy")) or "allow"
            ),
            effects_fork_policy=str(
                effects.get("fork_policy", data.get("effects_fork_policy")) or "reject"
            ),
            effects_standalone_fork_policy=str(
                effects.get(
                    "standalone_fork_policy", data.get("effects_standalone_fork_policy")
                )
                or "allow"
            ),
            effects_rules=tuple(effects.get("rules", data.get("effects_rules")) or ()),
            effects_on_unlisted=str(
                effects.get("on_unlisted", data.get("effects_on_unlisted")) or "reject"
            ),
            effects_opaque_effects=str(
                effects.get("opaque_effects", data.get("effects_opaque_effects")) or "allow"
            ),
            effects_max_queue_bytes=_as_int(
                effects.get("max_queue_bytes", data.get("effects_max_queue_bytes")),
                default=16 * 1024 * 1024,
            )
            or 16 * 1024 * 1024,
            effects_max_queue_entries=_as_int(
                effects.get("max_queue_entries", data.get("effects_max_queue_entries")),
                default=256,
            )
            or 256,
            network_expected_sandboxes=_as_int(
                network.get("expected_sandboxes", data.get("network_expected_sandboxes")),
                default=None,
            ),
            runc_options=runc_options,
            runc_paths=runc_paths,
            zfs_dataset_prefix=(
                None
                if data.get("zfs_dataset_prefix", zfs_data.get("dataset_prefix")) is None
                else str(data.get("zfs_dataset_prefix", zfs_data.get("dataset_prefix"))).rstrip("/")
            ),
            filesystem_backend=str(
                filesystem_data.get("backend", data.get("filesystem_backend", "zfs"))
            ).strip().lower(),
            btrfs_root=_resolve_config_path(
                btrfs_data.get("root", data.get("btrfs_root")),
                base_dir=config_base_dir,
            ),
            overlay_root=_resolve_config_path(
                overlay_data.get("root", data.get("overlay_root")),
                base_dir=config_base_dir,
            ),
            btrfs_qgroups_enabled=_as_bool(
                btrfs_data.get("qgroups_enabled", data.get("btrfs_qgroups_enabled")),
                default=False,
            ),
            runtime_root=runtime_root,
            image_cache_root=image_cache_root,
            work_dir_host_root=work_dir_host_root,
            agent_state_root=agent_state_root,
            default_image=str(data.get("default_image", "ubuntu:22.04")),
            scheduler_config=scheduler_config,
            executor_config=executor_config,
            telemetry_config=telemetry_config,
            log_file=log_file,
            log_level=log_level,
            log_file_mode=log_file_mode,
            host_inspector_launch_mode=host_inspector_launch_mode,
            host_inspector_host=host_inspector_host,
            host_inspector_port=host_inspector_port,
            host_inspector_log_level=host_inspector_log_level,
            host_inspector_log_file=host_inspector_log_file,
            extra={
                key: value
                for key, value in data.items()
                if key
                not in {
                    "agent_state_root",
                    "default_image",
                    "enable_interceptor",
                    "enable_sandbox_network",
                    "executor",
                    "forwarder",
                    "host_inspector",
                    "image_cache_root",
                    "interceptor",
                    "interceptor_host",
                    "interceptor_port",
                    "log_file",
                    "log_file_mode",
                    "log_level",
                    "logging",
                    "max_workers",
                    "network",
                    "network_expected_sandboxes",
                    "runc",
                    "runc_paths",
                    "runtime",
                    "run_id",
                    "runtime_root",
                    "scheduler",
                    "storage_planes",
                    "storage_root",
                    "telemetry",
                    "work_dir_host_root",
                    "zfs_dataset_prefix",
                }
            },
        )


class Engine:
    """Crab Engine.

    Two surfaces:
      - `Engine.connect(socket=...)` — what SDK callers use. Returns a
        `RemoteEngine` proxy backed by a running Crab daemon. Most
        callers just use `get_default_engine()` which picks up the
        default socket location.
      - `Engine.start(config)` — daemon-internal. The
        `crab.daemon.server` module calls this once to bring up an
        in-process Engine inside the daemon process. SDK callers should
        not invoke it directly; doing so would start a second Engine
        that races with the daemon on runc state, ZFS, and host paths.
    """

    def __init__(self, config: EngineConfig | str | os.PathLike[str] | Mapping[str, Any] | None = None) -> None:
        self._config = _coerce_engine_config(config)
        self._lock = threading.Lock()
        self._started = False
        self._owns_storage_dir = False
        self._tempdir: tempfile.TemporaryDirectory[str] | None = None
        self._storage_root: Path | None = None
        self._runtime_root: Path | None = None
        self._image_cache_root: Path | None = None
        self._work_dir_host_root: Path | None = None
        self._agent_state_root: Path | None = None
        self._runtime: Runtime | None = None
        self._system: CrabSystem | None = None
        self._interceptor: CrabRequestInterceptorServer | None = None
        self._request_state_store: InMemoryRequestStateStore | None = None
        self._forwarder: SdkLLMForwarder | None = None
        self._forwarder_server = None
        self._forwarder_thread: threading.Thread | None = None
        self._forwarder_base_url: str | None = None
        self._network_manager = None
        self._egress_proxy: Any = None
        self._tls_interceptor_ref: object | None = None
        self._host_inspector_client: HostInspectorServiceClient | None = None
        self._host_inspector_process: subprocess.Popen[str] | None = None
        self._host_inspector_server: Any = None
        self._host_inspector_process_log_path: Path | None = None
        self._sandboxes: "dict[SandboxId, Sandbox]" = {}
        self._sandbox_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @classmethod
    def start(cls, config: EngineConfig | str | os.PathLike[str] | Mapping[str, Any] | None = None) -> "Engine":
        """Daemon-internal: bring up the in-process Engine inside the daemon.

        Called by `crab.daemon.server.DaemonServer.start()`. SDK callers
        should use `Engine.connect(...)` (or the higher-level
        `get_default_engine()`) so the existing daemon stays the sole
        owner of runtime resources. Calling `Engine.start()` from the
        SDK process would create a second Engine that races with the
        daemon on runc state, the ZFS pool, and the host-inspector port.
        """
        engine = cls(config)
        engine._start()
        return engine

    @classmethod
    def connect(
        cls,
        socket: str | os.PathLike[str] | None = None,
        *,
        url: str | None = None,
        api_key: str | None = None,
        timeout_seconds: float = 30.0,
    ) -> "Engine":
        """Connect to a running Crab daemon (or gateway) — returns a `RemoteEngine`.

        The returned object exposes the same attribute surface SDK code
        already reads from a local Engine (`.runtime`, `.config`,
        `.storage_root` and friends, `.register_upstream`, …) — it just
        translates each call into the daemon's HTTP endpoints. Returned
        as `Engine` for static-typing convenience; the runtime type is
        `crab.remote_engine.RemoteEngine`.

        Two modes, dispatched on argument shape:

        - **Local daemon** (default): `socket` is a Unix-socket path,
          defaulting to `default_socket_path()` from `crab.daemon`. The
          daemon must already be running — start it with
          `crab daemon start` (or `python -m crab.daemon`).
        - **Cloud**: pass `url="https://gateway.example.com"` (or give an
          `http(s)://` string as the first positional argument) plus
          `api_key=` — or set `$CRAB_API_KEY`. Requests travel over TCP
          to a crab-gateway's `/v1` API with Bearer auth; gateway errors
          surface as the typed exceptions in `crab.cloud_client`."""
        from .remote_engine import RemoteEngine

        # Shape dispatch: an http(s):// string in the positional slot is
        # a gateway URL, not a socket path.
        if url is None and isinstance(socket, str) and socket.startswith(
            ("http://", "https://")
        ):
            url, socket = socket, None
        if url is not None:
            if socket is not None:
                raise ValueError("pass either socket= or url=, not both")
            from .cloud_client import CloudClient, CloudConnectionError, CloudRequestError

            cloud = CloudClient(url, api_key, timeout_seconds=timeout_seconds)
            try:
                info = cloud.get_json("/info")
            except (CloudRequestError, CloudConnectionError):
                raise
            except Exception as exc:
                raise RuntimeError(
                    f"failed to connect to crab gateway at {cloud.base_url}: {exc}"
                ) from exc
            return RemoteEngine(cloud, info=info)  # type: ignore[return-value]
        if api_key is not None:
            raise ValueError("api_key= is only meaningful with url= (cloud mode)")
        from .daemon import DaemonClient

        client = DaemonClient(socket, timeout_seconds=timeout_seconds)
        try:
            info = client.get_json("/info")
        except FileNotFoundError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"failed to connect to Crab daemon at {client.socket_path}: {exc}. "
                "Is the daemon running? Start it with `crab daemon start` "
                "(or `python -m crab.daemon`)."
            ) from exc
        return RemoteEngine(client, info=info)  # type: ignore[return-value]

    def _start(self) -> None:
        with self._lock:
            if self._started:
                return
            cfg = self._config
            self._configure_logging(cfg)
            storage_root = cfg.storage_root
            if storage_root is None:
                self._tempdir = tempfile.TemporaryDirectory(prefix="crab-engine-")
                storage_root = Path(self._tempdir.name)
                self._owns_storage_dir = True
            storage_root.mkdir(parents=True, exist_ok=True)
            self._storage_root = storage_root
            self._request_state_store = InMemoryRequestStateStore()

            self._runtime_root = (cfg.runtime_root or (storage_root / "runtime")).expanduser().resolve()
            self._image_cache_root = (cfg.image_cache_root or (self._runtime_root / "images")).expanduser().resolve()
            self._work_dir_host_root = (
                cfg.work_dir_host_root or (self._runtime_root / "work")
            ).expanduser().resolve()
            self._agent_state_root = (
                cfg.agent_state_root or (self._runtime_root / "agents")
            ).expanduser().resolve()
            for path in (
                self._runtime_root,
                self._image_cache_root,
                self._work_dir_host_root,
                self._agent_state_root,
            ):
                path.mkdir(parents=True, exist_ok=True)

            if cfg.enable_egress_recording and not cfg.enable_egress_proxy:
                raise RuntimeError(
                    "enable_egress_recording requires enable_egress_proxy=True "
                    "(the proxy is what sees the exchanges)"
                )
            if cfg.enable_egress_proxy and not cfg.enable_action_journal:
                # The journal is the effect ledger's only store: without it
                # the proxy would forward every flow and record none, which
                # looks exactly like working interception.
                raise RuntimeError(
                    "enable_egress_proxy requires enable_action_journal=True "
                    "(the journal is the effect ledger's only store)"
                )
            if cfg.runtime == "runc" and cfg.enable_sandbox_network:
                from integrations.sandboxes.runtime.network import BenchmarkNetworkManager

                network_manager = BenchmarkNetworkManager()
                network_manager.configure(expected_sandboxes=cfg.network_expected_sandboxes)
                if cfg.enable_interceptor or cfg.enable_egress_proxy:
                    network_manager.ensure_bridge()
                self._network_manager = network_manager
            elif cfg.enable_egress_proxy:
                raise RuntimeError(
                    "enable_egress_proxy requires runtime='runc' with "
                    "enable_sandbox_network=True (the bridge is the redirect hook point)"
                )

            if cfg.runtime == "runc":
                self._start_host_inspector_if_configured()
                self._system = self._build_runc_system(storage_root)
            else:
                self._system = build_default_system(
                    storage_root=storage_root,
                    runtime=cfg.runtime,
                    scheduler_config=cfg.scheduler_config,
                    executor_config=cfg.executor_config,
                    storage_config=cfg.storage_config,
                    runc_runtime_options=cfg.runc_options,
                    telemetry_config=cfg.telemetry_config,
                    request_state_store=self._request_state_store,
                )
            self._runtime = self._system.sandbox_manager

            if cfg.enable_interceptor:
                interceptor_host = self._effective_interceptor_host(cfg.interceptor_host)
                # Same architectural pattern as the benchmark harness:
                #   sandbox → CrabRequestInterceptorServer → forwarder → real LLM
                # The interceptor sees a single upstream URL (the forwarder).
                # The forwarder reads `X-Agent-Sandbox-Id` (set by the
                # interceptor's sandbox_id_resolver) and dispatches to the
                # per-sandbox real LLM URL registered via
                # `Engine.register_upstream`. The interceptor itself is
                # unchanged from the harness path.
                forwarder_server, forwarder = serve_sdk_llm_forwarder(
                    host=cfg.forwarder_host,
                    port=cfg.forwarder_port,
                )
                self._forwarder = forwarder
                self._forwarder_server = forwarder_server
                forwarder_host, forwarder_port = forwarder_server.server_address
                self._forwarder_base_url = f"http://{forwarder_host}:{int(forwarder_port)}"
                self._forwarder_thread = threading.Thread(
                    target=forwarder_server.serve_forever,
                    daemon=True,
                    name="crab-sdk-forwarder",
                )
                self._forwarder_thread.start()

                self._interceptor = CrabRequestInterceptorServer(
                    upstream_url=self._forwarder_base_url,
                    request_state_store=self._request_state_store,
                    response_gate_registry=self._system.response_gate_registry,
                    on_state_change=self._system.notify_interceptor_state_change,
                    on_response_ready=self._system.notify_live_response_ready,
                    sandbox_id_resolver=self._resolve_sandbox_id_for_request,
                    host=interceptor_host,
                    port=cfg.interceptor_port,
                )
                self._interceptor.start()
                logger.info(
                    "Engine started: runtime=%s storage_root=%s interceptor=%s forwarder=%s",
                    cfg.runtime,
                    storage_root,
                    self._interceptor.base_url,
                    self._forwarder_base_url,
                )
            else:
                logger.info(
                    "Engine started without interceptor: runtime=%s storage_root=%s",
                    cfg.runtime,
                    storage_root,
                )

            if cfg.enable_egress_proxy:
                # All sandbox TCP egress lands here (bridge REDIRECT);
                # host-bound flows are excluded by the rule, so the
                # interceptor path above is untouched.
                from .egress import CassetteRecorder, EgressProxyServer, EgressRule

                manager = self._network_manager
                assert manager is not None  # guarded above
                rules = tuple(
                    rule if isinstance(rule, EgressRule) else EgressRule.from_json(dict(rule))
                    for rule in cfg.egress_rules
                )
                cassette_recorder = None
                cassette_replayer = None
                if cfg.enable_egress_recording:
                    from .cassettes import CassetteStore
                    from .egress import CassetteReplayer

                    storage_cfg = cfg.storage_config or StorageConfig(root_dir=storage_root)
                    store = CassetteStore(
                        Path(storage_cfg.root_dir) / storage_cfg.cassettes_dirname
                    )
                    cassette_recorder = CassetteRecorder(
                        store,
                        max_body_bytes=cfg.egress_recording_max_body_bytes,
                        record_errors=cfg.egress_recording_record_errors,
                        record_partial=cfg.egress_recording_record_partial,
                        varying_headers=cfg.egress_recording_varying_headers,
                    )
                    cassette_replayer = CassetteReplayer(store)
                    # The system owns both for replay windows (D2) and for
                    # pruning a sandbox's cassettes when it is destroyed.
                    self._system.cassette_store = store
                    self._system.cassette_replayer = cassette_replayer
                from .effects import EffectGate

                effect_gate = EffectGate()
                self._system.effect_gate = effect_gate
                self._system.effect_policy_defaults = {
                    "default_policy": cfg.effects_default_policy,
                    "fork_policy": cfg.effects_fork_policy,
                    "standalone_fork_policy": cfg.effects_standalone_fork_policy,
                    "on_unlisted": cfg.effects_on_unlisted,
                    "opaque_effects": cfg.effects_opaque_effects,
                    "rules": cfg.effects_rules,
                    "max_queue_bytes": cfg.effects_max_queue_bytes,
                    "max_queue_entries": cfg.effects_max_queue_entries,
                }
                proxy = EgressProxyServer(
                    journal=getattr(self._system, "journal", None),
                    sandbox_id_resolver=manager.resolve_sandbox_id,
                    # REDIRECT rewrites the destination to the bridge's own
                    # address, so binding there catches every redirected
                    # flow without exposing the proxy on other interfaces.
                    host=manager.bridge_ip,
                    port=cfg.egress_proxy_port,
                    rules=rules,
                    cassette_recorder=cassette_recorder,
                    cassette_replayer=cassette_replayer,
                    replay_varying_headers=cfg.egress_recording_varying_headers,
                    effect_gate=effect_gate,
                    tls_interceptor=_build_tls_interceptor(cfg),
                )
                self._tls_interceptor_ref = proxy.tls_interceptor
                proxy.start()
                manager.enable_egress_redirect(proxy.port)
                # A previous run's deferred queue did not survive; close out
                # its journal rows so nothing looks pending forever.
                try:
                    self._system.backfill_lost_effects()
                except Exception:
                    logger.debug("Lost-effect backfill failed", exc_info=True)
                # The ledger re-derives classification when read, so the
                # query side needs the same rules as the recording side.
                self._system.egress_rules = rules
                self._egress_proxy = proxy
                logger.info("Egress interception active: proxy_port=%d", proxy.port)

            self._system.start()
            self._started = True

    def _configure_logging(self, cfg: EngineConfig) -> None:
        if cfg.log_file is None and not cfg.log_level:
            return
        level = getattr(logging, str(cfg.log_level or "INFO").upper(), logging.INFO)
        mode_value = str(cfg.log_file_mode or "append").lower()
        if mode_value not in {"append", "write", "a", "w"}:
            raise ValueError("EngineConfig.log_file_mode must be 'append' or 'write'")
        for logger_name in ("crab", "integrations.sandboxes", "integrations.agents"):
            logging.getLogger(logger_name).setLevel(level)
        if cfg.log_file is None:
            return
        log_path = cfg.log_file.expanduser().resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        existing = [
            handler
            for handler in logging.getLogger("crab").handlers
            if getattr(handler, "_crab_engine_log_file", None) == str(log_path)
        ]
        if existing:
            for handler in existing:
                handler.setLevel(level)
            return
        mode = "w" if mode_value in {"write", "w"} else "a"
        handler = logging.FileHandler(log_path, mode=mode, encoding="utf-8")
        handler.setLevel(level)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        setattr(handler, "_crab_engine_log_file", str(log_path))
        for logger_name in ("crab", "integrations.sandboxes", "integrations.agents"):
            logging.getLogger(logger_name).addHandler(handler)

    def _build_runc_system(self, storage_root: Path) -> CrabSystem:
        cfg = self._config
        assert self._request_state_store is not None
        assert self._runtime_root is not None
        backend = cfg.filesystem_backend.strip().lower()
        if backend not in {"zfs", "btrfs", "overlay"}:
            raise ValueError(
                f"unsupported filesystem_backend: {cfg.filesystem_backend!r} "
                "(expected 'zfs', 'btrfs' or 'overlay')"
            )
        btrfs_root = cfg.btrfs_root or RuncRuntimePaths().btrfs_root
        overlay_root = cfg.overlay_root or btrfs_root / "overlay"
        paths = cfg.runc_paths or RuncRuntimePaths(
            state_root=self._runtime_root / "runtime-state",
            bundle_root=self._runtime_root / "bundles",
            checkpoint_root=self._runtime_root / "checkpoints",
            metadata_root=self._runtime_root / "sandbox-meta",
            zfs_dataset_prefix=(
                ZfsProvider.resolve_dataset_prefix(cfg.zfs_dataset_prefix)
                if backend == "zfs"
                else RuncRuntimePaths().zfs_dataset_prefix
            ),
            btrfs_root=btrfs_root,
            overlay_root=overlay_root,
        )
        if backend == "zfs":
            ZfsProvider.ensure_parent_dataset(paths.zfs_dataset_prefix)
        elif backend == "btrfs":
            BtrfsProvider.ensure_root(paths.btrfs_root)
        else:
            OverlayProvider.ensure_root(paths.overlay_root or paths.btrfs_root / "overlay")
        runc_options = cfg.runc_options or RuncRuntimeOptions()
        if runc_options.filesystem_backend != backend or runc_options.btrfs_qgroups_enabled != cfg.btrfs_qgroups_enabled:
            runc_options = replace(
                runc_options,
                filesystem_backend=backend,
                btrfs_qgroups_enabled=cfg.btrfs_qgroups_enabled,
            )
        telemetry_cfg = cfg.telemetry_config or TelemetryConfig(enabled=True)
        telemetry = build_configured_telemetry_sink(
            telemetry_cfg,
            default_attributes={"run_id": cfg.run_id} if cfg.run_id else None,
            keep_in_memory_fallback=True,
        )
        runtime = RuncRuntime(
            paths=paths,
            options=runc_options,
            telemetry=telemetry,
            host_inspector_client=self._host_inspector_client,
        )
        storage_cfg = cfg.storage_config or StorageConfig(root_dir=storage_root)
        journal = None
        if cfg.enable_action_journal:
            journal = ActionJournal(storage_cfg.root_dir / storage_cfg.journal_dirname)
            # The runtime records exec attempts + launch markers itself; the
            # system records checkpoint/restore/fork/destroy markers.
            runtime.action_recorder = journal
        storage = LocalCheckpointManager(
            storage_cfg,
            runtime_image_path_in_use=runtime.runtime_image_path_in_use,
            destroy_filesystem_ref=runtime.destroy_filesystem_ref,
        )
        if self._host_inspector_client is not None:
            base_inspector = RemoteSandboxInspector(
                self._host_inspector_client,
                telemetry=telemetry,
            )
        else:
            base_inspector = EBPFSandboxInspector()
        inspector = RequestAwareSandboxInspector(
            base_inspector,
            self._request_state_store,
        )
        executor_cfg = cfg.executor_config or ExecutorConfig()
        executor = CRExecutor(
            executor_cfg,
            DefaultCWorker(
                AdapterProcessCWorker(runtime),
                AdapterFileSystemCWorker(runtime),
                storage,
                runtime,
                telemetry=telemetry,
                step_workers=executor_cfg.resolved_composite_step_workers,
            ),
            DefaultRWorker(
                AdapterProcessRWorker(runtime),
                AdapterFileSystemRWorker(runtime),
                storage,
                telemetry=telemetry,
                runtime=runtime,
            ),
            telemetry,
        )
        response_gate_registry = SandboxResponseGateRegistry()
        scheduler_cfg = cfg.scheduler_config or SchedulerConfig()
        system = CrabSystem(
            scheduler=CRScheduler(
                scheduler_cfg,
                inspector,
                runtime,
                InMemorySchedulerStateStore(),
                telemetry,
                FaultToleranceCheckpointingPolicy(scheduler_cfg),
            ),
            executor=executor,
            storage=storage,
            inspector=inspector,
            runtime=runtime,
            telemetry=telemetry,
            request_state_store=self._request_state_store,
            response_gate_registry=response_gate_registry,
            journal=journal,
        )
        # Fork-backed txns (B3) need engine-level machinery (lease
        # allocation, bundle replication, restore, kill-path teardown).
        system.configure_fork_txn_hooks(
            fork=self._fork_for_txn,
            destroy=self._destroy_txn_fork,
            lease_repair=self.repair_network_lease,
            lease_transfer=self.transfer_network_lease,
        )
        return system

    def _start_host_inspector_if_configured(self) -> None:
        cfg = self._config
        mode = (cfg.host_inspector_launch_mode or "in_process").strip().lower()
        if mode in {"", "in_process", "in-process", "inproc", "fake"}:
            return
        if mode not in {"process", "thread"}:
            raise ValueError(
                f"unsupported host_inspector.launch_mode={cfg.host_inspector_launch_mode!r}; "
                "expected 'in_process', 'process', or 'thread'"
            )
        assert self._runtime_root is not None
        runc_state_root = (
            cfg.runc_paths.state_root
            if cfg.runc_paths is not None
            else self._runtime_root / "runtime-state"
        )
        runc_state_root.mkdir(parents=True, exist_ok=True)
        host = cfg.host_inspector_host or "127.0.0.1"
        if mode == "process":
            url = self._launch_host_inspector_process(
                runc_state_root=runc_state_root,
                host=host,
                port=cfg.host_inspector_port,
            )
        else:
            url = self._launch_host_inspector_thread(
                runc_state_root=runc_state_root,
                host=host,
                port=cfg.host_inspector_port,
            )
        self._host_inspector_client = HostInspectorServiceClient(url)
        logger.info("Host inspector ready at %s (launch_mode=%s)", url, mode)

    def _launch_host_inspector_process(
        self,
        *,
        runc_state_root: Path,
        host: str,
        port: int,
    ) -> str:
        cfg = self._config
        resolved_port = port if port > 0 else _find_free_port()
        log_path = (
            cfg.host_inspector_log_file
            if cfg.host_inspector_log_file is not None
            else self._default_host_inspector_stderr_log()
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._host_inspector_process_log_path = log_path
        command = [
            sys.executable,
            "-m",
            "crab.host_inspector.server",
            "--host",
            host,
            "--port",
            str(resolved_port),
            "--runc-state-root",
            str(runc_state_root),
            "--max-workers",
            str(max(1, int(self._config.executor_config.max_workers if self._config.executor_config else 4))),
            "--log-level",
            cfg.host_inspector_log_level or "INFO",
        ]
        if cfg.host_inspector_log_file is not None:
            command.extend(["--log-file", str(cfg.host_inspector_log_file)])
        log_fh = open(log_path, "w", encoding="utf-8")
        try:
            self._host_inspector_process = subprocess.Popen(
                command,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                text=True,
            )
        finally:
            log_fh.close()
        url = f"http://{host}:{resolved_port}"
        try:
            _wait_for_http_json(f"{url}/healthz")
        except Exception as exc:
            if (
                self._host_inspector_process is not None
                and self._host_inspector_process.poll() is not None
            ):
                returncode = self._host_inspector_process.returncode
                stderr_tail = _read_log_tail(self._host_inspector_process_log_path)
                self._stop_host_inspector()
                raise RuntimeError(
                    f"host inspector failed to start exit_code={returncode} stderr={stderr_tail.strip()}"
                ) from exc
            self._stop_host_inspector()
            raise
        return url

    def _launch_host_inspector_thread(
        self,
        *,
        runc_state_root: Path,
        host: str,
        port: int,
    ) -> str:
        from .host_inspector.fs_helper import LibbpfFilesystemMonitor
        from .host_inspector.runtime_resolver import RuntimeResolver
        from .host_inspector.server import HostInspectorDaemon, HostInspectorServer

        daemon = HostInspectorDaemon(
            resolver=RuntimeResolver(runc_state_root=runc_state_root),
            fs_monitor=LibbpfFilesystemMonitor(),
        )
        max_workers = max(
            1,
            int(self._config.executor_config.max_workers if self._config.executor_config else 4),
        )
        server = HostInspectorServer(
            host=host,
            port=port,
            daemon=daemon,
            max_workers=max_workers,
        )
        server.start()
        self._host_inspector_server = server
        url = f"http://{host}:{server.port}"
        try:
            _wait_for_http_json(f"{url}/healthz")
        except Exception:
            self._stop_host_inspector()
            raise
        return url

    def _default_host_inspector_stderr_log(self) -> Path:
        if self._config.log_file is not None:
            return self._config.log_file.parent / "host-inspector.stderr.log"
        assert self._runtime_root is not None
        return self._runtime_root / "host-inspector.stderr.log"

    def _stop_host_inspector(self) -> None:
        server = self._host_inspector_server
        self._host_inspector_server = None
        if server is not None:
            try:
                server.stop()
            except Exception:
                logger.exception("Host inspector server stop failed")
        process = self._host_inspector_process
        self._host_inspector_process = None
        if process is not None:
            try:
                process.terminate()
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5.0)
            except Exception:
                logger.exception("Host inspector subprocess stop failed")
        client = self._host_inspector_client
        self._host_inspector_client = None
        if client is not None:
            try:
                client.close()
            except Exception:
                logger.debug("Host inspector client close failed", exc_info=True)

    def _effective_interceptor_host(self, configured_host: str) -> str:
        if configured_host not in {"127.0.0.1", "localhost"}:
            return configured_host
        manager = self._network_manager
        if manager is None:
            return configured_host
        bridge_ip = getattr(manager, "bridge_ip", None)
        return str(bridge_ip) if bridge_ip else configured_host

    def stop(self) -> None:
        with self._lock:
            if not self._started:
                return
            sandboxes = list(self._sandboxes.values())
        # Kill sandboxes outside the lock to avoid deadlocks with their
        # internal locking.
        for sbx in sandboxes:
            try:
                sbx.kill()
            except Exception:
                logger.debug("Best-effort sandbox kill failed during engine stop", exc_info=True)
        with self._lock:
            if self._egress_proxy is not None:
                if self._network_manager is not None:
                    try:
                        self._network_manager.disable_egress_redirect()
                    except Exception:
                        logger.exception("Egress redirect teardown failed")
                try:
                    self._egress_proxy.stop()
                except Exception:
                    logger.exception("Egress proxy stop failed")
                self._egress_proxy = None
            if self._interceptor is not None:
                try:
                    self._interceptor.stop()
                except Exception:
                    logger.exception("Interceptor stop failed")
                self._interceptor = None
            if self._forwarder_server is not None:
                try:
                    self._forwarder_server.shutdown()
                    self._forwarder_server.server_close()
                except Exception:
                    logger.exception("Forwarder shutdown failed")
                self._forwarder_server = None
            if self._forwarder_thread is not None:
                self._forwarder_thread.join(timeout=5.0)
                self._forwarder_thread = None
            self._forwarder = None
            self._forwarder_base_url = None
            if self._network_manager is not None:
                try:
                    self._network_manager.cleanup()
                except Exception:
                    logger.exception("Sandbox network cleanup failed")
                self._network_manager = None
            if self._system is not None:
                try:
                    self._system.stop()
                except Exception:
                    logger.exception("CrabSystem stop failed")
                self._system = None
            self._stop_host_inspector()
            if self._tempdir is not None and self._owns_storage_dir:
                try:
                    self._tempdir.cleanup()
                except Exception:
                    logger.debug("Tempdir cleanup failed", exc_info=True)
                self._tempdir = None
            self._started = False

    def __enter__(self) -> "Engine":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Accessors used by Sandbox
    # ------------------------------------------------------------------

    @property
    def started(self) -> bool:
        return self._started

    @property
    def config(self) -> EngineConfig:
        return self._config

    @property
    def runtime(self) -> Runtime:
        if self._runtime is None:
            raise RuntimeError("engine is not started")
        return self._runtime

    @property
    def system(self) -> CrabSystem:
        if self._system is None:
            raise RuntimeError("engine is not started")
        return self._system

    @property
    def storage_root(self) -> Path:
        if self._storage_root is None:
            raise RuntimeError("engine is not started")
        return self._storage_root

    @property
    def runtime_root(self) -> Path:
        if self._runtime_root is None:
            raise RuntimeError("engine is not started")
        return self._runtime_root

    @property
    def image_cache_root(self) -> Path:
        if self._image_cache_root is None:
            raise RuntimeError("engine is not started")
        return self._image_cache_root

    @property
    def work_dir_host_root(self) -> Path:
        if self._work_dir_host_root is None:
            raise RuntimeError("engine is not started")
        return self._work_dir_host_root

    @property
    def agent_state_root(self) -> Path:
        if self._agent_state_root is None:
            raise RuntimeError("engine is not started")
        return self._agent_state_root

    @property
    def interceptor_base_url(self) -> str | None:
        if self._interceptor is None:
            return None
        return self._interceptor.base_url

    def list_sandboxes(self) -> list[dict[str, Any]]:
        """Return the engine's registered sandboxes as raw dicts.

        Mirrors :meth:`RemoteEngine.list_sandboxes` so callers can rely on
        the same shape (``sandbox_id`` / ``runtime_name`` / ``status`` /
        ``metadata``) regardless of transport."""
        with self._lock:
            sandboxes = list(self._sandboxes.values())
        rows: list[dict[str, Any]] = []
        for sbx in sandboxes:
            sid = sbx.sandbox_id
            if sid is None:
                continue
            try:
                description = self.runtime.describe(sid)
                status = description.status
                runtime_name = description.runtime_name
                metadata = dict(getattr(description, "metadata", {}) or {})
            except Exception:
                status = "unknown"
                runtime_name = self.runtime.name if self._runtime is not None else ""
                metadata = {}
            rows.append(
                {
                    "sandbox_id": str(sid),
                    "runtime_name": runtime_name,
                    "status": status,
                    "metadata": metadata,
                }
            )
        return rows

    # ------------------------------------------------------------------
    # Per-sandbox upstream URL bookkeeping (delegates to the SDK forwarder
    # so the per-sandbox routing lives in one place — same architectural
    # pattern as the harness, which keeps per-sandbox dispatch in the
    # BenchmarkLLMRouter rather than in the interceptor).
    # ------------------------------------------------------------------

    def register_upstream(self, sandbox_id: SandboxId, url: str) -> None:
        if not url or self._forwarder is None:
            return
        self._forwarder.register(str(sandbox_id), url)

    def unregister_upstream(self, sandbox_id: SandboxId) -> None:
        if self._forwarder is None:
            return
        self._forwarder.unregister(str(sandbox_id))

    def _lookup_upstream(self, sandbox_id: SandboxId) -> str | None:
        if self._forwarder is None:
            return None
        return self._forwarder.resolve(str(sandbox_id))

    def _resolve_sandbox_id_for_request(
        self,
        client_host: str | None,
        headers: dict[str, str],
        body: bytes,
    ) -> str | None:
        header_value = headers.get("X-Agent-Sandbox-Id", "").strip()
        if header_value:
            return header_value
        try:
            payload = json.loads(body.decode("utf-8")) if body else {}
        except (UnicodeDecodeError, json.JSONDecodeError):
            payload = {}
        if isinstance(payload, dict):
            metadata = payload.get("metadata")
            if isinstance(metadata, dict):
                candidate = metadata.get("sandbox_id")
                if isinstance(candidate, str) and candidate.strip():
                    return candidate.strip()

        if self._network_manager is not None:
            try:
                sandbox_id = self._network_manager.resolve_sandbox_id(client_host)
            except Exception:
                sandbox_id = None
            if sandbox_id is not None:
                return str(sandbox_id)

        with self._sandbox_lock:
            sandboxes = list(self._sandboxes.values())
        for sandbox in sandboxes:
            try:
                runtime_state = self.runtime.inspect_runtime(sandbox.sandbox_id)
            except Exception:
                continue
            metadata = runtime_state.metadata or {}
            for key in ("guest_ip", "bridge_ip", "host_ip"):
                value = metadata.get(key)
                if isinstance(value, str) and value == client_host:
                    return str(sandbox.sandbox_id)

        # SDK runc sandboxes currently default to host networking unless the
        # caller supplies a network namespace. That makes the HTTP peer
        # 127.0.0.1 for all in-sandbox agents, so fall back to the single
        # registered upstream case. Multi-sandbox LLM routing should use an
        # explicit header or per-sandbox network identity.
        registered = [
            sandbox
            for sandbox in sandboxes
            if self._lookup_upstream(sandbox.sandbox_id) is not None
        ]
        if len(registered) == 1:
            return str(registered[0].sandbox_id)
        return None

    @property
    def tls_ca_cert_path(self) -> Path | None:
        """Host path to the CA certificate when TLS interception is active.

        Returns None when interception is disabled or the engine has not
        started.  Sandbox trust injection uses this to copy the cert into
        the rootfs and derive the env overlay.  No cryptography import
        occurs here — the path is read from the already-constructed
        TLSInterceptor's CAStore.
        """
        ti = self._tls_interceptor_ref
        if ti is None:
            return None
        # ti is a TLSInterceptor; access .ca_store.cert_path
        try:
            return Path(ti.ca_store.cert_path)
        except (AttributeError, TypeError):
            return None

    @property
    def forwarder_base_url(self) -> str | None:
        return self._forwarder_base_url

    @property
    def network_bridge_ip(self) -> str | None:
        manager = self._network_manager
        if manager is None:
            return None
        return str(manager.bridge_ip)

    def allocate_network_lease(self, sandbox_id: SandboxId):
        manager = self._network_manager
        if manager is None:
            raise RuntimeError(
                "sandbox network is not enabled; start Engine with "
                "EngineConfig(enable_sandbox_network=True)"
            )
        lease = manager.allocate_lease(sandbox_id)
        manager.register_guest_ip(lease.guest_ip, sandbox_id)
        return lease

    def release_network_lease(self, sandbox_id: SandboxId) -> None:
        manager = self._network_manager
        if manager is None:
            return
        try:
            manager.release_lease(sandbox_id)
        except Exception:
            logger.exception("Failed to release sandbox network lease: %s", sandbox_id)

    def repair_network_lease(self, sandbox_id: SandboxId) -> bool:
        manager = self._network_manager
        if manager is None:
            return False
        try:
            return bool(manager.repair_lease(sandbox_id))
        except Exception:
            logger.exception("Failed to repair sandbox network lease: %s", sandbox_id)
            return False

    def transfer_network_lease(
        self,
        from_sandbox_id: SandboxId,
        to_sandbox_id: SandboxId,
        *,
        probe: bool = False,
    ) -> bool:
        """Move a fork's network identity onto the source it is promoted into.

        Three things have to move together, which is why this is one call:
        the lease itself (so the address still exists), the target's bundle
        `netns` path (so runc restores into the namespace the image was
        dumped in), and the target's runtime metadata (so the interceptor's
        attribution fallback and `Sandbox.get_host` stop serving the dead
        address). Returns False when there is nothing to transfer, which
        tells the caller to keep the `repair_network_lease` path.

        With ``probe=True`` nothing is mutated and the return value only
        answers "would a transfer happen?" — the promotion needs that before
        it dumps the fork, because a transferred netns requires the fork's
        processes to leave it. The probe swallows its own errors (a
        pre-flight must not abort the promotion); the real call does not,
        because by then the fork is dumped-and-stopped and a silent False
        would send the caller down the repair path with a stopped fork and
        an image bound to the fork's address.
        """
        manager = self._network_manager
        if manager is None:
            return False
        if probe:
            try:
                return manager.lease_for(from_sandbox_id) is not None
            except Exception:
                logger.exception("Failed to probe sandbox network lease: %s", from_sandbox_id)
                return False
        lease = manager.transfer_lease(from_sandbox_id, to_sandbox_id)
        if lease is None:
            # The probe already confirmed a lease before the fork was
            # dumped, so None here means it vanished mid-promotion. The
            # fork is stopped and its image carries the fork's address;
            # falling back to repair would restore it into the wrong netns.
            # Surface it so _promote_fork_onto_source raises with the cause.
            raise RuntimeError(
                f"network lease for {from_sandbox_id} vanished between the "
                "promotion pre-flight and the transfer"
            )
        netns_path = str(lease.namespace_path)
        forking.retarget_bundle_network_namespace(
            self.runtime.bundle_path_for(to_sandbox_id), netns_path
        )
        update_metadata = getattr(self.runtime, "update_network_metadata", None)
        if update_metadata is not None:
            try:
                update_metadata(
                    to_sandbox_id,
                    guest_ip=str(lease.guest_ip),
                    network_namespace_path=netns_path,
                )
            except Exception:
                logger.exception(
                    "Failed to refresh network metadata after lease transfer sandbox=%s",
                    to_sandbox_id,
                )
        return True

    def fork_sandbox(
        self,
        source_sandbox_id: SandboxId,
        *,
        count: int = 1,
        lazy: bool = False,
        effects: str | None = None,
        gate_effects: bool = True,
    ) -> list[SandboxId]:
        """Fork a running sandbox `count` times via checkpoint+restore.

        Each fork gets its own bundle (source config.json copied with
        per-sandbox path rewrites), its own network lease when networking
        is enabled, a checkpoint-state clone (CrabSystem.fork_once, with
        incremental chain sharing when available), and a process restore —
        lazily via CRIU lazy-pages when ``lazy=True``.

        ``effects`` is the bare-fork effect policy (F1); it is resolved and
        validated once, before any fork exists, so a bad value cannot leave
        half a fleet behind. ``gate_effects=False`` is for callers that own
        the effect window themselves — a fork-backed transaction arms its
        own session on the fork (D3), and must not be given a standalone one
        on top of it.
        """
        if count < 1:
            raise ValueError("fork count must be >= 1")
        system = self.system
        fork_effect_policy: str | None = None
        if gate_effects:
            # Validate before anything is created (F1): a rejected policy
            # must not leave forks behind.
            fork_effect_policy = system.validate_standalone_fork_policy(effects)
        elif effects is not None:
            raise ValueError(
                "effects= is not accepted when the caller owns the effect window"
            )
        runtime = self.runtime
        source_bundle = runtime.bundle_path_for(source_sandbox_id)
        paths = getattr(runtime, "paths", None)
        if paths is None:
            raise RuntimeError("fork is only supported on the runc runtime")

        fork_ids: list[SandboxId] = []
        for _ in range(count):
            target_sandbox_id = SandboxId(f"{source_sandbox_id}-fork-{uuid.uuid4().hex[:8]}")
            target_bundle = paths.bundle_root / str(target_sandbox_id)
            target_bundle.mkdir(parents=True, exist_ok=True)
            source_cfg = source_bundle / "config.json"
            if source_cfg.is_file():
                shutil.copy2(source_cfg, target_bundle / "config.json")
            forking.replicate_bundle_config(
                source_bundle,
                target_bundle,
                source_sandbox_id,
                target_sandbox_id,
            )
            if self._network_manager is not None:
                try:
                    lease = self.allocate_network_lease(target_sandbox_id)
                    # The spec was copied from the source, so it still names
                    # the source's netns: without retargeting, the fork
                    # shares the source's network stack and its egress is
                    # attributed to the source.
                    if lease is not None:
                        forking.retarget_bundle_network_namespace(
                            target_bundle, str(lease.namespace_path)
                        )
                except Exception:
                    logger.exception("Failed to allocate network lease for fork %s", target_sandbox_id)

            result = system.fork_once(
                source_sandbox_id,
                target_sandbox_id,
                target_rootfs_path=target_bundle / "rootfs",
            )
            # Arm the gate before the fork's processes are restored: once
            # they run, an ungated write could already be on the wire.
            if fork_effect_policy is not None:
                system.arm_fork_effect_session(target_sandbox_id, fork_effect_policy)
            try:
                restore_result = system.restore_once(
                    target_sandbox_id,
                    result.checkpoint_id,
                    restore_metadata={"lazy_pages": True} if lazy else None,
                )
                if restore_result.status.value != "succeeded":
                    raise RuntimeError(
                        f"fork restore failed for {target_sandbox_id}: status={restore_result.status.value}"
                    )
            except Exception:
                # The session was armed a moment ago for a fork that never
                # came up; leaving it behind would gate a dead id forever.
                if fork_effect_policy is not None:
                    system._release_effect_session(target_sandbox_id)
                raise
            self.repair_network_lease(target_sandbox_id)
            fork_ids.append(target_sandbox_id)
        return fork_ids

    def _fork_for_txn(self, source_sandbox_id: SandboxId) -> SandboxId:
        """Fork hook for fork-backed transactions (B3): one fork via the
        standard pipeline, inspector seeded the way Sandbox.fork does
        (there is no SDK Sandbox object on this path)."""
        [fork_id] = self.fork_sandbox(source_sandbox_id, count=1, gate_effects=False)
        try:
            self.system.inspector.upsert_snapshot(
                SandboxSnapshot(
                    sandbox_id=fork_id,
                    runtime_name=self.runtime.name,
                    is_running=True,
                    process_changed=False,
                    filesystem_changed=False,
                    observed_at=utc_now(),
                )
            )
        except Exception:
            logger.debug("Failed to seed inspector for txn fork=%s", fork_id, exc_info=True)
        return fork_id

    def _destroy_txn_fork(self, fork_sandbox_id: SandboxId) -> None:
        """Teardown hook mirroring Sandbox.kill's fork cleanup chain —
        each step independently best-effort."""
        try:
            self.system.release_fork(fork_sandbox_id)
        except Exception:
            logger.exception("Txn fork release failed fork=%s", fork_sandbox_id)
        try:
            self.runtime.delete(fork_sandbox_id)
        except Exception:
            logger.exception("Txn fork runtime delete failed fork=%s", fork_sandbox_id)
        try:
            self.unregister_upstream(fork_sandbox_id)
        except Exception:
            logger.debug("Txn fork upstream cleanup failed", exc_info=True)
        try:
            self.release_network_lease(fork_sandbox_id)
        except Exception:
            logger.debug("Txn fork lease cleanup failed", exc_info=True)

    # ------------------------------------------------------------------
    # Sandbox registry — used so engine.stop() can clean up.
    # ------------------------------------------------------------------

    def _register_sandbox(self, sandbox: "Sandbox") -> None:
        with self._sandbox_lock:
            self._sandboxes[sandbox.sandbox_id] = sandbox

    def _unregister_sandbox(self, sandbox: "Sandbox") -> None:
        with self._sandbox_lock:
            self._sandboxes.pop(sandbox.sandbox_id, None)


# ---------------------------------------------------------------------------
# Default engine — used when the user calls `Sandbox(...)` without
# explicitly providing one. Lazily connects to the running daemon.
# Auto-starting the daemon is intentionally not supported (matches
# docker's behaviour: the user is responsible for `crab daemon start`).
# ---------------------------------------------------------------------------


_DEFAULT_ENGINE: Engine | None = None
_DEFAULT_ENGINE_LOCK = threading.Lock()


def get_default_engine(socket: str | os.PathLike[str] | None = None) -> Engine:
    """Return the process-wide default engine, lazily connecting to the
    running Crab daemon.

    Raises `FileNotFoundError` (or a `RuntimeError` describing the
    daemon's error) if the daemon isn't reachable. There is no
    in-process fallback — start the daemon with
    `crab daemon start` (or `python -m crab.daemon`) before
    creating any `Sandbox`."""
    global _DEFAULT_ENGINE
    with _DEFAULT_ENGINE_LOCK:
        if _DEFAULT_ENGINE is None:
            _DEFAULT_ENGINE = Engine.connect(socket)
        return _DEFAULT_ENGINE


def set_default_engine(engine: Engine | None) -> Engine | None:
    """Replace the process-wide default engine. Returns the previous one
    (caller is responsible for stopping it if needed)."""
    global _DEFAULT_ENGINE
    with _DEFAULT_ENGINE_LOCK:
        previous = _DEFAULT_ENGINE
        _DEFAULT_ENGINE = engine
    return previous


def shutdown_default_engine() -> None:
    """Drop the default engine handle. Safe to call multiple times.

    This does **not** stop the daemon — the daemon's lifecycle is
    independent of the SDK process and is managed via
    `crab daemon stop` (or `POST /shutdown`)."""
    global _DEFAULT_ENGINE
    with _DEFAULT_ENGINE_LOCK:
        engine = _DEFAULT_ENGINE
        _DEFAULT_ENGINE = None
    if engine is not None:
        try:
            engine.stop()
        except Exception:
            logger.debug("default engine stop failed", exc_info=True)


__all__ = [
    "Engine",
    "EngineConfig",
    "get_default_engine",
    "set_default_engine",
    "shutdown_default_engine",
]
