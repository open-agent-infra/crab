#!/usr/bin/env python3
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import copy
import errno
import hashlib
import json
import logging
import os
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crab import (
    CrabRequestInterceptorServer,
    CrabSystem,
    AdapterFileSystemCWorker,
    AdapterFileSystemRWorker,
    AdapterProcessCWorker,
    AdapterProcessRWorker,
    CRExecutor,
    CRScheduler,
    CheckpointId,
    CheckpointJob,
    CheckpointManager,
    CheckpointManifest,
    CompositeRequestInterceptorHook,
    DefaultCWorker,
    DefaultRWorker,
    EBPFSandboxInspector,
    HostInspectorServiceClient,
    RemoteSandboxInspector,
    ExecutorConfig,
    InMemoryRequestStateStore,
    InMemorySchedulerStateStore,
    LocalCheckpointManager,
    RequestInterceptorHook,
    RequestAwareSandboxInspector,
    RuncRestoreOptions,
    RuncRuntime,
    RuncRuntimePaths,
    RuncRuntimeOptions,
    SandboxDescription,
    SandboxExecResult,
    SandboxId,
    SandboxSnapshot,
    SchedulerConfig,
    StorageConfig,
    TelemetryConfig,
    TelemetrySink,
    TelemetryRequestInterceptorHook,
    JobId,
    build_configured_telemetry_sink,
)
from crab.models import ArtifactPayload, FailureCode, JobStatus, RestoreResult, utc_now
from crab.telemetry import start_operation
from crab.workers.composite import resolve_restore_manifest
from crab.host_inspector.fs_helper import LibbpfFilesystemMonitor
from crab.host_inspector.runtime_resolver import RuntimeResolver
from crab.host_inspector.server import HostInspectorDaemon, HostInspectorServer
from integrations.agents import BaseAgent, SandboxHandle, TaskConfig, TaskDescription, build_agent_registry
from integrations.llm_services import (
    BenchmarkLLMRouterClient,
    default_llm_service_type_for_agent,
    serve_benchmark_llm_router,
    validate_llm_service_type,
)
from integrations.sandboxes.runtime import launcher as sandbox_launcher
from integrations.sandboxes.runtime import network as sandbox_network
from integrations.sandboxes.runtime import bundle as sandbox_bundle
from integrations.sandboxes.runtime.bundle import SandboxResourceLimits
from integrations.sandboxes.runtime import compose as sandbox_compose
from integrations.sandboxes.runtime import image as sandbox_image
from integrations.sandboxes import swebench as swebench_support
from integrations.sandboxes.iflow import DOCKERFILE_PATH as IFLOW_DOCKERFILE_PATH
from integrations.sandboxes.simulated import DOCKERFILE_PATH as SIMULATED_DOCKERFILE_PATH
from benchmarks import core as benchmark_core
from benchmarks.monitoring import BenchmarkResourceMonitor
from benchmarks import support as benchmark_support

logger = logging.getLogger(__name__)

_HOST_INSPECTOR_HOST = "127.0.0.1"
_HOST_INSPECTOR_PORT = 9782
_DEFAULT_IMAGE_CACHE_ROOT = ROOT / ".cache" / "crab" / "images"
_SHARED_ROOTFS_KEY_METADATA_KEY = "shared_rootfs_key"
_SHARED_ROOTFS_PERSIST_METADATA_KEY = "shared_rootfs_persist"
_RUNTIME_LAUNCH_METADATA_LIST_KEYS = frozenset(
    {
        "rootfs_init_dirs",
        "rootfs_copy_paths",
        "rootfs_post_clone_copy_paths",
        "host_inspector_ignore_process_rules",
        "host_inspector_ignored_path_prefixes",
    }
)
_VERIFICATION_UV_TRANSIENT_ERROR_FRAGMENTS = (
    "Temporary failure resolving",
    "Failed to fetch",
    "Could not get lock",
    "Unable to lock directory",
    "No installation candidate",
    "Unable to locate package",
)
_VERIFICATION_NETWORK_READY_TIMEOUT_SECONDS = 60.0


def checkpoint_guard_from_inspector(inspector):
    def guard(job):
        try:
            snapshot = inspector.inspect(job.sandbox_id)
        except Exception:
            return True, None
        if snapshot.is_running:
            return True, None
        return False, "sandbox_not_running"

    return guard


def require_binaries() -> None:
    required = ["docker", "runc", "criu", "zfs", "zpool", "ip", "iptables", "sysctl"]
    missing = [name for name in required if shutil.which(name) is None]
    if missing:
        raise SystemExit(f"missing required binaries: {', '.join(missing)}")


def wait_for_http_json(url: str, *, timeout_s: float = 30.0) -> dict[str, object]:
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


def _host_resolv_conf_path() -> Path | None:
    host_resolv_conf = Path("/run/systemd/resolve/resolv.conf")
    if host_resolv_conf.is_file():
        return host_resolv_conf
    host_resolv_conf = Path("/etc/resolv.conf")
    if host_resolv_conf.is_file():
        return host_resolv_conf
    return None


def _merge_runtime_launch_metadata(*parts: dict[str, object] | None) -> dict[str, object]:
    merged: dict[str, object] = {}
    for part in parts:
        if not part:
            continue
        for key, value in part.items():
            if key not in _RUNTIME_LAUNCH_METADATA_LIST_KEYS:
                merged[key] = value
                continue
            incoming = value if isinstance(value, list) else []
            current = list(merged.get(key, []))
            for item in incoming:
                if item not in current:
                    current.append(item)
            merged[key] = current
    return merged


def _normalize_benchmark_run_name(raw_value: object) -> str | None:
    if raw_value is None:
        return None
    value = str(raw_value).strip()
    path = Path(value)
    if not value or path.is_absolute() or len(path.parts) != 1 or value in {".", ".."}:
        raise ValueError("benchmark_run_name must be a single non-empty directory name")
    return value


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _zpool_exists(pool_name: str) -> bool:
    result = subprocess.run(
        ["zpool", "list", "-H", "-o", "name", pool_name],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def _zfs_dataset_exists(dataset: str) -> bool:
    result = subprocess.run(
        ["zfs", "list", "-H", "-o", "name", dataset],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0

@dataclass(frozen=True)
class AgentSandboxImage:
    agent_type: str
    image_tag: str
    exported_rootfs: Path
    image_defaults: sandbox_image.ImageRuntimeDefaults


@dataclass
class PreparedBenchmarkSandbox:
    handle: SandboxHandle
    sandbox_name: str
    task_record: benchmark_support.BenchmarkTaskRecord
    prelaunch_task_run: BaseAgent | None = None
    work_dir_host_path: Path | None = None
    runtime_prepared: bool = False
    wait_for_ready_before_task_start: bool = False
    wait_for_ready_after_task_start: bool = False
    emit_ready_event: bool = True
    runtime_launched: bool = False


class _SpeculativeSandboxController:
    # Class-level throttle: caps total in-flight preforks across ALL active
    # sandboxes so the background warming work cannot saturate disk/CPU and
    # starve the active workload. Initialized lazily by `_acquire_global_slot`
    # off the first kick that supplies a non-zero `prefork_max_concurrent`.
    # Per-sandbox dedup (`_prefork_in_flight`) and min-interval still apply
    # in addition to this cap.
    _global_prefork_lock: "threading.Lock" = threading.Lock()
    _global_prefork_semaphore: "threading.BoundedSemaphore | None" = None
    _global_prefork_max: int = 0

    @classmethod
    def _acquire_global_slot(cls, max_concurrent: int) -> bool:
        """Try to claim a global prefork slot. Non-blocking: returns False
        when the cap is full (caller should skip this prefork). Returns True
        unconditionally when ``max_concurrent <= 0`` (throttle disabled)."""
        if max_concurrent <= 0:
            return True
        with cls._global_prefork_lock:
            if cls._global_prefork_semaphore is None or cls._global_prefork_max != max_concurrent:
                cls._global_prefork_semaphore = threading.BoundedSemaphore(max_concurrent)
                cls._global_prefork_max = max_concurrent
            sem = cls._global_prefork_semaphore
        return sem.acquire(blocking=False)

    @classmethod
    def _release_global_slot(cls, max_concurrent: int) -> None:
        if max_concurrent <= 0:
            return
        with cls._global_prefork_lock:
            sem = cls._global_prefork_semaphore
        if sem is None:
            return
        try:
            sem.release()
        except ValueError:
            # Released more than acquired (e.g. throttle reconfigured mid-run);
            # nothing to do.
            pass

    def __init__(
        self,
        harness: "RealHostScenarioHarness",
        active_sandbox: SandboxHandle,
        *,
        fork_reuse_enabled: bool = False,
        eager_cleanup_on_reject: bool = False,
        background_prefork_enabled: bool = False,
        prefork_min_interval_seconds: float = 0.0,
        prefork_wait_timeout_seconds: float = 0.0,
        prefork_max_concurrent_global: int = 0,
    ) -> None:
        self._harness = harness
        self._active_sandbox = active_sandbox
        self._fork_name_root = str(active_sandbox.sandbox_id)
        self._lock = threading.Lock()
        self._cached_fork: SandboxHandle | None = None
        # Tracks which checkpoint the cached fork was restored from. Set by
        # both the prefork path and a successful synchronous ensure_fork; used
        # by ensure_fork to detect a stale cache when a newer checkpoint has
        # landed between caching and the next spec turn.
        self._cached_fork_checkpoint_id: CheckpointId | None = None
        self._fork_counter = 0
        self._fork_epoch = 0
        self._fork_reuse_enabled = bool(fork_reuse_enabled)
        self.eager_cleanup_on_reject = bool(eager_cleanup_on_reject)
        self._background_prefork_enabled = bool(background_prefork_enabled)
        self._prefork_min_interval_seconds = max(0.0, float(prefork_min_interval_seconds))
        self._prefork_wait_timeout_seconds = max(0.0, float(prefork_wait_timeout_seconds))
        self._prefork_max_concurrent_global = max(0, int(prefork_max_concurrent_global))
        self._prefork_skipped_throttled = 0
        self._prefork_executor: ThreadPoolExecutor | None = None
        if self._background_prefork_enabled:
            self._prefork_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix=f"crab-prefork-{self._fork_name_root}",
            )
        self._prefork_future: Future[None] | None = None
        self._prefork_in_flight = False
        self._last_prefork_at = 0.0
        self._prefork_hits = 0
        self._prefork_wasted = 0

    def ensure_fork(self):
        # If a background prefork is in flight, give it a brief chance to
        # land in the cache so the synchronous-clone path doesn't race
        # ahead and double the work. Skipped (timeout=0) when prefork is
        # off or the caller hasn't opted in.
        wait_timeout = self._prefork_wait_timeout_seconds
        if (
            self._background_prefork_enabled
            and wait_timeout > 0
            and self._prefork_future is not None
            and not self._prefork_future.done()
        ):
            try:
                self._prefork_future.result(timeout=wait_timeout)
            except (FutureTimeoutError, Exception):
                pass
        with self._lock:
            storage = self._harness.storage
            if storage is None:
                return None
            checkpoints = storage.list_checkpoints(self._active_sandbox.sandbox_id)
            if not checkpoints:
                # No checkpoints yet — discard any stale cached fork to
                # avoid handing back a fork rooted at a vanished base.
                stale = self._cached_fork
                self._cached_fork = None
                self._cached_fork_checkpoint_id = None
                stale_fork_to_destroy = stale
                if stale_fork_to_destroy is None:
                    return None
                # Fall through; we'll destroy below outside the lock.
                self._fork_epoch += 1
            else:
                stale_fork_to_destroy = None
            cached_fork = self._cached_fork
            cached_id = self._cached_fork_checkpoint_id
            if cached_fork is not None:
                latest = checkpoints[-1]
                if cached_id is None or cached_id == latest:
                    # Treat untagged caches (legacy fork-reuse path) as
                    # fresh; tagged caches must match the latest checkpoint
                    # or we drop them.
                    self._prefork_hits += 1
                    return cached_fork, False, True
                # Stale cached prefork — newer checkpoint landed since.
                # Drop the cache; destroy outside the lock.
                stale_fork_to_destroy = cached_fork
                self._cached_fork = None
                self._cached_fork_checkpoint_id = None
                self._fork_epoch += 1
                self._prefork_wasted += 1
        if stale_fork_to_destroy is not None:
            try:
                self._harness.destroy_sandbox_dataset(stale_fork_to_destroy)
            except Exception:
                logger.exception(
                    "Failed to destroy stale cached fork sandbox=%s",
                    stale_fork_to_destroy.sandbox_id,
                )
        with self._lock:
            storage = self._harness.storage
            if storage is None:
                return None
            checkpoints = storage.list_checkpoints(self._active_sandbox.sandbox_id)
            if not checkpoints:
                return None
            checkpoint_id = checkpoints[-1]
            # Refuse to base a fork on a partial (fs-only) checkpoint.
            # `resolve_checkpoint_copy_plan` would walk back to find an
            # older full checkpoint and stitch its process image to the
            # newer fs snapshot, but the older process state's open file
            # descriptors / mmap'd inodes (`var/lib/dpkg/updates/tmp.i`,
            # the bash-mmap'd libc/ld-linux/gconv-modules ghost dumps,
            # etc.) frequently no longer match the newer fs's view of
            # the same paths once the agent has run anything that
            # rewrites system state (`apt-get install`, `make install`,
            # dpkg update, …). CRIU then aborts the restore with
            # `Can't open file ... on restore: No such file or
            # directory` / `Unable to open fd=N id=0xM` — observed in
            # the spec.compcert smoke as 17/19 fork restores failing.
            # Even when the composite restore happens to *succeed*
            # (10-sandbox spec.spec_friendly), the resulting fork's
            # process state is older than the active's actual state,
            # so a subsequent promote rolls the in-flight `make all`
            # back to the older composite point and the build never
            # reaches `make install`. Skip spec for this turn instead
            # — the next turn will still see the latest full checkpoint
            # if the inspector has emitted process_changed since then.
            try:
                latest_manifest = storage.get_manifest(
                    self._active_sandbox.sandbox_id, checkpoint_id
                )
            except FileNotFoundError:
                return None
            if not latest_manifest.process_artifacts:
                logger.debug(
                    "Skipping spec fork creation: latest checkpoint is fs-only "
                    "(process state stale relative to fs snapshot) "
                    "active=%s checkpoint=%s",
                    self._active_sandbox.sandbox_id,
                    checkpoint_id,
                )
                return None
            self._fork_counter += 1
            fork_epoch = self._fork_epoch
            fork_name = f"{self._fork_name_root}-spec-{self._fork_counter}"
        fork = self._harness.clone_checkpoint_to_fork(
            self._active_sandbox,
            checkpoint_id,
            fork_name,
        )
        if fork.task_run is not None:
            fork.task_run.configure_bundle()
        try:
            restore_result = self._harness.restore_once(fork, checkpoint_id)
        except Exception:
            logger.exception(
                "Failed to restore speculative fork source=%s target=%s checkpoint=%s",
                self._active_sandbox.sandbox_id,
                fork.sandbox_id,
                checkpoint_id,
            )
            self._harness.destroy_sandbox_dataset(fork)
            return None
        if restore_result.status.value != "succeeded":
            logger.warning(
                "Speculative fork restore did not succeed source=%s target=%s checkpoint=%s status=%s message=%s",
                self._active_sandbox.sandbox_id,
                fork.sandbox_id,
                checkpoint_id,
                restore_result.status.value,
                restore_result.message,
            )
            self._harness.destroy_sandbox_dataset(fork)
            return None
        invalidate_stale_fork = False
        with self._lock:
            if fork_epoch != self._fork_epoch:
                invalidate_stale_fork = True
            else:
                self._cached_fork = fork
                self._cached_fork_checkpoint_id = checkpoint_id
        if invalidate_stale_fork:
            logger.debug(
                "Discarding stale speculative fork after cache invalidation source=%s target=%s checkpoint=%s",
                self._active_sandbox.sandbox_id,
                fork.sandbox_id,
                checkpoint_id,
            )
            self._harness.destroy_sandbox_dataset(fork)
            return None
        return fork, True, False

    def promote_fork(self, fork: SandboxHandle) -> None:
        with self._lock:
            self._harness._copy_llm_router_state(
                source_sandbox_id=self._active_sandbox.sandbox_id,
                target_sandbox_id=fork.sandbox_id,
            )
            # Accepted speculative forks are ZFS clones of the current active
            # sandbox. Promote the clone before we swap handles so releasing the
            # old sandbox can delete its dataset without cascading into the new
            # active sandbox.
            self._harness._promote_filesystem_dataset(fork.sandbox_id)
            self._harness._swap_speculative_sandbox_state(self._active_sandbox, fork)
            if self._fork_reuse_enabled:
                self._cached_fork = fork

    def state_changed(self, sandbox: SandboxHandle) -> bool:
        return self._harness.sandbox_state_changed(sandbox)

    def cancel_fork_runtime(self, fork: SandboxHandle) -> None:
        """Force-kill the fork's container immediately, without
        cleaning up its ZFS dataset / network lease / inspector
        registration. Used on the spec-mismatch critical path so the
        wasted spec_exec on the fork stops competing with the about-
        to-run oracle exec on the active sandbox for host CPU/IO. The
        rest of the cleanup (dataset destroy, lease release, etc.) is
        left to the deferred `_finalize_speculative_fork` →
        `release_fork` chain, which runs off the agent's critical
        path."""
        if fork is None:
            return
        delete = getattr(self._harness, "_delete_runtime", None)
        if not callable(delete):
            return
        try:
            delete(fork.sandbox_id)
        except Exception:
            logger.debug(
                "Failed to cancel fork runtime sandbox=%s",
                fork.sandbox_id,
                exc_info=True,
            )

    def invalidate_fork(self) -> None:
        with self._lock:
            fork = self._cached_fork
            self._cached_fork = None
            self._cached_fork_checkpoint_id = None
            self._fork_epoch += 1
        if fork is None:
            return
        try:
            self._harness.destroy_sandbox_dataset(fork)
        except Exception:
            logger.exception("Failed to invalidate speculative fork sandbox=%s", fork.sandbox_id)

    def release_fork(self, fork: SandboxHandle, *, reusable: bool, accepted: bool) -> None:
        _ = accepted
        with self._lock:
            if reusable and self._fork_reuse_enabled:
                self._cached_fork = fork
                # Legacy fork-reuse caches stay untagged (None); ensure_fork
                # treats those as fresh for backward compatibility.
                self._cached_fork_checkpoint_id = None
                return
            if self._cached_fork is fork:
                self._cached_fork = None
                self._cached_fork_checkpoint_id = None
        try:
            self._harness.destroy_sandbox_dataset(fork)
        except Exception:
            logger.exception("Failed to delete speculative fork sandbox=%s", fork.sandbox_id)

    def kick_prefork(self) -> None:
        """Speculatively prepare a fork from the latest checkpoint so the
        next ``ensure_fork()`` returns instantly instead of paying clone +
        restore latency. Safe to call repeatedly: deduplicates against an
        in-flight prefork, an already-warm cache, and a min-interval
        debounce. No-op when ``background_prefork_enabled`` is False.
        """
        if not self._background_prefork_enabled:
            return
        executor = self._prefork_executor
        if executor is None:
            return
        storage = self._harness.storage
        if storage is None:
            return
        with self._lock:
            if self._prefork_in_flight:
                return
            if self._cached_fork is not None:
                # Already have a fork ready; nothing to prefetch.
                return
            now = time.monotonic()
            if (
                self._prefork_min_interval_seconds > 0
                and (now - self._last_prefork_at) < self._prefork_min_interval_seconds
            ):
                return
            try:
                checkpoints = storage.list_checkpoints(self._active_sandbox.sandbox_id)
            except Exception:
                logger.debug(
                    "Skipping prefork: storage list_checkpoints failed sandbox=%s",
                    self._active_sandbox.sandbox_id,
                    exc_info=True,
                )
                return
            if not checkpoints:
                return
            target_checkpoint_id = checkpoints[-1]
            try:
                manifest = storage.get_manifest(
                    self._active_sandbox.sandbox_id, target_checkpoint_id
                )
            except FileNotFoundError:
                return
            if not manifest.process_artifacts:
                # Same fs-only-skip rule as ensure_fork.
                return
            # Per-sandbox state is committed; now compete for the global
            # in-flight slot. If full, skip this kick: another sandbox is
            # already burning the disk/CPU on its own prefork, and queueing
            # a second one would only deepen the contention that hurts
            # active-sandbox throughput. Counted in telemetry as a
            # throttle-skip rather than a wasted prefork.
            if not self._acquire_global_slot(self._prefork_max_concurrent_global):
                self._prefork_skipped_throttled += 1
                return
            self._fork_counter += 1
            fork_epoch = self._fork_epoch
            fork_name = f"{self._fork_name_root}-prefork-{self._fork_counter}"
            self._prefork_in_flight = True
            self._last_prefork_at = now
        future = executor.submit(
            self._run_prefork,
            target_checkpoint_id,
            fork_name,
            fork_epoch,
        )
        self._prefork_future = future

    def _run_prefork(
        self,
        checkpoint_id: CheckpointId,
        fork_name: str,
        fork_epoch: int,
    ) -> None:
        fork: SandboxHandle | None = None
        try:
            fork = self._harness.clone_checkpoint_to_fork(
                self._active_sandbox,
                checkpoint_id,
                fork_name,
            )
            if fork.task_run is not None:
                fork.task_run.configure_bundle()
            restore_result = self._harness.restore_once(fork, checkpoint_id)
            if restore_result.status.value != "succeeded":
                logger.warning(
                    "Prefork restore did not succeed source=%s target=%s checkpoint=%s status=%s",
                    self._active_sandbox.sandbox_id,
                    fork.sandbox_id,
                    checkpoint_id,
                    restore_result.status.value,
                )
                self._harness.destroy_sandbox_dataset(fork)
                return
        except Exception:
            logger.exception(
                "Prefork preparation failed sandbox=%s checkpoint=%s",
                self._active_sandbox.sandbox_id,
                checkpoint_id,
            )
            if fork is not None:
                try:
                    self._harness.destroy_sandbox_dataset(fork)
                except Exception:
                    logger.exception(
                        "Failed to destroy failed prefork sandbox=%s",
                        fork.sandbox_id,
                    )
            return
        finally:
            self._release_global_slot(self._prefork_max_concurrent_global)
            with self._lock:
                self._prefork_in_flight = False
        # Commit the cache only if no invalidation has bumped the epoch
        # since prefork started. Otherwise destroy the just-restored fork.
        commit = False
        with self._lock:
            if fork_epoch == self._fork_epoch and self._cached_fork is None:
                self._cached_fork = fork
                self._cached_fork_checkpoint_id = checkpoint_id
                commit = True
        if not commit:
            self._prefork_wasted += 1
            try:
                self._harness.destroy_sandbox_dataset(fork)
            except Exception:
                logger.exception(
                    "Failed to destroy stale prefork sandbox=%s",
                    fork.sandbox_id,
                )

    def shutdown(self) -> None:
        """Idempotent shutdown of the prefork executor. Called by the
        harness when the active sandbox is destroyed so we don't leak
        the worker thread or block process exit."""
        executor = self._prefork_executor
        self._prefork_executor = None
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)


class RealHostScenarioHarness:
    REPLAY_COMPLETION_TASK_FUTURE_GRACE_SECONDS = 10.0

    @staticmethod
    def _verification_uv_python_shim_script() -> str:
        return """
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback


def _resolve_python(requested: str | None) -> str:
    candidates: list[str] = []
    if requested:
        candidates.append(requested)
        if requested and requested[0].isdigit():
            candidates.append(f"python{requested}")
    venv_root = os.environ.get("VIRTUAL_ENV")
    if venv_root:
        candidates.extend(
            [
                str(Path(venv_root) / "bin" / "python"),
                str(Path(venv_root) / "bin" / "python3"),
            ]
        )
    candidates.extend(["python3", "python", sys.executable])
    for candidate in candidates:
        if not candidate:
            continue
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    raise SystemExit("uv shim could not find a python interpreter")


def _python_version_tag(python: str) -> str:
    result = subprocess.run(
        [
            python,
            "-c",
            "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _venv_site_packages(venv_root: str, python: str) -> Path:
    version_tag = _python_version_tag(python)
    return Path(venv_root) / "lib" / f"python{version_tag}" / "site-packages"


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def _create_lightweight_venv(destination: str, python: str) -> int:
    venv_root = Path(destination)
    shutil.rmtree(venv_root, ignore_errors=True)
    bin_dir = venv_root / "bin"
    site_packages = _venv_site_packages(destination, python)
    bin_dir.mkdir(parents=True, exist_ok=True)
    site_packages.mkdir(parents=True, exist_ok=True)
    (venv_root / ".crab_fake_venv").write_text("", encoding="utf-8")
    python_wrapper = (
        "#!/bin/sh\\n"
        f'PYTHONPATH="{site_packages}:${{PYTHONPATH:-}}" exec "{python}" "$@"\\n'
    )
    _write_executable(bin_dir / "python", python_wrapper)
    _write_executable(bin_dir / "python3", python_wrapper)
    _write_executable(
        bin_dir / "pytest",
        (
            "#!/bin/sh\\n"
            f'exec "{bin_dir / "python"}" -m pytest "$@"\\n'
        ),
    )
    (bin_dir / "activate").write_text(
        "\\n".join(
            [
                f'VIRTUAL_ENV="{venv_root}"',
                "export VIRTUAL_ENV",
                'PATH="$VIRTUAL_ENV/bin:$PATH"',
                "export PATH",
                f'PYTHONPATH="{site_packages}:${{PYTHONPATH:-}}"',
                "export PYTHONPATH",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return 0


def _consume_python_flag(args: list[str], index: int) -> tuple[str | None, int]:
    arg = args[index]
    if arg in {"-p", "--python"}:
        if index + 1 >= len(args):
            raise SystemExit("uv shim expected a value after --python/-p")
        return args[index + 1], index + 2
    if arg.startswith("--python="):
        return arg.split("=", 1)[1], index + 1
    return None, index


def _run_venv(args: list[str]) -> int:
    python = None
    forwarded: list[str] = []
    i = 0
    while i < len(args):
        requested, next_i = _consume_python_flag(args, i)
        if next_i != i:
            python = requested or python
            i = next_i
            continue
        forwarded.append(args[i])
        i += 1
    if not forwarded:
        raise SystemExit("uv shim: venv requires a destination")
    resolved_python = _resolve_python(python)
    status = subprocess.call([resolved_python, "-m", "venv", *forwarded])
    if status == 0:
        return 0
    return _create_lightweight_venv(forwarded[-1], resolved_python)


def _run_pip(args: list[str]) -> int:
    python = None
    forwarded: list[str] = []
    i = 0
    while i < len(args):
        requested, next_i = _consume_python_flag(args, i)
        if next_i != i:
            python = requested or python
            i = next_i
            continue
        if args[i] == "--system":
            i += 1
            continue
        forwarded.append(args[i])
        i += 1
    if not forwarded:
        raise SystemExit("uv shim: pip requires arguments")
    resolved_python = _resolve_python(python)
    venv_root = os.environ.get("VIRTUAL_ENV")
    if forwarded[:1] == ["install"] and venv_root:
        site_packages = _venv_site_packages(venv_root, resolved_python)
        site_packages.mkdir(parents=True, exist_ok=True)
        return subprocess.call(
            [
                resolved_python,
                "-m",
                "pip",
                "install",
                "--target",
                str(site_packages),
                *forwarded[1:],
            ]
        )
    return subprocess.call([resolved_python, "-m", "pip", *forwarded])


def _run_python(args: list[str]) -> int:
    if args[:1] == ["pin"]:
        return 0
    return subprocess.call([_resolve_python(None), *args])


def _using_fake_venv() -> bool:
    venv_root = os.environ.get("VIRTUAL_ENV")
    return bool(venv_root and (Path(venv_root) / ".crab_fake_venv").exists())


def _run_command(args: list[str]) -> int:
    if not args:
        raise SystemExit("uv shim: run requires a command")
    if args[0] == "pytest":
        pytest_executable = shutil.which("pytest")
        if pytest_executable is not None:
            return subprocess.call(args)
        if os.environ.get("VIRTUAL_ENV"):
            return subprocess.call([_resolve_python(None), "-m", "pytest", *args[1:]])
        return _run_pytest_fallback(args[1:])
    return subprocess.call(args)


def _run_pytest_fallback(args: list[str]) -> int:
    test_files = [arg for arg in args if arg.endswith(".py") and not arg.startswith("-")]
    if not test_files:
        return 0
    failures: list[tuple[str, str, BaseException]] = []
    test_results: list[tuple[str, str, bool, str | None]] = []
    total = 0
    for index, test_file in enumerate(test_files):
        module_path = Path(test_file).resolve()
        spec = importlib.util.spec_from_file_location(
            f"crab_pytest_fallback_{index}",
            module_path,
        )
        if spec is None or spec.loader is None:
            raise SystemExit(f"uv shim: unable to load test module {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        for name in sorted(dir(module)):
            if not name.startswith("test_"):
                continue
            candidate = getattr(module, name)
            if not callable(candidate):
                continue
            total += 1
            try:
                candidate()
                test_results.append((str(module_path), name, True, None))
            except BaseException as exc:  # noqa: BLE001
                failures.append((str(module_path), name, exc))
                test_results.append(
                    (str(module_path), name, False, f"{exc.__class__.__name__}: {exc}")
                )
                traceback.print_exc()
    print("=========================== short test summary info ============================")
    for module_path, name, passed, details in test_results:
        if passed:
            print(f"PASSED {module_path}::{name}")
        else:
            print(f"FAILED {module_path}::{name} - {details}")
    if failures:
        print(f"{len(failures)}/{total} fallback tests failed", file=sys.stderr)
        return 1
    print(f"{total} fallback tests passed")
    return 0


def main(argv: list[str]) -> int:
    if not argv:
        raise SystemExit("uv shim: missing subcommand")
    command, *args = argv
    if command == "venv":
        return _run_venv(args)
    if command == "pip":
        return _run_pip(args)
    if command == "run":
        return _run_command(args)
    if command == "python":
        return _run_python(args)
    raise SystemExit(f"uv shim: unsupported subcommand {command!r}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
""".strip()

    @staticmethod
    def _verification_uv_bootstrap_script() -> str:
        python_shim = RealHostScenarioHarness._verification_uv_python_shim_script()
        return (
            """
set -euo pipefail
shim_bin="$HOME/.local/crab-verification/bin"
wait_for_apt_lock() {
  while pgrep -x apt-get >/dev/null 2>&1 || pgrep -x apt >/dev/null 2>&1 || pgrep -x dpkg >/dev/null 2>&1; do
    sleep 1
  done
}
need_python_packages=0
# `python3 -m venv --help` succeeds even when ensurepip is missing (the
# `venv` module ships with the stdlib, but actually creating a venv
# requires the python3-venv package's ensurepip data files). Probe
# ensurepip directly so we don't silently leave the container in a
# state where the verification's `uv venv` later dies with
# "ensurepip is not available".
if ! python3 -c "import ensurepip" >/dev/null 2>&1; then
  need_python_packages=1
fi
if ! python3 -m pip --version >/dev/null 2>&1; then
  need_python_packages=1
fi
install -d -m 755 "$shim_bin"
export PATH="$shim_bin:$PATH"
cat > "$shim_bin/apt-get" <<'EOF'
#!/bin/sh
REAL_APT_GET=/usr/bin/apt-get
APT_UPDATE_MARKER=/var/lib/crab-verification/apt-update.ok
first_cmd=""
for arg in "$@"; do
  case "$arg" in
    -*) ;;
    *)
      first_cmd="$arg"
      break
      ;;
  esac
done
if [ "$first_cmd" = "update" ]; then
  # Skip caller-driven `apt-get update` to keep verify fast — but we'll
  # still run it on demand below before any real install, so the cached
  # package index can't pin us to a version Ubuntu has already retired
  # (observed: libpython3.12-dev_3.12.3-1ubuntu0.12 → 404 once 0.13
  # rolled out). One `apt-get update` per verify, only when we'd
  # otherwise hit the network anyway.
  exit 0
fi
if [ "$first_cmd" = "install" ]; then
  can_skip=1
  for arg in "$@"; do
    case "$arg" in
      install|-y|-q|-qq|--yes|--no-install-recommends)
        ;;
      curl)
        if ! command -v /usr/bin/curl >/dev/null 2>&1 && ! command -v curl >/dev/null 2>&1; then
          can_skip=0
        fi
        ;;
      python3-venv)
        if ! python3 -c "import ensurepip" >/dev/null 2>&1; then
          can_skip=0
        fi
        ;;
      python3-pip)
        if ! python3 -m pip --version >/dev/null 2>&1; then
          can_skip=0
        fi
        ;;
      *)
        can_skip=0
        ;;
    esac
  done
  if [ "$can_skip" -eq 1 ]; then
    exit 0
  fi
  if [ ! -f "$APT_UPDATE_MARKER" ]; then
    install -d -m 755 "$(dirname "$APT_UPDATE_MARKER")" 2>/dev/null || true
    if "$REAL_APT_GET" update -qq >/dev/null 2>&1; then
      : > "$APT_UPDATE_MARKER" 2>/dev/null || true
    fi
  fi
fi
exec "$REAL_APT_GET" "$@"
EOF
chmod 755 "$shim_bin/apt-get"
cat > "$shim_bin/curl" <<'EOF'
#!/bin/sh
for arg in "$@"; do
  case "$arg" in
    *astral.sh/uv/*/install.sh*)
      printf '#!/bin/sh\nexit 0\n'
      exit 0
      ;;
  esac
done
exec /usr/bin/curl "$@"
EOF
chmod 755 "$shim_bin/curl"
if [ "$need_python_packages" -eq 1 ]; then
  export DEBIAN_FRONTEND=noninteractive
  wait_for_apt_lock
  apt-get update >/dev/null
  wait_for_apt_lock
  apt-get install -y python3-venv python3-pip >/dev/null
fi
install -d -m 755 "$HOME/.local/bin"
cat > "$HOME/.local/bin/env" <<'EOF'
export PATH="$HOME/.local/crab-verification/bin:$HOME/.local/bin:$PATH"
EOF
if [ ! -x "$HOME/.local/bin/uv" ]; then
cat > "$HOME/.local/bin/uv" <<'EOF'
#!/usr/bin/env python3
__CRAB_VERIFICATION_UV_PYTHON_SHIM__
EOF
chmod 755 "$HOME/.local/bin/uv"
fi
""".replace("__CRAB_VERIFICATION_UV_PYTHON_SHIM__", python_shim).strip()
        )

    @staticmethod
    def _verification_network_probe_script() -> str:
        return """
python3 - <<'PY'
import socket
import sys

targets = [
    ("archive.ubuntu.com", 80),
    ("astral.sh", 443),
]
errors = []
for host, port in targets:
    try:
        socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError as exc:
        errors.append(f"{host}:{port} {exc}")
if errors:
    print("\\n".join(errors), file=sys.stderr)
    raise SystemExit(1)
PY
""".strip()

    @property
    def _active_runtime(self):
        if self.runtime is not None:
            return self.runtime
        if self.system is None:
            return None
        runtime = getattr(self.system, "runtime", None)
        if runtime is not None:
            return runtime
        return getattr(self.system, "sandbox_manager", None)

    @property
    def sandbox_manager(self):
        return self.runtime

    @sandbox_manager.setter
    def sandbox_manager(self, value) -> None:
        self.runtime = value

    def __init__(
        self,
        *,
        provider: str,
        transfer_delay_ms: float,
        scheduler_config: SchedulerConfig,
        scheduler_policy,
        checkpoint_manager_factory,
        max_workers: int,
        executor_config: ExecutorConfig | None = None,
        auto_cr: bool = False,
        work_dir_host_root: Path | None = None,
        telemetry_output: Path | None = None,
        telemetry_detail_level: str = "basic",
        telemetry_capture_command_output: bool = False,
        telemetry_max_text_attribute_bytes: int = 2048,
        telemetry_keep_in_memory_copy: bool | None = None,
        telemetry_writer_mode: str = "async",
        telemetry_queue_capacity: int = 16384,
        telemetry_batch_max_records: int = 256,
        telemetry_flush_interval_ms: int = 50,
        telemetry_overflow_policy: str = "drop_new",
        telemetry_serializer: str = "auto",
        benchmark_root: Path | None = None,
        benchmark_root_home: Path | None = None,
        benchmark_run_name: str | None = None,
        zpool_size: str = "10G",
        zpool_name: str | None = None,
        zpool_image: Path | None = None,
        reuse_zpool: bool = False,
        runtime_command_timeout_seconds: float = 60.0,
        runtime_zfs_prepare_timeout_seconds: float = 300.0,
        image_cache_root: Path | None = None,
        run_id: str | None = None,
        monitoring_enabled: bool = True,
        monitoring_sample_interval_ms: int = 1000,
        monitoring_include_host: bool = True,
        monitoring_include_sandboxes: bool = True,
        llm_server_launch_mode: str = "process",
        host_inspector_launch_mode: str = "process",
        host_inspector_log_level: str = "INFO",
        host_inspector_log_file: bool = False,
        runtime_root: Path | None = None,
        storage_root: Path | None = None,
        agent_host_root: Path | None = None,
        expected_sandboxes: int | None = None,
        rootfs_reuse_enabled: bool = True,
        sandbox_resource_limits: SandboxResourceLimits | None = None,
        fork_reuse_enabled: bool = False,
        eager_fork_cleanup_on_reject: bool = False,
        fork_chain_sharing_enabled: bool = False,
        background_prefork_enabled: bool = False,
        prefork_min_interval_seconds: float = 0.0,
        prefork_wait_timeout_seconds: float = 0.0,
        prefork_max_concurrent_global: int = 0,
        lazy_restore_enabled: bool = False,
        fast_forward_idle_waits: bool = True,
        relaunch_on_restore_failure: bool = False,
    ) -> None:
        self.provider = provider
        self.transfer_delay_ms = transfer_delay_ms
        self.scheduler_config = scheduler_config
        self.scheduler_policy = scheduler_policy
        self.checkpoint_manager_factory = checkpoint_manager_factory
        self.max_workers = max_workers
        self.executor_config = executor_config or ExecutorConfig(max_workers=max(1, self.max_workers))
        self.auto_cr = auto_cr
        self.work_dir_host_root = work_dir_host_root
        self.telemetry_output = telemetry_output
        self.telemetry_detail_level = telemetry_detail_level
        self.telemetry_capture_command_output = telemetry_capture_command_output
        self.telemetry_max_text_attribute_bytes = telemetry_max_text_attribute_bytes
        self.telemetry_keep_in_memory_copy = telemetry_keep_in_memory_copy
        self.telemetry_writer_mode = telemetry_writer_mode
        self.telemetry_queue_capacity = telemetry_queue_capacity
        self.telemetry_batch_max_records = telemetry_batch_max_records
        self.telemetry_flush_interval_ms = telemetry_flush_interval_ms
        self.telemetry_overflow_policy = telemetry_overflow_policy
        self.telemetry_serializer = telemetry_serializer
        configured_benchmark_root = (
            None if benchmark_root is None else Path(benchmark_root).expanduser().resolve()
        )
        configured_benchmark_root_home = (
            None if benchmark_root_home is None else Path(benchmark_root_home).expanduser().resolve()
        )
        if (
            configured_benchmark_root is not None
            and configured_benchmark_root_home is not None
            and configured_benchmark_root != configured_benchmark_root_home
        ):
            raise ValueError(
                "benchmark_root and benchmark_root_home are aliases; specify only one "
                "or give them the same value"
        )
        self.configured_benchmark_root_home = configured_benchmark_root_home or configured_benchmark_root
        self.configured_benchmark_root = self.configured_benchmark_root_home
        self.configured_benchmark_run_name = _normalize_benchmark_run_name(benchmark_run_name)
        self.configured_runtime_root = None if runtime_root is None else runtime_root.expanduser().resolve()
        self.configured_storage_root = None if storage_root is None else storage_root.expanduser().resolve()
        self.configured_agent_host_root = None if agent_host_root is None else agent_host_root.expanduser().resolve()
        self.zpool_size = zpool_size
        self.configured_zpool_name = zpool_name
        self.configured_zpool_image = zpool_image
        self.reuse_zpool = reuse_zpool
        self.runtime_command_timeout_seconds = float(runtime_command_timeout_seconds)
        self.runtime_zfs_prepare_timeout_seconds = float(runtime_zfs_prepare_timeout_seconds)
        self.image_cache_root = (image_cache_root or _DEFAULT_IMAGE_CACHE_ROOT).resolve()
        self.run_id = "" if run_id is None else str(run_id)
        self.monitoring_enabled = monitoring_enabled
        self.monitoring_sample_interval_ms = monitoring_sample_interval_ms
        self.monitoring_include_host = monitoring_include_host
        self.monitoring_include_sandboxes = monitoring_include_sandboxes
        self.llm_server_launch_mode = llm_server_launch_mode
        self.host_inspector_launch_mode = host_inspector_launch_mode
        self.host_inspector_log_level = host_inspector_log_level
        self.host_inspector_log_file = host_inspector_log_file
        self.expected_sandboxes = expected_sandboxes
        self.rootfs_reuse_enabled = bool(rootfs_reuse_enabled)
        self.sandbox_resource_limits = sandbox_resource_limits
        self.fork_reuse_enabled = bool(fork_reuse_enabled)
        self.eager_fork_cleanup_on_reject = bool(eager_fork_cleanup_on_reject)
        self.fork_chain_sharing_enabled = bool(fork_chain_sharing_enabled)
        self.background_prefork_enabled = bool(background_prefork_enabled)
        self.prefork_min_interval_seconds = float(prefork_min_interval_seconds)
        self.prefork_wait_timeout_seconds = float(prefork_wait_timeout_seconds)
        self.prefork_max_concurrent_global = max(0, int(prefork_max_concurrent_global))
        self.lazy_restore_enabled = bool(lazy_restore_enabled)
        self.fast_forward_idle_waits = bool(fast_forward_idle_waits)
        self.relaunch_on_restore_failure = bool(relaunch_on_restore_failure)
        self._tmpdir: tempfile.TemporaryDirectory[str] | None = None
        self.uses_temporary_root = False
        self.root: Path | None = None
        self.runtime_root: Path | None = None
        self.storage_root: Path | None = None
        self.agent_host_root: Path | None = None
        self.runtime_bundle_root: Path | None = None
        self.runtime_checkpoint_root: Path | None = None
        self.runtime_metadata_root: Path | None = None
        self.pool_name = ""
        self.runtime_state_root: Path | None = None
        self.host_inspector_url: str = ""
        self._host_inspector_server: HostInspectorServer | None = None
        self._host_inspector_process: subprocess.Popen[str] | None = None
        self._host_inspector_process_log_path: Path | None = None
        self.host_inspector_client: HostInspectorServiceClient = None
        self.telemetry: TelemetrySink | None = None
        self.telemetry_path: Path | None = None
        self.request_state_store: InMemoryRequestStateStore | None = None
        self.base_inspector: EBPFSandboxInspector | None = None
        self.inspector: RequestAwareSandboxInspector | None = None
        self.runtime: RuncRuntime | None = None
        self.storage: CheckpointManager | None = None
        self.executor: CRExecutor | None = None
        self.system: CrabSystem | None = None
        self.interceptor: CrabRequestInterceptorServer | None = None
        self.interceptor_hook = CompositeRequestInterceptorHook()
        self.llm_server = None
        self.llm_thread: threading.Thread | None = None
        self.llm_process: subprocess.Popen[str] | None = None
        self._llm_process_log_path: Path | None = None
        self.llm_server_base_url: str = ""
        self.llm_router_client: BenchmarkLLMRouterClient | None = None
        self.sandboxes: list[SandboxHandle] = []
        self._sandbox_by_id: dict[SandboxId, SandboxHandle] = {}
        self.network_manager = sandbox_network.BenchmarkNetworkManager()
        self._compose_image_tags: set[str] = set()
        self._agent_registry = build_agent_registry()
        self._sandbox_images: dict[str, AgentSandboxImage] = {}
        self._sandbox_image_lock = threading.Lock()
        self._task_executor = ThreadPoolExecutor(max_workers=max(1, self.max_workers))
        self._zpool_image_path: Path | None = None
        self._task_attempts: dict[str, int] = {}
        self._resource_monitor: BenchmarkResourceMonitor | None = None
        self._rootfs_reuse_session_key = self.run_id or uuid.uuid4().hex
        self._fork_chain_pins: dict[SandboxId, tuple[SandboxId, CheckpointId]] = {}
        self._speculation_controllers_by_sandbox: dict[SandboxId, "_SpeculativeSandboxController"] = {}

    @property
    def benchmark_bridge_ip(self) -> str:
        return self.network_manager.bridge_ip

    def _effective_runtime_root(self) -> Path:
        if self.runtime_root is not None:
            return self.runtime_root
        if self.root is not None:
            return self.root
        raise AssertionError("runtime root is not initialized")

    def _effective_runtime_bundle_root(self) -> Path:
        if self.runtime_bundle_root is not None:
            return self.runtime_bundle_root
        return self._effective_runtime_root() / "bundles"

    def _effective_runtime_checkpoint_root(self) -> Path:
        if self.runtime_checkpoint_root is not None:
            return self.runtime_checkpoint_root
        return self._effective_runtime_root() / "checkpoints"

    def _effective_agent_host_root(self) -> Path:
        if self.agent_host_root is not None:
            return self.agent_host_root
        if self.root is not None:
            return self.root
        return self._effective_runtime_root()

    def _resolve_runtime_plane_root(self, *, zpool_image_path: Path) -> Path:
        _ = zpool_image_path
        if self.configured_runtime_root is not None:
            return self.configured_runtime_root
        assert self.root is not None
        return self.root

    def _start_host_inspector_server(self) -> str:
        assert self.runtime_state_root is not None
        if self._host_inspector_server is not None:
            return f"http://{_HOST_INSPECTOR_HOST}:{self._host_inspector_server.port}"
        if self.host_inspector_launch_mode not in {"process", "thread"}:
            raise ValueError(
                f"unsupported host_inspector_launch_mode={self.host_inspector_launch_mode!r}; expected 'process' or 'thread'"
            )

        self.runtime_state_root.mkdir(parents=True, exist_ok=True)
        if self.host_inspector_launch_mode == "process":
            port = _find_free_port()
            log_dir = self.root if self.root is not None else Path(tempfile.gettempdir())
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / "host-inspector.stderr.log"
            self._host_inspector_process_log_path = log_path
            log_fh = open(log_path, "w")
            try:
                self._host_inspector_process = subprocess.Popen(
                    self._host_inspector_subprocess_command(port=port),
                    stdout=log_fh,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            finally:
                log_fh.close()
            url = f"http://{_HOST_INSPECTOR_HOST}:{port}"
            try:
                wait_for_http_json(f"{url}/healthz")
            except Exception as exc:
                if self._host_inspector_process is not None and self._host_inspector_process.poll() is not None:
                    returncode = self._host_inspector_process.returncode
                    stderr_output = self._read_subprocess_log_tail(self._host_inspector_process_log_path)
                    self._stop_host_inspector_server()
                    raise RuntimeError(
                        f"host inspector failed to start exit_code={returncode} stderr={stderr_output.strip()}"
                    ) from exc
                self._stop_host_inspector_server()
                raise
            return url

        daemon = HostInspectorDaemon(
            resolver=RuntimeResolver(runc_state_root=self.runtime_state_root),
            fs_monitor=LibbpfFilesystemMonitor(),
        )
        try:
            self._host_inspector_server = HostInspectorServer(
                host=_HOST_INSPECTOR_HOST,
                port=_HOST_INSPECTOR_PORT,
                daemon=daemon,
                max_workers=max(1, self.max_workers),
            )
        except OSError as exc:
            if exc.errno != errno.EADDRINUSE:
                raise
            logger.warning(
                "Host inspector port %d is already in use; falling back to an ephemeral port",
                _HOST_INSPECTOR_PORT,
            )
            self._host_inspector_server = HostInspectorServer(
                host=_HOST_INSPECTOR_HOST,
                port=0,
                daemon=daemon,
                max_workers=max(1, self.max_workers),
            )
        logger.info(
            "Starting host inspector server in-process host=%s port=%d runc_state_root=%s",
            _HOST_INSPECTOR_HOST,
            self._host_inspector_server.port,
            self.runtime_state_root,
        )
        self._host_inspector_server.start()
        url = f"http://{_HOST_INSPECTOR_HOST}:{self._host_inspector_server.port}"
        try:
            wait_for_http_json(f"{url}/healthz")
        except Exception:
            self._stop_host_inspector_server()
            raise
        return url

    def _stop_host_inspector_server(self) -> None:
        server = self._host_inspector_server
        self._host_inspector_server = None
        if server is not None:
            server.stop()
        if self._host_inspector_process is not None:
            self._host_inspector_process.terminate()
            try:
                self._host_inspector_process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._host_inspector_process.kill()
                self._host_inspector_process.wait(timeout=5.0)
            self._host_inspector_process = None

    @staticmethod
    def _generated_benchmark_run_name() -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    @staticmethod
    def _read_subprocess_log_tail(path: Path | None, *, max_bytes: int = 8192) -> str:
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

    def _resolve_benchmark_run_root(self) -> tuple[Path, bool]:
        benchmark_root_home = self.configured_benchmark_root_home
        if benchmark_root_home is None:
            bench_dir = os.environ.get("CRAB_BENCH_DIR", None)
            if bench_dir and bench_dir.lower() not in ["tmpdir", "tmp"]:
                benchmark_root_home = Path(bench_dir).expanduser().resolve()
        if benchmark_root_home is None:
            if self.configured_benchmark_run_name is not None:
                raise ValueError(
                    "benchmark_run_name requires benchmark_root_home, benchmark_root, "
                    "or CRAB_BENCH_DIR"
                )
            if self._tmpdir is None:
                raise RuntimeError("temporary benchmark root has not been initialized")
            return Path(self._tmpdir.name), True
        run_name = self.configured_benchmark_run_name or self._generated_benchmark_run_name()
        return benchmark_root_home / run_name, False

    def __enter__(self) -> "RealHostScenarioHarness":
        require_binaries()
        self.network_manager.configure(expected_sandboxes=self.expected_sandboxes)
        logger.info(
            "Selected benchmark network cidr=%s bridge_ip=%s",
            self.network_manager.network_cidr,
            self.network_manager.bridge_ip,
        )
        self._tmpdir = tempfile.TemporaryDirectory(prefix="crab_scenario_bench_")
        self.root, self.uses_temporary_root = self._resolve_benchmark_run_root()
        unique_suffix = uuid.uuid4().hex[:10]
        self.pool_name = self.configured_zpool_name or f"crabbench{unique_suffix}"
        zpool_image_path = self.configured_zpool_image or (self.root / "zpool.img")
        self.runtime_root = self._resolve_runtime_plane_root(zpool_image_path=zpool_image_path)
        self.runtime_state_root = self.runtime_root / "runtime-state"
        self.runtime_bundle_root = self.runtime_root / "bundles"
        self.runtime_checkpoint_root = self.runtime_root / "checkpoints"
        self.runtime_metadata_root = self.runtime_root / "sandbox-meta"
        self.storage_root = self.configured_storage_root or (self.root / "storage")
        self.agent_host_root = self.configured_agent_host_root or self.root
        self.host_inspector_url = self._start_host_inspector_server()
        telemetry_path = self.telemetry_output or (self.root / "telemetry.jsonl")
        self.telemetry_path = telemetry_path
        self.telemetry = build_configured_telemetry_sink(
            TelemetryConfig(
                enabled=True,
                jsonl_path=telemetry_path,
                keep_in_memory_copy=self.telemetry_keep_in_memory_copy,
                detail_level=self.telemetry_detail_level,
                capture_command_output=self.telemetry_capture_command_output,
                max_text_attribute_bytes=self.telemetry_max_text_attribute_bytes,
                writer_mode=self.telemetry_writer_mode,
                queue_capacity=self.telemetry_queue_capacity,
                batch_max_records=self.telemetry_batch_max_records,
                flush_interval_ms=self.telemetry_flush_interval_ms,
                overflow_policy=self.telemetry_overflow_policy,
                serializer=self.telemetry_serializer,
            ),
            default_attributes={"run_id": self.run_id} if self.run_id else None,
            keep_in_memory_fallback=False,
        )
        self.request_state_store = InMemoryRequestStateStore()
        self._start_llm_server()
        self._ensure_zpool()

        self.host_inspector_client = HostInspectorServiceClient(self.host_inspector_url)
        self.base_inspector = RemoteSandboxInspector(self.host_inspector_client, telemetry=self.telemetry)
        self.inspector = RequestAwareSandboxInspector(self.base_inspector, self.request_state_store)
        runc_restore_options = RuncRestoreOptions(lazy_pages=self.lazy_restore_enabled)
        self.runtime = RuncRuntime(
            paths=RuncRuntimePaths(
                state_root=self.runtime_state_root,
                bundle_root=self.runtime_bundle_root,
                checkpoint_root=self.runtime_checkpoint_root,
                metadata_root=self.runtime_metadata_root,
                zfs_dataset_prefix=f"{self.pool_name}/crab",
            ),
            options=RuncRuntimeOptions(
                command_timeout_seconds=self.runtime_command_timeout_seconds,
                zfs_prepare_timeout_seconds=self.runtime_zfs_prepare_timeout_seconds,
                restore=runc_restore_options,
            ),
            restore_options=runc_restore_options,
            host_inspector_client=self.host_inspector_client,
            telemetry=self.telemetry,
        )
        base_storage = LocalCheckpointManager(
            StorageConfig(root_dir=self.storage_root),
            # Wire the runtime's lazy-pages daemon registry into storage
            # so retention defers pruning a runtime checkpoint tree while
            # an in-flight ``criu lazy-pages`` daemon is still serving
            # userfault page faults from it. Without this gate, a
            # well-timed ``LatestOnlyCheckpointManager._prune_unprotected``
            # would SIGBUS the restored fork. See the docstring on
            # ``Runtime.runtime_image_path_in_use`` for the contract.
            runtime_image_path_in_use=self.runtime.runtime_image_path_in_use,
        )
        self.storage = self.checkpoint_manager_factory(base_storage)
        self.executor = CRExecutor(
            self.executor_config,
            DefaultCWorker(
                AdapterProcessCWorker(self.runtime),
                AdapterFileSystemCWorker(self.runtime),
                self.storage,
                self.runtime,
                checkpoint_guard=checkpoint_guard_from_inspector(self.inspector),
                telemetry=self.telemetry,
                step_workers=self.executor_config.resolved_composite_step_workers,
            ),
            DefaultRWorker(
                AdapterProcessRWorker(self.runtime),
                AdapterFileSystemRWorker(self.runtime),
                self.storage,
                telemetry=self.telemetry,
            ),
            self.telemetry,
        )
        self.system = CrabSystem(
            scheduler=CRScheduler(
                self.scheduler_config,
                self.inspector,
                self.runtime,
                InMemorySchedulerStateStore(),
                self.telemetry,
                self.scheduler_policy,
            ),
            executor=self.executor,
            storage=self.storage,
            inspector=self.inspector,
            runtime=self.runtime,
            telemetry=self.telemetry,
            request_state_store=self.request_state_store,
            relaunch_handler=self._relaunch_sandbox if self.auto_cr else None,
            extra_checkpoint_metadata_provider=self._llm_service_checkpoint_metadata,
            restore_metadata_handler=self._restore_llm_service_state,
            recovery_delay_seconds=self.transfer_delay_ms / 1000.0 if self.auto_cr else 0.0,
            relaunch_on_restore_failure=self.relaunch_on_restore_failure,
        )
        # Auto-mode checkpoints land via the system's internal coordinator,
        # not via harness wrappers. Wrap the scheduler's per-completion hook
        # so the speculation controllers' kick_prefork fires regardless of
        # who triggered the checkpoint. Manual paths additionally call
        # `_notify_speculation_checkpoint_complete` directly; the controller
        # dedupes prefork via `_prefork_in_flight`, so duplicate notify is
        # safe.
        _orig_mark_complete = self.system.scheduler.mark_checkpoint_complete

        def _mark_complete_with_prefork_notify(sandbox_id_arg, *args, **kwargs):
            _orig_mark_complete(sandbox_id_arg, *args, **kwargs)
            self._notify_speculation_checkpoint_complete(sandbox_id_arg, None)

        self.system.scheduler.mark_checkpoint_complete = _mark_complete_with_prefork_notify  # type: ignore[method-assign]
        if self.telemetry is not None and self.runtime is not None and self.monitoring_enabled:
            self._resource_monitor = BenchmarkResourceMonitor(
                telemetry=self.telemetry,
                runtime=self.runtime,
                sandboxes=lambda: list(self.sandboxes),
                sample_interval_ms=self.monitoring_sample_interval_ms,
                include_host=self.monitoring_include_host,
                include_sandboxes=self.monitoring_include_sandboxes,
            )
            self._resource_monitor.start()
        self.interceptor_hook.add_hook(TelemetryRequestInterceptorHook(self.telemetry))
        self.interceptor = CrabRequestInterceptorServer(
            upstream_url=self.llm_server_base_url,
            request_state_store=self.request_state_store,
            hook=self.interceptor_hook,
            telemetry=self.telemetry,
            on_state_change=self.system.notify_interceptor_state_change,
            on_response_ready=self.system.notify_live_response_ready,
            response_gate_registry=self.system.response_gate_registry,
            sandbox_id_resolver=self.resolve_interceptor_sandbox_id,
            host="0.0.0.0",
            port=0,
            # Spec scenarios fire draft+oracle PER sandbox concurrently —
            # 2× the request fan-out of nofault. Sizing the interceptor
            # pool to just `max_workers` causes the second request of each
            # spec pair to queue behind the first inside the interceptor's
            # PooledHTTPServer, completely defeating the design's
            # "draft and oracle run in parallel" promise. (Smoking gun:
            # the second request's `interceptor.request.received` lands
            # within ms of the first request's `interceptor.response.released`,
            # not within ms of the first's `received`.) The router pool
            # already uses `max(32, max_workers*4)` for the same reason —
            # apply the same headroom here so the interceptor is never the
            # serialization point for parallel-by-design request pairs.
            max_workers=max(32, self.max_workers * 4),
        )
        self.interceptor.start()
        wait_for_http_json(f"http://127.0.0.1:{self.interceptor.port}/healthz")
        if self.auto_cr:
            self.system.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # input("wait for signal to cleanup")
        def _run_teardown_step(step: str, fn) -> None:
            logger.info("Benchmark teardown step start step=%s", step)
            started_at = time.perf_counter()
            try:
                fn()
            finally:
                logger.info(
                    "Benchmark teardown step end step=%s duration_s=%.3f",
                    step,
                    max(0.0, time.perf_counter() - started_at),
                )

        def _stop_resource_monitor() -> None:
            if self._resource_monitor is not None:
                self._resource_monitor.stop()
                self._resource_monitor = None

        _run_teardown_step("resource_monitor.stop", _stop_resource_monitor)
        _run_teardown_step(
            "task_runs.request_stop",
            lambda: [
                sandbox.task_run.request_stop()
                for sandbox in self.sandboxes
                if sandbox.task_run is not None
            ],
        )
        _run_teardown_step(
            "system.stop",
            lambda: self.system.stop() if self.system is not None and self.auto_cr else None,
        )
        _run_teardown_step(
            "interceptor.stop",
            lambda: self.interceptor.stop() if self.interceptor is not None else None,
        )
        # task_executor.shutdown MUST run before runtime.delete_all.
        # Pre-fix, runtimes were deleted while in-flight terminus
        # tasks were still mid-turn; the next `runc exec` then failed
        # with "container does not exist" and the task entered a 300s
        # restore-wait, hanging task_executor.shutdown. Observed in
        # benchmarks 20260429_031243 and 20260429_063012, both stuck
        # at task_executor.shutdown for 17+ minutes.
        #
        # request_stop() above signals each task to exit at its next
        # turn check; task_executor.shutdown(wait=True,
        # cancel_futures=True) cancels pending futures and waits for
        # in-flight tasks to actually return. With containers still
        # alive, the agents can finish a turn cleanly. Only after
        # they're all done do we tear runtimes down.
        _run_teardown_step(
            "task_executor.shutdown",
            lambda: self._task_executor.shutdown(wait=True, cancel_futures=True),
        )
        _run_teardown_step(
            "runtime.delete_all",
            lambda: [
                self.runtime.delete_runtime(sandbox.sandbox_id, force=True, ignore_missing=True)
                for sandbox in self.sandboxes
            ]
            if self.runtime is not None
            else None,
        )
        _run_teardown_step(
            "executor.shutdown",
            lambda: self.executor.shutdown() if self.executor is not None else None,
        )

        def _close_storage() -> None:
            if self.storage is None:
                return
            flush_storage = getattr(self.storage, "flush", None)
            if callable(flush_storage):
                flush_storage()
            close_storage = getattr(self.storage, "close", None)
            if callable(close_storage):
                close_storage()

        _run_teardown_step("storage.close", _close_storage)
        _run_teardown_step("network.cleanup", self.network_manager.cleanup)
        _run_teardown_step(
            "llm_service.unregister_all",
            lambda: [self._unregister_llm_service(sandbox.sandbox_id) for sandbox in self.sandboxes],
        )

        def _destroy_benchmark_dataset() -> None:
            if not self.pool_name:
                return
            dataset = f"{self.pool_name}/crab"
            subprocess.run(
                ["zfs", "destroy", "-r", dataset],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if not self.reuse_zpool:
                subprocess.run(
                    ["zpool", "destroy", "-f", self.pool_name],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

        _run_teardown_step("zfs.destroy", _destroy_benchmark_dataset)
        # for image in self._sandbox_images.values():
            # subprocess.run(
            #     ["docker", "rmi", "-f", image.image_tag],
            #     check=False,
            #     stdout=subprocess.DEVNULL,
            #     stderr=subprocess.DEVNULL,
            # )
        # for image_tag in sorted(self._compose_image_tags):
            # subprocess.run(
            #     ["docker", "rmi", "-f", image_tag],
            #     check=False,
            #     stdout=subprocess.DEVNULL,
            #     stderr=subprocess.DEVNULL,
            # )
        _run_teardown_step("llm_server.stop", self._stop_llm_server)
        _run_teardown_step("host_inspector.stop", self._stop_host_inspector_server)

        def _close_host_inspector_client() -> None:
            if self.host_inspector_client is not None:
                self.host_inspector_client.close()
                self.host_inspector_client = None

        _run_teardown_step("host_inspector_client.close", _close_host_inspector_client)
        _run_teardown_step(
            "telemetry.close",
            lambda: self.telemetry.close() if self.telemetry is not None else None,
        )
        _run_teardown_step(
            "tmpdir.cleanup",
            lambda: self._tmpdir.cleanup() if self._tmpdir is not None else None,
        )

    def _llm_router_subprocess_command(self, *, port: int) -> list[str]:
        router_workers = self._llm_router_max_workers()
        command = [
            sys.executable,
            "-m",
            "integrations.llm_services.router",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--telemetry-detail-level",
            self.telemetry_detail_level,
            "--telemetry-max-text-attribute-bytes",
            str(self.telemetry_max_text_attribute_bytes),
            "--telemetry-writer-mode",
            self.telemetry_writer_mode,
            "--telemetry-queue-capacity",
            str(self.telemetry_queue_capacity),
            "--telemetry-batch-max-records",
            str(self.telemetry_batch_max_records),
            "--telemetry-flush-interval-ms",
            str(self.telemetry_flush_interval_ms),
            "--telemetry-overflow-policy",
            self.telemetry_overflow_policy,
            "--telemetry-serializer",
            self.telemetry_serializer,
            "--max-workers",
            str(router_workers),
        ]
        if self.telemetry_path is not None:
            command.extend(["--telemetry-jsonl", str(self.telemetry_path)])
        if self.run_id:
            command.extend(["--run-id", self.run_id])
        if self.telemetry_capture_command_output:
            command.append("--telemetry-capture-command-output")
        if self.telemetry_keep_in_memory_copy:
            command.append("--telemetry-keep-in-memory-copy")
        return command

    def _host_inspector_subprocess_command(self, *, port: int) -> list[str]:
        assert self.runtime_state_root is not None
        command = [
            sys.executable,
            "-m",
            "crab.host_inspector.server",
            "--host",
            _HOST_INSPECTOR_HOST,
            "--port",
            str(port),
            "--runc-state-root",
            str(self.runtime_state_root),
            "--max-workers",
            str(max(1, self.max_workers)),
            "--log-level",
            self.host_inspector_log_level,
        ]
        if self.host_inspector_log_file and self.root is not None:
            log_file = self.root / "host-inspector.log"
            log_file.parent.mkdir(parents=True, exist_ok=True)
            command.extend(["--log-file", str(log_file)])
        return command

    def _llm_router_max_workers(self) -> int:
        # Speculative replay can issue both draft and oracle requests per active
        # sandbox while agents also poll the control plane for router state.
        # Keep the router pool comfortably above benchmark concurrency so
        # control endpoints are not starved by delayed replay responses.
        return max(32, max(1, self.max_workers) * 4)

    def _start_llm_server(self) -> None:
        if self.llm_server_launch_mode not in {"process", "thread"}:
            raise ValueError(
                f"unsupported llm_server_launch_mode={self.llm_server_launch_mode!r}; expected 'process' or 'thread'"
            )
        if self.llm_server_launch_mode == "thread":
            self.llm_server = serve_benchmark_llm_router(
                host="127.0.0.1",
                port=0,
                telemetry=self.telemetry,
                max_workers=self._llm_router_max_workers(),
            )
            self.llm_thread = threading.Thread(target=self.llm_server.serve_forever, daemon=True)
            self.llm_thread.start()
            port = int(self.llm_server.server_address[1])
        else:
            port = _find_free_port()
            log_dir = self.root if self.root is not None else Path(tempfile.gettempdir())
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / "llm-router.log"
            self._llm_process_log_path = log_path
            log_fh = open(log_path, "w")
            try:
                self.llm_process = subprocess.Popen(
                    self._llm_router_subprocess_command(port=port),
                    stdout=log_fh,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            finally:
                log_fh.close()
        self.llm_server_base_url = f"http://127.0.0.1:{port}"
        self.llm_router_client = BenchmarkLLMRouterClient(self.llm_server_base_url)
        try:
            wait_for_http_json(f"{self.llm_server_base_url}/healthz")
        except Exception as exc:
            if self.llm_process is not None and self.llm_process.poll() is not None:
                returncode = self.llm_process.returncode
                stderr_output = self._read_subprocess_log_tail(self._llm_process_log_path)
                self._stop_llm_server()
                raise RuntimeError(
                    f"benchmark llm router failed to start exit_code={returncode} stderr={stderr_output.strip()}"
                ) from exc
            self._stop_llm_server()
            raise

    def _stop_llm_server(self) -> None:
        if self.llm_router_client is not None:
            self.llm_router_client.close()
        self.llm_router_client = None
        self.llm_server_base_url = ""
        if self.llm_server is not None:
            self.llm_server.shutdown()
            self.llm_server.server_close()
            self.llm_server = None
        if self.llm_thread is not None:
            self.llm_thread.join(timeout=5.0)
            self.llm_thread = None
        if self.llm_process is not None:
            self.llm_process.terminate()
            try:
                self.llm_process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self.llm_process.kill()
                self.llm_process.wait(timeout=5.0)
            self.llm_process = None

    def _register_llm_service(
        self,
        sandbox_id: SandboxId,
        *,
        llm_service_type: str,
        llm_service_config: dict[str, object] | None = None,
    ) -> None:
        if self.llm_router_client is not None:
            self.llm_router_client.register_sandbox(
                sandbox_id=str(sandbox_id),
                llm_service_type=llm_service_type,
                llm_service_config=llm_service_config,
            )
            return
        if self.llm_server is not None:
            self.llm_server.benchmark_llm_router.register_sandbox(  # type: ignore[attr-defined]
                sandbox_id=str(sandbox_id),
                llm_service_type=llm_service_type,
                llm_service_config=llm_service_config,
            )
            return
        raise RuntimeError("llm router is not initialized")

    def _unregister_llm_service(self, sandbox_id: SandboxId) -> None:
        try:
            if self.llm_router_client is not None:
                self.llm_router_client.unregister_sandbox(str(sandbox_id))
            elif self.llm_server is not None:
                self.llm_server.benchmark_llm_router.unregister_sandbox(str(sandbox_id))  # type: ignore[attr-defined]
        except Exception:
            logger.debug("Failed to unregister llm service for sandbox=%s", sandbox_id, exc_info=True)

    def _llm_control_base_url(self) -> str:
        if self.llm_server_base_url:
            return self.llm_server_base_url
        if self.llm_server is not None:
            return f"http://127.0.0.1:{self.llm_server.server_address[1]}"
        return ""

    def _snapshot_llm_services(self, sandbox_id: SandboxId | None = None) -> dict[str, object] | None:
        resolved_sandbox_id = None if sandbox_id is None else str(sandbox_id)
        if self.llm_router_client is not None:
            return self.llm_router_client.snapshot(resolved_sandbox_id)
        if self.llm_server is not None:
            snapshot = self.llm_server.benchmark_llm_router.snapshot(  # type: ignore[attr-defined]
                sandbox_id=resolved_sandbox_id,
                include_events=False,
            )
            return None if snapshot is None else dict(snapshot)
        return None

    def _reset_llm_router_state(self, sandbox_id: SandboxId) -> None:
        if self.llm_router_client is not None:
            self.llm_router_client.reset_sandbox(str(sandbox_id))
            return
        if self.llm_server is not None:
            self.llm_server.benchmark_llm_router.reset_sandbox(str(sandbox_id))  # type: ignore[attr-defined]
            return
        raise RuntimeError("llm router is not initialized")

    def _restore_llm_router_state(self, sandbox_id: SandboxId, *, consumed_response_count: int) -> None:
        if self.llm_router_client is not None:
            self.llm_router_client.restore_sandbox(
                str(sandbox_id),
                consumed_response_count=consumed_response_count,
            )
            return
        if self.llm_server is not None:
            self.llm_server.benchmark_llm_router.restore_sandbox(  # type: ignore[attr-defined]
                str(sandbox_id),
                consumed_response_count=consumed_response_count,
            )
            return
        raise RuntimeError("llm router is not initialized")

    def _copy_llm_router_state(self, *, source_sandbox_id: SandboxId, target_sandbox_id: SandboxId) -> None:
        if self.llm_router_client is not None:
            self.llm_router_client.copy_sandbox_state(
                source_sandbox_id=str(source_sandbox_id),
                target_sandbox_id=str(target_sandbox_id),
            )
            return
        if self.llm_server is not None:
            self.llm_server.benchmark_llm_router.copy_sandbox_state(  # type: ignore[attr-defined]
                source_sandbox_id=str(source_sandbox_id),
                target_sandbox_id=str(target_sandbox_id),
            )
            return
        raise RuntimeError("llm router is not initialized")

    def sandbox_state_changed(self, sandbox: SandboxHandle) -> bool:
        assert self.base_inspector is not None
        snapshot = self.base_inspector.inspect(sandbox.sandbox_id)
        return bool(snapshot.process_changed or snapshot.filesystem_changed)

    def _swap_speculative_sandbox_state(self, active: SandboxHandle, fork: SandboxHandle) -> None:
        # If the fork was prepped in chain-sharing mode, its runtime checkpoint
        # tree contains symlinks into the about-to-be-destroyed active's
        # tree. Inline those bytes now (under the fork's current sandbox id,
        # before the id swap below) so subsequent incremental checkpoints
        # taken by the new-active's CRIU don't follow `--parent-path` into a
        # destroyed source. Also materialize the storage-side artifact JSON
        # for symmetry, in case storage-layer linking is ever wired in.
        self._inline_fork_chain_artifacts(fork.sandbox_id)
        self._fork_chain_pins.pop(fork.sandbox_id, None)
        # Re-key the speculation controller so subsequent checkpoint
        # completions on the now-active sandbox (which carries the fork's
        # old id post-swap) still find their controller.
        controller = self._speculation_controllers_by_sandbox.pop(active.sandbox_id, None)
        if controller is not None:
            self._speculation_controllers_by_sandbox[fork.sandbox_id] = controller
        active_id = active.sandbox_id
        fork_id = fork.sandbox_id
        active_bundle_dir = active.bundle_dir
        active_status_port = active.status_port
        active_status_host = active.status_host
        active_llm_base_url = active.llm_base_url
        active_llm_control_base_url = active.llm_control_base_url
        active_last_status = dict(active.last_status)
        active_launch_source = active.launch_source
        active_launch_metadata = dict(active.launch_metadata)

        active.sandbox_id = fork_id
        active.bundle_dir = fork.bundle_dir
        active.status_port = fork.status_port
        active.status_host = fork.status_host
        active.llm_base_url = fork.llm_base_url
        active.llm_control_base_url = fork.llm_control_base_url
        active.last_status = dict(fork.last_status)
        active.launch_source = fork.launch_source
        active.launch_metadata = dict(fork.launch_metadata)

        fork.sandbox_id = active_id
        fork.bundle_dir = active_bundle_dir
        fork.status_port = active_status_port
        fork.status_host = active_status_host
        fork.llm_base_url = active_llm_base_url
        fork.llm_control_base_url = active_llm_control_base_url
        fork.last_status = active_last_status
        fork.launch_source = active_launch_source
        fork.launch_metadata = active_launch_metadata
        fork.task_run = None
        fork.task_future = None

        self._sandbox_by_id[active.sandbox_id] = active
        self._sandbox_by_id[fork.sandbox_id] = fork

    def _ensure_zpool(self) -> None:
        assert self.root is not None
        dataset = f"{self.pool_name}/crab"
        zpool_image_path = self.configured_zpool_image or (self.root / "zpool.img")
        self._zpool_image_path = zpool_image_path
        if self.reuse_zpool:
            if _zpool_exists(self.pool_name):
                logger.info("Reusing existing benchmark zpool name=%s", self.pool_name)
            else:
                zpool_image_path.parent.mkdir(parents=True, exist_ok=True)
                if not zpool_image_path.exists():
                    logger.info(
                        "Creating reusable benchmark zpool image path=%s size=%s",
                        zpool_image_path,
                        self.zpool_size,
                    )
                    subprocess.run(["truncate", "-s", self.zpool_size, str(zpool_image_path)], check=True)
                logger.info("Creating reusable benchmark zpool name=%s image=%s", self.pool_name, zpool_image_path)
                subprocess.run(["zpool", "create", "-f", self.pool_name, str(zpool_image_path)], check=True)
            if _zfs_dataset_exists(dataset):
                subprocess.run(
                    ["zfs", "destroy", "-r", dataset],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            if _zfs_dataset_exists(dataset):
                logger.warning(
                    "Reusing existing benchmark dataset name=%s after destroy attempt did not remove it",
                    dataset,
                )
            else:
                subprocess.run(["zfs", "create", dataset], check=True)
            return

        subprocess.run(["truncate", "-s", self.zpool_size, str(zpool_image_path)], check=True)
        subprocess.run(["zpool", "create", "-f", self.pool_name, str(zpool_image_path)], check=True)
        subprocess.run(["zfs", "create", dataset], check=True)

    def get_agent_class(self, agent_type: str) -> type[BaseAgent]:
        try:
            return self._agent_registry[agent_type]
        except KeyError as exc:
            raise ValueError(f"unsupported benchmark agent type: {agent_type}") from exc

    def build_task_run(
        self,
        agent_type: str,
        sandbox: SandboxHandle,
        task_description: TaskDescription,
        task_config: TaskConfig,
    ) -> BaseAgent:
        extra_kwargs: dict[str, object] = {}
        if (agent_type == "mini_swe" and sandbox.llm_service_type == "mini_swe_spec_trace_replay") or (
            agent_type == "terminus" and sandbox.llm_service_type == "terminus_spec_trace_replay"
        ):
            controller = _SpeculativeSandboxController(
                self,
                sandbox,
                fork_reuse_enabled=self.fork_reuse_enabled,
                eager_cleanup_on_reject=self.eager_fork_cleanup_on_reject,
                background_prefork_enabled=self.background_prefork_enabled,
                prefork_min_interval_seconds=self.prefork_min_interval_seconds,
                prefork_wait_timeout_seconds=self.prefork_wait_timeout_seconds,
                prefork_max_concurrent_global=self.prefork_max_concurrent_global,
            )
            self._speculation_controllers_by_sandbox[sandbox.sandbox_id] = controller
            extra_kwargs["speculation_controller"] = controller
        if agent_type == "terminus":
            extra_kwargs["fast_forward_idle_waits"] = self.fast_forward_idle_waits
        return self.get_agent_class(agent_type)(
            sandbox,
            task_description,
            task_config,
            runtime_state_root=self.runtime_state_root,
            runtime=self.runtime,
            agent_host_dir=self._effective_agent_host_root() / agent_type / str(sandbox.sandbox_id),
            llm_base_url=sandbox.llm_base_url,
            telemetry=self.telemetry,
            **extra_kwargs,
        )

    def _agent_requires_benchmark_network(self, agent_type: str) -> bool:
        return bool(self.get_agent_class(agent_type).requires_network_namespace)

    def _sandbox_requires_benchmark_network(self, sandbox: SandboxHandle) -> bool:
        if self._agent_requires_benchmark_network(sandbox.agent_type):
            return True
        lease_for = getattr(self.network_manager, "lease_for", None)
        if callable(lease_for) and lease_for(sandbox.sandbox_id) is not None:
            return True
        config_path = sandbox.bundle_dir / "config.json"
        if not config_path.is_file():
            return False
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            logger.debug(
                "Failed to inspect sandbox bundle for network namespace usage sandbox=%s",
                sandbox.sandbox_id,
                exc_info=True,
            )
            return False
        namespaces = payload.get("linux", {}).get("namespaces", [])
        if not isinstance(namespaces, list):
            return False
        return any(isinstance(namespace, dict) and namespace.get("type") == "network" for namespace in namespaces)

    def resolve_llm_service_type(self, *, agent_type: str, llm_service_type: str | None) -> str:
        resolved = llm_service_type or default_llm_service_type_for_agent(agent_type)
        validate_llm_service_type(provider=self.provider, llm_service_type=resolved)
        return resolved

    def _sandbox_dockerfile_path(self, agent_type: str) -> Path:
        if agent_type == "iflow":
            return IFLOW_DOCKERFILE_PATH
        if agent_type == "simulated":
            return SIMULATED_DOCKERFILE_PATH
        raise ValueError(f"unsupported benchmark agent type for sandbox image: {agent_type}")

    def _sandbox_build_context_path(self, agent_type: str) -> Path:
        if agent_type == "iflow":
            return IFLOW_DOCKERFILE_PATH.parent
        if agent_type == "simulated":
            return SIMULATED_DOCKERFILE_PATH.parent
        raise ValueError(f"unsupported benchmark agent type for sandbox image: {agent_type}")

    def _sandbox_image_tag(self, agent_type: str) -> str:
        return f"crab-{sandbox_image.docker_tag_component(agent_type)}-bench:workspace"

    def ensure_sandbox_image(self, agent_type: str) -> AgentSandboxImage:
        with self._sandbox_image_lock:
            cached = self._sandbox_images.get(agent_type)
            if cached is not None:
                return cached
            image_tag = self._sandbox_image_tag(agent_type)
            sandbox_image.build_image(
                tag=image_tag,
                build_context=self._sandbox_build_context_path(agent_type),
                dockerfile_path=self._sandbox_dockerfile_path(agent_type),
                telemetry=self.telemetry,
            )
            image_defaults = sandbox_image.inspect_image_runtime_defaults(
                tag=image_tag,
                cache_root=self.image_cache_root,
                telemetry=self.telemetry,
            )
            exported_rootfs = sandbox_image.export_image_rootfs(
                tag=image_tag,
                output_dir=self._effective_runtime_root() / "image" / agent_type,
                cache_root=self.image_cache_root,
                telemetry=self.telemetry,
            )
            built = AgentSandboxImage(
                agent_type=agent_type,
                image_tag=image_tag,
                exported_rootfs=exported_rootfs,
                image_defaults=image_defaults,
            )
            self._sandbox_images[agent_type] = built
            return built

    def _ensure_task_record_inputs(
        self,
        sandbox_name: str,
        task_record: benchmark_support.BenchmarkTaskRecord,
    ) -> None:
        _ = sandbox_name
        if task_record.docker_compose_file is not None:
            return
        self.ensure_sandbox_image(task_record.agent_type)

    def get_sandbox_handle(self, sandbox_id: str | SandboxId) -> SandboxHandle:
        target = SandboxId(str(sandbox_id))
        return self._sandbox_by_id[target]

    def load_dataset(self, path: Path) -> list[benchmark_support.BenchmarkTaskRecord]:
        return benchmark_core.load_task_dataset(path)

    def _resolve_dataset_service_config(
        self,
        dataset_root: Path,
        raw_value: object,
    ) -> dict[str, object] | None:
        if raw_value is None:
            return None
        if not isinstance(raw_value, dict):
            raise ValueError(f"llm_service_config must be an object, got {raw_value!r}")
        config = dict(raw_value)
        trace_path = config.get("trace_path")
        if isinstance(trace_path, str):
            config["trace_path"] = str((dataset_root / trace_path).resolve())
        return config

    def select_task_record(
        self,
        dataset: list[benchmark_support.BenchmarkTaskRecord] | None,
        *,
        sandbox_index: int,
        default_agent_type: str,
        default_llm_service_type: str | None,
        default_task_description: TaskDescription,
        default_task_config: TaskConfig,
    ) -> benchmark_support.BenchmarkTaskRecord:
        return benchmark_core.select_task_record(
            dataset,
            sandbox_index=sandbox_index,
            default_agent_type=default_agent_type,
            default_llm_service_type=default_llm_service_type,
            default_task_description=default_task_description,
            default_task_config=default_task_config,
        )

    def launch_sandbox(
        self,
        sandbox_name: str,
        *,
        agent_type: str = "simulated",
        llm_service_type: str | None = None,
        llm_service_config: dict[str, object] | None = None,
        task_description: TaskDescription | None = None,
        task_config: TaskConfig | None = None,
    ) -> SandboxHandle:
        assert self.root is not None
        assert self.base_inspector is not None
        assert self.system is not None
        assert self.interceptor is not None

        if (task_description is None) != (task_config is None):
            raise ValueError("task_description and task_config must be provided together")

        resolved_task_description = task_description or TaskDescription("")
        resolved_task_config = task_config or TaskConfig()
        resolved_llm_service_type = self.resolve_llm_service_type(agent_type=agent_type, llm_service_type=llm_service_type)
        sandbox_image = self.ensure_sandbox_image(agent_type)
        network_lease = (
            self.network_manager.allocate_lease(SandboxId(sandbox_name))
            if self._agent_requires_benchmark_network(agent_type)
            else None
        )
        handle, work_dir_host_path = self._prepare_sandbox_handle(
            sandbox_name,
            interceptor_host=self.network_manager.bridge_ip if network_lease is not None else "127.0.0.1",
            network_lease=network_lease,
            agent_type=agent_type,
            llm_service_type=resolved_llm_service_type,
            llm_service_config=llm_service_config,
            image_defaults=sandbox_image.image_defaults,
            image_rootfs_dir=sandbox_image.exported_rootfs,
        )
        task_run = self.build_task_run(agent_type, handle, resolved_task_description, resolved_task_config)
        task_run.prepare_sandbox()
        task_run.configure_bundle()
        sandbox_id = handle.sandbox_id
        self.base_inspector.upsert_snapshot(
            SandboxSnapshot(
                sandbox_id=sandbox_id,
                runtime_name="runc",
                is_running=True,
                process_changed=False,
                filesystem_changed=False,
                observed_at=utc_now(),
            )
        )
        if network_lease is not None:
            self.network_manager.register_guest_ip(network_lease.guest_ip, sandbox_id)
        launch_metadata = _merge_runtime_launch_metadata(
            {
                "sandbox_id": sandbox_name,
                "bundle_path": str(handle.bundle_dir),
                "work_dir_host_path": None if work_dir_host_path is None else str(work_dir_host_path),
                "rootfs_init_dirs": task_run.rootfs_init_dirs(),
                "rootfs_copy_paths": [{"source": str(sandbox_image.exported_rootfs), "destination": "/"}],
            },
            task_run.extra_launch_metadata(),
            handle.launch_metadata.get("runtime", {}),
        )
        self._attach_shared_rootfs_launch_metadata(
            launch_metadata,
            agent_type=agent_type,
            compose_file=None,
            service_name=None,
        )
        runtime = self._active_runtime
        if runtime is None:
            raise RuntimeError("runtime is not initialized")
        runtime.launch("runc", launch_metadata)
        task_run.wait_for_task_ready()
        self.emit_benchmark_event("benchmark.task.ready", handle)
        logger.info(
            "Launched benchmark sandbox name=%s sandbox_id=%s status_host=%s status_port=%d auto_cr=%s agent_type=%s",
            sandbox_name,
            sandbox_id,
            handle.status_host,
            handle.status_port,
            self.auto_cr,
            agent_type,
        )
        return handle

    def benchmark_telemetry_attributes(
        self,
        sandbox: SandboxHandle,
        *,
        iteration: int | None = None,
        event_type: str | None = None,
        checkpoint_id: CheckpointId | None = None,
        request_id: str | None = None,
        extra: dict[str, object] | None = None,
    ) -> dict[str, object]:
        attributes: dict[str, object] = {
            "component": "benchmark",
            "sandbox_id": str(sandbox.sandbox_id),
            "task_id": benchmark_core.task_id_for_sandbox(sandbox),
            "task_run_id": benchmark_core.task_run_id_for_sandbox(sandbox),
            "agent_type": sandbox.agent_type,
            "llm_service_type": sandbox.llm_service_type,
            "task_attempt": self._task_attempts.get(str(sandbox.sandbox_id), 0),
        }
        if iteration is not None:
            attributes["iteration"] = int(iteration)
        if event_type is not None:
            attributes["event_type"] = event_type
        if checkpoint_id is not None:
            attributes["checkpoint_id"] = str(checkpoint_id)
        if request_id is not None:
            attributes["request_id"] = request_id
        if extra:
            attributes.update(extra)
        return attributes

    def emit_benchmark_metric(
        self,
        name: str,
        value: float,
        sandbox: SandboxHandle,
        *,
        iteration: int | None = None,
        event_type: str | None = None,
        checkpoint_id: CheckpointId | None = None,
        extra: dict[str, object] | None = None,
    ) -> None:
        if self.telemetry is None:
            return
        self.telemetry.emit_metric(
            name,
            float(value),
            self.benchmark_telemetry_attributes(
                sandbox,
                iteration=iteration,
                event_type=event_type,
                checkpoint_id=checkpoint_id,
                extra=extra,
            ),
        )

    def emit_benchmark_event(
        self,
        name: str,
        sandbox: SandboxHandle,
        *,
        iteration: int | None = None,
        event_type: str | None = None,
        checkpoint_id: CheckpointId | None = None,
        extra: dict[str, object] | None = None,
    ) -> None:
        if self.telemetry is None:
            return
        self.telemetry.emit_event(
            name,
            self.benchmark_telemetry_attributes(
                sandbox,
                iteration=iteration,
                event_type=event_type,
                checkpoint_id=checkpoint_id,
                extra=extra,
            ),
        )

    def _bind_task_future_telemetry(self, task_future, *, operation) -> None:
        if operation is None:
            return

        def _finalize(future) -> None:
            try:
                future.result()
            except Exception:
                operation.finish(status="failed")
                return
            operation.finish(status="succeeded")

        task_future.add_done_callback(_finalize)

    def _wrap_task_future_execution(
        self,
        task_run: BaseAgent,
        *,
        task_attributes: dict[str, object],
    ):
        submitted_ns = time.perf_counter_ns()

        def _run_task() -> None:
            queue_wait_ms = max(0.0, (time.perf_counter_ns() - submitted_ns) / 1_000_000.0)
            run_operation = None
            if self.telemetry is not None:
                metric_attributes = dict(task_attributes)
                self.telemetry.emit_metric(
                    "benchmark.task.queue_wait_ms",
                    queue_wait_ms,
                    metric_attributes,
                )
                run_operation = start_operation(
                    self.telemetry,
                    "benchmark.task.run",
                    {
                        **task_attributes,
                        "queue_wait_ms": queue_wait_ms,
                    },
                )
            try:
                task_run.perform_task()
            except Exception:
                if run_operation is not None:
                    run_operation.finish(status="failed")
                raise
            if run_operation is not None:
                run_operation.finish(status="succeeded")

        return _run_task

    def _run_benchmark_setup_step(
        self,
        sandbox: SandboxHandle,
        *,
        name: str,
        setup_step: str,
        fn,
        extra: dict[str, object] | None = None,
    ):
        operation = None
        if self.telemetry is not None:
            attributes = {
                "phase": "setup",
                "phase_scope": "sandbox",
                "setup_step": setup_step,
            }
            if extra:
                attributes.update(extra)
            operation = start_operation(
                self.telemetry,
                name,
                self.benchmark_telemetry_attributes(sandbox, extra=attributes),
            )
        try:
            result = fn()
        except Exception:
            if operation is not None:
                operation.finish(status="failed")
            raise
        if operation is not None:
            operation.finish(status="succeeded")
        return result

    def launch_task(
        self,
        agent_type: str,
        task_description: TaskDescription,
        task_config: TaskConfig,
        sandbox_id: str,
    ) -> BaseAgent:
        handle = self.get_sandbox_handle(sandbox_id)
        if handle.task_future is not None and not handle.task_future.done():
            if handle.task_run is not None:
                handle.task_run.request_stop()
            handle.task_future.cancel()
        handle.agent_type = agent_type
        handle.task_description = task_description
        handle.task_config = task_config
        task_run = self.build_task_run(agent_type, handle, task_description, task_config)
        handle.task_run = task_run
        self._task_attempts[str(handle.sandbox_id)] = self._task_attempts.get(str(handle.sandbox_id), 0) + 1
        task_attributes = self.benchmark_telemetry_attributes(handle)
        task_operation = None if self.telemetry is None else start_operation(
            self.telemetry,
            "benchmark.task",
            dict(task_attributes),
        )
        handle.task_future = self._task_executor.submit(
            self._wrap_task_future_execution(task_run, task_attributes=task_attributes)
        )
        self._bind_task_future_telemetry(handle.task_future, operation=task_operation)
        return task_run

    def launch_sandbox_and_task(
        self,
        sandbox_name: str,
        *,
        agent_type: str,
        llm_service_type: str | None = None,
        llm_service_config: dict[str, object] | None = None,
        task_description: TaskDescription,
        task_config: TaskConfig,
    ) -> SandboxHandle:
        handle = self.launch_sandbox(
            sandbox_name,
            agent_type=agent_type,
            llm_service_type=llm_service_type,
            llm_service_config=llm_service_config,
            task_description=task_description,
            task_config=task_config,
        )
        self.launch_task(agent_type, task_description, task_config, str(handle.sandbox_id))
        return handle

    def setup_task_record(
        self,
        sandbox_name: str,
        task_record: benchmark_support.BenchmarkTaskRecord,
    ) -> PreparedBenchmarkSandbox:
        self._ensure_task_record_inputs(sandbox_name, task_record)
        if task_record.docker_compose_file is not None:
            prepared = self._prepare_compose_task_record(
                sandbox_name=sandbox_name,
                task_record=task_record,
            )
        else:
            prepared = self._prepare_runc_task_record(
                sandbox_name=sandbox_name,
                task_record=task_record,
            )
        self._set_benchmark_launch_metadata(prepared.handle, sandbox_name=sandbox_name, task_record=task_record)
        self._prepare_prepared_runtime(prepared)
        return prepared

    def _set_benchmark_launch_metadata(
        self,
        handle: SandboxHandle,
        *,
        sandbox_name: str,
        task_record: benchmark_support.BenchmarkTaskRecord,
    ) -> None:
        handle.launch_metadata["benchmark"] = {
            "task_id": task_record.task_id or (task_record.task_root.name if task_record.task_root is not None else sandbox_name),
            "task_run_id": sandbox_name,
            "trace_response_count": task_record.trace_response_count,
            "trace_malformed_line_count": task_record.trace_malformed_line_count,
            "llm_service_config": None if task_record.llm_service_config is None else dict(task_record.llm_service_config),
        }

    def _normalized_rootfs_reuse_recipe(
        self,
        runtime_metadata: dict[str, object],
        *,
        agent_type: str,
        compose_file: Path | None,
        service_name: str | None,
        persist_across_runs: bool,
    ) -> dict[str, object]:
        init_dirs = sorted({str(item).lstrip("/") for item in runtime_metadata.get("rootfs_init_dirs", [])})
        copy_paths: list[dict[str, str]] = []
        for item in runtime_metadata.get("rootfs_copy_paths", []):
            if not isinstance(item, dict):
                raise ValueError(f"unsupported rootfs copy item for reuse key: {item!r}")
            copy_paths.append(
                {
                    "source": str(Path(str(item["source"])).expanduser().resolve()),
                    "destination": f"/{str(item['destination']).lstrip('/')}",
                }
            )
        copy_paths.sort(key=lambda item: (item["destination"], item["source"]))
        recipe: dict[str, object] = {
            "version": 1,
            "agent_type": agent_type,
            "docker_compose_file": None if compose_file is None else str(compose_file.expanduser().resolve()),
            "service_name": "" if service_name is None else str(service_name),
            "rootfs_init_dirs": init_dirs,
            "rootfs_copy_paths": copy_paths,
        }
        if not persist_across_runs:
            recipe["session_key"] = self._rootfs_reuse_session_key
        return recipe

    def _attach_shared_rootfs_launch_metadata(
        self,
        runtime_metadata: dict[str, object],
        *,
        agent_type: str,
        compose_file: Path | None,
        service_name: str | None,
    ) -> None:
        if not self.rootfs_reuse_enabled:
            return
        persist_across_runs = self.reuse_zpool and compose_file is not None
        recipe = self._normalized_rootfs_reuse_recipe(
            runtime_metadata,
            agent_type=agent_type,
            compose_file=compose_file,
            service_name=service_name,
            persist_across_runs=persist_across_runs,
        )
        recipe_json = json.dumps(recipe, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        runtime_metadata[_SHARED_ROOTFS_KEY_METADATA_KEY] = hashlib.sha256(
            recipe_json.encode("utf-8")
        ).hexdigest()[:32]
        runtime_metadata[_SHARED_ROOTFS_PERSIST_METADATA_KEY] = persist_across_runs
        sandbox_id = str(runtime_metadata.get("sandbox_id", ""))
        logger.info(
            "Configured shared rootfs reuse sandbox=%s agent=%s key=%s persist=%s compose_file=%s service=%s",
            sandbox_id,
            agent_type,
            runtime_metadata[_SHARED_ROOTFS_KEY_METADATA_KEY],
            persist_across_runs,
            "" if compose_file is None else compose_file.expanduser().resolve(),
            "" if service_name is None else service_name,
        )
        logger.debug(
            "Shared rootfs reuse recipe sandbox=%s init_dirs=%s copy_paths=%s",
            sandbox_id,
            recipe["rootfs_init_dirs"],
            recipe["rootfs_copy_paths"],
        )

    def _prepare_runc_task_record(
        self,
        *,
        sandbox_name: str,
        task_record: benchmark_support.BenchmarkTaskRecord,
    ) -> PreparedBenchmarkSandbox:
        assert self.base_inspector is not None
        assert self.system is not None
        assert self.interceptor is not None
        sandbox_image = self.ensure_sandbox_image(task_record.agent_type)
        resolved_llm_service_type = self.resolve_llm_service_type(
            agent_type=task_record.agent_type,
            llm_service_type=task_record.llm_service_type,
        )
        network_lease = (
            self.network_manager.allocate_lease(SandboxId(sandbox_name))
            if self._agent_requires_benchmark_network(task_record.agent_type)
            else None
        )
        handle, work_dir_host_path = self._prepare_sandbox_handle(
            sandbox_name,
            interceptor_host=self.network_manager.bridge_ip if network_lease is not None else "127.0.0.1",
            network_lease=network_lease,
            agent_type=task_record.agent_type,
            llm_service_type=resolved_llm_service_type,
            llm_service_config=task_record.llm_service_config,
            image_defaults=sandbox_image.image_defaults,
            image_rootfs_dir=sandbox_image.exported_rootfs,
        )
        handle.task_description = task_record.task_description
        handle.task_config = task_record.task_config
        prelaunch_task_run = self.build_task_run(
            task_record.agent_type,
            handle,
            task_record.task_description,
            task_record.task_config,
        )
        self._run_benchmark_setup_step(
            handle,
            name="benchmark.setup.prepare_sandbox",
            setup_step="prepare_sandbox",
            fn=prelaunch_task_run.prepare_sandbox,
        )
        self._run_benchmark_setup_step(
            handle,
            name="benchmark.setup.configure_bundle",
            setup_step="configure_bundle",
            fn=prelaunch_task_run.configure_bundle,
        )
        handle.launch_metadata = _merge_runtime_launch_metadata(
            {
                "sandbox_id": sandbox_name,
                "bundle_path": str(handle.bundle_dir),
                "work_dir_host_path": None if work_dir_host_path is None else str(work_dir_host_path),
                "rootfs_init_dirs": prelaunch_task_run.rootfs_init_dirs(),
                "rootfs_copy_paths": [{"source": str(sandbox_image.exported_rootfs), "destination": "/"}],
            },
            prelaunch_task_run.extra_launch_metadata(),
            handle.launch_metadata.get("runtime", {}),
        )
        self._attach_shared_rootfs_launch_metadata(
            handle.launch_metadata,
            agent_type=task_record.agent_type,
            compose_file=None,
            service_name=None,
        )
        return PreparedBenchmarkSandbox(
            handle=handle,
            sandbox_name=sandbox_name,
            task_record=task_record,
            prelaunch_task_run=prelaunch_task_run,
            work_dir_host_path=work_dir_host_path,
            wait_for_ready_before_task_start=True,
        )

    def _prepare_compose_task_record(
        self,
        *,
        sandbox_name: str,
        task_record: benchmark_support.BenchmarkTaskRecord,
    ) -> PreparedBenchmarkSandbox:
        assert task_record.docker_compose_file is not None
        compose_env = self._build_termnius_compose_env(
            sandbox_name=sandbox_name,
            task_root=task_record.task_root,
        )
        service_name, service = sandbox_compose.load_compose_service(
            compose_file=task_record.docker_compose_file,
            env_file=task_record.env_file,
            extra_env=compose_env,
            service_name=task_record.service_name,
        )
        network_lease = self.network_manager.allocate_lease(SandboxId(sandbox_name))
        handle, work_dir_host_path = self._prepare_sandbox_handle(
            sandbox_name,
            interceptor_host=self.network_manager.bridge_ip,
            network_lease=network_lease,
            agent_type=task_record.agent_type,
            llm_service_type=self.resolve_llm_service_type(
                agent_type=task_record.agent_type,
                llm_service_type=task_record.llm_service_type,
            ),
            llm_service_config=task_record.llm_service_config,
            status_host=network_lease.guest_ip,
        )
        handle.task_description = task_record.task_description
        handle.task_config = task_record.task_config
        prelaunch_task_run = self.build_task_run(
            task_record.agent_type,
            handle,
            task_record.task_description,
            task_record.task_config,
        )
        self._run_benchmark_setup_step(
            handle,
            name="benchmark.setup.prepare_sandbox",
            setup_step="prepare_sandbox",
            fn=prelaunch_task_run.prepare_sandbox,
        )
        translation = self._run_benchmark_setup_step(
            handle,
            name="benchmark.setup.compose_translate",
            setup_step="compose_translate",
            fn=lambda: sandbox_compose.translate_compose_service(
                compose_file=task_record.docker_compose_file,
                service_name=service_name,
                service=service,
                bundle_dir=handle.bundle_dir,
                sandbox_id=str(handle.sandbox_id),
                work_dir_host_path=work_dir_host_path,
                compose_image_root=self.image_cache_root,
                compose_image_tags=self._compose_image_tags,
                telemetry=self.telemetry,
            ),
            extra={
                "compose_file": str(task_record.docker_compose_file),
                "compose_service": service_name,
            },
        )
        handle.launch_source = "compose"
        handle.launch_metadata["runtime"] = dict(translation.runtime_launch_metadata)
        handle.launch_metadata["compose"] = dict(translation.compose_launch_metadata)
        if task_record.task_root is not None:
            handle.launch_metadata["task_root"] = str(task_record.task_root)
            self._extend_termnius_rootfs_materialization(handle.launch_metadata["runtime"], task_record.task_root)
        self._ensure_termnius_dns_materialization(
            handle.launch_metadata["runtime"], handle.bundle_dir
        )
        self._configure_termnius_bundle_privileges(handle.bundle_dir)
        handle.launch_metadata["runtime"] = _merge_runtime_launch_metadata(
            handle.launch_metadata["runtime"],
            {"rootfs_init_dirs": prelaunch_task_run.rootfs_init_dirs()},
            prelaunch_task_run.extra_launch_metadata(),
        )
        runtime_metadata = handle.launch_metadata["runtime"]
        self._attach_shared_rootfs_launch_metadata(
            runtime_metadata,
            agent_type=task_record.agent_type,
            compose_file=task_record.docker_compose_file,
            service_name=service_name,
        )
        self._run_benchmark_setup_step(
            handle,
            name="benchmark.setup.configure_bundle",
            setup_step="configure_bundle",
            fn=prelaunch_task_run.configure_bundle,
        )
        return PreparedBenchmarkSandbox(
            handle=handle,
            sandbox_name=sandbox_name,
            task_record=task_record,
            prelaunch_task_run=prelaunch_task_run,
            work_dir_host_path=work_dir_host_path,
            wait_for_ready_after_task_start=True,
        )

    def run_prepared_task_record(
        self,
        prepared: PreparedBenchmarkSandbox,
    ) -> SandboxHandle:
        self._prepare_prepared_runtime(prepared)
        self._launch_prepared_runtime(prepared)
        handle = prepared.handle
        if handle.task_description is not None and handle.task_config is not None:
            self.launch_task(
                handle.agent_type,
                handle.task_description,
                handle.task_config,
                str(handle.sandbox_id),
            )
        if prepared.wait_for_ready_after_task_start and handle.task_run is not None:
            try:
                handle.task_run.wait_for_task_ready()
            except RuntimeError:
                pass
            else:
                if prepared.emit_ready_event:
                    self.emit_benchmark_event("benchmark.task.ready", handle)
        return handle

    def _prepare_prepared_runtime(
        self,
        prepared: PreparedBenchmarkSandbox,
    ) -> None:
        if prepared.runtime_prepared:
            logger.info("Benchmark runtime already prepared sandbox=%s", prepared.handle.sandbox_id)
            return
        runtime = self._active_runtime
        if runtime is None:
            raise RuntimeError("runtime is not initialized")
        prepare_launch = getattr(runtime, "prepare_launch", None)
        if not callable(prepare_launch):
            return
        handle = prepared.handle
        runtime_metadata = handle.launch_metadata.get("runtime", handle.launch_metadata)
        logger.info("Benchmark preparing runtime sandbox=%s phase=setup", handle.sandbox_id)
        self._run_benchmark_setup_step(
            handle,
            name="benchmark.setup.runtime_prepare",
            setup_step="runtime_prepare",
            fn=lambda: prepare_launch("runc", runtime_metadata),
            extra={"launch_source": handle.launch_source},
        )
        prepared.runtime_prepared = True
        logger.info("Benchmark prepared runtime sandbox=%s phase=setup", handle.sandbox_id)

    def _launch_prepared_runtime(
        self,
        prepared: PreparedBenchmarkSandbox,
    ) -> None:
        if prepared.runtime_launched:
            logger.info("Benchmark runtime already launched sandbox=%s", prepared.handle.sandbox_id)
            return
        assert self.base_inspector is not None
        runtime = self._active_runtime
        if runtime is None:
            raise RuntimeError("runtime is not initialized")
        handle = prepared.handle
        logger.info("Benchmark launching prepared runtime sandbox=%s phase=run", handle.sandbox_id)
        network_lease = self.network_manager.lease_for(handle.sandbox_id)
        if network_lease is not None:
            self.network_manager.register_guest_ip(network_lease.guest_ip, handle.sandbox_id)
        self.base_inspector.upsert_snapshot(
            SandboxSnapshot(
                sandbox_id=handle.sandbox_id,
                runtime_name="runc",
                is_running=True,
                process_changed=False,
                filesystem_changed=False,
                observed_at=utc_now(),
            )
        )
        runtime_metadata = handle.launch_metadata.get("runtime", handle.launch_metadata)
        runtime.launch("runc", runtime_metadata)
        handle.last_status = {}
        if prepared.prelaunch_task_run is not None and prepared.wait_for_ready_before_task_start:
            prepared.prelaunch_task_run.wait_for_task_ready()
            if prepared.emit_ready_event:
                self.emit_benchmark_event("benchmark.task.ready", handle)
        prepared.runtime_launched = True
        logger.info("Benchmark launched prepared runtime sandbox=%s phase=run", handle.sandbox_id)

    def launch_task_record(
        self,
        sandbox_name: str,
        task_record: benchmark_support.BenchmarkTaskRecord,
    ) -> SandboxHandle:
        if task_record.docker_compose_file is not None and self.root is None:
            handle = self.launch_sandbox_from_docker_compose_file(
                task_record.docker_compose_file,
                task_record.env_file,
                sandbox_name=sandbox_name,
                service_name=task_record.service_name,
                agent_type=task_record.agent_type,
                llm_service_type=task_record.llm_service_type,
                llm_service_config=task_record.llm_service_config,
                task_description=task_record.task_description,
                task_config=task_record.task_config,
                task_root=task_record.task_root,
            )
            self._set_benchmark_launch_metadata(handle, sandbox_name=sandbox_name, task_record=task_record)
            return handle
        prepared = self.setup_task_record(sandbox_name, task_record)
        handle = self.run_prepared_task_record(prepared)
        return handle

    def launch_sandbox_from_docker_compose_file(
        self,
        compose_file: Path,
        env_file: Path | None,
        *,
        sandbox_name: str,
        service_name: str | None = None,
        status_host: str | None = None,
        status_port: int | None = None,
        agent_type: str | None = None,
        llm_service_type: str | None = None,
        llm_service_config: dict[str, object] | None = None,
        task_description: TaskDescription | None = None,
        task_config: TaskConfig | None = None,
        task_root: Path | None = None,
    ) -> SandboxHandle:
        compose_env = self._build_termnius_compose_env(
            sandbox_name=sandbox_name,
            task_root=task_root,
        )
        service_name, service = sandbox_compose.load_compose_service(
            compose_file=compose_file,
            env_file=env_file,
            extra_env=compose_env,
            service_name=service_name,
        )
        network_lease = self.network_manager.allocate_lease(SandboxId(sandbox_name))
        handle, work_dir_host_path = self._prepare_sandbox_handle(
            sandbox_name,
            interceptor_host=self.network_manager.bridge_ip,
            network_lease=network_lease,
            agent_type=agent_type or "simulated",
            llm_service_type=self.resolve_llm_service_type(
                agent_type=agent_type or "simulated",
                llm_service_type=llm_service_type,
            ),
            llm_service_config=llm_service_config,
            status_port=status_port,
            status_host=status_host if status_host is not None else network_lease.guest_ip,
        )
        prelaunch_task_run = None
        if agent_type is not None and task_description is not None and task_config is not None:
            handle.task_description = task_description
            handle.task_config = task_config
            prelaunch_task_run = self.build_task_run(agent_type, handle, task_description, task_config)
            prelaunch_task_run.prepare_sandbox()
        translation = sandbox_compose.translate_compose_service(
            compose_file=compose_file,
            service_name=service_name,
            service=service,
            bundle_dir=handle.bundle_dir,
            sandbox_id=str(handle.sandbox_id),
            work_dir_host_path=work_dir_host_path,
            compose_image_root=self.image_cache_root,
            compose_image_tags=self._compose_image_tags,
            telemetry=self.telemetry,
        )
        assert self.base_inspector is not None
        assert self.system is not None
        handle.launch_source = "compose"
        handle.launch_metadata["runtime"] = dict(translation.runtime_launch_metadata)
        handle.launch_metadata["compose"] = dict(translation.compose_launch_metadata)
        if task_root is not None:
            handle.launch_metadata["task_root"] = str(task_root)
            self._extend_termnius_rootfs_materialization(handle.launch_metadata["runtime"], task_root)
        self._ensure_termnius_dns_materialization(
            handle.launch_metadata["runtime"], handle.bundle_dir
        )
        self._configure_termnius_bundle_privileges(handle.bundle_dir)
        if prelaunch_task_run is not None:
            handle.launch_metadata["runtime"] = _merge_runtime_launch_metadata(
                handle.launch_metadata["runtime"],
                {"rootfs_init_dirs": prelaunch_task_run.rootfs_init_dirs()},
                prelaunch_task_run.extra_launch_metadata(),
            )
            prelaunch_task_run.configure_bundle()
        self._attach_shared_rootfs_launch_metadata(
            handle.launch_metadata["runtime"],
            agent_type=agent_type or "simulated",
            compose_file=compose_file,
            service_name=service_name,
        )
        self.network_manager.register_guest_ip(network_lease.guest_ip, handle.sandbox_id)
        self.base_inspector.upsert_snapshot(
            SandboxSnapshot(
                sandbox_id=handle.sandbox_id,
                runtime_name="runc",
                is_running=True,
                process_changed=False,
                filesystem_changed=False,
                observed_at=utc_now(),
            )
        )
        runtime = self._active_runtime
        if runtime is None:
            raise RuntimeError("runtime is not initialized")
        runtime.launch("runc", handle.launch_metadata["runtime"])
        handle.last_status = {}
        if agent_type is not None and task_description is not None and task_config is not None:
            self.launch_task(agent_type, task_description, task_config, str(handle.sandbox_id))
            if handle.task_run is not None:
                try:
                    handle.task_run.wait_for_task_ready()
                except RuntimeError:
                    pass
        return handle

    def _build_termnius_compose_env(
        self,
        *,
        sandbox_name: str,
        task_root: Path | None,
    ) -> dict[str, str]:
        task_id = "task" if task_root is None else task_root.name
        host_logs_root = self._effective_agent_host_root() / "termnius-logs" / sandbox_name
        task_logs_path = host_logs_root / "logs"
        task_agent_logs_path = host_logs_root / "agent-logs"
        task_logs_path.mkdir(parents=True, exist_ok=True)
        task_agent_logs_path.mkdir(parents=True, exist_ok=True)
        image_component = sandbox_image.docker_tag_component(f"{task_id}")
        return {
            "T_BENCH_CONTAINER_LOGS_PATH": "/logs",
            "T_BENCH_CONTAINER_AGENT_LOGS_PATH": "/agent-logs",
            "T_BENCH_TASK_LOGS_PATH": str(task_logs_path),
            "T_BENCH_TASK_AGENT_LOGS_PATH": str(task_agent_logs_path),
            "T_BENCH_TASK_DOCKER_CLIENT_CONTAINER_NAME": f"crab-{image_component}",
            "T_BENCH_TASK_DOCKER_CLIENT_IMAGE_NAME": f"crab-termnius-{image_component}",
            "T_BENCH_TEST_DIR": "/tests",
        }

    def _extend_termnius_rootfs_materialization(
        self,
        runtime_metadata: dict[str, object],
        task_root: Path,
    ) -> None:
        tests_dir = task_root / "tests"
        run_tests = task_root / "run-tests.sh"
        if not run_tests.is_file():
            raise FileNotFoundError(f"missing task run-tests.sh: {run_tests}")
        copy_paths = list(runtime_metadata.get("rootfs_copy_paths", []))
        if tests_dir.is_dir():
            copy_paths.append({"source": str(tests_dir), "destination": "/tests"})
        copy_paths.append({"source": str(run_tests), "destination": "/tests/run-tests.sh"})
        runtime_metadata["rootfs_copy_paths"] = copy_paths
        init_dirs = set(runtime_metadata.get("rootfs_init_dirs", []))
        init_dirs.add("tests")
        runtime_metadata["rootfs_init_dirs"] = sorted(init_dirs)

    def _ensure_termnius_dns_materialization(
        self, runtime_metadata: dict[str, object], bundle_dir: Path
    ) -> None:
        from integrations.sandboxes.runtime.baseline import add_dns_materialization

        add_dns_materialization(runtime_metadata, bundle_dir=bundle_dir)

    def _configure_termnius_bundle_privileges(self, bundle_dir: Path) -> None:
        from integrations.sandboxes.runtime.baseline import (
            apply_sandbox_bundle_baseline,
        )

        apply_sandbox_bundle_baseline(bundle_dir)

    def add_interceptor_hook(self, hook: RequestInterceptorHook) -> None:
        self.interceptor_hook.add_hook(hook)
        logger.debug("Registered interceptor hook %s", type(hook).__name__)

    def resolve_interceptor_sandbox_id(
        self,
        client_host: str | None,
        headers: dict[str, str],
        body: bytes,
    ) -> str | None:
        _ = body
        sandbox_id_from_header = str(headers.get("X-Agent-Sandbox-Id", "")).strip()
        if sandbox_id_from_header:
            return sandbox_id_from_header
        sandbox_id = self.network_manager.resolve_sandbox_id(client_host)
        if sandbox_id is None:
            return None
        return str(sandbox_id)

    def drain_request_state_changes(self) -> int:
        if self.request_state_store is None:
            return 0
        drained = 0
        while self.request_state_store.wait_for_change(timeout=0.0) is not None:
            drained += 1
        if drained:
            logger.info("Drained %d queued request-state changes", drained)
        return drained

    def set_snapshot_metadata(self, sandbox: SandboxHandle, **metadata: object) -> None:
        self.set_snapshot_metadata_by_id(sandbox.sandbox_id, **metadata)

    def set_snapshot_metadata_by_id(self, sandbox_id: SandboxId, **metadata: object) -> None:
        assert self.base_inspector is not None
        snapshot = self.base_inspector.inspect(sandbox_id)
        merged = {**snapshot.metadata, **metadata}
        self.base_inspector.upsert_snapshot(
            replace(
                snapshot,
                observed_at=utc_now(),
                metadata=merged,
            )
        )

    def clear_snapshot_metadata(self, sandbox: SandboxHandle, *keys: str) -> None:
        self.clear_snapshot_metadata_by_id(sandbox.sandbox_id, *keys)

    def clear_snapshot_metadata_by_id(self, sandbox_id: SandboxId, *keys: str) -> None:
        assert self.base_inspector is not None
        snapshot = self.base_inspector.inspect(sandbox_id)
        metadata = dict(snapshot.metadata)
        for key in keys:
            metadata.pop(key, None)
        self.base_inspector.upsert_snapshot(
            replace(
                snapshot,
                observed_at=utc_now(),
                metadata=metadata,
            )
        )

    def checkpoint_manual(self, sandbox: SandboxHandle, leave_running: bool=False):
        assert self.system is not None
        if sandbox.launch_source not in {"runc", "compose"}:
            raise RuntimeError(f"checkpoint_manual unsupported for launch_source={sandbox.launch_source}")
        logger.debug("Benchmark requesting checkpoint_manual for sandbox=%s", sandbox.sandbox_id)
        result = self.system.checkpoint_once(sandbox.sandbox_id, leave_running=leave_running)
        self._notify_speculation_checkpoint_complete(sandbox.sandbox_id, result)
        return result

    def _notify_speculation_checkpoint_complete(
        self,
        sandbox_id: SandboxId,
        result: object,
    ) -> None:
        """Kick the per-sandbox speculation controller's prefork. Idempotent
        per controller; the controller itself dedupes against an in-flight
        prefork or a still-warm cache. Called from every harness path that
        produces a new checkpoint on the active sandbox."""
        controller = self._speculation_controllers_by_sandbox.get(sandbox_id)
        if controller is None:
            return
        # Be permissive about result shape: scenarios that pass a non-result
        # (or no result at all) still benefit from the prefork hint.
        succeeded = True
        if result is not None:
            status = getattr(result, "status", None)
            value = getattr(status, "value", None)
            if value is not None and value != "succeeded":
                succeeded = False
        if not succeeded:
            return
        try:
            controller.kick_prefork()
        except Exception:
            logger.exception(
                "Failed to kick prefork for sandbox=%s", sandbox_id
            )

    def checkpoint_manual_filesystem_only(self, sandbox: SandboxHandle, leave_running: bool=False):
        assert self.system is not None
        if sandbox.launch_source not in {"runc", "compose"}:
            raise RuntimeError(f"checkpoint_manual_filesystem_only unsupported for launch_source={sandbox.launch_source}")
        logger.debug("Benchmark requesting filesystem-only checkpoint for sandbox=%s", sandbox.sandbox_id)
        pending_request = self.system._next_pending_live_request(sandbox.sandbox_id)
        paused = self.system._pause_for_manual_checkpoint(sandbox.sandbox_id)
        result = None
        job = None
        try:
            job = CheckpointJob(
                job_id=JobId.new(),
                sandbox_id=sandbox.sandbox_id,
                requested_at=utc_now(),
                reason="manual_filesystem_only",
                checkpoint_process=False,
                checkpoint_filesystem=True,
                leave_running=leave_running,
                metadata=self.system._build_checkpoint_metadata(
                    sandbox.sandbox_id,
                    pending_request=pending_request,
                ),
            )
            result = self.system.executor.run_checkpoint(job)
            if result.status.value == "succeeded":
                self.system.scheduler.mark_checkpoint_complete(sandbox.sandbox_id, result.finished_at)
                self.system.inspector.mark_checkpoint_complete(
                    sandbox.sandbox_id,
                    process=False,
                    filesystem=True,
                    at=result.finished_at,
                )
        finally:
            if paused and self.system._should_resume_after_checkpoint(job, result):
                self.system._resume_sandbox(sandbox.sandbox_id)
            self.system._release_response_gate(sandbox.sandbox_id, pending_request)
            self.system._refresh_interceptor_pending_state(sandbox.sandbox_id)
        assert result is not None
        return result

    def checkpoint_if_due(self, sandbox: SandboxHandle):
        assert self.system is not None
        if sandbox.launch_source not in {"runc", "compose"}:
            raise RuntimeError(f"checkpoint_if_due unsupported for launch_source={sandbox.launch_source}")
        logger.debug("Benchmark requesting checkpoint_if_due for sandbox=%s", sandbox.sandbox_id)
        result = self.system.checkpoint_if_due(sandbox.sandbox_id)
        self._notify_speculation_checkpoint_complete(sandbox.sandbox_id, result)
        return result

    def restore_once(self, sandbox: SandboxHandle, checkpoint_id: CheckpointId):
        assert self.system is not None
        if sandbox.launch_source not in {"runc", "compose"}:
            raise RuntimeError(f"restore_once unsupported for launch_source={sandbox.launch_source}")
        logger.info(
            "Benchmark requesting restore sandbox=%s checkpoint=%s transfer_delay_ms=%.1f",
            sandbox.sandbox_id,
            checkpoint_id,
            self.transfer_delay_ms,
        )
        if self.transfer_delay_ms > 0:
            time.sleep(self.transfer_delay_ms / 1000.0)
        result = self.system.restore_once(sandbox.sandbox_id, checkpoint_id)
        if result.status.value == "succeeded":
            self._repair_sandbox_network_after_recovery(sandbox)
            if sandbox.task_run is not None:
                sandbox.task_run.on_restore_complete()
        return result

    def restore_tree_search_fork(self, sandbox: SandboxHandle, checkpoint_id: CheckpointId):
        result = self.restore_once(sandbox, checkpoint_id)
        if result.status.value == "succeeded":
            return result
        if not self._should_fallback_tree_search_restore_to_relaunch(sandbox, result):
            return result
        manifest = self._resolve_tree_search_restore_manifest(sandbox.sandbox_id, checkpoint_id)
        if manifest is None:
            return result
        logger.warning(
            "Falling back to relaunch for tree-search fork restore sandbox=%s checkpoint=%s message=%s",
            sandbox.sandbox_id,
            checkpoint_id,
            result.message,
        )
        self._restore_llm_service_state(sandbox.sandbox_id, manifest)
        self.relaunch_sandbox(sandbox, preserve_filesystem_state=True)
        self._repair_sandbox_network_after_recovery(sandbox)
        return RestoreResult(
            job_id=result.job_id,
            sandbox_id=result.sandbox_id,
            checkpoint_id=result.checkpoint_id,
            status=JobStatus.SUCCEEDED,
            started_at=result.started_at,
            finished_at=utc_now(),
            failure_code=FailureCode.NONE,
            message="restore_failed_relaunch_fallback",
            operation_statuses=result.operation_statuses,
        )

    def _should_fallback_tree_search_restore_to_relaunch(
        self,
        sandbox: SandboxHandle,
        result: RestoreResult,
    ) -> bool:
        # Same opt-in as the system-level restore-failure handler: relaunch is
        # an availability fallback that can hide real bugs, so it stays off
        # unless the operator explicitly enables it.
        if not self.relaunch_on_restore_failure:
            return False
        if sandbox.launch_source != "compose":
            return False
        if not benchmark_support.is_replay_llm_service_type(sandbox.llm_service_type):
            return False
        if sandbox.agent_type is None or not self._agent_requires_benchmark_network(sandbox.agent_type):
            return False
        message = (result.message or "").lower()
        return "can't bind inet socket" in message or "cannot assign requested address" in message

    def _resolve_tree_search_restore_manifest(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> CheckpointManifest | None:
        if self.storage is None:
            return None
        try:
            manifest = self.storage.get_manifest(sandbox_id, checkpoint_id)
            return resolve_restore_manifest(self.storage, manifest)
        except Exception:
            logger.exception(
                "Failed to resolve tree-search restore manifest sandbox=%s checkpoint=%s",
                sandbox_id,
                checkpoint_id,
            )
            return None

    def notify_fault(self, sandbox: SandboxHandle, *, reason: str = "fault") -> None:
        assert self.system is not None
        logger.info("Benchmark notifying fault sandbox=%s reason=%s", sandbox.sandbox_id, reason)
        self.system.notify_fault(sandbox.sandbox_id, reason=reason)

    def notify_preemption(self, sandbox: SandboxHandle, *, grace_remaining_seconds: float) -> None:
        assert self.system is not None
        logger.info(
            "Benchmark notifying preemption sandbox=%s grace_remaining_seconds=%.3f",
            sandbox.sandbox_id,
            grace_remaining_seconds,
        )
        self.system.notify_preemption(sandbox.sandbox_id, grace_remaining_seconds=grace_remaining_seconds)

    def wait_for_recovery(
        self,
        sandbox: SandboxHandle,
        *,
        event_type: str,
        observed_after,
        timeout_s: float | None = None,
    ):
        assert self.system is not None
        if timeout_s is None:
            timeout_s = max(60.0, benchmark_support.task_timeout_seconds(sandbox.task_config or TaskConfig()))

        def _matching_record():
            record = self.system.get_last_recovery_record(sandbox.sandbox_id)
            if record is None:
                return None
            if record.event_type != event_type:
                return None
            if record.started_at < observed_after:
                return None
            return record

        benchmark_support.wait_for(lambda: _matching_record() is not None, timeout_s=timeout_s)
        record = _matching_record()
        if record is not None and record.status in {"restored", "relaunched"} and sandbox.task_run is not None:
            self._repair_sandbox_network_after_recovery(sandbox)
            sandbox.task_run.on_restore_complete()
            wait_for_task_ready = getattr(sandbox.task_run, "wait_for_task_ready", None)
            if callable(wait_for_task_ready):
                wait_for_task_ready()
                self.emit_benchmark_event("benchmark.task.ready", sandbox, event_type=event_type)
        logger.info(
            "Observed recovery record sandbox=%s event_type=%s status=%s checkpoint=%s",
            sandbox.sandbox_id,
            event_type,
            record.status if record is not None else "missing",
            "" if record is None or record.checkpoint_id is None else record.checkpoint_id,
        )
        return record

    def list_checkpoint_manifests(self, sandbox_id: SandboxId) -> list[CheckpointManifest]:
        assert self.storage is not None
        manifests: list[CheckpointManifest] = []
        for checkpoint_id in self.storage.list_checkpoints(sandbox_id):
            try:
                manifests.append(self.storage.get_manifest(sandbox_id, checkpoint_id))
            except FileNotFoundError:
                logger.debug(
                    "Skipped checkpoint manifest that disappeared during enumeration sandbox=%s checkpoint=%s",
                    sandbox_id,
                    checkpoint_id,
                )
        return manifests

    def _repair_sandbox_network_after_recovery(self, sandbox: SandboxHandle) -> None:
        if not sandbox.agent_type:
            return
        if not self._agent_requires_benchmark_network(sandbox.agent_type):
            return
        repaired = self.network_manager.repair_lease(sandbox.sandbox_id)
        if not repaired:
            logger.debug(
                "Skipped benchmark network repair sandbox=%s reason=no_lease_or_repair_failed",
                sandbox.sandbox_id,
            )

    def collect_tree_search_checkpoints(
        self,
        sandbox_id: SandboxId,
        *,
        initial_steps: int | None = None,
        require_complete: bool = False,
    ) -> dict[int, benchmark_support.TreeSearchCheckpointRecord]:
        return benchmark_support.build_tree_search_checkpoint_index(
            self.list_checkpoint_manifests(sandbox_id),
            initial_steps=initial_steps,
            require_complete=require_complete,
        )

    def wait_for_tree_search_checkpoints(
        self,
        sandbox_id: SandboxId,
        *,
        initial_steps: int,
        require_complete: bool = True,
        timeout_s: float = 45.0,
    ) -> dict[int, benchmark_support.TreeSearchCheckpointRecord]:
        collected: dict[int, benchmark_support.TreeSearchCheckpointRecord] = {}
        last_error = f"missing tree-search checkpoints for steps {list(range(1, initial_steps + 1))}"

        if not require_complete:
            self.wait_for_checkpoint_count_stable(sandbox_id, timeout_s=timeout_s)
            collected = self.collect_tree_search_checkpoints(
                sandbox_id,
                initial_steps=initial_steps,
                require_complete=False,
            )
            logger.info(
                "Collected sparse tree-search checkpoints sandbox=%s steps=%s",
                sandbox_id,
                sorted(collected.keys()),
            )
            return collected

        def _ready() -> bool:
            nonlocal collected
            nonlocal last_error
            try:
                collected = self.collect_tree_search_checkpoints(
                    sandbox_id,
                    initial_steps=initial_steps,
                    require_complete=True,
                )
            except ValueError as exc:
                message = str(exc)
                if message.startswith("missing tree-search checkpoints"):
                    last_error = message
                    return False
                raise
            return True

        if not benchmark_support.wait_for(_ready, timeout_s=timeout_s, interval_s=0.2, raise_on_timeout=False):
            raise RuntimeError(f"timed out waiting for tree-search checkpoints for sandbox {sandbox_id}: {last_error}")
        logger.info(
            "Collected tree-search checkpoints sandbox=%s steps=%s",
            sandbox_id,
            sorted(collected.keys()),
        )
        return collected

    def wait_for_checkpoint_count(
        self,
        sandbox_id: SandboxId,
        *,
        minimum: int,
        timeout_s: float = 45.0,
    ) -> int:
        assert self.storage is not None
        benchmark_support.wait_for(lambda: len(self.storage.list_checkpoints(sandbox_id)) >= minimum, timeout_s=timeout_s)
        return len(self.storage.list_checkpoints(sandbox_id))

    def wait_for_checkpoint_count_stable(
        self,
        sandbox_id: SandboxId,
        *,
        stable_period_s: float = 1.0,
        timeout_s: float = 15.0,
    ) -> int:
        assert self.storage is not None
        logger.debug(
            "Waiting for checkpoint count to stabilize sandbox=%s stable_period_s=%.1f timeout_s=%.1f",
            sandbox_id,
            stable_period_s,
            timeout_s,
        )
        deadline = time.time() + timeout_s
        stable_since = time.time()
        last_count = len(self.storage.list_checkpoints(sandbox_id))
        while time.time() < deadline:
            current = len(self.storage.list_checkpoints(sandbox_id))
            if current != last_count:
                logger.debug(
                    "Checkpoint count changed sandbox=%s previous=%d current=%d",
                    sandbox_id,
                    last_count,
                    current,
                )
                last_count = current
                stable_since = time.time()
            elif time.time() - stable_since >= stable_period_s:
                logger.info("Checkpoint count stabilized sandbox=%s count=%d", sandbox_id, current)
                return current
            time.sleep(0.2)
        raise RuntimeError(f"checkpoint count did not stabilize for sandbox {sandbox_id}")

    def deactivate_sandbox_runtime(self, sandbox: SandboxHandle) -> None:
        logger.info("Deactivating sandbox runtime sandbox=%s", sandbox.sandbox_id)
        self._delete_runtime(sandbox.sandbox_id)
        self._set_sandbox_running_state(sandbox.sandbox_id, is_running=False)

    def _fault_injection_ready(self, sandbox: SandboxHandle) -> bool:
        request_state = None if self.request_state_store is None else self.request_state_store.get(sandbox.sandbox_id)
        request_in_flight = False if request_state is None else request_state.llm_request_in_flight
        checkpoint_active = False if self.executor is None else self.executor.has_active_checkpoint(sandbox.sandbox_id)
        return not request_in_flight and not checkpoint_active

    @staticmethod
    def _fault_injection_target_task_finished(sandbox: SandboxHandle) -> bool:
        if str(sandbox.last_status.get("state", "")).lower() == "finished":
            return True
        task_future = sandbox.task_future
        return isinstance(task_future, Future) and task_future.done()

    def wait_for_fault_injection_window(self, sandbox: SandboxHandle, *, timeout_s: float = 60.0) -> bool:
        logger.info(
            "Waiting for fault injection window sandbox=%s timeout_s=%.1f",
            sandbox.sandbox_id,
            timeout_s,
        )
        if self._fault_injection_target_task_finished(sandbox):
            logger.info(
                "Skipping fault injection for sandbox=%s because task already finished state=%s task_future_done=%s",
                sandbox.sandbox_id,
                str(sandbox.last_status.get("state", "")),
                bool(sandbox.task_future is not None and sandbox.task_future.done()),
            )
            return False
        ready = benchmark_support.wait_for(
            lambda: self._fault_injection_ready(sandbox) or self._fault_injection_target_task_finished(sandbox),
            timeout_s=timeout_s,
            interval_s=0.01,
            raise_on_timeout=False,
        )
        if self._fault_injection_target_task_finished(sandbox):
            logger.info(
                "Skipping fault injection for sandbox=%s because task finished before the injection window was used "
                "(state=%s task_future_done=%s)",
                sandbox.sandbox_id,
                str(sandbox.last_status.get("state", "")),
                bool(sandbox.task_future is not None and sandbox.task_future.done()),
            )
            return False
        if ready:
            logger.info("Fault injection window ready sandbox=%s", sandbox.sandbox_id)
            return True
        request_state = None if self.request_state_store is None else self.request_state_store.get(sandbox.sandbox_id)
        request_in_flight = False if request_state is None else request_state.llm_request_in_flight
        checkpoint_active = False if self.executor is None else self.executor.has_active_checkpoint(sandbox.sandbox_id)
        logger.error(
            "Timed out waiting for fault injection window, skipping fault injection "
            "(sandbox=%s request_in_flight=%s checkpoint_active=%s)",
            sandbox.sandbox_id,
            request_in_flight,
            checkpoint_active,
        )
        return False

    def inject_fault(self, sandbox: SandboxHandle) -> None:
        logger.info("Injecting fault into sandbox=%s", sandbox.sandbox_id)
        if not self.wait_for_fault_injection_window(sandbox):
            return
        self._delete_runtime(sandbox.sandbox_id)
        logger.info("Injected fault into sandbox=%s", sandbox.sandbox_id)
        self._set_sandbox_running_state(sandbox.sandbox_id, is_running=False)

    def destroy_sandbox_dataset(self, sandbox: SandboxHandle) -> None:
        assert self.runtime is not None
        self._release_fork_chain_pin(sandbox.sandbox_id)
        controller = self._speculation_controllers_by_sandbox.pop(sandbox.sandbox_id, None)
        if controller is not None:
            try:
                controller.shutdown()
            except Exception:
                logger.exception(
                    "Failed to shut down speculation controller sandbox=%s",
                    sandbox.sandbox_id,
                )
        self._delete_runtime(sandbox.sandbox_id)
        try:
            self.runtime.describe(sandbox.sandbox_id)
        except KeyError:
            self._unregister_host_inspector(sandbox.sandbox_id)
            self._unregister_llm_service(sandbox.sandbox_id)
            self.network_manager.release_lease(sandbox.sandbox_id)
            self._sandbox_by_id.pop(sandbox.sandbox_id, None)
            self._drop_sandbox_from_registry(sandbox)
            return
        self._destroy_filesystem_dataset(sandbox.sandbox_id)
        self._unregister_host_inspector(sandbox.sandbox_id)
        self._unregister_llm_service(sandbox.sandbox_id)
        self.network_manager.release_lease(sandbox.sandbox_id)
        self._sandbox_by_id.pop(sandbox.sandbox_id, None)
        self._drop_sandbox_from_registry(sandbox)

    def _inline_fork_chain_artifacts(self, fork_sandbox_id: SandboxId) -> None:
        """Replace any chain-shared symlinks under ``fork_sandbox_id`` with
        byte copies. Called before promoting the fork into the active role
        (the original active sandbox is about to be destroyed) and before
        any other ownership transfer that would orphan the symlinks.
        Best-effort: failures are logged but do not abort the swap.
        """
        materialize_runtime = getattr(self.runtime, "materialize_linked_pre_dumps", None)
        if callable(materialize_runtime):
            try:
                materialize_runtime(fork_sandbox_id)
            except Exception:
                logger.exception(
                    "Failed to materialize linked pre-dump dirs for sandbox=%s",
                    fork_sandbox_id,
                )
        materialize_storage = getattr(self.storage, "materialize_linked_artifacts", None)
        if callable(materialize_storage):
            try:
                materialize_storage(fork_sandbox_id)
            except Exception:
                logger.exception(
                    "Failed to materialize linked storage artifacts for sandbox=%s",
                    fork_sandbox_id,
                )

    def _release_fork_chain_pin(self, fork_sandbox_id: SandboxId) -> None:
        pin = self._fork_chain_pins.pop(fork_sandbox_id, None)
        if pin is None:
            return
        source_sandbox_id, leaf_checkpoint_id = pin
        unpin_chain = getattr(self.storage, "unpin_chain", None)
        if not callable(unpin_chain):
            return
        try:
            unpin_chain(source_sandbox_id, leaf_checkpoint_id)
        except Exception:
            logger.exception(
                "Failed to unpin chain for fork=%s source=%s leaf=%s",
                fork_sandbox_id,
                source_sandbox_id,
                leaf_checkpoint_id,
            )

    def _drop_sandbox_from_registry(self, sandbox: SandboxHandle) -> None:
        try:
            self.sandboxes.remove(sandbox)
        except ValueError:
            pass

    def _relaunch_sandbox(
        self,
        sandbox_id: SandboxId,
        event_type: str,
        preserve_filesystem_state: bool = False,
    ) -> None:
        _ = event_type
        handle = self._sandbox_by_id[sandbox_id]
        self.relaunch_sandbox(handle, preserve_filesystem_state=preserve_filesystem_state)

    def relaunch_sandbox(
        self,
        sandbox: SandboxHandle,
        *,
        preserve_filesystem_state: bool = False,
    ) -> dict[str, object]:
        assert self.base_inspector is not None
        assert self.runtime is not None

        description = self.runtime.describe(sandbox.sandbox_id)
        metadata = dict(description.metadata)
        logger.info("Relaunching sandbox=%s after recovery fallback", sandbox.sandbox_id)
        preserve_task_run = (
            sandbox.task_run is not None
            and sandbox.task_future is not None
            and not sandbox.task_future.done()
            and sandbox.task_run.survives_fault_relaunch()
            and not (sandbox.launch_source == "compose" and sandbox.llm_service_type == "iflow_trace_replay")
        )
        self._delete_runtime(sandbox.sandbox_id)
        if preserve_filesystem_state:
            metadata["_crab_runtime_reuse_existing_rootfs"] = True
        else:
            self._destroy_filesystem_dataset(sandbox.sandbox_id)
        self.runtime.launch("runc", metadata)
        if not preserve_task_run:
            self._reset_llm_service_state(sandbox.sandbox_id)
        self.base_inspector.upsert_snapshot(
            SandboxSnapshot(
                sandbox_id=sandbox.sandbox_id,
                runtime_name="runc",
                is_running=True,
                process_changed=False,
                filesystem_changed=False,
                observed_at=utc_now(),
                metadata={},
            )
        )
        payload = sandbox.last_status
        if sandbox.task_run is not None and not preserve_task_run:
            sandbox.task_run.request_stop()
        if sandbox.task_description is not None and sandbox.task_config is not None:
            if not preserve_task_run:
                self.launch_task(
                    sandbox.agent_type,
                    sandbox.task_description,
                    sandbox.task_config,
                    str(sandbox.sandbox_id),
                )
            assert sandbox.task_run is not None
            sandbox.task_run.wait_for_task_ready()
            if preserve_task_run:
                sandbox.task_run.on_restore_complete()
            try:
                payload = sandbox.task_run.poll_status()
            except RuntimeError:
                payload = sandbox.last_status
        sandbox.last_status = payload
        logger.info("Relaunched sandbox=%s and recovered status endpoint", sandbox.sandbox_id)
        return payload

    def clone_checkpoint_to_fork(
        self,
        source: SandboxHandle,
        checkpoint_id: CheckpointId,
        fork_name: str,
    ) -> SandboxHandle:
        assert self.root is not None
        assert self.storage is not None
        assert self.runtime is not None
        assert self.base_inspector is not None

        network_lease = (
            self.network_manager.allocate_lease(SandboxId(fork_name))
            if self._sandbox_requires_benchmark_network(source)
            else None
        )
        target, work_dir_host_path = self._prepare_sandbox_handle(
            fork_name,
            interceptor_host=self.network_manager.bridge_ip if network_lease is not None else "127.0.0.1",
            network_lease=network_lease,
            agent_type=source.agent_type,
            llm_service_type=source.llm_service_type,
            llm_service_config=self._sandbox_llm_service_config(source),
        )
        if network_lease is not None:
            self.network_manager.register_guest_ip(network_lease.guest_ip, target.sandbox_id)
        self._clone_host_work_dir(source.sandbox_id, target.sandbox_id)

        # When the source sandbox was launched via docker-compose translation,
        # its bundle has compose-derived bind mounts (e.g. /logs). The fork's
        # bare bundle from _prepare_sandbox_handle is missing those, so CRIU's
        # restore can't map the original mountpoint IDs and aborts with
        # `Error (criu/mount.c:3024): mnt: No mapping for X:(null) mountpoint`.
        # Copy the source bundle's mounts list onto the fork before any restore
        # so the in-memory mount layout the process image references exists.
        self._replicate_source_bundle_mounts(source=source, target=target)

        rootfs_path = target.bundle_dir / "rootfs"
        rootfs_path.mkdir(parents=True, exist_ok=True)
        manifests = {manifest.checkpoint_id: manifest for manifest in self.list_checkpoint_manifests(source.sandbox_id)}
        checkpoint_order = list(manifests.keys())
        copy_plan = benchmark_support.resolve_checkpoint_copy_plan(checkpoint_order, manifests, checkpoint_id)
        filesystem_checkpoint_id = next(copy_id for copy_id, _, copy_filesystem in reversed(copy_plan) if copy_filesystem)
        source_dataset = self._dataset_name_for(source.sandbox_id)
        target_dataset = self._clone_filesystem_snapshot(
            source.sandbox_id,
            filesystem_checkpoint_id,
            target.sandbox_id,
            target_rootfs_path=rootfs_path,
        )
        logger.info(
            "Cloning checkpoint to fork source=%s target=%s selected_checkpoint=%s filesystem_checkpoint=%s copy_plan=%s",
            source.sandbox_id,
            target.sandbox_id,
            checkpoint_id,
            filesystem_checkpoint_id,
            [(str(copy_id), copy_process, copy_filesystem) for copy_id, copy_process, copy_filesystem in copy_plan],
        )

        chain_sharing_active = self._chain_sharing_active(manifests, checkpoint_id)
        if chain_sharing_active:
            pin_chain = getattr(self.storage, "pin_chain", None)
            if callable(pin_chain) and pin_chain(source.sandbox_id, checkpoint_id):
                self._fork_chain_pins[target.sandbox_id] = (
                    source.sandbox_id,
                    checkpoint_id,
                )
            else:
                # Pin not available (e.g. KeepAll manager) or leaf manifest
                # vanished between resolve and pin. Without a pin the source's
                # ancestor blobs could be pruned mid-restore by the
                # LatestOnly manager; fall back to copy mode for safety.
                logger.warning(
                    "Chain-sharing pin failed; falling back to copy mode source=%s target=%s checkpoint=%s",
                    source.sandbox_id,
                    target.sandbox_id,
                    checkpoint_id,
                )
                chain_sharing_active = False
        chain_sharing_links = 0
        chain_sharing_bytes_saved = 0
        for copy_id, copy_process, copy_filesystem in copy_plan:
            source_manifest = manifests[copy_id]
            is_leaf = copy_id == checkpoint_id
            link_this_entry = chain_sharing_active and copy_process and not is_leaf
            process_refs = []
            filesystem_refs = []
            if copy_process:
                if link_this_entry:
                    self.runtime.link_ancestor_pre_dump(
                        source.sandbox_id, target.sandbox_id, copy_id
                    )
                for reference in source_manifest.process_artifacts:
                    payload = self.storage.get_artifact(source.sandbox_id, copy_id, reference)
                    if link_this_entry:
                        rewritten = self._rewrite_process_artifact_linked(
                            payload,
                            target.sandbox_id,
                            copy_id,
                        )
                        chain_sharing_links += 1
                        chain_sharing_bytes_saved += int(reference.size_bytes or 0)
                    else:
                        rewritten = self._rewrite_process_artifact(
                            payload,
                            source.sandbox_id,
                            target.sandbox_id,
                            copy_id,
                            preserve_symlinks=chain_sharing_active and is_leaf,
                        )
                    process_refs.append(
                        self.storage.put_artifact(
                            target.sandbox_id,
                            copy_id,
                            ArtifactPayload(
                                kind=reference.kind,
                                name=reference.name,
                                data=rewritten,
                                metadata=dict(reference.metadata),
                            ),
                        )
                    )
            if copy_filesystem:
                for reference in source_manifest.filesystem_artifacts:
                    payload = self.storage.get_artifact(source.sandbox_id, copy_id, reference)
                    filesystem_refs.append(
                        self.storage.put_artifact(
                            target.sandbox_id,
                            copy_id,
                            ArtifactPayload(
                                kind=reference.kind,
                                name=reference.name,
                                data=self._rewrite_filesystem_artifact(
                                    payload,
                                    source.sandbox_id,
                                    target.sandbox_id,
                                    copy_id,
                                ),
                                metadata=dict(reference.metadata),
                            ),
                        )
                    )
            manifest = CheckpointManifest(
                schema_version=source_manifest.schema_version,
                checkpoint_id=source_manifest.checkpoint_id,
                sandbox_id=target.sandbox_id,
                created_at=source_manifest.created_at,
                runtime_name=source_manifest.runtime_name,
                runtime_version=source_manifest.runtime_version,
                process_artifacts=process_refs,
                filesystem_artifacts=filesystem_refs,
                metadata=dict(source_manifest.metadata),
            ).with_integrity()
            self.storage.put_manifest(manifest)

        if chain_sharing_active:
            # The leaf's `pre_dump/parent` symlink (preserved via
            # symlinks=True above) resolves relatively into the fork's
            # runtime checkpoint root: <fork>/ckpt-PARENT/pre_dump. We
            # plant a symlink at <fork>/ckpt-PARENT pointing at
            # <source>/ckpt-PARENT for every ancestor in the chain so
            # CRIU's chain walk during runc restore resolves into the
            # source's bytes without copying. Walk via storage manifests
            # rather than the copy_plan because resolve_checkpoint_copy_plan
            # short-circuits when the leaf has both process+filesystem
            # artifacts (the common case for incremental forks).
            try:
                cursor_id: CheckpointId | None = manifests[checkpoint_id].parent_checkpoint_id
                seen: set[CheckpointId] = set()
                while cursor_id is not None and cursor_id not in seen:
                    seen.add(cursor_id)
                    self.runtime.link_ancestor_pre_dump(
                        source.sandbox_id, target.sandbox_id, cursor_id
                    )
                    chain_sharing_links += 1
                    parent_manifest = manifests.get(cursor_id)
                    if parent_manifest is None:
                        break
                    chain_sharing_bytes_saved += sum(
                        int(ref.size_bytes or 0) for ref in parent_manifest.process_artifacts
                    )
                    cursor_id = parent_manifest.parent_checkpoint_id
            except Exception:
                logger.exception(
                    "Failed to plant ancestor symlinks for chain sharing source=%s target=%s checkpoint=%s",
                    source.sandbox_id,
                    target.sandbox_id,
                    checkpoint_id,
                )
        if chain_sharing_active and chain_sharing_links > 0:
            logger.info(
                "Fork chain sharing applied source=%s target=%s checkpoint=%s "
                "links=%d bytes_saved=%d",
                source.sandbox_id,
                target.sandbox_id,
                checkpoint_id,
                chain_sharing_links,
                chain_sharing_bytes_saved,
            )
            self.emit_benchmark_metric(
                "benchmark.fork.chain_sharing_links",
                float(chain_sharing_links),
                target,
                checkpoint_id=checkpoint_id,
            )
            self.emit_benchmark_metric(
                "benchmark.fork.chain_sharing_bytes_saved",
                float(chain_sharing_bytes_saved),
                target,
                checkpoint_id=checkpoint_id,
            )

        description = SandboxDescription(
            sandbox_id=target.sandbox_id,
            runtime_name="runc",
            status="stopped",
            metadata={
                "sandbox_id": str(target.sandbox_id),
                "bundle_path": str(target.bundle_dir),
                "rootfs_path": str(rootfs_path),
                "zfs_dataset": target_dataset,
                "work_dir_host_path": None if work_dir_host_path is None else str(work_dir_host_path),
            },
        )
        self.runtime._items[target.sandbox_id] = description
        self.runtime._persist(description)
        inherited_checkpoint_at = manifests[checkpoint_id].created_at
        self.base_inspector.upsert_snapshot(
            SandboxSnapshot(
                sandbox_id=target.sandbox_id,
                runtime_name="runc",
                is_running=False,
                process_changed=False,
                filesystem_changed=False,
                observed_at=utc_now(),
                last_checkpoint_at=inherited_checkpoint_at,
            )
        )
        if self.system is not None:
            self.system.scheduler.mark_checkpoint_complete(
                target.sandbox_id,
                inherited_checkpoint_at,
            )
        target.agent_type = source.agent_type
        target.llm_service_type = source.llm_service_type
        target.llm_base_url = source.llm_base_url
        target.task_description = source.task_description
        target.task_config = source.task_config
        target.launch_source = source.launch_source
        target.launch_metadata = dict(source.launch_metadata)
        target.task_run = None
        target.task_future = None
        if target.task_description is not None and target.task_config is not None:
            target.task_run = self.build_task_run(
                target.agent_type,
                target,
                target.task_description,
                target.task_config,
            )
            target.task_run.prepare_sandbox()
            extra_launch_metadata = dict(target.task_run.extra_launch_metadata())
            if extra_launch_metadata:
                description = replace(
                    description,
                    metadata=_merge_runtime_launch_metadata(description.metadata, extra_launch_metadata),
                )
                self.runtime._items[target.sandbox_id] = description
                self.runtime._persist(description)
        return target

    def _replicate_source_bundle_mounts(
        self,
        *,
        source: SandboxHandle,
        target: SandboxHandle,
    ) -> None:
        """Replicate compose-derived bundle config from source onto fork.

        The fork's bundle is built by `_prepare_sandbox_handle` (the runc-direct
        path), which produces a *minimal* spec. The source sandbox went through
        compose translation, which adds extra mounts (e.g. /logs) and a richer
        `process.capabilities` set (CAP_SETUID/SETGID/DAC_OVERRIDE/etc.). When
        the fork is later restored from the source's checkpoint or runs apt-get
        in verification, those deltas matter:

        - **mounts**: CRIU's process image references mountpoint IDs from the
          source spec; a mismatch fails restore with `mnt: No mapping for
          X:(null) mountpoint`.
        - **process.capabilities**: apt-get's package-install sub-process
          drops privileges to `nobody` via setgroups/setegid, which needs
          CAP_SETGID/CAP_SETUID. The bare-bundle's three-cap default fails
          with `setgroups 65534 failed - setgroups (1: Operation not permitted)`.
        - **process.user**: copied for forward compatibility with non-root
          containers.

        Per-sandbox host paths inside mount sources (`.../sandbox-X/logs/`)
        are rewritten to the fork's equivalents so a checkpoint+restore on
        the fork doesn't accidentally write into the source sandbox's state.
        """
        source_cfg_path = source.bundle_dir / "config.json"
        target_cfg_path = target.bundle_dir / "config.json"
        if not source_cfg_path.is_file() or not target_cfg_path.is_file():
            return
        try:
            source_cfg = json.loads(source_cfg_path.read_text(encoding="utf-8"))
            target_cfg = json.loads(target_cfg_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        changed = False
        source_mounts = source_cfg.get("mounts")
        if isinstance(source_mounts, list) and source_mounts:
            source_token = f"/{source.sandbox_id}/"
            target_token = f"/{target.sandbox_id}/"
            rewritten: list[dict[str, object]] = []
            for mount in source_mounts:
                if not isinstance(mount, dict):
                    continue
                new_mount = dict(mount)
                raw_source = new_mount.get("source")
                if isinstance(raw_source, str) and source_token in raw_source:
                    rewritten_source = raw_source.replace(source_token, target_token)
                    rewritten_path = Path(rewritten_source)
                    try:
                        rewritten_path.mkdir(parents=True, exist_ok=True)
                        new_mount["source"] = rewritten_source
                    except OSError:
                        pass
                rewritten.append(new_mount)
            target_cfg["mounts"] = rewritten
            changed = True
        source_process = source_cfg.get("process")
        if isinstance(source_process, dict):
            target_process = target_cfg.setdefault("process", {})
            if isinstance(target_process, dict):
                source_caps = source_process.get("capabilities")
                if isinstance(source_caps, dict):
                    target_process["capabilities"] = copy.deepcopy(source_caps)
                    changed = True
                source_user = source_process.get("user")
                if isinstance(source_user, dict):
                    target_process["user"] = copy.deepcopy(source_user)
                    changed = True
                # Replicate cwd / env / noNewPrivileges so agent commands
                # resolve relative paths against the same working directory the
                # source uses (compose-derived workdir, e.g. /app). After
                # promotion the harness swaps active.bundle_dir to the fork's
                # bundle, so a fork with cwd=/work would silently make every
                # subsequent `sed -i ars.R` operate on /work/ars.R instead of
                # /app/ars.R, leaving the source path frozen at its pre-fork
                # state. Same pattern for env (PATH/TEST_DIR etc) and
                # noNewPrivileges (must stay False to allow capability use).
                source_cwd = source_process.get("cwd")
                if isinstance(source_cwd, str) and source_cwd:
                    target_process["cwd"] = source_cwd
                    changed = True
                source_env = source_process.get("env")
                if isinstance(source_env, list):
                    target_process["env"] = list(source_env)
                    changed = True
                if "noNewPrivileges" in source_process:
                    target_process["noNewPrivileges"] = bool(
                        source_process["noNewPrivileges"]
                    )
                    changed = True
        if not changed:
            return
        target_cfg_path.write_text(json.dumps(target_cfg, indent=2), encoding="utf-8")
        logger.debug(
            "Replicated source bundle layout to fork source=%s target=%s",
            source.sandbox_id,
            target.sandbox_id,
        )

    def _clone_host_work_dir(self, source_sandbox_id: SandboxId, target_sandbox_id: SandboxId) -> None:
        source_work_dir = benchmark_support.resolve_work_dir_host_path(self.work_dir_host_root, str(source_sandbox_id))
        target_work_dir = benchmark_support.resolve_work_dir_host_path(self.work_dir_host_root, str(target_sandbox_id))
        if source_work_dir is None or target_work_dir is None:
            return
        if target_work_dir.exists():
            shutil.rmtree(target_work_dir, ignore_errors=True)
        if not source_work_dir.exists():
            target_work_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(
                "Prepared empty fork work dir because source work dir is missing source=%s target=%s",
                source_work_dir,
                target_work_dir,
            )
            return
        shutil.copytree(source_work_dir, target_work_dir)
        logger.debug(
            "Cloned host work dir for fork source_sandbox=%s target_sandbox=%s source=%s target=%s",
            source_sandbox_id,
            target_sandbox_id,
            source_work_dir,
            target_work_dir,
        )

    def clone_tree_search_checkpoint_to_fork(
        self,
        source: SandboxHandle,
        checkpoint_id: CheckpointId,
        fork_name: str,
    ) -> SandboxHandle:
        target = self.clone_checkpoint_to_fork(source, checkpoint_id, fork_name)
        network_lease = self.network_manager.lease_for(target.sandbox_id)
        work_dir_host_path = benchmark_support.resolve_work_dir_host_path(self.work_dir_host_root, str(target.sandbox_id))
        # Compose-backed agents (mini_swe, claude_code, terminus) materialize
        # their rootfs from the task's docker-compose file, not from a per-agent
        # Dockerfile, so there's no sandbox image to build. write_bundle_config
        # already tolerates None for image_defaults / image_rootfs_dir; only
        # call ensure_sandbox_image for runc-direct agents.
        if target.agent_type is not None and target.launch_source != "compose":
            sandbox_image = self.ensure_sandbox_image(target.agent_type)
        else:
            sandbox_image = None
        sandbox_bundle.write_bundle_config(
            bundle_dir=target.bundle_dir,
            llm_base_url=target.llm_base_url or "",
            provider=self.provider,
            sandbox_name=str(target.sandbox_id),
            status_port=source.status_port,
            cgroup_path=f"crab-bench/{self.pool_name}/{target.sandbox_id}",
            work_dir_host_path=work_dir_host_path,
            network_namespace_path=None if network_lease is None else network_lease.namespace_path,
            image_defaults=None if sandbox_image is None else sandbox_image.image_defaults,
            image_rootfs_dir=None if sandbox_image is None else sandbox_image.exported_rootfs,
            resource_limits=self.sandbox_resource_limits,
        )
        if target.task_run is not None and target.agent_type != "simulated":
            target.task_run.configure_bundle()
        self._preserve_tree_search_restore_bundle_compatibility(source=source, target=target)
        target.status_host = "127.0.0.1" if network_lease is None else network_lease.guest_ip
        target.status_port = source.status_port
        if network_lease is not None:
            self.network_manager.register_guest_ip(network_lease.guest_ip, target.sandbox_id)
        logger.info(
            "Prepared tree-search fork sandbox=%s status_host=%s status_port=%d source=%s checkpoint=%s",
            target.sandbox_id,
            target.status_host,
            target.status_port,
            source.sandbox_id,
            checkpoint_id,
        )
        return target

    def _preserve_tree_search_restore_bundle_compatibility(
        self,
        *,
        source: SandboxHandle,
        target: SandboxHandle,
    ) -> None:
        source_config_path = source.bundle_dir / "config.json"
        target_config_path = target.bundle_dir / "config.json"
        if not source_config_path.is_file() or not target_config_path.is_file():
            return

        source_cfg = json.loads(source_config_path.read_text(encoding="utf-8"))
        target_cfg = json.loads(target_config_path.read_text(encoding="utf-8"))

        source_mounts = source_cfg.get("mounts", [])
        target_mounts = target_cfg.get("mounts", [])
        if not isinstance(source_mounts, list) or not isinstance(target_mounts, list):
            return

        existing_destinations = {
            str(mount.get("destination"))
            for mount in target_mounts
            if isinstance(mount, dict) and mount.get("destination")
        }
        preserved_destinations: list[str] = []
        for mount in source_mounts:
            if not isinstance(mount, dict):
                continue
            destination = mount.get("destination")
            if not isinstance(destination, str) or not destination or destination in existing_destinations:
                continue
            target_mounts.append(dict(mount))
            existing_destinations.add(destination)
            preserved_destinations.append(destination)
        if preserved_destinations:
            logger.info(
                "Preserved source-only mounts for tree-search restore compatibility "
                "source=%s target=%s mounts=%s",
                source.sandbox_id,
                target.sandbox_id,
                preserved_destinations,
            )
        target_cfg["mounts"] = target_mounts

        source_process = source_cfg.get("process", {})
        target_process = target_cfg.get("process", {})
        if isinstance(source_process, dict) and isinstance(target_process, dict):
            source_cwd = source_process.get("cwd")
            if isinstance(source_cwd, str) and source_cwd:
                target_process["cwd"] = source_cwd
            target_cfg["process"] = target_process

        target_config_path.write_text(json.dumps(target_cfg, indent=2), encoding="utf-8")

    def _sandbox_llm_service_config(self, sandbox: SandboxHandle) -> dict[str, object] | None:
        benchmark_metadata = sandbox.launch_metadata.get("benchmark")
        if not isinstance(benchmark_metadata, dict):
            return None
        raw_config = benchmark_metadata.get("llm_service_config")
        if not isinstance(raw_config, dict):
            return None
        return dict(raw_config)

    def sandbox_bundle_config(self, sandbox: SandboxHandle) -> dict[str, object]:
        config_path = sandbox.bundle_dir / "config.json"
        return json.loads(config_path.read_text(encoding="utf-8"))

    def _bundle_spec_writer(self):
        manager = self._active_runtime
        if manager is None or not hasattr(manager, "write_bundle_spec"):
            raise RuntimeError("runtime with bundle spec support is required")
        return manager.write_bundle_spec

    def _delete_runtime(self, sandbox_id: SandboxId) -> None:
        if self.runtime is None or not hasattr(self.runtime, "delete_runtime"):
            return
        self.runtime.delete_runtime(sandbox_id, force=True, ignore_missing=True)

    def _destroy_filesystem_dataset(self, sandbox_id: SandboxId) -> None:
        if self.runtime is None or not hasattr(self.runtime, "destroy_filesystem_dataset"):
            return
        self.runtime.destroy_filesystem_dataset(sandbox_id)

    def _unregister_host_inspector(self, sandbox_id: SandboxId) -> None:
        client = self.host_inspector_client
        if client is None:
            return
        try:
            client.unregister_sandbox(sandbox_id)
        except Exception:
            logger.exception("Failed to unregister sandbox %s from host inspector", sandbox_id)

    def _promote_filesystem_dataset(self, sandbox_id: SandboxId) -> None:
        if self.runtime is None or not hasattr(self.runtime, "promote_filesystem_dataset"):
            return
        self.runtime.promote_filesystem_dataset(sandbox_id)

    def _dataset_name_for(self, sandbox_id: SandboxId) -> str:
        if self.runtime is not None and hasattr(self.runtime, "dataset_name_for"):
            return self.runtime.dataset_name_for(sandbox_id)
        return f"{self.pool_name}/crab/{sandbox_id}"

    def _clone_filesystem_snapshot(
        self,
        source_sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
        target_sandbox_id: SandboxId,
        *,
        target_rootfs_path: Path,
    ) -> str:
        if self.runtime is None or not hasattr(self.runtime, "clone_filesystem_snapshot"):
            return f"{self.pool_name}/crab/{target_sandbox_id}"
        return self.runtime.clone_filesystem_snapshot(
            source_sandbox_id,
            checkpoint_id,
            target_sandbox_id,
            target_rootfs_path=target_rootfs_path,
        )

    def sandbox_process_cwd(self, sandbox: SandboxHandle) -> str:
        process = self.sandbox_bundle_config(sandbox).get("process", {})
        if not isinstance(process, dict):
            return "/app"
        cwd = process.get("cwd")
        return str(cwd) if isinstance(cwd, str) and cwd else "/app"

    def _sandbox_process_env(self, sandbox: SandboxHandle) -> list[str]:
        process = self.sandbox_bundle_config(sandbox).get("process", {})
        if not isinstance(process, dict):
            return []
        env = process.get("env", [])
        if not isinstance(env, list):
            return []
        return [str(item) for item in env]

    def _sandbox_process_user(self, sandbox: SandboxHandle) -> str | None:
        process = self.sandbox_bundle_config(sandbox).get("process", {})
        if not isinstance(process, dict):
            return None
        user = process.get("user")
        if not isinstance(user, dict):
            return None
        uid = user.get("uid")
        gid = user.get("gid")
        if uid is None:
            return None
        return f"{uid}:{gid}" if gid is not None else str(uid)

    def exec_in_sandbox(
        self,
        sandbox: SandboxHandle,
        command: list[str],
        *,
        cwd: str | None = None,
        env: dict[str, object] | None = None,
        timeout_s: float | None = None,
        capture_output: bool = True,
    ) -> SandboxExecResult:
        assert self.runtime is not None
        merged_env = sandbox_bundle.merge_environment_defaults(
            self._sandbox_process_env(sandbox),
            [] if env is None else [f"{key}={value}" for key, value in env.items()],
        )
        resolved_cwd = cwd or self.sandbox_process_cwd(sandbox)
        user = self._sandbox_process_user(sandbox)
        env_map: dict[str, object] = {}
        for item in merged_env:
            key, _, value = item.partition("=")
            env_map[key] = value
        return self.runtime.exec(
            sandbox.sandbox_id,
            command,
            cwd=resolved_cwd,
            env=env_map,
            user=user,
            timeout_s=timeout_s,
            capture_output=capture_output,
        )

    def wait_for_task_completion(self, sandbox: SandboxHandle, *, timeout_s: float | None = None) -> None:
        if sandbox.task_future is None:
            future = None
        else:
            future = sandbox.task_future
        deadline = None if timeout_s is None else time.monotonic() + max(0.0, float(timeout_s))
        trace_response_count = benchmark_core.trace_response_count_for_sandbox(sandbox)
        replay_completion_wait_started_at: float | None = None
        replay_stop_requested = False
        while True:
            if benchmark_support.is_replay_llm_service_type(sandbox.llm_service_type):
                status = benchmark_core.poll_sandbox_status(sandbox)
                if benchmark_core.replay_status_is_complete(
                    status,
                    trace_response_count=trace_response_count,
                ):
                    if self.system is not None and self.system.has_pending_interceptor_signal(sandbox.sandbox_id):
                        replay_completion_wait_started_at = None
                        logger.debug(
                            "Delaying replay task completion short-circuit until pending interceptor work clears "
                            "sandbox=%s replay_final_trace_cursor=%d trace_response_count=%d",
                            sandbox.sandbox_id,
                            benchmark_core.replay_trace_cursor(status),
                            trace_response_count,
                        )
                        time.sleep(0.05)
                        continue
                    if future is not None and not future.done():
                        now = time.monotonic()
                        if replay_completion_wait_started_at is None:
                            replay_completion_wait_started_at = now
                        wait_s = max(0.0, now - replay_completion_wait_started_at)
                        if (
                            not replay_stop_requested
                            and wait_s >= self.REPLAY_COMPLETION_TASK_FUTURE_GRACE_SECONDS
                            and sandbox.task_run is not None
                        ):
                            logger.warning(
                                "Replay is complete but task future is still running; requesting stop "
                                "sandbox=%s replay_final_trace_cursor=%d trace_response_count=%d wait_s=%.3f",
                                sandbox.sandbox_id,
                                benchmark_core.replay_trace_cursor(status),
                                trace_response_count,
                                wait_s,
                            )
                            sandbox.task_run.request_stop()
                            replay_stop_requested = True
                        logger.debug(
                            "Delaying replay task completion short-circuit until task future completes "
                            "sandbox=%s replay_final_trace_cursor=%d trace_response_count=%d wait_s=%.3f",
                            sandbox.sandbox_id,
                            benchmark_core.replay_trace_cursor(status),
                            trace_response_count,
                            wait_s,
                        )
                        time.sleep(0.05)
                        continue
                    replay_completion_wait_started_at = None
                    if self.runtime is not None:
                        try:
                            self.runtime.resume(sandbox.sandbox_id)
                        except Exception:
                            logger.debug(
                                "Best-effort runtime resume before replay completion short-circuit failed "
                                "sandbox=%s",
                                sandbox.sandbox_id,
                                exc_info=True,
                            )
                    logger.info(
                        "Benchmark task completion short-circuit sandbox=%s replay_final_trace_cursor=%d "
                        "trace_response_count=%d",
                        sandbox.sandbox_id,
                        benchmark_core.replay_trace_cursor(status),
                        trace_response_count,
                    )
                    return
            if future is None:
                return
            wait_timeout = 1.0
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError
                wait_timeout = min(wait_timeout, remaining)
            try:
                future.result(timeout=wait_timeout)
                return
            except FutureTimeoutError:
                continue
            except Exception:
                if benchmark_support.is_replay_llm_service_type(sandbox.llm_service_type):
                    status = benchmark_core.poll_sandbox_status(sandbox)
                    if benchmark_core.replay_status_is_complete(
                        status,
                        trace_response_count=trace_response_count,
                    ):
                        if self.system is not None and self.system.has_pending_interceptor_signal(sandbox.sandbox_id):
                            logger.debug(
                                "Ignoring replay completion exception but waiting for pending interceptor work "
                                "to clear sandbox=%s replay_final_trace_cursor=%d trace_response_count=%d",
                                sandbox.sandbox_id,
                                benchmark_core.replay_trace_cursor(status),
                                trace_response_count,
                            )
                            time.sleep(0.05)
                            continue
                        if self.runtime is not None:
                            try:
                                self.runtime.resume(sandbox.sandbox_id)
                            except Exception:
                                logger.debug(
                                    "Best-effort runtime resume before ignoring replay completion exception "
                                    "failed sandbox=%s",
                                    sandbox.sandbox_id,
                                    exc_info=True,
                                )
                        logger.info(
                            "Ignoring replay task completion exception after replay completed sandbox=%s "
                            "replay_final_trace_cursor=%d trace_response_count=%d",
                            sandbox.sandbox_id,
                            benchmark_core.replay_trace_cursor(status),
                            trace_response_count,
                        )
                        return
                raise

    def _ensure_verification_uv_timeout_seconds(self, timeout_s: float | None) -> float:
        if timeout_s is None:
            return 30.0
        try:
            return max(30.0, float(timeout_s))
        except (TypeError, ValueError):
            return 30.0

    def _verification_uv_failure_is_transient(self, stderr: str) -> bool:
        return any(fragment in stderr for fragment in _VERIFICATION_UV_TRANSIENT_ERROR_FRAGMENTS)

    def _wait_for_verification_network(self, sandbox: SandboxHandle, *, timeout_s: float | None = None) -> None:
        if not self._agent_requires_benchmark_network(sandbox.agent_type):
            return
        total_timeout_s = min(
            _VERIFICATION_NETWORK_READY_TIMEOUT_SECONDS,
            self._ensure_verification_uv_timeout_seconds(timeout_s),
        )
        deadline = time.monotonic() + total_timeout_s
        last_stderr = ""
        attempt = 0
        refreshed_resolv_conf = False
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            attempt += 1
            result = self.exec_in_sandbox(
                sandbox,
                ["/bin/bash", "-lc", self._verification_network_probe_script()],
                cwd=self.sandbox_process_cwd(sandbox),
                timeout_s=max(1.0, min(5.0, remaining)),
            )
            if result.returncode == 0:
                if attempt > 1:
                    logger.info(
                        "Verification network became ready sandbox=%s attempts=%d",
                        sandbox.sandbox_id,
                        attempt,
                    )
                return
            last_stderr = result.stderr.rstrip()
            logger.debug(
                "Waiting for verification network readiness sandbox=%s attempt=%d stderr=%s",
                sandbox.sandbox_id,
                attempt,
                last_stderr,
            )
            if (
                not refreshed_resolv_conf
                and "Temporary failure" in last_stderr
                and self._refresh_sandbox_resolv_conf(sandbox)
            ):
                refreshed_resolv_conf = True
            time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
        logger.warning(
            "Verification network readiness probe timed out sandbox=%s stderr=%s",
            sandbox.sandbox_id,
            last_stderr,
        )

    def _refresh_sandbox_resolv_conf(self, sandbox: SandboxHandle) -> bool:
        host_resolv_conf = _host_resolv_conf_path()
        if host_resolv_conf is None:
            return False
        contents = host_resolv_conf.read_text(encoding="utf-8")
        result = self.exec_in_sandbox(
            sandbox,
            [
                "/bin/bash",
                "-lc",
                (
                    "python3 - <<'PY'\n"
                    "from pathlib import Path\n"
                    f"Path('/etc/resolv.conf').write_text({contents!r}, encoding='utf-8')\n"
                    "PY"
                ),
            ],
            cwd=self.sandbox_process_cwd(sandbox),
            timeout_s=10.0,
        )
        if result.returncode != 0:
            logger.warning(
                "Failed to refresh sandbox resolv.conf sandbox=%s source=%s exit_code=%s stderr=%s",
                sandbox.sandbox_id,
                host_resolv_conf,
                result.returncode,
                result.stderr.rstrip(),
            )
            return False
        logger.info(
            "Refreshed sandbox resolv.conf before verification sandbox=%s source=%s",
            sandbox.sandbox_id,
            host_resolv_conf,
        )
        return True

    def _ensure_verification_uv(self, sandbox: SandboxHandle, *, timeout_s: float | None = None) -> None:
        total_timeout_s = self._ensure_verification_uv_timeout_seconds(timeout_s)
        deadline = time.monotonic() + total_timeout_s
        last_result = None
        for attempt in range(1, 4):
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            result = self.exec_in_sandbox(
                sandbox,
                ["/bin/bash", "-lc", self._verification_uv_bootstrap_script()],
                cwd=self.sandbox_process_cwd(sandbox),
                timeout_s=max(1.0, remaining),
            )
            if result.returncode == 0:
                return
            last_result = result
            stderr = result.stderr.rstrip()
            if attempt >= 3 or not self._verification_uv_failure_is_transient(stderr):
                break
            backoff_s = min(float(attempt), max(0.0, deadline - time.monotonic()))
            logger.warning(
                "Retrying transient verification uv bootstrap failure sandbox=%s attempt=%d exit_code=%s stderr=%s",
                sandbox.sandbox_id,
                attempt,
                result.returncode,
                stderr,
            )
            if backoff_s > 0.0:
                time.sleep(backoff_s)
        if last_result is None:
            raise RuntimeError(
                f"failed to prepare verification uv shim sandbox={sandbox.sandbox_id} before timeout elapsed"
            )
        raise RuntimeError(
            "failed to prepare verification uv shim "
            f"sandbox={sandbox.sandbox_id} exit_code={last_result.returncode} "
            f"stderr={last_result.stderr.rstrip()}"
        )

    @staticmethod
    def _legacy_cython_pin_script() -> str:
        # swebench sklearn testbeds for commits ~0.20–0.22 ship a Python 3.6
        # conda env. The replay agent's trajectory for tasks like
        # scikit-learn-14710 runs `pip install cython` without a version
        # pin, which upgrades the testbed env to Cython 3.x. A subsequent
        # `pip install -e .` then regenerates .pyx→.c with Cython 3.x,
        # producing sources that start with `#error Cython requires
        # Python 3.8+.` swebench's verification `pip install -e .` then
        # fails to compile against the testbed's Python 3.6. We fix this
        # surgically before verification by (1) forcing Cython<3 into the
        # testbed env and (2) deleting any .c files that carry the
        # Cython-3.x Python-3.8 marker so setup.py's build_src
        # regenerates them via the pinned Cython.
        return r"""
set -uo pipefail
_run() {
  local TESTBED_PY=/opt/miniconda3/envs/testbed/bin/python
  local TESTBED_PIP=/opt/miniconda3/envs/testbed/bin/pip
  if [ ! -x "$TESTBED_PY" ]; then echo "SKIP: no testbed python at $TESTBED_PY"; return 0; fi
  local py
  py=$("$TESTBED_PY" -c 'import sys; print(sys.version_info[0]*100 + sys.version_info[1])' 2>&1) || { echo "SKIP: python version probe failed: $py"; return 0; }
  echo "testbed_python_version_code=$py"
  if [ "$py" -ge 308 ]; then echo "SKIP: python >= 3.8"; return 0; fi
  local cur
  cur=$("$TESTBED_PY" -c 'import Cython; print(Cython.__version__)' 2>&1) || cur=absent
  echo "current_cython=$cur"
  local need_pin=0
  case "$cur" in
    0.*) echo "cython already 0.x, skip pip install" ;;
    *) need_pin=1 ;;
  esac
  if [ $need_pin -eq 1 ]; then
    echo "INSTALL: pip install cython<3"
    local pip_out rc
    pip_out=$("$TESTBED_PIP" install --no-input --disable-pip-version-check 'cython<3' 2>&1)
    rc=$?
    echo "pip_rc=$rc"
    echo "$pip_out" | tail -5
    local after
    after=$("$TESTBED_PY" -c 'import Cython; print(Cython.__version__)' 2>&1) || after=?
    echo "cython_after_install=$after"
  fi
  # Scrub poisoned Cython outputs left by the agent's Cython 3.x build.
  # These begin with `#error Cython requires Python 3.8+.` and appear in
  # both .c and .cpp form depending on whether the .pyx targets C or
  # C++. Deleting them forces numpy.distutils.build_src to re-cythonize
  # from .pyx using the pinned testbed Cython.
  local src=/testbed
  if [ -d "$src" ]; then
    local scrubbed=0 gen_file
    while IFS= read -r -d '' gen_file; do
      if head -n 80 "$gen_file" 2>/dev/null | grep -q "Cython requires Python 3.8+"; then
        rm -f "$gen_file"
        scrubbed=$((scrubbed + 1))
      fi
    done < <(find "$src" \( -name '*.c' -o -name '*.cpp' -o -name '*.cxx' \) -print0 2>/dev/null)
    echo "scrubbed_poisoned_sources=$scrubbed"
  fi
}
_run
""".strip()

    def _ensure_legacy_cython_pin(
        self, sandbox: SandboxHandle, *, timeout_s: float | None = None
    ) -> None:
        budget = 180.0 if timeout_s is None else max(30.0, min(float(timeout_s), 240.0))
        result = self.exec_in_sandbox(
            sandbox,
            ["/bin/bash", "-lc", self._legacy_cython_pin_script()],
            cwd=self.sandbox_process_cwd(sandbox),
            timeout_s=budget,
        )
        logger.info(
            "legacy cython pin prelude sandbox=%s exit_code=%s log=%s",
            sandbox.sandbox_id,
            result.returncode,
            result.stdout.rstrip().replace("\n", " | "),
        )

    @staticmethod
    def _psf_requests_local_httpbin_wrapper(inner_command: str) -> str:
        # psf/requests tests read HTTPBIN from
        # `os.environ.get('HTTPBIN_URL', 'http://httpbin.org/')`, defaulting
        # to public httpbin.org. At high benchmark concurrency that service
        # rate-limits and returns 502, flaking PASS_TO_PASS tests like
        # test_unicode_multipart_post and test_mixed_case_scheme_acceptable.
        # Swap in a per-sandbox loopback httpbin. test_mixed_case_scheme_acceptable
        # probes both http:// and HTTPS:// schemes against the same netloc, so
        # we bind one process listening on 127.0.0.1:80 (HTTP) and 127.0.0.1:443
        # (HTTPS via self-signed cert) — matching how public httpbin.org exposes
        # both schemes on the same hostname. The testbed env's requests CA
        # bundle is extended with our self-signed cert so HTTPS verifies cleanly.
        # The sandbox has CAP_NET_BIND_SERVICE so privileged ports are bindable
        # as root. Install + cert are no-ops after the first run because the
        # base env and /tmp state persist inside the rootfs.
        return rf"""
set -uo pipefail
BASE_PY=/opt/miniconda3/bin/python
BASE_PIP=/opt/miniconda3/bin/pip
HTTPBIN_PID=""
HTTPBIN_ACTIVE=0
if [ ! -x "$BASE_PY" ]; then
  echo "psf_requests_httpbin: SKIP no base python"
else
  if ! "$BASE_PY" -c 'import httpbin, flask, cryptography' >/dev/null 2>&1; then
    echo "psf_requests_httpbin: installing httpbin+flask+cryptography"
    "$BASE_PIP" install --no-input --disable-pip-version-check --quiet \
      'httpbin' 'Flask<3' 'Werkzeug<3' 'brotli' 'cryptography' 'pyOpenSSL' 2>&1 | tail -5 || true
  fi
  # v2 cert path: older urllib3 vendored in psf/requests only checks DNS SANs
  # (ignores IP SANs) and never falls back to CN when any DNS SAN is present.
  # To satisfy both old and new stacks we omit DNS SAN entries entirely so old
  # urllib3 falls back to CN="127.0.0.1" while modern stacks match IP SAN.
  CERT=/tmp/crab-httpbin-v2.crt
  KEY=/tmp/crab-httpbin-v2.key
  if [ ! -s "$CERT" ] || [ ! -s "$KEY" ]; then
    "$BASE_PY" - <<'PYCERT' >/tmp/crab-httpbin-cert.log 2>&1 || true
import datetime, ipaddress
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
now = datetime.datetime.utcnow()
subj = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "127.0.0.1")])
cert = (x509.CertificateBuilder()
    .subject_name(subj).issuer_name(subj)
    .public_key(key.public_key())
    .serial_number(x509.random_serial_number())
    .not_valid_before(now - datetime.timedelta(days=1))
    .not_valid_after(now + datetime.timedelta(days=3650))
    .add_extension(x509.SubjectAlternativeName([
        x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
    ]), critical=False)
    .sign(key, hashes.SHA256()))
open("/tmp/crab-httpbin-v2.key","wb").write(key.private_bytes(
    serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL,
    serialization.NoEncryption()))
open("/tmp/crab-httpbin-v2.crt","wb").write(cert.public_bytes(serialization.Encoding.PEM))
PYCERT
  fi
  if [ -s "$CERT" ] && [ -s "$KEY" ]; then
    setsid "$BASE_PY" - >/tmp/crab-httpbin.log 2>&1 <<PYRUN &
import threading, ssl
from httpbin import app
from werkzeug.serving import make_server
ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
ctx.load_cert_chain("$CERT", "$KEY")
http_srv = make_server("127.0.0.1", 80, app, threaded=True)
https_srv = make_server("127.0.0.1", 443, app, threaded=True, ssl_context=ctx)
t1 = threading.Thread(target=http_srv.serve_forever, daemon=True)
t2 = threading.Thread(target=https_srv.serve_forever, daemon=True)
t1.start(); t2.start()
t1.join(); t2.join()
PYRUN
    HTTPBIN_PID=$!
    up_http=0; up_https=0
    for i in $(seq 1 150); do
      if [ "$up_http" -eq 0 ] && curl -s -f -o /dev/null "http://127.0.0.1/get"; then up_http=1; fi
      if [ "$up_https" -eq 0 ] && curl -s -f -k -o /dev/null "https://127.0.0.1/get"; then up_https=1; fi
      if [ "$up_http" -eq 1 ] && [ "$up_https" -eq 1 ]; then break; fi
      sleep 0.1
    done
    if [ "$up_http" -eq 1 ] && [ "$up_https" -eq 1 ]; then
      echo "psf_requests_httpbin: ready http+https pid=$HTTPBIN_PID"
      export HTTPBIN_URL="http://127.0.0.1/"
      TESTBED_PY=/opt/miniconda3/envs/testbed/bin/python
      if [ -x "$TESTBED_PY" ]; then
        CA_BUNDLE=$("$TESTBED_PY" -c 'import requests; print(requests.certs.where())' 2>/dev/null || echo "")
        if [ -n "$CA_BUNDLE" ] && [ -f "$CA_BUNDLE" ]; then
          if ! grep -q "crab-httpbin" "$CA_BUNDLE" 2>/dev/null; then
            printf '\n# crab-httpbin self-signed\n' >> "$CA_BUNDLE"
            cat "$CERT" >> "$CA_BUNDLE"
          fi
          export REQUESTS_CA_BUNDLE="$CA_BUNDLE"
          export SSL_CERT_FILE="$CA_BUNDLE"
        fi
      fi
      HTTPBIN_ACTIVE=1
      trap 'kill -9 $HTTPBIN_PID 2>/dev/null || true' EXIT
    else
      echo "psf_requests_httpbin: FAIL http=$up_http https=$up_https pid=$HTTPBIN_PID"
      tail -40 /tmp/crab-httpbin.log 2>/dev/null || true
      kill -9 $HTTPBIN_PID 2>/dev/null || true
    fi
  else
    echo "psf_requests_httpbin: FAIL to generate cert"
    cat /tmp/crab-httpbin-cert.log 2>/dev/null || true
  fi
fi
set +e
{inner_command}
rc=$?
exit $rc
""".strip()

    def verify_task_accuracy(
        self,
        sandbox: SandboxHandle,
        *,
        timeout_s: float | None = None,
    ) -> dict[str, object]:
        swebench_instance_id = ""
        if sandbox.task_config is not None:
            raw_instance_id = sandbox.task_config.options.get("swebench_instance_id")
            if isinstance(raw_instance_id, str):
                swebench_instance_id = raw_instance_id
        command_started = time.perf_counter()
        verify_operation = None if self.telemetry is None else start_operation(
            self.telemetry,
            "benchmark.task.verify",
            self.benchmark_telemetry_attributes(sandbox),
        )
        self._wait_for_verification_network(sandbox, timeout_s=timeout_s)
        self._ensure_verification_uv(sandbox, timeout_s=timeout_s)
        if swebench_instance_id.startswith("scikit-learn__"):
            self._ensure_legacy_cython_pin(sandbox, timeout_s=timeout_s)
        verification_command = "bash /tests/run-tests.sh"
        if swebench_instance_id:
            verification_command = "bash /tests/run-tests.sh 2>&1"
        if swebench_instance_id.startswith("psf__requests-"):
            verification_command = self._psf_requests_local_httpbin_wrapper(verification_command)
        thread_cap = "4"
        result = self.exec_in_sandbox(
            sandbox,
            ["/bin/bash", "-lc", verification_command],
            cwd=self.sandbox_process_cwd(sandbox),
            env={
                "TEST_DIR": "/tests",
                "PATH": (
                    "/root/.local/crab-verification/bin:/root/.local/bin:"
                    "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
                ),
                "OMP_NUM_THREADS": thread_cap,
                "OPENBLAS_NUM_THREADS": thread_cap,
                "MKL_NUM_THREADS": thread_cap,
                "NUMEXPR_NUM_THREADS": thread_cap,
                "BLIS_NUM_THREADS": thread_cap,
                "VECLIB_MAXIMUM_THREADS": thread_cap,
                # joblib/loky read cgroup v2 cpu.max to size the Pool used by
                # sklearn's cythonize; bundle.py strips the cgroup namespace so
                # /sys/fs/cgroup/cpu.max inside the container shows the host
                # root value "max", giving joblib os.cpu_count() (64). Pool(64)
                # then OOMs under our 8 GB memory cgroup and deadlocks in
                # _handle_tasks. LOKY_MAX_CPU_COUNT is read verbatim by joblib.
                "LOKY_MAX_CPU_COUNT": thread_cap,
            },
            timeout_s=timeout_s,
        )
        verification_ms = (time.perf_counter() - command_started) * 1000.0
        stdout = result.stdout.rstrip()
        stderr = result.stderr.rstrip()
        verification_status = "passed" if result.returncode == 0 else "failed"
        verification_details: dict[str, object] = {}
        if swebench_instance_id:
            instance = swebench_support.load_verified_dataset_row(swebench_instance_id)
            grading = swebench_support.grade_verification_log(
                instance=instance,
                log_text=result.stdout,
            )
            verification_status = "passed" if bool(grading["resolved"]) else "failed"
            verification_details = {
                "verification_swebench_resolved": bool(grading["resolved"]),
                "verification_swebench_patch_applied": bool(grading["patch_successfully_applied"]),
                "verification_swebench_report": json.dumps(grading["tests_status"], sort_keys=True),
            }
        logger.info(
            "Completed run-tests.sh sandbox=%s exit_code=%s command=%s",
            sandbox.sandbox_id,
            result.returncode,
            " ".join(shlex.quote(part) for part in result.args),
        )
        if stdout:
            logger.info("run-tests stdout sandbox=%s\n%s", sandbox.sandbox_id, stdout)
        if stderr:
            logger.warning("run-tests stderr sandbox=%s\n%s", sandbox.sandbox_id, stderr)
        if verify_operation is not None:
            verify_operation.finish(status="succeeded" if verification_status == "passed" else "failed")
        return {
            "verification_status": verification_status,
            "verification_exit_code": result.returncode,
            "verification_ms": verification_ms,
            "verification_stdout": result.stdout,
            "verification_stderr": result.stderr,
            "verification_command": " ".join(shlex.quote(part) for part in result.args),
            **verification_details,
        }

    def _prepare_sandbox_handle(
        self,
        sandbox_name: str,
        *,
        interceptor_host: str,
        network_lease: sandbox_network.BenchmarkNetworkLease | None = None,
        agent_type: str = "simulated",
        llm_service_type: str = "simulated",
        llm_service_config: dict[str, object] | None = None,
        status_port: int | None = None,
        status_host: str | None = None,
        image_defaults: sandbox_image.ImageRuntimeDefaults | None = None,
        image_rootfs_dir: Path | None = None,
    ) -> tuple[SandboxHandle, Path | None]:
        assert self.interceptor is not None
        llm_control_base_url = self._llm_control_base_url()
        if not llm_control_base_url:
            raise RuntimeError("llm router is not initialized")
        work_dir_host_path = benchmark_support.resolve_work_dir_host_path(self.work_dir_host_root, sandbox_name)
        prepared = sandbox_launcher.prepare_bundle_launch(
            bundle_root=self._effective_runtime_bundle_root(),
            sandbox_name=sandbox_name,
            provider=self.provider,
            pool_name=self.pool_name,
            interceptor_host=interceptor_host,
            interceptor_port=self.interceptor.port,
            status_host=status_host if status_host is not None else ("127.0.0.1" if network_lease is None else network_lease.guest_ip),
            status_port=status_port,
            work_dir_host_path=work_dir_host_path,
            network_namespace_path=None if network_lease is None else network_lease.namespace_path,
            image_defaults=image_defaults,
            image_rootfs_dir=image_rootfs_dir,
            bundle_spec_writer=self._bundle_spec_writer(),
            resource_limits=self.sandbox_resource_limits,
        )
        handle = SandboxHandle(
            sandbox_id=SandboxId(sandbox_name),
            bundle_dir=prepared.bundle_dir,
            status_port=prepared.status_port,
            last_status={},
            status_host=prepared.status_host,
            agent_type=agent_type,
            llm_service_type=llm_service_type,
            llm_base_url=prepared.llm_base_url,
            llm_control_base_url=llm_control_base_url,
        )
        self.sandboxes.append(handle)
        self._sandbox_by_id[handle.sandbox_id] = handle
        self._register_llm_service(
            handle.sandbox_id,
            llm_service_type=llm_service_type,
            llm_service_config=llm_service_config,
        )
        return handle, prepared.work_dir_host_path

    def _llm_service_checkpoint_metadata(self, sandbox_id: SandboxId) -> dict[str, object]:
        sandbox = self._sandbox_by_id.get(sandbox_id)
        if sandbox is not None and sandbox.task_run is not None:
            try:
                status = sandbox.task_run.poll_status()
            except Exception:
                logger.debug("Failed to poll task status for llm checkpoint metadata sandbox=%s", sandbox_id, exc_info=True)
            else:
                trace_cursor = self._benchmark_trace_cursor_from_status(status)
                if trace_cursor is not None:
                    return {"benchmark_trace_cursor": trace_cursor}
        try:
            snapshot = self._snapshot_llm_services(sandbox_id)
        except Exception:
            logger.debug("Failed to capture llm service checkpoint metadata for sandbox=%s", sandbox_id, exc_info=True)
            return {}
        if snapshot is None:
            return {}
        sandbox_snapshot = snapshot if isinstance(snapshot, dict) and "state" in snapshot else snapshot.get(str(sandbox_id))
        if not isinstance(sandbox_snapshot, dict):
            return {}
        if sandbox_snapshot.get("llm_service_type") not in {
            "iflow_trace_replay",
            "mini_swe_trace_replay",
            "mini_swe_spec_trace_replay",
            "claude_code_trace_replay",
            "terminus_trace_replay",
            "terminus_spec_trace_replay",
        }:
            return {}
        state = sandbox_snapshot.get("state")
        if not isinstance(state, dict):
            return {}
        trace_cursor = self._benchmark_trace_cursor_from_snapshot_state(state)
        if trace_cursor is None:
            return {}
        return {"benchmark_trace_cursor": trace_cursor}

    def _reset_llm_service_state(self, sandbox_id: SandboxId) -> None:
        if self.llm_router_client is None and self.llm_server is None:
            return
        try:
            self._reset_llm_router_state(sandbox_id)
        except Exception:
            logger.exception("Failed to reset llm service state for sandbox=%s", sandbox_id)

    @staticmethod
    def _benchmark_trace_cursor_from_metadata(metadata: dict[str, object]) -> int | None:
        raw_value = metadata.get("benchmark_trace_cursor")
        try:
            return max(0, int(raw_value))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _benchmark_trace_cursor_from_status(status: object) -> int | None:
        if not isinstance(status, dict):
            return None
        raw_value = status.get("replay_trace_cursor")
        try:
            return max(0, int(raw_value))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _benchmark_trace_cursor_from_snapshot_state(state: object) -> int | None:
        if not isinstance(state, dict):
            return None
        # Claude Code replay exposes trace_cursor as committed main-loop replay
        # progress only. Helper/count-tokens requests are auxiliary and must not
        # affect checkpoint metadata.
        raw_value = state.get("trace_cursor", state.get("consumed_response_count"))
        try:
            return max(0, int(raw_value))
        except (TypeError, ValueError):
            return None

    def _restore_task_run_trace_cursor(self, sandbox_id: SandboxId, manifest: CheckpointManifest) -> None:
        sandbox = self._sandbox_by_id.get(sandbox_id)
        if sandbox is None or sandbox.task_run is None:
            return
        metadata = manifest.metadata if isinstance(manifest.metadata, dict) else {}
        restore_trace_cursor = None
        raw_restore_checkpoint_id = metadata.get("filesystem_restore_checkpoint_id")
        if raw_restore_checkpoint_id is not None and self.storage is not None:
            try:
                filesystem_manifest = self.storage.get_manifest(sandbox_id, CheckpointId(str(raw_restore_checkpoint_id)))
            except Exception:
                logger.debug(
                    "Failed to resolve filesystem restore cursor for sandbox=%s checkpoint=%s",
                    sandbox_id,
                    raw_restore_checkpoint_id,
                    exc_info=True,
                )
            else:
                restore_trace_cursor = self._benchmark_trace_cursor_from_metadata(
                    filesystem_manifest.metadata if isinstance(filesystem_manifest.metadata, dict) else {}
                )
        if restore_trace_cursor is None:
            restore_trace_cursor = self._benchmark_trace_cursor_from_metadata(metadata)
        if restore_trace_cursor is None:
            return
        recorder = getattr(sandbox.task_run, "record_restore_trace_cursor", None)
        if not callable(recorder):
            return
        try:
            recorder(restore_trace_cursor)
        except Exception:
            logger.exception(
                "Failed to record restore trace cursor for sandbox=%s trace_cursor=%s",
                sandbox_id,
                restore_trace_cursor,
            )

    def _restore_llm_service_state(self, sandbox_id: SandboxId, manifest: CheckpointManifest) -> None:
        if self.llm_router_client is None and self.llm_server is None:
            return
        metadata = manifest.metadata if isinstance(manifest.metadata, dict) else {}
        raw_value_p = metadata.get("process_restore_trace_cursor")
        raw_value_f = metadata.get("filesystem_restore_trace_cursor")
        try:
            raw_value = max(int(raw_value_p), int(raw_value_f))
            consumed_response_count = max(0, int(raw_value))
        except (TypeError, ValueError):
            logger.warning(
                "Skipping llm service state restore for sandbox=%s because process_restore_trace_cursor and filesystem_restore_trace_cursor is missing or invalid",
                sandbox_id,
            )
            return
        try:
            self._restore_llm_router_state(
                sandbox_id,
                consumed_response_count=consumed_response_count,
            )
        except Exception:
            logger.exception(
                "Failed to restore llm service state for sandbox=%s consumed_response_count=%s",
                sandbox_id,
                consumed_response_count,
            )

        # self._restore_task_run_trace_cursor(sandbox_id, manifest)
        sandbox = self._sandbox_by_id.get(sandbox_id)
        if sandbox is None or sandbox.task_run is None:
            return
        recorder = getattr(sandbox.task_run, "record_restore_trace_cursor", None)
        if not callable(recorder):
            return
        try:
            recorder(raw_value)
        except Exception:
            logger.exception(
                "Failed to record restore trace cursor for sandbox=%s trace_cursor=%s",
                sandbox_id,
                raw_value,
            )

    def _set_sandbox_running_state(self, sandbox_id: SandboxId, *, is_running: bool) -> None:
        assert self.base_inspector is not None
        snapshot = self.base_inspector.inspect(sandbox_id)
        self.base_inspector.upsert_snapshot(
            replace(
                snapshot,
                is_running=is_running,
                observed_at=utc_now(),
            )
        )

    def _chain_sharing_active(
        self,
        manifests: dict[CheckpointId, CheckpointManifest],
        leaf_checkpoint_id: CheckpointId,
    ) -> bool:
        """Decide whether chain-ancestor sharing is in play for this fork.

        Requires the harness opt-in flag, a runtime that advertises
        incremental-process support (so ``link_ancestor_pre_dump`` is
        meaningful), and a leaf manifest with a parent — otherwise there
        are no ancestors to share.
        """
        if not getattr(self, "fork_chain_sharing_enabled", False):
            return False
        if self.runtime is None:
            return False
        try:
            capabilities = self.runtime.capabilities()
        except Exception:
            return False
        if not getattr(capabilities, "supports_incremental_process", False):
            return False
        leaf = manifests.get(leaf_checkpoint_id)
        if leaf is None:
            return False
        return leaf.parent_checkpoint_id is not None

    def _rewrite_process_artifact(
        self,
        payload: bytes,
        source_sandbox_id: SandboxId,
        target_sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
        *,
        preserve_symlinks: bool = False,
    ) -> bytes:
        bundle_root = self._effective_runtime_bundle_root()
        checkpoint_root = self._effective_runtime_checkpoint_root()
        data = json.loads(payload.decode("utf-8"))
        process_root = checkpoint_root / str(target_sandbox_id) / str(checkpoint_id)
        # When chain-sharing is active, preserve the leaf's `pre_dump/parent`
        # symlink (which points relatively at ``../../ckpt-PARENT/pre_dump``)
        # instead of dereferencing it. Default ``copytree`` behavior would
        # inline every ancestor's pre-dump bytes underneath
        # ``<target>/ckpt-LEAF/pre_dump/parent/...``, which voids the whole
        # point of chain sharing. The relative symlink resolves correctly
        # once we plant the per-ancestor ``<target>/ckpt-PARENT`` symlinks
        # below via ``Runtime.link_ancestor_pre_dump``.
        shutil.copytree(
            checkpoint_root / str(source_sandbox_id) / str(checkpoint_id),
            process_root,
            dirs_exist_ok=True,
            symlinks=bool(preserve_symlinks),
        )
        data["sandbox_id"] = str(target_sandbox_id)
        data["process_checkpoint_location"] = str(process_root / "process")
        status = data.get("status", {})
        if isinstance(status, dict):
            metadata = status.get("metadata", {})
            if isinstance(metadata, dict):
                metadata["sandbox_id"] = str(target_sandbox_id)
                metadata["checkpoint_id"] = str(checkpoint_id)
                metadata["bundle_path"] = str(bundle_root / str(target_sandbox_id))
                metadata["image_path"] = str(process_root / "process")
                metadata["work_path"] = str(process_root / "work")
        return json.dumps(data, sort_keys=True, indent=2).encode("utf-8")

    def _rewrite_process_artifact_linked(
        self,
        payload: bytes,
        target_sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> bytes:
        """Same JSON rewrite as ``_rewrite_process_artifact`` but skips the
        ``shutil.copytree`` of the runtime checkpoint dir. Caller must have
        already invoked ``Runtime.link_ancestor_pre_dump`` so the target
        runtime path resolves to the source's image bytes via symlink. Used
        when ``fork_chain_sharing_enabled`` is on for chain-ancestor entries.
        """
        bundle_root = self._effective_runtime_bundle_root()
        checkpoint_root = self._effective_runtime_checkpoint_root()
        data = json.loads(payload.decode("utf-8"))
        process_root = checkpoint_root / str(target_sandbox_id) / str(checkpoint_id)
        data["sandbox_id"] = str(target_sandbox_id)
        data["process_checkpoint_location"] = str(process_root / "process")
        status = data.get("status", {})
        if isinstance(status, dict):
            metadata = status.get("metadata", {})
            if isinstance(metadata, dict):
                metadata["sandbox_id"] = str(target_sandbox_id)
                metadata["checkpoint_id"] = str(checkpoint_id)
                metadata["bundle_path"] = str(bundle_root / str(target_sandbox_id))
                metadata["image_path"] = str(process_root / "process")
                metadata["work_path"] = str(process_root / "work")
        return json.dumps(data, sort_keys=True, indent=2).encode("utf-8")

    def _rewrite_filesystem_artifact(
        self,
        payload: bytes,
        source_sandbox_id: SandboxId,
        target_sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> bytes:
        _ = source_sandbox_id
        bundle_root = self._effective_runtime_bundle_root()
        data = json.loads(payload.decode("utf-8"))
        target_snapshot = f"{self.pool_name}/crab/{target_sandbox_id}@{checkpoint_id}"
        filesystem = data.get("filesystem", {})
        if isinstance(filesystem, dict):
            filesystem["dataset"] = f"{self.pool_name}/crab/{target_sandbox_id}"
            filesystem["snapshot"] = target_snapshot
            filesystem["mountpoint"] = str(bundle_root / str(target_sandbox_id) / "rootfs")
        status = data.get("status", {})
        if isinstance(status, dict):
            metadata = status.get("metadata", {})
            if isinstance(metadata, dict):
                metadata["sandbox_id"] = str(target_sandbox_id)
                metadata["checkpoint_id"] = str(checkpoint_id)
                metadata["dataset"] = f"{self.pool_name}/crab/{target_sandbox_id}"
                metadata["snapshot"] = target_snapshot
                metadata["mountpoint"] = str(bundle_root / str(target_sandbox_id) / "rootfs")
        data["sandbox_id"] = str(target_sandbox_id)
        return json.dumps(data, sort_keys=True, indent=2).encode("utf-8")
