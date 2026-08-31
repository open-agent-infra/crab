from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from integrations.sandboxes.runtime.baseline import apply_sandbox_process_baseline
from integrations.sandboxes.runtime.image import ImageRuntimeDefaults


DEFAULT_CPU_PERIOD_US = 100_000


@dataclass(frozen=True)
class SandboxResourceLimits:
    """Optional per-sandbox cgroup resource limits.

    Unset fields leave the corresponding OCI spec section untouched so the
    sandbox inherits host defaults (no limit).
    """

    cpus: int | None = None
    memory_bytes: int | None = None
    pids_limit: int | None = None
    cpu_period_us: int = DEFAULT_CPU_PERIOD_US
    cpu_set: str | None = None
    auto_cpu_set: bool = True

    def is_empty(self) -> bool:
        return (
            self.cpus is None
            and self.memory_bytes is None
            and self.pids_limit is None
            and self.cpu_set is None
        )


_SANDBOX_INDEX_RE = re.compile(r"(\d+)")


def derive_default_cpu_set(sandbox_name: str, cpus: int) -> str | None:
    """Pick a cpuset for the sandbox so `sched_getaffinity()` and
    `len(os.sched_getaffinity(0))`-aware tools (joblib/loky, pytest-xdist,
    `nproc`, OpenMP) see exactly `cpus` CPUs instead of the host count.

    The cgroup `cpu.max` quota alone does not constrain affinity-based
    pool sizing — `os.cpu_count()` and `nproc` read host visibility, not
    the quota. Pinning a cpuset slot per sandbox keeps the worker pools
    bounded without forcing every sandbox onto the same physical cores.

    The first integer component of `sandbox_name` is used as the slot
    index so a parent sandbox and its speculative forks (e.g. `spec-3`
    and `spec-3-spec-1`) share the same physical cores. Slots wrap modulo
    the host CPU count.
    """
    if cpus is None or cpus <= 0:
        return None
    host_cpus = os.cpu_count() or 0
    if host_cpus <= 0:
        return None
    cpus = min(int(cpus), host_cpus)
    match = _SANDBOX_INDEX_RE.search(sandbox_name)
    slot = int(match.group(1)) if match is not None else 0
    start = (slot * cpus) % host_cpus
    end = start + cpus - 1
    if end >= host_cpus:
        # Wrap-around windows would split the cpuset; collapse to the
        # head of the host range so we always emit a contiguous span.
        start = 0
        end = cpus - 1
    return f"{start}-{end}" if cpus > 1 else f"{start}"


_CPU_VISIBILITY_TARGETS = (
    "/sys/devices/system/cpu/online",
    "/sys/devices/system/cpu/possible",
    "/sys/devices/system/cpu/present",
)


def _write_cpu_visibility_overlay(bundle_dir: Path, cpus: int) -> Path | None:
    """Write a fake `/sys/devices/system/cpu/online`-style file in the bundle
    so glibc's `__get_nprocs()` (used by `sysconf(_SC_NPROCESSORS_ONLN)`,
    `os.cpu_count()`, `multiprocessing.cpu_count()`, `nproc`'s fallback,
    and most other "how many CPUs are there" callers) sees `cpus` CPUs
    instead of the host count.

    cgroup `cpuset.cpus` already constrains `sched_getaffinity()`, but
    glibc 2.39 reads `/sys/devices/system/cpu/online` directly for
    `__get_nprocs()` — it does not consult affinity. Without this overlay,
    `os.cpu_count()` returns the host count even with cpuset applied,
    which is what blows up `concurrent.futures.ProcessPoolExecutor`'s
    default worker count (e.g. PyStan/httpstan's forkserver pool).
    """
    if cpus is None or cpus <= 0:
        return None
    overlay_dir = bundle_dir / "cpu-visibility"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    target = overlay_dir / "online"
    body = f"0-{cpus - 1}\n" if cpus > 1 else "0\n"
    target.write_text(body)
    return target


def _cpu_visibility_mounts(overlay_path: Path) -> list[dict[str, object]]:
    return [
        {
            "destination": destination,
            "source": str(overlay_path),
            "type": "bind",
            "options": ["rbind", "ro"],
        }
        for destination in _CPU_VISIBILITY_TARGETS
    ]


def _apply_resource_limits(linux_cfg: dict, limits: SandboxResourceLimits | None) -> None:
    if limits is None or limits.is_empty():
        return
    resources = dict(linux_cfg.get("resources") or {})
    cpu_cfg = dict(resources.get("cpu") or {})
    if limits.cpus is not None:
        cpus = int(limits.cpus)
        if cpus <= 0:
            raise ValueError(f"resource_limits.cpus must be positive, got {cpus}")
        period = int(limits.cpu_period_us)
        if period <= 0:
            raise ValueError(f"resource_limits.cpu_period_us must be positive, got {period}")
        cpu_cfg["period"] = period
        cpu_cfg["quota"] = cpus * period
    if limits.cpu_set:
        cpu_cfg["cpus"] = str(limits.cpu_set)
    if cpu_cfg:
        resources["cpu"] = cpu_cfg
    if limits.memory_bytes is not None:
        memory = int(limits.memory_bytes)
        if memory <= 0:
            raise ValueError(f"resource_limits.memory_bytes must be positive, got {memory}")
        memory_cfg = dict(resources.get("memory") or {})
        memory_cfg["limit"] = memory
        resources["memory"] = memory_cfg
    if limits.pids_limit is not None:
        pids = int(limits.pids_limit)
        if pids <= 0:
            raise ValueError(f"resource_limits.pids_limit must be positive, got {pids}")
        pids_cfg = dict(resources.get("pids") or {})
        pids_cfg["limit"] = pids
        resources["pids"] = pids_cfg
    linux_cfg["resources"] = resources


def concurrency_env_for_cpu_limit(cpus: int | None) -> dict[str, str]:
    """Env vars that advertise the CPU limit to workloads whose process/thread
    pool sizing is not affinity-aware. cgroup cpuset alone is insufficient
    because Linux `sysconf(_SC_NPROCESSORS_ONLN)` — used by Python's
    `multiprocessing.cpu_count()` and friends — reports the host count.
    """
    if cpus is None:
        return {}
    value = str(int(cpus))
    return {
        "DJANGO_TEST_PROCESSES": value,
        "OMP_NUM_THREADS": value,
        "OPENBLAS_NUM_THREADS": value,
        "MKL_NUM_THREADS": value,
        "NUMEXPR_MAX_THREADS": value,
        # joblib/loky reads /sys/fs/cgroup/cpu.max to size Pools; because we
        # drop the cgroup namespace (see write_bundle_config) the container
        # sees the host root cgroup ("max") and falls through to
        # os.cpu_count(), sizing Pools to host core count regardless of our
        # cpu cgroup quota. Cap explicitly.
        "LOKY_MAX_CPU_COUNT": value,
    }


def parse_env_assignments(values: Iterable[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for entry in values:
        key, sep, value = str(entry).partition("=")
        if not sep:
            raise ValueError(f"invalid environment assignment: {entry!r}")
        parsed[key] = value
    return parsed


def merge_environment_defaults(defaults: Iterable[str], overrides: Iterable[str]) -> list[str]:
    merged = parse_env_assignments(defaults)
    merged.update(parse_env_assignments(overrides))
    return [f"{key}={value}" for key, value in merged.items()]


def parse_passwd_file(path: Path) -> tuple[dict[str, tuple[int, int]], dict[int, tuple[int, str]]]:
    by_name: dict[str, tuple[int, int]] = {}
    by_uid: dict[int, tuple[int, str]] = {}
    if not path.is_file():
        return by_name, by_uid
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line or raw_line.startswith("#"):
            continue
        parts = raw_line.split(":")
        if len(parts) < 4:
            continue
        name = parts[0]
        try:
            uid = int(parts[2])
            gid = int(parts[3])
        except ValueError:
            continue
        by_name[name] = (uid, gid)
        by_uid[uid] = (gid, name)
    return by_name, by_uid


def parse_group_file(path: Path) -> tuple[dict[str, int], dict[int, str]]:
    by_name: dict[str, int] = {}
    by_gid: dict[int, str] = {}
    if not path.is_file():
        return by_name, by_gid
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line or raw_line.startswith("#"):
            continue
        parts = raw_line.split(":")
        if len(parts) < 3:
            continue
        name = parts[0]
        try:
            gid = int(parts[2])
        except ValueError:
            continue
        by_name[name] = gid
        by_gid[gid] = name
    return by_name, by_gid


def resolve_process_user_from_rootfs(*, rootfs_dir: Path, user_spec: str) -> dict[str, object]:
    value = user_spec.strip()
    if not value:
        raise ValueError("compose/image user must not be empty")
    passwd_by_name, passwd_by_uid = parse_passwd_file(rootfs_dir / "etc" / "passwd")
    group_by_name, _ = parse_group_file(rootfs_dir / "etc" / "group")

    user_part, sep, group_part = value.partition(":")
    if not user_part:
        raise ValueError(f"unsupported user specification: {user_spec!r}")

    uid: int
    gid: int | None = None
    if user_part.isdigit():
        uid = int(user_part)
        if uid in passwd_by_uid:
            gid = passwd_by_uid[uid][0]
    else:
        if user_part not in passwd_by_name:
            raise ValueError(f"user {user_part!r} not found in rootfs {rootfs_dir}")
        uid, gid = passwd_by_name[user_part]

    if sep:
        if not group_part:
            raise ValueError(f"unsupported user specification: {user_spec!r}")
        if group_part.isdigit():
            gid = int(group_part)
        else:
            if group_part not in group_by_name:
                raise ValueError(f"group {group_part!r} not found in rootfs {rootfs_dir}")
            gid = group_by_name[group_part]

    if gid is None:
        gid = 0
    return {"uid": uid, "gid": gid}


def write_bundle_config(
    *,
    bundle_dir: Path,
    llm_base_url: str,
    provider: str,
    sandbox_name: str,
    status_port: int,
    cgroup_path: str,
    work_dir_host_path: Path | None = None,
    network_namespace_path: Path | None = None,
    image_defaults: ImageRuntimeDefaults | None = None,
    image_rootfs_dir: Path | None = None,
    resource_limits: SandboxResourceLimits | None = None,
) -> None:
    config_path = bundle_dir / "config.json"
    cfg = json.loads(config_path.read_text())
    linux_cfg = cfg.get("linux", {})
    namespaces = []
    network_namespace_found = False
    for namespace in linux_cfg.get("namespaces", []):
        ns_type = namespace.get("type")
        if ns_type == "cgroup":
            continue
        if ns_type == "network":
            network_namespace_found = True
            if network_namespace_path is None:
                continue
            namespace = {**namespace, "path": str(network_namespace_path)}
        namespaces.append(namespace)
    if network_namespace_path is not None and not network_namespace_found:
        namespaces.append({"type": "network", "path": str(network_namespace_path)})
    linux_cfg["namespaces"] = namespaces
    linux_cfg["cgroupsPath"] = cgroup_path
    linux_cfg.pop("seccomp", None)
    if (
        resource_limits is not None
        and resource_limits.cpus is not None
        and resource_limits.cpu_set is None
        and resource_limits.auto_cpu_set
    ):
        derived = derive_default_cpu_set(sandbox_name, int(resource_limits.cpus))
        if derived is not None:
            from dataclasses import replace as _dc_replace

            resource_limits = _dc_replace(resource_limits, cpu_set=derived)
    _apply_resource_limits(linux_cfg, resource_limits)
    cfg["linux"] = linux_cfg
    excluded_destinations = {"/work", *_CPU_VISIBILITY_TARGETS}
    mounts = [mount for mount in cfg.get("mounts", []) if mount.get("destination") not in excluded_destinations]
    if work_dir_host_path is not None:
        work_dir_host_path.mkdir(parents=True, exist_ok=True)
        mounts.append(
            {
                "destination": "/work",
                "source": str(work_dir_host_path),
                "type": "bind",
                "options": ["rbind", "rw"],
            }
        )
    if resource_limits is not None and resource_limits.cpus is not None:
        overlay_path = _write_cpu_visibility_overlay(bundle_dir, int(resource_limits.cpus))
        if overlay_path is not None:
            mounts.extend(_cpu_visibility_mounts(overlay_path))
    cfg["mounts"] = mounts
    cfg["process"]["terminal"] = False
    cfg["process"]["cwd"] = image_defaults.working_dir if image_defaults and image_defaults.working_dir else "/work"
    cfg["process"]["args"] = [
        "/bin/sh",
        "-lc",
        f"exec /usr/local/bin/agent-cli run --provider {provider} >/dev/null 2>/dev/null",
    ]
    runtime_env = [
        "PATH=/usr/local/bin:/usr/bin:/bin",
        "PYTHONUNBUFFERED=1",
        f"CRAB_LLM_BASE_URL={llm_base_url}",
        f"STATUS_PORT={status_port}",
        "POLL_INTERVAL_S=0.2",
        "AGENT_WORK_DIR=/work",
        f"AGENT_SANDBOX_ID={sandbox_name}",
        f"AGENT_PROVIDER={provider}",
    ]
    concurrency_env = concurrency_env_for_cpu_limit(
        resource_limits.cpus if resource_limits is not None else None
    )
    runtime_env.extend(f"{key}={value}" for key, value in concurrency_env.items())
    cfg["process"]["env"] = merge_environment_defaults(
        image_defaults.environment if image_defaults is not None else (),
        runtime_env,
    )
    if image_defaults is not None and image_defaults.user is not None:
        if image_rootfs_dir is None:
            raise ValueError("image_rootfs_dir is required when image defaults specify a user")
        cfg["process"]["user"] = resolve_process_user_from_rootfs(
            rootfs_dir=image_rootfs_dir,
            user_spec=image_defaults.user,
        )
    cfg["root"]["path"] = "rootfs"
    cfg["root"]["readonly"] = False
    apply_sandbox_process_baseline(cfg)
    # The default `runc spec` leaves hostname as "runc"; use the sandbox id so
    # each sandbox is individually identifiable from inside (e.g. `hostname`).
    cfg["hostname"] = sandbox_name
    config_path.write_text(json.dumps(cfg, indent=2))
