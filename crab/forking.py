from __future__ import annotations

import copy
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

from .ids import CheckpointId, SandboxId
from .models import CheckpointManifest

logger = logging.getLogger(__name__)

_LOGICAL_CHECKPOINT = "logical_checkpoint"
_PROCESS_RESTORE_CHECKPOINT_ID = "process_restore_checkpoint_id"
_FILESYSTEM_RESTORE_CHECKPOINT_ID = "filesystem_restore_checkpoint_id"


@dataclass(frozen=True)
class ForkResult:
    """Outcome of CrabSystem.fork_once (checkpoint-state clone only; the
    fork is left stopped — process restore is the caller's step)."""

    source_sandbox_id: SandboxId
    target_sandbox_id: SandboxId
    checkpoint_id: CheckpointId
    filesystem_checkpoint_id: CheckpointId
    chain_shared: bool
    chain_links: int
    chain_bytes_saved: int


def resolve_checkpoint_copy_plan(
    checkpoint_order: list[CheckpointId],
    manifests: dict[CheckpointId, CheckpointManifest],
    checkpoint_id: CheckpointId,
) -> list[tuple[CheckpointId, bool, bool]]:
    """Which (checkpoint, copy_process, copy_filesystem) tuples a fork needs
    so the leaf checkpoint is restorable on the fork side. Logical manifests
    use their explicit component mappings; legacy manifests retain the
    historical nearest-artifact scan. ``benchmarks.support`` re-exports this
    helper as part of the fork wiring (task A3)."""
    manifest = manifests[checkpoint_id]
    if bool(manifest.metadata.get(_LOGICAL_CHECKPOINT, False)):
        return _resolve_logical_checkpoint_copy_plan(
            checkpoint_order,
            manifests,
            checkpoint_id,
            manifest,
        )

    plan: list[tuple[CheckpointId, bool, bool]] = [
        (checkpoint_id, bool(manifest.process_artifacts), bool(manifest.filesystem_artifacts))
    ]
    need_process = not bool(manifest.process_artifacts)
    need_filesystem = not bool(manifest.filesystem_artifacts)
    if not need_process and not need_filesystem:
        return plan

    try:
        current_index = checkpoint_order.index(checkpoint_id)
        candidates = list(reversed(checkpoint_order[:current_index]))
    except ValueError:
        candidates = list(reversed(checkpoint_order))

    for candidate_id in candidates:
        if not need_process and not need_filesystem:
            break
        candidate = manifests[candidate_id]
        copy_process = need_process and bool(candidate.process_artifacts)
        copy_filesystem = need_filesystem and bool(candidate.filesystem_artifacts)
        if not copy_process and not copy_filesystem:
            continue
        plan.insert(0, (candidate_id, copy_process, copy_filesystem))
        if copy_process:
            need_process = False
        if copy_filesystem:
            need_filesystem = False

    if need_process or need_filesystem:
        raise ValueError(f"unable to resolve restore dependencies for checkpoint {checkpoint_id}")
    return plan


def _resolve_logical_checkpoint_copy_plan(
    checkpoint_order: list[CheckpointId],
    manifests: dict[CheckpointId, CheckpointManifest],
    checkpoint_id: CheckpointId,
    manifest: CheckpointManifest,
) -> list[tuple[CheckpointId, bool, bool]]:
    """Resolve a logical leaf from its exact component mappings.

    The legacy planner scans backwards for the nearest artifacts. Logical
    manifests deliberately carry explicit mappings, so scanning can copy a
    different recovery point when an older logical id is forked after newer
    physical checkpoints have been recorded.
    """
    process_source_raw = manifest.metadata.get(_PROCESS_RESTORE_CHECKPOINT_ID)
    filesystem_source_raw = manifest.metadata.get(
        _FILESYSTEM_RESTORE_CHECKPOINT_ID
    )
    if process_source_raw is None or filesystem_source_raw is None:
        raise ValueError(
            f"logical checkpoint {checkpoint_id} is missing explicit "
            "process/filesystem restore sources"
        )

    required: dict[CheckpointId, list[bool]] = {
        checkpoint_id: [
            bool(manifest.process_artifacts),
            bool(manifest.filesystem_artifacts),
        ]
    }
    for source_raw, component_index, component_name in (
        (process_source_raw, 0, "process"),
        (filesystem_source_raw, 1, "filesystem"),
    ):
        source_id = CheckpointId(str(source_raw))
        source = manifests.get(source_id)
        if source is None:
            raise ValueError(
                f"logical checkpoint {checkpoint_id} references missing "
                f"{component_name} source {source_id}"
            )
        artifacts = (
            source.process_artifacts
            if component_index == 0
            else source.filesystem_artifacts
        )
        if not artifacts:
            raise ValueError(
                f"logical checkpoint {checkpoint_id} references {component_name} "
                f"source {source_id} without {component_name} artifacts"
            )
        flags = required.setdefault(source_id, [False, False])
        flags[component_index] = True

    plan: list[tuple[CheckpointId, bool, bool]] = []
    ordered_ids = set(checkpoint_order)
    missing_from_order = set(required).difference(ordered_ids)
    if missing_from_order:
        raise ValueError(
            f"logical checkpoint {checkpoint_id} restore sources are missing "
            f"from checkpoint order: {sorted(str(item) for item in missing_from_order)}"
        )
    for candidate_id in checkpoint_order:
        if candidate_id == checkpoint_id or candidate_id not in required:
            continue
        copy_process, copy_filesystem = required[candidate_id]
        plan.append((candidate_id, copy_process, copy_filesystem))
    copy_process, copy_filesystem = required[checkpoint_id]
    plan.append((checkpoint_id, copy_process, copy_filesystem))
    return plan


def rewrite_process_artifact(
    payload: bytes,
    *,
    source_sandbox_id: SandboxId,
    target_sandbox_id: SandboxId,
    checkpoint_id: CheckpointId,
    bundle_root: Path,
    checkpoint_root: Path,
    preserve_symlinks: bool = False,
) -> bytes:
    """Copy a process checkpoint's runtime tree to the fork and rewrite the
    artifact payload's sandbox-scoped paths. Sunk from the benchmark
    harness; parameters that the harness read off ``self`` are explicit.

    When chain-sharing is active, ``preserve_symlinks=True`` keeps the
    leaf's ``pre_dump/parent`` relative symlink instead of dereferencing it
    (default ``copytree`` would inline every ancestor's pre-dump bytes,
    voiding chain sharing). The relative link resolves once the caller
    plants per-ancestor symlinks via ``Runtime.link_ancestor_pre_dump``.
    """
    data = json.loads(payload.decode("utf-8"))
    process_root = checkpoint_root / str(target_sandbox_id) / str(checkpoint_id)
    shutil.copytree(
        checkpoint_root / str(source_sandbox_id) / str(checkpoint_id),
        process_root,
        dirs_exist_ok=True,
        symlinks=bool(preserve_symlinks),
    )
    return _rewrite_process_payload(data, target_sandbox_id, checkpoint_id, bundle_root, process_root)


def rewrite_process_artifact_linked(
    payload: bytes,
    *,
    target_sandbox_id: SandboxId,
    checkpoint_id: CheckpointId,
    bundle_root: Path,
    checkpoint_root: Path,
) -> bytes:
    """Same JSON rewrite as ``rewrite_process_artifact`` but without the
    ``copytree``: the caller must have already invoked
    ``Runtime.link_ancestor_pre_dump`` so the target runtime path resolves
    to the source's image bytes via symlink (chain-ancestor entries)."""
    data = json.loads(payload.decode("utf-8"))
    process_root = checkpoint_root / str(target_sandbox_id) / str(checkpoint_id)
    return _rewrite_process_payload(data, target_sandbox_id, checkpoint_id, bundle_root, process_root)


def _rewrite_process_payload(
    data: dict[str, object],
    target_sandbox_id: SandboxId,
    checkpoint_id: CheckpointId,
    bundle_root: Path,
    process_root: Path,
) -> bytes:
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


def rewrite_filesystem_artifact(
    payload: bytes,
    *,
    target_sandbox_id: SandboxId,
    checkpoint_id: CheckpointId,
    filesystem_metadata: dict[str, object],
) -> bytes:
    """Rewrite a filesystem artifact payload for the fork.

    Backend-neutral generalization of the harness helper (which hardcoded
    zfs ``pool/crab/<id>@<ckpt>`` naming): the caller supplies the fork's
    ``runtime.filesystem_checkpoint_metadata(target, ckpt)`` and its
    dataset/snapshot/mountpoint/fs_ref values are stamped over both the
    ``filesystem`` block and ``status.metadata``. This also fixes the
    payload's ``fs_ref`` so fork-side retention destroys the fork's
    snapshot, not the source's.
    """
    data = json.loads(payload.decode("utf-8"))
    stamped_keys = ("dataset", "snapshot", "mountpoint", "fs_ref")
    filesystem = data.get("filesystem", {})
    if isinstance(filesystem, dict):
        for key in stamped_keys:
            if key in filesystem_metadata:
                filesystem[key] = filesystem_metadata[key]
    status = data.get("status", {})
    if isinstance(status, dict):
        metadata = status.get("metadata", {})
        if isinstance(metadata, dict):
            metadata["sandbox_id"] = str(target_sandbox_id)
            metadata["checkpoint_id"] = str(checkpoint_id)
            for key in stamped_keys:
                if key in filesystem_metadata:
                    metadata[key] = filesystem_metadata[key]
    data["sandbox_id"] = str(target_sandbox_id)
    return json.dumps(data, sort_keys=True, indent=2).encode("utf-8")


def retarget_bundle_network_namespace(target_bundle_dir: Path, netns_path: str) -> bool:
    """Point the fork's spec at its own network namespace.

    The fork's config.json is copied from the source, so without this it
    keeps the *source's* netns path: the fork then shares the source's
    network stack and its egress is attributed to the source (which also
    made the fork's allocated lease dead weight). Returns True when the
    spec was changed.
    """
    config_path = target_bundle_dir / "config.json"
    if not config_path.is_file():
        return False
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    namespaces = config.get("linux", {}).get("namespaces")
    if not isinstance(namespaces, list):
        return False
    changed = False
    for namespace in namespaces:
        if isinstance(namespace, dict) and namespace.get("type") == "network":
            if namespace.get("path") != netns_path:
                namespace["path"] = netns_path
                changed = True
    if not changed:
        return False
    try:
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    except OSError:
        logger.warning("Failed to retarget fork network namespace", exc_info=True)
        return False
    return True


def replicate_bundle_config(
    source_bundle_dir: Path,
    target_bundle_dir: Path,
    source_sandbox_id: SandboxId,
    target_sandbox_id: SandboxId,
) -> None:
    """Replicate the runtime-relevant parts of the source bundle spec onto
    the fork's bundle, rewriting per-sandbox host paths.

    CRIU's process image references mountpoint IDs from the source spec; a
    mismatch fails restore with ``mnt: No mapping for X:(null) mountpoint``.
    Capabilities/user/cwd/env/noNewPrivileges are copied so the fork's exec
    environment matches the source's (sunk from the harness's
    ``_replicate_source_bundle_mounts``; see that docstring's war stories).
    """
    source_cfg_path = source_bundle_dir / "config.json"
    target_cfg_path = target_bundle_dir / "config.json"
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
        source_token = f"/{source_sandbox_id}/"
        target_token = f"/{target_sandbox_id}/"
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
                    if Path(raw_source).is_file():
                        # File bind (e.g. the cpu-visibility overlay written
                        # for cpu-limited sandboxes): mkdir would plant a
                        # directory where runc expects a file, so copy the
                        # source file instead.
                        rewritten_path.parent.mkdir(parents=True, exist_ok=True)
                        if not rewritten_path.exists():
                            shutil.copy2(raw_source, rewritten_path)
                    else:
                        rewritten_path.mkdir(parents=True, exist_ok=True)
                    new_mount["source"] = rewritten_source
                except OSError:
                    pass
            rewritten.append(new_mount)
        target_cfg["mounts"] = rewritten
        changed = True
    source_linux = source_cfg.get("linux")
    if isinstance(source_linux, dict):
        cgroups_path = source_linux.get("cgroupsPath")
        if isinstance(cgroups_path, str) and str(source_sandbox_id) in cgroups_path:
            # CRIU restores into the spec's cgroup; inheriting the source's
            # path fails with "container's cgroup is not empty" while the
            # source is still running.
            target_linux = target_cfg.setdefault("linux", {})
            if isinstance(target_linux, dict):
                target_linux["cgroupsPath"] = cgroups_path.replace(
                    str(source_sandbox_id), str(target_sandbox_id)
                )
                changed = True
        source_resources = source_linux.get("resources")
        if isinstance(source_resources, dict) and source_resources:
            # S3: forks inherit the source's cgroup limits — the fork's own
            # cgroup (rewritten above) gets the same `linux.resources`.
            # Absent on the source -> absent on the fork (no limits).
            target_linux = target_cfg.setdefault("linux", {})
            if isinstance(target_linux, dict):
                target_linux["resources"] = copy.deepcopy(source_resources)
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
            source_cwd = source_process.get("cwd")
            if isinstance(source_cwd, str) and source_cwd:
                target_process["cwd"] = source_cwd
                changed = True
            source_env = source_process.get("env")
            if isinstance(source_env, list):
                target_process["env"] = list(source_env)
                changed = True
            if "noNewPrivileges" in source_process:
                target_process["noNewPrivileges"] = bool(source_process["noNewPrivileges"])
                changed = True
    if not changed:
        return
    target_cfg_path.write_text(json.dumps(target_cfg, indent=2), encoding="utf-8")
    logger.debug(
        "Replicated source bundle layout to fork source=%s target=%s",
        source_sandbox_id,
        target_sandbox_id,
    )
