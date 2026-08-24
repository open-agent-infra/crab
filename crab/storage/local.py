from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
import shutil
from typing import Callable

from ..config import StorageConfig
from ..contracts import CheckpointManager
from ..ids import CheckpointId, SandboxId
from ..json_codec import get_json_codec
from ..models import ArtifactPayload, ArtifactReference, CheckpointManifest

logger = logging.getLogger(__name__)
_STORAGE_JSON_CODEC = get_json_codec("auto")


def _stable_json_bytes(payload: object) -> bytes:
    return _STORAGE_JSON_CODEC.dumps_bytes(payload, sort_keys=True)


def _safe_name(value: str) -> str:
    if not value:
        raise ValueError("artifact name must be non-empty")
    return value.replace("/", "_")


class LocalCheckpointManager(CheckpointManager):
    def __init__(
        self,
        config: StorageConfig,
        *,
        runtime_image_path_in_use: Callable[[Path], bool] | None = None,
        destroy_filesystem_ref: Callable[[str], None] | None = None,
    ):
        self._config = config
        self._root = Path(config.root_dir)
        self._manifests_root = self._root / config.manifests_dirname
        self._artifacts_root = self._root / config.artifacts_dirname
        # Optional runtime-side predicate consulted before ``shutil.rmtree``-ing
        # a runtime checkpoint tree. Returns True when the tree (or a
        # descendant) is the on-disk image source for an active runtime
        # operation whose mid-flight failure would manifest as a fatal
        # signal rather than a clean error — concretely, an in-flight
        # ``criu lazy-pages`` daemon serving userfaultfd faults. When the
        # predicate says "in use" we leave the runtime tree in place and
        # log a warning; the next retention pass after the daemon exits
        # will reclaim it. None means "no safety check installed" (e.g.
        # in-memory tests, runtimes without lazy-pages support).
        self._runtime_image_path_in_use = runtime_image_path_in_use
        # Runtime-side hook that destroys a filesystem checkpoint by its
        # opaque fs_ref (routed to the runtime's filesystem provider).
        # Storage used to shell out `zfs destroy` itself; backend-specific
        # commands now live behind the provider. None means "no hook
        # installed" (in-memory tests, runtimes without fs checkpoints) —
        # retention then skips snapshot cleanup and logs at debug.
        self._destroy_filesystem_ref = destroy_filesystem_ref
        self._manifests_root.mkdir(parents=True, exist_ok=True)
        self._artifacts_root.mkdir(parents=True, exist_ok=True)

    def set_destroy_filesystem_ref(
        self,
        callback: Callable[[str], None] | None,
    ) -> None:
        """Late-bind the filesystem-ref destroy hook. Used when storage is
        constructed before the runtime instance is available (same
        pattern as ``set_runtime_image_path_in_use``)."""
        self._destroy_filesystem_ref = callback

    def set_runtime_image_path_in_use(
        self,
        callback: Callable[[Path], bool] | None,
    ) -> None:
        """Late-bind the runtime safety predicate. Used when storage is
        constructed before the runtime instance is available (e.g., the
        benchmark harness builds storage first)."""
        self._runtime_image_path_in_use = callback

    def _manifest_path(self, sandbox_id: SandboxId, checkpoint_id: CheckpointId) -> Path:
        return self._manifests_root / str(sandbox_id) / f"{checkpoint_id}.json"

    def _artifact_path(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
        kind: str,
        name: str,
    ) -> Path:
        return (
            self._artifacts_root
            / str(sandbox_id)
            / str(checkpoint_id)
            / kind
            / _safe_name(name)
        )

    def put_manifest(self, manifest: CheckpointManifest) -> None:
        manifest.validate_schema()
        manifest.validate_integrity()
        path = self._manifest_path(manifest.sandbox_id, manifest.checkpoint_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(manifest.to_json_bytes())
        tmp.replace(path)
        logger.debug(
            "Stored manifest for sandbox=%s checkpoint=%s at %s",
            manifest.sandbox_id,
            manifest.checkpoint_id,
            path,
        )

    def get_manifest(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> CheckpointManifest:
        path = self._manifest_path(sandbox_id, checkpoint_id)
        if not path.exists():
            raise FileNotFoundError(f"manifest not found: {path}")
        raw = _STORAGE_JSON_CODEC.loads(path.read_bytes())
        logger.debug("Loaded manifest for sandbox=%s checkpoint=%s from %s", sandbox_id, checkpoint_id, path)
        return CheckpointManifest.from_dict(raw)

    def put_artifact(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
        artifact: ArtifactPayload,
    ) -> ArtifactReference:
        path = self._artifact_path(
            sandbox_id,
            checkpoint_id,
            artifact.kind.value,
            artifact.name,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(artifact.data)
        tmp.replace(path)

        digest = hashlib.sha256(artifact.data).hexdigest()
        size = len(artifact.data)
        rel = str(path.relative_to(self._root))
        reference = ArtifactReference(
            kind=artifact.kind,
            name=artifact.name,
            relative_path=rel,
            size_bytes=size,
            sha256=digest,
            metadata=dict(artifact.metadata),
        )
        logger.debug(
            "Stored artifact %s for sandbox=%s checkpoint=%s at %s (%d bytes)",
            artifact.name,
            sandbox_id,
            checkpoint_id,
            path,
            size,
        )
        return reference

    def link_ancestor_artifact(
        self,
        source_sandbox_id: SandboxId,
        target_sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
        reference: ArtifactReference,
    ) -> ArtifactReference:
        """Materialize ``reference`` (which lives under ``source_sandbox_id``'s
        storage tree) at ``target_sandbox_id``'s artifact path as a relative
        symlink instead of a byte copy. Returns a new ``ArtifactReference``
        whose ``relative_path`` points at the target-side symlink so the fork
        manifest references its own storage tree (transparent to readers and
        survives source materialization). The byte payload is unchanged so
        ``size_bytes`` and ``sha256`` carry over."""
        source_path = self._root / reference.relative_path
        if not source_path.exists():
            raise FileNotFoundError(f"source artifact not found: {source_path}")
        target_path = self._artifact_path(
            target_sandbox_id,
            checkpoint_id,
            reference.kind.value,
            reference.name,
        )
        target_path.parent.mkdir(parents=True, exist_ok=True)
        # Idempotent: if a previous link or copy is already in place at the
        # target path (e.g. a retried fork prep), drop it before linking.
        if target_path.exists() or target_path.is_symlink():
            target_path.unlink()
        rel_target = os.path.relpath(source_path, target_path.parent)
        os.symlink(rel_target, target_path)
        new_rel = str(target_path.relative_to(self._root))
        logger.debug(
            "Linked ancestor artifact %s for source=%s target=%s checkpoint=%s "
            "(target_path=%s -> %s)",
            reference.name,
            source_sandbox_id,
            target_sandbox_id,
            checkpoint_id,
            target_path,
            rel_target,
        )
        return ArtifactReference(
            kind=reference.kind,
            name=reference.name,
            relative_path=new_rel,
            size_bytes=reference.size_bytes,
            sha256=reference.sha256,
            metadata=dict(reference.metadata),
        )

    def is_linked_artifact(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
        reference: ArtifactReference,
    ) -> bool:
        _ = (sandbox_id, checkpoint_id)
        path = self._root / reference.relative_path
        return path.is_symlink()

    def materialize_linked_artifacts(self, sandbox_id: SandboxId) -> int:
        """Replace any symlinked artifacts under ``sandbox_id`` with byte
        copies of their targets. Used before destroying a source sandbox
        whose ancestor blobs are still referenced by active fork storage:
        the fork calls this on itself to inline the borrowed bytes before
        the source's storage tree disappears.

        Returns the count of symlinks replaced. Idempotent: real files
        and dangling symlinks are skipped.
        """
        sandbox_dir = self._artifacts_root / str(sandbox_id)
        if not sandbox_dir.exists():
            return 0
        materialized = 0
        for path in sandbox_dir.rglob("*"):
            if not path.is_symlink():
                continue
            try:
                resolved = path.resolve(strict=True)
            except FileNotFoundError:
                logger.warning(
                    "Skipping dangling symlink during materialization: %s", path
                )
                continue
            data = resolved.read_bytes()
            tmp = path.with_suffix(path.suffix + ".materializing")
            tmp.write_bytes(data)
            path.unlink()
            tmp.replace(path)
            materialized += 1
        if materialized:
            logger.info(
                "Materialized %d linked artifact(s) under sandbox=%s",
                materialized,
                sandbox_id,
            )
        return materialized

    def get_artifact(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
        reference: ArtifactReference,
    ) -> bytes:
        # `sandbox_id` and `checkpoint_id` are part of the API contract for callers;
        # they also make intent explicit when reading by reference.
        _ = (sandbox_id, checkpoint_id)
        path = self._root / reference.relative_path
        if not path.exists():
            raise FileNotFoundError(f"artifact not found: {path}")
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != reference.sha256:
            raise ValueError(f"artifact digest mismatch for {reference.name}")
        if len(payload) != reference.size_bytes:
            raise ValueError(f"artifact size mismatch for {reference.name}")
        logger.debug(
            "Loaded artifact %s for sandbox=%s checkpoint=%s from %s",
            reference.name,
            sandbox_id,
            checkpoint_id,
            path,
        )
        return payload

    def list_checkpoints(self, sandbox_id: SandboxId) -> list[CheckpointId]:
        sandbox_dir = self._manifests_root / str(sandbox_id)
        if not sandbox_dir.exists():
            return []

        manifests: list[tuple[str, CheckpointId]] = []
        for path in sandbox_dir.glob("*.json"):
            try:
                raw = _STORAGE_JSON_CODEC.loads(path.read_bytes())
            except FileNotFoundError:
                logger.debug("Skipping manifest that disappeared during listing: %s", path)
                continue
            except ValueError:
                logger.warning("Skipping unreadable manifest while listing checkpoints: %s", path)
                continue
            checkpoint_id = raw.get("checkpoint_id")
            if checkpoint_id is None:
                logger.warning("Skipping manifest missing checkpoint_id while listing checkpoints: %s", path)
                continue
            created_at = str(raw.get("created_at", ""))
            ckpt = CheckpointId(str(checkpoint_id))
            manifests.append((created_at, ckpt))
        manifests.sort(key=lambda x: x[0])
        logger.debug("Listed %d checkpoints for sandbox=%s", len(manifests), sandbox_id)
        return [x[1] for x in manifests]

    def delete_checkpoint(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
        *,
        cascade: bool = False,
    ) -> None:
        manifest: CheckpointManifest | None = None
        manifest_path = self._manifest_path(sandbox_id, checkpoint_id)
        if manifest_path.exists():
            manifest = self.get_manifest(sandbox_id, checkpoint_id)

        descendants = self.descendants(sandbox_id, checkpoint_id)
        if descendants:
            if not cascade:
                raise RuntimeError(
                    "refusing to delete checkpoint with live incremental descendants: "
                    f"sandbox={sandbox_id} checkpoint={checkpoint_id} "
                    f"descendants={[str(d) for d in descendants]} "
                    "(pass cascade=True to drop the whole chain)"
                )
            # Cascade: delete descendants first (they hold the actual page
            # data we'd otherwise orphan).
            for descendant in descendants:
                self.delete_checkpoint(sandbox_id, descendant, cascade=False)

        if manifest is not None:
            self._delete_runtime_references(manifest)

        artifact_dir = self._artifacts_root / str(sandbox_id) / str(checkpoint_id)
        if artifact_dir.exists():
            shutil.rmtree(artifact_dir, ignore_errors=True)
        if manifest_path.exists():
            manifest_path.unlink()
        self._prune_empty_parents(manifest_path.parent, stop=self._manifests_root)
        self._prune_empty_parents(artifact_dir.parent, stop=self._artifacts_root)
        logger.debug("Deleted checkpoint sandbox=%s checkpoint=%s", sandbox_id, checkpoint_id)

    def delete_all_checkpoints(self, sandbox_id: SandboxId) -> None:
        # Reverse order = newest first. Combined with the descendants check
        # in delete_checkpoint, this evicts incremental tails before their
        # ancestors so cascade is never needed.
        for checkpoint_id in reversed(self.list_checkpoints(sandbox_id)):
            self.delete_checkpoint(sandbox_id, checkpoint_id)

    def descendants(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> list[CheckpointId]:
        """Transitive descendants whose `parent_checkpoint_id` chain
        bottoms out at ``checkpoint_id``. Returned newest-first so callers
        deleting them in order never orphan a child.
        """
        sandbox_dir = self._manifests_root / str(sandbox_id)
        if not sandbox_dir.exists():
            return []

        parent_by_child: dict[CheckpointId, CheckpointId] = {}
        order: dict[CheckpointId, str] = {}
        for path in sandbox_dir.glob("*.json"):
            try:
                raw = _STORAGE_JSON_CODEC.loads(path.read_bytes())
            except (FileNotFoundError, ValueError):
                continue
            cid_raw = raw.get("checkpoint_id")
            if cid_raw is None:
                continue
            cid = CheckpointId(str(cid_raw))
            order[cid] = str(raw.get("created_at", ""))
            parent_raw = raw.get("parent_checkpoint_id")
            if parent_raw is not None:
                parent_by_child[cid] = CheckpointId(str(parent_raw))

        target = checkpoint_id
        descendants: set[CheckpointId] = set()
        # BFS on the reverse parent map. Cheap because chains are short.
        frontier = {target}
        while frontier:
            next_frontier: set[CheckpointId] = set()
            for cid, parent in parent_by_child.items():
                if parent in frontier and cid not in descendants and cid != target:
                    descendants.add(cid)
                    next_frontier.add(cid)
            frontier = next_frontier
        return sorted(descendants, key=lambda c: order.get(c, ""), reverse=True)

    def handle_checkpoint_complete(self, manifest: CheckpointManifest) -> None:
        _ = manifest

    def handle_restore_complete(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> None:
        _ = (sandbox_id, checkpoint_id)

    def _delete_runtime_references(self, manifest: CheckpointManifest) -> None:
        for reference in [*manifest.process_artifacts, *manifest.filesystem_artifacts]:
            payload = self._load_artifact_payload(manifest.sandbox_id, manifest.checkpoint_id, reference)
            if payload is None:
                continue
            if reference.kind.value == "process":
                self._delete_process_runtime_paths(payload)
            elif reference.kind.value == "filesystem":
                self._delete_filesystem_runtime_paths(payload)

    def _load_artifact_payload(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
        reference: ArtifactReference,
    ) -> dict[str, object] | None:
        try:
            raw = self.get_artifact(sandbox_id, checkpoint_id, reference)
        except FileNotFoundError:
            return None
        try:
            payload = _STORAGE_JSON_CODEC.loads(raw)
        except ValueError:
            return None
        return payload if isinstance(payload, dict) else None

    def _delete_process_runtime_paths(self, payload: dict[str, object]) -> None:
        checkpoint_location = payload.get("process_checkpoint_location")
        if checkpoint_location:
            self._remove_runtime_dir(Path(str(checkpoint_location)).parent)
        status = payload.get("status", {})
        if not isinstance(status, dict):
            return
        metadata = status.get("metadata", {})
        if not isinstance(metadata, dict):
            return
        image_path = metadata.get("image_path")
        if image_path:
            self._remove_runtime_dir(Path(str(image_path)).parent)

    def _remove_runtime_dir(self, path: Path) -> None:
        # The fork-side runtime checkpoint dir for a chain ancestor is a
        # symlink pointing at the source's image dir (created by
        # ``Runtime.link_ancestor_pre_dump``). ``shutil.rmtree`` on a
        # symlink-to-dir raises NotADirectoryError; ``ignore_errors=True``
        # would mask that and leak the symlink. Detect symlinks first and
        # ``os.unlink`` them so we never traverse into the source's bytes.
        if path.is_symlink():
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            return
        if not path.exists():
            return
        # Defer pruning when the runtime says this tree (or a descendant
        # of it) is the on-disk image source for an in-flight lazy-pages
        # daemon. Removing it now would SIGBUS the restored process the
        # next time the kernel raises a userfault for an unloaded page.
        # The next retention pass after the daemon exits will reclaim
        # the tree.
        check = self._runtime_image_path_in_use
        if check is not None:
            try:
                in_use = bool(check(path))
            except Exception:
                logger.exception(
                    "Runtime image-path safety check raised for path=%s; "
                    "treating as not in use to avoid wedging retention.",
                    path,
                )
                in_use = False
            if in_use:
                logger.warning(
                    "Deferring runtime checkpoint dir prune; in use by an "
                    "active runtime operation (lazy-pages daemon serving "
                    "userfault faults from this image source). path=%s",
                    path,
                )
                return
        shutil.rmtree(path, ignore_errors=True)

    def _delete_filesystem_runtime_paths(self, payload: dict[str, object]) -> None:
        # Prefer the backend-neutral fs_ref; fall back to the legacy bare
        # `snapshot` key so checkpoints written before the fs_ref cutover
        # (and their retention) keep working. The provider accepts both
        # spellings.
        fs_ref = None
        filesystem = payload.get("filesystem", {})
        if isinstance(filesystem, dict):
            fs_ref = filesystem.get("fs_ref") or filesystem.get("snapshot")
        if fs_ref is None:
            status = payload.get("status", {})
            if isinstance(status, dict):
                metadata = status.get("metadata", {})
                if isinstance(metadata, dict):
                    fs_ref = metadata.get("fs_ref") or metadata.get("snapshot")
        if fs_ref is None:
            return
        destroy = self._destroy_filesystem_ref
        if destroy is None:
            logger.debug(
                "No destroy_filesystem_ref hook installed; skipping filesystem checkpoint cleanup for %s",
                fs_ref,
            )
            return
        try:
            destroy(str(fs_ref))
        except Exception:
            # Retention is best-effort; a failed snapshot destroy must not
            # wedge manifest/artifact deletion.
            logger.exception("destroy_filesystem_ref hook raised for %s", fs_ref)

    def _prune_empty_parents(self, start: Path, *, stop: Path) -> None:
        current = start
        while current != stop and current.exists():
            try:
                current.rmdir()
            except OSError:
                return
            current = current.parent
