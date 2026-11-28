from __future__ import annotations

import copy
import fcntl
import fnmatch
import hashlib
import json
import os
import posixpath
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, IO

if TYPE_CHECKING:
    from crab.contracts import TelemetrySink


def _telemetry_sink(telemetry: TelemetrySink | None) -> TelemetrySink:
    if telemetry is not None:
        return telemetry
    # Keep this module importable while ``crab`` is still initializing. The
    # built-in iFlow agent imports the image helpers from crab.__init__.
    from crab.telemetry import NoopTelemetrySink

    return NoopTelemetrySink()


@dataclass(frozen=True)
class ImageRuntimeDefaults:
    environment: tuple[str, ...] = ()
    working_dir: str | None = None
    user: str | None = None
    entrypoint: tuple[str, ...] = ()
    command: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResolvedImage:
    reference: str
    normalized_reference: str
    registry: str
    image_id: str
    digest: str | None
    os: str
    architecture: str
    size_bytes: int
    pulled: bool
    cache_hit: bool


class DockerCommandClient:
    """Small injectable seam around Docker CLI process execution."""

    def run(
        self,
        args: list[str],
        *,
        check: bool = False,
        capture_output: bool = False,
        text: bool = False,
        stdout: IO[bytes] | int | None = None,
        stderr: int | None = None,
        timeout: float | None = None,
    ) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["docker", *args],
            check=check,
            capture_output=capture_output,
            text=text,
            stdout=stdout,
            stderr=stderr,
            timeout=timeout,
        )


def normalize_public_image_reference(reference: str) -> tuple[str, str]:
    from crab.errors import ImageReferenceError

    raw = str(reference).strip()
    if not raw or any(ch.isspace() for ch in raw) or "://" in raw:
        raise ImageReferenceError(reference, f"malformed image reference: {reference!r}")
    if raw.count("@") > 1:
        raise ImageReferenceError(reference, f"malformed image reference: {reference!r}")
    name_part, separator, digest_part = raw.partition("@")
    if separator and not re.fullmatch(r"sha256:[0-9a-fA-F]{64}", digest_part):
        raise ImageReferenceError(
            reference,
            "only sha256 digest-pinned image references are supported",
        )
    if not re.fullmatch(
        r"[A-Za-z0-9]+(?:[._-][A-Za-z0-9]+)*(?:/[A-Za-z0-9]+(?:[._-][A-Za-z0-9]+)*)*(?::[A-Za-z0-9_][A-Za-z0-9_.-]{0,127})?",
        name_part,
    ):
        raise ImageReferenceError(reference, f"malformed image reference: {reference!r}")

    repository_name = name_part
    last_component = repository_name.rsplit("/", 1)[-1]
    if ":" in last_component:
        repository_name = repository_name.rsplit(":", 1)[0]
    if any(character.isupper() for character in repository_name):
        raise ImageReferenceError(
            reference, "Docker repository and registry names must be lowercase"
        )

    parts = name_part.split("/")
    first = parts[0]
    qualified = len(parts) > 1 and (
        "." in first or ":" in first or first == "localhost"
    )
    if qualified:
        registry = first.lower()
        repository = "/".join(parts[1:])
    else:
        registry = "docker.io"
        repository = name_part
    if registry in {"index.docker.io", "registry-1.docker.io"}:
        registry = "docker.io"
    if registry == "docker.io" and "/" not in repository:
        repository = f"library/{repository}"
    normalized = f"{registry}/{repository}"
    if separator:
        normalized = f"{normalized}@{digest_part.lower()}"
    return normalized, registry


def resolve_image(
    *,
    reference: str,
    cache_root: Path,
    pull_policy: str = "if-not-present",
    allowed_registries: tuple[str, ...] = ("docker.io",),
    allowed_references: tuple[str, ...] = (),
    pull_timeout_seconds: float = 600.0,
    max_image_bytes: int = 8 * 1024 * 1024 * 1024,
    min_free_bytes: int = 2 * 1024 * 1024 * 1024,
    telemetry: TelemetrySink | None = None,
    docker: DockerCommandClient | None = None,
) -> ResolvedImage:
    """Resolve or pull one policy-approved Linux/amd64 public image."""

    from crab.errors import (
        ImageCompatibilityError,
        ImageNotFoundError,
        ImagePlatformError,
        ImagePolicyError,
        ImagePullError,
        ImagePullTimeoutError,
        ImageReferenceError,
        ImageTooLargeError,
    )

    sink = _telemetry_sink(telemetry)
    client = docker or DockerCommandClient()
    normalized, registry = normalize_public_image_reference(reference)
    allowed_registry_set = {
        "docker.io" if item in {"index.docker.io", "registry-1.docker.io"} else item.lower()
        for item in allowed_registries
    }
    if registry not in allowed_registry_set:
        raise ImagePolicyError(
            reference,
            f"image registry {registry!r} is not allowed; allowed registries: "
            f"{', '.join(sorted(allowed_registry_set))}",
        )
    if allowed_references and not any(
        fnmatch.fnmatchcase(normalized, pattern)
        or fnmatch.fnmatchcase(reference, pattern)
        for pattern in allowed_references
    ):
        raise ImagePolicyError(
            reference,
            f"image {reference!r} is not included in images.allowed_references",
        )
    if pull_policy not in {"if-not-present", "never"}:
        raise ImagePolicyError(
            reference,
            f"unsupported image pull policy: {pull_policy!r}",
        )

    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    _require_free_disk(
        reference,
        cache_root,
        minimum_free_bytes=int(min_free_bytes),
        required_bytes=0,
    )
    lock_root = cache_root / ".pull-locks"
    lock_root.mkdir(parents=True, exist_ok=True)
    lock_name = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    lock_path = lock_root / f"{lock_name}.lock"
    started = time.perf_counter()
    with lock_path.open("w", encoding="utf-8") as lock_fh:
        wait_started = time.perf_counter()
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
        sink.emit_metric(
            "image.pull_lock_wait_ms",
            (time.perf_counter() - wait_started) * 1000.0,
            {"reference": normalized},
        )
        inspected = _docker_inspect(client, normalized)
        cache_hit = inspected is not None
        pulled = False
        if inspected is None:
            if pull_policy == "never":
                raise ImageNotFoundError(
                    reference,
                    f"image {reference!r} is not present locally and pull policy is 'never'",
                )
            pull_started = time.perf_counter()
            try:
                result = client.run(
                    ["pull", "--platform", "linux/amd64", normalized],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=float(pull_timeout_seconds),
                )
            except subprocess.TimeoutExpired as exc:
                sink.emit_event(
                    "image.pull_failed",
                    {"reference": normalized, "failure": "timeout"},
                )
                raise ImagePullTimeoutError(
                    reference,
                    f"pulling image {reference!r} exceeded {pull_timeout_seconds:.0f}s",
                ) from exc
            except OSError as exc:
                raise ImagePullError(
                    reference,
                    f"docker pull could not start for {reference!r}: {exc}",
                ) from exc
            if result.returncode != 0:
                error = _classify_pull_error(
                    reference,
                    f"{result.stdout or ''}\n{result.stderr or ''}",
                )
                sink.emit_event(
                    "image.pull_failed",
                    {
                        "reference": normalized,
                        "failure": error.error_type,
                        "duration_ms": (time.perf_counter() - pull_started) * 1000.0,
                    },
                )
                raise error
            pulled = True
            inspected = _docker_inspect(client, normalized)
            if inspected is None:
                raise ImagePullError(
                    reference,
                    f"docker pull reported success but {normalized!r} cannot be inspected",
                )

        assert inspected is not None
        image_id = str(inspected.get("Id") or "").removeprefix("sha256:").lower()
        image_os = str(inspected.get("Os") or "").lower()
        architecture = str(inspected.get("Architecture") or "").lower()
        try:
            size_bytes = int(inspected.get("Size") or 0)
        except (TypeError, ValueError) as exc:
            raise ImageCompatibilityError(
                reference,
                f"image {reference!r} reports a malformed size: {inspected.get('Size')!r}",
            ) from exc
        if not re.fullmatch(r"[0-9a-f]{64}", image_id):
            raise ImageCompatibilityError(
                reference,
                f"image {reference!r} has a malformed immutable image ID",
            )
        if image_os != "linux" or architecture not in {"amd64", "x86_64"}:
            if pulled:
                _remove_pulled_reference(client, normalized)
            raise ImagePlatformError(
                reference,
                f"image {reference!r} is {image_os or 'unknown'}/{architecture or 'unknown'}; "
                "Crab currently supports linux/amd64 only",
            )
        if size_bytes <= 0:
            raise ImagePullError(reference, f"image {reference!r} reports an invalid size")
        if size_bytes > int(max_image_bytes):
            if pulled:
                _remove_pulled_reference(client, normalized)
            raise ImageTooLargeError(
                reference,
                f"image {reference!r} is {size_bytes} bytes, above the configured "
                f"limit of {int(max_image_bytes)} bytes",
            )
        _require_free_disk(
            reference,
            cache_root,
            minimum_free_bytes=int(min_free_bytes),
            required_bytes=size_bytes,
        )
        repo_digests = inspected.get("RepoDigests") or []
        digest = None
        requested_repository = _normalized_repository(normalized)
        fallback_digest = None
        if isinstance(repo_digests, list):
            for value in repo_digests:
                raw_name, separator, raw_digest = str(value).partition("@")
                if raw_digest.startswith("sha256:"):
                    fallback_digest = fallback_digest or raw_digest
                    if separator:
                        try:
                            repo_normalized, _ = normalize_public_image_reference(
                                raw_name
                            )
                        except ImageReferenceError:
                            continue
                        if _normalized_repository(repo_normalized) == requested_repository:
                            digest = raw_digest
                            break
        if "@sha256:" in normalized:
            digest = normalized.partition("@")[2]
        else:
            digest = digest or fallback_digest

        duration_ms = (time.perf_counter() - started) * 1000.0
        attributes = {
            "reference": normalized,
            "resolved_digest": digest,
            "image_id": image_id,
            "cache_hit": cache_hit,
            "pulled": pulled,
            "bytes": size_bytes,
        }
        sink.emit_metric("image.pull_ms", duration_ms, attributes)
        sink.emit_event("image.pull", {**attributes, "duration_ms": duration_ms})
        return ResolvedImage(
            reference=reference,
            normalized_reference=normalized,
            registry=registry,
            image_id=image_id,
            digest=digest,
            os=image_os,
            architecture="amd64" if architecture == "x86_64" else architecture,
            size_bytes=size_bytes,
            pulled=pulled,
            cache_hit=cache_hit,
        )


def _docker_inspect(
    client: DockerCommandClient, reference: str
) -> dict[str, Any] | None:
    from crab.errors import ImagePullError

    try:
        result = client.run(
            ["image", "inspect", reference],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise ImagePullError(
            reference, f"docker image inspect could not start for {reference!r}: {exc}"
        ) from exc
    if result.returncode != 0:
        output = f"{result.stdout or ''}\n{result.stderr or ''}".strip()
        lowered = output.lower()
        if "no such image" in lowered or lowered == "not found":
            return None
        raise ImagePullError(
            reference,
            f"docker image inspect failed for {reference!r}: {output[-1000:]}",
        )
    try:
        payload = json.loads(result.stdout or "[]")
    except json.JSONDecodeError as exc:
        raise ImagePullError(
            reference, f"docker returned invalid inspect JSON for {reference!r}: {exc}"
        ) from exc
    if not isinstance(payload, list) or not payload or not isinstance(payload[0], dict):
        raise ImagePullError(reference, f"docker returned no inspect record for {reference!r}")
    return payload[0]


def _normalized_repository(normalized_reference: str) -> str:
    """Strip a tag or digest from one already-normalized image reference."""

    repository = normalized_reference.partition("@")[0]
    slash = repository.rfind("/")
    colon = repository.rfind(":")
    if colon > slash:
        repository = repository[:colon]
    return repository


def _classify_pull_error(reference: str, output: str):
    from crab.errors import (
        ImageAuthenticationError,
        ImageInsufficientDiskError,
        ImageNotFoundError,
        ImagePlatformError,
        ImagePullError,
        ImageRateLimitError,
    )

    lowered = output.lower()
    if "no space left on device" in lowered or "insufficient" in lowered and "space" in lowered:
        return ImageInsufficientDiskError(
            reference, f"insufficient disk while pulling image {reference!r}"
        )
    if "toomanyrequests" in lowered or "rate limit" in lowered:
        return ImageRateLimitError(
            reference, f"Docker Hub rate limited the pull for {reference!r}"
        )
    if any(
        marker in lowered
        for marker in (
            "authentication required",
            "unauthorized:",
            "requested access to the resource is denied",
        )
    ):
        return ImageAuthenticationError(
            reference,
            f"image {reference!r} requires registry authentication; private credentials are not supported",
        )
    if any(
        marker in lowered
        for marker in ("manifest unknown", "not found", "pull access denied")
    ) or (
        "failed to resolve reference" in lowered
        and "/manifests/" in lowered
        and "403 forbidden" in lowered
    ):
        # Some anonymous Docker Hub mirrors intentionally collapse an absent
        # repository/tag and an inaccessible one into a manifest HEAD 403.
        # The initial Crab contract does not accept private credentials, so
        # expose that non-enumerating response as public-image not-found just
        # like Docker's standard `pull access denied` wording.
        return ImageNotFoundError(
            reference,
            f"public image {reference!r} was not found on Docker Hub",
        )
    if "no matching manifest" in lowered or "does not match the specified platform" in lowered:
        return ImagePlatformError(
            reference,
            f"image {reference!r} has no linux/amd64 manifest",
        )
    return ImagePullError(
        reference,
        f"failed to pull public image {reference!r}: {output.strip()[-1000:]}",
    )


def _remove_pulled_reference(client: DockerCommandClient, reference: str) -> None:
    client.run(
        ["image", "rm", reference],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _require_free_disk(
    reference: str,
    path: Path,
    *,
    minimum_free_bytes: int,
    required_bytes: int,
) -> None:
    from crab.errors import ImageInsufficientDiskError

    free = shutil.disk_usage(path).free
    required_free = int(minimum_free_bytes) + int(required_bytes)
    if free < required_free:
        raise ImageInsufficientDiskError(
            reference,
            f"insufficient disk for image {reference!r}: free={free} bytes, "
            f"required={required_free} bytes (including configured reserve)",
        )


def container_rootfs_tar_filter(member: tarfile.TarInfo, destination: str | Path) -> tarfile.TarInfo | None:
    """Reject path escapes while preserving container rootfs metadata.

    Docker root filesystems legitimately contain absolute link targets because
    those links are interpreted inside the container root. Python 3.14's
    default ``data`` filter rejects them and also strips ownership, sticky
    bits, and group/world write bits. Those metadata are part of a container
    image's runtime semantics (for example, ``/tmp`` must remain ``01777``).
    Rewrite absolute link targets inside the rootfs, then use Python's ``tar``
    filter to reject absolute/member traversal without altering OCI metadata.
    """

    if member.isdev():
        # runc supplies a controlled /dev mount. Materializing device nodes or
        # FIFOs from an untrusted public image into the host-side cache adds no
        # supported semantics and creates an avoidable host resource boundary.
        raise tarfile.SpecialFileError(member)
    if member.issym() and posixpath.isabs(member.linkname):
        member = copy.copy(member)
        link_target = member.linkname.lstrip("/")
        link_parent = posixpath.dirname(member.name.lstrip("/")) or "."
        member.linkname = posixpath.relpath(link_target, start=link_parent)
    elif member.islnk() and posixpath.isabs(member.linkname):
        member = copy.copy(member)
        member.linkname = member.linkname.lstrip("/")
    filtered = tarfile.tar_filter(member, destination)
    if filtered is None:
        return None
    filtered = copy.copy(filtered)
    filtered.mode = member.mode
    filtered.uid = member.uid
    filtered.gid = member.gid
    filtered.uname = member.uname
    filtered.gname = member.gname
    return filtered


def docker_tag_component(raw: str) -> str:
    normalized = re.sub(r"[^a-z0-9_.-]+", "-", raw.lower()).strip("-.")
    return normalized or "image"


def image_exists(*, tag: str) -> bool:
    result = subprocess.run(
        ["docker", "image", "inspect", tag],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def inspect_image_id(*, tag: str, telemetry: TelemetrySink | None = None) -> str:
    sink = _telemetry_sink(telemetry)
    started = time.perf_counter()
    raw_output = subprocess.run(
        ["docker", "image", "inspect", tag, "--format", "{{.Id}}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    duration_ms = (time.perf_counter() - started) * 1000.0
    image_id = raw_output.replace("sha256:", "")
    sink.emit_metric(
        "image.inspect_ms",
        duration_ms,
        {"tag": tag, "image_id": image_id},
    )
    sink.emit_event("image.inspect", {"tag": tag, "image_id": image_id})
    return image_id


def inspect_image_runtime_defaults(
    *,
    tag: str,
    cache_root: Path | None = None,
    telemetry: TelemetrySink | None = None,
    image_id: str | None = None,
    docker: DockerCommandClient | None = None,
) -> ImageRuntimeDefaults:
    from crab.errors import ImageCompatibilityError, ImagePullError

    sink = _telemetry_sink(telemetry)
    client = docker or DockerCommandClient()
    image_id = image_id or inspect_image_id(tag=tag, telemetry=sink)
    cache_dir = None if cache_root is None else cache_root / image_id
    cache_path = None if cache_dir is None else cache_dir / "runtime_defaults.json"
    if cache_path is not None and cache_path.is_file():
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("defaults cache is not an object")
            defaults = _runtime_defaults_from_cache(payload)
        except (OSError, ValueError, json.JSONDecodeError):
            try:
                cache_path.unlink()
            except OSError:
                pass
        else:
            sink.emit_event("image.defaults_cache_hit", {"tag": tag, "image_id": image_id, "path": str(cache_path)})
            return defaults

    started = time.perf_counter()
    try:
        inspect_result = client.run(
            ["image", "inspect", tag, "--format", "{{json .Config}}"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise ImagePullError(
            tag, f"docker runtime-default inspection could not start for {tag!r}: {exc}"
        ) from exc
    if inspect_result.returncode != 0:
        raise ImagePullError(
            tag,
            f"docker could not inspect runtime defaults for {tag!r}: "
            f"{str(inspect_result.stderr or inspect_result.stdout).strip()[-1000:]}",
        )
    raw_output = str(inspect_result.stdout or "").strip()
    try:
        config = json.loads(raw_output) if raw_output else {}
    except json.JSONDecodeError as exc:
        raise ImageCompatibilityError(
            tag, f"image {tag!r} has malformed Docker config JSON: {exc}"
        ) from exc
    if not isinstance(config, dict):
        raise ImageCompatibilityError(
            tag, f"image {tag!r} has a non-object Docker config"
        )

    def _string_list(key: str) -> tuple[str, ...]:
        value = config.get(key)
        if value is None:
            return ()
        if not isinstance(value, list) or not all(
            isinstance(item, str) for item in value
        ):
            raise ImageCompatibilityError(
                tag,
                f"image {tag!r} has unsupported Docker config {key}: {value!r}",
            )
        return tuple(value)

    working_dir = config.get("WorkingDir")
    if working_dir is not None and not isinstance(working_dir, str):
        raise ImageCompatibilityError(
            tag,
            f"image {tag!r} has unsupported Docker WorkingDir: {working_dir!r}",
        )
    user = config.get("User")
    if user is not None and not isinstance(user, str):
        raise ImageCompatibilityError(
            tag, f"image {tag!r} has unsupported Docker User: {user!r}"
        )

    defaults = ImageRuntimeDefaults(
        environment=_string_list("Env"),
        working_dir=working_dir or None,
        user=user or None,
        entrypoint=_string_list("Entrypoint"),
        command=_string_list("Cmd"),
    )
    if cache_path is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=".runtime-defaults.", dir=cache_dir
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(asdict(defaults), handle, sort_keys=True, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, cache_path)
        finally:
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass
    duration_ms = (time.perf_counter() - started) * 1000.0
    sink.emit_metric(
        "image.inspect_defaults_ms",
        duration_ms,
        {"tag": tag, "image_id": image_id, "cache_hit": False},
    )
    sink.emit_event("image.defaults_cache_miss", {"tag": tag, "image_id": image_id})
    return defaults


def _runtime_defaults_from_cache(payload: dict[str, Any]) -> ImageRuntimeDefaults:
    def string_tuple(key: str) -> tuple[str, ...]:
        value = payload.get(key, [])
        if not isinstance(value, list) or not all(
            isinstance(item, str) for item in value
        ):
            raise ValueError(f"invalid cached image default {key}")
        return tuple(value)

    working_dir = payload.get("working_dir")
    user = payload.get("user")
    if working_dir is not None and not isinstance(working_dir, str):
        raise ValueError("invalid cached image working_dir")
    if user is not None and not isinstance(user, str):
        raise ValueError("invalid cached image user")
    return ImageRuntimeDefaults(
        environment=string_tuple("environment"),
        working_dir=working_dir or None,
        user=user or None,
        entrypoint=string_tuple("entrypoint"),
        command=string_tuple("command"),
    )


def build_image(
    *,
    tag: str,
    build_context: Path,
    dockerfile_path: Path,
    telemetry: TelemetrySink | None = None,
    skip_if_exists: bool = True,
) -> None:
    sink = _telemetry_sink(telemetry)
    if skip_if_exists and image_exists(tag=tag):
        sink.emit_event("image.build_cache_hit", {"tag": tag, "build_context": str(build_context)})
        sink.emit_metric("image.build_ms", 0.0, {"tag": tag, "cache_hit": True})
        return
    started = time.perf_counter()
    subprocess.run(
        ["docker", "build", "-t", tag, "-f", str(dockerfile_path), str(build_context)],
        check=True,
    )
    duration_ms = (time.perf_counter() - started) * 1000.0
    sink.emit_metric("image.build_ms", duration_ms, {"tag": tag, "cache_hit": False})
    sink.emit_event("image.build", {"tag": tag, "build_context": str(build_context), "dockerfile_path": str(dockerfile_path)})


def export_image_rootfs(
    *,
    tag: str,
    output_dir: Path,
    cache_root: Path | None = None,
    telemetry: TelemetrySink | None = None,
    image_id: str | None = None,
    image_size_bytes: int | None = None,
    max_image_bytes: int = 8 * 1024 * 1024 * 1024,
    cache_max_bytes: int = 20 * 1024 * 1024 * 1024,
    min_free_bytes: int = 2 * 1024 * 1024 * 1024,
    cache_retention_seconds: float = 30 * 24 * 60 * 60,
    docker: DockerCommandClient | None = None,
) -> Path:
    from crab.errors import (
        ImageCompatibilityError,
        ImageInsufficientDiskError,
        ImagePullError,
        ImageTooLargeError,
    )

    sink = _telemetry_sink(telemetry)
    client = docker or DockerCommandClient()
    image_id = image_id or inspect_image_id(tag=tag, telemetry=sink)
    resolved_output_dir = output_dir if cache_root is None else cache_root / image_id
    rootfs_dir = resolved_output_dir / "rootfs"
    backup_dir = resolved_output_dir / "rootfs.previous"
    lock_path = resolved_output_dir / ".export.lock"
    export_container_name = (
        f"crab-export-{image_id[:12]}-"
        f"{hashlib.sha256(str(resolved_output_dir.resolve()).encode('utf-8')).hexdigest()[:16]}"
    )
    export_container_marker = resolved_output_dir / ".export-container"
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    lock_wait_started = time.perf_counter()
    with lock_path.open("w", encoding="utf-8") as lock_fh:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
        lock_wait_ms = (time.perf_counter() - lock_wait_started) * 1000.0
        sink.emit_metric("image.cache_lock_wait_ms", lock_wait_ms, {"tag": tag, "image_id": image_id})
        # Recover or discard artifacts left by a process that died between the
        # two atomic renames below.  The export lock makes these paths stale:
        # no live exporter for this image can still own them.
        if not rootfs_dir.exists() and backup_dir.exists():
            backup_dir.replace(rootfs_dir)
        for stale_dir in resolved_output_dir.glob("rootfs-export-*"):
            _remove_cache_path(stale_dir)
        if export_container_marker.exists():
            cleanup_error = _remove_export_container(
                client, export_container_name
            )
            if cleanup_error is not None:
                raise ImagePullError(
                    tag,
                    f"cannot clean stale image-export container "
                    f"{export_container_name!r}: {cleanup_error}",
                )
            export_container_marker.unlink(missing_ok=True)

        if rootfs_dir.exists():
            try:
                if not rootfs_dir.is_dir() or not any(rootfs_dir.iterdir()):
                    raise ImageCompatibilityError(
                        tag, f"cached rootfs for {tag!r} is empty or malformed"
                    )
                _validate_image_rootfs(tag, rootfs_dir)
            except ImageCompatibilityError:
                # A previously interrupted/corrupted published directory is
                # not a cache hit. Re-export once under the same lock; a truly
                # incompatible image will fail the fresh validation below.
                _remove_cache_path(rootfs_dir)
                if backup_dir.exists():
                    backup_dir.replace(rootfs_dir)
                    try:
                        _validate_image_rootfs(tag, rootfs_dir)
                    except ImageCompatibilityError:
                        _remove_cache_path(rootfs_dir)
                    else:
                        _touch_cache_access(resolved_output_dir)
                        sink.emit_event(
                            "image.export_cache_recovered",
                            {"tag": tag, "image_id": image_id},
                        )
                        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
                        return rootfs_dir
                sink.emit_event(
                    "image.export_cache_invalid",
                    {"tag": tag, "image_id": image_id},
                )
            else:
                if backup_dir.exists():
                    _remove_cache_path(backup_dir)
                _touch_cache_access(resolved_output_dir)
                sink.emit_event("image.export_cache_hit", {"tag": tag, "image_id": image_id, "rootfs_dir": str(rootfs_dir)})
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
                return rootfs_dir

        if backup_dir.exists():
            _remove_cache_path(backup_dir)

        sink.emit_event("image.export_cache_miss", {"tag": tag, "image_id": image_id, "rootfs_dir": str(rootfs_dir)})
        if cache_root is not None:
            _ensure_cache_capacity(
                reference=tag,
                cache_root=Path(cache_root),
                protected_image_id=image_id,
                incoming_bytes=int(image_size_bytes or 0),
                max_cache_bytes=int(cache_max_bytes),
                min_free_bytes=int(min_free_bytes),
                retention_seconds=float(cache_retention_seconds),
            )
        export_container_marker.write_text(
            export_container_name + "\n", encoding="utf-8"
        )
        create_result = client.run(
            [
                "create",
                "--name",
                export_container_name,
                "--entrypoint",
                "/bin/sh",
                tag,
                "-c",
                "true",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if create_result.returncode != 0:
            cleanup_error = _remove_export_container(client, export_container_name)
            if cleanup_error is None:
                export_container_marker.unlink(missing_ok=True)
            raise ImagePullError(
                tag,
                f"docker could not create an export container for {tag!r}: "
                f"{str(create_result.stderr or create_result.stdout).strip()[-1000:]}"
                + (
                    ""
                    if cleanup_error is None
                    else f"; cleanup of {export_container_name!r} also failed: {cleanup_error}"
                ),
            )
        container_id = str(create_result.stdout).strip()
        if not container_id:
            cleanup_error = _remove_export_container(client, export_container_name)
            if cleanup_error is None:
                export_container_marker.unlink(missing_ok=True)
            raise ImagePullError(
                tag,
                f"docker create returned no container ID for {tag!r}"
                + (
                    ""
                    if cleanup_error is None
                    else f"; cleanup of {export_container_name!r} also failed: {cleanup_error}"
                ),
            )
        staging_dir: Path | None = None
        started = time.perf_counter()
        try:
            staging_dir = Path(
                tempfile.mkdtemp(prefix="rootfs-export-", dir=resolved_output_dir)
            )
            tar_path = staging_dir / "rootfs.tar"
            staging_rootfs_dir = staging_dir / "rootfs"
            with tar_path.open("wb") as fh:
                export_result = client.run(
                    ["export", container_id],
                    check=False,
                    stdout=fh,
                    stderr=subprocess.PIPE,
                )
            if export_result.returncode != 0:
                export_error = _output_string(export_result.stderr)
                if "no space left on device" in export_error.lower():
                    raise ImageInsufficientDiskError(
                        tag, f"insufficient disk while exporting image {tag!r}"
                    )
                raise ImagePullError(
                    tag,
                    f"docker export failed for {tag!r}: {export_error.strip()[-1000:]}",
                )
            staging_rootfs_dir.mkdir(parents=True, exist_ok=True)
            with tarfile.open(tar_path) as tf:
                members = tf.getmembers()
                expanded_bytes = sum(
                    max(0, int(member.size)) for member in members if member.isfile()
                )
                if expanded_bytes > int(max_image_bytes):
                    raise ImageTooLargeError(
                        tag,
                        f"expanded rootfs for {tag!r} is {expanded_bytes} bytes, above "
                        f"the configured limit of {int(max_image_bytes)} bytes",
                    )
                _require_free_disk(
                    tag,
                    staging_dir,
                    minimum_free_bytes=int(min_free_bytes),
                    required_bytes=expanded_bytes,
                )
                try:
                    tf.extractall(
                        staging_rootfs_dir,
                        members=members,
                        filter=container_rootfs_tar_filter,
                    )
                except tarfile.FilterError as exc:
                    raise ImageCompatibilityError(
                        tag,
                        f"image {tag!r} contains an unsafe rootfs archive entry: {exc}",
                    ) from exc
            # The exported tar is transient and can nearly double the apparent
            # cache footprint.  Remove it before enforcing the final expanded
            # rootfs budget, then account for all concurrently staged exports
            # while holding the global cache lock.
            tar_path.unlink()
            _validate_image_rootfs(tag, staging_rootfs_dir)
            if cache_root is not None:
                _ensure_cache_capacity(
                    reference=tag,
                    cache_root=Path(cache_root),
                    protected_image_id=image_id,
                    incoming_bytes=0,
                    max_cache_bytes=int(cache_max_bytes),
                    min_free_bytes=int(min_free_bytes),
                    retention_seconds=float(cache_retention_seconds),
                )
            try:
                if backup_dir.exists():
                    _remove_cache_path(backup_dir)
                if rootfs_dir.exists():
                    rootfs_dir.replace(backup_dir)
                staging_rootfs_dir.replace(rootfs_dir)
                _touch_cache_access(resolved_output_dir)
            except Exception:
                if backup_dir.exists() and not rootfs_dir.exists():
                    backup_dir.replace(rootfs_dir)
                raise
            finally:
                if backup_dir.exists():
                    _remove_cache_path(backup_dir)
        except OSError as exc:
            if getattr(exc, "errno", None) == 28:
                raise ImageInsufficientDiskError(
                    tag, f"insufficient disk while materializing image {tag!r}"
                ) from exc
            raise
        finally:
            cleanup_error = _remove_export_container(client, export_container_name)
            if cleanup_error is None:
                export_container_marker.unlink(missing_ok=True)
            if staging_dir is not None:
                shutil.rmtree(staging_dir, ignore_errors=True)
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
            if cleanup_error is not None:
                active_error = sys.exc_info()[1]
                detail = (
                    ""
                    if active_error is None
                    else f" after {type(active_error).__name__}: {active_error}"
                )
                raise ImagePullError(
                    tag,
                    f"image export{detail}; cleanup of Docker container "
                    f"{export_container_name!r} also failed: {cleanup_error}",
                ) from active_error
        duration_ms = (time.perf_counter() - started) * 1000.0
        sink.emit_metric("image.export_ms", duration_ms, {"tag": tag, "image_id": image_id, "cache_hit": False})
    return rootfs_dir


def _output_string(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _remove_export_container(
    client: DockerCommandClient, container_name: str
) -> str | None:
    """Remove one deterministic Crab export container, reporting failures."""

    try:
        result = client.run(
            ["rm", "-f", container_name],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        return str(exc)
    if result.returncode == 0:
        return None
    output = _output_string(result.stderr or result.stdout).strip()
    if "no such container" in output.lower():
        return None
    return output[-1000:] or f"docker rm exited {result.returncode}"


def _validate_image_rootfs(reference: str, rootfs_dir: Path) -> None:
    from crab.errors import ImageCompatibilityError

    shell = rootfs_dir / "bin" / "sh"
    if not shell.exists():
        raise ImageCompatibilityError(
            reference,
            f"image {reference!r} is incompatible: /bin/sh is required",
        )
    sleep_candidates = (
        rootfs_dir / "bin" / "sleep",
        rootfs_dir / "usr" / "bin" / "sleep",
    )
    if not any(path.exists() for path in sleep_candidates):
        raise ImageCompatibilityError(
            reference,
            f"image {reference!r} is incompatible: a sleep executable is required "
            "for Crab's long-running init",
        )


def _touch_cache_access(cache_dir: Path) -> None:
    marker = Path(cache_dir) / ".last_access"
    try:
        marker.touch(exist_ok=True)
    except OSError:
        pass


def _remove_cache_path(path: Path) -> None:
    """Remove one known cache artifact whether it is a file or directory."""

    try:
        if path.is_symlink() or not path.is_dir():
            path.unlink()
        else:
            shutil.rmtree(path)
    except FileNotFoundError:
        pass


def _ensure_cache_capacity(
    *,
    reference: str,
    cache_root: Path,
    protected_image_id: str,
    incoming_bytes: int,
    max_cache_bytes: int,
    min_free_bytes: int,
    retention_seconds: float,
) -> None:
    """Evict inactive exported-rootfs entries under one global GC lock."""

    from crab.errors import ImageInsufficientDiskError

    cache_root.mkdir(parents=True, exist_ok=True)
    gc_lock_path = cache_root / ".cache-gc.lock"
    with gc_lock_path.open("w", encoding="utf-8") as gc_fh:
        fcntl.flock(gc_fh.fileno(), fcntl.LOCK_EX)
        now = time.time()
        candidates: list[tuple[float, int, Path]] = []
        current_bytes = 0
        for entry in cache_root.iterdir():
            if not entry.is_dir() or not re.fullmatch(r"[0-9a-f]{32,64}", entry.name):
                continue
            size = _directory_size(entry)
            current_bytes += size
            marker = entry / ".last_access"
            try:
                accessed = marker.stat().st_mtime if marker.exists() else entry.stat().st_mtime
            except OSError:
                accessed = now
            if entry.name != protected_image_id:
                candidates.append((accessed, size, entry))
        candidates.sort(key=lambda item: item[0])

        for accessed, size, entry in candidates:
            over_retention = now - accessed >= max(0.0, retention_seconds)
            over_budget = current_bytes + max(0, incoming_bytes) > max_cache_bytes
            free_shortfall = (
                shutil.disk_usage(cache_root).free
                < min_free_bytes + max(0, incoming_bytes)
            )
            if not (over_retention or over_budget or free_shortfall):
                continue
            if not _try_remove_cache_entry(entry):
                continue
            current_bytes = max(0, current_bytes - size)

        free = shutil.disk_usage(cache_root).free
        if current_bytes + max(0, incoming_bytes) > max_cache_bytes:
            raise ImageInsufficientDiskError(
                reference,
                f"image cache budget exhausted: used={current_bytes} incoming={incoming_bytes} "
                f"limit={max_cache_bytes} bytes",
            )
        if free < min_free_bytes + max(0, incoming_bytes):
            raise ImageInsufficientDiskError(
                reference,
                f"image cache disk reserve would be crossed: free={free} "
                f"incoming={incoming_bytes} reserve={min_free_bytes} bytes",
            )


def _try_remove_cache_entry(entry: Path) -> bool:
    lock_path = entry / ".export.lock"
    try:
        lock_fh = lock_path.open("a+", encoding="utf-8")
    except OSError:
        return False
    try:
        try:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        shutil.rmtree(entry, ignore_errors=False)
        return True
    except OSError:
        return False
    finally:
        try:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        lock_fh.close()


def _directory_size(root: Path) -> int:
    total = 0
    try:
        entries = list(root.rglob("*"))
    except OSError:
        return 0
    for entry in entries:
        try:
            if entry.is_file() and not entry.is_symlink():
                total += entry.stat().st_size
        except OSError:
            continue
    return total
