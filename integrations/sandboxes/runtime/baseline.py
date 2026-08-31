from __future__ import annotations

import ipaddress
import json
import os
import tempfile
from pathlib import Path
from typing import Iterable


# Bump this whenever files copied into a shared image base, or other rootfs
# preparation semantics, change.  Runc incorporates it into every shared-base
# cache key so a daemon upgrade cannot silently reuse an older preparation.
SANDBOX_ROOTFS_PREPARATION_SCHEMA = "v2"


# Docker's ordinary, non-privileged capability baseline.  It is deliberately
# explicit: package managers need SETUID/SETGID and related ownership
# transitions, while privileged capabilities such as SYS_ADMIN stay absent.
SANDBOX_BASELINE_CAPABILITIES: tuple[str, ...] = (
    "CAP_AUDIT_WRITE",
    "CAP_CHOWN",
    "CAP_DAC_OVERRIDE",
    "CAP_FOWNER",
    "CAP_FSETID",
    "CAP_KILL",
    "CAP_MKNOD",
    "CAP_NET_BIND_SERVICE",
    "CAP_NET_RAW",
    "CAP_SETFCAP",
    "CAP_SETGID",
    "CAP_SETPCAP",
    "CAP_SETUID",
    "CAP_SYS_CHROOT",
)


DEFAULT_RESOLVER_CANDIDATES: tuple[Path, ...] = (
    # systemd-resolved's upstream resolver file.  Prefer it over the stub
    # symlink at /etc/resolv.conf, which commonly points at 127.0.0.53.
    Path("/run/systemd/resolve/resolv.conf"),
    # NetworkManager installations may expose an upstream-only file here.
    Path("/run/NetworkManager/resolv.conf"),
    Path("/etc/resolv.conf"),
)

_HOST_NETWORK_RESOLVER_CANDIDATES: tuple[Path, ...] = (
    # A host-network sandbox can reach the host's systemd-resolved stub and
    # should preserve its split-DNS/routing policy when one is configured.
    Path("/etc/resolv.conf"),
    Path("/run/systemd/resolve/resolv.conf"),
    Path("/run/NetworkManager/resolv.conf"),
)


class SandboxBaselineError(RuntimeError):
    """A required sandbox runtime baseline could not be materialized."""


def version_shared_rootfs_key(key: str) -> str:
    """Return a schema-qualified shared-rootfs key.

    Callers provide the content identity (usually an immutable image ID plus
    any baked-in CA state).  The runtime preparation schema is kept here so
    all local, daemon, Compose, ZFS, btrfs, and overlay paths invalidate old
    shared bases together.
    """

    raw = str(key).strip()
    if not raw:
        return raw
    prefix = f"{SANDBOX_ROOTFS_PREPARATION_SCHEMA}-"
    return raw if raw.startswith(prefix) else f"{prefix}{raw}"


def apply_sandbox_process_baseline(config: dict[str, object]) -> None:
    """Install Crab's ordinary non-privileged OCI process baseline."""

    process = config.get("process")
    if not isinstance(process, dict):
        raise SandboxBaselineError("OCI bundle has no process object")
    capabilities = {
        vector: list(SANDBOX_BASELINE_CAPABILITIES)
        for vector in ("bounding", "effective", "permitted", "inheritable", "ambient")
    }
    process["capabilities"] = capabilities
    # Root must be able to transition to image/package-manager users.  This is
    # deliberately false rather than inherited from `runc spec` by accident.
    process["noNewPrivileges"] = False


def apply_sandbox_bundle_baseline(bundle_dir: Path) -> None:
    """Apply the shared OCI process baseline to ``bundle_dir/config.json``."""

    config_path = Path(bundle_dir) / "config.json"
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SandboxBaselineError(f"OCI bundle config is missing: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise SandboxBaselineError(f"OCI bundle config is invalid JSON: {config_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SandboxBaselineError(f"OCI bundle config is not an object: {config_path}")
    apply_sandbox_process_baseline(payload)
    _atomic_write_text(config_path, json.dumps(payload, indent=2))


def materialize_resolver_config(
    bundle_dir: Path,
    *,
    candidates: Iterable[Path] = DEFAULT_RESOLVER_CANDIDATES,
    allow_loopback: bool = False,
) -> Path:
    """Write a non-loopback resolver file beside an OCI bundle.

    The returned host path is intended for a *post shared-clone* rootfs copy.
    Keeping the generated file beside the per-sandbox bundle both freezes the
    create-time input and prevents host DNS data from entering the persistent
    image/rootfs cache.
    """

    diagnostics: list[str] = []
    for raw_candidate in candidates:
        candidate = Path(raw_candidate)
        try:
            raw = candidate.read_text(encoding="utf-8")
        except FileNotFoundError:
            diagnostics.append(f"{candidate}: missing")
            continue
        except OSError as exc:
            diagnostics.append(f"{candidate}: {exc}")
            continue

        try:
            rendered, nameservers = _sanitize_resolver_text(
                raw, allow_loopback=allow_loopback
            )
        except ValueError as exc:
            diagnostics.append(f"{candidate}: {exc}")
            continue
        if not nameservers:
            diagnostics.append(f"{candidate}: no reachable non-loopback nameserver")
            continue

        destination = Path(bundle_dir) / "crab-resolv.conf"
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Resolver clients commonly run after dropping privileges (`_apt`,
        # package-manager helpers, application users). The mkstemp default is
        # 0600, so publish this runtime file with the conventional 0644 mode.
        _atomic_write_text(destination, rendered, mode=0o644)
        return destination

    details = "; ".join(diagnostics) if diagnostics else "no resolver candidates configured"
    raise SandboxBaselineError(
        "cannot create sandbox DNS configuration: no usable upstream resolver; "
        f"checked {details}"
    )


def add_dns_materialization(
    runtime_metadata: dict[str, object],
    *,
    bundle_dir: Path,
    candidates: Iterable[Path] | None = None,
    isolated: bool | None = None,
) -> Path:
    """Add the shared post-clone DNS directive to launch metadata."""

    if isolated is None:
        isolated = _bundle_uses_network_namespace(bundle_dir)
    selected_candidates = (
        tuple(candidates)
        if candidates is not None
        else (
            DEFAULT_RESOLVER_CANDIDATES
            if isolated
            else _HOST_NETWORK_RESOLVER_CANDIDATES
        )
    )
    resolver_path = materialize_resolver_config(
        bundle_dir,
        candidates=selected_candidates,
        allow_loopback=not isolated,
    )
    copy_paths = list(runtime_metadata.get("rootfs_post_clone_copy_paths", []))
    item = {
        "source": str(resolver_path),
        "destination": "/etc/resolv.conf",
        "replace": True,
    }
    if item not in copy_paths:
        copy_paths.append(item)
    runtime_metadata["rootfs_post_clone_copy_paths"] = copy_paths
    runtime_metadata["rootfs_preparation_schema"] = SANDBOX_ROOTFS_PREPARATION_SCHEMA
    return resolver_path


def _bundle_uses_network_namespace(bundle_dir: Path) -> bool:
    """Infer mode for Compose/benchmark callers from their completed spec.

    Missing or unreadable specs choose the isolated-safe policy: copying an
    upstream resolver may also work in host mode, while copying a loopback
    stub into an isolated namespace cannot work.
    """

    try:
        payload = json.loads(
            (Path(bundle_dir) / "config.json").read_text(encoding="utf-8")
        )
        namespaces = (payload.get("linux") or {}).get("namespaces", [])
    except (FileNotFoundError, OSError, json.JSONDecodeError, AttributeError):
        return True
    return any(
        isinstance(item, dict) and item.get("type") == "network"
        for item in namespaces
    )


def _sanitize_resolver_text(
    raw: str, *, allow_loopback: bool = False
) -> tuple[str, tuple[str, ...]]:
    output: list[str] = []
    nameservers: list[str] = []
    for raw_line in raw.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith(("#", ";")):
            output.append(raw_line)
            continue
        fields = stripped.split()
        if fields[0].lower() != "nameserver":
            output.append(raw_line)
            continue
        if len(fields) < 2:
            continue
        address = fields[1].split("%", 1)[0]
        try:
            parsed = ipaddress.ip_address(address)
        except ValueError as exc:
            raise ValueError(f"invalid nameserver address {fields[1]!r}") from exc
        if (parsed.is_loopback and not allow_loopback) or parsed.is_unspecified:
            continue
        nameservers.append(fields[1])
        output.append(f"nameserver {fields[1]}")

    if not nameservers:
        return "", ()
    rendered = "\n".join(output).rstrip() + "\n"
    return rendered, tuple(nameservers)


def _atomic_write_text(path: Path, text: str, *, mode: int | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            if mode is not None:
                os.fchmod(handle.fileno(), mode)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass


__all__ = [
    "DEFAULT_RESOLVER_CANDIDATES",
    "SANDBOX_BASELINE_CAPABILITIES",
    "SANDBOX_ROOTFS_PREPARATION_SCHEMA",
    "SandboxBaselineError",
    "add_dns_materialization",
    "apply_sandbox_bundle_baseline",
    "apply_sandbox_process_baseline",
    "materialize_resolver_config",
    "version_shared_rootfs_key",
]
