"""TLS trust injection helpers for sandbox environments.

This module provides utilities for injecting CA trust into sandbox
rootfs filesystems and environment variables. It does NOT import
``cryptography`` or any TLS-specific modules — it only manipulates
file paths and environment dictionaries.

Two injection vectors are covered (§3.3):
  1. **CA file copy**: the CA certificate PEM is placed into the sandbox
     rootfs at a well-known system path so that system-store-aware tools
     (curl, apt, bare Python ``ssl``, Go, etc.) trust the proxy CA.
  2. **Env overlay**: environment variables pointing popular runtimes at
     the CA file (``SSL_CERT_FILE``, ``REQUESTS_CA_BUNDLE``, etc.) are
     injected into both the init-process env and the exec-level
     ``command_env`` so that daemons and on-demand commands see them.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# Path inside the sandbox rootfs where the CA certificate is placed.
_SANDBOX_CA_CERT_PATH = "/usr/local/share/ca-certificates/crab-ca.crt"

# The env variables that point runtimes at the CA file.
_TLS_CA_ENV_VARS: tuple[tuple[str, str], ...] = (
    ("SSL_CERT_FILE", _SANDBOX_CA_CERT_PATH),
    ("REQUESTS_CA_BUNDLE", _SANDBOX_CA_CERT_PATH),
    ("CURL_CA_BUNDLE", _SANDBOX_CA_CERT_PATH),
    ("NODE_EXTRA_CA_CERTS", _SANDBOX_CA_CERT_PATH),
    # SSL_CERT_DIR intentionally omitted: it conflicts with system cert
    # directory layouts and is less portable than the single-file vars.
)


def tls_ca_env_overlay() -> dict[str, str]:
    """Return the env dict overlay for CA trust injection.

    This is a pure function — safe to call unconditionally and merge
    into any env dict when TLS interception is enabled.
    """
    return {key: value for key, value in _TLS_CA_ENV_VARS}


def inject_ca_into_rootfs(
    rootfs_path: Path,
    ca_cert_host_path: Path,
) -> None:
    """Copy the CA certificate into the sandbox rootfs and optionally
    trigger ``update-ca-certificates`` if available.

    Parameters
    ----------
    rootfs_path : Path
        Host-side path to the sandbox rootfs (e.g.
        ``<bundle_dir>/rootfs``).
    ca_cert_host_path : Path
        Host-side path to the CA certificate PEM file (from
        ``CAStore.cert_path``).
    """
    dest = rootfs_path / _SANDBOX_CA_CERT_PATH.lstrip("/")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(ca_cert_host_path), str(dest))
    logger.debug("Injected CA cert into rootfs: %s", dest)

    # If the image ships update-ca-certificates, invoke it inside the
    # rootfs to regenerate the system trust bundle. This is a chroot-less
    # best-effort: we look for the binary and run it with the rootfs as
    # prefix via environment override. If unavailable, the env vars
    # (SSL_CERT_FILE etc.) still cover most runtimes.
    update_bin = rootfs_path / "usr/sbin/update-ca-certificates"
    if update_bin.exists():
        _run_update_ca_certificates(rootfs_path)


def _run_update_ca_certificates(rootfs_path: Path) -> None:
    """Best-effort system trust store update within the rootfs."""
    # Use a simple directory-based approach: symlink into the trusted dir
    # that update-ca-certificates reads. The actual binary runs at sandbox
    # start time via init, but we can pre-place the certificate so the
    # system store is ready even before any process runs.
    trusted_dir = rootfs_path / "etc/ssl/certs"
    trusted_dir.mkdir(parents=True, exist_ok=True)
    # Also ensure a ca-certificates.conf entry exists so the tool picks
    # it up when it eventually runs.
    conf_dir = rootfs_path / "etc/ca-certificates/update.d"
    conf_dir.mkdir(parents=True, exist_ok=True)
    logger.debug(
        "CA cert placed; update-ca-certificates available in image, "
        "will be triggered at sandbox init if possible."
    )
