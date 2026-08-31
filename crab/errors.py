from __future__ import annotations

import subprocess
from typing import Sequence


class SandboxExecTimeout(subprocess.TimeoutExpired):
    """An attached sandbox exec crossed its daemon-enforced deadline.

    Subclassing :class:`subprocess.TimeoutExpired` preserves compatibility for
    existing direct-runtime callers while giving daemon/gateway/SDK code one
    stable type.  ``stdout`` and ``stderr`` contain the partial captured output
    available after the complete payload cgroup has been reaped.
    """

    error_type = "exec_timeout"

    def __init__(
        self,
        cmd: Sequence[str] | str,
        timeout: float,
        *,
        stdout: str = "",
        stderr: str = "",
    ) -> None:
        super().__init__(cmd=cmd, timeout=timeout, output=stdout, stderr=stderr)


class SandboxExecCleanupError(RuntimeError):
    """Crab could not prove that a timed-out exec payload was fully reaped."""

    error_type = "exec_cleanup_failed"

    def __init__(
        self,
        message: str,
        *,
        cmd: Sequence[str] | str,
        timeout: float | None,
        stdout: str = "",
        stderr: str = "",
        payload_pid: int | None = None,
        cgroup_path: str | None = None,
    ) -> None:
        super().__init__(message)
        self.cmd = cmd
        self.timeout = timeout
        self.stdout = stdout
        self.stderr = stderr
        self.payload_pid = payload_pid
        self.cgroup_path = cgroup_path


class SandboxCreateCleanupError(RuntimeError):
    """Sandbox creation failed and one or more per-sandbox resources leaked."""

    error_type = "create_cleanup_failed"

    def __init__(
        self,
        sandbox_id: str,
        cause: BaseException,
        cleanup_errors: Sequence[str],
        *,
        resources: Sequence[str] = (),
    ) -> None:
        details = "; ".join(str(item) for item in cleanup_errors)
        resource_text = ", ".join(str(item) for item in resources) or sandbox_id
        super().__init__(
            f"sandbox creation failed: {type(cause).__name__}: {cause}; "
            f"cleanup also failed for {resource_text}: {details}. "
            "Remove or repair the listed resources manually before retrying"
        )
        self.sandbox_id = sandbox_id
        self.cause = cause
        self.cleanup_errors = tuple(str(item) for item in cleanup_errors)
        self.resources = tuple(str(item) for item in resources)


class SandboxImageError(RuntimeError):
    """Base class for actionable sandbox image resolution failures."""

    error_type = "image_error"

    def __init__(self, reference: str, message: str) -> None:
        super().__init__(message)
        self.reference = reference


class ImageNotFoundError(SandboxImageError):
    error_type = "image_not_found"


class ImageAuthenticationError(SandboxImageError):
    error_type = "image_authentication_required"


class ImageRateLimitError(SandboxImageError):
    error_type = "image_rate_limited"


class ImagePullTimeoutError(SandboxImageError):
    error_type = "image_pull_timeout"


class ImagePlatformError(SandboxImageError):
    error_type = "image_incompatible_platform"


class ImageCompatibilityError(SandboxImageError):
    error_type = "image_incompatible"


class ImageReferenceError(SandboxImageError):
    error_type = "image_malformed_reference"


class ImagePolicyError(SandboxImageError):
    error_type = "image_policy_denied"


class ImageInsufficientDiskError(SandboxImageError):
    error_type = "image_insufficient_disk"


class ImageTooLargeError(SandboxImageError):
    error_type = "image_too_large"


class ImagePullError(SandboxImageError):
    error_type = "image_pull_failed"


__all__ = [
    "SandboxCreateCleanupError",
    "SandboxExecCleanupError",
    "SandboxExecTimeout",
    "SandboxImageError",
    "ImageAuthenticationError",
    "ImageCompatibilityError",
    "ImageInsufficientDiskError",
    "ImageNotFoundError",
    "ImagePlatformError",
    "ImagePolicyError",
    "ImagePullError",
    "ImagePullTimeoutError",
    "ImageRateLimitError",
    "ImageReferenceError",
    "ImageTooLargeError",
]
