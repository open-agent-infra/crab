"""Sandbox resource claims — normalization shared by the SDK and gateway.

`Sandbox(resources=...)` accepts the user-facing shape
(`{"cpus": 2, "memory": "512M", "pids": 256}`); this module normalizes
it once, at construction, into the wire-format *claim*
(`{"cpus": 2, "memory_bytes": 536870912, "pids": 256}`) that travels in
the launch metadata under the `"resources"` key. The gateway re-validates
the claim on its side (`validate_claim`) without importing the runc
bundle machinery — this module is intentionally stdlib-only.

Semantics (track design doc §4 S3): limits are per-sandbox floors
decided at launch; there is no live resize in v0. An empty/omitted
`resources` means "no limits" and leaves every launch path byte-for-byte
identical to the unlimited behavior.
"""
from __future__ import annotations

import re
from typing import Any, Mapping

CLAIM_KEYS = ("cpus", "memory_bytes", "pids")
"""Wire-format claim keys, all positive ints."""

_USER_KEYS = ("cpus", "memory", "pids")

_MEMORY_RE = re.compile(r"^\s*(\d+)\s*([KMGT])?(?:I?B)?\s*$", re.IGNORECASE)

_MEMORY_MULTIPLIERS = {
    None: 1,
    "K": 1024,
    "M": 1024**2,
    "G": 1024**3,
    "T": 1024**4,
}


def _require_positive_int(value: Any, field: str) -> int:
    # bool is an int subclass; True/False are always caller mistakes here.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"resources.{field} must be a positive integer, got {value!r}")
    if value <= 0:
        raise ValueError(f"resources.{field} must be positive, got {value!r}")
    return int(value)


def parse_memory_bytes(value: Any, field: str = "memory") -> int:
    """Positive byte count from an int or a string with an optional binary
    suffix: `"512M"`, `"2G"`, `"1024KB"`, `"1GiB"` (K/M/G/T are 1024-based).
    """
    if isinstance(value, bool):
        raise ValueError(f"resources.{field} must be bytes or a size string, got {value!r}")
    if isinstance(value, int):
        if value <= 0:
            raise ValueError(f"resources.{field} must be positive, got {value!r}")
        return int(value)
    if isinstance(value, str):
        match = _MEMORY_RE.match(value)
        if match is None:
            raise ValueError(
                f"resources.{field} must look like '512M', '2G' or a byte count, got {value!r}"
            )
        amount = int(match.group(1))
        if amount <= 0:
            raise ValueError(f"resources.{field} must be positive, got {value!r}")
        suffix = match.group(2)
        return amount * _MEMORY_MULTIPLIERS[suffix.upper() if suffix else None]
    raise ValueError(f"resources.{field} must be bytes or a size string, got {value!r}")


def normalize_resources(resources: Mapping[str, Any] | None) -> dict[str, int]:
    """User-facing `resources` mapping -> wire-format claim.

    Accepted keys: `cpus` (positive int), `memory` (bytes or size string),
    `pids` (positive int). Unknown keys and invalid values raise
    `ValueError` loudly — resources are enforced as of S3, so a typo must
    not silently become "no limit". Returns `{}` for `None`/empty input.
    """
    if not resources:
        return {}
    unknown = sorted(set(resources) - set(_USER_KEYS))
    if unknown:
        raise ValueError(
            f"unknown resources keys: {', '.join(unknown)} (expected any of {', '.join(_USER_KEYS)})"
        )
    claim: dict[str, int] = {}
    if "cpus" in resources:
        claim["cpus"] = _require_positive_int(resources["cpus"], "cpus")
    if "memory" in resources:
        claim["memory_bytes"] = parse_memory_bytes(resources["memory"])
    if "pids" in resources:
        claim["pids"] = _require_positive_int(resources["pids"], "pids")
    return claim


def validate_claim(claim: Any) -> dict[str, int]:
    """Wire-side validation of a normalized claim (gateway create path).

    Accepts `None`/`{}` (no limits). Raises `ValueError` on non-mapping
    input, unknown keys, or non-positive-int values.
    """
    if claim is None:
        return {}
    if not isinstance(claim, Mapping):
        raise ValueError(f"resources must be a JSON object, got {type(claim).__name__}")
    unknown = sorted(set(claim) - set(CLAIM_KEYS))
    if unknown:
        raise ValueError(
            f"unknown resources keys: {', '.join(unknown)} (expected any of {', '.join(CLAIM_KEYS)})"
        )
    return {key: _require_positive_int(claim[key], key) for key in CLAIM_KEYS if key in claim}
