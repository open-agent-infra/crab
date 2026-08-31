#!/usr/bin/env python3
"""Compare one URL from a Crab service VM and one fresh sandbox.

Run this inside the service VM after the daemon is available.  The output is
one credential-free JSON document suitable for attaching to an incident.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from egress_probe import probe, redact_url


def _network_argument(raw: str) -> bool | None:
    if raw == "auto":
        return None
    return raw == "isolated"


def _run_probe(url: str, *, timeout: float) -> dict[str, Any]:
    started_at = time.time()
    try:
        return {
            "ok": True,
            "started_at_unix": started_at,
            **probe(url, timeout=timeout),
        }
    except Exception as exc:
        safe_url = redact_url(url)
        return {
            "ok": False,
            "started_at_unix": started_at,
            "requested_url": safe_url,
            "error_type": type(exc).__name__,
            "error": str(exc).replace(url, safe_url),
        }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="measure service-VM and sandbox egress for the same URL"
    )
    parser.add_argument("url")
    parser.add_argument("--socket", default="/run/crab/crab.sock")
    parser.add_argument("--image", default="python:3.12-slim")
    parser.add_argument(
        "--network",
        choices=("auto", "host", "isolated"),
        default="auto",
        help="sandbox network selection (default: daemon policy)",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    # Imports are delayed so the host-side measurement is still returned if
    # Crab itself cannot be imported or reached.
    output: dict[str, Any] = {
        "schema": "crab-egress-diagnostic-v1",
        "url": redact_url(args.url),
        "service_vm": _run_probe(args.url, timeout=args.timeout),
        "sandbox": None,
    }
    sandbox = None
    try:
        from crab import Engine, Sandbox

        engine = Engine.connect(socket=args.socket)
        sandbox = Sandbox(
            image=args.image,
            network=_network_argument(args.network),
            engine=engine,
        )
        probe_source = (Path(__file__).with_name("egress_probe.py")).read_text(
            encoding="utf-8"
        )
        sandbox.files.write("/tmp/crab-egress-probe.py", probe_source)
        result = sandbox.commands.run(
            [
                "python3",
                "/tmp/crab-egress-probe.py",
                args.url,
                "--timeout",
                str(args.timeout),
            ],
            timeout=args.timeout + 15.0,
            capture_output=True,
            check=False,
        )
        try:
            measured = json.loads(result.stdout)
        except json.JSONDecodeError:
            measured = {
                "ok": False,
                "error_type": "InvalidProbeOutput",
                "error": result.stderr or result.stdout,
            }
        description = sandbox.describe()
        output["sandbox"] = {
            "image": args.image,
            "requested_network": args.network,
            "sandbox_id": description.sandbox_id,
            "metadata": {
                key: description.metadata.get(key)
                for key in (
                    "image_reference",
                    "image_digest",
                    "network_mode",
                    "guest_ip",
                    "network_namespace_path",
                )
                if key in description.metadata
            },
            "command_returncode": result.returncode,
            "measurement": measured,
        }
    except Exception as exc:
        output["sandbox"] = {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "image": args.image,
            "requested_network": args.network,
        }
    finally:
        if sandbox is not None:
            sandbox.kill()

    print(json.dumps(output, indent=2, sort_keys=True))
    sandbox_result = output.get("sandbox") or {}
    sandbox_measurement = sandbox_result.get("measurement") or {}
    return 0 if output["service_vm"].get("ok") and sandbox_measurement.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
