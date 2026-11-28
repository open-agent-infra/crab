"""Unit tests for S3 resource enforcement plumbing (track S, S3):
claim normalization (`crab.resources`), the Sandbox constructor's
loud validation and claim -> `SandboxResourceLimits` mapping, bundle
spec generation for claim-derived limits, and fork inheritance of
`linux.resources` (including the file-bind mount rewrite the
cpu-visibility overlay needs). Host-runnable — no daemon, no root.

The real cgroup enforcement path (OOM-kill at the limit) is covered by
`tests/test_resources_real.py`, gated on `CRAB_REAL_HOST_TESTS`.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from crab.forking import replicate_bundle_config
from crab.ids import SandboxId
from crab.resources import normalize_resources, parse_memory_bytes, validate_claim
from crab.sandbox import Sandbox
from integrations.sandboxes.runtime import bundle as sandbox_bundle

MiB = 1024 * 1024
GiB = 1024 * MiB


# ---------------------------------------------------------------------------
# crab.resources — normalization shared by the SDK and gateway
# ---------------------------------------------------------------------------


class ParseMemoryBytesTests(unittest.TestCase):
    def test_plain_ints_and_digit_strings(self) -> None:
        self.assertEqual(parse_memory_bytes(123456), 123456)
        self.assertEqual(parse_memory_bytes("123456"), 123456)

    def test_binary_suffixes(self) -> None:
        self.assertEqual(parse_memory_bytes("1K"), 1024)
        self.assertEqual(parse_memory_bytes("512M"), 512 * MiB)
        self.assertEqual(parse_memory_bytes("2G"), 2 * GiB)
        self.assertEqual(parse_memory_bytes("1T"), 1024 * GiB)
        # KB/KiB/kib spellings are all the same 1024-based unit.
        self.assertEqual(parse_memory_bytes("512MB"), 512 * MiB)
        self.assertEqual(parse_memory_bytes("512MiB"), 512 * MiB)
        self.assertEqual(parse_memory_bytes("512mib"), 512 * MiB)

    def test_invalid_values_are_loud(self) -> None:
        for bad in (0, -1, "0", "-5M", "", "lots", "1.5G", "5X", None, True, 1.5):
            with self.assertRaises(ValueError, msg=repr(bad)):
                parse_memory_bytes(bad)


class NormalizeResourcesTests(unittest.TestCase):
    def test_none_and_empty_mean_no_limits(self) -> None:
        self.assertEqual(normalize_resources(None), {})
        self.assertEqual(normalize_resources({}), {})

    def test_full_claim(self) -> None:
        claim = normalize_resources({"cpus": 2, "memory": "512M", "pids": 256})
        self.assertEqual(claim, {"cpus": 2, "memory_bytes": 512 * MiB, "pids": 256})

    def test_partial_claims_keep_only_declared_keys(self) -> None:
        self.assertEqual(normalize_resources({"memory": 1 * GiB}), {"memory_bytes": GiB})
        self.assertEqual(normalize_resources({"cpus": 1}), {"cpus": 1})

    def test_unknown_keys_are_loud(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            normalize_resources({"cpu": 2})  # typo for "cpus"
        self.assertIn("cpu", str(ctx.exception))

    def test_invalid_values_are_loud(self) -> None:
        for bad in (
            {"cpus": 0},
            {"cpus": -1},
            {"cpus": True},
            {"cpus": "2"},
            {"cpus": 1.5},
            {"memory": 0},
            {"memory": "x"},
            {"pids": 0},
            {"pids": False},
        ):
            with self.assertRaises(ValueError, msg=repr(bad)):
                normalize_resources(bad)


class ValidateClaimTests(unittest.TestCase):
    def test_none_and_empty_pass(self) -> None:
        self.assertEqual(validate_claim(None), {})
        self.assertEqual(validate_claim({}), {})

    def test_valid_claim_roundtrips(self) -> None:
        claim = {"cpus": 2, "memory_bytes": 512 * MiB, "pids": 64}
        self.assertEqual(validate_claim(claim), claim)

    def test_non_mapping_rejected(self) -> None:
        for bad in ("resources", 7, ["cpus"]):
            with self.assertRaises(ValueError, msg=repr(bad)):
                validate_claim(bad)

    def test_unknown_keys_and_bad_values_rejected(self) -> None:
        for bad in (
            {"memory": 512 * MiB},  # user-facing key, not wire format
            {"cpus": "2"},
            {"cpus": 0},
            {"memory_bytes": True},
            {"pids": -1},
        ):
            with self.assertRaises(ValueError, msg=repr(bad)):
                validate_claim(bad)


# ---------------------------------------------------------------------------
# Sandbox constructor — loud normalization, claim -> SandboxResourceLimits
# ---------------------------------------------------------------------------


class _EngineStub:
    """The constructor only stores the engine when autostart=False."""


class SandboxClaimTests(unittest.TestCase):
    def _sandbox(self, resources: dict | None) -> Sandbox:
        return Sandbox(engine=_EngineStub(), autostart=False, resources=resources)

    def test_claim_normalized_and_raw_metadata_kept(self) -> None:
        sbx = self._sandbox({"cpus": 2, "memory": "512M", "pids": 128})
        self.assertEqual(
            sbx._resource_claim, {"cpus": 2, "memory_bytes": 512 * MiB, "pids": 128}
        )
        # `Sandbox.metadata` keeps the user's original spelling.
        self.assertEqual(sbx.metadata["resources"], {"cpus": 2, "memory": "512M", "pids": 128})

    def test_invalid_resources_fail_at_construction(self) -> None:
        with self.assertRaises(ValueError):
            self._sandbox({"memory": "eleventy"})
        with self.assertRaises(ValueError):
            self._sandbox({"gpu": 1})

    def test_bundle_resource_limits_mapping(self) -> None:
        sbx = self._sandbox({"cpus": 3, "memory": 2 * GiB, "pids": 99})
        limits = sbx._bundle_resource_limits()
        self.assertIsInstance(limits, sandbox_bundle.SandboxResourceLimits)
        self.assertEqual(limits.cpus, 3)
        self.assertEqual(limits.memory_bytes, 2 * GiB)
        self.assertEqual(limits.pids_limit, 99)

    def test_partial_claim_leaves_other_limits_unset(self) -> None:
        limits = self._sandbox({"memory": "256M"})._bundle_resource_limits()
        self.assertEqual(limits.memory_bytes, 256 * MiB)
        self.assertIsNone(limits.cpus)
        self.assertIsNone(limits.pids_limit)

    def test_no_resources_means_no_limits_object(self) -> None:
        self.assertIsNone(self._sandbox(None)._bundle_resource_limits())
        self.assertIsNone(self._sandbox({})._bundle_resource_limits())


# ---------------------------------------------------------------------------
# Bundle spec generation — claim-derived limits land in linux.resources
# ---------------------------------------------------------------------------


def _write_minimal_config(bundle_dir: Path) -> Path:
    config_path = bundle_dir / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "linux": {"namespaces": []},
                "mounts": [],
                "process": {"terminal": True, "cwd": "/", "args": [], "env": []},
                "root": {"path": "rootfs", "readonly": True},
            }
        ),
        encoding="utf-8",
    )
    return config_path


class SpecGenerationTests(unittest.TestCase):
    def _write(self, bundle_dir: Path, limits) -> dict:
        config_path = _write_minimal_config(bundle_dir)
        sandbox_bundle.write_bundle_config(
            bundle_dir=bundle_dir,
            llm_base_url="",
            provider="openai",
            sandbox_name="sbx-s3",
            status_port=9001,
            cgroup_path="crab-sdk/sbx-s3",
            resource_limits=limits,
        )
        return json.loads(config_path.read_text(encoding="utf-8"))

    def test_claim_fields_map_to_oci_resources(self) -> None:
        claim = normalize_resources({"cpus": 2, "memory": "512M", "pids": 64})
        limits = sandbox_bundle.SandboxResourceLimits(
            cpus=claim.get("cpus"),
            memory_bytes=claim.get("memory_bytes"),
            pids_limit=claim.get("pids"),
        )
        with tempfile.TemporaryDirectory() as tmp:
            payload = self._write(Path(tmp), limits)
        resources = payload["linux"]["resources"]
        self.assertEqual(resources["memory"]["limit"], 512 * MiB)
        self.assertEqual(resources["pids"]["limit"], 64)
        self.assertEqual(resources["cpu"]["period"], 100_000)
        self.assertEqual(resources["cpu"]["quota"], 200_000)
        self.assertEqual(payload["linux"]["cgroupsPath"], "crab-sdk/sbx-s3")

    def test_memory_only_claim_leaves_cpu_and_pids_unset(self) -> None:
        limits = sandbox_bundle.SandboxResourceLimits(memory_bytes=256 * MiB)
        with tempfile.TemporaryDirectory() as tmp:
            payload = self._write(Path(tmp), limits)
        resources = payload["linux"]["resources"]
        self.assertEqual(resources["memory"]["limit"], 256 * MiB)
        self.assertNotIn("cpu", resources)
        self.assertNotIn("pids", resources)

    def test_no_limits_leaves_spec_without_resources_section(self) -> None:
        # Zero-breakage: the no-resources spec is identical to main's.
        with tempfile.TemporaryDirectory() as tmp:
            payload = self._write(Path(tmp), None)
        self.assertNotIn("resources", payload["linux"])

    def test_invalid_limit_values_are_loud(self) -> None:
        for limits in (
            sandbox_bundle.SandboxResourceLimits(memory_bytes=0),
            sandbox_bundle.SandboxResourceLimits(memory_bytes=-1),
            sandbox_bundle.SandboxResourceLimits(pids_limit=0),
            sandbox_bundle.SandboxResourceLimits(cpus=-2, auto_cpu_set=False),
        ):
            with tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(ValueError, msg=repr(limits)):
                    self._write(Path(tmp), limits)


# ---------------------------------------------------------------------------
# Fork inheritance — replicate_bundle_config copies linux.resources
# ---------------------------------------------------------------------------


class ForkInheritanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        self.source_id = SandboxId("src-1")
        self.target_id = SandboxId("src-1-fork-aa")
        self.source_dir = root / "bundles" / str(self.source_id)
        self.target_dir = root / "bundles" / str(self.target_id)
        self.source_dir.mkdir(parents=True)
        self.target_dir.mkdir(parents=True)

    def _write_configs(self, source_cfg: dict, target_cfg: dict | None = None) -> None:
        (self.source_dir / "config.json").write_text(
            json.dumps(source_cfg), encoding="utf-8"
        )
        (self.target_dir / "config.json").write_text(
            json.dumps(target_cfg if target_cfg is not None else source_cfg),
            encoding="utf-8",
        )

    def _replicate(self) -> dict:
        replicate_bundle_config(
            self.source_dir, self.target_dir, self.source_id, self.target_id
        )
        return json.loads((self.target_dir / "config.json").read_text(encoding="utf-8"))

    def test_fork_inherits_linux_resources(self) -> None:
        resources = {
            "memory": {"limit": 512 * MiB},
            "cpu": {"period": 100_000, "quota": 200_000},
            "pids": {"limit": 64},
        }
        self._write_configs(
            {
                "linux": {
                    "cgroupsPath": f"crab-sdk/{self.source_id}",
                    "resources": resources,
                },
                "mounts": [],
                "process": {"cwd": "/work", "env": []},
            },
            {
                "linux": {"cgroupsPath": f"crab-sdk/{self.source_id}"},
                "mounts": [],
                "process": {"cwd": "/work", "env": []},
            },
        )
        target = self._replicate()
        self.assertEqual(target["linux"]["resources"], resources)
        # The fork keeps its own cgroup — limits travel, the path doesn't.
        self.assertEqual(target["linux"]["cgroupsPath"], f"crab-sdk/{self.target_id}")

    def test_source_without_resources_writes_none(self) -> None:
        self._write_configs(
            {
                "linux": {"cgroupsPath": f"crab-sdk/{self.source_id}"},
                "mounts": [],
                "process": {"cwd": "/work", "env": []},
            }
        )
        target = self._replicate()
        self.assertNotIn("resources", target["linux"])

    def test_file_bind_mount_is_copied_not_mkdired(self) -> None:
        # The cpu-visibility overlay is a *file* bind; the per-sandbox
        # path rewrite must copy it, not plant a directory in its place.
        overlay = self.source_dir / "cpu-visibility" / "online"
        overlay.parent.mkdir(parents=True)
        overlay.write_text("0-1\n", encoding="utf-8")
        work_dir = self.source_dir / "work"
        work_dir.mkdir()
        self._write_configs(
            {
                "linux": {"cgroupsPath": f"crab-sdk/{self.source_id}"},
                "mounts": [
                    {
                        "destination": "/sys/devices/system/cpu/online",
                        "source": str(overlay),
                        "type": "bind",
                        "options": ["rbind", "ro"],
                    },
                    {
                        "destination": "/work",
                        "source": str(work_dir),
                        "type": "bind",
                        "options": ["rbind", "rw"],
                    },
                ],
                "process": {"cwd": "/work", "env": []},
            }
        )
        target = self._replicate()
        by_dest = {m["destination"]: m for m in target["mounts"]}
        rewritten_overlay = Path(by_dest["/sys/devices/system/cpu/online"]["source"])
        self.assertIn(str(self.target_id), str(rewritten_overlay))
        self.assertTrue(rewritten_overlay.is_file())
        self.assertEqual(rewritten_overlay.read_text(encoding="utf-8"), "0-1\n")
        rewritten_work = Path(by_dest["/work"]["source"])
        self.assertIn(str(self.target_id), str(rewritten_work))
        self.assertTrue(rewritten_work.is_dir())


if __name__ == "__main__":
    unittest.main()
