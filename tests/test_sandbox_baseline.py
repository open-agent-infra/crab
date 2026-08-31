from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from integrations.sandboxes.runtime.baseline import (
    SANDBOX_BASELINE_CAPABILITIES,
    SANDBOX_ROOTFS_PREPARATION_SCHEMA,
    SandboxBaselineError,
    add_dns_materialization,
    apply_sandbox_bundle_baseline,
    apply_sandbox_process_baseline,
    materialize_resolver_config,
    version_shared_rootfs_key,
)


class SandboxBaselineTests(unittest.TestCase):
    def test_resolver_skips_loopback_stub_and_uses_upstream(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            stub = root / "stub.conf"
            upstream = root / "upstream.conf"
            bundle = root / "bundle"
            stub.write_text("nameserver 127.0.0.53\n", encoding="utf-8")
            upstream.write_text(
                "search example.test\nnameserver 10.0.2.3\noptions edns0\n",
                encoding="utf-8",
            )

            destination = materialize_resolver_config(
                bundle, candidates=(stub, upstream)
            )

            self.assertEqual(destination, bundle / "crab-resolv.conf")
            rendered = destination.read_text(encoding="utf-8")
            self.assertIn("nameserver 10.0.2.3", rendered)
            self.assertNotIn("127.0.0.53", rendered)
            self.assertIn("search example.test", rendered)
            self.assertEqual(destination.stat().st_mode & 0o777, 0o644)

    def test_host_network_preserves_reachable_systemd_stub(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            stub = root / "stub.conf"
            stub.write_text(
                "nameserver 127.0.0.53\noptions edns0 trust-ad\n",
                encoding="utf-8",
            )

            metadata: dict[str, object] = {}
            destination = add_dns_materialization(
                metadata,
                bundle_dir=root / "bundle",
                candidates=(stub,),
                isolated=False,
            )

            self.assertEqual(
                destination.read_text(encoding="utf-8"),
                "nameserver 127.0.0.53\noptions edns0 trust-ad\n",
            )

    def test_resolver_failure_is_actionable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            stub = root / "stub.conf"
            stub.write_text("nameserver ::1\n", encoding="utf-8")
            with self.assertRaisesRegex(
                SandboxBaselineError, "no usable upstream resolver"
            ):
                materialize_resolver_config(root / "bundle", candidates=(stub,))

    def test_dns_is_post_clone_and_schema_versioned(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            resolver = root / "resolver.conf"
            resolver.write_text("nameserver 192.0.2.53\n", encoding="utf-8")
            metadata: dict[str, object] = {
                "rootfs_copy_paths": [
                    {"source": "/immutable/image", "destination": "/"}
                ]
            }

            generated = add_dns_materialization(
                metadata,
                bundle_dir=root / "bundle",
                candidates=(resolver,),
            )

            self.assertEqual(
                metadata["rootfs_copy_paths"],
                [{"source": "/immutable/image", "destination": "/"}],
            )
            self.assertEqual(
                metadata["rootfs_post_clone_copy_paths"],
                [
                    {
                        "source": str(generated),
                        "destination": "/etc/resolv.conf",
                        "replace": True,
                    }
                ],
            )
            self.assertEqual(
                metadata["rootfs_preparation_schema"],
                SANDBOX_ROOTFS_PREPARATION_SCHEMA,
            )
            self.assertEqual(
                version_shared_rootfs_key("abc"),
                f"{SANDBOX_ROOTFS_PREPARATION_SCHEMA}-abc",
            )

    def test_capability_profile_is_explicit_and_non_privileged(self) -> None:
        config: dict[str, object] = {"process": {"noNewPrivileges": True}}

        apply_sandbox_process_baseline(config)

        process = config["process"]
        self.assertIsInstance(process, dict)
        capabilities = process["capabilities"]
        self.assertEqual(
            set(capabilities),
            {"bounding", "effective", "permitted", "inheritable", "ambient"},
        )
        for values in capabilities.values():
            self.assertEqual(tuple(values), SANDBOX_BASELINE_CAPABILITIES)
            self.assertIn("CAP_SETUID", values)
            self.assertIn("CAP_SETGID", values)
            self.assertNotIn("CAP_SYS_ADMIN", values)
        self.assertIs(process["noNewPrivileges"], False)

    def test_bundle_helper_applies_same_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle = Path(temp_dir)
            config_path = bundle / "config.json"
            config_path.write_text(
                json.dumps({"process": {}, "root": {"path": "rootfs"}}),
                encoding="utf-8",
            )

            apply_sandbox_bundle_baseline(bundle)

            process = json.loads(config_path.read_text(encoding="utf-8"))["process"]
            self.assertEqual(
                process["capabilities"]["effective"],
                list(SANDBOX_BASELINE_CAPABILITIES),
            )


if __name__ == "__main__":
    unittest.main()
