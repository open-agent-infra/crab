from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from crab import RuncRuntime, RuncRuntimePaths, SandboxId
from crab.models import SandboxDescription
from crab.runtime.runc import _repair_postfix_rootfs_permissions


class RuncRuntimePrepareTests(unittest.TestCase):
    def test_repair_postfix_rootfs_permissions_normalizes_queue_owners(self) -> None:
        with tempfile.TemporaryDirectory(prefix="crab_runc_prepare_") as tmp:
            root = Path(tmp)
            (root / "etc").mkdir(parents=True, exist_ok=True)
            (root / "var" / "spool" / "postfix").mkdir(parents=True, exist_ok=True)
            (root / "var" / "lib" / "postfix").mkdir(parents=True, exist_ok=True)
            (root / "etc" / "passwd").write_text("postfix:x:101:103::/var/spool/postfix:/usr/sbin/nologin\n", encoding="utf-8")
            (root / "etc" / "group").write_text("postfix:x:103:\npostdrop:x:104:\n", encoding="utf-8")

            for name in ("active", "defer", "private", "maildrop", "public", "pid"):
                path = root / "var" / "spool" / "postfix" / name
                path.mkdir(parents=True, exist_ok=True)
                path.chmod(0o755)
                os.chown(path, 65534, 65534)
            restart_marker = root / "var" / "spool" / "postfix" / "restart"
            restart_marker.write_text("", encoding="utf-8")
            os.chown(restart_marker, 65534, 65534)

            _repair_postfix_rootfs_permissions(root)

            self.assertEqual((root / "var" / "spool" / "postfix" / "active").stat().st_uid, 101)
            self.assertEqual((root / "var" / "spool" / "postfix" / "active").stat().st_gid, 103)
            self.assertEqual((root / "var" / "spool" / "postfix" / "active").stat().st_mode & 0o7777, 0o700)

            self.assertEqual((root / "var" / "spool" / "postfix" / "maildrop").stat().st_uid, 101)
            self.assertEqual((root / "var" / "spool" / "postfix" / "maildrop").stat().st_gid, 104)
            self.assertEqual((root / "var" / "spool" / "postfix" / "maildrop").stat().st_mode & 0o7777, 0o1730)

            self.assertEqual((root / "var" / "spool" / "postfix" / "public").stat().st_uid, 101)
            self.assertEqual((root / "var" / "spool" / "postfix" / "public").stat().st_gid, 104)
            self.assertEqual((root / "var" / "spool" / "postfix" / "public").stat().st_mode & 0o7777, 0o2710)

            self.assertEqual((root / "var" / "spool" / "postfix" / "pid").stat().st_uid, 0)
            self.assertEqual((root / "var" / "spool" / "postfix" / "pid").stat().st_gid, 0)
            self.assertEqual((root / "var" / "spool" / "postfix" / "pid").stat().st_mode & 0o7777, 0o755)

            self.assertEqual((root / "var" / "spool" / "postfix" / "restart").stat().st_uid, 0)
            self.assertEqual((root / "var" / "spool" / "postfix" / "restart").stat().st_gid, 0)
            self.assertEqual((root / "var" / "spool" / "postfix" / "restart").stat().st_mode & 0o7777, 0o644)

    def test_persist_uses_unique_temp_paths_for_concurrent_metadata_updates(self) -> None:
        with tempfile.TemporaryDirectory(prefix="crab_runc_prepare_") as tmp:
            root = Path(tmp)
            runtime = RuncRuntime(
                paths=RuncRuntimePaths(
                    state_root=root / "state",
                    bundle_root=root / "bundles",
                    checkpoint_root=root / "checkpoints",
                    metadata_root=root / "metadata",
                    zfs_dataset_prefix="pool/crab",
                )
            )
            sandbox_id = SandboxId("sbx-test")
            descriptions = [
                SandboxDescription(
                    sandbox_id=sandbox_id,
                    runtime_name="runc",
                    status="running",
                    metadata={"generation": 1},
                ),
                SandboxDescription(
                    sandbox_id=sandbox_id,
                    runtime_name="runc",
                    status="running",
                    metadata={"generation": 2},
                ),
            ]
            barrier = threading.Barrier(2)
            original_replace = Path.replace

            def _replace_with_barrier(path: Path, target: Path) -> Path:
                if path.suffix == ".tmp":
                    barrier.wait(timeout=2.0)
                return original_replace(path, target)

            errors: list[Exception] = []

            def _persist(description: SandboxDescription) -> None:
                try:
                    runtime._persist(description)
                except Exception as exc:  # pragma: no cover - asserted below
                    errors.append(exc)

            with patch.object(Path, "replace", autospec=True, side_effect=_replace_with_barrier):
                threads = [threading.Thread(target=_persist, args=(description,)) for description in descriptions]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join(timeout=2.0)

            self.assertEqual(errors, [])
            payload = json.loads((root / "metadata" / "sbx-test.json").read_text(encoding="utf-8"))
            self.assertIn(payload["metadata"]["generation"], {1, 2})


class UpdateNetworkMetadataTests(unittest.TestCase):
    """PR-N1 decision 9: promotion adopts the fork's lease, so the source's
    launch-time network metadata goes dead. Two readers depend on it (the
    interceptor attribution fallback and Sandbox.get_host), so the swap must
    rewrite it — a transfer that fixes only the lease passes the socket
    assertions while silently breaking attribution."""

    def _runtime(self, root: Path) -> RuncRuntime:
        return RuncRuntime(
            paths=RuncRuntimePaths(
                state_root=root / "state",
                bundle_root=root / "bundles",
                checkpoint_root=root / "checkpoints",
                metadata_root=root / "metadata",
                zfs_dataset_prefix="pool/crab",
            )
        )

    def test_rewrites_guest_ip_and_netns_preserving_other_metadata(self) -> None:
        with tempfile.TemporaryDirectory(prefix="crab_runc_meta_") as tmp:
            root = Path(tmp)
            runtime = self._runtime(root)
            sandbox_id = SandboxId("sbx-src")
            runtime.adopt_sandbox_description(
                sandbox_id,
                runtime_name="runc",
                status="running",
                metadata={
                    "guest_ip": "10.250.0.2",
                    "network_namespace_path": "/var/run/netns/ts-old",
                    "bridge_ip": "10.250.0.1",
                },
            )

            runtime.update_network_metadata(
                sandbox_id,
                guest_ip="10.250.0.3",
                network_namespace_path="/var/run/netns/ts-new",
            )

            described = runtime.describe(sandbox_id)
            self.assertEqual(described.metadata["guest_ip"], "10.250.0.3")
            self.assertEqual(
                described.metadata["network_namespace_path"], "/var/run/netns/ts-new"
            )
            # Untouched keys survive the rewrite.
            self.assertEqual(described.metadata["bridge_ip"], "10.250.0.1")
            # The change is persisted, not just in-memory: this is what the
            # interceptor's inspect_runtime attribution fallback reads back.
            payload = json.loads(
                (root / "metadata" / "sbx-src.json").read_text(encoding="utf-8")
            )
            self.assertEqual(payload["metadata"]["guest_ip"], "10.250.0.3")
            self.assertEqual(
                payload["metadata"]["network_namespace_path"], "/var/run/netns/ts-new"
            )

    def test_missing_sandbox_is_ignored(self) -> None:
        # The caller is mid-swap; a sandbox with no description has no
        # metadata to correct and must not raise.
        with tempfile.TemporaryDirectory(prefix="crab_runc_meta_") as tmp:
            runtime = self._runtime(Path(tmp))
            runtime.update_network_metadata(
                SandboxId("sbx-absent"),
                guest_ip="10.250.0.9",
                network_namespace_path="/var/run/netns/ts-x",
            )  # no raise


if __name__ == "__main__":
    unittest.main()
