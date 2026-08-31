from __future__ import annotations

import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from crab.engine import EngineConfig, resolve_sandbox_network_mode
from crab.errors import SandboxCreateCleanupError
from crab.ids import SandboxId
from integrations.sandboxes.runtime.network import BenchmarkNetworkManager


class NetworkSelectionTests(unittest.TestCase):
    def test_explicit_values_win_over_daemon_default(self) -> None:
        config = EngineConfig(
            runtime="runc",
            enable_sandbox_network=True,
            enable_interceptor=True,
        )
        self.assertTrue(
            resolve_sandbox_network_mode(
                config, runtime_name="runc", requested=True
            )
        )
        self.assertFalse(
            resolve_sandbox_network_mode(
                config, runtime_name="runc", requested=False
            )
        )

    def test_omitted_value_defaults_to_isolated_when_networking_is_enabled(self) -> None:
        isolated = EngineConfig(
            runtime="runc",
            enable_sandbox_network=True,
            enable_interceptor=False,
            enable_egress_proxy=False,
        )
        host = EngineConfig(
            runtime="runc",
            enable_sandbox_network=False,
            enable_interceptor=False,
            enable_egress_proxy=False,
        )
        self.assertTrue(
            resolve_sandbox_network_mode(
                isolated, runtime_name="runc", requested=None
            )
        )
        self.assertFalse(
            resolve_sandbox_network_mode(host, runtime_name="runc", requested=None)
        )
        self.assertFalse(
            resolve_sandbox_network_mode(
                isolated, runtime_name="docker", requested=None
            )
        )

    def test_non_boolean_value_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "true, false, or null"):
            resolve_sandbox_network_mode(
                EngineConfig(), runtime_name="runc", requested="true"  # type: ignore[arg-type]
            )

    def test_packaged_config_defaults_runc_sandboxes_to_isolated(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "config" / "crab.yaml"
        config = EngineConfig.from_file(config_path)

        self.assertTrue(config.enable_sandbox_network)
        self.assertTrue(
            resolve_sandbox_network_mode(
                config, runtime_name="runc", requested=None
            )
        )


class NetworkLeaseRollbackTests(unittest.TestCase):
    def test_partial_setup_failure_removes_owned_plumbing_and_recycles_ip(self) -> None:
        manager = BenchmarkNetworkManager()
        manager._bridge_name = "crab-test-bridge"
        commands: list[tuple[str, ...]] = []

        def run(command, **kwargs):
            _ = kwargs
            commands.append(tuple(command))
            if command[:3] == ["ip", "link", "set"] and "master" in command:
                raise subprocess.CalledProcessError(1, command)
            return subprocess.CompletedProcess(command, 0, "", "")

        with patch(
            "integrations.sandboxes.runtime.network.subprocess.run",
            side_effect=run,
        ):
            with self.assertRaises(subprocess.CalledProcessError):
                manager.allocate_lease(SandboxId("sbx-fail"))

        self.assertEqual(manager._leases, {})
        self.assertEqual(manager._free_ip_indices, [2])
        self.assertTrue(
            any(command[:3] == ("ip", "link", "delete") for command in commands)
        )
        self.assertTrue(
            any(command[:3] == ("ip", "netns", "del") for command in commands)
        )

    def test_veth_name_collision_does_not_delete_someone_elses_link(self) -> None:
        manager = BenchmarkNetworkManager()
        manager._bridge_name = "crab-test-bridge"
        commands: list[tuple[str, ...]] = []

        def run(command, **kwargs):
            _ = kwargs
            commands.append(tuple(command))
            if command[:3] == ["ip", "link", "add"]:
                raise subprocess.CalledProcessError(2, command)
            return subprocess.CompletedProcess(command, 0, "", "")

        with patch(
            "integrations.sandboxes.runtime.network.subprocess.run",
            side_effect=run,
        ):
            with self.assertRaises(subprocess.CalledProcessError):
                manager.allocate_lease(SandboxId("sbx-collision"))

        self.assertFalse(
            any(command[:3] == ("ip", "link", "delete") for command in commands)
        )
        self.assertTrue(
            any(command[:3] == ("ip", "netns", "del") for command in commands)
        )
        self.assertEqual(manager._free_ip_indices, [2])

    def test_partial_setup_reports_resources_when_rollback_also_fails(self) -> None:
        manager = BenchmarkNetworkManager()
        manager._bridge_name = "crab-test-bridge"

        def run(command, **kwargs):
            _ = kwargs
            if command[:3] == ["ip", "link", "set"] and "master" in command:
                raise subprocess.CalledProcessError(1, command)
            if command[:3] in (["ip", "link", "delete"], ["ip", "netns", "del"]):
                return subprocess.CompletedProcess(command, 2, "", "cleanup failed")
            return subprocess.CompletedProcess(command, 0, "", "")

        with patch(
            "integrations.sandboxes.runtime.network.subprocess.run",
            side_effect=run,
        ):
            with self.assertRaises(SandboxCreateCleanupError) as raised:
                manager.allocate_lease(SandboxId("sbx-leak"))

        error = raised.exception
        self.assertIsInstance(error.cause, subprocess.CalledProcessError)
        self.assertTrue(any(item.startswith("host-veth=") for item in error.resources))
        self.assertTrue(any(item.startswith("netns=") for item in error.resources))
        self.assertIn("manually before retrying", str(error))
        self.assertEqual(manager._free_ip_indices, [2])


if __name__ == "__main__":
    unittest.main()
