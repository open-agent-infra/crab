from __future__ import annotations

import ipaddress
import heapq
import logging
import os
from dataclasses import dataclass
from pathlib import Path
import subprocess
import threading
import uuid

from crab import SandboxId

logger = logging.getLogger(__name__)

_DEFAULT_BENCHMARK_NETWORK_CIDR = "10.250.0.0/24"
_DEFAULT_BENCHMARK_NETWORK_PREFIXLEN = 24
_SMALLEST_SUPPORTED_BENCHMARK_NETWORK_PREFIXLEN = 30
_DEFAULT_BENCHMARK_GUEST_CAPACITY = 253


def parse_ipv4_route_networks(raw_routes: str) -> list[ipaddress.IPv4Network]:
    networks: list[ipaddress.IPv4Network] = []
    for raw_line in raw_routes.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        destination = line.split()[0]
        if destination == "default":
            continue
        try:
            network = ipaddress.ip_network(destination, strict=False)
        except ValueError:
            continue
        if isinstance(network, ipaddress.IPv4Network) and network.prefixlen < 32:
            networks.append(network)
    return networks


def benchmark_network_guest_capacity(network: ipaddress.IPv4Network) -> int:
    if network.prefixlen > _SMALLEST_SUPPORTED_BENCHMARK_NETWORK_PREFIXLEN:
        return 0
    return max(0, network.num_addresses - 3)


def benchmark_network_prefixlen_for_guest_capacity(required_guest_capacity: int) -> int:
    if required_guest_capacity <= 0:
        raise ValueError(
            f"required_guest_capacity must be positive, got {required_guest_capacity}"
        )
    for prefixlen in range(_SMALLEST_SUPPORTED_BENCHMARK_NETWORK_PREFIXLEN, -1, -1):
        network = ipaddress.ip_network(f"10.0.0.0/{prefixlen}", strict=False)
        if benchmark_network_guest_capacity(network) >= required_guest_capacity:
            return prefixlen
    raise RuntimeError(
        f"unable to determine benchmark network prefix for guest capacity {required_guest_capacity}"
    )


def select_benchmark_network(
    *,
    existing_routes: str,
    candidate_pool: str = "10.250.0.0/16",
    required_guest_capacity: int = _DEFAULT_BENCHMARK_GUEST_CAPACITY,
) -> tuple[str, str]:
    pool = ipaddress.ip_network(candidate_pool, strict=False)
    if not isinstance(pool, ipaddress.IPv4Network):
        raise ValueError(f"candidate pool must be IPv4, got {candidate_pool}")
    if pool.prefixlen > _SMALLEST_SUPPORTED_BENCHMARK_NETWORK_PREFIXLEN:
        raise ValueError(
            "candidate pool must be at most "
            f"/{_SMALLEST_SUPPORTED_BENCHMARK_NETWORK_PREFIXLEN}, got {candidate_pool}"
        )
    preferred_prefixlen = min(
        _DEFAULT_BENCHMARK_NETWORK_PREFIXLEN,
        benchmark_network_prefixlen_for_guest_capacity(required_guest_capacity),
    )
    target_prefixlen = max(pool.prefixlen, preferred_prefixlen)
    target_capacity = benchmark_network_guest_capacity(
        ipaddress.ip_network(f"10.0.0.0/{target_prefixlen}", strict=False)
    )
    if target_capacity < required_guest_capacity:
        raise ValueError(
            f"candidate pool {candidate_pool} supports at most "
            f"{benchmark_network_guest_capacity(pool)} guest sandboxes, need {required_guest_capacity}"
        )
    existing_networks = parse_ipv4_route_networks(existing_routes)
    candidates = [pool] if pool.prefixlen == target_prefixlen else list(pool.subnets(new_prefix=target_prefixlen))
    for network in candidates:
        if any(network.overlaps(existing) for existing in existing_networks):
            continue
        return str(next(network.hosts())), str(network)
    raise RuntimeError(
        f"unable to find an available benchmark /{target_prefixlen} inside {candidate_pool}"
    )


@dataclass(frozen=True)
class BenchmarkNetworkLease:
    sandbox_id: SandboxId
    namespace_name: str
    namespace_path: Path
    host_veth_name: str
    guest_veth_name: str
    guest_ip: str


class BenchmarkNetworkManager:
    def __init__(self) -> None:
        self._bridge_name: str | None = None
        self._bridge_ip = "10.250.0.1"
        self._network_cidr = _DEFAULT_BENCHMARK_NETWORK_CIDR
        self._ip_cursor = 2
        self._free_ip_indices: list[int] = []
        self._leases: dict[SandboxId, BenchmarkNetworkLease] = {}
        self._ip_to_sandbox: dict[str, SandboxId] = {}
        self._lock = threading.Lock()
        self._nat_rules_configured = False

    @property
    def bridge_ip(self) -> str:
        return self._bridge_ip

    @property
    def network_cidr(self) -> str:
        return self._network_cidr

    def configure(self, *, expected_sandboxes: int | None = None) -> None:
        required_guest_capacity = (
            _DEFAULT_BENCHMARK_GUEST_CAPACITY
            if expected_sandboxes is None
            else max(1, int(expected_sandboxes))
        )
        configured_cidr = os.environ.get("CRAB_BENCHMARK_NETWORK_CIDR", "").strip()
        if configured_cidr:
            network = ipaddress.ip_network(configured_cidr, strict=False)
            if not isinstance(network, ipaddress.IPv4Network):
                raise ValueError(f"benchmark network must be IPv4, got {configured_cidr}")
            supported_capacity = benchmark_network_guest_capacity(network)
            if supported_capacity <= 0:
                raise ValueError(
                    "benchmark network must be between /0 and "
                    f"/{_SMALLEST_SUPPORTED_BENCHMARK_NETWORK_PREFIXLEN}, got {configured_cidr}"
                )
            if supported_capacity < required_guest_capacity:
                raise ValueError(
                    f"benchmark network {configured_cidr} supports at most {supported_capacity} "
                    f"guest sandboxes, need {required_guest_capacity}"
                )
            self._network_cidr = str(network)
            self._bridge_ip = str(next(network.hosts()))
            return
        route_result = subprocess.run(
            ["ip", "-4", "route", "show", "table", "main"],
            check=True,
            capture_output=True,
            text=True,
        )
        self._bridge_ip, self._network_cidr = select_benchmark_network(
            existing_routes=route_result.stdout,
            required_guest_capacity=required_guest_capacity,
        )

    def ensure_bridge(self) -> None:
        with self._lock:
            if self._bridge_name is not None:
                return
            network = ipaddress.ip_network(self._network_cidr, strict=False)
            bridge_name = f"acb{uuid.uuid4().hex[:8]}"
            subprocess.run(["ip", "link", "add", bridge_name, "type", "bridge"], check=True)
            subprocess.run(
                ["ip", "addr", "add", f"{self._bridge_ip}/{network.prefixlen}", "dev", bridge_name],
                check=True,
            )
            subprocess.run(["ip", "link", "set", bridge_name, "up"], check=True)
            self._bridge_name = bridge_name
            self._ensure_outbound_connectivity_rules(bridge_name)
            logger.info("Created benchmark bridge name=%s bridge_ip=%s", bridge_name, self._bridge_ip)

    def _ensure_outbound_connectivity_rules(self, bridge_name: str) -> None:
        subprocess.run(["sysctl", "-w", "net.ipv4.ip_forward=1"], check=True, stdout=subprocess.DEVNULL)
        nat_rule = ["iptables", "-t", "nat", "-C", "POSTROUTING", "-s", self._network_cidr, "-j", "MASQUERADE"]
        if subprocess.run(nat_rule, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0:
            subprocess.run(
                ["iptables", "-t", "nat", "-A", "POSTROUTING", "-s", self._network_cidr, "-j", "MASQUERADE"],
                check=True,
            )
        forward_from_bridge = ["iptables", "-C", "FORWARD", "-i", bridge_name, "-j", "ACCEPT"]
        if subprocess.run(forward_from_bridge, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0:
            subprocess.run(["iptables", "-A", "FORWARD", "-i", bridge_name, "-j", "ACCEPT"], check=True)
        forward_to_bridge = [
            "iptables",
            "-C",
            "FORWARD",
            "-o",
            bridge_name,
            "-m",
            "conntrack",
            "--ctstate",
            "RELATED,ESTABLISHED",
            "-j",
            "ACCEPT",
        ]
        if subprocess.run(forward_to_bridge, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0:
            subprocess.run(
                [
                    "iptables",
                    "-A",
                    "FORWARD",
                    "-o",
                    bridge_name,
                    "-m",
                    "conntrack",
                    "--ctstate",
                    "RELATED,ESTABLISHED",
                    "-j",
                    "ACCEPT",
                ],
                check=True,
            )
        self._nat_rules_configured = True

    def allocate_lease(self, sandbox_id: SandboxId) -> BenchmarkNetworkLease:
        self.ensure_bridge()
        with self._lock:
            assert self._bridge_name is not None
            network = ipaddress.ip_network(self._network_cidr, strict=False)
            if self._free_ip_indices:
                guest_index = heapq.heappop(self._free_ip_indices)
            else:
                if self._ip_cursor >= network.num_addresses - 1:
                    raise RuntimeError(
                        f"benchmark network exhausted guest IP capacity "
                        f"cidr={self._network_cidr} max_guests={benchmark_network_guest_capacity(network)}"
                    )
                guest_index = self._ip_cursor
                self._ip_cursor += 1
            guest_ip = str(network[guest_index])
            suffix = uuid.uuid4().hex[:8]
            namespace_name = f"ts-{suffix}"
            # Use the full 8-char suffix (not suffix[:6]) so the veth-name keyspace
            # matches the namespace keyspace. Truncating to 6 hex chars collapsed
            # the keyspace to ~16.7M, and at ~5k concurrent leases birthday
            # collisions became >50% likely, triggering spurious
            # `ip link add … File exists` failures. vh+8 = 10 chars, well within
            # IFNAMSIZ-1 (15).
            host_veth_name = f"vh{suffix}"
            guest_veth_name = f"vg{suffix}"
            subprocess.run(["ip", "netns", "add", namespace_name], check=True)
            subprocess.run(
                ["ip", "link", "add", host_veth_name, "type", "veth", "peer", "name", guest_veth_name],
                check=True,
            )
            subprocess.run(["ip", "link", "set", host_veth_name, "master", self._bridge_name], check=True)
            subprocess.run(["ip", "link", "set", host_veth_name, "up"], check=True)
            subprocess.run(["ip", "link", "set", guest_veth_name, "netns", namespace_name], check=True)
            subprocess.run(["ip", "netns", "exec", namespace_name, "ip", "link", "set", "lo", "up"], check=True)
            subprocess.run(
                ["ip", "netns", "exec", namespace_name, "ip", "link", "set", guest_veth_name, "name", "eth0"],
                check=True,
            )
            subprocess.run(
                [
                    "ip",
                    "netns",
                    "exec",
                    namespace_name,
                    "ip",
                    "addr",
                    "add",
                    f"{guest_ip}/{network.prefixlen}",
                    "dev",
                    "eth0",
                ],
                check=True,
            )
            subprocess.run(["ip", "netns", "exec", namespace_name, "ip", "link", "set", "eth0", "up"], check=True)
            subprocess.run(
                ["ip", "netns", "exec", namespace_name, "ip", "route", "replace", "default", "via", self._bridge_ip],
                check=True,
            )
            lease = BenchmarkNetworkLease(
                sandbox_id=sandbox_id,
                namespace_name=namespace_name,
                namespace_path=Path("/var/run/netns") / namespace_name,
                host_veth_name=host_veth_name,
                guest_veth_name=guest_veth_name,
                guest_ip=guest_ip,
            )
            self._leases[sandbox_id] = lease
            logger.info(
                "Allocated benchmark network lease sandbox=%s guest_ip=%s namespace=%s",
                sandbox_id,
                guest_ip,
                namespace_name,
            )
            return lease

    def register_guest_ip(self, guest_ip: str, sandbox_id: SandboxId) -> None:
        with self._lock:
            self._ip_to_sandbox[guest_ip] = sandbox_id

    def resolve_sandbox_id(self, client_host: str | None) -> SandboxId | None:
        if client_host is None:
            return None
        with self._lock:
            return self._ip_to_sandbox.get(client_host)

    def lease_for(self, sandbox_id: SandboxId) -> BenchmarkNetworkLease | None:
        with self._lock:
            return self._leases.get(sandbox_id)

    def repair_lease(self, sandbox_id: SandboxId) -> bool:
        with self._lock:
            lease = self._leases.get(sandbox_id)
            bridge_name = self._bridge_name
            network_prefixlen = ipaddress.ip_network(self._network_cidr, strict=False).prefixlen
        if lease is None or bridge_name is None:
            return False
        try:
            self._ensure_outbound_connectivity_rules(bridge_name)
            subprocess.run(["ip", "link", "set", lease.host_veth_name, "up"], check=True)
            subprocess.run(["ip", "link", "set", lease.host_veth_name, "master", bridge_name], check=True)
            subprocess.run(["ip", "netns", "exec", lease.namespace_name, "ip", "link", "set", "lo", "up"], check=True)
            subprocess.run(["ip", "netns", "exec", lease.namespace_name, "ip", "link", "set", "eth0", "up"], check=True)
            subprocess.run(
                [
                    "ip",
                    "netns",
                    "exec",
                    lease.namespace_name,
                    "ip",
                    "addr",
                    "replace",
                    f"{lease.guest_ip}/{network_prefixlen}",
                    "dev",
                    "eth0",
                ],
                check=True,
            )
            subprocess.run(
                ["ip", "netns", "exec", lease.namespace_name, "ip", "route", "replace", "default", "via", self._bridge_ip],
                check=True,
            )
        except subprocess.CalledProcessError:
            logger.warning(
                "Failed to repair benchmark network lease sandbox=%s guest_ip=%s namespace=%s",
                sandbox_id,
                lease.guest_ip,
                lease.namespace_name,
                exc_info=True,
            )
            return False
        logger.info(
            "Repaired benchmark network lease sandbox=%s guest_ip=%s namespace=%s",
            sandbox_id,
            lease.guest_ip,
            lease.namespace_name,
        )
        return True

    def release_lease(self, sandbox_id: SandboxId) -> None:
        with self._lock:
            lease = self._leases.pop(sandbox_id, None)
            if lease is not None:
                self._ip_to_sandbox.pop(lease.guest_ip, None)
                network = ipaddress.ip_network(self._network_cidr, strict=False)
                try:
                    guest_index = int(ipaddress.ip_address(lease.guest_ip)) - int(network.network_address)
                except ValueError:
                    guest_index = -1
                if 2 <= guest_index < network.num_addresses - 1:
                    heapq.heappush(self._free_ip_indices, guest_index)
        if lease is None:
            return
        subprocess.run(["ip", "netns", "del", lease.namespace_name], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["ip", "link", "delete", lease.host_veth_name], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        logger.info(
            "Released benchmark network lease sandbox=%s guest_ip=%s namespace=%s",
            sandbox_id,
            lease.guest_ip,
            lease.namespace_name,
        )

    def cleanup(self) -> None:
        with self._lock:
            sandbox_ids = list(self._leases)
            bridge_name = self._bridge_name
        for sandbox_id in sandbox_ids:
            self.release_lease(sandbox_id)
        if bridge_name is not None:
            if self._nat_rules_configured:
                subprocess.run(
                    ["iptables", "-t", "nat", "-D", "POSTROUTING", "-s", self._network_cidr, "-j", "MASQUERADE"],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                subprocess.run(
                    ["iptables", "-D", "FORWARD", "-i", bridge_name, "-j", "ACCEPT"],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                subprocess.run(
                    [
                        "iptables",
                        "-D",
                        "FORWARD",
                        "-o",
                        bridge_name,
                        "-m",
                        "conntrack",
                        "--ctstate",
                        "RELATED,ESTABLISHED",
                        "-j",
                        "ACCEPT",
                    ],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                self._nat_rules_configured = False
            subprocess.run(["ip", "link", "delete", bridge_name], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            with self._lock:
                if self._bridge_name == bridge_name:
                    self._bridge_name = None
