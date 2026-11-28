from __future__ import annotations

import ipaddress
import heapq
import logging
import os
from dataclasses import dataclass, replace
from pathlib import Path
import subprocess
import threading
import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from crab.ids import SandboxId

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
        self._egress_redirect_port: int | None = None

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

    def enable_egress_redirect(self, proxy_port: int) -> None:
        """Redirect all sandbox-originated TCP egress into the host-side
        egress proxy (roadmap D1). Traffic aimed at the host itself is
        excluded, so the LLM interceptor/forwarder/daemon paths stay
        byte-identical — that exclusion is what makes "LLM interception
        unchanged" a property of the rule rather than a hope.

        Idempotent; must be called after ``ensure_bridge``.
        """
        with self._lock:
            bridge_name = self._bridge_name
            if bridge_name is None:
                raise RuntimeError("egress redirect requires the bridge (call ensure_bridge first)")
            if self._egress_redirect_port == int(proxy_port):
                return
            rule = self._egress_redirect_rule(bridge_name, int(proxy_port))
            if subprocess.run(
                ["iptables", "-t", "nat", "-C", *rule],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode != 0:
                subprocess.run(["iptables", "-t", "nat", "-A", *rule], check=True)
            self._egress_redirect_port = int(proxy_port)
        logger.info(
            "Enabled egress redirect bridge=%s proxy_port=%d (host-bound traffic excluded)",
            bridge_name,
            int(proxy_port),
        )

    def disable_egress_redirect(self) -> None:
        with self._lock:
            bridge_name = self._bridge_name
            port = self._egress_redirect_port
            self._egress_redirect_port = None
        if bridge_name is None or port is None:
            return
        subprocess.run(
            ["iptables", "-t", "nat", "-D", *self._egress_redirect_rule(bridge_name, port)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def _egress_redirect_rule(self, bridge_name: str, proxy_port: int) -> list[str]:
        return [
            "PREROUTING",
            "-i",
            bridge_name,
            "-p",
            "tcp",
            "!",
            "-d",
            self._bridge_ip,
            "-j",
            "REDIRECT",
            "--to-ports",
            str(proxy_port),
        ]

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
            namespace_created = False
            veth_created = False
            try:
                subprocess.run(["ip", "netns", "add", namespace_name], check=True)
                namespace_created = True
                subprocess.run(
                    ["ip", "link", "add", host_veth_name, "type", "veth", "peer", "name", guest_veth_name],
                    check=True,
                )
                veth_created = True
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
            except Exception as exc:
                # The lease is published only after every setup command has
                # succeeded, so outer cleanup cannot see a partial lease. Tear
                # down only resources this attempt positively created: an
                # `ip link add` collision must never delete another lease's
                # interface.
                cleanup_errors: list[str] = []
                if veth_created:
                    try:
                        result = subprocess.run(
                            ["ip", "link", "delete", host_veth_name],
                            check=False,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                        if result.returncode != 0:
                            cleanup_errors.append(
                                f"delete host veth {host_veth_name}: exit {result.returncode}"
                            )
                    except Exception as cleanup_exc:
                        cleanup_errors.append(
                            f"delete host veth {host_veth_name}: {cleanup_exc}"
                        )
                if namespace_created:
                    try:
                        result = subprocess.run(
                            ["ip", "netns", "del", namespace_name],
                            check=False,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                        if result.returncode != 0:
                            cleanup_errors.append(
                                f"delete network namespace {namespace_name}: exit {result.returncode}"
                            )
                    except Exception as cleanup_exc:
                        cleanup_errors.append(
                            f"delete network namespace {namespace_name}: {cleanup_exc}"
                        )
                heapq.heappush(self._free_ip_indices, guest_index)
                if cleanup_errors:
                    from crab.errors import SandboxCreateCleanupError

                    leaked_resources: list[str] = []
                    if veth_created:
                        leaked_resources.append(f"host-veth={host_veth_name}")
                    if namespace_created:
                        leaked_resources.append(f"netns={namespace_name}")
                    raise SandboxCreateCleanupError(
                        str(sandbox_id),
                        exc,
                        cleanup_errors,
                        resources=tuple(leaked_resources),
                    ) from exc
                raise
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

    def transfer_lease(
        self, from_sandbox_id: SandboxId, to_sandbox_id: SandboxId
    ) -> BenchmarkNetworkLease | None:
        """Move ``from``'s lease onto ``to``'s identity, destroying whatever
        lease ``to`` held before. Returns the re-keyed lease, or ``None``
        when ``from`` holds none (which makes a repeated call a no-op).

        Promotion (B3 commit, C4 promote) restores a fork's CRIU image onto
        the source's identity. The image's sockets are bound to the *fork's*
        guest IP, so unless that address moves with them CRIU's ``soccr``
        fails to bind them back (``EADDRNOTAVAIL``) and the restore dies.

        Ordering is load-bearing: `release_lease` pops by sandbox id and
        then tears down whatever it popped, so the outgoing record must
        leave the table *before* the incoming one takes its key. Re-keying
        first and releasing afterwards would destroy the netns just
        transferred in.
        """
        with self._lock:
            incoming = self._leases.get(from_sandbox_id)
            if incoming is None:
                return None
            # 1. the outgoing record leaves the table first.
            outgoing = self._leases.pop(to_sandbox_id, None)
            if outgoing is not None:
                self._ip_to_sandbox.pop(outgoing.guest_ip, None)
                self._recycle_guest_ip_locked(outgoing.guest_ip)
            # 2. re-key the incoming record onto the target identity.
            del self._leases[from_sandbox_id]
            transferred = replace(incoming, sandbox_id=to_sandbox_id)
            self._leases[to_sandbox_id] = transferred
            # 3. attribution follows the address.
            self._ip_to_sandbox[transferred.guest_ip] = to_sandbox_id
        # 4. only the popped record's plumbing is destroyed, outside the lock.
        if outgoing is not None:
            self._destroy_lease_plumbing(outgoing)
        logger.info(
            "Transferred benchmark network lease from=%s to=%s guest_ip=%s namespace=%s",
            from_sandbox_id,
            to_sandbox_id,
            transferred.guest_ip,
            transferred.namespace_name,
        )
        return transferred

    def release_lease(self, sandbox_id: SandboxId) -> None:
        with self._lock:
            lease = self._leases.pop(sandbox_id, None)
            if lease is not None:
                self._ip_to_sandbox.pop(lease.guest_ip, None)
                self._recycle_guest_ip_locked(lease.guest_ip)
        if lease is None:
            return
        self._destroy_lease_plumbing(lease)
        logger.info(
            "Released benchmark network lease sandbox=%s guest_ip=%s namespace=%s",
            sandbox_id,
            lease.guest_ip,
            lease.namespace_name,
        )

    def _recycle_guest_ip_locked(self, guest_ip: str) -> None:
        """Return a freed address to the allocation pool. Caller holds the lock."""
        network = ipaddress.ip_network(self._network_cidr, strict=False)
        try:
            guest_index = int(ipaddress.ip_address(guest_ip)) - int(network.network_address)
        except ValueError:
            guest_index = -1
        if 2 <= guest_index < network.num_addresses - 1:
            heapq.heappush(self._free_ip_indices, guest_index)

    @staticmethod
    def _destroy_lease_plumbing(lease: BenchmarkNetworkLease) -> None:
        subprocess.run(["ip", "netns", "del", lease.namespace_name], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["ip", "link", "delete", lease.host_veth_name], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def cleanup(self) -> None:
        self.disable_egress_redirect()
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
