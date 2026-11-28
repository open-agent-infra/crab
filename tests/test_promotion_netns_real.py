"""Real-host end-to-end for the roadmap's known hardening gap:
**cross-netns restore with sandbox networking on**.

Since PR-D2.2 a fork runs in its own network namespace
(`retarget_bundle_network_namespace`), so both promotion paths —

- B3's commit-by-promotion (`txn.commit()` on `isolation="fork"`), and
- C4's promotion migration (`merge_processes(strategy="promote")`)

— now CRIU-dump a fork inside netns A and restore that image onto the
source's identity, whose bundle still names netns B. Every existing
promotion E2E runs with `enable_sandbox_network=False`, where the dumped
processes hold no network resources at all, so the combination below has
no coverage.

This file is forensic: it asserts the behavior a networked deployment
would need (the promoted process survives, its sockets still work, the
source's network is usable afterwards) so that a failure names the
mechanism that breaks. Four workloads, ordered by how much of the
netns they capture:

1. no network resources — the baseline that already works unnetworked;
2. a listening socket bound to `0.0.0.0` — a bind address that is valid
   in any netns;
3. the same listener bound to the fork's own `eth0` address — which
   literally does not exist in the source's netns;
4. an established TCP connection to a host-side service — dumped under
   `--tcp-established` (the runtime default).

The interceptor and the egress proxy stay off on purpose: the variable
under test is the network namespace, not D1's redirect. Because
`Sandbox._requires_network_namespace()` keys off those two features, the
sandboxes here pass `network=True` explicitly, and every test asserts
the bridge lease exists before trusting a result — a host-networked
sandbox holds no netns and would pass vacuously.

Self-skipping outside the crab-dev VM.
"""
from __future__ import annotations

import os
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path

from crab import Engine, EngineConfig, Sandbox
from crab.ids import SandboxId

_LISTEN_PORT = 18080


def _real_stack_available() -> bool:
    if os.geteuid() != 0:
        return False
    tools = ("docker", "runc", "criu", "zfs", "iptables", "ip")
    return all(shutil.which(tool) is not None for tool in tools)


def _host_lan_ip() -> str | None:
    """An address of this host a sandbox on the bridge can reach which is
    not the bridge IP itself (same helper as the D1 egress E2E)."""
    result = subprocess.run(
        ["ip", "-4", "-o", "addr", "show", "scope", "global"],
        capture_output=True,
        text=True,
        check=False,
    )
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) < 4 or fields[1].startswith(("acb", "lo", "vh")):
            continue
        return fields[3].split("/")[0]
    return None


class _HoldingTCPServer:
    """Accepts connections and holds them open, so a dumped sandbox can
    own a genuinely established flow across the promotion."""

    def __init__(self) -> None:
        self._sock = socket.socket()
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("0.0.0.0", 0))
        self._sock.listen(8)
        self.port = self._sock.getsockname()[1]
        self._held: list[socket.socket] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return
            self._held.append(conn)

    @property
    def connections(self) -> int:
        return len(self._held)

    def close(self) -> None:
        self._stop.set()
        for conn in self._held:
            try:
                conn.close()
            except OSError:
                pass
        self._sock.close()
        self._thread.join(timeout=2.0)


# A daemon that binds, listens, and idles. `bind_host` is resolved inside
# the sandbox so the eth0 variant names the fork's own address. The UDP
# connect sends nothing — it just asks the kernel which source address the
# default route would use, so no bridge address is hardcoded here.
_LISTENER_SCRIPT = """
import socket, sys, time
bind_host = sys.argv[1]
if bind_host == "eth0":
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    probe.connect(("8.8.8.8", 9))
    bind_host = probe.getsockname()[0]
    probe.close()
srv = socket.socket()
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind((bind_host, int(sys.argv[2])))
srv.listen(4)
open("/probe/listen.addr", "w").write(bind_host)
while True:
    time.sleep(1)
"""

# Opens a connection to the host service and keeps it open.
_CLIENT_SCRIPT = """
import socket, sys, time
conn = socket.create_connection((sys.argv[1], int(sys.argv[2])), timeout=10)
open("/probe/client.local", "w").write("%s:%d" % conn.getsockname())
while True:
    time.sleep(1)
"""

_ESTABLISHED_PROBE = (
    "python3 -c \"import sys;"
    "rows=open('/proc/net/tcp').read().splitlines()[1:];"
    "print(sum(1 for r in rows if r.split()[3]=='01'))\""
)


class _PromotionNetnsMixin:
    """Scenarios shared by the two promotion paths; concrete classes bind
    `_promote_from_fork` to B3's commit or C4's merge."""

    _IMAGE = "python:3.11-slim"

    def setUp(self) -> None:
        if not _real_stack_available():
            self.skipTest("docker/runc/criu/zfs/iptables/root not available")
        self.host_ip = _host_lan_ip()
        if self.host_ip is None:
            self.skipTest("no global-scope host address for the external service")
        self._tmp = tempfile.TemporaryDirectory(prefix="crab_promo_netns_e2e_")
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def _engine(self) -> Engine:
        engine = Engine.start(
            EngineConfig(
                runtime="runc",
                enable_sandbox_network=True,
                enable_interceptor=False,
                enable_egress_proxy=False,
                storage_root=self.root / "storage",
                runtime_root=self.root / "runtime",
                host_inspector_launch_mode="thread",
            )
        )
        self.addCleanup(engine.stop)
        return engine

    def _networked_sandbox(self, engine: Engine) -> Sandbox:
        sandbox = Sandbox(image=self._IMAGE, network=True, engine=engine)
        self.addCleanup(sandbox.kill)
        # Without a bridge lease the sandbox shares the host network, the
        # dumped processes hold no netns-local resources, and every
        # assertion below would pass without testing anything.
        self.assertIsNotNone(
            engine._network_manager.lease_for(sandbox.sandbox_id),
            "source sandbox has no bridge lease; the netns variable is absent",
        )
        self._run(sandbox, "mkdir -p /probe")
        return sandbox

    def _run(self, target, script: str) -> str:
        """`target` is either a Sandbox (`.commands.run`) or a Transaction
        (`.exec`, which routes the command to the fork). Both return the
        same result shape, so the workloads stay path-agnostic."""
        commands = getattr(target, "commands", None)
        result = commands.run(script) if commands is not None else target.exec(script)
        self.assertEqual(result.returncode, 0, msg=f"command failed: {script!r}: {result.stderr}")
        return result.stdout.strip()

    def _write_script(self, target, path: str, body: str) -> None:
        import base64

        blob = base64.b64encode(body.encode()).decode()
        self._run(target, f"python3 -c \"import base64,sys;open('{path}','wb')"
                          f".write(base64.b64decode('{blob}'))\"")

    def _start_daemon(self, target, script_path: str, args: str, pidfile: str) -> None:
        self._run(
            target,
            f"sh -c 'nohup python3 {script_path} {args} >/dev/null 2>&1 & echo $! > {pidfile}'",
        )
        # Let the child reach its bind/connect before the dump.
        time.sleep(2.0)

    def _assert_alive(self, sandbox: Sandbox, pidfile: str, label: str) -> None:
        self.assertEqual(
            self._run(sandbox, f"test -d /proc/$(cat {pidfile}) && echo alive || echo gone"),
            "alive",
            f"{label} did not survive the promotion",
        )

    def _assert_network_usable(self, sandbox: Sandbox, port: int) -> None:
        """The promoted source must still reach the outside world — this is
        what `_fork_txn_lease_repair` exists to guarantee."""
        self.assertEqual(
            self._run(
                sandbox,
                "python3 -c \"import socket;"
                f"s=socket.create_connection(('{self.host_ip}',{port}),timeout=10);"
                "print('reachable');s.close()\"",
            ),
            "reachable",
        )

    # --- the four workloads ---------------------------------------------

    def test_promotion_without_network_resources(self) -> None:
        """Baseline: a netns exists but the dumped process holds nothing
        from it. Isolates 'networking on' from 'network resources dumped'."""
        engine = self._engine()
        sandbox = self._networked_sandbox(engine)
        holder = _HoldingTCPServer()
        self.addCleanup(holder.close)

        def workload(target) -> None:
            self._run(
                target,
                "sh -c 'nohup sleep 300 >/dev/null 2>&1 & echo $! > /probe/bg.pid'",
            )

        self._promote_from_fork(engine, sandbox, workload)

        self._assert_alive(sandbox, "/probe/bg.pid", "background sleep")
        self._assert_network_usable(sandbox, holder.port)

    def test_promotion_with_wildcard_listening_socket(self) -> None:
        """A listener on 0.0.0.0: the bind address is netns-independent,
        so this isolates 'a socket was dumped' from 'its address is gone'."""
        engine = self._engine()
        sandbox = self._networked_sandbox(engine)

        def workload(target) -> None:
            self._write_script(target, "/probe/listen.py", _LISTENER_SCRIPT)
            self._start_daemon(
                target, "/probe/listen.py", f"0.0.0.0 {_LISTEN_PORT}", "/probe/listen.pid"
            )

        self._promote_from_fork(engine, sandbox, workload)

        self._assert_alive(sandbox, "/probe/listen.pid", "wildcard listener")
        self.assertEqual(
            self._run(
                sandbox,
                "python3 -c \"import socket;"
                f"s=socket.create_connection(('127.0.0.1',{_LISTEN_PORT}),timeout=5);"
                "print('accepting');s.close()\"",
            ),
            "accepting",
            "the promoted listener no longer accepts connections",
        )

    def test_promotion_with_socket_bound_to_fork_address(self) -> None:
        """The harsh case: the socket is bound to the *fork's* eth0
        address, which does not exist in the source's netns. If promotion
        keeps the source's own netns, CRIU has no address to restore onto."""
        engine = self._engine()
        sandbox = self._networked_sandbox(engine)
        holder = _HoldingTCPServer()
        self.addCleanup(holder.close)

        def workload(target) -> None:
            self._write_script(target, "/probe/listen.py", _LISTENER_SCRIPT)
            self._start_daemon(
                target, "/probe/listen.py", f"eth0 {_LISTEN_PORT}", "/probe/listen.pid"
            )
            self.assertTrue(
                self._run(target, "cat /probe/listen.addr"),
                "the fork listener never recorded its bind address",
            )

        self._promote_from_fork(engine, sandbox, workload)

        self._assert_alive(sandbox, "/probe/listen.pid", "fork-address listener")
        promoted_addr = self._run(sandbox, "cat /probe/listen.addr")
        source_addr = self._run(
            sandbox,
            "python3 -c \"import socket;"
            "p=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);"
            "p.connect(('8.8.8.8',9));print(p.getsockname()[0])\"",
        )
        # Whichever way the fix goes, these must agree after the swap:
        # either the source adopted the fork's address (lease transfer) or
        # the socket was rebound. A mismatch is the bug this file hunts.
        self.assertEqual(
            promoted_addr,
            source_addr,
            "the promoted listener is bound to an address the sandbox no longer owns",
        )
        self.assertEqual(
            self._run(
                sandbox,
                "python3 -c \"import socket;"
                f"s=socket.create_connection(('{promoted_addr}',{_LISTEN_PORT}),timeout=5);"
                "print('accepting');s.close()\"",
            ),
            "accepting",
        )
        # The adopted netns is not just self-consistent: its default route
        # still reaches outside the sandbox after the swap.
        self._assert_network_usable(sandbox, holder.port)

    def test_promotion_with_established_connection(self) -> None:
        """An established flow to a host service, dumped under
        `--tcp-established` (the runtime default). Its local address is the
        fork's; the peer is outside the sandbox entirely."""
        engine = self._engine()
        sandbox = self._networked_sandbox(engine)
        holder = _HoldingTCPServer()
        self.addCleanup(holder.close)

        def workload(target) -> None:
            self._write_script(target, "/probe/client.py", _CLIENT_SCRIPT)
            self._start_daemon(
                target,
                "/probe/client.py",
                f"{self.host_ip} {holder.port}",
                "/probe/client.pid",
            )
            self.assertEqual(
                self._run(target, _ESTABLISHED_PROBE),
                "1",
                "the fork never established the connection under test",
            )

        self._promote_from_fork(engine, sandbox, workload)

        self._assert_alive(sandbox, "/probe/client.pid", "connected client")
        self.assertEqual(
            self._run(sandbox, _ESTABLISHED_PROBE),
            "1",
            "the established connection did not survive the promotion",
        )
        # A fresh outbound connection also works, proving the adopted netns
        # has a live egress path and not merely a preserved socket.
        self._assert_network_usable(sandbox, holder.port)


class CommitPromotionNetnsRealTests(_PromotionNetnsMixin, unittest.TestCase):
    """B3: `txn.commit()` on a fork-backed transaction."""

    def _promote_from_fork(self, engine: Engine, sandbox: Sandbox, workload) -> None:
        txn = sandbox.begin(isolation="fork")
        fork_id = txn.fork_sandbox_id
        self.assertTrue(fork_id)
        # `fork_sandbox_id` rides the txn description as a plain string;
        # the lease table is keyed by SandboxId (a dataclass, not a str
        # subclass), so an unwrapped lookup silently misses.
        self.assertIsNotNone(
            engine._network_manager.lease_for(SandboxId(fork_id)),
            "the fork holds no lease of its own; D2.2's netns retarget is inactive",
        )
        workload(txn)
        result = txn.commit()
        self.assertTrue(result.promoted_checkpoint_id)


class MergePromotionNetnsRealTests(_PromotionNetnsMixin, unittest.TestCase):
    """C4: `merge_processes(strategy="promote")`."""

    def _promote_from_fork(self, engine: Engine, sandbox: Sandbox, workload) -> None:
        [fork] = sandbox.fork()
        self.addCleanup(fork.kill)
        self.assertIsNotNone(
            engine._network_manager.lease_for(fork.sandbox_id),
            "the fork holds no lease of its own; D2.2's netns retarget is inactive",
        )
        workload(fork)
        report = sandbox.merge_processes(fork, strategy="promote", force=True)
        self.assertEqual(report.strategy, "promote")


if __name__ == "__main__":
    unittest.main()
