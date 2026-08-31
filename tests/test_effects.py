"""Unit tests for D3.1: the effect gate — policy decisions, the defer
allow-list, the queue, and the proxy integration (202/503 shapes).
Host-runnable; the proxy is driven over loopback."""
from __future__ import annotations

import socket
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from crab.effects import (
    DECISION_DEFER,
    DEFERRED_RESPONSE,
    REJECTED_RESPONSE,
    DECISION_PASS,
    DECISION_REJECT,
    DECISION_SEAL,
    DeferredRequest,
    EffectGate,
    EffectRule,
    build_deferred_request,
    read_remaining_body,
)
from crab.egress import CassetteRecorder, EgressProxyServer, EgressRule
from crab.cassettes import CassetteStore
from crab.http_wire import parse_head
from crab.ids import CheckpointId, SandboxId
from crab.journal import ActionJournal
from crab.system import logger as _system_logger
from crab.models import EgressFlow, EgressLedger


class RuleMatchingTests(unittest.TestCase):
    def test_globs_and_method(self) -> None:
        rule = EffectRule(
            host_glob="*.internal.example", method="POST", path_glob="/events*"
        )
        self.assertTrue(
            rule.matches(host="metrics.internal.example", method="post", path="/events/1")
        )
        self.assertFalse(
            rule.matches(host="metrics.internal.example", method="PUT", path="/events/1")
        )
        self.assertFalse(
            rule.matches(host="api.example.com", method="POST", path="/events/1")
        )
        self.assertFalse(
            rule.matches(host="metrics.internal.example", method="POST", path="/other")
        )

    def test_defaults_match_everything(self) -> None:
        rule = EffectRule()
        self.assertTrue(rule.matches(host="anything", method="DELETE", path="/x"))

    def test_from_json_and_defer_false(self) -> None:
        rule = EffectRule.from_json(
            {"host_glob": "*.x", "method": "post", "path_glob": "/a*", "defer": False}
        )
        self.assertFalse(rule.defer)
        self.assertEqual(rule.method, "post")
        self.assertTrue(rule.matches(host="h.x", method="POST", path="/ab"))


class PolicyDecisionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gate = EffectGate()
        self.sandbox_id = "sbx-1"

    def _decide(self, **overrides) -> str:
        kwargs = {"host": "api.example.com", "method": "POST", "path": "/orders"}
        kwargs.update(overrides)
        decision, _ = self.gate.decide_write(self.sandbox_id, **kwargs)
        return decision

    def test_no_session_is_inert(self) -> None:
        self.assertIsNone(
            self.gate.decide_write(
                "sbx-none", host="api.example.com", method="POST", path="/x"
            )
        )
        self.assertIsNone(self.gate.decide_opaque("sbx-none"))

    def test_allow_passes(self) -> None:
        self.gate.begin(self.sandbox_id, policy="allow")
        self.assertEqual(self._decide(), DECISION_PASS)
        self.assertEqual(self.gate.counters(self.sandbox_id), (0, 0, 1, False))

    def test_reject_refuses(self) -> None:
        self.gate.begin(self.sandbox_id, policy="reject")
        self.assertEqual(self._decide(), DECISION_REJECT)
        self.assertEqual(self.gate.counters(self.sandbox_id), (0, 1, 0, False))

    def test_seal_passes_but_marks_the_txn(self) -> None:
        self.gate.begin(self.sandbox_id, policy="seal")
        self.assertFalse(self.gate.sealed(self.sandbox_id))
        self.assertEqual(self._decide(), DECISION_SEAL)
        self.assertTrue(self.gate.sealed(self.sandbox_id))

    def test_defer_needs_the_allow_list(self) -> None:
        # Empty list + defer: refused, never silently queued.
        self.gate.begin(self.sandbox_id, policy="defer")
        self.assertEqual(self._decide(), DECISION_REJECT)

        self.gate.begin(
            self.sandbox_id,
            policy="defer",
            rules=(EffectRule(host_glob="api.example.com", method="POST", path_glob="/orders*"),),
        )
        self.assertEqual(self._decide(), DECISION_DEFER)
        # A different endpoint still falls to on_unlisted.
        self.assertEqual(self._decide(path="/refunds"), DECISION_REJECT)

    def test_on_unlisted_allow(self) -> None:
        self.gate.begin(self.sandbox_id, policy="defer", on_unlisted="allow")
        self.assertEqual(self._decide(), DECISION_PASS)

    def test_rule_can_exempt_an_endpoint_from_deferral(self) -> None:
        self.gate.begin(
            self.sandbox_id,
            policy="defer",
            rules=(EffectRule(host_glob="*", defer=False),),
        )
        self.assertEqual(self._decide(), DECISION_PASS)

    def test_first_matching_rule_wins(self) -> None:
        self.gate.begin(
            self.sandbox_id,
            policy="defer",
            rules=(
                EffectRule(host_glob="*", defer=False),
                EffectRule(host_glob="api.example.com", defer=True),
            ),
        )
        self.assertEqual(self._decide(), DECISION_PASS)

    def test_opaque_behaviors(self) -> None:
        self.gate.begin(self.sandbox_id, policy="reject", opaque_effects="allow")
        decision, _ = self.gate.decide_opaque(self.sandbox_id)
        self.assertEqual(decision, DECISION_PASS)  # HTTPS keeps working

        self.gate.begin(self.sandbox_id, policy="reject", opaque_effects="reject")
        decision, _ = self.gate.decide_opaque(self.sandbox_id)
        self.assertEqual(decision, DECISION_REJECT)

        self.gate.begin(self.sandbox_id, policy="allow", opaque_effects="seal")
        decision, _ = self.gate.decide_opaque(self.sandbox_id)
        self.assertEqual(decision, DECISION_SEAL)
        self.assertTrue(self.gate.sealed(self.sandbox_id))

    def test_validation(self) -> None:
        with self.assertRaises(ValueError):
            self.gate.begin(self.sandbox_id, policy="hold")
        with self.assertRaises(ValueError):
            self.gate.begin(self.sandbox_id, policy="defer", on_unlisted="queue")
        with self.assertRaises(ValueError):
            self.gate.begin(self.sandbox_id, policy="defer", opaque_effects="defer")

    def test_session_lifecycle(self) -> None:
        session = self.gate.begin(self.sandbox_id, policy="reject", txn_id="txn-1")
        self.assertIs(self.gate.session_for(self.sandbox_id), session)
        self.assertEqual(session.txn_id, "txn-1")
        self.assertIs(self.gate.end(self.sandbox_id), session)
        self.assertIsNone(self.gate.session_for(self.sandbox_id))
        self.assertIsNone(self.gate.end(self.sandbox_id))  # idempotent


class QueueTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gate = EffectGate()
        self.gate.begin("sbx-1", policy="defer", rules=(EffectRule(),), txn_id="txn-9")

    def _request(self, path: str) -> DeferredRequest:
        return DeferredRequest(
            method="POST", host="api.example.com", port=80, path=path,
            headers=(("Content-Type", "application/json"),), body=b"{}",
            txn_id="txn-9",
        )

    def test_enqueue_and_drain_preserve_order(self) -> None:
        self.assertEqual(self.gate.enqueue("sbx-1", self._request("/a")), 1)
        self.assertEqual(self.gate.enqueue("sbx-1", self._request("/b")), 2)
        self.assertEqual(self.gate.counters("sbx-1")[0], 2)
        drained = self.gate.drain("sbx-1")
        self.assertEqual([entry.path for entry in drained], ["/a", "/b"])
        # Draining empties the queue (commit flushes, abort discards).
        self.assertEqual(self.gate.drain("sbx-1"), [])

    def test_enqueue_without_session_is_a_noop(self) -> None:
        self.assertEqual(self.gate.enqueue("sbx-other", self._request("/a")), 0)
        self.assertEqual(self.gate.drain("sbx-other"), [])

    def test_body_digest(self) -> None:
        request = self._request("/a")
        self.assertEqual(len(request.body_sha256), 64)


class ConcurrencyTests(unittest.TestCase):
    """One thread per connection in the proxy, so the queue and counters
    are touched concurrently — the D2.2 review's lesson applied up front."""

    def test_counters_and_queue_survive_parallel_writers(self) -> None:
        gate = EffectGate()
        gate.begin("sbx-1", policy="defer", rules=(EffectRule(),))
        rounds, workers = 150, 8

        def hammer() -> None:
            for index in range(rounds):
                gate.decide_write(
                    "sbx-1", host="api.example.com", method="POST", path=f"/p{index}"
                )
                gate.enqueue(
                    "sbx-1",
                    DeferredRequest(
                        method="POST", host="api.example.com", port=80,
                        path=f"/p{index}", headers=(), body=b"",
                    ),
                )

        threads = [threading.Thread(target=hammer) for _ in range(workers)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        attempts = rounds * workers
        deferred, rejected, passed, sealed = gate.counters("sbx-1")
        # Nothing is lost and nothing is double-counted: every attempt is
        # either queued or refused by the ceiling, never dropped silently.
        self.assertEqual(deferred + rejected, attempts)
        self.assertEqual(deferred, min(attempts, gate.session_for("sbx-1").max_queue_entries))
        self.assertEqual((passed, sealed), (0, False))
        self.assertEqual(len(gate.drain("sbx-1")), deferred)


class BodyReadingTests(unittest.TestCase):
    """A deferred write must be queued whole: only the first peek reached
    the proxy, so a body that is still on the socket has to be drained
    before the request can be replayed at commit."""

    def _pair(self):
        left, right = socket.socketpair()
        self.addCleanup(left.close)
        self.addCleanup(right.close)
        return left, right

    def test_reads_the_rest_of_a_framed_body(self) -> None:
        head = parse_head(b"POST /x HTTP/1.1\r\nHost: h\r\nContent-Length: 10\r\n\r\nabc")
        left, right = self._pair()
        right.sendall(b"defghij")
        body, complete = read_remaining_body(left, head, head.rest, limit=1024)
        self.assertEqual(body, b"abcdefghij")
        self.assertTrue(complete)

    def test_already_complete_body_is_returned_as_is(self) -> None:
        head = parse_head(b"POST /x HTTP/1.1\r\nHost: h\r\nContent-Length: 3\r\n\r\nabc")
        left, _ = self._pair()
        body, complete = read_remaining_body(left, head, head.rest, limit=1024)
        self.assertEqual((body, complete), (b"abc", True))

    def test_oversized_body_is_incomplete(self) -> None:
        head = parse_head(b"POST /x HTTP/1.1\r\nHost: h\r\nContent-Length: 5000\r\n\r\nabc")
        left, _ = self._pair()
        body, complete = read_remaining_body(left, head, head.rest, limit=64)
        self.assertFalse(complete)

    def test_unframed_body_cannot_be_deferred(self) -> None:
        chunked = parse_head(
            b"POST /x HTTP/1.1\r\nHost: h\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nhello\r\n"
        )
        left, _ = self._pair()
        _, complete = read_remaining_body(left, chunked, chunked.rest, limit=1024)
        self.assertFalse(complete)
        # A bodyless request (no Content-Length, nothing buffered) is fine.
        bodyless = parse_head(b"DELETE /x HTTP/1.1\r\nHost: h\r\n\r\n")
        body, complete = read_remaining_body(left, bodyless, bodyless.rest, limit=1024)
        self.assertEqual((body, complete), (b"", True))

    def test_build_strips_credentials(self) -> None:
        head = parse_head(
            b"POST /x HTTP/1.1\r\nHost: h\r\nAuthorization: Bearer secret\r\n"
            b"Cookie: sid=1\r\nContent-Type: application/json\r\n\r\n"
        )
        request = build_deferred_request(
            parsed_head=head, body=b'{"a":1}', host="h", port=80, method="post",
            path="/x", txn_id="txn-1", enqueued_at="now",
        )
        names = {name.lower() for name, _ in request.headers}
        self.assertNotIn("authorization", names)
        self.assertNotIn("cookie", names)
        self.assertIn("content-type", names)
        self.assertEqual(request.method, "POST")
        self.assertEqual(request.body, b'{"a":1}')


class _Upstream:
    def __init__(self, response: bytes = b"HTTP/1.1 201 Created\r\nContent-Length: 2\r\n\r\nok"):
        self.response = response
        self.received: list[bytes] = []
        self._sock = socket.socket()
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(8)
        self.port = self._sock.getsockname()[1]
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self) -> None:
        while True:
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return
            with conn:
                try:
                    self.received.append(conn.recv(8192))
                    conn.sendall(self.response)
                except OSError:
                    pass

    def close(self) -> None:
        self._sock.close()


class ProxyEffectGateTests(unittest.TestCase):
    """End-to-end through the proxy: the decisive property is that a
    refused or deferred write never reaches the upstream."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="crab_effects_")
        self.addCleanup(self._tmp.cleanup)
        self.journal = ActionJournal(Path(self._tmp.name) / "journal")
        self.sandbox_id = SandboxId("sbx-effects")
        self.upstream = _Upstream()
        self.addCleanup(self.upstream.close)
        self.gate = EffectGate()
        self.proxy = EgressProxyServer(
            journal=self.journal,
            sandbox_id_resolver=lambda peer: self.sandbox_id,
            host="127.0.0.1",
            port=0,
            head_timeout_seconds=1.0,
            effect_gate=self.gate,
        )
        self.proxy.start()
        self.addCleanup(self.proxy.stop)
        patcher = mock.patch(
            "crab.egress.original_destination",
            return_value=("127.0.0.1", self.upstream.port),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _talk(self, payload: bytes) -> bytes:
        with socket.create_connection(("127.0.0.1", self.proxy.port), timeout=5.0) as sock:
            sock.sendall(payload)
            chunks = []
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                chunks.append(chunk)
        return b"".join(chunks)

    def _flows(self, *, expected: int = 1, timeout: float = 5.0) -> list[dict]:
        deadline = time.monotonic() + timeout
        rows: list = []
        while time.monotonic() < deadline:
            rows = self.journal.entries(self.sandbox_id, kind="egress")
            if len(rows) >= expected:
                break
            time.sleep(0.05)
        return [row.payload for row in rows]

    _POST = (
        b"POST /orders HTTP/1.1\r\nHost: api.example.com\r\n"
        b"Content-Length: 7\r\n\r\n{\"a\":1}"
    )

    def test_reject_never_reaches_upstream(self) -> None:
        self.gate.begin(self.sandbox_id, policy="reject", txn_id="txn-1")
        response = self._talk(self._POST)
        self.assertIn(b"503", response)
        self.assertIn(b"X-Crab-Effect: rejected", response)
        self.assertEqual(self.upstream.received, [])
        [flow] = self._flows()
        self.assertEqual(flow["effect"], "rejected")

    def test_defer_queues_the_whole_request_and_answers_202(self) -> None:
        self.gate.begin(
            self.sandbox_id, policy="defer", rules=(EffectRule(),), txn_id="txn-2"
        )
        response = self._talk(self._POST)
        self.assertIn(b"202", response)
        self.assertIn(b"X-Crab-Effect: deferred", response)
        self.assertEqual(self.upstream.received, [])  # nothing left the host

        [flow] = self._flows()
        self.assertEqual(flow["effect"], "deferred")
        self.assertEqual(flow["effect_queue_position"], 1)
        [queued] = self.gate.drain(self.sandbox_id)
        self.assertEqual(queued.method, "POST")
        self.assertEqual(queued.path, "/orders")
        self.assertEqual(queued.body, b'{"a":1}')  # body captured whole
        self.assertEqual(queued.txn_id, "txn-2")

    def test_reads_are_never_gated(self) -> None:
        self.upstream.response = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nhi"
        self.gate.begin(self.sandbox_id, policy="reject", txn_id="txn-3")
        response = self._talk(b"GET /things HTTP/1.1\r\nHost: api.example.com\r\n\r\n")
        self.assertIn(b"200", response)
        self.assertEqual(len(self.upstream.received), 1)
        [flow] = self._flows()
        self.assertIsNone(flow.get("effect"))

    def test_seal_lets_the_write_out_and_marks_the_session(self) -> None:
        self.gate.begin(self.sandbox_id, policy="seal", txn_id="txn-4")
        response = self._talk(self._POST)
        self.assertIn(b"201", response)
        self.assertEqual(len(self.upstream.received), 1)
        self.assertTrue(self.gate.sealed(self.sandbox_id))
        [flow] = self._flows()
        self.assertEqual(flow["effect"], "sealed")

    def test_no_session_behaves_exactly_as_before(self) -> None:
        response = self._talk(self._POST)
        self.assertIn(b"201", response)
        self.assertEqual(len(self.upstream.received), 1)
        [flow] = self._flows()
        self.assertIsNone(flow.get("effect"))

    def test_unqueueable_body_is_refused_not_queued(self) -> None:
        self.gate.begin(
            self.sandbox_id, policy="defer", rules=(EffectRule(),), txn_id="txn-5"
        )
        self.proxy.max_deferred_body_bytes = 4
        response = self._talk(self._POST)  # 7-byte body vs a 4-byte cap
        self.assertIn(b"503", response)
        self.assertEqual(self.upstream.received, [])
        self.assertEqual(self.gate.drain(self.sandbox_id), [])
        [flow] = self._flows()
        self.assertEqual(flow["effect"], "rejected")
        self.assertEqual(flow["effect_reason"], "unqueueable")

    def test_opaque_flow_follows_opaque_effects(self) -> None:
        self.gate.begin(self.sandbox_id, policy="reject", opaque_effects="reject")
        response = self._talk(b"\x16\x03\x01\x00\x50 not really tls")
        self.assertIn(b"503", response)
        self.assertEqual(self.upstream.received, [])


class LedgerEffectFieldsTests(unittest.TestCase):
    def test_counters_and_round_trip(self) -> None:
        def flow(seq: int, effect: str | None) -> EgressFlow:
            return EgressFlow(
                seq=seq, host="api.example.com", dst_ip="10.0.0.1", dst_port=80,
                scheme="http", classification="mutating", method="POST", path="/w",
                effect=effect,
            )

        ledger = EgressLedger(
            sandbox_id=SandboxId("sbx-1"),
            flows=(flow(1, "deferred"), flow(2, "rejected"), flow(3, "flushed"), flow(4, None)),
        )
        self.assertEqual((ledger.deferred, ledger.rejected, ledger.flushed), (1, 1, 1))
        payload = ledger.to_json()
        self.assertEqual(payload["deferred"], 1)
        self.assertEqual(EgressLedger.from_json(payload), ledger)

    def test_pre_d3_rows_have_no_effect(self) -> None:
        flow = EgressFlow.from_json(
            {"seq": 1, "host": "h", "dst_ip": "1.2.3.4", "dst_port": 80,
             "scheme": "http", "classification": "mutating"}
        )
        self.assertIsNone(flow.effect)
        self.assertIsNone(flow.effect_status)


class ConfigTests(unittest.TestCase):
    def test_defaults_and_nested_parsing(self) -> None:
        from crab.engine import EngineConfig

        cfg = EngineConfig()
        self.assertEqual(cfg.effects_default_policy, "allow")
        self.assertEqual(cfg.effects_fork_policy, "reject")
        self.assertEqual(cfg.effects_on_unlisted, "reject")
        self.assertEqual(cfg.effects_opaque_effects, "allow")
        self.assertEqual(cfg.effects_rules, ())

        parsed = EngineConfig.from_mapping(
            {
                "effects": {
                    "default_policy": "defer",
                    "fork_policy": "allow",
                    "on_unlisted": "allow",
                    "opaque_effects": "seal",
                    "rules": [{"host_glob": "*.internal", "method": "POST"}],
                }
            }
        )
        self.assertEqual(parsed.effects_default_policy, "defer")
        self.assertEqual(parsed.effects_fork_policy, "allow")
        self.assertEqual(parsed.effects_on_unlisted, "allow")
        self.assertEqual(parsed.effects_opaque_effects, "seal")
        self.assertEqual(len(parsed.effects_rules), 1)


if __name__ == "__main__":
    unittest.main()


class SyntheticResponseShapeTests(unittest.TestCase):
    """Byte-level shape of what the gate writes back. The first cut of the
    503 declared Content-Length: 41 for a 44-byte body — lenient clients
    hid it behind `Connection: close`, a strict one would mis-parse or
    hang. Both responses are now asserted header-against-body."""

    def _split(self, wire: bytes) -> tuple[dict, bytes]:
        head, _, body = wire.partition(b"\r\n\r\n")
        lines = head.split(b"\r\n")
        headers = {}
        for line in lines[1:]:
            name, _, value = line.partition(b":")
            headers[name.strip().lower()] = value.strip()
        return headers, body

    def test_declared_length_matches_the_body(self) -> None:
        for label, wire in (("deferred", DEFERRED_RESPONSE), ("rejected", REJECTED_RESPONSE)):
            with self.subTest(response=label):
                headers, body = self._split(wire)
                self.assertEqual(
                    int(headers[b"content-length"]),
                    len(body),
                    msg=f"{label}: Content-Length disagrees with the body",
                )
                self.assertEqual(headers[b"x-crab-effect"], label.encode())
                self.assertEqual(headers[b"connection"], b"close")

    def test_status_lines(self) -> None:
        self.assertTrue(DEFERRED_RESPONSE.startswith(b"HTTP/1.1 202 Accepted\r\n"))
        self.assertTrue(
            REJECTED_RESPONSE.startswith(b"HTTP/1.1 503 Service Unavailable\r\n")
        )

    def test_deferred_has_no_body(self) -> None:
        _, body = self._split(DEFERRED_RESPONSE)
        self.assertEqual(body, b"")

    def test_parsable_by_a_real_http_client(self) -> None:
        """A strict parser must accept both, and read exactly the declared
        number of bytes without blocking."""
        import http.client

        for label, wire in (("deferred", DEFERRED_RESPONSE), ("rejected", REJECTED_RESPONSE)):
            with self.subTest(response=label):
                left, right = socket.socketpair()
                self.addCleanup(left.close)
                self.addCleanup(right.close)
                right.sendall(wire)
                right.shutdown(socket.SHUT_WR)
                response = http.client.HTTPResponse(left)
                response.begin()
                body = response.read()
                self.assertEqual(len(body), int(response.getheader("Content-Length")))
                self.assertEqual(response.getheader("X-Crab-Effect"), label)


class QueueCeilingTests(unittest.TestCase):
    """A per-request body cap cannot bound a loop that posts many
    allow-listed writes, so the session carries whole-queue ceilings."""

    def _request(self, body: bytes) -> DeferredRequest:
        return DeferredRequest(
            method="POST", host="api.example.com", port=80, path="/w",
            headers=(), body=body,
        )

    def test_entry_ceiling_refuses_further_writes(self) -> None:
        gate = EffectGate()
        gate.begin("sbx-1", policy="defer", rules=(EffectRule(),), max_queue_entries=2)
        self.assertEqual(gate.enqueue("sbx-1", self._request(b"a")), 1)
        self.assertEqual(gate.enqueue("sbx-1", self._request(b"b")), 2)
        # Full: -1 tells the proxy to refuse rather than answer a fake 202.
        self.assertEqual(gate.enqueue("sbx-1", self._request(b"c")), -1)
        self.assertEqual(gate.queue_size("sbx-1"), (2, 2))
        deferred, rejected, _, _ = gate.counters("sbx-1")
        self.assertEqual((deferred, rejected), (2, 1))

    def test_byte_ceiling_refuses_further_writes(self) -> None:
        gate = EffectGate()
        gate.begin("sbx-1", policy="defer", rules=(EffectRule(),), max_queue_bytes=10)
        self.assertEqual(gate.enqueue("sbx-1", self._request(b"x" * 6)), 1)
        self.assertEqual(gate.enqueue("sbx-1", self._request(b"x" * 6)), -1)  # 12 > 10
        self.assertEqual(gate.enqueue("sbx-1", self._request(b"x" * 4)), 2)  # 10 fits
        self.assertEqual(gate.queue_size("sbx-1"), (2, 10))

    def test_draining_frees_the_budget(self) -> None:
        gate = EffectGate()
        gate.begin("sbx-1", policy="defer", rules=(EffectRule(),), max_queue_bytes=8)
        gate.enqueue("sbx-1", self._request(b"x" * 8))
        self.assertEqual(gate.enqueue("sbx-1", self._request(b"y")), -1)
        self.assertEqual(len(gate.drain("sbx-1")), 1)
        self.assertEqual(gate.queue_size("sbx-1"), (0, 0))
        self.assertEqual(gate.enqueue("sbx-1", self._request(b"y")), 1)

    def test_defaults_are_generous_but_finite(self) -> None:
        from crab.effects import DEFAULT_MAX_QUEUE_BYTES, DEFAULT_MAX_QUEUE_ENTRIES

        gate = EffectGate()
        session = gate.begin("sbx-1", policy="defer")
        self.assertEqual(session.max_queue_bytes, DEFAULT_MAX_QUEUE_BYTES)
        self.assertEqual(session.max_queue_entries, DEFAULT_MAX_QUEUE_ENTRIES)


class ProxyQueueCeilingTests(ProxyEffectGateTests):
    def test_queue_full_is_refused_not_falsely_accepted(self) -> None:
        self.gate.begin(
            self.sandbox_id,
            policy="defer",
            rules=(EffectRule(),),
            txn_id="txn-full",
            max_queue_entries=1,
        )
        first = self._talk(self._POST)
        self.assertIn(b"202", first)
        second = self._talk(self._POST)
        # The second write did not fit, so it must be refused - answering
        # 202 for something that was never queued would be a lie.
        self.assertIn(b"503", second)
        self.assertEqual(self.upstream.received, [])
        flows = self._flows(expected=2)
        self.assertEqual(flows[-1]["effect"], "rejected")
        self.assertEqual(flows[-1]["effect_reason"], "queue_full")
        self.assertEqual(len(self.gate.drain(self.sandbox_id)), 1)

    def test_session_closing_mid_flight_refuses_instead_of_faking(self) -> None:
        session_gate = self.gate

        class _RacingGate:
            """Mimics commit/abort closing the session between the decision
            and the enqueue."""

            def __getattr__(self, name):
                return getattr(session_gate, name)

            def enqueue(self, sandbox_id, request):
                return 0

        self.gate.begin(
            self.sandbox_id, policy="defer", rules=(EffectRule(),), txn_id="txn-race"
        )
        self.proxy.effect_gate = _RacingGate()
        response = self._talk(self._POST)
        self.assertIn(b"503", response)
        self.assertEqual(self.upstream.received, [])
        [flow] = self._flows()
        self.assertEqual(flow["effect_reason"], "session_closed")


# ---------------------------------------------------------------------------
# D3.2: transaction wiring
# ---------------------------------------------------------------------------

from crab.effects import flush_deferred_request
from crab.txn import EffectFlushReport, TxnNotAbortable


class _RecordingHTTPServer:
    """A real HTTP server: it frames the request properly, so a test can
    assert the queued body arrived intact (a raw one-shot recv would miss
    the body, which http.client writes as a separate segment)."""

    def __init__(self, status: int = 204) -> None:
        import http.server

        seen: list[tuple[str, str, bytes]] = []

        class _Handler(http.server.BaseHTTPRequestHandler):
            def _handle(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else b""
                seen.append((self.command, self.path, body))
                self.send_response(status)
                self.send_header("Content-Length", "0")
                self.end_headers()

            do_POST = _handle
            do_PUT = _handle

            def log_message(self, *args) -> None:
                pass

        self.seen = seen
        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5.0)


class FlushTransportTests(unittest.TestCase):
    """`flush_deferred_request` goes straight out from the host: not through
    the proxy, so it cannot be caught by the gate that queued it."""

    def setUp(self) -> None:
        self.upstream = _RecordingHTTPServer(status=204)
        self.addCleanup(self.upstream.close)

    def _request(self, **overrides) -> DeferredRequest:
        payload = {
            "method": "POST", "host": "127.0.0.1", "port": self.upstream.port,
            "path": "/orders", "headers": (("Content-Type", "application/json"),),
            "body": b'{"a":1}',
        }
        payload.update(overrides)
        return DeferredRequest(**payload)

    def test_sends_the_queued_request_verbatim(self) -> None:
        status, error = flush_deferred_request(self._request())
        self.assertEqual(status, 204)
        self.assertIsNone(error)
        [(method, path, body)] = self.upstream.seen
        self.assertEqual((method, path), ("POST", "/orders"))
        self.assertEqual(body, b'{"a":1}')  # the queued body arrived intact

    def test_unreachable_host_is_reported_not_raised(self) -> None:
        # Port 1 on loopback: nothing serves it, so the failure is genuine
        # (closing a listener still lets already-queued connections finish).
        status, error = flush_deferred_request(
            self._request(port=1), timeout=2.0
        )
        self.assertIsNone(status)
        self.assertIsInstance(error, str)
        self.assertTrue(error)


class _EffectSystemHarness(unittest.TestCase):
    """Real CrabSystem + real journal + a real EffectGate; the heavy txn
    machinery is patched at the system boundary."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="crab_d32_")
        self.addCleanup(self._tmp.cleanup)
        self.journal = ActionJournal(Path(self._tmp.name) / "journal")
        from tests.test_egress import _minimal_system

        self.system = _minimal_system(self.journal)
        self.gate = EffectGate()
        self.system.effect_gate = self.gate
        self.system.effect_policy_defaults = {
            "default_policy": "allow",
            "fork_policy": "reject",
            "on_unlisted": "reject",
            "opaque_effects": "allow",
            "rules": ({"host_glob": "*", "method": "POST"},),
        }
        self.sandbox_id = SandboxId("sbx-d32")


class PolicyResolutionTests(_EffectSystemHarness):
    def test_snapshot_and_fork_defaults(self) -> None:
        self.assertEqual(self.system._resolve_effect_policy("snapshot", None), "allow")
        self.assertEqual(self.system._resolve_effect_policy("fork", None), "reject")
        # An explicit request always wins.
        self.assertEqual(self.system._resolve_effect_policy("fork", "seal"), "seal")

    def test_arm_uses_configured_defaults_and_rules(self) -> None:
        self.system._arm_effect_session(
            self.sandbox_id, policy="defer", txn_id="txn-1", isolation="snapshot"
        )
        session = self.gate.session_for(self.sandbox_id)
        self.assertEqual(session.policy, "defer")
        self.assertEqual(session.on_unlisted, "reject")
        self.assertEqual(session.txn_id, "txn-1")
        self.assertEqual(len(session.rules), 1)
        # The dict from config became a real rule.
        self.assertTrue(
            session.rules[0].matches(host="api.example.com", method="POST", path="/x")
        )


class StandaloneForkPolicyTests(_EffectSystemHarness):
    """F1: a bare `sandbox.fork()` declares its intent. An independent
    branch (RL rollout, tree search) legitimately produces N effects from N
    forks and stays ungated; a speculative branch serves a main line and
    must produce its effect exactly once, so it opts into `reject`."""

    def setUp(self) -> None:
        super().setUp()
        self.fork_id = SandboxId("sbx-d32-fork-1")

    def test_default_is_allow_and_is_not_the_fork_txn_key(self) -> None:
        # `fork_policy` (reject) governs fork-BACKED TRANSACTIONS; a bare
        # fork reads `standalone_fork_policy`. Reading the wrong key would
        # silently gate today's freely writing forks.
        self.assertEqual(self.system.validate_standalone_fork_policy(None), "allow")
        self.assertEqual(self.system._resolve_effect_policy("fork", None), "reject")

    def test_configured_default_can_be_flipped(self) -> None:
        self.system.effect_policy_defaults["standalone_fork_policy"] = "reject"
        self.assertEqual(self.system.validate_standalone_fork_policy(None), "reject")
        # An explicit request still wins over the configured default.
        self.assertEqual(self.system.validate_standalone_fork_policy("allow"), "allow")

    def test_defer_and_seal_are_refused_not_degraded(self) -> None:
        # A bare fork has no commit to flush a queue into and no abort for a
        # seal to block; honoring either name would be a guarantee in name
        # only (F1 decision 5).
        with self.assertRaises(ValueError) as deferred:
            self.system.validate_standalone_fork_policy("defer")
        self.assertIn("defer", str(deferred.exception))
        with self.assertRaises(ValueError) as sealed:
            self.system.validate_standalone_fork_policy("seal")
        self.assertIn("seal", str(sealed.exception))

    def test_unknown_policy_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            self.system.validate_standalone_fork_policy("nonsense")

    def test_arming_a_fork_session_carries_a_null_txn_id(self) -> None:
        # First time the ledger sees an effect row with no txn: the query
        # surface must tolerate it (asserted in the ledger test below).
        self.system.arm_fork_effect_session(self.fork_id, "reject")
        session = self.gate.session_for(self.fork_id)
        self.assertEqual(session.policy, "reject")
        self.assertIsNone(session.txn_id)
        # Config-driven knobs still apply to a bare fork's session.
        self.assertEqual(session.on_unlisted, "reject")
        self.assertEqual(session.opaque_effects, "allow")

    def test_release_fork_disarms_the_session_and_is_idempotent(self) -> None:
        self.system.arm_fork_effect_session(self.fork_id, "reject")
        self.assertIsNotNone(self.gate.session_for(self.fork_id))

        self.system.release_fork(self.fork_id)

        self.assertIsNone(self.gate.session_for(self.fork_id))
        # A fork with no chain pin returns early inside release_fork, so the
        # disarm must not depend on that path; a second release is a no-op.
        self.system.release_fork(self.fork_id)
        self.assertIsNone(self.gate.session_for(self.fork_id))

    def test_an_ungated_fork_passes_writes(self) -> None:
        # Pins F1's narrowing of D3 decision 10: a fork created WITHOUT
        # effects= is not gated. D3 recorded this in prose only; this is its
        # first executable guarantee, so the default cannot regress silently.
        #
        # Note the shape: 'allow' still opens a session (counters stay
        # uniform with the txn paths) — what "ungated" means is that the
        # decision is PASS, not that no session exists.
        policy = self.system.validate_standalone_fork_policy(None)
        self.assertEqual(policy, "allow")
        self.system.arm_fork_effect_session(self.fork_id, policy)
        decision = self.gate.decide_write(
            self.fork_id, host="api.example.com", method="POST", path="/x"
        )
        self.assertIsNotNone(decision)
        self.assertEqual(decision[0], DECISION_PASS)

    def test_a_gated_fork_refuses_writes_but_never_reads(self) -> None:
        self.system.arm_fork_effect_session(self.fork_id, "reject")
        write = self.gate.decide_write(
            self.fork_id, host="api.example.com", method="POST", path="/x"
        )
        self.assertEqual(write[0], DECISION_REJECT)
        # Reads are never gated (D3 decision 9) — decide_write is only
        # consulted for mutating methods, and opaque flows follow
        # opaque_effects, which stays 'allow' (the documented TLS hole), so
        # the decision is PASS even under a rejecting policy.
        opaque = self.gate.decide_opaque(self.fork_id)
        self.assertIsNotNone(opaque)
        self.assertEqual(opaque[0], DECISION_PASS)


class ForkPipelineGatingTests(unittest.TestCase):
    """F1 wiring inside `Engine.fork_sandbox`: when the policy is armed
    relative to the fork's restore, and who owns the effect window."""

    def _engine(self, system):
        from crab.engine import Engine

        engine = Engine.__new__(Engine)  # bypass full construction
        engine._system = system
        return engine

    def test_policy_is_validated_before_any_fork_is_created(self) -> None:
        # A refused policy must not leave half a fleet behind, so validation
        # runs before the first bundle is touched.
        calls: list[str] = []

        class _System:
            def validate_standalone_fork_policy(_self, effects):
                calls.append("validate")
                raise ValueError("effects='seal' is not supported for a bare fork")

        engine = self._engine(_System())
        with self.assertRaises(ValueError):
            engine.fork_sandbox(SandboxId("src"), count=3, effects="seal")
        # Validation happened, and nothing past it ran (no runtime access:
        # the fake system has no other attributes to call).
        self.assertEqual(calls, ["validate"])

    def test_a_caller_owning_the_window_may_not_pass_effects(self) -> None:
        engine = self._engine(object())
        with self.assertRaises(ValueError):
            engine.fork_sandbox(
                SandboxId("src"), count=1, effects="reject", gate_effects=False
            )


class ForkArmOrderingTests(unittest.TestCase):
    """The load-bearing order inside `Engine.fork_sandbox`: the gate is armed
    BEFORE the fork's processes are restored (once they run, an ungated write
    could already be on the wire), and a restore that fails does not leave
    the session behind."""

    def _engine_and_log(self, *, restore_ok: bool):
        from crab.engine import Engine

        events: list[str] = []

        class _Runtime:
            name = "runc"
            paths = SimpleNamespace(bundle_root=Path("/tmp/crab-fake-bundles"))

            def bundle_path_for(_self, sandbox_id):
                return Path("/tmp/crab-fake-bundles") / str(sandbox_id)

        class _System:
            def validate_standalone_fork_policy(_self, effects):
                return "reject" if effects is None else str(effects)

            def arm_fork_effect_session(_self, fork_id, policy):
                events.append("arm")

            def _release_effect_session(_self, sandbox_id):
                events.append("release")

            def fork_once(_self, source, target, *, checkpoint_id=None, target_rootfs_path):
                events.append("fork_once")
                return SimpleNamespace(checkpoint_id=CheckpointId("ckpt-1"))

            def restore_once(_self, sandbox_id, checkpoint_id, *, restore_metadata=None):
                events.append("restore")
                return SimpleNamespace(
                    status=SimpleNamespace(value="succeeded" if restore_ok else "failed")
                )

        engine = Engine.__new__(Engine)
        engine._system = _System()
        engine._runtime = _Runtime()
        engine._network_manager = None
        engine.repair_network_lease = lambda sandbox_id: None
        return engine, events

    def _run_fork(self, engine):
        from crab import forking

        with mock.patch.object(forking, "replicate_bundle_config"), mock.patch(
            "shutil.copy2"
        ), mock.patch("pathlib.Path.mkdir"), mock.patch(
            "pathlib.Path.is_file", return_value=False
        ):
            return engine.fork_sandbox(SandboxId("src"), count=1, effects="reject")

    def test_arm_precedes_restore(self) -> None:
        engine, events = self._engine_and_log(restore_ok=True)
        self._run_fork(engine)
        # The whole point: arm lands before restore, never after.
        self.assertLess(events.index("arm"), events.index("restore"))
        self.assertNotIn("release", events)

    def test_a_failed_restore_releases_the_armed_session(self) -> None:
        engine, events = self._engine_and_log(restore_ok=False)
        with self.assertRaises(RuntimeError):
            self._run_fork(engine)
        # Otherwise the session would gate a fork id that never came up.
        self.assertEqual(events[-1], "release")


class MissingGateWarningTests(_EffectSystemHarness):
    """An explicit gating policy on an engine with no effect gate cannot be
    enforced. It must not fail the fork (callers legitimately ask for
    `reject` on unnetworked engines), but it must not pass silently either —
    that is the hollow guarantee this item refuses elsewhere."""

    def test_explicit_reject_without_a_gate_warns(self) -> None:
        self.system.effect_gate = None
        with self.assertLogs(_system_logger, level="WARNING") as captured:
            policy = self.system.validate_standalone_fork_policy("reject")
        self.assertEqual(policy, "reject")
        message = "\n".join(captured.output)
        self.assertIn("enable_egress_proxy", message)
        self.assertIn("NOT be gated", message)

    def test_the_config_default_path_stays_quiet(self) -> None:
        # A deployment that never asked per call has not been promised
        # anything, so the default resolution must not spam warnings.
        self.system.effect_gate = None
        self.system.effect_policy_defaults["standalone_fork_policy"] = "reject"
        with mock.patch.object(_system_logger, "warning") as warned:
            self.assertEqual(self.system.validate_standalone_fork_policy(None), "reject")
        warned.assert_not_called()

    def test_explicit_allow_without_a_gate_is_not_a_warning(self) -> None:
        # 'allow' promises nothing, so a missing gate is irrelevant to it.
        self.system.effect_gate = None
        with mock.patch.object(_system_logger, "warning") as warned:
            self.assertEqual(
                self.system.validate_standalone_fork_policy("allow"), "allow"
            )
        warned.assert_not_called()


class FlushAndDropTests(_EffectSystemHarness):
    def setUp(self) -> None:
        super().setUp()
        self.upstream = _RecordingHTTPServer(status=200)
        self.addCleanup(self.upstream.close)
        self.system._arm_effect_session(
            self.sandbox_id, policy="defer", txn_id="txn-f", isolation="snapshot"
        )

    def _queue(self, path: str, *, port: int | None = None) -> None:
        self.gate.enqueue(
            self.sandbox_id,
            DeferredRequest(
                method="POST", host="127.0.0.1", port=port or self.upstream.port,
                path=path, headers=(), body=b"x",
            ),
        )

    def test_flush_sends_in_order_and_journals_each_outcome(self) -> None:
        self._queue("/first")
        self._queue("/second")
        report = self.system._flush_deferred_effects(self.sandbox_id, "txn-f")
        self.assertEqual((report.attempted, report.succeeded, report.failed), (2, 2, 0))
        self.assertEqual([entry.path for entry in report.entries], ["/first", "/second"])
        self.assertEqual(
            [path for _, path, _ in self.upstream.seen], ["/first", "/second"]
        )  # enqueue order preserved
        rows = [
            row.payload for row in self.journal.entries(self.sandbox_id, kind="egress")
        ]
        self.assertEqual([row["effect"] for row in rows], ["flushed", "flushed"])
        self.assertEqual(rows[0]["effect_status"], 200)
        self.assertEqual(rows[0]["txn_id"] if "txn_id" in rows[0] else "txn-f", "txn-f")

    def test_flush_failure_is_reported_not_raised(self) -> None:
        self._queue("/gone", port=1)  # nothing serves loopback port 1
        report = self.system._flush_deferred_effects(self.sandbox_id, "txn-f")
        self.assertEqual((report.attempted, report.succeeded, report.failed), (1, 0, 1))
        self.assertIsNotNone(report.entries[0].error)
        [row] = [r.payload for r in self.journal.entries(self.sandbox_id, kind="egress")]
        self.assertEqual(row["effect"], "flush_failed")
        self.assertIsNotNone(row["effect_error"])

    def test_empty_queue_yields_no_report(self) -> None:
        self.assertIsNone(self.system._flush_deferred_effects(self.sandbox_id, "txn-f"))

    def test_drop_discards_without_sending(self) -> None:
        self._queue("/never")
        dropped = self.system._drop_deferred_effects(self.sandbox_id, "txn-f")
        self.assertEqual(dropped, 1)
        self.assertEqual(self.upstream.seen, [])  # nothing was sent
        [row] = [r.payload for r in self.journal.entries(self.sandbox_id, kind="egress")]
        self.assertEqual(row["effect"], "dropped")
        # Nothing left to flush afterwards.
        self.assertIsNone(self.system._flush_deferred_effects(self.sandbox_id, "txn-f"))

    def test_report_round_trip(self) -> None:
        self._queue("/rt")
        report = self.system._flush_deferred_effects(self.sandbox_id, "txn-f")
        self.assertEqual(EffectFlushReport.from_json(report.to_json()), report)


class BeginPolicyGuardTests(unittest.TestCase):
    """The one combination D3 refuses outright (decision 12)."""

    def test_fork_with_defer_is_refused(self) -> None:
        from types import SimpleNamespace

        system = SimpleNamespace()
        # Use the real method with a minimal object: the guard runs before
        # anything else in begin_txn.
        from crab.system import CrabSystem

        with self.assertRaises(ValueError) as ctx:
            CrabSystem.begin_txn(
                _StubSystem(policy="defer"), SandboxId("sbx-1"), isolation="fork"
            )
        self.assertIn("not supported with isolation='fork'", str(ctx.exception))

    def test_unknown_policy_is_refused(self) -> None:
        from crab.system import CrabSystem

        with self.assertRaises(ValueError):
            CrabSystem.begin_txn(_StubSystem(policy="hold"), SandboxId("sbx-1"))


class _StubSystem:
    """Just enough surface for begin_txn's argument validation."""

    def __init__(self, *, policy: str) -> None:
        self._policy = policy
        self.effect_policy_defaults = {}

    def _resolve_effect_policy(self, isolation, requested):
        return self._policy


class BareHostTests(unittest.TestCase):
    """The queued host comes from the Host header, which carries the port
    for non-default ports; a flush connects with (host, port), so leaving
    it in makes resolution fail every time (D2.1's request_key trap)."""

    def test_strips_a_numeric_port(self) -> None:
        from crab.effects import bare_host

        self.assertEqual(bare_host("10.0.2.15:45411"), "10.0.2.15")
        self.assertEqual(bare_host("api.example.com:8080"), "api.example.com")

    def test_strips_the_port_from_a_bracketed_ipv6_literal(self) -> None:
        # This assertion previously pinned the wrong behavior: leaving the
        # port on made every flush to an IPv6 host fail.
        from crab.effects import bare_host

        self.assertEqual(bare_host("[::1]:8080"), "[::1]")
        self.assertEqual(bare_host("[2001:db8::1]:443"), "[2001:db8::1]")

    def test_leaves_everything_else_alone(self) -> None:
        from crab.effects import bare_host

        self.assertEqual(bare_host("api.example.com"), "api.example.com")
        self.assertEqual(bare_host("[::1]"), "[::1]")  # bracketed, no port
        self.assertEqual(bare_host("fe80::1"), "fe80::1")  # bare IPv6, no port
        self.assertEqual(bare_host("host:notaport"), "host:notaport")
        self.assertEqual(bare_host("[::1"), "[::1")  # unterminated bracket

    def test_queue_entries_store_the_bare_host(self) -> None:
        head = parse_head(b"POST /w HTTP/1.1\r\nHost: 10.0.2.15:45411\r\n\r\n")
        request = build_deferred_request(
            parsed_head=head, body=b"", host="10.0.2.15:45411", port=45411,
            method="POST", path="/w", txn_id=None, enqueued_at=None,
        )
        self.assertEqual(request.host, "10.0.2.15")
        self.assertEqual(request.port, 45411)


class MutatingSentTests(unittest.TestCase):
    """`mutating_egress` must not count writes the gate held: reporting a
    deferred write as "already fired" would misstate the one guarantee
    `effects="defer"` provides."""

    def _flow(self, seq: int, effect: str | None) -> EgressFlow:
        return EgressFlow(
            seq=seq, host="api.example.com", dst_ip="10.0.0.1", dst_port=80,
            scheme="http", classification="mutating", method="POST", path="/w",
            effect=effect,
        )

    def test_held_writes_are_excluded(self) -> None:
        ledger = EgressLedger(
            sandbox_id=SandboxId("sbx-1"),
            flows=(
                self._flow(1, None),          # passed through
                self._flow(2, "sealed"),      # sent under seal
                self._flow(3, "flushed"),     # sent at commit
                self._flow(4, "flush_failed"),  # attempted; may have landed
                self._flow(5, "deferred"),    # queued, never sent
                self._flow(6, "rejected"),    # refused
                self._flow(7, "dropped"),     # discarded on abort
                self._flow(8, "lost"),        # queue died with the daemon
            ),
        )
        self.assertEqual(ledger.mutating, 8)  # all are mutating flows
        self.assertEqual(ledger.mutating_sent, 4)  # only these left the host


class LostEffectBackfillTests(_EffectSystemHarness):
    """The queue is in-memory only, so a restart loses it while the journal
    keeps the `deferred` rows. Left alone they look like writes that might
    still fire (`crab egress --pending`), so startup closes them out."""

    def _row(self, effect: str, *, txn_id: str = "txn-a", path: str = "/w") -> None:
        self.journal.record_egress(
            self.sandbox_id,
            payload={
                "host": "api.example.com", "dst_ip": "10.0.0.1", "dst_port": 80,
                "scheme": "http", "method": "POST", "path": path,
                "bytes_out": 3, "bytes_in": 0, "duration_ms": 1.0,
                "classification": "mutating", "effect": effect,
            },
            txn_id=txn_id,
        )

    def _effects(self) -> list[str]:
        return [
            row.payload["effect"]
            for row in self.journal.entries(self.sandbox_id, kind="egress")
            if row.payload.get("effect")
        ]

    def test_orphaned_deferred_row_gets_a_terminal_lost_row(self) -> None:
        self._row("deferred")
        self.assertEqual(self.system.backfill_lost_effects(), 1)
        self.assertEqual(self._effects(), ["deferred", "lost"])
        ledger = self.system.egress_ledger(self.sandbox_id)
        # The pending row is closed out, and a lost write is not "sent".
        self.assertEqual(ledger.deferred, 1)
        self.assertEqual(ledger.mutating_sent, 0)

    def test_is_idempotent(self) -> None:
        self._row("deferred")
        self.assertEqual(self.system.backfill_lost_effects(), 1)
        # A second startup must not pile on more rows.
        self.assertEqual(self.system.backfill_lost_effects(), 0)
        self.assertEqual(self._effects().count("lost"), 1)

    def test_resolved_queues_are_left_alone(self) -> None:
        self._row("deferred")
        self._row("flushed")
        self._row("deferred", txn_id="txn-b")
        self._row("dropped", txn_id="txn-b")
        self.assertEqual(self.system.backfill_lost_effects(), 0)
        self.assertNotIn("lost", self._effects())

    def test_partially_flushed_queue_only_closes_the_remainder(self) -> None:
        self._row("deferred", path="/one")
        self._row("deferred", path="/two")
        self._row("flushed", path="/one")  # the restart hit mid-flush
        self.assertEqual(self.system.backfill_lost_effects(), 1)
        self.assertEqual(self._effects().count("lost"), 1)

    def test_non_effect_rows_are_ignored(self) -> None:
        self.journal.record_egress(
            self.sandbox_id,
            payload={
                "host": "api.example.com", "dst_ip": "10.0.0.1", "dst_port": 80,
                "scheme": "http", "method": "GET", "path": "/r",
                "bytes_out": 0, "bytes_in": 2, "duration_ms": 1.0,
                "classification": "idempotent_read",
            },
        )
        self.assertEqual(self.system.backfill_lost_effects(), 0)

    def test_no_journal_is_a_noop(self) -> None:
        self.system.journal = None
        self.assertEqual(self.system.backfill_lost_effects(), 0)


class ReleaseTxnEffectCleanupTests(_EffectSystemHarness):
    """A sandbox dying with an open txn must discard its queue exactly as an
    abort would; otherwise the bodies stay in memory and the ledger keeps a
    pending row that can never fire."""

    def test_release_drops_the_queue_and_disarms(self) -> None:
        from crab.txn import TxnDescription

        description = TxnDescription(
            txn_id="txn-r", sandbox_id=str(self.sandbox_id), base_checkpoint_id="ckpt-1",
            base_was_fresh=True, started_at="now", effects="defer",
        )
        self.system._active_txns[self.sandbox_id] = description
        self.system._arm_effect_session(
            self.sandbox_id, policy="defer", txn_id="txn-r", isolation="snapshot"
        )
        self.gate.enqueue(
            self.sandbox_id,
            DeferredRequest(
                method="POST", host="api.example.com", port=80, path="/w",
                headers=(), body=b"x",
            ),
        )

        self.system.release_txn(self.sandbox_id)

        self.assertIsNone(self.gate.session_for(self.sandbox_id))
        self.assertEqual(self.gate.queue_size(self.sandbox_id), (0, 0))
        rows = [
            row.payload["effect"]
            for row in self.journal.entries(self.sandbox_id, kind="egress")
            if row.payload.get("effect")
        ]
        self.assertEqual(rows, ["dropped"])
        # Nothing is left pending, so no backfill is needed either.
        self.assertEqual(self.system.backfill_lost_effects(), 0)
