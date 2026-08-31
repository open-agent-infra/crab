"""Unit tests for D2.1 recording: the cassette store, request keys, the
HTTP wire helpers, and the proxy's recording gate. Host-runnable."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from crab.cassettes import (
    DEFAULT_VARYING_HEADERS,
    CassetteEntry,
    CassetteStore,
    filter_headers,
    request_key,
    sha256_hex,
)
from crab.egress import CassetteRecorder
from crab.http_wire import (
    ResponseAssembler,
    dechunk,
    parse_head,
    parse_status_line,
    serialize_response,
)


def _entry(**overrides) -> CassetteEntry:
    payload = {
        "request_key": "k" * 64,
        "method": "GET",
        "host": "api.example.com",
        "port": 80,
        "path": "/things",
        "status": 200,
        "reason": "OK",
        "response_headers": (("Content-Type", "application/json"),),
        "body": b'{"ok":true}',
        "body_sha256": sha256_hex(b'{"ok":true}'),
    }
    payload.update(overrides)
    return CassetteEntry(**payload)


class RequestKeyTests(unittest.TestCase):
    def _key(self, **overrides) -> str:
        payload = {
            "method": "GET",
            "host": "api.example.com",
            "port": 80,
            "path": "/things?page=2",
            "body_sha256": sha256_hex(b""),
            "headers": [("Accept", "application/json"), ("User-Agent", "curl/8")],
        }
        payload.update(overrides)
        return request_key(**payload)

    def test_stable_and_case_insensitive_on_host_and_method(self) -> None:
        self.assertEqual(self._key(), self._key())
        self.assertEqual(self._key(method="get"), self._key(method="GET"))
        self.assertEqual(self._key(host="API.example.com"), self._key())

    def test_incidental_headers_do_not_change_the_key(self) -> None:
        bumped = self._key(headers=[("Accept", "application/json"), ("User-Agent", "curl/9")])
        self.assertEqual(bumped, self._key())

    def test_varying_headers_do_change_the_key(self) -> None:
        other = self._key(headers=[("Accept", "text/csv"), ("User-Agent", "curl/8")])
        self.assertNotEqual(other, self._key())

    def test_query_string_and_body_participate(self) -> None:
        self.assertNotEqual(self._key(path="/things?page=3"), self._key())
        self.assertNotEqual(self._key(body_sha256=sha256_hex(b"x")), self._key())

    def test_range_only_matters_when_allow_listed(self) -> None:
        headers = [("Range", "bytes=0-99")]
        default = request_key(
            method="GET", host="h", port=80, path="/f", body_sha256=sha256_hex(b""),
            headers=headers,
        )
        other_range = request_key(
            method="GET", host="h", port=80, path="/f", body_sha256=sha256_hex(b""),
            headers=[("Range", "bytes=100-199")],
        )
        # This collision is exactly why 206 is not recorded by default.
        self.assertEqual(default, other_range)
        with_range = request_key(
            method="GET", host="h", port=80, path="/f", body_sha256=sha256_hex(b""),
            headers=headers, varying_headers=(*DEFAULT_VARYING_HEADERS, "range"),
        )
        other_with_range = request_key(
            method="GET", host="h", port=80, path="/f", body_sha256=sha256_hex(b""),
            headers=[("Range", "bytes=100-199")],
            varying_headers=(*DEFAULT_VARYING_HEADERS, "range"),
        )
        self.assertNotEqual(with_range, other_with_range)


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="crab_cass_")
        self.addCleanup(self._tmp.cleanup)
        self.store = CassetteStore(Path(self._tmp.name))

    def test_round_trip(self) -> None:
        entry = _entry()
        path = self.store.put("sbx-1", entry)
        self.assertTrue(path.exists())
        restored = self.store.get("sbx-1", entry.request_key)
        self.assertEqual(restored, entry)

    def test_miss_and_bucket_isolation(self) -> None:
        entry = _entry()
        self.store.put("sbx-1", entry)
        self.assertIsNone(self.store.get("sbx-1", "missing"))
        # Buckets are per sandbox: another sandbox sees nothing.
        self.assertIsNone(self.store.get("sbx-2", entry.request_key))

    def test_newer_response_wins_without_clobbering(self) -> None:
        first = _entry(body=b"one", body_sha256=sha256_hex(b"one"))
        self.store.put("sbx-1", first)
        second = _entry(body=b"two", body_sha256=sha256_hex(b"two"))
        self.store.put("sbx-1", second)
        directory = self.store.bucket("sbx-1") / first.request_key
        self.assertEqual(len(list(directory.glob("*.json"))), 2)
        self.assertEqual(self.store.get("sbx-1", first.request_key).body, b"two")
        # "current" is an explicit pointer, not an mtime guess.
        self.assertEqual(
            (directory / "latest").read_text(encoding="utf-8"), second.body_sha256
        )

    def test_corrupt_file_degrades_to_a_miss(self) -> None:
        entry = _entry()
        path = self.store.put("sbx-1", entry)
        path.write_text("{not json", encoding="utf-8")
        self.assertIsNone(self.store.get("sbx-1", entry.request_key))

    def test_no_temp_files_left_behind(self) -> None:
        self.store.put("sbx-1", _entry())
        self.assertEqual(list(self.store.root.rglob("*.tmp")), [])

    def test_count_and_prune(self) -> None:
        self.store.put("sbx-1", _entry())
        self.store.put("sbx-1", _entry(request_key="j" * 64))
        self.store.put("sbx-2", _entry())
        self.assertEqual(self.store.count("sbx-1"), 2)
        self.store.prune("sbx-1")
        self.assertEqual(self.store.count("sbx-1"), 0)
        self.assertEqual(self.store.count("sbx-2"), 1)
        self.store.prune("sbx-missing")  # idempotent

    def test_entry_json_round_trip_carries_binary_bodies(self) -> None:
        entry = _entry(body=b"\x00\x01\x02\xff", body_sha256=sha256_hex(b"\x00\x01\x02\xff"))
        restored = CassetteEntry.from_json(json.loads(json.dumps(entry.to_json())))
        self.assertEqual(restored.body, b"\x00\x01\x02\xff")


class HeaderFilterTests(unittest.TestCase):
    def test_deny_list_is_case_insensitive(self) -> None:
        headers = [("AUTHORIZATION", "Bearer x"), ("Accept", "*/*"), ("Cookie", "a=b")]
        kept = filter_headers(headers, {"authorization", "cookie"})
        self.assertEqual(kept, [("Accept", "*/*")])


class HttpWireTests(unittest.TestCase):
    def test_parse_head_and_rest(self) -> None:
        head = parse_head(b"GET /x HTTP/1.1\r\nHost: h\r\nAccept: */*\r\n\r\nbody")
        self.assertIsNotNone(head)
        self.assertEqual(head.start_line, "GET /x HTTP/1.1")
        self.assertEqual(head.get("host"), "h")
        self.assertEqual(head.get("HOST"), "h")  # case-insensitive
        self.assertEqual(head.rest, b"body")

    def test_parse_head_rejects_incomplete_or_garbage(self) -> None:
        self.assertIsNone(parse_head(b"GET /x HTTP/1.1\r\nHost: h\r\n"))  # no terminator
        self.assertIsNone(parse_head(b"GET /x HTTP/1.1\r\nbogus-line\r\n\r\n"))

    def test_status_line(self) -> None:
        self.assertEqual(parse_status_line("HTTP/1.1 204 No Content"), (204, "No Content"))
        self.assertEqual(parse_status_line("HTTP/1.0 200 OK"), (200, "OK"))
        self.assertIsNone(parse_status_line("garbage"))
        self.assertIsNone(parse_status_line("HTTP/1.1 abc OK"))

    def test_dechunk(self) -> None:
        self.assertEqual(dechunk(b"4\r\nabcd\r\n3\r\nefg\r\n0\r\n\r\n"), b"abcdefg")
        self.assertEqual(dechunk(b"4;ext=1\r\nabcd\r\n0\r\n\r\n"), b"abcd")
        self.assertIsNone(dechunk(b"4\r\nab"))  # incomplete
        self.assertIsNone(dechunk(b"zz\r\n"))  # malformed size

    def test_assembler_content_length(self) -> None:
        assembler = ResponseAssembler(limit=1024)
        assembler.feed(b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nhello")
        head, status, reason, body = assembler.result()
        self.assertEqual((status, reason, body), (200, "OK", b"hello"))
        self.assertEqual(head.get("content-length"), "5")

    def test_assembler_incomplete_body_is_not_recordable(self) -> None:
        assembler = ResponseAssembler(limit=1024)
        assembler.feed(b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nhel")
        self.assertIsNone(assembler.result())

    def test_assembler_chunked_and_close_delimited(self) -> None:
        chunked = ResponseAssembler(limit=1024)
        chunked.feed(
            b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nhello\r\n0\r\n\r\n"
        )
        self.assertEqual(chunked.result()[3], b"hello")
        closed = ResponseAssembler(limit=1024)
        closed.feed(b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n\r\nraw bytes")
        self.assertEqual(closed.result()[3], b"raw bytes")

    def test_assembler_truncates_past_the_limit(self) -> None:
        assembler = ResponseAssembler(limit=32)
        assembler.feed(b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\n" + b"x" * 100)
        self.assertTrue(assembler.truncated)
        self.assertIsNone(assembler.result())
        self.assertEqual(assembler.total_bytes, 140)  # 40-byte head + 100-byte body

    def test_serialize_response_rewrites_framing(self) -> None:
        wire = serialize_response(
            status=200,
            reason="OK",
            headers=[
                ("Content-Type", "text/plain"),
                ("Transfer-Encoding", "chunked"),  # dropped
                ("Content-Length", "999"),  # replaced
                ("Connection", "keep-alive"),  # dropped
            ],
            body=b"hello",
        )
        self.assertTrue(wire.startswith(b"HTTP/1.1 200 OK\r\n"))
        self.assertIn(b"Content-Type: text/plain\r\n", wire)
        self.assertNotIn(b"Transfer-Encoding", wire)
        self.assertIn(b"Content-Length: 5\r\n", wire)
        self.assertIn(b"Connection: close\r\n", wire)
        self.assertTrue(wire.endswith(b"\r\n\r\nhello"))


class RecorderGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="crab_rec_")
        self.addCleanup(self._tmp.cleanup)
        self.store = CassetteStore(Path(self._tmp.name))

    def _recorder(self, **kwargs) -> CassetteRecorder:
        return CassetteRecorder(self.store, **kwargs)

    def _record(self, recorder, *, request: bytes, response: bytes, limit: int = 4096):
        assembler = ResponseAssembler(limit=limit)
        assembler.feed(response)
        return recorder.record_exchange(
            sandbox_id="sbx-1",
            request_head=request,
            host="api.example.com",
            port=80,
            method="GET",
            path="/things",
            assembler=assembler,
            bytes_in=len(response),
        )

    _REQUEST = b"GET /things HTTP/1.1\r\nHost: api.example.com\r\nAccept: */*\r\n\r\n"
    _OK = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nhi"

    def test_records_and_returns_index_fields(self) -> None:
        recorder = self._recorder()
        meta = self._record(recorder, request=self._REQUEST, response=self._OK)
        self.assertTrue(meta["recorded"])
        self.assertEqual(meta["status"], 200)
        self.assertFalse(meta["truncated"])
        entry = self.store.get("sbx-1", meta["request_key"])
        self.assertEqual(entry.body, b"hi")
        self.assertEqual(entry.origin_sandbox_id, "sbx-1")
        self.assertTrue(entry.recorded_at)

    def test_credentials_never_reach_the_cassette(self) -> None:
        recorder = self._recorder()
        request = (
            b"GET /things HTTP/1.1\r\nHost: api.example.com\r\n"
            b"Authorization: Bearer super-secret\r\nCookie: sid=42\r\n\r\n"
        )
        response = (
            b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nSet-Cookie: sid=99\r\n\r\nhi"
        )
        meta = self._record(recorder, request=request, response=response)
        entry = self.store.get("sbx-1", meta["request_key"])
        names = {name.lower() for name, _ in entry.response_headers}
        self.assertNotIn("set-cookie", names)
        raw = (self.store.bucket("sbx-1") / meta["request_key"]).glob("*.json")
        blob = next(raw).read_text(encoding="utf-8")
        self.assertNotIn("super-secret", blob)
        self.assertNotIn("sid=99", blob)

    def test_status_matrix(self) -> None:
        recorder = self._recorder()
        for status, expected in ((200, True), (203, True), (204, True), (302, True)):
            meta = self._record(
                recorder,
                request=self._REQUEST,
                response=f"HTTP/1.1 {status} X\r\nContent-Length: 0\r\n\r\n".encode(),
            )
            self.assertEqual(bool(meta), expected, msg=f"status={status}")
        # Errors need the opt-in.
        error = b"HTTP/1.1 500 Boom\r\nContent-Length: 0\r\n\r\n"
        self.assertIsNone(self._record(recorder, request=self._REQUEST, response=error))
        self.assertTrue(
            self._record(
                self._recorder(record_errors=True), request=self._REQUEST, response=error
            )["recorded"]
        )

    def test_partial_needs_both_opt_ins(self) -> None:
        partial = b"HTTP/1.1 206 Partial\r\nContent-Length: 2\r\n\r\nhi"
        self.assertIsNone(
            self._record(self._recorder(), request=self._REQUEST, response=partial)
        )
        # record_partial alone is not enough: the key would collide.
        self.assertIsNone(
            self._record(
                self._recorder(record_partial=True), request=self._REQUEST, response=partial
            )
        )
        recorder = self._recorder(
            record_partial=True, varying_headers=(*DEFAULT_VARYING_HEADERS, "range")
        )
        self.assertTrue(
            self._record(recorder, request=self._REQUEST, response=partial)["recorded"]
        )

    def test_truncated_is_visible_but_not_recorded(self) -> None:
        recorder = self._recorder(max_body_bytes=16)
        meta = self._record(
            recorder,
            request=self._REQUEST,
            response=b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\n" + b"x" * 100,
            limit=16,
        )
        self.assertFalse(meta["recorded"])
        self.assertTrue(meta["truncated"])
        self.assertIsNone(self.store.get("sbx-1", meta["request_key"]))

    def test_unparsable_exchange_is_skipped(self) -> None:
        recorder = self._recorder()
        self.assertIsNone(
            self._record(recorder, request=b"not a request", response=self._OK)
        )
        self.assertIsNone(
            self._record(recorder, request=self._REQUEST, response=b"not a response\r\n\r\n")
        )

    def test_store_failure_degrades_to_not_recorded(self) -> None:
        recorder = self._recorder()
        with mock.patch.object(self.store, "put", side_effect=OSError("disk full")):
            self.assertIsNone(
                self._record(recorder, request=self._REQUEST, response=self._OK)
            )


if __name__ == "__main__":
    unittest.main()
