from __future__ import annotations

import unittest
from types import SimpleNamespace

from crab import SandboxInfo, SandboxState, sandbox_state_from_status
from crab.cloud_client import SandboxLost, SandboxNotFound
from crab.ids import SandboxId
from crab.sandbox import Sandbox


class SandboxStateMappingTests(unittest.TestCase):
    def test_maps_core_states(self) -> None:
        self.assertEqual(sandbox_state_from_status("running"), SandboxState.RUNNING)
        self.assertEqual(sandbox_state_from_status("paused"), SandboxState.PAUSED)
        self.assertEqual(sandbox_state_from_status("stopped"), SandboxState.STOPPED)
        self.assertEqual(sandbox_state_from_status("created"), SandboxState.CREATED)

    def test_paused_and_stopped_are_distinct(self) -> None:
        self.assertIsNot(sandbox_state_from_status("paused"), sandbox_state_from_status("stopped"))
        self.assertEqual(sandbox_state_from_status("paused"), SandboxState.PAUSED)
        self.assertEqual(sandbox_state_from_status("stopped"), SandboxState.STOPPED)

    def test_gone_states(self) -> None:
        self.assertEqual(sandbox_state_from_status("killed"), SandboxState.KILLED)
        self.assertEqual(sandbox_state_from_status("lost"), SandboxState.LOST)
        # A vanished runc container reads as killed from the user's view.
        self.assertEqual(sandbox_state_from_status("missing"), SandboxState.KILLED)
        self.assertEqual(sandbox_state_from_status("exited"), SandboxState.STOPPED)

    def test_case_and_whitespace_are_normalized(self) -> None:
        self.assertEqual(sandbox_state_from_status("  Running "), SandboxState.RUNNING)
        self.assertEqual(sandbox_state_from_status("PAUSED"), SandboxState.PAUSED)

    def test_unknown_and_none(self) -> None:
        self.assertEqual(sandbox_state_from_status("bogus"), SandboxState.UNKNOWN)
        self.assertEqual(sandbox_state_from_status(None), SandboxState.UNKNOWN)
        self.assertEqual(sandbox_state_from_status(""), SandboxState.UNKNOWN)


class _FakeRuntime:
    def __init__(
        self,
        *,
        runtime_status: str = "running",
        desc_status: str = "running",
        inspect_exc: Exception | None = None,
        describe_exc: Exception | None = None,
    ) -> None:
        self.runtime_status = runtime_status
        self.desc_status = desc_status
        self.inspect_exc = inspect_exc
        self.describe_exc = describe_exc

    def inspect_runtime(self, sandbox_id):
        if self.inspect_exc is not None:
            raise self.inspect_exc
        return SimpleNamespace(status=self.runtime_status, pid=123)

    def describe(self, sandbox_id):
        if self.describe_exc is not None:
            raise self.describe_exc
        return SimpleNamespace(status=self.desc_status, metadata={"image": "ubuntu:22.04"})


def _make_sandbox(runtime) -> Sandbox:
    sbx = Sandbox.__new__(Sandbox)
    sbx._engine = SimpleNamespace(runtime=runtime)
    sbx._sandbox_id = SandboxId("sbx-status")
    sbx._closed = False
    return sbx


class SandboxStatePropertyTests(unittest.TestCase):
    def test_state_reflects_live_runtime_status(self) -> None:
        for status, expected in [
            ("running", SandboxState.RUNNING),
            ("paused", SandboxState.PAUSED),
            ("stopped", SandboxState.STOPPED),
        ]:
            sbx = _make_sandbox(_FakeRuntime(runtime_status=status))
            self.assertEqual(sbx.state, expected)

    def test_state_prefers_live_over_stale_bookkeeping(self) -> None:
        # Bookkeeping says running, but runc reports paused -> trust the live one.
        sbx = _make_sandbox(_FakeRuntime(runtime_status="paused", desc_status="running"))
        self.assertEqual(sbx.state, SandboxState.PAUSED)

    def test_not_found_maps_to_killed(self) -> None:
        sbx = _make_sandbox(
            _FakeRuntime(inspect_exc=SandboxNotFound(404, "/sandboxes/sbx-status", b""))
        )
        self.assertEqual(sbx.state, SandboxState.KILLED)

    def test_lost_maps_to_lost(self) -> None:
        sbx = _make_sandbox(
            _FakeRuntime(inspect_exc=SandboxLost(410, "/sandboxes/sbx-status", b""))
        )
        self.assertEqual(sbx.state, SandboxState.LOST)

    def test_local_keyerror_maps_to_killed(self) -> None:
        sbx = _make_sandbox(_FakeRuntime(inspect_exc=KeyError("sbx-status")))
        self.assertEqual(sbx.state, SandboxState.KILLED)

    def test_closed_sandbox_is_killed(self) -> None:
        sbx = _make_sandbox(_FakeRuntime(runtime_status="running"))
        sbx._closed = True
        self.assertEqual(sbx.state, SandboxState.KILLED)

    def test_unset_sandbox_id_is_unknown(self) -> None:
        sbx = _make_sandbox(_FakeRuntime())
        sbx._sandbox_id = None
        self.assertEqual(sbx.state, SandboxState.UNKNOWN)

    def test_inspect_failure_falls_back_to_bookkeeping(self) -> None:
        sbx = _make_sandbox(
            _FakeRuntime(inspect_exc=RuntimeError("boom"), desc_status="stopped")
        )
        self.assertEqual(sbx.state, SandboxState.STOPPED)


class SandboxDescribeTests(unittest.TestCase):
    def test_describe_returns_both_statuses(self) -> None:
        sbx = _make_sandbox(_FakeRuntime(runtime_status="running", desc_status="running"))
        info = sbx.describe()
        self.assertIsInstance(info, SandboxInfo)
        self.assertEqual(info.state, SandboxState.RUNNING)
        self.assertEqual(info.status, "running")
        self.assertEqual(info.runtime_status, "running")
        self.assertEqual(info.pid, 123)
        self.assertEqual(info.metadata["image"], "ubuntu:22.04")

    def test_describe_distinguishes_paused_and_stopped(self) -> None:
        paused = _make_sandbox(_FakeRuntime(runtime_status="paused")).describe()
        stopped = _make_sandbox(_FakeRuntime(runtime_status="stopped")).describe()
        self.assertEqual(paused.state, SandboxState.PAUSED)
        self.assertEqual(stopped.state, SandboxState.STOPPED)
        self.assertNotEqual(paused.state, stopped.state)

    def test_describe_lost(self) -> None:
        sbx = _make_sandbox(
            _FakeRuntime(
                inspect_exc=SandboxLost(410, "/sandboxes/sbx-status", b""),
                describe_exc=SandboxLost(410, "/sandboxes/sbx-status", b""),
            )
        )
        self.assertEqual(sbx.describe().state, SandboxState.LOST)

    def test_describe_killed(self) -> None:
        sbx = _make_sandbox(
            _FakeRuntime(
                inspect_exc=SandboxNotFound(404, "/sandboxes/sbx-status", b""),
                describe_exc=SandboxNotFound(404, "/sandboxes/sbx-status", b""),
            )
        )
        self.assertEqual(sbx.describe().state, SandboxState.KILLED)


if __name__ == "__main__":
    unittest.main()
