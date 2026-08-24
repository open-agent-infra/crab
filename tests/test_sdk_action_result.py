"""SDK unit tests for the rich return-value surface of
``commands.run()`` / ``commands.stream()`` (feat/sdk-improvements).

Covers the new ``ActionResult`` wrapper, the ``AsyncCheckpoint`` /
``AsyncChangeset`` background handles (with client-preallocated
``ckpt-*`` ids), the read-only inspector peek, sandbox-level
``auto_checkpoint``, and the ``ExecStream`` iterate-then-.result shape.

Host-runnable: the engine/runtime/system are fakes, so no runc/CRIU/zfs
and no daemon socket. The fakes mirror the real duck-typed contracts the
SDK depends on (``runtime.exec`` / ``runtime.stream_exec`` /
``system.checkpoint_once`` / ``system.fork_changeset`` /
``system.inspector.inspect``).
"""
from __future__ import annotations

import threading
import unittest
from types import SimpleNamespace

from crab.ids import CheckpointId, SandboxId
from crab.models import (
    ChangesetEntry,
    ChangesetResult,
    ExecDone,
    ExecEvent,
    JobStatus,
    SandboxExecResult,
    SandboxSnapshot,
    utc_now,
)
from crab.sandbox import (
    ActionResult,
    AsyncChangeset,
    AsyncCheckpoint,
    ExecStream,
    Sandbox,
)


# ---------------------------------------------------------------------------
# Fakes: a minimal engine/runtime/system that duck-types onto what the
# SDK's commands namespace and _build_action_result reach for.
# ---------------------------------------------------------------------------


class _FakeInspector:
    def __init__(self, *, filesystem_changed=False, process_changed=False) -> None:
        self.filesystem_changed = filesystem_changed
        self.process_changed = process_changed
        self.inspect_calls: list[SandboxId] = []

    def inspect(self, sandbox_id: SandboxId) -> SandboxSnapshot:
        self.inspect_calls.append(sandbox_id)
        return SandboxSnapshot(
            sandbox_id=sandbox_id,
            runtime_name="docker",
            is_running=True,
            process_changed=self.process_changed,
            filesystem_changed=self.filesystem_changed,
            observed_at=utc_now(),
        )

    def upsert_snapshot(self, *_, **__) -> None:
        return None

    def mark_changed(self, *_, **__) -> None:
        return None


class _FakeSystem:
    def __init__(self, inspector: _FakeInspector | None = None) -> None:
        self.inspector = inspector or _FakeInspector()
        self.checkpoint_calls: list[dict] = []
        self.fork_calls: list[tuple[SandboxId, bool]] = []
        self.since_calls: list[tuple[SandboxId, CheckpointId, bool]] = []
        self.checkpoint_barrier: threading.Event | None = None

    def checkpoint_once(self, sandbox_id, *, leave_running=True, checkpoint_id=None):
        if self.checkpoint_barrier is not None:
            self.checkpoint_barrier.wait(timeout=5.0)
        self.checkpoint_calls.append(
            {
                "sandbox_id": sandbox_id,
                "leave_running": leave_running,
                "checkpoint_id": checkpoint_id,
            }
        )
        # The composite worker honours a client-supplied id; mirror that.
        resolved = checkpoint_id or "ckpt-server-minted"
        ckpt = CheckpointId(str(resolved))
        return SimpleNamespace(
            checkpoint_id=ckpt,
            manifest=SimpleNamespace(checkpoint_id=ckpt),
            status=JobStatus.SUCCEEDED,
            message="",
        )

    def fork_changeset(self, sandbox_id, *, force=False):
        self.fork_calls.append((sandbox_id, force))
        return ChangesetResult(
            sandbox_id=sandbox_id,
            base_checkpoint_id=CheckpointId("ckpt-forkpoint"),
            entries=(ChangesetEntry(path="/new.txt", change="added"),),
        )

    def changeset_since(self, sandbox_id, checkpoint_id, *, use_inspector_gate=True):
        self.since_calls.append((sandbox_id, checkpoint_id, use_inspector_gate))
        return ChangesetResult(
            sandbox_id=sandbox_id,
            base_checkpoint_id=checkpoint_id,
            entries=(ChangesetEntry(path="/since.txt", change="added"),),
        )


class _FakeRuntime:
    name = "docker"

    def __init__(self) -> None:
        self.exec_calls: list[list[str]] = []

    def exec(self, sandbox_id, argv, *, cwd=None, env=None, user=None,
             timeout_s=None, capture_output=True):
        self.exec_calls.append(argv)
        return SandboxExecResult(
            args=tuple(argv),
            returncode=0,
            stdout="ok",
            stderr="",
        )

    def stream_exec(self, sandbox_id, argv, *, cwd=None, env=None, user=None,
                    timeout_s=None):
        yield ExecEvent(channel="stdout", text="line1\n")
        yield ExecEvent(channel="stdout", text="line2\n")
        yield ExecDone(returncode=0)


class _FakeEngine:
    def __init__(self, system: _FakeSystem, runtime: _FakeRuntime) -> None:
        self.system = system
        self.runtime = runtime

    def _register_sandbox(self, sandbox) -> None:
        pass


def _make_sandbox(*, auto_checkpoint=False, inspector=None):
    system = _FakeSystem(inspector=inspector)
    runtime = _FakeRuntime()
    engine = _FakeEngine(system, runtime)
    sbx = Sandbox(engine=engine, autostart=False, auto_checkpoint=auto_checkpoint)
    sbx._sandbox_id = SandboxId("sbx-test")
    return sbx, system, runtime


# ---------------------------------------------------------------------------
# ActionResult basics
# ---------------------------------------------------------------------------


class ActionResultBasicsTests(unittest.TestCase):
    def test_plain_run_returns_action_result(self) -> None:
        sbx, _system, runtime = _make_sandbox()
        result = sbx.commands.run("echo hi")
        self.assertIsInstance(result, ActionResult)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "ok")
        self.assertEqual(result.args, ("/bin/sh", "-c", "echo hi"))
        # No enrichment requested -> all optional fields None.
        self.assertIsNone(result.checkpoint)
        self.assertIsNone(result.changeset)
        self.assertIsNone(result.filesystem_changed)
        self.assertIsNone(result.process_changed)
        self.assertEqual(runtime.exec_calls, [["/bin/sh", "-c", "echo hi"]])

    def test_argv_bypasses_shell(self) -> None:
        sbx, _system, runtime = _make_sandbox()
        result = sbx.commands.run(argv=["ls", "-la"])
        self.assertEqual(result.args, ("ls", "-la"))
        self.assertEqual(runtime.exec_calls, [["ls", "-la"]])

    def test_check_raises_on_nonzero(self) -> None:
        sbx, _system, runtime = _make_sandbox()

        def _boom(*_a, **_k):
            return SandboxExecResult(args=("false",), returncode=1, stdout="", stderr="nope")

        runtime.exec = _boom  # type: ignore[assignment]
        with self.assertRaises(RuntimeError):
            sbx.commands.run("false", check=True)


# ---------------------------------------------------------------------------
# checkpoint=True -> AsyncCheckpoint with pre-allocated id
# ---------------------------------------------------------------------------


class AsyncCheckpointTests(unittest.TestCase):
    def test_checkpoint_true_preallocates_id_and_waits(self) -> None:
        sbx, system, _runtime = _make_sandbox()
        result = sbx.commands.run("make", checkpoint=True)
        self.assertIsInstance(result.checkpoint, AsyncCheckpoint)
        # id is available immediately, before the background job finishes.
        self.assertTrue(result.checkpoint.checkpoint_id.startswith("ckpt-"))
        returned_id = result.checkpoint.wait(timeout=5.0)
        self.assertEqual(returned_id, result.checkpoint.checkpoint_id)
        self.assertTrue(result.checkpoint.done)
        # The daemon received the client-preallocated id verbatim.
        self.assertEqual(len(system.checkpoint_calls), 1)
        self.assertEqual(
            system.checkpoint_calls[0]["checkpoint_id"],
            result.checkpoint.checkpoint_id,
        )

    def test_done_is_false_until_barrier_released(self) -> None:
        sbx, system, _runtime = _make_sandbox()
        system.checkpoint_barrier = threading.Event()
        result = sbx.commands.run("make", checkpoint=True)
        handle = result.checkpoint
        assert handle is not None
        self.assertFalse(handle.done)
        with self.assertRaises(TimeoutError):
            handle.wait(timeout=0.1)
        system.checkpoint_barrier.set()
        handle.wait(timeout=5.0)
        self.assertTrue(handle.done)

    def test_checkpoint_failure_propagates_on_wait(self) -> None:
        sbx, system, _runtime = _make_sandbox()

        def _fail(*_a, **_k):
            raise RuntimeError("checkpoint blew up")

        system.checkpoint_once = _fail  # type: ignore[assignment]
        result = sbx.commands.run("make", checkpoint=True)
        assert result.checkpoint is not None
        with self.assertRaises(RuntimeError):
            result.checkpoint.wait(timeout=5.0)


class AutoCheckpointTests(unittest.TestCase):
    def test_sandbox_auto_checkpoint_triggers_without_flag(self) -> None:
        sbx, system, _runtime = _make_sandbox(auto_checkpoint=True)
        result = sbx.commands.run("echo hi")
        self.assertIsInstance(result.checkpoint, AsyncCheckpoint)
        result.checkpoint.wait(timeout=5.0)
        self.assertEqual(len(system.checkpoint_calls), 1)

    def test_auto_and_explicit_do_not_double_checkpoint(self) -> None:
        sbx, system, _runtime = _make_sandbox(auto_checkpoint=True)
        result = sbx.commands.run("echo hi", checkpoint=True)
        assert result.checkpoint is not None
        result.checkpoint.wait(timeout=5.0)
        # A single run performs a single checkpoint even when both the
        # sandbox-level flag and the per-call flag are set.
        self.assertEqual(len(system.checkpoint_calls), 1)

    def test_no_checkpoint_when_disabled(self) -> None:
        sbx, system, _runtime = _make_sandbox(auto_checkpoint=False)
        result = sbx.commands.run("echo hi")
        self.assertIsNone(result.checkpoint)
        self.assertEqual(system.checkpoint_calls, [])


# ---------------------------------------------------------------------------
# changeset=True -> AsyncChangeset (async default) / list (sync)
# ---------------------------------------------------------------------------


class ChangesetActionTests(unittest.TestCase):
    def test_changeset_async_default_no_prev_falls_back_to_fork(self) -> None:
        # No prior checkpoint -> the async changeset can only diff against
        # the fork point, so it falls back to fork_changeset (which errors
        # on non-fork sandboxes in reality, but our fake returns entries).
        sbx, system, _runtime = _make_sandbox()
        result = sbx.commands.run("touch x", changeset=True)
        self.assertIsInstance(result.changeset, AsyncChangeset)
        entries = result.changeset.wait(timeout=5.0)
        self.assertEqual(entries, [{"path": "/new.txt", "change": "added"}])
        self.assertTrue(result.changeset.done)
        # No prev checkpoint -> fork_changeset(force=True), no since call.
        self.assertEqual(system.fork_calls, [(SandboxId("sbx-test"), True)])
        self.assertEqual(system.since_calls, [])

    def test_changeset_async_uses_previous_checkpoint_as_since(self) -> None:
        # After a checkpoint=True run, the next changeset=True must diff
        # against the *previous* checkpoint id, not the fresh one being
        # created concurrently (semantics: "what did this action change?").
        sbx, system, _runtime = _make_sandbox()
        # Establish a prior checkpoint (populates last_checkpoint_id).
        first = sbx.commands.run("echo prep", checkpoint=True)
        first_ckpt = first.checkpoint.wait(timeout=5.0)
        self.assertEqual(sbx.last_checkpoint_id, first_ckpt)
        # Now issue a changeset-only run: it must call changeset_since
        # with `first_ckpt` and skip fork_changeset entirely.
        result = sbx.commands.run("touch x", changeset=True)
        entries = result.changeset.wait(timeout=5.0)
        self.assertEqual(entries, [{"path": "/since.txt", "change": "added"}])
        self.assertEqual(len(system.since_calls), 1)
        called_sid, called_ckpt, use_gate = system.since_calls[0]
        self.assertEqual(called_sid, SandboxId("sbx-test"))
        self.assertEqual(str(called_ckpt), first_ckpt)
        self.assertFalse(use_gate)  # force=True flips use_inspector_gate off
        # fork_changeset must NOT be called on the second run.
        self.assertEqual(system.fork_calls, [])

    def test_changeset_uses_prev_when_checkpoint_and_changeset_both_true(self) -> None:
        # checkpoint=True + changeset=True on the same run: the changeset
        # must use the *previous* checkpoint as since, NOT the fresh one
        # allocated by this run (which hasn't captured pre-exec state).
        sbx, system, _runtime = _make_sandbox()
        first = sbx.commands.run("echo prep", checkpoint=True)
        first_ckpt = first.checkpoint.wait(timeout=5.0)
        result = sbx.commands.run("touch x", checkpoint=True, changeset=True)
        # A fresh checkpoint id was pre-allocated, and last_checkpoint_id
        # advanced. But the changeset was tied to the prior id.
        new_ckpt = result.checkpoint.checkpoint_id
        self.assertNotEqual(new_ckpt, first_ckpt)
        self.assertEqual(sbx.last_checkpoint_id, new_ckpt)
        result.checkpoint.wait(timeout=5.0)
        entries = result.changeset.wait(timeout=5.0)
        self.assertEqual(entries, [{"path": "/since.txt", "change": "added"}])
        called_ckpt = system.since_calls[0][1]
        self.assertEqual(str(called_ckpt), first_ckpt)

    def test_changeset_sync_returns_list(self) -> None:
        # Sync mode with a prior checkpoint uses changeset_since directly.
        sbx, system, _runtime = _make_sandbox()
        first = sbx.commands.run("echo prep", checkpoint=True)
        first_ckpt = first.checkpoint.wait(timeout=5.0)
        result = sbx.commands.run(
            "touch x", changeset=True, changeset_sync=True
        )
        self.assertIsInstance(result.changeset, list)
        self.assertEqual(result.changeset, [{"path": "/since.txt", "change": "added"}])
        self.assertEqual(system.since_calls[0][1], CheckpointId(first_ckpt))

    def test_changeset_sync_falls_back_to_fork_without_prev(self) -> None:
        # Sync mode without a prior checkpoint also falls back to fork.
        sbx, system, _runtime = _make_sandbox()
        result = sbx.commands.run("touch x", changeset=True, changeset_sync=True)
        self.assertIsInstance(result.changeset, list)
        self.assertEqual(result.changeset, [{"path": "/new.txt", "change": "added"}])
        self.assertEqual(system.fork_calls, [(SandboxId("sbx-test"), True)])

    def test_no_changeset_when_not_requested(self) -> None:
        sbx, system, _runtime = _make_sandbox()
        result = sbx.commands.run("noop")
        self.assertIsNone(result.changeset)
        self.assertEqual(system.fork_calls, [])
        self.assertEqual(system.since_calls, [])


# ---------------------------------------------------------------------------
# observe=True -> inspector peek (read-only, no reset)
# ---------------------------------------------------------------------------


class ObserveTests(unittest.TestCase):
    def test_observe_populates_flags(self) -> None:
        inspector = _FakeInspector(filesystem_changed=True, process_changed=False)
        sbx, _system, _runtime = _make_sandbox(inspector=inspector)
        result = sbx.commands.run("touch x", observe=True)
        self.assertTrue(result.filesystem_changed)
        self.assertFalse(result.process_changed)
        self.assertEqual(inspector.inspect_calls, [SandboxId("sbx-test")])

    def test_observe_absent_by_default(self) -> None:
        inspector = _FakeInspector(filesystem_changed=True)
        sbx, _system, _runtime = _make_sandbox(inspector=inspector)
        result = sbx.commands.run("touch x")
        self.assertIsNone(result.filesystem_changed)
        self.assertIsNone(result.process_changed)
        self.assertEqual(inspector.inspect_calls, [])

    def test_peek_runs_before_async_checkpoint_reset(self) -> None:
        # Regression: `observe=True` combined with `checkpoint=True` used
        # to race with the background checkpoint's `mark_checkpoint_complete`
        # → inspector `reset()` (scorched-earth wipe of the dirty cursors).
        # For remote SDK clients the peek RTT (SDK → gateway → daemon) is
        # slower than the daemon-local reset, so the reset won and the
        # ActionResult reported `filesystem_changed=False` for an action
        # that clearly mutated the FS. The fix reorders `_build_action_result`
        # to peek BEFORE `_start_async_checkpoint` spawns its worker, so
        # the peek is guaranteed to see the pre-reset state.
        #
        # We express the invariant deterministically by having the fake
        # `checkpoint_once` mirror the real reset side-effect (zeroing the
        # inspector's dirty flags). If peek ran after the checkpoint thread
        # had already run, it would observe the post-reset False; because
        # peek now runs first on the main thread, it captures True.
        inspector = _FakeInspector(filesystem_changed=True, process_changed=True)
        sbx, system, _runtime = _make_sandbox(inspector=inspector)

        call_order: list[str] = []
        orig_inspect = inspector.inspect

        def tracking_inspect(sandbox_id):
            call_order.append("peek")
            return orig_inspect(sandbox_id)

        inspector.inspect = tracking_inspect  # type: ignore[assignment]

        orig_checkpoint = system.checkpoint_once

        def resetting_checkpoint(sandbox_id, *, leave_running=True, checkpoint_id=None):
            # Mirror host-inspector's `reset()` triggered by
            # `mark_checkpoint_complete`: scorched-earth wipe of both
            # dirty axes. Any peek reading the inspector after this
            # point would incorrectly see False.
            inspector.filesystem_changed = False
            inspector.process_changed = False
            call_order.append("checkpoint")
            return orig_checkpoint(
                sandbox_id,
                leave_running=leave_running,
                checkpoint_id=checkpoint_id,
            )

        system.checkpoint_once = resetting_checkpoint  # type: ignore[assignment]

        result = sbx.commands.run("touch x", checkpoint=True, observe=True)
        assert result.checkpoint is not None
        result.checkpoint.wait(timeout=5.0)

        # Peek must be recorded before the checkpoint side-effect. This
        # is deterministic under the fix: peek runs synchronously on the
        # main thread before `_start_async_checkpoint` spawns the worker.
        self.assertEqual(
            call_order[0],
            "peek",
            f"observe peek must precede the checkpoint reset; got {call_order}",
        )
        # And the ActionResult snapshots the pre-reset value.
        self.assertTrue(result.filesystem_changed)
        self.assertTrue(result.process_changed)
        # Sanity: the reset side-effect did in fact run (via the bg thread),
        # proving this scenario would corrupt the peek result if peek were
        # ordered after `_start_async_checkpoint`.
        self.assertFalse(inspector.filesystem_changed)
        self.assertFalse(inspector.process_changed)
        self.assertEqual(inspector.inspect_calls, [SandboxId("sbx-test")])
        self.assertEqual(len(system.checkpoint_calls), 1)


# ---------------------------------------------------------------------------
# ExecStream: iterate, then read .result
# ---------------------------------------------------------------------------


class ExecStreamTests(unittest.TestCase):
    def test_stream_yields_events_then_result(self) -> None:
        sbx, system, _runtime = _make_sandbox()
        stream = sbx.commands.stream("make test", checkpoint=True, observe=True)
        self.assertIsInstance(stream, ExecStream)
        texts = []
        for event in stream:
            if isinstance(event, ExecEvent):
                texts.append(event.text)
        self.assertEqual(texts, ["line1\n", "line2\n"])
        result = stream.result
        self.assertIsInstance(result, ActionResult)
        self.assertEqual(result.returncode, 0)
        self.assertIsInstance(result.checkpoint, AsyncCheckpoint)
        result.checkpoint.wait(timeout=5.0)
        self.assertEqual(len(system.checkpoint_calls), 1)

    def test_result_before_consumption_raises(self) -> None:
        sbx, _system, _runtime = _make_sandbox()
        stream = sbx.commands.stream("make test")
        with self.assertRaises(RuntimeError):
            _ = stream.result

    def test_stream_changeset_sync(self) -> None:
        sbx, system, _runtime = _make_sandbox()
        stream = sbx.commands.stream("build", changeset=True, changeset_sync=True)
        list(stream)  # drain
        self.assertEqual(stream.result.changeset, [{"path": "/new.txt", "change": "added"}])
        self.assertEqual(system.fork_calls, [(SandboxId("sbx-test"), True)])


if __name__ == "__main__":
    unittest.main()
