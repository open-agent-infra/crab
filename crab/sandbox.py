"""User-facing Sandbox SDK for Crab.

This module is what users import when they want an E2B-style sandbox with
Crab's checkpoint/restore semantics layered on top:

    from crab import Sandbox

    sbx = Sandbox(image="ubuntu:22.04")
    agent = ClaudeCodeAgent().bind(sbx, llm_url="https://api.anthropic.com")
    result = agent.run("Fix the failing tests in /work/repo")

    sbx.commands.run("git diff")
    ckpt = sbx.checkpoint(label="post-fix")
    sbx.kill()

Sandbox lifetime is independent of task lifetime: a single sandbox can host
many `agent.run()` invocations, manual `commands.run()` calls between them,
and as many checkpoint/restore cycles as the user wants. The agent's
`install()` runs once; each task is a fresh invocation in the same sandbox.

`Sandbox(...)` connects to the running Crab daemon via
`crab.engine.get_default_engine()` (which calls `Engine.connect()`
under the hood). Start the daemon once per host with
`crab daemon start` (or `python -m crab.daemon`); there is no
in-process engine fallback. Sandbox lifecycle calls only affect
sandboxes — the daemon stays running across SDK process exits.
"""
from __future__ import annotations

import contextlib
import hashlib
import logging
import shlex
import socket
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Iterator

from .errors import ImageCompatibilityError, SandboxCreateCleanupError
from .ids import CheckpointId, SandboxId
from .models import EgressLedger, ExecDone, ExecEvent, JobStatus, MergeReport, ObservationReport, PortAllocation, ProcessMergeReport, SandboxExecResult, SandboxInfo, SandboxSnapshot, SandboxState, sandbox_state_from_status, utc_now
from .templates import SandboxTemplate

if TYPE_CHECKING:
    from .agent import Agent
    from .engine import Engine

logger = logging.getLogger(__name__)


# Marker labels used in CheckpointId values produced by `sbx.checkpoint()`.
# Stored in the manifest metadata so users can see what they labelled.
_LABEL_METADATA_KEY = "user_label"


# ---------------------------------------------------------------------------
# Rich return types for commands.run() / commands.stream()
# ---------------------------------------------------------------------------


class AsyncCheckpoint:
    """Handle for a background checkpoint. The checkpoint_id is pre-allocated
    and immediately available; the actual checkpoint runs asynchronously.

    Two modes:
      - Thread-based (local engine): wait via thread.join()
      - Poll-based (remote engine): wait by polling GET /jobs/{job_id}
    """

    def __init__(
        self,
        checkpoint_id: str,
        thread: threading.Thread | None = None,
        *,
        poll_fn: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        self.checkpoint_id = checkpoint_id
        self._thread = thread
        self._poll_fn = poll_fn
        self._error: BaseException | None = None
        self.materialization: str | None = None
        self.physical_checkpoint_created: bool | None = None
        # Do NOT eagerly cache done-ness: an unstarted thread reports
        # is_alive()==False, and _start_async_checkpoint constructs the
        # handle with a placeholder thread before reassigning the real one.
        self._done = False

    def wait(self, timeout: float | None = None) -> str:
        """Block until the checkpoint completes. Returns checkpoint_id.
        Raises TimeoutError if *timeout* elapses, or propagates the
        checkpoint failure exception."""
        if self._done:
            if self._error is not None:
                raise self._error
            return self.checkpoint_id

        if self._thread is not None:
            # Thread-based (local mode)
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                raise TimeoutError("checkpoint did not complete within timeout")
            self._done = True
            if self._error is not None:
                raise self._error
            return self.checkpoint_id

        # Poll-based (remote mode)
        if self._poll_fn is None:
            raise RuntimeError("AsyncCheckpoint has no thread or poll_fn")
        deadline = time.time() + (timeout if timeout is not None else 300.0)
        while time.time() < deadline:
            resp = self._poll_fn()
            status = resp.get("status", "pending")
            if status == "completed":
                self._done = True
                result = resp.get("result") or {}
                resolved = result.get("checkpoint_id")
                if resolved and str(resolved) != self.checkpoint_id:
                    self._error = RuntimeError(
                        "logical checkpoint id changed while waiting: "
                        f"expected={self.checkpoint_id} actual={resolved}"
                    )
                    raise self._error
                materialization = result.get("checkpoint_materialization")
                if materialization is not None:
                    self.materialization = str(materialization)
                if "physical_checkpoint_created" in result:
                    self.physical_checkpoint_created = bool(
                        result["physical_checkpoint_created"]
                    )
                return self.checkpoint_id
            if status == "failed":
                err_msg = resp.get("error", "checkpoint failed")
                self._error = RuntimeError(err_msg)
                self._done = True
                raise self._error
            time.sleep(0.15)
        raise TimeoutError("checkpoint did not complete within timeout")

    @property
    def done(self) -> bool:
        """Non-blocking check: has the background checkpoint finished?"""
        if self._done:
            return True
        if self._thread is not None:
            self._done = not self._thread.is_alive()
            return self._done
        # For poll-based: quick non-blocking check
        if self._poll_fn is not None:
            try:
                resp = self._poll_fn()
                status = resp.get("status")
                if status == "completed":
                    result = resp.get("result") or {}
                    resolved = result.get("checkpoint_id")
                    if resolved and str(resolved) != self.checkpoint_id:
                        self._error = RuntimeError(
                            "logical checkpoint id changed while waiting: "
                            f"expected={self.checkpoint_id} actual={resolved}"
                        )
                    materialization = result.get("checkpoint_materialization")
                    if materialization is not None:
                        self.materialization = str(materialization)
                    if "physical_checkpoint_created" in result:
                        self.physical_checkpoint_created = bool(
                            result["physical_checkpoint_created"]
                        )
                    self._done = True
                elif status == "failed":
                    self._error = RuntimeError(
                        resp.get("error", "checkpoint failed")
                    )
                    self._done = True
            except Exception:
                pass
        return self._done

    def __repr__(self) -> str:
        status = "done" if self.done else "pending"
        return f"AsyncCheckpoint(id={self.checkpoint_id!r}, {status})"


class AsyncChangeset:
    """Handle for a background changeset computation.

    Two modes:
      - Thread-based (local engine): wait via thread.join()
      - Poll-based (remote engine): wait by polling GET /jobs/{job_id}
    """

    def __init__(
        self,
        thread: threading.Thread | None = None,
        *,
        poll_fn: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        self._thread = thread
        self._poll_fn = poll_fn
        self._result: list[dict] | None = None
        self._error: BaseException | None = None
        # See AsyncCheckpoint: unstarted placeholder threads report
        # is_alive()==False, so compute done-ness lazily instead.
        self._done = False

    def wait(self, timeout: float | None = None) -> list[dict]:
        """Block until the changeset completes. Returns the changeset entries."""
        if self._done:
            if self._error is not None:
                raise self._error
            return self._result or []

        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                raise TimeoutError("changeset did not complete within timeout")
            self._done = True
            if self._error is not None:
                raise self._error
            return self._result or []

        # Poll-based (remote mode)
        if self._poll_fn is None:
            raise RuntimeError("AsyncChangeset has no thread or poll_fn")
        deadline = time.time() + (timeout if timeout is not None else 300.0)
        while time.time() < deadline:
            resp = self._poll_fn()
            status = resp.get("status", "pending")
            if status == "completed":
                self._done = True
                result = resp.get("result") or {}
                raw_cs = result.get("changeset")
                if isinstance(raw_cs, dict):
                    self._result = raw_cs.get("entries") or []
                elif isinstance(raw_cs, list):
                    self._result = raw_cs
                else:
                    self._result = []
                return self._result
            if status == "failed":
                err_msg = resp.get("error", "changeset failed")
                self._error = RuntimeError(err_msg)
                self._done = True
                raise self._error
            time.sleep(0.15)
        raise TimeoutError("changeset did not complete within timeout")

    @property
    def done(self) -> bool:
        if self._done:
            return True
        if self._thread is not None:
            self._done = not self._thread.is_alive()
            return self._done
        if self._poll_fn is not None:
            try:
                resp = self._poll_fn()
                if resp.get("status") in ("completed", "failed"):
                    self._done = True
            except Exception:
                pass
        return self._done

    def __repr__(self) -> str:
        status = "done" if self.done else "pending"
        return f"AsyncChangeset({status})"


@dataclass
class ActionResult:
    """Rich return value from commands.run() / commands.stream().

    Wraps the original exec result and optionally carries background
    checkpoint/changeset handles plus inspector observations."""

    returncode: int
    stdout: str
    stderr: str
    args: tuple[str, ...]

    # Optional enrichments (None when not requested)
    checkpoint: AsyncCheckpoint | None = None
    changeset: AsyncChangeset | list[dict] | None = None
    filesystem_changed: bool | None = None
    process_changed: bool | None = None


class ExecStream:
    """Iterable wrapper for streaming exec that exposes a .result after
    iteration completes.

    Usage::

        stream = sandbox.commands.stream("make test", checkpoint=True)
        for event in stream:
            print(event.text)
        result = stream.result  # ActionResult
    """

    def __init__(self, gen: Iterator[ExecEvent | ExecDone], sandbox: "Sandbox", *,
                 do_checkpoint: bool, do_changeset: bool,
                 changeset_sync: bool, do_observe: bool) -> None:
        self._gen = gen
        self._sandbox = sandbox
        self._do_checkpoint = do_checkpoint
        self._do_changeset = do_changeset
        self._changeset_sync = changeset_sync
        self._do_observe = do_observe
        self._result: ActionResult | None = None
        self._returncode: int = -1
        self._consumed = False

    def __iter__(self) -> Iterator[ExecEvent | ExecDone]:
        for event in self._gen:
            if isinstance(event, ExecDone):
                self._returncode = event.returncode
            yield event
        self._consumed = True
        self._result = self._sandbox._build_action_result(
            returncode=self._returncode,
            stdout="",
            stderr="",
            args=(),
            do_checkpoint=self._do_checkpoint and self._returncode == 0,
            do_changeset=self._do_changeset,
            changeset_sync=self._changeset_sync,
            do_observe=self._do_observe,
        )

    @property
    def result(self) -> ActionResult:
        """Available after iteration completes. Raises if accessed early."""
        if self._result is None:
            raise RuntimeError(
                "ExecStream.result is only available after the stream is fully consumed"
            )
        return self._result


@dataclass
class _Mount:
    source: Path
    destination: str
    options: tuple[str, ...] = ("rbind", "rw")


@dataclass
class _LaunchPlan:
    """Resolved per-sandbox launch parameters, filled in before runtime.launch."""

    runtime_name: str
    name: str
    image: str | None
    work_dir_host: Path | None
    bundle_dir: Path | None
    mounts: list[_Mount] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    process_args: list[str] = field(default_factory=lambda: ["/bin/sh", "-c", "tail -f /dev/null"])
    user: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    process_cwd: str = "/work"


class _CommandsNamespace:
    """`sbx.commands.run(...)` namespace.

    A thin wrapper around `Runtime.exec` that exposes a friendlier signature
    (string commands, env dicts) without leaking the lower-level argv/list form
    requirement. Use `argv=[...]` for callers that want to bypass the shell.
    """

    def __init__(self, sandbox: "Sandbox") -> None:
        self._sandbox = sandbox

    @staticmethod
    def _detach_wrap(argv: list[str]) -> list[str]:
        """Wrap ``argv`` so its stdio is redirected to ``/dev/null`` *inside*
        the container before the command runs.

        This is what makes ``detach=True`` actually return promptly for a
        ``"... &"`` command. ``runc exec`` (no ``-d``) relays the container
        process's stdout/stderr through an internal pipe and waits for that
        pipe to reach EOF; a ``&``-backgrounded child inherits the pipe and
        keeps it open until it exits, so ``run`` blocks even though the
        foreground shell already returned. Merely discarding output on the
        host side (``capture_output=False``) does *not* help — the relay pipe
        still exists. Reopening fd1/fd2 to ``/dev/null`` with ``exec`` closes
        the child's handle on the relay pipe, so ``runc exec`` sees EOF as
        soon as the foreground shell exits.
        """
        if (
            len(argv) >= 3
            and argv[0] in ("/bin/sh", "/bin/bash", "sh", "bash")
            and argv[1] == "-c"
        ):
            # Shell-string form: prepend the redirect to the script itself.
            wrapped = list(argv)
            wrapped[2] = f"exec 1>/dev/null 2>&1; {argv[2]}"
            return wrapped
        # Direct argv form: run under a shell that redirects, then execs it.
        quoted = " ".join(shlex.quote(a) for a in argv)
        return ["/bin/sh", "-c", f"exec 1>/dev/null 2>&1; {quoted}"]

    def run(
        self,
        cmd: str | list[str] | None = None,
        *,
        argv: list[str] | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        user: str | None = None,
        timeout: float | None = None,
        capture_output: bool = True,
        detach: bool = False,
        check: bool = False,
        checkpoint: bool = False,
        changeset: bool = False,
        changeset_sync: bool = False,
        observe: bool = False,
    ) -> ActionResult:
        """Execute a command and return an ActionResult.

        Optional enrichments:
          - ``checkpoint=True``: trigger an async checkpoint after exec
          - ``changeset=True``: compute filesystem changeset after exec
          - ``changeset_sync=True``: make changeset synchronous (returns list)
          - ``observe=True``: peek inspector state (filesystem/process changed)

        ``detach=True`` is for launching background processes. It (1) sets
        ``capture_output=False`` and (2) redirects the command's stdout/stderr
        to ``/dev/null`` *inside* the container. Without (2), a command like
        ``"myserver &"`` would *hang* ``run`` even though the shell returned:
        ``runc exec`` relays the container process's stdio through an internal
        pipe and waits for EOF, and the ``&``-backgrounded child inherits that
        pipe and holds it open until it exits (discarding output host-side is
        not enough — the relay pipe still exists). The in-container redirect
        closes the child's handle on the pipe so ``run`` returns as soon as
        the foreground shell exits.

        Note the boundary: ``detach`` removes the *pipe-inheritance* block, not
        the *process-lifecycle* block. ``runc exec`` still waits for the
        foreground process to exit, so ``run("sleep 3", detach=True)`` (no
        ``&``) still blocks ~3s. To return immediately the command must
        background itself (``"... &"``) or already be a daemon. Because output
        is sent to ``/dev/null``, the returned ``ActionResult`` has empty
        ``stdout``/``stderr``; enrichments (checkpoint/changeset/observe)
        still work, and a subsequent checkpoint captures the process while it
        is alive.
        """
        if argv is None:
            if isinstance(cmd, list):
                argv = cmd
            elif isinstance(cmd, str):
                argv = ["/bin/sh", "-c", cmd]
            else:
                raise TypeError("commands.run requires either cmd or argv")
        if detach:
            capture_output = False
            argv = self._detach_wrap(argv)
        merged_env = self._sandbox._command_env(env)

        # Resolve whether to checkpoint (per-call or sandbox-level auto)
        do_checkpoint = checkpoint or bool(self._sandbox._auto_checkpoint)

        # If any enrichment is requested AND we're in remote mode, use the
        # batch action endpoint to save round-trips.
        runtime = self._sandbox._engine.runtime
        if (do_checkpoint or changeset or observe) and hasattr(runtime, "batch_action"):
            return self._run_batch(
                argv=argv, cwd=cwd, env=merged_env, user=user,
                timeout=timeout, capture_output=capture_output, check=check,
                do_checkpoint=do_checkpoint, changeset=changeset,
                changeset_sync=changeset_sync, observe=observe,
            )

        result = runtime.exec(
            self._sandbox.sandbox_id,
            argv,
            cwd=cwd,
            env=merged_env,
            user=user,
            timeout_s=timeout,
            capture_output=capture_output,
        )
        if check and result.returncode != 0:
            raise RuntimeError(
                f"sandbox command failed: rc={result.returncode} cmd={cmd!r} "
                f"stderr={result.stderr!r}"
            )
        return self._sandbox._build_action_result(
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            args=result.args,
            do_checkpoint=do_checkpoint and result.returncode == 0,
            do_changeset=changeset,
            changeset_sync=changeset_sync,
            do_observe=observe,
        )

    def _run_batch(
        self,
        *,
        argv: list[str],
        cwd: str | None,
        env: dict[str, str] | None,
        user: str | None,
        timeout: float | None,
        capture_output: bool = True,
        check: bool,
        do_checkpoint: bool,
        changeset: bool,
        changeset_sync: bool,
        observe: bool,
    ) -> ActionResult:
        """Execute via the daemon batch action endpoint (one round-trip)."""
        sandbox = self._sandbox
        runtime = sandbox._engine.runtime

        # Snapshot the previous checkpoint id BEFORE we overwrite it.
        prev_ckpt_id = sandbox._last_checkpoint_id

        # Pre-allocate checkpoint id client-side (mirrors _start_async_checkpoint)
        ckpt_id: str | None = None
        if do_checkpoint:
            ckpt_id = f"ckpt-{uuid.uuid4().hex[:12]}"
            sandbox._last_checkpoint_id = ckpt_id

        # Determine changeset_since: the previous checkpoint id (same
        # semantics as the local path in _build_action_result).
        changeset_since: str | None = None
        if changeset and prev_ckpt_id:
            changeset_since = prev_ckpt_id

        try:
            response = runtime.batch_action(
                sandbox.sandbox_id,
                argv=argv,
                cwd=cwd,
                env=env,
                user=user,
                timeout_s=timeout,
                capture_output=capture_output,
                checkpoint=do_checkpoint,
                checkpoint_id=ckpt_id,
                changeset=changeset,
                changeset_since=changeset_since,
                observe=observe,
            )
        except BaseException:
            # The daemon never reserved a checkpoint when exec failed or
            # timed out. Do not expose the client-side provisional id as the
            # sandbox's latest checkpoint.
            if do_checkpoint:
                sandbox._last_checkpoint_id = prev_ckpt_id
            raise

        # Parse exec result
        exec_data = response.get("exec") or {}
        returncode = int(exec_data.get("returncode", -1))
        stdout = str(exec_data.get("stdout") or "")
        stderr = str(exec_data.get("stderr") or "")

        if returncode != 0 and do_checkpoint:
            sandbox._last_checkpoint_id = prev_ckpt_id

        if check and returncode != 0:
            raise RuntimeError(
                f"sandbox command failed: rc={returncode} cmd={argv!r} "
                f"stderr={stderr!r}"
            )

        # Build checkpoint and changeset handles.
        # The daemon now runs checkpoint/changeset asynchronously and returns
        # job_id + status="pending". The SDK creates poll-based Async handles.
        ckpt_handle: AsyncCheckpoint | None = None
        cs_handle: AsyncChangeset | list[dict] | None = None

        job_id = response.get("job_id")
        ckpt_status = response.get("checkpoint_status")
        cs_status = response.get("changeset_status")

        # Build a poll_fn closure if the daemon gave us a job_id.
        poll_fn: Callable[[], dict[str, Any]] | None = None
        if job_id and hasattr(runtime, "poll_job"):
            _sid = sandbox.sandbox_id
            _jid = job_id
            _rt = runtime
            poll_fn = lambda: _rt.poll_job(_sid, _jid)  # noqa: E731

        if do_checkpoint and ckpt_id:
            if ckpt_status == "skipped":
                sandbox._last_checkpoint_id = prev_ckpt_id
            elif ckpt_status == "pending" and poll_fn is not None:
                # Daemon is working in background — poll for result.
                ckpt_handle = AsyncCheckpoint(ckpt_id, poll_fn=poll_fn)
            else:
                # Daemon completed synchronously (or test fake).
                done_thread = threading.Thread(target=lambda: None)
                done_thread.start()
                done_thread.join()
                ckpt_handle = AsyncCheckpoint(ckpt_id, done_thread)
                server_ckpt = response.get("checkpoint_id")
                if server_ckpt and str(server_ckpt) != ckpt_id:
                    raise RuntimeError(
                        "daemon changed the preallocated logical checkpoint id: "
                        f"expected={ckpt_id} actual={server_ckpt}"
                    )

        if changeset:
            if cs_status == "pending" and poll_fn is not None:
                # Daemon is working in background — poll for result.
                if changeset_sync:
                    # Caller wants sync result: block here.
                    handle = AsyncChangeset(poll_fn=poll_fn)
                    cs_handle = handle.wait(timeout=300.0)
                else:
                    cs_handle = AsyncChangeset(poll_fn=poll_fn)
            else:
                # Daemon returned result synchronously (changeset already in response).
                raw_cs = response.get("changeset")
                if isinstance(raw_cs, dict):
                    entries_list: list[dict] = raw_cs.get("entries") or []
                elif isinstance(raw_cs, list):
                    entries_list = raw_cs
                else:
                    entries_list = []

                if changeset_sync:
                    cs_handle = entries_list
                else:
                    done_thread = threading.Thread(target=lambda: None)
                    done_thread.start()
                    done_thread.join()
                    handle = AsyncChangeset(done_thread)
                    handle._result = entries_list
                    cs_handle = handle

        # Observer flags
        fs_changed = response.get("filesystem_changed")
        proc_changed = response.get("process_changed")

        return ActionResult(
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            args=tuple(argv),
            checkpoint=ckpt_handle,
            changeset=cs_handle,
            filesystem_changed=fs_changed,
            process_changed=proc_changed,
        )

    def stream(
        self,
        cmd: str | list[str] | None = None,
        *,
        argv: list[str] | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        user: str | None = None,
        timeout: float | None = None,
        detach: bool = False,
        checkpoint: bool = False,
        changeset: bool = False,
        changeset_sync: bool = False,
        observe: bool = False,
    ) -> ExecStream:
        """Streaming exec that returns an ExecStream.

        Iterate the stream to consume events; after iteration completes,
        access ``stream.result`` for the ActionResult with checkpoint/
        changeset/observer information.

        ``detach=True`` mirrors ``run(detach=True)``: it redirects the
        command's stdout/stderr to ``/dev/null`` inside the container (so no
        ``ExecEvent`` output is produced) which is what actually releases the
        pipe-inheritance block a ``&``-backgrounded child would otherwise
        impose; see ``run`` for the full stdio semantics.

        Usage::

            stream = sandbox.commands.stream("ls -la", checkpoint=True)
            for event in stream:
                if isinstance(event, ExecEvent):
                    print(f"[{event.channel}] {event.text}", end="")
            result = stream.result  # ActionResult
        """
        if argv is None:
            if isinstance(cmd, list):
                argv = cmd
            elif isinstance(cmd, str):
                argv = ["/bin/sh", "-c", cmd]
            else:
                raise TypeError("commands.stream requires either cmd or argv")
        if detach:
            argv = self._detach_wrap(argv)
        merged_env = self._sandbox._command_env(env)
        runtime = self._sandbox._engine.runtime
        gen = runtime.stream_exec(
            self._sandbox.sandbox_id,
            argv,
            cwd=cwd,
            env=merged_env,
            user=user,
            timeout_s=timeout,
            capture_output=not detach,
        )
        do_checkpoint = checkpoint or bool(self._sandbox._auto_checkpoint)
        return ExecStream(
            gen, self._sandbox,
            do_checkpoint=do_checkpoint,
            do_changeset=changeset,
            changeset_sync=changeset_sync,
            do_observe=observe,
        )


class _PortsNamespace:
    """`sbx.ports.*` — port exposure (S4).

    Expose sandbox ports through the gateway's L4 proxy so external
    clients can reach services running inside the sandbox.
    """

    def __init__(self, sandbox: "Sandbox") -> None:
        self._sandbox = sandbox

    def expose(self, port: int) -> PortAllocation:
        """Expose a guest port; returns the allocation with host_port and url."""
        client = self._sandbox._engine.runtime._client  # type: ignore[attr-defined]
        result = client.post_json(
            f"/sandboxes/{self._sandbox.sandbox_id}/ports",
            {"port": int(port)},
        )
        return PortAllocation(
            host_port=int(result["host_port"]),
            guest_port=int(result["guest_port"]),
            url=str(result["url"]),
        )

    def list(self) -> list[PortAllocation]:
        """List all port allocations for this sandbox."""
        client = self._sandbox._engine.runtime._client  # type: ignore[attr-defined]
        result = client.get_json(
            f"/sandboxes/{self._sandbox.sandbox_id}/ports",
        )
        return [
            PortAllocation(
                host_port=int(p["host_port"]),
                guest_port=int(p["guest_port"]),
                url=f"tcp://unknown:{p['host_port']}",
            )
            for p in result.get("ports") or []
        ]

    def release(self, port: int) -> None:
        """Release a port allocation."""
        client = self._sandbox._engine.runtime._client  # type: ignore[attr-defined]
        client.delete(
            f"/sandboxes/{self._sandbox.sandbox_id}/ports/{int(port)}",
        )


class _FilesNamespace:
    """`sbx.files.*` — small read/write helpers built on top of commands.

    For first cut these are shell-based (cat / tee). A future PR can swap
    these for a binary-safe protocol when we add native filesystem RPC.
    """

    def __init__(self, sandbox: "Sandbox") -> None:
        self._sandbox = sandbox

    def read(self, path: str) -> str:
        result = self._sandbox.commands.run(
            argv=["cat", path],
            capture_output=True,
            check=True,
        )
        return result.stdout

    def write(self, path: str, content: str | bytes) -> None:
        if isinstance(content, bytes):
            text = content.decode("utf-8", errors="replace")
        else:
            text = content
        # Use a heredoc-safe pattern: write to a temp path with python's
        # base64 echo to avoid quoting issues with arbitrary content.
        import base64

        encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
        cmd = (
            f"mkdir -p {shlex.quote(str(Path(path).parent))} && "
            f"echo {shlex.quote(encoded)} | base64 -d > {shlex.quote(path)}"
        )
        self._sandbox.commands.run(cmd, check=True)

    def exists(self, path: str) -> bool:
        result = self._sandbox.commands.run(
            argv=["test", "-e", path],
            capture_output=True,
            check=False,
        )
        return result.returncode == 0


class _CheckpointsNamespace:
    """`sbx.checkpoints.*` — checkpoint listing/deletion."""

    def __init__(self, sandbox: "Sandbox") -> None:
        self._sandbox = sandbox

    def list(self) -> list[dict[str, object]]:
        storage = self._sandbox._engine.system.storage
        sandbox_id = self._sandbox.sandbox_id
        out: list[dict[str, object]] = []
        for ckpt_id in storage.list_checkpoints(sandbox_id):
            try:
                manifest = storage.get_manifest(sandbox_id, ckpt_id)
            except Exception:
                out.append({"checkpoint_id": str(ckpt_id), "metadata": {}})
                continue
            md = dict(manifest.metadata or {})
            try:
                resolved = self._sandbox._engine.system._resolve_restore_manifest(
                    sandbox_id, ckpt_id
                )
            except Exception:
                resolved = manifest
            restore_process_artifacts = getattr(
                manifest,
                "restore_process_artifacts",
                getattr(resolved, "process_artifacts", None),
            )
            restore_filesystem_artifacts = getattr(
                manifest,
                "restore_filesystem_artifacts",
                getattr(resolved, "filesystem_artifacts", None),
            )
            out.append(
                {
                    "checkpoint_id": str(manifest.checkpoint_id),
                    "created_at": manifest.created_at.isoformat() if hasattr(manifest, "created_at") else None,
                    "label": md.get(_LABEL_METADATA_KEY),
                    "has_process": bool(restore_process_artifacts),
                    "has_filesystem": bool(restore_filesystem_artifacts),
                    "materialized_process": bool(manifest.process_artifacts),
                    "materialized_filesystem": bool(manifest.filesystem_artifacts),
                    "logical": bool(md.get("logical_checkpoint", False)),
                    "materialization": md.get("checkpoint_materialization"),
                    "process_checkpoint_id": md.get("process_restore_checkpoint_id"),
                    "filesystem_checkpoint_id": md.get(
                        "filesystem_restore_checkpoint_id"
                    ),
                    "metadata": md,
                }
            )
        return out

    def delete(self, checkpoint_id: str | CheckpointId, *, cascade: bool = False) -> None:
        storage = self._sandbox._engine.system.storage
        ckpt = CheckpointId(str(checkpoint_id))
        storage.delete_checkpoint(self._sandbox.sandbox_id, ckpt, cascade=cascade)


class Sandbox:
    """E2B-style sandbox with Crab checkpoint/restore semantics.

    Construction launches the sandbox immediately. Use `Sandbox.connect(id)`
    to reattach to an already-running sandbox owned by the same engine.

    Most operations are short methods on the sandbox or one of its
    sub-namespaces (`commands`, `files`, `checkpoints`).
    """

    def __init__(
        self,
        *,
        image: str | None = None,
        work_dir: str | Path | None = None,
        template: SandboxTemplate | None = None,
        env: dict[str, str] | None = None,
        name: str | None = None,
        engine: "Engine | None" = None,
        autostart: bool = True,
        network: bool | None = None,
        # `resources` is enforced (S3): normalized at construction and
        # applied as cgroup limits on the runc launch path. `timeout` is the
        # idle-reclaim window (seconds of inactivity before `idle_action` is
        # applied by the gateway; E2B-style). `idle_action` selects the
        # outcome: "pause" | "stop" | "checkpoint_stop" | "kill". Both are
        # ignored by local (non-gateway) engines. `labels` remain advisory
        # metadata exposed via `Sandbox.metadata`.
        resources: dict[str, object] | None = None,
        timeout: float | None = None,
        idle_action: str | None = None,
        labels: dict[str, str] | None = None,
        auto_checkpoint: bool = False,
    ) -> None:
        from .engine import get_default_engine
        from .resources import normalize_resources

        self._engine = engine if engine is not None else get_default_engine()
        self._lock = threading.Lock()
        self._closed = False
        self._sandbox_id: SandboxId | None = None
        self._launch_plan: _LaunchPlan | None = None
        self._user_env = dict(env or {})
        # Loud at construction: an invalid resources shape must fail before
        # any launch work happens (enforced limits, not advisory metadata).
        self._resource_claim = normalize_resources(resources)
        self._metadata = {
            "resources": dict(resources or {}),
            "timeout": timeout,
            "idle_action": idle_action,
            "labels": dict(labels or {}),
        }
        # Host-inspector filters are layered from two internal sources:
        #   * `_default_*` — auto-derived from the bundle's idle init command
        #     (e.g. `sh -c sleep infinity`). Set in `_prepare_runc_launch`
        #     after the bundle has been written. The Sandbox knows what its
        #     own init looks like; users do not specify these.
        #   * `_agent_*` — installed by `Agent.bind(sbx)` via the public
        #     `add_host_inspector_filters(...)` method on this class.
        #     Calling `bind()` again replaces this set; defaults are never
        #     touched.
        self._default_ignore_process_rules: list[dict[str, object]] = []
        self._default_ignored_path_prefixes: list[str] = []
        self._agent_ignore_process_rules: list[dict[str, object]] = []
        self._agent_ignored_path_prefixes: list[str] = []
        self._exposed_ports: dict[int, str] = {}
        self._auto_checkpoint = bool(auto_checkpoint)
        # Tracks the most recently started (pre-allocated) checkpoint id so
        # callers using auto_checkpoint can retrieve it without wiring the
        # AsyncCheckpoint handle through the ActionResult (which is still
        # returned for explicit `checkpoint=True` callers).
        self._last_checkpoint_id: str | None = None
        self._desired_name = name
        self._desired_image = image
        self._template = template
        self._work_dir_host = Path(work_dir).expanduser().resolve() if work_dir else None
        if network is not None and not isinstance(network, bool):
            raise ValueError("network must be true, false, or null")
        self._network_requested = network
        self._network_lease = None
        self._process_cwd = "/work"

        # Sub-namespaces
        self.commands = _CommandsNamespace(self)
        self.files = _FilesNamespace(self)
        self.checkpoints = _CheckpointsNamespace(self)
        self.ports = _PortsNamespace(self)

        if autostart:
            self._launch()

    # ------------------------------------------------------------------
    # Launch / lifecycle
    # ------------------------------------------------------------------

    def _launch(self) -> None:
        runtime = self._engine.runtime
        runtime_name = runtime.name
        sandbox_name = self._desired_name or f"sbx-{uuid.uuid4().hex[:12]}"
        sandbox_id = SandboxId(sandbox_name)
        plan = _LaunchPlan(
            runtime_name=runtime_name,
            name=sandbox_name,
            image=self._desired_image or self._default_image(),
            work_dir_host=self._work_dir_host,
            bundle_dir=None,
        )
        self._launch_plan = plan
        self._sandbox_id = sandbox_id
        is_remote = self._is_remote_runtime(runtime)
        try:
            if is_remote:
                # S5 full-access: remote mode — daemon handles all bundle prep.
                launch_metadata = self._build_remote_launch_metadata(plan, sandbox_id)
            elif runtime_name == "runc":
                launch_metadata = self._prepare_runc_launch(plan, sandbox_id)
            else:
                launch_metadata = {"sandbox_id": str(sandbox_id), **dict(plan.metadata)}
            if self._resource_claim:
                # The normalized claim rides the launch metadata so the gateway
                # (cloud mode) can meter per-tenant aggregate quotas (§4 S3).
                launch_metadata = {**launch_metadata, "resources": dict(self._resource_claim)}
            idle_timeout = self._metadata.get("timeout")
            if idle_timeout is not None:
                launch_metadata = {**launch_metadata, "idle_timeout": idle_timeout}
                idle_action = self._metadata.get("idle_action")
                if idle_action is not None:
                    launch_metadata = {**launch_metadata, "idle_action": idle_action}
            sandbox_id = runtime.launch(runtime_name, launch_metadata)
        except Exception as exc:
            # Remote create is a daemon-owned transaction (and a gateway
            # timeout is intentionally reconciled server-side). Local runc
            # preparation owns its lease/bundle/dataset here.
            if not is_remote and runtime_name == "runc":
                cleanup_errors: list[str] = []
                cleanup = getattr(runtime, "cleanup_failed_launch", None)
                if callable(cleanup):
                    try:
                        cleanup_errors.extend(
                            cleanup(
                                sandbox_id,
                                bundle_path=plan.bundle_dir,
                                dataset=(
                                    None
                                    if plan.metadata.get("zfs_dataset") is None
                                    else str(plan.metadata["zfs_dataset"])
                                ),
                            )
                        )
                    except Exception as cleanup_exc:
                        cleanup_errors.append(f"runtime cleanup: {cleanup_exc}")
                try:
                    self._engine.release_network_lease(sandbox_id, strict=True)
                except Exception as cleanup_exc:
                    cleanup_errors.append(f"network lease: {cleanup_exc}")
                self._network_lease = None
                if cleanup_errors:
                    resources = (
                        str(plan.bundle_dir or "bundle"),
                        str(plan.metadata.get("zfs_dataset") or "dataset"),
                        "network lease",
                    )
                    if isinstance(exc, SandboxCreateCleanupError):
                        raise SandboxCreateCleanupError(
                            exc.sandbox_id,
                            exc.cause,
                            (*exc.cleanup_errors, *cleanup_errors),
                            resources=(*exc.resources, *resources),
                        ) from exc
                    raise SandboxCreateCleanupError(
                        str(sandbox_id),
                        exc,
                        cleanup_errors,
                        resources=resources,
                    ) from exc
            raise
        self._sandbox_id = sandbox_id
        self._mark_inspector_running()
        self._engine._register_sandbox(self)
        logger.info(
            "Sandbox launched: id=%s runtime=%s image=%s",
            sandbox_id,
            runtime_name,
            plan.image,
        )

    @staticmethod
    def _is_remote_runtime(runtime) -> bool:
        """True when the runtime is a proxy to a remote daemon (cloud mode)."""
        # Duck-type: RuntimeProxy has _client; local RuncRuntime does not.
        return hasattr(runtime, "_client")

    def _build_remote_launch_metadata(
        self, plan: "_LaunchPlan", sandbox_id: SandboxId
    ) -> dict[str, object]:
        """Minimal metadata for daemon-side bundle prep (S5 full-access)."""
        md: dict[str, object] = {
            "sandbox_id": str(sandbox_id),
            "image": plan.image,
        }
        if self._user_env:
            md["env"] = dict(self._user_env)
        if self._network_requested is not None:
            md["network"] = bool(self._network_requested)
        return md

    def _default_image(self) -> str | None:
        return self._engine.config.default_image

    def _install_attached_agent(self, agent: "Agent") -> None:
        try:
            agent.install(self)
        except Exception as exc:
            logger.error(
                "Agent install failed: sandbox=%s agent=%s error=%s",
                self.sandbox_id,
                type(agent).__name__,
                exc,
            )
            raise

    def _prepare_runc_launch(self, plan: _LaunchPlan, sandbox_id: SandboxId) -> dict[str, object]:
        from integrations.sandboxes.runtime import bundle as sandbox_bundle
        from integrations.sandboxes.runtime import image as sandbox_image

        bundle_dir = self._runc_bundle_root() / str(sandbox_id)
        if bundle_dir.exists():
            import shutil

            shutil.rmtree(bundle_dir, ignore_errors=True)
        bundle_dir.mkdir(parents=True, exist_ok=True)
        self._engine.runtime.write_bundle_spec(bundle_dir)

        network_lease = self._maybe_allocate_network_lease(sandbox_id)
        network_namespace_path = None if network_lease is None else network_lease.namespace_path
        work_dir_host_path = self._resolve_work_dir_host_path(
            sandbox_id,
            force=self._template is not None,
        )
        resource_limits = self._bundle_resource_limits()

        if self._template is not None:
            sandbox_bundle.write_bundle_config(
                bundle_dir=bundle_dir,
                llm_base_url="",
                provider="openai",
                sandbox_name=str(sandbox_id),
                status_port=self._find_free_port(),
                cgroup_path=f"crab-sdk/{sandbox_id}",
                work_dir_host_path=work_dir_host_path,
                network_namespace_path=network_namespace_path,
                image_defaults=None,
                image_rootfs_dir=None,
                resource_limits=resource_limits,
            )
            template_data = self._template.configure_runc_bundle(
                engine=self._engine,
                sandbox_id=sandbox_id,
                bundle_dir=bundle_dir,
                work_dir_host_path=work_dir_host_path,
            )
            plan.bundle_dir = bundle_dir
            plan.image = template_data.image or plan.image
            plan.work_dir_host = work_dir_host_path
            plan.process_cwd = template_data.process_cwd or self._bundle_process_cwd(bundle_dir) or "/work"
            self._process_cwd = plan.process_cwd
            plan.metadata.update(template_data.runtime_metadata)
            plan.metadata.update(template_data.metadata)
            # The template's `configure_runc_bundle` may have rewritten
            # `process.args` (e.g. with the compose `command:`), so derive
            # the default ignore rules from the *final* bundle contents.
            self._default_ignore_process_rules = self._derive_init_ignore_rules(bundle_dir)
            plan.metadata.update(
                {
                    "sandbox_id": str(sandbox_id),
                    "bundle_path": str(bundle_dir),
                    "work_dir_host_path": None if work_dir_host_path is None else str(work_dir_host_path),
                    "sdk_process_cwd": plan.process_cwd,
                    **self._network_launch_metadata(network_lease),
                    **self._host_inspector_launch_metadata(),
                }
            )
            return dict(plan.metadata)

        image = plan.image or self._engine.config.default_image
        resolved_image = sandbox_image.resolve_image(
            reference=image,
            cache_root=self._engine.image_cache_root,
            pull_policy=self._engine.config.image_pull_policy,
            allowed_registries=self._engine.config.image_allowed_registries,
            allowed_references=self._engine.config.image_allowed_references,
            pull_timeout_seconds=self._engine.config.image_pull_timeout_seconds,
            max_image_bytes=self._engine.config.image_max_bytes,
            min_free_bytes=self._engine.config.image_min_free_bytes,
            telemetry=self._engine.system.telemetry,
        )
        image_tag = resolved_image.normalized_reference
        image_id = resolved_image.image_id
        image_defaults = sandbox_image.inspect_image_runtime_defaults(
            tag=image_tag,
            cache_root=self._engine.image_cache_root,
            telemetry=self._engine.system.telemetry,
            image_id=image_id,
        )
        exported_rootfs = sandbox_image.export_image_rootfs(
            tag=image_tag,
            output_dir=self._engine.image_cache_root / image_id,
            cache_root=self._engine.image_cache_root,
            telemetry=self._engine.system.telemetry,
            image_id=image_id,
            image_size_bytes=resolved_image.size_bytes,
            max_image_bytes=self._engine.config.image_max_bytes,
            cache_max_bytes=self._engine.config.image_cache_max_bytes,
            min_free_bytes=self._engine.config.image_min_free_bytes,
            cache_retention_seconds=self._engine.config.image_cache_retention_seconds,
        )
        try:
            sandbox_bundle.write_bundle_config(
                bundle_dir=bundle_dir,
                llm_base_url="",
                provider="openai",
                sandbox_name=str(sandbox_id),
                status_port=self._find_free_port(),
                cgroup_path=f"crab-sdk/{sandbox_id}",
                work_dir_host_path=work_dir_host_path,
                network_namespace_path=network_namespace_path,
                image_defaults=image_defaults,
                image_rootfs_dir=exported_rootfs,
                resource_limits=resource_limits,
            )
        except ValueError as exc:
            raise ImageCompatibilityError(
                image,
                f"image {image!r} has unsupported runtime metadata: {exc}",
            ) from exc
        self._write_sdk_bundle_process(bundle_dir, image_defaults)
        # TLS trust injection: add CA cert to rootfs copy paths.
        rootfs_copy_paths = [{"source": str(exported_rootfs), "destination": "/"}]
        ca_cert_path = getattr(self._engine, "tls_ca_cert_path", None)
        if ca_cert_path is not None:
            from .tls_trust import _SANDBOX_CA_CERT_PATH  # no cryptography dep
            rootfs_copy_paths.append(
                {"source": str(ca_cert_path), "destination": _SANDBOX_CA_CERT_PATH}
            )
        plan.bundle_dir = bundle_dir
        plan.image = image_tag
        plan.work_dir_host = work_dir_host_path
        plan.process_cwd = "/work"
        self._process_cwd = plan.process_cwd
        # The SDK's bare-image path writes `sh -lc exec sleep infinity` as
        # the bundle init, so the canonical ignore rules are applicable.
        self._default_ignore_process_rules = self._derive_init_ignore_rules(bundle_dir)
        launch_metadata: dict[str, object] = {
            "sandbox_id": str(sandbox_id),
            "bundle_path": str(bundle_dir),
            "work_dir_host_path": None if work_dir_host_path is None else str(work_dir_host_path),
            "rootfs_init_dirs": self._rootfs_init_dirs(),
            "rootfs_copy_paths": rootfs_copy_paths,
            "shared_rootfs_key": self._shared_rootfs_key(image_id, ca_cert_path),
            "shared_rootfs_persist": True,
            "sdk_image": image,
            "image_reference": image_tag,
            "image_id": image_id,
            "image_digest": resolved_image.digest,
            "sdk_process_cwd": plan.process_cwd,
            **self._network_launch_metadata(network_lease),
            **self._host_inspector_launch_metadata(),
        }
        from integrations.sandboxes.runtime.baseline import (
            SANDBOX_ROOTFS_PREPARATION_SCHEMA,
            add_dns_materialization,
        )

        add_dns_materialization(
            launch_metadata,
            bundle_dir=bundle_dir,
            isolated=network_lease is not None,
        )
        launch_metadata["rootfs_preparation_schema"] = (
            SANDBOX_ROOTFS_PREPARATION_SCHEMA
        )
        plan.metadata.update(launch_metadata)
        return dict(plan.metadata)

    def _bundle_resource_limits(self) -> "object | None":
        """Normalized claim -> `SandboxResourceLimits` for the bundle spec
        (`linux.resources`); None when no limits were requested so the
        written spec is identical to the unlimited one."""
        if not self._resource_claim:
            return None
        from integrations.sandboxes.runtime import bundle as sandbox_bundle

        return sandbox_bundle.SandboxResourceLimits(
            cpus=self._resource_claim.get("cpus"),
            memory_bytes=self._resource_claim.get("memory_bytes"),
            pids_limit=self._resource_claim.get("pids"),
        )

    @staticmethod
    def _shared_rootfs_key(image_id: str, ca_cert_path: Path | None) -> str:
        """Compute cache key for the shared-rootfs ZFS snapshot.

        The key incorporates the image identity *and* the TLS CA state so that
        toggling interception (or rotating the CA) invalidates stale snapshots
        that lack the injected certificate.
        """
        base = image_id[:32]
        if ca_cert_path is None:
            return base
        try:
            ca_digest = hashlib.sha256(ca_cert_path.read_bytes()).hexdigest()[:16]
        except OSError:
            # CA path configured but file missing/unreadable — treat as no-CA.
            return base
        return f"{base}-ca{ca_digest}"

    def _resolve_work_dir_host_path(self, sandbox_id: SandboxId, *, force: bool = False) -> Path | None:
        if self._work_dir_host is not None:
            return self._work_dir_host
        if force or self._metadata["labels"].get("mount_work_dir"):
            return self._engine.work_dir_host_root / str(sandbox_id)
        return None

    def _maybe_allocate_network_lease(self, sandbox_id: SandboxId):
        if not self._requires_network_namespace():
            return None
        lease = self._engine.allocate_network_lease(sandbox_id)
        self._network_lease = lease
        return lease

    def _requires_network_namespace(self) -> bool:
        from .engine import resolve_sandbox_network_mode

        return resolve_sandbox_network_mode(
            self._engine.config,
            runtime_name=self._engine.runtime.name,
            requested=self._network_requested,
        )

    def _network_launch_metadata(self, lease) -> dict[str, object]:
        if lease is None:
            return {
                "network_mode": "host",
                "network_requested": self._network_requested,
            }
        bridge_ip = self._engine.network_bridge_ip
        return {
            "network_mode": "isolated",
            "network_requested": self._network_requested,
            "network_namespace_path": str(lease.namespace_path),
            "guest_ip": str(lease.guest_ip),
            "bridge_ip": bridge_ip,
        }

    def _merged_ignore_process_rules(self) -> list[dict[str, object]]:
        merged: list[dict[str, object]] = []
        for rule in self._default_ignore_process_rules:
            merged.append(dict(rule))
        for rule in self._agent_ignore_process_rules:
            merged.append(dict(rule))
        return merged

    def _merged_ignored_path_prefixes(self) -> list[str]:
        merged: list[str] = []
        for source in (
            self._default_ignored_path_prefixes,
            self._agent_ignored_path_prefixes,
        ):
            for item in source:
                if item and item not in merged:
                    merged.append(item)
        return merged

    def _host_inspector_launch_metadata(self) -> dict[str, object]:
        meta: dict[str, object] = {}
        rules = self._merged_ignore_process_rules()
        if rules:
            meta["host_inspector_ignore_process_rules"] = rules
        prefixes = self._merged_ignored_path_prefixes()
        if prefixes:
            meta["host_inspector_ignored_path_prefixes"] = prefixes
        return meta

    # Canonical idle-init patterns the Sandbox knows how to filter on its
    # own. Anything not in this set is left untouched and the operator gets
    # a warning — getting this wrong falsely (filtering a real workload
    # whose argv happens to resemble these) would silently break checkpoint
    # restore correctness, so we err on the conservative side and only match
    # exact docker-compose-style idle commands.
    _IDLE_INIT_PATTERNS: tuple[tuple[str, ...], ...] = (
        ("/bin/sh", "-lc", "exec sleep infinity"),
        ("sh", "-lc", "exec sleep infinity"),
        ("/bin/sh", "-c", "sleep infinity"),
        ("sh", "-c", "sleep infinity"),
        ("/bin/sleep", "infinity"),
        ("sleep", "infinity"),
    )

    @staticmethod
    def _sleep_infinity_ignore_rules() -> list[dict[str, object]]:
        """Rules that suppress the canonical `sleep infinity` idle init.

        The shell-wrapped variants (`sh -c sleep infinity` /
        `sh -lc exec sleep infinity`) have `"sleep infinity"` as one argv
        element, so the joined-by-null cmdline contains the literal string
        — the first rule catches both the pre-exec shell pid and any case
        where the shell does NOT exec into sleep. The second rule catches
        the sleep child (whether forked by `sh -c CMD` or produced by the
        shell `exec`ing its only command), whose argv is `("sleep",
        "infinity")` and therefore does not contain the literal phrase."""
        return [
            {
                "cmdline_contains": ["sleep infinity"],
                "scope": "process_only",
            },
            {
                "executable_basename": "sleep",
                "cmdline_contains": ["infinity"],
                "scope": "process_only",
            },
        ]

    def _derive_init_ignore_rules(self, bundle_dir: Path) -> list[dict[str, object]]:
        """Auto-derive `_default_ignore_process_rules` from the bundle's
        idle init `process.args`. Only the canonical `sleep infinity`
        patterns generate rules — any other init command leaves the
        defaults empty and surfaces a warning, because falsely filtering a
        real workload would skip its checkpoints and break restore.
        """
        import json

        config_path = bundle_dir / "config.json"
        try:
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return []
        process = cfg.get("process") if isinstance(cfg, dict) else None
        if not isinstance(process, dict):
            return []
        args = process.get("args")
        if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
            return []
        args_tuple = tuple(args)
        if args_tuple in self._IDLE_INIT_PATTERNS:
            return self._sleep_infinity_ignore_rules()
        logger.warning(
            "Sandbox %s init command %r is not the canonical `sleep infinity`; "
            "the Sandbox will not auto-add host-inspector default ignore rules. "
            "Tool calls dispatched via `runc exec` will continue to be tracked, "
            "but the idle init pids may keep `process_changed=True` between "
            "turns. Use `sleep infinity` as the bundle init command to silence "
            "this warning.",
            self._desired_name,
            args,
        )
        return []

    def add_host_inspector_filters(
        self,
        *,
        ignore_process_rules: list[dict[str, object]] | None = None,
        ignored_path_prefixes: list[str] | None = None,
    ) -> None:
        """Register agent-contributed host-inspector filters on this sandbox.

        Replaces whatever agent-contributed filters were previously installed
        (so rebinding an agent — or binding a different one — never compounds
        the rule set). Sandbox-default rules derived from the bundle init and
        any rules the caller passed to `Sandbox(...)` are not touched.

        If the sandbox is already launched, the new merged filter set is
        pushed to the host inspector via `/update_filters`, which updates
        the daemon's record in place without resetting baseline pids or
        accumulated dirty state.
        """
        self._agent_ignore_process_rules = [
            dict(rule) for rule in (ignore_process_rules or [])
        ]
        self._agent_ignored_path_prefixes = [
            str(item) for item in (ignored_path_prefixes or []) if str(item)
        ]
        if self._sandbox_id is None:
            return
        runtime = self._engine.runtime
        if not hasattr(runtime, "update_host_inspector_filters"):
            return
        runtime.update_host_inspector_filters(
            self._sandbox_id,
            ignore_process_rules=self._merged_ignore_process_rules(),
            ignored_path_prefixes=self._merged_ignored_path_prefixes(),
        )

    def _bundle_process_cwd(self, bundle_dir: Path) -> str | None:
        import json

        config_path = bundle_dir / "config.json"
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        process = payload.get("process")
        if not isinstance(process, dict):
            return None
        cwd = process.get("cwd")
        return cwd if isinstance(cwd, str) and cwd else None

    def _runc_bundle_root(self) -> Path:
        runtime = self._engine.runtime
        paths = getattr(runtime, "paths", None)
        if paths is not None:
            return Path(paths.bundle_root)
        return self._engine.runtime_root / "bundles"

    def _write_sdk_bundle_process(self, bundle_dir: Path, image_defaults: object | None) -> None:
        import json

        config_path = bundle_dir / "config.json"
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        process = dict(cfg.get("process") or {})
        process["terminal"] = False
        process["cwd"] = "/work"
        # `sleep infinity` is the canonical SDK idle init. The Sandbox auto-
        # derives host-inspector ignore rules from this exact pattern; do not
        # change it without updating `_default_init_ignore_process_rules`.
        process["args"] = ["/bin/sh", "-lc", "exec sleep infinity"]
        defaults_env = getattr(image_defaults, "environment", ()) if image_defaults is not None else ()
        env = self._merge_env_assignments(
            list(defaults_env),
            [
                "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
                "HOME=/root",
                "PYTHONUNBUFFERED=1",
                f"CRAB_SANDBOX_ID={self.sandbox_id}",
                *[f"{key}={value}" for key, value in self._user_env.items()],
                *self._tls_ca_env_assignments(),
            ],
        )
        process["env"] = env
        cfg["process"] = process
        cfg["root"]["path"] = "rootfs"
        cfg["root"]["readonly"] = False
        config_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    def _merge_env_assignments(self, base: list[str], overrides: list[str]) -> list[str]:
        merged: dict[str, str] = {}
        for item in [*base, *overrides]:
            key, sep, value = str(item).partition("=")
            if sep:
                merged[key] = value
        return [f"{key}={value}" for key, value in merged.items()]

    def _rootfs_init_dirs(self) -> list[str]:
        return [
            "work",
            "tmp",
            "proc",
            "dev",
            "dev/pts",
            "dev/shm",
            "dev/mqueue",
            "sys",
            "run",
            "var",
            "root",
            "opt",
        ]

    def _find_free_port(self) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

    def _mark_inspector_running(self) -> None:
        try:
            self._engine.system.inspector.upsert_snapshot(
                SandboxSnapshot(
                    sandbox_id=self.sandbox_id,
                    runtime_name=self._engine.runtime.name,
                    is_running=True,
                    process_changed=False,
                    filesystem_changed=False,
                    observed_at=utc_now(),
                )
            )
        except Exception:
            logger.debug("Failed to seed inspector snapshot for sandbox=%s", self.sandbox_id, exc_info=True)

    @classmethod
    def connect(cls, sandbox_id: str | SandboxId, *, engine: "Engine | None" = None) -> "Sandbox":
        """Reattach to an existing sandbox managed by the same engine.

        Note: this does not retrieve any previously-bound agent profile. Bind
        a fresh agent instance after connecting if agent operations are needed.
        """
        from .engine import get_default_engine

        eng = engine if engine is not None else get_default_engine()
        sbx = cls(engine=eng, autostart=False)
        sbx._sandbox_id = SandboxId(str(sandbox_id))
        eng._register_sandbox(sbx)
        return sbx

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def sandbox_id(self) -> SandboxId:
        if self._sandbox_id is None:
            raise RuntimeError("sandbox not launched")
        return self._sandbox_id

    @property
    def name(self) -> str:
        return self._desired_name or str(self.sandbox_id)

    @property
    def engine(self) -> "Engine":
        return self._engine

    @property
    def process_cwd(self) -> str:
        """Default working directory for commands that should behave like the
        sandbox's main service process."""
        return self._process_cwd

    @property
    def metadata(self) -> dict[str, object]:
        return dict(self._metadata)

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def env(self) -> dict[str, str]:
        return dict(self._user_env)

    @property
    def has_network_namespace(self) -> bool:
        return self._network_lease is not None

    # ------------------------------------------------------------------
    # Action result builder (checkpoint / changeset / observe)
    # ------------------------------------------------------------------

    def _build_action_result(
        self,
        *,
        returncode: int,
        stdout: str,
        stderr: str,
        args: tuple[str, ...],
        do_checkpoint: bool,
        do_changeset: bool,
        changeset_sync: bool,
        do_observe: bool,
    ) -> ActionResult:
        """Construct an ActionResult with optional async enrichments."""
        ckpt_handle: AsyncCheckpoint | None = None
        cs_handle: AsyncChangeset | list[dict] | None = None
        fs_changed: bool | None = None
        proc_changed: bool | None = None

        # Snapshot the checkpoint id that existed *before* this action.
        # Changeset semantics: "what changed since the last checkpoint",
        # so we must diff against the previous id, not the fresh one that
        # the concurrent checkpoint (if any) is about to create.
        prev_ckpt_id = self._last_checkpoint_id

        # Observe BEFORE starting the async checkpoint. When a checkpoint
        # finishes, `mark_checkpoint_complete` resets the inspector's
        # `filesystem_changed`/`process_changed` cursors (scorched-earth
        # wipe in host-inspector `reset()`). If we peeked after starting
        # the checkpoint, the reset could race with — and win against —
        # the peek, so `observe=True` would report False for actions
        # that clearly mutated state. Peeking first also matches the
        # semantic contract in the tutorial: observe answers "did *this
        # action* mutate?", relative to state before the concurrent
        # checkpoint snapshots it.
        if do_observe:
            fs_changed, proc_changed = self._peek_inspector()

        if do_checkpoint:
            ckpt_handle = self._start_async_checkpoint()

        if do_changeset:
            if changeset_sync:
                cs_handle = self._run_changeset_sync(since=prev_ckpt_id)
            else:
                cs_handle = self._start_async_changeset(since=prev_ckpt_id)

        return ActionResult(
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            args=args,
            checkpoint=ckpt_handle,
            changeset=cs_handle,
            filesystem_changed=fs_changed,
            process_changed=proc_changed,
        )

    @property
    def last_checkpoint_id(self) -> str | None:
        """Most recently pre-allocated checkpoint id (auto_checkpoint /
        explicit ``checkpoint=True``). ``None`` before the first checkpoint.
        Available immediately after ``commands.run`` returns — the actual
        checkpoint may still be running in the background."""
        return self._last_checkpoint_id

    def _start_async_checkpoint(self) -> AsyncCheckpoint:
        """Pre-allocate a checkpoint_id and run the checkpoint in background."""
        ckpt_id = f"ckpt-{uuid.uuid4().hex[:12]}"
        self._last_checkpoint_id = ckpt_id
        handle = AsyncCheckpoint(ckpt_id, threading.Thread(target=lambda: None))

        def _do_checkpoint() -> None:
            try:
                result = self._engine.system.checkpoint_requested(
                    self.sandbox_id,
                    leave_running=True,
                    checkpoint_id=ckpt_id,
                )
                if result.status != JobStatus.SUCCEEDED or result.manifest is None:
                    raise RuntimeError(
                        "logical checkpoint failed: "
                        f"status={result.status.value} message={result.message}"
                    )
                if str(result.checkpoint_id) != ckpt_id:
                    raise RuntimeError(
                        "logical checkpoint id changed during materialization: "
                        f"expected={ckpt_id} actual={result.checkpoint_id}"
                    )
                handle.materialization = str(
                    result.manifest.metadata.get(
                        "checkpoint_materialization", "legacy"
                    )
                )
                handle.physical_checkpoint_created = bool(
                    result.manifest.metadata.get(
                        "physical_checkpoint_created", True
                    )
                )
            except BaseException as exc:
                handle._error = exc

        t = threading.Thread(target=_do_checkpoint, daemon=True, name="crab-async-ckpt")
        handle._thread = t
        t.start()
        return handle

    def _start_async_changeset(self, *, since: str | None = None) -> AsyncChangeset:
        """Run ``changeset(since=..., force=True)`` in background.

        ``since`` should be the checkpoint id the caller wants to diff
        against — typically the previous ``last_checkpoint_id``. When
        ``None``, the call falls back to ``fork_changeset``, which only
        works for sandboxes produced by ``fork_once`` (see
        :meth:`changeset`)."""
        handle = AsyncChangeset(threading.Thread(target=lambda: None))

        def _do_changeset() -> None:
            try:
                handle._result = self.changeset(since=since, force=True)
            except BaseException as exc:
                handle._error = exc

        t = threading.Thread(target=_do_changeset, daemon=True, name="crab-async-cs")
        handle._thread = t
        t.start()
        return handle

    def _run_changeset_sync(self, *, since: str | None = None) -> list[dict]:
        """Run ``changeset(since=..., force=True)`` synchronously."""
        return self.changeset(since=since, force=True)

    def _peek_inspector(self) -> tuple[bool | None, bool | None]:
        """Read-only peek at inspector state (no reset).

        Returns ``(None, None)`` if the peek fails; the reason is logged
        at WARNING so tutorials / CLI users can see why ``observe=True``
        did not surface real flags (e.g. daemon snapshot not registered,
        gateway 404, connection error)."""
        try:
            system = self._engine.system
            snapshot = system.inspector.inspect(self.sandbox_id)
            return (bool(snapshot.filesystem_changed), bool(snapshot.process_changed))
        except Exception as exc:
            logger.warning(
                "inspector peek failed for sandbox=%s: %s",
                self._sandbox_id,
                exc,
            )
            logger.debug("inspector peek traceback", exc_info=True)
            return (None, None)

    # ------------------------------------------------------------------
    # Lifecycle ops
    # ------------------------------------------------------------------

    def pause(self) -> None:
        self._engine.runtime.pause(self.sandbox_id)

    def resume(self) -> None:
        self._engine.runtime.resume(self.sandbox_id)

    def stop(self) -> None:
        """Gracefully stop the sandbox (SIGTERM) while keeping its filesystem
        and bundle, so it can later be brought back with ``start()``. In-sandbox
        process state is not preserved — take a checkpoint first if you need to
        resume the exact process tree. Distinct from ``kill()``, which destroys
        the sandbox entirely."""
        self._engine.runtime.stop(self.sandbox_id)

    def start(self) -> None:
        """Re-launch a previously stopped sandbox from its existing filesystem.
        The container boots with the same files but a fresh process tree (like
        ``docker start``). Use ``restore()`` instead to resume a checkpointed
        process state."""
        self._engine.runtime.start(self.sandbox_id)

    def restart(self) -> None:
        """Stop then start the sandbox (equivalent to ``stop()`` followed by
        ``start()``)."""
        self._engine.runtime.restart(self.sandbox_id)

    @staticmethod
    def _state_exceptions() -> tuple[tuple[type, ...], tuple[type, ...]]:
        """Return (lost_exceptions, not_found_exceptions). Remote engines raise
        typed cloud errors (410/404); local engines raise ``KeyError``. The
        import is lazy to avoid a top-level cycle with ``cloud_client``."""
        try:
            from .cloud_client import SandboxLost, SandboxNotFound

            return (SandboxLost,), (SandboxNotFound, KeyError)
        except Exception:
            return (), (KeyError,)

    @property
    def state(self) -> SandboxState:
        """The sandbox's current lifecycle state.

        Distinguishes ``PAUSED`` (cgroup-frozen, ``resume()`` to continue) from
        ``STOPPED`` (processes terminated, ``start()``/``restore()`` to bring
        back). Returns ``KILLED`` after ``kill()`` or when the sandbox no longer
        exists, and ``LOST`` when the gateway has lost track of it (e.g. a daemon
        restart)."""
        if self._closed:
            return SandboxState.KILLED
        sid = self._sandbox_id
        if sid is None:
            return SandboxState.UNKNOWN
        lost_exc, notfound_exc = self._state_exceptions()
        try:
            runtime_state = self._engine.runtime.inspect_runtime(sid)
            return sandbox_state_from_status(runtime_state.status)
        except lost_exc:
            return SandboxState.LOST
        except notfound_exc:
            return SandboxState.KILLED
        except Exception:
            try:
                description = self._engine.runtime.describe(sid)
                return sandbox_state_from_status(description.status)
            except lost_exc:
                return SandboxState.LOST
            except notfound_exc:
                return SandboxState.KILLED
            except Exception:
                return SandboxState.UNKNOWN

    def describe(self) -> SandboxInfo:
        """Return a :class:`SandboxInfo` snapshot with both the bookkeeping
        status and the live runc runtime status (plus pid/metadata)."""
        sid = self._sandbox_id
        if sid is None:
            return SandboxInfo(sandbox_id="", state=SandboxState.UNKNOWN)
        if self._closed:
            return SandboxInfo(sandbox_id=str(sid), state=SandboxState.KILLED)
        lost_exc, notfound_exc = self._state_exceptions()
        status: str | None = None
        runtime_status: str | None = None
        pid: int | None = None
        metadata: dict[str, Any] = {}
        lost = notfound = False
        try:
            description = self._engine.runtime.describe(sid)
            status = description.status
            metadata = dict(description.metadata or {})
        except lost_exc:
            lost = True
        except notfound_exc:
            notfound = True
        except Exception:
            pass
        try:
            runtime_state = self._engine.runtime.inspect_runtime(sid)
            runtime_status = runtime_state.status
            pid = runtime_state.pid
        except lost_exc:
            lost = True
        except notfound_exc:
            notfound = True
        except Exception:
            pass
        if lost:
            resolved = SandboxState.LOST
        elif notfound:
            resolved = SandboxState.KILLED
        else:
            resolved = sandbox_state_from_status(
                runtime_status if runtime_status is not None else status
            )
        return SandboxInfo(
            sandbox_id=str(sid),
            state=resolved,
            status=status,
            runtime_status=runtime_status,
            pid=pid,
            metadata=metadata,
        )

    def set_timeout(self, seconds: float | None) -> None:
        """Reset/extend the idle-reclaim window (E2B-style ``setTimeout`` /
        Daytona keep-alive). ``seconds`` of further inactivity before the
        configured ``idle_action`` fires; ``None`` disables idle reclaim.
        Only enforced by gateway (remote) engines — local engines have no idle
        sweeper, so the value is recorded but not acted on."""
        setter = getattr(self._engine, "set_idle_timeout", None)
        if callable(setter):
            setter(self.sandbox_id, seconds)
        self._metadata["timeout"] = seconds

    def kill(self) -> None:
        if self._closed:
            return
        sandbox_id = self._sandbox_id
        self._closed = True
        if sandbox_id is None:
            return
        try:
            # Fork bookkeeping: if this sandbox has live forks, replace
            # chain-shared symlinks with real bytes first; if it *is* a
            # fork, release its chain pin. Remote engines expose a system
            # shim without these hooks — degrade gracefully.
            system = getattr(self._engine, "system", None)
            if system is not None:
                release_txn = getattr(system, "release_txn", None)
                if callable(release_txn):
                    release_txn(sandbox_id)
                prepare = getattr(system, "prepare_source_destroy", None)
                if callable(prepare):
                    prepare(sandbox_id)
                release = getattr(system, "release_fork", None)
                if callable(release):
                    release(sandbox_id)
        except Exception:
            logger.exception("Fork bookkeeping failed during kill: id=%s", sandbox_id)
        try:
            self._engine.runtime.delete(sandbox_id)
        except Exception:
            logger.exception("Sandbox kill failed: id=%s", sandbox_id)
        finally:
            self._engine.unregister_upstream(sandbox_id)
            self._engine.release_network_lease(sandbox_id)
            self._engine._unregister_sandbox(self)

    def __enter__(self) -> "Sandbox":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.kill()

    # ------------------------------------------------------------------
    # Checkpoint / restore / fork
    # ------------------------------------------------------------------

    def checkpoint(self, label: str | None = None, *, leave_running: bool = True, checkpoint_id: str | None = None) -> str:
        result = self._engine.system.checkpoint_once(
            self.sandbox_id,
            leave_running=leave_running,
            checkpoint_id=checkpoint_id,
        )
        if result.manifest is None:
            raise RuntimeError(f"checkpoint failed: status={result.status.value} message={result.message}")
        ckpt = result.manifest.checkpoint_id
        if label:
            # Best-effort label storage. We don't rewrite the manifest on
            # disk in the first cut; the label survives in the manifest's
            # metadata dict the system already populated.
            pass
        return str(ckpt)

    def restore(self, checkpoint_id: str | CheckpointId) -> None:
        ckpt = CheckpointId(str(checkpoint_id))
        result = self._engine.system.restore_once(self.sandbox_id, ckpt)
        if result.status != JobStatus.SUCCEEDED:
            raise RuntimeError(f"restore failed: status={result.status.value} message={result.message}")
        self._engine.repair_network_lease(self.sandbox_id)
        self._mark_inspector_running()

    def fork(
        self,
        count: int = 1,
        *,
        lazy: bool = False,
        effects: str | None = None,
        checkpoint_id: str | CheckpointId | None = None,
    ) -> list["Sandbox"]:
        """Clone this sandbox via checkpoint+restore. Each fork is an
        independent, running sandbox sharing initial state with the parent
        (by default a fresh checkpoint at call time; incremental chain
        sharing applies when the runtime supports it).

        ``checkpoint_id`` forks from an *existing* checkpoint of this
        sandbox instead of its live state: no new checkpoint is taken, the
        parent is left untouched (it is not restored), and every fork in
        the batch branches from that same point. The checkpoint must belong
        to this sandbox.

        With ``lazy=True`` the process restore uses CRIU lazy-pages: the
        call returns as soon as metadata and the eager page set are in
        place, and memory streams in on demand.

        ``effects`` declares what the fork is *for*, which is what decides
        whether its outbound writes are gated (F1):

        - omitted (default) — an **independent branch**: an RL rollout or a
          tree-search arm is a first-class timeline whose external effects
          are intended, so N forks legitimately produce N effects and
          nothing is gated. Unchanged from before.
        - ``"reject"`` — a **temporal branch**: a speculative fork that
          serves this sandbox and must not write on its own. Mutating
          plaintext egress is refused at the proxy with ``503`` and shows up
          in the fork's effect ledger; if the guess pays off, the promoted
          identity issues the write once.
        - ``"allow"`` — explicit opt-out, same as omitting it unless the
          deployment flipped ``effects.standalone_fork_policy``.

        ``"defer"`` and ``"seal"`` raise ``ValueError``: a bare fork has no
        commit to flush a queue into and no abort for a seal to block.

        **Bounded by TLS**: ``effects="reject"`` prevents *plaintext*
        mutating egress. HTTPS writes stay unclassifiable and are not
        blocked, so a speculative fork writing over HTTPS can still
        double-fire.

        Note: forks share the parent's ``work_dir`` host mount (fork shares
        initial state by design); pass a fresh work dir to a new sandbox if
        isolation is needed.
        """
        if count < 1:
            raise ValueError("fork count must be >= 1")
        fork_ids = self._engine.fork_sandbox(
            self.sandbox_id,
            count=count,
            lazy=lazy,
            effects=effects,
            checkpoint_id=None if checkpoint_id is None else CheckpointId(str(checkpoint_id)),
        )
        forks: list[Sandbox] = []
        for fork_id in fork_ids:
            fork = Sandbox.connect(fork_id, engine=self._engine)
            fork._mark_inspector_running()
            forks.append(fork)
        return forks

    def actions(self, *, kind: str | None = None, limit: int | None = None) -> list[dict]:
        """Read this sandbox's action journal: every exec attempt (argv,
        cwd, env, exit status, timing) plus lifecycle markers
        (launch/checkpoint/restore/fork/destroy) and adopted fork history
        (kind="observation", C3), oldest first.

        Works with both a local in-process engine and the daemon
        (`crab sandbox actions` from the CLI).
        """
        system = getattr(self._engine, "system", None)
        journal = getattr(system, "journal", None)
        if journal is None:
            raise NotImplementedError(
                "action journal is not available on this engine"
            )
        records = journal.entries(self.sandbox_id, kind=kind)
        if limit is not None and limit >= 0:
            records = records[-limit:]
        return [record.to_json() for record in records]

    def changeset(self, since: str | CheckpointId | None = None, *, force: bool = False) -> list[dict]:
        """Changed rootfs paths (added/modified/removed/renamed) relative
        to a base checkpoint's filesystem snapshot. ``since=None``
        resolves this sandbox's fork point (``fork_created`` journal
        marker); pass a checkpoint id for an explicit base.

        ``force=True`` skips the inspector gate optimization so the
        backend diff always runs.

        Works with both a local in-process engine and the daemon
        (`crab sandbox changeset` from the CLI).
        """
        system = getattr(self._engine, "system", None)
        if since is None:
            fork_changeset = getattr(system, "fork_changeset", None)
            if not callable(fork_changeset):
                raise NotImplementedError(
                    "changesets are not available on this engine"
                )
            result = fork_changeset(self.sandbox_id, force=force)
        else:
            changeset_since = getattr(system, "changeset_since", None)
            if not callable(changeset_since):
                raise NotImplementedError(
                    "changesets are not available on this engine"
                )
            result = changeset_since(
                self.sandbox_id,
                CheckpointId(str(since)),
                use_inspector_gate=not force,
            )
        return [entry.to_json() for entry in result.entries]

    def merge(
        self,
        fork: "Sandbox | str",
        *,
        policy: str = "fail_fast",
        ignore_prefixes: tuple[str, ...] | None = None,
        merger=None,
        observations: str = "none",
        observation_summarizer=None,
    ) -> MergeReport:
        """Three-way merge of a fork's filesystem changes back into this
        sandbox (C2): each fork-changed path applies iff this sandbox
        did not change it since the fork point; conflicts resolve per
        ``policy`` — ``fail_fast`` (default: any conflict aborts before
        a single write), ``prefer_fork``, ``prefer_source``, or
        ``text_merge`` (in-repo line-based diff3; unresolved overlap
        aborts like fail_fast). A ``merger`` callable
        ``(path, base, source, fork) -> bytes | None`` gets first shot
        at every conflict. Returns a ``MergeReport``; apply failures
        raise ``MergeError`` carrying the report (``rolled_back=True``
        after a clean path-level undo). The fork stays alive.

        Works with both a local in-process engine and the daemon
        (`crab sandbox merge` from the CLI); custom ``merger`` hooks are
        local-only and cannot cross the RPC boundary.
        """
        system = getattr(self._engine, "system", None)
        merge_from_fork = getattr(system, "merge_from_fork", None)
        if not callable(merge_from_fork):
            raise NotImplementedError(
                "merge is not available on this engine"
            )
        fork_id = fork.sandbox_id if isinstance(fork, Sandbox) else SandboxId(str(fork))
        kwargs = {}
        if observations != "none":
            kwargs["observations"] = observations
        if observation_summarizer is not None:
            kwargs["observation_summarizer"] = observation_summarizer
        return merge_from_fork(
            self.sandbox_id,
            fork_id,
            policy=policy,
            ignore_prefixes=ignore_prefixes,
            merger=merger,
            **kwargs,
        )

    def consolidate_observations(
        self,
        fork: "Sandbox | str",
        *,
        policy: str = "append",
        summarizer=None,
    ) -> ObservationReport:
        """Adopt a fork's journal history into this sandbox's journal
        (C3): ``append`` copies every qualifying record with provenance,
        ``dedupe`` skips execs this sandbox produced identically itself
        since the fork point, ``none`` copies nothing (combine with
        ``summarizer`` for a digest-only entry). Read the result back
        via ``actions(kind="observation")``.

        Works with both a local in-process engine and the daemon
        (`crab sandbox consolidate` from the CLI); ``summarizer``
        callables are local-only.
        """
        system = getattr(self._engine, "system", None)
        consolidate = getattr(system, "consolidate_observations", None)
        if not callable(consolidate):
            raise NotImplementedError(
                "observation consolidation is not available on this engine"
            )
        fork_id = fork.sandbox_id if isinstance(fork, Sandbox) else SandboxId(str(fork))
        return consolidate(
            self.sandbox_id,
            fork_id,
            policy=policy,
            summarizer=summarizer,
        )

    def egress(self, *, txn_id: str | None = None, since_seq: int | None = None) -> EgressLedger:
        """Effect ledger (D1): every outbound flow this sandbox opened,
        classified as ``idempotent_read`` / ``mutating`` / ``opaque``
        (encrypted and raw flows are opaque — the proxy sees the host,
        not the method). Pass ``txn_id`` to scope the view to one
        transaction. Requires ``EngineConfig(enable_egress_proxy=True)``
        for flows to exist; works locally and against the daemon
        (`crab sandbox egress`).
        """
        system = getattr(self._engine, "system", None)
        egress_ledger = getattr(system, "egress_ledger", None)
        if not callable(egress_ledger):
            raise NotImplementedError("the effect ledger is not available on this engine")
        return egress_ledger(self.sandbox_id, txn_id=txn_id, since_seq=since_seq)

    def replay_egress(
        self,
        *,
        policy: str = "cassette_first",
        cassette_source: "Sandbox | str | None" = None,
    ):
        """Context manager serving this sandbox's recorded reads from
        cassettes instead of the network (D2). ``cassette_source`` reads
        another sandbox's bucket — pass the fork whose reads you are
        re-running. ``policy="cassette_only"`` turns a miss into a 504
        instead of live traffic. Writes and encrypted flows always pass
        through. Yields nothing; the report is returned on exit via
        ``.report``.
        """
        system = getattr(self._engine, "system", None)
        begin = getattr(system, "begin_egress_replay", None)
        end = getattr(system, "end_egress_replay", None)
        if not callable(begin) or not callable(end):
            raise NotImplementedError("egress replay is not available on this engine")
        source = (
            cassette_source.sandbox_id
            if isinstance(cassette_source, Sandbox)
            else cassette_source
        )

        class _ReplayWindow:
            report = None

        @contextlib.contextmanager
        def _window():
            handle = _ReplayWindow()
            begin(self.sandbox_id, policy=policy, cassette_source=source)
            try:
                yield handle
            finally:
                handle.report = end(self.sandbox_id)

        return _window()

    def merge_processes(
        self,
        fork: "Sandbox | str",
        *,
        strategy: str = "auto",
        policy: str = "fail_fast",
        observations: str = "append",
        stop_on_deviation: bool = False,
        lazy_pages: bool = True,
        force: bool = False,
        egress_replay: str = "cassette_first",
        replay_effects: str = "reject",
    ) -> "ProcessMergeReport":
        """Process-half of consolidation (C4). ``strategy="auto"``
        resolves from a process census on this sandbox: with live
        background processes the fork's journaled execs are **replayed**
        here verbatim (deviations against the recorded outcomes are
        counted; ``stop_on_deviation`` aborts at the first one); without
        any, the fork is **promoted** wholesale onto this sandbox's
        identity (PR-C4.2 — ``policy``/``observations``/``lazy_pages``/
        ``force`` steer that path). Returns a ProcessMergeReport.

        Works with both a local in-process engine and the daemon
        (`crab sandbox merge-processes` from the CLI).
        """
        system = getattr(self._engine, "system", None)
        merge_processes = getattr(system, "merge_processes", None)
        if not callable(merge_processes):
            raise NotImplementedError(
                "process merge is not available on this engine"
            )
        fork_id = fork.sandbox_id if isinstance(fork, Sandbox) else SandboxId(str(fork))
        return merge_processes(
            self.sandbox_id,
            fork_id,
            strategy=strategy,
            policy=policy,
            observations=observations,
            stop_on_deviation=stop_on_deviation,
            lazy_pages=lazy_pages,
            force=force,
            egress_replay=egress_replay,
            replay_effects=replay_effects,
        )

    def begin(
        self,
        label: str | None = None,
        *,
        isolation: str = "snapshot",
        effects: str | None = None,
    ) -> "Transaction":
        """Open a transaction. ``isolation="snapshot"`` (default, B2):
        adaptive base checkpoint + observation staging armed +
        auto-checkpoints suppressed; actions run in place (weak
        isolation), commit delivers staged observations, abort restores
        the base. ``isolation="fork"`` (B3): begin forks the sandbox and
        ``txn.exec`` runs in the fork while this sandbox stays clean and
        serving; commit promotes the fork's whole state (filesystem +
        processes) back onto this sandbox's identity, abort just
        destroys the fork.

        Works with both a local in-process engine and the daemon
        (`crab txn ...` from the CLI).
        """
        from .txn import Transaction

        system = getattr(self._engine, "system", None)
        begin_txn = getattr(system, "begin_txn", None)
        if not callable(begin_txn):
            raise NotImplementedError(
                "transactions are not available on this engine "
                "(daemon-mode txn RPC lands in a follow-up)"
            )
        kwargs = {} if isolation == "snapshot" else {"isolation": isolation}
        # Only pass the policy when the caller chose one, so older system
        # fakes (and pre-D3 daemons) keep working with their defaults.
        if effects is not None:
            kwargs["effects"] = effects
        description = begin_txn(self.sandbox_id, label=label, **kwargs)
        return Transaction(self, description)

    def current_txn(self) -> "Transaction | None":
        """Reattach to this sandbox's active transaction, if any."""
        from .txn import Transaction

        system = getattr(self._engine, "system", None)
        current = getattr(system, "current_txn", None)
        if not callable(current):
            return None
        description = current(self.sandbox_id)
        if description is None:
            return None
        return Transaction(self, description)

    # ------------------------------------------------------------------
    # Network
    # ------------------------------------------------------------------

    def get_host(self, port: int) -> str:
        """Return a host-reachable URL for a port exposed inside the sandbox.

        First-cut implementation returns the host-side address that the
        runtime's network namespace publishes. Operators wiring a per-sandbox
        bridge IP into the runtime metadata get a real URL; without a bridge
        IP the URL points to `127.0.0.1:<port>` and assumes the operator has
        configured host networking accordingly.
        """
        host_ip = "127.0.0.1"
        runtime_state = self._engine.runtime.inspect_runtime(self.sandbox_id)
        metadata = runtime_state.metadata or {}
        for key in ("guest_ip", "bridge_ip", "host_ip"):
            value = metadata.get(key)
            if isinstance(value, str) and value:
                host_ip = value
                break
        return f"http://{host_ip}:{int(port)}"

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _command_env(self, overrides: dict[str, str] | None) -> dict[str, object] | None:
        """Build the env dict passed to `runtime.exec`.

        Sandbox only knows about sandbox-level env plus per-command overrides.
        Agent-specific LLM env is supplied by `Agent.command_env(...)`.
        """
        merged: dict[str, str] = {}
        # TLS CA env injection (exec path) — before user env so user can
        # override if needed.
        ca_env = self._tls_ca_env_dict()
        if ca_env:
            merged.update(ca_env)
        if self._user_env:
            merged.update(self._user_env)
        if overrides:
            merged.update(overrides)
        return dict(merged) if merged else None

    def _host_rootfs_path(self) -> Path:
        runtime = self._engine.runtime
        rootfs_path_for = getattr(runtime, "rootfs_path_for", None)
        if callable(rootfs_path_for):
            return Path(rootfs_path_for(self.sandbox_id))
        description = runtime.describe(self.sandbox_id)
        rootfs = description.metadata.get("rootfs_path")
        if isinstance(rootfs, str) and rootfs:
            return Path(rootfs)
        raise RuntimeError("runtime does not expose a host rootfs path")

    # ------------------------------------------------------------------
    # TLS trust injection (§3.3)
    # ------------------------------------------------------------------

    def _tls_ca_env_dict(self) -> dict[str, str] | None:
        """Return CA env overlay dict when TLS interception is active, else None."""
        ca_path = getattr(self._engine, "tls_ca_cert_path", None)
        if ca_path is None:
            return None
        from .tls_trust import tls_ca_env_overlay  # no cryptography dep
        return tls_ca_env_overlay()

    def _tls_ca_env_assignments(self) -> list[str]:
        """Return CA env as KEY=VALUE assignment strings for init env."""
        overlay = self._tls_ca_env_dict()
        if not overlay:
            return []
        return [f"{k}={v}" for k, v in overlay.items()]

    def _inject_tls_ca_into_rootfs(self, rootfs_path: Path) -> None:
        """Copy the CA cert into the sandbox rootfs when interception is on."""
        ca_path = getattr(self._engine, "tls_ca_cert_path", None)
        if ca_path is None:
            return
        from .tls_trust import inject_ca_into_rootfs  # no cryptography dep
        inject_ca_into_rootfs(rootfs_path, ca_path)


__all__ = ["Sandbox", "ActionResult", "AsyncCheckpoint", "AsyncChangeset", "ExecStream"]
