"""Transaction API (roadmap B2 + B3).

A transaction wraps a span of sandbox work between a base checkpoint
and an explicit resolution. Two isolation modes:

- ``snapshot`` (B2, default): actions run *in place* on the sandbox;
  commit delivers staged observations and drops a freshly-taken base;
  abort drops observations and restores the base (concurrent readers
  see the dirt while the txn is open).
- ``fork`` (B3): begin forks the sandbox and actions run in the fork;
  the source stays clean and serving. Commit promotes the fork's whole
  state (filesystem + processes) back onto the source's identity;
  abort just destroys the fork — the source is never restored.

Either way the airtight part is observation staging: nothing gated
escapes an uncommitted transaction.

All transaction state lives in ``CrabSystem`` (one active txn per
sandbox); the :class:`Transaction` object here is a thin SDK handle so
daemon-mode clients can reattach via ``Sandbox.current_txn()``.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import SandboxExecResult
    from .sandbox import Sandbox


class TxnError(RuntimeError):
    """Base class for transaction failures."""


class TxnActiveError(TxnError):
    """A transaction is already active for the sandbox (one per sandbox)."""


class TxnMismatchError(TxnError):
    """The given txn_id does not match the sandbox's active transaction."""


class TxnResolvedError(TxnError):
    """The SDK handle was already committed or aborted."""


class TxnAbortError(TxnError):
    """Abort could not restore the base checkpoint; the transaction stays
    open (observations are already dropped — dropping is idempotent, so
    retrying abort is safe)."""

    def __init__(self, message: str, *, restore_result: object | None = None) -> None:
        super().__init__(message)
        self.restore_result = restore_result


class TxnCommitConflict(TxnError):
    """Fork-backed commit refused: the source changed since the fork
    point, so promoting the fork would silently discard those writes.
    Commit with ``force=True`` to discard them, or abort the txn."""


class TxnNotAbortable(TxnError):
    """Abort refused: under ``effects="seal"`` this txn already sent a
    mutating request, so rolling the filesystem back would leave the world
    ahead of the sandbox. Commit it, or abort with ``force=True`` to
    accept that the external write stands."""


def new_txn_id() -> str:
    return f"txn-{uuid.uuid4().hex[:12]}"


@dataclass(frozen=True)
class TxnDescription:
    """System-side record of an active transaction."""

    txn_id: str
    sandbox_id: str
    base_checkpoint_id: str | None
    base_was_fresh: bool
    started_at: str
    label: str | None = None
    isolation: str = "snapshot"
    fork_sandbox_id: str | None = None
    effects: str = "allow"
    """Egress effect policy for this txn (D3): allow / defer / reject /
    seal. See crab.effects for what each one does to a mutating flow."""


@dataclass(frozen=True)
class FlushedEffect:
    """One deferred write's flush outcome (D3)."""

    method: str
    host: str
    path: str
    status: int | None = None
    error: str | None = None

    def to_json(self) -> dict[str, object]:
        return {
            "method": self.method,
            "host": self.host,
            "path": self.path,
            "status": self.status,
            "error": self.error,
        }

    @classmethod
    def from_json(cls, payload: dict) -> "FlushedEffect":
        status = payload.get("status")
        error = payload.get("error")
        return cls(
            method=str(payload.get("method", "")),
            host=str(payload.get("host", "")),
            path=str(payload.get("path", "")),
            status=None if status is None else int(status),
            error=None if error is None else str(error),
        )


@dataclass(frozen=True)
class EffectFlushReport:
    """What the commit did with the deferred queue (D3). A failure here
    never unwinds the commit — the filesystem is already committed, so the
    honest move is to report it."""

    attempted: int = 0
    succeeded: int = 0
    failed: int = 0
    entries: tuple[FlushedEffect, ...] = ()

    def to_json(self) -> dict[str, object]:
        return {
            "attempted": self.attempted,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "entries": [entry.to_json() for entry in self.entries],
        }

    @classmethod
    def from_json(cls, payload: dict) -> "EffectFlushReport":
        return cls(
            attempted=int(payload.get("attempted", 0)),
            succeeded=int(payload.get("succeeded", 0)),
            failed=int(payload.get("failed", 0)),
            entries=tuple(
                FlushedEffect.from_json(entry) for entry in (payload.get("entries") or [])
            ),
        )


@dataclass(frozen=True)
class TxnCommitResult:
    txn_id: str
    released_observations: int
    base_dropped: bool
    promoted_checkpoint_id: str | None = None
    observations_consolidated: int | None = None
    effects: "EffectFlushReport | None" = None


@dataclass(frozen=True)
class TxnAbortResult:
    txn_id: str
    discarded_observations: int
    restored_checkpoint_id: str | None
    mutating_egress: int = 0
    """Outbound mutating flows already fired during this txn (D1). The
    filesystem rollback cannot undo them; the count surfaces to scripts
    and callers so they know the abort was not total. Blocking/deferring
    is D3's charter."""
    deferred_dropped: int = 0
    """Deferred writes discarded by this abort (D3). These never reached
    the world at all — that is the point of ``effects="defer"``."""


class Transaction:
    """SDK handle for an active transaction.

    Context-manager semantics: commit on clean exit, abort when the block
    raises (the exception still propagates). Resolving twice raises
    :class:`TxnResolvedError`; the system-side registry stays the source
    of truth for what is actually active.
    """

    def __init__(self, sandbox: "Sandbox", description: TxnDescription) -> None:
        self._sandbox = sandbox
        self._description = description
        self._resolved: str | None = None
        self._fork_sandbox: "Sandbox | None" = None

    @property
    def txn_id(self) -> str:
        return self._description.txn_id

    @property
    def base_checkpoint_id(self) -> str | None:
        return self._description.base_checkpoint_id

    @property
    def label(self) -> str | None:
        return self._description.label

    @property
    def isolation(self) -> str:
        return self._description.isolation

    @property
    def fork_sandbox_id(self) -> str | None:
        """The fork actions run in (fork-backed txns only)."""
        return self._description.fork_sandbox_id

    @property
    def effects(self) -> str:
        """Egress effect policy in force for this txn (D3): allow / defer /
        reject / seal. Worth knowing before issuing a write — under
        ``reject`` it will fail, under ``defer`` it returns ``202`` and
        fires at commit."""
        return getattr(self._description, "effects", "allow")

    @property
    def resolved(self) -> str | None:
        """None while open; "committed" or "aborted" afterwards."""
        return self._resolved

    def _exec_target(self) -> "Sandbox":
        fork_id = self._description.fork_sandbox_id
        if fork_id is None:
            return self._sandbox
        if self._fork_sandbox is None:
            from .sandbox import Sandbox

            self._fork_sandbox = Sandbox.connect(fork_id, engine=self._sandbox._engine)
        return self._fork_sandbox

    def exec(self, cmd=None, *, argv=None, **kwargs) -> "SandboxExecResult":
        """Sugar over ``sandbox.commands.run`` so txn code reads
        ``txn.exec(...)``. Snapshot txns run in place; fork-backed txns
        route to the fork. Every exec during the txn is journal-tagged
        with the txn_id regardless of which API issued it."""
        self._ensure_open()
        return self._exec_target().commands.run(cmd, argv=argv, **kwargs)

    def commit(self, *, force: bool = False) -> TxnCommitResult:
        """``force`` only affects fork-backed txns: promote even when the
        source changed since the fork point (its writes are discarded)."""
        self._ensure_open()
        kwargs = {"force": True} if force else {}
        result = self._sandbox._engine.system.commit_txn(
            self._sandbox.sandbox_id, self.txn_id, **kwargs
        )
        self._resolved = "committed"
        return result

    def abort(self, *, force: bool = False) -> TxnAbortResult:
        """Roll back. Under ``effects="seal"`` a txn that already sent a
        mutating request raises :class:`TxnNotAbortable`; ``force=True``
        aborts anyway, accepting that the external write stands."""
        self._ensure_open()
        # Only pass the kwarg when set, so older system fakes keep working.
        kwargs = {"force": True} if force else {}
        result = self._sandbox._engine.system.abort_txn(
            self._sandbox.sandbox_id, self.txn_id, **kwargs
        )
        self._resolved = "aborted"
        return result

    def _ensure_open(self) -> None:
        if self._resolved is not None:
            raise TxnResolvedError(
                f"transaction {self.txn_id} already {self._resolved}"
            )

    def __enter__(self) -> "Transaction":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._resolved is not None:
            return None
        if exc_type is None:
            self.commit()
        else:
            self.abort()
        return None


__all__ = [
    "Transaction",
    "TxnAbortError",
    "TxnAbortResult",
    "TxnActiveError",
    "TxnCommitConflict",
    "TxnCommitResult",
    "TxnDescription",
    "TxnError",
    "TxnMismatchError",
    "TxnResolvedError",
    "new_txn_id",
]
