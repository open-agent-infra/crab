"""Per-sandbox append-only action journal (roadmap B1).

The journal is the durable record of *what the agent did* to a sandbox:
every exec attempt (argv/cwd/env verbatim, exit status, timing) plus
lifecycle markers (launch/checkpoint/restore/fork/destroy) that later
tracks use to delimit replay windows (C4 replays the exec records after a
fork's `fork_created` marker).

Design notes (see `.cache/tasks/journal-staging.md`):
- One JSONL file per sandbox at `{storage_root}/{journal_dirname}/{sid}.jsonl`,
  appended under an exclusive flock (same write pattern as
  `JsonlTelemetrySink`). The journal is NOT a checkpoint artifact: it spans
  checkpoints and restores (a restore appends a marker, never truncates).
- stdout/stderr bodies stay out of the journal — sizes + sha256 only.
  Env values are recorded verbatim: replay needs them, and the journal
  lives in the same root-owned tree as checkpoint images which already
  contain the same secrets.
- `txn_id` is carried on every record but stays None until the B2
  transaction API tags actions.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from .ids import SandboxId
from .models import utc_now

logger = logging.getLogger(__name__)

JOURNAL_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ActionRecord:
    """One journal line. `payload` is kind-specific:

    kind="exec": argv, cwd, env, user, timeout_s, capture_output,
        returncode (None when timed out), duration_ms, timed_out,
        stdout_len/stdout_sha256, stderr_len/stderr_sha256.
    kind="lifecycle": event (launch/checkpoint/restore/fork_source/
        fork_created/destroy/...), plus event-specific metadata.
    kind="observation": adopted history from a consolidated fork (C3):
        fork_sandbox_id, origin_seq/kind/txn_id/timestamps,
        origin_payload (verbatim), reason — or origin_kind="summary"
        with a summarizer digest.
    """

    seq: int
    kind: str
    sandbox_id: str
    txn_id: str | None
    started_at: str
    finished_at: str | None
    payload: dict

    def to_json(self) -> dict:
        return {
            "schema": JOURNAL_SCHEMA_VERSION,
            "seq": self.seq,
            "kind": self.kind,
            "sandbox_id": self.sandbox_id,
            "txn_id": self.txn_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "payload": self.payload,
        }

    @classmethod
    def from_json(cls, data: dict) -> "ActionRecord":
        return cls(
            seq=int(data["seq"]),
            kind=str(data["kind"]),
            sandbox_id=str(data["sandbox_id"]),
            txn_id=None if data.get("txn_id") is None else str(data["txn_id"]),
            started_at=str(data["started_at"]),
            finished_at=None if data.get("finished_at") is None else str(data["finished_at"]),
            payload=dict(data.get("payload") or {}),
        )


def _text_digest(text: str | None) -> tuple[int | None, str | None]:
    """(byte length, sha256 hex) of captured output; (None, None) when the
    stream was not captured."""
    if text is None:
        return None, None
    encoded = text.encode("utf-8", errors="replace")
    return len(encoded), hashlib.sha256(encoded).hexdigest()


class ActionJournal:
    """Append-only per-sandbox JSONL journal.

    Thread-safe within a process (single lock; exec latency dwarfs a JSONL
    append) and safe across processes via flock on the per-sandbox file.
    Implements the `ActionRecorder` contract so the runtime can record
    exec attempts without importing storage.
    """

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._next_seq: dict[str, int] = {}
        # Active transaction per sandbox (B2): while set, records without
        # an explicit txn_id are stamped with it — the journal reflects
        # reality, not which API issued the action.
        self._active_txn_ids: dict[str, str] = {}

    @property
    def root(self) -> Path:
        return self._root

    def path_for(self, sandbox_id: SandboxId | str) -> Path:
        return self._root / f"{sandbox_id}.jsonl"

    def set_active_txn(self, sandbox_id: SandboxId | str, txn_id: str | None) -> None:
        """Set (or clear with None) the transaction id stamped onto this
        sandbox's records while no explicit txn_id is provided."""
        key = str(sandbox_id)
        with self._lock:
            if txn_id is None:
                self._active_txn_ids.pop(key, None)
            else:
                self._active_txn_ids[key] = str(txn_id)

    def active_txn(self, sandbox_id: SandboxId | str) -> str | None:
        with self._lock:
            return self._active_txn_ids.get(str(sandbox_id))

    # ------------------------------------------------------------------
    # ActionRecorder contract
    # ------------------------------------------------------------------

    def record_exec(
        self,
        sandbox_id: SandboxId,
        *,
        argv: list[str],
        cwd: str | None,
        env: dict[str, object] | None,
        user: str | None,
        timeout_s: float | None,
        capture_output: bool,
        returncode: int | None,
        duration_ms: float,
        stdout: str | None,
        stderr: str | None,
        started_at: str,
        finished_at: str,
        timed_out: bool = False,
        txn_id: str | None = None,
    ) -> ActionRecord:
        stdout_len, stdout_sha256 = _text_digest(stdout if capture_output else None)
        stderr_len, stderr_sha256 = _text_digest(stderr if capture_output else None)
        payload = {
            "argv": [str(item) for item in argv],
            "cwd": cwd,
            # str(value) is exactly what the runtime passes to `runc exec
            # --env KEY=VALUE`, so the journal round-trips replay-faithfully.
            "env": {str(key): str(value) for key, value in (env or {}).items()},
            "user": user,
            "timeout_s": timeout_s,
            "capture_output": bool(capture_output),
            "returncode": returncode,
            "duration_ms": round(float(duration_ms), 3),
            "timed_out": bool(timed_out),
            "stdout_len": stdout_len,
            "stdout_sha256": stdout_sha256,
            "stderr_len": stderr_len,
            "stderr_sha256": stderr_sha256,
        }
        return self._append(
            sandbox_id,
            kind="exec",
            payload=payload,
            started_at=started_at,
            finished_at=finished_at,
            txn_id=txn_id,
        )

    def record_lifecycle(
        self,
        sandbox_id: SandboxId,
        event: str,
        *,
        metadata: dict[str, object] | None = None,
        txn_id: str | None = None,
    ) -> ActionRecord:
        now = utc_now().isoformat()
        payload: dict[str, object] = {"event": str(event)}
        if metadata:
            payload["metadata"] = {str(key): value for key, value in metadata.items()}
        return self._append(
            sandbox_id,
            kind="lifecycle",
            payload=payload,
            started_at=now,
            finished_at=now,
            txn_id=txn_id,
        )

    def record_egress(
        self,
        sandbox_id: SandboxId | str,
        *,
        payload: dict,
        txn_id: str | None = None,
    ) -> ActionRecord:
        """Effect ledger (D1): one record per completed egress flow with
        the host/destination/protocol facts the proxy could observe
        without decrypting anything. The journal is the ledger's only
        store, so flows inherit ordering and the active-txn stamp."""
        now = utc_now().isoformat()
        return self._append(
            sandbox_id,
            kind="egress",
            payload=payload,
            started_at=now,
            finished_at=now,
            txn_id=txn_id,
        )

    def record_observation(
        self,
        sandbox_id: SandboxId | str,
        *,
        payload: dict,
        txn_id: str | None = None,
    ) -> ActionRecord:
        """Adopted history (C3): a record copied from another sandbox's
        journal (typically a fork being consolidated). The payload keeps
        the origin record verbatim under ``origin_payload`` plus
        provenance fields (fork id, origin seq/kind/txn/timestamps) so
        first-hand and adopted history stay distinguishable."""
        now = utc_now().isoformat()
        return self._append(
            sandbox_id,
            kind="observation",
            payload=payload,
            started_at=now,
            finished_at=now,
            txn_id=txn_id,
        )

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def known_sandbox_ids(self) -> list[str]:
        """Every sandbox with a journal on disk, newest file last.

        Startup reconciliation needs this: after a restart the in-memory
        state is gone, but the journal still holds rows describing work
        that was in flight.
        """
        try:
            files = [path for path in self._root.iterdir() if path.suffix == ".jsonl"]
        except OSError:
            return []
        return [path.stem for path in sorted(files, key=lambda p: p.name)]

    def entries(
        self,
        sandbox_id: SandboxId | str,
        *,
        kind: str | None = None,
        since_seq: int | None = None,
    ) -> list[ActionRecord]:
        path = self.path_for(sandbox_id)
        if not path.is_file():
            return []
        records: list[ActionRecord] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = ActionRecord.from_json(json.loads(line))
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    logger.warning("Skipping malformed journal line in %s", path)
                    continue
                if kind is not None and record.kind != kind:
                    continue
                if since_seq is not None and record.seq <= since_seq:
                    continue
                records.append(record)
        return records

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _append(
        self,
        sandbox_id: SandboxId | str,
        *,
        kind: str,
        payload: dict,
        started_at: str,
        finished_at: str | None,
        txn_id: str | None,
    ) -> ActionRecord:
        key = str(sandbox_id)
        path = self.path_for(sandbox_id)
        with self._lock:
            if txn_id is None:
                txn_id = self._active_txn_ids.get(key)
            seq = self._resolve_next_seq(key, path)
            record = ActionRecord(
                seq=seq,
                kind=kind,
                sandbox_id=key,
                txn_id=txn_id,
                started_at=started_at,
                finished_at=finished_at,
                payload=payload,
            )
            line = json.dumps(record.to_json(), sort_keys=True) + "\n"
            fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                try:
                    data = line.encode("utf-8")
                    while data:
                        written = os.write(fd, data)
                        data = data[written:]
                finally:
                    fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)
            self._next_seq[key] = seq + 1
        return record

    def _resolve_next_seq(self, key: str, path: Path) -> int:
        cached = self._next_seq.get(key)
        if cached is not None:
            return cached
        if not path.is_file():
            return 0
        # Recover the counter from the last well-formed line so appends
        # across process restarts stay monotonic.
        last_seq = -1
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        last_seq = int(json.loads(line).get("seq", last_seq))
                    except (json.JSONDecodeError, TypeError, ValueError):
                        continue
        except OSError:
            logger.exception("Failed to read journal for seq recovery: %s", path)
        return last_seq + 1


__all__ = [
    "ActionJournal",
    "ActionRecord",
    "JOURNAL_SCHEMA_VERSION",
]
