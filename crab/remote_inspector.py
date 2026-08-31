from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from datetime import datetime
from threading import Lock

from .contracts import SandboxInspector, TelemetrySink
from .http_utils import HttpStatusError, ThreadLocalHttpClient
from .ids import SandboxId
from .models import SandboxSnapshot, utc_now
from .telemetry import NoopTelemetrySink

logger = logging.getLogger(__name__)


def _parse_ts(raw: str | None) -> datetime | None:
    if raw is None:
        return None
    return datetime.fromisoformat(raw)


@dataclass(frozen=True)
class HostInspectorServiceClient:
    base_url: str
    # Must comfortably exceed the daemon's fs_sync timeout (5s in
    # HostInspectorDaemon.status). If the client bound equals the sync
    # bound, a sync that legitimately completes just under 5s still
    # races the client into a TimeoutError.
    timeout_s: float = 15.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_http_client",
            ThreadLocalHttpClient(self.base_url, timeout_seconds=self.timeout_s),
        )

    def _post(self, path: str, payload: dict[str, object]) -> dict[str, object]:
        return self._http_client.post_json(path, payload)

    def register_sandbox(
        self,
        sandbox_id: SandboxId,
        runtime: str,
        object_id: str,
        *,
        ignore_process_rules: list[dict[str, object]] | None = None,
        ignored_path_prefixes: list[str] | None = None,
    ) -> dict[str, object]:
        payload: dict[str, object] = {"sandbox_id": str(sandbox_id), "runtime": runtime, "object_id": object_id}
        if ignore_process_rules is not None:
            payload["ignore_process_rules"] = ignore_process_rules
        if ignored_path_prefixes is not None:
            payload["ignored_path_prefixes"] = ignored_path_prefixes
        return self._post("/register", payload)

    def unregister_sandbox(self, sandbox_id: SandboxId) -> dict[str, object]:
        try:
            return self._post("/unregister", {"sandbox_id": str(sandbox_id)})
        except HttpStatusError as exc:
            if exc.status_code != 404:
                raise
            # Teardown and create rollback are deliberately idempotent.  A
            # missing registration means the desired postcondition already
            # holds (including failures that happen before register()).
            return {
                "ok": True,
                "sandbox_id": str(sandbox_id),
                "unregistered": False,
                "already_absent": True,
            }

    def update_filters(
        self,
        sandbox_id: SandboxId,
        *,
        ignore_process_rules: list[dict[str, object]] | None = None,
        ignored_path_prefixes: list[str] | None = None,
    ) -> dict[str, object]:
        """Update ignore rules / path prefixes on an already-registered
        sandbox without resetting baseline pids or accumulated dirty state.

        Use this when an Agent attaches to an existing Sandbox and wants
        to add its own filters on top of the sandbox-default set; for the
        initial registration use `register_sandbox` instead."""
        payload: dict[str, object] = {"sandbox_id": str(sandbox_id)}
        if ignore_process_rules is not None:
            payload["ignore_process_rules"] = ignore_process_rules
        if ignored_path_prefixes is not None:
            payload["ignored_path_prefixes"] = ignored_path_prefixes
        return self._post("/update_filters", payload)

    def get_proc_and_fs_status(self, sandbox_id: SandboxId) -> dict[str, object]:
        return self._post("/get_proc_and_fs_status", {"sandbox_id": str(sandbox_id)})

    def reset_sandbox(
        self,
        sandbox_id: SandboxId,
        at: datetime | None,
        *,
        captures_process: bool = False,
    ) -> dict[str, object]:
        payload: dict[str, object] = {"sandbox_id": str(sandbox_id)}
        if at is not None:
            payload["at"] = at.isoformat()
        if captures_process:
            payload["captures_process"] = True
        return self._post("/reset", payload)

    def mark_filesystem_changed(
        self,
        sandbox_id: SandboxId,
        *,
        reason: str,
    ) -> dict[str, object]:
        """Conservatively invalidate the filesystem-clean baseline.

        This is used when a command intentionally runs in a descendant
        cgroup that the eBPF monitor cannot attribute to the sandbox's
        registered parent cgroup.  The marker is cleared by the same reset
        that clears ordinary filesystem events after a checkpoint.
        """

        return self._post(
            "/mark_filesystem_changed",
            {"sandbox_id": str(sandbox_id), "reason": str(reason)},
        )

    def close(self) -> None:
        self._http_client.close()


class RemoteSandboxInspector(SandboxInspector):
    def __init__(
        self,
        service_client: HostInspectorServiceClient,
        *,
        telemetry: TelemetrySink | None = None,
    ) -> None:
        self._service_client = service_client
        self._lock = Lock()
        self._snapshots: dict[SandboxId, SandboxSnapshot] = {}
        # (reset_at, captures_process). captures_process is sticky-OR
        # across deferrals: if any superseded pending reset followed a
        # full checkpoint, the eventual replay must still refresh the
        # daemon's `acknowledged_deleted_mmaps` baseline.
        self._pending_reset_at: dict[SandboxId, tuple[datetime, bool]] = {}
        self._telemetry: TelemetrySink = telemetry or NoopTelemetrySink()
        self._sync_timeout_count_lock = Lock()
        self._sync_timeout_count = 0

    @property
    def sync_timeout_count(self) -> int:
        with self._sync_timeout_count_lock:
            return self._sync_timeout_count

    def upsert_snapshot(self, snapshot: SandboxSnapshot) -> None:
        with self._lock:
            self._snapshots[snapshot.sandbox_id] = snapshot

        if snapshot.process_changed or snapshot.filesystem_changed:
            with self._lock:
                self._pending_reset_at.pop(snapshot.sandbox_id, None)
            return

        reset_at = snapshot.last_checkpoint_at or snapshot.observed_at
        try:
            # upsert_snapshot resets are baseline-clearing, NOT
            # checkpoint-completion resets — they reflect the inspector's
            # own clean snapshot, with no fresh process image dumped. So
            # captures_process stays False here; only mark_checkpoint_complete
            # may flip it on for the post-checkpoint reset.
            self._service_client.reset_sandbox(snapshot.sandbox_id, reset_at)
        except Exception as exc:  # noqa: BLE001
            if self._is_unknown_sandbox_error(exc):
                with self._lock:
                    self._defer_reset_locked(snapshot.sandbox_id, reset_at, captures_process=False)
                logger.debug(f"pending a sandbox reset for sandbox={snapshot.sandbox_id}")
                return
            logger.debug(
                "Failed to sync clean snapshot baseline for sandbox %s",
                snapshot.sandbox_id,
                exc_info=True,
            )
        else:
            with self._lock:
                self._pending_reset_at.pop(snapshot.sandbox_id, None)

    def inspect(self, sandbox_id: SandboxId) -> SandboxSnapshot:
        self._apply_pending_reset(sandbox_id)
        with self._lock:
            local_snapshot = self._snapshots.get(sandbox_id)
        try:
            remote_snapshot = self._read_remote_snapshot(sandbox_id)
        except Exception as exc:  # noqa: BLE001
            if local_snapshot is not None:
                return local_snapshot
            return SandboxSnapshot(
                sandbox_id=sandbox_id,
                runtime_name="remote-inspector-unavailable",
                is_running=True,
                process_changed=True,
                filesystem_changed=True,
                observed_at=utc_now(),
                last_checkpoint_at=None,
                metadata={"inspector_error": str(exc)},
            )
        return self._merge_snapshots(local_snapshot, remote_snapshot)

    def mark_checkpoint_complete(
        self,
        sandbox_id: SandboxId,
        *,
        process: bool,
        filesystem: bool,
        at: datetime,
    ) -> None:
        if not process and not filesystem:
            return
        # The checkpoint has already succeeded in storage; the daemon-side
        # reset is a best-effort baseline refresh so the next inspect() can
        # report a clean dirty signal. If the daemon call fails (unknown
        # sandbox, transient timeout), fall back to pending_reset_at so the
        # next inspect retries the reset — mirroring upsert_snapshot. A
        # raised exception here would escape _execute_checkpoint_flow past
        # operation.finish and create an orphan flow.start in telemetry.
        #
        # Pass `captures_process` so the daemon can refresh its
        # `acknowledged_deleted_mmaps` baseline for full checkpoints (the
        # process image just dumped any `(deleted)` mmap content inline, so
        # those paths should no longer fire mmap_invalidation). For
        # filesystem-only checkpoints we leave the baseline alone — the
        # process image is unchanged, so its set of frozen content is too.
        try:
            self._service_client.reset_sandbox(sandbox_id, at, captures_process=process)
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self._defer_reset_locked(sandbox_id, at, captures_process=process)
            if self._is_unknown_sandbox_error(exc):
                logger.debug(
                    "Deferring mark_checkpoint_complete reset for unknown sandbox=%s",
                    sandbox_id,
                )
            else:
                logger.warning(
                    "mark_checkpoint_complete daemon reset failed for sandbox=%s; deferring",
                    sandbox_id,
                    exc_info=True,
                )
        else:
            with self._lock:
                self._pending_reset_at.pop(sandbox_id, None)
        with self._lock:
            snapshot = self._snapshots.get(sandbox_id)
            if snapshot is None:
                return
            self._snapshots[sandbox_id] = replace(
                snapshot,
                process_changed=False if process else snapshot.process_changed,
                filesystem_changed=False if filesystem else snapshot.filesystem_changed,
                observed_at=max(snapshot.observed_at, at),
                last_checkpoint_at=at,
            )

    def _defer_reset_locked(
        self,
        sandbox_id: SandboxId,
        reset_at: datetime,
        *,
        captures_process: bool,
    ) -> None:
        # Caller holds self._lock. captures_process is sticky-OR across
        # successive deferrals so a full-checkpoint reset can never get
        # silently downgraded by a later fs-only-checkpoint reset queueing
        # behind it (the daemon's `acknowledged_deleted_mmaps` baseline must
        # still be refreshed when the deferred reset eventually replays).
        existing = self._pending_reset_at.get(sandbox_id)
        if existing is None:
            self._pending_reset_at[sandbox_id] = (reset_at, captures_process)
            return
        existing_at, existing_captures = existing
        merged_at = max(existing_at, reset_at)
        merged_captures = existing_captures or captures_process
        self._pending_reset_at[sandbox_id] = (merged_at, merged_captures)

    def _apply_pending_reset(self, sandbox_id: SandboxId) -> None:
        with self._lock:
            pending = self._pending_reset_at.get(sandbox_id)
        if pending is None:
            return
        reset_at, captures_process = pending
        try:
            self._service_client.reset_sandbox(
                sandbox_id, reset_at, captures_process=captures_process
            )
        except Exception as exc:  # noqa: BLE001
            if not self._is_unknown_sandbox_error(exc):
                logger.debug(
                    "Failed to apply deferred remote reset for sandbox %s",
                    sandbox_id,
                    exc_info=True,
                )
            return
        with self._lock:
            self._pending_reset_at.pop(sandbox_id, None)

    def _read_remote_snapshot(self, sandbox_id: SandboxId) -> SandboxSnapshot:
        payload = self._service_client.get_proc_and_fs_status(sandbox_id)
        status = dict(payload["status"])
        metadata = dict(status.get("metadata", {}))
        last_reset_at = _parse_ts(status.get("last_reset_at"))
        if last_reset_at is not None:
            metadata["host_last_reset_at"] = last_reset_at.isoformat()
        if bool(metadata.get("fs_sync_timeout", False)):
            with self._sync_timeout_count_lock:
                self._sync_timeout_count += 1
                total = self._sync_timeout_count
            logger.warning(
                "host-inspector fs sync timeout for sandbox=%s (cumulative=%d)",
                sandbox_id,
                total,
            )
            self._telemetry.emit_event(
                "host_inspector.sync_timeout",
                {
                    "sandbox_id": str(sandbox_id),
                    "cumulative_count": total,
                },
            )
            self._telemetry.emit_metric(
                "host_inspector.sync_timeout_count",
                1.0,
                {"sandbox_id": str(sandbox_id)},
            )
        return SandboxSnapshot(
            sandbox_id=sandbox_id,
            runtime_name=str(status["runtime_name"]),
            is_running=bool(status["is_running"]),
            process_changed=bool(status["process_changed"]),
            filesystem_changed=bool(status["filesystem_changed"]),
            observed_at=_parse_ts(status.get("observed_at")) or utc_now(),
            # Host-inspector resets also happen for non-checkpoint baselines.
            # Only explicit mark_checkpoint_complete() calls should advance the
            # scheduler-visible checkpoint timestamp.
            last_checkpoint_at=None,
            metadata=metadata,
        )

    def _merge_snapshots(
        self,
        local_snapshot: SandboxSnapshot | None,
        remote_snapshot: SandboxSnapshot,
    ) -> SandboxSnapshot:
        if local_snapshot is None:
            return remote_snapshot

        prefer_local = local_snapshot.observed_at >= remote_snapshot.observed_at
        if prefer_local:
            runtime_name = local_snapshot.runtime_name
            is_running = local_snapshot.is_running
        else:
            runtime_name = remote_snapshot.runtime_name
            is_running = remote_snapshot.is_running

        checkpoints = [x for x in (local_snapshot.last_checkpoint_at, remote_snapshot.last_checkpoint_at) if x is not None]
        last_checkpoint_at = max(checkpoints) if checkpoints else None
        return SandboxSnapshot(
            sandbox_id=remote_snapshot.sandbox_id,
            runtime_name=runtime_name,
            is_running=is_running,
            process_changed=local_snapshot.process_changed or remote_snapshot.process_changed,
            filesystem_changed=local_snapshot.filesystem_changed or remote_snapshot.filesystem_changed,
            observed_at=max(local_snapshot.observed_at, remote_snapshot.observed_at),
            last_checkpoint_at=last_checkpoint_at,
            metadata={**remote_snapshot.metadata, **local_snapshot.metadata},
        )

    def _is_unknown_sandbox_error(self, exc: Exception) -> bool:
        if isinstance(exc, KeyError):
            return True
        if isinstance(exc, HttpStatusError):
            return exc.status_code == 404
        return False
