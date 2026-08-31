from __future__ import annotations

import argparse
import logging
import os
import socket
import time
from dataclasses import replace
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from threading import Event, Lock, Thread
from typing import Any

from ..http_utils import PooledHTTPServer
from ..json_codec import get_json_codec
from ..models import utc_now
from .fs_helper import LibbpfFilesystemMonitor
from .process_filter import (
    PidIdentityCache,
    classify_pid_against_rules,
    parse_process_ignore_rules,
    pid_matches_ignore_rules,
)
from .process_monitor import (
    MmapPathCache,
    all_deleted_mmap_paths,
    dirty_pids,
    list_cgroup_pids,
    path_invalidates_mmap,
    reset_soft_dirty_for_pids,
)
from .runtime_resolver import ResolvedSandbox, RuntimeResolver
from .state import SandboxRecord

logger = logging.getLogger(__name__)
_JSON_CODEC = get_json_codec("auto")

_OPEN_SYSCALLS = {"open", "openat", "openat2", "creat"}
# Write-family syscalls that mutate via an existing fd. BPF captures the
# fd in fd_from_enter, so _is_countable_fs_event can require fd >= 0 and
# filter on fd_kind to drop writes to terminals/pipes.
_WRITE_FD_SYSCALLS = {"write", "pwrite64", "writev", "pwritev", "pwritev2", "ftruncate"}
# truncate(2) is path-based, not fd-based: BPF captures arg0 as the
# primary path and never populates fd. It belongs with the other
# unconditional path-based mutations below, not with the fd-gated ones.
_WRITE_PATH_SYSCALLS = {"truncate"}
_METADATA_PATH_SYSCALLS = {
    "chmod",
    "fchmodat",
    "chown",
    "fchownat",
    "lchown",
    "setxattr",
    "lsetxattr",
    "removexattr",
    "lremovexattr",
}
_METADATA_FD_SYSCALLS = {"fchmod", "fchown", "fsetxattr", "fremovexattr"}
_CREATE_PATH_SYSCALLS = {"mkdir", "mkdirat", "mknod", "mknodat"}
_DELETE_SYSCALLS = {"unlink", "unlinkat", "rmdir"}
_RENAME_SYSCALLS = {"rename", "renameat", "renameat2"}
_SECONDARY_TARGET_SYSCALLS = {"link", "linkat", "symlink", "symlinkat"}
_DEVICE_OR_STREAM_FD_KINDS = {"char", "block", "fifo", "socket"}
_MUTATING_FD_KINDS = {"regular", "directory", "symlink", "unknown"}
# Path prefixes that are known to be non-persistent pseudo-filesystems. A
# mutating open against one of these should be ignored even when fd_kind looks
# "regular", and conversely, a path outside of this set is almost certainly a
# real file regardless of what post-hoc fd_kind resolution reports.
_NON_PERSISTENT_PATH_PREFIXES = ("/dev/", "/proc/", "/sys/")
_NON_PERSISTENT_PATH_EXACT = {"/dev", "/proc", "/sys"}
_RECENT_EVENT_LIMIT = 8

# Lock-contention instrumentation thresholds. Sustained hold/wait times
# above these are actionable — they bleed into the fs_monitor.sync()
# budget and cause fs_sync_timeouts. 50ms picks up a single runc-state
# subprocess fork+exec and anything slower.
_LOCK_HOLD_WARN_MS = 50.0
_LOCK_WAIT_WARN_MS = 50.0


def _isoformat(ts: datetime | None) -> str | None:
    return None if ts is None else ts.isoformat()


def _parse_ignored_path_prefixes(raw: object | None) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise TypeError("ignored_path_prefixes must be a list of strings")
    out: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            raise TypeError("ignored_path_prefixes entries must be strings")
        cleaned = item.strip()
        if cleaned:
            out.append(cleaned)
    return tuple(out)


def _parse_ts(raw: str | None) -> datetime | None:
    if raw is None:
        return None
    return datetime.fromisoformat(raw)


class HostInspectorDaemon:
    def __init__(
        self,
        *,
        resolver: RuntimeResolver | None = None,
        fs_monitor: LibbpfFilesystemMonitor | None = None,
        process_poll_interval_s: float = 1.0,
    ) -> None:
        self._resolver = resolver or RuntimeResolver()
        self._fs_monitor = fs_monitor or LibbpfFilesystemMonitor()
        self._process_poll_interval_s = process_poll_interval_s
        self._records_lock = Lock()
        self._sandbox_locks_guard = Lock()
        self._sandbox_locks: dict[str, Lock] = {}
        self._records: dict[str, SandboxRecord] = {}
        self._bpf_ignored_pids: set[int] = set()
        self._bpf_ignored_pids_lock = Lock()
        # Cache pid → ProcessIdentity reads on the high-rate fs-event hot
        # path. Without this, every fs event triggered 2 syscalls in
        # `read_process_identity` (and a parent-chain walk for
        # ancestor-scoped rules), and `_handle_fs_event` ran
        # ~10–40k times/sec under tmux pane workloads — that proc-read
        # fan-out was the dominant cost responsible for the
        # `fs_monitor.sync(timeout_s=5.0)` barrier missing its budget.
        self._pid_identity_cache = PidIdentityCache()
        # Cache /proc/<pid>/maps reads on the per-event hot path. The
        # mmap-invalidation check in `_handle_fs_event` was the second
        # dominant per-worker cost (after pid identity reads), since
        # every fs event re-read the maps file for every live pid in
        # the cgroup. Caching for 1s lets bursty event streams
        # against the same pid set reuse one read; the kernel-truth
        # fallback in status() reads `(deleted)` mmaps uncached, so a
        # stale event-path miss is recovered on the next status()
        # call without correctness loss.
        self._mmap_path_cache = MmapPathCache()

    def _sandbox_lock(self, sandbox_id: str) -> Lock:
        with self._sandbox_locks_guard:
            lock = self._sandbox_locks.get(sandbox_id)
            if lock is None:
                lock = Lock()
                self._sandbox_locks[sandbox_id] = lock
            return lock

    def _sync_bpf_ignored_pids(self, new_ignored_pids: set[int]) -> None:
        """Reconcile BPF ignored_pids map with the per-sandbox ignored set.

        `new_ignored_pids` is the set currently computed for one sandbox.
        We maintain a global set `_bpf_ignored_pids` that mirrors the BPF
        map state. Any newly-ignored pid is added; any previously-ignored
        pid that's no longer ignored for any sandbox is removed. Pids are
        kernel-global so collisions across sandboxes are a non-issue —
        an ignored pid is ignored regardless of which sandbox saw it.
        """
        with self._bpf_ignored_pids_lock:
            to_add = new_ignored_pids - self._bpf_ignored_pids
            # Union of fs-eligible ignored sets across all sandboxes (the
            # caller's new set is already included in new_ignored_pids).
            # `record.fs_ignored_pids` excludes pids ignored under
            # scope=process_only, so they never enter the kernel filter.
            with self._records_lock:
                union = set(new_ignored_pids)
                for record in self._records.values():
                    union |= set(record.fs_ignored_pids)
            to_remove = self._bpf_ignored_pids - union
            for pid in to_add:
                try:
                    self._fs_monitor.add_ignored_pid(pid)
                    self._bpf_ignored_pids.add(pid)
                except Exception:
                    logger.exception("failed to add ignored pid %s to BPF map", pid)
            for pid in to_remove:
                try:
                    self._fs_monitor.remove_ignored_pid(pid)
                    self._bpf_ignored_pids.discard(pid)
                except Exception:
                    logger.exception("failed to remove ignored pid %s from BPF map", pid)

    def start(self) -> None:
        self._fs_monitor.start(self._handle_fs_event)
        logger.info(
            "process change detection is on-demand; process_poll_interval=%s is accepted for compatibility only",
            self._process_poll_interval_s,
        )

    def stop(self) -> None:
        self._fs_monitor.stop()

    def register(
        self,
        sandbox_id: str,
        runtime: str,
        object_id: str,
        *,
        ignore_process_rules: object | None = None,
        ignored_path_prefixes: object | None = None,
    ) -> dict[str, object]:
        resolved = self._resolver.resolve(runtime, object_id)
        observed_at = utc_now()
        parsed_ignore_rules = parse_process_ignore_rules(ignore_process_rules)
        parsed_path_prefixes = _parse_ignored_path_prefixes(ignored_path_prefixes)
        sandbox_lock = self._sandbox_lock(sandbox_id)
        with sandbox_lock:
            with self._records_lock:
                previous = self._records.get(sandbox_id)
        all_current_pids = list_cgroup_pids(resolved.cgroup_path)
        tracked_pids, ignored_pids, fs_ignored_pids = self._split_pids(all_current_pids, parsed_ignore_rules)
        record = SandboxRecord(
            sandbox_id=sandbox_id,
            runtime=runtime,
            object_id=object_id,
            runtime_name=resolved.runtime_name,
            is_running=resolved.is_running,
            init_pid=resolved.init_pid,
            cgroup_path=resolved.cgroup_path,
            cgroup_id=resolved.cgroup_id,
            ignore_process_rules=parsed_ignore_rules,
            ignored_path_prefixes=parsed_path_prefixes,
            tracked_pids=tracked_pids,
            ignored_pids=ignored_pids,
            fs_ignored_pids=fs_ignored_pids,
            current_pids=tracked_pids,
            process_changed=False,
            filesystem_changed=False,
            observed_at=observed_at,
        )
        with sandbox_lock:
            with self._records_lock:
                self._records[sandbox_id] = record
            # Snapshot under sandbox_lock: _handle_fs_event mutates the
            # record's live_dirty_entries / recent_fs_events /
            # unreconciled_fs_events in place under the same lock, so
            # iterating them after release would race.
            response = self._record_to_response(record)
        if previous is not None and previous.cgroup_id is not None and previous.cgroup_id != resolved.cgroup_id:
            self._fs_monitor.remove_sandbox(sandbox_id)
        if resolved.cgroup_id is not None:
            self._fs_monitor.upsert_sandbox(sandbox_id, resolved.cgroup_id)
        # Push the path-prefix filter into the C helper so matching
        # events get dropped before they cross the JSON/parse boundary.
        # Must come AFTER upsert_sandbox so the helper can find the
        # registration record by sandbox_id.
        if hasattr(self._fs_monitor, "set_ignored_path_prefixes"):
            try:
                self._fs_monitor.set_ignored_path_prefixes(sandbox_id, parsed_path_prefixes)
            except Exception:
                logger.exception(
                    "Failed to push ignored_path_prefixes to fs helper sandbox=%s",
                    sandbox_id,
                )
        # Push fs-eligible (scope=all) ignore rules to the helper so it
        # can drop matched events before they cross IPC. The Python
        # `_handle_fs_event` still re-evaluates the same rules as
        # belt-and-suspenders for any path that bypasses the helper.
        if hasattr(self._fs_monitor, "set_ignore_process_rules"):
            try:
                self._fs_monitor.set_ignore_process_rules(sandbox_id, parsed_ignore_rules)
            except Exception:
                logger.exception(
                    "Failed to push ignore_process_rules to fs helper sandbox=%s",
                    sandbox_id,
                )
        # Only the fs-ignored subset goes into the kernel filter; pids
        # ignored under scope=process_only must keep delivering fs events.
        self._sync_bpf_ignored_pids(fs_ignored_pids)
        logger.debug(
            "register sandbox=%s runtime=%s object=%s cgroup_id=%s init_pid=%s previous_cgroup_id=%s observed_at=%s",
            sandbox_id,
            runtime,
            object_id,
            resolved.cgroup_id,
            resolved.init_pid,
            (previous.cgroup_id if previous is not None else None),
            _isoformat(observed_at),
        )
        # Do not call status() here: status() drains the BPF sync pipeline
        # (up to 5s) and the client's HTTP timeout is on the same order,
        # so registering 128 sandboxes in parallel at startup would race
        # the client into giving up before the server responded. The
        # freshly-built record is already the baseline the caller needs.
        return response

    def unregister(self, sandbox_id: str) -> dict[str, object]:
        sandbox_lock = self._sandbox_lock(sandbox_id)
        with sandbox_lock:
            with self._records_lock:
                record = self._records.pop(sandbox_id, None)
        if record is None:
            raise KeyError(sandbox_id)
        self._fs_monitor.remove_sandbox(sandbox_id)
        # After dropping the record, any pids that were ignored only for
        # this sandbox should leave the BPF map. Passing an empty set
        # prompts the reconciler to recompute the union over remaining
        # records and remove anything stale.
        self._sync_bpf_ignored_pids(set())
        return {"sandbox_id": sandbox_id, "unregistered": True}

    def update_filters(
        self,
        sandbox_id: str,
        *,
        ignore_process_rules: object | None = None,
        ignored_path_prefixes: object | None = None,
    ) -> dict[str, object]:
        """Update an already-registered sandbox's ignore rules in place.

        Unlike `register`, this does NOT reset baseline_pids, last_reset_at,
        live_dirty_entries, or the process/filesystem-changed cached signals
        — only the filter sets and any derived tracked/ignored partitions
        are recomputed against the *current* cgroup pid set. Existing
        baseline pids that move from tracked → ignored under the new rules
        are also dropped from `baseline_pids`, so the next status call's
        `tracked_pids != baseline_pids` comparison stays sound.
        """
        parsed_ignore_rules = parse_process_ignore_rules(ignore_process_rules)
        parsed_path_prefixes = _parse_ignored_path_prefixes(ignored_path_prefixes)
        sandbox_lock = self._sandbox_lock(sandbox_id)
        with sandbox_lock:
            with self._records_lock:
                record = self._records.get(sandbox_id)
                if record is None:
                    raise KeyError(sandbox_id)
            resolved = self._resolver.resolve(record.runtime, record.object_id)
            all_current_pids = list_cgroup_pids(resolved.cgroup_path)
            tracked_pids, ignored_pids, fs_ignored_pids = self._split_pids(
                all_current_pids, parsed_ignore_rules
            )
            # Drop any baseline entries that are now ignored under the new
            # rules; otherwise `tracked_pids != baseline_pids` would latch
            # process_changed=True forever on the next status() call.
            new_baseline = {pid for pid in record.baseline_pids if pid not in ignored_pids}
            updated = replace(
                record,
                ignore_process_rules=parsed_ignore_rules,
                ignored_path_prefixes=parsed_path_prefixes,
                tracked_pids=tracked_pids,
                ignored_pids=ignored_pids,
                fs_ignored_pids=fs_ignored_pids,
                current_pids=tracked_pids,
                baseline_pids=new_baseline,
            )
            with self._records_lock:
                self._records[sandbox_id] = updated
            response = self._record_to_response(updated)
        # Push the new filters to the C helper (same flow as register).
        if hasattr(self._fs_monitor, "set_ignored_path_prefixes"):
            try:
                self._fs_monitor.set_ignored_path_prefixes(sandbox_id, parsed_path_prefixes)
            except Exception:
                logger.exception(
                    "Failed to push ignored_path_prefixes to fs helper sandbox=%s",
                    sandbox_id,
                )
        if hasattr(self._fs_monitor, "set_ignore_process_rules"):
            try:
                self._fs_monitor.set_ignore_process_rules(sandbox_id, parsed_ignore_rules)
            except Exception:
                logger.exception(
                    "Failed to push ignore_process_rules to fs helper sandbox=%s",
                    sandbox_id,
                )
        self._sync_bpf_ignored_pids(fs_ignored_pids)
        logger.debug(
            "update_filters sandbox=%s rules=%d prefixes=%d new_tracked=%d new_ignored=%d",
            sandbox_id,
            len(parsed_ignore_rules),
            len(parsed_path_prefixes),
            len(tracked_pids),
            len(ignored_pids),
        )
        return response

    def mark_filesystem_changed(
        self,
        sandbox_id: str,
        *,
        reason: str,
    ) -> dict[str, object]:
        """Latch a conservative dirty marker until the next reset.

        The kernel monitor keys filesystem events by the task's current
        cgroup ID.  A timeout-isolated exec runs in a child cgroup, so its
        events cannot be matched to the sandbox's registered parent cgroup.
        Recording one synthetic unresolved event avoids a false-clean result;
        the authoritative backend diff still determines the actual changes.
        """

        when = utc_now()
        marker = {
            "kind": "conservative_invalidation",
            "counts_as_change": True,
            "reason": str(reason),
            "timestamp": when.isoformat(),
        }
        sandbox_lock = self._sandbox_lock(sandbox_id)
        with sandbox_lock:
            with self._records_lock:
                existing = self._records.get(sandbox_id)
                if existing is None:
                    raise KeyError(sandbox_id)
            unresolved = [*existing.unreconciled_fs_events, marker]
            if len(unresolved) > _RECENT_EVENT_LIMIT:
                unresolved = unresolved[-_RECENT_EVENT_LIMIT:]
            updated = replace(
                existing,
                filesystem_changed=True,
                observed_at=max(existing.observed_at or when, when),
                unreconciled_fs_events=unresolved,
            )
            with self._records_lock:
                self._records[sandbox_id] = updated
            return self._record_to_response(updated)

    def status(self, sandbox_id: str) -> dict[str, object]:
        # Drain in-flight BPF events before reading state. Under load the
        # kernel→helper→daemon pipeline lags by several seconds, so without
        # the sync we would return stale state and the scheduler would
        # decline a checkpoint immediately after a command that did mutate
        # the filesystem. The sync() call blocks until the fs helper has
        # processed every event currently waiting in the ring buffer AND
        # _handle_fs_event has applied each to the sandbox record.
        #
        # Timeout sizing: at 128-sandbox load the Python reader legitimately
        # needs 3-5s to drain a full event backlog (measured: 6 timeouts/s
        # sustained at a 2s timeout). Setting the bound at 5s absorbs the
        # observed worst case without masking a genuine pipeline stall; a
        # sync that cannot complete in 5s signals real breakage rather than
        # steady-state backpressure.
        #
        # We pass `sandbox_id` so the barrier fences only the worker
        # bucket that owns this sandbox. Without it, a single noisy
        # sandbox's event burst (apt-get, tar, build) would block every
        # other sandbox's status() until its bucket drained — the
        # cross-sandbox HOL blocking that produced the bulk of the
        # cumulative fs_sync_timeout warnings in benchmark runs.
        fs_sync_timeout = not self._fs_monitor.sync(sandbox_id, timeout_s=5.0)
        if fs_sync_timeout:
            # If sync did not complete in time, we cannot trust the
            # cached filesystem_changed signal: in-flight events may
            # include a mutation that the scheduler must see. Force
            # filesystem_changed=True so the scheduler errs on the side
            # of taking a checkpoint. Log loudly (WARNING) so this
            # shows in stdout during benchmark runs, and expose the
            # flag in the response metadata so clients can count and
            # surface sync timeouts via their own telemetry.
            logger.warning(
                "host-inspector fs sync timed out for sandbox=%s; forcing filesystem_changed=True",
                sandbox_id,
            )
        sandbox_lock = self._sandbox_lock(sandbox_id)
        lock_hold_start = time.monotonic()
        with sandbox_lock:
            with self._records_lock:
                record = self._records.get(sandbox_id)
                if record is None:
                    raise KeyError(sandbox_id)
            if record.last_reset_at is None:
                response = self._record_to_response(record)
                if fs_sync_timeout:
                    response["filesystem_changed"] = True
                    _metadata = dict(response.get("metadata", {}) or {})
                    _metadata["fs_sync_timeout"] = True
                    response["metadata"] = _metadata
                hold_ms = (time.monotonic() - lock_hold_start) * 1000.0
                if hold_ms > _LOCK_HOLD_WARN_MS:
                    logger.warning(
                        "sandbox_lock held %.1fms in status(no-baseline) sandbox=%s",
                        hold_ms, sandbox_id,
                    )
                return response

            # Note: the three calls below (resolver.resolve, list_cgroup_pids,
            # dirty_pids) run INSIDE sandbox_lock, and resolver.resolve in
            # particular fork-execs a `runc state` subprocess that typically
            # costs tens of milliseconds. The fs-event worker for this
            # sandbox blocks on the same lock, which is why fs_monitor.sync
            # barriers stall. Timed below so the magnitude shows up in the
            # host-inspector log.
            resolve_start = time.monotonic()
            resolved = self._resolver.resolve(record.runtime, record.object_id)
            resolve_ms = (time.monotonic() - resolve_start) * 1000.0
            all_current_pids = list_cgroup_pids(resolved.cgroup_path)
            tracked_pids, ignored_pids, fs_ignored_pids = self._split_pids(all_current_pids, record.ignore_process_rules)
            dirty = dirty_pids(tracked_pids)
            live_dirty_entries = self._reconcile_live_dirty_entries(record.live_dirty_entries, resolved.init_pid)
            filesystem_changed = bool(live_dirty_entries) or any(
                bool(item.get("counts_as_change")) for item in record.unreconciled_fs_events
            )
            if fs_sync_timeout:
                filesystem_changed = True
            mmap_invalidated = record.mmap_invalidated
            mmap_invalidated_path = record.mmap_invalidated_path
            if not mmap_invalidated:
                # Kernel-truth fallback for the eBPF-event-driven detection in
                # _handle_fs_event: at high concurrency the perf ringbuf can
                # drop file-write events for a sandbox (apt-get install often
                # rewrites both libc.so.6 and ld-linux-x86-64.so.2 within a
                # tight burst), and a missed event leaves mmap_invalidated
                # False. The kernel marks any /proc/{pid}/maps entry whose
                # backing file was unlinked with a ` (deleted)` suffix, so
                # scanning the live pids' maps here recovers the signal even
                # when the corresponding eBPF event never made it through.
                #
                # Scan EVERY pid that CRIU will dump (tracked + ignored +
                # init), not just tracked. The whole point of CRIU's process
                # checkpoint is that it captures the entire cgroup, so a
                # `(deleted)` mmap on ANY of those pids is what breaks
                # restore from a stale process image. Excluding ignored pids
                # here was a fault-7 footgun: with the tmux ignore rules
                # active, `tracked_pids` is empty (sleep init / tmux server
                # / pane bash all matched), so this fallback never fired
                # even though sleep — the actual init that mmaps libc.so.6
                # — sat in `ignored_pids` waiting to be detected.
                #
                # Latching is suppressed by `acknowledged_deleted_mmaps`:
                # once a full process checkpoint runs, reset() captures the
                # current `(deleted)` set, and only NEW paths fire here
                # afterwards. CRIU stores unlinked file content inline in
                # the dump image, so a path already captured won't break
                # any future restore from that image.
                live_pids = set(tracked_pids) | set(ignored_pids)
                if resolved.init_pid:
                    live_pids.add(int(resolved.init_pid))
                deleted_path: str | None = None
                if live_pids:
                    for candidate in all_deleted_mmap_paths(live_pids):
                        if candidate not in record.acknowledged_deleted_mmaps:
                            deleted_path = candidate
                            break
                if deleted_path is not None:
                    mmap_invalidated = True
                    mmap_invalidated_path = deleted_path
                    logger.info(
                        "status mmap_invalidation_via_proc_maps sandbox=%s path=%s live_pids=%s acknowledged=%d",
                        sandbox_id,
                        deleted_path,
                        sorted(live_pids),
                        len(record.acknowledged_deleted_mmaps),
                    )
            updated = replace(
                record,
                runtime_name=resolved.runtime_name,
                is_running=resolved.is_running,
                init_pid=resolved.init_pid,
                cgroup_path=resolved.cgroup_path,
                cgroup_id=resolved.cgroup_id,
                current_pids=tracked_pids,
                dirty_pids=dirty,
                tracked_pids=tracked_pids,
                ignored_pids=ignored_pids,
                fs_ignored_pids=fs_ignored_pids,
                process_changed=(
                    (tracked_pids != record.baseline_pids)
                    or bool(dirty)
                    or mmap_invalidated
                ),
                filesystem_changed=filesystem_changed,
                live_dirty_entries=live_dirty_entries,
                mmap_invalidated=mmap_invalidated,
                mmap_invalidated_path=mmap_invalidated_path,
                observed_at=max(record.observed_at or utc_now(), utc_now()),
                last_error=None,
            )
            with self._records_lock:
                self._records[sandbox_id] = updated
            # Snapshot under sandbox_lock so _handle_fs_event's in-place
            # mutation of live_dirty_entries / unreconciled_fs_events /
            # recent_fs_events cannot race with this iteration.
            response = self._record_to_response(updated)
        hold_ms = (time.monotonic() - lock_hold_start) * 1000.0
        if hold_ms > _LOCK_HOLD_WARN_MS:
            logger.warning(
                "sandbox_lock held %.1fms in status() sandbox=%s resolver_ms=%.1f pids=%d",
                hold_ms, sandbox_id, resolve_ms, len(all_current_pids),
            )
        if resolved.cgroup_id != record.cgroup_id:
            self._fs_monitor.remove_sandbox(sandbox_id)
            if resolved.cgroup_id is not None:
                self._fs_monitor.upsert_sandbox(sandbox_id, resolved.cgroup_id)
        self._sync_bpf_ignored_pids(fs_ignored_pids)
        logger.debug(
            "status sandbox=%s cgroup_id=%s init_pid=%s process_changed=%s filesystem_changed=%s "
            "live_dirty=%d unreconciled=%d last_reset_at=%s observed_at=%s",
            sandbox_id,
            resolved.cgroup_id,
            resolved.init_pid,
            updated.process_changed,
            updated.filesystem_changed,
            len(updated.live_dirty_entries),
            len(updated.unreconciled_fs_events),
            _isoformat(updated.last_reset_at) if updated.last_reset_at else None,
            _isoformat(updated.observed_at) if updated.observed_at else None,
        )
        if fs_sync_timeout:
            _metadata = dict(response.get("metadata", {}) or {})
            _metadata["fs_sync_timeout"] = True
            response["metadata"] = _metadata
        return response

    def reset(
        self,
        sandbox_id: str,
        at: datetime | None = None,
        *,
        captures_process: bool = False,
    ) -> dict[str, object]:
        when = at or utc_now()
        sandbox_lock = self._sandbox_lock(sandbox_id)
        lock_hold_start = time.monotonic()
        with sandbox_lock:
            with self._records_lock:
                existing = self._records.get(sandbox_id)
            if existing is None:
                raise KeyError(sandbox_id)
            logger.debug(
                "reset.enter sandbox=%s when=%s before_live=%d before_unreconciled=%d "
                "before_filesystem_changed=%s last_reset_at=%s observed_at=%s",
                sandbox_id,
                _isoformat(when),
                len(existing.live_dirty_entries),
                len(existing.unreconciled_fs_events),
                existing.filesystem_changed,
                _isoformat(existing.last_reset_at) if existing.last_reset_at else None,
                _isoformat(existing.observed_at) if existing.observed_at else None,
            )

            # Same bottleneck shape as status(): runc subprocess under lock.
            resolve_start = time.monotonic()
            resolved = self._resolver.resolve(existing.runtime, existing.object_id)
            resolve_ms = (time.monotonic() - resolve_start) * 1000.0
            all_current_pids = list_cgroup_pids(resolved.cgroup_path)
            tracked_pids, ignored_pids, fs_ignored_pids = self._split_pids(all_current_pids, existing.ignore_process_rules)
            soft_dirty_start = time.monotonic()
            # When this reset follows a full process checkpoint, CRIU dumped
            # the tasks with --leave-running; its parasite/resume writes can
            # leave a few residual soft-dirty pages that land around this
            # clear. Stabilize (re-clear until clean or attempts exhausted) so
            # an otherwise idle checkpointed sandbox does not latch a false
            # process_changed=True. Genuinely busy processes keep re-dirtying
            # and are still reported by the next status() poll.
            baseline_pids = reset_soft_dirty_for_pids(tracked_pids, stabilize=captures_process)
            soft_dirty_ms = (time.monotonic() - soft_dirty_start) * 1000.0

            # When this reset follows a full process checkpoint, snapshot
            # the `(deleted)` mmap paths CRIU just captured inline. Future
            # mmap_invalidation checks will skip these — the dump image
            # already has their content frozen, so they cannot break a
            # future restore from that image. For fs-only checkpoints we
            # preserve the existing baseline (the process image is unchanged
            # by an fs-only checkpoint, so its frozen content set is too).
            if captures_process:
                live_pids = set(tracked_pids) | set(ignored_pids)
                if resolved.init_pid:
                    live_pids.add(int(resolved.init_pid))
                acknowledged = (
                    frozenset(all_deleted_mmap_paths(live_pids))
                    if live_pids
                    else frozenset()
                )
            else:
                acknowledged = existing.acknowledged_deleted_mmaps

            updated = replace(
                existing,
                runtime_name=resolved.runtime_name,
                is_running=resolved.is_running,
                init_pid=resolved.init_pid,
                cgroup_path=resolved.cgroup_path,
                cgroup_id=resolved.cgroup_id,
                baseline_pids=baseline_pids,
                current_pids=tracked_pids,
                dirty_pids=set(),
                tracked_pids=tracked_pids,
                ignored_pids=ignored_pids,
                fs_ignored_pids=fs_ignored_pids,
                process_changed=False,
                filesystem_changed=False,
                mmap_invalidated=False,
                mmap_invalidated_path=None,
                acknowledged_deleted_mmaps=acknowledged,
                last_reset_at=when,
                observed_at=max(existing.observed_at or when, when),
                fs_event_count_since_reset=0,
                recent_fs_events=[],
                live_dirty_entries={},
                dirty_by_path={},
                dirty_by_inode={},
                unreconciled_fs_events=[],
                last_error=None,
            )
            with self._records_lock:
                self._records[sandbox_id] = updated
            # Snapshot under sandbox_lock so _handle_fs_event's in-place
            # mutation of the record's containers cannot race with this
            # iteration.
            response = self._record_to_response(updated)
            logger.debug(
                "reset.exit sandbox=%s when=%s filesystem_changed=%s",
                sandbox_id,
                _isoformat(when),
                updated.filesystem_changed,
            )
        if resolved.cgroup_id is not None:
            self._fs_monitor.upsert_sandbox(sandbox_id, resolved.cgroup_id)
        self._sync_bpf_ignored_pids(fs_ignored_pids)
        hold_ms = (time.monotonic() - lock_hold_start) * 1000.0
        if hold_ms > _LOCK_HOLD_WARN_MS:
            logger.warning(
                "sandbox_lock held %.1fms in reset() sandbox=%s resolver_ms=%.1f soft_dirty_ms=%.1f pids=%d",
                hold_ms, sandbox_id, resolve_ms, soft_dirty_ms, len(all_current_pids),
            )
        return response

    def _handle_fs_event(self, event) -> None:
        # Fast path: filter events whose countability can be determined from the
        # event payload alone (syscall, fd, fd_kind, flags, path). These checks
        # do not need the per-sandbox record, so running them before the
        # sandbox_lock avoids serializing uncountable events behind state
        # updates — critical when the BPF ring buffer is delivering thousands
        # of events per second and the Python reader is the pipeline bottleneck
        # gating sync_ack delivery.
        if not self._is_countable_fs_event(event):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "fs_event drop=not_countable sandbox=%s syscall=%s fd=%s fd_kind=%s flags=%s path=%s",
                    event.sandbox_id,
                    getattr(event, "syscall", None),
                    getattr(event, "fd", None),
                    getattr(event, "fd_kind", None),
                    getattr(event, "flags", None),
                    getattr(event, "path", None),
                )
            return

        # Lock-free record peek: dict.get is atomic under the GIL. The record
        # is only mutated here and in register/unregister, so a stale read at
        # worst drops an event for a just-unregistered sandbox. We re-read
        # under the lock before mutating.
        record = self._records.get(event.sandbox_id)
        if record is None:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "fs_event drop=no_record sandbox=%s syscall=%s pid=%s fd=%s fd_kind=%s path=%s",
                    event.sandbox_id,
                    getattr(event, "syscall", None),
                    getattr(event, "pid", None),
                    getattr(event, "fd", None),
                    getattr(event, "fd_kind", None),
                    getattr(event, "path", None),
                )
            return
        # Path-prefix filter: drop events whose target path is on a
        # known host-side path the runtime declared at register time.
        # Catches CRIU/runc transient-cgroup attribution where the PID is
        # too short-lived for the proc-identity-based ignore rule to fire
        # in time. Cheap O(N_prefixes) string check before the more
        # expensive proc-read path.
        if record.ignored_path_prefixes:
            for candidate in (
                getattr(event, "path", None),
                getattr(event, "path_secondary", None),
            ):
                if not candidate:
                    continue
                cstr = str(candidate)
                if any(cstr.startswith(prefix) for prefix in record.ignored_path_prefixes):
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            "fs_event drop=ignored_path_prefix sandbox=%s syscall=%s pid=%s path=%s",
                            event.sandbox_id,
                            getattr(event, "syscall", None),
                            getattr(event, "pid", None),
                            cstr,
                        )
                    return
        if int(event.pid or 0) > 0 and pid_matches_ignore_rules(
            int(event.pid or 0),
            record.ignore_process_rules,
            fs_only=True,
            cache=self._pid_identity_cache,
        ):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "fs_event drop=ignore_rule sandbox=%s syscall=%s pid=%s path=%s",
                    event.sandbox_id,
                    getattr(event, "syscall", None),
                    getattr(event, "pid", None),
                    getattr(event, "path", None),
                )
            return

        now = utc_now()
        event_ts = _parse_ts(event.timestamp) or now
        sandbox_lock = self._sandbox_lock(event.sandbox_id)
        # Measure time the worker spends blocked on sandbox_lock. When
        # status()/reset() hold the lock across a runc-state subprocess,
        # the worker serving this sandbox stalls — directly blowing the
        # fs_monitor.sync barrier budget.
        wait_start = time.monotonic()
        with sandbox_lock:
            wait_ms = (time.monotonic() - wait_start) * 1000.0
            if wait_ms > _LOCK_WAIT_WARN_MS:
                logger.warning(
                    "sandbox_lock wait %.1fms in _handle_fs_event sandbox=%s syscall=%s",
                    wait_ms, event.sandbox_id, getattr(event, "syscall", None),
                )
            with self._records_lock:
                record = self._records.get(event.sandbox_id)
            if record is None:
                return
            # Mutate the record's mutable containers in place under
            # sandbox_lock. The previous implementation deep-copied
            # live_dirty_entries on every event as part of an immutable
            # record-swap contract, but with N=10k dirty entries per
            # sandbox (tar extraction / make-install workloads) that
            # became O(N*M) over an event burst and stalled the
            # per-sandbox event worker for ~5s, blowing the fs_sync
            # barrier. Readers outside sandbox_lock (status/reset/
            # register) now snapshot via _record_to_response before
            # releasing the lock; iterations on the shared containers
            # therefore cannot race with this in-place mutation.
            self._apply_fs_event(event, record, event_ts)
            excess = len(record.unreconciled_fs_events) - _RECENT_EVENT_LIMIT
            if excess > 0:
                del record.unreconciled_fs_events[:excess]
            record.recent_fs_events.append(self._event_to_dict(event))
            excess = len(record.recent_fs_events) - _RECENT_EVENT_LIMIT
            if excess > 0:
                del record.recent_fs_events[:excess]
            record.filesystem_changed = bool(record.live_dirty_entries) or any(
                bool(item.get("counts_as_change")) for item in record.unreconciled_fs_events
            )
            if not record.mmap_invalidated:
                # A write/delete against a path currently mmapped by any live
                # process in the sandbox cgroup invalidates that process's prior
                # CRIU checkpoint (the file's build-ID on disk is now out of
                # sync with the recorded vma). We must check the FULL set
                # CRIU dumps — tracked + ignored pids + init — because the
                # long-lived "ignored" pid (e.g. the container's
                # `sleep infinity` PID 1, the tmux server, or any pane-bash
                # descendant) is exactly the process whose mmap'd libc.so.6
                # apt-get install rewrites under. With tmux ignore rules
                # active, `current_pids` (= tracked) is empty, so falling
                # back to `{init_pid}` alone would miss any deletion seen
                # only via the tmux server / pane bash. Surface as
                # process_changed in status() so the scheduler promotes the
                # next checkpoint to a full one.
                #
                # `acknowledged_deleted_mmaps` is the snapshot captured by
                # the most recent full process checkpoint; CRIU stored the
                # content of those paths inline in the dump image, so they
                # cannot break a future restore from that image — skip them
                # here to keep one libc rewrite from latching every
                # subsequent checkpoint as full forever.
                live_pids = (
                    set(record.current_pids)
                    | set(record.ignored_pids)
                )
                if record.init_pid:
                    live_pids.add(int(record.init_pid))
                if live_pids:
                    for candidate in (
                        getattr(event, "path", None),
                        getattr(event, "path_secondary", None),
                    ):
                        candidate_str = str(candidate or "")
                        if not candidate_str:
                            continue
                        matched = path_invalidates_mmap(live_pids, candidate_str, cache=self._mmap_path_cache)
                        if matched is None:
                            continue
                        # Suppression keys on the MATCHED path — the
                        # canonical file that's actually mmap'd. The event
                        # path itself can be a transient name (e.g.
                        # `.dpkg-new`) that gets renamed onto the real
                        # filename, so checking the candidate would never
                        # hit the baseline. Once a path is acknowledged,
                        # CRIU has its content frozen inline; further
                        # rewrites of the same path don't break restore.
                        if matched in record.acknowledged_deleted_mmaps:
                            continue
                        record.mmap_invalidated = True
                        record.mmap_invalidated_path = matched
                        logger.info(
                            "fs_event mmap_invalidation sandbox=%s syscall=%s path=%s live_pids=%s acknowledged=%d",
                            event.sandbox_id,
                            getattr(event, "syscall", None),
                            matched,
                            sorted(live_pids),
                            len(record.acknowledged_deleted_mmaps),
                        )
                        break
            record.observed_at = max(
                record.observed_at or now,
                _parse_ts(event.timestamp) or now,
            )
            record.fs_event_count_since_reset += 1
            logger.debug(
                "fs_event accept sandbox=%s syscall=%s pid=%s fd=%s fd_kind=%s path=%s path2=%s event_ts=%s "
                "live_dirty=%d unreconciled=%d filesystem_changed=%s",
                event.sandbox_id,
                getattr(event, "syscall", None),
                getattr(event, "pid", None),
                getattr(event, "fd", None),
                getattr(event, "fd_kind", None),
                getattr(event, "path", None),
                getattr(event, "path_secondary", None),
                _isoformat(event_ts),
                len(record.live_dirty_entries),
                len(record.unreconciled_fs_events),
                record.filesystem_changed,
            )

    def _is_countable_fs_event(self, event) -> bool:
        syscall = str(event.syscall)
        if syscall in _OPEN_SYSCALLS:
            if not self._has_mutating_open_flags(int(event.flags or 0)):
                return False
            # When the BPF event carries an absolute path to a non-pseudo
            # filesystem, trust it over `fd_kind`. The helper resolves
            # `fd_kind` by stat()ing `/proc/<pid>/fd/<fd>` after the syscall
            # has returned, which races with dup2/close/exec in the tracee —
            # e.g. `cat > file << 'EOF'` opens the real file, dup2s it onto
            # fd 1, closes the original fd, then sets up a heredoc pipe that
            # reuses the same fd slot. By the time the helper stats the fd,
            # it sees a fifo/socket and reports fd_kind=fifo for a write that
            # actually landed on a regular file. Dropping such events causes
            # real writes to disappear from the dirty-state signal.
            path = str(event.path or "")
            if self._path_is_likely_persistent(path):
                return True
            return event.fd_kind not in _DEVICE_OR_STREAM_FD_KINDS
        if syscall in _WRITE_FD_SYSCALLS or syscall in _METADATA_FD_SYSCALLS:
            fd = int(event.fd or -1)
            if fd < 0:
                return False
            resolved_path = str(event.path or "")
            if resolved_path and not self._path_is_likely_persistent(resolved_path):
                return False
            # Do not filter by fd number (1/2).  Shell redirection like
            # `cat > file` makes fd 1 point to a regular file, not a
            # terminal.  Rely on fd_kind instead — char/fifo/socket fds
            # are excluded below, so true stdout/stderr writes to
            # terminals and pipes are still ignored.
            return str(event.fd_kind or "unknown") in _MUTATING_FD_KINDS
        if syscall in _WRITE_PATH_SYSCALLS | _METADATA_PATH_SYSCALLS | _CREATE_PATH_SYSCALLS | _DELETE_SYSCALLS | _RENAME_SYSCALLS | _SECONDARY_TARGET_SYSCALLS:
            return True
        return bool(event.path or event.path_secondary)

    def _has_mutating_open_flags(self, flags: int) -> bool:
        if flags & (os.O_CREAT | os.O_TRUNC):
            return True
        tmpfile_flags = getattr(os, "O_TMPFILE", 0)
        return bool(tmpfile_flags) and (flags & tmpfile_flags) == tmpfile_flags

    def _path_is_likely_persistent(self, path: str) -> bool:
        """Return True if `path` looks like a real file on a persistent
        filesystem inside the sandbox. Used to bypass racy post-hoc fd_kind
        checks: if we have an absolute path to something that is not a
        pseudo-filesystem entry, mutating opens against it count regardless
        of what `/proc/<pid>/fd/<fd>` reports after the fact.
        """
        if not path or not path.startswith("/"):
            return False
        if path in _NON_PERSISTENT_PATH_EXACT:
            return False
        return not path.startswith(_NON_PERSISTENT_PATH_PREFIXES)

    def _apply_fs_event(
        self,
        event,
        record: SandboxRecord,
        event_ts: datetime | None = None,
    ) -> None:
        syscall = str(event.syscall)
        if syscall in _RENAME_SYSCALLS:
            self._apply_rename_event(event, record, event_ts)
            return
        if syscall in _DELETE_SYSCALLS:
            self._apply_delete_event(event, record, event_ts)
            return
        self._apply_mutation_event(event, record, event_ts)

    def _apply_mutation_event(
        self,
        event,
        record: SandboxRecord,
        event_ts: datetime | None = None,
    ) -> None:
        syscall = str(event.syscall)
        target_path = self._event_target_path(event)
        target_key = self._event_target_key(event, target_path)
        if target_key is None:
            self._append_unreconciled(
                record.unreconciled_fs_events,
                event,
                reason="missing_identity",
                counts_as_change=False,
            )
            return
        existing_key = self._lookup_entry_key(
            record,
            preferred_key=target_key,
            path=target_path,
            device=event.device,
            inode=event.inode,
        )
        created_after_reset = self._event_creates_new_entry(event)
        entry = (
            self._dirty_pop(record, existing_key)
            if existing_key is not None
            else self._new_dirty_entry(
                key=target_key,
                path=target_path,
                device=event.device,
                inode=event.inode,
                syscall=syscall,
                created_after_reset=created_after_reset,
            )
        )
        entry["path"] = target_path
        entry["device"] = event.device
        entry["inode"] = event.inode
        entry["last_syscall"] = syscall
        entry["event_count"] = int(entry.get("event_count", 0)) + 1
        entry["created_after_reset"] = bool(entry.get("created_after_reset")) or created_after_reset
        if event_ts is not None:
            entry["observed_at"] = _isoformat(event_ts)
        self._dirty_put(record, target_key, entry)

    def _apply_delete_event(
        self,
        event,
        record: SandboxRecord,
        event_ts: datetime | None = None,
    ) -> None:
        delete_path = self._event_target_path(event)
        delete_key = self._event_target_key(event, delete_path)
        existing_key = self._lookup_entry_key(
            record,
            preferred_key=delete_key,
            path=delete_path,
            device=event.device,
            inode=event.inode,
        )
        if existing_key is not None:
            entry = self._dirty_pop(record, existing_key)
            if entry is None:
                return
            if bool(entry.get("created_after_reset")):
                return
            entry["path"] = delete_path
            entry["deleted"] = True
            entry["last_syscall"] = str(event.syscall)
            entry["event_count"] = int(entry.get("event_count", 0)) + 1
            if event_ts is not None:
                entry["observed_at"] = _isoformat(event_ts)
            self._dirty_put(record, existing_key, entry)
            return
        if delete_key is None:
            self._append_unreconciled(
                record.unreconciled_fs_events,
                event,
                reason="delete_without_identity",
                counts_as_change=True,
            )
            return
        new_entry = self._new_dirty_entry(
            key=delete_key,
            path=delete_path,
            device=event.device,
            inode=event.inode,
            syscall=str(event.syscall),
            created_after_reset=False,
            deleted=True,
        )
        if event_ts is not None:
            new_entry["observed_at"] = _isoformat(event_ts)
        self._dirty_put(record, delete_key, new_entry)

    def _apply_rename_event(
        self,
        event,
        record: SandboxRecord,
        event_ts: datetime | None = None,
    ) -> None:
        old_path = self._clean_path(event.path)
        new_path = self._clean_path(event.path_secondary)
        new_key = self._event_secondary_key(event, new_path)
        existing_key = self._lookup_entry_key(
            record,
            preferred_key=self._path_key(old_path),
            path=old_path,
            device=None,
            inode=None,
        )
        if existing_key is None:
            existing_key = self._lookup_entry_key(
                record,
                preferred_key=new_key,
                path=new_path,
                device=event.device,
                inode=event.inode,
            )
        if existing_key is None and new_key is None:
            self._append_unreconciled(
                record.unreconciled_fs_events,
                event,
                reason="rename_without_identity",
                counts_as_change=True,
            )
            return
        if existing_key is None:
            new_entry = self._new_dirty_entry(
                key=new_key,
                path=new_path,
                device=event.device,
                inode=event.inode,
                syscall=str(event.syscall),
                created_after_reset=False,
            )
            new_entry["original_path"] = old_path
            if event_ts is not None:
                new_entry["observed_at"] = _isoformat(event_ts)
            self._dirty_put(record, new_key, new_entry)
            return
        entry = self._dirty_pop(record, existing_key)
        if entry is None:
            return
        entry["path"] = new_path or old_path
        entry["device"] = event.device
        entry["inode"] = event.inode
        entry["last_syscall"] = str(event.syscall)
        entry["event_count"] = int(entry.get("event_count", 0)) + 1
        entry.setdefault("original_path", old_path)
        if event_ts is not None:
            entry["observed_at"] = _isoformat(event_ts)
        self._dirty_put(record, new_key or existing_key, entry)

    def _event_to_dict(self, event) -> dict[str, object]:
        return {
            "syscall": str(event.syscall),
            "pid": None if event.pid is None else int(event.pid),
            "fd": None if event.fd is None else int(event.fd),
            "fd_kind": None if event.fd_kind is None else str(event.fd_kind),
            "flags": None if event.flags is None else int(event.flags),
            "path": self._clean_path(event.path),
            "path_secondary": self._clean_path(event.path_secondary),
            "inode": None if event.inode is None else int(event.inode),
            "device": None if event.device is None else int(event.device),
        }

    def _append_unreconciled(
        self,
        unreconciled_fs_events: list[dict[str, object]],
        event,
        *,
        reason: str,
        counts_as_change: bool,
    ) -> None:
        item = self._event_to_dict(event)
        item["reason"] = reason
        item["counts_as_change"] = counts_as_change
        unreconciled_fs_events.append(item)

    def _event_target_path(self, event) -> str | None:
        syscall = str(event.syscall)
        if syscall in _SECONDARY_TARGET_SYSCALLS:
            return self._clean_path(event.path_secondary)
        return self._clean_path(event.path)

    def _event_target_key(self, event, target_path: str | None) -> str | None:
        syscall = str(event.syscall)
        if syscall in _SECONDARY_TARGET_SYSCALLS:
            return self._identity_key(event.device, event.inode, target_path)
        return self._identity_key(event.device, event.inode, target_path)

    def _event_secondary_key(self, event, target_path: str | None) -> str | None:
        return self._identity_key(event.device, event.inode, target_path)

    def _event_creates_new_entry(self, event) -> bool:
        syscall = str(event.syscall)
        if syscall in _CREATE_PATH_SYSCALLS | _SECONDARY_TARGET_SYSCALLS:
            return True
        if syscall in _OPEN_SYSCALLS:
            flags = int(event.flags or 0)
            if flags & os.O_CREAT:
                return True
            tmpfile_flags = getattr(os, "O_TMPFILE", 0)
            return bool(tmpfile_flags) and (flags & tmpfile_flags) == tmpfile_flags
        return False

    def _new_dirty_entry(
        self,
        *,
        key: str,
        path: str | None,
        device: int | None,
        inode: int | None,
        syscall: str,
        created_after_reset: bool,
        deleted: bool = False,
    ) -> dict[str, object]:
        return {
            "key": key,
            "path": path,
            "device": device,
            "inode": inode,
            "created_after_reset": created_after_reset,
            "deleted": deleted,
            "first_syscall": syscall,
            "last_syscall": syscall,
            "event_count": 1,
        }

    def _lookup_entry_key(
        self,
        record: SandboxRecord,
        *,
        preferred_key: str | None,
        path: str | None,
        device: int | None,
        inode: int | None,
    ) -> str | None:
        # O(1) via dirty_by_path / dirty_by_inode reverse indexes. These
        # are maintained by _dirty_put / _dirty_pop; the previous
        # implementation scanned live_dirty_entries.items() on every
        # event, which was the dominant remaining cost on bursts of new
        # file creations (tar/make install) where the preferred_key
        # systematically misses.
        entries = record.live_dirty_entries
        if preferred_key is not None and preferred_key in entries:
            return preferred_key
        if device is not None and inode is not None:
            indexed = record.dirty_by_inode.get((int(device), int(inode)))
            if indexed is not None and indexed in entries:
                return indexed
        if path:
            indexed = record.dirty_by_path.get(path)
            if indexed is not None and indexed in entries:
                return indexed
        return None

    def _dirty_put(
        self,
        record: SandboxRecord,
        key: str,
        entry: dict[str, object],
    ) -> None:
        """Insert (or replace) `entry` under `key` and update the reverse
        indexes. Callers must hold sandbox_lock.
        """
        record.live_dirty_entries[key] = entry
        path = entry.get("path")
        if isinstance(path, str) and path:
            record.dirty_by_path[path] = key
        device = entry.get("device")
        inode = entry.get("inode")
        if device is not None and inode is not None:
            record.dirty_by_inode[(int(device), int(inode))] = key

    def _dirty_pop(
        self,
        record: SandboxRecord,
        key: str,
    ) -> dict[str, object] | None:
        """Remove the entry at `key` and strip any reverse-index rows that
        still point at it. Returns the removed entry (or None if `key`
        was missing). Callers must hold sandbox_lock.
        """
        entry = record.live_dirty_entries.pop(key, None)
        if entry is None:
            return None
        path = entry.get("path")
        if isinstance(path, str) and path:
            if record.dirty_by_path.get(path) == key:
                record.dirty_by_path.pop(path, None)
        device = entry.get("device")
        inode = entry.get("inode")
        if device is not None and inode is not None:
            index_key = (int(device), int(inode))
            if record.dirty_by_inode.get(index_key) == key:
                record.dirty_by_inode.pop(index_key, None)
        return entry

    def _identity_key(self, device: int | None, inode: int | None, path: str | None) -> str | None:
        if device is not None and inode is not None:
            return f"ino:{int(device)}:{int(inode)}"
        return self._path_key(path)

    def _path_key(self, path: str | None) -> str | None:
        cleaned = self._clean_path(path)
        if not cleaned:
            return None
        return f"path:{cleaned}"

    def _clean_path(self, path: str | None) -> str | None:
        if path is None:
            return None
        cleaned = str(path).strip()
        return cleaned or None

    def _reconcile_live_dirty_entries(
        self,
        live_dirty_entries: dict[str, dict[str, object]],
        init_pid: int | None,
    ) -> dict[str, dict[str, object]]:
        # Historically this step did an `os.lstat` against
        # `/proc/<init_pid>/root/<path>` and silently dropped any
        # `created_after_reset` entry whose lstat failed. That turned out to
        # lose real signal: kernel tracepoints captured the write, but the host
        # side then cancelled it because the path string was relative (C helper
        # fallback), the init_pid was stale, /proc was briefly unavailable, or
        # the file was an ephemeral artefact whose delete event also reached us
        # but via an unrelated code path. Kernel events are authoritative for
        # "did the sandbox mutate its filesystem?" — do not second-guess them
        # here. We still return a fresh dict so callers can mutate freely.
        _ = init_pid
        reconciled: dict[str, dict[str, object]] = {}
        for key, value in live_dirty_entries.items():
            reconciled[key] = dict(value)
        return reconciled

    def _sandbox_path_for_stat(self, path: str, init_pid: int | None) -> str:
        if init_pid is None or not path.startswith("/"):
            return path
        return f"/proc/{int(init_pid)}/root{path}"

    def _split_pids(
        self,
        pids: set[int],
        ignore_process_rules,
    ) -> tuple[set[int], set[int], set[int]]:
        """Split `pids` into (tracked, ignored, fs_ignored).

        - `tracked`: pids that contribute to the process-changed signal.
        - `ignored`: pids removed from process tracking (union of scope=all
          and scope=process_only matches).
        - `fs_ignored`: subset of `ignored` whose fs events should also be
          dropped. This is the set we sync to the eBPF kernel filter and
          consult in `_handle_fs_event`. scope=process_only rules
          deliberately do NOT contribute here, so file writes from a
          process-only-ignored PID still surface as filesystem changes.
        """
        tracked: set[int] = set()
        ignored: set[int] = set()
        fs_ignored: set[int] = set()
        if not ignore_process_rules:
            tracked = set(pids)
            return tracked, ignored, fs_ignored
        for pid in sorted(pids):
            matched_any, matched_fs = classify_pid_against_rules(
                pid, ignore_process_rules, cache=self._pid_identity_cache,
            )
            if matched_any:
                ignored.add(pid)
                if matched_fs:
                    fs_ignored.add(pid)
            else:
                tracked.add(pid)
        return tracked, ignored, fs_ignored

    def _record_to_response(self, record: SandboxRecord) -> dict[str, object]:
        return {
            "sandbox_id": record.sandbox_id,
            "runtime_name": record.runtime_name,
            "is_running": record.is_running,
            # "process_changed": record.is_running, # record.process_changed,
            # "filesystem_changed": record.is_running, # record.filesystem_changed,
            "process_changed": record.process_changed,
            "filesystem_changed": record.filesystem_changed,
            "observed_at": _isoformat(record.observed_at),
            "last_reset_at": _isoformat(record.last_reset_at),
            "metadata": record.metadata(),
        }


class HostInspectorServer:
    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        daemon: HostInspectorDaemon | None = None,
        max_workers: int | None = None,
    ) -> None:
        self._daemon = daemon or HostInspectorDaemon()
        self._server = PooledHTTPServer((host, port), self._build_handler(), max_workers=max_workers)
        self._thread: Thread | None = None

    @property
    def port(self) -> int:
        return int(self._server.server_address[1])

    def start(self) -> None:
        self._daemon.start()
        self._thread = Thread(target=self._server.serve_forever, name="host-inspector-http", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        self._daemon.stop()

    def _build_handler(self):
        daemon = self._daemon

        class _ClientDisconnectedError(Exception):
            pass

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def end_headers(self) -> None:
                self.send_header("Connection", "close")
                self.close_connection = True
                super().end_headers()

            def _read_json(self) -> dict[str, Any]:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0:
                    return {}
                payload = self.rfile.read(length)
                result = _JSON_CODEC.loads(payload)
                if not isinstance(result, dict):
                    raise ValueError("request body must decode to a JSON object")
                return dict(result)

            def _write_json(self, code: int, payload: dict[str, Any]) -> None:
                body = _JSON_CODEC.dumps_bytes(payload)
                try:
                    self.send_response(code)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError) as exc:
                    raise _ClientDisconnectedError from exc

            def do_GET(self) -> None:  # noqa: N802
                try:
                    if self.path == "/healthz":
                        self._write_json(HTTPStatus.OK, {"ok": True})
                        return
                    self._write_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})
                except _ClientDisconnectedError:
                    logger.debug("Host inspector client disconnected during GET response")

            def do_POST(self) -> None:  # noqa: N802
                try:
                    if self.path == "/register":
                        payload = self._read_json()
                        result = daemon.register(
                            sandbox_id=str(payload["sandbox_id"]),
                            runtime=str(payload["runtime"]),
                            object_id=str(payload["object_id"]),
                            ignore_process_rules=payload.get("ignore_process_rules"),
                            ignored_path_prefixes=payload.get("ignored_path_prefixes"),
                        )
                        logger.debug(f"host-inspector: register {payload=} {result=}")
                        self._write_json(HTTPStatus.OK, {"ok": True, "status": result})
                        return
                    if self.path == "/get_proc_and_fs_status":
                        payload = self._read_json()
                        result = daemon.status(str(payload["sandbox_id"]))
                        def without_key(d, key):
                            return {k: v for k, v in d.items() if k != key}
                        logger.debug(f"host-inspector: get_proc_and_fs_status {payload=} result={without_key(result, 'metadata')}")
                        self._write_json(HTTPStatus.OK, {"ok": True, "status": result})
                        return
                    if self.path == "/reset":
                        payload = self._read_json()
                        result = daemon.reset(
                            str(payload["sandbox_id"]),
                            at=_parse_ts(payload.get("at")),
                            captures_process=bool(payload.get("captures_process", False)),
                        )
                        logger.debug(f"host-inspector: reset {payload=} {result=}")
                        self._write_json(HTTPStatus.OK, {"ok": True, "status": result})
                        return
                    if self.path == "/mark_filesystem_changed":
                        payload = self._read_json()
                        result = daemon.mark_filesystem_changed(
                            str(payload["sandbox_id"]),
                            reason=str(payload.get("reason") or "external"),
                        )
                        self._write_json(
                            HTTPStatus.OK, {"ok": True, "status": result}
                        )
                        return
                    if self.path == "/unregister":
                        payload = self._read_json()
                        result = daemon.unregister(str(payload["sandbox_id"]))
                        logger.debug(f"host-inspector: unregister {payload=} {result=}")
                        self._write_json(HTTPStatus.OK, {"ok": True, **result})
                        return
                    if self.path == "/update_filters":
                        payload = self._read_json()
                        result = daemon.update_filters(
                            sandbox_id=str(payload["sandbox_id"]),
                            ignore_process_rules=payload.get("ignore_process_rules"),
                            ignored_path_prefixes=payload.get("ignored_path_prefixes"),
                        )
                        logger.debug(f"host-inspector: update_filters {payload=} {result=}")
                        self._write_json(HTTPStatus.OK, {"ok": True, "status": result})
                        return
                    self._write_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})
                except KeyError as exc:
                    self._write_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": f"unknown sandbox: {exc.args[0]}"})
                except _ClientDisconnectedError:
                    logger.debug("Host inspector client disconnected during POST response")
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Host inspector request failed")
                    try:
                        self._write_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "error": str(exc)})
                    except _ClientDisconnectedError:
                        logger.debug("Host inspector client disconnected while sending POST error response")

            def log_message(self, fmt: str, *args) -> None:
                logger.debug("host-inspector: " + fmt, *args)

        return Handler


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="crab host inspector server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9782)
    parser.add_argument("--process-poll-interval", type=float, default=1.0)
    parser.add_argument("--helper-path", default=None)
    parser.add_argument("--runc-state-root", default=None)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--max-workers", type=int, default=32)
    args = parser.parse_args(argv)

    level = getattr(logging, args.log_level.upper(), logging.INFO)
    handlers: list[logging.Handler] = []
    if args.log_file:
        handlers.append(logging.FileHandler(args.log_file, mode="a"))
    else:
        handlers.append(logging.StreamHandler())
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=handlers,
        force=True,
    )
    daemon = HostInspectorDaemon(
        resolver=RuntimeResolver(runc_state_root=args.runc_state_root),
        fs_monitor=LibbpfFilesystemMonitor(helper_path=args.helper_path),
        process_poll_interval_s=args.process_poll_interval,
    )
    server = HostInspectorServer(
        host=args.host,
        port=args.port,
        daemon=daemon,
        max_workers=args.max_workers,
    )
    server.start()
    logger.info("host inspector listening on %s:%s", args.host, server.port)
    stop = Event()

    try:
        while not stop.wait(1.0):
            continue
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
