from __future__ import annotations

import os
import struct
import time
from collections import OrderedDict
from collections.abc import Iterable
from pathlib import Path
from threading import Lock


PAGEMAP_ENTRY_SIZE = 8
SOFT_DIRTY_BIT = 55
PAGE_SIZE = os.sysconf("SC_PAGE_SIZE")


def list_cgroup_pids(cgroup_path: str | None) -> set[int]:
    if not cgroup_path:
        return set()
    path = Path("/sys/fs/cgroup") / cgroup_path.lstrip("/") / "cgroup.procs"
    if not path.exists():
        return set()
    return {int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def clear_soft_dirty(pid: int) -> None:
    Path(f"/proc/{pid}/clear_refs").write_text("4\n", encoding="utf-8")


def reset_soft_dirty_for_pids(
    pids: set[int],
    *,
    stabilize: bool = False,
    stabilize_attempts: int = 3,
    stabilize_delay: float = 0.05,
) -> set[int]:
    cleared: set[int] = set()
    for pid in sorted(pids):
        try:
            clear_soft_dirty(pid)
            cleared.add(pid)
        except FileNotFoundError:
            continue
    if stabilize and cleared:
        _stabilize_soft_dirty(cleared, attempts=stabilize_attempts, delay=stabilize_delay)
    return cleared


def _stabilize_soft_dirty(pids: set[int], *, attempts: int, delay: float) -> None:
    """Re-clear soft-dirty until the writable set reads clean or attempts run out.

    A checkpoint dumps the tasks with CRIU --leave-running while the container
    is paused; the parasite teardown on the subsequent *resume* dirties 1-3
    residual pages -- and it does so just after the baseline clear. Those pages
    are a one-time artifact of the checkpoint, not workload activity, so a
    follow-up clear settles them permanently.

    We sleep BEFORE each scan so a residual page that lands slightly after the
    baseline clear is still observed: a scan-first loop could read clean before
    the resume writes arrive and stop too early, leaving the residue latched.

    A genuinely busy process re-dirties pages continuously: it still reads dirty
    after every re-clear and simply exhausts `attempts`, after which we leave it
    alone. The next status() poll re-scans and reports it dirty, so real memory
    writes are never masked -- this loop only removes checkpoint residue.
    """
    for _ in range(attempts):
        if delay > 0:
            time.sleep(delay)
        still = dirty_pids(pids)
        if not still:
            return
        for pid in sorted(still):
            try:
                clear_soft_dirty(pid)
            except FileNotFoundError:
                pids.discard(pid)


def parse_writable_ranges(pid: int) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    with open(f"/proc/{pid}/maps", "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 2:
                continue
            if "w" not in parts[1]:
                continue
            start_hex, end_hex = parts[0].split("-", 1)
            start = int(start_hex, 16)
            end = int(end_hex, 16)
            if end > start:
                ranges.append((start, end))
    return ranges


def range_has_soft_dirty(pid: int, start: int, end: int) -> bool:
    start_page = start // PAGE_SIZE
    end_page = (end + PAGE_SIZE - 1) // PAGE_SIZE
    remaining = end_page - start_page
    if remaining <= 0:
        return False

    chunk_pages = 4096
    with open(f"/proc/{pid}/pagemap", "rb", buffering=0) as handle:
        current_page = start_page
        while remaining > 0:
            pages_now = min(remaining, chunk_pages)
            handle.seek(current_page * PAGEMAP_ENTRY_SIZE)
            data = handle.read(pages_now * PAGEMAP_ENTRY_SIZE)
            if len(data) != pages_now * PAGEMAP_ENTRY_SIZE:
                return False
            for offset in range(0, len(data), PAGEMAP_ENTRY_SIZE):
                entry = struct.unpack_from("Q", data, offset)[0]
                if (entry >> SOFT_DIRTY_BIT) & 1:
                    return True
            current_page += pages_now
            remaining -= pages_now
    return False


def pid_has_soft_dirty(pid: int) -> bool:
    try:
        for start, end in parse_writable_ranges(pid):
            if range_has_soft_dirty(pid, start, end):
                return True
    except FileNotFoundError:
        return False
    return False


def dirty_pids(pids: set[int]) -> set[int]:
    dirty: set[int] = set()
    for pid in sorted(pids):
        if pid_has_soft_dirty(pid):
            dirty.add(pid)
    return dirty


_MMAP_DELETED_SUFFIX = " (deleted)"


def parse_mapped_paths(pid: int) -> set[str]:
    """File paths currently mmapped by `pid`.

    The path is what the kernel resolves in the process's mount namespace, so
    inside a container it is the in-container absolute path. Anonymous and
    pseudo mappings (`[heap]`, `[stack]`, `[vdso]`, etc.) are skipped. If the
    file backing a mapping has been unlinked the kernel suffixes the path with
    " (deleted)"; the suffix is stripped so callers can compare against the
    canonical path the eBPF FS event reports.
    """
    paths: set[str] = set()
    try:
        with open(f"/proc/{pid}/maps", "r", encoding="utf-8") as handle:
            for line in handle:
                # Format: addr perms offset dev inode pathname
                parts = line.split(maxsplit=5)
                if len(parts) < 6:
                    continue
                pathname = parts[5].rstrip("\n")
                if not pathname or pathname.startswith("["):
                    continue
                if pathname.endswith(_MMAP_DELETED_SUFFIX):
                    pathname = pathname[: -len(_MMAP_DELETED_SUFFIX)]
                paths.add(pathname)
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return paths
    return paths


class MmapPathCache:
    """TTL cache for `parse_mapped_paths(pid)` results.

    The `_handle_fs_event` hot path calls `path_invalidates_mmap` per
    fs event, which iterates over every live pid in the sandbox cgroup
    and does an open + read + parse of `/proc/<pid>/maps` per pid. At
    54-sandbox scale on tmux pane workloads, the helper sees ~30k
    events/sec and each sandbox has multiple live pids, so this fan-
    out was the dominant per-worker cost — observed in benchmark
    20260429_063012 where 25% of fs_sync_timeouts fired with
    target_depth < 5K events queued, i.e. the worker was stuck inside
    a single event's mmap scan.

    Caching mapped-paths per pid for ~1s lets a burst of events
    against the same path reuse the same /proc/<pid>/maps read
    instead of re-issuing it per event. The kernel-truth fallback in
    `HostInspectorDaemon.status()` uses `all_deleted_mmap_paths`
    UNCACHED so a stale event-path miss is recovered on the next
    status() call — the cache only accelerates the soft event-driven
    detection path; correctness is preserved by the fresh fallback.
    """

    def __init__(self, *, ttl_s: float = 1.0, max_entries: int = 4096) -> None:
        self._ttl_s = float(ttl_s)
        self._max = int(max_entries)
        self._lock = Lock()
        self._cache: OrderedDict[int, tuple[float, frozenset[str]]] = OrderedDict()

    def get(self, pid: int) -> frozenset[str]:
        if pid <= 0:
            return frozenset()
        now = time.monotonic()
        with self._lock:
            entry = self._cache.get(pid)
            if entry is not None:
                deadline, paths = entry
                if deadline > now:
                    self._cache.move_to_end(pid)
                    return paths
                del self._cache[pid]
        # Miss / expired: read uncached. Note the read happens
        # OUTSIDE the lock so concurrent gets for OTHER pids don't
        # serialize behind a slow /proc read.
        paths = frozenset(parse_mapped_paths(pid))
        with self._lock:
            self._cache[pid] = (now + self._ttl_s, paths)
            self._cache.move_to_end(pid)
            while len(self._cache) > self._max:
                self._cache.popitem(last=False)
        return paths

    def invalidate(self, pid: int) -> None:
        with self._lock:
            self._cache.pop(pid, None)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


def path_invalidates_mmap(
    pids: Iterable[int], path: str, *, cache: MmapPathCache | None = None
) -> str | None:
    """Return the matched pid path if `path` is mmapped by any pid in `pids`.

    A write or unlink against a path that is currently mmap-backed by a live
    process invalidates any prior CRIU process checkpoint of that process,
    because CRIU records the mapping by file path + build-ID and refuses to
    restore when the on-disk file no longer matches.

    Pass `cache` (a `MmapPathCache`) on the per-event hot path to avoid
    re-reading `/proc/<pid>/maps` for every event. Without a cache, this
    becomes the dominant per-worker cost on burst-y fs workloads.
    """
    if not path:
        return None
    for pid in pids:
        if pid <= 0:
            continue
        mapped = cache.get(pid) if cache is not None else parse_mapped_paths(pid)
        if path in mapped:
            return path
    return None


def all_deleted_mmap_paths(pids: Iterable[int]) -> set[str]:
    """Return every distinct ` (deleted)`-suffixed file mmap path across `pids`.

    The kernel suffixes a /proc/{pid}/maps entry with ` (deleted)` once the
    backing file's last directory entry is unlinked — exactly what dpkg /
    apt-get / atomic-rename installers do when they replace shared libraries
    like `libc.so.6` underneath a running process. This is the same condition
    that breaks CRIU process restore (build-ID mismatch on the current on-disk
    file), so callers can scan here for a kernel-truth signal even when the
    eBPF-event-driven detection missed the write (e.g. ringbuf overflow under
    load).

    Also used by the daemon to snapshot the set of unlinked-but-mmap'd files
    that a full CRIU process checkpoint just captured. CRIU stores the content
    of such mappings inline in the dump image, so any subsequent restore from a
    checkpoint that includes that process image won't need the on-disk file to
    match — its build-ID can drift freely. The daemon uses this baseline to
    suppress repeat mmap_invalidation signals for the same path so that one
    rewrite doesn't force every subsequent checkpoint to be full forever.
    Anonymous and pseudo mappings are skipped.
    """
    paths: set[str] = set()
    for pid in pids:
        if pid <= 0:
            continue
        try:
            with open(f"/proc/{pid}/maps", "r", encoding="utf-8") as handle:
                for line in handle:
                    parts = line.split(maxsplit=5)
                    if len(parts) < 6:
                        continue
                    pathname = parts[5].rstrip("\n")
                    if not pathname or pathname.startswith("["):
                        continue
                    if pathname.endswith(_MMAP_DELETED_SUFFIX):
                        paths.add(pathname[: -len(_MMAP_DELETED_SUFFIX)])
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return paths
