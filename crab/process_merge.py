"""Process-half of consolidation (roadmap C4).

Strategy constants, the container-side process census, and the journal
replay engine live here; ``CrabSystem.merge_processes`` orchestrates
(the promotion path, PR-C4.2, rides B3's swap machinery there).

Replay consumes the fork's raw journal exec records — ground truth in
origin order (C3's adopted ``observation`` rows are a reading surface,
not a replay source). v1 ships no nondeterminism classifier: each
replayed exec is diffed against the journal's recorded outcome
(returncode + stdout sha256) and deviations are reported honestly; D2
record/replay is the roadmap's answer for network-dependent commands.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Callable, Sequence

from .models import ReplayEntry

logger = logging.getLogger(__name__)

PROCESS_MERGE_STRATEGIES: tuple[str, ...] = ("auto", "replay", "promote")

# Container-side process census. Pure shell builtins (glob + positional
# count) so the probe spawns no children of its own — a pipeline like
# `ls | wc -l` shows up as three extra PIDs and races its own teardown.
# The probe shell itself counts, so the quiescent baseline is 2
# (container init + probe): anything above it means background
# processes are present.
PROCESS_PROBE_ARGV: tuple[str, ...] = ("sh", "-c", "set -- /proc/[0-9]*; echo $#")
PROCESS_PROBE_BASELINE = 2


class ProcessMergeConflict(RuntimeError):
    """Process merge refused: promoting would kill the source's live
    background processes (retry with ``force=True`` to accept that
    loss), or the promotion's reverse fs apply hit conflicts under
    ``fail_fast`` (PR-C4.2)."""


def replay_fork_execs(
    exec_fn: Callable[..., object],
    records: Sequence,
    *,
    stop_on_deviation: bool = False,
) -> tuple[list[ReplayEntry], bool]:
    """Re-run the fork's journal exec ``records`` through ``exec_fn``
    (already bound to the source sandbox), verbatim: argv, cwd, env,
    user and timeout replay exactly as recorded. Output is always
    captured (the diff needs it, even for records that originally ran
    with ``capture_output=False`` — for those the journal carries no
    digest and ``stdout_matched`` stays None).

    A deviation is a returncode mismatch, or a stdout sha256 mismatch
    where the original captured output. Returns
    ``(entries, stopped_early)``.
    """
    entries: list[ReplayEntry] = []
    stopped_early = False
    for record in records:
        payload = record.payload
        argv = [str(item) for item in (payload.get("argv") or [])]
        result = exec_fn(
            argv,
            cwd=payload.get("cwd"),
            env=payload.get("env") or None,
            user=payload.get("user"),
            timeout_s=payload.get("timeout_s"),
            capture_output=True,
        )
        returncode = getattr(result, "returncode", None)
        expected_returncode = payload.get("returncode")
        expected_sha = payload.get("stdout_sha256")
        stdout_matched: bool | None = None
        if expected_sha is not None:
            stdout = getattr(result, "stdout", "") or ""
            digest = hashlib.sha256(stdout.encode("utf-8", errors="replace")).hexdigest()
            stdout_matched = digest == expected_sha
        deviated = returncode != expected_returncode or stdout_matched is False
        entries.append(
            ReplayEntry(
                origin_seq=record.seq,
                argv=tuple(argv),
                returncode=None if returncode is None else int(returncode),
                expected_returncode=(
                    None if expected_returncode is None else int(expected_returncode)
                ),
                stdout_matched=stdout_matched,
                deviated=deviated,
            )
        )
        if deviated and stop_on_deviation:
            stopped_early = True
            logger.info(
                "Replay stopped at first deviation seq=%s argv=%s", record.seq, argv
            )
            break
    return entries, stopped_early


__all__ = [
    "PROCESS_MERGE_STRATEGIES",
    "PROCESS_PROBE_ARGV",
    "PROCESS_PROBE_BASELINE",
    "ProcessMergeConflict",
    "replay_fork_execs",
]
