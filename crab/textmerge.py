"""Line-based three-way text merge for the C2 fs merge engine.

Stdlib only: composes two two-way diffs (base→ours, base→theirs) from
``difflib.SequenceMatcher`` into one merged text. Hunks whose base line
ranges do not overlap compose; both sides making the *identical* change
coalesces; overlapping differing hunks make the merge unresolvable
(``None``) — no conflict markers are ever written into content.

Properties (pinned by tests/test_textmerge.py):
    merge3(b, b, x) == x
    merge3(b, x, b) == x
    merge3(b, x, x) == x
"""

from __future__ import annotations

import difflib

__all__ = ["merge3"]

# A hunk is (base_start, base_end, replacement_lines, side).
_Hunk = tuple[int, int, tuple[str, ...], int]


def _changed_hunks(base_lines: list[str], other_lines: list[str], side: int) -> list[_Hunk]:
    matcher = difflib.SequenceMatcher(a=base_lines, b=other_lines, autojunk=False)
    hunks: list[_Hunk] = []
    for tag, a0, a1, b0, b1 in matcher.get_opcodes():
        if tag != "equal":
            hunks.append((a0, a1, tuple(other_lines[b0:b1]), side))
    return hunks


def _overlaps(a0: int, a1: int, b0: int, b1: int) -> bool:
    if a0 < b1 and b0 < a1:
        return True
    # Insertions occupy an empty base range; two insertions at the same
    # point cannot be ordered and must count as overlapping.
    return a0 == a1 == b0 == b1


def merge3(base: str, ours: str, theirs: str) -> str | None:
    """Merge ``ours`` and ``theirs`` against their common ``base``.

    Returns the merged text, or ``None`` when changes overlap and the
    merge is unresolvable at line granularity.
    """
    if ours == theirs:
        return ours
    base_lines = base.splitlines(keepends=True)
    hunks = sorted(
        _changed_hunks(base_lines, ours.splitlines(keepends=True), 0)
        + _changed_hunks(base_lines, theirs.splitlines(keepends=True), 1),
        key=lambda hunk: (hunk[0], hunk[1], hunk[3]),
    )

    merged: list[str] = []
    cursor = 0
    index = 0
    while index < len(hunks):
        # Group transitively-overlapping hunks. Hunks are sorted by
        # base start, so tracking the group's range is sufficient.
        group = [hunks[index]]
        group_start, group_end = hunks[index][0], hunks[index][1]
        scan = index + 1
        while scan < len(hunks) and _overlaps(group_start, group_end, hunks[scan][0], hunks[scan][1]):
            group.append(hunks[scan])
            group_end = max(group_end, hunks[scan][1])
            scan += 1
        index = scan

        if len(group) == 1:
            start, end, replacement, _side = group[0]
        elif len(group) == 2 and group[0][:3] == group[1][:3]:
            # Both sides made the identical change — take it once.
            start, end, replacement, _side = group[0]
        else:
            return None

        merged.extend(base_lines[cursor:start])
        merged.extend(replacement)
        cursor = end
    merged.extend(base_lines[cursor:])
    return "".join(merged)
