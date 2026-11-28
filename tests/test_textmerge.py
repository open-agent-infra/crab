"""Unit vector suite for the in-repo line-based diff3 (crab/textmerge.py).

Pins compose/coalesce/conflict semantics and the three identity
properties the C2 design guarantees. Host-runnable."""
from __future__ import annotations

import unittest

from crab.textmerge import merge3

BASE = "one\ntwo\nthree\nfour\nfive\n"


class Merge3Tests(unittest.TestCase):
    def test_identity_properties(self) -> None:
        changed = "one\nTWO\nthree\nfour\nfive\n"
        self.assertEqual(merge3(BASE, BASE, changed), changed)
        self.assertEqual(merge3(BASE, changed, BASE), changed)
        self.assertEqual(merge3(BASE, changed, changed), changed)
        self.assertEqual(merge3(BASE, BASE, BASE), BASE)

    def test_disjoint_edits_compose(self) -> None:
        ours = "ONE\ntwo\nthree\nfour\nfive\n"
        theirs = "one\ntwo\nthree\nfour\nFIVE\n"
        self.assertEqual(merge3(BASE, ours, theirs), "ONE\ntwo\nthree\nfour\nFIVE\n")

    def test_adjacent_hunks_compose(self) -> None:
        ours = "one\nTWO\nthree\nfour\nfive\n"
        theirs = "one\ntwo\nTHREE\nfour\nfive\n"
        self.assertEqual(merge3(BASE, ours, theirs), "one\nTWO\nTHREE\nfour\nfive\n")

    def test_insertion_composes_with_distant_edit(self) -> None:
        ours = "zero\none\ntwo\nthree\nfour\nfive\n"
        theirs = "one\ntwo\nthree\nfour\nFIVE\n"
        self.assertEqual(merge3(BASE, ours, theirs), "zero\none\ntwo\nthree\nfour\nFIVE\n")

    def test_identical_change_coalesces_next_to_unique_edit(self) -> None:
        ours = "one\nTWO\nthree\nfour\nfive\n"
        theirs = "one\nTWO\nthree\nfour\nFIVE\n"
        self.assertEqual(merge3(BASE, ours, theirs), "one\nTWO\nthree\nfour\nFIVE\n")

    def test_overlapping_different_edits_conflict(self) -> None:
        ours = "one\nTWO\nthree\nfour\nfive\n"
        theirs = "one\nDOS\nthree\nfour\nfive\n"
        self.assertIsNone(merge3(BASE, ours, theirs))

    def test_same_point_insertions_conflict(self) -> None:
        ours = "one\ntwo\nA\nthree\nfour\nfive\n"
        theirs = "one\ntwo\nB\nthree\nfour\nfive\n"
        self.assertIsNone(merge3(BASE, ours, theirs))

    def test_deletion_composes_with_distant_edit(self) -> None:
        ours = "one\nthree\nfour\nfive\n"
        theirs = "one\ntwo\nthree\nfour\nFIVE\n"
        self.assertEqual(merge3(BASE, ours, theirs), "one\nthree\nfour\nFIVE\n")

    def test_delete_vs_edit_of_same_line_conflicts(self) -> None:
        ours = "one\nthree\nfour\nfive\n"
        theirs = "one\nTWO\nthree\nfour\nfive\n"
        self.assertIsNone(merge3(BASE, ours, theirs))

    def test_missing_trailing_newline_edge(self) -> None:
        base = "a\nb"
        ours = "a\nb\n"
        theirs = "A\nb"
        self.assertEqual(merge3(base, ours, theirs), "A\nb\n")

    def test_empty_base_identical_additions(self) -> None:
        self.assertEqual(merge3("", "x\n", "x\n"), "x\n")

    def test_empty_base_different_additions_conflict(self) -> None:
        self.assertIsNone(merge3("", "x\n", "y\n"))

    def test_one_side_truncates_everything(self) -> None:
        self.assertEqual(merge3(BASE, BASE, ""), "")


if __name__ == "__main__":
    unittest.main()
