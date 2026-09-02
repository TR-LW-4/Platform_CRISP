"""Unit tests for batch Continue helpers (no worker process)."""

from __future__ import annotations

import unittest

from web.backend.jobs import _merge_unique


class MergeUniqueTests(unittest.TestCase):
    def test_keeps_previous_then_appends_new(self) -> None:
        merged = _merge_unique(
            ["/results/a.json", "/results/b.json"],
            ["/results/b.json", "/results/c.json"],
        )
        self.assertEqual(
            merged,
            ["/results/a.json", "/results/b.json", "/results/c.json"],
        )

    def test_empty_incoming_preserves_existing(self) -> None:
        existing = ["/results/a.json"]
        self.assertEqual(_merge_unique(existing, []), existing)


if __name__ == "__main__":
    unittest.main()
