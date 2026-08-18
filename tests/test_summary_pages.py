"""Tests for npm-free HTML benchmark summary helpers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from web.backend.summary_pages import (
    render_summary_html,
    summarize_saved_runs,
)


class SummaryPagesTests(unittest.TestCase):
    def test_render_contains_class_and_stats(self) -> None:
        html = render_summary_html(
            {
                "classes": [{
                    "label": "3×3",
                    "n": 40,
                    "metrics": {
                        "relocations": {"mean": 5.1, "std": 2.0},
                    },
                }],
                "ungrouped": 0,
                "total_runs": 40,
            },
            title="Demo",
            subtitle="test",
        )
        self.assertIn("3×3", html)
        self.assertIn("relocations mean", html)
        self.assertIn("5.1", html)

    def test_summarize_saved_runs_reads_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "CRP-R" / "Caserta_(2012)_HEUR"
            folder.mkdir(parents=True)
            for i, rel in enumerate((4.0, 6.0)):
                (folder / f"run{i}.json").write_text(
                    json.dumps({
                        "prob_config": {
                            "layout_file_path": f"/x/data3-3-{i+1}.dat",
                        },
                        "metrics": {"relocations": rel},
                    }),
                    encoding="utf-8",
                )
            with mock.patch(
                "web.backend.summary_pages.results_root",
                return_value=root,
            ):
                summary = summarize_saved_runs(
                    problem="CRP-R",
                    algorithm="Caserta (2012) HEUR",
                    limit=10,
                )
            self.assertEqual(summary["total_runs"], 2)
            self.assertEqual(summary["classes"][0]["label"], "3×3")
            self.assertAlmostEqual(
                summary["classes"][0]["metrics"]["relocations"]["mean"], 5.0
            )


if __name__ == "__main__":
    unittest.main()
