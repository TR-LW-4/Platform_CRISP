"""Unit tests for class-wise benchmark aggregation (no algorithm runs)."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def _load_benchmark_summary():
    """Load module without importing heavy ``core/__init__.py`` (numpy, etc.)."""
    if "core" not in sys.modules:
        pkg = types.ModuleType("core")
        pkg.__path__ = [str(_ROOT / "core")]
        sys.modules["core"] = pkg
    name = "core.benchmark_summary"
    path = _ROOT / "core" / "benchmark_summary.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_summary = _load_benchmark_summary()
summarize_run_dicts = _summary.summarize_run_dicts
summarize_result_files = _summary.summarize_result_files


class BenchmarkSummaryTests(unittest.TestCase):
    def test_groups_caserta_by_hw_and_computes_mean_std(self) -> None:
        runs = [
            {
                "prob_config": {
                    "layout_file_path": "/data/bench/data3-3-1.dat",
                },
                "metrics": {"relocations": 4.0, "time": 0.1},
            },
            {
                "prob_config": {
                    "layout_file_path": "/data/bench/data3-3-2.dat",
                },
                "metrics": {"relocations": 6.0, "time": 0.3},
            },
            {
                "prob_config": {
                    "layout_file_path": "/data/bench/data3-4-1.dat",
                },
                "metrics": {"relocations": 10.0},
            },
        ]
        summary = summarize_run_dicts(runs)
        self.assertEqual(summary["total_runs"], 3)
        self.assertEqual(summary["ungrouped"], 0)
        self.assertEqual(len(summary["classes"]), 2)

        by_label = {row["label"]: row for row in summary["classes"]}
        self.assertIn("3×3", by_label)
        self.assertIn("3×4", by_label)

        c33 = by_label["3×3"]
        self.assertEqual(c33["n"], 2)
        self.assertEqual(c33["h"], 3)
        self.assertEqual(c33["w"], 3)
        self.assertAlmostEqual(c33["metrics"]["relocations"]["mean"], 5.0)
        self.assertAlmostEqual(c33["metrics"]["relocations"]["std"], 2**0.5, places=5)
        self.assertAlmostEqual(c33["metrics"]["time"]["mean"], 0.2)

        c34 = by_label["3×4"]
        self.assertEqual(c34["n"], 1)
        self.assertAlmostEqual(c34["metrics"]["relocations"]["mean"], 10.0)
        self.assertAlmostEqual(c34["metrics"]["relocations"]["std"], 0.0)

    def test_caserta_classes_sorted_by_numeric_hw(self) -> None:
        runs = [
            {
                "prob_config": {"layout_file_path": "/data/bench/data10-10-1.dat"},
                "metrics": {"relocations": 1.0},
            },
            {
                "prob_config": {"layout_file_path": "/data/bench/data5-10-1.dat"},
                "metrics": {"relocations": 1.0},
            },
            {
                "prob_config": {"layout_file_path": "/data/bench/data5-4-1.dat"},
                "metrics": {"relocations": 1.0},
            },
            {
                "prob_config": {"layout_file_path": "/data/bench/data3-3-1.dat"},
                "metrics": {"relocations": 1.0},
            },
        ]
        labels = [row["label"] for row in summarize_run_dicts(runs)["classes"]]
        self.assertEqual(labels, ["3×3", "5×4", "5×10", "10×10"])

    def test_groups_stow_pro_files_by_brlp_class(self) -> None:
        runs = [
            {
                "prob_config": {
                    "layout_file_path": (
                        f"/data/crp_stow/Data/Gen/Bay-3-20-38-6_{seed}.pro"
                    ),
                },
                "metrics": {"relocations": value},
            }
            for seed, value in ((0, 10.0), (1, 14.0))
        ]
        summary = summarize_run_dicts(runs)
        self.assertEqual(summary["ungrouped"], 0)
        self.assertEqual(len(summary["classes"]), 1)
        row = summary["classes"][0]
        self.assertEqual(row["label"], "A=3 · VS=20 · YS=38 · YT=6")
        self.assertEqual(row["n"], 2)
        self.assertEqual(row["vessel_stacks"], 20)
        self.assertAlmostEqual(row["metrics"]["relocations"]["mean"], 12.0)

    def test_summarize_result_files_reads_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "run.json"
            path.write_text(
                json.dumps({
                    "prob_config": {
                        "layout_file_path": str(root / "data5-8-7.dat"),
                    },
                    "metrics": {"relocations": 12},
                }),
                encoding="utf-8",
            )
            summary = summarize_result_files([str(path)])
            self.assertEqual(summary["total_runs"], 1)
            self.assertEqual(summary["classes"][0]["label"], "5×8")
            self.assertEqual(summary["classes"][0]["n"], 1)

    def test_ungrouped_without_layout(self) -> None:
        summary = summarize_run_dicts([{"metrics": {"relocations": 1}}])
        self.assertEqual(summary["ungrouped"], 1)
        self.assertEqual(summary["classes"], [])


if __name__ == "__main__":
    unittest.main()
