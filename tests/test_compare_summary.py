"""Unit tests for multi-algorithm compare aggregation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from core.compare_summary import (
    compare_algorithms,
    delete_algorithm_folder,
    list_result_folders,
)


def _write_run(
    folder: Path,
    *,
    problem: str,
    algorithm: str,
    layout: str,
    relocations: float,
    name: str,
) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    payload = {
        "problem": problem,
        "algorithm": algorithm,
        "category": "Heuristic",
        "timestamp": name,
        "prob_config": {"layout_file_path": layout},
        "metrics": {"relocations": relocations},
        "history": [],
    }
    (folder / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")


class CompareSummaryTests(unittest.TestCase):
    def test_compare_highlights_best_mean(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = root / "CRP-R" / "Caserta_(2012)_HEUR"
            b = root / "CRP-R" / "Kim–Hong_(2006)_ENAR"
            _write_run(
                a,
                problem="CRP-R",
                algorithm="Caserta (2012) HEUR",
                layout="/data/x/data3-3-1.dat",
                relocations=4.0,
                name="20260101_000001",
            )
            _write_run(
                a,
                problem="CRP-R",
                algorithm="Caserta (2012) HEUR",
                layout="/data/x/data3-3-2.dat",
                relocations=6.0,
                name="20260101_000002",
            )
            _write_run(
                b,
                problem="CRP-R",
                algorithm="Kim–Hong (2006) ENAR",
                layout="/data/x/data3-3-1.dat",
                relocations=8.0,
                name="20260101_000003",
            )
            _write_run(
                b,
                problem="CRP-R",
                algorithm="Kim–Hong (2006) ENAR",
                layout="/data/x/data3-3-2.dat",
                relocations=10.0,
                name="20260101_000004",
            )

            folders = list_result_folders("CRP-R", base_dir=root)
            self.assertEqual(len(folders), 2)

            table = compare_algorithms(
                "CRP-R",
                ["Caserta (2012) HEUR", "Kim–Hong (2006) ENAR"],
                metric="relocations",
                base_dir=root,
            )
            self.assertEqual(table["metric"], "relocations")
            self.assertEqual(len(table["rows"]), 1)
            row = table["rows"][0]
            self.assertEqual(row["label"], "3×3")
            cas = row["cells"]["Caserta (2012) HEUR"]
            kim = row["cells"]["Kim–Hong (2006) ENAR"]
            self.assertAlmostEqual(cas["mean"], 5.0)
            self.assertAlmostEqual(kim["mean"], 9.0)
            self.assertTrue(cas["best"])
            self.assertFalse(kim["best"])

            removed = delete_algorithm_folder(
                "CRP-R", "Caserta (2012) HEUR", base_dir=root,
            )
            self.assertEqual(removed, 2)
            self.assertEqual(len(list_result_folders("CRP-R", base_dir=root)), 1)

    def test_compare_rows_sorted_by_numeric_hw(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = root / "CRP-R" / "Algo_A"
            b = root / "CRP-R" / "Algo_B"
            _write_run(
                a,
                problem="CRP-R",
                algorithm="Algo A",
                layout="/data/x/data10-10-1.dat",
                relocations=1.0,
                name="20260101_000001",
            )
            _write_run(
                a,
                problem="CRP-R",
                algorithm="Algo A",
                layout="/data/x/data5-4-1.dat",
                relocations=1.0,
                name="20260101_000002",
            )
            _write_run(
                b,
                problem="CRP-R",
                algorithm="Algo B",
                layout="/data/x/data3-3-1.dat",
                relocations=1.0,
                name="20260101_000003",
            )
            table = compare_algorithms(
                "CRP-R",
                ["Algo A", "Algo B"],
                metric="relocations",
                base_dir=root,
            )
            self.assertEqual(
                [row["label"] for row in table["rows"]],
                ["3×3", "5×4", "10×10"],
            )


if __name__ == "__main__":
    unittest.main()
