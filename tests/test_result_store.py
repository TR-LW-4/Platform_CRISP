"""Tests for PlatEMO-style result naming / overwrite paths."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from core.result_store import save_run, stable_result_stem


class StableStemTests(unittest.TestCase):
    def test_caserta_layout(self) -> None:
        stem = stable_result_stem({
            "layout_file_path": "/data/bench/Caserta_dataset/data3-3-7.dat",
        })
        self.assertEqual(stem, "caserta_3x3_07")

    def test_zhu_layout(self) -> None:
        stem = stable_result_stem({
            "layout_file_path": "/data/bench/Zhu_dataset/5-8-39/06101.txt",
        })
        self.assertEqual(stem, "zhu_5-8-39_06101")

    def test_random_fallback(self) -> None:
        self.assertEqual(stable_result_stem({}, seed=42), "random_seed42")

    def test_save_overwrites_same_instance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            cfg = {"layout_file_path": str(base / "data3-4-2.dat")}
            p1 = save_run(
                problem="CRP-R",
                algorithm="Caserta (2012) HEUR",
                category="Heuristic",
                seed=1,
                prob_config=cfg,
                algo_config={},
                metrics={"relocations": 9.0},
                history=[],
                base_dir=base,
            )
            p2 = save_run(
                problem="CRP-R",
                algorithm="Caserta (2012) HEUR",
                category="Heuristic",
                seed=1,
                prob_config=cfg,
                algo_config={},
                metrics={"relocations": 3.0},
                history=[],
                base_dir=base,
            )
            self.assertEqual(p1, p2)
            self.assertEqual(p1.name, "caserta_3x4_02.json")
            data = json.loads(p2.read_text(encoding="utf-8"))
            self.assertEqual(data["metrics"]["relocations"], 3.0)
            folder = p1.parent
            runs = [p for p in folder.glob("*.json")]
            self.assertEqual(len(runs), 1)


if __name__ == "__main__":
    unittest.main()
