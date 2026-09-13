"""Tests for PlatEMO-style result naming / overwrite paths."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from core.result_store import (
    completed_layout_paths,
    result_path_for,
    save_run,
    stable_result_stem,
)
from core.results.aggregate import summarize_run_dicts


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

    def test_stow_layout(self) -> None:
        stem = stable_result_stem({
            "layout_file_path": (
                "/data/benchmark/crp_stow/Data/Gen/"
                "Bay-3-20-38-6_23.pro"
            ),
        })
        self.assertEqual(stem, "stow_3-20-38-6_23")

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

    def test_result_path_for_matches_save_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            layout = base / "data5-4-19.dat"
            saved = save_run(
                problem="CRP-R",
                algorithm="LA-N (2013)",
                category="Heuristic",
                seed=0,
                prob_config={"layout_file_path": str(layout)},
                algo_config={},
                metrics={"relocations": 4.0},
                history=[],
                base_dir=base,
            )
            looked_up = result_path_for(
                "CRP-R", "LA-N (2013)", layout, base_dir=base, mkdir=False,
            )
            self.assertEqual(saved, looked_up)

    def test_completed_layout_paths_skips_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            done = base / "data3-4-1.dat"
            pending = base / "data3-4-2.dat"
            save_run(
                problem="CRP-R",
                algorithm="Caserta (2012) HEUR",
                category="Heuristic",
                seed=0,
                prob_config={"layout_file_path": str(done)},
                algo_config={},
                metrics={"relocations": 1.0},
                history=[],
                base_dir=base,
            )
            found = completed_layout_paths(
                "CRP-R",
                "Caserta (2012) HEUR",
                [done, pending],
                base_dir=base,
            )
            self.assertEqual(found, [str(done)])

    def test_continue_starts_at_first_unfinished(self) -> None:
        """350-instance mental model: after 1..3 exist, remaining starts at 4."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            layouts = [base / f"data3-4-{i}.dat" for i in range(1, 6)]
            for layout in layouts[:3]:
                save_run(
                    problem="CRP-R",
                    algorithm="LA-N (2013)",
                    category="Heuristic",
                    seed=0,
                    prob_config={"layout_file_path": str(layout)},
                    algo_config={},
                    metrics={"relocations": 1.0},
                    history=[],
                    base_dir=base,
                )
            done = set(completed_layout_paths(
                "CRP-R",
                "LA-N (2013)",
                layouts,
                base_dir=base,
            ))
            remaining = [str(p) for p in layouts if str(p) not in done]
            self.assertEqual(Path(remaining[0]).name, "data3-4-4.dat")
            self.assertEqual(len(remaining), 2)

    def test_lookup_does_not_create_folders(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            path = result_path_for(
                "CRP-R",
                "Missing Algo",
                base / "data3-3-1.dat",
                base_dir=base,
                mkdir=False,
            )
            self.assertFalse(path.parent.exists())
            self.assertFalse(path.exists())

    def test_time_objectives_use_distinct_result_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            layout = base / "data3-4-1.dat"
            f2_path = result_path_for(
                "CRP-Time",
                "Time Algo",
                layout,
                prob_config={
                    "objective_mode": "crane_time",
                    "time_model": "f2",
                    "stack_s_per_stack": 1.2,
                    "pickup_place_s": 30.0,
                },
                base_dir=base,
                mkdir=False,
            )
            vertical_path = result_path_for(
                "CRP-Time",
                "Time Algo",
                layout,
                prob_config={
                    "objective_mode": "crane_time",
                    "time_model": "f2_vertical",
                    "stack_s_per_stack": 1.2,
                    "empty_vertical_s_per_tier": 2.59,
                    "loaded_vertical_s_per_tier": 5.18,
                },
                base_dir=base,
                mkdir=False,
            )
            self.assertNotEqual(f2_path, vertical_path)

    def test_aggregation_does_not_mix_time_objectives(self) -> None:
        layout = "/bench/Caserta_dataset/data3-4-1.dat"
        runs = [
            {
                "prob_config": {
                    "layout_file_path": layout,
                    "objective_mode": "crane_time",
                    "time_model": model,
                },
                "metrics": {"objective_value": value},
            }
            for model, value in (("f2", 10.0), ("f2_vertical", 20.0))
        ]
        summary = summarize_run_dicts(runs)
        self.assertEqual(len(summary["classes"]), 2)


if __name__ == "__main__":
    unittest.main()
