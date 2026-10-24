"""
Checks for Azari2017CSUM (docs/validation/azari_2017.md §4): the worked examples
of Fig. 1 and Fig. 4, a bay without a GBH solution (A2), and the fallback to the
GBH solution when CSUM finds none.

  python -m unittest algorithms.CRP_Time.heuristic.azari_2017.test_azari   (from the repo root)

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
corresponding original algorithm paper.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import contextlib
import io
import queue
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from algorithms.CRP_Time.utils.bay import free_destinations
from algorithms.CRP_Time.utils.located import min_due
from core.base_algorithm import AlgorithmConfig
from core.benchmarks.caserta import DEFAULT_CASERTA_DIR, problem_config_for_caserta_dat
from problems.CRP_Time import CRP_Time

from . import algorithm
from .branching import best_bb, best_bg, gbh_branches, set_b_branches
from .csum import csum


def _train(cfg, **extra):
    q = queue.Queue()
    alg = algorithm.Azari2017CSUM(AlgorithmConfig(extra=extra))
    alg.train(lambda: CRP_Time(config=cfg), q, threading.Event())
    records = []
    while not q.empty():
        records.append(q.get_nowait())
    return records


class TestWorkedExamples(unittest.TestCase):

    def test_fig1_best_bb_and_bg_stacks(self):
        # Fig. 1 after container 1 is retrieved; stack k of the paper is index k - 1. H = 5, so S = {1, 2, 4}.
        stacks = [[12, 13], [3, 6, 5], [9, 2, 8], [10, 7, 11, 4]]
        empty = 14
        q, src = 8, 2
        lows = [min_due(st, empty) for st in stacks]
        candidates = free_destinations(stacks, src, 5)
        self.assertEqual(candidates, [0, 1, 3])
        self.assertEqual(best_bb(q, lows, candidates), 3)
        self.assertEqual(best_bg(q, lows, candidates), 0)
        self.assertEqual(gbh_branches(q, src, stacks, 5, empty), [0, 3])

    def test_fig4_set_b_branches(self):
        # Fig. 4: each stack holds only its lowest number; the PS (stack 3) holds 1 below q = 10; stack 8 is empty.
        stacks = [[14], [13], [1, 10], [12], [9], [15], [11], []]
        self.assertEqual(set_b_branches(10, 2, stacks, 5, 16), [1, 3, 6])


class TestNoSolution(unittest.TestCase):

    def test_gbh_without_solution_raises(self):
        # A2: container 1 is under 2 and the only other stack is full (H = 2).
        with tempfile.TemporaryDirectory() as tmp:
            dat = Path(tmp) / "data2-2-1.dat"
            dat.write_text("2 4\n2 1 2\n2 3 4\n")
            cfg = problem_config_for_caserta_dat(dat, {"time_model": "f2"})
            cfg.max_tiers = 2
            with self.assertRaisesRegex(RuntimeError, r"GBH found no complete solution within gbh_time_limit_s = 1\.0 s"):
                _train(cfg, gbh_time_limit_s=1.0)

    def test_csum_without_solution_returns_gbh_solution(self):
        # data5-5-1 with the CSUM clock expired at the start: the GBH solution is returned.
        cfg = problem_config_for_caserta_dat(DEFAULT_CASERTA_DIR / "data5-5-1.dat", {"time_model": "f2"})
        expired = lambda *args: csum(*args[:-1], 0.0)
        err = io.StringIO()
        with mock.patch.object(algorithm, "csum", expired), contextlib.redirect_stderr(err):
            records = _train(cfg)
        self.assertIn("returning the GBH solution", err.getvalue())
        self.assertEqual([r.extra["csum_solution_found"] for r in records], [False] * 2)
        self.assertEqual(len(records[-1].extra["moves"]), 48)   # the saved plan is the GBH solution
        metrics = records[0].metrics
        self.assertEqual(metrics["total_moves"], 48)
        self.assertAlmostEqual(metrics["crane_time_f2"], 1725.60, places=6)


if __name__ == "__main__":
    unittest.main()
