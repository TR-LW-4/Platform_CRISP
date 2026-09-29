"""
Checks for LopezPlata2019Heuristic.

  python -m unittest algorithms.CRP_Time.heuristic.lopez_plata_2019.test_lopez_plata   (from the repo root)

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
corresponding original algorithm paper.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import queue
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from core.base_algorithm import AlgorithmConfig
from core.benchmarks.caserta import DEFAULT_CASERTA_DIR, parse_caserta_dat, problem_config_for_caserta_dat
from core.plan import RelocationPlan
from problems.CRP_Time import CRP_Time

from .algorithm import LopezPlata2019Heuristic
from .operating_cost import Bay, CostModel, f

# Fig. 2 (p. 438): n_S = 6, n_T = 4, stacks bottom -> top.
FIG2 = [[9, 1, 8], [11, 3, 2], [5, 4], [13, 6], [12, 10, 7], []]
# Relocate 8 from stack 1 to 6, then retrieve 1..13.
MIN_MOVES = [(1, 6), (1, 0), (2, 0), (2, 0), (3, 0), (3, 0), (4, 0), (5, 0),
             (6, 0), (1, 0), (5, 0), (2, 0), (5, 0), (4, 0)]
# Relocate 8 from 1 to 2, retrieve 1, relocate 8 from 2 to 1, then retrieve 2..13.
MIN_COST = [(1, 2), (1, 0), (2, 1), (2, 0), (2, 0), (3, 0), (3, 0), (4, 0),
            (5, 0), (1, 0), (1, 0), (5, 0), (2, 0), (5, 0), (4, 0)]

# Weights are always passed explicitly. W_PRINTED: the numbers at Fig. 2 and on p. 445;
# W_LOADED: loaded twice as heavy (Sec. 7.7, Table 5), the default since O01.
W_PRINTED = dict(eta1=2, gamma1=2, eta2=1, gamma2=1)
W_LOADED = dict(eta1=1, gamma1=1, eta2=2, gamma2=2)
PRINTED = CostModel(n_T=4, **W_PRINTED)
LOADED = CostModel(n_T=4, **W_LOADED)


class TestOperatingCost(unittest.TestCase):

    def test_fig2(self):
        """V01: 531 (14 moves) and 513 (15 moves), p. 438."""
        self.assertEqual((len(MIN_MOVES), f(FIG2, MIN_MOVES, PRINTED)), (14, 531))
        self.assertEqual((len(MIN_COST), f(FIG2, MIN_COST, PRINTED)), (15, 513))

    def test_fig2_both_weight_readings(self):
        """O01 does not change f of a restricted plan (V02 in the log)."""
        self.assertEqual(f(FIG2, MIN_MOVES, LOADED), 531)
        self.assertEqual(f(FIG2, MIN_COST, LOADED), 513)

    def test_trips_asymmetric_weights(self):
        """m uses eta1/gamma1 and m' eta2/gamma2 (hand values, n_T + epsilon = 6)."""
        cost = CostModel(n_T=4, eta1=1, gamma1=3, eta2=5, gamma2=7)
        self.assertEqual(cost.m(0, 1, 2, 3), 5 * 1 + 2 * 3 + 3 * 1)
        self.assertEqual(cost.m_loaded(2, 3, 5, 4), 3 * 5 + 3 * 7 + 2 * 5)

    def test_partial_and_incremental(self):
        """g of a partial solution: the first move of Fig. 2 costs m(0,1,1,3) + m'(1,3,6,1) = 18 + 13."""
        self.assertEqual(f(FIG2, MIN_MOVES[:1], PRINTED), 31)
        bay = Bay(FIG2, PRINTED)
        steps = [bay.apply(mv) for mv in MIN_COST]
        for k in range(len(MIN_COST) + 1):
            self.assertEqual(sum(steps[:k]), f(FIG2, MIN_COST[:k], PRINTED))
        self.assertIsNone(bay.target)

    def test_fig2_per_move(self):
        """Hand-checked terms of eq. 3, min-moves solution (n_T + epsilon = 6, printed weights).
        Move 1, relocate 8 from (1,3) to (6,1): m(0,1,1,3) = 5*2 + 1*2 + 3*2 = 18, m'(1,3,6,1) = 3 + 5 + 5 = 13.
        Move 2, retrieve 1 from (1,2); the crane is still at (6,1): m(6,1,1,2) = 5*2 + 5*2 + 4*2 = 28,
        m'(1,2,0,1) = 4 + 1 + 5 = 10.
        Move 9, retrieve 8 from (6,1): m(0,1,6,1) = 5*2 + 6*2 + 5*2 = 32, m'(6,1,0,1) = 5 + 6 + 5 = 16."""
        bay = Bay(FIG2, PRINTED)
        steps = [bay.apply(mv) for mv in MIN_MOVES]
        self.assertEqual((steps[0], steps[1], steps[8]), (31, 38, 48))

    def test_copy_is_independent(self):
        bay = Bay(FIG2, PRINTED)
        other = bay.copy()
        bay.apply((1, 6))
        self.assertEqual((other.stacks, other.crane, other.target), (FIG2, (0, 1), 1))

    def test_impossible_moves(self):
        cases = [
            (FIG2, PRINTED, [], (1, 0)),                    # c* = 1 is blocked by 8
            (FIG2, PRINTED, [], (2, 0)),                    # 2 is not the next to retrieve
            (FIG2, PRINTED, [], (5, 6)),                    # not restricted: stack 5 does not hold c*
            (FIG2, PRINTED, [(1, 6)], (1, 2)),              # c* on top must be retrieved
            (FIG2, PRINTED, [], (1, 1)),                    # relocation within its own stack
            (FIG2, PRINTED, [], (6, 0)),                    # empty stack
            (FIG2, PRINTED, [], (7, 0)),                    # no stack 7
            (FIG2, PRINTED, [], (1, 7)),
            (FIG2, PRINTED, [], (0, 1)),
            ([[1, 3], [4, 5], []], CostModel(n_T=2, **W_LOADED), [], (1, 2)),   # stack 2 is full
        ]
        for stacks, cost, prefix, move in cases:
            with self.subTest(move=move, prefix=prefix):
                bay = Bay(stacks, cost)
                for mv in prefix:
                    bay.apply(mv)
                with self.assertRaises(ValueError):
                    bay.apply(move)

    def test_stack_above_n_T(self):
        with self.assertRaises(ValueError):
            Bay(FIG2, CostModel(n_T=2, **W_LOADED))


TS, TPP = 1.2, 30.0


def _config(path, max_tiers=None):
    cfg = problem_config_for_caserta_dat(
        Path(path), {"time_model": "f2", "stack_s_per_stack": TS, "pickup_place_s": TPP})
    if max_tiers:
        cfg.max_tiers = max_tiers
    return cfg


def _run(cfg):
    algo = LopezPlata2019Heuristic(AlgorithmConfig())
    q = queue.Queue()
    algo.train(lambda: CRP_Time(config=cfg), q, threading.Event())
    return algo, q.get_nowait()


def _f2_formula(phi):
    """Voß-Schwarze f2 per move, stacks counted from the truck: retrieval from s: 2*ts*s + tpp,
    relocation s -> s': 2*ts*|s - s'| + tpp. Independent of the platform evaluator."""
    return sum(2 * TS * (s if d == 0 else abs(s - d)) + TPP for s, d in phi)


def _replay(cfg, actions):
    """Replay env actions with the env's own auto-retrieval, check the executed plan with validate_plan
    on a fresh env, and return it as (s, s')."""
    env = CRP_Time(config=cfg)
    assert env.validate_actions(actions)["feasible"], env.get_last_validation_errors()
    movements = list(env._plan.movements)
    check = CRP_Time(config=cfg)
    assert check.validate_plan(RelocationPlan(movements))["feasible"], check.get_last_validation_errors()
    return [(m.from_pos[0], 0 if m.to_pos is None else m.to_pos[0]) for m in movements]


class TestCompletePlan(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.fig2 = Path(cls.tmp.name) / "fig2.dat"
        cls.fig2.write_text("\n".join(["6 13"] + [" ".join(map(str, [len(s)] + s)) for s in FIG2]) + "\n")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_fig2_complete(self):
        """A complete plan that validate_plan accepts; operating_cost is f of that plan on a fresh Bay."""
        cfg = _config(self.fig2, max_tiers=4)
        algo, rec = _run(cfg)
        moves = _replay(cfg, algo.get_best_solution())
        self.assertEqual(sum(d == 0 for _, d in moves), 13)
        self.assertAlmostEqual(rec.extra["operating_cost"], f(FIG2, moves, CostModel(n_T=4, **W_LOADED)))

    def test_fig2_stack_mapping(self):
        """Paper stack s is bay s, at distance s from the truck: f2 equals the formula on the executed plan."""
        cfg = _config(self.fig2, max_tiers=4)
        algo, rec = _run(cfg)
        moves = _replay(cfg, algo.get_best_solution())
        self.assertAlmostEqual(rec.metrics["crane_time_f2"], _f2_formula(moves))

    def test_containers_on_top_at_start(self):
        """data3-3-5: blocks 1-4 are on top at the start; their retrievals are in the plan and in f2."""
        path = DEFAULT_CASERTA_DIR / "data3-3-5.dat"
        _, _, stacks = parse_caserta_dat(path)
        self.assertIn(1, [s[-1] for s in stacks])
        cfg = _config(path)
        algo, rec = _run(cfg)
        moves = _replay(cfg, algo.get_best_solution())
        self.assertEqual([d for _, d in moves[:4]], [0, 0, 0, 0])
        self.assertEqual(sum(d == 0 for _, d in moves), 9)
        self.assertEqual(rec.metrics["relocations"], sum(d != 0 for _, d in moves))
        self.assertAlmostEqual(rec.metrics["crane_time_f2"], _f2_formula(moves))
        self.assertAlmostEqual(rec.extra["operating_cost"],
                               f(stacks, moves, CostModel(n_T=cfg.max_tiers, **W_LOADED)))

    def test_truncated_phi_raises(self):
        cfg = _config(self.fig2, max_tiers=4)
        orig = LopezPlata2019Heuristic._construct_solution
        for cut in (lambda phi: phi[:-1], lambda phi: phi[1:]):
            def patched(self, *args, cut=cut):
                phi, cost = orig(self, *args)
                return cut(phi), cost
            with self.subTest(cut=cut), mock.patch.object(LopezPlata2019Heuristic, "_construct_solution", patched):
                with self.assertRaises(RuntimeError):
                    _run(cfg)


if __name__ == "__main__":
    unittest.main()
