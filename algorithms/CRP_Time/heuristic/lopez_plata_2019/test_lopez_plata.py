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

import math
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
from .measures import beta, greedy, h, non_located, p, q
from .operating_cost import Bay, CostModel, f
from .step1 import algorithm1, select

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


# Greedy (Sec. 7.1, reading R1) on Fig. 2, traced by hand with W_LOADED: m'(s, n(s), s', n(s')+1).
# 8 from (1,3): stack 2 costs 6+2+4 = 12, the cheapest. 8 from (2,4): stack 3 costs 4+2+6 = 12.
# 8 from (3,3): stacks 4 and 5 both cost 14, the tie goes to 4. 8 from (4,3): stack 5 costs 6+2+4 = 12.
# 8 from (5,4): stack 4 costs 4+2+8 = 14. All other moves are retrievals.
GREEDY_FIG2 = [(1, 2), (1, 0), (2, 3), (2, 0), (2, 0), (3, 4), (3, 0), (3, 0), (4, 5), (4, 0),
               (5, 4), (5, 0), (4, 0), (1, 0), (5, 0), (2, 0), (5, 0), (4, 0)]


def _remaining_optimum(stacks, cost):
    """Exact remaining cost of every state (configuration, crane position) reachable from (stacks, (0, 1)).
    The restricted state graph is acyclic: a relocation lowers the number of blocks above c*, a retrieval
    advances c*. Unsolvable states get infinity."""
    memo = {}

    def V(st, crane):
        if (st, crane) in memo:
            return memo[st, crane]
        best = 0.0 if not any(st) else math.inf
        if any(st):
            target = min(c for x in st for c in x)
            s = next(i for i, x in enumerate(st, 1) if target in x)
            pick = len(st[s - 1])
            options = [(0, 1)] if st[s - 1][-1] == target else [
                (d, len(st[d - 1]) + 1) for d in range(1, len(st) + 1) if d != s and len(st[d - 1]) < cost.n_T]
            for d, drop in options:
                new = list(st)
                new[s - 1] = st[s - 1][:-1]
                if d:
                    new[d - 1] = st[d - 1] + (st[s - 1][-1],)
                best = min(best, cost.m(*crane, s, pick) + cost.m_loaded(s, pick, d, drop) + V(tuple(new), (d, drop)))
        memo[st, crane] = best
        return best

    V(tuple(tuple(x) for x in stacks), (0, 1))
    return memo


class TestMeasures(unittest.TestCase):

    def test_h_lower_bound(self):
        """T1: h <= the exact remaining optimum in every reachable state of Fig. 2 and the 40 Caserta 3x3
        instances, under both weight readings (admissibility claim, p. 442-443)."""
        cases = [(FIG2, 4)] + [(parse_caserta_dat(DEFAULT_CASERTA_DIR / f"data3-3-{i}.dat")[2], 5) for i in range(1, 41)]
        for w in (W_LOADED, W_PRINTED):
            for stacks, n_T in cases:
                cost = CostModel(n_T=n_T, **w)
                for (st, _), v in _remaining_optimum(stacks, cost).items():
                    if not math.isinf(v):
                        self.assertLessEqual(h(Bay([list(x) for x in st], cost)), v + 1e-9, (w, st))

    def test_h_by_hand(self):
        """T2, eq. 21-22 with n_T + epsilon = 6. With W_LOADED, EstRet of c at (s, t) is
        (6-t)*1 + (6-t)*2 + 2s + 5*2 = 3(6-t) + 2s + 10.
        Stack 1 = [9, 1, 8]: EstRet 27 + 24 + 21; 8 is non-located on stack 1, so its relocation goes to
        stack 2 at tier n_T = 4: m(1,6,1,3) + m'(1,3,2,4) = 3 + (6 + 2 + 4) = 15; total 87.
        Fig. 2: stacks 1-5 give 87 + 78 + 59 + 63 + 96 = 383 (only 8 is non-located).
        Stack 3 = [3, 5] with eta1, gamma1, eta2, gamma2 = 1, 3, 5, 7: EstRet(3) = 5 + 25 + 21 + 25 = 76;
        EstRet(5) = 4 + 20 + 21 + 25 = 70, plus m(3,6,3,2) + m'(3,2,2,4) = 4 + (20 + 7 + 10) = 41; total 187."""
        self.assertEqual(h(Bay([[9, 1, 8], [], [], [], [], []], LOADED)), 87)
        self.assertEqual(h(Bay(FIG2, LOADED)), 383)
        self.assertEqual(h(Bay([[], [], [3, 5]], CostModel(n_T=4, eta1=1, gamma1=3, eta2=5, gamma2=7))), 187)
        self.assertEqual(h(Bay([[], []], LOADED)), 0)

    def test_non_located(self):
        """O21 (p. 437): non-located if some block below has an earlier retrieval order."""
        self.assertEqual(non_located([5, 1]), [False, False])
        self.assertEqual(non_located([2, 6]), [False, True])
        self.assertEqual(non_located([1, 3, 2]), [False, True, True])

    def test_q_by_hand(self):
        """T3, eq. 23-26: per block q_b + q_c / n_C + q_r / maxDist, maxDist = n_S = 2.
        [5, 1], [2, 6], n_C = 6: 5: 0 + 4/6 + 1/2; 1: 0 + 0 + 1/2; 2: 0 + 0 + 2/2; 6: 1 + 4/6 + 2/2; sum 16/3.
        [3], [1, 2], n_C = 3, c* = 1: 3 is alone, q_c = 3 - 1 = 2 (O10): 0 + 2/3 + 1/2; 1: 0 + 0 + 1;
        2: 1 + 1/3 + 1; sum 9/2.
        [1, 3, 2], [], n_C = 3: 1: 0 + 0 + 1/2; 3: 1 + 2/3 + 1/2; 2: 1 + 1/3 + 1/2; sum 9/2."""
        self.assertAlmostEqual(q(Bay([[5, 1], [2, 6]], LOADED), 6), 16 / 3)
        self.assertAlmostEqual(q(Bay([[3], [1, 2]], LOADED), 3), 9 / 2)
        self.assertAlmostEqual(q(Bay([[1, 3, 2], []], LOADED), 3), 9 / 2)

    def test_greedy_fig2(self):
        """T4: the hand trace above, including the tie; R1 gives the same moves under both weight readings."""
        bay = Bay(FIG2, LOADED)
        phi, cost = greedy(bay)
        self.assertEqual(phi, GREEDY_FIG2)
        self.assertEqual(cost, f(FIG2, phi, LOADED))
        self.assertEqual(greedy(Bay(FIG2, PRINTED))[0], GREEDY_FIG2)
        self.assertEqual((bay.stacks, bay.crane), (FIG2, (0, 1)))

    def test_greedy_criterion(self):
        """T4, R1 (O19): only m' decides. 5 above c* = 1 on stack 2; eta1, gamma1, eta2, gamma2 = 1, 10, 1, 2.
        m'(2,2,4,4) = 4 + 2*2 + 2 = 10 against m'(2,2,1,1) = m'(2,2,3,1) = 4 + 2 + 5 = 11, so R1 picks stack 4.
        With the empty return trip (R2), stack 4 costs 10 + m(4,4,2,1) = 10 + 27 and stack 1 costs 11 + 20."""
        bay = Bay([[], [1, 5], [], [2, 3, 4]], CostModel(n_T=4, eta1=1, gamma1=10, eta2=1, gamma2=2))
        self.assertEqual(greedy(bay)[0][0], (2, 4))

    def test_greedy_caserta(self):
        """T4: complete, feasible (f replays it through Bay) and ub = f(phi) on the 40 Caserta 3x3 instances;
        |phi| counts the retrievals at the start (data3-3-5 has 1-4 on top)."""
        for i in range(1, 41):
            _, n, stacks = parse_caserta_dat(DEFAULT_CASERTA_DIR / f"data3-3-{i}.dat")
            cost = CostModel(n_T=5, **W_LOADED)
            phi, ub = greedy(Bay(stacks, cost))
            self.assertEqual(sum(d == 0 for _, d in phi), n)
            self.assertAlmostEqual(ub, f(stacks, phi, cost))
            if i == 5:
                self.assertEqual([d for _, d in phi[:4]], [0, 0, 0, 0])

    def test_p_and_beta(self):
        """T5, eq. 29-30 with the ends of eq. 30 swapped (O02 (c)): p is 1.5 up to n_C = 9, the printed
        ramp 1.5 - (n_C - 9)/16 in between and 0.5 from 25; beta = floor(|phi| * theta/100 * p)."""
        self.assertEqual([p(n) for n in (5, 9, 10, 24, 25, 60)], [1.5, 1.5, 1.4375, 0.5625, 0.5, 0.5])
        self.assertEqual(beta(18, 13), 9)             # 18 * 0.4 * 1.25 = 9 exactly
        self.assertEqual(beta(29, 15), 13)            # 29 * 0.4 * 1.125 = 13.05
        self.assertEqual(beta(25, 9), 15)             # 25 * 0.4 * 1.5 = 15 exactly
        self.assertEqual(beta(40, 25), 8)             # 40 * 0.4 * 0.5 = 8 exactly
        self.assertEqual(beta(18, 13, theta=10), 2)   # 18 * 0.1 * 1.25 = 2.25
        self.assertEqual(beta(29, 10), 16)            # 29 * 0.4 * 1.4375 = 16.675: floor, not round


def _sequences(bay, k):
    """Brute force: every restricted move sequence of length k from bay that does not end in or
    pass through the complete configuration."""
    if k == 0:
        yield tuple(bay.phi)
        return
    if bay.target is None:
        return
    s = bay.target_stack()
    moves = [(s, 0)] if bay.stacks[s - 1][-1] == bay.target else [
        (s, d) for d in range(1, len(bay.stacks) + 1) if d != s and len(bay.stacks[d - 1]) < bay.cost.n_T]
    for mv in moves:
        child = bay.copy()
        child.apply(mv)
        yield from _sequences(child, k - 1)


def _algorithm1_reference(root, ub, depth):
    """Recursive transcription of Algorithm 1 with the depth limit, written apart from step1.py."""
    collected, state = [], {"ub": ub, "best": None}

    def visit(n):
        if len(n.phi) == depth:
            collected.append(tuple(n.phi))
            return
        s = n.target_stack()
        if n.stacks[s - 1][-1] == n.target:
            child = n.copy()
            child.apply((s, 0))
            if child.target is None:
                if child.g < state["ub"]:
                    state["ub"], state["best"] = child.g, tuple(child.phi)
            else:
                visit(child)
            return
        kids = []
        for d in range(1, len(n.stacks) + 1):
            if d != s and len(n.stacks[d - 1]) < n.cost.n_T:
                c = n.copy()
                c.apply((s, d))
                if c.g + h(c) < state["ub"]:
                    kids.append(c)
        for c in sorted(kids, key=lambda c: c.g):
            visit(c)

    visit(root)
    return collected, state["best"], state["ub"]


class TestStep1(unittest.TestCase):

    def test_algorithm1_exact(self):
        """Without a depth limit, Algorithm 1 with ub = f(greedy) is exact: min(f(best), ub) equals the
        exact optimum on Fig. 2 (513) and on each Caserta 3x3 instance (average 497.40, V06)."""
        for w in (W_LOADED, W_PRINTED):
            root = Bay(FIG2, CostModel(n_T=4, **w))
            _, ub = greedy(root)
            _, best, _ = algorithm1(root, ub)
            self.assertEqual(min(best.g if best else math.inf, ub), 513)
            values = []
            for i in range(1, 41):
                stacks = parse_caserta_dat(DEFAULT_CASERTA_DIR / f"data3-3-{i}.dat")[2]
                cost = CostModel(n_T=5, **w)
                root = Bay(stacks, cost)
                _, ub = greedy(root)
                _, best, _ = algorithm1(root, ub)
                value = min(best.g if best else math.inf, ub)
                self.assertAlmostEqual(value, _remaining_optimum(stacks, cost)[tuple(map(tuple, stacks)), (0, 1)])
                values.append(value)
            self.assertAlmostEqual(sum(values) / 40, 497.40)

    def test_n_beta_brute_force(self):
        """With ub = infinity and no leaf within beta, N_beta is every restricted sequence of length beta."""
        for depth in (3, 6):
            nodes, best, ub = algorithm1(Bay(FIG2, LOADED), math.inf, depth)
            self.assertEqual(sorted(tuple(n.phi) for n in nodes), sorted(_sequences(Bay(FIG2, LOADED), depth)))
            self.assertEqual((best, ub), (None, math.inf))

    def test_n_beta_reference(self):
        """With ub = f(greedy), N_beta, the best leaf and the final ub equal the recursive transcription,
        in the same order; at beta = 15 leaves (14 moves and more) lower ub during the search.
        Every collected node has depth exactly beta, is not complete, and g = f(phi)."""
        for w in (W_LOADED, W_PRINTED):
            cost = CostModel(n_T=4, **w)
            _, ub = greedy(Bay(FIG2, cost))
            for depth in (3, 9, 15):
                nodes, best, final_ub = algorithm1(Bay(FIG2, cost), ub, depth)
                ref_nodes, ref_best, ref_ub = _algorithm1_reference(Bay(FIG2, cost), ub, depth)
                self.assertEqual([tuple(n.phi) for n in nodes], ref_nodes)
                self.assertEqual((tuple(best.phi) if best else None, final_ub), (ref_best, ref_ub))
                for n in nodes:
                    self.assertEqual(len(n.phi), depth)
                    self.assertIsNotNone(n.target)
                    self.assertAlmostEqual(n.g, f(FIG2, n.phi, cost))
                if depth == 15:
                    self.assertIsNotNone(best)
                    self.assertLess(final_ub, ub)

    def test_select_by_hand(self):
        """N_g + N_q (O06 (a), O07). n_C = 2, maxDist = 2; q by hand (eq. 23-26):
        A [1], [2]: 2.0; B [1, 2], []: 2.5; C [], [1, 2]: 3.5; D [2, 1], []: 1.5; E [], [2, 1]: 2.5.
        g: A 10, B 5, C 5, D 7, E 20; discovery order A, B, C, D, E. Ties keep that order."""
        layouts = {"A": [[1], [2]], "B": [[1, 2], []], "C": [[], [1, 2]], "D": [[2, 1], []], "E": [[], [2, 1]]}
        g = {"A": 10, "B": 5, "C": 5, "D": 7, "E": 20}
        nodes = []
        for name in "ABCDE":
            n = Bay(layouts[name], LOADED)
            n.g = g[name]
            nodes.append(n)
        names = {id(n): name for n, name in zip(nodes, "ABCDE")}
        for lam, expected in ((1, "BD"), (2, "BCDA"), (3, "BCDA"), (10, "BCDAE")):
            self.assertEqual("".join(names[id(n)] for n in select(nodes, lam, 2)), expected, lam)

    def test_deterministic(self):
        """Step 1 and the whole run give the same result twice and for another seed."""
        stacks = parse_caserta_dat(DEFAULT_CASERTA_DIR / "data3-4-1.dat")[2]
        cost = CostModel(n_T=5, **W_LOADED)
        runs = []
        for _ in range(2):
            root = Bay(stacks, cost)
            phi, ub = greedy(root)
            nodes, best, _ = algorithm1(root, ub, beta(len(phi), 12))
            runs.append(([tuple(n.phi) for n in select(nodes, 500, 12)], best and tuple(best.phi)))
        self.assertEqual(runs[0], runs[1])
        cfg = _config(DEFAULT_CASERTA_DIR / "data3-4-1.dat")
        plans = []
        for seed in (0, 7):
            algo = LopezPlata2019Heuristic(AlgorithmConfig(seed=seed))
            algo.train(lambda: CRP_Time(config=cfg), queue.Queue(), threading.Event())
            plans.append(algo.get_best_solution())
        self.assertEqual(plans[0], plans[1])


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
