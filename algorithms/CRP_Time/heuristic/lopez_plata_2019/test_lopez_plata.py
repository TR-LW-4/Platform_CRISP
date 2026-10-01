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
import random
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
from .step1 import RunningSelection, algorithm1, partial_solutions, select
from .step2 import alphas, children, draw, expand, q_m, roulette, sequences
from .step3 import improve_once, postprocess, refit, temporary_moves

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
        """Step 1 gives the same result twice; so does the whole run with the same seed (another seed
        may differ through the roulette of step 2, see TestStep2)."""
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
        for _ in range(2):
            algo = LopezPlata2019Heuristic(AlgorithmConfig(seed=7, extra={"rho": RHO_TEST}))
            algo.train(lambda: CRP_Time(config=cfg), queue.Queue(), threading.Event())
            plans.append(algo.get_best_solution())
        self.assertEqual(plans[0], plans[1])


def _plain(stacks, cost, depth, lam=500):
    """Step 1 without a budget: algorithm1 to depth `depth` with the running selection."""
    n_C = sum(len(s) for s in stacks)
    sel, stats = RunningSelection(lam, n_C), {}
    _, best, ub = algorithm1(Bay(stacks, cost), greedy(Bay(stacks, cost))[1], depth, sel, stats)
    return _summary(sel.result(), best, ub, n_C), stats["visited"], sel.count


def _summary(nodes, best, ub, n_C):
    return [(tuple(n.phi), n.g, q(n, n_C)) for n in nodes], best and tuple(best.phi), ub


def _budgeted(stacks, cost, node_budget, lam=500):
    n_C = sum(len(s) for s in stacks)
    root = Bay(stacks, cost)
    phi, ub = greedy(root)
    depth = beta(len(phi), n_C)
    nodes, best, final_ub, info = partial_solutions(root, ub, depth, lam, n_C, node_budget)
    return _summary(nodes, best, final_ub, n_C), info, depth


class TestIterativeDeepening(unittest.TestCase):
    """Step 1 under a node budget (L18)."""

    def _cases(self):
        cases = [(FIG2, CostModel(n_T=4, **W_LOADED))]
        for size, ids in (("3-3", range(1, 11)), ("3-4", (1, 2)), ("3-5", (1, 3))):
            for i in ids:
                stacks = parse_caserta_dat(DEFAULT_CASERTA_DIR / f"data{size}-{i}.dat")[2]
                cases.append((stacks, CostModel(n_T=int(size[0]) + 2, **W_LOADED)))
        return cases

    def test_no_hit_equals_plain(self):
        """(a) Budget not hit: N (nodes, order, g, q), best leaf and ub equal the plain run to beta."""
        for stacks, cost in self._cases():
            got, info, depth = _budgeted(stacks, cost, 10 ** 7)
            want, visited, n_beta = _plain(stacks, cost, depth)
            self.assertFalse(info["budget_hit"])
            self.assertEqual((info["beta_eff"], info["visited_last"], info["n_beta"]), (depth, visited, n_beta))
            self.assertEqual(got, want)

    def test_no_hit_with_leaves(self):
        """(a) Fig. 2 to depth 15: leaves (14 and 15 moves) lower ub inside the deepest iterations; every
        iteration must still start from f(greedy), so the result equals the plain run to depth 15."""
        cost = CostModel(n_T=4, **W_LOADED)
        root = Bay(FIG2, cost)
        _, ub = greedy(root)
        nodes, best, final_ub, info = partial_solutions(root, ub, 15, 500, 13, 10 ** 7)
        sel = RunningSelection(500, 13)
        _, want_best, want_ub = algorithm1(Bay(FIG2, cost), ub, 15, sel)
        self.assertEqual(_summary(nodes, best, final_ub, 13), _summary(sel.result(), want_best, want_ub, 13))
        self.assertLess(final_ub, ub)
        self.assertEqual(info["beta_eff"], 15)

    def test_hit_equals_plain_at_beta_eff(self):
        """(b) Budget hit: the result equals the plain run to beta_eff, and exactly node_budget nodes are
        visited. data3-5-2 has beta = 15; about 255 000 nodes complete depths 1-14, depth 15 needs 359 019 more."""
        stacks = parse_caserta_dat(DEFAULT_CASERTA_DIR / "data3-5-2.dat")[2]
        cost = CostModel(n_T=5, **W_LOADED)
        for budget in (300_000, 20_000, 50, 1):
            got, info, depth = _budgeted(stacks, cost, budget)
            self.assertTrue(info["budget_hit"])
            self.assertLess(info["beta_eff"], depth)
            self.assertEqual(info["visited_total"], budget)
            want, visited, n_beta = _plain(stacks, cost, info["beta_eff"])
            self.assertEqual(got, want)
            if info["beta_eff"]:
                self.assertEqual((info["visited_last"], info["n_beta"]), (visited, n_beta))
        self.assertEqual(_budgeted(stacks, cost, 300_000)[1]["beta_eff"], 14)

    def test_deterministic(self):
        """(c) Same result and the same visited-node counts twice, with and without hitting the budget."""
        stacks = parse_caserta_dat(DEFAULT_CASERTA_DIR / "data3-5-2.dat")[2]
        cost = CostModel(n_T=5, **W_LOADED)
        for budget in (10 ** 7, 20_000):
            self.assertEqual(_budgeted(stacks, cost, budget), _budgeted(stacks, cost, budget))

    def test_metrics(self):
        """(d) The run reports beta, beta_eff, budget_hit and visited_total; beta_eff <= beta, with
        equality without a hit."""
        cfg = _config(DEFAULT_CASERTA_DIR / "data3-5-2.dat")
        for budget, hit in ((10 ** 7, False), (20_000, True)):
            algo = LopezPlata2019Heuristic(AlgorithmConfig(extra={"node_budget": budget, "rho": 2}))
            q_ = queue.Queue()
            algo.train(lambda: CRP_Time(config=cfg), q_, threading.Event())
            m = q_.get_nowait().metrics
            for key in ("operating_cost", "beta", "beta_eff", "budget_hit", "visited_total"):
                self.assertIn(key, m)
            for key in ("theta_eff_min", "theta_eff_max", "visited_last", "n_beta"):
                self.assertNotIn(key, m)
            self.assertEqual(bool(m["budget_hit"]), hit)
            self.assertLessEqual(m["visited_total"], budget)
            if hit:
                self.assertLess(m["beta_eff"], m["beta"])
            else:
                self.assertEqual(m["beta_eff"], m["beta"])


TS, TPP = 1.2, 30.0
RHO_TEST = 5   # step-2 iterations in the unit tests that run train(); the default is 300

# Fig. 4 (p. 444), read by Thom from the page image: n_S = 6, n_T = 4, n_C = 13, c* = 1; the move shown is
# block 5 from stack 1 onto stack 3.
FIG4 = [[9, 1, 5], [11, 3, 2], [8, 4], [13, 6], [12, 10, 7], []]


FIG5_ORIGINAL = [(1, 3), (1, 0), (2, 0), (2, 0), (3, 1), (3, 0)]   # Fig. 5 (p. 445), on the layout of Fig. 4
FIG5_FIX = [(1, 2), (1, 0), (2, 1), (2, 0), (2, 0), (3, 0)]


def _random_restricted(rng, n_S, n_T, n_blocks):
    """A random layout and a random complete restricted plan (random destination for every c+)."""
    prios = list(range(1, n_blocks + 1))
    rng.shuffle(prios)
    stacks = [[] for _ in range(n_S)]
    for c in prios:
        stacks[rng.choice([i for i in range(n_S) if len(stacks[i]) < n_T])].append(c)
    bay, phi = Bay(stacks, CostModel(n_T=n_T, **W_LOADED)), []
    while bay.target is not None:
        s = bay.target_stack()
        if bay.stacks[s - 1][-1] == bay.target:
            move = (s, 0)
        else:
            move = (s, rng.choice([d for d in range(1, n_S + 1) if d != s and len(bay.stacks[d - 1]) < n_T]))
        bay.apply(move)
        phi.append(move)
    return stacks, phi


class TestStep3(unittest.TestCase):

    def test_fig5(self):
        """Fig. 5 (p. 445), weights (b), crane from (0, 1). Original: 5 1->3 (25), R1 (29), R2 (30), R3 (33),
        5 3->1 (29), R4 (34) = 180. Fix: 5 1->2 (21), R1 (27), 5 2->1 (23), R2 (28), R3 (33), R4 (36) = 168;
        same end configuration and crane position (hand calculation by Thom, checked independently).
        The move of 5 onto stack 3 is temporary; under O13 (d) only target 2 is accepted (on 4, 5, 6 block 5
        is not forced to move before its original second move), and the postprocessing yields the fix."""
        for phi, steps in ((FIG5_ORIGINAL, [25, 29, 30, 33, 29, 34]), (FIG5_FIX, [21, 27, 23, 28, 33, 36])):
            bay = Bay(FIG4, LOADED)
            self.assertEqual([bay.apply(m) for m in phi], steps)
        self.assertEqual(temporary_moves(FIG4, FIG5_ORIGINAL, LOADED), [(0, 4)])
        self.assertEqual(refit(FIG4, FIG5_ORIGINAL, LOADED, 0, 4, 2), (FIG5_FIX, 168))
        for target in (4, 5, 6):
            self.assertIsNone(refit(FIG4, FIG5_ORIGINAL, LOADED, 0, 4, target))
        self.assertEqual(postprocess(FIG4, FIG5_ORIGINAL, LOADED), (FIG5_FIX, 168))
        self.assertEqual(postprocess(FIG4, FIG5_FIX, LOADED), (FIG5_FIX, 168))

    def test_covered_block_is_not_temporary(self):
        """Condition (ii): 5 goes onto stack 3, then 6 is put on top of it before 5 moves again, so that
        pair is not temporary; the later pair of 6 (onto 3, then onto 4) is."""
        stacks = [[1, 5], [2, 6], [3], []]
        phi = [(1, 3), (1, 0), (2, 3), (2, 0), (3, 4), (3, 1), (3, 0), (1, 0), (4, 0)]
        cost = CostModel(n_T=4, **W_LOADED)
        self.assertEqual(f(stacks, phi, cost), f(stacks, phi, cost))   # the plan is feasible
        self.assertEqual(temporary_moves(stacks, phi, cost), [(2, 4)])

    def test_late_forcing_is_rejected(self):
        """O13 (d): block 5 goes onto stack 1 and back onto stack 3 (the pair (0, 2)). With target 2 it would
        only be forced to move after its original second move (when 4 is due, not when 2 is), so the
        repair is rejected, although the rest of the plan would stay feasible (n_T = 3)."""
        stacks = [[3, 2], [4], [1, 5]]
        phi = [(3, 1), (3, 0), (1, 3), (1, 0), (1, 0), (2, 0), (3, 0)]
        cost = CostModel(n_T=3, **W_LOADED)
        self.assertEqual(temporary_moves(stacks, phi, cost), [(0, 2)])
        self.assertIsNone(refit(stacks, phi, cost, 0, 2, 2))

    def test_improve_once_choice(self):
        """On random plans, one improvement step repairs the first temporary movement that some target
        improves strictly, with the target of lowest f (ties to the lowest index), and nothing else."""
        rng, checked = random.Random(11), 0
        for _ in range(300):
            n_S, n_T = rng.randint(3, 5), rng.randint(3, 4)
            stacks, phi = _random_restricted(rng, n_S, n_T, rng.randint(n_S, n_S * n_T - n_T + 1))
            cost = CostModel(n_T=n_T, **W_LOADED)
            f_phi = f(stacks, phi, cost)
            expected = None
            for i, j in temporary_moves(stacks, phi, cost):
                fits = [refit(stacks, phi, cost, i, j, t) for t in range(1, n_S + 1) if t not in phi[i]]
                fits = [x for x in fits if x and x[1] < f_phi]
                if fits:
                    expected = min(fits, key=lambda x: x[1])
                    break
            self.assertEqual(improve_once(stacks, phi, cost, f_phi), expected)
            checked += expected is not None
        self.assertGreater(checked, 20)

    def test_random_plans(self):
        """On 400 random restricted plans: f never increases, the length stays the same, the result is a
        complete restricted plan (replayed through Bay), and some plans do improve."""
        rng, improved = random.Random(2026), 0
        for _ in range(400):
            n_S, n_T = rng.randint(3, 5), rng.randint(3, 4)
            stacks, phi = _random_restricted(rng, n_S, n_T, rng.randint(n_S, n_S * n_T - n_T + 1))
            cost = CostModel(n_T=n_T, **W_LOADED)
            before = f(stacks, phi, cost)
            new_phi, after = postprocess(stacks, phi, cost)
            self.assertLessEqual(after, before)
            self.assertEqual(len(new_phi), len(phi))
            bay = Bay(stacks, cost)
            for move in new_phi:
                bay.apply(move)
            self.assertIsNone(bay.target)
            self.assertAlmostEqual(bay.g, after)
            improved += after < before
        self.assertGreater(improved, 20)


def _brute_sequences(node, kappa, tau, n_C):
    """Every restricted continuation (any destination) up to min(kappa, blocks left) retrievals, kept only
    if each relocation goes to one of the tau children of its node; with its sum of alpha."""
    goal = min(kappa, sum(len(st) for st in node.stacks))
    out = []

    def walk(bay, retrieved, total):
        if retrieved == goal:
            out.append((tuple(bay.phi), total))
            return
        s = bay.target_stack()
        if bay.stacks[s - 1][-1] == bay.target:
            child = bay.copy()
            child.apply((s, 0))
            walk(child, retrieved + 1, total)
            return
        allowed = {d for _, d in children(bay, tau, n_C)}
        alpha = {d: a for a, d in alphas(bay, n_C)}
        for d in range(1, len(bay.stacks) + 1):
            if d != s and len(bay.stacks[d - 1]) < bay.cost.n_T and d in allowed:
                child = bay.copy()
                child.apply((s, d))
                walk(child, retrieved, total + alpha[d])

    walk(node.copy(), 0, 0.0)
    return out


class TestStep2(unittest.TestCase):

    def test_alpha_by_hand(self):
        """1. Eq. 27-28 on Fig. 4 (p. 444): block 5 on stack 1 (c* = 1), n_T = 4, epsilon = 2, W_LOADED, n_C = 13,
        maxDist = n_S = 6, each term separately (mq masks). q_m = m'(1,3,d,t') + m(d,t',1,2) = 10 + 3|d-1| + 3(6-t'):
        19 (to 2, t' = 4), 25 (to 3, t' = 3, the move of Fig. 4), 28 (to 4, t' = 3), 28 (to 5, t' = 4), 40 (to 6, t' = 1);
        maxMov = 40. q_b: 1 on 2 and 3 (blocks 2 and 4 below are earlier), 0 on 4, 5, 6. q_c: |5-2| = 3, |5-4| = 1,
        0 on 4 and 5 (5 is the earliest there), 4 on the empty stack 6 (alone: r(c) - r(c*) = 5 - 1).
        alpha: 4481/1560, 307/104, 31/30, 67/60, 47/26 (hand calculation by Thom, checked independently)."""
        bay = Bay(FIG4, LOADED)
        self.assertEqual([q_m(bay, 1, d) for d in range(2, 7)], [19, 25, 28, 28, 40])
        terms = {name: dict((d, a) for a, d in alphas(bay, 13, mask))
                 for name, mask in (("m", (1, 0, 0, 0)), ("b", (0, 1, 0, 0)), ("c", (0, 0, 1, 0)), ("r", (0, 0, 0, 1)))}
        self.assertEqual(terms["m"], {2: 19 / 40, 3: 25 / 40, 4: 28 / 40, 5: 28 / 40, 6: 1.0})
        self.assertEqual(terms["b"], {2: 1, 3: 1, 4: 0, 5: 0, 6: 0})
        self.assertEqual(terms["c"], {2: 3 / 13, 3: 1 / 13, 4: 0.0, 5: 0.0, 6: 4 / 13})
        self.assertEqual(terms["r"], {d: d / 6 for d in range(2, 7)})
        total = dict((d, a) for a, d in alphas(bay, 13))
        for d, (num, den) in {2: (4481, 1560), 3: (307, 104), 4: (31, 30), 5: (67, 60), 6: (47, 26)}.items():
            self.assertAlmostEqual(total[d], num / den, places=12)

    def test_alpha_full_stacks_and_q_b(self):
        """1. [1, 5], [2, 7], [3, 14, 15], [4, 6, 8, 9], [10, 11, 12, 13], n_T = 4: stacks 4 and 5 are full.
        Candidates 2 and 3: q_m = (8+2+6) + (3+1+5) = 25 and (8+4+4) + (2+2+5) = 25, so maxMov = 25; the full
        stack 5 would give 16 + 3*4 = 28 and must not enter maxMov. q_b = 1 on both: blocks 2 and 3 below
        are earlier than 5, although the tops 7 and 15 are later."""
        bay = Bay([[1, 5], [2, 7], [3, 14, 15], [4, 6, 8, 9], [10, 11, 12, 13]], LOADED)
        self.assertEqual(dict((d, a) for a, d in alphas(bay, 15, (1, 0, 0, 0))), {2: 1.0, 3: 1.0})
        self.assertEqual(dict((d, a) for a, d in alphas(bay, 15, (0, 1, 0, 0))), {2: 1, 3: 1})

    def test_tau_children(self):
        """2. The tau lowest alpha, ties to the lowest stack index. Fig. 4: alpha 2.87 (2), 2.95 (3), 1.03 (4),
        1.12 (5), 1.81 (6), so tau = 3 gives stacks 4, 5, 6 (not the move drawn in Fig. 4). Tie: [3], [1, 6], [5], [2, 4, 7, 8] (stack 4 full), n_C = 8: 6 goes to 1 or 3, q_m = 28
        both (maxMov term 1), q_b = 1 both; alpha = 1 + 2 + 3/8 + 0.5 * 1/4 = 3.5 on stack 1 and
        1 + 2 + 1/8 + 0.5 * 3/4 = 3.5 on stack 3."""
        bay = Bay(FIG4, LOADED)
        self.assertEqual([d for _, d in children(bay, 1, 13)], [4])
        self.assertEqual([d for _, d in children(bay, 3, 13)], [4, 5, 6])
        self.assertEqual([d for _, d in children(bay, 5, 13)], [4, 5, 6, 2, 3])
        tie = Bay([[3], [1, 6], [5], [2, 4, 7, 8]], LOADED)
        self.assertEqual(children(tie, 3, 8), [(3.5, 1), (3.5, 3)])
        self.assertEqual(children(tie, 1, 8), [(3.5, 1)])

    def test_sequences(self):
        """3. Every sequence ends right after min(kappa, blocks left) retrievals; at most tau different
        next moves after any prefix; the set equals a brute-force enumeration filtered on the tau children."""
        cases = [(Bay(FIG2, LOADED), 13), (Bay(parse_caserta_dat(DEFAULT_CASERTA_DIR / "data3-4-1.dat")[2],
                                               CostModel(n_T=5, **W_LOADED)), 12)]
        end = Bay(FIG2, LOADED)
        for mv in MIN_MOVES[:-2]:
            end.apply(mv)                               # two blocks left: 12 and 13
        cases.append((end, 13))
        for node, n_C in cases:
            for kappa, tau in ((1, 1), (2, 2), (2, 3), (3, 3)):
                seqs = sequences(node, kappa, tau, n_C)
                goal = min(kappa, sum(len(st) for st in node.stacks))
                start = len(node.phi)
                got = [(tuple(b.phi), a) for b, a in seqs]
                for phi, _ in got:
                    added = phi[start:]
                    self.assertEqual(sum(d == 0 for _, d in added), goal)
                    self.assertEqual(added[-1][1], 0)
                branching = {}
                for phi, _ in got:
                    for k in range(start, len(phi)):
                        branching.setdefault(phi[:k], set()).add(phi[k])
                self.assertLessEqual(max(len(v) for v in branching.values()), tau)
                brute = _brute_sequences(node, kappa, tau, n_C)
                self.assertEqual(sorted(got), sorted(brute))

    def test_roulette(self):
        """4. Probabilities proportional to 1 / (sum of alpha): sums 1, 2, 4 give 4/7, 2/7, 1/7; 30 000
        draws with a fixed seed match within 0.01. Only the mu best sequences are ever drawn."""
        self.assertEqual([round(x, 12) for x in roulette([1, 2, 4])], [round(4 / 7, 12), round(2 / 7, 12), round(1 / 7, 12)])
        rng = random.Random(1)
        counts = {"a": 0, "b": 0, "c": 0}
        for _ in range(30_000):
            counts[draw([("a", 1.0), ("b", 2.0), ("c", 4.0)], rng)] += 1
        for key, want in (("a", 4 / 7), ("b", 2 / 7), ("c", 1 / 7)):
            self.assertAlmostEqual(counts[key] / 30_000, want, delta=0.01)
        self.assertEqual(draw([("only", 0.0)], rng), "only")
        root = Bay(FIG2, LOADED)
        seqs = sorted(sequences(root, 2, 3, 13), key=lambda s: s[1])
        self.assertGreater(len(seqs), 2)
        allowed = {tuple(b.phi) for b, _ in seqs[:2]}
        seen = {tuple(expand(root, rng, 2, 3, 2, 13).phi) for _ in range(300)}
        self.assertEqual(seen, allowed)

    def test_seeds(self):
        """5. The same seed gives the same completions; another seed gives different ones (V04)."""
        stacks = parse_caserta_dat(DEFAULT_CASERTA_DIR / "data3-4-1.dat")[2]
        cost = CostModel(n_T=5, **W_LOADED)
        root = Bay(stacks, cost)
        phi, ub = greedy(root)
        nodes, _, _, _ = partial_solutions(root, ub, beta(len(phi), 12), 500, 12, 10 ** 7)

        def completions(seed):
            rng, out = random.Random(seed), []
            for node in nodes[:20]:
                while node.target is not None:
                    node = expand(node, rng, 2, 3, 5, 12)
                out.append((tuple(node.phi), node.g))
            return out

        self.assertEqual(completions(0), completions(0))
        self.assertNotEqual(completions(0), completions(1))

    def test_whole_run(self):
        """6. Fig. 2 and the 40 Caserta 3x3 instances: complete, accepted by validate_plan (train raises
        otherwise), and f(phi_best) <= f(greedy)."""
        cases = [(self.fig2_path(), FIG2, 4)] + [
            (DEFAULT_CASERTA_DIR / f"data3-3-{i}.dat", parse_caserta_dat(DEFAULT_CASERTA_DIR / f"data3-3-{i}.dat")[2], 5)
            for i in range(1, 41)]
        for path, stacks, n_T in cases:
            cfg = _config(path, max_tiers=n_T)
            algo, rec = _run(cfg)
            moves = _replay(cfg, algo.get_best_solution())
            self.assertEqual(sum(d == 0 for _, d in moves), sum(len(s) for s in stacks))
            greedy_f = greedy(Bay(stacks, CostModel(n_T=n_T, **W_LOADED)))[1]
            self.assertLessEqual(rec.extra["operating_cost"], greedy_f + 1e-9)

    def fig2_path(self):
        if not hasattr(self, "_tmp"):
            self._tmp = tempfile.TemporaryDirectory()
            self.addCleanup(self._tmp.cleanup)
        path = Path(self._tmp.name) / "fig2.dat"
        path.write_text("\n".join(["6 13"] + [" ".join(map(str, [len(s)] + s)) for s in FIG2]) + "\n")
        return path


def _config(path, max_tiers=None):
    cfg = problem_config_for_caserta_dat(
        Path(path), {"time_model": "f2", "stack_s_per_stack": TS, "pickup_place_s": TPP})
    if max_tiers:
        cfg.max_tiers = max_tiers
    return cfg


def _run(cfg, seed=0):
    algo = LopezPlata2019Heuristic(AlgorithmConfig(seed=seed, extra={"rho": RHO_TEST}))
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
        orig = LopezPlata2019Heuristic._step2
        for cut in (lambda phi: phi[:-1], lambda phi: phi[1:]):
            def patched(*args, cut=cut):
                phi, f_best = orig(*args)
                return cut(phi), f_best
            with self.subTest(cut=cut), mock.patch.object(LopezPlata2019Heuristic, "_step2", staticmethod(patched)):
                with self.assertRaises(RuntimeError):
                    _run(cfg)



class TestPlatform(unittest.TestCase):
    """Schema, metadata and the path from the UI fields to the steps."""

    PAPER = {"theta": 40, "lam": 500, "node_budget": 10 ** 7, "rho": 300, "kappa": 2, "mu": 5, "tau": 3}

    def test_schema(self):
        schema = LopezPlata2019Heuristic.config_schema()
        self.assertEqual({k: schema[k]["default"] for k in self.PAPER}, self.PAPER)
        self.assertEqual(schema["theta"]["type"], "int")
        self.assertEqual((schema["theta"]["min"], schema["theta"]["max"], schema["lam"]["max"]), (0, 100, 500))
        for key in ("delta", "num_expansions", "max_partial_nodes", "alpha1", "beta1", "alpha2", "beta2",
                    "eta1", "gamma1", "eta2", "gamma2"):
            self.assertNotIn(key, schema)

    def test_metadata(self):
        from core.algorithm_meta import validate_meta
        cls = LopezPlata2019Heuristic
        self.assertEqual(validate_meta(cls), [])
        self.assertEqual((cls.objectives, cls.fidelity, cls.geometry), (["fixed-rule"], "faithful", "single-bay"))
        for word in ("restricted", "operating cost", "L18", "L19"):
            self.assertIn(word, cls.description)
        import algorithms.CRP_Time.heuristic.lopez_plata_2019.algorithm as module
        for author in ("López-Plata", "Expósito-Izquierdo", "Lalla-Ruiz", "Melián-Batista", "Moreno-Vega"):
            self.assertIn(author, module.__doc__)

    def _captured(self, values):
        """Run train() with the configuration the web backend builds from `values` and record what reaches
        step 1 and step 2 (step 2 itself is skipped)."""
        from web.backend.api import _config_values
        from web.backend.config import algorithm_config
        import algorithms.CRP_Time.heuristic.lopez_plata_2019.algorithm as module
        seen = {}

        def step1(root, ub, depth, lam, n_C, node_budget):
            seen.update(depth=depth, lam=lam, node_budget=node_budget, n_C=n_C)
            return partial_solutions(root, ub, depth, lam, n_C, node_budget)

        def step2(partials, phi_best, f_best, rng, rho, kappa, tau, mu, n_C):
            seen.update(rho=rho, kappa=kappa, tau=tau, mu=mu, draw=rng.random())
            return phi_best, f_best

        cfg_a = algorithm_config(_config_values(LopezPlata2019Heuristic.config_schema(), values))
        cfg = _config(DEFAULT_CASERTA_DIR / "data3-4-1.dat")
        with mock.patch.object(module, "partial_solutions", step1), mock.patch.object(
                LopezPlata2019Heuristic, "_step2", staticmethod(step2)):
            LopezPlata2019Heuristic(cfg_a).train(lambda: CRP_Time(config=cfg), queue.Queue(), threading.Event())
        return seen

    def test_ui_defaults_reach_the_steps(self):
        seen = self._captured({})
        stacks = parse_caserta_dat(DEFAULT_CASERTA_DIR / "data3-4-1.dat")[2]
        depth = beta(len(greedy(Bay(stacks, CostModel(n_T=5)))[0]), 12, 40)
        self.assertEqual({k: seen[k] for k in ("depth", "lam", "node_budget", "rho", "kappa", "tau", "mu")},
                         {"depth": depth, **{k: v for k, v in self.PAPER.items() if k != "theta"}})
        self.assertEqual(seen["draw"], random.Random(0).random())

    def test_ui_values_reach_the_steps(self):
        seen = self._captured({"theta": 0, "lam": 7, "node_budget": 12345, "rho": 3, "kappa": 1, "mu": 2,
                               "tau": 4, "seed": 9})
        self.assertEqual({k: seen[k] for k in ("depth", "lam", "node_budget", "rho", "kappa", "tau", "mu")},
                         {"depth": 0, "lam": 7, "node_budget": 12345, "rho": 3, "kappa": 1, "tau": 4, "mu": 2})
        self.assertEqual(seen["draw"], random.Random(9).random())

    def test_ui_rejects_removed_fields(self):
        from web.backend.api import _config_values
        for key in ("delta", "max_partial_nodes", "alpha1", "eta1"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                _config_values(LopezPlata2019Heuristic.config_schema(), {key: 1})

    def test_weights_reach_the_cost_model(self):
        """The code weights (not in the UI) are passed to CostModel: operating_cost is f under the weights
        given, and a weight <= 0 is refused."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "fig2.dat"
        path.write_text("\n".join(["6 13"] + [" ".join(map(str, [len(s)] + s)) for s in FIG2]) + "\n")
        cfg = _config(path, max_tiers=4)
        for w in (W_PRINTED, W_LOADED):
            algo = LopezPlata2019Heuristic(AlgorithmConfig(extra={"rho": RHO_TEST, **w}))
            q_ = queue.Queue()
            algo.train(lambda: CRP_Time(config=cfg), q_, threading.Event())
            moves = _replay(cfg, algo.get_best_solution())
            self.assertAlmostEqual(q_.get_nowait().extra["operating_cost"], f(FIG2, moves, CostModel(n_T=4, **w)))
        with self.assertRaises(AssertionError):
            CostModel(n_T=4, eta1=1, gamma1=0, eta2=2, gamma2=2)
        algo = LopezPlata2019Heuristic(AlgorithmConfig(extra={"rho": RHO_TEST, "eta2": 0}))
        with self.assertRaises(AssertionError):
            algo.train(lambda: CRP_Time(config=cfg), queue.Queue(), threading.Event())


if __name__ == "__main__":
    unittest.main()
