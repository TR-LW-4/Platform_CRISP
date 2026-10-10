"""
Checks for JovanovicACO_CRPTime: the crane-time lower bounds against the exact
remaining time in every reachable state (docs/validation/jovanovic_2019.md §3.3).

  python -m unittest algorithms.CRP_Time.heuristic.jovanovic_2019_aco.test_jovanovic   (from the repo root)

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
corresponding original algorithm paper.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import math
import random
import unittest

from algorithms.CRP_Time.utils.crane_time import move_time
from algorithms.CRP_Time.utils.located import non_located
from core.benchmarks.caserta import DEFAULT_CASERTA_DIR, parse_caserta_dat
from core.objectives import ObjectiveSpec
from core.plan import Movement

from .lower_bound import lb_time

# CRISP defaults (Voß & Schwarze 2019), O_f,5 of Table 5, tpp = 0, horizontal and vertical dominating.
CRISP = dict(stack_s_per_stack=1.2, pickup_place_s=30.0, empty_vertical_s_per_tier=2.59,
             loaded_vertical_s_per_tier=5.18, outside_height=1.5)
PARAMETER_SETS = {
    "crisp": CRISP,
    "tpp5": {**CRISP, "pickup_place_s": 5.0},
    "tpp0": dict(stack_s_per_stack=2.35, pickup_place_s=0.0, empty_vertical_s_per_tier=3.33,
                 loaded_vertical_s_per_tier=1.07, outside_height=0.75),
    "wide": dict(stack_s_per_stack=50.0, pickup_place_s=0.5, empty_vertical_s_per_tier=0.01,
                 loaded_vertical_s_per_tier=0.02, outside_height=0.0),
    "tall": dict(stack_s_per_stack=0.01, pickup_place_s=0.0, empty_vertical_s_per_tier=40.0,
                 loaded_vertical_s_per_tier=80.0, outside_height=3.0),
}
FIG2 = [[9, 1, 8], [11, 3, 2], [5, 4], [13, 6], [12, 10, 7], []]   # López-Plata et al. (2019), Fig. 2
COUNTER = [[], [], [1, 2]]                                          # §3.6: the paper's LB_f fails here
EPS = 1e-9   # the bound and the exact time are float sums in different orders


def _spec(model, n_T, params):
    return ObjectiveSpec(time_model=model, max_tiers=n_T, num_rows=1, **params)


def _cases():
    """(stacks, H): Fig. 2, the counterexample, the 40 Caserta 3x3 instances with H = T + 2, and 20 random
    layouts."""
    rng = random.Random(11)
    cases = [(FIG2, 4), (COUNTER, 2)] + [
        (parse_caserta_dat(DEFAULT_CASERTA_DIR / f"data3-3-{i}.dat")[2], 5) for i in range(1, 41)]
    for _ in range(20):
        n_S, n_T = rng.randint(2, 4), rng.randint(2, 4)
        n = rng.randint(1, (n_S - 1) * n_T)
        prios = list(range(1, n + 1))
        rng.shuffle(prios)
        stacks = [[] for _ in range(n_S)]
        for c in prios:
            stacks[rng.choice([i for i in range(n_S) if len(stacks[i]) < n_T])].append(c)
        cases.append((stacks, n_T))
    return cases


def _remaining_time(stacks, spec):
    """Exact remaining time of every configuration reachable from stacks under the restricted rule, with the
    per-move times of the shared evaluator. f2 and f2vert do not depend on the crane position, so the
    configuration is the state. Configurations that cannot be emptied get infinity."""
    memo = {}

    def V(st):
        if st in memo:
            return memo[st]
        best = 0.0 if not any(st) else math.inf
        if any(st):
            target = min(c for x in st for c in x)
            s = next(i for i, x in enumerate(st) if target in x)
            top = st[s][-1]
            if top == target:
                options = [None]
            else:
                options = [d for d in range(len(st)) if d != s and len(st[d]) < spec.max_tiers]
            for d in options:
                new = list(st)
                new[s] = st[s][:-1]
                if d is None:
                    move = Movement(top, (s + 1, 1), None, from_tier=len(st[s]))
                else:
                    new[d] = st[d] + (top,)
                    move = Movement(top, (s + 1, 1), (d + 1, 1), from_tier=len(st[s]), to_tier=len(new[d]))
                best = min(best, move_time(move, spec) + V(tuple(new)))
        memo[st] = best
        return best

    V(tuple(tuple(x) for x in stacks))
    return memo


def _literal_lb_f(stacks, spec):
    """Mutant: the paper's LB_f as printed, with tpp + ts for every non-well-located container."""
    ts, tpp = spec.stack_s_per_stack, spec.pickup_place_s
    return sum(tpp + 2.0 * s * ts + ((tpp + ts) if nw else 0.0)
               for s, stack in enumerate(stacks, 1) for nw in non_located(stack))


def _violations(model, params, bound):
    """Number of reachable states with bound > exact remaining time, and the number of states checked."""
    bad = total = 0
    for stacks, n_T in _cases():
        spec = _spec(model, n_T, params)
        for st, v in _remaining_time(stacks, spec).items():
            if not math.isinf(v):
                total += 1
                bad += bound([list(x) for x in st], spec) > v + EPS
    return bad, total


class TestLowerBound(unittest.TestCase):

    def test_lb_f_is_a_lower_bound(self):
        """LB_f (the paper's without the ts term) <= the exact remaining time in every reachable state,
        for every parameter set."""
        for name, params in PARAMETER_SETS.items():
            with self.subTest(params=name):
                bad, total = _violations("f2", params, lb_time)
                self.assertGreater(total, 5000)
                self.assertEqual(bad, 0)

    def test_literal_eq44_fails(self):
        """The mutant: the paper's LB_f as printed exceeds the exact remaining time in some state, for every parameter
        set (all have ts > 0)."""
        for name, params in PARAMETER_SETS.items():
            with self.subTest(params=name):
                self.assertGreater(_violations("f2", params, _literal_lb_f)[0], 0)

    def test_by_hand(self):
        """Counterexample of §3.6 (H = 2), CRISP defaults: relocate 2 to stack 1, retrieve 1 and 2 costs
        (4 ts + tpp) + (6 ts + tpp) + (2 ts + tpp) = 12 ts + 3 tpp = 104.4, the optimum. LB_f: container 1
        (6 ts + tpp), container 2 (6 ts + tpp) plus tpp as non-well-located = 104.4; the paper's LB_f adds
        ts: 105.6. Fig. 2 (H = 4): only container 8 is non-well-located; the sum of s over the 13 containers
        is 38, so LB_f = 2 ts 38 + 14 tpp = 511.2."""
        spec = _spec("f2", 2, CRISP)
        v = _remaining_time(COUNTER, spec)[tuple(tuple(x) for x in COUNTER)]
        self.assertAlmostEqual(v, 104.4, places=9)
        self.assertAlmostEqual(lb_time(COUNTER, spec), 104.4, places=9)
        self.assertAlmostEqual(_literal_lb_f(COUNTER, spec), 105.6, places=9)
        self.assertAlmostEqual(lb_time(FIG2, _spec("f2", 4, CRISP)), 511.2, places=9)

    def test_lb_v_unchanged(self):
        """LB_v (as in the paper, not changed) holds in every state with the CRISP defaults, tpp = 0 and
        vertical time dominating, and fails when horizontal time dominates (§3.6)."""
        for name in ("crisp", "tpp0", "tall"):
            with self.subTest(params=name):
                self.assertEqual(_violations("f2_vertical", PARAMETER_SETS[name], lb_time)[0], 0)
        self.assertGreater(_violations("f2_vertical", PARAMETER_SETS["wide"], lb_time)[0], 0)

    def test_metadata(self):
        """The label stays faithful; the deviation in LB_f is named in the description."""
        from .algorithm import JovanovicACO_CRPTime as cls
        self.assertEqual(cls.fidelity, "faithful")
        for words in ("Deviation from the paper", "LB_f without the extra ts"):
            self.assertIn(words, cls.description)


if __name__ == "__main__":
    unittest.main()
