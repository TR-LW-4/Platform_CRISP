"""
Tests of the Lin, Lee & Lee (2015) heuristic on the multi-bay layer.

  python -m unittest algorithms.CRP_Time.crp_time_u.lin_2015.test_lin_2015      (from the repo root)
"""

from __future__ import annotations

import random
import unittest
from unittest import mock

from algorithms.CRP_Time.crp_time_u.lin_2015.heuristic import solve
from algorithms.CRP_Time.utils.multibay import MultiBayLayout, check_multibay_plan, load_lee_lee
from core.benchmarks.lee_lee import list_lee_lee_files


def layout(name, n_bays, n_rows, H, stacks):
    order = tuple(sorted(c for s in stacks.values() for c in s))
    full = {(b, r): tuple(stacks.get((b, r), ())) for b in range(1, n_bays + 1) for r in range(1, n_rows + 1)}
    return MultiBayLayout(name, n_bays, n_rows, H, full, order)


def moves(plan):
    return [(m.container_id, m.from_pos, m.to_pos) for m in plan.movements]


class FirstOfTies(random.Random):
    """Resolves every tie to the first of the sorted candidates, i.e. the stack with the lowest (bay, row)."""
    def choice(self, seq):
        return seq[0]


# Worked example, pp. 138-139: one bay, 6 rows, H = 6, 12 containers (ID = group), Pr = Pb = 0.
# Fig. 4 (p. 138), stacks bottom to top, as read from the figure by Thom (23 Nov 2026); equal to the earlier
# reconstruction from Fig. 8, and consistent with the text: after retrieving 1, ming(1) = 9, ming(4) = 10, ming(6) = 8.
FIG4 = layout("Lin Fig. 4", 1, 6, 6, {(1, 1): (9, 12, 1), (1, 2): (5, 2, 6), (1, 3): (4, 7), (1, 4): (10,),
                                      (1, 5): (3, 11), (1, 6): (8,)})
# Fig. 8 (p. 139), (c, s, U), (c, s', D) pairs as moves; s' = 0 is the truck (None here).
FIG8 = [(1, 1, 0), (7, 3, 6), (6, 2, 6), (2, 2, 0), (11, 5, 4), (3, 5, 0), (4, 3, 0), (5, 2, 0), (6, 6, 0), (7, 6, 0),
        (8, 6, 0), (12, 1, 2), (9, 1, 0), (11, 4, 2), (10, 4, 0), (11, 2, 0), (12, 2, 0)]
FIG8_MOVES = [(c, (1, s), None if t == 0 else (1, t)) for c, s, t in FIG8]


class TestWorkedExample(unittest.TestCase):
    def test_fig8_move_by_move(self):
        """Iteration 2: Rule I picks stack 6 (ming 8; ming(1) = 9, ming(4) = 10), Rule II moves k = 7 into it before 6.
        Iteration 3: Rule III moves 11 to stack 4 (largest ming, 10). Iteration 9: 12 blocks 9; the empty stacks 2,
        3, 5, 6 all have SSI = G + 1 = 13 (Pr = Pb = 0), a tie the paper does not address; Fig. 8 takes stack 2,
        which is the lowest position, so ties are resolved to the lowest position here. Iteration 10: 11 blocks 10;
        stack 2 (ming 12) beats the empty stacks (13)."""
        with mock.patch("algorithms.CRP_Time.crp_time_u.lin_2015.heuristic.random.Random", FirstOfTies):
            plan = solve(FIG4, P_r=0, P_b=0)
        self.assertEqual(moves(plan), FIG8_MOVES)

    def test_seeded_ties(self):
        """With seeded ties every plan is valid, has Fig. 8's 17 moves, and equals Fig. 8 up to the tie (move 11)."""
        for seed in range(10):
            plan = solve(FIG4, P_r=0, P_b=0, seed=seed)
            check_multibay_plan(plan, FIG4)
            self.assertEqual(len(plan.movements), 17)
            self.assertEqual(moves(plan)[:11], FIG8_MOVES[:11])
            self.assertIn(plan.movements[11].to_pos, [(1, 2), (1, 3), (1, 5), (1, 6)])


class TestRules(unittest.TestCase):
    def test_rule_i_bay_penalty(self):
        """2 bays x 2 rows, H = 2: (1,1) [1, 3], (2,2) [2]; G = 3, an empty stack has ming 4.
        Target 1 is blocked by d = 3. Ideal stacks (ming > 3, not full, not s(c)): (1,2) and (2,1), both empty.
        Pr = 30, Pb = 300: SSI(1,2) = 4 + 30*2 + 0 = 64, SSI(2,1) = 4 + 30*1 + 300*1 = 334 -> (1,2).
        Pb = 0: SSI(1,2) = 64, SSI(2,1) = 34 -> (2,1).
        Rule II: the only other top is 2, which blocks nothing (type A) -> no pre-relocation."""
        lay = layout("rule I", 2, 2, 2, {(1, 1): (1, 3), (2, 2): (2,)})
        self.assertEqual(moves(solve(lay))[0], (3, (1, 1), (1, 2)))
        self.assertEqual(moves(solve(lay, P_b=0))[0], (3, (1, 1), (2, 1)))

    def test_rule_ii_twice(self):
        """1 bay x 4 rows, H = 4: (1,1) [5, 1, 4], (1,2) [2, 7], (1,3) [3, 6], (1,4) [8]; n = G = 8.
        Target 1 is blocked by d = 4. Not full and not s(c): (1,2) ming 2, (1,3) ming 3, (1,4) ming 8; only (1,4) has
        ming > 4, so Rule I gives t = (1,4).
        Rule II, 1st pass: h(t) = 1 <= H - 2; ming(t) = 8. Tops: 7 on (1,2) blocks 2, 7 > 4, 3 < 7 < 8; 6 on (1,3)
        blocks 3, 6 > 4, 3 < 6 < 8; the largest is 7 -> 7 into t.
        2nd pass (Fig. 3 loop): h(t) = 2 <= H - 2; ming(t) = 7. 6: blocks 3, 6 > 4, 2 < 6 < 7 -> 6 into t.
        3rd check: h(t) = 3 > H - 2 (one empty slot) -> stop; d = 4 into t (h becomes 4 = H).
        Then 1, 2, 3, 4, 5, 6, 7, 8 are all on top when their turn comes."""
        lay = layout("rule II twice", 1, 4, 4, {(1, 1): (5, 1, 4), (1, 2): (2, 7), (1, 3): (3, 6), (1, 4): (8,)})
        self.assertEqual(moves(solve(lay)), [(7, (1, 2), (1, 4)), (6, (1, 3), (1, 4)), (4, (1, 1), (1, 4)),
                                             (1, (1, 1), None), (2, (1, 2), None), (3, (1, 3), None),
                                             (4, (1, 4), None), (5, (1, 1), None), (6, (1, 4), None),
                                             (7, (1, 4), None), (8, (1, 4), None)])

    def test_rule_ii_into_empty_stack(self):
        """1 bay x 3 rows, H = 3: (1,1) [1, 4], (1,2) [3, 2, 5], (1,3) []; n = G = 5, an empty stack has ming 6.
        Target 1 is blocked by d = 4. (1,2) is full, so the only ideal stack is (1,3) (ming 6 > 4): t = (1,3).
        h(t) = 0 <= H - 2, so Rule II applies: k = 5 is on top of (1,2), blocks 2 and 3 (type B), g(5) > g(4), and
        ming(t) - 5 = 1 < 5 < 6 = ming(t). So 5 goes to (1,3) first, then 4; then 1, 2, 3, 4, 5 are retrieved.
        (With an empty stack's min group set to 10 n = 50, as in CRP_RL lin2015.py, condition 4 would need
        45 < g(k) < 50 and no pre-relocation would be made.)"""
        lay = layout("rule II", 1, 3, 3, {(1, 1): (1, 4), (1, 2): (3, 2, 5)})
        self.assertEqual(moves(solve(lay)), [(5, (1, 2), (1, 3)), (4, (1, 1), (1, 3)), (1, (1, 1), None),
                                             (2, (1, 2), None), (3, (1, 2), None), (4, (1, 3), None),
                                             (5, (1, 3), None)])


@unittest.skipUnless(list_lee_lee_files(), "benchmark/lee_lee is missing")
class TestLeeLeeExample1(unittest.TestCase):
    def test_valid_plan(self):
        """Lee & Lee's Example 1 (2 bays x 3 stacks, not part of the benchmark sets): solve returns a plan that
        check_multibay_plan accepts (solve checks it too). No comparison with any reported value."""
        path = next(p for p in list_lee_lee_files() if p.stem == "R020306_0020_001")
        lay = load_lee_lee(path)
        summary = check_multibay_plan(solve(lay), lay)
        self.assertEqual(summary.retrievals, 20)


if __name__ == "__main__":
    unittest.main()
