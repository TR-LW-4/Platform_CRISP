"""
Tests of the Shin et al. (2026) crane-time functions (hand-computed values).

  python -m unittest algorithms.CRP_Time.utils.test_shin_time      (from the repo root)
"""

from __future__ import annotations

import unittest

from core.plan import Movement, RelocationPlan

from algorithms.CRP_Time.utils.crane_time import ObjectiveSpec, movement_f2, movement_f2_vertical
from algorithms.CRP_Time.utils.multibay import MultiBayLayout, PlanError
from algorithms.CRP_Time.utils.shin_time import lb_shin, shin_f2vert, shin_f2vert_parts, shin_move_times, shin_time, trip


class TestTrip(unittest.TestCase):
    def test_eq1(self):
        self.assertAlmostEqual(trip((1, 3), (1, 0)), 3.6)                 # 3 rows
        self.assertAlmostEqual(trip((1, 0), (3, 2)), 40 + 7.0 + 2.4)       # 2 bays, 2 rows
        self.assertEqual(trip((2, 5), (2, 5)), 0.0)


class TestLowerBound(unittest.TestCase):
    def test_theorem2_by_hand(self):
        """Bay 1: row 1 [3, 1], row 2 [2]; bay 2: row 1 [4], row 2 [5, 6] (bottom to top).

        Positions: 1 (1,1), 2 (1,2), 3 (1,1), 4 (2,1), 5 (2,2), 6 (2,2).
        LB_ret, n = 1..5: 1.2 + 2.4 | 2.4 + 1.2 | 1.2 + (40 + 3.5 + 1.2) | 1.2 + 2.4 | 2.4 + 2.4,
        container 6: 2.4, plus 6 * 30: 3.6 + 3.6 + 45.9 + 3.6 + 4.8 + 2.4 + 180 = 243.9.
        LB_rel: only 6 has a smaller order below it: (2 * 1.2 + 30) * 1 = 32.4. LB = 276.3.
        """
        layout = (((3, 1), (2,)), ((4,), (5, 6)))
        self.assertAlmostEqual(lb_shin(layout), 276.3)

    def test_duplicate_order_refused(self):
        with self.assertRaises(ValueError):
            lb_shin((((1, 2), (2,)),))


def _plan(*moves):
    p = RelocationPlan()
    for c, src, dst in moves:
        p.add(Movement(c, src, dst))
    return p


class TestShinF2vert(unittest.TestCase):
    # One bay, rows 1-3, H = 3, bottom to top: row 1 [1, 3], row 2 [2], row 3 [].
    BAY = MultiBayLayout("one bay", 1, 3, 3, {(1, 1): (1, 3), (1, 2): (2,), (1, 3): ()}, (1, 2, 3))
    PLAN = [(3, (1, 1), (1, 3)), (1, (1, 1), None), (2, (1, 2), None), (3, (1, 3), None)]
    TIERS = [(2, 1), (1, None), (1, None), (1, None)]          # (from_tier, to_tier), by hand

    def test_single_bay_against_f2vert(self):
        """Vertical part, move by move, equals single-bay f2vert with the horizontal time set to 0 (stack_s_per_stack
        = 0): h_max = 4, tr0 + tr1 = 7.77, h_out = 1.5: 5 x 7.77 = 38.85, then 5.5 x 7.77 = 42.735 three times.

        Horizontal parts. Rows s_i (source) and e_i (end; 0 = the retrieval point, Shin's row 0 = f2's stack 0); ts =
        gamma_row = 1.2 and no bay term in one bay.
        - Shin: h_i = ts (|e_(i-1) - s_i| + |s_i - e_i|), with e_0 = s_1 (first trip free).
        - f2:   h_i = 2 ts |s_i - e_i| (movement_f2 without tpp): every move is charged as a round trip.
        So shin - f2 = ts (sum_(i>=2) |e_(i-1) - s_i| - sum_i |s_i - e_i|). This is not only a start-position term:
        f2 charges each move's empty return to its own source, Shin the empty trip from the previous end to the next
        source; only the first move (Shin: free) is a start-position difference.
        Here s = 1, 1, 2, 3 and e = 3, 0, 0, 0: Shin 2.4 + 3.6 + 4.8 + 7.2 = 18.0, f2 4.8 + 2.4 + 4.8 + 7.2 = 19.2;
        formula 1.2 x ((2 + 2 + 3) - (2 + 1 + 2 + 3)) = -1.2. Totals 185.055 and 186.255."""
        plan = _plan(*self.PLAN)
        parts = shin_f2vert_parts(plan, self.BAY)
        spec_v = ObjectiveSpec(stack_s_per_stack=0.0, max_tiers=3, num_rows=3)
        spec_h = ObjectiveSpec(pickup_place_s=0.0, max_tiers=3, num_rows=3)
        f2v_h, f2v_v = [], []
        for (c, src, dst), (ft, tt) in zip(self.PLAN, self.TIERS):
            m = Movement(c, src, dst, from_tier=ft, to_tier=tt)
            f2v_v.append(movement_f2_vertical(m, spec_v))
            f2v_h.append(movement_f2(m, spec_h))
        for (h, v), want in zip(parts, f2v_v):
            self.assertAlmostEqual(v, want)
        for got, want in zip(f2v_v, [38.85, 42.735, 42.735, 42.735]):
            self.assertAlmostEqual(got, want)
        for (h, v), want in zip(parts, [2.4, 3.6, 4.8, 7.2]):
            self.assertAlmostEqual(h, want)
        s, e = [1, 1, 2, 3], [3, 0, 0, 0]
        expected = 1.2 * (sum(abs(e[i - 1] - s[i]) for i in range(1, 4)) - sum(abs(a - b) for a, b in zip(s, e)))
        self.assertAlmostEqual(expected, -1.2)
        self.assertAlmostEqual(shin_f2vert(plan, self.BAY) - (sum(f2v_h) + sum(f2v_v)), expected)
        self.assertAlmostEqual(shin_f2vert(plan, self.BAY), 185.055)

    def test_horizontal_equals_shin_time_without_pd(self):
        """Multi-bay: the horizontal parts are shin_move_times minus gamma_pd, move by move."""
        lay = MultiBayLayout("hand", 2, 2, 2, {(1, 1): (1, 3), (1, 2): (2,), (2, 1): (4,), (2, 2): ()}, (1, 2, 3, 4))
        plan = _plan((3, (1, 1), (2, 2)), (1, (1, 1), None), (2, (1, 2), None), (3, (2, 2), None), (4, (2, 1), None))
        for (h, _v), st in zip(shin_f2vert_parts(plan, lay), shin_move_times(plan, lay)):
            self.assertAlmostEqual(h, st - 30.0)

    def test_invalid_plan_refused(self):
        with self.assertRaises(PlanError):
            shin_f2vert(_plan(*self.PLAN[:-1]), self.BAY)


if __name__ == "__main__":
    unittest.main()
