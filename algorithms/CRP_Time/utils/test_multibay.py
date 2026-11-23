"""
Tests of the multi-bay layer (utils/multibay.py) and of shin_time on it (hand-computed values).

  python -m unittest algorithms.CRP_Time.utils.test_multibay      (from the repo root)

Block used throughout: 2 bays x 2 rows, H = 2, bottom to top
    bay 1: row 1 [1, 3], row 2 [2];   bay 2: row 1 [4], row 2 [].
"""

from __future__ import annotations

import unittest

from core.benchmarks.lee_lee import list_lee_lee_files, parse_lee_lee
from core.plan import Movement, RelocationPlan

from algorithms.CRP_Time.utils.multibay import MultiBayLayout, PlanError, load_lee_lee, replay
from algorithms.CRP_Time.utils.shin_time import lb_shin, shin_move_times, shin_time

LAYOUT = MultiBayLayout("hand", 2, 2, 2, {(1, 1): (1, 3), (1, 2): (2,), (2, 1): (4,), (2, 2): ()}, (1, 2, 3, 4))


def plan(*moves):
    p = RelocationPlan()
    for c, src, dst in moves:
        p.add(Movement(c, src, dst))
    return p


# Restricted: 3 to (2, 2), then retrieve 1, 2, 3, 4.
PLAN_A = [(3, (1, 1), (2, 2)), (1, (1, 1), None), (2, (1, 2), None), (3, (2, 2), None), (4, (2, 1), None)]
# Unrestricted: 4 to (2, 2) first (not from the target's stack (1, 1)), then 3 to (2, 1).
PLAN_B = [(4, (2, 1), (2, 2)), (3, (1, 1), (2, 1)), (1, (1, 1), None), (2, (1, 2), None),
          (3, (2, 1), None), (4, (2, 2), None)]


class TestReplay(unittest.TestCase):
    def assertFails(self, moves, step, restricted=False):
        with self.assertRaises(PlanError) as cm:
            replay(plan(*moves), LAYOUT, restricted)
        self.assertEqual(cm.exception.step, step)

    def test_valid_plans(self):
        self.assertEqual(replay(plan(*PLAN_A), LAYOUT, restricted=True), (1, 4, 0))
        self.assertEqual(replay(plan(*PLAN_B), LAYOUT), (2, 4, 1))

    def test_only_top_moves(self):
        self.assertFails([(1, (1, 1), None)], 0)                       # 3 lies on 1
        self.assertFails([(3, (1, 1), (2, 2)), (4, (2, 2), (1, 1))], 1)  # 4 is not on (2, 2)

    def test_height(self):
        self.assertFails([(3, (1, 1), (1, 2)), (4, (2, 1), (1, 2))], 1)  # (1, 2) already holds 2 = H

    def test_retrieval_order(self):
        self.assertFails([(2, (1, 2), None)], 0)

    def test_empty_at_end(self):
        self.assertFails(PLAN_A[:-1], 4)

    def test_destination_equals_source(self):
        self.assertFails([(3, (1, 1), (1, 1))], 0)

    def test_restricted(self):
        self.assertFails(PLAN_B, 0, restricted=True)


class TestShinTime(unittest.TestCase):
    def test_plan_a_by_hand(self):
        """First trip free; gamma_row 1.2, gamma_bay 3.5, gamma_acc 40, gamma_pd 30.
        3 (1,1)->(2,2): 0 + (40 + 3.5 + 1.2) + 30 = 74.7, crane at (2,2)
        1 from (1,1): (40 + 3.5 + 1.2) + 1.2 + 30 = 75.9, crane at (1,0)
        2 from (1,2): 2.4 + 2.4 + 30 = 34.8, crane at (1,0)
        3 from (2,2): (40 + 3.5 + 2.4) + 2.4 + 30 = 78.3, crane at (2,0)
        4 from (2,1): 1.2 + 1.2 + 30 = 32.4. Total 296.1."""
        times = shin_move_times(plan(*PLAN_A), LAYOUT)
        for got, want in zip(times, [74.7, 75.9, 34.8, 78.3, 32.4]):
            self.assertAlmostEqual(got, want)
        self.assertAlmostEqual(shin_time(plan(*PLAN_A), LAYOUT), 296.1)

    def test_invalid_plan_refused(self):
        with self.assertRaises(PlanError):
            shin_time(plan(*PLAN_A[:-1]), LAYOUT)


@unittest.skipUnless(list_lee_lee_files(), "benchmark/lee_lee is missing")
class TestLeeLeeLayout(unittest.TestCase):
    def test_orders_match_reader(self):
        for path in list_lee_lee_files()[:5]:
            layout = load_lee_lee(path)
            self.assertEqual(layout.orders(), parse_lee_lee(path).ids)   # group = id in every file
            self.assertEqual(lb_shin(layout.orders()), lb_shin(parse_lee_lee(path).ids))


if __name__ == "__main__":
    unittest.main()
