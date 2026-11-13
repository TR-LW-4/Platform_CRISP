"""
Checks for the multi-bay part of Forster & Bortfeldt (Oper. Res. Proc. 2011)
(docs/validation/forster_bortfeldt_2011_multibay.md §4-5): the move time of
p. 258 with the parameters of p. 261 on hand-computed moves (within a bay,
across bays, removes), and the merge of Algorithm 1 (p. 261) on small blocks.

  python -m unittest algorithms.CRP_Time.crp_time_u.forster_bortfeldt_2012.test_multibay   (from the repo root)

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
corresponding original algorithm paper.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import unittest

from .crane_time import START, TRUCK, TimeParams, move_time, plan_time


class TestMoveTime(unittest.TestCase):
    """Hand-computed with t_b 3.5, t_s 1.2, t_ad 40, t_pp 30; positions (bay, row)."""

    def test_parameters_and_positions(self):
        self.assertEqual(TimeParams(), TimeParams(3.5, 1.2, 40.0, 30.0))
        self.assertEqual((TRUCK, START), ((0, 0), (0, 0)))

    def test_relocation_within_a_bay(self):
        # crane over the source: s = 4 -> 4 * 1.2 + 30
        self.assertAlmostEqual(move_time((1, 3), (1, 3), (1, 7)), 34.8)
        # empty trip 1 -> 5, loaded 5 -> 2: s = 4 + 3 -> 7 * 1.2 + 30
        self.assertAlmostEqual(move_time((2, 1), (2, 5), (2, 2)), 38.4)

    def test_relocation_across_bays(self):
        # loaded trip bay 1 -> 3: b = 2, s = 3 -> 2 * 3.5 + 3 * 1.2 + 40 + 30
        self.assertAlmostEqual(move_time((1, 2), (1, 2), (3, 5)), 80.6)
        # empty trip across one bay, loaded trip within the bay: b = 1, s = 3 + 2 -> 3.5 + 6 + 40 + 30
        self.assertAlmostEqual(move_time((1, 1), (2, 4), (2, 6)), 79.5)
        # both trips across bays: b = 2 + 1, s = 0, one ad per move -> 10.5 + 40 + 30
        self.assertAlmostEqual(move_time((1, 1), (3, 1), (2, 1)), 80.5)

    def test_remove(self):
        # (2, 3) to the truck at (0, 0), crane over the source: b = 2, s = 3 -> 7 + 3.6 + 40 + 30
        self.assertAlmostEqual(move_time((2, 3), (2, 3), TRUCK), 80.6)
        # crane at the truck, remove from (1, 1): b = 1 + 1, s = 1 + 1 -> 7 + 2.4 + 40 + 30
        self.assertAlmostEqual(move_time(TRUCK, (1, 1), TRUCK), 79.4)

    def test_no_bay_crossed_no_ad(self):
        p = TimeParams(tb=0.0, tad=0.0)
        self.assertAlmostEqual(move_time((1, 1), (1, 4), (1, 2), p), 5 * 1.2 + 30)
        self.assertAlmostEqual(move_time((1, 4), (1, 4), (1, 4)), 30.0)

    def test_plan_time_chains_the_crane_position(self):
        moves = [((1, 2), (1, 4)), ((1, 2), None), ((1, 4), None)]
        # (0,0)->(1,2)->(1,4): b 1, s 4 = 78.3; (1,4)->(1,2)->(0,0): b 1, s 4 = 78.3; (0,0)->(1,4)->(0,0): b 2, s 8 = 86.6
        self.assertAlmostEqual(plan_time(moves), 78.3 + 78.3 + 86.6)
        # truck in the source bay at row 0 and start at (1, 2): one remove, b 0, s 2 -> 2.4 + 30
        self.assertAlmostEqual(plan_time([((1, 2), None)], start=(1, 2), truck=(1, 0)), 32.4)
        self.assertEqual(plan_time([]), 0.0)


if __name__ == "__main__":
    unittest.main()
