"""
Tests of the Shin et al. (2026) crane-time functions (hand-computed values).

  python -m unittest algorithms.CRP_Time.utils.test_shin_time      (from the repo root)
"""

from __future__ import annotations

import unittest

from algorithms.CRP_Time.utils.shin_time import lb_shin, trip


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


if __name__ == "__main__":
    unittest.main()
