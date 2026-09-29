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

import unittest

from .operating_cost import Bay, CostModel, f

# Fig. 2 (p. 438): n_S = 6, n_T = 4, stacks bottom -> top.
FIG2 = [[9, 1, 8], [11, 3, 2], [5, 4], [13, 6], [12, 10, 7], []]
# Relocate 8 from stack 1 to 6, then retrieve 1..13.
MIN_MOVES = [(1, 6), (1, 0), (2, 0), (2, 0), (3, 0), (3, 0), (4, 0), (5, 0),
             (6, 0), (1, 0), (5, 0), (2, 0), (5, 0), (4, 0)]
# Relocate 8 from 1 to 2, retrieve 1, relocate 8 from 2 to 1, then retrieve 2..13.
MIN_COST = [(1, 2), (1, 0), (2, 1), (2, 0), (2, 0), (3, 0), (3, 0), (4, 0),
            (5, 0), (1, 0), (1, 0), (5, 0), (2, 0), (5, 0), (4, 0)]

PRINTED = CostModel(n_T=4)                                          # eta1 = gamma1 = 2, eta2 = gamma2 = 1
SENTENCE = CostModel(n_T=4, eta1=1, gamma1=1, eta2=2, gamma2=2)     # "costs twice when loaded"


class TestOperatingCost(unittest.TestCase):

    def test_fig2(self):
        """V01: 531 (14 moves) and 513 (15 moves), p. 438."""
        self.assertEqual((len(MIN_MOVES), f(FIG2, MIN_MOVES, PRINTED)), (14, 531))
        self.assertEqual((len(MIN_COST), f(FIG2, MIN_COST, PRINTED)), (15, 513))

    def test_fig2_both_weight_readings(self):
        """O01 does not change f of a restricted plan (V02 in the log)."""
        self.assertEqual(f(FIG2, MIN_MOVES, SENTENCE), 531)
        self.assertEqual(f(FIG2, MIN_COST, SENTENCE), 513)

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
            ([[1, 3], [4, 5], []], CostModel(n_T=2), [], (1, 2)),   # stack 2 is full
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
            Bay(FIG2, CostModel(n_T=2))


if __name__ == "__main__":
    unittest.main()
