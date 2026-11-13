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

from core.benchmarks.lee_lee import DEFAULT_LEE_LEE_DIR, parse_lee_lee

from .crane_time import START, TRUCK, TimeParams, move_time, plan_time
from .layout import lower_bound
from .multibay import check_block_plan, merge, ranked, solve_bay, solve_block, to_positions
from .plan import check_plan
from .search import Params, TreeSearch
from .state import Bay

EXAMPLE1_FILE = next(DEFAULT_LEE_LEE_DIR.rglob("R020306_0020_001.txt"), None) if DEFAULT_LEE_LEE_DIR.is_dir() else None
# Two hand-made 2-bay blocks (bottom to top), groups unique per block.
BLOCK_A = ([[[5, 1, 9], [3], [12, 7]], [[2, 10], [8, 4, 11], [6]]], 4)
BLOCK_B = ([[[4, 2], [1, 6]], [[3, 5], []]], 3)
NO_LIMIT = Params(time_limit=None)

# Three bays of two stacks, H = 3, groups 1-8, with hand-made bay solutions (d, r, g).
BLOCK = [[[3, 6], [1]], [[2, 5], [4]], [[7], [8]]]
SOLUTIONS = [
    [(1, None, 1), (0, 1, 6), (0, None, 3), (1, None, 6)],
    [(0, 1, 5), (0, None, 2), (1, 0, 5), (1, None, 4), (0, None, 5)],
    [(0, None, 7), (1, None, 8)],
]
MERGED = [((0, 1), None, 1),                                   # group 1: bay 0
          ((1, 0), (1, 1), 5), ((1, 0), None, 2),              # group 2: bay 1, relocation first
          ((0, 0), (0, 1), 6), ((0, 0), None, 3),              # group 3: bay 0
          ((1, 1), (1, 0), 5), ((1, 1), None, 4),              # group 4: bay 1
          ((1, 0), None, 5),                                   # group 5: bay 1
          ((0, 1), None, 6),                                   # group 6: bay 0
          ((2, 0), None, 7), ((2, 1), None, 8)]                # groups 7, 8: bay 2
FIG6 = [[[10, 4, 18], [9, 17, 8, 20], [5, 19, 2]], [[16, 14, 7, 13, 3, 6], [1, 11], [15, 12]]]
EXAMPLE1 = [(11, 5, 6), (1, 5, 0), (6, 4, 2), (2, 3, 0), (3, 4, 0), (19, 3, 5), (18, 1, 5), (4, 1, 0), (5, 3, 0),
            (6, 2, 0), (13, 4, 5), (7, 4, 0), (20, 2, 3), (8, 2, 0), (17, 2, 3), (9, 2, 0), (10, 1, 0), (11, 6, 0),
            (12, 6, 0), (13, 5, 0), (14, 4, 0), (15, 6, 0), (16, 4, 0), (17, 3, 0), (18, 5, 0), (19, 5, 0), (20, 3, 0)]


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


class TestMerge(unittest.TestCase):
    def test_bay_solutions_are_valid(self):
        for bay, sol in zip(BLOCK, SOLUTIONS):
            check_plan(bay, 3, sol)

    def test_merged_sequence(self):
        self.assertEqual(merge(BLOCK, SOLUTIONS), MERGED)

    def test_removes_in_group_order_and_block_empty(self):
        plan = merge(BLOCK, SOLUTIONS)
        self.assertEqual([g for _s, dst, g in plan if dst is None], list(range(1, 9)))
        self.assertEqual(check_block_plan(BLOCK, 3, plan), 3)

    def test_each_bay_keeps_its_own_move_order(self):
        plan = merge(BLOCK, SOLUTIONS)
        for b, sol in enumerate(SOLUTIONS):
            own = [(d, None if dst is None else dst[1], g) for (bb, d), dst, g in plan if bb == b]
            self.assertEqual(own, sol)
            self.assertTrue(all(dst is None or dst[0] == b for (bb, _d), dst, _g in plan if bb == b))

    def test_one_remove_per_bay_and_pass(self):
        # lowestGroupBlock is fixed per pass: every bay holding it gives one remove, in bay order (Algorithm 1)
        self.assertEqual(merge([[[1], [2]], [[1]]], [[(0, None, 1), (1, None, 2)], [(0, None, 1)]]),
                         [((0, 0), None, 1), ((1, 0), None, 1), ((0, 1), None, 2)])
        self.assertEqual(merge([[[1, 1]], [[1]]], [[(0, None, 1), (0, None, 1)], [(0, None, 1)]]),
                         [((0, 0), None, 1), ((1, 0), None, 1), ((0, 0), None, 1)])

    def test_merge_rejects_bad_solutions(self):
        with self.assertRaises(ValueError):          # bay 1's solution ends before its remove
            merge(BLOCK, [SOLUTIONS[0], SOLUTIONS[1][:1], SOLUTIONS[2]])
        with self.assertRaises(ValueError):          # 6 is not on top of stack 1 of bay 0
            merge(BLOCK, [[(1, 0, 6)] + SOLUTIONS[0][1:], SOLUTIONS[1], SOLUTIONS[2]])
        with self.assertRaises(ValueError):          # moves left after the block is empty
            merge(BLOCK, [SOLUTIONS[0], SOLUTIONS[1], SOLUTIONS[2] + [(0, 1, 7)]])
        with self.assertRaises(ValueError):
            merge(BLOCK, SOLUTIONS[:2])

    def test_block_check_rejects(self):
        cases = [
            [((1, 0), (1, 1), 5), ((1, 0), None, 2)] + MERGED[1:],           # 2 removed before 1
            [((0, 1), (0, 0), 1)] + MERGED,                                   # stack (0, 0) full at H = 2
            MERGED[:-1],                                                      # 8 left in the block
            [((0, 0), None, 3)] + MERGED,                                     # 3 is not on top
        ]
        for H, plan in zip((3, 2, 3, 3), cases):
            with self.subTest(plan=plan[:2]), self.assertRaises(ValueError):
                check_block_plan(BLOCK, H, plan)

    def test_example_1_of_lee_lee(self):
        # global stack k = bay (k - 1) // 3, stack (k - 1) % 3; moves between bays are allowed in the check
        plan = [(((a - 1) // 3, (a - 1) % 3), None if b == 0 else ((b - 1) // 3, (b - 1) % 3), c) for c, a, b in EXAMPLE1]
        self.assertEqual(check_block_plan(FIG6, 6, plan), 7)
        # P4-P7, not Lee & Lee's 2346 s (log §4.1); regression value of step1b_evaluator.txt
        self.assertAlmostEqual(plan_time(to_positions(plan)), 2243.2)

    def test_time_of_the_merged_plan(self):
        # from (0, 0), positions (bay, row) from 1; per move b, s -> time:
        #  1 rem 1 (1,2): b 2 s 4 -> 81.8      2 (2,1)->(2,2): b 2 s 2 -> 79.4   3 rem 2 (2,1): b 2 s 2 -> 79.4
        #  4 (1,1)->(1,2): b 1 s 2 -> 75.9     5 rem 3 (1,1): b 1 s 2 -> 75.9    6 (2,2)->(2,1): b 2 s 3 -> 80.6
        #  7 rem 4 (2,2): b 2 s 3 -> 80.6      8 rem 5 (2,1): b 4 s 2 -> 86.4    9 rem 6 (1,2): b 2 s 4 -> 81.8
        # 10 rem 7 (3,1): b 6 s 2 -> 93.4     11 rem 8 (3,2): b 6 s 4 -> 95.8   total 911.0
        self.assertAlmostEqual(plan_time(to_positions(merge(BLOCK, SOLUTIONS))), 911.0)


if __name__ == "__main__":
    unittest.main()


class TestSolveBlock(unittest.TestCase):
    """Step 2 (log §9): block plans of the multi-bay algorithm, without a time limit (deterministic)."""

    def check_block(self, bays, H):
        plan, runs = solve_block(bays, H, NO_LIMIT)
        check_block_plan(bays, H, plan)                                   # top only, group order, H, empty
        self.assertTrue(all(dst is None or dst[0] == src[0] for src, dst, _g in plan))   # P9
        lb = sum(lower_bound(ranked(bay)[0]) for bay in bays)
        self.assertEqual(lb, sum(r.lower_bound for r in runs))
        self.assertGreaterEqual(len(plan), lb)
        for b, bay in enumerate(bays):
            layout, back = ranked(bay)
            n = sum(map(len, bay))
            alone = TreeSearch(Bay.from_stacks(layout, H, G=n), NO_LIMIT).run() if n else []
            own = [(d, None if dst is None else dst[1], g) for (bb, d), dst, g in plan if bb == b]
            self.assertEqual(own, [(d, r, back[g]) for d, r, g in alone])     # = standalone single-bay run
            self.assertEqual(own, runs[b].moves)
        return plan, runs

    def test_ranked(self):
        layout, back = ranked([[5, 9], [], [2]])
        self.assertEqual(layout, [[2, 3], [], [1]])
        self.assertEqual(back, {1: 2, 2: 5, 3: 9})

    def test_default_parameters(self):
        self.assertEqual(Params(), Params(10, 150, 3, 2, 60.0))                # P10, P14 (T1)

    def test_empty_bay(self):
        self.assertEqual(solve_bay([[], []], 3).moves, [])

    def test_block_a(self):
        self.check_block(*BLOCK_A)

    def test_block_b_with_an_empty_stack(self):
        self.check_block(*BLOCK_B)

    @unittest.skipUnless(EXAMPLE1_FILE, f"R020306_0020_001 not found in {DEFAULT_LEE_LEE_DIR}")
    def test_example_1(self):
        ins = parse_lee_lee(EXAMPLE1_FILE)
        self.assertEqual([[list(s) for s in bay] for bay in ins.groups], FIG6)
        plan, _runs = self.check_block(ins.groups, ins.max_height)
        self.assertGreaterEqual(len(plan), 27)                                 # Lee & Lee's L (p. 1145)
