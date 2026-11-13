"""
Checks for the tree search of Forster & Bortfeldt (2012)
(docs/validation/forster_bortfeldt_2012.md §4): the incremental state against
the plain definitions of layout.py, the bound of Fig. 3 and against exact optima,
the move types of Sec. 4.2, one case per priority group of the greedy (Fig. 4),
the productive moves (Fig. 7), the plan checker, and greedy and search on the
Caserta 3x3 and 3x4 instances (valid plans, never below the bound, never worse
than the greedy, never below the exact optimum at H = T + 2).

  python -m unittest algorithms.CRP_Time.crp_time_u.forster_bortfeldt_2012.test_forster_bortfeldt   (from the repo root)

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
corresponding original algorithm paper.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import itertools
import random
import unittest

from core.benchmarks.caserta import DEFAULT_CASERTA_DIR, parse_caserta_dat

from .layout import badly_placed, lower_bound, lower_bound_parts, n_badly_placed
from .plan import check_plan
from .search import Params, TreeSearch, greedy, productive_moves
from .state import Bay

FIG3 = [[12, 8, 11, 2], [9, 10, 3, 14], [1, 16, 7, 15], [5, 4, 13, 6]]


def caserta(tiers: int, stacks: int, k: int):
    return parse_caserta_dat(DEFAULT_CASERTA_DIR / f"data{tiers}-{stacks}-{k}.dat")[2]


def clean_supply(stacks, H, G):
    """Def. 2 and Table 1 (p. 302), slot by slot."""
    total = 0
    for s in stacks:
        if not s:
            total += H * G
        elif not any(badly_placed(s)):
            total += sum(s[-1] for _h in range(len(s), H))
    return total


def exact_moves(stacks, H):
    """Minimum number of moves by IDA* with the bound n + n_b (Kim & Hong); independent of state.Bay."""
    start = tuple(tuple(s) for s in stacks)

    def h(st):
        return n_badly_placed(st) + sum(map(len, st))

    def succ(st):
        low = min(min(s) for s in st if s)
        for d, s in enumerate(st):
            if s and s[-1] == low:
                yield st[:d] + (s[:-1],) + st[d + 1:]
                return
        for d, s in enumerate(st):
            if not s:
                continue
            for r, t in enumerate(st):
                if r != d and len(t) < H:
                    nxt = list(st)
                    nxt[d], nxt[r] = s[:-1], t + (s[-1],)
                    yield tuple(nxt)

    def dfs(st, g, bound, seen):
        f = g + h(st)
        if f > bound:
            return f
        if not any(st):
            return -1
        best = float("inf")
        for nxt in succ(st):
            if seen.get(nxt, 1 << 30) <= g + 1:
                continue
            seen[nxt] = g + 1
            t = dfs(nxt, g + 1, bound, seen)
            if t == -1:
                return -1
            best = min(best, t)
        return best

    bound = h(start)
    while True:
        t = dfs(start, 0, bound, {start: 0})
        if t == -1:
            return bound
        bound = t


class TestState(unittest.TestCase):

    def test_incremental_values_equal_the_definitions(self):
        rng = random.Random(7)
        for _ in range(300):
            S, H = rng.randint(2, 5), rng.randint(3, 6)
            n = rng.randint(1, S * H - 1)
            groups = [rng.randint(1, n) for _ in range(n)]      # repeated groups too
            stacks = [[] for _ in range(S)]
            for g in groups:
                rng.choice([s for s in stacks if len(s) < H]).append(g)
            bay = Bay.from_stacks(stacks, H, G=n)
            for _step in range(40):
                ref = [list(s) for s in bay.stacks]
                self.assertEqual(bay.nb, n_badly_placed(ref))
                self.assertEqual(bay.lower_bound(), lower_bound(ref))
                self.assertEqual(bay.clean_supply(), clean_supply(ref, H, n))
                moves = bay.removes() or [m for m, *_ in bay.relocations()]
                bay = bay.apply(rng.choice(moves))
                if not bay.n:
                    break


class TestLowerBound(unittest.TestCase):

    def test_fig3(self):
        self.assertEqual(lower_bound_parts(FIG3), (16, 8, 1))
        self.assertEqual(Bay.from_stacks(FIG3, 6).lower_bound(), 25)

    def test_never_above_the_exact_optimum_on_3x3(self):
        for k in range(1, 41):
            st = caserta(3, 3, k)
            self.assertLessEqual(lower_bound(st), exact_moves(st, 5), k)


class TestMoveTypes(unittest.TestCase):
    """Sec. 4.2 (p. 302), stacks from 0."""

    @staticmethod
    def kind(stacks, move, H=6):
        for m, before, after, flg in Bay.from_stacks(stacks, H).relocations():
            if m == move:
                return ("FLG_" if flg else "FOG_") * before + "GB"[before] + "GB"[after]
        raise AssertionError(move)

    def test_fig3_and_small_layouts(self):
        self.assertEqual(self.kind(FIG3, (2, 0, 15)), "FLG_BB")
        self.assertEqual(self.kind(FIG3, (3, 0, 6)), "FOG_BB")
        self.assertEqual(self.kind(FIG3, (0, 1, 2)), "GG")
        self.assertEqual(self.kind(FIG3, (0, 2, 2)), "GB")
        self.assertEqual(self.kind([[1, 5], [6]], (0, 1, 5)), "FLG_BG")
        self.assertEqual(self.kind([[2, 3, 5], [1, 4], [6]], (0, 2, 5)), "FOG_BG")
        self.assertEqual(self.kind([[2, 3, 5], [1, 4], []], (0, 2, 5)), "FOG_BG")

    def test_full_stack_is_no_receiver(self):
        self.assertNotIn((0, 1, 5), [m for m, *_ in Bay.from_stacks([[1, 5], [7, 6]], 2).relocations()])


class TestGreedy(unittest.TestCase):
    """The first move of Fig. 4 (p. 303) per priority group."""

    def first(self, stacks, H=4):
        return greedy(Bay.from_stacks(stacks, H))[0]

    def test_1_remove(self):
        self.assertEqual(greedy(Bay.from_stacks([[2, 1], [3]], 3)), [(0, None, 1), (0, None, 2), (1, None, 3)])

    def test_2_bg_to_non_empty_stack_with_smallest_difference(self):
        self.assertEqual(self.first([[1, 5], [6], [9]]), (0, 1, 5))
        self.assertEqual(self.first([[1, 5], [2, 7], [6], [8]]), (0, 2, 5))   # tie 5->6, 7->8: first (F8)

    def test_3_bg_to_empty_stack_with_smallest_group(self):
        self.assertEqual(self.first([[1, 7], [2, 8], [], [3]]), (0, 2, 7))

    def test_4_flg_bb_with_smallest_difference_to_lowest_group_of_target(self):
        # F9 (B): 9 - 3 = 6 onto stack 2 beats 9 - 2 = 7 onto stack 1 (Fig. 4 as printed would pick stack 1).
        self.assertEqual(self.first([[1, 9], [2, 8], [3, 7]]), (0, 2, 9))
        # (B) and the text (p. 304, max |g - top|) differ: (B) 9 - 4 = 5 onto stack 2; the text |9 - 6| = 3 onto stack 1.
        self.assertEqual(self.first([[1, 9], [2, 6], [4, 8]]), (0, 2, 9))

    def test_fig3_plan_is_valid(self):
        self.assertEqual(check_plan(FIG3, 6, greedy(Bay.from_stacks(FIG3, 6))), 13)


class TestProductiveMoves(unittest.TestCase):
    """Fig. 7 (p. 306)."""

    def test_removes_only(self):
        self.assertEqual(productive_moves(Bay.from_stacks([[3, 1], [2, 5]], 4), Params()), [(0, None, 1)])

    def test_all_bg_when_there_is_one(self):
        bay = Bay.from_stacks([[1, 5], [2, 7], [6], [8]], 4)
        self.assertEqual(productive_moves(bay, Params()), [(0, 2, 5), (0, 3, 5), (1, 3, 7)])

    def test_flg_bb_and_gg_sorted_and_cut(self):
        bay = Bay.from_stacks([[1, 9], [7, 3], [6, 4], [8, 5], [2, 10]], 5)
        # FLG_BB of 9 by 9 - lowest group of the target: 4, 5, 6 (7 cut by maxFlg_bb = 3);
        # GG by (g - below) - (g - top) (F11): 3->4: -3, 3->5: -2 (4->5: -1 cut by maxGg = 2).
        self.assertEqual(productive_moves(bay, Params()),
                         [(0, 3, 9), (0, 2, 9), (0, 1, 9), (1, 2, 3), (1, 3, 3)])

    def test_gg_from_single_item_stack_uses_group_G(self):
        bay = Bay.from_stacks([[1, 9], [3], [6, 4], [8, 5], [2, 10]], 5)
        # 3->4: 4 - G = -6, 3->5: 5 - G = -5, 4->5: 5 - 6 = -1 (F12, G = 10)
        self.assertEqual(productive_moves(bay, Params(max_flg_bb=0, max_gg=3)), [(1, 2, 3), (1, 3, 3), (2, 3, 4)])


class TestCheckPlan(unittest.TestCase):

    def test_violations(self):
        st = [[2, 1], [3]]
        for plan in ([], [(0, None, 2)], [(0, None, 1), (1, None, 3)], [(0, 1, 1), (1, None, 1)],
                     [(0, None, 1), (0, None, 2)]):
            with self.assertRaises(ValueError):
                check_plan(st, 2, plan)
        with self.assertRaises(ValueError):          # stack 1 full at H = 1
            check_plan([[1], [2]], 1, [(1, 0, 2)])
        self.assertEqual(check_plan(st, 2, [(0, None, 1), (0, None, 2), (1, None, 3)]), 0)


class TestSearch(unittest.TestCase):

    def run_class(self, tiers, stacks, exact=False):
        for k in range(1, 41):
            st = caserta(tiers, stacks, k)
            n = sum(map(len, st))
            for H in (n, tiers + 2):                # unlimited (F7) and T + 2 (F6)
                bay = Bay.from_stacks(st, H)
                ts = TreeSearch(bay, Params(time_limit=None))
                plan = ts.run()
                rel = check_plan(st, H, plan)
                self.assertGreaterEqual(len(plan), bay.lower_bound(), (k, H))
                self.assertLessEqual(len(plan), ts.greedy_moves, (k, H))
                self.assertEqual(check_plan(st, H, greedy(bay)) + n, ts.greedy_moves)
                self.assertEqual(rel + n, len(plan))
                if exact and H == tiers + 2:
                    self.assertGreaterEqual(len(plan), exact_moves(st, H), (k, H))

    def test_caserta_3x3(self):
        self.run_class(3, 3, exact=True)

    def test_caserta_3x4(self):
        self.run_class(3, 4)

    def test_tight_pruning_returns_the_plan_of_the_printed_test(self):
        """F20: without a time limit, the test of Fig. 6 as printed (n'_m(L') > maxMoves) and the tight test
        (> maxMoves - 1) give the same plan. The printed test is obtained by passing maxMoves + 1 to the first-level
        call of determine_compound_moves; the recursion then compares with maxMoves, maxMoves - 1, ... as printed."""

        class Printed(TreeSearch):
            def _determine(self, bay, s, cm_work, max_moves, v, cm):
                super()._determine(bay, s, cm_work, max_moves + (not cm_work), v, cm)

        for tiers, stacks in ((3, 3), (3, 4)):
            for k in range(1, 41):
                st = caserta(tiers, stacks, k)
                n = sum(map(len, st))
                for H in (n, tiers + 2):
                    tight = TreeSearch(Bay.from_stacks(st, H), Params(time_limit=None))
                    printed = Printed(Bay.from_stacks(st, H), Params(time_limit=None))
                    self.assertEqual(tight.run(), printed.run(), (tiers, stacks, k, H))
                    self.assertLessEqual(tight.nodes, printed.nodes, (tiers, stacks, k, H))

    def test_time_limit_returns_the_greedy(self):
        st = caserta(5, 5, 1)
        bay = Bay.from_stacks(st, 25)
        ticks = itertools.count()
        ts = TreeSearch(bay, Params(time_limit=0.5), clock=lambda: next(ticks))
        plan = ts.run()
        self.assertTrue(ts.timed_out)
        self.assertEqual(ts.nodes, 0)
        self.assertEqual(plan, greedy(bay))

    def test_stops_at_the_bound(self):
        bay = Bay.from_stacks(caserta(3, 4, 2), 12)
        ts = TreeSearch(bay, Params())
        self.assertEqual(len(ts.run()), bay.lower_bound())
        self.assertEqual(ts.nodes, 0)


if __name__ == "__main__":
    unittest.main()
