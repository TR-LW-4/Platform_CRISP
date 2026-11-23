"""
Tests of the Kim, Kim & Lee (2016) heuristic on the multi-bay layer.

  python -m unittest algorithms.CRP_Time.crp_time_u.kim_2016.test_kim_2016      (from the repo root)

Figs. 2 and 5 are rebuilt from the move sequences printed in the text; Figs. 6 and 7 were read from the
figures by Thom (23 Nov 2026).
"""

from __future__ import annotations

import random
import unittest
from unittest import mock

from algorithms.CRP_Time.crp_time_u.kim_2016.heuristic import is_critical, is_ideal, solve
from algorithms.CRP_Time.utils.multibay import MultiBayLayout, YardState, check_multibay_plan, load_lee_lee
from core.benchmarks.lee_lee import list_lee_lee_files


def layout(name, n_rows, H, stacks):
    order = tuple(sorted(c for s in stacks.values() for c in s))
    full = {(1, r): tuple(stacks.get(r, ())) for r in range(1, n_rows + 1)}
    return MultiBayLayout(name, 1, n_rows, H, full, order)


def moves(plan):
    return [(m.container_id, m.from_pos[1], 0 if m.to_pos is None else m.to_pos[1]) for m in plan.movements]


class FirstOfTies(random.Random):
    """Resolves every tie to the first of the sorted candidates (the lowest stack)."""
    def choice(self, seq):
        return seq[0]


class TestDefinitions(unittest.TestCase):
    def test_fig2_classification(self):
        """Fig. 2 (p. 354), rebuilt from the first move sequence on p. 353 (each container's first pick-up stack, in
        pick-up order): stacks 1-6 = [2, 8, 6], [7, 11], [4, 9], [12, 3], [10], [5, 13, 1] (bottom to top).
        Text, p. 354: "stacks 4 and 5 are ideal stacks", "stacks 1, 2, 3, and 6 are unideal stacks. Among these
        stacks, stacks 1, 2, and 3 are critical stacks. The TNC of these critical stacks is 6, 11, and 9"."""
        stacks = [[2, 8, 6], [7, 11], [4, 9], [12, 3], [10], [5, 13, 1]]
        self.assertEqual([is_ideal(s) for s in stacks], [False, False, False, True, True, False])
        self.assertEqual([is_critical(s) for s in stacks], [True, True, True, False, False, False])
        self.assertEqual([s[-1] for s in stacks if is_critical(s)], [6, 11, 9])


class TestExamples(unittest.TestCase):
    def test_case2_fig5_full_sequence(self):
        """Fig. 5 (p. 355): "Following to the proposed algorithm, the entire process is as follows; (3:1, 2), (1:1, 0),
        (4:3, 1), (2:3, 0), (3:2, 0), (4:1, 0), (6:2, 1), (5:2, 0), (6:1, 0)". Layout rebuilt from that sequence:
        stack 1 [1, 3], stack 2 [5, 6], stack 3 [2, 4]; consistent with "The minimum order of stack 2 is 5, whereas that
        of stack 3 is 2". H is not given in the text; H = 3 (any H >= 3 gives the same moves).
        (6:2, 1): stacks 1 and 3 are both empty candidates (TN = big M), a tie the paper does not address; the
        sequence takes stack 1, the lowest, so ties are resolved to the lowest stack here."""
        lay = layout("Kim Fig. 5", 3, 3, {1: (1, 3), 2: (5, 6), 3: (2, 4)})
        with mock.patch("algorithms.CRP_Time.crp_time_u.kim_2016.heuristic.random.Random", FirstOfTies):
            plan = solve(lay)
        self.assertEqual(moves(plan), [(3, 1, 2), (1, 1, 0), (4, 3, 1), (2, 3, 0), (3, 2, 0), (4, 1, 0), (6, 2, 1),
                                       (5, 2, 0), (6, 1, 0)])

    def assertExample(self, stacks, expected, after):
        lay = layout("Kim example", 3, 4, stacks)
        plan = solve(lay)
        self.assertEqual(moves(plan)[:3], expected)
        state = YardState(lay)
        for m in plan.movements[:3]:
            state.retrieve(m.from_pos) if m.to_pos is None else state.move(m.from_pos, m.to_pos)
        self.assertEqual({r: state.stacks[(1, r)] for r in (1, 2, 3)}, after)
        check_multibay_plan(plan, lay)

    def test_case3_fig6(self):
        """Fig. 6 (p. 356, read by Thom), bottom to top: 1 = [7, 6], 2 = [5, 2, 4], 3 = [1, 3]; H = 4.
        cmin = 1 under 3: tnccmin = 3. Ideal: stack 1 (TN 6 > 3, the only candidate) -> Case 3, selected stack 1.
        Critical stacks with 3 < TNC < 6: stack 2 (2 below 4, TNC 4) -> 4 into stack 1; stack 1 then has one empty
        slot (K4) -> stop; 3 into stack 1; 1 retrieved. Text (p. 355): (4:2, 1), (3:3, 1), (1:3, 0); the figure
        after: 1 = [7, 6, 4, 3], 2 = [5, 2], 3 empty."""
        self.assertExample({1: (7, 6), 2: (5, 2, 4), 3: (1, 3)}, [(4, 2, 1), (3, 3, 1), (1, 3, 0)],
                           {1: [7, 6, 4, 3], 2: [5, 2], 3: []})

    def test_case4_fig7(self):
        """Fig. 7 (p. 356, read by Thom), bottom to top: 1 = [7, 4], 2 = [5, 2, 3], 3 = [1, 6]; H = 4.
        cmin = 1 under 6: tnccmin = 6. Ideal: stack 1 (TN 4); no candidate -> Case 4, selected stack 1 (largest TN).
        Critical stacks with TNC < 4: stack 2 (2 below 3, TNC 3) -> 3 into stack 1; one empty slot left (K4) -> stop;
        6 into stack 1; 1 retrieved. Text (p. 355): (3:2, 1), (6:3, 1), (1,3, 0); the figure after:
        1 = [7, 4, 3, 6], 2 = [5, 2], 3 empty."""
        self.assertExample({1: (7, 4), 2: (5, 2, 3), 3: (1, 6)}, [(3, 2, 1), (6, 3, 1), (1, 3, 0)],
                           {1: [7, 4, 3, 6], 2: [5, 2], 3: []})


class TestVariants(unittest.TestCase):
    """The sensitivity flags (V1 search_once, V2 fixed_tn) default to the pseudocode reading."""

    FIG5 = layout("Kim Fig. 5", 3, 3, {1: (1, 3), 2: (5, 6), 3: (2, 4)})
    FIG6 = layout("Kim Fig. 6", 3, 4, {1: (7, 6), 2: (5, 2, 4), 3: (1, 3)})
    FIG7 = layout("Kim Fig. 7", 3, 4, {1: (7, 4), 2: (5, 2, 3), 3: (1, 6)})

    def test_default_flags_reproduce_figs_5_to_7(self):
        with mock.patch("algorithms.CRP_Time.crp_time_u.kim_2016.heuristic.random.Random", FirstOfTies):
            plan5 = solve(self.FIG5, search_once=False, fixed_tn=False)
        self.assertEqual(moves(plan5), [(3, 1, 2), (1, 1, 0), (4, 3, 1), (2, 3, 0), (3, 2, 0), (4, 1, 0), (6, 2, 1),
                                        (5, 2, 0), (6, 1, 0)])
        self.assertEqual(moves(solve(self.FIG6, search_once=False, fixed_tn=False))[:3], [(4, 2, 1), (3, 3, 1), (1, 3, 0)])
        self.assertEqual(moves(solve(self.FIG7, search_once=False, fixed_tn=False))[:3], [(3, 2, 1), (6, 3, 1), (1, 3, 0)])

    def test_variants_give_valid_plans(self):
        for lay in (self.FIG5, self.FIG6, self.FIG7):
            for once, fixed in ((True, False), (False, True), (True, True)):
                check_multibay_plan(solve(lay, search_once=once, fixed_tn=fixed), lay)


@unittest.skipUnless(list_lee_lee_files(), "benchmark/lee_lee is missing")
class TestLeeLeeExample1(unittest.TestCase):
    def test_valid_plan(self):
        """Lee & Lee's Example 1 (not part of the benchmark sets): a plan that check_multibay_plan accepts."""
        path = next(p for p in list_lee_lee_files() if p.stem == "R020306_0020_001")
        lay = load_lee_lee(path)
        self.assertEqual(check_multibay_plan(solve(lay), lay).retrievals, 20)


if __name__ == "__main__":
    unittest.main()
