"""
Tests of the Kim, Kim & Lee (2016) heuristic on the multi-bay layer.

  python -m unittest algorithms.CRP_Time.crp_time_u.kim_2016.test_kim_2016      (from the repo root)

The figures of the paper (Figs. 2, 5, 6, 7) are images that were not readable in the extracted text. Fig. 2 and Fig. 5
are rebuilt from the move sequences printed in the text; the Fig. 6 and Fig. 7 layouts are NOT known: the layouts
below are the smallest ones consistent with every statement of the text about them, so these two tests check the
printed moves, not the figures.
"""

from __future__ import annotations

import random
import unittest
from unittest import mock

from algorithms.CRP_Time.crp_time_u.kim_2016.heuristic import is_critical, is_ideal, solve
from algorithms.CRP_Time.utils.multibay import MultiBayLayout, check_multibay_plan, load_lee_lee
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

    def test_case3_example(self):
        """Case 3 (p. 355): "In Fig. 6, the TN of stack 1 selected for relocation is 6, and tnccmin is 3. Thus,
        container 4 in stack 2 is moved to stack 1, and then container 3 in stack 3 is moved to stack 1 as well ...
        the generated movements in the example are (4:2, 1), (3:3, 1), and (1:3, 0)"; afterwards "stack 1 remains as
        an ideal stack and ... stack 2 actually becomes an ideal stack".
        Constructed layout (Fig. 6 unknown): stack 1 [6], stack 2 [2, 4], stack 3 [5, 1, 3], H = 4.
        cmin = 1 under 3: tnccmin = 3. Ideal: stack 1 (TN 6 > 3, the only candidate) -> Case 3, selected stack 1.
        Critical stacks with 3 < TNC < 6: stack 2 (2 below 4, TNC 4) -> 4 into stack 1; stack 2 = [2] is now ideal and
        no TNC lies in (3, 4) -> stop; 3 into stack 1; 1 retrieved. The rest are retrievals."""
        lay = layout("Kim Case 3", 3, 4, {1: (6,), 2: (2, 4), 3: (5, 1, 3)})
        self.assertEqual(moves(solve(lay)), [(4, 2, 1), (3, 3, 1), (1, 3, 0), (2, 2, 0), (3, 1, 0), (4, 1, 0),
                                             (5, 3, 0), (6, 1, 0)])

    def test_case4_example(self):
        """Case 4 (p. 355): "In Fig. 7, TN of stack 1 selected for relocation is 4. Thus, container 3 in stack 2 is
        moved to stack 1, and then container 6 in stack 3 is moved to stack 1 as well ... the movements (3:2, 1),
        (6:3, 1), and (1,3, 0) are generated"; afterwards "stack 1 is an almost ideal stack except for container 6 on
        top". Constructed layout (Fig. 7 unknown): stack 1 [4], stack 2 [2, 3], stack 3 [5, 1, 6], H = 4.
        cmin = 1 under 6: tnccmin = 6. Ideal: stack 1 (TN 4); no candidate (4 < 6) -> Case 4, selected stack 1 (largest
        TN). Critical stacks with TNC < 4: stack 2 (TNC 3) -> 3 into stack 1; then none with TNC < 3 -> 6 into stack 1;
        1 retrieved; 2 retrieved. cmin = 3 under 6: ideal stacks 2 (empty, TN big M) and 3 ([5]); candidate stack 2
        (big M > 6) -> Case 3, no critical stack with 6 < TNC < big M -> 6 into stack 2. The rest are retrievals."""
        lay = layout("Kim Case 4", 3, 4, {1: (4,), 2: (2, 3), 3: (5, 1, 6)})
        self.assertEqual(moves(solve(lay)), [(3, 2, 1), (6, 3, 1), (1, 3, 0), (2, 2, 0), (6, 1, 2), (3, 1, 0),
                                             (4, 1, 0), (5, 3, 0), (6, 2, 0)])


@unittest.skipUnless(list_lee_lee_files(), "benchmark/lee_lee is missing")
class TestLeeLeeExample1(unittest.TestCase):
    def test_valid_plan(self):
        """Lee & Lee's Example 1 (not part of the benchmark sets): a plan that check_multibay_plan accepts."""
        path = next(p for p in list_lee_lee_files() if p.stem == "R020306_0020_001")
        lay = load_lee_lee(path)
        self.assertEqual(check_multibay_plan(solve(lay), lay).retrievals, 20)


if __name__ == "__main__":
    unittest.main()
