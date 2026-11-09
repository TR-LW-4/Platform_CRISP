"""
Checks for the Difference heuristics (docs/validation/unluyurt_2012.md §3): the
first relocation of Fig. 9 and of the cases E1-E7 and D1-D4, the error when no
stack can receive a container (O6), complete plans through the problem, the same
plan under f2 and f2vert (O8), Difference1 under CRP-R with the same plan as under
CRP-Time (O9), and Objective2 as f2 with tpp = A = 5 and ts = B = 1 on the examples
of Fig. 3 and Fig. 6(c).

  python -m unittest algorithms.CRP_Time.heuristic.unluyurt_2012.test_unluyurt   (from the repo root)

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
corresponding original algorithm paper.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import queue
import tempfile
import threading
import unittest
from pathlib import Path

from algorithms.CRP_Time.utils.crane_time import ObjectiveSpec, evaluate, movement_f2
from algorithms.CRP_Time.utils.plan import replay
from core.base_algorithm import AlgorithmConfig
from core.benchmarks.caserta import DEFAULT_CASERTA_DIR, problem_config_for_caserta_dat
from core.plan import Movement
from problems.CRP_R import CRP_R
from problems.CRP_Time import CRP_Time

from .difference1 import UnluyurtAydinDifference1
from .difference2 import UnluyurtAydinDifference2
from .retrieve import finish_relocations, retrieve
from .rules import difference1, difference2

# (case, H, stacks bottom to top with stack 1 first, expected stack of Difference1, of Difference2), §3.
# Stacks are numbered from 1 as in the log; container 1 is the target and x the top of its stack.
CASES = [
    ("Fig. 9 A", 4, [[1, 4, 2], [3, 5], [6]], 2, 2),
    ("Fig. 9 B", 4, [[1, 6, 5], [3, 2], [4]], 3, 3),
    ("Fig. 9 C", 4, [[1, 7, 4], [3, 5], [2, 6]], 2, 2),
    ("E1", 4, [[1, 2, 3], [4, 7], [6, 5]], 2, 2),
    ("E2", 4, [[1, 5], [2, 4], [3]], 2, 2),
    ("E3", 4, [[1, 4], [3, 6], [2, 5]], 3, 3),
    ("E4", 4, [[1, 5], [2, 3], [4], []], 4, 4),
    ("E5", 4, [[1, 3], [2, 4], [5], []], 3, 3),
    ("E6", 4, [[1, 3], [2, 4], [], []], 3, 3),
    ("E7", 3, [[1, 4, 3], [5, 6, 7], [2]], 3, 3),
    ("D1", 4, [[3, 2], [4], [1, 5], [6]], 4, 4),
    ("D2", 4, [[6], [1, 4], [5], [3, 2]], 3, 1),
    ("D3", 4, [[2, 6], [1, 4], [5, 3]], 3, 3),
    ("D4", 4, [[2, 6], [1, 4], [3, 5]], 3, 1),
]


def _first_destination(select, stacks, max_tiers):
    """Stack (numbered from 1) that select chooses for the top container of container 1's stack."""
    src = next(i for i, st in enumerate(stacks) if 1 in st)
    empty = sum(len(st) for st in stacks) + 1
    return select(stacks[src][-1], src, stacks, max_tiers, empty) + 1


def _config(tmp, stacks, max_tiers=None, time_model="f2"):
    dat = Path(tmp) / "case.dat"
    dat.write_text(f"{len(stacks)} {sum(len(st) for st in stacks)}\n"
                   + "".join(f"{len(st)} {' '.join(map(str, st))}\n" for st in stacks))
    cfg = problem_config_for_caserta_dat(dat, {"time_model": time_model})
    if max_tiers is not None:
        cfg.max_tiers = max_tiers
    return cfg


def _train(cls, cfg, problem=CRP_Time):
    q = queue.Queue()
    alg = cls(AlgorithmConfig())
    alg.train(lambda: problem(config=cfg), q, threading.Event())
    records = []
    while not q.empty():
        records.append(q.get_nowait())
    return records


class TestFirstRelocation(unittest.TestCase):

    def test_difference1(self):
        for case, max_tiers, stacks, expected, _ in CASES:
            with self.subTest(case=case):
                self.assertEqual(_first_destination(difference1, stacks, max_tiers), expected)

    def test_difference2(self):
        for case, max_tiers, stacks, _, expected in CASES:
            with self.subTest(case=case):
                self.assertEqual(_first_destination(difference2, stacks, max_tiers), expected)


class TestNoDestination(unittest.TestCase):

    def test_raises_when_every_other_stack_is_full(self):
        # O6: container 1 is under 2 and the only other stack is full (H = 2).
        for select in (difference1, difference2):
            with self.subTest(select=select.__name__):
                with self.assertRaisesRegex(RuntimeError, "no stack can receive container 2"):
                    retrieve([[1, 2], [3, 4]], 2, select)


class TestThroughProblem(unittest.TestCase):

    def test_complete_plan(self):
        # data3-5-1 has container 1 on top at the start: its retrieval must be a move of the plan.
        cfg = problem_config_for_caserta_dat(DEFAULT_CASERTA_DIR / "data3-5-1.dat", {"time_model": "f2"})
        for cls in (UnluyurtAydinDifference1, UnluyurtAydinDifference2):
            with self.subTest(algorithm=cls.name):
                records = _train(cls, cfg)
                self.assertEqual(len(records), 1)
                moves = records[0].extra["moves"]
                retrievals = [m["container_id"] for m in moves if m["kind"] == "retrieve"]
                self.assertEqual(retrievals, list(range(1, cfg.num_containers + 1)))
                metrics = records[0].metrics
                self.assertEqual(metrics["relocations"], len(moves) - cfg.num_containers)
                self.assertEqual(metrics["objective_value"], metrics["crane_time_f2"])
                self.assertEqual(metrics["steps"], len(moves))

    def test_same_plan_under_f2_and_f2vert(self):
        # O8: the rules do not read the time model.
        path = DEFAULT_CASERTA_DIR / "data4-6-1.dat"
        for cls in (UnluyurtAydinDifference1, UnluyurtAydinDifference2):
            with self.subTest(algorithm=cls.name):
                f2 = _train(cls, problem_config_for_caserta_dat(path, {"time_model": "f2"}))[0]
                vert = _train(cls, problem_config_for_caserta_dat(path, {"time_model": "f2_vertical"}))[0]
                self.assertEqual(f2.extra["moves"], vert.extra["moves"])
                self.assertEqual(vert.metrics["objective_value"], vert.metrics["crane_time_vertical"])

    def test_no_destination_raises_through_problem(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _config(tmp, [[1, 2], [3, 4]], max_tiers=2)
            with self.assertRaisesRegex(RuntimeError, "no stack can receive container 2"):
                _train(UnluyurtAydinDifference1, cfg)


class TestCrpR(unittest.TestCase):
    """O9 (Wei, 9 Nov 2026): Difference1 also under CRP-R, with the metrics of validate_plan."""

    @staticmethod
    def _config(path):
        defaults = {k: v["default"] for k, v in CRP_R.config_schema().items() if "default" in v}
        return problem_config_for_caserta_dat(path, defaults)

    def test_listed_problems(self):
        self.assertEqual(UnluyurtAydinDifference1.compatible_problems, ["CRP-Time", "CRP-R"])
        self.assertEqual(UnluyurtAydinDifference2.compatible_problems, ["CRP-Time"])

    def test_same_plan_as_under_crp_time(self):
        for name in ("data3-5-1.dat", "data4-6-1.dat"):
            path = DEFAULT_CASERTA_DIR / name
            with self.subTest(instance=name):
                cfg = self._config(path)
                records = _train(UnluyurtAydinDifference1, cfg, CRP_R)
                self.assertEqual(len(records), 1)
                record, moves = records[0], records[0].extra["moves"]
                retrievals = [m["container_id"] for m in moves if m["kind"] == "retrieve"]
                self.assertEqual(retrievals, list(range(1, cfg.num_containers + 1)))
                self.assertEqual(record.metrics["feasible"], 1.0)
                self.assertEqual(record.metric, record.metrics["relocations"])
                self.assertEqual(record.metrics["steps"], len(moves))
                time = _train(UnluyurtAydinDifference1,
                              problem_config_for_caserta_dat(path, {"time_model": "f2"}))[0]
                self.assertEqual(moves, time.extra["moves"])
                self.assertEqual(record.metrics["relocations"], time.metrics["relocations"])

    def test_infeasible_plan_raises(self):
        stacks = [[2], [1]]
        plan = replay(stacks, [(1, 1), (2, 1)], [(0, None)])   # retrieves 2 before 1
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _config(tmp, stacks)
            with self.assertRaisesRegex(RuntimeError, "plan rejected by validate_plan: move 0: expected priority 1"):
                finish_relocations(lambda: CRP_R(config=cfg), plan, "test", 0.0)


class TestObjective2(unittest.TestCase):
    """Objective2 (p. 382, A = 5 and B = 1, p. 388) is f2 with tpp = A and ts = B (§2, Objective2)."""

    SPEC = ObjectiveSpec(stack_s_per_stack=1.0, pickup_place_s=5.0, max_tiers=4)
    POSITIONS = [(1, 1), (2, 1), (3, 1)]

    def test_fig3(self):
        # p. 381-382: [<1,4,5><3,2><6>]; 5 and 4 onto container 2, then retrieve 1: 3A + 6B.
        plan = replay([[1, 4, 5], [3, 2], [6]], self.POSITIONS, [(0, 1), (0, 1), (0, None)])
        self.assertEqual(evaluate(plan, self.SPEC)["crane_time_f2"], 3 * 5 + 6 * 1)

    def test_fig6(self):
        # p. 382, 384: from Fig. 6(a) = [<1,4,5><3,2><6>], 5 to stack 3 (4 units) and 4 to stack 2 (2 units).
        plan = replay([[1, 4, 5], [3, 2], [6]], self.POSITIONS, [(0, 2), (0, 1)])
        self.assertEqual(evaluate(plan, self.SPEC)["crane_time_f2"], 2 * 5 + 6 * 1)
        # Fig. 6(c) = [<1><3,2,4><6,5>]: the six retrievals take 2 + 3*4 + 2*6 = 26 units.
        stack_of = {1: 1, 2: 2, 3: 2, 4: 2, 5: 3, 6: 3}
        retrievals = [Movement(c, (s, 1), None) for c, s in stack_of.items()]
        self.assertEqual(sum(movement_f2(m, self.SPEC) for m in retrievals), 6 * 5 + 26 * 1)


if __name__ == "__main__":
    unittest.main()
