"""
Tests of the CRP-Time utility functions, and of the agreement between each
algorithm's header and its config_schema().

  python -m unittest algorithms.CRP_Time.utils.test_utils      (from the repo root)
"""

from __future__ import annotations

import copy
import importlib
import re
import unittest

from core.base_algorithm import AlgorithmConfig, BaseAlgorithm
from core.benchmarks.caserta import DEFAULT_CASERTA_DIR, parse_caserta_dat, problem_config_for_caserta_dat
from core.objectives import (
    KinematicsModel,
    ObjectiveSpec,
    annotate_plan_tiers,
    evaluate_plan_objectives,
    movement_objective_cost,
)
from core.plan import RelocationPlan
from problems.CRP_Time import CRP_Time

from .bay import free_destinations, read_bay, target_stack
from .crane_time import check_objective, move_time
from .located import min_due, non_located, well_placed
from .params import parameters
from .plan import actions, replay

ALGORITHMS = [
    ("algorithms.CRP_Time.heuristic.azari_2017.algorithm", "Azari2017CSUM"),
    ("algorithms.CRP_Time.heuristic.jovanovic_2019_aco.algorithm", "JovanovicACO_CRPTime"),
    ("algorithms.CRP_Time.heuristic.lopez_plata_2019.algorithm", "LopezPlata2019Heuristic"),
]
HEADER_PARAMETER = re.compile(r"^(\w+)\s+---\s+(\S+)\s+---\s+\S")


def _env(name="data3-5-1.dat", **values):
    return CRP_Time(config=problem_config_for_caserta_dat(DEFAULT_CASERTA_DIR / name, values))


def _plan_by_lowest_stack(stacks, max_tiers):
    """A complete restricted plan: every blocker goes to the first free stack."""
    st, moves = [list(s) for s in stacks], []
    for c in range(1, sum(len(s) for s in st) + 1):
        src = target_stack(st, c)
        while st[src][-1] != c:
            dst = free_destinations(st, src, max_tiers)[0]
            st[dst].append(st[src].pop())
            moves.append((src, dst))
        st[src].pop()
        moves.append((src, None))
    return moves


class TestBay(unittest.TestCase):

    def test_read_bay_keeps_containers_on_top(self):
        for name in ("data3-5-1.dat", "data4-6-7.dat"):
            env = _env(name)
            stacks, positions = read_bay(env)
            self.assertEqual(stacks, parse_caserta_dat(DEFAULT_CASERTA_DIR / name)[2])
            self.assertEqual(positions, sorted(env.yard.stacks))

    def test_target_stack_and_free_destinations(self):
        stacks = [[3, 1], [2], [], [4, 5, 6]]
        self.assertEqual(target_stack(stacks, 1), 0)
        self.assertEqual(target_stack(stacks, 6), 3)
        self.assertEqual(target_stack(stacks, 7), -1)
        self.assertEqual(free_destinations(stacks, 0, 3), [1, 2])
        self.assertEqual(free_destinations(stacks, 2, 2), [1])
        self.assertEqual(free_destinations(stacks, 1, 1), [2])


class TestLocated(unittest.TestCase):

    def test_min_due(self):
        self.assertEqual(min_due([4, 2, 7], 9), 2)
        self.assertEqual(min_due([], 9), 9)

    def test_non_located(self):
        self.assertEqual(non_located([5, 3, 4, 1, 2]), [False, False, True, False, True])
        self.assertEqual(non_located([]), [])

    def test_well_placed(self):
        self.assertTrue(well_placed(3, []))
        self.assertTrue(well_placed(3, [5, 4]))
        self.assertFalse(well_placed(3, [5, 2]))


class TestCraneTime(unittest.TestCase):

    def test_move_time_equals_the_evaluator(self):
        for model in ("f2", "f2_vertical"):
            env = _env("data4-6-7.dat", time_model=model)
            stacks, positions = read_bay(env)
            plan = replay(stacks, positions, _plan_by_lowest_stack(stacks, env.config.max_tiers))
            spec = ObjectiveSpec.from_config(env.config)
            kin = KinematicsModel.from_config_extra(env.config.extra)
            total = 0.0
            for move in plan.movements:
                t = move_time(move, spec)
                self.assertEqual(t, movement_objective_cost(move, spec, kin, (1, 1))[0])
                total += t
            metrics = evaluate_plan_objectives(plan, spec, kinematics=kin)
            self.assertEqual(total, metrics["crane_time"])

    def test_check_objective(self):
        check_objective(ObjectiveSpec(time_model="f2"), "A")
        check_objective(ObjectiveSpec(time_model="f2_vertical"), "A")
        with self.assertRaisesRegex(ValueError, "rmgc_current"):
            check_objective(ObjectiveSpec(time_model="rmgc_current"), "A")
        with self.assertRaisesRegex(ValueError, "relocations"):
            check_objective(ObjectiveSpec(mode="relocations"), "A")
        check_objective(ObjectiveSpec(mode="relocations"), "A", ("crane_time", "relocations"))
        with self.assertRaisesRegex(ValueError, "rmgc_current"):
            check_objective(ObjectiveSpec(mode="relocations", time_model="rmgc_current"), "A",
                            ("crane_time", "relocations"))
        with self.assertRaisesRegex(ValueError, "weighted"):
            check_objective(ObjectiveSpec(mode="weighted"), "A", ("crane_time", "relocations"))


class TestPlan(unittest.TestCase):

    def test_replay_sets_the_tiers_of_the_platform(self):
        env = _env("data4-6-7.dat")
        stacks, positions = read_bay(env)
        initial_yard = copy.deepcopy(env.yard)
        moves = _plan_by_lowest_stack(stacks, env.config.max_tiers)
        plan = replay(stacks, positions, moves)
        bare = RelocationPlan()
        for m in plan.movements:
            bare.add(type(m)(m.container_id, m.from_pos, m.to_pos))
        annotated = annotate_plan_tiers(initial_yard, bare)
        self.assertEqual([(m.from_tier, m.to_tier) for m in plan.movements],
                         [(m.from_tier, m.to_tier) for m in annotated.movements])
        self.assertTrue(env.validate_plan(plan)["feasible"])
        self.assertEqual(actions(plan, env.config.num_rows), [d for _, d in moves if d is not None])

    def test_replay_refuses_an_empty_source(self):
        with self.assertRaises(ValueError):
            replay([[1], []], [(1, 1), (2, 1)], [(1, 0)])


class TestParams(unittest.TestCase):

    def test_defaults_come_from_the_schema(self):
        module, name = ALGORITHMS[1]
        cls = getattr(importlib.import_module(module), name)
        values = parameters(AlgorithmConfig(seed=3, extra={"n_ants": 4}), cls)
        self.assertEqual(values["n_ants"], 4)
        self.assertEqual(values["q0"], cls.config_schema()["q0"]["default"])
        self.assertNotIn("seed", values)


class TestHeaders(unittest.TestCase):

    def test_header_and_schema_agree(self):
        base = BaseAlgorithm.config_schema()
        for module, name in ALGORITHMS:
            with self.subTest(algorithm=name):
                mod = importlib.import_module(module)
                header = dict(m.groups()[:2] for m in map(HEADER_PARAMETER.match, mod.__doc__.splitlines()) if m)
                schema = {k: s for k, s in getattr(mod, name).config_schema().items() if k not in base}
                self.assertEqual(list(header), list(schema))
                for key, spec in schema.items():
                    self.assertEqual(float(header[key]), float(spec["default"]), key)


if __name__ == "__main__":
    unittest.main()
