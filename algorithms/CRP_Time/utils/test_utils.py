"""
Tests of the CRP-Time utility functions, and of the agreement between each
algorithm's header and its config_schema().

  python -m unittest algorithms.CRP_Time.utils.test_utils      (from the repo root)
"""

from __future__ import annotations

import copy
import importlib
import re
import time
import unittest

from core.base_algorithm import AlgorithmConfig, BaseAlgorithm
from core.benchmarks.caserta import DEFAULT_CASERTA_DIR, parse_caserta_dat, problem_config_for_caserta_dat
from core.objectives import (
    PRACTICE_PARAMETERS,
    KinematicsModel,
    ObjectiveSpec,
    annotate_plan_tiers,
    compute_crane_time_accel,
    evaluate_plan_objectives,
    is_practice,
    movement_accel,
    movement_f2_vertical,
    movement_objective_cost,
    movement_time,
)
from core.plan import Movement, RelocationPlan
from core.results.aggregate import summarize_run_dicts
from problems.CRP_MO import CRP_MO
from problems.CRP_Time import CRP_Time

from .bay import free_destinations, read_bay, target_stack
from .crane_time import check_objective, move_time
from .finish import finish
from .located import min_due, non_located, well_placed
from .params import parameters
from .plan import actions, replay
from .timer import Stopwatch

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


class TestFinish(unittest.TestCase):

    def test_finish_equals_the_evaluator(self):
        for model, mode in (("f2", "crane_time"), ("f2_vertical", "crane_time"), ("f2", "relocations")):
            with self.subTest(time_model=model, mode=mode):
                env = _env("data4-6-7.dat", time_model=model, objective_mode=mode)
                stacks, positions = read_bay(env)
                initial_yard = copy.deepcopy(env.yard)
                plan = replay(stacks, positions, _plan_by_lowest_stack(stacks, env.config.max_tiers))
                spec = ObjectiveSpec.from_config(env.config)
                want = evaluate_plan_objectives(plan, spec, kinematics=KinematicsModel.from_config_extra(
                    env.config.extra), initial_yard=initial_yard)
                metrics, _ = finish(lambda: _env("data4-6-7.dat", time_model=model, objective_mode=mode),
                                    plan, spec, "A", 0.25)
                self.assertEqual({k: metrics[k] for k in want}, want)
                self.assertEqual(metrics["steps"], float(plan.num_moves()))
                self.assertEqual(metrics["progress"], 1.0)
                self.assertEqual(metrics["runtime_s"], 0.25)

    def test_finish_refuses_an_infeasible_plan(self):
        env = _env("data3-5-1.dat")
        stacks, positions = read_bay(env)
        moves = _plan_by_lowest_stack(stacks, env.config.max_tiers)
        plan = replay(stacks, positions, moves[:-1])   # the last container stays in the bay
        with self.assertRaisesRegex(RuntimeError, "A: plan rejected by validate_plan"):
            finish(lambda: _env("data3-5-1.dat"), plan, ObjectiveSpec.from_config(env.config), "A", 0.0)


class TestPracticePreset(unittest.TestCase):
    """The CRP-Time option f2_vertical_practice: f2vert with the practice parameters, no f2."""

    def _values(self, time_model):
        schema = CRP_Time.config_schema()
        defaults = {k: s["default"] for k, s in schema.items() if "default" in s}
        return CRP_Time.resolve_values({**defaults, "time_model": time_model})

    def test_resolves_to_f2vert_with_the_practice_parameters(self):
        values = self._values("f2_vertical_practice")
        self.assertEqual(values["time_model"], "f2_vertical")
        self.assertEqual({k: values[k] for k in PRACTICE_PARAMETERS}, PRACTICE_PARAMETERS)
        self.assertEqual(values["pickup_place_s"], CRP_Time.config_schema()["pickup_place_s"]["default"])
        self.assertTrue(is_practice(values))
        for model in ("f2", "f2_vertical"):
            self.assertEqual(self._values(model)["time_model"], model)
            self.assertFalse(is_practice(self._values(model)))

    def test_one_move_by_hand(self):
        """Retrieval from tier 3 of stack 2 with H = 5 (h_max = 6): 2 * 2.328 * 2 + (12 - 3 - 1.5) * (1.727 + 3.455)
        = 9.312 + 38.865 = 48.177 s."""
        spec = ObjectiveSpec(time_model="f2_vertical", max_tiers=5, num_rows=1, **PRACTICE_PARAMETERS)
        self.assertAlmostEqual(movement_f2_vertical(Movement(7, (2, 1), None, from_tier=3), spec), 48.177, places=9)

    def _finish(self, time_model):
        values = self._values(time_model)
        env = _env("data4-6-7.dat", **values)
        stacks, positions = read_bay(env)
        plan = replay(stacks, positions, _plan_by_lowest_stack(stacks, env.config.max_tiers))
        return finish(lambda: _env("data4-6-7.dat", **values), plan, ObjectiveSpec.from_config(env.config), "A", 0.0)[0]

    def test_finish_reports_f2vert_and_no_f2(self):
        practice, f2vert = self._finish("f2_vertical_practice"), self._finish("f2_vertical")
        self.assertNotIn("crane_time_f2", practice)
        self.assertEqual(set(practice) | {"crane_time_f2"}, set(f2vert))
        self.assertEqual(practice["objective_value"], practice["crane_time_vertical"])
        self.assertEqual(practice["relocations"], f2vert["relocations"])
        self.assertNotEqual(practice["crane_time_vertical"], f2vert["crane_time_vertical"])

    def test_class_label(self):
        path = str(DEFAULT_CASERTA_DIR / "data3-5-1.dat")
        runs = [{"prob_config": {**self._values(model), "layout_file_path": path}, "metrics": {"relocations": 1.0}}
                for model in ("f2_vertical", "f2_vertical_practice")]
        labels = sorted(row["display_label"] for row in summarize_run_dicts(runs)["classes"])
        self.assertEqual(labels, ["3×5 · f2vert", "3×5 · f2vert (practice)"])


class TestCraneTimeAccel(unittest.TestCase):
    """Crane time with acceleration (evaluation only), against values computed by hand. Row 2.54 m, tier 2.591 m;
    model 1: trolley 1.0 m/s, 0.33 m/s², spreader 0.3 m/s²; model 2: trolley 1.2 m/s, 0.35 m/s², spreader 0.5 m/s²;
    spreader 1.5 m/s empty, 0.75 m/s laden."""

    def test_movement_time(self):
        # One trolley row, v²/a above the distance: 2 sqrt(2.54 / 0.33) = 5.549 and 2 sqrt(2.54 / 0.35) = 5.388.
        self.assertAlmostEqual(movement_time(2.54, 1.0, 0.33), 5.5487, places=4)
        self.assertAlmostEqual(movement_time(2.54, 1.2, 0.35), 5.3878, places=4)
        # 3.5 tiers empty, model 1, top speed reached: 9.0685 / 1.5 + 1.5 / 0.3 = 11.0457.
        self.assertAlmostEqual(movement_time(3.5 * 2.591, 1.5, 0.3), 11.0457, places=4)
        # One tier laden, model 1: 2.591 / 0.75 + 0.75 / 0.3 = 5.9547.
        self.assertAlmostEqual(movement_time(2.591, 0.75, 0.3), 5.9547, places=4)
        self.assertEqual(movement_time(0.0, 1.0, 0.33), 0.0)

    def test_moves_by_hand(self):
        """H = 3, h_max = 4. Relocation from tier 3 of stack 1 to tier 2 of stack 2: trolley 2 x 5.4682, spreader
        1 tier empty down 5.2152, 1 tier laden up 5.4547, 2 tiers laden down 8.9093, 2 tiers empty up 7.3834 (means of
        the two models): 37.899 s. Retrieval from tier 1 of stack 2 (hout 1.5): model 1 2 x 8.1103 + 10.1820 +
        12.8640 + 11.1367 + 9.2934 = 59.6967, model 2 2 x 7.6619 + 8.1820 + 11.8640 + 10.1367 + 7.3183 = 52.8248;
        mean 56.2608 s."""
        spec = ObjectiveSpec(time_model="f2", max_tiers=3, num_rows=1)
        self.assertAlmostEqual(movement_accel(Movement(1, (1, 1), (2, 1), from_tier=3, to_tier=2), spec), 37.899,
                               delta=5e-4)
        self.assertAlmostEqual(movement_accel(Movement(1, (2, 1), None, from_tier=1), spec), 56.2608, delta=5e-4)

    def test_reported_for_the_final_plan_only_as_a_metric(self):
        """finish() reports it, equal under f2, f2vert and the practice preset (it depends on the plan only), and
        objective_value is unchanged; CRP-MO does not report it."""
        values = {}
        for model in ("f2", "f2_vertical", "f2_vertical_practice"):
            schema = CRP_Time.config_schema()
            run = CRP_Time.resolve_values({**{k: s["default"] for k, s in schema.items() if "default" in s},
                                           "time_model": model})
            env = _env("data4-6-7.dat", **run)
            stacks, positions = read_bay(env)
            plan = replay(stacks, positions, _plan_by_lowest_stack(stacks, env.config.max_tiers))
            spec = ObjectiveSpec.from_config(env.config)
            metrics, _ = finish(lambda: _env("data4-6-7.dat", **run), plan, spec, "A", 0.0)
            self.assertEqual(metrics["crane_time_accel"], compute_crane_time_accel(plan, spec))   # replay sets tiers
            self.assertEqual(metrics["objective_value"], metrics[spec.time_metric])
            values[model] = metrics["crane_time_accel"]
        self.assertEqual(len(set(values.values())), 1)
        mo = CRP_MO(config=problem_config_for_caserta_dat(DEFAULT_CASERTA_DIR / "data3-5-1.dat", {}))
        mo.reset()
        self.assertNotIn("crane_time_accel", mo.get_metrics())


class TestStopwatch(unittest.TestCase):

    def test_paused_time_is_not_counted(self):
        timer = Stopwatch()
        timer.start()
        time.sleep(0.05)
        with timer.paused():
            time.sleep(0.2)
        time.sleep(0.05)
        self.assertGreaterEqual(timer.stop(), 0.1)
        self.assertLess(timer.elapsed, 0.2)

    def test_stopped_and_paused_when_stopped(self):
        timer = Stopwatch()
        with timer.paused():   # not running: stays stopped afterwards
            pass
        self.assertEqual(timer.stop(), 0.0)
        timer.start()
        first = timer.stop()
        time.sleep(0.05)
        self.assertEqual(timer.stop(), first)


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
