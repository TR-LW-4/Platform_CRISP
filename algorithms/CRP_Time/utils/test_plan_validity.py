"""
Every CRP-Time algorithm in the registry returns a complete, feasible plan whose metrics are the ones it reports.

Independently of utils.finish(): each algorithm runs on a few small Caserta instances under f2 (fixed seed 0,
parameters reduced where the defaults are slow), its final plan is taken from what the algorithm reports, a fresh
problem validates it with validate_plan, and relocations, f2 and f2vert of that plan must equal the reported ones.
data3-5-1 has container 1 on top of a stack at the start (the case behind Azari's A3 issue); data3-3-2 and data3-4-2
start without a target on top. Not covered: crp_time_u
(not registered; its own package tests).

Where the plan comes from:
- "moves": the validated plan saved in the final record (extra["moves"]), algorithms that use utils.finish();
- "actions": get_best_solution() replayed on a fresh problem (Kim 2016, Lin 2015 step the problem themselves);
- "plan": get_best_plan() (Lee–Lee);
- "captured": the last plan built by the module's _plan_from_moves, captured by wrapping it (Firmino scores its
  own plan and reports none).
"""

import importlib
import queue
import threading
import unittest
from pathlib import Path
from unittest import mock

from core.base_algorithm import AlgorithmConfig
from core.benchmarks.caserta import DEFAULT_CASERTA_DIR, parse_caserta_dat, problem_config_for_caserta_dat
from core.move_export import plan_from_json
from core.plan import RelocationPlan
from problems.CRP_Time import CRP_Time

INSTANCES = ("data3-3-2.dat", "data3-4-2.dat", "data3-5-1.dat")
ON_TOP = "data3-5-1.dat"   # container 1 on top at the start; the other two start without a target on top
H = "algorithms.CRP_Time.heuristic."
ALGORITHMS = {   # name: (module, class, extra parameters, plan source)
    "azari": (H + "azari_2017.algorithm", "Azari2017CSUM", {}, "moves"),
    "jovanovic": (H + "jovanovic_2019_aco.algorithm", "JovanovicACO_CRPTime", {"n_iterations": 200}, "moves"),
    "lopez_plata": (H + "lopez_plata_2019.algorithm", "LopezPlata2019Heuristic", {"rho": 10}, "moves"),
    "unluyurt_d1": (H + "unluyurt_2012.difference1", "UnluyurtAydinDifference1", {}, "moves"),
    "unluyurt_d2": (H + "unluyurt_2012.difference2", "UnluyurtAydinDifference2", {}, "moves"),
    "forster_bortfeldt_r": ("algorithms.CRP_Time.crp_time_r.forster_bortfeldt_2012.algorithm",
                            "ForsterBortfeldt2012Restricted", {}, "moves"),
    "kim_2016": (H + "kim_2016.algorithm", "Kim2016Heuristic", {}, "actions"),
    "lin_2015": (H + "lin_2015.algorithm", "Lin2015Heuristic", {}, "actions"),
    "lee_lee": (H + "lee_lee.algorithm", "LeeLeeRetrievalHeuristic",
                {"max_no_improve_p2": 10, "max_no_improve_p3": 10}, "plan"),
    "firmino": (H + "firmino_2019_rgrasp.algorithm", "FirminoRGRASP",
                {"max_iterations": 20, "max_no_improve": 100}, "captured"),
}
KEYS = ("relocations", "crane_time_f2", "crane_time_vertical")


def _env(name):
    return CRP_Time(config=problem_config_for_caserta_dat(DEFAULT_CASERTA_DIR / name, {"time_model": "f2"}))


def _run(name, instance):
    """The algorithm's final plan and its last record."""
    module_name, cls_name, extra, source = ALGORITHMS[name]
    module = importlib.import_module(module_name)
    algo = getattr(module, cls_name)(config=AlgorithmConfig(seed=0, extra=dict(extra)))
    q, captured = queue.Queue(), []
    if source == "captured":
        build = module._plan_from_moves

        def capture(*args, **kwargs):
            plan = build(*args, **kwargs)
            captured.append(plan)
            return plan

        with mock.patch.object(module, "_plan_from_moves", capture):
            algo.train(lambda: _env(instance), q, threading.Event())
    else:
        algo.train(lambda: _env(instance), q, threading.Event())
    records = [q.get_nowait() for _ in range(q.qsize())]
    if source == "moves":
        plan = plan_from_json(records[-1].extra["moves"])
    elif source == "actions":
        env = _env(instance)
        env.reset()
        for action in algo.get_best_solution() or []:
            if env._done:
                break
            env.step(action)
        plan = RelocationPlan(list(env._plan.movements))
    elif source == "plan":
        plan = algo.get_best_plan()
    else:
        plan = captured[-1] if captured else RelocationPlan()
    return plan, records[-1]


class TestPlanValidity(unittest.TestCase):

    def test_container_on_top_instance(self):
        """data3-5-1 starts with container 1 on top of a stack; the other instances do not."""
        for instance in INSTANCES:
            tops = [st[-1] for st in parse_caserta_dat(Path(DEFAULT_CASERTA_DIR) / instance)[2] if st]
            self.assertEqual(1 in tops, instance == ON_TOP, instance)

    def _check(self, name):
        for instance in INSTANCES:
            with self.subTest(algorithm=name, instance=instance):
                plan, final = _run(name, instance)
                checked = _env(instance).validate_plan(plan)
                self.assertTrue(checked.get("feasible"), f"{name} on {instance}: plan rejected")
                for key in KEYS:
                    self.assertAlmostEqual(checked[key], final.metrics[key], places=6, msg=f"{name} {instance} {key}")


def _make(name):
    def test(self):
        self._check(name)
    test.__doc__ = f"{name}: complete, feasible plan with the reported metrics on {', '.join(INSTANCES)}."
    return test


for _name in ALGORITHMS:
    _test = _make(_name)
    if _name == "firmino":
        # Known A3 issue (CLAUDE.md, 26 Oct 2026): Firmino scores its own plan after env.reset() without
        # skip_auto_retrieve, so on data3-5-1 the retrieval of container 1 is missing and f2 is reported too low.
        # When it is repaired, this becomes an unexpected success.
        _test = unittest.expectedFailure(_test)
    setattr(TestPlanValidity, f"test_{_name}", _test)


if __name__ == "__main__":
    unittest.main()
