"""
Checks for the CRP-Time class of the restricted tree search
(docs/validation/forster_bortfeldt_2012_restricted.md §10): the registry finds
it (the registry drops modules that fail to import without a message) and not
the unrestricted package (I5); on Caserta 3-3 #1 the class reproduces the plan
of the validated package run (step 2a, §8), so reading the platform's yard as
one bay (I2) changes nothing; the plan does not depend on the time model (I1).

  python -m unittest algorithms.CRP_Time.crp_time_r.forster_bortfeldt_2012.test_algorithm   (from the repo root)

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
corresponding original algorithm paper.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import queue
import threading
import unittest

from core.algorithm_meta import validate_meta
from core.base_algorithm import AlgorithmConfig
from core.benchmarks.caserta import DEFAULT_CASERTA_DIR, problem_config_for_caserta_dat
from core.registry import compatible_algorithms, get_algorithm_class, list_algorithms
from problems.CRP_Time import CRP_Time

from .algorithm import ForsterBortfeldt2012Restricted

# data3-3-1 in step2a_run.jsonl (data repo, CRISP 8731c2e, H = 5, 60 s): 6 relocations, search exhausted;
# moves (d, r, g) with stacks from 0, r None for a retrieval.
STEP2A_3_3_1 = [(0, None, 1), (1, 2, 5), (1, 2, 6), (1, None, 2), (0, 1, 7), (0, None, 3), (2, 1, 6), (2, 1, 5),
                (2, None, 4), (1, None, 5), (1, None, 6), (1, None, 7), (2, 0, 9), (2, None, 8), (0, None, 9)]


def run(instance: str, time_model: str):
    """The final record of the class on one Caserta instance."""
    values = CRP_Time.resolve_values({"time_model": time_model})   # as the web API and main.py do
    cfg = problem_config_for_caserta_dat(DEFAULT_CASERTA_DIR / instance, values)
    q = queue.Queue()
    ForsterBortfeldt2012Restricted(AlgorithmConfig()).train(lambda: CRP_Time(config=cfg), q, threading.Event())
    records = [q.get_nowait() for _ in range(q.qsize())]
    return records[-1]


def as_bay_moves(saved):
    """Saved moves (positions (bay, 1) on Caserta) as (d, r, g) on stack indices from 0."""
    return [(m["from"][0] - 1, None if m["to"] is None else m["to"][0] - 1, m["container_id"]) for m in saved]


class TestRegistry(unittest.TestCase):

    def test_registered_for_crp_time(self):
        name = ForsterBortfeldt2012Restricted.name
        self.assertIn(name, list_algorithms())
        self.assertIs(get_algorithm_class(name), ForsterBortfeldt2012Restricted)
        self.assertIn(name, compatible_algorithms("CRP-Time"))
        self.assertEqual(validate_meta(ForsterBortfeldt2012Restricted), [])

    def test_unrestricted_package_not_registered(self):
        modules = [get_algorithm_class(n).__module__ for n in list_algorithms()]
        self.assertFalse([m for m in modules if ".crp_time_u." in m])


class TestCaserta(unittest.TestCase):

    def test_3_3_1_equals_the_validated_package_run(self):
        final = run("data3-3-1.dat", "f2")
        self.assertEqual(final.metrics["relocations"], 6.0)
        self.assertEqual(as_bay_moves(final.extra["moves"]), STEP2A_3_3_1)

    def test_plan_does_not_depend_on_the_time_model(self):
        plans = [run("data3-3-1.dat", t).extra["moves"] for t in ("f2", "f2_vertical", "f2_vertical_practice")]
        self.assertEqual(plans[0], plans[1])
        self.assertEqual(plans[0], plans[2])


if __name__ == "__main__":
    unittest.main()
