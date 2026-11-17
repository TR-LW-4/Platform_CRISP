"""
ForsterBortfeldt2012Restricted
<2012> <heuristic> <relocations> <single-bay> <CRP-Time>
Tree search minimising the number of moves, restricted relocations
time_limit_s --- 60 --- Time limit (s)

Integration decisions I1-I5: docs/validation/forster_bortfeldt_2012_restricted.md
§10. The search (search.py, state.py) is the restricted package validated in
that log; this class only reads the bay, runs it and reports the plan.

------------------------------- Reference --------------------------------
F. Forster, A. Bortfeldt,
"A tree search procedure for the container relocation problem",
Computers & Operations Research 39 (2012) 299-309.
------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import multiprocessing as mp
from typing import Callable, Dict, List, Optional

from algorithms.CRP_Time.utils.bay import read_bay
from algorithms.CRP_Time.utils.crane_time import ObjectiveSpec, check_objective
from algorithms.CRP_Time.utils.finish import finish
from algorithms.CRP_Time.utils.params import parameters
from algorithms.CRP_Time.utils.plan import actions, replay, saved_moves
from algorithms.CRP_Time.utils.timer import Stopwatch
from core.base_algorithm import AlgorithmConfig, BaseAlgorithm

from .search import Params, TreeSearch
from .state import Bay


class ForsterBortfeldt2012Restricted(BaseAlgorithm):

    name = "Forster–Bortfeldt (2012) Tree Search, restricted"
    category = "Heuristic"
    description = (
        "Tree search of Forster & Bortfeldt (2012): a greedy start, then compound moves ranked by the paper's "
        "lower bound, keeping the plan with the fewest moves (nSucc 10, cmStopThreshold 150, maxFlg_bb 3, "
        "maxGg 2; p. 306). Relocations are restricted to the containers above the next one to retrieve (the "
        "paper's method is unrestricted). As in the paper, the number of relocations is the objective; crane "
        "time under the selected model is evaluated afterwards. The "
        "search is deterministic, but the time limit is wall-clock, so on larger instances results can differ "
        "between runs and machines."
    )
    compatible_problems = ["CRP-Time"]
    geometry = "single-bay"
    objectives = ["relocations"]
    fidelity = "adapted"
    relocation_rule = "restricted"

    def __init__(self, config: Optional[AlgorithmConfig] = None):
        super().__init__(config)

    @classmethod
    def config_schema(cls) -> Dict:
        base = super().config_schema()
        # max_iterations and report_interval are not read; the search is deterministic, so no seed.
        for key in ("max_iterations", "report_interval", "seed"):
            base.pop(key)
        base.update({
            "time_limit_s": {
                "type": "float", "default": 60.0, "min": 0.1, "max": 600.0,
                "label": "Time limit (s)",
                "help": "Wall-clock limit of the search per instance, the greedy start included (paper: 60 s).",
            },
        })
        return base

    def train(self, problem_factory: Callable, result_queue: mp.Queue, stop_event: mp.Event) -> None:
        if stop_event.is_set():
            return
        env = problem_factory()
        # I2: the S stacks of the platform's yard (S bays of one row on Caserta) as one bay of S stacks.
        stacks, positions = read_bay(env)
        spec = ObjectiveSpec.from_config(env.config)
        check_objective(spec, self.name)
        time_limit = max(0.1, float(parameters(self.config, type(self))["time_limit_s"]))

        timer = Stopwatch()
        timer.start()
        search = TreeSearch(Bay.from_stacks(stacks, int(env.config.max_tiers)), Params(time_limit=time_limit))
        moves = search.run()
        timer.stop()

        plan = replay(stacks, positions, [(d, r) for d, r, _g in moves])
        metrics, env_final = finish(problem_factory, plan, spec, self.name, timer.elapsed)
        self._best_metric = float(metrics["objective_value"])
        self._best_solution = actions(plan, env.config.num_rows)
        self._push(result_queue, step=1, metric=self._best_metric, metrics=metrics, progress=1.0,
                   snapshot=env_final.get_state_snapshot(),
                   extra={"moves": saved_moves(plan), "time_limit_hit": search.timed_out,
                          "lower_bound_reached": search.optimal})

    def get_best_solution(self) -> Optional[List[int]]:
        return self._best_solution
