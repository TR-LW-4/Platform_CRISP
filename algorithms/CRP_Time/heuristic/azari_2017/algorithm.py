"""
Azari2017CSUM
<2017> <heuristic> <time> <single-bay> <CRP-Time>
CSUM tree search minimizing crane working time
gbh_time_limit_s --- 3 --- GBH time limit T (s)
csum_time_limit_s --- 5 --- CSUM time limit (s)

------------------------------- Reference --------------------------------
E. Azari, H. Eskandari, A. Nourmohammadi,
"Decreasing the crane working time in retrieving the containers from a bay",
Scientia Iranica E 24 (2017) 309–318.
------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import copy
import multiprocessing as mp
import sys
from typing import Callable, Dict, List, Optional

import numpy as np

from algorithms.CRP_Time.utils.bay import read_bay
from algorithms.CRP_Time.utils.crane_time import check_objective
from algorithms.CRP_Time.utils.params import parameters
from algorithms.CRP_Time.utils.plan import actions, replay
from core.base_algorithm import AlgorithmConfig, BaseAlgorithm
from core.objectives import KinematicsModel, ObjectiveSpec, evaluate_plan_objectives

from .csum import csum
from .gbh import gbh


class Azari2017CSUM(BaseAlgorithm):

    name = "Azari et al. (2017) CSUM"
    category = "Heuristic"
    description = (
        "CSUM heuristic for CRP-Time (Scientia Iranica 2017). "
        "Pipeline: GBH initial solution -> A/B partition by relocation count -> "
        "CSUM tree search minimizing crane working time (TCW). "
        "A-set containers follow GBH branching; B-set containers follow "
        "constant-summation branching. "
        "Implementation is self-contained and does not depend on other algorithms."
    )
    compatible_problems = ["CRP-Time"]
    geometry = "single-bay"
    objectives = ["crane_time"]
    fidelity = "faithful"
    def __init__(self, config: Optional[AlgorithmConfig] = None):
        super().__init__(config)

    def train(
        self,
        problem_factory: Callable,
        result_queue: mp.Queue,
        stop_event: mp.Event,
    ) -> None:
        cfg = self.config
        n_seeds = 1  # multi-seed eval removed; single run only
        par = parameters(cfg, type(self))
        gbh_limit = max(0.1, float(par["gbh_time_limit_s"]))
        csum_limit = max(0.1, float(par["csum_time_limit_s"]))
        all_metrics: List[Dict[str, float]] = []

        for seed_idx in range(n_seeds):
            if stop_event.is_set():
                break

            env = problem_factory()
            env.config.seed = int(cfg.seed) + seed_idx
            stacks, positions = read_bay(env)
            initial_yard = copy.deepcopy(env.yard)
            objective_spec = ObjectiveSpec.from_config(env.config)
            check_objective(objective_spec, self.name)
            # Only for the legacy RMGC metric of the shared evaluator.
            kinematics = KinematicsModel.from_config_extra(env.config.extra)
            max_tiers = int(env.config.max_tiers)

            moves, found = [], True
            if any(stacks):
                initial = gbh(stacks, max_tiers, gbh_limit)
                if initial is None:
                    raise RuntimeError(f"GBH found no complete solution within gbh_time_limit_s = {gbh_limit} s")
                moves = csum(stacks, positions, max_tiers, initial, objective_spec, csum_limit)
                found = moves is not None
                if not found:
                    # Not covered by the paper: fall back to the GBH solution.
                    moves = initial
                    print(
                        "[Azari2017CSUM] CSUM found no complete solution within "
                        "csum_time_limit_s; returning the GBH solution.",
                        file=sys.stderr, flush=True,
                    )
            extra = {"csum_solution_found": found}

            plan = replay(stacks, positions, moves)
            metrics = evaluate_plan_objectives(
                plan, objective_spec, kinematics=kinematics, initial_yard=initial_yard,
            )
            metrics["steps"] = float(plan.num_moves())
            all_metrics.append(metrics)

            primary = float(metrics["objective_value"])
            if primary < self._best_metric:
                self._best_metric = primary
                self._best_solution = actions(plan, env.config.num_rows)

            self._push(
                result_queue,
                step=seed_idx + 1,
                metric=primary,
                metrics=metrics,
                progress=(seed_idx + 1) / n_seeds,
                snapshot=env.get_state_snapshot(),
                extra=extra,
            )

        if all_metrics:
            agg = {
                k: float(np.mean([m[k] for m in all_metrics if k in m]))
                for k in all_metrics[0]
            }
            self._push(
                result_queue,
                step=n_seeds,
                metric=self._best_metric,
                metrics=agg,
                progress=1.0,
                extra=extra,
            )

    def get_best_solution(self) -> Optional[List[int]]:
        return self._best_solution

    @classmethod
    def config_schema(cls) -> Dict:
        base = super().config_schema()
        # Not read by this method.
        for key in ("max_iterations", "report_interval"):
            base.pop(key)
        base["seed"] = {
            **base["seed"],
            "help": (
                "No effect: GBH and CSUM have no random component, and the "
                "layout is read from the Caserta file."
            ),
        }
        base.update({
            "gbh_time_limit_s": {
                "type": "float", "default": 3.0, "min": 0.1, "max": 120.0,
                "label": "GBH time limit T (s)",
                "help": "Wall-clock limit for the GBH search that yields the initial solution (paper: 3 s).",
            },
            "csum_time_limit_s": {
                "type": "float", "default": 5.0, "min": 0.1, "max": 120.0,
                "label": "CSUM time limit (s)",
                "help": "Wall-clock limit for the CSUM search, starting after GBH (paper: 5 s).",
            },
        })
        return base
