"""
Caserta2012BRPII
<2012> <exact> <restricted> <single-bay> <CRP-R>
BRP-II item-indexed 0-1 integer program
time_limit_s --- 3600 --- Gurobi time limit (s)

------------------------------- Reference --------------------------------
M. Caserta, S. Schwarze, S. Voß,
"A mathematical formulation and complexity considerations for the
 blocks relocation problem",
European Journal of Operational Research 219 (2012) 96–104.
------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import multiprocessing as mp
import sys
import time
from typing import Callable, Dict, List, Optional

import numpy as np

from core.base_algorithm import BaseAlgorithm, AlgorithmConfig
from core.plan import RelocationPlan
from algorithms.CRP_R.solver_ip.common import (
    merge_solver_and_plan_metrics,
    plan_from_brp_ii_vars,
    plan_to_moves_extra,
    sequential_retrieval_plan,
)

from .model import (
    _initial_layout,
    _stacks_as_lists,
    _minmax_ub,
    _solve_brp_ii,
)


# ================================================================ #
#  BaseAlgorithm subclass                                            #
# ================================================================ #

class Caserta2012BRPII(BaseAlgorithm):

    name                = "Caserta (2012) BRP-II [Gurobi]"
    category            = "Exact"
    requires_solver     = True
    solver_backend      = "gurobi"
    description         = "Caserta et al. (EJOR 2012) BRP-II integer program."
    compatible_problems = ["CRP-R"]
    geometry            = "single-bay"
    objectives          = ["relocations"]
    fidelity            = "faithful"
    def __init__(self, config: Optional[AlgorithmConfig] = None) -> None:
        super().__init__(config)

    def train(
        self,
        problem_factory: Callable,
        result_queue:    mp.Queue,
        stop_event:      mp.Event,
    ) -> None:
        cfg   = self.config
        extra = cfg.extra or {}

        time_limit_s = float(extra.get("time_limit_s", 3600.0))
        output_flag  = int(extra.get("output_flag", 0))
        n_seeds = 1  # multi-seed eval removed; single run only
        all_metrics: List[Dict] = []

        for seed in range(n_seeds):
            if stop_event.is_set():
                break

            env = problem_factory()
            env.reset(seed=seed, options={"skip_auto_retrieve": True})

            yard  = env.yard
            N     = len(env.containers)
            W     = env.config.num_bays * env.config.num_rows
            H     = env.config.max_tiers

            t0 = time.perf_counter()

            try:
                import gurobipy  # noqa: F401
            except ImportError:
                print("[Caserta2012BRPII] ERROR: gurobipy not installed.",
                      file=sys.stderr, flush=True)
                self._push(result_queue, step=seed+1, metric=float("inf"),
                           metrics={"relocations": float("inf"), "error": 1.0},
                           progress=(seed+1)/n_seeds)
                continue

            stacks      = _stacks_as_lists(yard)
            initial_pos = _initial_layout(yard, W)

            heuristic_ub = _minmax_ub(stacks, N, W, H)
            plan: Optional[RelocationPlan] = None
            result: Dict = {
                "obj": 0, "optimal": True, "time_out": False,
                "n_vars": 0, "n_constrs": 0, "solve_time": 0.0,
            }

            if heuristic_ub == 0:
                plan = sequential_retrieval_plan(yard)
            else:
                result = _solve_brp_ii(
                    initial_pos, N, W, H, time_limit_s, output_flag,
                )
                if result["obj"] is not None:
                    plan = plan_from_brp_ii_vars(
                        yard, result.get("x_vals") or (), result.get("y_vals") or (), N,
                    )

            elapsed = time.perf_counter() - t0
            if plan is not None:
                metrics = merge_solver_and_plan_metrics(
                    result, env.validate_plan(plan),
                )
            else:
                metrics = merge_solver_and_plan_metrics(
                    result,
                    {
                        "relocations": float("inf"),
                        "crane_time": float("inf"),
                        "time": float("inf"),
                        "feasible": 0.0,
                        "completed": 0.0,
                        "validated": 1.0,
                        "validation_conflicts": 1.0,
                    },
                )
            metrics["heuristic_ub"] = float(heuristic_ub)
            metrics["solve_time_s"] = round(float(result.get("solve_time") or 0.0), 4)
            all_metrics.append(metrics)

            primary = float(metrics.get("relocations", float("inf")))
            if primary < self._best_metric:
                self._best_metric = primary
                self._best_solution = [
                    (m.to_pos[0] - 1) * env.config.num_rows + (m.to_pos[1] - 1)
                    for m in (plan.movements if plan is not None else [])
                    if m.to_pos is not None
                ]

            extra = {
                "moves": plan_to_moves_extra(plan) if plan is not None else [],
                "validation_errors": env.get_last_validation_errors(),
            }
            self._push(
                result_queue,
                step     = seed + 1,
                metric   = primary,
                metrics  = metrics,
                progress = (seed + 1) / n_seeds,
                extra    = extra,
            )
            print(
                f"[Caserta2012BRPII] seed={seed}  N={N} W={W} H={H}  "
                f"reloc={metrics.get('relocations')}  "
                f"crane={metrics.get('crane_time')}  "
                f"opt={result.get('optimal')}  "
                f"vars={result.get('n_vars')}  t={elapsed:.2f}s",
                file=sys.stderr, flush=True,
            )

        if all_metrics:
            agg = {k: float(np.mean([mm[k] for mm in all_metrics if k in mm]))
                   for k in all_metrics[0]}
            self._push(result_queue, step=n_seeds, metric=self._best_metric,
                       metrics=agg, progress=1.0)

    def get_best_solution(self) -> Optional[List[int]]:
        return self._best_solution

    @classmethod
    def config_schema(cls) -> Dict:
        base = super().config_schema()
        base.update({
            "time_limit_s": {
                "type": "float", "default": 3600.0, "min": 1.0, "max": 86400.0,
                "label": "Time limit (s)",
                "help": "Per-instance Gurobi time limit. Paper uses 60 min (3600 s) or 1 day (86400 s).",
            },
        })
        return base
