"""
Galle2018CRPI
<2018> <exact> <restricted> <single-bay> <CRP-R>
CRP-I binary integer program
time_limit_s --- 3600 --- Gurobi time limit (s)

------------------------------- Reference --------------------------------
V. Galle, C. Barnhart, P. Jaillet,
"A new binary formulation of the restricted Container Relocation Problem
 based on a binary encoding of configurations",
European Journal of Operational Research 267 (2018) 467–477.
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
from algorithms.CRP_R.solver_ip.common import (
    merge_solver_and_plan_metrics,
    occupancy_from_yard,
    occupancy_stages_to_plan,
    plan_to_moves_extra,
    priority_to_container_id,
    stack_keys,
)

from .model import (
    _solve_ip,
)


# ================================================================ #
#  BaseAlgorithm subclass                                            #
# ================================================================ #

class GalleCRPI2018(BaseAlgorithm):

    name                = "Galle (2018) CRP-I [Gurobi]"
    category            = "Exact"
    requires_solver     = True
    solver_backend      = "gurobi"
    description         = "Galle et al. (EJOR 2018) CRP-I binary integer program."
    compatible_problems = ["CRP-R"]
    def __init__(self, config: Optional[AlgorithmConfig] = None) -> None:
        super().__init__(config)

    # ---------------------------------------------------------------- #
    # train                                                              #
    # ---------------------------------------------------------------- #

    def train(
        self,
        problem_factory: Callable,
        result_queue:    mp.Queue,
        stop_event:      mp.Event,
    ) -> None:
        cfg   = self.config
        extra = cfg.extra or {}

        time_limit_s     = float(extra.get("time_limit_s",   3600.0))
        use_gap          = bool(extra.get("use_gap",          True))
        use_incumbent    = bool(extra.get("use_incumbent",    True))
        use_preprocessing = bool(extra.get("use_preprocessing", True))
        memory_limit     = float(extra.get("memory_limit",   8e10))
        output_flag      = int(extra.get("output_flag",      0))
        n_seeds = 1  # multi-seed eval removed; single run only

        all_metrics: List[Dict] = []

        for seed in range(n_seeds):
            if stop_event.is_set():
                break

            env = problem_factory()
            env.reset(seed=seed, options={"skip_auto_retrieve": True})

            yard = env.yard
            C    = len(env.containers)
            cfg_ = env.config
            S    = cfg_.num_bays * cfg_.num_rows
            T    = cfg_.max_tiers

            t0 = time.perf_counter()

            # Guard: Gurobi import check
            try:
                import gurobipy  # noqa: F401
            except ImportError:
                print(
                    "[GalleCRPI2018] ERROR: gurobipy not installed. "
                    "Run:  pip install gurobipy",
                    file=sys.stderr, flush=True,
                )
                self._push(result_queue, step=seed+1, metric=float("inf"),
                           metrics={"relocations": float("inf"),
                                    "error": 1.0},
                           progress=(seed+1)/n_seeds)
                continue

            result = _solve_ip(
                yard=yard, C=C, S=S, T=T,
                time_limit_s=time_limit_s,
                use_gap=use_gap,
                use_incumbent=use_incumbent,
                use_preprocessing=use_preprocessing,
                memory_limit=memory_limit,
                output_flag=output_flag,
            )

            elapsed = time.perf_counter() - t0
            keys = stack_keys(yard)
            pri = priority_to_container_id(yard)
            plan = None
            if result["obj"] is not None:
                occupancy = result.get("occupancy") or {1: occupancy_from_yard(yard)}
                plan = occupancy_stages_to_plan(
                    occupancy,
                    result.get("y_vals") or {},
                    C,
                    keys,
                    pri,
                    max_tiers=T,
                )
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
            metrics["memory_out"] = 1.0 if result.get("memory_out") else 0.0
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

            self._push(
                result_queue,
                step     = seed + 1,
                metric   = primary,
                metrics  = metrics,
                progress = (seed + 1) / n_seeds,
                extra    = {
                    "moves": plan_to_moves_extra(plan) if plan is not None else [],
                    "validation_errors": env.get_last_validation_errors(),
                },
            )

            status_str = (
                "optimal" if result["optimal"]
                else ("time_out" if result["time_out"]
                      else ("memory_out" if result["memory_out"]
                            else "no_solution"))
            )
            print(
                f"[GalleCRPI2018] seed={seed}  C={C} S={S} T={T}  "
                f"relocations={metrics.get('relocations')}  "
                f"crane={metrics.get('crane_time')}  status={status_str}  "
                f"vars={result['n_vars']}  constrs={result['n_constrs']}  "
                f"t={elapsed:.3f}s",
                file=sys.stderr, flush=True,
            )

        if all_metrics:
            agg = {
                k: float(np.mean([m[k] for m in all_metrics if k in m]))
                for k in all_metrics[0]
            }
            self._push(
                result_queue,
                step     = n_seeds,
                metric   = self._best_metric,
                metrics  = agg,
                progress = 1.0,
            )

    def get_best_solution(self) -> Optional[List[int]]:
        return self._best_solution

    # ---------------------------------------------------------------- #
    # config_schema (for GUI parameter widgets)                          #
    # ---------------------------------------------------------------- #

    @classmethod
    def config_schema(cls) -> Dict:
        base = super().config_schema()
        base.update({
            "time_limit_s": {
                "type": "float", "default": 3600.0, "min": 1.0, "max": 86400.0,
                "label": "Time limit (s)",
                "help": "Per-instance Gurobi time limit in seconds.",
            },
            "use_gap": {
                "type": "bool", "default": True,
                "label": "Use MIP gap (faster)",
                "help": "Set MIPGapAbs=0.99 — safe for integer objectives, speeds convergence.",
            },
            "use_incumbent": {
                "type": "bool", "default": True,
                "label": "Warm-start with MinMax heuristic",
                "help": "Provide Caserta 2012 MinMax solution as Gurobi first incumbent.",
            },
            "use_preprocessing": {
                "type": "bool", "default": True,
                "label": "Pre-processing (fix unchanged columns)",
                "help": "Fix binary encoding columns that cannot change before first move.",
            },
            "memory_limit": {
                "type": "float", "default": 8e10, "min": 1e8, "max": 1e12,
                "label": "Memory limit (#vars × #constrs)",
                "help": "Skip instance if #vars × #constrs exceeds this threshold.",
            },
        })
        return base
