"""
DeMeloSilva2018RBRP
<2018> <exact> <restricted> <single-bay> <CRP-R>
Time-indexed r-BRP MIP (m1 / m2)
variant --- m1 --- m1 stronger LP; m2 more compact
time_limit_s --- 3600 --- Gurobi time limit (s)

------------------------------- Reference --------------------------------
M. de Melo da Silva, S. Toulouse, R. Wolfler Calvo,
"A new effective unified model for solving the Pre-marshalling and
 Block Relocation Problems",
European Journal of Operational Research 271 (2018) 40–56.
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
    plan_to_moves_extra,
    sequential_retrieval_plan,
)

from .model import (
    _minmax_upper_bound,
    _plan_from_demelo_flow,
    _solve_rbrp_m1,
    _solve_rbrp_m2,
)


# ================================================================ #
#  BaseAlgorithm subclass                                            #
# ================================================================ #

class DeMeloSilva2018RBRP(BaseAlgorithm):

    name                = "de Melo da Silva (2018) r-BRP [Gurobi]"
    category            = "Exact"
    requires_solver     = True
    solver_backend      = "gurobi"
    description         = "de Melo da Silva et al. (EJOR 2018) r-BRP m1/m2 integer program."
    compatible_problems = ["CRP-R"]
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

        variant      = str(extra.get("variant",      "m1"))
        time_limit_s = float(extra.get("time_limit_s", 3600.0))
        output_flag  = int(extra.get("output_flag",   0))
        n_seeds = 1  # multi-seed eval removed; single run only
        all_metrics: List[Dict] = []

        for seed in range(n_seeds):
            if stop_event.is_set():
                break

            env = problem_factory()
            env.reset(seed=seed, options={"skip_auto_retrieve": True})

            yard  = env.yard
            C     = len(env.containers)
            S     = env.config.num_bays * env.config.num_rows
            H     = env.config.max_tiers
            G     = C  # unique priorities → G = N

            T_ub = _minmax_upper_bound(yard, C, S, H)
            plan = None
            result: Dict = {
                "obj": 0, "optimal": True, "time_out": False,
                "n_vars": 0, "n_constrs": 0, "solve_time": 0.0,
            }
            if T_ub <= 0:
                plan = sequential_retrieval_plan(yard)
            else:
                t0 = time.perf_counter()
                try:
                    import gurobipy  # noqa: F401
                except ImportError:
                    print("[DeMeloSilva2018RBRP] ERROR: gurobipy not installed.",
                          file=sys.stderr, flush=True)
                    self._push(result_queue, step=seed+1, metric=float("inf"),
                               metrics={"relocations": float("inf"), "error": 1.0},
                               progress=(seed+1)/n_seeds)
                    continue

                if variant == "m2":
                    result = _solve_rbrp_m2(yard, C, S, H, G, T_ub,
                                            time_limit_s, output_flag)
                else:
                    result = _solve_rbrp_m1(yard, C, S, H, G, T_ub,
                                            time_limit_s, output_flag)
                elapsed = time.perf_counter() - t0
                plan = result.get("plan")

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
            metrics["T_ub"] = float(max(T_ub, 0))
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
            print(
                f"[DeMeloSilva2018RBRP/{variant}] seed={seed}  "
                f"C={C} S={S} H={H} T_ub={T_ub}  "
                f"reloc={metrics.get('relocations')}  "
                f"crane={metrics.get('crane_time')}  "
                f"opt={result.get('optimal')}  "
                f"t={float(result.get('solve_time') or 0.0):.2f}s",
                file=sys.stderr, flush=True,
            )

        if all_metrics:
            agg = {k: float(np.mean([m[k] for m in all_metrics if k in m]))
                   for k in all_metrics[0]}
            self._push(result_queue, step=n_seeds, metric=self._best_metric,
                       metrics=agg, progress=1.0)

    def get_best_solution(self) -> Optional[List[int]]:
        return self._best_solution

    @classmethod
    def config_schema(cls) -> Dict:
        base = super().config_schema()
        base.update({
            "variant": {
                "type": "str", "default": "m1", "options": ["m1", "m2"],
                "label": "Model variant",
                "help": (
                    "m1: stronger LP relaxation (≈82% instances solved, Caserta bench). "
                    "m2: more compact (faster per node, weaker LP, ≈66% instances solved)."
                ),
            },
            "time_limit_s": {
                "type": "float", "default": 3600.0, "min": 1.0, "max": 86400.0,
                "label": "Time limit (s)",
            },
        })
        return base
