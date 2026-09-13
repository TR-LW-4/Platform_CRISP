"""
Tang2015ILP
<2015> <exact> <restricted> <single-bay> <CRP-R>
ILP / ILP-DK integer program
k --- 0 --- 0 = exact ILP; ≥1 = rolling horizon
time_limit_s --- 3600 --- Gurobi time limit (s)

------------------------------- Reference --------------------------------
L. Tang, W. Jiang, J. Liu, Y. Dong,
"Research into container reshuffling and stacking problems in container
 terminal yards",
IIE Transactions 47 (2015) 751–766.
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
)

from .model import (
    _run_full_ilp,
    _run_rolling_horizon,
)


# ================================================================ #
#  BaseAlgorithm subclass                                            #
# ================================================================ #

class Tang2015ILP(BaseAlgorithm):

    name                = "Tang et al. (2015) ILP [Gurobi]"
    category            = "Exact"
    requires_solver     = True
    solver_backend      = "gurobi"
    description         = "Tang et al. (IIE Trans. 2015) ILP / ILP-DK integer program."
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

        k            = int(extra.get("k", 0))
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
            C     = env.config.num_bays * env.config.num_rows
            H     = env.config.max_tiers

            t0 = time.perf_counter()

            try:
                import gurobipy  # noqa: F401
            except ImportError:
                print("[Tang2015ILP] ERROR: gurobipy not installed.",
                      file=sys.stderr, flush=True)
                self._push(result_queue, step=seed+1, metric=float("inf"),
                           metrics={"relocations": float("inf"), "error": 1.0},
                           progress=(seed+1)/n_seeds)
                continue

            use_exact = (k <= 0 or k >= N - 1)
            if use_exact:
                result = _run_full_ilp(yard, N, C, H, time_limit_s, output_flag)
            else:
                result = _run_rolling_horizon(yard, N, C, H, k, time_limit_s, output_flag)

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

            mode_str = "ILP(exact)" if use_exact else f"ILP{k}(rolling)"
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
                f"[Tang2015ILP/{mode_str}] seed={seed}  N={N} C={C} H={H}  "
                f"reloc={metrics.get('relocations')}  "
                f"crane={metrics.get('crane_time')}  "
                f"opt={result.get('optimal', False)}  "
                f"vars={result.get('n_vars', 0)}  t={elapsed:.2f}s",
                file=sys.stderr, flush=True,
            )

        if all_metrics:
            agg = {k2: float(np.mean([mm[k2] for mm in all_metrics if k2 in mm]))
                   for k2 in all_metrics[0]}
            self._push(result_queue, step=n_seeds, metric=self._best_metric,
                       metrics=agg, progress=1.0)

    def get_best_solution(self) -> Optional[List[int]]:
        return self._best_solution

    @classmethod
    def config_schema(cls) -> Dict:
        base = super().config_schema()
        base.update({
            "k": {
                "type": "int", "default": 0, "min": 0, "max": 100,
                "label": "k  (0 = exact ILP; >=1 = ILP-DK rolling horizon)",
                "help": (
                    "k=0: solve full ILP to optimality. "
                    "k=1..N-1: rolling horizon -- at each retrieval step solve a "
                    "k-stage sub-ILP; much faster, near-optimal. "
                    "Paper's dynamic experiments use k=5..8 (Table 5)."
                ),
            },
            "time_limit_s": {
                "type": "float", "default": 3600.0, "min": 1.0, "max": 86400.0,
                "label": "Time limit (s)",
                "help": (
                    "For k=0 (exact): total solve time limit. "
                    "For k>0 (rolling): per-step budget = time_limit / (N-1)."
                ),
            },
        })
        return base
