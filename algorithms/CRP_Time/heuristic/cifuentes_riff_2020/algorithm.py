"""
G-CREM (Cifuentes & Riff, 2020) — Main GRASP driver for CRP-Time.

Reference
---------
C.D. Cifuentes and M.C. Riff,
"G-CREM: A GRASP approach to solve the container relocation problem
for multibays", Applied Soft Computing 97 (2020) 106721.

Role on the platform
--------------------
Main target problem: ``CRP-Time`` (multi-bay RMGC, Lee & Lee 2010
kinematics).  Same problem family as Lin, Lee & Lee (2015) and the
current platform's Lee–Lee (2010) three-phase heuristic.

Algorithm structure
-------------------
For each random seed (``num_eval_seeds``), and for each GRASP restart
on that seed (``max_restarts``):

    x+  ← constructive_phase(initial_yard, k, rng)
    x*  ← x+
    for _ in range(max_inner_iters):
        x+ ← hill-climb(x+) with RIL repair
        if Eval(x+) < Eval(x*): x* = x+

The weighted evaluation function is  α · (# moves) + β · crane_time,
with defaults  ``α = 0.3, β = 0.005``  as tuned by ParamILS in the paper.
``k = 3`` for the RCL, also from the paper.
"""

from __future__ import annotations

import copy
import multiprocessing as mp
import random
from typing import Callable, Dict, List, Optional

import numpy as np

from core.base_algorithm import BaseAlgorithm, AlgorithmConfig
from core.objectives     import (
    KinematicsModel,
    compute_crane_time,
    lower_bound_relocations,
)
from core.plan           import RelocationPlan
from .constructive       import constructive_phase
from .local_search       import eval_plan, local_search


class CifuentesRiff2020GRASP(BaseAlgorithm):

    name                = "Cifuentes–Riff (2020) G-CREM"
    category            = "Heuristic"
    description         = (
        "[native multi-bay]  "
        "GRASP for multi-bay CRP (Cifuentes & Riff, ASOC 2020). "
        "Constructive phase uses a myopic function (remaining stack "
        "capacity) + size-adaptive RCL (top-k distinct values). "
        "Post-processing hill-climbs by editing the most-relocated "
        "container's last relocation and repairing with RIL "
        "(Wu & Ting 2010).  Objective = α·moves + β·crane_time; "
        "defaults α=0.3, β=0.005, k=3 follow the paper's ParamILS tune."
    )
    compatible_problems = ["CRP-Time", "BRP-Fixed"]
    step_label          = "Seed"

    def __init__(self, config: Optional[AlgorithmConfig] = None):
        super().__init__(config)
        self._best_plan: Optional[RelocationPlan] = None

    # ---------------------------------------------------------------- #
    # Train                                                              #
    # ---------------------------------------------------------------- #

    def train(
        self,
        problem_factory: Callable,
        result_queue:    mp.Queue,
        stop_event:      mp.Event,
    ) -> None:
        cfg       = self.config
        rng_seed  = cfg.seed
        n_seeds   = max(1, cfg.num_eval_seeds)

        k         = int  (cfg.extra.get("k",               3))
        alpha     = float(cfg.extra.get("alpha",           0.3))
        beta      = float(cfg.extra.get("beta",            0.005))
        max_rst   = int  (cfg.extra.get("max_restarts",    5))
        max_inner = int  (cfg.extra.get("max_inner_iters", 50))

        all_metrics: List[Dict] = []

        for seed_idx in range(n_seeds):
            if stop_event.is_set():
                break

            env = problem_factory()
            env.config.seed = rng_seed + seed_idx
            env.reset()

            kin          = KinematicsModel.from_config_extra(env.config.extra)
            initial_yard = copy.deepcopy(env.yard)
            containers   = list(env.containers)
            max_tiers    = env.config.max_tiers
            lb           = lower_bound_relocations(initial_yard)

            seed_rng = random.Random(rng_seed + seed_idx)

            best_plan_local:  Optional[RelocationPlan] = None
            best_eval_local:  float                    = float("inf")

            # ── GRASP restarts ───────────────────────────────────── #
            for _ in range(max_rst):
                if stop_event.is_set():
                    break

                constructed = constructive_phase(
                    initial_yard, containers, k, max_tiers, seed_rng
                )
                improved, eval_val = local_search(
                    constructed,
                    initial_yard,
                    containers,
                    max_tiers,
                    kin,
                    alpha,
                    beta,
                    max_inner,
                )
                if eval_val < best_eval_local:
                    best_eval_local = eval_val
                    best_plan_local = improved

            if best_plan_local is None:
                continue

            # ── Seed-level metrics ───────────────────────────────── #
            n_relocs   = best_plan_local.num_relocations()
            crane_time = compute_crane_time(best_plan_local, kin)
            metrics = {
                "relocations": float(n_relocs),
                "crane_time":  float(crane_time),
                "total_moves": float(best_plan_local.num_moves()),
                "lower_bound": float(lb),
                "lb_ratio":    float(n_relocs / max(lb, 1)),
                "eval":        float(best_eval_local),
                "time":        float(crane_time),
                "steps":       float(best_plan_local.num_moves()),
            }
            all_metrics.append(metrics)

            primary = float(crane_time)
            if primary < self._best_metric:
                self._best_metric   = primary
                self._best_plan     = best_plan_local
                self._best_solution = _plan_to_action_list(best_plan_local, env)

            self._push(
                result_queue,
                step     = seed_idx + 1,
                metric   = primary,
                metrics  = metrics,
                progress = (seed_idx + 1) / n_seeds,
                snapshot = env.get_state_snapshot(),
                extra    = {
                    "seed":          seed_idx,
                    "k":             k,
                    "alpha":         alpha,
                    "beta":          beta,
                    "max_restarts":  max_rst,
                    "inner_iters":   max_inner,
                },
            )

        if all_metrics:
            agg = {
                kname: float(np.mean([m[kname] for m in all_metrics if kname in m]))
                for kname in all_metrics[0]
            }
            self._push(
                result_queue,
                step     = n_seeds,
                metric   = self._best_metric,
                metrics  = agg,
                progress = 1.0,
            )

    # ---------------------------------------------------------------- #
    # Public accessors                                                   #
    # ---------------------------------------------------------------- #

    def get_best_solution(self) -> Optional[List[int]]:
        return self._best_solution

    def get_best_plan(self) -> Optional[RelocationPlan]:
        return self._best_plan

    # ---------------------------------------------------------------- #
    # Config schema                                                      #
    # ---------------------------------------------------------------- #

    @classmethod
    def config_schema(cls) -> Dict:
        base = super().config_schema()
        base.update({
            "num_eval_seeds": {
                "type": "int", "default": 5, "min": 1, "max": 100,
                "label": "Evaluation seeds",
                "help": "Number of random initial layouts to evaluate over.",
            },
            "k": {
                "type": "int", "default": 3, "min": 1, "max": 20,
                "label": "RCL distinct values (k)",
                "help": (
                    "Top-k distinct myopic values kept in the Restricted "
                    "Candidate List during construction (paper: k=3)."
                ),
            },
            "alpha": {
                "type": "float", "default": 0.3, "min": 0.0, "max": 100.0,
                "label": "α (weight on moves)",
                "help": "Weight on the number of moves in the evaluation "
                        "function (paper ParamILS tune: α=0.3).",
            },
            "beta": {
                "type": "float", "default": 0.005, "min": 0.0, "max": 1.0,
                "label": "β (weight on crane_time)",
                "help": "Weight on total crane working time (paper "
                        "ParamILS tune: β=0.005).",
            },
            "max_restarts": {
                "type": "int", "default": 5, "min": 1, "max": 50,
                "label": "GRASP restarts",
                "help": "Number of GRASP restarts per seed (paper: 10).",
            },
            "max_inner_iters": {
                "type": "int", "default": 50, "min": 1, "max": 1000,
                "label": "Local-search iters",
                "help": "Max hill-climb iterations per restart "
                        "(paper: 200).",
            },
        })
        return base


# ================================================================ #
#  Helpers                                                            #
# ================================================================ #

def _plan_to_action_list(plan: RelocationPlan, env) -> List[int]:
    """Flatten the plan's relocations into BRPFixed.step() action indices.

    Retrieval moves are skipped (auto-retrieved inside the environment).
    """
    num_rows = env.config.num_rows
    actions  = []
    for m in plan.movements:
        if m.to_pos is None:
            continue
        bay, row = m.to_pos
        actions.append((bay - 1) * num_rows + (row - 1))
    return actions
