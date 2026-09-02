"""
LeeLeeRetrievalHeuristic
<2010> <heuristic> <time> <multi-bay> <CRP-Time>
Three-phase retrieval heuristic for crane working time
max_no_improve_p2 --- 200 --- Phase-2 stagnation cutoff
max_no_improve_p3 --- 500 --- Phase-3 stagnation cutoff

------------------------------- Reference --------------------------------
Y. Lee, Y.-J. Lee,
"A heuristic for retrieving containers from a yard",
Computers & Operations Research 37 (2010) 1139–1147.
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
import random
from typing import Callable, Dict, List, Optional

import numpy as np

from core.base_algorithm import BaseAlgorithm, AlgorithmConfig
from core.plan import RelocationPlan
from core.objectives import (
    KinematicsModel,
    compute_crane_time,
    lower_bound_relocations,
)
from .phase1 import phase1_greedy
from .phase2 import phase2_reduce_moves
from .phase3 import phase3_reduce_time


class LeeLeeRetrievalHeuristic(BaseAlgorithm):
    """
    Three-phase retrieval heuristic (Lee & Lee, COR 2010).

    Builds a complete RelocationPlan; reported metrics include relocations
    and crane_time (Phase 3 shortens RMGC time without increasing move count).
    """

    name                = "Lee–Lee (2010) Retrieval"
    category            = "Heuristic"
    description         = (
        "[native multi-bay]  "
        "Three-phase heuristic for CRP-Time (Lee & Lee, COR 2010).  "
        "Phase 1: greedy feasible sequence (faithful to paper).  "
        "Phase 2: iterative path shortening via random single-container "
        "neighbourhood search (approximates the paper's BIP).  "
        "Phase 3: crane-time reduction via alternate waypoints "
        "(approximates the paper's MIP).  "
        "Note: Phase 2/3 do not call any solver; paper Table 1/3 "
        "numerics cannot be reproduced with this implementation."
    )
    compatible_problems = ["CRP-R", "CRP-Time"]
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
        cfg         = self.config
        rng_seed    = cfg.seed
        max_p2      = int(cfg.extra.get("max_no_improve_p2", 200))
        max_p3      = int(cfg.extra.get("max_no_improve_p3", 500))
        n_seeds = 1  # multi-seed eval removed; single run only
        total_steps = n_seeds * 3
        all_metrics: List[Dict] = []

        for seed_idx in range(n_seeds):
            if stop_event.is_set():
                break

            seed = rng_seed + seed_idx
            random.seed(seed)
            np.random.seed(seed)

            env = problem_factory()
            env.reset(options={"skip_auto_retrieve": True})

            kin          = KinematicsModel.from_config_extra(env.config.extra)
            initial_yard = copy.deepcopy(env.yard)
            containers   = list(env.containers)
            max_tiers    = env.config.max_tiers
            n_containers = env.config.num_containers
            env._finish_reset_after_layout_loaded()
            lb           = lower_bound_relocations(initial_yard)

            # ── Phase 1 ──────────────────────────────────────────── #
            plan = phase1_greedy(initial_yard, containers, n_containers)

            self._push(
                result_queue,
                step     = seed_idx * 3 + 1,
                metric   = float(plan.num_relocations()),
                metrics  = _plan_metrics(plan, kin, lb),
                progress = (seed_idx * 3 + 1) / total_steps,
                snapshot = env.get_state_snapshot(),
                extra    = {"phase": 1, "seed": seed_idx},
            )

            if stop_event.is_set():
                break

            # ── Phase 2 ──────────────────────────────────────────── #
            plan, _ = phase2_reduce_moves(
                plan, initial_yard, containers, max_tiers,
                max_no_improve=max_p2,
                stop_event=stop_event,
            )

            self._push(
                result_queue,
                step     = seed_idx * 3 + 2,
                metric   = float(plan.num_relocations()),
                metrics  = _plan_metrics(plan, kin, lb),
                progress = (seed_idx * 3 + 2) / total_steps,
                snapshot = env.get_state_snapshot(),
                extra    = {"phase": 2, "seed": seed_idx},
            )

            if stop_event.is_set():
                break

            # ── Phase 3 ──────────────────────────────────────────── #
            plan, _ = phase3_reduce_time(
                plan, initial_yard, max_tiers, kin,
                max_no_improve=max_p3,
                stop_event=stop_event,
            )

            metrics = _plan_metrics(plan, kin, lb)
            all_metrics.append(metrics)
            primary = float(plan.num_relocations())

            if primary <= self._best_metric:
                self._best_metric   = primary
                self._best_plan     = plan
                self._best_solution = _plan_to_action_list(plan, env)

            self._push(
                result_queue,
                step     = seed_idx * 3 + 3,
                metric   = primary,
                metrics  = metrics,
                progress = (seed_idx * 3 + 3) / total_steps,
                snapshot = env.get_state_snapshot(),
                extra    = {"phase": 3, "seed": seed_idx},
            )

        if all_metrics:
            agg = {
                k: float(np.mean([m[k] for m in all_metrics if k in m]))
                for k in all_metrics[0]
            }
            self._push(
                result_queue,
                step     = total_steps,
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
        """Return the best RelocationPlan (richer than the action-index list)."""
        return self._best_plan

    # ---------------------------------------------------------------- #
    # Config schema                                                      #
    # ---------------------------------------------------------------- #

    @classmethod
    def config_schema(cls) -> Dict:
        base = super().config_schema()
        base.update({
            "max_no_improve_p2": {
                "type": "int", "default": 200, "min": 10, "max": 2000,
                "label": "Phase 2 – max non-improving iters",
                "help": "Phase 2 stops after this many consecutive non-improving iterations.",
            },
            "max_no_improve_p3": {
                "type": "int", "default": 500, "min": 10, "max": 5000,
                "label": "Phase 3 – max non-improving iters",
                "help": "Phase 3 stops after this many consecutive non-improving iterations.",
            },
        })
        return base


# ================================================================ #
#  Private helpers (module-level, algorithm-specific)               #
# ================================================================ #

def _plan_metrics(
    plan:      RelocationPlan,
    kin:       KinematicsModel,
    lb_relocs: int,
) -> Dict:
    n_relocs   = plan.num_relocations()
    crane_time = compute_crane_time(plan, kin)
    return {
        "relocations": float(n_relocs),
        "crane_time":  crane_time,
        "total_moves": float(plan.num_moves()),
        "lower_bound": float(lb_relocs),
        "lb_ratio":    float(n_relocs / max(lb_relocs, 1)),
        "time":        crane_time,
        "steps":       float(plan.num_moves()),
    }


def _plan_to_action_list(plan: RelocationPlan, env) -> List[int]:
    """
    Convert a RelocationPlan to flat action indices for the environment ``step()``.
    Retrieval moves (to_pos=None) are skipped — the env applies auto-retrieval when the target is on top.
    """
    num_rows = env.config.num_rows
    actions  = []
    for m in plan.movements:
        if m.to_pos is None:
            continue
        bay, row = m.to_pos
        actions.append((bay - 1) * num_rows + (row - 1))
    return actions
