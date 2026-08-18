"""
LANHeuristic
<2013> <heuristic> <restricted> <single-bay> <CRP-R>
Look-ahead N with optional cleaning moves for upcoming targets
N --- 2 --- Look-ahead depth for cleaning moves

------------------------------- Reference --------------------------------
M.E.H. Petering, M.I. Hussein,
"A new mixed integer program and extended look-ahead heuristic algorithm
 for the block relocation problem",
European Journal of Operational Research 231 (2013) 120–130.
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
from typing import Callable, Dict, List, Optional

import numpy as np

from core.base_algorithm import BaseAlgorithm, AlgorithmConfig
from core.objectives import lower_bound_relocations
from core.plan import RelocationPlan
from .planner import build_lan_plan


class LANHeuristic(BaseAlgorithm):
    """
    Look-Ahead N (LA-N) heuristic for CRP-R.

    Petering & Hussein (2013) show LA-N consistently outperforms
    Kim–Hong (2006) and Lee–Lee (2010) on relocation count
    across small, medium, and large instances.
    """

    name                = "LA-N Look-Ahead"
    category            = "Heuristic"
    description         = (
        "[single-bay origin]  "
        "Look-Ahead N heuristic for CRP-R (Petering & Hussein, EJOR 2013).  "
        "N=1: basic LA (no cleaning moves).  "
        "N>1: proactive cleaning moves for the next N targets.  "
        "Generally outperforms Kim–Hong and Lee–Lee on relocations."
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
        cfg      = self.config
        N        = int(cfg.extra.get("N", 2))
        n_seeds = 1  # multi-seed eval removed; single run only
        rng_seed = cfg.seed
        all_metrics: List[Dict] = []

        for seed_idx in range(n_seeds):
            if stop_event.is_set():
                break

            env = problem_factory()
            env.config.seed = rng_seed + seed_idx
            env.reset(options={"skip_auto_retrieve": True})

            initial_yard = copy.deepcopy(env.yard)
            containers   = list(env.containers)
            max_tiers    = env.config.max_tiers
            n_containers = env.config.num_containers
            env._finish_reset_after_layout_loaded()
            lb           = lower_bound_relocations(initial_yard)

            # Build plan with LA-N
            plan = build_lan_plan(initial_yard, containers, N, max_tiers)

            metrics = _plan_metrics(plan, lb)
            all_metrics.append(metrics)
            primary = float(plan.num_relocations())

            if primary <= self._best_metric:
                self._best_metric   = primary
                self._best_plan     = plan
                self._best_solution = _plan_to_action_list(plan, env)

            self._push(
                result_queue,
                step     = seed_idx + 1,
                metric   = primary,
                metrics  = metrics,
                progress = (seed_idx + 1) / n_seeds,
                snapshot = env.get_state_snapshot(),
                extra    = {"N": N, "seed": seed_idx},
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

    # ---------------------------------------------------------------- #
    # Public accessors                                                   #
    # ---------------------------------------------------------------- #

    def get_best_solution(self) -> Optional[List[int]]:
        return self._best_solution

    def get_best_plan(self) -> Optional[RelocationPlan]:
        """Return the best RelocationPlan (richer than action-index list)."""
        return self._best_plan

    # ---------------------------------------------------------------- #
    # Config schema                                                      #
    # ---------------------------------------------------------------- #

    @classmethod
    def config_schema(cls) -> Dict:
        base = super().config_schema()
        base.update({
            "N": {
                "type": "int", "default": 2, "min": 1, "max": 20,
                "label": "Look-ahead N",
                "help": (
                    "Number of future targets considered for cleaning moves. "
                    "N=1: basic LA (no cleaning moves). "
                    "N=S-1: maximum look-ahead (best for small/medium instances). "
                    "For very large instances (>40×40), N=1 is often best."
                ),
            },
        })
        return base


# ================================================================ #
#  Private helpers                                                   #
# ================================================================ #

def _plan_metrics(plan: RelocationPlan, lb_relocs: int) -> Dict:
    n_relocs = plan.num_relocations()
    return {
        "relocations":  float(n_relocs),
        "retrievals":   float(plan.num_retrievals()),
        "total_moves":  float(plan.num_moves()),
        "lower_bound":  float(lb_relocs),
        "lb_ratio":     float(n_relocs / max(lb_relocs, 1)),
    }


def _plan_to_action_list(plan: RelocationPlan, env) -> List[int]:
    """
    Convert plan to flat destination-stack indices for CRP_R.step().
    Retrieval moves (to_pos=None) are skipped – CRP_R auto-retrieves.
    """
    num_rows = env.config.num_rows
    actions  = []
    for m in plan.movements:
        if m.to_pos is None:
            continue
        bay, row = m.to_pos
        actions.append((bay - 1) * num_rows + (row - 1))
    return actions
