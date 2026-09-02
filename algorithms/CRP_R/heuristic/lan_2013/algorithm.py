"""
LANHeuristic
<2013> <heuristic> <restricted> <single-bay> <CRP-R>
Look-ahead N with optional cleaning moves for upcoming targets
N --- 1 --- Restricted look-ahead (cleaning moves disabled)

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

from core.base_algorithm import BaseAlgorithm, AlgorithmConfig
from core.plan import RelocationPlan
from .planner import build_lan_plan


class LANHeuristic(BaseAlgorithm):
    
    name                = "LA-N Look-Ahead"
    category            = "Heuristic"
    description         = "Petering & Hussein (EJOR 2013) restricted LA-1 heuristic."
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
        if stop_event.is_set():
            return

        cfg = self.config
        N = int(cfg.extra.get("N", 1))
        if N != 1:
            raise ValueError(
                "CRP-R is restricted: LA-N must use N=1. "
                "N>1 enables cleaning moves from non-target stacks."
            )

        env = problem_factory()
        env.reset(options={"skip_auto_retrieve": True})

        initial_yard = copy.deepcopy(env.yard)
        containers   = list(env.containers)
        max_tiers    = env.config.max_tiers
        env._finish_reset_after_layout_loaded()
        plan    = build_lan_plan(initial_yard, containers, N, max_tiers)
        metrics = env.validate_plan(plan)
        primary = float(metrics["relocations"])
        self._best_plan     = plan
        self._best_solution = _plan_to_action_list(plan, env)

        self._push(
            result_queue,
            step     = 1,
            metric   = primary,
            metrics  = metrics,
            progress = 1.0,
            extra    = {
                "N": N,
                "moves": [
                    {
                        "container_id": m.container_id,
                        "from": list(m.from_pos),
                        "to": list(m.to_pos) if m.to_pos is not None else None,
                        "kind": "retrieve" if m.to_pos is None else "relocate",
                    }
                    for m in plan.movements
                ],
                "validation_errors": env.get_last_validation_errors(),
            },
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
                "type": "int", "default": 1, "min": 1, "max": 1,
                "label": "Look-ahead N (restricted)",
                "help": (
                    "Fixed to N=1 for CRP-R. N>1 enables cleaning moves from "
                    "non-target stacks and therefore belongs to unrestricted BRP."
                ),
            },
        })
        return base


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
