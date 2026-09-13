"""
CasertaLAH
<2009> <heuristic> <restricted> <single-bay> <CRP-R>
Randomised look-ahead over Min–Max with roulette-wheel restarts
n_restarts --- 200 --- Number of trajectory restarts

------------------------------- Reference --------------------------------
M. Caserta, S. Schwarze, S. Voß,
"A New Binary Description of the Blocks Relocation Problem and Benefits
 in a Look Ahead Heuristic",
Evolutionary Computation in Combinatorial Optimization (EvoCOP 2009),
LNCS 5482, pp. 37–48, Springer, 2009.
------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import multiprocessing as mp
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from core.base_algorithm import BaseAlgorithm, AlgorithmConfig
from core.layout_trace import trace_layout
from core.move_export import export_yard_moves
from .scoring import greedy_trajectory, run_trajectory


class CasertaLAH(BaseAlgorithm):

    name                = "Caserta et al. (2009) LAH"
    category            = "Heuristic"
    description         = "Caserta et al. (EvoCOP 2009) look-ahead heuristic."
    compatible_problems = ["CRP-R"]
    geometry            = "single-bay"
    objectives          = ["relocations"]
    fidelity            = "faithful"
    step_label          = "Restart"

    def __init__(self, config: Optional[AlgorithmConfig] = None):
        super().__init__(config)

    # ---------------------------------------------------------------- #
    # Training loop                                                      #
    # ---------------------------------------------------------------- #

    def train(
        self,
        problem_factory: Callable,
        result_queue:    mp.Queue,
        stop_event:      mp.Event,
    ) -> None:
        if stop_event.is_set():
            return

        cfg          = self.config
        n_restarts   = int(cfg.extra.get("n_restarts", 200))
        report_every = max(1, n_restarts // 20)

        rng = np.random.RandomState(cfg.seed)
        np.random.seed(cfg.seed)

        trace_layout(f"CasertaLAH.train: n_restarts={n_restarts}")

        env = problem_factory()
        env.reset(options={"skip_auto_retrieve": True})

        n_total   = int(env.config.num_containers)
        max_tiers = int(env.config.max_tiers)
        n_stacks  = env.config.num_bays * env.config.num_rows

        all_keys:    List[Any] = []
        stacks_init: Dict[Any, List[int]] = {}

        for a in range(n_stacks):
            key = env._action_to_stack(a)
            all_keys.append(key)
            stk = env.yard.stacks.get(key)
            stacks_init[key] = (
                [int(c.priority) for c in stk.containers] if stk else []
            )

        best_cost, best_dsts = greedy_trajectory(
            stacks_init, n_total, max_tiers, all_keys
        )

        for restart in range(n_restarts):
            if stop_event.is_set():
                break

            cost, dsts = run_trajectory(
                stacks_init, n_total, max_tiers, all_keys, rng, best_cost
            )
            if cost < best_cost:
                best_cost = cost
                best_dsts = dsts

            if (restart + 1) % report_every == 0:
                self._push(
                    result_queue,
                    step     = restart + 1,
                    metric   = float(best_cost),
                    metrics  = {
                        "relocations": float(best_cost),
                        "restart":     float(restart + 1),
                    },
                    progress = (restart + 1) / n_restarts,
                )

        action_by_key = {key: action for action, key in enumerate(all_keys)}
        solution = [action_by_key[dst] for dst in best_dsts]
        metrics = env.validate_actions(solution)
        metrics["search_relocations"] = float(best_cost)
        metrics["progress"] = 1.0
        self._best_solution = solution[:]

        self._push(
            result_queue,
            step     = n_restarts,
            metric   = float(metrics["relocations"]),
            metrics  = metrics,
            progress = 1.0,
            extra={
                "solution": solution[:],
                "moves": export_yard_moves(env.yard),
                "validation_errors": env.get_last_validation_errors(),
            },
        )

    def get_best_solution(self) -> Optional[List[int]]:
        return self._best_solution

    # ---------------------------------------------------------------- #
    # Configuration schema                                               #
    # ---------------------------------------------------------------- #

    @classmethod
    def config_schema(cls) -> Dict:
        base = super().config_schema()
        base.update({
            "n_restarts": {
                "type":    "int",
                "default": 200,
                "min":     1,
                "max":     5000,
                "label":   "Restarts",
                "help": (
                    "Number of trajectory restarts (outer loop). "
                    "Each restart builds one complete relocation sequence via "
                    "randomised look-ahead; the best is retained. "
                    "More restarts → better solutions, longer runtime."
                ),
            },
        })
        return base
