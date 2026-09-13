"""
CasertaHeuristic
<2012> <heuristic> <restricted> <single-bay> <CRP-R>
Destination-stack scoring by min priority

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
from typing import Callable, List, Optional

from core.base_algorithm import BaseAlgorithm, AlgorithmConfig
from core.move_export import attach_crane_time, export_yard_moves
from .scoring import caserta_select_action


class CasertaHeuristic(BaseAlgorithm):

    name = "Caserta (2012) HEUR"
    category = "Heuristic"
    description = "Caserta et al. (EJOR 2012) min-priority stack-score heuristic."
    compatible_problems = ["CRP-R", "CRP-Time"]
    geometry = "single-bay"
    objectives = ["relocations"]
    fidelity = "faithful"

    def __init__(self, config: Optional[AlgorithmConfig] = None):
        super().__init__(config)

    def train(
        self,
        problem_factory: Callable,
        result_queue: mp.Queue,
        stop_event: mp.Event,
    ) -> None:
        if stop_event.is_set():
            return

        env = problem_factory()
        env.reset()

        solution: List[int] = []
        done = False
        while not done:
            if stop_event.is_set():
                break
            info = env._get_info()
            action = caserta_select_action(env, info.get("action_mask"))
            _, _, done, _, _ = env.step(action)
            solution.append(action)

        metrics = attach_crane_time(
            env.get_metrics(),
            env.yard,
            getattr(env.config, "extra", None) or {},
        )
        metrics["time"] = metrics["crane_time"]
        metrics["feasible"] = float(done)
        metrics["completed"] = float(done)
        metrics["validated"] = 1.0
        metrics["validation_conflicts"] = 0.0 if done else 1.0
        if not done:
            metrics["executed_relocations"] = float(
                metrics.get("relocations", 0.0)
            )
            metrics["relocations"] = float("inf")
        primary = float(
            metrics.get("objective_value", metrics.get("relocations", 0.0))
        )
        self._best_solution = solution[:]
        moves = export_yard_moves(env.yard)

        self._push(
            result_queue,
            step=1,
            metric=primary,
            metrics=metrics,
            progress=1.0,
            extra={
                "moves": moves,
                "solution": solution[:],
            },
        )

    def get_best_solution(self) -> Optional[List[int]]:
        return self._best_solution
