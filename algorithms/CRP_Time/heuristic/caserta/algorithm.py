"""
Caserta et al. (2012) HEUR heuristic for CRP-R.

Algorithm (Section 4, Algorithm 1)
------------------------------------
For each target block n = 1 … N:
  While blocks R above n are non-empty:
    r  ← topmost block in R
    s* ← destination chosen by Eq. (11) scoring rule  (see scoring.py)
    move r → s*
  Retrieve block n

Objective : minimise total RELOCATIONS.
Interface : CRP_R.step()  (same as Kim–Hong, Greedy).
Compatible: CRP-R only (assumption A1 — relocate only blocks
            above the current target).

Reference
---------
M. Caserta, S. Schwarze, S. Voß,
"A mathematical formulation and complexity considerations for the
 blocks relocation problem",
European Journal of Operational Research 219 (2012) 96–104.
"""

from __future__ import annotations

import multiprocessing as mp
from typing import Callable, Dict, List, Optional

import numpy as np

from core.base_algorithm import BaseAlgorithm, AlgorithmConfig
from .scoring import caserta_select_action


class CasertaHeuristic(BaseAlgorithm):

    # Distinct from algorithms/CRP_R/heuristic/caserta (same paper; registry uses ``name`` as key).
    name                = "Caserta (2012) HEUR (CRP-Time)"
    category            = "Heuristic"
    description         = (
        "[single-bay origin]  "
        "Caserta et al. (EJOR 2012) stack-score heuristic for CRP-R. "
        "Relocates blockers to the stack with the smallest min-priority "
        "greater than the blocker (good fit), or the highest min-priority "
        "if no good stack exists (delay re-relocation). "
        "Average gap to optimum ≈1.9%; outperforms Kim–Hong."
    )
    compatible_problems = ["CRP-R", "CRP-Time"]
    step_label          = "Seed"

    def __init__(self, config: Optional[AlgorithmConfig] = None):
        super().__init__(config)

    # ---------------------------------------------------------------- #
    # Train                                                              #
    # ---------------------------------------------------------------- #

    def train(
        self,
        problem_factory: Callable,
        result_queue:    mp.Queue,
        stop_event:      mp.Event,
    ) -> None:
        cfg     = self.config
        n_seeds = max(1, cfg.num_eval_seeds)
        all_metrics: List[Dict] = []

        for seed in range(n_seeds):
            if stop_event.is_set():
                break

            env = problem_factory()
            env.config.seed = seed
            env.reset()

            solution: List[int] = []
            done = False

            while not done:
                info   = env._get_info()
                action = caserta_select_action(env, info.get("action_mask"))
                _, _, done, _, _ = env.step(action)
                solution.append(action)

            metrics = env.get_metrics()
            all_metrics.append(metrics)
            primary = float(metrics.get("relocations", 0.0))

            if primary < self._best_metric:
                self._best_solution = solution[:]

            self._push(
                result_queue,
                step     = seed + 1,
                metric   = primary,
                metrics  = metrics,
                progress = (seed + 1) / n_seeds,
                snapshot = env.get_state_snapshot(),
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

    @classmethod
    def config_schema(cls) -> Dict:
        base = super().config_schema()
        base.update({
            "num_eval_seeds": {
                "type": "int", "default": 10, "min": 1, "max": 100,
                "label": "Evaluation seeds",
                "help": "Number of random initial layouts to evaluate over.",
            },
        })
        return base
