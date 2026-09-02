"""
Lin2015Heuristic
<2015> <heuristic> <time> <multi-bay> <CRP-Time>
SSI priority-rule heuristic for retrieval with relocation
P_r --- 30 --- Row-distance weight in SSI
P_b --- 300 --- Bay-distance weight in SSI
restricted --- False --- Disable unrestricted pre-moves

------------------------------- Reference --------------------------------
D.-Y. Lin, Y.-J. Lee, Y. Lee,
"The container retrieval problem with respect to relocation",
Transportation Research Part C 52 (2015) 132–143.
------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import multiprocessing as mp
from typing import Callable, Dict, List, Optional

import numpy as np

from core.base_algorithm import BaseAlgorithm, AlgorithmConfig
from .scoring import lin_compute_moves


class Lin2015Heuristic(BaseAlgorithm):

    name                = "Lin–Lee–Lee (2015) Rule"
    category            = "Heuristic"
    description         = (
        "[native multi-bay]  "
        "Lin, Lee & Lee (TRC 2015) SSI heuristic for CRP-Time. "
        "Rule 1: ideal stacks scored by SSI = min_prio + P_r·row + P_b·bay_dist. "
        "Rule 2 (unrestricted): pre-moves into dest before main relocation. "
        "Rule 3: no ideal stacks → max min_priority destination. "
        "Defaults P_r=30, P_b=300 follow Shin et al. (TRC 2026)."
    )
    compatible_problems = ["CRP-Time", "CRP-R"]
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
        cfg        = self.config
        rng_seed   = cfg.seed
        n_seeds = 1  # multi-seed eval removed; single run only
        P_r        = float(cfg.extra.get("P_r",        30.0))
        P_b        = float(cfg.extra.get("P_b",        300.0))
        restricted = bool (cfg.extra.get("restricted", False))

        all_metrics: List[Dict] = []

        for seed_idx in range(n_seeds):
            if stop_event.is_set():
                break

            env = problem_factory()
            env.config.seed = rng_seed + seed_idx
            _, info = env.reset()

            solution: List[int] = []
            done = False

            while not done:
                mask  = info.get("action_mask")
                moves = lin_compute_moves(env, mask, P_r=P_r, P_b=P_b,
                                          restricted=restricted)

                if not moves:
                    # Nothing to do (target already on top; env will
                    # auto-retrieve on next step with a dummy action 0)
                    _, _, done, _, info = env.step(0)
                    solution.append(0)
                    continue

                for action in moves:
                    _, _, done, _, info = env.step(action)
                    solution.append(int(action))
                    if done:
                        break

            metrics = env.get_metrics()
            all_metrics.append(metrics)

            primary = float(
                metrics.get("crane_time", metrics.get("relocations", 0.0))
            )

            if primary < self._best_metric:
                self._best_metric   = primary
                self._best_solution = solution[:]

            self._push(
                result_queue,
                step     = seed_idx + 1,
                metric   = primary,
                metrics  = metrics,
                progress = (seed_idx + 1) / n_seeds,
                snapshot = env.get_state_snapshot(),
                extra    = {"P_r": P_r, "P_b": P_b,
                            "restricted": restricted, "seed": seed_idx},
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

    # ---------------------------------------------------------------- #
    # Config schema                                                      #
    # ---------------------------------------------------------------- #

    @classmethod
    def config_schema(cls) -> Dict:
        base = super().config_schema()
        base.update({
            "P_r": {
                "type": "float", "default": 30.0, "min": 0.0, "max": 10_000.0,
                "label": "P_r (row-index weight)",
                "help": (
                    "SSI row weight: added as P_r * row_index(dest). "
                    "Default 30 follows Shin et al. (TRC 2026)."
                ),
            },
            "P_b": {
                "type": "float", "default": 300.0, "min": 0.0, "max": 100_000.0,
                "label": "P_b (bay-distance weight)",
                "help": (
                    "SSI bay weight: added as P_b * |bay(dest) - bay(target)|. "
                    "Default 300 follows Shin et al. (TRC 2026)."
                ),
            },
            "restricted": {
                "type": "bool", "default": False,
                "label": "Restricted (skip Rule 2 pre-moves)",
                "help": (
                    "If True, skip Rule-2 pre-moves and only execute the "
                    "direct blocker relocation (restricted retrieval sequence)."
                ),
            },
        })
        return base
