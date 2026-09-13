"""
ExpositoDSK
<2014> <heuristic> <unrestricted> <single-bay> <CRP-U>
Domain-specific knowledge scores for unrestricted relocation
alpha --- 1 --- Top-α randomization (1 = greedy)

------------------------------- Reference --------------------------------
C. Expósito-Izquierdo, B. Melián-Batista, J.M. Moreno-Vega,
"A domain-specific knowledge-based heuristic for the Blocks Relocation Problem",
Applied Soft Computing 14 (2014) 1–20.
------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import multiprocessing as mp
import random
from typing import Callable, Dict, List, Optional

import numpy as np

from core.base_algorithm import BaseAlgorithm, AlgorithmConfig
from .scoring import dsk_select_action


class ExpositoDSK(BaseAlgorithm):
    """
    DSK (Domain-Specific Knowledge) heuristic for CRP-U.

    Greedily places each blocker in the destination stack where it will be
    well-located (not blocking any earlier-priority container).  When multiple
    destinations tie, the one whose minimum-priority item is largest is
    preferred (least likely to cause future conflicts).

    With α > 1 and multiple evaluation seeds, the top-α randomisation allows
    the algorithm to explore diverse solution trajectories across restarts and
    report the best one.
    """

    name                = "Expósito-Izquierdo et al. (2014) DSK"
    category            = "Heuristic"
    description         = (
        "[single-bay origin]  "
        "Expósito-Izquierdo, Melián-Batista & Moreno-Vega (2014) domain-specific "
        "knowledge-based heuristic for CRP-U. "
        "Scores each destination by the number of containers that would be blocked "
        "by the relocated container (0 = well-located = ideal). "
        "Supports top-α randomisation for multi-restart diversity. "
        "DOI: 10.1016/j.asoc.2014.04.007"
    )
    compatible_problems = ["CRP-U"]
    geometry            = "single-bay"
    objectives          = ["relocations"]
    fidelity            = "faithful"
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
        n_seeds = 1  # multi-seed eval removed; single run only
        alpha   = int(cfg.extra.get("alpha", 1))
        base_seed = cfg.seed

        all_metrics: List[Dict] = []

        for seed_idx in range(n_seeds):
            if stop_event.is_set():
                break

            env = problem_factory()
            env.config.seed = base_seed + seed_idx
            env.reset()

            # Each seed gets its own RNG so runs are independent
            rng = random.Random(base_seed + seed_idx * 997)

            solution: List[int] = []
            done = False
            step_i = 0

            while not done:
                action = dsk_select_action(env, alpha=alpha, rng=rng)
                if action is None:
                    # No valid move — should not happen on a non-done env
                    break

                _, _, done, _, _ = env.step(action)

                step_i += 1
                src_i = action // env._num_stacks()
                dst_i = action  % env._num_stacks()
                src_key = env._action_to_stack(src_i)
                dst_key = env._action_to_stack(dst_i)
                m = env.get_metrics()
                print(
                    f"[DSK] seed={seed_idx}  step={step_i}"
                    f"  src={src_key}→dst={dst_key}"
                    f"  relocations={m['relocations']}  steps={m['steps']}",
                    flush=True,
                )
                solution.append(action)

            metrics = env.get_metrics()
            all_metrics.append(metrics)
            primary = float(metrics.get("relocations", 0.0))

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
                extra    = {"seed": seed_idx, "alpha": alpha},
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
    # Accessors                                                          #
    # ---------------------------------------------------------------- #

    def get_best_solution(self) -> Optional[List[int]]:
        return self._best_solution

    # ---------------------------------------------------------------- #
    # Config schema (GUI / CLI)                                          #
    # ---------------------------------------------------------------- #

    @classmethod
    def config_schema(cls) -> Dict:
        base = super().config_schema()
        base.update({
            "alpha": {
                "type":    "int",
                "default": 1,
                "min":     1,
                "max":     20,
                "label":   "Top-α candidates (α)",
                "help":    (
                    "Size of the candidate pool for random selection at each step. "
                    "α=1 → fully deterministic greedy (paper default). "
                    "α>1 → random pick from the α best moves, enabling diverse restarts."
                ),
            },
        })
        return base
