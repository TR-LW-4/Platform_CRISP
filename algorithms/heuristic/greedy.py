"""
Greedy heuristic baseline.

Strategy
--------
At each step, pick the action from the valid mask that results in the
fewest immediate relocations (look-ahead depth = 1).

Useful as a baseline to compare RL and GA results against.
"""

from __future__ import annotations

import multiprocessing as mp
from typing import Callable, Dict, List, Optional

import numpy as np

from core.base_algorithm import BaseAlgorithm, AlgorithmConfig


class GreedyHeuristic(BaseAlgorithm):

    name        = "Greedy Heuristic"
    category    = "Heuristic"
    description = ("Greedy baseline: always picks the action with the lowest "
                   "immediate relocation cost (depth-1 look-ahead).")
    compatible_problems: List[str] = []

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
        best_sol:    Optional[List[int]] = None
        best_metric  = float("inf")

        for seed in range(n_seeds):
            if stop_event.is_set():
                break

            env = problem_factory()
            env.config.seed = seed
            obs, info = env.reset()

            solution: List[int] = []
            total_reward        = 0.0
            done                = False

            while not done:
                mask   = info.get("action_mask", None)
                action = self._greedy_action(env, obs, mask)
                obs, reward, done, _, info = env.step(action)
                solution.append(action)
                total_reward += reward

            metrics = env.get_metrics()
            all_metrics.append(metrics)

            primary = -total_reward
            if primary < best_metric:
                best_metric    = primary
                best_sol       = solution[:]
                self._best_solution = best_sol

            # ── Report ─────────────────────────────────────────── #
            self._push(
                result_queue,
                step     = seed + 1,
                metric   = primary,
                metrics  = metrics,
                progress = (seed + 1) / n_seeds,
                snapshot = env.get_state_snapshot(),
            )

        # Summary
        if all_metrics:
            agg = {
                k: float(np.mean([m[k] for m in all_metrics if k in m]))
                for k in all_metrics[0]
            }
            self._push(
                result_queue,
                step     = n_seeds,
                metric   = best_metric,
                metrics  = agg,
                progress = 1.0,
            )

    # ---------------------------------------------------------------- #
    # Greedy action selection                                            #
    # ---------------------------------------------------------------- #

    def _greedy_action(
        self,
        env,
        obs:  np.ndarray,
        mask: Optional[np.ndarray],
    ) -> int:
        """
        Choose the valid action that minimises immediate reward penalty
        (i.e. maximises immediate reward).

        If the environment supports look-ahead via clone, try each valid
        action on a copy; otherwise fall back to the first valid action.
        """
        import copy

        n = env.action_space.n
        valid = list(range(n)) if mask is None else list(np.where(mask)[0])

        if not valid:
            return 0

        best_action = valid[0]
        best_reward = -float("inf")

        for a in valid:
            try:
                env_copy = copy.deepcopy(env)
                _, r, _, _, _ = env_copy.step(a)
                if r > best_reward:
                    best_reward = r
                    best_action = a
            except Exception:
                return valid[0]

        return best_action

    def get_best_solution(self) -> Optional[List[int]]:
        return self._best_solution

    step_label = "Seed"

    @classmethod
    def config_schema(cls) -> Dict:
        return {
            "num_eval_seeds": {"type": "int", "default": 10, "min": 1, "max": 100,
                               "label": "Evaluation seeds (num_eval_seeds)",
                               "help": "Number of random initial states to evaluate."},
            "seed":           {"type": "int", "default": 0,  "min": 0, "max": 9999,
                               "label": "Random seed"},
        }
