"""
Leveling heuristic for CRP-Time.

Reference
---------
Shin et al. (TRC 2026) ``baselines/leveling.py`` — used as a baseline
in the benchmark evaluation.

Algorithm (matches leveling.py exactly)
----------------------------------------
At each step, the blocker on top of the target stack is moved to:
  1. Prefer the same bay as the target: among valid stacks in the same bay,
     pick the one with the minimum stack height (ties broken by travel time).
  2. If no valid stack in the same bay, look across all valid stacks:
     pick the one with the minimum stack height (ties broken by travel time).

Tie-breaking by travel time uses the gantry kinematics constants
(t_acc, t_bay, t_row) from env.config.extra (Lee-Lee defaults).
"""

from __future__ import annotations

import math
import multiprocessing as mp
import random
from typing import Callable, Dict, List, Optional

import numpy as np

from core.base_algorithm import BaseAlgorithm, AlgorithmConfig


# ================================================================ #
#  Travel-time tie-breaker (mirrors leveling.py choose_stack_by_travel_time)
# ================================================================ #

def _choose_by_travel_time(
    actions: List[int],
    curr_bay: int,
    curr_row: int,
    n_rows: int,
    t_acc: float,
    t_bay: float,
    t_row: float,
) -> int:
    best_idx    = None
    best_cost   = float("inf")
    best_tied: List[int] = []

    for a in actions:
        dst_bay  = a // n_rows + 1
        dst_row  = a % n_rows + 1
        bay_diff = abs(dst_bay - curr_bay)
        row_diff = abs(dst_row - curr_row)

        bay_cost = (t_acc + bay_diff * t_bay) if bay_diff > 0 else 0.0
        row_cost = row_diff * t_row
        cost     = bay_cost + row_cost

        if cost < best_cost:
            best_cost = cost
            best_tied = [a]
        elif cost == best_cost:
            best_tied.append(a)

    return random.choice(best_tied)


# ================================================================ #
#  Core decision logic                                              #
# ================================================================ #

def leveling_select_action(env, mask) -> int:
    """Return the flat destination action for the current blocker."""
    target = env._get_target_container()
    if target is None:
        return 0

    src_stack = env.yard._find_stack(target)
    if src_stack is None or src_stack.top == target:
        return 0

    n_rows    = env.config.num_rows
    n_bays    = env.config.num_bays
    max_tiers = env.config.max_tiers
    n_stacks  = n_bays * n_rows

    # Kinematics constants
    kin_extra = env.config.extra or {}
    t_acc = float(kin_extra.get("t_acc", 40.0))
    t_bay = float(kin_extra.get("t_bay", 3.5))
    t_row = float(kin_extra.get("t_row", 1.2))

    src_action = (src_stack.bay - 1) * n_rows + (src_stack.row - 1)
    valid_actions: List[int] = (
        list(range(n_stacks))
        if mask is None
        else [int(i) for i in range(n_stacks) if mask[i]]
    )

    # Exclude source stack
    candidates = [a for a in valid_actions if a != src_action]
    if not candidates:
        return 0

    def get_stk(action: int):
        return env.yard.stacks.get(env._action_to_stack(action))

    def height(a: int) -> int:
        stk = get_stk(a)
        return len(stk.containers) if stk else max_tiers

    def bay_of(a: int) -> int:
        return a // n_rows + 1

    target_bay = src_stack.bay
    curr_bay   = src_stack.bay
    curr_row   = src_stack.row

    # Same-bay valid stacks
    same_bay = [a for a in candidates if bay_of(a) == target_bay]

    if same_bay:
        pool = same_bay
    else:
        pool = candidates

    min_h = min(height(a) for a in pool)
    tied  = [a for a in pool if height(a) == min_h]

    if len(tied) > 1:
        return _choose_by_travel_time(
            tied, curr_bay, curr_row, n_rows, t_acc, t_bay, t_row
        )
    return tied[0]


# ================================================================ #
#  BaseAlgorithm wrapper                                            #
# ================================================================ #

class LevelingHeuristic(BaseAlgorithm):

    name                = "Leveling Heuristic"
    category            = "Heuristic"
    description         = (
        "[native multi-bay]  "
        "Leveling heuristic for CRP-Time: prefers same-bay destination "
        "with minimum stack height; ties broken by crane travel time. "
        "Matches baselines/leveling.py in Shin et al. (TRC 2026)."
    )
    compatible_problems = ["CRP-Time"]
    step_label          = "Seed"

    def __init__(self, config: Optional[AlgorithmConfig] = None):
        super().__init__(config)

    def train(
        self,
        problem_factory: Callable,
        result_queue:    mp.Queue,
        stop_event:      mp.Event,
    ) -> None:
        cfg      = self.config
        rng_seed = cfg.seed
        n_seeds  = max(1, cfg.num_eval_seeds)

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
                mask   = info.get("action_mask")
                action = leveling_select_action(env, mask)
                _, _, done, _, info = env.step(action)
                solution.append(int(action))

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
                extra    = {"seed": seed_idx},
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
            },
        })
        return base
