"""
Ji2015
<2015> <heuristic> <stowage> <multi-bay> <CRP-Stow>
Nearest-stack and optimization rehandling strategies

------------------------------- Reference --------------------------------
M. Ji, W. Guo, H. Zhu, Y. Yang,
"Optimization of loading sequence and rehandling strategy for multi-quay
 crane operations in container terminals",
Transportation Research Part E 80 (2015) 1–19.
------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import multiprocessing as mp
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from core.base_algorithm import BaseAlgorithm, AlgorithmConfig


# ================================================================ #
#  High phase – shared sequencing rule                             #
# ================================================================ #

def _select_high(env) -> int:
    """
    Fewest-blockers sequencing rule (H_L, Jovanović 2019 §5.2).

    Among stacks containing a RETRIEVABLE container (cdd==0):
    - Prefer the stack where the highest (topmost) retrievable container
      has the fewest blockers above it (0 = already on top).
    - Tie-break: smallest (bay, row).

    Returns flat action index.
    """
    n_stacks = env._n_stacks

    best_action   = -1
    best_blockers = float("inf")
    best_key      = (float("inf"), float("inf"))

    for i in range(n_stacks):
        key = env._idx_to_stack(i)
        stk = env.yard.stacks.get(key)
        if stk is None or stk.is_empty:
            continue
        for depth, c in enumerate(reversed(stk.containers)):
            if env._is_retrievable(c):
                if depth < best_blockers or (depth == best_blockers and key < best_key):
                    best_action   = i
                    best_blockers = depth
                    best_key      = key
                break

    return max(best_action, 0)


# ================================================================ #
#  Low phase – JiNearest                                           #
# ================================================================ #

def _select_low_nearest(env, src_key: Tuple[int, int]) -> int:
    """
    Nearest Stack Strategy (Ji et al. §3.1).

    Choose the non-full, non-source stack with minimum Manhattan distance
    (|Δbay| + |Δrow|) to the source stack.

    Tie-break 1: smallest height (most room).
    Tie-break 2: smallest (bay, row) index.
    """
    n_stacks = env._n_stacks
    src_bay, src_row = src_key

    best_action = -1
    best_dist   = float("inf")
    best_height = float("inf")
    best_key    = (float("inf"), float("inf"))

    for i in range(n_stacks):
        key = env._idx_to_stack(i)
        if key == src_key:
            continue
        stk = env.yard.stacks.get(key)
        if stk is None or stk.is_full:
            continue

        bay, row = key
        dist   = abs(bay - src_bay) + abs(row - src_row)
        height = stk.height

        if (dist < best_dist
                or (dist == best_dist and height < best_height)
                or (dist == best_dist and height == best_height and key < best_key)):
            best_action = i
            best_dist   = dist
            best_height = height
            best_key    = key

    return max(best_action, 0)


# ================================================================ #
#  Low phase – JiOptimization                                      #
# ================================================================ #

def _select_low_optimization(env, src_key: Tuple[int, int]) -> int:
    """
    Optimization Strategy (Ji et al. §3.3).

    A destination stack is "safe" if it contains NO container designated to the
    same vessel stack as the current target — placing the blocker there will not
    cause an immediate secondary rehandle during that vessel-stack's loading.

    Priority: safe stacks sorted by height ascending (lowest first), then (bay, row).
    Fallback:  if no safe stack, select the lowest (emptiest) stack overall.
    """
    n_stacks = env._n_stacks
    k        = env._target_container.group if env._target_container is not None else -1

    safe:     List[Tuple] = []   # (height, bay, row, action)
    fallback: List[Tuple] = []   # (height, bay, row, action)

    for i in range(n_stacks):
        key = env._idx_to_stack(i)
        if key == src_key:
            continue
        stk = env.yard.stacks.get(key)
        if stk is None or stk.is_full:
            continue

        bay, row = key
        height   = stk.height
        has_k    = (not stk.is_empty) and any(c.group == k for c in stk.containers)

        if has_k:
            fallback.append((height, bay, row, i))
        else:
            safe.append((height, bay, row, i))

    if safe:
        safe.sort()
        return safe[0][3]

    if fallback:
        fallback.sort()
        return fallback[0][3]

    return 0   # no valid destination (should not happen with valid mask)


# ================================================================ #
#  Episode runner                                                   #
# ================================================================ #

def _run_episode(env, selector_fn: Callable) -> Tuple[List[int], Dict]:
    """
    Run one complete episode.  Deadlock guard: stop if no valid action exists.
    """
    env.reset()
    solution: List[int] = []
    done = False
    cfg  = env.config
    max_steps = env._n_stacks * getattr(cfg, "max_tiers", 10) * 4 + 200

    while not done and len(solution) < max_steps:
        mask = env._build_action_mask() if hasattr(env, "_build_action_mask") else None
        if mask is not None and not mask.any():
            break

        action = selector_fn(env)
        _, _, terminated, truncated, _ = env.step(action)
        solution.append(action)
        done = terminated or truncated

    return solution, env.get_metrics()


# ================================================================ #
#  Base algorithm class                                            #
# ================================================================ #

class _JiBase(BaseAlgorithm):
    """Shared train() scaffold for Ji (2015) heuristics."""

    compatible_problems = ["CRP-Stow", "CRP-D"]
    def __init__(self, config: Optional[AlgorithmConfig] = None):
        super().__init__(config)

    def _select(self, env) -> int:
        raise NotImplementedError

    def train(
        self,
        problem_factory: Callable,
        result_queue:    mp.Queue,
        stop_event:      mp.Event,
    ) -> None:
        cfg     = self.config
        n_seeds = 1  # multi-seed eval removed; single run only
        all_metrics: List[Dict] = []

        for seed in range(n_seeds):
            if stop_event.is_set():
                break

            env = problem_factory()
            env.config.seed = cfg.seed + seed

            solution, metrics = _run_episode(env, self._select)
            all_metrics.append(metrics)

            primary = float(metrics.get("relocations", metrics.get("shifters", 0.0)))
            if primary < self._best_metric:
                self._best_metric   = primary
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


# ================================================================ #
#  JiNearest                                                       #
# ================================================================ #

class JiNearest(_JiBase):

    name        = "Ji (2015) Nearest Stack"
    category    = "Heuristic"
    description = (
        "Ji et al. (2015) Nearest Stack Strategy for CRP-Stow / CRP-D. "
        "High phase: pick the source stack with fewest blockers above a "
        "target-group container. "
        "Low phase: relocate the blocker to the spatially closest non-full "
        "stack (minimum Manhattan distance |Δbay| + |Δrow|). "
        "Ties broken by smallest height, then by (bay, row) index."
    )

    def _select(self, env) -> int:
        if env._mode == "high":
            return _select_high(env)
        return _select_low_nearest(env, env._idx_to_stack(env._source_stack_idx))


# ================================================================ #
#  JiOptimization                                                  #
# ================================================================ #

class JiOptimization(_JiBase):

    name        = "Ji (2015) Optimization Strategy"
    category    = "Heuristic"
    description = (
        "Ji et al. (2015) Optimization Strategy for CRP-Stow / CRP-D. "
        "High phase: same fewest-blockers rule as Nearest. "
        "Low phase: prefer stacks that do NOT contain the current target "
        "group k (safe stacks), sorted by height. "
        "Avoids creating secondary rehandles during group-k retrieval. "
        "Fallback: emptiest stack overall when no safe stack exists."
    )

    def _select(self, env) -> int:
        if env._mode == "high":
            return _select_high(env)
        return _select_low_optimization(env, env._idx_to_stack(env._source_stack_idx))
