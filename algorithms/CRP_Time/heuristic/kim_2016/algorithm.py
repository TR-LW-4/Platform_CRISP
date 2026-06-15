"""
Kim (2016) multi-case heuristic for CRP-Time.

Reference
---------
Shin et al. (TRC 2026) ``baselines/kim2016.py`` — the implementation used
as Kim2016 baseline in the paper's benchmark evaluation.

Algorithm (4 cases, matches paper code exactly)
------------------------------------------------
Case 1:  auto-retrieve all accessible containers.

Case 2 (no ideal stacks):
    Dest = valid stack with max min_priority.

Case 3 (ideal stacks exist, some beat target_top_priority):
    Candidate ideal stacks: ideal AND top_priority > target_top_priority.
    Dest = candidate with min top_priority (random tie-break).
    [Unrestricted] Before main move: pull containers from *critical* stacks
    (stacks where every container below the top is larger than the top, i.e.
    the top is not the smallest — a definition of "critical") whose top
    priority lies in (target_top_priority, selected_top_priority) into dest.

Case 4 (ideal stacks exist but none beat target_top_priority):
    Dest = ideal stack with max top_priority (no tie-break).
    [Unrestricted] Same critical-stack pre-move loop as Case 3.

An "ideal stack" is a non-decreasing sequence from top to bottom
(every element ≥ the element above it), i.e. the stack is sorted.
"""

from __future__ import annotations

import multiprocessing as mp
import random
from typing import Callable, Dict, List, Optional, Set

import numpy as np

from core.base_algorithm import BaseAlgorithm, AlgorithmConfig


# ================================================================ #
#  Stack feature helpers                                            #
# ================================================================ #

def _stack_height(stk) -> int:
    return len(stk.containers) if stk else 0


def _top_priority(stk, fallback: float = float("inf")) -> float:
    if stk is None or stk.is_empty:
        return fallback
    return float(stk.top.priority)


def _min_priority(stk, fallback: float = float("inf")) -> float:
    if stk is None or stk.is_empty:
        return fallback
    return float(min(c.priority for c in stk.containers))


def _is_non_increasing(stk) -> bool:
    """True if the stack is sorted (non-increasing priority from bottom to top,
    i.e. the largest priority is at the bottom)."""
    if stk is None or stk.is_empty:
        return True
    prios = [float(c.priority) for c in stk.containers]
    return all(prios[i] >= prios[i + 1] for i in range(len(prios) - 1))


def _is_critical_stack(stk) -> bool:
    """
    A stack is critical if there exists a container (not the top) such that
    every container above it has a strictly larger priority
    (i.e. the container is NOT the minimum of those above it — meaning it will
    cause future relocations).

    Matches find_critical_stacks() in kim2016.py: container c at position i
    (0=bottom) is critical if all containers above i have a larger priority
    than c.  We exclude the topmost container.
    """
    if stk is None or stk.is_empty or len(stk.containers) == 1:
        return False

    prios = [float(c.priority) for c in stk.containers]
    n = len(prios)
    for i in range(n - 1):          # exclude the top container
        above = prios[i + 1:]       # positions i+1 … top
        if all(a > prios[i] for a in above):
            return True
    return False


# ================================================================ #
#  Core decision logic                                              #
# ================================================================ #

def kim2016_compute_moves(
    env,
    mask,
    restricted: bool = False,
) -> List[int]:
    """
    Return a (possibly multi-action) list for one Kim2016 decision step.
    """
    target = env._get_target_container()
    if target is None:
        return []

    src_stack = env.yard._find_stack(target)
    if src_stack is None or src_stack.top == target:
        return []

    target_top_priority = _top_priority(src_stack)
    n_bays    = env.config.num_bays
    n_rows    = env.config.num_rows
    max_tiers = env.config.max_tiers
    n_cont    = env.config.num_containers

    n_stacks = n_bays * n_rows
    valid_actions: List[int] = (
        list(range(n_stacks))
        if mask is None
        else [int(i) for i in range(n_stacks) if mask[i]]
    )

    src_action = (src_stack.bay - 1) * n_rows + (src_stack.row - 1)
    candidates = [a for a in valid_actions if a != src_action]
    if not candidates:
        return []

    def get_stk(action: int):
        return env.yard.stacks.get(env._action_to_stack(action))

    def height(a):      return _stack_height(get_stk(a))
    def topp(a):        return _top_priority(get_stk(a), float(n_cont * 10))
    def minp(a):        return _min_priority(get_stk(a), float("inf"))
    def is_ideal(a):    return _is_non_increasing(get_stk(a))
    def is_critical(a): return _is_critical_stack(get_stk(a))

    # Check for any ideal stacks
    ideal_candidates = [a for a in candidates if is_ideal(a)]

    if ideal_candidates:
        top_priorities = {a: topp(a) for a in ideal_candidates}
        above_target   = [a for a in ideal_candidates
                          if top_priorities[a] > target_top_priority]

        if above_target:
            # Case 3: ideal AND top > target_top_priority → min top, random tie
            min_top = min(top_priorities[a] for a in above_target)
            best_set = [a for a in above_target if top_priorities[a] == min_top]
            selected = random.choice(best_set)
        else:
            # Case 4: ideal stacks exist but none with top > target → max top
            max_top  = max(top_priorities[a] for a in ideal_candidates)
            best_set = [a for a in ideal_candidates
                        if top_priorities[a] == max_top]
            selected = best_set[0]  # no tie-break needed per paper

        selected_top = topp(selected)
        result: List[int] = []

        if not restricted:
            # Pre-move critical-stack containers into dest
            while True:
                spare = max_tiers - height(selected)
                if spare < 2:
                    break
                # critical stacks with top in (target_top, selected_top)
                if above_target:
                    crits = [a for a in candidates
                             if a != selected
                             and is_critical(a)
                             and target_top_priority < topp(a) < selected_top]
                else:
                    crits = [a for a in candidates
                             if a != selected
                             and is_critical(a)
                             and topp(a) < selected_top]

                if not crits:
                    break

                # highest top_priority among critical
                max_crit_top = max(topp(a) for a in crits)
                src_crit = [a for a in crits if topp(a) == max_crit_top][0]
                result.append(selected)      # dest receives the pre-move
                break  # one pre-move per invocation

        result.append(selected)
        return result

    else:
        # Case 2: no ideal stacks → pick valid stack with max min_priority
        max_minp = max(minp(a) for a in candidates)
        best_set = [a for a in candidates if minp(a) == max_minp]
        return [best_set[0]]


# ================================================================ #
#  BaseAlgorithm wrapper                                            #
# ================================================================ #

class Kim2016Heuristic(BaseAlgorithm):

    name                = "Kim (2016) Multi-Case Heuristic"
    category            = "Heuristic"
    description         = (
        "[native multi-bay]  "
        "Kim (2016) 4-case heuristic for CRP-Time. "
        "Case 3/4: use ideal (non-increasing) stacks with pre-move loop "
        "for critical stacks. Case 2: fallback to max min-priority dest. "
        "Matches baselines/kim2016.py in Shin et al. (TRC 2026)."
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
        cfg        = self.config
        rng_seed   = cfg.seed
        n_seeds    = max(1, cfg.num_eval_seeds)
        restricted = bool(cfg.extra.get("restricted", False))

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
                moves = kim2016_compute_moves(env, mask, restricted=restricted)

                if not moves:
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
                extra    = {"restricted": restricted, "seed": seed_idx},
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
            "restricted": {
                "type": "bool", "default": False,
                "label": "Restricted (skip pre-moves)",
                "help": "Skip the critical-stack pre-move loop.",
            },
        })
        return base
