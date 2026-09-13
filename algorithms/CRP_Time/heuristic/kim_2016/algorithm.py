"""
Kim2016Heuristic
<2016> <heuristic> <time> <multi-bay> <CRP-Time>
Four-case destination rule for RMGC retrieval
restricted --- False --- Restrict relocation to the current blocker

------------------------------- Reference --------------------------------
Y. Kim, T. Kim, H.C. Lee,
"Heuristic algorithm for retrieving containers",
Computers & Industrial Engineering 101 (2016) 352–360.
------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import multiprocessing as mp
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
    restricted: bool = False,  # kept for API compatibility; no effect in restricted model
) -> List[int]:
    """
    Decide the destination for the *current forced relocation* (the top
    container above cmin) according to the 4 cases in Kim et al. (2016).

    Because the platform CRP-Time uses the restricted model (only the
    direct blocker above the current target may be relocated), we cannot
    literally execute the paper's pre-relocations of other critical stacks.
    We apply the paper's case logic to select the best destination for the
    relocation that *must* happen now.

    Returns a list containing at most one destination action index.
    The caller executes it and then re-invokes this function, allowing
    re-identification of the yard state (matching the paper's loop).
    """
    target = env._get_target_container()
    if target is None:
        return []

    src_stack = env.yard._find_stack(target)
    if src_stack is None or src_stack.top == target:
        # Case 1 in paper: cmin is already on top → nothing to relocate here.
        # The outer loop will handle retrieval via step(0).
        return []

    # tnccmin in paper = priority of the current top of cmin's stack
    target_top_priority = _top_priority(src_stack)

    n_bays = env.config.num_bays
    n_rows = env.config.num_rows
    n_cont = env.config.num_containers

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

    def topp(a): return _top_priority(get_stk(a), float(n_cont * 10))
    def minp(a): return _min_priority(get_stk(a), float("inf"))
    def is_ideal(a): return _is_non_increasing(get_stk(a))

    ideal_candidates = [a for a in candidates if is_ideal(a)]

    if ideal_candidates:
        top_priorities = {a: topp(a) for a in ideal_candidates}
        above_target = [a for a in ideal_candidates
                        if top_priorities[a] > target_top_priority]

        if above_target:
            # Case 3 (paper):
            # There exists at least one ideal stack whose TN > tnccmin.
            # Choose the one with the *smallest* such TN (min top priority).
            min_top = min(top_priorities[a] for a in above_target)
            best_set = [a for a in above_target if top_priorities[a] == min_top]
            # Deterministic tie-break (PlatEMO style): smallest action index
            selected = min(best_set)
        else:
            # Case 4 (paper):
            # Ideal stacks exist, but none with TN > tnccmin.
            # Choose the ideal stack with the *largest* TN.
            max_top = max(top_priorities[a] for a in ideal_candidates)
            best_set = [a for a in ideal_candidates if top_priorities[a] == max_top]
            selected = min(best_set)  # deterministic on ties

        # In the paper's unrestricted setting we would now:
        #   - (Case 3) move tops of critical stacks with tnccmin < TNC < TN_selected
        #     (in descending TNC order) into the selected stack,
        #   - then move the current blocker (top of cmin's stack) into it.
        # Because we are in the restricted model we can only relocate the
        # current forced blocker. We therefore return the destination chosen
        # according to the paper's rule for this case.
        return [selected]

    else:
        # Case 2 (paper):
        # No ideal stack at all. Move the current blocker to the (unideal)
        # stack that has the largest "minimum movement priority"
        # (i.e. the stack whose smallest priority is the largest).
        if not candidates:
            return []
        max_minp = max(minp(a) for a in candidates)
        best_set = [a for a in candidates if minp(a) == max_minp]
        selected = min(best_set)  # deterministic
        return [selected]


# ================================================================ #
#  BaseAlgorithm wrapper                                            #
# ================================================================ #

class Kim2016Heuristic(BaseAlgorithm):

    name                = "Kim (2016) Multi-Case Heuristic"
    category            = "Heuristic"
    description         = (
        "[fixed rule; selected objective is posterior-only] "
        "Kim, Kim & Lee (C&IE 2016) 4-case heuristic for CRP-Time. "
        "Case 1: retrieve when cmin on top. "
        "Case 2: no ideal stack → dest with max min_priority. "
        "Case 3: candidate ideal (top > tnccmin) exists → dest = min-TN candidate. "
        "Case 4: ideal exists but no candidate → dest = max-TN ideal. "
        "Deterministic tie-breaking. "
        "Adapted to the platform's restricted relocation model "
        "(only the direct blocker above the current target can be moved); "
        "the paper's literal pre-relocations of other critical stacks are not "
        "executable here but the destination-selection logic follows the paper."
    )
    compatible_problems = ["CRP-Time"]
    geometry            = "multi-bay"
    objectives          = ["fixed-rule"]
    fidelity            = "faithful"
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
        n_seeds = 1  # multi-seed eval removed; single run only
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
                metrics.get(
                    "objective_value",
                    metrics.get("crane_time", metrics.get("relocations", 0.0)),
                )
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
            "restricted": {
                "type": "bool", "default": False,
                "label": "Restricted (skip pre-moves)",
                "help": "Skip the critical-stack pre-move loop.",
            },
        })
        return base
