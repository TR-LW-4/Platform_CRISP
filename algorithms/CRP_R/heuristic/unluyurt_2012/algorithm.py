"""
Ünlüyurt & Aydın (2012) — Difference heuristic for CRP-R.

Reference
---------
T. Ünlüyurt, C. Aydın,
"Improved rehandling strategies for the container retrieval process",
Journal of Advanced Transportation 46 (2012) 378–393.
https://doi.org/10.1002/atr.1193

Algorithm — Difference1 (§4.2)
-------------------------------
At each step, the top blocker X of the target stack must be relocated.
The destination stack is chosen by the following three rules, applied in
order until one succeeds:

Rule 1 — "Ideal" destination (avoids creating a new recognised rehandle):
    Candidate stacks where `top_priority(s) > priority(X)`.
    An "ideal" stack absorbs X without creating a blocking; when X is later
    retrieved, it will already be sitting on top of a container that will be
    retrieved after it.
    Among all ideal candidates, select the one with `top_priority(s)` closest
    to (but still above) `priority(X)` — i.e., minimise `top_priority(s) - X`.
    This minimises the number of containers that could potentially be placed
    between X and its new neighbour, reducing future re-blockings.

Rule 2 — "Reverse-order buffer" (no ideal stack found):
    Candidate stacks where `top_priority(s) < priority(X)`.
    Placing X on such a stack does create a blocking, but containers are
    stacked in reverse retrieval order, which may make them reordered later.
    Among these candidates, minimise `priority(X) - top_priority(s)`.

Rule 3 — Fallback (neither Rule 1 nor Rule 2 succeeds):
    No restriction on relative order; simply minimise
    `|top_priority(s) - priority(X)|`.
    Empty stacks: treated as having `top_priority = N + 1` (highest order).

For all rules: full stacks and the source stack are excluded.

Objective / compatible problems
--------------------------------
CRP-R (primary): minimises the number of relocations.
The algorithm is purely rule-based; it does not compute crane time.
Primary metric reported: ``relocations``.
"""

from __future__ import annotations

import multiprocessing as mp
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from core.base_algorithm import BaseAlgorithm, AlgorithmConfig


# ================================================================ #
#  Core decision logic                                               #
# ================================================================ #

def _top_priority(stk_prios: List[int], n_total: int) -> int:
    """
    Minimum priority currently in the stack = container retrieved first.
    Empty stack → N+1 (sentinel: will be retrieved last of all).
    """
    return min(stk_prios) if stk_prios else (n_total + 1)


def difference_select(
    c:         int,            # priority of the container being relocated
    src_key:   Any,            # key of the source stack (to exclude)
    stacks:    Dict[Any, List[int]],   # full yard (priorities, bottom→top)
    all_keys:  List[Any],
    n_total:   int,
    max_tiers: int,
) -> Optional[Any]:
    """
    Ünlüyurt & Aydın (2012) Difference1 destination selector.

    Returns the chosen destination stack key, or None if no valid stack
    exists (should not happen in a well-formed instance).
    """
    # Eligible destinations: not the source stack, not full
    eligible = [
        k for k in all_keys
        if k != src_key and len(stacks[k]) < max_tiers
    ]
    if not eligible:
        return None

    tp = {k: _top_priority(stacks[k], n_total) for k in eligible}

    # ── Rule 1: top_priority(s) > c → ideal stack ─────────────── #
    rule1 = [k for k in eligible if tp[k] > c]
    if rule1:
        return min(rule1, key=lambda k: tp[k] - c)

    # ── Rule 2: top_priority(s) < c → reverse-order buffer ──────── #
    rule2 = [k for k in eligible if tp[k] < c]
    if rule2:
        return min(rule2, key=lambda k: c - tp[k])

    # ── Rule 3: fallback — minimise absolute difference ──────────── #
    return min(eligible, key=lambda k: abs(tp[k] - c))


def _run_one_episode(env) -> Tuple[List[int], Dict]:
    """
    Execute Difference1 on a single environment episode.
    Returns (action_sequence, final_metrics).
    """
    n_total   = int(env.config.num_containers)
    max_tiers = int(env.config.max_tiers)
    n_stacks  = env.config.num_bays * env.config.num_rows

    all_keys: List[Any] = [env._action_to_stack(a) for a in range(n_stacks)]

    env.reset()
    solution: List[int] = []
    done = False

    while not done:
        # Build current yard snapshot (priorities, bottom-to-top per stack)
        stacks: Dict[Any, List[int]] = {}
        for key in all_keys:
            stk = env.yard.stacks.get(key)
            stacks[key] = (
                [int(c.priority) for c in stk.containers] if stk else []
            )

        # Identify the current target container and its blocker
        target_pri = env._current_target_priority
        src_key: Optional[Any] = None
        blocker_pri: Optional[int] = None

        for key, prios in stacks.items():
            if target_pri in prios:
                src_key = key
                # Top of stack
                blocker_pri = prios[-1] if prios[-1] != target_pri else None
                break

        if src_key is None or blocker_pri is None:
            # Target is accessible — env.step with any valid action triggers retrieval
            # Use action 0 as a dummy; env will auto-retrieve
            _, _, done, _, _ = env.step(0)
            solution.append(0)
            continue

        dst_key = difference_select(
            blocker_pri, src_key, stacks, all_keys, n_total, max_tiers
        )

        if dst_key is None:
            _, _, done, _, _ = env.step(0)
            solution.append(0)
            continue

        action = (dst_key[0] - 1) * env.config.num_rows + (dst_key[1] - 1)
        _, _, done, _, _ = env.step(action)
        solution.append(action)

    return solution, env.get_metrics()


# ================================================================ #
#  Algorithm class                                                   #
# ================================================================ #

class UnluyurtDifference(BaseAlgorithm):

    name     = "Ünlüyurt–Aydın (2012) Difference"
    category = "Heuristic"
    description = (
        "[single-bay origin]  "
        "Ünlüyurt & Aydın (J. Adv. Transp. 2012) Difference1 heuristic. "
        "When relocating container X, selects the destination stack by "
        "three ordered rules: "
        "(1) ideal stack — top_priority > X, pick closest; "
        "(2) reverse-buffer — top_priority < X, pick closest; "
        "(3) fallback — minimise |top_priority − X|. "
        "Optimises for minimum relocations (CRP-R primary metric). "
        "Achieves ~72.6 %% of instances at optimal, average gap 1.83 %% "
        "on the paper's benchmark (outperforms Kim–Hong EAR at 8.04 %% gap)."
    )
    compatible_problems = ["CRP-R"]
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
            env.config.seed = cfg.seed + seed

            solution, metrics = _run_one_episode(env)
            all_metrics.append(metrics)

            primary = float(metrics.get("relocations", 0.0))

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

    # ---------------------------------------------------------------- #
    # Config schema                                                      #
    # ---------------------------------------------------------------- #

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
