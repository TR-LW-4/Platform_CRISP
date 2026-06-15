"""
Ünlüyurt & Aydın (2012) — Difference heuristic for CRP-Time.

Reference
---------
T. Ünlüyurt, C. Aydın,
"Improved rehandling strategies for the container retrieval process",
Journal of Advanced Transportation 46 (2012) 378–393.
https://doi.org/10.1002/atr.1193

Algorithm — Difference1 (§4.2), CRP-Time packaging
----------------------------------------------------
The Difference heuristic selects the destination stack for each blocker
container using three priority rules based on relative retrieval order:

Rule 1 — Ideal stack (top_priority > X): no new blocking created.
    Select the ideal stack whose top_priority is closest to X from above.

Rule 2 — Reverse-order buffer (top_priority < X): blocking unavoidable,
    but containers are stacked in reverse order, easing future reordering.
    Select the stack whose top_priority is closest to X from below.

Rule 3 — Fallback: minimise |top_priority − X| unconditionally.
    Empty stacks: treated as top_priority = N + 1.

Paper context (§2)
------------------
The paper focuses on a **single bay** with the objective of minimising
`A × pickups + B × horizontal_distance` (Objective2 in the paper).
The Difference1 selection rules were designed for Objective1 (minimise
relocations) and are algorithm-level decisions that do NOT depend on the
time formula used to evaluate solutions.

CRP-Time adaptation
-------------------
The decision rules (Rules 1–3) are applied unchanged.
The **primary metric** reported here is `crane_time` (seconds), computed
by the CRP-Time environment via Lee & Lee (2010) kinematics
(`compute_crane_time`), which is the standard for this platform.
The algorithm does NOT explicitly optimise for crane_time; it minimises
relocations through the Difference rules, and the environment computes
the resulting crane_time automatically.

Bay support
-----------
[single-bay origin]  The paper assumes a single bay.
On multi-bay instances, Rules 1–3 consider all valid stacks across bays
without penalising cross-bay gantry travel. Crane-time is still correctly
calculated by the environment (including gantry costs), but the heuristic
decision ignores bay distance. Use num_bays=1 for faithful replication of
the paper's intended behaviour.
"""

from __future__ import annotations

import multiprocessing as mp
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from core.base_algorithm import BaseAlgorithm, AlgorithmConfig


# ================================================================ #
#  Core decision logic (self-contained, no import from CRP_R)        #
# ================================================================ #

def _top_priority(stk_prios: List[int], n_total: int) -> int:
    """Minimum priority in the stack (= retrieved first). Empty → N+1."""
    return min(stk_prios) if stk_prios else (n_total + 1)


def _difference_select(
    c:         int,
    src_key:   Any,
    stacks:    Dict[Any, List[int]],
    all_keys:  List[Any],
    n_total:   int,
    max_tiers: int,
) -> Optional[Any]:
    """
    Difference1 destination selector.
    Rules 1 → 2 → 3 applied in order.
    Returns destination stack key, or None if no valid stack exists.
    """
    eligible = [
        k for k in all_keys
        if k != src_key and len(stacks[k]) < max_tiers
    ]
    if not eligible:
        return None

    tp = {k: _top_priority(stacks[k], n_total) for k in eligible}

    rule1 = [k for k in eligible if tp[k] > c]
    if rule1:
        return min(rule1, key=lambda k: tp[k] - c)

    rule2 = [k for k in eligible if tp[k] < c]
    if rule2:
        return min(rule2, key=lambda k: c - tp[k])

    return min(eligible, key=lambda k: abs(tp[k] - c))


def _run_one_episode(env) -> Tuple[List[int], Dict]:
    """Execute Difference1 on a single CRP-Time episode."""
    n_total   = int(env.config.num_containers)
    max_tiers = int(env.config.max_tiers)
    n_stacks  = env.config.num_bays * env.config.num_rows

    all_keys: List[Any] = [env._action_to_stack(a) for a in range(n_stacks)]

    env.reset()
    solution: List[int] = []
    done = False

    while not done:
        stacks: Dict[Any, List[int]] = {}
        for key in all_keys:
            stk = env.yard.stacks.get(key)
            stacks[key] = (
                [int(c.priority) for c in stk.containers] if stk else []
            )

        target_pri  = env._current_target_priority
        src_key:    Optional[Any] = None
        blocker_pri: Optional[int] = None

        for key, prios in stacks.items():
            if target_pri in prios:
                src_key = key
                blocker_pri = prios[-1] if prios[-1] != target_pri else None
                break

        if src_key is None or blocker_pri is None:
            _, _, done, _, _ = env.step(0)
            solution.append(0)
            continue

        dst_key = _difference_select(
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

class UnluyurtDifferenceCRPTime(BaseAlgorithm):

    name     = "Ünlüyurt–Aydın (2012) Difference (CRP-Time)"
    category = "Heuristic"
    description = (
        "[single-bay origin]  "
        "Ünlüyurt & Aydın (J. Adv. Transp. 2012) Difference1 heuristic, "
        "packaged for CRP-Time. "
        "Selects the relocation destination by three ordered rules: "
        "(1) ideal stack — top_priority > X, pick closest; "
        "(2) reverse-buffer — top_priority < X, pick closest; "
        "(3) fallback — minimise |top_priority − X|. "
        "Decision logic minimises relocations; reported primary metric is "
        "crane_time (Lee & Lee 2010 kinematics) computed by the environment. "
        "On multi-bay instances the rules ignore bay distance; "
        "use num_bays=1 for faithful single-bay comparison."
    )
    compatible_problems = ["CRP-Time"]
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

            primary = float(
                metrics.get("crane_time", metrics.get("relocations", 0.0))
            )

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
