"""
Ünlüyurt & Aydın (2012) — Difference1 heuristic for CRP-R / CRP-Time.

Reference
---------
T. Ünlüyurt, C. Aydın,
"Improved rehandling strategies for the container retrieval process",
Journal of Advanced Transportation 46 (2012) 378–393.
https://doi.org/10.1002/atr.1193

Algorithm — Difference1 (§4.2)
-------------------------------
When container X must be relocated, choose the destination stack by three
priority rules applied in order:

  Rule 1 — Ideal stack (top_priority > X):
    The relocating container will be retrieved before the current top of
    the destination stack, so no new blocking is created.
    Among all ideal stacks, select the one whose top_priority is closest
    to X from above (minimise top_priority − X).

  Rule 2 — Reverse-order buffer (top_priority < X):
    No ideal stack is available; a future rehandle is unavoidable.
    Stack containers in reverse order so they are easiest to re-sort later.
    Among these stacks, select the one whose top_priority is closest to X
    from below (minimise X − top_priority).

  Rule 3 — Fallback:
    Neither Rule 1 nor Rule 2 applies. Minimise |top_priority − X|
    unconditionally. Empty stacks are treated as top_priority = N + 1.

Paper context (§2, §4.2)
-------------------------
The paper focuses on a **single bay** with two objectives:
  Objective1: min number of relocations  (= CRP-R objective)
  Objective2: min A × pickups + B × horizontal_distance

Difference1 was designed for Objective1.  The three Rules do not depend
on any time formula; they solely use retrieval priority order.
Primary metric here is ``relocations``.  On CRP-Time environments the
environment additionally computes ``crane_time`` via Lee & Lee (2010)
kinematics.

Bay support
-----------
[single-bay origin]  The paper assumes a single bay.
On multi-bay instances, Rules 1–3 consider all valid stacks across bays
without penalising cross-bay gantry travel. The relocation count is still
correct, but crane_time on multi-bay yards is only approximated.
Use num_bays=1 for faithful replication of the paper's intended behaviour.

Numerical results (Table VI, paper)
------------------------------------
Objective1 (min relocations): avg optimality gap 1.83 %, finds optimal
for 72.55 % of instances — best among all three proposed heuristics.
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
    Difference1 destination selector (§4.2).
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
    """Execute Difference1 on a single CRP-R / CRP-Time episode."""
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

        target_pri   = env._current_target_priority
        src_key:     Optional[Any] = None
        blocker_pri: Optional[int] = None

        for key, prios in stacks.items():
            if target_pri in prios:
                src_key     = key
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

class UnluyurtAydinDifference(BaseAlgorithm):

    name     = "Ünlüyurt–Aydın (2012) Difference1"
    category = "Heuristic"
    description = (
        "[single-bay origin]  "
        "Ünlüyurt & Aydın (J. Adv. Transp. 2012) Difference1 heuristic (§4.2). "
        "Relocates blockers using three priority-order rules to minimise future "
        "rehandles: (1) ideal stack — top_priority > X, pick closest from above; "
        "(2) reverse-buffer — top_priority < X, pick closest from below; "
        "(3) fallback — minimise |top_priority − X|. "
        "Primary metric: relocations (Objective1 in the paper). "
        "Best-performing heuristic for Objective1: avg gap 1.83 %, "
        "finds optimal for 72.55 % of instances (Table VI). "
        "On multi-bay instances, bay distance is not penalised by the rules; "
        "use num_bays=1 for single-bay faithful comparison."
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
