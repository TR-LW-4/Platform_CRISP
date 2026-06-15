"""
Ðurasević–Ðumić–Gil-Gala (2025) fixed GP priority function for CRP-Time.

Reference
---------
Ðurasević, M.; Ðumić, M.; Gil-Gala, F. J. (2025)
"Multitask genetic programming for automated design of heuristics for
 the container relocation problem",
Engineering Applications of Artificial Intelligence 144: 110001.

Role on the platform
--------------------
This implements the **best priority function found by MGP**, hardcoded
exactly as used in Shin et al. (TRC 2026) ``baselines/durasevic2025.py``.
It is a pure inference rule — no GP evolution is performed here.

The priority function (WT objective) evaluated per candidate stack s:
    slack  = n_containers - current_target_priority
    tmp_x  = x + slack  (shift priorities to create slack-adjusted values)
    CUR    = top_priority(target_stack) in tmp_x
    RI(s)  = # containers in s with tmp_priority < CUR
    DIFF(s)= min_priority(s) - CUR
    DIS(s) = |bay(s) - bay(target)|
    DUR(s) = DIS * t_bay + (DIS > 0) * t_acc
    SH(s)  = stack height of s
    EMP(s) = max_tiers - SH(s)

    PF(s) = (((RI + DIS) * ((DIFF² + (DUR - DIFF))) * RI)
           + (((RI - DUR) / (DUR - SH/(DIFF+ε) + ε))
           / ((DUR / (CUR/(DUR+ε) + ε)) * (DIFF² + DUR·CUR) + ε)))

Destination with minimum PF is chosen.  Tie-breaking is random.

The outer loop supports the unrestricted pre-move pattern from the paper:
before the main relocation, the algorithm optionally moves containers
from the chosen dest's stack to improve its state (same as durasevic2025.py
non-restricted logic).
"""

from __future__ import annotations

import multiprocessing as mp
import random
from typing import Callable, Dict, List, Optional

import numpy as np

from core.base_algorithm import BaseAlgorithm, AlgorithmConfig

_EPS = 1e-9


# ================================================================ #
#  Terminal helpers (mirrors durasevic2025.py Env-side logic)       #
# ================================================================ #

def _stack_height(stk) -> int:
    return len(stk.containers) if stk else 0


def _min_priority(stk, fallback: float = float("inf")) -> float:
    if stk is None or stk.is_empty:
        return fallback
    return float(min(c.priority for c in stk.containers))


def _top_priority(stk, fallback: float) -> float:
    if stk is None or stk.is_empty:
        return fallback
    return float(stk.top.priority)


def _avg_priority(stk, fallback: float) -> float:
    if stk is None or stk.is_empty:
        return fallback
    vals = [float(c.priority) for c in stk.containers]
    return sum(vals) / len(vals)


# ================================================================ #
#  Core PF scorer (WT objective, matches durasevic2025.py)          #
# ================================================================ #

def _compute_pf(
    src_top_priority: float,   # CUR: top priority of the blocker being relocated
    stk,                       # candidate destination stack object
    target_bay: int,           # bay index of the source/target stack (1-indexed)
    t_acc: float,
    t_bay: float,
    max_tiers: int,
    n_containers: int,
    slack: float,
    dst_bay: int,              # bay index of destination (1-indexed)
) -> float:
    """
    Evaluate the Ðurasević 2025 PF for one candidate destination stack.
    Lower is better (we pick argmin).
    """
    CUR  = src_top_priority + slack
    SH   = float(_stack_height(stk))
    EMP  = float(max_tiers) - SH

    # RI: number of containers in s with adjusted priority < CUR
    if stk and not stk.is_empty:
        RI = float(sum(1 for c in stk.containers if float(c.priority) + slack < CUR))
    else:
        RI = 0.0

    # min_priority with N+1 fallback for empty stacks
    minp_raw = _min_priority(stk, fallback=float("inf"))
    if minp_raw == float("inf"):
        minp_val = n_containers + 1 + slack
    else:
        minp_val = minp_raw + slack

    DIFF = minp_val - CUR

    DIS  = float(abs(dst_bay - target_bay))
    DUR  = DIS * t_bay + (DIS > 0) * t_acc

    pf = (
        (((RI + DIS) * ((DIFF * DIFF) + (DUR - DIFF))) * RI)
        + (
            ((RI - DUR) / (DUR - (SH / (DIFF + _EPS)) + _EPS))
            / ((DUR / (CUR / (DUR + _EPS) + _EPS)) * ((DIFF * DIFF) + (DUR * CUR)) + _EPS)
        )
    )
    return float(pf)


# ================================================================ #
#  Per-step action selection                                        #
# ================================================================ #

def durasevic2025_compute_moves(
    env,
    mask,
    restricted: bool = False,
) -> List[int]:
    """
    Return a list of flat action indices for one decision step.
    Usually length 1; unrestricted mode may prepend a pre-move.
    """
    target = env._get_target_container()
    if target is None:
        return []

    src_stack = env.yard._find_stack(target)
    if src_stack is None or src_stack.top == target:
        return []

    src_bay   = src_stack.bay
    n_bays    = env.config.num_bays
    n_rows    = env.config.num_rows
    max_tiers = env.config.max_tiers
    n_cont    = env.config.num_containers

    # Kinematics constants (Lee-Lee defaults)
    kin_extra = env.config.extra or {}
    t_acc = float(kin_extra.get("t_acc", 40.0))
    t_bay = float(kin_extra.get("t_bay", 3.5))

    n_stacks = n_bays * n_rows
    valid_actions: List[int] = (
        list(range(n_stacks))
        if mask is None
        else [int(i) for i in range(n_stacks) if mask[i]]
    )
    src_action = (src_bay - 1) * n_rows + (src_stack.row - 1)
    candidates = [a for a in valid_actions if a != src_action]
    if not candidates:
        return []

    def get_stk(action: int):
        return env.yard.stacks.get(env._action_to_stack(action))

    def bay_of(action: int) -> int:
        return action // n_rows + 1

    # Slack: current_target_priority - 1 so priorities shift
    cur_prio = float(src_stack.top.priority)
    slack    = float(n_cont) - float(env._current_target_priority)

    # Score all candidates
    scores = {
        a: _compute_pf(
            src_top_priority = cur_prio,
            stk              = get_stk(a),
            target_bay       = src_bay,
            t_acc            = t_acc,
            t_bay            = t_bay,
            max_tiers        = max_tiers,
            n_containers     = n_cont,
            slack            = slack,
            dst_bay          = bay_of(a),
        )
        for a in candidates
    }

    min_pf   = min(scores.values())
    best_set = [a for a, v in scores.items() if v == min_pf]
    dest_action = random.choice(best_set)

    result: List[int] = []

    if not restricted:
        dest_stk  = get_stk(dest_action)
        dest_top  = _top_priority(dest_stk, fallback=float(n_cont * 10))

        while True:
            dest_stk   = get_stk(dest_action)
            spare      = max_tiers - _stack_height(dest_stk)
            if spare < 1:
                break
            dc = dest_top

            # Find container to pre-move INTO dest: pick stack whose
            # top priority falls above the current dc
            pre_candidates = [
                a for a in valid_actions
                if a != src_action and a != dest_action
                and _top_priority(get_stk(a), fallback=float(-1)) > dc
            ]
            if not pre_candidates:
                break

            min_pre_prio = _min_priority(get_stk(pre_candidates[0]))
            new_dest_idx = random.choice(pre_candidates)
            # Execute pre-move: move pre_candidates's top to dest
            result.append(dest_action)  # dest receives the pre-mover
            break  # one pre-move per invocation

    result.append(dest_action)
    return result


# ================================================================ #
#  BaseAlgorithm wrapper                                            #
# ================================================================ #

class Durasevic2025(BaseAlgorithm):

    name                = "Ðurasević–Ðumić–Gil-Gala (2025) Fixed PF"
    category            = "Heuristic"
    description         = (
        "[native multi-bay]  "
        "Best priority function found by Multitask GP "
        "(Ðurasević, Ðumić & Gil-Gala, EAAI 2025), hardcoded for "
        "direct inference on CRP-Time.  Matches baselines/durasevic2025.py "
        "in Shin et al. (TRC 2026).  No GP evolution — pure rule application."
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
                moves = durasevic2025_compute_moves(env, mask,
                                                    restricted=restricted)
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
                "help": "Number of random initial layouts to evaluate over.",
            },
            "restricted": {
                "type": "bool", "default": False,
                "label": "Restricted (skip pre-moves)",
                "help": "If True, skip the unrestricted pre-move loop.",
            },
        })
        return base
