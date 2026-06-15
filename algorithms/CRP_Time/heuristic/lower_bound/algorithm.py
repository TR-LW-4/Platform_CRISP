"""
Working-Time Lower Bound for CRP-Time.

Reference
---------
Shin et al. (TRC 2026) ``baselines/lowerbound.py`` — ``get_wt_lb(x)``.

This is not a constructive heuristic: it computes a theoretical lower bound
on the total crane working time for a given layout and reports it.
It is useful for measuring the optimality gap of other algorithms.

Formula (matches lowerbound.py exactly)
----------------------------------------
LB = LB1 + LB2

LB1 – Sequential-retrieval lower bound:
    Assume containers are retrieved in priority order.  For each retrieval,
    add the travel cost from the crane's current position to the container's
    stack, plus the spreader pick-up time (t_pd).  After retrieval the crane
    is at row 0 of that bay.

LB2 – Disorder penalty:
    For each stack, count the number of containers that have a smaller
    priority (= earlier retrieval) than some container below them
    (disorder).  Each disorder costs at least 2 * t_row + t_pd
    (one relocation travel + spreader).

Constants (Lee-Lee defaults, overridden by env.config.extra):
    t_pd  = 30  s   (spreader pickup + place-down)
    t_acc = 40  s   (gantry bay-change acceleration)
    t_bay = 3.5 s/bay (gantry travel per bay)
    t_row = 1.2 s/row (trolley travel per row)
"""

from __future__ import annotations

import copy
import multiprocessing as mp
from typing import Callable, Dict, List, Optional

import numpy as np

from core.base_algorithm import BaseAlgorithm, AlgorithmConfig


# ================================================================ #
#  Core lower-bound computation (mirrors lowerbound.py)             #
# ================================================================ #

def _count_disorder(stacks_prio: List[List[float]]) -> int:
    """
    For each stack, count containers c such that there exists a container
    below c in the same stack with a smaller priority (= disorder).
    The flag is set per container (not per pair), matching lowerbound.py.
    """
    total = 0
    for prios in stacks_prio:
        n = len(prios)
        for i in range(n):           # i = tier index (0 = bottom)
            # Is prios[i] in disorder? i.e., exists j < i with prios[j] < prios[i]
            # lowerbound.py: container at position i has any "larger" value above it
            # (x_expanded > x_below means the value AT i is larger than something
            # below it -- but wait, let me re-read)
            # In lowerbound.py the tensor x is (rows, tiers) and index [i,j]:
            # x_expanded = x[:, :, idx i of tiers]   (a container)
            # x_below    = x[:, :, idx j of tiers]   (below)
            # compare = (x_expanded > x_below) & (j < i)  [mask means j < i]
            # disorder_flag[stack, tier_i] = any(compare[stack, :i, tier_i])
            # So: container at tier i is "disordered" if it has a LARGER value
            # than some container below it (j < i → j is a lower tier index).
            # In our convention containers list is [bottom ... top], so
            # index 0 is bottom. tier_i above tier_j means i > j.
            # "x_expanded > x_below" with j < i means prios[i] > prios[j]
            # (the container at i has higher priority number than something below).
            below = prios[:i]
            if any(prios[i] > b for b in below):
                total += 1
    return total


def compute_wt_lb(env) -> float:
    """
    Compute the working-time lower bound for the current yard layout.

    Parameters
    ----------
    env : CRP_Time or CRP_R environment (after reset())

    Returns
    -------
    Lower bound on total crane working time (seconds).
    """
    kin_extra = env.config.extra or {}
    t_pd  = float(kin_extra.get("spreader_s", 30.0))
    t_acc = float(kin_extra.get("t_acc",      40.0))
    t_bay = float(kin_extra.get("t_bay",      3.5))
    t_row = float(kin_extra.get("t_row",      1.2))
    n_rows = env.config.num_rows

    # Collect all containers with their stack positions, sorted by priority
    items: List[tuple] = []
    for key, stk in env.yard.stacks.items():
        for c in stk.containers:
            items.append((float(c.priority), key[0], key[1]))  # (prio, bay, row)

    items.sort(key=lambda x: x[0])  # ascending priority = retrieval order

    # LB1: sequential travel lower bound
    lb1       = 0.0
    curr_bay  = None
    curr_row  = None

    for prio, bay, row in items:
        next_bay = bay
        next_row = row

        if curr_bay is None:
            curr_bay = next_bay
        if curr_row is None:
            curr_row = next_row

        if curr_bay != next_bay:
            lb1 += t_acc + t_bay * abs(curr_bay - next_bay)
        lb1 += t_row * abs(curr_row - next_row)
        lb1 += t_row * next_row    # travel from row 0 to container row
        lb1 += t_pd

        curr_bay = next_bay
        curr_row = 0               # crane rests at row 0 after retrieval

    # LB2: disorder penalty
    stacks_prio: List[List[float]] = []
    for stk in env.yard.stacks.values():
        if not stk.is_empty:
            # containers list: bottom = index 0
            stacks_prio.append([float(c.priority) for c in stk.containers])

    disorder = _count_disorder(stacks_prio)
    lb2 = (2.0 * t_row + t_pd) * disorder

    return lb1 + lb2


# ================================================================ #
#  BaseAlgorithm wrapper                                            #
# ================================================================ #

class WorkingTimeLowerBound(BaseAlgorithm):

    name                = "Working-Time Lower Bound (Shin 2026)"
    category            = "Heuristic"
    description         = (
        "[native multi-bay]  "
        "Theoretical lower bound on total crane working time "
        "(Shin et al., TRC 2026, lowerbound.py).  "
        "Not a constructive algorithm: reports LB = LB1 + LB2 where "
        "LB1 is a sequential-retrieval travel bound and LB2 penalises "
        "stack disorder.  Use to measure the optimality gap of other methods."
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

        all_lbs: List[float] = []

        for seed_idx in range(n_seeds):
            if stop_event.is_set():
                break

            env = problem_factory()
            env.config.seed = rng_seed + seed_idx
            env.reset()

            lb = compute_wt_lb(env)
            all_lbs.append(lb)

            if lb < self._best_metric:
                self._best_metric = lb

            self._push(
                result_queue,
                step     = seed_idx + 1,
                metric   = lb,
                metrics  = {
                    "lower_bound": lb,
                    "crane_time":  lb,
                    "time":        lb,
                },
                progress = (seed_idx + 1) / n_seeds,
                snapshot = env.get_state_snapshot(),
                extra    = {"seed": seed_idx},
            )

        if all_lbs:
            mean_lb = float(np.mean(all_lbs))
            self._push(
                result_queue,
                step     = n_seeds,
                metric   = mean_lb,
                metrics  = {
                    "lower_bound": mean_lb,
                    "crane_time":  mean_lb,
                    "time":        mean_lb,
                },
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
                "help": "Number of random layouts to compute LB over.",
            },
        })
        return base
