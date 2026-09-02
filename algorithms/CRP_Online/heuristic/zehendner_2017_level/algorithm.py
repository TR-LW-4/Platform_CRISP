"""
LevelingHeuristic
<2017> <heuristic> <online> <single-bay> <CRP-Online>
Online leveling L with random R and right-neighbour M baselines
lookahead_h --- 0 --- Revealed future targets beyond the current one

------------------------------- Reference --------------------------------
E. Zehendner, D. Feillet, P. Jaillet,
"An algorithm with performance guarantee for the Online Container
 Relocation Problem",
European Journal of Operational Research 259 (2017) 48–62.
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

from core.base_algorithm import AlgorithmConfig, BaseAlgorithm
from core.layout_trace import trace_layout

from .policies import POLICIES, leveling_competitive_ratio


StackKey = Tuple[int, int]


def _stack_to_action(dst: StackKey, num_rows: int) -> int:
    return (int(dst[0]) - 1) * int(num_rows) + (int(dst[1]) - 1)


def _snapshot_priorities(env) -> Dict[StackKey, List[int]]:
    return {
        key: [int(c.priority) for c in stack.containers]
        for key, stack in env.yard.stacks.items()
    }


def _generate_actions(env, pick_fn, rng: np.random.RandomState) -> List[int]:
    """Drive one online trajectory and return the destination actions."""
    env.reset(options={"skip_auto_retrieve": True})
    env._finish_reset_after_layout_loaded()

    actions: List[int] = []
    while not env._done:
        target = env._get_target_container()
        if target is None:
            break
        src_stack = env.yard._find_stack(target)
        if src_stack is None:
            break
        if src_stack.top == target:
            env._advance_auto_retrievals()
            continue

        src = (src_stack.bay, src_stack.row)
        stacks = _snapshot_priorities(env)
        dst = pick_fn(
            stacks,
            max_tiers=int(env.config.max_tiers),
            src=src,
            blocker_priority=int(src_stack.top.priority),
            rng=rng,
        )
        if dst is None:
            break
        action = _stack_to_action(dst, int(env.config.num_rows))
        actions.append(action)
        env.step(action)
    return actions


def _run_one(env, pick_fn, rng: np.random.RandomState) -> Tuple[Dict[str, float], List[int], Dict]:
    actions = _generate_actions(env, pick_fn, rng=rng)
    snapshot = env.get_state_snapshot()

    metrics = env.validate_actions(actions)
    metrics["competitive_ratio_bound"] = float(
        leveling_competitive_ratio(
            int(env.config.num_containers),
            int(env.config.num_bays) * int(env.config.num_rows),
        )
    )
    metrics["lookahead_h"] = float(int(env.config.extra.get("lookahead_h", 0) or 0))
    metrics["metric"] = float(metrics.get("relocations", float("inf")))
    return metrics, actions, snapshot


class _ZehendnerBase(BaseAlgorithm):

    category = "Heuristic"
    compatible_problems = ["CRP-Online"]
    _rule_key: str = "L"

    def __init__(self, config: Optional[AlgorithmConfig] = None):
        super().__init__(config)

    def _pick_fn(self):
        return POLICIES[self._rule_key]

    def train(
        self,
        problem_factory: Callable,
        result_queue: mp.Queue,
        stop_event: mp.Event,
    ) -> None:
        pick_fn = self._pick_fn()

        if stop_event.is_set():
            return

        trace_layout(
            f"{self.__class__.__name__}.train: online rule={self._rule_key}"
        )
        env = problem_factory()
        rng = np.random.RandomState(int(self.config.seed))
        metrics, actions, snapshot = _run_one(env, pick_fn, rng=rng)
        primary = float(metrics.get("relocations", float("inf")))
        self._best_solution = list(actions)
        self._best_metric = primary

        self._push(
            result_queue,
            step=1,
            metric=primary,
            metrics=metrics,
            progress=1.0,
            snapshot=snapshot,
        )

    def get_best_solution(self) -> Optional[List[int]]:
        return self._best_solution

    @classmethod
    def config_schema(cls) -> Dict:
        return super().config_schema()


class LevelingHeuristic(_ZehendnerBase):

    name = "Zehendner et al. (2017) Leveling L"
    description = (
        "Online CRP leveling heuristic (Zehendner, Feillet & Jaillet, EJOR 2017): "
        "relocate blockers to the lowest-height stack (leftmost on ties). "
        "Simple, feasible-when-feasible, and enjoys a proven "
        "competitive ratio 2·ceil(N/W) - 1 against the offline optimum "
        "(paper Theorem 1)."
    )
    _rule_key = "L"


class RandomStackHeuristic(_ZehendnerBase):

    name = "Zehendner et al. (2017) Random R"
    description = (
        "Baseline for Zehendner et al. 2017: pick a uniformly random "
        "non-full, non-source destination for the current blocker. "
        "Used as an empirical lower reference in the paper's experiments."
    )
    _rule_key = "R"


class RightNeighbourHeuristic(_ZehendnerBase):

    name = "Zehendner et al. (2017) Right-Neighbour M"
    description = (
        "Baseline for Zehendner et al. 2017: relocate the current blocker "
        "to the next stack in natural (bay, row) order (with wrap-around). "
        "Emulates the paper's M heuristic on multi-row yards."
    )
    _rule_key = "M"
