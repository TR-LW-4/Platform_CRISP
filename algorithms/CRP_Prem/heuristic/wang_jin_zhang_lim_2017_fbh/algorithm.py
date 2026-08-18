"""
Wang, Jin, Zhang, Lim (2017) feasibility-based heuristic (FBH) for the
container pre-marshalling problem.

Reference
---------
N. Wang, B. Jin, Z. Zhang, A. Lim, "A feasibility-based heuristic for the
container pre-marshalling problem", European Journal of Operational
Research 256 (2017) 90-101.

Embedding scope
----------------
* §4 state representation, container/state stability, surplus vector
  ``Delta = R - D``, extreme/pre-extreme/dead-end state predicates.
* §5.2 the six-tuple lexicographic task-selection rule, including the
  tier-protection dead-end-avoidance mechanism (thresholds ``P1``, ``P2``
  exposed as config parameters).
* §5.3 STAP: the exact helper machinery (alpha, EvalMove, Interim /
  InterimFull, BiSender / BiReceiver) and the full immediate / internal
  (I1-I3) / external (E1-E4) case analysis for accomplishing one task.

Only the main FBH greedy loop (Algorithm 1) is embedded; the paper has no
evolutionary/beam-search extension, so there is nothing else to add here.

Documented simplification: for internal tasks (target and aim slot in the
same stack) in the tight-capacity regime (cases I2/I3), STAP's paper
pseudocode routes surplus containers through a second temp-stack hand-off
tied to an exact slot-count formula that could not be reconstructed
unambiguously from the printed algorithm boxes without risking silent
mis-tracking of the target container. We use a provably retrieval-safe
reconstruction instead (see ``stap.py`` module docstring for the full
rationale); it matches the paper's move-count formulas whenever they are
achievable and otherwise reports the task as infeasible gracefully rather
than mis-placing a container. External tasks (E1-E4) have no such
ambiguity (the drained aim stack itself is always available as overflow)
and are implemented exactly as printed.
"""

from __future__ import annotations

import multiprocessing as mp
import time as _time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from core.base_algorithm import AlgorithmConfig, BaseAlgorithm
from core.layout_trace import trace_layout

from .fbh import fbh_solve
from .state import Stacks, infer_num_groups, make_state


def _encode_action(src_idx: int, dst_idx: int, n_stacks: int) -> int:
    dst_flat = dst_idx if dst_idx < src_idx else dst_idx - 1
    return src_idx * (n_stacks - 1) + dst_flat


class WangJinZhangLim2017FBH(BaseAlgorithm):

    name = "Wang-Jin-Zhang-Lim (2017) FBH"
    category = "Heuristic"
    description = (
        "[single-bay origin] Wang, Jin, Zhang & Lim (EJOR 2017) "
        "feasibility-based heuristic for the CPMP. Fixes containers one at "
        "a time, always picking the task minimizing a six-tuple (tier "
        "protection, STAP move count, disturbed stable containers, "
        "affected demand, aim-stack fixed height, group value); every "
        "candidate task is pre-filtered through a necessary state-"
        "feasibility (surplus-vector) condition, and dead-end states are "
        "avoided proactively rather than discovered by backtracking."
    )
    compatible_problems = ["CRP-Prem"]
    requires_solver = False
    solver_backend = None

    def __init__(self, config: Optional[AlgorithmConfig] = None):
        super().__init__(config)

    # ---------------------------------------------------------------- #
    # Training loop                                                      #
    # ---------------------------------------------------------------- #

    def train(
        self,
        problem_factory: Callable,
        result_queue: mp.Queue,
        stop_event: mp.Event,
    ) -> None:
        cfg = self.config
        n_seeds = 1  # multi-seed eval removed; single run only
        p1 = int(cfg.extra.get("tier_protection_p1", 0))
        p2 = int(cfg.extra.get("tier_protection_p2", 1))

        all_metrics: List[Dict[str, float]] = []

        for seed in range(n_seeds):
            if stop_event.is_set():
                break

            env = problem_factory()
            env.reset()

            stack_keys = list(env.yard.stacks.keys())
            key_to_idx = {k: i for i, k in enumerate(stack_keys)}
            n_stacks = len(stack_keys)
            max_tiers = int(env.config.max_tiers)

            stacks_init: Stacks = {
                key_to_idx[k]: list(st.priority_snapshot())
                for k, st in env.yard.stacks.items()
            }
            num_groups = max(
                int(getattr(env.config, "num_groups", 0) or 0),
                infer_num_groups(stacks_init),
            )

            trace_layout(
                f"WangJinZhangLim2017FBH seed={seed+1}/{n_seeds} "
                f"P1={p1} P2={p2} num_groups={num_groups}"
            )

            state = make_state(stacks_init, max_tiers=max_tiers, num_groups=num_groups)

            t0 = _time.perf_counter()
            result = fbh_solve(state, p1=p1, p2=p2)
            elapsed = _time.perf_counter() - t0

            moves_list = list(result.moves)
            solved = bool(result.solved)

            metrics = {
                "moves": float(len(moves_list)),
                "time": float(elapsed),
                "solved": 1.0 if solved else 0.0,
                "infeasible": 0.0 if solved else 1.0,
            }
            all_metrics.append(metrics)

            primary = float(len(moves_list) if solved else len(moves_list) + 1000 * (len(moves_list) + 1))
            if primary < self._best_metric:
                self._best_metric = primary
                self._best_solution = [
                    _encode_action(s, d, n_stacks) for (s, d) in moves_list
                ]

            self._push(
                result_queue,
                step=seed + 1,
                metric=primary,
                metrics=metrics,
                progress=(seed + 1) / n_seeds,
                snapshot=env.get_state_snapshot(),
            )

        if all_metrics:
            agg = {
                k: float(np.mean([m[k] for m in all_metrics if k in m]))
                for k in all_metrics[0]
            }
            self._push(
                result_queue,
                step=n_seeds,
                metric=self._best_metric,
                metrics=agg,
                progress=1.0,
            )

    def get_best_solution(self) -> Optional[List[int]]:
        return self._best_solution

    # ---------------------------------------------------------------- #
    # Configuration schema                                               #
    # ---------------------------------------------------------------- #

    @classmethod
    def config_schema(cls) -> Dict[str, Dict[str, Any]]:
        base = super().config_schema()
        base.update({
            "tier_protection_p1": {
                "type": "int", "default": 0, "min": 0, "max": 20,
                "label": "Tier protection: max fixed height (P1)",
                "help": (
                    "A task aiming at a stack with f(s) <= P1 and affected "
                    "demand >= P2 is deprioritized (paper §5.2.2)."
                ),
            },
            "tier_protection_p2": {
                "type": "int", "default": 1, "min": 0, "max": 50,
                "label": "Tier protection: min affected demand (P2)",
                "help": "See tier_protection_p1.",
            },
        })
        return base
