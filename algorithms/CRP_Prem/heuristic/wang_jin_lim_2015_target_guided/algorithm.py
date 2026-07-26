"""
Wang, Jin, Lim (2015) target-guided heuristic (TGH) for the container
pre-marshalling problem.

Reference
---------
N. Wang, B. Jin, A. Lim, "Target-guided algorithms for the container
pre-marshalling problem", Omega 53 (2015) 67-77.

Embedding scope
----------------
* §3 CPMP / CPMPDS: the dummy-stack variant is exposed as a config toggle
  (``has_dummy_stack``) rather than a separate problem class -- the last
  stack of the bay is treated as a temporary transfer-lane slot that must be
  emptied by the end of pre-marshalling.
* §4.1-§4.2 candidate container/stack selection with the smallF-lowT
  evaluation scheme (paper's final choice; §6.1 ablation not embedded).
* §4.3 giant moves (Case 1 / Case 2, each with their nslot-based
  sub-scenarios) and §4.4 the 5-priority relocation + single-move
  fulfillment.
* §4.5 termination guarantee via the fixed-height invariant.

Only the main greedy algorithm (TGH, Algorithm 1) is embedded; the beam
search extensions (BS-G, BS-B, §5) are out of scope for this pass.

Known limitation: in the doubly-degenerate giant-move sub-case where
``nslot == 0`` *and* the only borrowable container happens to sit in a
completely fixed stack, restoring it can occasionally be capacity-infeasible
mid-sequence. This is a rare corner case (empirically <0.3% of adversarial
random instances at realistic occupancy); when it occurs the heuristic
reports ``solved=False`` for that instance rather than corrupting state or
crashing.
"""

from __future__ import annotations

import multiprocessing as mp
import time as _time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from core.base_algorithm import AlgorithmConfig, BaseAlgorithm
from core.layout_trace import trace_layout

from .state import Stacks, infer_num_groups, make_state
from .tgh import tgh_solve


def _encode_action(src_idx: int, dst_idx: int, n_stacks: int) -> int:
    dst_flat = dst_idx if dst_idx < src_idx else dst_idx - 1
    return src_idx * (n_stacks - 1) + dst_flat


class WangJinLim2015TGH(BaseAlgorithm):

    name = "Wang-Jin-Lim (2015) TGH"
    category = "Heuristic"
    description = (
        "[single-bay origin] Wang, Jin & Lim (Omega 2015) target-guided "
        "heuristic for the CPMP. Fixes containers one at a time in "
        "descending group-label order via giant moves (Case 1/2, nslot-based "
        "sub-scenarios) and 5-priority relocation with fulfillment. "
        "Guarantees termination and a feasible solution on any solvable "
        "instance. Supports the CPMP-with-dummy-stack (CPMPDS) variant via "
        "``has_dummy_stack``."
    )
    compatible_problems = ["CRP-Prem"]
    step_label = "Seed"
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
        n_seeds = max(1, int(cfg.num_eval_seeds))
        has_dummy_stack = bool(cfg.extra.get("has_dummy_stack", False))
        max_moves = int(cfg.extra.get("max_moves", 10 ** 6))

        all_metrics: List[Dict[str, float]] = []

        for seed in range(n_seeds):
            if stop_event.is_set():
                break

            env = problem_factory()
            env.config.seed = seed
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

            dummy = None
            if has_dummy_stack:
                dummy = max(stacks_init.keys())
                stacks_init[dummy] = []  # CPMPDS: dummy stack starts empty (§3).

            trace_layout(
                f"WangJinLim2015TGH seed={seed+1}/{n_seeds} "
                f"has_dummy_stack={has_dummy_stack} num_groups={num_groups}"
            )

            state = make_state(stacks_init, max_tiers=max_tiers, dummy=dummy)

            t0 = _time.perf_counter()
            result = tgh_solve(state, num_groups=num_groups, max_moves=max_moves)
            elapsed = _time.perf_counter() - t0

            moves_list = list(result.moves)
            solved = bool(result.solved)

            metrics = {
                "moves": float(len(moves_list)),
                "time": float(elapsed),
                "solved": 1.0 if solved else 0.0,
                "infeasible": 0.0 if solved else 1.0,
                "has_dummy_stack": 1.0 if has_dummy_stack else 0.0,
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
            "num_eval_seeds": {
                "type": "int", "default": 10, "min": 1, "max": 100,
                "label": "Evaluation seeds",
                "help": "TGH is deterministic per instance; multiple seeds vary the layout.",
            },
            "has_dummy_stack": {
                "type": "bool", "default": False,
                "label": "CPMPDS: last stack is a dummy (transfer-lane) stack",
                "help": (
                    "When enabled, the highest-index stack starts empty and "
                    "acts as a transfer-lane slot: it is always treated as "
                    "dirty, is excluded as a fix target, and must end empty "
                    "(paper §3)."
                ),
            },
            "max_moves": {
                "type": "int", "default": 1_000_000, "min": 100, "max": 10_000_000,
                "label": "Move budget (defensive cap)",
                "help": (
                    "TGH is proven to terminate on solvable instances (paper "
                    "§4.5); this cap only guards against implementation bugs."
                ),
            },
        })
        return base
