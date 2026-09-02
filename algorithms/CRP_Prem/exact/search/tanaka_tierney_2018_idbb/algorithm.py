"""
TanakaTierney2018IDBB
<2018> <exact> <premarshalling> <single-bay> <CRP-Prem>
Iterative-deepening branch-and-bound for pre-marshalling
time_limit_s --- 30 --- Per-instance wall-clock limit (s)

------------------------------- Reference --------------------------------
S. Tanaka, K. Tierney,
"Solving real-world sized container pre-marshalling problems with an
 iterative deepening branch-and-bound algorithm",
European Journal of Operational Research 264 (2018) 165–180.
------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import multiprocessing as mp
import time as _time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from core.base_algorithm import AlgorithmConfig, BaseAlgorithm
from core.layout_trace import trace_layout

from .core import Stacks
from .solve import solve_idbb


def _encode_action(src_idx: int, dst_idx: int, n_stacks: int) -> int:
    dst_flat = dst_idx if dst_idx < src_idx else dst_idx - 1
    return src_idx * (n_stacks - 1) + dst_flat


class TanakaTierney2018IDBB(BaseAlgorithm):

    name = "Tanaka & Tierney (2018) IDBB [vendored C]"
    category = "Exact"
    description = (
        "Iterative-deepening branch-and-bound for the Pre-marshalling "
        "Problem. Tanaka & Tierney -- EJOR 264 (2018). Vendors the "
        "authors' own public C implementation (Bortfeldt-Forster lower "
        "bound with IBF1/IBF2 tightenings, TYPE1 dominance pruning, greedy "
        "upper-bound seeding); no MIP solver required."
    )
    compatible_problems = ["CRP-Prem"]
    requires_solver = False

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
        time_limit_s = float(cfg.extra.get("time_limit_s", 30.0))

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

            trace_layout(
                f"TanakaTierney2018IDBB seed={seed+1}/{n_seeds} "
                f"time_limit_s={time_limit_s}"
            )

            t0 = _time.perf_counter()
            result = solve_idbb(stacks_init, max_tiers=max_tiers, time_limit_s=time_limit_s)
            elapsed = _time.perf_counter() - t0

            moves_list = list(result.moves)
            solved = bool(result.solved)

            metrics = {
                "moves": float(len(moves_list)),
                "remaining_not_well_located": float(result.remaining_bad_overlaps),
                "time": float(elapsed),
                "solved": 1.0 if solved else 0.0,
                "proved_optimal": 1.0 if result.proved_optimal else 0.0,
                "timed_out": 1.0 if result.timed_out else 0.0,
                "n_relocation_reported": float(result.n_relocation or 0),
            }
            all_metrics.append(metrics)

            primary = float(
                len(moves_list) if solved else len(moves_list) + 1000.0 * (result.remaining_bad_overlaps + 1)
            )
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
            "time_limit_s": {
                "type": "float", "default": 30.0, "min": 1.0, "max": 3600.0,
                "label": "Per-instance time limit (s)",
                "help": "Passed to the vendored solver's -t flag; on expiry it returns the best solution found so far (not necessarily optimal).",
            },
        })
        return base
