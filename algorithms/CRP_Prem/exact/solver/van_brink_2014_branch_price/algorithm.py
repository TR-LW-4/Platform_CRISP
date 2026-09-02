"""
VanBrinkVanDerZwaan2014BP
<2014> <exact> <premarshalling> <single-bay> <CRP-Prem>
Branch-and-price restricted master for pre-marshalling
bp_mip_time_limit_s --- 8 --- Restricted-master MIP limit (s)
bp_global_time_limit_s --- 120 --- Overall time limit (s)

------------------------------- Reference --------------------------------
J. van Brink, R. van der Zwaan,
"A branch and price approach for the container pre-marshalling problem",
2014.
------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import multiprocessing as mp
import random
import time as _time
from typing import Callable, Dict, List, Optional

import numpy as np

from core.base_algorithm import AlgorithmConfig, BaseAlgorithm
from core.layout_trace import trace_layout
from .common import (
    StackColumn,
    Stacks,
    clone_stacks,
    greedy_candidate_moves,
    lower_bound_wrongly_plus_free,
    sequence_to_columns,
    total_bad_overlaps,
)
from .master_lp import solve_master_mip


class VanBrinkVanDerZwaan2014BP(BaseAlgorithm):
    name = "van Brink & van der Zwaan (2014) B&P"
    category = "Exact"
    description = (
        "[single-bay origin] Branch-and-price style exact algorithm for "
        "pre-marshalling (van Brink & van der Zwaan, 2014). "
        "Uses a Gurobi-backed restricted master model with iterative time horizon."
    )
    compatible_problems = ["CRP-Prem"]
    requires_solver = True
    solver_backend = "gurobi"

    def __init__(self, config: Optional[AlgorithmConfig] = None):
        super().__init__(config)

    def train(
        self,
        problem_factory: Callable,
        result_queue: mp.Queue,
        stop_event: mp.Event,
    ) -> None:
        cfg = self.config
        n_seeds = 1  # multi-seed eval removed; single run only
        all_metrics: List[Dict[str, float]] = []

        n_pool_sequences = int(cfg.extra.get("bp_pool_sequences", 120))
        per_seq_steps = int(cfg.extra.get("bp_seq_max_steps", 300))
        mip_time_limit = float(cfg.extra.get("bp_mip_time_limit_s", 8.0))
        global_time_limit = float(cfg.extra.get("bp_global_time_limit_s", 120.0))
        horizon_padding = int(cfg.extra.get("bp_horizon_padding", 80))

        for seed in range(n_seeds):
            if stop_event.is_set():
                break

            env = problem_factory()
            env.reset()

            stack_keys = list(env.yard.stacks.keys())
            key_to_idx = {k: i for i, k in enumerate(stack_keys)}
            max_tiers = int(env.config.max_tiers)
            n_containers = int(env.config.num_containers)
            priorities = list(range(1, n_containers + 1))

            stacks0: Stacks = {
                key_to_idx[k]: list(st.priority_snapshot())
                for k, st in env.yard.stacks.items()
            }

            lb = lower_bound_wrongly_plus_free(stacks0)
            ub = lb + horizon_padding
            t_start = _time.perf_counter()
            rng = random.Random(int(cfg.seed) + seed)

            trace_layout(
                f"vanBrink2014BP seed={seed + 1}/{n_seeds} LB={lb} UB={ub} pool={n_pool_sequences}"
            )

            columns_by_stack: Dict[int, List[StackColumn]] = {s: [] for s in stacks0}
            for _ in range(max(1, n_pool_sequences)):
                if stop_event.is_set():
                    break
                seq = greedy_candidate_moves(
                    stacks_init=stacks0,
                    max_tiers=max_tiers,
                    max_steps=per_seq_steps,
                    rng=rng,
                )
                cols, sorted_ok = sequence_to_columns(stacks0, seq, max_tiers)
                if not sorted_ok:
                    continue
                for s, c in cols.items():
                    columns_by_stack[s].append(c)

            for s in columns_by_stack:
                if not columns_by_stack[s]:
                    columns_by_stack[s].append(
                        StackColumn(stack_idx=s, adds={}, rems={}, cost=0, max_time_used=0)
                    )

            best_obj: Optional[float] = None
            best_horizon: Optional[int] = None
            solved = False

            for T in range(max(0, lb), max(0, ub) + 1):
                if stop_event.is_set():
                    break
                if (_time.perf_counter() - t_start) > global_time_limit:
                    break

                obj, _sol = solve_master_mip(
                    columns_by_stack=columns_by_stack,
                    priorities=priorities,
                    time_horizon=T,
                    time_limit_s=mip_time_limit,
                )
                if obj is None:
                    continue
                best_obj = obj
                best_horizon = T
                if obj <= float(T) + 1e-9:
                    solved = True
                    break

            elapsed = _time.perf_counter() - t_start
            if best_obj is None:
                best_obj = float("inf")
            if best_horizon is None:
                best_horizon = -1

            metrics = {
                "moves": float(best_obj if np.isfinite(best_obj) else 1e9),
                "bad_overlaps": float(total_bad_overlaps(clone_stacks(stacks0))),
                "time": float(elapsed),
                "lower_bound": float(lb),
                "best_horizon": float(best_horizon),
                "solved": 1.0 if solved else 0.0,
            }
            all_metrics.append(metrics)

            primary = metrics["moves"]
            if primary < self._best_metric:
                self._best_metric = primary
                self._best_solution = []

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

    @classmethod
    def config_schema(cls) -> Dict:
        base = super().config_schema()
        base.update({
            "bp_pool_sequences": {
                "type": "int", "default": 120, "min": 10, "max": 2000,
                "label": "Initial column-pool sequences",
            },
            "bp_seq_max_steps": {
                "type": "int", "default": 300, "min": 10, "max": 20000,
                "label": "Max steps per sampled sequence",
            },
            "bp_horizon_padding": {
                "type": "int", "default": 80, "min": 0, "max": 10000,
                "label": "Upper-bound horizon padding",
            },
            "bp_mip_time_limit_s": {
                "type": "float", "default": 8.0, "min": 0.1, "max": 600.0,
                "label": "Gurobi time per horizon (s)",
            },
            "bp_global_time_limit_s": {
                "type": "float", "default": 120.0, "min": 1.0, "max": 86400.0,
                "label": "Global time limit per seed (s)",
            },
        })
        return base

