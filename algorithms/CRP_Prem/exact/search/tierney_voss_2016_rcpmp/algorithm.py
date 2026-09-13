"""
TierneyVoss2016RCPMP
<2016> <exact> <premarshalling> <single-bay> <CRP-Prem>
Robust pre-marshalling with blocking-matrix IDA*
robust_interval_window --- 2 --- Priority-class uncertainty window
ida_time_limit_s --- 60 --- IDA* time limit (s)

------------------------------- Reference --------------------------------
K. Tierney, S. Voß,
"Solving the Robust Container Pre-Marshalling Problem",
ICCL 2016, LNCS 9855, pp. 131–145.
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
from .core import (
    Stacks,
    build_blocking_matrix_from_intervals,
    build_intervals_from_priorities,
    clone_stacks,
    is_robust_sorted,
    relaxation_groups_from_blocking_matrix,
    solve_rcpmp_ida_star,
)


Move = Tuple[int, int]


def _encode_action(src_idx: int, dst_idx: int, n_stacks: int) -> int:
    dst_flat = dst_idx if dst_idx < src_idx else dst_idx - 1
    return src_idx * (n_stacks - 1) + dst_flat


def _robust_bad_pairs(stacks: Stacks, b: Dict[Tuple[int, int], int]) -> int:
    bad = 0
    for arr in stacks.values():
        for i in range(1, len(arr)):
            above = arr[i]
            for j in range(i):
                below = arr[j]
                if b.get((above, below), 0) == 1:
                    bad += 1
    return bad


def _apply_sequence(stacks_init: Stacks, seq: List[Move], max_tiers: int) -> Stacks:
    st = clone_stacks(stacks_init)
    for src, dst in seq:
        if src == dst:
            continue
        if src not in st or dst not in st:
            continue
        if not st[src] or len(st[dst]) >= max_tiers:
            continue
        x = st[src].pop()
        st[dst].append(x)
    return st


class TierneyVoss2016RCPMP(BaseAlgorithm):
    name = "Tierney–Voß (2016) RCPMP"
    category = "Exact"
    description = (
        "[single-bay origin] Robust pre-marshalling with blocking-matrix objective. "
        "Uses relaxation-guided IDA* search inspired by Tierney & Voss (2016)."
    )
    compatible_problems = ["CRP-Prem"]
    geometry = "single-bay"
    objectives = ["relocations"]
    fidelity = "faithful"
    # No external MILP/CP backend is required for this embedding.
    requires_solver = False
    solver_backend = None

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
        robust_window = int(cfg.extra.get("robust_interval_window", 2))
        time_limit_s = float(cfg.extra.get("ida_time_limit_s", 60.0))
        depth_padding = int(cfg.extra.get("ida_depth_padding", 120))
        all_metrics: List[Dict[str, float]] = []

        for seed in range(n_seeds):
            if stop_event.is_set():
                break

            env = problem_factory()
            env.reset()

            stack_keys = list(env.yard.stacks.keys())
            n_stacks = len(stack_keys)
            key_to_idx = {k: i for i, k in enumerate(stack_keys)}
            max_tiers = int(env.config.max_tiers)
            n_containers = int(env.config.num_containers)

            stacks0: Stacks = {
                key_to_idx[k]: list(st.priority_snapshot())
                for k, st in env.yard.stacks.items()
            }
            ids = list(range(1, n_containers + 1))
            intervals = build_intervals_from_priorities(ids, robust_window=robust_window)
            blocking = build_blocking_matrix_from_intervals(intervals)
            groups = relaxation_groups_from_blocking_matrix(ids, blocking)

            trace_layout(
                f"TierneyVoss2016RCPMP seed={seed+1}/{n_seeds} "
                f"window={robust_window} t_limit={time_limit_s}s"
            )

            res = solve_rcpmp_ida_star(
                stacks_init=stacks0,
                max_tiers=max_tiers,
                blocking_matrix=blocking,
                relax_groups=groups,
                time_limit_s=time_limit_s,
                depth_padding=depth_padding,
            )

            final_stacks = _apply_sequence(stacks0, res.moves, max_tiers=max_tiers)
            bad_pairs = _robust_bad_pairs(final_stacks, blocking)
            solved = bool(res.solved and is_robust_sorted(final_stacks, blocking))
            moves = float(len(res.moves))
            primary = moves if solved else moves + 1000.0 * (bad_pairs + 1.0)

            actions = [_encode_action(s, d, n_stacks) for (s, d) in res.moves]
            if primary < self._best_metric:
                self._best_metric = primary
                self._best_solution = actions

            metrics = {
                "moves": moves,
                "bad_overlaps": float(bad_pairs),
                "robust_bad_pairs": float(bad_pairs),
                "lower_bound": float(res.lower_bound),
                "expanded_nodes": float(res.expanded_nodes),
                "time": float(res.elapsed_s),
                "solved": 1.0 if solved else 0.0,
            }
            all_metrics.append(metrics)

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
            "robust_interval_window": {
                "type": "int", "default": 2, "min": 0, "max": 100,
                "label": "Robust interval half-window",
                "help": (
                    "Builds interval [p-w, p+w] around each priority p to "
                    "derive the blocking matrix used by the robust objective."
                ),
            },
            "ida_time_limit_s": {
                "type": "float", "default": 60.0, "min": 0.1, "max": 86400.0,
                "label": "IDA* time limit per seed (s)",
            },
            "ida_depth_padding": {
                "type": "int", "default": 120, "min": 0, "max": 10000,
                "label": "IDA* search depth padding",
                "help": (
                    "Maximum extra depth explored beyond the initial lower "
                    "bound threshold."
                ),
            },
        })
        return base

