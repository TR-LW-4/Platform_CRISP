"""
ParrenoTorresAlvarezValdesRuizTierney2020CPMPCT
<2020> <exact> <premarshalling> <single-bay> <CRP-Prem>
Exact crane-time branch-and-bound for CPMPCT
bb_time_limit_s --- 120 --- Branch-and-bound time limit (s)
backend --- auto --- Search backend

------------------------------- Reference --------------------------------
C. Parreño-Torres, R. Alvarez-Valdes, R. Ruiz, K. Tierney,
"Minimizing crane times in pre-marshalling problems",
Transportation Research Part E 137 (2020) 101917.
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
from .branch_and_bound import solve_cpmpct_branch_and_bound
from .core import Move, Stacks
from .crane_time import CraneParams
from .mip_model import solve_cpmpct_mip


def _encode_action(src_idx: int, dst_idx: int, n_stacks: int) -> int:
    dst_flat = dst_idx if dst_idx < src_idx else dst_idx - 1
    return src_idx * (n_stacks - 1) + dst_flat


class ParrenoTorresAlvarezValdesRuizTierney2020CPMPCT(BaseAlgorithm):
    name = "Parreno-Torres et al. (2020) CPMPCT"
    category = "Exact"
    description = (
        "[single-bay origin] Exact crane-time minimization for pre-marshalling "
        "(CPMPCT), with acceleration-aware crane kinematics and exact B&B search."
    )
    compatible_problems = ["CRP-Prem"]
    geometry = "single-bay"
    objectives = ["relocations"]
    fidelity = "faithful"
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
        backend = str(cfg.extra.get("backend", "auto")).strip().lower()
        bb_time_limit = float(cfg.extra.get("bb_time_limit_s", 120.0))
        bb_max_extra_depth = int(cfg.extra.get("bb_max_extra_depth", 120))

        params = CraneParams(
            vmax_v_loaded=float(cfg.extra.get("vmax_v_loaded", 0.5)),
            vmax_v_unloaded=float(cfg.extra.get("vmax_v_unloaded", 1.0)),
            vmax_r_loaded=float(cfg.extra.get("vmax_r_loaded", 1.16)),
            vmax_r_unloaded=float(cfg.extra.get("vmax_r_unloaded", 2.16)),
            d_v=float(cfg.extra.get("d_v", 2.65)),
            d_r=float(cfg.extra.get("d_r", 2.50)),
            container_h=float(cfg.extra.get("container_h", 2.591)),
            container_w=float(cfg.extra.get("container_w", 2.438)),
            margin_w=float(cfg.extra.get("margin_w", 0.300)),
            top_sep=float(cfg.extra.get("top_sep", 2.000)),
            start_hsep=float(cfg.extra.get("start_hsep", 1.000)),
        )

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

            stacks0: Stacks = {
                key_to_idx[k]: list(st.priority_snapshot())
                for k, st in env.yard.stacks.items()
            }
            trace_layout(
                f"Parreno2020CPMPCT seed={seed+1}/{n_seeds} "
                f"backend={backend} max_tiers={max_tiers}"
            )

            if backend in {"mip"}:
                res_m = solve_cpmpct_mip(
                    stacks_init=stacks0,
                    max_tiers=max_tiers,
                    p=params,
                    time_limit_s=bb_time_limit,
                )
                moves: List[Move] = list(res_m.moves)
                crane_time = float(res_m.crane_time)
                solved = bool(res_m.solved)
                elapsed = float(res_m.elapsed_s)
                expanded_nodes = 0.0
                mip_routed = 1.0
            else:
                # auto / bb
                res = solve_cpmpct_branch_and_bound(
                    stacks_init=stacks0,
                    max_tiers=max_tiers,
                    p=params,
                    time_limit_s=bb_time_limit,
                    max_extra_depth=bb_max_extra_depth,
                )
                moves = list(res.moves)
                crane_time = float(res.crane_time)
                solved = bool(res.solved)
                elapsed = float(res.elapsed_s)
                expanded_nodes = float(res.expanded_nodes)
                mip_routed = 0.0

            metrics = {
                "crane_time_seconds": float(crane_time),
                "moves": float(len(moves)),
                "bad_overlaps": 0.0 if solved else 1.0,
                "expanded_nodes": expanded_nodes,
                "time": elapsed,
                "solved": 1.0 if solved else 0.0,
                "mip_backend_routed_to_bb": mip_routed,
            }
            all_metrics.append(metrics)

            # Primary objective = crane time.
            primary = float(crane_time if solved else crane_time + 1e6)
            if primary < self._best_metric:
                self._best_metric = primary
                self._best_solution = [_encode_action(s, d, n_stacks) for (s, d) in moves]

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
            "backend": {
                "type": "str", "default": "auto",
                "label": "Backend (auto/bb/mip)",
            },
            "bb_time_limit_s": {
                "type": "float", "default": 120.0, "min": 0.1, "max": 86400.0,
                "label": "B&B time limit per seed (s)",
            },
            "bb_max_extra_depth": {
                "type": "int", "default": 120, "min": 0, "max": 5000,
                "label": "B&B max extra depth over move lower bound",
            },
            "vmax_v_loaded": {
                "type": "float", "default": 0.5, "min": 0.01, "max": 10.0,
                "label": "Vertical speed loaded (m/s)",
            },
            "vmax_v_unloaded": {
                "type": "float", "default": 1.0, "min": 0.01, "max": 10.0,
                "label": "Vertical speed unloaded (m/s)",
            },
            "vmax_r_loaded": {
                "type": "float", "default": 1.16, "min": 0.01, "max": 20.0,
                "label": "Horizontal speed loaded (m/s)",
            },
            "vmax_r_unloaded": {
                "type": "float", "default": 2.16, "min": 0.01, "max": 20.0,
                "label": "Horizontal speed unloaded (m/s)",
            },
            "d_v": {
                "type": "float", "default": 2.65, "min": 0.01, "max": 100.0,
                "label": "Vertical accel distance (m)",
            },
            "d_r": {
                "type": "float", "default": 2.50, "min": 0.01, "max": 100.0,
                "label": "Horizontal accel distance (m)",
            },
            "container_h": {
                "type": "float", "default": 2.591, "min": 0.1, "max": 10.0,
                "label": "Container height (m)",
            },
            "container_w": {
                "type": "float", "default": 2.438, "min": 0.1, "max": 10.0,
                "label": "Container width (m)",
            },
            "margin_w": {
                "type": "float", "default": 0.300, "min": 0.0, "max": 10.0,
                "label": "Inter-container gap (m)",
            },
            "top_sep": {
                "type": "float", "default": 2.000, "min": 0.0, "max": 20.0,
                "label": "Top lane to top tier separation (m)",
            },
            "start_hsep": {
                "type": "float", "default": 1.000, "min": 0.0, "max": 20.0,
                "label": "Initial point horizontal separation (m)",
            },
        })
        return base
