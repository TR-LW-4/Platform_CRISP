"""
DeMeloSilva2018PMP
<2018> <exact> <premarshalling> <single-bay> <CRP-Prem>
Time-indexed PMPm1 MIP for pre-marshalling
lifo_strengthening --- True --- Section 3.2 LIFO cuts
per_solve_time_limit_s --- 30 --- Per-T Gurobi limit (s)

------------------------------- Reference --------------------------------
M. de Melo da Silva, S. Toulouse, R. Wolfler Calvo,
"A new effective unified model for solving the Pre-marshalling and
 Block Relocation Problems",
European Journal of Operational Research 271 (2018) 40–56.
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
from .solve import solve_pmp


def _encode_action(src_idx: int, dst_idx: int, n_stacks: int) -> int:
    dst_flat = dst_idx if dst_idx < src_idx else dst_idx - 1
    return src_idx * (n_stacks - 1) + dst_flat


class DeMeloSilva2018PMP(BaseAlgorithm):

    name = "de Melo da Silva et al. (2018) PMPm1 [Gurobi]"
    category = "Exact"
    description = (
        "Time-indexed MIP (PMPm1) for the Pre-marshalling Problem. "
        "de Melo da Silva, Toulouse & Wolfler Calvo -- EJOR 271 (2018). "
        "Uses the paper's own Section 3.3 greedy heuristic to size the time "
        "horizon and its Section 3.2 LIFO/flow strengthening constraints "
        "(default on), reported in the paper to dominate Lee & Hsu (2007) "
        "on LP-relaxation quality and instances solved to optimality. "
        "[Requires Gurobi license]"
    )
    compatible_problems = ["CRP-Prem"]
    requires_solver = True
    solver_backend = "gurobi"

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

        lifo_strengthening = bool(cfg.extra.get("lifo_strengthening", True))
        flow_strengthening = bool(cfg.extra.get("flow_strengthening", True))
        per_solve_time_limit_s = float(cfg.extra.get("per_solve_time_limit_s", 30.0))
        overall_time_budget_s = float(cfg.extra.get("overall_time_budget_s", 120.0))
        max_groups = int(cfg.extra.get("max_groups", 20))
        max_binaries = int(cfg.extra.get("max_binaries", 250_000))
        t_escalation_step = int(cfg.extra.get("t_escalation_step", 2))
        t_escalation_max_attempts = int(cfg.extra.get("t_escalation_max_attempts", 5))

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
                f"DeMeloSilva2018PMP seed={seed+1}/{n_seeds} "
                f"lifo_strengthening={lifo_strengthening} flow_strengthening={flow_strengthening}"
            )

            t0 = _time.perf_counter()
            result = solve_pmp(
                stacks_init,
                max_tiers=max_tiers,
                lifo_strengthening=lifo_strengthening,
                flow_strengthening=flow_strengthening,
                per_solve_time_limit_s=per_solve_time_limit_s,
                overall_time_budget_s=overall_time_budget_s,
                max_types=max_groups,
                max_binaries=max_binaries,
                t_escalation_step=t_escalation_step,
                t_escalation_max_attempts=t_escalation_max_attempts,
            )
            elapsed = _time.perf_counter() - t0

            moves_list = list(result.moves)
            solved = bool(result.solved)

            metrics = {
                "moves": float(len(moves_list)),
                "remaining_not_well_located": float(result.remaining_bad_overlaps),
                "time": float(elapsed),
                "solved": 1.0 if solved else 0.0,
                "proved_optimal": 1.0 if result.proved_optimal else 0.0,
                "heuristic_T": float(result.heuristic_T or 0),
                "T_used": float(result.T_used or 0),
                "n_groups": float(result.n_groups),
                "n_binaries": float(result.n_binaries),
                "attempts": float(result.attempts),
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
            "lifo_strengthening": {
                "type": "bool", "default": True,
                "label": "Section 3.2 LIFO strengthening (Eq. 15-18)",
                "help": "Off falls back to the plain no-floating-container rule (Eq. 4) instead.",
            },
            "flow_strengthening": {
                "type": "bool", "default": True,
                "label": "Section 3.2 flow strengthening (Eq. 20-22)",
                "help": "Tightens the LP relaxation by matching pick-ups/drop-offs across stacks.",
            },
            "per_solve_time_limit_s": {
                "type": "float", "default": 30.0, "min": 1.0, "max": 3600.0,
                "label": "Gurobi time limit per T attempt (s)",
            },
            "overall_time_budget_s": {
                "type": "float", "default": 120.0, "min": 1.0, "max": 86400.0,
                "label": "Overall wall-clock budget per seed (s)",
            },
            "max_groups": {
                "type": "int", "default": 20, "min": 1, "max": 200,
                "label": "Max container groups before bucket-coarsening",
                "help": "Instances with more distinct priorities are bucketed to keep the model tractable.",
            },
            "max_binaries": {
                "type": "int", "default": 250_000, "min": 1000, "max": 20_000_000,
                "label": "Binary-variable cap per T attempt",
                "help": "T candidates whose estimated model size exceeds this are skipped instead of hanging the solver.",
            },
            "t_escalation_step": {
                "type": "int", "default": 2, "min": 1, "max": 50,
                "label": "T escalation step (safety net)",
                "help": "The Section 3.3 heuristic already gives a valid T; this only matters if that bound is exceeded due to coarsening.",
            },
            "t_escalation_max_attempts": {
                "type": "int", "default": 5, "min": 1, "max": 20,
                "label": "Max T escalation attempts (safety net)",
            },
        })
        return base
