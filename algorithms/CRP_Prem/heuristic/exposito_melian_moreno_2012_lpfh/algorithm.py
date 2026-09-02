"""
ExpositoMelianMoreno2012LPFH
<2012> <heuristic> <premarshalling> <single-bay> <CRP-Prem>
Lowest Priority First Heuristic with multistart
max_iterations --- 150 --- Multistart iterations
no_improve_limit --- 100 --- Stagnation cutoff

------------------------------- Reference --------------------------------
C. Expósito-Izquierdo, B. Melián-Batista, M. Moreno-Vega,
"Pre-Marshalling Problem: Heuristic solution method and instances generator",
Expert Systems with Applications 39 (2012) 8337–8349.
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
from .core import LPFHResult, Stacks, run_lpfh_multistart


def _encode_action(src_idx: int, dst_idx: int, n_stacks: int) -> int:
    dst_flat = dst_idx if dst_idx < src_idx else dst_idx - 1
    return src_idx * (n_stacks - 1) + dst_flat


class ExpositoMelianMoreno2012LPFH(BaseAlgorithm):
    name = "Expósito-Melián-Moreno (2012) LPFH"
    category = "Heuristic"
    description = (
        "[single-bay origin] Lowest Priority First Heuristic (LPFH) with multistart "
        "for the pre-marshalling problem."
    )
    compatible_problems = ["CRP-Prem"]
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
        k1 = max(1, int(cfg.extra.get("k1", 1)))
        k2_mode = str(cfg.extra.get("k2_mode", "quarter_s")).strip().lower()
        k3_mode = str(cfg.extra.get("k3_mode", "half_s")).strip().lower()
        max_iterations = max(1, int(cfg.extra.get("max_iterations", 150)))
        no_improve_limit = max(1, int(cfg.extra.get("no_improve_limit", 100)))
        move_budget_factor = float(cfg.extra.get("move_budget_factor", 50.0))
        use_stack_filling = bool(cfg.extra.get("use_stack_filling", True))

        all_metrics: List[Dict[str, float]] = []
        for seed in range(n_seeds):
            if stop_event.is_set():
                break
            env = problem_factory()
            env.reset()

            stack_keys = list(env.yard.stacks.keys())
            key_to_idx = {k: i for i, k in enumerate(stack_keys)}
            max_tiers = int(env.config.max_tiers)
            n_stacks = len(stack_keys)

            stacks0: Stacks = {
                key_to_idx[k]: list(st.priority_snapshot())
                for k, st in env.yard.stacks.items()
            }
            n_containers = sum(len(v) for v in stacks0.values())
            move_budget = max(1, int(move_budget_factor * max(1, n_containers)))

            if k2_mode == "one":
                k2 = 1
            elif k2_mode == "all_s":
                k2 = n_stacks
            elif k2_mode == "half_s":
                k2 = max(1, int(round(0.5 * n_stacks)))
            else:  # quarter_s
                k2 = max(1, int(round(0.25 * n_stacks)))

            if k3_mode == "one":
                k3 = 1
            elif k3_mode == "all_s":
                k3 = n_stacks
            elif k3_mode == "quarter_s":
                k3 = max(1, int(round(0.25 * n_stacks)))
            else:  # half_s
                k3 = max(1, int(round(0.5 * n_stacks)))

            trace_layout(
                f"Exposito2012LPFH seed={seed+1}/{n_seeds} k1={k1} k2={k2} k3={k3} "
                f"iter={max_iterations} noImp={no_improve_limit}"
            )

            t0 = _time.perf_counter()
            res: LPFHResult = run_lpfh_multistart(
                stacks_init=stacks0,
                max_tiers=max_tiers,
                k1=k1,
                k2=k2,
                k3=k3,
                max_iterations=max_iterations,
                no_improve_limit=no_improve_limit,
                use_stack_filling=use_stack_filling,
                rng_seed=int(cfg.seed) + seed * 10007 + random.randint(0, 999),
                move_budget=move_budget,
            )
            elapsed = _time.perf_counter() - t0
            solved = bool(res.solved)
            moves = float(len(res.moves))
            bad = float(res.remaining_bad)
            primary = moves if solved else moves + 1000.0 * (bad + 1.0)

            if primary < self._best_metric:
                self._best_metric = primary
                self._best_solution = [_encode_action(s, d, n_stacks) for (s, d) in res.moves]

            metrics = {
                "moves": moves,
                "bad_overlaps": bad,
                "time": float(elapsed),
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
            "k1": {
                "type": "int", "default": 1, "min": 1, "max": 100,
                "label": "Target-candidate count k1",
            },
            "k2_mode": {
                "type": "str", "default": "quarter_s",
                "label": "Destination-candidate mode k2",
            },
            "k3_mode": {
                "type": "str", "default": "half_s",
                "label": "Temporary-stack candidate mode k3",
            },
            "max_iterations": {
                "type": "int", "default": 150, "min": 1, "max": 100000,
                "label": "Multistart max iterations (a)",
            },
            "no_improve_limit": {
                "type": "int", "default": 100, "min": 1, "max": 100000,
                "label": "No-improvement cutoff (b)",
            },
            "move_budget_factor": {
                "type": "float", "default": 50.0, "min": 1.0, "max": 10000.0,
                "label": "Move budget factor",
            },
            "use_stack_filling": {
                "type": "bool", "default": True,
                "label": "Enable stack-filling improvement",
            },
        })
        return base
