"""
HuangLin2012LabellingTypeA
<2012> <heuristic> <premarshalling> <single-bay> <CRP-Prem>
Type-A R/W stack-labelling heuristic
b --- 0.4 --- Buffer-stack fraction
num_restarts --- 1 --- Randomized restarts

------------------------------- Reference --------------------------------
S.-H. Huang, T.-H. Lin,
"Heuristic algorithms for container pre-marshalling problems",
Computers & Industrial Engineering 62 (2012) 13–20.
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
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from core.base_algorithm import AlgorithmConfig, BaseAlgorithm
from core.layout_trace import trace_layout
from .core import Move, Stacks, heuristic_a_type_a


def _encode_action(src_idx: int, dst_idx: int, n_stacks: int) -> int:
    dst_flat = dst_idx if dst_idx < src_idx else dst_idx - 1
    return src_idx * (n_stacks - 1) + dst_flat


class HuangLin2012LabellingTypeA(BaseAlgorithm):
    name = "Huang-Lin (2012) Labelling-A"
    category = "Heuristic"
    description = (
        "[single-bay origin] Type-A labelling heuristic for pre-marshalling "
        "(R/W stack labelling; complete/deconstruct strategy with buffers)."
    )
    compatible_problems = ["CRP-Prem"]
    geometry = "single-bay"
    objectives = ["relocations"]
    fidelity = "faithful"
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
        b = float(cfg.extra.get("b", 0.4))
        max_moves_factor = float(cfg.extra.get("max_moves_factor", 20.0))
        num_restarts = int(cfg.extra.get("num_restarts", 1))
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

            stacks0: Stacks = {
                key_to_idx[k]: list(st.priority_snapshot())
                for k, st in env.yard.stacks.items()
            }
            n_containers = sum(len(v) for v in stacks0.values())
            max_moves = max(1, int(max_moves_factor * max(1, n_containers)))

            trace_layout(
                f"HuangLin2012TypeA seed={seed+1}/{n_seeds} b={b:.3f} "
                f"restarts={num_restarts} max_moves={max_moves}"
            )

            best_moves: List[Move] = []
            best_rem = 10**9
            best_solved = False
            t0 = _time.perf_counter()

            for rr in range(max(1, num_restarts)):
                if stop_event.is_set():
                    break
                rng = random.Random(int(cfg.seed) + seed * 1009 + rr)
                seq, rem, solved = heuristic_a_type_a(
                    stacks_init=stacks0,
                    max_tiers=max_tiers,
                    b=b,
                    max_moves=max_moves,
                    rng=rng,
                )
                if (rem, len(seq)) < (best_rem, len(best_moves) if best_moves else 10**9):
                    best_moves = list(seq)
                    best_rem = int(rem)
                    best_solved = bool(solved)
                if best_solved:
                    break

            elapsed = _time.perf_counter() - t0
            metric = float(len(best_moves) if best_solved else len(best_moves) + 1000 * (best_rem + 1))
            metrics = {
                "moves": float(len(best_moves)),
                "bad_overlaps": float(best_rem),
                "time": float(elapsed),
                "solved": 1.0 if best_solved else 0.0,
            }
            all_metrics.append(metrics)

            if metric < self._best_metric:
                self._best_metric = metric
                self._best_solution = [_encode_action(s, d, n_stacks) for (s, d) in best_moves]

            self._push(
                result_queue,
                step=seed + 1,
                metric=metric,
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
            "b": {
                "type": "float", "default": 0.4, "min": 0.05, "max": 0.99,
                "label": "High-stack threshold ratio b",
                "help": "Stacks with height >= ceil(b*H) are treated as high-R in Type-A.",
            },
            "num_restarts": {
                "type": "int", "default": 1, "min": 1, "max": 100,
                "label": "Randomized restarts per seed",
            },
            "max_moves_factor": {
                "type": "float", "default": 20.0, "min": 1.0, "max": 500.0,
                "label": "Move budget factor (x number of containers)",
            },
        })
        return base
