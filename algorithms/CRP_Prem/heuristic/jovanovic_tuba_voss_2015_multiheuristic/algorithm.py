"""
JovanovicTubaVoss2015MultiHeuristic
<2015> <heuristic> <premarshalling> <single-bay> <CRP-Prem>
Deterministic 48-combination multi-heuristic for the CPMP
full_48_search --- True --- Enumerate all stage-heuristic combinations
safe_slack --- 1 --- Deadlock-avoidance slack

------------------------------- Reference --------------------------------
R. Jovanovic, M. Tuba, S. Voss,
"A multi-heuristic approach for solving the pre-marshalling problem",
Central European Journal of Operations Research 25 (2017) 1–28.
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
from .multiheuristic import HB_SET, HF_SET, HS_SET, HW_SET, all_combos, run_multiheuristic


def _encode_action(src_idx: int, dst_idx: int, n_stacks: int) -> int:
    dst_flat = dst_idx if dst_idx < src_idx else dst_idx - 1
    return src_idx * (n_stacks - 1) + dst_flat


class JovanovicTubaVoss2015MultiHeuristic(BaseAlgorithm):

    name = "Jovanovic-Tuba-Voss (2015) Multi-Heuristic"
    category = "Heuristic"
    description = (
        "[single-bay origin] Jovanovic, Tuba & Voss (CEJOR 2015) "
        "deterministic multi-heuristic approach for the CPMP. Extends the "
        "Exposito-Izquierdo (2012) four-stage greedy skeleton with "
        "competing heuristics per stage (block selection, destination "
        "selection, relocation target, filling), enumerates all 48 "
        "combinations and keeps the best solution, avoiding randomization "
        "entirely. Includes formalized deadlock avoidance and a "
        "move-sequence correction pass."
    )
    compatible_problems = ["CRP-Prem"]
    geometry = "single-bay"
    objectives = ["relocations"]
    fidelity = "faithful"
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
        safe_slack = int(cfg.extra.get("safe_slack", 1))
        full_search = bool(cfg.extra.get("full_48_search", True))

        combos = None
        if not full_search:
            hb = str(cfg.extra.get("hb", HB_SET[0]))
            hs = str(cfg.extra.get("hs", HS_SET[0]))
            hw = str(cfg.extra.get("hw", HW_SET[0]))
            hf = str(cfg.extra.get("hf", HF_SET[0]))
            combos = [(hb, hs, hw, hf)]

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
                f"JovanovicTubaVoss2015MultiHeuristic seed={seed+1}/{n_seeds} "
                f"full_48_search={full_search} n_combos={len(combos) if combos else len(all_combos())}"
            )

            t0 = _time.perf_counter()
            result = run_multiheuristic(
                stacks_init,
                max_tiers=max_tiers,
                rng_seed=int(cfg.seed) + seed * 10007,
                safe_slack=safe_slack,
                combos=combos,
            )
            elapsed = _time.perf_counter() - t0

            moves_list = list(result.moves)
            solved = bool(result.solved)
            remaining = int(result.remaining_not_well_located)

            metrics = {
                "moves": float(len(moves_list)),
                "remaining_not_well_located": float(remaining),
                "time": float(elapsed),
                "solved": 1.0 if solved else 0.0,
            }
            all_metrics.append(metrics)

            primary = float(len(moves_list) if solved else len(moves_list) + 1000.0 * (remaining + 1))
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
            "full_48_search": {
                "type": "bool", "default": True,
                "label": "Run the full 48-combination multi-heuristic search",
                "help": (
                    "When disabled, runs a single fixed (hb, hs, hw, hf) "
                    "combination instead (useful for ablation)."
                ),
            },
            "hb": {
                "type": "str", "default": "descending",
                "label": "Ablation only: block-selection heuristic (descending|lookahead)",
            },
            "hs": {
                "type": "str", "default": "w",
                "label": "Ablation only: destination-stack heuristic (w|w_hat)",
            },
            "hw": {
                "type": "str", "default": "MinMax",
                "label": "Ablation only: relocation heuristic (TLP|LPI|MinMax)",
            },
            "hf": {
                "type": "str", "default": "Standard",
                "label": "Ablation only: filling heuristic (None|Standard|Safe|Stop)",
            },
            "safe_slack": {
                "type": "int", "default": 1, "min": 0, "max": 20,
                "label": "Safe-filling slack threshold (a)",
                "help": "Safe filling is only committed if it leaves at most this many empty tiers.",
            },
        })
        return base
