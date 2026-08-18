"""
KimHong2006ENARHeuristic
<2006> <heuristic> <restricted> <single-bay> <CRP-R>
ENAR-style destination scoring with inversion and height tie-breaks

------------------------------- Reference --------------------------------
K.H. Kim, G.P. Hong,
"A heuristic rule for relocating blocks",
Computers & Operations Research 33 (2006) 940–954.
------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import multiprocessing as mp
import sys
from typing import Callable, Dict, List, Optional

import numpy as np

from core.base_algorithm import BaseAlgorithm, AlgorithmConfig
from core.layout_trace import trace_layout
from .scoring import select_action


class KimHong2006ENARHeuristic(BaseAlgorithm):

    name        = "Kim–Hong (2006) ENAR"
    category    = "Heuristic"
    description = (
        "[single-bay origin]  "
        "Kim & Hong (COR 2006) rule: minimise containers in the destination that "
        "must be retrieved before the relocating block (ENAR-inspired), "
        "with inversion/height tie-breaking."
    )
    compatible_problems = ["CRP-R", "CRP-Time"]

    def __init__(self, config: Optional[AlgorithmConfig] = None):
        super().__init__(config)

    def train(
        self,
        problem_factory: Callable,
        result_queue:    mp.Queue,
        stop_event:      mp.Event,
    ) -> None:
        cfg     = self.config
        n_seeds = 1  # multi-seed eval removed; single run only
        all_metrics: List[Dict] = []

        for seed in range(n_seeds):
            if stop_event.is_set():
                break

            trace_layout(
                f"KimHongHeuristic.train: inner loop seed {seed + 1}/{n_seeds} "
                f"(each seed ⇒ new env + reset(); same layout file while problem_factory is unchanged)"
            )

            env = problem_factory()
            env.reset()

            print(
                f"step=0 (initial yard)  seed={seed}",
                file=sys.stderr,
                flush=True,
            )
            print(env.yard, file=sys.stderr, flush=True)

            solution: List[int] = []
            done = False
            step_i = 0

            while not done:
                info   = env._get_info()
                action = select_action(env, info.get("action_mask"))
                _, _, done, _, info = env.step(action)

                step_i += 1
                dst = env._action_to_stack(action)
                m = env.get_metrics()
                print(
                    f"step={step_i}  action={action}  dst_stack={dst}  "
                    f"relocations={m['relocations']}  steps={m['steps']}",
                    file=sys.stderr,
                    flush=True,
                )
                print(env.yard, file=sys.stderr, flush=True)
                solution.append(action)

            metrics = env.get_metrics()
            all_metrics.append(metrics)
            primary = float(metrics.get("relocations", 0.0))

            if primary < self._best_metric:
                self._best_solution = solution[:]

            self._push(
                result_queue,
                step     = seed + 1,
                metric   = primary,
                metrics  = metrics,
                progress = (seed + 1) / n_seeds,
                snapshot = env.get_state_snapshot(),
            )

        if all_metrics:
            agg = {
                k: float(np.mean([m[k] for m in all_metrics if k in m]))
                for k in all_metrics[0]
            }
            self._push(
                result_queue,
                step     = n_seeds,
                metric   = self._best_metric,
                metrics  = agg,
                progress = 1.0,
            )

    def get_best_solution(self) -> Optional[List[int]]:
        return self._best_solution
