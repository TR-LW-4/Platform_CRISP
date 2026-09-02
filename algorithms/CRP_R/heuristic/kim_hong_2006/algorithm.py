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
from typing import Callable, List, Optional

from core.base_algorithm import BaseAlgorithm, AlgorithmConfig
from core.layout_trace import trace_layout
from core.move_export import attach_crane_time, export_yard_moves
from .scoring import select_action


class KimHong2006ENARHeuristic(BaseAlgorithm):

    name        = "Kim–Hong (2006) ENAR"
    category    = "Heuristic"
    description = "Kim & Hong (COR 2006) ENAR-inspired destination heuristic."
    compatible_problems = ["CRP-R", "CRP-Time"]

    def __init__(self, config: Optional[AlgorithmConfig] = None):
        super().__init__(config)

    def train(
        self,
        problem_factory: Callable,
        result_queue:    mp.Queue,
        stop_event:      mp.Event,
    ) -> None:
        if stop_event.is_set():
            return

        trace_layout("KimHongHeuristic.train: single run")

        env = problem_factory()
        env.reset()

        print("step=0 (initial yard)", file=sys.stderr, flush=True)
        print(env.yard, file=sys.stderr, flush=True)

        solution: List[int] = []
        done = False
        step_i = 0

        while not done:
            if stop_event.is_set():
                break
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

        metrics = attach_crane_time(
            env.get_metrics(),
            env.yard,
            getattr(env.config, "extra", None) or {},
        )
        metrics["time"] = metrics["crane_time"]
        metrics["feasible"] = float(done)
        metrics["completed"] = float(done)
        metrics["validated"] = 1.0
        metrics["validation_conflicts"] = 0.0 if done else 1.0
        if not done:
            metrics["executed_relocations"] = float(
                metrics.get("relocations", 0.0)
            )
            metrics["relocations"] = float("inf")
        primary = float(metrics.get("relocations", 0.0))
        self._best_solution = solution[:]

        self._push(
            result_queue,
            step     = 1,
            metric   = primary,
            metrics  = metrics,
            progress = 1.0,
            extra    = {
                "solution": solution[:],
                "moves": export_yard_moves(env.yard),
            },
        )

    def get_best_solution(self) -> Optional[List[int]]:
        return self._best_solution
