"""
JovanovicVoss2014Chain
<2014> <heuristic> <restricted> <single-bay> <CRP-R>
Chain look-ahead over Min–Max destination choice
use_chain_f --- True --- Enable Chain-F full-stack correction

------------------------------- Reference --------------------------------
R. Jovanović, S. Voß,
"A chain heuristic for the Blocks Relocation Problem",
Computers & Industrial Engineering 75 (2014) 79–86.
------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import multiprocessing as mp
from typing import Callable, Dict, List, Optional

from core.base_algorithm import BaseAlgorithm, AlgorithmConfig
from core.layout_trace import trace_layout
from core.move_export import attach_crane_time, export_yard_moves
from .scoring import chain_select_action


class JovanovicVoss2014Chain(BaseAlgorithm):

    name                = "Jovanović & Voß (2014) Chain"
    category            = "Heuristic"
    description         = "Jovanović & Voß (C&IE 2014) chain look-ahead heuristic."
    compatible_problems = ["CRP-R"]
    def __init__(self, config: Optional[AlgorithmConfig] = None):
        super().__init__(config)

    # ---------------------------------------------------------------- #
    # Training loop                                                      #
    # ---------------------------------------------------------------- #

    def train(
        self,
        problem_factory: Callable,
        result_queue:    mp.Queue,
        stop_event:      mp.Event,
    ) -> None:
        if stop_event.is_set():
            return

        cfg         = self.config
        use_chain_f = bool(cfg.extra.get("use_chain_f", True))

        trace_layout(f"JovanovicVoss2014Chain.train: use_chain_f={use_chain_f}")

        env = problem_factory()
        env.reset()

        solution: List[int] = []
        done = False

        while not done:
            if stop_event.is_set():
                break
            action = chain_select_action(env, use_chain_f=use_chain_f)
            _, _, done, _, _ = env.step(action)
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

    # ---------------------------------------------------------------- #
    # Configuration schema (GUI + CLI)                                  #
    # ---------------------------------------------------------------- #

    @classmethod
    def config_schema(cls) -> Dict:
        base = super().config_schema()
        base.update({
            "use_chain_f": {
                "type":    "bool",
                "default": True,
                "label":   "Enable Chain-F correction",
                "help": (
                    "When True (Chain F), penalises destination stacks that would "
                    "become full when a deadlock is unavoidable. "
                ),
            },
        })
        return base
