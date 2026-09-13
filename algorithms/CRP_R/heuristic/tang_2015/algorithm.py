"""
TangEtAl2015
<2015> <heuristic> <restricted> <single-bay> <CRP-R>
H1/H2 reshuffling rules with optional full-sequence extended choice
rule --- H2 --- Base rule: H1 reshuffle-index or H2 blocking-index
use_extended --- True --- Evaluate destinations by remaining-sequence rollout

------------------------------- Reference --------------------------------
L. Tang, W. Jiang, J. Liu, Y. Dong,
"Research into container reshuffling and stacking problems in container
 terminal yards",
IIE Transactions 47 (2015) 751–766.
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
from .scoring import tang_select_action


class TangEtAl2015(BaseAlgorithm):

    name                = "Tang et al. (2015) H1/H2"
    category            = "Heuristic"
    description         = "Tang et al. (IIE Trans. 2015) H1/H2 reshuffling heuristic."
    compatible_problems = ["CRP-R"]
    geometry            = "single-bay"
    objectives          = ["relocations"]
    fidelity            = "faithful"

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

        cfg          = self.config
        rule         = str(cfg.extra.get("rule", "H2"))
        use_extended = bool(cfg.extra.get("use_extended", True))

        trace_layout(f"TangEtAl2015.train: rule={rule}  use_extended={use_extended}")

        env = problem_factory()
        env.reset()

        solution: List[int] = []
        done = False

        while not done:
            if stop_event.is_set():
                break
            action = tang_select_action(env, rule=rule, use_extended=use_extended)
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
    # Configuration schema (drives both GUI widgets and CLI defaults)   #
    # ---------------------------------------------------------------- #

    @classmethod
    def config_schema(cls) -> Dict:
        base = super().config_schema()
        base.update({
            "rule": {
                "type":    "str",
                "default": "H2",
                "options": ["H1", "H2"],
                "label":   "Heuristic rule",
                "help": (
                    "H1 scores a stack by how many containers are retrieved sooner "
                    "than the one being moved. "
                    "H2 scores it by how deeply the most urgent container would be buried. "
                    "Both first try a stack that does not create a new blockage."
                ),
            },
            "use_extended": {
                "type":    "bool",
                "default": True,
                "label":   "Extended variant (*-E)",
                "help": (
                    "When enabled, every feasible destination is evaluated by "
                    "simulating the remaining retrieval sequence; "
                    "the one with fewest reshuffles is chosen."
                ),
            },
        })
        return base
