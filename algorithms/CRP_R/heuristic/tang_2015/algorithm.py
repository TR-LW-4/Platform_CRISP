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

import numpy as np

from core.base_algorithm import BaseAlgorithm, AlgorithmConfig
from core.layout_trace import trace_layout
from .scoring import tang_select_action


class TangEtAl2015(BaseAlgorithm):

    name                = "Tang et al. (2015) H1/H2"
    category            = "Heuristic"
    description         = (
        "[single-bay origin]  "
        "Tang, Jiang, Liu & Dong (IIE Trans. 2015) reshuffling heuristics for CRP-R. "
        "H1 uses the Reshuffle Index (RI); H2 uses the Blocking Index (BI). "
        "Both prefer destinations where the relocated container will be retrieved "
        "before the current stack's most urgent container (nc > k). "
        "The Extended variant (*-E, default) picks the destination that minimises "
        "total future reshuffles via full-sequence simulation — H2-E achieves the "
        "best average performance across all tested bay configurations."
    )
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
        cfg          = self.config
        n_seeds = 1  # multi-seed eval removed; single run only
        rule         = str(cfg.extra.get("rule", "H2"))
        use_extended = bool(cfg.extra.get("use_extended", True))
        all_metrics: List[Dict] = []

        for seed in range(n_seeds):
            if stop_event.is_set():
                break

            trace_layout(
                f"TangEtAl2015.train: seed {seed + 1}/{n_seeds}  "
                f"rule={rule}  use_extended={use_extended}"
            )

            env = problem_factory()
            env.reset()

            solution: List[int] = []
            done = False

            while not done:
                action = tang_select_action(env, rule=rule, use_extended=use_extended)
                _, _, done, _, _ = env.step(action)
                solution.append(action)

            metrics = env.get_metrics()
            all_metrics.append(metrics)
            primary = float(metrics.get("relocations", 0.0))

            if primary < self._best_metric:
                self._best_metric   = primary
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
                    "H1: Reshuffle-Index based — best at 100% bay utilisation. "
                    "H2: Blocking-Index based — best at ~80% utilisation and in "
                    "dynamic environments (paper Tables 2–5). "
                    "Both first try to place the blocker where nc > k (no future deadlock)."
                ),
            },
            "use_extended": {
                "type":    "bool",
                "default": True,
                "label":   "Extended variant (*-E)",
                "help": (
                    "When enabled, every feasible destination is evaluated by "
                    "simulating the full remaining retrieval sequence with the base "
                    "heuristic; the destination yielding fewest reshuffles is chosen. "
                    "H2-E is the top performer across static and dynamic benchmarks "
                    "(paper Tables 3 and 5). Negligible extra run-time."
                ),
            },
        })
        return base
