"""
Jovanović & Voß (2014) Chain heuristic for CRP-R.

Algorithm overview
------------------
This heuristic improves the Min–Max rule (Caserta et al., 2012) by introducing
a one-step look-ahead: when deciding where to relocate block rn, the algorithm
also evaluates where block rn+1 would have to go, and chooses the order that
yields a better combined outcome.

Two variants are provided via the `use_chain_f` switch:
  • Chain    (use_chain_f=False) – pure chain look-ahead (Eq. 11)
  • Chain F  (use_chain_f=True)  – chain + correction for stacks about to become
                                   full (Eq. 5), default; best in paper tables

Config parameters
-----------------
  num_eval_seeds : int   – number of random seeds (layouts) to evaluate
  use_chain_f    : bool  – enable Chain-F full-stack correction (default True)

Compatible problems
-------------------
  CRP-R only (restricted relocation, minimize total relocations)

Reference
---------
R. Jovanović, S. Voß,
"A chain heuristic for the Blocks Relocation Problem",
Computers & Industrial Engineering 75 (2014) 79–86.
https://doi.org/10.1016/j.cie.2014.06.010
"""

from __future__ import annotations

import multiprocessing as mp
from typing import Callable, Dict, List, Optional

import numpy as np

from core.base_algorithm import BaseAlgorithm, AlgorithmConfig
from core.layout_trace import trace_layout
from .scoring import chain_select_action


class JovanovicVoss2014Chain(BaseAlgorithm):

    name                = "Jovanović & Voß (2014) Chain"
    category            = "Heuristic"
    description         = (
        "[single-bay origin]  "
        "Jovanović & Voß (C&IE 2014) chain look-ahead heuristic for CRP-R. "
        "Extends the Min–Max rule by considering the NEXT block to be relocated "
        "when choosing the destination for the current one. "
        "Two modes: Chain (look-ahead only) and Chain F (+ full-stack correction). "
        "Chain F improves on Min–Max by ≈5% on average and up to 8% on large instances."
    )
    compatible_problems = ["CRP-R"]
    step_label          = "Seed"

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
        n_seeds      = max(1, cfg.num_eval_seeds)
        use_chain_f  = bool(cfg.extra.get("use_chain_f", True))
        all_metrics: List[Dict] = []

        for seed in range(n_seeds):
            if stop_event.is_set():
                break

            trace_layout(
                f"JovanovicVoss2014Chain.train: seed {seed + 1}/{n_seeds}  "
                f"use_chain_f={use_chain_f}"
            )

            env = problem_factory()
            env.config.seed = seed
            env.reset()

            solution: List[int] = []
            done = False

            while not done:
                info   = env._get_info()
                action = chain_select_action(env, use_chain_f=use_chain_f)
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
    # Configuration schema (GUI + CLI)                                  #
    # ---------------------------------------------------------------- #

    @classmethod
    def config_schema(cls) -> Dict:
        base = super().config_schema()
        base.update({
            "num_eval_seeds": {
                "type":    "int",
                "default": 10,
                "min":     1,
                "max":     200,
                "label":   "Evaluation seeds",
                "help":    "Number of random initial layouts to evaluate.",
            },
            "use_chain_f": {
                "type":    "bool",
                "default": True,
                "label":   "Enable Chain-F correction",
                "help": (
                    "When True (Chain F), penalises destination stacks that would "
                    "become full when a deadlock is unavoidable (Eq. 5). "
                    "Improves average relocations by ≈5% over plain Chain on large "
                    "instances. Set False for the basic Chain heuristic."
                ),
            },
        })
        return base
