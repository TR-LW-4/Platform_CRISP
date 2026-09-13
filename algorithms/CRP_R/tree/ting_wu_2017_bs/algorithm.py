"""
TingWuBS
<2017> <heuristic> <restricted> <single-bay> <CRP-R>
Beam search with virtual relocation heuristic
beam_width --- 5 --- Nodes retained at each level
evaluator --- VRH --- Evaluation heuristic: VRH or SDH

------------------------------- Reference --------------------------------
C.-J. Ting, K.-C. Wu,
"Optimizing container relocation operations at container yards with
 beam search",
Transportation Research Part E 103 (2017) 17–31.
------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import multiprocessing as mp
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from core.base_algorithm import BaseAlgorithm, AlgorithmConfig
from core.layout_trace import trace_layout
from .scoring import beam_search, vrh_simulate, sdh_simulate


class TingWuBS(BaseAlgorithm):

    name                = "Ting & Wu (2017) BS"
    category            = "Heuristic"
    description         = "Ting & Wu (TRE 2017) beam search with VRH."
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
        cfg          = self.config
        n_seeds = 1  # multi-seed eval removed; single run only
        beam_width   = int(cfg.extra.get("beam_width", 5))
        evaluator    = str(cfg.extra.get("evaluator", "VRH"))
        all_metrics: List[Dict] = []

        for seed in range(n_seeds):
            if stop_event.is_set():
                break

            trace_layout(
                f"TingWuBS.train: seed {seed+1}/{n_seeds}  "
                f"beam_width={beam_width}  evaluator={evaluator}"
            )

            env = problem_factory()
            env.reset(options={"skip_auto_retrieve": True})

            n_total   = int(env.config.num_containers)
            max_tiers = int(env.config.max_tiers)
            n_stacks  = env.config.num_bays * env.config.num_rows

            # Build plain-dict stacks
            all_keys:    List[Any] = []
            stacks_init: Dict[Any, List[int]] = {}

            for a in range(n_stacks):
                key = env._action_to_stack(a)
                all_keys.append(key)
                stk = env.yard.stacks.get(key)
                stacks_init[key] = (
                    [int(c.priority) for c in stk.containers] if stk else []
                )

            # Run beam search
            relocations = beam_search(
                stacks_init=stacks_init,
                n_total=n_total,
                max_tiers=max_tiers,
                all_keys=all_keys,
                beam_width=beam_width,
                evaluator=evaluator,
            )

            metrics = {
                "relocations": float(relocations),
                "steps":       float(relocations),
                "time":        float(relocations),
                "progress":    1.0,
            }
            all_metrics.append(metrics)

            if float(relocations) < self._best_metric:
                self._best_metric   = float(relocations)
                self._best_solution = []

            self._push(
                result_queue,
                step     = seed + 1,
                metric   = float(relocations),
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
    # Configuration schema                                               #
    # ---------------------------------------------------------------- #

    @classmethod
    def config_schema(cls) -> Dict:
        base = super().config_schema()
        base.update({
            "beam_width": {
                "type":    "int",
                "default": 5,
                "min":     1,
                "max":     50,
                "label":   "Beam width (b)",
                "help": (
                    "Number of promising nodes retained at each level. "
                    "Paper tests b = 5 / 10 / 15 / 20.  "
                    "b = 5 already achieves near-optimal on all benchmarks "
                    "and runs in < 0.1 s per instance."
                ),
            },
            "evaluator": {
                "type":    "str",
                "default": "VRH",
                "options": ["VRH", "SDH"],
                "label":   "Evaluation heuristic",
                "help": (
                    "VRH (Virtual Relocation Heuristic) — considers all blocking "
                    "containers simultaneously via a two-phase decision sequence. "
                    "Outperforms Min–Max/Chain on all 48 test sizes (Table 1). "
                    "SDH (Smallest Difference Heuristic) — equivalent to Caserta "
                    "Min–Max; included as a simpler baseline."
                ),
            },
        })
        return base
