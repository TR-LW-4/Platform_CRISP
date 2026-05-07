"""
Lin, Lee & Lee (2015) priority-rule heuristic for CRP-Time / CRP-R.

Reference
---------
D.-Y. Lin, Y.-J. Lee, Y. Lee,
"The container retrieval problem with respect to relocation",
Transportation Research Part C 52 (2015) 132–143.

Role on the platform
--------------------
Main target problem: ``CRP-Time`` (multi-bay RMGC, Lee & Lee kinematics).
Shin et al. (TRC 2026) use this rule — with ``P_r = 30`` and
``P_b = 300`` — as the strongest classical baseline for the CRP.  The
degenerate ``num_bays = 1`` case makes it usable on ``CRP-R`` too.

Per-step decision flow (see ``scoring.lin_select_action``)
----------------------------------------------------------
1. Read the current blocker above the priority-based target.
2. Score every admissible destination stack:

       score(s) = P_b * 1[not well_placed] + P_r * severity + carry_time

3. Pick the argmin.  The platform then calls ``env.step(action)`` which
   records a ``Movement`` via `CRP_Time._hook_after_relocate` and feeds
   it to ``compute_crane_time`` on terminal step.

Seed loop matches the other rule-based baselines (``caserta``,
``kim_hong``) so Experiment / Compare Tabs can plot it side-by-side.
"""

from __future__ import annotations

import multiprocessing as mp
from typing import Callable, Dict, List, Optional

import numpy as np

from core.base_algorithm import BaseAlgorithm, AlgorithmConfig
from .scoring import lin_select_action


class Lin2015Heuristic(BaseAlgorithm):

    name                = "Lin–Lee–Lee (2015) Rule"
    category            = "Heuristic"
    description         = (
        "[native multi-bay]  "
        "Lin, Lee & Lee (TRC 2015) priority-rule heuristic for CRP-Time. "
        "Composite score per candidate destination: "
        "P_b·(not-well-placed) + P_r·severity + carry-time "
        "(Lee–Lee RMGC kinematics).  Defaults P_r=30, P_b=300 follow "
        "Shin et al. (TRC 2026).  Also usable on CRP-R (single-bay)."
    )
    compatible_problems = ["CRP-Time", "CRP-R"]
    step_label          = "Seed"

    def __init__(self, config: Optional[AlgorithmConfig] = None):
        super().__init__(config)

    # ---------------------------------------------------------------- #
    # Train                                                              #
    # ---------------------------------------------------------------- #

    def train(
        self,
        problem_factory: Callable,
        result_queue:    mp.Queue,
        stop_event:      mp.Event,
    ) -> None:
        cfg      = self.config
        rng_seed = cfg.seed
        n_seeds  = max(1, cfg.num_eval_seeds)
        P_r      = float(cfg.extra.get("P_r", 30.0))
        P_b      = float(cfg.extra.get("P_b", 300.0))

        all_metrics: List[Dict] = []

        for seed_idx in range(n_seeds):
            if stop_event.is_set():
                break

            env = problem_factory()
            env.config.seed = rng_seed + seed_idx
            _, info = env.reset()

            solution: List[int] = []
            done = False

            while not done:
                mask   = info.get("action_mask")
                action = lin_select_action(env, mask, P_r=P_r, P_b=P_b)
                _, _, done, _, info = env.step(action)
                solution.append(int(action))

            metrics = env.get_metrics()
            all_metrics.append(metrics)

            # Primary metric: crane_time when the problem is CRP-Time,
            # otherwise fall back to relocations.  This keeps the "best"
            # comparison aligned with the problem's own objective.
            primary = float(
                metrics.get("crane_time", metrics.get("relocations", 0.0))
            )

            if primary < self._best_metric:
                self._best_metric   = primary
                self._best_solution = solution[:]

            self._push(
                result_queue,
                step     = seed_idx + 1,
                metric   = primary,
                metrics  = metrics,
                progress = (seed_idx + 1) / n_seeds,
                snapshot = env.get_state_snapshot(),
                extra    = {"P_r": P_r, "P_b": P_b, "seed": seed_idx},
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

    # ---------------------------------------------------------------- #
    # Public accessors                                                   #
    # ---------------------------------------------------------------- #

    def get_best_solution(self) -> Optional[List[int]]:
        return self._best_solution

    # ---------------------------------------------------------------- #
    # Config schema                                                      #
    # ---------------------------------------------------------------- #

    @classmethod
    def config_schema(cls) -> Dict:
        base = super().config_schema()
        base.update({
            "num_eval_seeds": {
                "type": "int", "default": 10, "min": 1, "max": 100,
                "label": "Evaluation seeds",
                "help": "Number of random initial layouts to evaluate over.",
            },
            "P_r": {
                "type": "float", "default": 30.0, "min": 0.0, "max": 10_000.0,
                "label": "P_r (severity weight)",
                "help": (
                    "Weight on severity = max(0, blocker_priority - min_priority(dst)). "
                    "Default 30 follows Shin et al. (TRC 2026)."
                ),
            },
            "P_b": {
                "type": "float", "default": 300.0, "min": 0.0, "max": 100_000.0,
                "label": "P_b (not-well-placed weight)",
                "help": (
                    "Penalty when the blocker would NOT be well-placed at the "
                    "chosen stack (= needs re-relocation later). "
                    "Default 300 follows Shin et al. (TRC 2026)."
                ),
            },
        })
        return base
