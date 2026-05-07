"""
Tang, Jiang, Liu & Dong (2015) H1 / H2 reshuffling heuristics for CRP-R.

Paper contribution
------------------
The paper proposes five construction heuristics (H1–H5) and five extended
variants (H1-E – H5-E) for the static container reshuffling problem.

Why only H1 and H2 are implemented here
-----------------------------------------
H1 and H2 are the two foundational rules from which H3, H4, and H5 are
derived:
  • H3 = H1 with a modified decision order and a same-column restriction in
         a narrow special case. Paper results show only marginal improvement.
  • H4 = H1 with adjusted RI values that account for already-assigned blockers
         when multiple blockers share the same relocation stage.
  • H5 = H4 using BI (like H2) instead of RI.
H3/H4/H5 add algorithmic complexity while the performance gap is small (see
Tables 2 and 3 in the paper). H1 and H2 already represent the RI-based and
BI-based design choices and, together with the Extended variant, reproduce
the paper's best results. H3–H5 can be added later as additional `rule`
options without changing the architecture.

Why H2-E is the default
------------------------
From the paper's experiments:
  • Static problem (Table 2):
      – H2 is best at ~80% bay utilisation (the common case in practice).
      – H1 is best at 100% utilisation.
  • Extended variants (Table 3): H1-E and H2-E both perform near-optimally;
    H2-E has a slight edge in more cases.
  • Dynamic simulation (Table 5): H2-E achieves the best average across ALL
    tested bay configurations (6-column, 2–5 tiers), with negligible extra
    computation time compared with the base heuristic.
H2-E therefore offers the best overall performance for CRP-R benchmarks and
is the most practically relevant configuration for a research baseline.

Config
------
  rule         : "H1" | "H2"   (dropdown in GUI)
  use_extended : bool           (checkbox in GUI, default True → *-E variant)
  num_eval_seeds : int

Compatible problems
-------------------
  CRP-R

Reference
---------
L. Tang, W. Jiang, J. Liu, Y. Dong,
"Research into container reshuffling and stacking problems in container
 terminal yards",
IIE Transactions, 47(7), 751–766, 2015.
https://doi.org/10.1080/0740817X.2014.971201
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
            env.config.seed = seed
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
            "num_eval_seeds": {
                "type":    "int",
                "default": 10,
                "min":     1,
                "max":     200,
                "label":   "Evaluation seeds",
                "help":    "Number of random initial layouts to evaluate.",
            },
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
