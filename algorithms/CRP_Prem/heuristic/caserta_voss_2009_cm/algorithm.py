"""
Caserta & Voß (2009) — Corridor Method (CM) for the Pre-marshalling Problem.

Algorithm overview
-------------------
A stochastic metaheuristic that reshuffles a bay so that every stack ends
up internally sorted (no container sits above a more urgent one), using
the minimum number of moves.  Each iteration is made up of four phases:

  1. Corridor Definition/Selection — roulette-wheel pick of a source stack
     (weighted by its forced-relocation count), classification of every
     other stack into empty / no-deadlock / deadlock groups relative to
     the block being relocated, and a stochastic draw of a "corridor" of
     candidate destination stacks (parameter ``delta``).
  2. Neighbourhood Search — the bay configurations reachable by placing
     the block on each corridor stack.
  3. Move Evaluation and Selection — GRASP-style: keep the best
     ``elite_quantile`` fraction (fewest resulting forced relocations),
     then roulette-select one.
  4. Local Search Improvement — two fast operators that opportunistically
     finish sorting stacks (``heuristic_swap``,
     ``heuristic_subsequence_building``) without needing the corridor
     machinery.

Because every phase is randomized, the algorithm is run multiple times per
instance (``num_restarts``, bounded by ``time_limit``) and the best
fully-sorted solution is kept — mirroring the original C++ implementation,
which restarts until its time budget is exhausted.

This algorithm is self-contained (see ``scoring.py``): it only depends on
the platform's generic ``core`` package and does not import from any other
algorithm module, so it can be added, modified or removed independently.

Config parameters
------------------
  delta          : int   — corridor width, i.e. number of candidate
                           destination stacks considered per move (paper
                           default: unrestricted / large)
  w0, w1, w2     : float — attractiveness weights for empty / no-deadlock /
                           deadlock destination stacks (Eq. 1)
  elite_quantile : float — fraction of corridor destinations kept as elite
                           candidates before the final roulette-wheel pick
  num_restarts   : int   — independent randomized attempts per instance;
                           the best fully-sorted solution is kept
  time_limit     : float — wall-clock budget (seconds) per instance,
                           shared across all restarts
                           to evaluate

Compatible problems
--------------------
  CRP-Prem

Reference
---------
M. Caserta, S. Voß,
"A Corridor Method-Based Algorithm for the Pre-marshalling Problem",
in: M. Giacobini et al. (Eds.), EvoWorkshops 2009, LNCS 5484, pp. 788-797,
Springer, 2009.
https://doi.org/10.1007/978-3-642-01129-0_89
"""

from __future__ import annotations

import multiprocessing as mp
import time as _time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from core.base_algorithm import BaseAlgorithm, AlgorithmConfig
from core.layout_trace import trace_layout
from .scoring import corridor_method_solve


class CasertaVossCM(BaseAlgorithm):

    name                = "Caserta & Voß (2009) CM"
    category            = "Heuristic"
    description         = (
        "[single-bay origin]  "
        "Caserta & Voß (EvoWorkshops 2009) Corridor Method for the pre-marshalling "
        "problem. Four-phase stochastic metaheuristic: roulette-wheel corridor "
        "selection over empty/no-deadlock/deadlock stacks, GRASP-style elite move "
        "sampling, and two local-search operators (sorted-run building, "
        "subsequence chaining). Multiple randomized restarts keep the best "
        "fully-sorted solution found within the time budget."
    )
    compatible_problems = ["CRP-Prem"]
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
        cfg            = self.config
        n_seeds = 1  # multi-seed eval removed; single run only
        delta          = int(cfg.extra.get("delta", 6))
        w0             = float(cfg.extra.get("w0", 0.4))
        w1             = float(cfg.extra.get("w1", 0.3))
        w2             = float(cfg.extra.get("w2", 0.3))
        elite_quantile = float(cfg.extra.get("elite_quantile", 0.6))
        num_restarts   = int(cfg.extra.get("num_restarts", 30))
        time_limit     = float(cfg.extra.get("time_limit", 10.0))
        all_metrics: List[Dict] = []

        for seed in range(n_seeds):
            if stop_event.is_set():
                break

            trace_layout(
                f"CasertaVossCM.train: seed {seed+1}/{n_seeds}  "
                f"delta={delta}  restarts={num_restarts}  time_limit={time_limit}s"
            )

            env = problem_factory()
            env.reset()

            n_total   = int(env.config.num_containers)
            max_tiers = int(env.config.max_tiers)

            # Build plain-dict stacks (priorities, bottom→top) — independent
            # of the problem's gym action encoding.
            stacks_init: Dict[Any, List[int]] = {
                key: stack.priority_snapshot()
                for key, stack in env.yard.stacks.items()
            }

            t0 = _time.perf_counter()
            moves, remaining, solved = corridor_method_solve(
                stacks_init=stacks_init,
                n_total=n_total,
                max_tiers=max_tiers,
                delta=delta,
                w0=w0, w1=w1, w2=w2,
                elite_quantile=elite_quantile,
                num_restarts=num_restarts,
                time_limit=time_limit,
                seed=seed,
            )
            elapsed = _time.perf_counter() - t0

            metrics = {
                "moves":        float(moves),
                "bad_overlaps": float(remaining),
                "time":         float(elapsed),
                "solved":       1.0 if solved else 0.0,
                "progress":     1.0,
            }
            all_metrics.append(metrics)

            # Unsolved attempts are penalised so they never look better than
            # a fully-sorted solution when the platform tracks "best_metric".
            primary = float(moves) if solved else float(moves + 1000 * (remaining + 1))
            if primary < self._best_metric:
                self._best_metric   = primary
                self._best_solution = []

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
    # Configuration schema                                               #
    # ---------------------------------------------------------------- #

    @classmethod
    def config_schema(cls) -> Dict:
        base = super().config_schema()
        base.update({
            "delta": {
                "type":    "int",
                "default": 6,
                "min":     1,
                "max":     30,
                "label":   "Corridor width (δ)",
                "help": (
                    "Number of candidate destination stacks stochastically "
                    "admitted into the corridor at each move. Larger δ "
                    "explores more of the neighbourhood per step at extra "
                    "cost; the paper effectively uses an unrestricted "
                    "corridor (δ ≈ number of stacks)."
                ),
            },
            "w0": {
                "type":    "float",
                "default": 0.4,
                "min":     0.0,
                "max":     1.0,
                "label":   "Empty-stack weight (w0)",
                "help": "Attractiveness weight for empty destination stacks (Eq. 1).",
            },
            "w1": {
                "type":    "float",
                "default": 0.3,
                "min":     0.0,
                "max":     1.0,
                "label":   "No-deadlock weight (w1)",
                "help": (
                    "Attractiveness weight for stacks where the relocated "
                    "block creates no future forced relocation."
                ),
            },
            "w2": {
                "type":    "float",
                "default": 0.3,
                "min":     0.0,
                "max":     1.0,
                "label":   "Deadlock weight (w2)",
                "help": (
                    "Attractiveness weight for stacks where the relocation "
                    "is unavoidable but can be postponed."
                ),
            },
            "elite_quantile": {
                "type":    "float",
                "default": 0.6,
                "min":     0.1,
                "max":     1.0,
                "label":   "Elite quantile",
                "help": (
                    "Fraction of corridor destinations kept as elite "
                    "candidates before the final roulette-wheel pick "
                    "(GRASP-style; paper uses the best 50% quantile)."
                ),
            },
            "num_restarts": {
                "type":    "int",
                "default": 30,
                "min":     1,
                "max":     200,
                "label":   "Randomized restarts per instance",
                "help": (
                    "Every phase of the algorithm is stochastic, so distinct "
                    "restarts explore different trajectories. The best "
                    "fully-sorted solution across restarts is kept."
                ),
            },
            "time_limit": {
                "type":    "float",
                "default": 5.0,
                "min":     0.5,
                "max":     300.0,
                "label":   "Time limit per instance (s)",
                "help": (
                    "Wall-clock budget shared across all restarts for a "
                    "single instance. Paper uses 20 s for its random-instance "
                    "experiments."
                ),
            },
        })
        return base
