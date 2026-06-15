"""
Caserta, Schwarze, Voß (2009) — Look-Ahead Heuristic (LAH) for CRP-R.

Algorithm overview
------------------
A randomised metaheuristic built on top of the Min–Max greedy heuristic.
The key idea comes from the *pilot method*: instead of deterministically
picking the best destination for the current blocker, the algorithm

  1. Enumerates every feasible destination s' ∈ S  (at most W−1 options).
  2. Evaluates each resulting configuration A' by running the deterministic
     greedy heuristic (Min–Max) to completion — the resulting relocation
     count is the *greedy score* g(A') (look-ahead depth = 1).
  3. Selects the next move via *roulette-wheel* sampling where the
     attractiveness of A' = max_g − g(A') + 1, so lower-score neighbours
     are preferred but not exclusively chosen.

The full algorithm (Matrix-Algorithm, Section 4) repeats this trajectory
construction until a stopping criterion is met, keeping the best solution
found across all restarts.  A *trajectory-fathoming* test abandons any
partial trajectory whose current cost already equals or exceeds the best
known upper bound.

Relation to Caserta (2012) HEUR
---------------------------------
The deterministic greedy rule used here (Section 3) is identical to
Caserta et al. (2012) Eq.(11).  The 2009 paper wraps that rule in a
randomised look-ahead framework, trading determinism for solution quality
across multiple restarts.

Why this belongs in CRP-R / Heuristic
--------------------------------------
The algorithm is single-bay, restricted (Assumption A1), minimises
relocations, and does not require training — it is a pure constructive
metaheuristic, consistent with the other heuristics in this family.

Config parameters
-----------------
  n_restarts     : int   — number of trajectory restarts (default 200)
  num_eval_seeds : int   — independent runs per layout (default 1 for
                           fixed benchmark; increase for random-layout mode)

Reference
---------
M. Caserta, S. Schwarze, S. Voß,
"A New Binary Description of the Blocks Relocation Problem and Benefits
 in a Look Ahead Heuristic",
Evolutionary Computation in Combinatorial Optimization (EvoCOP 2009),
LNCS 5482, pp. 37–48, Springer, Berlin Heidelberg, 2009.
https://doi.org/10.1007/978-3-642-01009-5_4
"""

from __future__ import annotations

import multiprocessing as mp
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from core.base_algorithm import BaseAlgorithm, AlgorithmConfig
from core.layout_trace import trace_layout
from .scoring import greedy_simulate, run_trajectory


class CasertaLAH(BaseAlgorithm):

    name                = "Caserta et al. (2009) LAH"
    category            = "Heuristic"
    description         = (
        "[single-bay origin]  "
        "Caserta, Schwarze, Voß (EvoCOP 2009) Look-Ahead Heuristic for CRP-R. "
        "Pilot-method inspired: enumerates all W−1 one-step neighbours, scores "
        "each with the greedy Min–Max heuristic (look-ahead depth 1), and selects "
        "via roulette-wheel sampling. Multi-restart with trajectory fathoming. "
        "Outperforms Kim–Hong and Corridor Method on large instances (Table 1)."
    )
    compatible_problems = ["CRP-R"]
    step_label          = "Restart"

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
        cfg         = self.config
        n_seeds     = max(1, cfg.num_eval_seeds)
        n_restarts  = int(cfg.extra.get("n_restarts", 200))
        report_every = max(1, n_restarts // 20)  # ~20 GUI pushes per seed

        all_metrics: List[Dict] = []

        for seed in range(n_seeds):
            if stop_event.is_set():
                break

            np.random.seed(cfg.seed + seed * 1000)
            rng = np.random.RandomState(cfg.seed + seed * 1000)

            trace_layout(
                f"CasertaLAH.train: seed {seed+1}/{n_seeds}  "
                f"n_restarts={n_restarts}"
            )

            env = problem_factory()
            env.config.seed = seed
            env.reset(options={"skip_auto_retrieve": True})

            n_total   = int(env.config.num_containers)
            max_tiers = int(env.config.max_tiers)
            n_stacks  = env.config.num_bays * env.config.num_rows

            all_keys:    List[Any] = []
            stacks_init: Dict[Any, List[int]] = {}

            for a in range(n_stacks):
                key = env._action_to_stack(a)
                all_keys.append(key)
                stk = env.yard.stacks.get(key)
                stacks_init[key] = (
                    [int(c.priority) for c in stk.containers] if stk else []
                )

            # ── Warm-start: deterministic greedy upper bound ─────────── #
            best_cost = greedy_simulate(stacks_init, n_total, max_tiers, all_keys)

            # ── Multi-restart loop (Matrix-Algorithm, lines 3–24) ─────── #
            for restart in range(n_restarts):
                if stop_event.is_set():
                    break

                cost = run_trajectory(
                    stacks_init, n_total, max_tiers, all_keys, rng, best_cost
                )

                if cost < best_cost:
                    best_cost = cost

                if (restart + 1) % report_every == 0:
                    frac = (seed * n_restarts + restart + 1) / (n_seeds * n_restarts)
                    self._push(
                        result_queue,
                        step     = restart + 1,
                        metric   = float(best_cost),
                        metrics  = {
                            "relocations": float(best_cost),
                            "restart":     float(restart + 1),
                        },
                        progress = min(frac, (seed + 1) / n_seeds),
                    )

            # ── End of seed ───────────────────────────────────────────── #
            metrics = {
                "relocations": float(best_cost),
                "steps":       float(best_cost),
                "time":        float(best_cost),
                "progress":    1.0,
            }
            all_metrics.append(metrics)

            if float(best_cost) < self._best_metric:
                self._best_metric   = float(best_cost)
                self._best_solution = []

            self._push(
                result_queue,
                step     = seed + 1,
                metric   = float(best_cost),
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
            "num_eval_seeds": {
                "type":    "int",
                "default": 1,
                "min":     1,
                "max":     50,
                "label":   "Evaluation seeds",
                "help": (
                    "Number of independent runs per layout. "
                    "Use 1 for fixed benchmark files. "
                    "Increase for random-layout mode to assess stability."
                ),
            },
            "n_restarts": {
                "type":    "int",
                "default": 200,
                "min":     1,
                "max":     5000,
                "label":   "Restarts",
                "help": (
                    "Number of trajectory restarts (outer loop). "
                    "Each restart builds one complete relocation sequence via "
                    "randomised look-ahead; the best is retained. "
                    "More restarts → better solutions, longer runtime."
                ),
            },
        })
        return base
