"""
Zhu, Qin, Lim & Zhang (2012) IDA* algorithm for the Restricted BRP (CRP-R).

Reference
---------
W. Zhu, H. Qin, A. Lim, H. Zhang, "Iterative deepening A* algorithms for
the container relocation problem", IEEE Transactions on Automation
Science and Engineering, 9(4) (2012) 710-722.

Algorithm overview
-------------------
Iterative deepening A* (IDA*), *not* branch-and-bound: successive depth-
first passes explore all nodes with cost estimate ``f(n) = g(n) + h(n)``
up to an increasing threshold, using an admissible lower bound ``h(n)``
(Section V-A: LB1/LB2/LB3) and a greedy "probe" heuristic (Section V-B:
PR1-PR4) to find good solutions early and narrow the search window.  The
paper reports this outperforms the branch-and-bound approach of Kim &
Hong (2006) (``exact/search/kim_hong_2006``) by a wide margin under
tight (1 s) time limits.

Best configuration from the paper (IDA*-R, Section VII-C):
``IDAStar(pi, PR+, LB3, PR4)`` -- initial upper bound from the best of
PR1-PR4 ("PR+"), LB3 for branch-and-bound pruning, PR4 for probing
children on the search frontier.  All exposed as configurable options
below.

Independence
------------
Self-contained: depends only on ``core.base_algorithm``, ``core.yard``
and its own ``ida_core`` module (the IDA* engine).  Does not import
from any other algorithm package.
"""

from __future__ import annotations

import multiprocessing as mp
import sys
import time
from typing import Callable, Dict, List, Optional

import numpy as np

from core.base_algorithm import BaseAlgorithm, AlgorithmConfig
from core.yard import Yard

from .ida_core import LB1, LB2, LB3, PR1, PR2, PR3, PR4, ida_star_restricted, state_from_stacks


# ================================================================ #
#  Yard -> lightweight state conversion                              #
# ================================================================ #

def _yard_to_stacks(yard: Yard) -> List[List[int]]:
    """Return [[priority, ...], ...] bottom -> top, ordered by (bay, row)."""
    stack_list = sorted(yard.stacks.values(), key=lambda s: (s.bay, s.row))
    return [[c.priority for c in st.containers] for st in stack_list]


# ================================================================ #
#  BaseAlgorithm subclass                                            #
# ================================================================ #

class Zhu2012IDAStarR(BaseAlgorithm):
    """
    Exact IDA* search for the Restricted BRP / CRP-R.

    Reference: Zhu, Qin, Lim & Zhang (2012), IEEE T-ASE 9(4), 710-722.

    Unlike ``exact/solver`` algorithms, this is not an IP formulation --
    it is a hand-crafted iterative-deepening search directly over bay
    states, using the paper's LB1/LB2/LB3 lower bounds and PR1-PR4
    probe heuristics.

    Parameters
    ----------
    time_limit_s : float
        Per-seed wall-clock time limit in seconds (default 300).  If
        the search is stopped early, the best solution found so far is
        reported and ``optimal_proven`` is set to 0.0.
    lb_mode : str
        Lower bound used for pruning: "LB1" | "LB2" | "LB3" (default
        "LB3", the paper's strongest and best-performing bound for
        larger instances).
    probe_mode : str
        Probe heuristic used on frontier nodes: "PR1" | "PR2" | "PR3" |
        "PR4" (default "PR4").
    """

    name                = "Zhu et al. (2012) IDA*-R"
    category            = "Exact"
    description         = (
        "Iterative deepening A* for the Restricted BRP (Zhu, Qin, Lim & "
        "Zhang, IEEE T-ASE 2012). Admissible lower bounds LB1/LB2/LB3 "
        "for pruning; probe heuristics PR1-PR4 narrow the search window. "
        "Guarantees the optimal number of relocations if it completes "
        "within the time limit."
    )
    compatible_problems = ["CRP-R"]
    step_label          = "Seed"

    _LB_MODES    = (LB1, LB2, LB3)
    _PROBE_MODES = (PR1, PR2, PR3, PR4)

    def __init__(self, config: Optional[AlgorithmConfig] = None) -> None:
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
        cfg        = self.config
        extra      = cfg.extra or {}
        time_limit = float(extra.get("time_limit_s", 300.0))
        lb_mode    = str(extra.get("lb_mode", LB3))
        probe_mode = str(extra.get("probe_mode", PR4))
        if lb_mode not in self._LB_MODES:
            lb_mode = LB3
        if probe_mode not in self._PROBE_MODES:
            probe_mode = PR4

        n_seeds     = max(1, cfg.num_eval_seeds)
        all_metrics: List[Dict] = []

        for seed in range(n_seeds):
            if stop_event.is_set():
                break

            t_start = time.perf_counter()

            env = problem_factory()
            env.config.seed = seed
            env.reset(seed=seed, options={"skip_auto_retrieve": True})

            n_total = len(env.containers)
            max_tiers = env.config.max_tiers
            state0 = state_from_stacks(_yard_to_stacks(env.yard))

            result = ida_star_restricted(
                state0, 1, n_total, max_tiers,
                lb_mode=lb_mode, probe_mode=probe_mode,
                time_limit_s=time_limit, stop_event=stop_event,
            )

            t_elapsed = time.perf_counter() - t_start
            n_reloc = result["best"]

            metrics: Dict = {
                "relocations":    float(n_reloc),
                "steps":          float(n_reloc),
                "time":           float(n_reloc),
                "nodes_explored": float(result["nodes"]),
                "root_lower_bound": float(result["root_lb"]),
                "optimal_proven": 1.0 if result["optimal"] else 0.0,
                "solve_time_s":   round(t_elapsed, 4),
            }
            all_metrics.append(metrics)
            primary = float(n_reloc)

            if primary < self._best_metric:
                self._best_metric   = primary
                self._best_solution = []  # move list not tracked (state-space search)

            self._push(
                result_queue,
                step     = seed + 1,
                metric   = primary,
                metrics  = metrics,
                progress = (seed + 1) / n_seeds,
                snapshot = env.get_state_snapshot(),
            )

            print(
                f"[Zhu2012IDAStarR] seed={seed}  relocations={n_reloc}  "
                f"optimal={result['optimal']}  nodes={result['nodes']}  "
                f"t={t_elapsed:.3f}s",
                file=sys.stderr,
                flush=True,
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
    # Config schema (for GUI / API)                                      #
    # ---------------------------------------------------------------- #

    @classmethod
    def config_schema(cls) -> Dict:
        base = super().config_schema()
        base.update({
            "num_eval_seeds": {
                "type": "int", "default": 1, "min": 1, "max": 20,
                "label": "Seeds",
                "help": "Number of independent seeds to evaluate.",
            },
            "time_limit_s": {
                "type": "float", "default": 300.0, "min": 1.0, "max": 86400.0,
                "label": "Time limit (s)",
                "help": (
                    "Per-seed wall-clock time limit. The paper uses a strict "
                    "1 s budget per instance under practical time constraints; "
                    "a larger budget increases the chance of proving optimality."
                ),
            },
            "lb_mode": {
                "type": "str", "default": LB3, "choices": list(cls._LB_MODES),
                "label": "Lower bound",
                "help": (
                    "LB1 (Kim & Hong 2006) / LB2 / LB3 (Zhu 2012, Eqs. 1-4). "
                    "LB3 dominates LB2 dominates LB1 but costs more per node; "
                    "the paper recommends LB3 for larger instances."
                ),
            },
            "probe_mode": {
                "type": "str", "default": PR4, "choices": list(cls._PROBE_MODES),
                "label": "Probe heuristic",
                "help": (
                    "Greedy heuristic (PR1-PR4, Section V-B.1) used to "
                    "complete promising frontier nodes during search."
                ),
            },
        })
        return base
