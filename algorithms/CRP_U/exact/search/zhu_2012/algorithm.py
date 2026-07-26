"""
Zhu, Qin, Lim & Zhang (2012) IDA* algorithm for the Unrestricted BRP (CRP-U).

Reference
---------
W. Zhu, H. Qin, A. Lim, H. Zhang, "Iterative deepening A* algorithms for
the container relocation problem", IEEE Transactions on Automation
Science and Engineering, 9(4) (2012) 710-722.

Algorithm overview
-------------------
Same iterative-deepening A* framework as ``exact/search/zhu_2012`` in
CRP-R (see that package's docstring for Algorithm 1 / Algorithm 2), but
with the *unrestricted* branching rule: a relocation may move the top
container of **any** stack (not only a blocker above the current
target) to any other non-full stack.  Three configurations from
Section VII-D are exposed:

  - **IDA*-U**  (``lb_mode="LB1"``, ``use_visited_map=False``): exact,
    admissible LB1 only, no transposition table.
  - **IDA*-UM** (``lb_mode="LB1"``, ``use_visited_map=True``, default):
    exact; a transposition table avoids re-exploring layouts already
    seen at an equal-or-shallower depth within the same iteration.
  - **IDA*-UM3** (``lb_mode="LB3"``): reuses the restricted variant's
    LB3 bound for more aggressive pruning.  The paper notes LB3 is
    **not proven admissible for CRP-U** -- solutions found in this
    mode are never reported as ``optimal_proven`` even if the search
    appears to converge.

Independence
------------
Self-contained: depends only on ``core.base_algorithm``, ``core.yard``
and its own ``ida_core`` module.  Duplicates (rather than imports) the
small amount of logic shared in spirit with the CRP-R package -- see
that package's ``ida_core.py`` docstring for the rationale.
"""

from __future__ import annotations

import multiprocessing as mp
import sys
import time
from typing import Callable, Dict, List, Optional

import numpy as np

from core.base_algorithm import BaseAlgorithm, AlgorithmConfig
from core.yard import Yard

from .ida_core import LB1, LB3, PU1, PU2, ida_star_unrestricted, state_from_stacks


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

class Zhu2012IDAStarU(BaseAlgorithm):
    """
    Exact (or, in aggressive mode, near-exact) IDA* search for the
    Unrestricted BRP / CRP-U.

    Reference: Zhu, Qin, Lim & Zhang (2012), IEEE T-ASE 9(4), 710-722.

    Parameters
    ----------
    time_limit_s : float
        Per-seed wall-clock time limit in seconds (default 300).
    lb_mode : str
        "LB1" (admissible, default) or "LB3" (aggressive, reused from
        the restricted analysis, **not proven admissible for CRP-U**).
    probe_mode : str
        Probe heuristic used on frontier nodes: "PU1" | "PU2" (default
        "PU2"); both build on the restricted PR3/PR4 destination rule
        with the "relocate a due helper stack out of the way first"
        refinement (Section V-B.2).
    use_visited_map : bool
        Maintain a per-iteration transposition table to skip
        re-exploring layouts already seen at an equal-or-shallower
        depth (IDA*-UM vs plain IDA*-U). Default True.
    """

    name                = "Zhu et al. (2012) IDA*-U"
    category            = "Exact"
    description         = (
        "Iterative deepening A* for the Unrestricted BRP (Zhu, Qin, Lim & "
        "Zhang, IEEE T-ASE 2012). LB1 (admissible) or LB3 (aggressive, not "
        "proven admissible for CRP-U) for pruning; PU1/PU2 probe heuristics "
        "narrow the search window; optional transposition table (IDA*-UM)."
    )
    compatible_problems = ["CRP-U"]
    step_label          = "Seed"

    _LB_MODES    = (LB1, LB3)
    _PROBE_MODES = (PU1, PU2)

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
        lb_mode    = str(extra.get("lb_mode", LB1))
        probe_mode = str(extra.get("probe_mode", PU2))
        use_map    = bool(extra.get("use_visited_map", True))
        if lb_mode not in self._LB_MODES:
            lb_mode = LB1
        if probe_mode not in self._PROBE_MODES:
            probe_mode = PU2

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

            result = ida_star_unrestricted(
                state0, 1, n_total, max_tiers,
                lb_mode=lb_mode, probe_mode=probe_mode, use_map=use_map,
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
                f"[Zhu2012IDAStarU] seed={seed}  relocations={n_reloc}  "
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
                    "1 s budget under practical time constraints; the "
                    "unrestricted variant's larger branching factor makes "
                    "proven optimality much harder for W,H > 4-5."
                ),
            },
            "lb_mode": {
                "type": "str", "default": LB1, "choices": list(cls._LB_MODES),
                "label": "Lower bound",
                "help": (
                    "LB1 (admissible for CRP-U, default -> IDA*-U/-UM) or "
                    "LB3 (aggressive, reused from the restricted analysis; "
                    "NOT proven admissible for CRP-U -> IDA*-UM3; solutions "
                    "are never marked optimal in this mode)."
                ),
            },
            "probe_mode": {
                "type": "str", "default": PU2, "choices": list(cls._PROBE_MODES),
                "label": "Probe heuristic",
                "help": "PU1 (base PR3) or PU2 (base PR4), Section V-B.2.",
            },
            "use_visited_map": {
                "type": "bool", "default": True,
                "label": "Use transposition table (IDA*-UM)",
                "help": (
                    "Skip re-exploring layouts already visited at an "
                    "equal-or-shallower depth within the current iteration. "
                    "Matches the paper's IDA*-UM (vs. plain IDA*-U)."
                ),
            },
        })
        return base
