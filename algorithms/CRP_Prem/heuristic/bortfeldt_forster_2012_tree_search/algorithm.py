"""
BortfeldtForster2012TreeSearch
<2012> <heuristic> <premarshalling> <single-bay> <CRP-Prem>
Compound-move tree search for the CPMP
n_succ --- 5 --- Successor compound moves per node
time_limit_s --- 20 --- Search time limit (s)

------------------------------- Reference --------------------------------
A. Bortfeldt, F. Forster,
"A tree search procedure for the container pre-marshalling problem",
European Journal of Operational Research 217 (2012) 531–540.
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

from core.base_algorithm import AlgorithmConfig, BaseAlgorithm
from core.layout_trace import trace_layout

from .core import Stacks, infer_num_groups
from .lower_bound import lb_bg_only, lb_moves
from .tree_search import perform_tree_search


def _encode_action(src_idx: int, dst_idx: int, n_stacks: int) -> int:
    dst_flat = dst_idx if dst_idx < src_idx else dst_idx - 1
    return src_idx * (n_stacks - 1) + dst_flat


class BortfeldtForster2012TreeSearch(BaseAlgorithm):

    name = "Bortfeldt-Forster (2012) Tree Search"
    category = "Heuristic"
    description = (
        "[single-bay origin] Bortfeldt & Forster (EJOR 2012) heuristic tree "
        "search for the CPMP. Uses natural BG/BB/GG/GB move classification, "
        "the Proposition 1 lower bound n0_m = n0_BX + n0_GX, and branches on "
        "compound moves (normal = BG-only max length; extra = non-BG "
        "prefixes) with three-rule filtering per branch. Paper §7.5 "
        "ablation switches V1-V4 exposed via config.extra."
    )
    compatible_problems = ["CRP-Prem"]
    requires_solver = False
    solver_backend = None

    def __init__(self, config: Optional[AlgorithmConfig] = None):
        super().__init__(config)

    # ---------------------------------------------------------------- #
    # Training loop                                                      #
    # ---------------------------------------------------------------- #

    def train(
        self,
        problem_factory: Callable,
        result_queue: mp.Queue,
        stop_event: mp.Event,
    ) -> None:
        cfg = self.config
        n_seeds = 1  # multi-seed eval removed; single run only

        # Paper §7 parameters.
        n_succ = int(cfg.extra.get("n_succ", 5))
        pub = float(cfg.extra.get("pub", 1.75))
        time_limit_s = float(cfg.extra.get("time_limit_s", 20.0))

        # Paper §7.5 ablation switches (V0 = all True; V1..V4 flip one flag).
        use_full_lb = bool(cfg.extra.get("use_full_lb", True))                    # V1 -> False
        use_compound_moves = bool(cfg.extra.get("use_compound_moves", True))      # V2 -> False
        sort_compound_moves = bool(cfg.extra.get("sort_compound_moves", True))    # V3 -> False
        use_filtering_rules = bool(cfg.extra.get("use_filtering_rules", True))    # V4 -> False

        lb_fn = (
            (lambda st, ng, ht: lb_moves(st, ng, ht))
            if use_full_lb
            else (lambda st, ng, ht: lb_bg_only(st))
        )

        all_metrics: List[Dict[str, float]] = []

        for seed in range(n_seeds):
            if stop_event.is_set():
                break

            env = problem_factory()
            env.reset()

            stack_keys = list(env.yard.stacks.keys())
            key_to_idx = {k: i for i, k in enumerate(stack_keys)}
            n_stacks = len(stack_keys)
            max_tiers = int(env.config.max_tiers)

            stacks_init: Stacks = {
                key_to_idx[k]: list(st.priority_snapshot())
                for k, st in env.yard.stacks.items()
            }
            num_groups = max(
                int(getattr(env.config, "num_groups", 0) or 0),
                infer_num_groups(stacks_init),
            )

            trace_layout(
                f"BortfeldtForster2012 seed={seed+1}/{n_seeds} "
                f"n_succ={n_succ} pub={pub} time_limit_s={time_limit_s} "
                f"V0={use_full_lb and use_compound_moves and sort_compound_moves and use_filtering_rules}"
            )

            result = perform_tree_search(
                initial_stacks=stacks_init,
                max_tiers=max_tiers,
                num_groups=num_groups,
                lb_fn=lb_fn,
                n_succ=n_succ,
                pub=pub,
                time_limit_s=time_limit_s,
                use_filtering=use_filtering_rules,
                truncate_to_single_move=(not use_compound_moves),
                sort_output=sort_compound_moves,
            )

            moves_list = list(result.moves)
            solved = bool(result.solved)
            n_moves = len(moves_list)
            lb_init = int(result.lower_bound_init)
            lb_gap = max(0, n_moves - lb_init) if solved else -1

            metrics = {
                "moves": float(n_moves),
                "lower_bound": float(lb_init),
                "lb_gap": float(lb_gap if lb_gap >= 0 else -1.0),
                "time": float(result.time_seconds),
                "solved": 1.0 if solved else 0.0,
                "aborted_by_time": 1.0 if result.aborted_by_time else 0.0,
                "nodes_expanded": float(result.n_nodes_expanded),
                "compound_moves_evaluated": float(result.n_compound_moves_evaluated),
                "compound_moves_normal": float(result.n_normal_compound_moves),
                "compound_moves_extra": float(result.n_extra_compound_moves),
            }
            all_metrics.append(metrics)

            primary = float(n_moves if solved else n_moves + 1000 * (n_moves + 1))
            if primary < self._best_metric:
                self._best_metric = primary
                self._best_solution = [
                    _encode_action(s, d, n_stacks) for (s, d) in moves_list
                ]

            self._push(
                result_queue,
                step=seed + 1,
                metric=primary,
                metrics=metrics,
                progress=(seed + 1) / n_seeds,
                snapshot=env.get_state_snapshot(),
            )

        if all_metrics:
            agg = {
                k: float(np.mean([m[k] for m in all_metrics if k in m]))
                for k in all_metrics[0]
            }
            self._push(
                result_queue,
                step=n_seeds,
                metric=self._best_metric,
                metrics=agg,
                progress=1.0,
            )

    def get_best_solution(self) -> Optional[List[int]]:
        return self._best_solution

    # ---------------------------------------------------------------- #
    # Configuration schema                                               #
    # ---------------------------------------------------------------- #

    @classmethod
    def config_schema(cls) -> Dict[str, Dict[str, Any]]:
        base = super().config_schema()
        base.update({
            "n_succ": {
                "type": "int", "default": 5, "min": 1, "max": 50,
                "label": "n_succ (max branches per node)",
                "help": "Paper §7: n_succ = 5.",
            },
            "pub": {
                "type": "float", "default": 1.75, "min": 1.0, "max": 10.0,
                "label": "pub (initial upper-bound multiplier)",
                "help": (
                    "When no best solution exists yet, target = round(pub * lb_init) "
                    "for the acceptance criterion. Paper §7: pub = 1.75."
                ),
            },
            "time_limit_s": {
                "type": "float", "default": 20.0, "min": 0.1, "max": 3600.0,
                "label": "Time limit per instance (s)",
                "help": (
                    "Paper §7 uses 20 s for LC/CV benchmarks and 60 s for the "
                    "new BF-32 benchmark set."
                ),
            },
            "use_full_lb": {
                "type": "bool", "default": True,
                "label": "V0/V1: use full lower bound n0_m",
                "help": (
                    "V0 (True, default) uses Proposition 1 n0_m = n0_BX + n0_GX. "
                    "V1 (False) uses only the naive n0_BG = nb (paper §7.5)."
                ),
            },
            "use_compound_moves": {
                "type": "bool", "default": True,
                "label": "V0/V2: enable compound moves",
                "help": (
                    "V0 (True) uses compound moves; V2 (False) truncates each "
                    "compound move to its first single move (paper §7.5)."
                ),
            },
            "sort_compound_moves": {
                "type": "bool", "default": True,
                "label": "V0/V3: sort compound-move branches",
                "help": (
                    "V0 (True) sorts branches by (length, clean supply) as in "
                    "§6.1/§6.2. V3 (False) leaves branches unsorted (paper §7.5)."
                ),
            },
            "use_filtering_rules": {
                "type": "bool", "default": True,
                "label": "V0/V4: apply per-move filtering rules",
                "help": (
                    "V0 (True) applies the 3 filtering rules of §6.1 (normal) "
                    "and §6.2 (extra) to select each move within a compound "
                    "move. V4 (False) picks the first legal move (paper §7.5)."
                ),
            },
        })
        return base
