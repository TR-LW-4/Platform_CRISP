"""
Tierney, Pacino & Voß (2017) A*/IDA* for the pre-marshalling problem.

Paper:
K. Tierney, D. Pacino, S. Voß, "Solving the Pre-Marshalling Problem to
Optimality with A* and IDA*", Flexible Services and Manufacturing Journal
29(2), 223-259, 2017.

Embedding scope (v1):
  - A* (Algorithm 1) and IDA* (Algorithm 2) exact graph search
  - "direct" and EMO (Bortfeldt & Forster 2012) cost estimation heuristics,
    with the paper's Section 4.3 parent-clamp fix for the EMO bound's lack
    of consistency
  - Section 5 branching rules: move reversal prevention, unrelated move
    symmetry breaking (directly-successive / successive, both directions),
    transitive move avoidance (directly-successive / successive), and
    empty stack symmetry breaking
  - A*-specific state memoization and two-stage tie breaking (Section 4.4)

No external MIP/CP solver is used anywhere in this embedding -- both
algorithms are pure graph search. This module is self-contained and does
not import from any other algorithm folder.
"""

from __future__ import annotations

import multiprocessing as mp
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from core.base_algorithm import AlgorithmConfig, BaseAlgorithm
from core.layout_trace import trace_layout
from .core import Stacks, apply_move_sequence, is_sorted, mis_overlay_count
from .search import solve_astar, solve_idastar


def _encode_action(src_idx: int, dst_idx: int, n_stacks: int) -> int:
    dst_flat = dst_idx if dst_idx < src_idx else dst_idx - 1
    return src_idx * (n_stacks - 1) + dst_flat


class TierneyPacinoVoss2017AStarIDAStar(BaseAlgorithm):
    name = "Tierney-Pacino-Voß (2017) A*/IDA*"
    category = "Exact"
    description = (
        "[single-bay origin] Exact pre-marshalling via A*/IDA* graph search "
        "with the EMO (Bortfeldt & Forster) lower bound and novel unrelated-"
        "move / transitive-move / empty-stack symmetry breaking rules."
    )
    compatible_problems = ["CRP-Prem"]
    requires_solver = False
    solver_backend = None

    def __init__(self, config: Optional[AlgorithmConfig] = None):
        super().__init__(config)

    def train(
        self,
        problem_factory: Callable,
        result_queue: mp.Queue,
        stop_event: mp.Event,
    ) -> None:
        cfg = self.config
        n_seeds = 1  # multi-seed eval removed; single run only
        backend = str(cfg.extra.get("backend", "astar")).strip().lower()
        time_limit_s = float(cfg.extra.get("time_limit_s", 30.0))
        use_emo = bool(cfg.extra.get("use_emo_bound", True))
        unrelated_mode = str(cfg.extra.get("unrelated_mode", "successive")).strip().lower()
        transitive_mode = str(cfg.extra.get("transitive_mode", "successive")).strip().lower()
        direction = str(cfg.extra.get("direction", "lt")).strip().lower()
        empty_stack_symmetry = bool(cfg.extra.get("empty_stack_symmetry", True))
        use_memoization = bool(cfg.extra.get("astar_memoization", True))
        use_tie_breaking = bool(cfg.extra.get("astar_tie_breaking", True))
        depth_padding = int(cfg.extra.get("idastar_depth_padding", 200))
        max_expansions = int(cfg.extra.get("astar_max_expansions", 500_000))

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

            stacks0: Stacks = {
                key_to_idx[k]: list(st.priority_snapshot())
                for k, st in env.yard.stacks.items()
            }

            trace_layout(
                f"TierneyPacinoVoss2017 seed={seed+1}/{n_seeds} "
                f"backend={backend} emo={use_emo} t_limit={time_limit_s}s"
            )

            if backend == "astar":
                res = solve_astar(
                    stacks_init=stacks0,
                    max_tiers=max_tiers,
                    time_limit_s=time_limit_s,
                    use_emo=use_emo,
                    unrelated_mode=unrelated_mode,
                    transitive_mode=transitive_mode,
                    direction=direction,
                    empty_stack_symmetry=empty_stack_symmetry,
                    use_memoization=use_memoization,
                    use_tie_breaking=use_tie_breaking,
                    max_expansions=max_expansions,
                )
            else:
                res = solve_idastar(
                    stacks_init=stacks0,
                    max_tiers=max_tiers,
                    time_limit_s=time_limit_s,
                    use_emo=use_emo,
                    unrelated_mode=unrelated_mode,
                    transitive_mode=transitive_mode,
                    direction=direction,
                    empty_stack_symmetry=empty_stack_symmetry,
                    depth_padding=depth_padding,
                )

            final_stacks = apply_move_sequence(stacks0, res.moves, max_tiers)
            solved = bool(res.solved and is_sorted(final_stacks))
            bad_overlaps = mis_overlay_count(final_stacks)
            moves = float(len(res.moves))
            primary = moves if solved else moves + 1000.0 * (bad_overlaps + 1.0)

            actions = [_encode_action(s, d, n_stacks) for (s, d) in res.moves]
            if primary < self._best_metric:
                self._best_metric = primary
                self._best_solution = actions

            metrics = {
                "moves": moves,
                "bad_overlaps": float(bad_overlaps),
                "lower_bound": float(res.lower_bound),
                "expanded_nodes": float(res.expanded_nodes),
                "time": float(res.elapsed_s),
                "solved": 1.0 if solved else 0.0,
                "proved_optimal": 1.0 if res.proved_optimal else 0.0,
                "timed_out": 1.0 if res.timed_out else 0.0,
            }
            all_metrics.append(metrics)

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

    @classmethod
    def config_schema(cls) -> Dict:
        base = super().config_schema()
        base.update({
            "backend": {
                "type": "str", "default": "astar",
                "label": "Backend (astar/idastar)",
                "help": (
                    "A* (default) keeps an explicit open/closed list and is "
                    "faster in this pure-Python implementation; IDA* trades "
                    "speed for a constant memory footprint (Section 4 of "
                    "the paper), which matters more for very large bays."
                ),
            },
            "time_limit_s": {
                "type": "float", "default": 30.0, "min": 0.1, "max": 86400.0,
                "label": "Search time limit per seed (s)",
            },
            "use_emo_bound": {
                "type": "bool", "default": True,
                "label": "Use EMO lower bound (vs. direct count)",
            },
            "unrelated_mode": {
                "type": "str", "default": "successive",
                "label": "Unrelated-move rule (off/direct/successive)",
            },
            "transitive_mode": {
                "type": "str", "default": "successive",
                "label": "Transitive-move rule (off/direct/successive)",
            },
            "direction": {
                "type": "str", "default": "lt",
                "label": "Symmetry-breaking direction (lt/gt)",
            },
            "empty_stack_symmetry": {
                "type": "bool", "default": True,
                "label": "Empty stack symmetry breaking",
            },
            "astar_memoization": {
                "type": "bool", "default": True,
                "label": "A* state memoization",
            },
            "astar_tie_breaking": {
                "type": "bool", "default": True,
                "label": "A* two-stage tie breaking",
            },
            "astar_max_expansions": {
                "type": "int", "default": 500_000, "min": 1000, "max": 20_000_000,
                "label": "A* max node expansions",
            },
            "idastar_depth_padding": {
                "type": "int", "default": 200, "min": 0, "max": 10000,
                "label": "IDA* max extra depth over lower bound",
            },
        })
        return base
