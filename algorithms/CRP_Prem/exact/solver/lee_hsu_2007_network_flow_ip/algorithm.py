"""
Lee & Hsu (2007) network-flow integer-programming model for the container
pre-marshalling problem.

Reference
---------
Y. Lee, N.-Y. Hsu, "An optimization model for the container pre-marshalling
problem", Computers & Operations Research 34 (2007) 3295-3313.

Embedding scope ("main algorithm" focus)
-----------------------------------------
The paper's central contribution is an exact model on a time-expanded
multi-commodity flow network: each stack/tier/time-point is a node, and
five arc types (upward, downward, internal/stationary-in-slot, stationary
across a segment, and crane movement) carry 0/1 flow of container "types".
This embedding implements:

* ``mip_model.py`` -- the basic model, constraints (1)-(26) and objective
  (26), built directly as a Gurobi MIP (paper used CPLEX; this platform
  standardizes exact/solver modules on Gurobi).
* the paper's own practical relaxation (Sec. 5): replacing the "one
  movement per time segment" constraint (4) with a cap K (constraints
  (33)/(34) instead of (3)/(5)/(21)), plus the 2- and 3-stack
  cycle-breaking constraints (31)/(32). Without this, T must equal the
  number of moves and the model is -- as the paper itself reports --
  impractically slow even on 12-30 container instances.
* ``ordering.py`` -- Sec. 5's own procedure for turning the (possibly
  simultaneous, possibly cyclic) per-segment movement arcs into an
  executable move sequence, breaking any residual >= 4-length cycle by
  rerouting one move through a spare stack.
* ``extensions.py`` -- Eq. (27) exact final layout and Eq. (28)
  one-type-per-stack, both optional config toggles.
* ``solve.py`` -- since the model needs a fixed time horizon T chosen
  up front, this drives a small ascending T search under a wall-clock
  budget (the paper's own Table 3 experiments do the same by hand).

Documented simplifications / out-of-scope for this embedding
--------------------------------------------------------------
* Container "types" C default to the distinct priority values present in
  the instance (matches CRP-Prem's actual well-located definition exactly
  when types are unique); when an instance has more distinct values than
  ``max_types``, they are bucketed into ``max_types`` contiguous groups
  (paper explicitly treats C as a user-chosen size/fidelity trade-off).
  Bucketing is a documented approximation: containers sharing a bucket
  become interchangeable, so the final "solved" metric always re-checks
  the *original* (non-bucketed) priorities, never the coarse ones.
* The Eq. (29)-(30) "concurrent unloading" extension is not implemented:
  it introduces a new arc/variable whose time index is stated
  inconsistently between Eq. (29) (TIME\\{1}) and the pop-out conservation
  constraint it is meant to replace, Eq. (19) (TIME\\{T}), and it models a
  scenario (containers leaving the yard *during* premarshalling) that
  CRP-Prem itself does not have.
* The Sec. 6 two-phase matheuristic (heuristic construction + MIP repair)
  is not implemented; this embedding is the exact model only.
* Because binary-variable count grows as O(T x S x H x C), this is, as in
  the paper's own experiments, only practical on small instances. A hard
  ``max_binaries`` guard skips oversized (T, C) combinations rather than
  hanging the solver.
"""

from __future__ import annotations

import multiprocessing as mp
import time as _time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from core.base_algorithm import AlgorithmConfig, BaseAlgorithm
from core.layout_trace import trace_layout

from .core import Stacks
from .solve import solve_lee_hsu


def _encode_action(src_idx: int, dst_idx: int, n_stacks: int) -> int:
    dst_flat = dst_idx if dst_idx < src_idx else dst_idx - 1
    return src_idx * (n_stacks - 1) + dst_flat


class LeeHsu2007NetworkFlowIP(BaseAlgorithm):

    name = "Lee-Hsu (2007) Network-Flow IP"
    category = "Exact"
    description = (
        "[single-bay origin] Lee & Hsu (COR 2007) exact time-expanded "
        "multi-commodity network-flow model for the pre-marshalling "
        "problem, solved with Gurobi. Uses the paper's own multi-move "
        "relaxation and cycle-breaking constraints to keep the time "
        "horizon small, with an ascending time-horizon search driver and "
        "a post-processing step that reconstructs an executable move "
        "sequence from the solved flow."
    )
    compatible_problems = ["CRP-Prem"]
    requires_solver = True
    solver_backend = "gurobi"

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

        t_start = int(cfg.extra.get("t_start", 4))
        t_max = int(cfg.extra.get("t_max", 12))
        t_step = int(cfg.extra.get("t_step", 2))
        auto_size_t = bool(cfg.extra.get("auto_size_t", True))
        max_moves_per_segment = int(cfg.extra.get("max_moves_per_segment", 3))
        cycle_breaking_level = int(cfg.extra.get("cycle_breaking_level", 3))
        per_solve_time_limit_s = float(cfg.extra.get("per_solve_time_limit_s", 30.0))
        overall_time_budget_s = float(cfg.extra.get("overall_time_budget_s", 120.0))
        max_types = int(cfg.extra.get("max_types", 12))
        max_binaries = int(cfg.extra.get("max_binaries", 250_000))
        one_type_per_stack = bool(cfg.extra.get("one_type_per_stack", False))

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

            trace_layout(
                f"LeeHsu2007NetworkFlowIP seed={seed+1}/{n_seeds} "
                f"T=[{t_start}..{t_max}:{t_step}] K={max_moves_per_segment} "
                f"cycle_break={cycle_breaking_level}"
            )

            t0 = _time.perf_counter()
            result = solve_lee_hsu(
                stacks_init,
                max_tiers=max_tiers,
                t_start=t_start,
                t_max=t_max,
                t_step=t_step,
                auto_size_t=auto_size_t,
                max_moves_per_segment=max_moves_per_segment,
                cycle_breaking_level=cycle_breaking_level,
                per_solve_time_limit_s=per_solve_time_limit_s,
                overall_time_budget_s=overall_time_budget_s,
                max_types=max_types,
                max_binaries=max_binaries,
                one_type_per_stack=one_type_per_stack,
            )
            elapsed = _time.perf_counter() - t0

            moves_list = list(result.moves)
            solved = bool(result.solved)

            metrics = {
                "moves": float(len(moves_list)),
                "remaining_not_well_located": float(result.remaining_bad_overlaps),
                "time": float(elapsed),
                "solved": 1.0 if solved else 0.0,
                "proved_optimal": 1.0 if result.proved_optimal else 0.0,
                "T_used": float(result.T_used or 0),
                "n_types": float(result.n_types),
                "n_binaries": float(result.n_binaries),
                "attempts": float(result.attempts),
            }
            all_metrics.append(metrics)

            primary = float(
                len(moves_list) if solved else len(moves_list) + 1000.0 * (result.remaining_bad_overlaps + 1)
            )
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
            "t_start": {
                "type": "int", "default": 4, "min": 1, "max": 200,
                "label": "Initial time horizon T",
            },
            "t_max": {
                "type": "int", "default": 12, "min": 1, "max": 400,
                "label": "Maximum time horizon T tried",
            },
            "t_step": {
                "type": "int", "default": 2, "min": 1, "max": 50,
                "label": "T increment between attempts",
            },
            "auto_size_t": {
                "type": "bool", "default": True,
                "label": "Seed T from a cheap move-count heuristic",
                "help": "Adds one extra T candidate sized from a quick (non-paper) greedy upper bound.",
            },
            "max_moves_per_segment": {
                "type": "int", "default": 3, "min": 1, "max": 20,
                "label": "Max simultaneous moves per time segment (K)",
                "help": "K=1 reproduces the paper's literal basic model (Eq. 4); K>1 applies the (33)/(34) relaxation.",
            },
            "cycle_breaking_level": {
                "type": "int", "default": 3, "min": 0, "max": 3,
                "label": "In-model cycle breaking (0=off, 2=2-cycle Eq.31, 3=+3-cycle Eq.32)",
            },
            "per_solve_time_limit_s": {
                "type": "float", "default": 30.0, "min": 1.0, "max": 3600.0,
                "label": "Gurobi time limit per T attempt (s)",
            },
            "overall_time_budget_s": {
                "type": "float", "default": 120.0, "min": 1.0, "max": 86400.0,
                "label": "Overall wall-clock budget per seed (s)",
            },
            "max_types": {
                "type": "int", "default": 12, "min": 1, "max": 200,
                "label": "Max container types before bucket-coarsening",
                "help": "Instances with more distinct priorities are bucketed to keep the model tractable (paper treats C as a user-chosen parameter).",
            },
            "max_binaries": {
                "type": "int", "default": 250_000, "min": 1000, "max": 20_000_000,
                "label": "Binary-variable cap per T attempt",
                "help": "T candidates whose estimated model size exceeds this are skipped instead of hanging the solver.",
            },
            "one_type_per_stack": {
                "type": "bool", "default": False,
                "label": "Extension: require a single type per stack in the final layout (Eq. 28)",
            },
        })
        return base
