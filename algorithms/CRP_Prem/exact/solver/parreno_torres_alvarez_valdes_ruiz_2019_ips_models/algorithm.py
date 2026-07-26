"""
Parreno-Torres, Alvarez-Valdes & Ruiz (2019) "Integer programming models
for the pre-marshalling problem", European Journal of Operational Research
274 (2019) 142-154.

Embedding scope: the paper's **IPS6** model (Section 5.4) -- the authors'
own best-performing formulation among the eight IP/IPS variants compared in
the paper -- solved via the paper's own **Section 6 iterative T-search**
(ascend T starting from the "blocking containers" lower bound; the first
feasible T is provably optimal), rather than a single oversized-T solve.
This is deliberately a different sizing strategy from this platform's other
two PMP embeddings (Lee & Hsu 2007's ascending search over a different
weaker bound, and de Melo da Silva et al. 2018's own from-scratch greedy
heuristic), each kept faithful to its own paper.

This folder is fully self-contained: it does not import from, and is not
imported by, ``lee_hsu_2007_network_flow_ip/``, ``de_melo_silva_2018/``
(EJOR 271, 2018 -- a different paper), or
``exact/search/parreno_torres_alvarez_valdes_ruiz_tierney_2020_cpmpct/`` (Transp. Res.
Part E 137, 2020 -- a later, different paper by an overlapping author list
with a crane-time objective instead of a relocation-count objective).

What is implemented here
-------------------------
* ``core.py`` -- stack helpers plus the Section 6 "blocking containers"
  lower bound used to seed the T-search.
* ``mip_model.py`` -- the IPS6 variables (x/w/z) and constraints (26)-(38).
* ``extensions.py`` -- the four optional Section 8 final-layout extensions
  (balance, no empty/full stacks, stability bonus, same-priority bonus),
  all off by default and each provably unable to change the optimal move
  count.
* ``solve.py`` -- the Section 6 iterative T-search driver.

Documented simplifications / out of scope
------------------------------------------
* Priority groups map 1:1 to each instance's distinct container priority
  values (matching ``core.yard.Stack.is_sorted_by_priority``), same
  convention as this platform's other two PMP embeddings. Instances with
  more distinct priorities than ``max_priorities`` are bucketed into
  contiguous groups to keep the model tractable (a documented
  approximation, not part of the original paper).
* K=1 throughout (Eq. 28): the paper's PMP models do not consider
  multi-move segments.
* Eq. (29) ("earliest time" push) is implemented but left off by default,
  matching the paper's own Section 6 remark that it is unnecessary once
  the ascending iterative T-search is used.
"""

from __future__ import annotations

import multiprocessing as mp
import time as _time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from core.base_algorithm import AlgorithmConfig, BaseAlgorithm
from core.layout_trace import trace_layout

from .core import Stacks
from .solve import solve_ips6


def _encode_action(src_idx: int, dst_idx: int, n_stacks: int) -> int:
    dst_flat = dst_idx if dst_idx < src_idx else dst_idx - 1
    return src_idx * (n_stacks - 1) + dst_flat


class ParrenoTorresAlvarezValdesRuiz2019IPS6(BaseAlgorithm):

    name = "Parreno-Torres, Alvarez-Valdes & Ruiz (2019) IPS6 [Gurobi]"
    category = "Exact"
    description = (
        "Time-indexed MIP (IPS6, the paper's best model) for the "
        "Pre-marshalling Problem, solved with the paper's own ascending "
        "iterative T-search (starts at the 'blocking containers' lower "
        "bound; first feasible T is optimal). "
        "Parreno-Torres, Alvarez-Valdes & Ruiz -- EJOR 274 (2019). "
        "[Requires Gurobi license]"
    )
    compatible_problems = ["CRP-Prem"]
    step_label = "Seed"
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
        n_seeds = max(1, int(cfg.num_eval_seeds))

        transitive_move_breaking = bool(cfg.extra.get("transitive_move_breaking", True))
        fix_last_segment_vars = bool(cfg.extra.get("fix_last_segment_vars", True))
        continuous_move_vars = bool(cfg.extra.get("continuous_move_vars", False))
        balance_kappa_raw = cfg.extra.get("balance_kappa", 0)
        balance_kappa = int(balance_kappa_raw) if int(balance_kappa_raw) > 0 else None
        no_empty_or_full_stacks = bool(cfg.extra.get("no_empty_or_full_stacks", False))
        stability_bonus = bool(cfg.extra.get("stability_bonus", False))
        same_priority_bonus = bool(cfg.extra.get("same_priority_bonus", False))
        per_solve_time_limit_s = float(cfg.extra.get("per_solve_time_limit_s", 30.0))
        overall_time_budget_s = float(cfg.extra.get("overall_time_budget_s", 120.0))
        max_priorities = int(cfg.extra.get("max_priorities", 20))
        max_binaries = int(cfg.extra.get("max_binaries", 250_000))
        max_t_attempts = int(cfg.extra.get("max_t_attempts", 40))

        all_metrics: List[Dict[str, float]] = []

        for seed in range(n_seeds):
            if stop_event.is_set():
                break

            env = problem_factory()
            env.config.seed = seed
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
                f"ParrenoTorresAlvarezValdesRuiz2019IPS6 seed={seed+1}/{n_seeds} "
                f"transitive_move_breaking={transitive_move_breaking}"
            )

            t0 = _time.perf_counter()
            result = solve_ips6(
                stacks_init,
                max_tiers=max_tiers,
                transitive_move_breaking=transitive_move_breaking,
                fix_last_segment_vars=fix_last_segment_vars,
                continuous_move_vars=continuous_move_vars,
                balance_kappa=balance_kappa,
                no_empty_or_full_stacks=no_empty_or_full_stacks,
                stability_bonus=stability_bonus,
                same_priority_bonus=same_priority_bonus,
                per_solve_time_limit_s=per_solve_time_limit_s,
                overall_time_budget_s=overall_time_budget_s,
                max_priorities=max_priorities,
                max_binaries=max_binaries,
                max_t_attempts=max_t_attempts,
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
                "lower_bound": float(result.lower_bound or 0),
                "n_time_points_used": float(result.n_time_points_used or 0),
                "n_priorities": float(result.n_priorities),
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
            "num_eval_seeds": {
                "type": "int", "default": 5, "min": 1, "max": 100,
                "label": "Evaluation seeds",
                "help": "Each seed solves its own MIP(s); keep small since this is an exact method.",
            },
            "transitive_move_breaking": {
                "type": "bool", "default": True,
                "label": "Eq. (35)/(36) transitive-move + same-priority symmetry breaking",
                "help": "Forbids immediately undoing/re-doing a move on the same stack across consecutive segments.",
            },
            "fix_last_segment_vars": {
                "type": "bool", "default": True,
                "label": "Eq. (37)/(38) last-segment variable fixing",
                "help": "The top-priority group can never be the subject of the very last move.",
            },
            "continuous_move_vars": {
                "type": "bool", "default": False,
                "label": "Relax w/z to continuous (Proposition 1)",
                "help": "Provably loses no integrality, but the paper keeps them binary since it helps Gurobi detect infeasible T faster during the iterative search.",
            },
            "balance_kappa": {
                "type": "int", "default": 0, "min": 0, "max": 50,
                "label": "Section 8.1 balance cap (0 = off)",
                "help": "Caps the container-count difference between adjacent stacks in the final layout.",
            },
            "no_empty_or_full_stacks": {
                "type": "bool", "default": False,
                "label": "Section 8.2: forbid empty/full stacks in final layout",
            },
            "stability_bonus": {
                "type": "bool", "default": False,
                "label": "Section 8.3: prefer balanced final-layout heights",
                "help": "Adds a < 1 objective term; never changes the optimal move count.",
            },
            "same_priority_bonus": {
                "type": "bool", "default": False,
                "label": "Section 8.4: prefer stacking same-priority containers together",
                "help": "Adds a < 1 objective term; never changes the optimal move count.",
            },
            "per_solve_time_limit_s": {
                "type": "float", "default": 30.0, "min": 1.0, "max": 3600.0,
                "label": "Gurobi time limit per T attempt (s)",
            },
            "overall_time_budget_s": {
                "type": "float", "default": 120.0, "min": 1.0, "max": 86400.0,
                "label": "Overall wall-clock budget per seed (s)",
            },
            "max_priorities": {
                "type": "int", "default": 20, "min": 1, "max": 200,
                "label": "Max priority groups before bucket-coarsening",
                "help": "Instances with more distinct priorities are bucketed to keep the model tractable.",
            },
            "max_binaries": {
                "type": "int", "default": 250_000, "min": 1000, "max": 20_000_000,
                "label": "Binary-variable cap per T attempt",
                "help": "T candidates whose estimated model size exceeds this are skipped instead of hanging the solver.",
            },
            "max_t_attempts": {
                "type": "int", "default": 40, "min": 1, "max": 200,
                "label": "Max T-search attempts",
                "help": "Upper bound on how many ascending T values to try before giving up.",
            },
        })
        return base
