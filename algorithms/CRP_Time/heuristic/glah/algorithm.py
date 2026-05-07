"""
GLAH – Greedy Look-Ahead Heuristic for CRP-R.

Port of MainProcess.java + Probing.java (Bo Jin, 2015).
Reference: Jin, Zhu & Lim, EJOR 240 (2015) 837–847.

Architecture
------------
Phase 1: Run evaluation_heuristic alone → initial solution (IS).
Phase 2: Greedy outer loop + look-ahead tree:
         while bay not empty:
             if LB + current_relocs ≥ best → break early
             op = Lookahead.most_promising_relocation(state)
             execute op, auto-retrieve, continue
Phase 3: Record best solution found during Phase 2 (updated whenever
         evaluation_heuristic is called at a leaf node).

Platform interface
------------------
Packaged for **CRP-Time** (crane-time objective).  Same port as CRP-D / CRP-U
GLAH copies; registry keys differ so all can load.

Uses GlahLayout internally; outputs RelocationPlan via evaluate_plan().

NOTE:  Keep in sync with ``algorithms/CRP_U/heuristic/glah`` and
``algorithms/CRP_D/heuristic/glah`` or symlink.
"""

from __future__ import annotations

import copy
import multiprocessing as mp
import os
from typing import Callable, Dict, List, Optional

import numpy as np

from core.base_algorithm import BaseAlgorithm, AlgorithmConfig
from core.objectives import KinematicsModel, compute_crane_time, lower_bound_relocations

from .layout import (
    GlahLayout, GlahState,
    lower_bound, ops_to_relocation_plan,
)
from .evaluate import evaluation_heuristic
from .lookahead import Lookahead


class GLAHHeuristic(BaseAlgorithm):
    """
    Greedy Look-Ahead Heuristic (Jin, Zhu & Lim 2015) for CRP-R.

    Significantly outperforms Kim–Hong and Caserta on large instances.
    Uses a limited look-ahead tree (depth D) to choose the next relocation,
    guided by an evaluation heuristic at each leaf.
    """

    name                = "Jin (2015) GLAH (CRP-Time)"
    category            = "Heuristic"
    description         = (
        "[CRP-Time packaging] [single-bay origin]  "
        "Greedy Look-Ahead Heuristic for CRP-R (Jin, Zhu & Lim, EJOR 2015). "
        "Classifies relocations into 6 types, builds a depth-D look-ahead tree, "
        "and uses an evaluation heuristic at leaves."
    )
    compatible_problems = ["CRP-Time", "CRP-R"]
    step_label          = "Seed"

    def __init__(self, config: Optional[AlgorithmConfig] = None):
        super().__init__(config)
        self._best_plan = None

    # ---------------------------------------------------------------- #
    # Train                                                              #
    # ---------------------------------------------------------------- #

    def train(
        self,
        problem_factory: Callable,
        result_queue:    mp.Queue,
        stop_event:      mp.Event,
    ) -> None:
        cfg         = self.config
        n_seeds     = max(1, cfg.num_eval_seeds)
        rng_seed    = cfg.seed
        depth       = int(cfg.extra.get("D",        3))
        ftbg_n      = int(cfg.extra.get("n_ftbg",   5))
        nfbg_n      = int(cfg.extra.get("n_nfbg",   5))
        ftbb_n      = int(cfg.extra.get("n_ftbb",   3))
        nfbb_n      = int(cfg.extra.get("n_nfbb",   3))
        gg_n        = int(cfg.extra.get("n_gg",     1))
        gb_n        = int(cfg.extra.get("n_gb",     1))

        all_metrics: List[Dict] = []

        for seed_idx in range(n_seeds):
            if stop_event.is_set():
                break

            env = problem_factory()
            env.config.seed = rng_seed + seed_idx
            env.reset()

            kin          = KinematicsModel.from_config_extra(env.config.extra)
            initial_yard = copy.deepcopy(env.yard)
            containers   = list(env.containers)
            lb_val       = lower_bound_relocations(initial_yard)

            # Build GlahLayout from platform Yard
            glah_layout = GlahLayout.build_from_yard(initial_yard, containers)

            # ── Phase 1: evaluation heuristic → initial solution ─── #
            state = GlahState(glah_layout.copy())
            evaluation_heuristic(state)
            # best is now set inside state
            if _should_print_plan(cfg):
                p1_ops = state.best_ops or state.ops
                if p1_ops:
                    plan_p1 = ops_to_relocation_plan(p1_ops, glah_layout)
                    _print_plan_moves(plan_p1, "Phase-1 plan (after evaluation_heuristic)")

            # ── Phase 2: greedy + look-ahead ─────────────────────── #
            la    = Lookahead(depth, ftbg_n, nfbg_n, ftbb_n, nfbb_n, gg_n, gb_n)
            state2 = GlahState(glah_layout.copy())
            state2.best_ops   = state.best_ops
            state2.best_reloc = state.best_reloc

            state2.try_retrievals()

            step_i = 0
            while not state2.is_empty():
                lb_now = lower_bound(state2.inst)
                if lb_now + state2.reloc_count >= state2.best_reloc:
                    print(
                        "[GLAH] phase2 stop: LB prune  "
                        f"lb_now={lb_now}  reloc_count={state2.reloc_count}  "
                        f"best_reloc={state2.best_reloc}",
                        flush=True,
                    )
                    break
                if stop_event.is_set():
                    break

                op = la.most_promising_relocation(state2)
                if op is None:
                    print("[GLAH] phase2 stop: no relocation candidate", flush=True)
                    break

                state2.go_one_step(op)
                step_i += 1
                kind = "reloc" if op.is_relocation else "retrieve"
                print(
                    f"[GLAH] step={step_i}  {kind}  from_stack={op.from_s}  to={op.to_s}  "
                    f"prior={op.gc.priority}  reloc_count={state2.reloc_count}",
                    flush=True,
                )
                state2.try_retrievals()

            # ── Phase 3: collect best solution ───────────────────── #
            best_ops   = state2.best_ops   or state.best_ops
            best_reloc = state2.best_reloc if state2.best_ops else state.best_reloc

            if best_ops is None:
                # fallback: use whatever state2 accumulated
                best_ops   = state2.ops
                best_reloc = state2.reloc_count

            plan = ops_to_relocation_plan(best_ops, glah_layout)
            if _should_print_plan(cfg):
                _print_plan_moves(plan, "Final plan (metrics / export)")

            metrics = _plan_metrics(plan, kin, lb_val)
            all_metrics.append(metrics)
            primary = float(best_reloc)

            if primary <= self._best_metric:
                self._best_metric   = primary
                self._best_plan     = plan
                self._best_solution = _plan_to_action_list(plan, env)

            self._push(
                result_queue,
                step     = seed_idx + 1,
                metric   = primary,
                metrics  = metrics,
                progress = (seed_idx + 1) / n_seeds,
                snapshot = env.get_state_snapshot(),
                extra    = {"seed": seed_idx, "D": depth},
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
    # Accessors                                                          #
    # ---------------------------------------------------------------- #

    def get_best_solution(self) -> Optional[List[int]]:
        return self._best_solution

    def get_best_plan(self):
        return self._best_plan

    # ---------------------------------------------------------------- #
    # Config schema                                                      #
    # ---------------------------------------------------------------- #

    @classmethod
    def config_schema(cls) -> Dict:
        base = super().config_schema()
        base.update({
            "num_eval_seeds": {
                "type": "int", "default": 5, "min": 1, "max": 50,
                "label": "Evaluation seeds",
                "help": "Number of random initial layouts to evaluate.",
            },
            "D": {
                "type": "int", "default": 3, "min": 1, "max": 6,
                "label": "Look-ahead depth D",
                "help": "Depth of the look-ahead tree (D=3 is paper default; D=4 better but slower).",
            },
            "n_ftbg": {
                "type": "int", "default": 5, "min": 1, "max": 20,
                "label": "FT-BG candidates",
                "help": "Max FT-BG relocations per tree node.",
            },
            "n_nfbg": {
                "type": "int", "default": 5, "min": 1, "max": 20,
                "label": "NF-BG candidates",
            },
            "n_ftbb": {
                "type": "int", "default": 3, "min": 1, "max": 20,
                "label": "FT-BB candidates",
            },
            "n_nfbb": {
                "type": "int", "default": 3, "min": 1, "max": 20,
                "label": "NF-BB candidates",
            },
            "n_gg": {
                "type": "int", "default": 1, "min": 0, "max": 10,
                "label": "GG candidates",
            },
            "n_gb": {
                "type": "int", "default": 1, "min": 0, "max": 10,
                "label": "GB candidates",
            },
            "glah_print_plan": {
                "type": "bool", "default": False,
                "label": "Print full movement sequence",
                "help": "Print Phase-1 and final RelocationPlan step lists (or set env GLAH_PRINT_PLAN=1).",
            },
        })
        return base


# ================================================================ #
#  Private helpers                                                  #
# ================================================================ #

def _should_print_plan(cfg: AlgorithmConfig) -> bool:
    v = (cfg.extra or {}).get("glah_print_plan")
    if v is True:
        return True
    if isinstance(v, str) and v.strip().lower() in ("1", "true", "yes", "on"):
        return True
    return os.environ.get("GLAH_PRINT_PLAN", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _print_plan_moves(plan, header: str) -> None:
    """Human-readable list of Movement rows (stdout, flushed)."""
    n_m = plan.num_moves()
    n_r = plan.num_relocations()
    print(
        f"[GLAH] {header}  ({n_m} moves, {n_r} relocations)",
        flush=True,
    )
    for i, m in enumerate(plan.movements, start=1):
        print(f"  [{i:4d}]  {m!r}", flush=True)


def _plan_metrics(plan, kin: KinematicsModel, lb: int) -> Dict:
    from core.plan import RelocationPlan
    n_relocs   = plan.num_relocations()
    crane_time = compute_crane_time(plan, kin)
    return {
        "relocations": float(n_relocs),
        "crane_time":  crane_time,
        "total_moves": float(plan.num_moves()),
        "lower_bound": float(lb),
        "lb_ratio":    float(n_relocs / max(lb, 1)),
        "time":        crane_time,
        "steps":       float(plan.num_moves()),
    }


def _plan_to_action_list(plan, env) -> List[int]:
    """Convert RelocationPlan → flat action indices for CRP_R.step()."""
    num_rows = env.config.num_rows
    actions  = []
    for m in plan.movements:
        if m.to_pos is None:
            continue
        bay, row = m.to_pos
        actions.append((bay - 1) * num_rows + (row - 1))
    return actions
