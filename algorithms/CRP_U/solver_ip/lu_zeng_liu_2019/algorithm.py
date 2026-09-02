"""
LuZengLiu2019BRP
<2019> <exact> <unrestricted> <single-bay> <CRP-U>
LB4, BRP-m3, and IS* iterative MIP solver
mode --- IS* --- LB4-only or IS* exact mode
time_limit_s --- 3600 --- Gurobi time limit (s)

------------------------------- Reference --------------------------------
C. Lu, B. Zeng, S. Liu,
"A Study on the Block Relocation Problem: Lower Bound Derivations
 and Strong Formulations",
IEEE Transactions on Automation Science and Engineering, 2019.
------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import copy
import sys
import time
import multiprocessing as mp
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from core.base_algorithm import BaseAlgorithm, AlgorithmConfig

from .state import BRPState
from .lb4 import compute_lb4, compute_all_lower_bounds
from .brp_m3 import solve_brp_m3, minmax_ub
from .is_star import run_is, run_is_star


class LuZengLiu2019BRP(BaseAlgorithm):
    """
    Lu, Zeng & Liu (2019) — LB4 + BRP-m3 / IS* for CRP-U.

    Provides the strongest known lower bound (LB4) and a new MIP
    formulation (BRP-m3) with a novel iterative solving scheme (IS*).

    Requires: Gurobi (except mode='LB4-only').
    """

    name                = "Lu, Zeng & Liu (2019) BRP-m3 / IS* [Gurobi]"
    category            = "Exact"
    requires_solver     = True
    solver_backend      = "gurobi"
    description         = (
        "[CRP-U]  "
        "Lu, Zeng, Liu (IEEE TASE 2019): LB4 (strongest lower bound via "
        "virtual-layer structures) + BRP-m3 (new MIP with block-adjacency "
        "variables) + IS* (MIP-relaxation iterative exact solver). "
        "mode='LB4-only': compute LB4 without Gurobi. "
        "mode='IS*' (default): iterative exact solver, ~1–2 MIP iterations. "
        "arXiv:1904.03347"
    )
    compatible_problems = ["CRP-U"]
    def __init__(self, config: Optional[AlgorithmConfig] = None) -> None:
        super().__init__(config)

    # ------------------------------------------------------------------ #
    # Training loop                                                        #
    # ------------------------------------------------------------------ #

    def train(
        self,
        problem_factory: Callable,
        result_queue:    mp.Queue,
        stop_event:      mp.Event,
    ) -> None:
        cfg          = self.config
        extra        = cfg.extra or {}
        n_seeds = 1  # multi-seed eval removed; single run only
        mode         = str(extra.get("mode",          "IS*"))
        time_limit_s = float(extra.get("time_limit_s", 3600.0))
        output_flag  = int(extra.get("output_flag",   0))
        verbose      = bool(extra.get("verbose",      False))

        all_metrics: List[Dict] = []

        for seed in range(n_seeds):
            if stop_event.is_set():
                break

            env = problem_factory()
            env.reset()

            initial_yard = copy.deepcopy(env.yard)
            all_keys = sorted(initial_yard.stacks.keys())
            W        = len(all_keys)
            H        = initial_yard.max_tiers
            stacks   = [
                [c.priority for c in initial_yard.stacks[k].containers]
                for k in all_keys
            ]
            n = sum(len(s) for s in stacks)

            t0 = time.perf_counter()

            # ── Compute LB4 (always) ──────────────────────────────── #
            lb4_val = compute_lb4(stacks, n, W, H)
            all_lbs = compute_all_lower_bounds(stacks, n, W, H)

            if verbose:
                print(
                    f"[LuZengLiu2019] seed={seed}  n={n} W={W} H={H}  "
                    f"LB1={all_lbs['LB1']} LB2={all_lbs['LB2']} "
                    f"LB3={all_lbs['LB3']} LB4={all_lbs['LB4']}",
                    flush=True,
                )

            if mode == "LB4-only":
                metrics = _lb4_only_metrics(all_lbs)
                self._push(
                    result_queue,
                    step     = seed + 1,
                    metric   = float(lb4_val),
                    metrics  = metrics,
                    progress = (seed + 1) / n_seeds,
                    snapshot = env.get_state_snapshot(),
                    extra    = {"seed": seed, "mode": mode, **all_lbs},
                )
                all_metrics.append(metrics)
                continue

            # ── MIP solve ─────────────────────────────────────────── #
            try:
                import gurobipy  # noqa: F401
            except ImportError:
                print(
                    "[LuZengLiu2019] ERROR: gurobipy not installed. "
                    "Use mode='LB4-only' to run without Gurobi.",
                    file=sys.stderr, flush=True,
                )
                self._push(
                    result_queue,
                    step=seed + 1, metric=float("inf"),
                    metrics={"relocations": float("inf"), "error": 1.0},
                    progress=(seed + 1) / n_seeds,
                )
                continue

            if mode == "BRP-m3":
                T_ub = minmax_ub(stacks, n, W, H)
                result = solve_brp_m3(
                    stacks, n, W, H,
                    L=lb4_val, T=max(T_ub, lb4_val),
                    time_limit_s=time_limit_s,
                    output_flag=output_flag,
                )
                n_relocs = result["obj"]
                ops = result.get("ops")
                optimal = result["optimal"]
                iters = 1

            elif mode == "IS":
                result = run_is(
                    stacks, n, W, H,
                    time_limit_s=time_limit_s,
                    output_flag=output_flag,
                )
                n_relocs = result["obj"]
                ops = result.get("ops")
                optimal = result["optimal"]
                iters = result["iterations"]

            else:  # IS*
                result = run_is_star(
                    stacks, n, W, H,
                    time_limit_s=time_limit_s,
                    output_flag=output_flag,
                )
                n_relocs = result["obj"]
                ops = result.get("ops")
                optimal = result["optimal"]
                iters = result["iterations"]

            elapsed = time.perf_counter() - t0

            # ── Convert ops to env action list ─────────────────────── #
            action_list = _ops_to_actions(ops, W) if ops else []

            # ── Simulate env for standard metrics ─────────────────── #
            metrics = _simulate_env(env, action_list, lb4_val, n_relocs, optimal,
                                    elapsed, iters, all_lbs, mode)
            all_metrics.append(metrics)

            primary = float(n_relocs) if n_relocs is not None else float("inf")
            if primary < self._best_metric:
                self._best_metric   = primary
                self._best_solution = action_list

            if verbose:
                print(
                    f"[LuZengLiu2019/{mode}] seed={seed}  "
                    f"reloc={n_relocs}  lb4={lb4_val}  opt={optimal}  "
                    f"iters={iters}  t={elapsed:.2f}s",
                    flush=True,
                )

            self._push(
                result_queue,
                step     = seed + 1,
                metric   = primary,
                metrics  = metrics,
                progress = (seed + 1) / n_seeds,
                snapshot = env.get_state_snapshot(),
                extra    = {
                    "seed":        seed,
                    "mode":        mode,
                    "relocations": n_relocs,
                    "lb4":         lb4_val,
                    "optimal":     optimal,
                    "iterations":  iters,
                    "solve_time":  round(elapsed, 4),
                    **all_lbs,
                },
            )

        if all_metrics:
            agg = {
                k: float(np.mean([m[k] for m in all_metrics if k in m]))
                for k in all_metrics[0]
            }
            self._push(
                result_queue,
                step=n_seeds, metric=self._best_metric,
                metrics=agg, progress=1.0,
            )

    # ------------------------------------------------------------------ #
    # Accessors                                                            #
    # ------------------------------------------------------------------ #

    def get_best_solution(self) -> Optional[List[int]]:
        return self._best_solution

    # ------------------------------------------------------------------ #
    # Config schema                                                        #
    # ------------------------------------------------------------------ #

    @classmethod
    def config_schema(cls) -> Dict:
        base = super().config_schema()
        base.update({
            "mode": {
                "type":    "str",
                "default": "IS*",
                "label":   "Algorithm mode",
                "help":    (
                    "'IS*': enhanced iterative exact solver (recommended). "
                    "'IS': basic iterative scheme. "
                    "'BRP-m3': single direct MIP solve (Gurobi). "
                    "'LB4-only': compute LB4 lower bound only, no MIP required."
                ),
            },
            "time_limit_s": {
                "type":    "float",
                "default": 3600.0,
                "min":     1.0,
                "max":     86400.0,
                "label":   "Time limit (s)",
                "help":    "Per-seed time budget for Gurobi / IS* iterations.",
            },
            "output_flag": {
                "type":    "int",
                "default": 0,
                "min":     0,
                "max":     1,
                "label":   "Gurobi output",
                "help":    "0 = silent; 1 = print Gurobi solve log.",
            },
            "verbose": {
                "type":    "bool",
                "default": False,
                "label":   "Verbose logging",
                "help":    "Print per-seed summary (LBs, relocations, time).",
            },
        })
        return base


# ═══════════════════════════════════════════════════════════════════════ #
#  Private helpers                                                        #
# ═══════════════════════════════════════════════════════════════════════ #

def _ops_to_actions(
    ops: List[Tuple[int, int]],
    W: int,
) -> List[int]:
    """
    Convert (from_stack, to_stack) relocation list to flat env action integers.
    CRP-U encoding: action = from_stack * W + to_stack  (0-indexed).
    """
    return [from_s * W + to_s for from_s, to_s in ops]


def _simulate_env(
    env,
    action_list: List[int],
    lb4: int,
    n_relocs: Optional[int],
    optimal: bool,
    elapsed: float,
    iters: int,
    all_lbs: Dict[str, int],
    mode: str,
) -> Dict:
    """Replay action_list through env and collect standard metrics."""
    env.reset()
    for action in action_list:
        if env.is_done():
            break
        try:
            env.step(action)
        except Exception:
            pass

    m = env.get_metrics()
    relocs   = float(m.get("relocations", n_relocs or 0))
    ct       = float(m.get("crane_time", m.get("time", 0.0)))
    total    = float(m.get("steps", m.get("total_moves", relocs)))

    return {
        "relocations":    relocs,
        "crane_time":     ct,
        "total_moves":    total,
        "lower_bound_lb4": float(lb4),
        "lower_bound_lb1": float(all_lbs.get("LB1", 0)),
        "lower_bound_lb3": float(all_lbs.get("LB3", 0)),
        "lb_ratio_lb4":   relocs / max(lb4, 1),
        "optimal_proven": 1.0 if optimal else 0.0,
        "solve_time_s":   round(elapsed, 4),
        "is_iterations":  float(iters),
        "time":           ct,
        "steps":          total,
    }


def _lb4_only_metrics(all_lbs: Dict[str, int]) -> Dict:
    """Metrics for LB4-only mode."""
    return {
        "relocations":    float(all_lbs.get("LB4", 0)),
        "crane_time":     0.0,
        "total_moves":    0.0,
        "lower_bound_lb1": float(all_lbs.get("LB1", 0)),
        "lower_bound_lb2": float(all_lbs.get("LB2", 0)),
        "lower_bound_lb3": float(all_lbs.get("LB3", 0)),
        "lower_bound_lb4": float(all_lbs.get("LB4", 0)),
        "optimal_proven": 0.0,
        "solve_time_s":   0.0,
        "is_iterations":  0.0,
        "time":           0.0,
        "steps":          0.0,
    }
