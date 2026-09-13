"""
FeillletLS
<2019> <heuristic> <unrestricted> <single-bay> <CRP-U>
OPT(n) dynamic-programming local search for uBRP
max_iter --- 0 --- Outer iterations (0 = until convergence)
init --- greedy --- Initial solution: greedy or given

------------------------------- Reference --------------------------------
D. Feillet, S.N. Parragh, F. Tricoire,
"A local-search based heuristic for the unrestricted block relocation problem",
Computers & Operations Research 108 (2019) 44–56.
------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import copy
import multiprocessing as mp
from typing import Callable, Dict, List, Optional

import numpy as np

from core.base_algorithm import BaseAlgorithm, AlgorithmConfig
from core.objectives import KinematicsModel, compute_crane_time, lower_bound_relocations

from .solution import Step, build_greedy
from .opt_operator import opt_n


class FeillletLS(BaseAlgorithm):
    """
    Feillet, Parragh & Tricoire (2019) local-search heuristic for CRP-U.

    Uses OPT(n) — a DP operator on the reduced state-space graph S^−n — to
    iteratively improve the relocation sequence of each container.

    Compatible with CRP-U (unrestricted BRP, fixed retrieval order).

    Configuration extras
    --------------------
    max_iter  (int, default 0)  Max LS passes; 0 = run until convergence.
    init      (str, default "greedy")  Initial solution source:
                "greedy" — built-in MinMax greedy (no external coupling)
                "glah"   — warm-start from GLAH Phase-1 evaluation heuristic
    verbose   (bool, default False)  Print per-container improvement log.
    """

    name                = "Feillet et al. (2019) LS"
    category            = "Heuristic"
    description         = (
        "[single-bay origin]  "
        "Feillet, Parragh, Tricoire (COR 2019) iterative local search for CRP-U. "
        "OPT(n) operator: DP shortest-path on the S^−n state-space graph reoptimises "
        "the relocation sub-sequence of each container n in turn until convergence. "
        "Self-contained: initial solution from an internal MinMax greedy "
        "(no GLAH coupling required). "
        "DOI: 10.1016/j.cor.2019.04.005"
    )
    compatible_problems = ["CRP-U"]
    geometry            = "single-bay"
    objectives          = ["relocations"]
    fidelity            = "faithful"
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
        cfg      = self.config
        n_seeds = 1  # multi-seed eval removed; single run only
        max_iter = int(cfg.extra.get("max_iter", 0))
        init_src = str(cfg.extra.get("init",     "greedy")).lower()
        verbose  = bool(cfg.extra.get("verbose",  False))

        all_metrics: List[Dict] = []

        for seed_idx in range(n_seeds):
            if stop_event.is_set():
                break

            env = problem_factory()
            env.config.seed = cfg.seed + seed_idx
            env.reset()

            kin          = KinematicsModel.from_config_extra(env.config.extra)
            initial_yard = copy.deepcopy(env.yard)
            containers   = list(env.containers)
            lb_val       = lower_bound_relocations(initial_yard)

            # ── Extract bay geometry ─────────────────────────────────── #
            all_keys = sorted(initial_yard.stacks.keys())
            W        = len(all_keys)
            H_max    = initial_yard.max_tiers
            N        = sum(stk.height for stk in initial_yard.stacks.values())

            # init_stacks[s] = [bottom..top] priority list, 1-indexed, s=1..W
            init_stacks: List[Optional[List[int]]] = [None]
            for key in all_keys:
                init_stacks.append(
                    [c.priority for c in initial_yard.stacks[key].containers]
                )

            # ── Phase 1: initial solution ────────────────────────────── #
            if init_src == "glah":
                steps = _build_glah_initial(initial_yard, containers, W, all_keys, verbose)
                if steps is None:
                    print(
                        "[FeillletLS] GLAH init failed, falling back to greedy",
                        flush=True,
                    )
                    steps = build_greedy(init_stacks, W, H_max, N)
            else:
                steps = build_greedy(init_stacks, W, H_max, N)

            relocs_init = sum(1 for s in steps if s.dst != 0)
            if verbose:
                print(
                    f"[FeillletLS] seed={seed_idx}  init_relocs={relocs_init}  "
                    f"lb={lb_val}  source={init_src}",
                    flush=True,
                )

            # ── Phase 2: local search ────────────────────────────────── #
            it = 0
            while not stop_event.is_set():
                improved = False
                for n in range(1, N + 1):
                    new_steps = opt_n(n, steps, init_stacks, W, H_max, N)
                    if new_steps is not None:
                        old_count = sum(1 for s in steps    if s.dst != 0)
                        new_count = sum(1 for s in new_steps if s.dst != 0)
                        if verbose:
                            print(
                                f"[FeillletLS] seed={seed_idx}  iter={it}  n={n}  "
                                f"relocs {old_count}→{new_count}",
                                flush=True,
                            )
                        steps    = new_steps
                        improved = True

                it += 1
                if not improved:
                    break
                if max_iter > 0 and it >= max_iter:
                    break

            relocs_final = sum(1 for s in steps if s.dst != 0)
            print(
                f"[FeillletLS] seed={seed_idx}  iters={it}  "
                f"init_relocs={relocs_init}  final_relocs={relocs_final}  lb={lb_val}",
                flush=True,
            )

            # ── Convert to action list for env ───────────────────────── #
            action_list = _steps_to_actions(steps, W)

            # Build RelocationPlan-compatible metrics by simulating env
            metrics, plan = _simulate_env(env, action_list, kin, lb_val)
            all_metrics.append(metrics)
            primary = float(relocs_final)

            if primary < self._best_metric:
                self._best_metric   = primary
                self._best_solution = action_list

            self._push(
                result_queue,
                step     = seed_idx + 1,
                metric   = primary,
                metrics  = metrics,
                progress = (seed_idx + 1) / n_seeds,
                snapshot = env.get_state_snapshot(),
                extra    = {
                    "seed":         seed_idx,
                    "ls_iters":     it,
                    "init_relocs":  relocs_init,
                    "final_relocs": relocs_final,
                    "lower_bound":  lb_val,
                },
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

    # ---------------------------------------------------------------- #
    # Config schema                                                      #
    # ---------------------------------------------------------------- #

    @classmethod
    def config_schema(cls) -> Dict:
        base = super().config_schema()
        base.update({
            "max_iter": {
                "type":    "int",
                "default": 0,
                "min":     0,
                "max":     100,
                "label":   "Max LS iterations (0 = until convergence)",
                "help":    "0 means run until no container can be improved.",
            },
            "init": {
                "type":    "str",
                "default": "greedy",
                "label":   "Initial solution",
                "help":    (
                    "'greedy' — internal MinMax greedy (fully decoupled, fast). "
                    "'glah'   — GLAH evaluation heuristic Phase-1 warm-start."
                ),
            },
            "verbose": {
                "type":    "bool",
                "default": False,
                "label":   "Verbose logging",
                "help":    "Print per-container improvement events.",
            },
        })
        return base


# ═══════════════════════════════════════════════════════════════════ #
#  Private helpers                                                    #
# ═══════════════════════════════════════════════════════════════════ #

def _steps_to_actions(steps: List[Step], W: int) -> List[int]:
    """
    Convert a List[Step] to a flat list of env action integers.

    In CRP-U: action = src_i * W + dst_i  (both 0-indexed).
    Retrieval steps (dst == 0) are omitted; the env handles them automatically.
    """
    actions: List[int] = []
    for step in steps:
        if step.dst == 0:
            continue
        src_i = step.src - 1   # 1-indexed → 0-indexed
        dst_i = step.dst - 1
        actions.append(src_i * W + dst_i)
    return actions


def _simulate_env(
    env,
    action_list: List[int],
    kin:         KinematicsModel,
    lb:          int,
) -> tuple:
    """
    Replay the action list through the env to compute standard metrics.
    Returns (metrics_dict, None).
    """
    env.reset()
    for action in action_list:
        if env.is_done():
            break
        try:
            env.step(action)
        except Exception:
            pass

    m          = env.get_metrics()
    n_relocs   = float(m.get("relocations", 0))
    crane_time = float(m.get("crane_time",  m.get("time", 0.0)))
    total_moves = float(m.get("steps", m.get("total_moves", n_relocs)))

    metrics = {
        "relocations": n_relocs,
        "crane_time":  crane_time,
        "total_moves": total_moves,
        "lower_bound": float(lb),
        "lb_ratio":    n_relocs / max(lb, 1),
        "time":        crane_time,
        "steps":       total_moves,
    }
    return metrics, None


def _build_glah_initial(
    initial_yard,
    containers: list,
    W:          int,
    all_keys:   list,
    verbose:    bool,
) -> Optional[List[Step]]:
    """
    Optional GLAH warm-start (only imported when init='glah').

    Runs GLAH's evaluation_heuristic (Phase 1) and converts the resulting
    GlahOp list to a List[Step].  Returns None on any failure.
    """
    try:
        from algorithms.CRP_U.heuristic.glah.layout import GlahLayout, GlahState  # type: ignore
        from algorithms.CRP_U.heuristic.glah.evaluate import evaluation_heuristic   # type: ignore
    except ImportError:
        return None

    key_to_1based: Dict[tuple, int] = {k: (i + 1) for i, k in enumerate(all_keys)}

    glah_layout = GlahLayout.build_from_yard(initial_yard, containers)
    state       = GlahState(glah_layout.copy())
    evaluation_heuristic(state)

    ops = state.best_ops or state.ops
    if not ops:
        return None

    steps: List[Step] = []
    for op in ops:
        src_s = key_to_1based.get(glah_layout._s_to_key.get(op.from_s, ()), 0)
        dst_s = key_to_1based.get(glah_layout._s_to_key.get(op.to_s,   ()), 0) if op.to_s != 0 else 0
        if src_s == 0:
            continue
        steps.append(Step(op.gc.priority, src_s, dst_s))

    if verbose:
        n_rel = sum(1 for s in steps if s.dst != 0)
        print(f"[FeillletLS] GLAH Phase-1 initial solution: {n_rel} relocations", flush=True)

    return steps or None
