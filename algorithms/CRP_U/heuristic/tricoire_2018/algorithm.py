"""
Tricoire, Parragh, Feillet (2018) — Heuristics for CRP-U (uBRP).

Algorithm summary
-----------------
The paper proposes a family of greedy heuristics and metaheuristics for the
unrestricted Block Relocation Problem (CRP-U), all based on two "safe move"
primitives:

  SM-1   (safe 1-relocate) — place the top item of a misplaced stack onto a
         destination whose minimum is ≥ the item, so no new conflict is created.
  SM-2   (safe 2-relocate) — a two-move sequence: first safely relocate one item
         to unblock a stack-top so that a safe 1-relocate becomes possible.

Greedy heuristics (mode options)
---------------------------------
  SM-1      — apply best safe 1-relocate at each step; forced move otherwise.
  SM-2      — apply best safe 1- or safe 2-relocate; forced move otherwise.
  SmSEQ-1   — if the blocking items form a decreasing sequence (≥ 2 items),
               prepare a target stack for the whole sequence; else SM-1.
  SmSEQ-2   — same but falls back to SM-2 instead of SM-1.

Condensation post-processor (applied to SmSEQ solutions)
----------------------------------------------------------
  classic  — Jin et al. condensation: merge (s1→s2, s2→s3) if s3 is untouched.
  improved — Tricoire improved condensation: allows s3 to receive/lose items
             between the two relocations, subject to capacity constraints.

Metaheuristics
--------------
  RakeSearch   — BFS over partial solutions up to width W; apply fast-meta
                 (all four greedy heuristics) on each leaf.
  PilotMethod  — iteratively expand the best candidate using all valid single
                 relocations, evaluate each with RakeSearch(w=hub_width).

Reference
---------
F. Tricoire, S. Parragh, D. Feillet,
"New insights on the block relocation problem",
Computers & Operations Research 89 (2018) 127–139.
https://doi.org/10.1016/j.cor.2017.08.009

C++ codebase: https://github.com/ftricoire/block-relocation-master

Platform configuration (extra parameters)
------------------------------------------
mode          : str   'SmSEQ-2' (default) | 'SM-1' | 'SM-2' | 'SmSEQ-1'
                      | 'RakeSearch' | 'PilotMethod'
condensation  : str   'improved' (default, Tricoire) | 'classic' (Jin) | 'none'
                      Applied only to SmSEQ modes and as completion step in
                      RakeSearch / PilotMethod.
width         : int   100 (default). BFS width for RakeSearch; outer width for
                      PilotMethod.  Larger → better quality, slower.
hub_width     : int   2 (default). RakeSearch width used as look-ahead hub
                      inside PilotMethod.
verbose       : bool  False (default). Print per-seed relocation counts.

Correspondence to the original C++ command-line flags (README.txt)
------------------------------------------------------------------
  -m SM-1 / SM-2 / SmSEQ-1 / SmSEQ-2   ←→  mode = '...'
  -m RS-<N>                              ←→  mode = 'RakeSearch', width = N
  -m PM-<N>                             ←→  mode = 'PilotMethod', width = N
  -hub RS-<N>                           ←→  hub_width = N
  -cp none / jin / tricoire             ←→  condensation = 'none' / 'classic' / 'improved'
  -m DFBB / BB                          ←→  (not implemented; platform has stronger exact solvers)

Excluded from this port (deliberately)
----------------------------------------
  Exact algorithms  – DFBB, Branch-and-Bound (dfbb.cpp / branchandbound.cpp).
                      The platform already provides stronger exact solvers at
                      CRP_U/tree/ (Tanaka & Mizuno 2018, Jin & Tanaka 2023).
  Baselines         – GLAH (→ CRP_U/heuristic/glah/),
                      LA-N  (→ CRP_U/heuristic/lan_*/),
                      JZW, ZHU (not in Platform_CRISP scope).
  SubsequencePolicy – plain 'SSEQ' mode; superseded by SmSEQ variants which
                      are used in all main experiments and FastMeta.
"""

from __future__ import annotations

import copy
import multiprocessing as mp
from typing import Callable, Dict, List, Optional

import numpy as np

from core.base_algorithm import BaseAlgorithm, AlgorithmConfig
from core.objectives import KinematicsModel, lower_bound_relocations

from .state import BRPState
from .heuristics import solve_brp


class TricoireHeuristic(BaseAlgorithm):
    """
    Tricoire, Parragh & Feillet (2018) heuristics / metaheuristics for CRP-U.

    Provides six variants via the ``mode`` extra parameter:

    ===========================  =============================================
    mode                         Description
    ===========================  =============================================
    ``SM-1``                     Safe 1-relocates only (fastest)
    ``SM-2``                     Safe 1- or 2-relocates
    ``SmSEQ-1``                  Decreasing-sequence + SM-1 + condensation
    ``SmSEQ-2`` (default)        Decreasing-sequence + SM-2 + condensation
    ``RakeSearch``               BFS metaheuristic with fast-meta completion
    ``PilotMethod``              Look-ahead metaheuristic (slow in Python)
    ===========================  =============================================

    Additional options
    ------------------
    condensation : 'improved' (default) | 'classic' | 'none'
        Post-processing condensation applied to SmSEQ and RakeSearch / PM
        completion results.
    width : int (default 100)
        BFS width for RakeSearch; outer-loop width for PilotMethod.
    hub_width : int (default 2)
        RakeSearch width used as look-ahead hub inside PilotMethod.
    verbose : bool (default False)
        Print per-seed relocation counts.
    """

    name                = "Tricoire et al. (2018) Heuristics"
    category            = "Heuristic"
    description         = (
        "[single-bay origin]  "
        "Tricoire, Parragh, Feillet (COR 2018): greedy safe-relocation heuristics "
        "for CRP-U.  Modes: SM-1/2 (safe-relocate only), SmSEQ-1/2 (decreasing "
        "sequences + condensation), RakeSearch BFS metaheuristic, Pilot Method. "
        "DOI: 10.1016/j.cor.2017.08.009"
    )
    compatible_problems = ["CRP-U"]
    def __init__(self, config: Optional[AlgorithmConfig] = None):
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
        cfg         = self.config
        n_seeds = 1  # multi-seed eval removed; single run only
        mode        = str(cfg.extra.get("mode",         "SmSEQ-2"))
        cond_mode   = str(cfg.extra.get("condensation", "improved"))
        width       = int(cfg.extra.get("width",        100))
        hub_width   = int(cfg.extra.get("hub_width",    2))
        verbose     = bool(cfg.extra.get("verbose",     False))

        all_metrics: List[Dict] = []

        for seed_idx in range(n_seeds):
            if stop_event.is_set():
                break

            env = problem_factory()
            env.config.seed = cfg.seed + seed_idx
            env.reset()

            kin          = KinematicsModel.from_config_extra(env.config.extra)
            initial_yard = copy.deepcopy(env.yard)
            lb_val       = lower_bound_relocations(initial_yard)

            # ── Extract yard geometry ───────────────────────────────── #
            all_keys = sorted(initial_yard.stacks.keys())
            W        = len(all_keys)
            H_max    = initial_yard.max_tiers

            # ── Build BRPState ──────────────────────────────────────── #
            init_state = BRPState.from_yard(initial_yard, all_keys, H_max)

            # ── Run selected algorithm ──────────────────────────────── #
            result_state = solve_brp(
                init_state,
                mode        = mode,
                width       = width,
                hub_width   = hub_width,
                condensation = cond_mode,
            )

            relocs = result_state.n_relocations
            if verbose:
                print(
                    f"[Tricoire2018] seed={seed_idx}  mode={mode}  "
                    f"relocations={relocs}  lb={lb_val}",
                    flush=True,
                )

            # ── Convert ops to env actions ──────────────────────────── #
            action_list = _ops_to_actions(result_state.ops, W)

            # ── Simulate through env to get standard metrics ─────────── #
            metrics = _simulate_env(env, action_list, lb_val)
            all_metrics.append(metrics)

            primary = float(relocs)
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
                    "seed":        seed_idx,
                    "mode":        mode,
                    "relocations": relocs,
                    "lower_bound": lb_val,
                    "condensation": cond_mode,
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
                "default": "SmSEQ-2",
                "label":   "Algorithm mode",
                "help":    (
                    "SM-1: safe 1-relocates only.  "
                    "SM-2: safe 1- or 2-relocates.  "
                    "SmSEQ-1: decreasing sequences + SM-1 + condensation.  "
                    "SmSEQ-2: decreasing sequences + SM-2 + condensation (best greedy).  "
                    "RakeSearch: BFS metaheuristic (best overall).  "
                    "PilotMethod: look-ahead metaheuristic (very slow in Python)."
                ),
            },
            "condensation": {
                "type":    "str",
                "default": "improved",
                "label":   "Condensation post-processor",
                "help":    (
                    "'improved' — Tricoire (2018) condensation (default, best).  "
                    "'classic'  — Jin et al. condensation.  "
                    "'none'     — no condensation."
                ),
            },
            "width": {
                "type":    "int",
                "default": 100,
                "min":     2,
                "max":     5000,
                "label":   "BFS width (RakeSearch / PilotMethod)",
                "help":    "Width threshold for Rake Search and Pilot Method.",
            },
            "hub_width": {
                "type":    "int",
                "default": 2,
                "min":     1,
                "max":     50,
                "label":   "Hub width (PilotMethod)",
                "help":    "RakeSearch width used as look-ahead inside PilotMethod.",
            },
            "verbose": {
                "type":    "bool",
                "default": False,
                "label":   "Verbose logging",
                "help":    "Print per-seed relocation counts.",
            },
        })
        return base


# ═══════════════════════════════════════════════════════════════════════ #
#  Private helpers                                                        #
# ═══════════════════════════════════════════════════════════════════════ #

def _ops_to_actions(ops: list, W: int) -> List[int]:
    """
    Convert the BRPState ops log to a flat list of env action integers.

    CRP-U action encoding: action = src_stack_idx * W + dst_stack_idx
    (both 0-indexed).  Retrieval ops (from_s == to_s) are omitted.
    """
    return [
        from_s * W + to_s
        for from_s, to_s in ops
        if from_s != to_s
    ]


def _simulate_env(env, action_list: List[int], lb: int) -> Dict:
    """
    Replay *action_list* through the env and collect standard metrics.
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
    total_moves = float(m.get("steps",      m.get("total_moves", n_relocs)))

    return {
        "relocations":  n_relocs,
        "crane_time":   crane_time,
        "total_moves":  total_moves,
        "lower_bound":  float(lb),
        "lb_ratio":     n_relocs / max(lb, 1),
        "time":         crane_time,
        "steps":        total_moves,
    }
