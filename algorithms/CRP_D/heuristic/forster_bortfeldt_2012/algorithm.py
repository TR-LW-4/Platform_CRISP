"""
ForsterBortfeldt2012
<2012> <heuristic> <grouped> <single-bay> <CRP-D>
Compound-move tree search with lower-bound pruning
time_limit_s --- 15 --- Search time limit (s)
n_succ --- 5 --- Successor compound moves per node

------------------------------- Reference --------------------------------
F. Forster, A. Bortfeldt,
"A tree search procedure for the container relocation problem",
Computers & Operations Research 39 (2012) 299–309.
------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import multiprocessing as mp
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from core.base_algorithm import AlgorithmConfig, BaseAlgorithm

from .layout import FBLayout
from .tree_search import _count_relocations, run_tree_search


# ================================================================ #
#  Environment → FBLayout conversion                               #
# ================================================================ #

def _build_layout(env) -> Tuple[FBLayout, int]:
    """
    Build an FBLayout from a CRP-D environment.

    The environment must expose:
      env.config.num_bays, env.config.num_rows, env.config.max_tiers
      env._idx_to_stack(i)  → stack key
      env.yard.stacks       → dict[key → stack_object]
      stack_object.is_empty, stack_object.containers
      container.priority    → int group/priority (1-based, lower = first)

    Returns
    -------
    (layout, S)  where S is the number of stacks.
    """
    S = env.config.num_bays * env.config.num_rows
    H = env.config.max_tiers

    stacks: List[List[int]] = []
    for i in range(S):
        key     = env._idx_to_stack(i)
        stk_obj = env.yard.stacks.get(key)
        if stk_obj is None or stk_obj.is_empty:
            stacks.append([])
        else:
            stacks.append([c.priority for c in stk_obj.containers])

    all_groups = [g for stk in stacks for g in stk]
    G          = max(all_groups) if all_groups else 1
    n          = len(all_groups)
    ng         = min(all_groups) if all_groups else G + 1

    return FBLayout(stacks=stacks, S=S, H=H, G=G, n=n, next_group=ng), S


# ================================================================ #
#  Move list → action list conversion                              #
# ================================================================ #

def _ops_to_actions(ops, S: int) -> List[int]:
    """
    Convert the tree-search move sequence to a platform action list.

    Only relocation moves produce env.step() calls; remove moves are
    handled automatically by the environment (auto-retrieval).

    action = src_idx * S + dst_idx
    """
    return [
        op[1] * S + op[2]
        for op in ops
        if op[0] == "rel"
    ]


# ================================================================ #
#  Platform algorithm class                                         #
# ================================================================ #

class ForsterBortfeldt2012(BaseAlgorithm):
    """
    Heuristic tree search for CRP (Forster & Bortfeldt, COR 2012).

    Compatible with
    ---------------
    CRP-D : grouping instances (duplicate priorities / group precedence).

    Configuration (via config.extra)
    ---------------------------------
    time_limit_s      : wall-clock seconds per instance (default 15.0;
                        paper uses 60 s with a Java implementation).
    n_succ            : max tree successors nSucc            (default 5; paper 10).
    cm_stop_threshold : compound-move depth controller        (default 150).
    max_flg_bb        : max FLG_BB productive moves/step      (default 3).
    max_gg            : max GG productive moves/step          (default 2).

    Decoupling
    ----------
    Self-contained: depends only on core.base_algorithm and this package's
    layout.py / tree_search.py.  No imports from other platform algorithms.
    """

    name                = "Forster–Bortfeldt (2012) Tree Search"
    category            = "Heuristic"
    description         = (
        "Heuristic tree search (Forster & Bortfeldt, COR 2012). "
        "Greedy initial solution + depth-limited compound-move tree search "
        "with §5 lower-bound pruning and 6-type move classification (§4.2)."
    )
    compatible_problems = ["CRP-D"]
    def __init__(self, config: Optional[AlgorithmConfig] = None) -> None:
        super().__init__(config)

    # ------------------------------------------------------------------ #
    # Training loop                                                         #
    # ------------------------------------------------------------------ #

    def train(
        self,
        problem_factory: Callable,
        result_queue:    mp.Queue,
        stop_event:      mp.Event,
    ) -> None:
        cfg    = self.config
        extra  = cfg.extra if cfg.extra else {}

        n_seeds = 1  # multi-seed eval removed; single run only
        time_limit    = float(extra.get("time_limit_s",       15.0))
        n_succ        = int  (extra.get("n_succ",              5))
        cm_threshold  = int  (extra.get("cm_stop_threshold", 150))
        max_flg_bb    = int  (extra.get("max_flg_bb",          3))
        max_gg        = int  (extra.get("max_gg",              2))

        all_metrics: List[Dict] = []

        for seed_idx in range(n_seeds):
            if stop_event.is_set():
                break

            env = problem_factory()
            env.config.seed = cfg.seed + seed_idx
            env.reset()

            layout, S = _build_layout(env)

            if layout.is_empty():
                metrics = {"relocations": 0.0, "steps": 0.0}
                all_metrics.append(metrics)
                self._push(
                    result_queue,
                    step     = seed_idx + 1,
                    metric   = 0.0,
                    metrics  = metrics,
                    progress = (seed_idx + 1) / n_seeds,
                )
                continue

            ops, n_relocs = run_tree_search(
                layout,
                time_limit        = time_limit,
                n_succ            = n_succ,
                cm_stop_threshold = cm_threshold,
                max_flg_bb        = max_flg_bb,
                max_gg            = max_gg,
            )

            actions = _ops_to_actions(ops, S)

            # Replay on env to collect official metrics
            env.reset()
            for action in actions:
                if stop_event.is_set():
                    break
                _, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break

            metrics = env.get_metrics()
            all_metrics.append(metrics)

            primary = float(metrics.get("relocations", n_relocs))
            if primary < self._best_metric:
                self._best_metric   = primary
                self._best_solution = actions[:]

            self._push(
                result_queue,
                step     = seed_idx + 1,
                metric   = primary,
                metrics  = metrics,
                progress = (seed_idx + 1) / n_seeds,
                snapshot = env.get_state_snapshot(),
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
    # Best solution                                                          #
    # ------------------------------------------------------------------ #

    def get_best_solution(self) -> Optional[List[int]]:
        return self._best_solution

    # ------------------------------------------------------------------ #
    # GUI configuration schema                                              #
    # ------------------------------------------------------------------ #

    @classmethod
    def config_schema(cls) -> Dict:
        base = super().config_schema()
        base.update({
            "time_limit_s": {
                "type": "float", "default": 15.0, "min": 1.0, "max": 300.0,
                "label": "Time limit per instance (s)",
                "help": (
                    "Wall-clock seconds allowed per instance. "
                    "Paper uses 60 s (Java). 15 s is recommended for Python."
                ),
            },
            "n_succ": {
                "type": "int", "default": 5, "min": 1, "max": 20,
                "label": "Max successors (nSucc)",
                "help": "Branching factor of the tree search. Paper value: 10.",
            },
            "cm_stop_threshold": {
                "type": "int", "default": 150, "min": 10, "max": 1000,
                "label": "cmStopThreshold",
                "help": (
                    "Controls compound-move depth. "
                    "Extension stops when cmStopValue ≥ threshold. "
                    "Paper default: 150."
                ),
            },
            "max_flg_bb": {
                "type": "int", "default": 3, "min": 1, "max": 10,
                "label": "Max FLG_BB moves",
                "help": "Max FLG_BB productive moves considered per step. Paper: 3.",
            },
            "max_gg": {
                "type": "int", "default": 2, "min": 0, "max": 10,
                "label": "Max GG moves",
                "help": "Max GG productive moves considered per step. Paper: 2.",
            },
        })
        return base
