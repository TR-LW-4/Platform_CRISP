"""
Wang2026GRASP
<2026> <heuristic> <stowage> <multi-bay> <CRP-Stow>
GRASP with LNS for POCRP with rolled containers

------------------------------- Reference --------------------------------
N. Wang, X. Ma, R. Yang, J. Hu,
"Optimization of partially ordered container relocation problem with
 rolled containers",
Computers & Operations Research 191 (2026) 107434.
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

from core.base_algorithm import BaseAlgorithm, AlgorithmConfig

from .construction import Move, run_greedy
from .lns import LNSParams, run_lns


# ================================================================ #
#  GRASP framework (Algorithm 2)                                   #
# ================================================================ #

def _run_grasp(
    env_factory:     Callable,
    max_iteration:   int,
    hood_num:        int,
    params:          LNSParams,
    seed:            int,
    stop_event       = None,
    progress_cb      = None,
) -> Tuple[List[int], Dict]:
    """One full GRASP run on ONE instance (single seed of env_factory)."""
    rng = np.random.RandomState(seed)
    best_actions: Optional[List[int]] = None
    best_metrics: Optional[Dict]      = None
    best_relocs                       = float("inf")

    for it in range(max_iteration):
        if stop_event is not None and stop_event.is_set():
            break

        env = env_factory()
        init_actions, init_records, _ = run_greedy(env, rng=rng)

        # Local search hood_num times over this initial solution.
        curr_actions = init_actions
        curr_records = init_records
        curr_relocs  = sum(1 for r in curr_records if r.kind == "low")

        for _ in range(hood_num):
            if stop_event is not None and stop_event.is_set():
                break
            new_actions, new_records, new_relocs = run_lns(
                env_factory, curr_actions, curr_records, rng, params,
            )
            if new_relocs < curr_relocs:
                curr_actions = new_actions
                curr_records = new_records
                curr_relocs  = new_relocs

        if curr_relocs < best_relocs:
            best_relocs  = curr_relocs
            best_actions = curr_actions
            # Evaluate to get full metrics dict.
            env2 = env_factory()
            best_metrics = env2.evaluate(curr_actions)

        if progress_cb is not None:
            progress_cb(it + 1, best_relocs, best_metrics)

    return best_actions or [], best_metrics or {}


# ================================================================ #
#  Platform algorithm class                                        #
# ================================================================ #

class Wang2026GRASP(BaseAlgorithm):
    """
    Wang, Ma, Yang, & Hu (2026) GRASP for POCRP-RC (Blocks Relocation Problem
    with Stowage Plan **and** Rolled Containers).

    Set ``env.config.rc_ratio > 0`` (via ProblemConfig / GUI) to activate
    the Rolled-Container extension the paper is designed for.  ``rc_ratio=0``
    degenerates to plain POCRP (Jovanović 2019 setting) — GRASP still works
    but its RC-specific heuristics have no leverage there.
    """

    name        = "Wang (2026) GRASP"
    category    = "Heuristic"
    description = (
        "Wang, Ma, Yang, & Hu (2026, C&OR 191:107434) GRASP for POCRP-RC. "
        "Constructive Greedy with TR (min NBC → max DC → min BC+RCB) and "
        "RC-aware RR (Task 1 for RCs, Task 2 for OCs), followed by LNS "
        "over four poor-move criteria (CMM, CMB, poor-target 2.1 and the "
        "paper's original 2.2). Requires CRP-Stow with rc_ratio>0 to "
        "exercise its RC-specific heuristics."
    )
    compatible_problems = ["CRP-Stow"]
    def __init__(self, config: Optional[AlgorithmConfig] = None):
        super().__init__(config)

    # ---------------------------------------------------------------- #
    # Config schema                                                    #
    # ---------------------------------------------------------------- #

    @classmethod
    def config_schema(cls) -> Dict:
        base = super().config_schema()
        base.update({
            "max_iteration": {
                "type": "int", "default": 10, "min": 1, "max": 500,
                "label": "GRASP outer iterations",
                "help":  "Paper §6.3.2 tuned value: 10 (marginal gain beyond).",
            },
            "hood_num": {
                "type": "int", "default": 10, "min": 1, "max": 200,
                "label": "LNS neighbors per iteration",
                "help":  "Paper §6.3.2 tuned value: 10.",
            },
            "noimprlimit": {
                "type": "int", "default": 3, "min": 1, "max": 20,
                "label": "LNS no-improve limit",
                "help":  "Paper §6.3.3 tuned value: 3.",
            },
            "Ma": {
                "type": "int", "default": 2, "min": 1, "max": 20,
                "label": "M_a (CMM threshold)",
                "help":  "First relocation of a container relocated ≥ M_a times is poor.",
            },
            "Mb": {
                "type": "int", "default": 2, "min": 1, "max": 20,
                "label": "M_b (CMB threshold)",
            },
            "Mc": {
                "type": "int", "default": 3, "min": 1, "max": 20,
                "label": "M_c (poor-target threshold)",
                "help":  "Target requiring > M_c blocker moves is a poor target.",
            },
        })
        return base

    def _make_params(self) -> Tuple[int, int, LNSParams]:
        extra = getattr(self.config, "extra", {}) or {}

        def _get(key: str, default):
            if key in extra:
                return type(default)(extra[key])
            return default

        max_iter  = _get("max_iteration", 10)
        hood_num  = _get("hood_num", 10)
        params = LNSParams(
            Ma           = _get("Ma", 2),
            Mb           = _get("Mb", 2),
            Mc           = _get("Mc", 3),
            noimprlimit  = _get("noimprlimit", 3),
            init_fitness = float(_get("init_fitness", 1.0)),
            fit_alpha    = float(_get("fit_alpha", 1.5)),
            init_temp    = float(_get("init_temp", 10.0)),
            final_temp   = float(_get("final_temp", 0.1)),
            cooling      = float(_get("cooling", 0.95)),
            boltzmann    = float(_get("boltzmann", 1.0)),
        )
        return max_iter, hood_num, params

    # ---------------------------------------------------------------- #
    # Training loop                                                    #
    # ---------------------------------------------------------------- #

    def train(
        self,
        problem_factory: Callable,
        result_queue:    mp.Queue,
        stop_event:      mp.Event,
    ) -> None:
        cfg     = self.config
        n_seeds = 1  # multi-seed eval removed; single run only
        max_iter, hood_num, params = self._make_params()
        all_metrics: List[Dict] = []

        for seed in range(n_seeds):
            if stop_event.is_set():
                break
            base_seed = cfg.seed + seed

            def factory(bs=base_seed):
                env = problem_factory()
                env.config.seed = bs
                return env

            sol, metrics = _run_grasp(
                factory,
                max_iteration = max_iter,
                hood_num      = hood_num,
                params        = params,
                seed          = base_seed,
                stop_event    = stop_event,
            )

            if metrics:
                all_metrics.append(metrics)
                primary = float(metrics.get("relocations", 0.0))
                if primary < self._best_metric:
                    self._best_metric   = primary
                    self._best_solution = list(sol)

                env_snap = factory()
                env_snap.reset()
                self._push(
                    result_queue,
                    step     = seed + 1,
                    metric   = primary,
                    metrics  = metrics,
                    progress = (seed + 1) / n_seeds,
                    snapshot = env_snap.get_state_snapshot(),
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
