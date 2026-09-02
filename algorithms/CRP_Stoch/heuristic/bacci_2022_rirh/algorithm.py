"""
RIRH
<2022> <heuristic> <stochastic> <single-bay> <CRP-Stoch>
Realization-independent reallocation and Expected MinMax
num_sample_realizations --- 0 --- Sampled realizations (0 = full tree)

------------------------------- Reference --------------------------------
T. Bacci, S. Mattia, P. Ventura,
"The realization-independent reallocation heuristic for the stochastic
 container relocation problem",
Soft Computing, 2022.
------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import multiprocessing as mp
import time
from typing import Callable, Dict, List, Optional

from core.base_algorithm import AlgorithmConfig, BaseAlgorithm
from core.layout_trace import trace_layout

from .rirh_policy import build_em_policy_tree, build_rirh_policy_tree


# ────────────────────────────────────────────────────────────────────
#  Base helper: run a policy-tree builder on one CRP-Stoch environment
# ────────────────────────────────────────────────────────────────────


def _run_once(env, builder) -> Dict[str, float]:
    """
    Build a policy tree with ``builder`` on ``env`` and evaluate it.

    ``env`` is expected to be a fresh (already ``reset``) ``CRP_Stoch``
    instance.  Returns the metrics dict from ``evaluate_policy``.
    """
    stacks = env.yard_snapshot_priorities()
    batches = env.get_batches()
    max_tiers = int(env.config.max_tiers)
    n_total = int(env.config.num_containers)

    t0 = time.perf_counter()
    tree = builder(
        initial_stacks=stacks,
        batches=batches,
        max_tiers=max_tiers,
        n_containers=n_total,
    )
    build_time = time.perf_counter() - t0

    mc_samples = int(env.config.extra.get("num_sample_realizations", 0) or 0)
    metrics = env.evaluate_policy(
        tree,
        num_sample_realizations=mc_samples,
        rng_seed=int(env.config.seed),
    )
    metrics["build_time"] = float(build_time)
    metrics["num_leaves"] = float(_count_leaves(tree))
    return metrics


def _count_leaves(tree) -> int:
    """Count leaf DecisionNodes (accounting for shared-object reuse)."""
    from core.stoch import ChanceNode, DecisionNode

    if tree is None or not tree.outcomes:
        return 0
    seen = set()

    def _walk(cn: ChanceNode) -> None:
        for dn in cn.outcomes.values():
            if id(dn) in seen:
                continue
            seen.add(id(dn))
            if dn.next_chance is None:
                # terminal — count it
                pass
            else:
                _walk(dn.next_chance)

    _walk(tree)
    return len(seen)


# ────────────────────────────────────────────────────────────────────
#  RIRH
# ────────────────────────────────────────────────────────────────────


class RIRH(BaseAlgorithm):

    name = "Bacci et al. (2022) RIRH"
    category = "Heuristic"
    description = (
        "Realization-Independent Reallocation Heuristic (Bacci, Mattia & "
        "Ventura, Soft Comp 2022): builds a commitment-style policy tree "
        "for SCRP by trying a realization-independent MinMax per batch "
        "(disjoint destination columns per batch member) and falling back "
        "to per-realization Expected-MinMax when needed.  Produces a "
        "compact tree whose expected number of reshuffles E[R] often "
        "matches the paper's Table 3 / 4 within a few percent while "
        "being drastically smaller than EM."
    )
    compatible_problems = ["CRP-Stoch"]
    def __init__(self, config: Optional[AlgorithmConfig] = None):
        super().__init__(config)

    # -------------------------------------------------------------- #
    # Training loop                                                    #
    # -------------------------------------------------------------- #

    def train(
        self,
        problem_factory: Callable,
        result_queue: mp.Queue,
        stop_event: mp.Event,
    ) -> None:
        cfg = self.config
        n_seeds = 1  # multi-seed eval removed; single run only
        all_metrics: List[Dict[str, float]] = []

        for seed in range(n_seeds):
            if stop_event.is_set():
                break
            trace_layout(f"RIRH.train: seed {seed + 1}/{n_seeds}")

            env = problem_factory()
            env.reset(options={"skip_auto_retrieve": True})

            metrics = _run_once(env, build_rirh_policy_tree)
            primary = float(metrics.get("expected_relocations", float("inf")))
            metrics["metric"] = primary
            all_metrics.append(metrics)

            if primary < self._best_metric:
                self._best_metric = primary
                self._best_solution = []  # policy tree isn't a flat list

            self._push(
                result_queue,
                step=seed + 1,
                metric=primary,
                metrics=metrics,
                progress=(seed + 1) / n_seeds,
                snapshot=env.get_state_snapshot(),
            )

        if all_metrics:
            keys = set().union(*[set(m.keys()) for m in all_metrics])
            agg = {
                k: float(sum(m.get(k, 0.0) for m in all_metrics) / len(all_metrics))
                for k in keys
            }
            self._push(
                result_queue,
                step=len(all_metrics),
                metric=self._best_metric,
                metrics=agg,
                progress=1.0,
            )

    def get_best_solution(self) -> Optional[List[int]]:
        return self._best_solution

    # -------------------------------------------------------------- #
    # Config schema                                                    #
    # -------------------------------------------------------------- #

    @classmethod
    def config_schema(cls) -> Dict:
        base = super().config_schema()
        base.update(
            {
            }
        )
        return base


# ────────────────────────────────────────────────────────────────────
#  Expected MinMax baseline (Galle et al. 2018)
# ────────────────────────────────────────────────────────────────────


class ExpectedMinMax(BaseAlgorithm):

    name = "Galle et al. (2018) Expected MinMax"
    category = "Heuristic"
    description = (
        "Expected MinMax (EM) baseline for SCRP (Galle, Manshadi, Barnhart & "
        "Jaillet, Transportation Science 2018, Algorithm 1): enumerate every "
        "realization at every batch boundary and drive Caserta/Galle MinMax "
        "on each branch.  |Ω_B| leaves.  Reproduces the EM column of "
        "Bacci 2022 Tables 1–4 and serves as the RIRH counterfactual."
    )
    compatible_problems = ["CRP-Stoch"]
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
        all_metrics: List[Dict[str, float]] = []

        for seed in range(n_seeds):
            if stop_event.is_set():
                break
            trace_layout(f"ExpectedMinMax.train: seed {seed + 1}/{n_seeds}")

            env = problem_factory()
            env.reset(options={"skip_auto_retrieve": True})

            metrics = _run_once(env, build_em_policy_tree)
            primary = float(metrics.get("expected_relocations", float("inf")))
            metrics["metric"] = primary
            all_metrics.append(metrics)

            if primary < self._best_metric:
                self._best_metric = primary
                self._best_solution = []

            self._push(
                result_queue,
                step=seed + 1,
                metric=primary,
                metrics=metrics,
                progress=(seed + 1) / n_seeds,
                snapshot=env.get_state_snapshot(),
            )

        if all_metrics:
            keys = set().union(*[set(m.keys()) for m in all_metrics])
            agg = {
                k: float(sum(m.get(k, 0.0) for m in all_metrics) / len(all_metrics))
                for k in keys
            }
            self._push(
                result_queue,
                step=len(all_metrics),
                metric=self._best_metric,
                metrics=agg,
                progress=1.0,
            )

    def get_best_solution(self) -> Optional[List[int]]:
        return self._best_solution

    @classmethod
    def config_schema(cls) -> Dict:
        base = super().config_schema()
        base.update(
            {
            }
        )
        return base
