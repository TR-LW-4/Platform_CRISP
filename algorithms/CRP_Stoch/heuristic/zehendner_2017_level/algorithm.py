"""
Zehendner, Feillet & Jaillet (2017) — Online Container Relocation
Problem heuristics (leveling ``L`` + baselines ``R`` and ``M``).

Setting on the platform
-----------------------
The Online CRP with **look-ahead H = 0** (Section 2 of the paper) is
naturally captured by our SCRP with ``batch_size = 1``: every container
is its own batch, so the next-to-retrieve container is *fully revealed*
just before retrieving it — matching the online setting.  All three
policies are single-realization, ``step()``-driven online rules.

The policies still work meaningfully when ``batch_size > 1`` (each
realization is a permutation of the intra-batch priorities); they simply
ignore any lookahead information — that is the whole point of the
paper.  Multiple realizations are evaluated via Monte-Carlo sampling
whenever ``num_sample_realizations > 0`` on the env config, and the
average number of relocations is reported alongside ``expected_relocations``
computed as an average across the sampled batches.

Reference
---------
E. Zehendner, D. Feillet, P. Jaillet,
"An algorithm with performance guarantee for the Online Container
Relocation Problem", European J. of Operational Research 259 (2017)
48-62.  DOI: 10.1016/j.ejor.2016.09.011.
"""

from __future__ import annotations

import multiprocessing as mp
import time
from itertools import permutations
from typing import Callable, Dict, List, Optional

import numpy as np

from core.base_algorithm import AlgorithmConfig, BaseAlgorithm
from core.layout_trace import trace_layout
from core.plan import Movement
from core.stoch import ChanceNode, DecisionNode, expected_relocations

from .policies import (
    POLICIES,
    leveling_competitive_ratio,
    leveling_pick,
    random_pick,
    right_neighbour_pick,
)


# ────────────────────────────────────────────────────────────────────
#  Shared driver: run a policy over one realization
# ────────────────────────────────────────────────────────────────────


def _run_realization(
    initial_stacks,
    batch,
    realization,
    max_tiers,
    pick_fn,
    batch_idx,
    rng: Optional[np.random.RandomState] = None,
) -> Optional[DecisionNode]:
    """Execute one batch under ``realization`` using ``pick_fn`` as the
    online reshuffle rule.  Returns a DecisionNode (moves list ends with
    each batch member's retrieval)."""
    stacks = {k: list(v) for k, v in initial_stacks.items()}
    moves: List[Movement] = []
    for target in realization:
        # Locate target in stacks
        src = None
        for k, s in stacks.items():
            if target in s:
                src = k
                break
        if src is None:
            return None
        # Relocate blockers on top of target, one at a time
        while stacks[src] and stacks[src][-1] != target:
            blocker = stacks[src][-1]
            dst = pick_fn(
                stacks,
                max_tiers=max_tiers,
                src=src,
                blocker_priority=blocker,
                rng=rng,
            )
            if dst is None:
                return None
            stacks[src].pop()
            stacks[dst].append(blocker)
            moves.append(Movement(blocker, src, dst))
        stacks[src].pop()
        moves.append(Movement(target, src, None))
    # Propagate resulting yard state back to caller
    for k in initial_stacks.keys():
        initial_stacks[k] = stacks[k]
    return DecisionNode(
        batch_idx=batch_idx,
        realization=tuple(realization),
        moves=moves,
        next_chance=None,
    )


def _build_online_tree(
    initial_stacks,
    batches,
    max_tiers,
    pick_fn,
    rng: Optional[np.random.RandomState] = None,
) -> ChanceNode:
    """
    Build a full |Ω_B|-branch policy tree by running ``pick_fn`` on
    every intra-batch permutation.  Used to produce E[R] for the online
    heuristics under CRP-Stoch batches.

    When batch_size = 1 there is exactly one branch per batch → the tree
    degenerates to a single deterministic path, and E[R] equals the plain
    online-heuristic reshuffle count on CRP-R.
    """
    root = ChanceNode(batch_idx=0)

    def _grow(cn: ChanceNode, stacks, depth: int) -> None:
        if depth >= len(batches):
            return
        b = list(batches[depth])
        for perm in permutations(b):
            local = {k: list(v) for k, v in stacks.items()}
            dn = _run_realization(
                initial_stacks=local,
                batch=b,
                realization=perm,
                max_tiers=max_tiers,
                pick_fn=pick_fn,
                batch_idx=depth,
                rng=rng,
            )
            if dn is None:
                continue
            cn.add_outcome(perm, dn)
            if depth + 1 < len(batches):
                dn.next_chance = ChanceNode(batch_idx=depth + 1)
                _grow(dn.next_chance, local, depth + 1)

    _grow(root, {k: list(v) for k, v in initial_stacks.items()}, 0)
    return root


def _run_one(env, pick_fn, rng: np.random.RandomState) -> Dict[str, float]:
    stacks = env.yard_snapshot_priorities()
    batches = env.get_batches()
    max_tiers = int(env.config.max_tiers)
    n_total = int(env.config.num_containers)
    n_stacks = int(env.config.num_bays) * int(env.config.num_rows)

    t0 = time.perf_counter()
    tree = _build_online_tree(stacks, batches, max_tiers, pick_fn, rng=rng)
    dt = time.perf_counter() - t0

    er = float(expected_relocations(tree))
    metrics = env.evaluate_policy(
        tree,
        num_sample_realizations=int(env.config.extra.get("num_sample_realizations", 0) or 0),
        rng_seed=int(env.config.seed),
    )
    metrics["build_time"] = float(dt)
    metrics["competitive_ratio_bound"] = float(
        leveling_competitive_ratio(n_total, n_stacks)
    )
    metrics["expected_relocations"] = er
    return metrics


# ────────────────────────────────────────────────────────────────────
#  Common BaseAlgorithm shell
# ────────────────────────────────────────────────────────────────────


class _ZehendnerBase(BaseAlgorithm):

    category = "Heuristic"
    compatible_problems = ["CRP-Stoch"]
    _rule_key: str = "L"  # override in subclasses

    def __init__(self, config: Optional[AlgorithmConfig] = None):
        super().__init__(config)

    def _pick_fn(self):
        return POLICIES[self._rule_key]

    def train(
        self,
        problem_factory: Callable,
        result_queue: mp.Queue,
        stop_event: mp.Event,
    ) -> None:
        cfg = self.config
        n_seeds = 1  # multi-seed eval removed; single run only
        all_metrics: List[Dict[str, float]] = []
        pick_fn = self._pick_fn()

        for seed in range(n_seeds):
            if stop_event.is_set():
                break

            trace_layout(
                f"{self.__class__.__name__}.train: seed {seed + 1}/{n_seeds}  rule={self._rule_key}"
            )

            env = problem_factory()
            env.reset(options={"skip_auto_retrieve": True})

            rng = np.random.RandomState(seed)
            metrics = _run_one(env, pick_fn, rng=rng)
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


# ────────────────────────────────────────────────────────────────────
#  Public classes
# ────────────────────────────────────────────────────────────────────


class LevelingHeuristic(_ZehendnerBase):

    name = "Zehendner et al. (2017) Leveling L"
    description = (
        "Online CRP leveling heuristic (Zehendner, Feillet & Jaillet, EJOR 2017): "
        "relocate blockers to the lowest-height stack (leftmost on ties). "
        "Simple, feasible-when-feasible, and enjoys a proven "
        "competitive ratio 2·⌈N/W⌉ − 1 against the offline optimum "
        "(paper Theorem 1)."
    )
    _rule_key = "L"


class RandomStackHeuristic(_ZehendnerBase):

    name = "Zehendner et al. (2017) Random R"
    description = (
        "Baseline for Zehendner et al. 2017: pick a uniformly random "
        "non-full, non-source destination for the current blocker.  "
        "Used as an empirical lower reference in the paper's experiments "
        "(Tables 1 & 4)."
    )
    _rule_key = "R"


class RightNeighbourHeuristic(_ZehendnerBase):

    name = "Zehendner et al. (2017) Right-Neighbour M"
    description = (
        "Baseline for Zehendner et al. 2017: relocate the current blocker "
        "to the next stack in natural (bay, row) order (with wrap-around).  "
        "Emulates the paper's \"M\" heuristic on multi-row yards."
    )
    _rule_key = "M"
