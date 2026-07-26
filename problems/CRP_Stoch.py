"""
CRP-Stoch — Stochastic Container Relocation Problem (SCRP).

Model
-----
Extends the deterministic ``CRP-R`` environment by partitioning the N
containers into ordered **batches** ``B_1, ..., B_b``:

- Containers in B_k must be retrieved before those in B_{k+1}.
- The intra-batch retrieval order is revealed only when the LAST
  container of B_{k-1} leaves the yard.
- All permutations of a batch are assumed equiprobable (uniform
  distribution — matches Bacci, Mattia & Ventura, Soft Computing 2022;
  Galle et al. 2018; Ku & Arthanari, EJOR 2016).

The **primary metric** is the *expected* number of relocations
``E[R]`` over all batch-based realizations.

Backward compatibility
----------------------
- ``batch_size = 1`` (the default) puts every container in its own
  batch, so every retrieval order is fully known ⇒ **identical** to
  CRP-R.  All CRP-R baselines can therefore be run on CRP-Stoch as
  sanity checks without changing anything.
- Existing metric keys (``relocations``, ``steps``, ``time``,
  ``crane_time``) are preserved.  A new key ``expected_relocations``
  is added and used as the primary metric when a policy tree has been
  evaluated.

Interfaces
----------
Two evaluation entry-points are provided:

- ``step(action)`` — inherited from CRP-R; drives a **single**
  realization of the retrieval sequence.  Suitable for online-style
  heuristics such as Zehendner et al. (EJOR 2017) leveling.

- ``evaluate_policy(tree)`` — takes a :class:`core.stoch.PolicyTree`
  and returns E[R] plus feasibility.  Used by Bacci et al. (Soft
  Comp 2022) RIRH and Galle et al. (2018) EM baselines.

Reference
---------
T. Bacci, S. Mattia, P. Ventura, "The realization-independent
reallocation heuristic for the stochastic container relocation
problem", Soft Computing (2022).
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.base_problem import ProblemConfig
from core.plan import Movement, RelocationPlan
from core.stoch import (
    ChanceNode,
    DecisionNode,
    expected_relocations,
    make_batches,
    priority_to_batch_map,
)
from problems.CRP_R import CRP_R


class CRP_Stoch(CRP_R):
    """
    Stochastic CRP (SCRP) environment.

    Config knobs (via ``ProblemConfig.extra``)
    -----------------------------------------
    - ``batch_size``: int, default 1
        Nominal containers per batch.  ``1`` → deterministic (== CRP-R);
        ``num_containers`` → single-batch worst uncertainty.  When
        ``num_containers`` is not divisible by ``batch_size`` the last
        batch is smaller (only its size — the rest keep ``batch_size``).
    - ``batch_sizes``: List[int], optional
        Explicit per-batch sizes; overrides ``batch_size``.  Must sum
        to ``num_containers``.
    - ``max_enumerated_realizations``: int, default 720
        Safety cap for full enumeration (6! = 720).  Larger batches are
        handled by Monte-Carlo sampling via ``num_sample_realizations``.
    - ``num_sample_realizations``: int, default 0
        When > 0, ``evaluate_policy`` also reports a MC-averaged
        ``avg_relocations`` metric.
    """

    name = "CRP-Stoch"
    description = (
        "Stochastic Container Relocation Problem (SCRP): "
        "containers are partitioned into ordered batches whose intra-batch "
        "retrieval order is revealed only after the previous batch is empty. "
        "Primary metric: expected number of relocations E[R] (Bacci et al. 2022; "
        "Galle et al. 2018; Ku & Arthanari 2016). "
        "Set batch_size=1 for deterministic CRP-R compatibility."
    )
    tags = ["crp", "stochastic", "batches", "yard-only"]
    metric_names = [
        "expected_relocations",
        "avg_relocations",
        "relocations",
        "steps",
        "time",
        "crane_time",
    ]

    # ---------------------------------------------------------------- #
    # Episode-level state                                                #
    # ---------------------------------------------------------------- #

    def _hooks_clear_episode(self) -> None:
        # Recorded moves (for single-realization step-driven simulation).
        self._plan = RelocationPlan()
        # Batch structure — populated in _build_episode()
        self._batches: List[List[int]] = []
        self._priority_to_batch: Dict[int, int] = {}
        # Cached results from the last evaluate_policy() call.
        self._last_expected_reloc: Optional[float] = None
        self._last_avg_reloc: Optional[float] = None
        self._last_feasible: Optional[bool] = None

    def _hook_after_relocate(
        self,
        src: Tuple[int, int],
        dst: Tuple[int, int],
        container_id: int,
    ) -> None:
        self._plan.add(Movement(container_id, src, dst))

    def _hook_after_retrieve(self, bay: int, row: int, container_id: int) -> None:
        self._plan.add(Movement(container_id, (bay, row), None))

    # ---------------------------------------------------------------- #
    # Episode construction                                              #
    # ---------------------------------------------------------------- #

    def _build_episode(self) -> None:
        """
        Standard CRP-R layout + partition of {1..N} into ordered batches.
        The single-realization retrieval sequence keeps using the fixed
        1…N priorities (this is why ``batch_size=1`` gives an exact
        CRP-R match).  Algorithms that reason over multiple realizations
        should introspect ``self._batches`` and drive a PolicyTree via
        :meth:`evaluate_policy`.
        """
        super()._build_episode()

        n = int(self.config.num_containers)
        extra = self.config.extra or {}
        explicit = extra.get("batch_sizes")
        batch_size = int(extra.get("batch_size", 1) or 1)
        if batch_size < 1:
            batch_size = 1

        self._batches = make_batches(
            n_containers=n,
            batch_size=batch_size,
            explicit_sizes=explicit,
        )
        self._priority_to_batch = priority_to_batch_map(self._batches)

    # ---------------------------------------------------------------- #
    # Read-only helpers exposed to algorithms                            #
    # ---------------------------------------------------------------- #

    def get_batches(self) -> List[List[int]]:
        """Return the list of batches (each is a list of priorities)."""
        return [list(b) for b in self._batches]

    def get_batch_of(self, priority: int) -> int:
        """Return 0-indexed batch index for a given priority (raises KeyError if unknown)."""
        return self._priority_to_batch[int(priority)]

    def num_batches(self) -> int:
        return len(self._batches)

    def num_realizations(self) -> int:
        """Total |Ω_B| = product of |B_k|! across batches."""
        import math

        r = 1
        for b in self._batches:
            r *= math.factorial(len(b))
        return r

    def yard_snapshot_priorities(self) -> Dict[Tuple[int, int], List[int]]:
        """
        Return ``{(bay,row): [priorities bottom→top]}`` for the current
        yard.  Convenience helper used by every SCRP algorithm.
        """
        out: Dict[Tuple[int, int], List[int]] = {}
        for key, stack in self.yard.stacks.items():
            out[key] = [int(c.priority) for c in stack.containers]
        return out

    # ---------------------------------------------------------------- #
    # Metrics                                                            #
    # ---------------------------------------------------------------- #

    def get_metrics(self) -> Dict[str, float]:
        m = super().get_metrics()
        # Add SCRP-specific keys; fall back to the single-realization
        # relocation count when a policy tree hasn't been evaluated yet.
        expected = (
            float(self._last_expected_reloc)
            if self._last_expected_reloc is not None
            else float(self._total_relocations)
        )
        avg = (
            float(self._last_avg_reloc)
            if self._last_avg_reloc is not None
            else float(self._total_relocations)
        )
        m.update(
            {
                "expected_relocations": expected,
                "avg_relocations": avg,
                "num_batches": float(self.num_batches()),
                "num_realizations": float(self.num_realizations()),
                # Keep crane_time slot for algorithms/GUI that expect it
                # (0.0 unless caller explicitly evaluates a plan for time).
                "crane_time": 0.0,
            }
        )
        return m

    # ---------------------------------------------------------------- #
    # Policy-tree evaluation                                            #
    # ---------------------------------------------------------------- #

    def evaluate_policy(
        self,
        tree: Optional[ChanceNode],
        num_sample_realizations: int = 0,
        rng_seed: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Evaluate a SCRP policy tree.

        Parameters
        ----------
        tree : PolicyTree (== root ``ChanceNode``) or ``None``.
        num_sample_realizations : optional Monte-Carlo sample size for
            the ``avg_relocations`` sanity metric (0 = skip).
        rng_seed : reproducibility seed for MC sampling.

        Returns
        -------
        Dict with keys: ``expected_relocations`` (primary), ``feasible``,
        ``num_realizations``, and optionally ``avg_relocations`` when MC
        sampling is requested.
        """
        er = float(expected_relocations(tree))
        feasible = tree is not None and _tree_is_feasible(tree, self._batches)

        self._last_expected_reloc = er
        self._last_feasible = feasible

        out: Dict[str, float] = {
            "expected_relocations": er,
            "feasible": float(feasible),
            "num_realizations": float(self.num_realizations()),
        }

        if num_sample_realizations > 0 and tree is not None:
            avg = _average_relocations_mc(
                tree,
                self._batches,
                num_samples=int(num_sample_realizations),
                rng_seed=rng_seed,
            )
            self._last_avg_reloc = avg
            out["avg_relocations"] = float(avg)

        return out

    # ---------------------------------------------------------------- #
    # Config schema (adds SCRP knobs on top of CRP-R schema)             #
    # ---------------------------------------------------------------- #

    @classmethod
    def config_schema(cls) -> Dict[str, Any]:
        schema = super().config_schema()
        schema.update(
            {
                "batch_size": {
                    "type": "int",
                    "default": 1,
                    "min": 1,
                    "max": 40,
                    "label": "Batch size (containers/batch)",
                    "help": (
                        "SCRP batch size. 1 → deterministic (== CRP-R). "
                        "Larger values group containers so their intra-batch "
                        "retrieval order becomes stochastic."
                    ),
                },
                "num_sample_realizations": {
                    "type": "int",
                    "default": 0,
                    "min": 0,
                    "max": 100_000,
                    "label": "Monte-Carlo samples for avg E[R]",
                    "help": (
                        "When > 0, evaluate_policy() additionally reports "
                        "a Monte-Carlo sampled average of relocations "
                        "over that many realizations."
                    ),
                },
            }
        )
        return schema


# ────────────────────────────────────────────────────────────────────
#  Helpers (private)
# ────────────────────────────────────────────────────────────────────


def _tree_is_feasible(
    tree: ChanceNode,
    batches: Sequence[Sequence[int]],
) -> bool:
    """
    A policy tree is feasible iff every possible realization has a
    matching outcome branch in every ChanceNode along its path.
    """
    from itertools import permutations

    def _walk(cn: Optional[ChanceNode], depth: int) -> bool:
        if cn is None:
            return depth == len(batches)
        if depth >= len(batches):
            return not cn.outcomes  # extra chance nodes = malformed
        b = batches[depth]
        expected_keys = {tuple(p) for p in permutations(b)}
        if not expected_keys.issubset(set(cn.outcomes.keys())):
            return False
        # Recurse: each DecisionNode's next_chance must be feasible for depth+1
        for key in expected_keys:
            dn = cn.outcomes[key]
            if not _walk(dn.next_chance, depth + 1):
                return False
        return True

    return _walk(tree, 0)


def _average_relocations_mc(
    tree: ChanceNode,
    batches: Sequence[Sequence[int]],
    num_samples: int,
    rng_seed: Optional[int] = None,
) -> float:
    """Monte-Carlo estimate of E[R] by sampling realizations."""
    from core.stoch.policy_tree import total_relocations_along

    rng = np.random.RandomState(rng_seed if rng_seed is not None else 0)
    total = 0
    good = 0
    for _ in range(num_samples):
        realization: List[int] = []
        for b in batches:
            perm = list(b)
            rng.shuffle(perm)
            realization.extend(perm)
        total += total_relocations_along(tree, realization)
        good += 1
    return total / max(good, 1)
