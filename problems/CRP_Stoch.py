"""
CRP-Stoch
<stochastic> <batch> <CRP-Stoch>
Batch-stochastic Container Relocation Problem (SCRP)

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP".
--------------------------------------------------------------------------
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.plan import Movement, RelocationPlan
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
    - ``num_sample_realizations``: int, default 0
        Read by Bacci's RIRH / Expected MinMax.  When > 0 those
        algorithms also report a Monte-Carlo ``avg_relocations``.
    """

    name = "CRP-Stoch"
    description = (
        "Batch-stochastic container retrieval with ordered batches and "
        "uniform intra-batch uncertainty."
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
        CRP-R match).  Algorithms that reason over several realizations
        read ``get_batches()``.
        """
        super()._build_episode()

        n = int(self.config.num_containers)
        extra = self.config.extra or {}
        explicit = extra.get("batch_sizes")
        batch_size = int(extra.get("batch_size", 1) or 1)
        if batch_size < 1:
            batch_size = 1

        self._batches = _make_batches(
            n_containers=n,
            batch_size=batch_size,
            explicit_sizes=explicit,
        )
        self._priority_to_batch = _priority_to_batch_map(self._batches)

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
        # Step-driven runs only see one realization.  Algorithms that
        # estimate E[R] themselves return that number in their own
        # metrics dict.
        realized = float(self._total_relocations)
        m.update(
            {
                "expected_relocations": realized,
                "avg_relocations": realized,
                "num_batches": float(self.num_batches()),
                "num_realizations": float(self.num_realizations()),
                "crane_time": 0.0,
            }
        )
        return m

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
                        "When > 0, Bacci RIRH and Expected MinMax also "
                        "report a Monte-Carlo average of relocations "
                        "over that many realizations."
                    ),
                },
            }
        )
        return schema


def _make_batches(
    n_containers: int,
    batch_size: int,
    explicit_sizes: Optional[Sequence[int]] = None,
) -> List[List[int]]:
    """Partition priorities {1, ..., N} into ordered batches.

    ``batch_size=3`` and ``N=7`` yields ``[[1, 2, 3], [4, 5, 6], [7]]``.
    A smaller batch index must be retrieved before a larger one.  Order
    inside a batch is not fixed here.
    """
    if n_containers <= 0:
        return []
    if explicit_sizes is not None:
        sizes = [int(s) for s in explicit_sizes]
        if any(s <= 0 for s in sizes):
            raise ValueError(f"explicit_sizes must be positive: {sizes}")
        if sum(sizes) != n_containers:
            raise ValueError(
                f"explicit_sizes sum {sum(sizes)} != n_containers {n_containers}"
            )
    else:
        b = max(1, int(batch_size))
        n_batches = math.ceil(n_containers / b)
        sizes = [b] * n_batches
        total = sum(sizes)
        if total > n_containers:
            sizes[-1] -= total - n_containers

    batches: List[List[int]] = []
    priority = 1
    for size in sizes:
        batches.append(list(range(priority, priority + size)))
        priority += size
    return batches


def _priority_to_batch_map(batches: Sequence[Sequence[int]]) -> Dict[int, int]:
    """Return {priority: batch_index} (0-indexed batch)."""
    out: Dict[int, int] = {}
    for batch_idx, priorities in enumerate(batches):
        for priority in priorities:
            out[int(priority)] = batch_idx
    return out
