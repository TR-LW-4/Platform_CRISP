"""
Batch generation helpers for the Stochastic Container Relocation Problem
(SCRP; Bacci, Mattia & Ventura, Soft Computing 2022 — Section 2).

Definitions (following Bacci 2022)
----------------------------------
- The container set I = {1, ..., N} is partitioned into b batches
  B_1, ..., B_b such that ALL containers of B_k must be retrieved
  strictly before ANY container of B_{k+1}.
- The intra-batch retrieval order is revealed only when the last
  container of the previous batch has left the yard.
- A "B-based order" ω is any permutation of {1..N} consistent with the
  batch precedence.
- |Ω_B| = |B_1|! * |B_2|! * ... * |B_b|!.

Semantics of ``batch_size`` in this platform
--------------------------------------------
We expose the knob as ``batch_size`` (containers per batch) instead of
``num_batches`` to avoid the confusion between "1 group of N containers"
and "N groups of 1 container".  A larger ``batch_size`` = more
uncertainty per batch and more possible realizations.

- ``batch_size = 1`` → each container in its own batch, |Ω_B| = 1
  → **deterministic** retrieval, equivalent to CRP-R.
- ``batch_size = N`` → one big batch with N! realizations
  → maximum uncertainty.
- ``batch_size = k``, ``1 < k < N`` → ⌈N/k⌉ batches, most of size k
  with a possibly smaller final batch.
"""

from __future__ import annotations

import math
from itertools import permutations
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


# ────────────────────────────────────────────────────────────────────
#  Batch construction
# ────────────────────────────────────────────────────────────────────


def make_batches(
    n_containers: int,
    batch_size: int,
    explicit_sizes: Optional[Sequence[int]] = None,
) -> List[List[int]]:
    """
    Partition priorities {1, ..., N} into ordered batches.

    Priorities within a batch are consecutive; a smaller batch index means
    "must be retrieved earlier". Each batch is returned as a list of the
    priorities it contains (bottom-of-set to top-of-set, but the intra-batch
    order is *not* meaningful — realizations are drawn later).

    Parameters
    ----------
    n_containers   : total number of containers N
    batch_size     : nominal containers per batch (last batch may be smaller)
    explicit_sizes : optional list of batch sizes; if given, must sum to N and
                     overrides ``batch_size``

    Returns
    -------
    List of batches; each batch is a list of priorities.

    Examples
    --------
    >>> make_batches(7, 3)
    [[1, 2, 3], [4, 5, 6], [7]]
    >>> make_batches(5, 1)
    [[1], [2], [3], [4], [5]]
    >>> make_batches(6, 2, explicit_sizes=[1, 2, 3])
    [[1], [2, 3], [4, 5, 6]]
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
        num_batches = math.ceil(n_containers / b)
        sizes = [b] * num_batches
        # trim last batch if it overshoots
        total = sum(sizes)
        if total > n_containers:
            sizes[-1] -= total - n_containers

    batches: List[List[int]] = []
    p = 1
    for s in sizes:
        batches.append(list(range(p, p + s)))
        p += s
    return batches


def priority_to_batch_map(batches: Sequence[Sequence[int]]) -> Dict[int, int]:
    """Return {priority: batch_index} lookup (0-indexed batch)."""
    out: Dict[int, int] = {}
    for b_idx, priorities in enumerate(batches):
        for p in priorities:
            out[int(p)] = b_idx
    return out


# ────────────────────────────────────────────────────────────────────
#  Realization enumeration (used by the exact evaluator)
# ────────────────────────────────────────────────────────────────────


def enumerate_realizations(
    batch: Sequence[int],
    max_permutations: Optional[int] = None,
) -> Iterable[Tuple[int, ...]]:
    """
    Yield all permutations of a single batch (intra-batch retrieval orders).

    For the SCRP with the uniform intra-batch distribution assumed in Bacci
    2022 (Section 3), every permutation has probability 1 / |batch|!.

    When ``max_permutations`` is set, only the first that many are yielded
    (useful for Monte-Carlo evaluation on very large batches where full
    enumeration would explode).
    """
    perms = permutations(batch)
    if max_permutations is None:
        yield from perms
    else:
        for i, p in enumerate(perms):
            if i >= max_permutations:
                return
            yield p


def num_realizations(batches: Sequence[Sequence[int]]) -> int:
    """Total |Ω_B| = product of |B_k|! across batches."""
    r = 1
    for b in batches:
        r *= math.factorial(len(b))
    return r
