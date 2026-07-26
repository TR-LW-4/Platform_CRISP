"""
Online reshuffle policies for the Online Container Relocation Problem
(Zehendner, Feillet & Jaillet, EJOR 2017).

Three policies are provided — all *look-ahead H = 0*, i.e. they use only
the identity of the container currently being retrieved and the current
yard state.  They match the paper's Section 3 and 5.

- ``leveling_pick``      : heuristic L (Section 3).  Relocate to the
  stack with the smallest current height (excluding the source stack).
  Break ties by preferring the leftmost stack (lowest (bay, row) key).
  Guarantees a feasible solution whenever one exists (Property 2 of the
  paper).
- ``random_pick``        : heuristic R.  Choose a random destination
  stack (uniform over non-full, non-source stacks).  Paper baseline.
- ``right_neighbour_pick``: heuristic M.  Move to the "right neighbour"
  stack ``(w+1)`` (wrapping W → 1).  Paper baseline.

All three take the SAME signature so they are interchangeable inside a
single ``step()``-driven driver loop.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


StackKey = Tuple[int, int]
Stacks = Dict[StackKey, List[int]]


# ────────────────────────────────────────────────────────────────────
#  Policies
# ────────────────────────────────────────────────────────────────────


def _candidate_stacks(
    stacks: Stacks,
    max_tiers: int,
    src: StackKey,
) -> List[StackKey]:
    """Return non-full, non-source stack keys in natural (bay, row) order."""
    return [
        k
        for k in sorted(stacks.keys())
        if k != src and len(stacks.get(k, [])) < max_tiers
    ]


def leveling_pick(
    stacks: Stacks,
    max_tiers: int,
    src: StackKey,
    blocker_priority: int,   # kept in signature for uniformity; unused
    rng: Optional[np.random.RandomState] = None,
) -> Optional[StackKey]:
    """Heuristic L: lowest-height stack (leftmost on tie)."""
    del blocker_priority, rng  # unused
    cands = _candidate_stacks(stacks, max_tiers, src)
    if not cands:
        return None
    # (height, natural order) — the natural order already tie-breaks left first
    cands.sort(key=lambda k: (len(stacks[k]), k))
    return cands[0]


def random_pick(
    stacks: Stacks,
    max_tiers: int,
    src: StackKey,
    blocker_priority: int,
    rng: Optional[np.random.RandomState] = None,
) -> Optional[StackKey]:
    """Heuristic R: uniform random destination."""
    del blocker_priority
    cands = _candidate_stacks(stacks, max_tiers, src)
    if not cands:
        return None
    r = rng if rng is not None else np.random.RandomState()
    return cands[int(r.randint(0, len(cands)))]


def right_neighbour_pick(
    stacks: Stacks,
    max_tiers: int,
    src: StackKey,
    blocker_priority: int,
    rng: Optional[np.random.RandomState] = None,
) -> Optional[StackKey]:
    """
    Heuristic M: move to the right neighbour stack of ``src``.

    The paper defines this on a single-row bay: stack ``w → w+1`` with
    wrap-around to ``1``.  For multi-row yards we scan the natural
    (bay, row) order and pick the next feasible neighbour after ``src``.
    """
    del blocker_priority, rng
    keys = sorted(stacks.keys())
    if src not in keys:
        return leveling_pick(stacks, max_tiers, src, blocker_priority=0)
    idx = keys.index(src)
    n = len(keys)
    for step in range(1, n):
        cand = keys[(idx + step) % n]
        if cand == src:
            continue
        if len(stacks.get(cand, [])) < max_tiers:
            return cand
    return None


POLICIES: Dict[str, Callable] = {
    "L": leveling_pick,
    "R": random_pick,
    "M": right_neighbour_pick,
}


# ────────────────────────────────────────────────────────────────────
#  Competitive-ratio bound (paper Theorem 1)
# ────────────────────────────────────────────────────────────────────


def leveling_competitive_ratio(n_containers: int, num_stacks: int) -> float:
    """
    Zehendner et al. 2017 Theorem 1 upper bound on the competitive ratio
    of heuristic L:  ``c_L = 2 * ceil(N / W) - 1``  where W is the number
    of stacks and N is the number of containers.
    """
    import math

    if num_stacks <= 0:
        return float("inf")
    return 2.0 * math.ceil(n_containers / num_stacks) - 1.0
