"""
MinMax heuristic of JovanovicACO_CRPTime: how undesirable a destination stack is
for a relocated container, and the extended minimum due date that indexes the
pheromone matrix. Section 6 of the paper keeps this heuristic unchanged for crane
time.

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
corresponding original algorithm paper.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import List


def dd_star(stack: List[int], n_total: int, s: int) -> int:
    """
    dd*(S) — extended dd for pheromone-matrix indexing.

    Non-empty stack: returns dd(S) = min due-date ∈ {1..N}.
    Empty stack s (1-based): returns N + s ∈ {N+1 .. N+W}.
      This differentiates empty stacks so the pheromone matrix can learn
      which empty slot is most beneficial.
    """
    if not stack:
        return n_total + s
    return min(stack)


def dif(c: int, d: int, n_total: int) -> int:
    """
    Undesirability of placing container c above a stack whose minimum
    due-date is d (N + 1 for an empty stack).

    Range: 1 .. 2N.  Smaller = more desirable destination.
    """
    if d > c:
        return d - c
    else:
        return 2 * n_total + 1 - d
