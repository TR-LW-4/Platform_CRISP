"""
------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
corresponding original algorithm paper.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def dd_star(stack_priorities: List[int], n_total: int, stack_1based_idx: int) -> int:
    """
    dd*(S) — extended dd for pheromone-matrix indexing.

    Non-empty stack: returns dd(S) = min due-date ∈ {1..N}.
    Empty stack    : returns N + stack_1based_idx ∈ {N+1 .. N+W}.
      This differentiates empty stacks from each other so the pheromone
      matrix can learn which empty slot is most beneficial.
    """
    if not stack_priorities:
        return n_total + stack_1based_idx
    return min(stack_priorities)


# ─────────────────────────────────────────────────────────────────────────────
# Heuristic scoring
# ─────────────────────────────────────────────────────────────────────────────

def dif(c: int, d: int, n_total: int) -> int:
    """
    Undesirability of placing container c above a stack whose
    minimum due-date is d.

    dif(c, d) ≤ N  →  c will be well-located (d > c, d retrieved after c)
    dif(c, d) > N  →  c will NOT be well-located (d < c, d retrieved before c)

    Range: 1 .. 2N.  Smaller = more desirable destination.

    Well-located case (d > c):
      dif = d − c  ← how many "slots" are wasted between c and d;
                     smaller = tighter fit = more future room preserved.

    Not-well-located case (d < c):
      dif = 2N + 1 − d  ← larger d → smaller dif → prefer stacks with
                          higher min-due-date (delays re-relocation longest).
    """
    if d > c:
        return d - c
    else:
        return 2 * n_total + 1 - d




# ─────────────────────────────────────────────────────────────────────────────
# Lower bound
# ─────────────────────────────────────────────────────────────────────────────

def compute_lb(stacks: Dict[Any, List[int]]) -> int:
    """
    Lower bound on remaining relocations: number of non-well-located
    containers in the current bay state.

    Container c (at tier i, bottom-to-top index) is non-well-located if
    ∃ d at tier j < i (below c) with d < c.
    (d is retrieved before c but c is blocking d → c needs ≥ 1 relocation.)
    """
    lb = 0
    for prios in stacks.values():
        for i in range(1, len(prios)):
            c = prios[i]
            if min(prios[:i]) < c:
                lb += 1
    return lb


