"""
Destination stacks of Azari2017CSUM for q, the container on top of the present
stack (PS): the best BG and best BB stacks (Def. 3, p. 312), the GBH branching
(Sec. 4.2, p. 313), and the set-B branching of CSUM (Fig. 3, p. 314). A move of
q is BG if q is good on the destination (lower number than the stack's lowest),
BB if it stays bad. lows[i] is the lowest number in stack i, `empty` (N + 1)
for an empty stack.

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
corresponding original algorithm paper.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

from algorithms.CRP_Time.utils.bay import free_destinations
from algorithms.CRP_Time.utils.located import min_due


def best_bg(q: int, lows: Sequence[int], candidates: Sequence[int]) -> Optional[int]:
    """The BG stack with the lowest lowest number; among empty stacks the lowest index."""
    bg = [i for i in candidates if q < lows[i]]
    return min(bg, key=lambda i: (lows[i], i)) if bg else None


def best_bb(q: int, lows: Sequence[int], candidates: Sequence[int]) -> Optional[int]:
    """The BB stack with the highest lowest number (a BB stack is never empty, so there are no ties)."""
    bb = [i for i in candidates if q > lows[i]]
    return max(bb, key=lambda i: (lows[i], -i)) if bb else None


def gbh_branches(q: int, src: int, stacks: Sequence[Sequence[int]], max_tiers: int, empty: int) -> List[int]:
    """[FDS, SDS] among the non-full stacks other than the PS: the best BG and the best BB
    stack if a BG move exists, otherwise the best and second-best BB stacks."""
    lows = [min_due(st, empty) for st in stacks]
    candidates = free_destinations(stacks, src, max_tiers)
    fds = best_bg(q, lows, candidates)
    if fds is not None:
        sds = best_bb(q, lows, candidates)
        return [fds] if sds is None else [fds, sds]
    bb = sorted((i for i in candidates if q > lows[i]), key=lambda i: (lows[i], -i), reverse=True)
    return bb[:2]


def set_b_branches(q: int, src: int, stacks: Sequence[Sequence[int]], max_tiers: int, empty: int) -> List[int]:
    """Q for a set-B container: the best BG stack between the truck lane and the PS (G1), then
    every BG stack beyond the PS (G2), in increasing order, whose lowest number is below all
    those already in Q. Only BG stacks are compared, as in the example of Fig. 4."""
    lows = [min_due(st, empty) for st in stacks]
    g1 = [i for i in range(src) if len(stacks[i]) < max_tiers and q < lows[i]]
    out: List[int] = []
    bound = math.inf
    if g1:
        s = min(g1, key=lambda i: (lows[i], i))
        out.append(s)
        bound = lows[s]
    # Fig. 3 reads "m := bound", with m the lowest number in the stack; "bound := m" is intended.
    for j in range(src + 1, len(stacks)):
        if len(stacks[j]) < max_tiers and q < lows[j] < bound:
            out.append(j)
            bound = lows[j]
    return out
