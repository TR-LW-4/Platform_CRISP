"""
GBH, the Good-Bad Heuristic of Azari2017CSUM (Sec. 4.2, p. 312-313, Fig. 2): a
depth-first search with backtracking for the solution with the fewest movements,
two branches per relocation, pruned with the lower bound. Its solution is the
initial solution of CSUM.

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
corresponding original algorithm paper.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import time
from typing import List, Optional, Sequence, Tuple

from algorithms.CRP_Time.utils.bay import target_stack

from .branching import gbh_branches
from .lower_bound import lower_bound


def gbh(stacks: Sequence[Sequence[int]], max_tiers: int, time_limit: float) -> Optional[List[Tuple[int, Optional[int]]]]:
    """The complete solution with the fewest movements found within time_limit seconds, as moves
    (src, dst) with dst None for a retrieval; None if no solution was found."""
    deadline = time.monotonic() + time_limit
    st = [list(s) for s in stacks]
    targets = sorted(c for s in st for c in s)
    empty = targets[-1] + 1
    ub = max_tiers * len(targets)
    best: Optional[List[Tuple[int, Optional[int]]]] = None
    path: List[Tuple[int, Optional[int]]] = []

    def search(k: int) -> None:
        nonlocal ub, best
        if time.monotonic() >= deadline:
            return
        target = targets[k]
        src = target_stack(st, target)
        if st[src][-1] == target:
            st[src].pop()
            path.append((src, None))
            if k == len(targets) - 1:
                # Later solutions must use fewer movements than this one.
                ub = len(path) - 1
                best = path[:]
            else:
                search(k + 1)
            path.pop()
            st[src].append(target)
            return
        q = st[src][-1]
        for dst in gbh_branches(q, src, st, max_tiers, empty):
            if time.monotonic() >= deadline:
                return
            st[dst].append(st[src].pop())
            path.append((src, dst))
            # Prune unless Nm + LB <= UB, with Nm the movements so far.
            if len(path) + lower_bound(st) <= ub:
                search(k)
            path.pop()
            st[src].append(st[dst].pop())

    search(0)
    return best
