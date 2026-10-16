"""
CSUM, the Constant Summation search of Azari2017CSUM (Sec. 4.3, p. 313-315,
Fig. 3): a depth-first search for the solution with the least crane time. A
blocker in set A (relocated more than once in the GBH solution) gets the GBH
branches, any other blocker the set-B branches. A relocation is added only while
the movements stay within NMOV, which starts at the movements of the GBH solution.

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
corresponding original algorithm paper.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import math
import time
from collections import Counter
from typing import List, Optional, Sequence, Set, Tuple

from algorithms.CRP_Time.utils.bay import target_stack
from algorithms.CRP_Time.utils.crane_time import move_time
from algorithms.CRP_Time.utils.plan import replay
from core.objectives import ObjectiveSpec
from core.plan import Movement

from .branching import gbh_branches, set_b_branches

_EPS = 1e-9


def set_a(stacks: Sequence[Sequence[int]], positions: Sequence[Tuple[int, int]],
          moves: Sequence[Tuple[int, Optional[int]]]) -> Set[int]:
    """The containers relocated more than once by the moves."""
    count = Counter(m.container_id for m in replay(stacks, positions, moves).movements if not m.is_retrieval)
    return {c for c, n in count.items() if n > 1}


def csum(stacks: Sequence[Sequence[int]], positions: Sequence[Tuple[int, int]], max_tiers: int,
         initial: Sequence[Tuple[int, Optional[int]]], spec: ObjectiveSpec,
         time_limit: float) -> Optional[List[Tuple[int, Optional[int]]]]:
    """The fastest complete solution found within time_limit seconds, starting from the GBH
    solution `initial`, as moves (src, dst) with dst None for a retrieval; None if none was found."""
    deadline = time.monotonic() + time_limit
    st = [list(s) for s in stacks]
    targets = sorted(c for s in st for c in s)
    empty = targets[-1] + 1
    a = set_a(stacks, positions, initial)
    nmov = len(initial)
    best_time = math.inf
    best: Optional[List[Tuple[int, Optional[int]]]] = None
    path: List[Tuple[int, Optional[int]]] = []

    def search(k: int, nm: int, tcw: float) -> None:
        nonlocal nmov, best_time, best
        if time.monotonic() >= deadline:
            return
        target = targets[k]
        src = target_stack(st, target)
        if st[src][-1] == target:
            t = tcw + move_time(Movement(target, positions[src], None, from_tier=len(st[src])), spec)
            st[src].pop()
            path.append((src, None))
            if k == len(targets) - 1:
                # tcw <= Tbest: a later solution with equal time replaces the earlier one.
                if t <= best_time + _EPS:
                    nmov, best_time, best = nm + 1, t, path[:]
            else:
                search(k + 1, nm + 1, t)
            path.pop()
            st[src].append(target)
            return
        q = st[src][-1]
        if q in a:
            branches = gbh_branches(q, src, st, max_tiers, empty)
        else:
            # Not in the paper: a set-B container without a BG stack gets the GBH branches.
            branches = set_b_branches(q, src, st, max_tiers, empty) or gbh_branches(q, src, st, max_tiers, empty)
        if time.monotonic() >= deadline:
            return
        for dst in branches:
            # Nm <= NMOV is checked only when a relocation is added (Fig. 3).
            if nm + 1 > nmov:
                continue
            dt = move_time(Movement(q, positions[src], positions[dst],
                                    from_tier=len(st[src]), to_tier=len(st[dst]) + 1), spec)
            st[dst].append(st[src].pop())
            path.append((src, dst))
            search(k, nm + 1, tcw + dt)
            path.pop()
            st[src].append(st[dst].pop())

    search(0, 0, 0.0)
    return best
