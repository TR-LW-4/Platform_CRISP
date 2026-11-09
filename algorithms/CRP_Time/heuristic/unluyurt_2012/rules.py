"""
The destination rules of the Difference heuristic (Ünlüyurt & Aydın 2012,
Sec. 4.2, p. 386-387) and the two selectors built from them, Difference1 and
Difference2. Stacks are lists of retrieval priorities, bottom to top; stack 0 is
nearest the truck lane. Decisions O2-O10: docs/validation/unluyurt_2012.md §2.

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section of difference1.py.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from algorithms.CRP_Time.utils.bay import free_destinations
from algorithms.CRP_Time.utils.located import min_due


def rule_choice(rule: int, x: int, candidates: Sequence[int],
                stacks: Sequence[Sequence[int]], empty: int) -> Optional[int]:
    """The stack among candidates that rule 1, 2 or 3 (p. 386) chooses for container x; None if none qualifies.

    Rule 1: the earliest-retrieved container Y of the stack is retrieved after x; the earliest Y wins.
    Rule 2: the top container Z is retrieved before x; the smallest x - Z wins.
    Rule 3: the smallest difference between x and the top container.
    An empty stack counts as container empty = N + 1 (O4); ties go to the lowest stack index (O3).
    """
    scored: List[Tuple[int, int]] = []
    for i in candidates:
        y = min_due(stacks[i], empty)
        top = stacks[i][-1] if stacks[i] else empty
        if rule == 1 and y > x:
            scored.append((y, i))
        elif rule == 2 and top < x:
            scored.append((x - top, i))
        elif rule == 3:
            scored.append((abs(top - x), i))
    return min(scored)[1] if scored else None


def first_choice(x: int, steps: Sequence[Tuple[int, Sequence[int]]],
                 stacks: Sequence[Sequence[int]], empty: int) -> Optional[int]:
    """The choice of the first (rule, candidates) step in which the rule finds a stack."""
    for rule, candidates in steps:
        dst = rule_choice(rule, x, candidates, stacks, empty)
        if dst is not None:
            return dst
    return None


def difference1(x: int, src: int, stacks: Sequence[Sequence[int]], max_tiers: int, empty: int) -> Optional[int]:
    """Destination of x, the top container of stack src (p. 386): rules 1, 2 and 3 in turn over
    every other stack with room. None if no other stack has room."""
    candidates = free_destinations(stacks, src, max_tiers)
    return first_choice(x, [(rule, candidates) for rule in (1, 2, 3)], stacks, empty)


def difference2(x: int, src: int, stacks: Sequence[Sequence[int]], max_tiers: int, empty: int) -> Optional[int]:
    """Destination of x under Difference2 (p. 387): at each rule level, first the stacks between src and
    the truck lane (lower index, O7), then the stacks beyond src (O2, reading (b))."""
    candidates = free_destinations(stacks, src, max_tiers)
    lane = [i for i in candidates if i < src]
    far = [i for i in candidates if i > src]
    return first_choice(x, [(rule, side) for rule in (1, 2, 3) for side in (lane, far)], stacks, empty)
