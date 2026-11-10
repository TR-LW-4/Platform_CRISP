"""
Layouts, item placement and the lower bound of the tree search of Forster &
Bortfeldt (2012), Sec. 4.1 and 5 (p. 301-303). A layout is a list of stacks,
each a list of item groups from bottom to top. Decisions F1-F...:
docs/validation/forster_bortfeldt_2012.md §2.

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite
F. Forster, A. Bortfeldt, "A tree search procedure for the container
relocation problem", Computers & Operations Research 39 (2012) 299-309.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

Stacks = Sequence[Sequence[int]]


def badly_placed(stack: Sequence[int]) -> List[bool]:
    """Per item, bottom to top: badly placed if a lower item of the stack has a smaller group (p. 302, def. 1)."""
    flags, low = [], None
    for g in stack:
        flags.append(low is not None and low < g)
        low = g if low is None else min(low, g)
    return flags


def n_badly_placed(stacks: Stacks) -> int:
    """n_b of Table 1 (p. 302)."""
    return sum(sum(badly_placed(s)) for s in stacks)


def n_items(stacks: Stacks) -> int:
    """n(L) of Table 1 (p. 302)."""
    return sum(len(s) for s in stacks)


def removable(stacks: Stacks) -> List[int]:
    """Stacks whose top item belongs to the currently lowest group (p. 302)."""
    if not any(stacks):
        return []
    low = min(min(s) for s in stacks if s)
    return [i for i, s in enumerate(stacks) if s and s[-1] == low]


def g_top(stack: Sequence[int], n: int) -> int:
    """g_top(s) of Table 1 (p. 302): the top item's group if it is badly placed, n(L) + 1 otherwise (F2)."""
    return stack[-1] if stack and badly_placed(stack)[-1] else n + 1


def g_min(stack: Sequence[int]) -> int:
    """g_min(s) of Table 1 (p. 302): the smallest group in the stack, 0 if it is empty."""
    return min(stack) if stack else 0


def lower_bound_parts(stacks: Stacks) -> Tuple[int, int, int]:
    """(n'_REM, n'_BG, n'_non-BG) of the Proposition in Sec. 5 (p. 302)."""
    n = n_items(stacks)
    if n == 0:
        return 0, 0, 0
    non_bg = int(not removable(stacks)
                 and all(stacks)
                 and min(g_top(s, n) for s in stacks) > max(g_min(s) for s in stacks))
    return n, n_badly_placed(stacks), non_bg


def lower_bound(stacks: Stacks) -> int:
    """n'_m = n'_REM + n'_BG + n'_non-BG, a lower bound on the moves that empty the layout (p. 302)."""
    return sum(lower_bound_parts(stacks))
