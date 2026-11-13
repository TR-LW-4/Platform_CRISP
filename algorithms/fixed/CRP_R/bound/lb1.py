"""
------------------------------- Reference --------------------------------
K.H. Kim, G.P. Hong,
"A heuristic rule for relocating blocks",
Computers & Operations Research 33 (2006) 940–954.
Zhu et al. (2012) call this lower bound LB1.
------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section.
--------------------------------------------------------------------------
"""

from typing import Optional, Sequence


def lb1(stacks: Sequence[Sequence[int]]) -> int:
    total = 0
    for stack in stacks:
        minimum_below: Optional[int] = None
        for priority in stack:
            if minimum_below is not None and priority > minimum_below:
                total += 1
            if minimum_below is None:
                minimum_below = priority
            else:
                minimum_below = min(minimum_below, priority)
    return total