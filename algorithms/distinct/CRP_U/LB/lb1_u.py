"""
------------------------------- Reference --------------------------------
K.H. Kim, G.P. Hong,
"A heuristic rule for relocating blocks",
Computers & Operations Research 33 (2006) 940–954.
Tricoire et al. (2018) call this lower bound LB1: the number of containers
that sit above a container with a smaller priority.
------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section.
--------------------------------------------------------------------------
"""


def lb1_u(stacks, max_tiers) -> int:
    lower_bound = 0
    for stack in stacks:
        minimum_below = None
        for priority in stack:
            if minimum_below is not None and priority > minimum_below:
                lower_bound += 1
            if minimum_below is None or priority < minimum_below:
                minimum_below = priority
    return lower_bound
