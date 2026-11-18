"""
Depth-first branch and bound for CRP-U
<2018> <exact> <unrestricted> <single-bay> <CRP-U>
lb --- 2 --- Lower bound used to prune nodes: 1 (LB1), 2 (LB2), or 3 (LB3)
loop --- false --- Raise the target one relocation at a time, starting from LB3
ub_width --- 500 --- Width of the rake search that gives the initial upper bound
time_limit --- 0 --- Time limit (s); 0 runs until optimality is proven

------------------------------- Reference --------------------------------
F. Tricoire, J. Scagnetti, A. Beham,
"New insights on the block relocation problem",
Computers & Operations Research 89 (2018) 127–139.
The initial upper bound is the rake search of the same paper, imported
from metaheuristic/rake/search.py.
------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section.
--------------------------------------------------------------------------
"""

from algorithms.ALGORITHM import Algorithm

from .search import dfbb


class Move:
    def __init__(self, container, source, destination):
        self.container = container
        self.source = source
        self.destination = destination


class DFBBU(Algorithm):

    def main(self):
        lb = self.parameter("lb", 2, int)
        loop = self.parameter("loop", "false", str).lower() in ("1", "true", "yes")
        ub_width = self.parameter("ub_width", 500, int)
        time_limit = self.parameter("time_limit", 0.0, float)
        ops = dfbb(self.problem.stacks, self.problem.max_tiers, lb, loop, ub_width, time_limit)
        return _as_moves(self.problem.stacks, ops)


def _as_moves(stacks, ops):
    state = [list(stack) for stack in stacks]
    moves = []
    for source, destination in ops:
        tier = len(state[source])
        container = state[source].pop()
        if source == destination:
            moves.append(Move(container, (source, tier), None))
        else:
            state[destination].append(container)
            moves.append(Move(container, (source, tier), (destination, len(state[destination]))))
    return moves
