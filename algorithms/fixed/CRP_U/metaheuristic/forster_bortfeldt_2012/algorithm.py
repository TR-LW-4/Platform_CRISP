"""
Forster
<2012> <heuristic> <unrestricted> <single-bay> <CRP-U>
Compound-move tree search for unrestricted BRP
time_limit_s --- 15 --- Search time limit (s)
n_succ --- 5 --- Successor compound moves per node

------------------------------- Reference --------------------------------
F. Forster, A. Bortfeldt,
"A tree search procedure for the container relocation problem",
Computers & Operations Research 39 (2012) 299–309.
------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section.
--------------------------------------------------------------------------
"""

from algorithms.ALGORITHM import Algorithm
from algorithms.fixed.CRP_U.move import Move

from .layout import FBLayout
from .tree_search import run_tree_search


class Forster(Algorithm):

    def main(self):
        time_limit = self.parameter("time_limit_s", 15.0, float)
        n_succ = self.parameter("n_succ", 5, int)
        stacks = [list(stack) for stack in self.problem.stacks]
        priorities = [priority for stack in stacks for priority in stack]
        groups = max(priorities) if priorities else 1
        layout = FBLayout(
            stacks,
            len(stacks),
            self.problem.max_tiers,
            groups,
            len(priorities),
            min(priorities) if priorities else groups + 1,
        )
        ops, _count = run_tree_search(layout, time_limit=time_limit, n_succ=n_succ)
        return _ops_to_moves(self.problem.stacks, ops)


def _ops_to_moves(stacks, ops):
    state = [list(stack) for stack in stacks]
    moves = []
    for op in ops:
        if op[0] == "rem":
            source = op[1]
            container = state[source].pop()
            moves.append(Move(container, source, None))
        else:
            source, destination = op[1], op[2]
            container = state[source].pop()
            state[destination].append(container)
            moves.append(Move(container, source, destination))
    return moves
