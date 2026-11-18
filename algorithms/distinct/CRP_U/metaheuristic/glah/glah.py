"""
Greedy look-ahead heuristic for CRP-U
<2015> <metaheuristic> <unrestricted> <single-bay> <CRP-U>
D --- 3 --- Look-ahead depth

------------------------------- Reference --------------------------------
B. Jin, W. Zhu, A. Lim,
"Solving the container relocation problem by an improved greedy look-ahead
 heuristic",
European Journal of Operational Research 240 (2015) 837–847.
------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section.
--------------------------------------------------------------------------
"""

from algorithms.ALGORITHM import Algorithm
from algorithms.distinct.CRP_U.LB.lb2_u import lb2_u

from .lookahead import Lookahead, retrieve


class Move:
    def __init__(self, container, source, destination):
        self.container = container
        self.source = source
        self.destination = destination


class GLAH(Algorithm):

    def main(self):
        depth = self.parameter("D", 3, int)
        stacks = [list(stack) for stack in self.problem.stacks]
        max_tiers = self.problem.max_tiers
        search = Lookahead(stacks, max_tiers, depth)
        steps, _ = search.rollout(stacks)
        if steps is not None:
            search.update(list(steps))
        state = [list(stack) for stack in stacks]
        retrieve(state, search.path)
        while any(state):
            if lb2_u(state, max_tiers) + search.relocations >= search.best_count:
                break
            move = search.most_promising(state)
            if move is None:
                break
            source, destination = move
            state[destination].append(state[source].pop())
            search.path.append(move)
            search.relocations += 1
            retrieve(state, search.path)
        if search.best is None:
            raise ValueError("GLAH found no feasible solution")
        return _as_moves(stacks, search.best)


def _as_moves(stacks, steps):
    state = [list(stack) for stack in stacks]
    moves = []
    for source, destination in steps:
        tier = len(state[source])
        container = state[source].pop()
        if destination is None:
            moves.append(Move(container, (source, tier), None))
        else:
            state[destination].append(container)
            moves.append(Move(container, (source, tier), (destination, len(state[destination]))))
    return moves
