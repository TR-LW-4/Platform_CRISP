"""
Compound-move tree search for CRP-U
<2012> <metaheuristic> <unrestricted> <single-bay> <CRP-U>
time_limit --- 60 --- Search time limit (s)
n_succ --- 10 --- nSucc: compound moves expanded per node
cm_stop --- 150 --- cmStopThreshold: limits the length of a compound move
max_flg_bb --- 3 --- maxFlg_bb: FLG_BB relocations offered per step
max_gg --- 2 --- maxGg: GG relocations offered per step

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

import sys

from algorithms.ALGORITHM import Algorithm
from algorithms.distinct.CRP_U.operator import fb

from .search import Search


class Move:
    def __init__(self, container, source, destination):
        self.container = container
        self.source = source
        self.destination = destination


class CMTS(Algorithm):

    def main(self):
        time_limit = self.parameter("time_limit", 60.0, float)
        n_succ = self.parameter("n_succ", 10, int)
        cm_stop = self.parameter("cm_stop", 150, int)
        max_flg_bb = self.parameter("max_flg_bb", 3, int)
        max_gg = self.parameter("max_gg", 2, int)
        stacks = [list(stack) for stack in self.problem.stacks]
        max_tiers = self.problem.max_tiers
        # Recursion depth grows with the number of moves in a solution.
        sys.setrecursionlimit(max(sys.getrecursionlimit(), 20000))
        search = Search(max_tiers, time_limit, n_succ, cm_stop, max_flg_bb, max_gg)
        best = search.run(stacks, _greedy(stacks, max_tiers))
        return _as_moves(stacks, best)


def _greedy(stacks, max_tiers):
    state = [list(stack) for stack in stacks]
    moves = []
    while any(state):
        lowest = min(priority for stack in state for priority in stack)
        removes = [source for source, stack in enumerate(state) if stack and stack[-1] == lowest]
        if removes:
            state[removes[0]].pop()
            moves.append((removes[0], None))
            continue
        chosen = fb.choose_move(state, max_tiers)
        if chosen is None:
            raise ValueError(f"greedy cannot retrieve container {lowest}")
        source, destination = chosen
        state[destination].append(state[source].pop())
        moves.append(chosen)
    return moves


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
