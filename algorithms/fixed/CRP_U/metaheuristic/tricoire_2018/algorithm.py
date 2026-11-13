"""
Tricoire
<2018> <heuristic> <unrestricted> <single-bay> <CRP-U>
Safe-move greedy, SmSEQ, RakeSearch, and Pilot methods
mode --- SmSEQ-2 --- SM-1/2, SmSEQ-1/2, RakeSearch, or Pilot
width --- 100 --- RakeSearch / Pilot beam width

------------------------------- Reference --------------------------------
F. Tricoire, S.N. Parragh, D. Feillet,
"New insights on the block relocation problem",
Computers & Operations Research 89 (2018) 127–139.
------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section.
--------------------------------------------------------------------------
"""

from algorithms.ALGORITHM import Algorithm
from algorithms.fixed.CRP_U.move import Move

from .heuristics import solve_brp
from .state import BRPState


class Tricoire(Algorithm):

    def main(self):
        mode = self.parameter("mode", "SmSEQ-2", str)
        condensation = self.parameter("condensation", "improved", str)
        width = self.parameter("width", 100, int)
        hub_width = self.parameter("hub_width", 2, int)
        stacks = [list(stack) for stack in self.problem.stacks]
        total = self.problem.num_containers
        state = BRPState(stacks, total, self.problem.max_tiers, len(stacks))
        solved = solve_brp(
            state,
            mode=mode,
            width=width,
            hub_width=hub_width,
            condensation=condensation,
        )
        return _ops_to_moves(stacks, solved.ops)


def _ops_to_moves(stacks, ops):
    state = [list(stack) for stack in stacks]
    moves = []
    for source, destination in ops:
        container = state[source].pop()
        if source == destination:
            moves.append(Move(container, source, None))
        else:
            state[destination].append(container)
            moves.append(Move(container, source, destination))
    return moves
