"""
Pilot method for CRP-U
<2018> <metaheuristic> <unrestricted> <single-bay> <CRP-U>
width --- 10 --- Best candidates kept per iteration
hub_width --- 2 --- Width of the rake search that scores each candidate

------------------------------- Reference --------------------------------
F. Tricoire, J. Scagnetti, A. Beham,
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

from .search import pilot
from .state import State, as_moves


class Pilot(Algorithm):

    def main(self):
        width = self.parameter("width", 10, int)
        hub_width = self.parameter("hub_width", 2, int)
        state = State(self.problem.stacks, self.problem.max_tiers)
        return as_moves(self.problem.stacks, pilot(state, width, hub_width).ops)
