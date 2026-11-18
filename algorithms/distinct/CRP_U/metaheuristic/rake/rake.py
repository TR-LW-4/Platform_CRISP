"""
Rake search for CRP-U
<2018> <metaheuristic> <unrestricted> <single-bay> <CRP-U>
width --- 500 --- Partial solutions grown breadth first before FastMeta finishes each one

This folder contains three algorithms: RakeSearch (rake.py), Pilot (pilot.py),
SmSEQ (smseq.py). Shared code: heuristic.py, search.py, state.py.

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

from .search import rake_search
from .state import State, as_moves


class RakeSearch(Algorithm):

    def main(self):
        width = self.parameter("width", 500, int)
        state = State(self.problem.stacks, self.problem.max_tiers)
        return as_moves(self.problem.stacks, rake_search(state, width).ops)
