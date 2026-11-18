"""
Decreasing-sequence heuristic SmSEQ for CRP-U
<2018> <heuristic> <unrestricted> <single-bay> <CRP-U>
variant --- 2 --- Safe moves used when no decreasing sequence exists: 1 (SmSEQ-1) or 2 (SmSEQ-2)
condensation --- tricoire --- Post-processing of the solution: tricoire, jin, or none

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

from .heuristic import smseq
from .state import State, as_moves


class SmSEQ(Algorithm):

    def main(self):
        variant = self.parameter("variant", 2, int)
        condensation = self.parameter("condensation", "tricoire", str).lower()
        state = State(self.problem.stacks, self.problem.max_tiers)
        return as_moves(self.problem.stacks, smseq(state, variant, condensation).ops)
