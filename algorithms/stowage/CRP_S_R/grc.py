"""
Greedy with correction for the restricted stowage-plan relocation problem.
<2019> <heuristic> <restricted> <single-bay> <CRP-Stow-R>
moved --- 2 --- A container relocated at least this many times is corrected
block --- 1 --- Block length that makes an earlier relocation suspicious

------------------------------- Reference --------------------------------
R. Jovanović, S. Tanaka, T. Nishi, S. Voß,
"A GRASP approach for solving the Blocks Relocation Problem with Stowage Plan",
Flexible Services and Manufacturing Journal 31 (2019) 702–729.
One deterministic construction with MinW4CB and MinMax4CB, then correction
of suspicious relocations. The same rule is repeated by GRASP.
------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section.
--------------------------------------------------------------------------
"""

from algorithms.ALGORITHM import Algorithm
from algorithms.stowage.CRP_S_R.operator.minw4cb import relocations


class GRC(Algorithm):
    compatible_problems = ["CRP-Stow-R", "CRP-Stow"]

    def main(self):
        moves = relocations(
            self.problem.stacks,
            self.problem.max_tiers,
            moved=self.parameter("moved", 2, int),
            block=self.parameter("block", 1, int),
        )
        return self.problem.replay(moves)
