"""
GRASP for the restricted stowage-plan relocation problem.
<2019> <metaheuristic> <restricted> <single-bay> <CRP-Stow-R>
iterations --- 100 --- Randomized constructions; the shortest one is kept
seed --- 0 --- Seed for tie breaks
moved --- 2 --- A container relocated at least this many times is corrected
block --- 1 --- Block length that makes an earlier relocation suspicious

------------------------------- Reference --------------------------------
R. Jovanović, S. Tanaka, T. Nishi, S. Voß,
"A GRASP approach for solving the Blocks Relocation Problem with Stowage Plan",
Flexible Services and Manufacturing Journal 31 (2019) 702–729.
Each iteration builds one GRC solution. Ties in MinW4CB and MinMax4CB are
broken at random; correction after each construction stays deterministic.
------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section.
--------------------------------------------------------------------------
"""

import random

from algorithms.ALGORITHM import Algorithm
from algorithms.stowage.CRP_S_R.operator.minw4cb import relocations


class GRASP(Algorithm):
    compatible_problems = ["CRP-Stow-R", "CRP-Stow"]

    def main(self):
        iterations = self.parameter("iterations", 100, int)
        rng = random.Random(self.parameter("seed", 0, int))
        moved = self.parameter("moved", 2, int)
        block = self.parameter("block", 1, int)
        best = None
        for _ in range(iterations):
            moves = relocations(
                self.problem.stacks,
                self.problem.max_tiers,
                moved=moved,
                block=block,
                rng=rng,
            )
            if best is None or len(moves) < len(best):
                best = moves
        return self.problem.replay(best or [])
