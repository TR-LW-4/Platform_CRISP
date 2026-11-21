"""
Branch-and-bound for the unrestricted block relocation problem with a stowage plan.
<2019> <exact> <unrestricted> <single-bay> <CRP-Stow-U>
lb_mode --- lbr --- Lower bound: lb2c, lb2c4c, or lbr
time_limit --- 0 --- Time limit (s); 0 runs until optimality is proven

------------------------------- Reference --------------------------------
S. Tanaka, S. Voß,
"An exact algorithm for the block relocation problem with a stowage plan",
European Journal of Operational Research 279 (2019) 767–781.
Any container on top of a yard stack may be relocated. The upper bound is
the Section 6.3 greedy heuristic, which is feasible for this variant as well.
------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section.
--------------------------------------------------------------------------
"""

from algorithms.ALGORITHM import Algorithm

from .search import solve


class IDBBSU(Algorithm):
    compatible_problems = ["CRP-Stow-U"]

    def main(self):
        moves, _proven = solve(
            self.problem.stacks,
            self.problem.max_tiers,
            self.problem.vessel_heights,
            lb_mode=self.parameter("lb_mode", "lbr", str),
            time_limit=self.parameter("time_limit", 0.0, float),
        )
        return self.problem.replay(moves)
