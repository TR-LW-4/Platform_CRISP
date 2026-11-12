"""
Zhu iterative deepening A* for CRP-R
<2012> <exact> <restricted> <single-bay> <CRP-R>
time_limit --- 300.0 --- Per-instance wall-clock limit (s)
lb_mode --- LB3 --- Lower bound: LB1, LB2, or LB3
probe_mode --- PR4 --- Probe heuristic: PR1, PR2, PR3, or PR4

------------------------------- Reference --------------------------------
W. Zhu, H. Qin, A. Lim, H. Zhang,
"Iterative deepening A* algorithms for the container relocation problem",
IEEE Transactions on Automation Science and Engineering 9 (2012) 710–722.
------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from algorithms.CRP_R.ALGORITHM import Algorithm

from .search import search


class Zhu2012IDAStarR(Algorithm):

    def main(self, problem):
        time_limit = self.parameter("time_limit", 300.0, float)
        return search(
            problem.stacks,
            problem.n,
            problem.max_tiers,
            time_limit,
            lb_mode=self.parameter("lb_mode", "LB3", str),
            probe_mode=self.parameter("probe_mode", "PR4", str),
            stop=self.stopped,
            on_progress=lambda elapsed, cost: self.report(
                min(elapsed / time_limit, 1.0), cost
            ),
        )
