"""
Bounded beam search for CRP-R
<2019> <metaheuristic> <restricted> <single-bay> <CRP-R>
time_limit --- 5.0 --- Time limit in seconds

------------------------------- Reference --------------------------------
T. Bacci, S. Mattia, P. Ventura,
"The bounded beam search algorithm for the block relocation problem",
Computers & Operations Research 103 (2019) 252–264.
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


class BBS(Algorithm):

    def main(self, problem):
        time_limit = self.parameter("time_limit", 5.0, float)
        return search(
            problem.stacks,
            problem.n,
            problem.max_tiers,
            time_limit,
            stop=self.stopped,
            on_progress=lambda elapsed, cost: self.report(
                min(elapsed / time_limit, 1.0), cost
            ),
        )
