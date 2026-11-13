"""
Kim and Hong branch-and-bound for CRP-R
<2006> <exact> <restricted> <single-bay> <CRP-R>
time_limit --- 300.0 --- Per-instance wall-clock limit (s)

------------------------------- Reference --------------------------------
K.H. Kim, G.P. Hong,
"A heuristic rule for relocating blocks",
Computers & Operations Research 33 (2006) 940–954.
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


class KimHong2006BB(Algorithm):

    def main(self, problem):
        time_limit = self.parameter("time_limit", 300.0, float)
        return search(
            problem.stacks,
            problem.max_tiers,
            time_limit,
            stop=self.stopped,
            on_progress=lambda elapsed, cost: self.report(
                min(elapsed / time_limit, 1.0), cost
            ),
        )
