"""
Greedy beam search for CRP-R
<2017> <metaheuristic> <restricted> <single-bay> <CRP-R>
beam_width --- 5 --- Nodes retained at each level
evaluator --- VRH --- Evaluation heuristic: VRH or SDH
time_limit --- 300.0 --- Time limit in seconds

------------------------------- Reference --------------------------------
C.-J. Ting, K.-C. Wu,
"Optimizing container relocation operations at container yards with
 beam search",
Transportation Research Part E 103 (2017) 17–31.
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


class GBS(Algorithm):

    def main(self, problem):
        time_limit = self.parameter("time_limit", 300.0, float)
        return search(
            problem.stacks,
            problem.n,
            problem.max_tiers,
            time_limit,
            width=self.parameter("beam_width", 5, int),
            evaluator=self.parameter("evaluator", "VRH", str),
            stop=self.stopped,
            on_progress=lambda elapsed, cost: self.report(
                min(elapsed / time_limit, 1.0), cost
            ),
        )
