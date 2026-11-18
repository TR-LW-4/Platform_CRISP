"""
Greedy beam search for CRP-R
<2017> <metaheuristic> <restricted> <single-bay> <CRP-R>
beam_width --- 5 --- Nodes retained at each level
operator --- vrh --- Completion rule: vrh or sdh

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

from algorithms.ALGORITHM import Algorithm

from .search import search


class GBS(Algorithm):

    def main(self):
        return search(
            self.problem.stacks,
            self.problem.num_containers,
            self.problem.max_tiers,
            width=self.parameter("beam_width", 5, int),
            evaluator=self.parameter("operator", "vrh", str),
        )