"""
Best-first branch-and-bound for CRP-R
<2015> <exact> <restricted> <single-bay> <CRP-R>
alpha --- 0 --- Maximum number of assignments kept at a node. 0 keeps every assignment.

------------------------------- Reference --------------------------------
C. Expósito-Izquierdo, B. Melián-Batista, J.M. Moreno-Vega,
"An exact approach for the Blocks Relocation Problem",
Expert Systems with Applications 42 (2015) 6408–6422.
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


class BFBB(Algorithm):

    def main(self):
        return search(
            self.problem.stacks,
            self.problem.num_containers,
            self.problem.max_tiers,
            alpha=self.parameter("alpha", 0, int),
        )