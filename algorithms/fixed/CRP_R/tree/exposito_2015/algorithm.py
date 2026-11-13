"""
Expósito-Izquierdo branch-and-bound for CRP-R
<2015> <exact> <restricted> <single-bay> <CRP-R>
alpha --- 0 --- Max children per node (0 = unlimited)
time_limit --- 600.0 --- Per-instance wall-clock limit (s)

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

from algorithms.CRP_R.ALGORITHM import Algorithm

from .search import search


class ExposioBB2015(Algorithm):

    def main(self):
        time_limit = self.parameter("time_limit", 600.0, float)
        return search(
            problem.stacks,
            problem.n,
            problem.max_tiers,
            time_limit,
            alpha=self.parameter("alpha", 0, int),
            stop=self.stopped,
            on_progress=lambda elapsed, cost: self.report(
                min(elapsed / time_limit, 1.0), cost
            ),
        )
