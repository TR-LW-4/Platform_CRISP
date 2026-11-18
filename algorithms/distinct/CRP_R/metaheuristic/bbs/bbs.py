"""
Bounded beam search for CRP-R
<2019> <metaheuristic> <restricted> <single-bay> <CRP-R>
time_limit --- 5.0 --- Time limit in seconds
operator --- chain_f --- Completion rule

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

import importlib

from algorithms.ALGORITHM import Algorithm

from .search import search


class BBS(Algorithm):

    def main(self):
        name = self.parameter("operator", "chain_f", str)
        module = importlib.import_module("algorithms.distinct.CRP_R.operator." + name)
        choose_destination = module.choose_destination
        time_limit = self.parameter("time_limit", 5.0, float)
        return search(
            self.problem.stacks,
            self.problem.num_containers,
            self.problem.max_tiers,
            time_limit,
            choose_destination,
        )
