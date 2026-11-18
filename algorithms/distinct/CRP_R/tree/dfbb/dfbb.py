"""
Kim and Hong branch-and-bound for CRP-R
<2006> <exact> <restricted> <single-bay> <CRP-R>

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

from algorithms.ALGORITHM import Algorithm

from .search import search


class DFBB(Algorithm):

    def main(self):
        return search(
            self.problem.stacks,
            self.problem.max_tiers,
        )