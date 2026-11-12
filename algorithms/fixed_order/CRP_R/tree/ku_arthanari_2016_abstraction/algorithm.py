"""
Ku and Arthanari abstraction search for CRP-R
<2016> <exact> <restricted> <single-bay> <CRP-R>
time_limit --- 300.0 --- Per-instance wall-clock limit (s)
pdb_depth --- 5 --- PDB remaining-container depth
cache_depth --- 20 --- Forward cache depth
max_cache_size --- 1000000 --- Max forward-cache entries
pdb_build_time_limit --- 60.0 --- PDB build time limit (s)

------------------------------- Reference --------------------------------
D. Ku, T.S. Arthanari,
"On the abstraction method for the container relocation problem",
Computers & Operations Research 68 (2016) 110–122.
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


class KuArthanari2016Abstraction(Algorithm):

    def main(self, problem):
        time_limit = self.parameter("time_limit", 300.0, float)
        return search(
            problem.stacks,
            problem.n,
            problem.max_tiers,
            time_limit,
            pdb_depth=self.parameter("pdb_depth", 5, int),
            cache_depth=self.parameter("cache_depth", 20, int),
            max_cache_size=self.parameter("max_cache_size", 1000000, int),
            pdb_build_time_limit_s=self.parameter("pdb_build_time_limit", 60.0, float),
            stop=self.stopped,
            on_progress=lambda elapsed, cost: self.report(
                min(elapsed / time_limit, 1.0), cost
            ),
        )
