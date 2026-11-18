"""
Pattern database search for CRP-R
<2016> <exact> <restricted> <single-bay> <CRP-R>
cache_level --- 15 --- Node caching depth n (CC_n)
pdb_depth --- 0 --- Pattern database size k in containers (PDB_k)
max_cache_size --- 20000000 --- Maximum cached abstract states (Pmax_size)
time_limit --- 0 --- Time limit in seconds. 0 runs until optimal.

------------------------------- Reference --------------------------------
D. Ku, T.S. Arthanari,
"On the abstraction method for the container relocation problem",
Computers & Operations Research 68 (2016) 110–122.
Initial upper bound: best of the PR1–PR4 probe heuristics of Zhu et al. (2012).
------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from algorithms.ALGORITHM import Algorithm

from .levels import load_pdb
from .search import search


class Move:
    def __init__(self, container, source, destination):
        self.container = container
        self.source = source
        self.destination = destination


class PDB(Algorithm):

    def main(self):
        levels = load_pdb(
            self.parameter("pdb_depth", 0, int),
            len(self.problem.stacks),
            self.problem.max_tiers,
        )
        raw, _cost = search(
            self.problem.stacks,
            self.problem.max_tiers,
            levels,
            cache_level=self.parameter("cache_level", 15, int),
            max_cache_size=self.parameter("max_cache_size", 20000000, int),
            time_limit=self.parameter("time_limit", 0.0, float),
        )
        return _as_moves(self.problem.stacks, raw)


def _as_moves(stacks, raw) -> list:
    state = [list(stack) for stack in stacks]
    moves = []
    for container, source, destination in raw:
        source_tier = len(state[source])
        state[source].pop()
        if destination is None:
            moves.append(Move(container, (source, source_tier), None))
            continue
        landing = len(state[destination]) + 1
        state[destination].append(container)
        moves.append(Move(container, (source, source_tier), (destination, landing)))
    return moves
