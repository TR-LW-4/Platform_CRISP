"""
Iterative deepening A* for CRP-R
<2012> <exact> <restricted> <single-bay> <CRP-R>
lb_mode --- lb3 --- Lower bound: lb1, lb2, or lb3
probe_mode --- pr4 --- Probe heuristic: pr1, pr2, pr3, or pr4

------------------------------- Reference --------------------------------
W. Zhu, H. Qin, A. Lim, H. Zhang,
"Iterative deepening A* algorithms for the container relocation problem",
IEEE Transactions on Automation Science and Engineering 9 (2012) 710–722.
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


class Move:
    def __init__(self, container, source, destination):
        self.container = container
        self.source = source
        self.destination = destination


class IDA(Algorithm):

    def main(self):
        raw, _cost = search(
            self.problem.stacks,
            self.problem.max_tiers,
            lb_mode=self.parameter("lb_mode", "lb3", str),
            probe_mode=self.parameter("probe_mode", "pr4", str),
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