"""
One Gre-C (or Gre-N) pass for the CRP-U ant colony.
Each record is (container, label, count, target).

------------------------------- Reference --------------------------------
R. Jovanović, M. Tuba, S. Voß,
"An efficient ant colony optimization algorithm for the blocks relocation
 problem",
European Journal of Operational Research 274 (2019) 78–90.
------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section.
--------------------------------------------------------------------------
"""

from .ant import label
from .pheromone import Solution


def greedy(stacks, max_tiers, n_total, max_moves, choose) -> Solution:
    state = [list(stack) for stack in stacks]
    moved = [0] * (n_total + 1)
    solution: Solution = []
    for target in range(1, n_total + 1):
        home = next(index for index, stack in enumerate(state) if target in stack)
        while state[home][-1] != target:
            chosen = choose(state, max_tiers)
            if chosen is None:
                raise ValueError(f"greedy cannot retrieve container {target}")
            source, destination = chosen
            container = state[source][-1]
            solution.append((container, label(state, destination, n_total),
                             min(moved[container], max_moves), target))
            state[destination].append(state[source].pop())
            moved[container] += 1
        state[home].pop()
    return solution
