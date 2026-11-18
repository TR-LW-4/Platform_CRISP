"""
One ant retrieves the bay. Each relocation is chosen from the Gre-C (or Gre-N)
candidate list by the pheromone transition rule.

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

from typing import Tuple

import numpy as np

from algorithms.distinct.CRP_U.LB.lb1_u import lb1_u
from algorithms.distinct.CRP_U.operator.grec import candidates

from .pheromone import Solution


def label(stacks, destination, n_total) -> int:
    return min(stacks[destination]) if stacks[destination] else n_total + destination + 1


def build_solution(stacks, max_tiers, pheromone, q0, max_moves, best_cost, extended,
                   rng) -> Tuple[Solution, bool]:
    n_total = pheromone.shape[0]
    state = [list(stack) for stack in stacks]
    where = {container: index for index, stack in enumerate(state) for container in stack}
    moved = [0] * (n_total + 1)
    bound = lb1_u(state, max_tiers)
    solution: Solution = []

    for target in range(1, n_total + 1):
        home = where[target]
        while state[home][-1] != target:
            options = candidates(state, max_tiers, extended)
            if not options:
                return solution, False
            weights, labels = [], []
            for source, destination, dife in options:
                container = state[source][-1]
                labels.append(label(state, destination, n_total))
                count = min(moved[container], max_moves)
                weights.append(pheromone[container - 1, labels[-1] - 1, count, target - 1] / (1.0 + dife))
            if rng.random() < q0:
                picked = int(np.argmax(weights))
            else:
                weights = np.asarray(weights, dtype=np.float64)
                picked = int(rng.choice(len(weights), p=weights / weights.sum()))
            source, destination, _ = options[picked]
            container = state[source].pop()
            if state[source] and min(state[source]) < container:
                bound -= 1
            if state[destination] and min(state[destination]) < container:
                bound += 1
            state[destination].append(container)
            where[container] = destination
            solution.append((container, labels[picked], min(moved[container], max_moves), target))
            moved[container] += 1
            if len(solution) + bound >= best_cost:
                return solution, False
        state[home].pop()
    return solution, True
