"""
Ant colony search for CRP-U
<2019> <metaheuristic> <unrestricted> <single-bay> <CRP-U>
n_iterations --- 1000 --- Colony iterations per layout
n_ants --- 10 --- Ants per iteration
q0 --- 0.9 --- Exploitation rate
rho --- 0.1 --- Global update rate
phi --- 0.9 --- Local evaporation
max_moves --- 10 --- MaxMoves (pheromone depth)
max_const_iter --- 100 --- Stagnation threshold (MaxConst) before reinitialising the pheromone
variant --- C --- C (ACO-C, built on Gre-C) or N (ACO-N, built on Gre-N)
seed --- 0 --- Random seed

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

import numpy as np

from algorithms.ALGORITHM import Algorithm
from algorithms.distinct.CRP_U.LB.lb1_u import lb1_u
from algorithms.distinct.CRP_U.operator import grec, gren

from .ant import build_solution
from .greedy import greedy
from .pheromone import global_update, initial, local_update, reinitialise


class Move:
    def __init__(self, container, source, destination):
        self.container = container
        self.source = source
        self.destination = destination


class ACOU(Algorithm):

    def main(self):
        n_iters = self.parameter("n_iterations", 1000, int)
        n_ants = self.parameter("n_ants", 10, int)
        q0 = self.parameter("q0", 0.9, float)
        rho = self.parameter("rho", 0.1, float)
        phi = self.parameter("phi", 0.9, float)
        max_moves = self.parameter("max_moves", 10, int)
        max_const = self.parameter("max_const_iter", 100, int)
        extended = self.parameter("variant", "C", str).upper() == "C"
        rng = np.random.default_rng(self.parameter("seed", 0, int))

        stacks = [list(stack) for stack in self.problem.stacks]
        n_total = self.problem.num_containers
        max_tiers = self.problem.max_tiers
        n_stacks = len(stacks)
        lower_bound = float(lb1_u(stacks, max_tiers))

        choose = grec.choose_move if extended else gren.choose_move
        best = greedy(stacks, max_tiers, n_total, max_moves, choose)
        best_cost = float(len(best))
        pheromone, tau_min = initial(best_cost, lower_bound, n_total, n_stacks, max_moves)
        idle = 0

        for _ in range(n_iters):
            improved = False
            for _ in range(n_ants):
                solution, valid = build_solution(stacks, max_tiers, pheromone, q0, max_moves, best_cost,
                                                 extended, rng)
                local_update(pheromone, solution, phi, tau_min)
                if valid and len(solution) < best_cost:
                    best = solution
                    best_cost = float(len(solution))
                    improved = True
                    idle = 0
            if not improved:
                idle += 1
            if idle >= max_const:
                reinitialise(pheromone, best_cost, lower_bound, n_stacks)
                idle = 0
            tau_min = global_update(pheromone, best, best_cost, lower_bound, rho, n_stacks)
        return _decode(best, stacks, n_total)


def _decode(solution, stacks, n_total):
    state = [list(stack) for stack in stacks]
    where = {container: index for index, stack in enumerate(state) for container in stack}
    by_target = {}
    for container, destination_label, _, target in solution:
        by_target.setdefault(target, []).append((container, destination_label))

    moves = []
    for target in range(1, n_total + 1):
        for container, destination_label in by_target.get(target, []):
            source = where[container]
            destination = (where[destination_label] if destination_label <= n_total
                           else destination_label - n_total - 1)
            moves.append(Move(container, (source, len(state[source])),
                              (destination, len(state[destination]) + 1)))
            state[source].pop()
            state[destination].append(container)
            where[container] = destination
        source = where[target]
        moves.append(Move(target, (source, len(state[source])), None))
        state[source].pop()
    return moves
