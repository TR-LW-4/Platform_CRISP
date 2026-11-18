"""
Ant colony search for CRP-R
<2019> <metaheuristic> <restricted> <single-bay> <CRP-R>
n_iterations --- 5000 --- Colony iterations per layout
n_ants --- 10 --- Ants per iteration
q0 --- 0.9 --- Exploitation rate
rho --- 0.1 --- Global update rate
phi --- 0.9 --- Local evaporation
max_moves --- 10 --- MaxMoves (pheromone depth)
max_const_iter --- 100 --- Stagnation threshold (MaxConst)
operator --- min_max --- Greedy completion rule
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

from __future__ import annotations

import numpy as np
import importlib

from algorithms.ALGORITHM import Algorithm

from .ant import build_solution, start
from .greedy import greedy
from .pheromone import global_update, initial, local_update, reinitialise
from algorithms.distinct.CRP_R.LB.lb1 import lb1

class Move:
    def __init__(self, container, source, destination):
        self.container = container
        self.source = source
        self.destination = destination


def decode(solution, stacks, n_total):
    state = [list(stack) for stack in stacks]
    loc = {
        container: index
        for index, stack in enumerate(state)
        for container in stack
    }
    by_target = {}
    for container, label, _count, target in solution:
        by_target.setdefault(target, []).append((container, label))

    moves = []
    for target in range(1, n_total + 1):
        for container, label in by_target.get(target, []):
            source = loc[container]
            destination = loc[label] if label <= n_total else label - n_total - 1
            moves.append(Move(
                container,
                (source, len(state[source])),
                (destination, len(state[destination]) + 1),
            ))
            state[source].pop()
            state[destination].append(container)
            loc[container] = destination
        source = loc[target]
        moves.append(Move(target, (source, len(state[source])), None))
        state[source].pop()
        loc[target] = None
    return moves

class ACO(Algorithm):

    def main(self):
        n_iters = self.parameter("n_iterations", 5000, int)
        n_ants = self.parameter("n_ants", 10, int)
        q0 = self.parameter("q0", 0.9, float)
        rho = self.parameter("rho", 0.1, float)
        phi = self.parameter("phi", 0.9, float)
        max_moves = self.parameter("max_moves", 10, int)
        max_const = self.parameter("max_const_iter", 100, int)
        np.random.seed(self.parameter("seed", 0, int))
        
        

        stacks_init = self.problem.stacks
        n_total = self.problem.num_containers
        max_tiers = self.problem.max_tiers
        n_stacks = len(stacks_init)
        
        lower_bound = float(lb1(stacks_init, max_tiers))
        
        name = self.parameter("operator", "min_max", str)
        module = importlib.import_module("algorithms.distinct.CRP_R.operator." + name)
        choose_destination = module.choose_destination
        best = greedy(stacks_init, n_total, max_tiers, max_moves, choose_destination)

        ant0 = start(stacks_init, max_tiers)
        best_cost = float(len(best))
        pheromone, tau_min = initial(best_cost, lower_bound, n_total, n_stacks, max_moves)
        idle = 0

        for iteration in range(n_iters):
            improved = False
            for _ in range(n_ants):
                solution, valid, _cost = build_solution(
                    ant0, pheromone, q0, max_moves, best_cost,
                )
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
        return decode(best, stacks_init, n_total)
