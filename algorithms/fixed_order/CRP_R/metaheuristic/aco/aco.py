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

from algorithms.CRP_R.ALGORITHM import Algorithm

from .ant import build_solution, start
from .greedy import greedy
from .pheromone import global_update, initial, local_update, reinitialise
from .scoring import compute_lb


class ACO(Algorithm):

    def main(self, problem):
        n_iters = self.parameter("n_iterations", 5000, int)
        n_ants = self.parameter("n_ants", 10, int)
        q0 = self.parameter("q0", 0.9, float)
        rho = self.parameter("rho", 0.1, float)
        phi = self.parameter("phi", 0.9, float)
        max_moves = self.parameter("max_moves", 10, int)
        max_const = self.parameter("max_const_iter", 100, int)
        report_every = max(1, n_iters // 50)
        np.random.seed(getattr(self.config, "seed", 0))

        stacks_init = problem.stacks
        n_total = problem.n
        n_stacks = len(stacks_init)
        max_tiers = problem.max_tiers
        lb = float(compute_lb({i: stack for i, stack in enumerate(stacks_init)}))
        ant0 = start(stacks_init, max_tiers)

        best = greedy(stacks_init, n_total, max_tiers, max_moves)
        best_cost = float(len(best))
        pheromone, tau_min = initial(best_cost, lb, n_total, n_stacks, max_moves)
        idle = 0

        for iteration in range(n_iters):
            if self.stopped():
                break
            improved = False
            for _ in range(n_ants):
                if self.stopped():
                    break
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
                reinitialise(pheromone, best_cost, lb, n_stacks)
                idle = 0
            tau_min = global_update(pheromone, best, best_cost, lb, rho, n_stacks)
            if (iteration + 1) % report_every == 0:
                self.report((iteration + 1) / n_iters, best_cost)

        return best, best_cost
