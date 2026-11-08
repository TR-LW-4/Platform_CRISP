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

import multiprocessing as mp
from typing import Callable, Optional

import numpy as np

from web.backend.header_schema import parse_header

from .ant import build_solution, start
from .greedy import greedy
from .pheromone import global_update, initial, local_update, reinitialise
from .scoring import compute_lb


class ACO:

    def __init__(self, config=None):
        self.config = config
        self._best_solution = None

    def _take(self, header, name, cast):
        extra = getattr(self.config, "extra", None) or {}
        if name in extra and extra[name] not in (None, ""):
            return cast(extra[name])
        return cast(header[name]["default"])

    def train(
        self,
        problem_factory: Callable,
        result_queue: mp.Queue,
        stop_event: mp.Event,
    ) -> None:
        header = parse_header(__doc__ or "")
        n_iters = self._take(header, "n_iterations", int)
        n_ants = self._take(header, "n_ants", int)
        q0 = self._take(header, "q0", float)
        rho = self._take(header, "rho", float)
        phi = self._take(header, "phi", float)
        max_moves = self._take(header, "max_moves", int)
        max_const = self._take(header, "max_const_iter", int)
        report_every = max(1, n_iters // 50)
        np.random.seed(getattr(self.config, "seed", 0))

        env = problem_factory()
        stacks_init = [
            [int(c.priority) for c in stack.containers]
            for stack in env.yard.stacks.values()
        ]
        n_total = int(env.config.num_containers)
        n_stacks = len(stacks_init)
        max_tiers = int(env.config.max_tiers)
        lb = float(compute_lb({i: stack for i, stack in enumerate(stacks_init)}))
        ant0 = start(stacks_init, max_tiers)

        best = greedy(stacks_init, n_total, max_tiers, max_moves)
        best_cost = float(len(best))
        pheromone, tau_min = initial(best_cost, lb, n_total, n_stacks, max_moves)
        idle = 0

        for iteration in range(n_iters):
            if stop_event.is_set():
                break
            improved = False
            for _ in range(n_ants):
                if stop_event.is_set():
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
                result_queue.put({
                    "step": iteration + 1,
                    "metric": best_cost,
                    "progress": (iteration + 1) / n_iters,
                })

        self._best_solution = best
        result_queue.put({
            "step": n_iters,
            "metric": best_cost,
            "progress": 1.0,
            "solution": best,
        })

    def get_best_solution(self) -> Optional[list]:
        return self._best_solution