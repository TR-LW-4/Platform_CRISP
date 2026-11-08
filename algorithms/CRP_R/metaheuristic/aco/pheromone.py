"""
Pheromone matrix tau[c][d][n][t] of JovanovicACO_CRPTime and its updates
(Sec. 5, p. 83–84): the initial value from the greedy solution, the local update
after every ant, the global update with the best solution after every iteration,
and the reinitialisation after MaxConst iterations without improvement.

A solution is a list of (c, d, n, t): container c relocated onto a stack with
extended minimum due date d, its n-th relocation (at most MaxMoves), while target
t is being retrieved. All four are 1-based except n.

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
corresponding original algorithm paper.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

Solution = List[Tuple[int, int, int, int]]


def val(cost: float, lb: float) -> float:
    """Quality of a solution from its crane time and the lower bound of the initial bay."""
    return 1.0 / max(cost - lb + 1.0, 1.0)


def initial(cost: float, lb: float, n_total: int, n_stacks: int, max_moves: int) -> Tuple[np.ndarray, float]:
    """The matrix filled with tau_0 from the greedy solution's cost, and tau_min."""
    tau_0 = val(cost, lb) / n_stacks
    tau_min = tau_0 / n_stacks
    return np.full((n_total, n_total + n_stacks, max_moves + 1, n_total), tau_0, dtype=np.float32), tau_min


def _cells(pheromone: np.ndarray, solution: Solution):
    """Matrix index of each element of the solution that lies inside the matrix."""
    n_total, n_d = pheromone.shape[0], pheromone.shape[1]
    for c, d_val, mc_rec, t in solution:
        c_i, d_i, t_i = c - 1, d_val - 1, t - 1
        if 0 <= c_i < n_total and 0 <= d_i < n_d and 0 <= t_i < n_total:
            yield c_i, d_i, mc_rec, t_i


def local_update(pheromone: np.ndarray, solution: Solution, phi: float, tau_min: float) -> None:
    """After every ant: evaporate the pheromone on its solution by phi, not below tau_min."""
    for cell in _cells(pheromone, solution):
        pheromone[cell] = max(np.float32(tau_min), np.float32(pheromone[cell] * phi))


def global_update(pheromone: np.ndarray, best: Solution, cost: float, lb: float, rho: float,
                  n_stacks: int) -> float:
    """After every iteration: move the pheromone on the best solution towards its quality at rate rho,
    not below the new tau_min, which is returned."""
    val_b = val(cost, lb)
    delta = np.float32(val_b)
    tau_min = val_b / (n_stacks ** 2)
    for cell in _cells(pheromone, best):
        pheromone[cell] = max(np.float32(tau_min), np.float32((1.0 - rho) * pheromone[cell] + rho * delta))
    return tau_min


def reinitialise(pheromone: np.ndarray, cost: float, lb: float, n_stacks: int) -> None:
    """After MaxConst iterations without improvement: fill the matrix with the value of the best solution."""
    pheromone[:] = np.float32(val(cost, lb) / n_stacks)
