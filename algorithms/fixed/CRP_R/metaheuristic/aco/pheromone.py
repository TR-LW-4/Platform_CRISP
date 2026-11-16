"""
Pheromone matrix for the CRP-R ant colony.

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

from typing import List, Tuple

import numpy as np

Solution = List[Tuple[int, int, int, int]]


def val(cost: float, lb: float) -> float:
    return 1.0 / max(cost - lb + 1.0, 1.0)


def initial(cost: float, lb: float, n_total: int, n_stacks: int, max_moves: int) -> Tuple[np.ndarray, float]:
    tau_0 = val(cost, lb) / n_stacks
    tau_min = tau_0 / n_stacks
    return np.full((n_total, n_total + n_stacks, max_moves + 1, n_total), tau_0, dtype=np.float32), tau_min


def _cells(pheromone: np.ndarray, solution: Solution):
    n_total, n_d = pheromone.shape[0], pheromone.shape[1]
    for c, d_val, mc_rec, t in solution:
        c_i, d_i, t_i = c - 1, d_val - 1, t - 1
        if 0 <= c_i < n_total and 0 <= d_i < n_d and 0 <= t_i < n_total:
            yield c_i, d_i, mc_rec, t_i


def local_update(pheromone: np.ndarray, solution: Solution, phi: float, tau_min: float) -> None:
    for cell in _cells(pheromone, solution):
        pheromone[cell] = max(np.float32(tau_min), np.float32(pheromone[cell] * phi))


def global_update(pheromone: np.ndarray, best: Solution, cost: float, lb: float, rho: float,
                  n_stacks: int) -> float:
    val_b = val(cost, lb)
    delta = np.float32(val_b)
    tau_min = val_b / (n_stacks ** 2)
    for cell in _cells(pheromone, best):
        pheromone[cell] = max(np.float32(tau_min), np.float32((1.0 - rho) * pheromone[cell] + rho * delta))
    return tau_min


def reinitialise(pheromone: np.ndarray, cost: float, lb: float, n_stacks: int) -> None:
    pheromone[:] = np.float32(val(cost, lb) / n_stacks)