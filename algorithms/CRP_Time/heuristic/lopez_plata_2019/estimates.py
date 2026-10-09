"""
Estimates of López-Plata et al. (2019) used by the search: h, the estimate of the remaining cost
(eq. 21-22), and q, the quality of a configuration (eq. 23-26). Both work on a Bay; stacks are
numbered 1..n_S.
"""

from __future__ import annotations

from functools import lru_cache

from algorithms.CRP_Time.utils.located import non_located

from .bay import Bay
from .operating_cost import CostModel


@lru_cache(maxsize=None)
def block_term(cost: CostModel, s: int, t: int, bad: bool) -> float:
    """Term of the block at (s, t) in h, from the cost model (Sec. 5.1, p. 442)."""
    return cost.h_term(s, t, bad)


def h(bay: Bay) -> float:
    """Estimate of the remaining cost under the cost model: block_term summed over the blocks present."""
    return sum(block_term(bay.cost, s, t, bad)
               for s, stack in enumerate(bay.stacks, 1) for t, bad in enumerate(non_located(stack), 1))


def q(bay: Bay, n_C: int) -> float:
    """Quality of a configuration (eq. 23-26, p. 443-444), lower is better:
    sum over the blocks c of q_b(c) + q_c(c) / n_C + q_r(c) / maxDist, with n_C the
    number of blocks of the instance and maxDist = max_s d(s, 0) = n_S."""
    max_dist = len(bay.stacks)
    total = 0.0
    for s, stack in enumerate(bay.stacks, 1):
        for c, bad in zip(stack, non_located(stack)):
            q_c = abs(c - min(stack)) if len(stack) > 1 else c - bay.target   # eq. 24
            total += bad + q_c / n_C + s / max_dist                           # eq. 23, 25
    return total
