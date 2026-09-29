"""
Estimates and parameters of López-Plata et al. (2019) used by the search:
h (eq. 21-22), q (eq. 23-26), the greedy of Sec. 7.1 and beta (eq. 29-30).
All work on a Bay (operating_cost.py); stacks are numbered 1..n_S.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import List, Tuple

from .operating_cost import Bay, Move


def non_located(stack: List[int]) -> List[bool]:
    """Per block, bottom to top: True if some block below it has an earlier
    retrieval order (p. 437); a well-located block has only later ones below."""
    flags, lowest = [], math.inf
    for c in stack:
        flags.append(lowest < c)
        lowest = min(lowest, c)
    return flags


def h(bay: Bay) -> float:
    """Estimate of the remaining operating cost, summed over the blocks present (Sec. 5.1, p. 442).
    EstRet (eq. 21): descend from n_T + epsilon to c, carry c to the retrieval point.
    EstRel (eq. 22): EstRet plus one relocation to the top tier n_T of the adjacent stack
    towards the retrieval point, s(c) - 1, or s(c) + 1 if s(c) = 1."""
    cost = bay.cost
    top = cost.n_T + cost.epsilon
    total = 0.0
    for s, stack in enumerate(bay.stacks, 1):
        for t, bad in enumerate(non_located(stack), 1):
            descend = cost.m(s, top, s, t)
            total += descend + cost.m_loaded(s, t, 0, 1)
            if bad:
                total += descend + cost.m_loaded(s, t, s + 1 if s == 1 else s - 1, cost.n_T)
    return total


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


def greedy(bay: Bay) -> Tuple[List[Move], float]:
    """Greedy of Sec. 7.1 (p. 445) from the configuration of bay: relocate c+ to the
    stack with the lowest m' (eq. 2, drop tier n(s') + 1), ties to the lowest stack
    index (O19). Returns the moves and their cost (eq. 3)."""
    bay = bay.copy()
    cost, phi, total = bay.cost, [], 0.0
    while bay.target is not None:
        s = next(i for i, st in enumerate(bay.stacks, 1) if bay.target in st)
        stack = bay.stacks[s - 1]
        if stack[-1] == bay.target:
            move = (s, 0)
        else:
            dests = [d for d in range(1, len(bay.stacks) + 1)
                     if d != s and len(bay.stacks[d - 1]) < cost.n_T]
            if not dests:
                raise ValueError(f"greedy: no stack free for the block above {bay.target}")
            d = min(dests, key=lambda d: (cost.m_loaded(s, len(stack), d, len(bay.stacks[d - 1]) + 1), d))
            move = (s, d)
        total += bay.apply(move)
        phi.append(move)
    return phi, total


def p(n_C: int) -> Fraction:
    """Eq. 30 (p. 446) as printed; it jumps at n_C = 9 and 25 (O02)."""
    if n_C <= 9:
        return Fraction(1, 2)
    if n_C < 25:
        return Fraction(3, 2) - Fraction(n_C - 9, 25 - 9)
    return Fraction(3, 2)


def beta(phi_len: int, n_C: int, theta: float = 40) -> int:
    """Eq. 29 (p. 446): depth of step 1 in moves, from the length of the greedy solution."""
    return math.floor(phi_len * Fraction(theta) / 100 * p(n_C))
