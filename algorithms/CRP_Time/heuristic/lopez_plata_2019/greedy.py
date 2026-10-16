"""
The greedy of López-Plata et al. (2019), Sec. 7.1, which gives the upper bound for step 1, and
the depth beta of step 1 that follows from the length of its solution. Stacks are
numbered 1..n_S.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import List, Tuple

from .bay import Bay, Move


def greedy(bay: Bay) -> Tuple[List[Move], float]:
    """Greedy of Sec. 7.1 (p. 445) from the configuration of bay: relocate c+ to the
    stack with the lowest loaded-trip cost m' (drop tier n(s') + 1), ties to the lowest stack
    index (O19). Returns the moves and their cost."""
    bay = bay.copy()
    cost, phi, total = bay.cost, [], 0
    while bay.target is not None:
        s = bay.target_stack()
        stack = bay.stacks[s - 1]
        if stack[-1] == bay.target:
            move = (s, 0)
        else:
            dests = [d for d in range(1, len(bay.stacks) + 1)
                     if d != s and len(bay.stacks[d - 1]) < cost.n_T]
            if not dests:
                raise ValueError(f"greedy: no stack free for the block above {bay.target}")
            d = min(dests, key=lambda d: (cost.greedy_key(s, len(stack), d, len(bay.stacks[d - 1]) + 1), d))
            move = (s, d)
        total += bay.apply(move)
        phi.append(move)
    return phi, total


def p(n_C: int) -> Fraction:
    """Factor p of the depth beta (p. 446), with the end values swapped (O02 (c)): the printed ramp runs from 1.5
    at n_C = 9 to 0.5 at n_C = 25, so the printed ends 0.5 and 1.5 are read as a misprint."""
    if n_C <= 9:
        return Fraction(3, 2)
    if n_C < 25:
        return Fraction(3, 2) - Fraction(n_C - 9, 25 - 9)
    return Fraction(1, 2)


def beta(phi_len: int, n_C: int, theta: float = 40) -> int:
    """Depth beta of step 1 in moves (p. 446): theta percent of the length of the greedy solution, times p(n_C)."""
    return math.floor(phi_len * Fraction(theta) / 100 * p(n_C))
