"""
Crane working time of Shin et al. (2026) for multi-bay plans.

W.-J. Shin et al., Transportation Research Part C 183 (2026) 105496. The model
follows Shin's public code (github.com/operagang/CRP_RL, env/env.py at commit
0d99c0c), decided by Wei as the multi-bay comparison metric:
- gamma_row 1.2, gamma_bay 3.5, gamma_acc 40, gamma_pd 30 (env.py:7-10; p. 12);
- positions (bay, row) from 1; the retrieval point of a bay is its row 0 (p. 4, A1-A2);
- a trip from (i, j) to (i', j') takes gamma_acc * [i != i'] + gamma_bay * |i - i'|
  + gamma_row * |j - j'| (eq. (1), p. 5).

A layout is layout[b][s] = retrieval orders of the containers in bay b, stack (row) s,
bottom to top, both indices from 0; retrieval orders are unique (p. 4).

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite
W.-J. Shin et al., Transportation Research Part C 183 (2026) 105496.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

Position = Tuple[int, int]
Layout = Sequence[Sequence[Sequence[int]]]


@dataclass(frozen=True)
class ShinParams:
    """p. 12 and env.py:7-10."""
    row: float = 1.2
    bay: float = 3.5
    acc: float = 40.0
    pd: float = 30.0


def trip(a: Position, b: Position, p: ShinParams = ShinParams()) -> float:
    """Eq. (1), p. 5."""
    return (p.acc if a[0] != b[0] else 0.0) + p.bay * abs(a[0] - b[0]) + p.row * abs(a[1] - b[1])


def _positions(layout: Layout) -> Dict[int, Position]:
    """Retrieval order -> (bay, row) from 1."""
    pos: Dict[int, Position] = {}
    for b, bay in enumerate(layout):
        for s, stack in enumerate(bay):
            for x in stack:
                if x in pos:
                    raise ValueError(f"retrieval order {x} occurs twice")
                pos[x] = (b + 1, s + 1)
    return pos


def lb_shin(layout: Layout, p: ShinParams = ShinParams()) -> float:
    """Theorem 2, p. 6: LB = LB_ret + LB_rel.

    LB_ret: for n = 1..N-1 the trip from (b_n, r_n) to (b_n, 0) and on to (b_n+1, r_n+1),
    the trip of container N to its retrieval point, and N * gamma_pd.
    LB_rel: (2 gamma_row + gamma_pd) per container with a smaller retrieval order below it.
    """
    pos = _positions(layout)
    order = sorted(pos)
    lb_ret = 0.0
    for n, nxt in zip(order, order[1:]):
        b, r = pos[n]
        lb_ret += trip((b, r), (b, 0), p) + trip((b, 0), pos[nxt], p)
    if order:
        b, r = pos[order[-1]]
        lb_ret += trip((b, r), (b, 0), p)
    lb_ret += len(order) * p.pd
    blocking = sum(1 for bay in layout for stack in bay
                   for k in range(1, len(stack)) if min(stack[:k]) < stack[k])
    return lb_ret + (2 * p.row + p.pd) * blocking
