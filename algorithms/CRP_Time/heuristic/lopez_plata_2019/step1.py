"""
Step 1 of López-Plata et al. (2019): Algorithm 1 (p. 441-442) limited to depth
beta, and the selection of the most promising partial solutions (Sec. 6.1, p. 443).
Nodes are Bay objects carrying phi' and g (operating_cost.py).
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from .measures import block_term, h, q
from .operating_cost import Bay


def algorithm1(root: Bay, ub: float, beta: Optional[int] = None) -> Tuple[List[Bay], Optional[Bay], float]:
    """Algorithm 1 with children placed at the front of N, i.e. depth-first. Only relocation
    children are tested against ub (line 18, O17). With beta, a node at depth beta (moves from
    the root, retrievals included) is collected instead of expanded (Sec. 6.1). A leaf at any
    depth <= beta updates ub and best (lines 8-10, O05).
    Returns the collected nodes N_beta, the best leaf (None if none beat ub) and the final ub.

    Implemented as a recursion on one Bay with apply/undo: the children of a node are generated,
    tested with the ub of that moment and sorted before any of them is explored, which is the order
    of the list N in Algorithm 1. h is updated per move: a relocation only changes the term of the
    moved block, a retrieval only removes the term of c*."""
    bay, collected = root.copy(), []
    best: Optional[Bay] = None
    cost = bay.cost

    def visit(h_n: float) -> None:
        nonlocal ub, best
        if bay.target is None:
            return
        if beta is not None and len(bay.phi) == beta:
            collected.append(bay.copy())
            return
        s = bay.target_stack()
        stack = bay.stacks[s - 1]
        before = bay.crane, bay.target, bay.g
        if stack[-1] == bay.target:                             # c* = c+: retrieve it
            t = len(stack)
            bay.apply((s, 0))
            if bay.target is None:                              # (i) c* = n_C: a leaf
                if bay.g < ub:
                    ub, best = bay.g, bay.copy()
            else:                                               # (ii) explored next, not tested
                visit(h_n - block_term(cost, s, t, False))
            bay.undo((s, 0), *before)
            return
        block, t = stack[-1], len(stack)                        # (iii) relocate c+ to each other non-full stack
        h_rest = h_n - block_term(cost, s, t, min(stack[:-1], default=math.inf) < block)
        omega = []
        for d in range(1, len(bay.stacks) + 1):
            dest = bay.stacks[d - 1]
            if d != s and len(dest) < cost.n_T:
                h_child = h_rest + block_term(cost, d, len(dest) + 1, min(dest, default=math.inf) < block)
                bay.apply((s, d))
                g_child = bay.g
                bay.undo((s, d), *before)
                if g_child + h_child < ub:
                    omega.append((g_child, d, h_child))
        omega.sort(key=lambda child: child[0])                  # stable: ties keep the stack order (O07)
        for _, d, h_child in omega:                             # the cheapest child is explored first
            bay.apply((s, d))
            visit(h_child)
            bay.undo((s, d), *before)

    visit(h(bay))
    return collected, best, ub


def select(nodes: List[Bay], lam: int, n_C: int) -> List[Bay]:
    """N = N_g + N_q: the lam nodes with the lowest g and the lam nodes with the lowest q
    (O06 (a), so |N| <= 2 lam). Stable sorts keep the order of discovery on ties (O07)."""
    n_g = sorted(nodes, key=lambda n: n.g)[:lam]
    n_q = sorted(nodes, key=lambda n: q(n, n_C))[:lam]
    taken = {id(n) for n in n_g}
    return n_g + [n for n in n_q if id(n) not in taken]
