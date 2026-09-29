"""
Step 1 of López-Plata et al. (2019): Algorithm 1 (p. 441-442) limited to depth
beta, and the selection of the most promising partial solutions (Sec. 6.1, p. 443).
Nodes are Bay objects carrying phi' and g (operating_cost.py).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from .measures import h, q
from .operating_cost import Bay


def algorithm1(root: Bay, ub: float, beta: Optional[int] = None) -> Tuple[List[Bay], Optional[Bay], float]:
    """Algorithm 1 with children placed at the front of N, i.e. depth-first. Only relocation
    children are tested against ub (line 18, O17). With beta, a node at depth beta (moves from
    the root, retrievals included) is collected instead of expanded (Sec. 6.1). A leaf at any
    depth <= beta updates ub and best (lines 8-10, O05).
    Returns the collected nodes N_beta, the best leaf (None if none beat ub) and the final ub."""
    frontier, collected, best = [root], [], None
    while frontier:
        n = frontier.pop()                                  # first(N), removed from N
        if n.target is None:
            continue
        if beta is not None and len(n.phi) == beta:
            collected.append(n)
            continue
        s = n.target_stack()
        if n.stacks[s - 1][-1] == n.target:                 # c* = c+: retrieve it
            child = n.copy()
            child.apply((s, 0))
            if child.target is None:                        # (i) c* = n_C: a leaf
                if child.g < ub:
                    ub, best = child.g, child
            else:                                           # (ii) at the front, not tested
                frontier.append(child)
            continue
        omega = []                                          # (iii) relocate c+ to each other non-full stack
        for d in range(1, len(n.stacks) + 1):
            if d != s and len(n.stacks[d - 1]) < n.cost.n_T:
                child = n.copy()
                child.apply((s, d))
                if child.g + h(child) < ub:
                    omega.append(child)
        omega.sort(key=lambda c: c.g)                       # stable: ties keep the stack order (O07)
        frontier.extend(reversed(omega))                    # the cheapest child is explored next
    return collected, best, ub


def select(nodes: List[Bay], lam: int, n_C: int) -> List[Bay]:
    """N = N_g + N_q: the lam nodes with the lowest g and the lam nodes with the lowest q
    (O06 (a), so |N| <= 2 lam). Stable sorts keep the order of discovery on ties (O07)."""
    n_g = sorted(nodes, key=lambda n: n.g)[:lam]
    n_q = sorted(nodes, key=lambda n: q(n, n_C))[:lam]
    taken = {id(n) for n in n_g}
    return n_g + [n for n in n_q if id(n) not in taken]
