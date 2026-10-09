"""
Step 1 of López-Plata et al. (2019): Algorithm 1 (p. 441-442) limited to depth
beta, and the selection of the most promising partial solutions (Sec. 6.1, p. 443).
Nodes are Bay objects carrying phi' and g (operating_cost.py).
"""

from __future__ import annotations

import heapq
import math
from typing import Callable, Dict, List, Optional, Tuple

from .bay import Bay
from .estimates import block_term, h, q


class BudgetExceeded(Exception):
    """Raised by algorithm1 when max_visited nodes have been taken from N."""


def algorithm1(root: Bay, ub: float, beta: Optional[int] = None,
               collect: Optional[Callable[[Bay], None]] = None,
               stats: Optional[Dict[str, int]] = None,
               max_visited: Optional[int] = None) -> Tuple[List[Bay], Optional[Bay], float]:
    """Algorithm 1 with children placed at the front of N, i.e. depth-first. Only relocation
    children are tested against ub (line 18, O17). With beta, a node at depth beta (moves from
    the root, retrievals included) is collected instead of expanded (Sec. 6.1). A leaf at any
    depth <= beta updates ub and best (lines 8-10, O05).
    Returns the collected nodes N_beta, the best leaf (None if none beat ub) and the final ub.
    With collect, each node at depth beta is passed to it instead (the node is then modified
    further, so collect must copy what it keeps) and the returned list is empty. With stats,
    stats["visited"] counts the nodes taken from N. With max_visited, BudgetExceeded is raised
    instead of taking node max_visited + 1 from N.

    Implemented as a recursion on one Bay with apply/undo: the children of a node are generated,
    tested with the ub of that moment and sorted before any of them is explored, which is the order
    of the list N in Algorithm 1. h is updated per move: a relocation only changes the term of the
    moved block, a retrieval only removes the term of c*."""
    bay, collected = root.copy(), []
    best: Optional[Bay] = None
    cost = bay.cost
    if collect is None:
        collect = lambda node: collected.append(node.copy())  # noqa: E731
    if stats is not None:
        stats.setdefault("visited", 0)
    visited = 0

    def visit(h_n: float) -> None:
        nonlocal ub, best, visited
        if max_visited is not None and visited >= max_visited:
            raise BudgetExceeded
        visited += 1
        if stats is not None:
            stats["visited"] += 1
        if bay.target is None:
            return
        if beta is not None and len(bay.phi) == beta:
            collect(bay)
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


class RunningSelection:
    """Collector for algorithm1 that keeps only what select() would return: the lam nodes
    with the lowest g and the lam nodes with the lowest q seen so far, ties to the node found
    first. Memory stays of the order of lam however many nodes reach depth beta."""

    def __init__(self, lam: int, n_C: int):
        self.lam, self.n_C, self.count = lam, n_C, 0
        self._by_g: List[tuple] = []   # max-heaps of (-key, -index, node): the worst kept node on top
        self._by_q: List[tuple] = []

    def _keeps(self, heap: List[tuple], key: float, index: int) -> bool:
        return len(heap) < self.lam or (key, index) < (-heap[0][0], -heap[0][1])

    def _add(self, heap: List[tuple], key: float, index: int, node: Bay) -> None:
        entry = (-key, -index, node)
        if len(heap) < self.lam:
            heapq.heappush(heap, entry)
        else:
            heapq.heapreplace(heap, entry)

    def __call__(self, bay: Bay) -> None:
        index = self.count
        self.count += 1
        g, q_n = bay.g, q(bay, self.n_C)
        node = None
        if self._keeps(self._by_g, g, index):
            node = bay.copy()
            self._add(self._by_g, g, index, node)
        if self._keeps(self._by_q, q_n, index):
            self._add(self._by_q, q_n, index, node or bay.copy())

    def result(self) -> List[Bay]:
        """N = N_g + N_q in the order of select()."""
        n_g = [e[2] for e in sorted(self._by_g, key=lambda e: (-e[0], -e[1]))]
        n_q = [e[2] for e in sorted(self._by_q, key=lambda e: (-e[0], -e[1]))]
        taken = {id(n) for n in n_g}
        return n_g + [n for n in n_q if id(n) not in taken]


def partial_solutions(root: Bay, ub: float, beta: int, lam: int, n_C: int,
                      node_budget: int) -> Tuple[List[Bay], Optional[Bay], float, Dict[str, int]]:
    """Step 1 under a node budget (L18): iterative deepening d = 1, ..., beta, each iteration a full
    run of algorithm1 to depth d from ub with the running selection. The budget counts the nodes
    visited over all iterations; when it runs out during iteration d, the result of depth d - 1
    is returned (depth 0: the root). If it does not run out, the result is that of algorithm1 to
    depth beta. Returns N, the best leaf, the final ub and beta_eff, budget_hit, visited_total,
    visited_last (the completed iteration) and n_beta (|N_beta| at beta_eff)."""
    nodes, best, final_ub = ([root.copy()] if root.target is not None else []), None, ub
    info = {"beta_eff": 0, "budget_hit": False, "visited_total": 0, "visited_last": 0, "n_beta": len(nodes)}
    for d in range(1, beta + 1):
        selection, stats = RunningSelection(lam, n_C), {"visited": 0}
        try:
            _, leaf, leaf_ub = algorithm1(root, ub, d, selection, stats, node_budget - info["visited_total"])
        except BudgetExceeded:
            info["visited_total"] += stats["visited"]
            info["budget_hit"] = True
            break
        info["visited_total"] += stats["visited"]
        nodes, best, final_ub = selection.result(), leaf, leaf_ub
        info.update(beta_eff=d, visited_last=stats["visited"], n_beta=selection.count)
    return nodes, best, final_ub, info


def select(nodes: List[Bay], lam: int, n_C: int) -> List[Bay]:
    """N = N_g + N_q: the lam nodes with the lowest g and the lam nodes with the lowest q
    (O06 (a), so |N| <= 2 lam). Stable sorts keep the order of discovery on ties (O07)."""
    n_g = sorted(nodes, key=lambda n: n.g)[:lam]
    n_q = sorted(nodes, key=lambda n: q(n, n_C))[:lam]
    taken = {id(n) for n in n_g}
    return n_g + [n for n in n_q if id(n) not in taken]
