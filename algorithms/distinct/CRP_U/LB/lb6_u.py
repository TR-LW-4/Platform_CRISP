"""
LB-PS lower bound for CRP-U

------------------------------- Reference --------------------------------
B. Jin, S. Tanaka,
"An exact algorithm for the unrestricted container relocation problem with
 new lower bounds and dominance rules",
European Journal of Operational Research 304 (2023) 494–514.
Priority scan method (Algorithm 4).
------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section.
--------------------------------------------------------------------------
"""

import math

from algorithms.distinct.CRP_U.LB.lb1_u import lb1_u


def lb6_u(stacks, max_tiers) -> int:
    state = _retrieve(stacks)
    n_bad = lb1_u(state, max_tiers)
    if n_bad == 0 or not all(state):
        return n_bad
    p, q, w = _tables(state)
    cells = [(s, t) for s, stack in enumerate(state) for t in range(1, len(stack) + 1)]
    demand = {(s, t): math.inf if w[s][t] else p[s][t] for s, t in cells}
    resource = {(s, t): q[s][t] if t < max_tiers else 0 for s, t in cells}
    k = 0
    active, pending = set(), set(cells)
    while active or pending:
        r_next = max((resource[cell] for cell in active), default=-math.inf)
        d_next = max((demand[cell] for cell in pending), default=-math.inf)
        if r_next < d_next:
            arrived = {cell for cell in pending if demand[cell] == d_next}
            active |= arrived
            pending -= arrived
            continue
        while True:
            layer = _extract(q, w, resource, active, r_next, len(state))
            if layer is None:
                break
            k += 1
            active -= layer
        active = {cell for cell in active if resource[cell] != r_next}
    return n_bad + k


def _extract(q, w, resource, active, r_next, n_stacks):
    """A blocking layer in ``active`` whose largest resource is ``r_next``."""
    tiers = [[] for _ in range(n_stacks)]
    for s, t in active:
        tiers[s].append(t)
    if not all(tiers):
        return None
    bound = min(max(q[s][t] - w[s][t] for t in tiers[s]) for s in range(n_stacks))
    badly = [(s, t) for s, t in active if not w[s][t] and q[s][t] <= bound]
    if not badly:
        return None
    s_star, t_star = max(badly, key=resource.get)
    layer = {(s_star, t_star)}
    for s in range(n_stacks):
        if s == s_star:
            continue
        accepted = [t for t in tiers[s] if q[s][t] - w[s][t] >= q[s_star][t_star]]
        layer.add((s, max(accepted, key=lambda t: (resource[(s, t)], w[s][t]))))
    if max(resource[cell] for cell in layer) != r_next:
        return None
    return layer


def _tables(state):
    """Priority p, quality q and well-placed flag w per tier; tier 0 is the ground."""
    ground = max(priority for stack in state for priority in stack) + 1
    p, q, w = [], [], []
    for stack in state:
        priorities, qualities, placed = [ground], [ground], [1]
        for priority in stack:
            placed.append(1 if priority <= qualities[-1] else 0)
            qualities.append(min(qualities[-1], priority))
            priorities.append(priority)
        p.append(priorities)
        q.append(qualities)
        w.append(placed)
    return p, q, w


def _retrieve(stacks):
    state = [list(stack) for stack in stacks]
    while any(state):
        target = min(min(stack) for stack in state if stack)
        source = next((stack for stack in state if stack and stack[-1] == target), None)
        if source is None:
            break
        source.pop()
    return state
