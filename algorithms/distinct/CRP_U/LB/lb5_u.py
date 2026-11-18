"""
LB-TS lower bound for CRP-U

------------------------------- Reference --------------------------------
B. Jin, S. Tanaka,
"An exact algorithm for the unrestricted container relocation problem with
 new lower bounds and dominance rules",
European Journal of Operational Research 304 (2023) 494–514.
Tier scan method (Algorithm 3), following lb_ts in the authors' code.
------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section.
--------------------------------------------------------------------------
"""

import math

from algorithms.distinct.CRP_U.LB.lb1_u import lb1_u


def lb5_u(stacks, max_tiers) -> int:
    state = _retrieve(stacks)
    n_bad = lb1_u(state, max_tiers)
    if n_bad == 0 or not all(state):
        return n_bad
    p, q, w = _tables(state)
    n_stacks = len(state)
    h = [len(stack) for stack in state]
    remain = n_bad
    k = 0
    while True:
        s_min, q_min = _lowest(p, q, h)
        q_max = max((q[s][h[s]] for s in range(n_stacks) if h[s] < max_tiers), default=0)
        p_min = p_min_bad = math.inf
        v = 0
        while v < n_stacks:
            top = p[v][h[v]]
            if top == q_min:
                h[v] -= 1
                if h[v] == 0:
                    return n_bad + k
                if v == s_min and q[v][h[v]] > q_min:
                    s_min, q_min = _lowest(p, q, h)
                q_max = max(q_max, q[v][h[v]])
                if p_min <= q_min or p_min_bad <= q_max:
                    v, p_min, p_min_bad = 0, math.inf, math.inf
            elif not w[v][h[v]] and top <= q_max:
                remain -= 1
                h[v] -= 1
                if remain == 0 or h[v] == 0:
                    return n_bad + k
            else:
                p_min = min(p_min, top)
                if not w[v][h[v]]:
                    p_min_bad = min(p_min_bad, top)
                v += 1
        k += 1
        for s in range(n_stacks):
            if not w[s][h[s]]:
                remain -= 1
            h[s] -= 1
            if remain == 0 or h[s] == 0:
                return n_bad + k


def _lowest(p, q, h):
    """Stack of smallest quality, ties broken by the larger priority."""
    s_min, q_min = -1, math.inf
    for s in range(len(h)):
        if q[s][h[s]] < q_min or (q[s][h[s]] == q_min and p[s_min][h[s_min]] <= p[s][h[s]]):
            s_min, q_min = s, q[s][h[s]]
    return s_min, q_min


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
