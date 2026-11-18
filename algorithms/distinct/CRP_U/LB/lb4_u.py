"""
LB-N lower bound for CRP-U

------------------------------- Reference --------------------------------
S. Tanaka, F. Mizuno,
"An exact algorithm for the unrestricted block relocation problem",
Computers & Operations Research 95 (2018) 12–31.
------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section.
--------------------------------------------------------------------------
"""

import math

from algorithms.distinct.CRP_U.LB.lb1_u import lb1_u


def lb4_u(stacks, max_tiers) -> int:
    state = _retrieve(stacks)
    lower_bound = lb1_u(state, max_tiers)
    if lower_bound == 0:
        return lower_bound
    p, q, w = _tables(state)
    height = [len(stack) for stack in state]
    while any(height):
        target = min(q[s][height[s]] for s in range(len(state)))
        lowest = {}
        for s in range(len(state)):
            tier = next((t for t in range(1, height[s] + 1) if p[s][t] == target), None)
            if tier is not None:
                lowest[s] = tier
        full = {}
        for s, tier in lowest.items():
            full[s] = height[s]
            height[s] = tier - 1
        for s, tier in lowest.items():
            height[s] = full[s]
            if _check(p, q, w, height, s, tier, max_tiers):
                return lower_bound + 1
            height[s] = tier - 1
    return lower_bound


def _check(p, q, w, height, source, tier, max_tiers):
    """True if the containers above ``(source, tier)`` cannot all become
    well-placed by forced relocations only."""
    q_max = max((q[s][height[s]] for s in range(len(height)) if height[s] < max_tiers), default=-math.inf)
    quality = {
        s: q[s][height[s]]
        for s in range(len(height))
        if s != source and (
            height[s] < max_tiers
            or (not w[s][max_tiers] and p[s][max_tiers] <= q_max)
        )
    }
    for t in range(height[source], tier, -1):
        blocker = p[source][t]
        if blocker == p[source][tier]:
            continue
        if blocker > max(quality.values(), default=-math.inf):
            return True
        destination = min((s for s in quality if quality[s] >= blocker), key=quality.get)
        quality[destination] = blocker
    return False


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
