"""
Depth-first branch and bound of Tricoire et al. (2018). Each node first
retrieves what it can, is pruned by the chosen lower bound, and branches on
every relocation whose LB1-based bound beats the incumbent; branches are
explored in increasing bound order.

------------------------------- Reference --------------------------------
F. Tricoire, J. Scagnetti, A. Beham,
"New insights on the block relocation problem",
Computers & Operations Research 89 (2018) 127–139.
------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section.
--------------------------------------------------------------------------
"""

from time import perf_counter

from algorithms.distinct.CRP_U.LB.lb1_u import lb1_u
from algorithms.distinct.CRP_U.LB.lb2_u import lb2_u
from algorithms.distinct.CRP_U.LB.lb3_u import lb3_u
from algorithms.distinct.CRP_U.metaheuristic.rake.search import rake_search
from algorithms.distinct.CRP_U.metaheuristic.rake.state import State

BOUNDS = {1: lb1_u, 2: lb2_u, 3: lb3_u}


class _Context:
    def __init__(self, bound, deadline, best):
        self.bound = bound
        self.deadline = deadline
        self.best_ops = best.ops
        self.best = best.relocations
        self.objective = best.relocations


def dfbb(stacks, max_tiers, lb, loop, ub_width, time_limit):
    start = State(stacks, max_tiers)
    deadline = perf_counter() + time_limit if time_limit > 0 else None
    context = _Context(BOUNDS[lb], deadline, rake_search(start, ub_width))
    if not loop:
        _branch(start.copy(), -1, context)
        return context.best_ops
    level = lb3_u(stacks, max_tiers)
    upper = context.best
    while level < upper:
        context.objective = level + 1
        if not _branch(start.copy(), -1, context):
            break
        upper = min(upper, context.best)
        level += 1
    return context.best_ops


def _branch(state, last_to, context):
    if context.deadline is not None and perf_counter() > context.deadline:
        return False
    retrieved = state.retrieve_all()
    if retrieved:
        last_to = -1
    stacks, max_tiers = state.stacks, state.max_tiers
    if state.empty():
        if state.relocations < context.objective:
            context.objective = context.best = state.relocations
            context.best_ops = list(state.ops)
    elif state.relocations + context.bound(stacks, max_tiers) < context.objective:
        base = state.relocations + 1 + lb1_u(stacks, max_tiers)
        lows = state.lows()
        branches = []
        for source, stack in enumerate(stacks):
            if source == last_to or not stack:
                continue
            item = stack[-1]
            leave = -1 if lows[source] < item else 0
            for destination, other in enumerate(stacks):
                if destination == source or len(other) == max_tiers:
                    continue
                bound = base + leave + (1 if lows[destination] < item else 0)
                if bound < context.objective:
                    branches.append((bound, source, destination))
        branches.sort()
        for _, source, destination in branches:
            state.relocate(source, destination)
            going = _branch(state, destination, context)
            state.undo()
            if not going:
                return False
    for _ in range(retrieved):
        state.undo()
    return True
