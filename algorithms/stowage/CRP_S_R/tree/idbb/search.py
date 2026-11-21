"""Restricted iterative deepening for Tanaka and Voß (2019), Section 6.1.

The search increases the allowed number of relocations from the lower bound
up to the greedy upper bound. A partial path is pruned when its length plus
a lower bound exceeds the current limit. Empty destination stacks are
identical, so only one of them is branched on. After a relocation that
retrieves nothing, the next relocation stays on the same source stack.
"""

from __future__ import annotations

import time

from algorithms.stowage.CRP_S_R.operator.tv import relocations
from algorithms.stowage.CRP_S_U.LB import lower_bound
from algorithms.stowage.CRP_S_U.LB.lb2c import lower_bound as lb2c
from algorithms.stowage.CRP_S_U.model import freeze, loaded_from, normalize, top_target


def solve(stacks, max_tiers, heights, lb_mode="lbr", time_limit=0.0):
    """Return ``(relocations, proven)``."""
    heights = tuple(heights)
    start = freeze(normalize(stacks, heights))
    upper = relocations(start, max_tiers, heights)
    result_cache = {}
    lb_cache = {}
    root_bound = _estimate(start, heights, lb_mode, result_cache, lb_cache)
    if len(upper) <= root_bound:
        return upper, True

    deadline = None if not time_limit else time.perf_counter() + time_limit
    best = upper
    timed_out = False

    def limited(limit):
        nonlocal timed_out
        found = []
        seen = {}

        def dfs(state, path, lock):
            nonlocal timed_out
            if found or timed_out:
                return
            if deadline is not None and time.perf_counter() >= deadline:
                timed_out = True
                return
            used = len(path)
            if used > limit:
                return
            key = (state, lock)
            previous = seen.get(key)
            if previous is not None and previous <= used:
                return
            seen[key] = used
            if not any(state):
                found.append(list(path))
                return
            estimate = _estimate(state, heights, lb_mode, result_cache, lb_cache)
            if used + estimate > limit:
                return
            for child, source, destination, next_lock, _score in _children(
                state, max_tiers, heights, lock
            ):
                path.append((source, destination))
                dfs(child, path, next_lock)
                path.pop()
                if found or timed_out:
                    return

        dfs(start, [], None)
        return found[0] if found else None

    for limit in range(root_bound, len(best)):
        improved = limited(limit)
        if timed_out:
            return best, False
        if improved is not None:
            return improved, True
    return best, True


def _estimate(state, heights, lb_mode, result_cache, lb_cache):
    frozen = freeze(normalize(state, heights))
    key = (lb_mode, heights, frozen)
    saved = result_cache.get(key)
    if saved is not None:
        return saved
    value = lower_bound(lb_mode, frozen, heights, lb_cache)
    result_cache[key] = value
    return value


def _children(state, max_tiers, heights, lock):
    columns = [list(stack) for stack in state]
    if lock is not None:
        sources = [lock]
    else:
        loaded = loaded_from(columns, heights)
        sources = []
        for index, stack in enumerate(columns):
            target = top_target(stack, loaded)
            if target is not None and target < len(stack) - 1:
                sources.append(index)

    children = []
    for source in sources:
        if not columns[source]:
            continue
        saw_empty = False
        for destination in range(len(columns)):
            if destination == source or len(columns[destination]) >= max_tiers:
                continue
            if not columns[destination]:
                if saw_empty:
                    continue
                saw_empty = True
            trial = [list(stack) for stack in columns]
            trial[destination].append(trial[source].pop())
            before = sum(len(stack) for stack in columns)
            trial = normalize(trial, heights)
            after = sum(len(stack) for stack in trial)
            next_lock = source if after == before else None
            frozen = freeze(trial)
            children.append((frozen, source, destination, next_lock, lb2c(frozen, heights)))
    children.sort(key=lambda item: (item[4], item[1], item[2]))
    return children
