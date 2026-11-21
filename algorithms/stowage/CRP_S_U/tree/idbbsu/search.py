"""Unrestricted iterative deepening for Tanaka and Voß (2019), Section 6.1.

The search increases the allowed number of relocations from the lower bound
up to the greedy upper bound. A partial path is pruned when its length plus
a lower bound exceeds the current limit. Empty destination stacks are
identical, so only one of them is branched on.

Relocation-ordering rules follow ``DOMINANCE_CHECK`` in brpsp-1.0. A
container that has already been relocated is not moved again when an
equivalent earlier destination was available, and independent relocations
are kept in one stack-index order.
"""

from __future__ import annotations

import time

from algorithms.stowage.CRP_S_R.operator.tv import relocations
from algorithms.stowage.CRP_S_U.LB import lower_bound
from algorithms.stowage.CRP_S_U.LB.lb2c import lower_bound as lb2c
from algorithms.stowage.CRP_S_U.model import freeze, loaded_from, normalize


# Larger than any relocation depth. The reference packs two depths into one
# integer with the same split (source stamp, destination stamp, retrieval stamp).
_STAMP = 1_000_000
_TABLE = (
    (0, 1, 0, 0),
    (1, 1, 2, 1),
    (0, 2, 0, 0),
    (0, 1, 0, 0),
)


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
    modified0 = (0,) * len(start)

    def limited(limit):
        nonlocal timed_out
        found = []
        seen = {}

        def dfs(state, path, modified, last_of):
            nonlocal timed_out
            if found or timed_out:
                return
            if deadline is not None and time.perf_counter() >= deadline:
                timed_out = True
                return
            used = len(path)
            if used > limit:
                return
            key = (state, _dominance_key(state, modified, last_of, path))
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
            for child, source, destination, _score, next_modified, next_last in _children(
                state, max_tiers, heights, modified, last_of, path
            ):
                path.append((source, destination))
                dfs(child, path, next_modified, next_last)
                path.pop()
                if found or timed_out:
                    return

        dfs(start, [], modified0, {})
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


def _children(state, max_tiers, heights, modified, last_of, path):
    columns = [list(stack) for stack in state]
    sources = [index for index, stack in enumerate(columns) if stack]
    checks, anchors = _ordering(modified, path)
    children = []
    for source in sources:
        if not columns[source]:
            continue
        container = columns[source][-1]
        if _source_dominated(columns, source, modified, last_of, path, max_tiers):
            continue
        saw_empty = False
        for destination in range(len(columns)):
            if destination == source or len(columns[destination]) >= max_tiers:
                continue
            if not columns[destination]:
                if saw_empty:
                    continue
                saw_empty = True
            if _destination_dominated(
                source, destination, container, modified, last_of, checks, anchors
            ):
                continue
            trial = [list(stack) for stack in columns]
            trial[destination].append(trial[source].pop())
            next_modified = list(modified)
            next_last = dict(last_of)
            depth = len(path) + 1
            moved = trial[destination][-1]
            next_modified[source] = _STAMP + depth
            next_modified[destination] = depth
            next_last[moved] = depth
            if not _retrieve(trial, heights, next_modified, next_last, path + [(source, destination)], depth):
                continue
            frozen = freeze(trial)
            children.append((
                frozen, source, destination, lb2c(frozen, heights), tuple(next_modified), next_last
            ))
    children.sort(key=lambda item: (item[3], item[1], item[2]))
    return children


def _source_dominated(columns, source, modified, last_of, path, max_tiers):
    """True when relocating this top again is dominated by an earlier move."""
    moved = last_of.get(columns[source][-1], 0)
    if not moved:
        return False
    previous_source = path[moved - 1][0]
    if modified[previous_source] == _STAMP + moved:
        return True
    if modified[source] == moved:
        height = len(columns[source])
        for index in range(source):
            if len(columns[index]) <= height - 1 and _reduced(modified[index]) < moved:
                return True
        for index in range(source + 1, len(columns)):
            if len(columns[index]) < height - 1 and _reduced(modified[index]) < moved:
                return True
        return False
    return any(
        len(stack) < max_tiers and _reduced(modified[index]) < moved
        for index, stack in enumerate(columns)
    )


def _destination_dominated(source, destination, container, modified, last_of, checks, anchors):
    moved = last_of.get(container, 0)
    if moved and _reduced(modified[destination]) < moved:
        return True
    anchor = anchors[_TABLE[checks[source]][checks[destination]]]
    return source < anchor


def _ordering(modified, path):
    """Stack classes and the source indices of the latest non-interfering moves."""
    pairs = []
    for index, value in enumerate(modified):
        if value <= _STAMP:
            continue
        move_id = value - _STAMP
        _src, destination = path[move_id - 1]
        if modified[destination] == move_id:
            pairs.append((index, destination))
    pairs.sort(reverse=True)
    anchors = [-1, -1, -1]
    checks = [0] * len(modified)
    for slot, (src, destination) in enumerate(pairs[:3]):
        anchors[slot] = src
        mark = slot + 1
        checks[src] = mark
        checks[destination] = mark
    return checks, anchors


def _retrieve(columns, heights, modified, last_of, path, depth):
    """Retrieve exposed targets from high stack indices downward.

    Returns false when the reference dominance rule rejects this relocation
    while a previously relocated container is loaded.
    """
    loaded = loaded_from(columns, heights)

    def ready(index):
        stack = columns[index]
        if not stack:
            return False
        vessel_stack, vessel_tier = stack[-1]
        return vessel_tier == loaded[vessel_stack]

    index = len(columns)
    while index > 0:
        index -= 1
        if not ready(index):
            continue
        modified[index] = depth - _STAMP
        while ready(index):
            container = columns[index][-1]
            moved = last_of.get(container, 0)
            if moved and _retrieval_dominated(columns, index, moved, modified, path):
                return False
            columns[index].pop()
            loaded[container[0]] += 1
        index = len(columns)
    return True


def _retrieval_dominated(columns, index, moved, modified, path):
    previous_source = path[moved - 1][0]
    if modified[previous_source] == _STAMP + moved:
        return True
    height = len(columns[index])
    for other in range(index):
        if len(columns[other]) <= height - 1 and _reduced(modified[other]) < moved:
            return True
    for other in range(index + 1, len(columns)):
        if len(columns[other]) < height - 1 and _reduced(modified[other]) < moved:
            return True
    return False


def _dominance_key(state, modified, last_of, path):
    live = tuple(
        sorted(
            (container, last_of.get(container, 0))
            for stack in state
            for container in stack
        )
    )
    needed = set()
    for value in modified:
        move_id = _stamp_id(value)
        if move_id:
            needed.add(move_id)
    for _container, move_id in live:
        if move_id:
            needed.add(move_id)
    referenced = tuple(path[move_id - 1] for move_id in sorted(needed))
    return (modified, live, referenced)


def _stamp_id(value):
    if value > _STAMP:
        return value - _STAMP
    if value > 0:
        return value
    if value < 0:
        return value + _STAMP
    return 0


def _reduced(value):
    return (value + _STAMP) % _STAMP
