"""
Searches shared by RakeSearch, Pilot, and DFBBU: FastMeta, rake search,
and the pilot method.

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

from algorithms.distinct.CRP_U.LB.lb3_u import lb3_u
from algorithms.distinct.CRP_U.operator import sm1, sm2

from .heuristic import policy, smseq


def fast_meta(state):
    best = None
    for run in (lambda s: policy(s, sm1.choose_move),
                lambda s: policy(s, sm2.choose_move),
                lambda s: smseq(s, 1, "tricoire"),
                lambda s: smseq(s, 2, "tricoire")):
        result = run(state.copy())
        if best is None or result.relocations < best.relocations:
            best = result
    return best


def rake_search(state, width):
    level = [state.copy()]
    while len(level) < width:
        following, keys, slots = [], [], {}
        for current in level:
            current.retrieve_all()
            if current.empty():
                return current
            for source, destination in _rake_moves(current):
                successor = current.copy()
                successor.relocate(source, destination)
                _update_level(following, keys, slots, successor)
        level = following
    best = None
    for current in level:
        result = fast_meta(current)
        if best is None or result.relocations < best.relocations:
            best = result
    return best


def pilot(state, width, hub_width):
    best = rake_search(state, hub_width)
    queue = [state.copy()]
    while True:
        best_value, candidates = None, []
        for current in queue:
            if current.relocations + lb3_u(current.stacks, current.max_tiers) >= best.relocations:
                continue
            current.retrieve_all()
            if current.empty():
                return best
            for source, destination in _pilot_moves(current):
                successor = current.copy()
                successor.relocate(source, destination)
                move = sm2.safe_1(successor.stacks, successor.max_tiers) \
                    or sm2.safe_2(successor.stacks, successor.max_tiers)
                if move is not None:
                    successor.relocate(*move)
                result = rake_search(successor, hub_width)
                if result.relocations < best.relocations:
                    best = result
                if best_value is None or result.relocations < best_value:
                    best_value, candidates = result.relocations, [successor]
                elif result.relocations == best_value:
                    candidates.append(successor)
        if not candidates:
            return best
        queue = candidates[:width]


def _blockers(state):
    return [(index, stack[-1]) for index, stack in enumerate(state.stacks)
            if len(stack) > 1 and stack[-1] > min(stack[:-1])]


def _rake_moves(state):
    stacks, max_tiers = state.stacks, state.max_tiers
    lows = state.lows()
    blockers = _blockers(state)
    moves, seen = [], set()

    def add(move):
        if move not in seen:
            seen.add(move)
            moves.append(move)

    for source, item in blockers:
        for destination in range(len(stacks)):
            if destination != source and len(stacks[destination]) < max_tiers and lows[destination] >= item:
                add((source, destination))
    for origin, stack in enumerate(stacks):
        if not stack:
            continue
        below = min(stack[:-1]) if len(stack) > 1 else state.n + 1
        best, best_gap = None, 4 * state.n
        for source, item in blockers:
            if origin == source or stack[-1] >= item or below < item:
                continue
            landing, gap = None, state.n + 1
            for destination in range(len(stacks)):
                if (destination != origin and len(stacks[destination]) < max_tiers
                        and lows[destination] >= stack[-1] and lows[destination] - stack[-1] < gap):
                    landing, gap = destination, lows[destination] - stack[-1]
            if landing is not None and below - item + gap < best_gap:
                best, best_gap = (origin, landing), below - item + gap
        if best is not None:
            add(best)
    for item in range(state.next, min(state.next + 4, state.n) + 1):
        source = state.where[item]
        for destination in range(len(stacks)):
            if destination != source and len(stacks[destination]) < max_tiers:
                add((source, destination))
    return moves


# A dominated entry is overwritten by the last entry and the newcomer is
# appended, which keeps the queue order of the original code.
def _update_level(level, keys, slots, state):
    key = tuple(map(tuple, state.stacks))
    index = slots.get(key)
    if index is not None:
        if level[index].relocations <= state.relocations:
            return
        if index == len(level) - 1:
            level[index] = state
            return
        level[index], keys[index] = level.pop(), keys.pop()
        slots[keys[index]] = index
    slots[key] = len(level)
    level.append(state)
    keys.append(key)


def _pilot_moves(state):
    stacks, max_tiers = state.stacks, state.max_tiers
    moves = []
    for source, stack in enumerate(stacks):
        if source == state.last_to or not stack:
            continue
        to_empty = False
        for destination, other in enumerate(stacks):
            if destination == source or len(other) == max_tiers:
                continue
            if other or not to_empty:
                moves.append((source, destination))
            if not other:
                to_empty = True
    return moves
