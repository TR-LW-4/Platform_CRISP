"""
------------------------------- Reference --------------------------------
B. Jin, W. Zhu, A. Lim,
"Solving the container relocation problem by an improved greedy look-ahead
 heuristic",
European Journal of Operational Research 240 (2015) 837–847.
One relocation of the JZW evaluation rule, as used by Jin and Tanaka (2023).
------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section.
--------------------------------------------------------------------------
"""

from typing import Optional, Sequence, Tuple

from ._state import View, view


def choose_move(stacks: Sequence[Sequence[int]], max_tiers: int) -> Optional[Tuple[int, int]]:
    """Return one ``(source, destination)`` relocation of the JZW rule."""
    state = view(stacks)
    if state is None or state.n_bad == 0 or state.top_b(state.list[0]) == 0:
        return None

    n_stacks = state.n_stacks
    q_min = state.top_q(state.list[0])
    i_next = -1
    for index in range(n_stacks):
        stack = state.list[index]
        if state.top_q(stack) > q_min:
            break
        empty_slots = (n_stacks - 1) * max_tiers - (state.n_blocks - state.h[stack])
        if state.top_b(stack) <= empty_slots:
            i_next = index
            break
    if i_next < 0:
        return None

    i_max, q_max = _fullest_other(state, i_next, max_tiers)
    if i_max is None:
        return None

    has_multi_q_max = False
    if q_min < q_max:
        for index in range(i_max - 1, -1, -1):
            stack = state.list[index]
            if state.top_q(stack) < q_max:
                break
            if state.h[stack] < max_tiers:
                has_multi_q_max = True
                break

    source = state.list[i_next]
    if state.top_p(source) <= q_max:
        destination = _first_fit(state, i_next + 1, state.top_p(source), max_tiers)
        if destination is None:
            return None
        if state.h[destination] < max_tiers - 1:
            prepared = _prepared_source(state, source, destination)
            if prepared is not None:
                source = prepared
        return source, destination

    i_opt = _optional_source(state, i_max, q_min, q_max, source, has_multi_q_max)
    if i_opt is None:
        destination = state.list[i_max]
        if state.h[destination] == max_tiers - 1 and not _top_is_smallest(state, source):
            for index in range(i_max - 1, -1, -1):
                stack = state.list[index]
                if stack != source and state.h[stack] < max_tiers:
                    destination = stack
                    break
        if destination == source or state.h[destination] >= max_tiers:
            return None
        return source, destination

    source = state.list[i_opt]
    destination = _destination_around(state, i_opt, source, max_tiers)
    if destination is None:
        return None
    if state.h[destination] < max_tiers - 1:
        prepared = _prepared_source(state, source, destination)
        if prepared is not None:
            source = prepared
    return source, destination


def _fullest_other(state: View, i_next: int, max_tiers: int):
    for index in range(state.n_stacks - 1, -1, -1):
        stack = state.list[index]
        if index != i_next and state.h[stack] < max_tiers:
            return index, state.top_q(stack)
    return None, None


def _first_fit(state: View, start: int, priority: int, max_tiers: int) -> Optional[int]:
    for index in range(start, state.n_stacks):
        stack = state.list[index]
        if state.h[stack] < max_tiers and priority <= state.top_q(stack):
            return stack
    return None


def _prepared_source(state: View, source: int, destination: int) -> Optional[int]:
    prepared = None
    for index in range(state.rank[destination]):
        stack = state.list[index]
        if stack == source or state.top_b(stack) <= 0:
            continue
        top = state.top_p(stack)
        if state.top_p(source) <= top <= state.top_q(destination):
            if prepared is None or state.top_p(prepared) < top:
                prepared = stack
    return prepared


def _optional_source(state: View, i_max: int, q_min: int, q_max: int, source: int, has_multi_q_max: bool) -> Optional[int]:
    direction = 1
    index = i_max
    blocker = state.top_p(source)
    for _ in range(state.n_stacks * 2):
        if index == state.n_stacks or (index > i_max and state.top_q(state.list[index]) > q_max):
            direction = -1
            index = i_max - 1
            continue
        if index < 0:
            return None
        stack = state.list[index]
        if state.top_q(stack) == q_min:
            return None
        below = state.h[stack] - 1
        if (
            state.top_b(stack) == 0
            and below >= 0
            and blocker <= state.q[stack][below]
            and (index != i_max or has_multi_q_max)
        ):
            return index
        index += direction
    return None


def _top_is_smallest(state: View, source: int) -> bool:
    top = state.top_p(source)
    for depth in range(1, state.top_b(source)):
        if state.p[source][state.h[source] - depth] < top:
            return False
    return True


def _destination_around(state: View, i_opt: int, source: int, max_tiers: int) -> Optional[int]:
    direction = -1
    index = i_opt - 1
    blocker = state.top_p(source)
    for _ in range(state.n_stacks * 2):
        if index < 0 or index >= state.n_stacks:
            if direction == 1:
                return None
            direction = 1
            index = i_opt + 1
            continue
        if index < i_opt and state.top_q(state.list[index]) < blocker:
            direction = 1
            index = i_opt + 1
            continue
        stack = state.list[index]
        if state.h[stack] < max_tiers:
            return stack
        index += direction
    return None
