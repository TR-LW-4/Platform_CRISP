"""
------------------------------- Reference --------------------------------
F. Tricoire, J. Scagnetti, A. Beham,
"New insights on the block relocation problem",
Computers & Operations Research 89 (2018) 127–139.
One safe relocation of SM-2, as used by Jin and Tanaka (2023).
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
    """Return one ``(source, destination)`` relocation of the SM-2 rule."""
    state = view(stacks)
    if state is None or state.n_bad == 0 or state.top_b(state.list[0]) == 0:
        return None

    q_min = state.top_q(state.list[0])
    i_next = -1
    for index in range(state.n_stacks):
        stack = state.list[index]
        if state.top_q(stack) > q_min:
            break
        empty_slots = (state.n_stacks - 1) * max_tiers - (state.n_blocks - state.h[stack])
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

    source, destination = _safe_pair(state, i_max, q_min, q_max, has_multi_q_max, max_tiers)
    if source is None:
        source = state.list[i_next]
        destination = state.list[i_max]
    if destination is None or destination == source or state.h[destination] >= max_tiers:
        return None
    if not state.h[source]:
        return None
    return source, destination


def _fullest_other(state: View, i_next: int, max_tiers: int):
    for index in range(state.n_stacks - 1, -1, -1):
        stack = state.list[index]
        if index != i_next and state.h[stack] < max_tiers:
            return index, state.top_q(stack)
    return None, None


def _safe_pair(state: View, i_max: int, q_min: int, q_max: int, has_multi_q_max: bool, max_tiers: int):
    source = destination = None
    best_diff = None
    if q_min < q_max:
        source, destination, best_diff = _safe_one(state, i_max, q_max, max_tiers)
        if best_diff is None:
            source, destination, best_diff = _safe_two(
                state, i_max, q_max, has_multi_q_max, max_tiers
            )
    return source, destination


def _safe_one(state: View, i_max: int, q_max: int, max_tiers: int):
    source = destination = None
    best_diff = None
    for index in range(i_max):
        origin = state.list[index]
        if state.h[origin] == 0:
            break
        if state.top_b(origin) > 0 and state.top_p(origin) <= q_max:
            for later in range(index + 1, state.n_stacks):
                landing = state.list[later]
                diff = state.top_q(landing) - state.top_p(origin)
                if best_diff is not None and diff >= best_diff:
                    break
                if state.h[landing] < max_tiers and diff >= 0:
                    source, destination, best_diff = origin, landing, diff
                    break
    return source, destination, best_diff


def _safe_two(state: View, i_max: int, q_max: int, has_multi_q_max: bool, max_tiers: int):
    source = destination = None
    best_diff = None
    for index in range(state.n_stacks):
        origin = state.list[index]
        if state.top_q(origin) > q_max:
            break
        if state.top_b(origin) != 0 or not (index != i_max or has_multi_q_max):
            continue
        if state.h[origin] < 1:
            continue
        below = state.q[origin][state.h[origin] - 1]
        bad, alternate = _bad_stacks(state, below)
        if bad is None:
            continue
        landing, diff = _landing(state, index, origin, bad, best_diff, max_tiers)
        if landing is None:
            continue
        if landing != bad:
            source, destination, best_diff = origin, landing, diff
            continue
        if alternate is not None:
            alt_diff = below - state.top_p(alternate) + state.top_q(landing) - state.top_p(origin)
            if best_diff is None or alt_diff < best_diff:
                source, destination, best_diff = origin, landing, alt_diff
        shifted = _shift_landing(state, index, i_max, origin, bad, below, best_diff, max_tiers)
        if shifted is not None:
            source, destination, best_diff = origin, shifted[0], shifted[1]
    return source, destination, best_diff


def _bad_stacks(state: View, below: int):
    bad = alternate = None
    best_diff = None
    for index in range(state.n_stacks):
        stack = state.list[index]
        if state.top_q(stack) >= below:
            break
        diff = below - state.top_p(stack)
        if state.top_b(stack) > 0 and diff >= 0 and (best_diff is None or diff < best_diff):
            alternate = bad
            bad = stack
            best_diff = diff
        elif bad is not None and (
            alternate is None or state.top_p(alternate) < state.top_p(stack)
        ):
            alternate = stack
    return bad, alternate


def _landing(state: View, index: int, origin: int, bad: int, best_diff, max_tiers: int):
    below = state.q[origin][state.h[origin] - 1]
    direction = -1
    cursor = index - 1
    for _ in range(state.n_stacks * 2):
        if cursor < 0 or cursor >= state.n_stacks:
            if direction == 1:
                return None, None
            direction = 1
            cursor = index + 1
            continue
        if direction == -1 and state.top_q(state.list[cursor]) < state.top_p(origin):
            direction = 1
            cursor = index + 1
            continue
        stack = state.list[cursor]
        diff = below - state.top_p(bad) + state.top_q(stack) - state.top_p(origin)
        if best_diff is not None and diff >= best_diff:
            return None, None
        if state.h[stack] < max_tiers:
            return stack, diff
        cursor += direction
    return None, None


def _shift_landing(state, index, i_max, origin, bad, below, best_diff, max_tiers):
    direction = -1 if state.rank[bad] < index else 1
    cursor = state.rank[bad] + direction
    while 0 <= cursor <= i_max:
        if direction == -1 and state.top_q(state.list[cursor]) < state.top_p(origin):
            direction = 1
            cursor = index + 1
            continue
        stack = state.list[cursor]
        diff = below - state.top_p(bad) + state.top_q(stack) - state.top_p(origin)
        if best_diff is not None and diff >= best_diff:
            return None
        if state.h[stack] < max_tiers:
            return stack, diff
        cursor += direction
    return None
