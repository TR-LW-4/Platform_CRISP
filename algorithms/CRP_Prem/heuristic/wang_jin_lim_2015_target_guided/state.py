"""
Bay-state representation for WangJinLim2015TGH.

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
corresponding original algorithm paper.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

Stacks = Dict[int, List[int]]     # stack_idx -> group labels, bottom..top
Move = Tuple[int, int]            # (src, dst)
Pos = Tuple[int, int]             # (stack_idx, tier_idx), 0-based from bottom


@dataclass
class State:
    stacks: Stacks
    fixed_height: Dict[int, int] = field(default_factory=dict)
    dummy: Optional[int] = None
    max_tiers: int = 0

    def clone(self) -> "State":
        return State(
            stacks={k: list(v) for k, v in self.stacks.items()},
            fixed_height=dict(self.fixed_height),
            dummy=self.dummy,
            max_tiers=self.max_tiers,
        )


def make_state(stacks_init: Stacks, max_tiers: int, dummy: Optional[int]) -> State:
    return State(
        stacks={k: list(v) for k, v in stacks_init.items()},
        fixed_height={k: 0 for k in stacks_init},
        dummy=dummy,
        max_tiers=max_tiers,
    )


def is_dummy(state: State, s: int) -> bool:
    return state.dummy is not None and s == state.dummy


def stack_height(state: State, s: int) -> int:
    return len(state.stacks[s])


def num_empty(state: State, s: int) -> int:
    return state.max_tiers - stack_height(state, s)


def unfixed_count(state: State, s: int) -> int:
    return stack_height(state, s) - state.fixed_height[s]


def infer_num_groups(stacks: Stacks) -> int:
    g_max = 0
    for arr in stacks.values():
        for x in arr:
            if x > g_max:
                g_max = int(x)
    return max(1, g_max)


# ---------------------------------------------------------------- #
# Transitive clean/dirty predicate (paper §3)                        #
# ---------------------------------------------------------------- #

def first_dirty_tier(arr: List[int]) -> int:
    """
    Lowest 0-based tier at which the stack first becomes dirty. Returns
    ``len(arr)`` when the whole stack is clean (non-increasing bottom->top).
    """
    if len(arr) < 2:
        return len(arr)
    min_below = arr[0]
    for h in range(1, len(arr)):
        if arr[h] > min_below:
            return h
        if arr[h] < min_below:
            min_below = arr[h]
    return len(arr)


def is_clean_stack(state: State, s: int) -> bool:
    """Paper §3: the dummy stack is always dirty, whatever it holds."""
    if is_dummy(state, s) and stack_height(state, s) > 0:
        return False
    arr = state.stacks[s]
    return first_dirty_tier(arr) == len(arr)


def dirty_count(state: State, s: int) -> int:
    if is_dummy(state, s):
        return stack_height(state, s)
    arr = state.stacks[s]
    return len(arr) - first_dirty_tier(arr)


def largest_dirty_group(state: State, s: int) -> Optional[int]:
    """ldg(s): largest group label among dirty containers in s."""
    arr = state.stacks[s]
    if is_dummy(state, s):
        return max(arr) if arr else None
    fbt = first_dirty_tier(arr)
    if fbt >= len(arr):
        return None
    return max(arr[fbt:])


def is_solved(state: State) -> bool:
    if state.dummy is not None and stack_height(state, state.dummy) > 0:
        return False
    return all(is_clean_stack(state, s) for s in state.stacks)


def num_above(state: State, pos: Pos) -> int:
    s, t = pos
    return stack_height(state, s) - 1 - t


def group_of(state: State, pos: Pos) -> int:
    s, t = pos
    return int(state.stacks[s][t])


def is_fixed(state: State, pos: Pos) -> bool:
    s, t = pos
    return t < state.fixed_height[s]


# ---------------------------------------------------------------- #
# Move application                                                   #
# ---------------------------------------------------------------- #

def apply_move(state: State, src: int, dst: int) -> bool:
    """Pure stack pop/push. Does not touch ``fixed_height``."""
    if src == dst:
        return False
    if not state.stacks[src] or stack_height(state, dst) >= state.max_tiers:
        return False
    x = state.stacks[src].pop()
    state.stacks[dst].append(x)
    return True


def commit_fix(state: State, s: int) -> None:
    """Mark the container currently at tier ``fixed_height[s]`` as fixed."""
    state.fixed_height[s] += 1


def top_is_fixed(state: State, s: int) -> bool:
    """True iff the stack's topmost container is itself within the fixed prefix."""
    h = stack_height(state, s)
    return h > 0 and h == state.fixed_height[s]


def borrow_fixed_top(state: State, s: int, dst: int) -> bool:
    """
    Temporarily move a stack's fixed top container elsewhere (§4.3 nslot=0
    scenarios). Caller must later call ``return_fixed_top`` to restore it.
    """
    if not top_is_fixed(state, s):
        return False
    ok = apply_move(state, s, dst)
    if ok:
        state.fixed_height[s] -= 1
    return ok


def return_fixed_top(state: State, src: int, s: int) -> bool:
    """Move a previously borrowed fixed container back onto ``s`` and re-fix it."""
    ok = apply_move(state, src, s)
    if ok:
        state.fixed_height[s] += 1
    return ok
