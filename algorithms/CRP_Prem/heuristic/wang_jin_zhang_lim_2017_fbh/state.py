"""
Layout state for the Wang, Jin, Zhang, Lim (2017) feasibility-based
heuristic (FBH).

Reference: N. Wang, B. Jin, Z. Zhang, A. Lim, "A feasibility-based heuristic
for the container pre-marshalling problem", EJOR 256 (2017) 90-101.

State representation (paper §4, Definitions 1-2)
--------------------------------------------------
A state is a pair ``(L, f)``: a layout ``L`` (stacks of group labels, bottom
to top) and a fix vector ``f`` giving, per stack, the number of *fixed*
containers. Fixed containers are orderly containers that have been locked to
a slot; by construction they always form a contiguous prefix from the bottom
of a stack (exactly as in Wang, Jin & Lim 2015's TGH), so -- as in that
module -- we track only the integer ``fixed_height[s]``.

Group-label convention: a container is *orderly* if it is supported by the
ground or by another orderly container with an equal or larger group value;
i.e. group labels are non-increasing from bottom to top (same convention as
"clean" in the 2015 TGH paper and "well-placed" in Bortfeldt & Forster 2012).
The ground is treated as an occupied slot at (conceptual) tier 0 with group
value ``G`` (the paper's largest group value / weakest constraint).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

Stacks = Dict[int, List[int]]
Move = Tuple[int, int]
Pos = Tuple[int, int]


@dataclass
class State:
    stacks: Stacks
    fixed_height: Dict[int, int] = field(default_factory=dict)
    max_tiers: int = 0
    num_groups: int = 0

    def clone(self) -> "State":
        return State(
            stacks={k: list(v) for k, v in self.stacks.items()},
            fixed_height=dict(self.fixed_height),
            max_tiers=self.max_tiers,
            num_groups=self.num_groups,
        )


def infer_num_groups(stacks: Stacks) -> int:
    g_max = 0
    for arr in stacks.values():
        for x in arr:
            if x > g_max:
                g_max = int(x)
    return max(1, g_max)


def total_capacity(state: State) -> int:
    return len(state.stacks) * state.max_tiers


def total_containers(state: State) -> int:
    return sum(len(v) for v in state.stacks.values())


def total_empty(state: State) -> int:
    return total_capacity(state) - total_containers(state)


def stack_height(state: State, s: int) -> int:
    return len(state.stacks[s])


def num_empty(state: State, s: int) -> int:
    return state.max_tiers - stack_height(state, s)


def make_state(stacks_init: Stacks, max_tiers: int, num_groups: Optional[int] = None) -> State:
    stacks = {k: list(v) for k, v in stacks_init.items()}
    ng = num_groups if num_groups is not None else infer_num_groups(stacks)
    return State(
        stacks=stacks,
        fixed_height={k: 0 for k in stacks},
        max_tiers=max_tiers,
        num_groups=ng,
    )


def unreachable_tiers(state: State) -> int:
    """U = max(H - E, 0): tiers that are unreachable in dense instances (§5.1)."""
    return max(state.max_tiers - total_empty(state), 0)


def initialize_fixed_from_unreachable(state: State) -> None:
    """
    Pre-fix up to the bottom ``U`` tiers of every stack (the initial state
    ``(L0, U * 1)`` of Algorithm 1). Only called once, on a freshly built
    state with ``fixed_height`` all zero.

    Definition 1 only allows *orderly* containers to be fixed, so the
    pre-fix is additionally capped by each stack's orderly height: if a
    stack's bottom containers are not already in non-increasing order, we
    do not force-fix them (that would fix an invalid, unstable-by-construction
    prefix and can make the instance unsolvable by bookkeeping alone). The
    main loop's iteration budget is derived from the actual post-init fixed
    count, not from ``S * U``, so this is always consistent.
    """
    u = unreachable_tiers(state)
    for s in state.stacks:
        o_s = orderly_height(state.stacks[s])
        state.fixed_height[s] = min(u, o_s)


# ---------------------------------------------------------------- #
# Orderly / capability / messiness helpers (paper §3-4)               #
# ---------------------------------------------------------------- #

def orderly_height(arr: List[int]) -> int:
    """
    Number of orderly containers counted from the bottom (o(s)): the length
    of the maximal non-increasing prefix, i.e. the first index at which the
    "supported by an equal-or-larger value" rule breaks.
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


def is_orderly_tier(state: State, s: int, t: int) -> bool:
    """0-indexed tier ``t`` of stack ``s`` is orderly."""
    return t < orderly_height(state.stacks[s])


def q_value(state: State, s: int, t: int) -> int:
    """
    Capability of an *occupied* slot (paper §5.3.2): the group value of the
    container if orderly, else 0. The ground (t = -1, conceptually "tier 0"
    in the paper's 1-indexing) is always orderly with value G.
    """
    if t < 0:
        return state.num_groups
    return state.stacks[s][t] if is_orderly_tier(state, s, t) else 0


def capability(state: State, s: int) -> int:
    """
    g(s, f(s)): the group value of the topmost *fixed* container of s, or G
    if s has no fixed containers (skyline at the ground). Slots above the
    skyline may only receive containers with a group value <= this.
    """
    fh = state.fixed_height[s]
    if fh == 0:
        return state.num_groups
    return state.stacks[s][fh - 1]


def apply_move(state: State, src: int, dst: int) -> bool:
    """
    Pure top-to-top stack move. FBH's feasibility framework (§4-5) is
    designed so that STAP only ever needs to move currently-unfixed
    containers onto non-full stacks; callers are responsible for that
    invariant, this just performs the raw pop/push with a capacity check.
    """
    if src == dst:
        return False
    if not state.stacks[src] or stack_height(state, dst) >= state.max_tiers:
        return False
    x = state.stacks[src].pop()
    state.stacks[dst].append(x)
    return True


def top_value(state: State, s: int) -> int:
    """q(s, h(s)): capability-style value of the current top slot; G if empty."""
    h = stack_height(state, s)
    if h == 0:
        return state.num_groups
    return q_value(state, s, h - 1)


def messiness(state: State, s: int, sh_s: int) -> int:
    """
    m(s): the largest group value among the *unstable* containers of stack
    s, i.e. tiers ``[sh(s), h(s))``. Returns 0 if s has no unstable
    containers (caller should special-case ``sh(s) == h(s)`` beforehand).
    """
    arr = state.stacks[s]
    tail = arr[sh_s:]
    return max(tail) if tail else 0
