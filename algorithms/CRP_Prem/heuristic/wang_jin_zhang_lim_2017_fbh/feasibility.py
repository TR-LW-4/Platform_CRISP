"""
Surplus-vector feasibility test for WangJinZhangLim2017FBH.

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
corresponding original algorithm paper.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .state import State, capability, orderly_height, stack_height


@dataclass
class Surplus:
    r: List[int]       # r[g]: # slots with capability exactly g (index 1..G)
    d: List[int]       # d[g]: # unfixed containers with group value exactly g
    R: List[int]       # R[g] = sum_{i=g}^{G} r[i]  (index 1..G, R[G+1] = 0)
    D: List[int]       # D[g] = sum_{i=g}^{G} d[i]
    delta: List[int]   # delta[g] = R[g] - D[g]

    def is_feasible(self, num_groups: int) -> bool:
        return all(self.delta[g] >= 0 for g in range(1, num_groups + 1))

    def min_delta_in_range(self, lo_exclusive: int, hi_inclusive: int) -> int:
        """min_{phi in (lo_exclusive, hi_inclusive]} delta(phi); +inf if empty."""
        best = float("inf")
        for phi in range(lo_exclusive + 1, hi_inclusive + 1):
            if self.delta[phi] < best:
                best = self.delta[phi]
        return best

    def affected_demand(self, lo_exclusive: int, hi_inclusive: int) -> int:
        """sum_{phi in (lo_exclusive, hi_inclusive]} d(phi) = D(lo+1) - D(hi+1)."""
        return self.D[lo_exclusive + 1] - self.D[hi_inclusive + 1]


def _cumulative(vec: List[int], num_groups: int) -> List[int]:
    cum = [0] * (num_groups + 2)
    for g in range(num_groups, 0, -1):
        cum[g] = cum[g + 1] + vec[g]
    return cum


def compute_surplus(state: State, fixed_override: Optional[Dict[int, int]] = None) -> Surplus:
    G = state.num_groups
    r = [0] * (G + 1)
    d = [0] * (G + 1)
    for s, arr in state.stacks.items():
        fh = state.fixed_height[s] if fixed_override is None else fixed_override.get(s, state.fixed_height[s])
        cap = arr[fh - 1] if fh > 0 else G
        r[cap] += state.max_tiers - fh
        for t in range(fh, len(arr)):
            d[arr[t]] += 1
    R = _cumulative(r, G)
    D = _cumulative(d, G)
    delta = [0] * (G + 2)
    for g in range(1, G + 1):
        delta[g] = R[g] - D[g]
    return Surplus(r=r, d=d, R=R, D=D, delta=delta)


# ================================================================== #
# Container stability (Definition 4)                                  #
# ================================================================== #

def stable_height(state: State, s: int, base: Optional[Surplus] = None) -> int:
    """
    sh(s): largest tier (0-indexed count) up to which stack s could be
    hypothetically fixed while keeping Delta >= 0, searched incrementally
    from the current fixed height upward through the orderly prefix.
    """
    fh = state.fixed_height[s]
    o_s = orderly_height(state.stacks[s])
    sh = fh
    while sh < o_s:
        candidate = sh + 1
        surplus = compute_surplus(state, fixed_override={s: candidate})
        if surplus.is_feasible(state.num_groups):
            sh = candidate
        else:
            break
    return sh


def compute_all_stable_heights(state: State) -> Dict[int, int]:
    return {s: stable_height(state, s) for s in state.stacks}


def can_stabilize(state: State, dst: int, group_value: int) -> bool:
    """
    Would appending a container of ``group_value`` to the top of ``dst``
    make it stable? Requires ``dst`` to currently be entirely stable, the
    new container to land orderly (group_value <= current top value), and
    the resultant hypothetical fix (dst extended by one, then fixed) to keep
    Delta >= 0.
    """
    h = stack_height(state, dst)
    if stable_height(state, dst) != h:
        return False  # dst must currently be entirely stable to extend it
    top_val = state.stacks[dst][h - 1] if h > 0 else state.num_groups
    if group_value > top_val:
        return False
    # Simulate the append (compute_surplus reads state.stacks directly, so we
    # need an actual, if temporary, mutated copy rather than a fixed-height
    # override alone).
    tmp = state.clone()
    tmp.stacks[dst].append(group_value)
    tmp.fixed_height[dst] = h + 1
    surplus = compute_surplus(tmp)
    return surplus.is_feasible(state.num_groups)


# ================================================================== #
# Extreme / pre-extreme / dead-end states (Definitions 5-7)            #
# ================================================================== #

def free_stacks(state: State) -> List[int]:
    return [s for s in state.stacks if state.fixed_height[s] < state.max_tiers]


def is_extreme_state(state: State) -> bool:
    return len(free_stacks(state)) == 2


def extreme_state_feasible(state: State) -> bool:
    """Definitions 5-6: feasibility of an extreme state (exactly 2 free stacks)."""
    free = free_stacks(state)
    if len(free) != 2:
        return True  # not an extreme state at all; nothing to check here
    a, b = free

    def is_stack_orderly(s: int) -> bool:
        return orderly_height(state.stacks[s]) == stack_height(state, s)

    if is_stack_orderly(a) and is_stack_orderly(b):
        return True

    for x, y in ((a, b), (b, a)):
        if not is_stack_orderly(x) or is_stack_orderly(y):
            continue
        arr_y = state.stacks[y]
        oy = orderly_height(arr_y)
        disorderly_part = sorted(arr_y[oy:], reverse=True)
        if stack_height(state, x) + len(disorderly_part) > state.max_tiers:
            continue
        merged = state.stacks[x] + disorderly_part
        if orderly_height(merged) == len(merged):
            return True
    return False


def is_dead_end_state(state: State) -> bool:
    return is_extreme_state(state) and not extreme_state_feasible(state)


def is_pre_extreme_state(state: State) -> bool:
    """Definition 7."""
    n_total = sum(len(v) for v in state.stacks.values())
    n_full = sum(1 for s in state.stacks if state.fixed_height[s] == state.max_tiers)
    n_almost_full = sum(1 for s in state.stacks if state.fixed_height[s] == state.max_tiers - 1)
    sum_f = sum(state.fixed_height.values())
    return n_full == len(state.stacks) - 3 and n_almost_full >= 1 and sum_f < n_total - 2


# ================================================================== #
# Valid task filter (§5.2.1)                                          #
# ================================================================== #

def is_valid_task(state: State, c_pos, s: int, surplus: Surplus) -> bool:
    cs, ct = c_pos
    fh_c = state.fixed_height[cs]
    if ct < fh_c:
        return False  # container is fixed, cannot be a target
    fh_s = state.fixed_height[s]
    if fh_s >= state.max_tiers:
        return False
    g_c = state.stacks[cs][ct]
    cap_s = capability(state, s)
    if g_c > cap_s:
        return False
    if surplus.min_delta_in_range(g_c, cap_s) < state.max_tiers - fh_s:
        return False
    return True
