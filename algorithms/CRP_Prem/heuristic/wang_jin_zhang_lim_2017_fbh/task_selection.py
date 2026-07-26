"""
Valid-task set construction and the six-tuple lexicographic task ranking
(paper §5.2).

The six-tuple, minimized lexicographically over all valid tasks (c -> s):

1. tier-protection indicator (0 preferred): a task is "tier-protected" when
   its aim stack has ``f(s) <= P1`` and moving c there would touch an
   "affected demand" (of the group values it displaces) ``>= P2``. The
   paper describes tier-protection as removing such tasks from T outright;
   we instead encode it as the primary (least-significant-first, i.e. most
   dominant) tuple key so that T never becomes accidentally empty in a
   pathological state where *every* remaining valid task happens to be
   tier-protected -- a strictly safer implementation of the same intent.
2. the number of movements STAP would need to accomplish the task.
3. the number of currently-stable-but-unfixed containers on the aim stack,
   ``sh(s) - f(s)`` (fewer disturbed stable containers preferred).
4. the affected demand ``sum_{phi=g(c)+1}^{g(s,f(s))} d(phi)``.
5. the fixed height of the aim stack, ``f(s)`` (prefer less-fixed stacks).
6. the negative of ``g(c)`` (prefer larger group values first).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .feasibility import (
    Surplus,
    compute_all_stable_heights,
    compute_surplus,
    is_pre_extreme_state,
    is_valid_task,
)
from .stap import external_task_move_count, internal_task_move_count
from .state import Pos, State, capability


def candidate_positions(state: State) -> List[Pos]:
    out: List[Pos] = []
    for s, arr in state.stacks.items():
        for t in range(state.fixed_height[s], len(arr)):
            out.append((s, t))
    return out


def select_task(
    state: State,
    p1: int,
    p2: int,
    stable_heights: Optional[Dict[int, int]] = None,
    surplus: Optional[Surplus] = None,
) -> Optional[Tuple[Pos, int]]:
    if surplus is None:
        surplus = compute_surplus(state)
    if stable_heights is None:
        stable_heights = compute_all_stable_heights(state)
    pre_extreme = is_pre_extreme_state(state)

    best_key = None
    best_choice: Optional[Tuple[Pos, int]] = None

    for c_pos in candidate_positions(state):
        cs, ct = c_pos
        g_c = state.stacks[cs][ct]
        for s in state.stacks:
            fh_s = state.fixed_height[s]
            if pre_extreme and fh_s == state.max_tiers - 1:
                continue
            if not is_valid_task(state, c_pos, s, surplus):
                continue
            cap_s = capability(state, s)
            affected = surplus.affected_demand(g_c, cap_s)
            tier_protected = 1 if (fh_s <= p1 and affected >= p2) else 0
            if cs == s:
                move_count = internal_task_move_count(state, cs, ct)
            else:
                move_count = external_task_move_count(state, cs, ct, s)
            sh_minus_f = stable_heights.get(s, fh_s) - fh_s
            key = (tier_protected, move_count, sh_minus_f, affected, fh_s, -g_c)
            if best_key is None or key < best_key:
                best_key = key
                best_choice = (c_pos, s)

    return best_choice
