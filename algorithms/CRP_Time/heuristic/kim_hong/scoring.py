"""
Kim & Hong (2006) ENAR scoring functions.

Separated from the algorithm class so they can be tested independently
and reused by derived algorithms without importing the full training loop.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


def blocker_priority(env) -> Optional[int]:
    """Return the priority of the topmost blocker above the current target, or None."""
    target = env._get_target_container()
    if target is None:
        return None
    src = env.yard._find_stack(target)
    if src is None or src.is_empty or src.top == target:
        return None
    return int(src.top.priority)


def valid_dst_actions(env, mask: Optional[np.ndarray]) -> List[int]:
    """Return flat action indices of non-full stacks that differ from the blocker's stack."""
    target = env._get_target_container()
    if target is None:
        return []
    src = env.yard._find_stack(target)
    if src is None or src.top == target:
        return []
    src_key = (src.bay, src.row)
    n       = env.action_space.n
    cand    = list(range(n)) if mask is None else [int(i) for i in np.where(mask)[0]]
    return [a for a in cand if env._action_to_stack(int(a)) != src_key]


def bad_overlaps_count(priorities: List[int]) -> int:
    """Count consecutive (lower, upper) pairs where lower priority < upper priority."""
    return sum(
        1 for i in range(len(priorities) - 1)
        if priorities[i] < priorities[i + 1]
    )


def relocation_score(env, dst_action: int, p_blk: int) -> Tuple[int, int, int, int]:
    """
    Lexicographic key to minimise when choosing a destination stack.

    1) Fewer containers in dst with priority < p_blk  (direct future conflicts)
    2) Fewer new priority inversions after placing the blocker on top
    3) Lower resulting stack height
    4) Stable tie-break: smaller flat action index
    """
    dst   = env._action_to_stack(int(dst_action))
    stk   = env.yard.stacks[dst]
    below = [int(c.priority) for c in stk.containers]

    cnt_low   = sum(1 for p in below if p < p_blk)
    new_prios = below + [p_blk]
    bad       = bad_overlaps_count(new_prios)
    height    = len(new_prios)

    return (cnt_low, bad, height, int(dst_action))


def select_action(env, mask: Optional[np.ndarray] = None) -> int:
    """Pick the best destination stack index for the current blocker."""
    if mask is None:
        mask = env._get_info().get("action_mask")

    p_blk = blocker_priority(env)
    if p_blk is None:
        return 0

    valid = valid_dst_actions(env, mask)
    if not valid:
        return 0

    return min(valid, key=lambda a: relocation_score(env, a, p_blk))
