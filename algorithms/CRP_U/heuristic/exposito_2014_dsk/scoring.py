"""
Blocked-container destination scores for ExpositoDSK.

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
corresponding original algorithm paper.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import random
from typing import List, Optional, Tuple


def count_blocked_below(b_priority: int, dst_containers) -> int:
    """
    Number of containers in dst that would be blocked by placing b there.
    b blocks c when c.priority < b.priority (c retrieved earlier, b piled above).
    """
    return sum(1 for c in dst_containers if c.priority < b_priority)


def score_placement(b_priority: int, dst_stack) -> Tuple[int, float]:
    """
    Evaluate placing container b (priority b_priority) at the top of dst_stack.
    Returns (bad_count, tie_breaker); smaller is better.
    tie_breaker = -min_priority_in_dst (stacks with later-retrieved min item win).
    """
    containers = dst_stack.containers
    bad = count_blocked_below(b_priority, containers)
    min_prio = min((c.priority for c in containers), default=float("inf"))
    return bad, -min_prio


def dsk_select_action(env, alpha: int = 1, rng=None) -> Optional[int]:
    """
    Select the best relocation action for CRP-U using the DSK heuristic.

    Strategy
    --------
    1. Find the current target t and its topmost blocker b (physically on top
       of the target stack, above t).
    2. Score every valid non-full destination for b using score_placement().
    3. Pick the destination with the lowest score (0 = well-located = ideal).
    4. With alpha > 1, pick uniformly from the alpha best candidates.

    Returns
    -------
    action (int) = src_i * n_stacks + dst_i  (CRP_U.step() encoding),
    or None if no valid move exists.
    """
    yard   = env.yard
    n      = env._num_stacks()
    target = env._get_target_container()
    if target is None:
        return None

    # find which stack holds the target
    target_stack_key: Optional[Tuple[int, int]] = None
    for key, stk in yard.stacks.items():
        if target in stk.containers:
            target_stack_key = key
            break
    if target_stack_key is None:
        return None

    # topmost blocker = stack top (if it exists and is not the target itself)
    target_stk = yard.stacks[target_stack_key]
    top = target_stk.top
    if top is None or top.id == target.id:
        # target is accessible; env auto-retrieve should handle termination
        return None

    topmost_blocker = top
    src_i = _key_to_index(env, target_stack_key)

    # score every valid destination
    scored: List[Tuple[Tuple, int]] = []
    for dst_key, dst_stk in yard.stacks.items():
        if dst_key == target_stack_key:
            continue
        if dst_stk.is_full:
            continue
        bad, tie = score_placement(topmost_blocker.priority, dst_stk)
        dst_i = _key_to_index(env, dst_key)
        scored.append(((bad, tie), dst_i))

    if not scored:
        return None

    scored.sort(key=lambda x: x[0])

    if alpha <= 1:
        _, best_dst_i = scored[0]
    else:
        pool = scored[:alpha]
        _, best_dst_i = (rng or random).choice(pool)

    return src_i * n + best_dst_i


def _key_to_index(env, key: Tuple[int, int]) -> int:
    """(bay, row) -> linear stack index."""
    bay, row = key
    return (bay - 1) * env.config.num_rows + (row - 1)
