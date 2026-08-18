"""
Destination-stack selection helpers for CasertaHeuristic.

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
corresponding original algorithm paper.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


def _min_priority(stk, n_total: int) -> int:
    """
    Lowest-numbered (earliest-retrieved) block priority in a stack.
    Empty stacks return n_total + 1 (treated as always 'good').
    """
    if stk.is_empty:
        return n_total + 1
    return min(c.priority for c in stk.containers)


def caserta_select_action(env, mask: Optional[np.ndarray] = None) -> int:
    """
    Pick the destination stack index for the current topmost blocker
    using Caserta et al. (2012) min-priority scoring.

    Returns a flat action index compatible with CRP_R.step().
    """
    if mask is None:
        mask = env._get_info().get("action_mask")

    # Identify the target container and its stack
    target = env._get_target_container()
    if target is None:
        return 0

    src_stack = env.yard._find_stack(target)
    if src_stack is None or src_stack.top == target:
        return 0   # target already on top, auto-retrieved

    # r = topmost blocker above the target
    r = src_stack.top
    r_priority = int(r.priority)
    src_key = (src_stack.bay, src_stack.row)

    n_total = env.config.num_containers
    n_stacks = env.config.num_bays * env.config.num_rows

    # Candidate destination stacks (non-full, not source)
    valid = list(range(n_stacks)) if mask is None else [
        int(i) for i in np.where(mask)[0]
    ]
    candidates = [
        a for a in valid
        if env._action_to_stack(int(a)) != src_key
    ]
    if not candidates:
        return 0

    # Compute min(i) for each candidate
    def min_i(action: int) -> int:
        pos = env._action_to_stack(int(action))
        return _min_priority(env.yard.stacks[pos], n_total)

    # Good stacks: min(i) > r_priority
    good = [a for a in candidates if min_i(a) > r_priority]
    if good:
        return min(good, key=min_i)
    return max(candidates, key=min_i)
