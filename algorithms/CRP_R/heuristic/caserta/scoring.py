"""
Caserta et al. (2012) destination-stack scoring rule — Eq. (11).

Given the topmost relocating block r (priority = r_priority), choose:
  s* = argmin_{min(i) > r}  min(i)   if any stack qualifies
  s* = argmax_{i}            min(i)   otherwise

min(i) = priority of the lowest-numbered (earliest-retrieved) block
         in stack i.  Empty stacks are treated as min(i) = N+1
         (infinitely good — no future conflict).

Reference
---------
M. Caserta, S. Schwarze, S. Voß,
"A mathematical formulation and complexity considerations for the
 blocks relocation problem",
European Journal of Operational Research 219 (2012) 96–104.
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
    using Caserta et al. (2012) Eq. (11).

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
        # Choose good stack with LOWEST min(i) — conserve high-value slots
        return min(good, key=lambda a: (min_i(a), int(a)))
    else:
        # No good stack — choose stack with HIGHEST min(i) — delay re-relocation
        return max(candidates, key=lambda a: (min_i(a), -int(a)))
