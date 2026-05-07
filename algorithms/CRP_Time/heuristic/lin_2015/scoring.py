"""
Lin, Lee & Lee (2015) position-priority scoring rule.

Reference
---------
D.-Y. Lin, Y.-J. Lee, Y. Lee,
"The container retrieval problem with respect to relocation",
Transportation Research Part C 52 (2015) 132–143.

Paper context
-------------
Lin et al. extend Lee & Lee (2010) into a fast, rule-based heuristic that
explicitly considers RMGC working time (gantry / trolley / acceleration /
spreader) on top of the relocation count.  Their scoring rule evaluates
each candidate destination stack with a weighted combination of:

    1. A large penalty when the blocker would NOT be well-placed at s
       (i.e. it would need to be relocated again in the future).
    2. A severity term proportional to how badly the new placement would
       violate the "later-retrieved on top" invariant.
    3. The crane travel / carry time needed to deposit the blocker at s.

The destination minimising this composite score is chosen.  Defaults
``P_r = 30`` and ``P_b = 300`` follow the parameter values used in
subsequent literature (e.g. Shin et al., TRC 2026) as the canonical
"Lin" baseline.

This module is algorithm-agnostic: it only needs the CRP_R / CRP_Time
environment, its current blocker, and a ``KinematicsModel`` read from
``env.config.extra``.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from core.objectives import KinematicsModel


# ================================================================ #
#  Kinematics helpers                                                #
# ================================================================ #

def _carry_time(
    kin: KinematicsModel,
    src: Tuple[int, int],
    dst: Tuple[int, int],
) -> float:
    """
    Time to carry a picked-up blocker from *src* to *dst*:
        travel(src → dst) + spreader handling

    The reposition-to-src cost is shared by every candidate (the crane is
    already at src after picking up the blocker), so it is factored out of
    the ranking and omitted here.
    """
    return kin.travel_time(src, dst) + kin.spreader_s


# ================================================================ #
#  Stack features                                                    #
# ================================================================ #

def _min_priority(stk) -> int:
    """Lowest (earliest-retrieved) priority currently in the stack.

    Empty stacks return a very large sentinel so they are treated as
    'always well-placed' for any blocker.
    """
    if stk.is_empty:
        return 1 << 30
    return int(min(c.priority for c in stk.containers))


def _is_well_placed(stk, blocker_priority: int) -> bool:
    """Blocker is well-placed at ``stk`` iff every container already in
    ``stk`` has a strictly larger (later-retrieved) priority."""
    return _min_priority(stk) > blocker_priority


# ================================================================ #
#  Lin et al. (2015) destination score                               #
# ================================================================ #

def lin_score(
    env,
    dst_action: int,
    blocker_priority: int,
    src_key: Tuple[int, int],
    kin: KinematicsModel,
    P_r: float,
    P_b: float,
) -> Tuple[float, int]:
    """
    Composite score for a single candidate destination (lower is better).

        score = P_b * 1[not well_placed]
              + P_r * max(0, blocker_priority - min_priority(dst))
              + carry_time(src → dst)

    The second return value is a stable tie-break on the flat action index
    so results are deterministic under ties.
    """
    dst  = env._action_to_stack(int(dst_action))
    stk  = env.yard.stacks[dst]
    minp = _min_priority(stk)

    bad_flag = 0.0 if minp > blocker_priority else 1.0
    severity = max(0, blocker_priority - minp) if minp != (1 << 30) else 0

    travel = _carry_time(kin, src_key, dst)

    score = P_b * bad_flag + P_r * float(severity) + float(travel)
    return score, int(dst_action)


# ================================================================ #
#  Top-level action selector                                         #
# ================================================================ #

def lin_select_action(
    env,
    mask: Optional[np.ndarray] = None,
    P_r: float = 30.0,
    P_b: float = 300.0,
) -> int:
    """
    Choose the destination stack index for the current topmost blocker
    using the Lin, Lee & Lee (2015) rule.

    Parameters
    ----------
    env  : CRP_R or CRP_Time environment (priority-based fixed order)
    mask : optional boolean action mask (non-full stacks)
    P_r  : severity weight (default 30, Shin 2026)
    P_b  : well-placed violation weight (default 300, Shin 2026)

    Returns
    -------
    Flat action index compatible with CRP_R.step().
    """
    if mask is None:
        mask = env._get_info().get("action_mask")

    target = env._get_target_container()
    if target is None:
        return 0

    src_stack = env.yard._find_stack(target)
    if src_stack is None or src_stack.top == target:
        # Target is already on top; CRP_R will auto-retrieve, no choice needed.
        return 0

    blocker    = src_stack.top
    p_blk      = int(blocker.priority)
    src_key    = (src_stack.bay, src_stack.row)
    n_stacks   = env.action_space.n

    valid: List[int] = (
        list(range(n_stacks))
        if mask is None
        else [int(i) for i in np.where(mask)[0]]
    )
    candidates = [
        a for a in valid
        if env._action_to_stack(int(a)) != src_key
    ]
    if not candidates:
        return 0

    kin = KinematicsModel.from_config_extra(env.config.extra)

    return min(
        candidates,
        key=lambda a: lin_score(env, a, p_blk, src_key, kin, P_r, P_b),
    )
