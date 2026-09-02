"""
SSI destination scoring for Lin2015Heuristic.

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
corresponding original algorithm paper.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import List, Optional, Tuple


# ================================================================ #
#  Stack feature helpers                                            #
# ================================================================ #

def _min_priority(stk) -> float:
    """Minimum priority in *stk* (INF for empty stacks)."""
    if stk.is_empty:
        return float("inf")
    return float(min(c.priority for c in stk.containers))


def _top_priority(stk, n_containers: int) -> float:
    """Top-of-stack priority. Empty stacks return a large sentinel."""
    if stk.is_empty:
        return float(n_containers * 10)
    return float(stk.top.priority)


def _stack_height(stk) -> int:
    return len(stk.containers)


# ================================================================ #
#  Public entry point                                               #
# ================================================================ #

def lin_compute_moves(
    env,
    mask,
    P_r: float = 30.0,
    P_b: float = 300.0,
    restricted: bool = False,
) -> List[int]:
    """
    Compute the next action(s) for one 'decision step' of Lin (2015).

    Returns a list of flat destination-stack action indices to execute
    sequentially via env.step().  Usually length 1; Rule 2 can prepend
    additional pre-move actions.

    Parameters
    ----------
    env        : CRP_Time / CRP_R gymnasium environment
    mask       : boolean action mask (non-full stacks)
    P_r        : row-index weight for SSI scoring (paper default 30)
    P_b        : bay-distance weight for SSI scoring (paper default 300)
    restricted : if True, skip Rule 2 pre-moves
    """
    target = env._get_target_container()
    if target is None:
        return []

    src_stack = env.yard._find_stack(target)
    if src_stack is None or src_stack.top == target:
        return []

    target_top_priority = float(src_stack.top.priority)
    src_bay  = src_stack.bay
    n_bays   = env.config.num_bays
    n_rows   = env.config.num_rows
    max_tiers = env.config.max_tiers
    n_containers = env.config.num_containers

    n_stacks = n_bays * n_rows
    valid_actions: List[int] = (
        list(range(n_stacks))
        if mask is None
        else [int(i) for i in range(n_stacks) if mask[i]]
    )

    # Exclude the source stack from destinations
    src_action = (src_bay - 1) * n_rows + (src_stack.row - 1)
    candidates = [a for a in valid_actions if a != src_action]
    if not candidates:
        return []

    def get_stk(action: int):
        key = env._action_to_stack(action)
        return env.yard.stacks.get(key)

    def minp(action: int) -> float:
        stk = get_stk(action)
        return _min_priority(stk) if stk else float("inf")

    def topp(action: int) -> float:
        stk = get_stk(action)
        return _top_priority(stk, n_containers) if stk else float(n_containers * 10)

    def height(action: int) -> int:
        stk = get_stk(action)
        return _stack_height(stk) if stk else max_tiers

    def bay_of(action: int) -> int:
        return action // n_rows + 1   # 1-indexed

    def row_of(action: int) -> int:
        return action % n_rows + 1    # 1-indexed

    # ── Identify ideal stacks ──────────────────────────────────── #
    ideal = [a for a in candidates if minp(a) > target_top_priority]

    if ideal:
        # Rule 1: SSI = min_priority + P_r * row + P_b * |bay - src_bay|
        def ssi(a: int) -> float:
            return (minp(a)
                    + P_r * row_of(a)
                    + P_b * abs(bay_of(a) - src_bay))

        best_ssi = min(ssi(a) for a in ideal)
        tied = [a for a in ideal if ssi(a) == best_ssi]
        import random
        dest_action = random.choice(tied)

        result: List[int] = []

        if not restricted:
            # Rule 2: pre-move candidates from other stacks into dest
            while True:
                dest_stk = get_stk(dest_action)
                spare = max_tiers - _stack_height(dest_stk)
                if spare < 2:
                    break

                dest_minp = minp(dest_action)
                dest_top  = topp(dest_action)

                # Candidate pre-move sources: top_priority in
                # (target_top_priority, dest_top) AND within 5 below dest_minp
                pre_candidates = [
                    a for a in valid_actions
                    if a != src_action
                    and a != dest_action
                    and target_top_priority < topp(a) < dest_top
                    and dest_minp - 5 < topp(a) < dest_minp
                ]
                if not pre_candidates:
                    break

                # Pick pre-source with maximum top_priority
                pre_src = max(pre_candidates, key=topp)
                # Execute: move pre_src's top → dest
                result.append(dest_action)

                # Rebuild mask after imaginary move (approximate):
                # just update valid_actions to exclude newly-full stacks
                # Actual validity is handled by env.step()
                break  # one pre-move per invocation; outer loop re-calls

            result.append(dest_action)
        else:
            result.append(dest_action)

        return result

    else:
        # Rule 3: no ideal stacks → pick stack with maximum min_priority
        max_minp = max(minp(a) for a in candidates)
        best_r3  = [a for a in candidates if minp(a) == max_minp]
        import random
        return [random.choice(best_r3)]
