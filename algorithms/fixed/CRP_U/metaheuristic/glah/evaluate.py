"""
Evaluation heuristic for GLAHHeuristic.

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
corresponding original algorithm paper.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import List, Optional

from .layout import (
    GlahContainer, GlahLayout, GlahOp, GlahState,
    INF_PRI, FLOOR_PRI,
    _get_nearest_target, lower_bound,
)


# ================================================================ #
#  Stack selection helpers                                          #
# ================================================================ #

def _find_support_stack_min_capacity(
    inst:    GlahLayout,
    label:   int,
    without: List[int],
) -> int:
    """
    Java: findSupportStackMinCapacity.
    Return stack s with smallest support_capacity(s) ≥ label,
    not in `without`, not full.  Return -1 if none.
    """
    best_cap = INF_PRI + 1
    best_s   = -1
    for s in range(1, inst.S + 1):
        if s in without or inst.height[s] >= inst.H:
            continue
        cap = inst.support_capacity(s)
        if cap >= label and cap < best_cap:
            best_cap = cap
            best_s   = s
    return best_s


def _find_stack_max_capacity(
    inst:    GlahLayout,
    without: List[int],
) -> int:
    """
    Java: findStackMaxCapacity.
    Return non-full stack s (not in `without`) with largest support_capacity.
    Return -1 if none.
    """
    best_cap = -1
    best_s   = -1
    for s in range(1, inst.S + 1):
        if s in without or inst.height[s] >= inst.H:
            continue
        cap = inst.support_capacity(s)
        if cap > best_cap:
            best_cap = cap
            best_s   = s
    return best_s


# ================================================================ #
#  Gap utilization                                                  #
# ================================================================ #

def _gap_utilize(
    state:   GlahState,
    ut_s:    int,
    p_stack: int,
    blocker: GlahContainer,
) -> None:
    """
    Port of Probing.evaluationHeuristic inner "interlaying" loop.
    Before relocating `blocker` to `p_stack`, squeeze in additional
    BG relocations from other stacks to fill p_stack's gap.

    Condition: blocker.priority ≤ support_capacity(p_stack) AND height < H-1.
    """
    inst = state.inst
    while (blocker.priority <= inst.support_capacity(p_stack) and
           inst.height[p_stack] <= inst.H - 2):
        best_pri = 0
        best_s   = -1
        for i in range(1, inst.S + 1):
            if i == p_stack or i == ut_s or inst.height[i] == 0:
                continue
            top = inst.top_container(i)
            if top is None:
                continue
            if (not inst.is_top_well_placed(i) and
                    blocker.priority <= top.priority <= inst.support_capacity(p_stack) and
                    top.priority > best_pri):
                best_pri = top.priority
                best_s   = i
        if best_s == -1:
            break
        gc_squeeze = inst.top_container(best_s)
        state.go_one_step(GlahOp(gc_squeeze, best_s, p_stack))
        state.try_retrievals_from(best_s)


# ================================================================ #
#  Evaluation heuristic (main entry point)                         #
# ================================================================ #

def evaluation_heuristic(state: GlahState) -> None:
    """
    Port of Probing.evaluationHeuristic(state).
    Clears the bay greedily and updates state.best.
    """
    state.try_retrievals()
    inst = state.inst

    while not inst.is_empty():
        ut = _get_nearest_target(inst)
        if ut is None:
            break

        ut_uid   = ut.uid
        ut_tier  = inst.at_tier[ut_uid]
        ut_stack = inst.at_stack[ut_uid]

        # Relocate all blockers above c*
        while ut_tier < inst.height[ut_stack]:
            c = inst.top_container(ut_stack)
            if c is None:
                break

            p_stack = _find_support_stack_min_capacity(
                inst, c.priority, [ut_stack]
            )

            if p_stack == -1:
                # Case B: try to vacate a well-placed stack top
                cv         = None
                largest_cv = 0
                v_stack    = -1
                found_p    = -1

                for i in range(1, inst.S + 1):
                    if i == ut_stack:
                        continue
                    itop = inst.top_container(i)
                    if itop is None:
                        continue
                    if (inst.support_capacity_except_top(i) >= itop.priority and
                            inst.support_capacity_except_top(i) >= c.priority):
                        j = _find_support_stack_min_capacity(
                            inst, itop.priority, [ut_stack, i]
                        )
                        if j != -1 and itop.priority > largest_cv:
                            largest_cv = itop.priority
                            found_p    = i
                            v_stack    = j
                            cv         = itop

                if found_p != -1:
                    # vacate: move cv from found_p → v_stack, then c → found_p
                    state.go_one_step(GlahOp(cv, found_p, v_stack))
                    # (state.try_retrievals_from(found_p) skipped per Java comment)
                    p_stack = found_p
                else:
                    # Case C: fallback – stack with max support_capacity
                    p_stack = _find_stack_max_capacity(inst, [ut_stack])
                    if p_stack == -1:
                        break   # no room anywhere – shouldn't happen

                    # Special check (Java Case 3 equivalent):
                    # if p_stack is almost full AND c is not the smallest above c*
                    if (inst.height[p_stack] == inst.H - 1 and
                            inst.height[ut_stack] - ut_tier >= 2):
                        min_above = min(
                            inst.bay[ut_stack][jt].priority
                            for jt in range(ut_tier + 1, inst.height[ut_stack] + 1)
                            if inst.bay[ut_stack][jt] is not None
                        )
                        if c.priority != min_above:
                            alt = _find_stack_max_capacity(inst, [ut_stack, p_stack])
                            if alt != -1:
                                p_stack = alt

            # Gap utilization (interlaying) before relocating c → p_stack
            if p_stack != -1:
                _gap_utilize(state, ut_stack, p_stack, c)

            # Relocate c → p_stack
            if p_stack == -1:
                break
            state.go_one_step(GlahOp(c, ut_stack, p_stack))

        state.try_retrievals()

    state.update_best()
