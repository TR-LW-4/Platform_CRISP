"""
NBC/DC/BC task definitions for Wang2026GRASP.

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
corresponding original algorithm paper.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Tuple


# ================================================================ #
#  RC / stack-type helpers (Def 1)                                 #
# ================================================================ #

def is_rc(c) -> bool:
    """True iff container c is a Rolled Container."""
    return bool(c.attrs.get("is_rolled", False))


def stack_only_rcs(stk) -> bool:
    """True iff stack is non-empty AND contains only RCs (Def 1: rolled stack)."""
    if stk.is_empty:
        return False
    return all(is_rc(c) for c in stk.containers)


def stack_has_oc(stk) -> bool:
    """True iff stack contains at least one OC."""
    if stk.is_empty:
        return False
    return any(not is_rc(c) for c in stk.containers)


# ================================================================ #
#  Container-echelon / stack-echelon (Def 4-5)                     #
# ================================================================ #

_HUGE = 10**9   # sentinel for "empty / rolled stack" echelon (paper: C+1)


def cdd_of(c, vessel_loaded: Sequence[int]) -> int:
    """cdd(c) = vt(c) − loaded[vs(c)]; 0 → c is currently retrievable."""
    if is_rc(c):
        return _HUGE
    g = c.group
    if g >= len(vessel_loaded):
        return _HUGE
    return c.priority - vessel_loaded[g]


def ce_of(c, vessel_loaded: Sequence[int]) -> int:
    """Container echelon (Def 4): rank-1 = candidate.  RCs return HUGE."""
    if is_rc(c):
        return _HUGE
    return cdd_of(c, vessel_loaded) + 1


def se_of(stk, vessel_loaded: Sequence[int]) -> int:
    """Stack echelon (Def 5)."""
    if stk.is_empty or stack_only_rcs(stk):
        return _HUGE
    best = _HUGE
    for c in stk.containers:
        if is_rc(c):
            continue
        v = ce_of(c, vessel_loaded)
        if v < best:
            best = v
    return best


def sec_of(stk, group: int, vessel_loaded: Sequence[int]) -> int:
    """Stack-echelon restricted to OCs sharing *group* with target."""
    if stk.is_empty:
        return _HUGE
    best = _HUGE
    for c in stk.containers:
        if is_rc(c) or c.group != group:
            continue
        v = ce_of(c, vessel_loaded)
        if v < best:
            best = v
    return best


# ================================================================ #
#  Bad container (Def 2)                                           #
# ================================================================ #

def is_bad_container(c, containers_below: Sequence) -> bool:
    """
    OC d is BAD iff any OC c below d in the same stack shares d's group and
    has strictly smaller priority (i.e., c is retrieved before d).
    RCs are never labelled BAD by this function (they are handled separately).
    """
    if is_rc(c):
        return False
    g, p = c.group, c.priority
    for d in containers_below:
        if is_rc(d):
            continue
        if d.group == g and d.priority < p:
            return True
    return False


def is_bad_if_placed_on(c, dst_stk) -> bool:
    """Would c be a BAD container if it were pushed on top of dst_stk now?"""
    if is_rc(c):
        return False
    below = dst_stk.containers   # entire current stack is below the pushed c
    return is_bad_container(c, below)


def compute_bad_map(env) -> Dict[int, bool]:
    """Return {id(c): True/False} bad-ness for every container in the yard."""
    out: Dict[int, bool] = {}
    for stk in env.yard.stacks.values():
        below: List = []
        for c in stk.containers:
            out[id(c)] = is_bad_container(c, below)
            below.append(c)
    return out


# ================================================================ #
#  Deadlock (Def 3) — relevant 4-cycle over OCs                    #
# ================================================================ #

def _wl_of(c, bad_map: Dict[int, bool]) -> bool:
    """Well-located = not bad AND not RC (only OCs participate in 4-cycles)."""
    return (not is_rc(c)) and (not bad_map.get(id(c), False))


def creates_deadlock_if_placed(
    env,
    c_group:    int,
    c_priority: int,
    dst_key:    Tuple[int, int],
    bad_map:    Dict[int, bool],
) -> bool:
    """
    True iff placing an OC (c_group, c_priority) on top of dst_key would
    complete a deadlock (relevant 4-cycle):

        c →_y d →_v d' →_y x →_v c

    All four nodes must be non-bad (well-located) OCs.
    """
    dst = env.yard.stacks.get(dst_key)
    if dst is None or dst.is_empty:
        return False

    # c would be non-bad only if no OC below in dst has same group and smaller vt.
    for d in dst.containers:
        if (not is_rc(d)) and d.group == c_group and d.priority < c_priority:
            return False

    for d in dst.containers:
        if not _wl_of(d, bad_map):
            continue
        vs_d, vt_d = d.group, d.priority
        for stk2 in env.yard.stacks.values():
            cs2 = stk2.containers
            for j, dp in enumerate(cs2):
                if is_rc(dp) or dp.group != vs_d or dp.priority <= vt_d:
                    continue
                if not _wl_of(dp, bad_map):
                    continue
                for x in cs2[:j]:
                    if (not is_rc(x)
                            and x.group == c_group
                            and x.priority < c_priority
                            and _wl_of(x, bad_map)):
                        return True
    return False


def is_deadlock_container(
    env,
    c,
    c_pos:   int,
    stk_key: Tuple[int, int],
    bad_map: Dict[int, bool],
) -> bool:
    """
    True iff non-bad OC c (at position c_pos, 0-indexed from bottom of stk_key)
    participates in a deadlock (as the "blocking" node of a relevant 4-cycle).
    """
    if is_rc(c) or not _wl_of(c, bad_map):
        return False
    vs_c, vt_c = c.group, c.priority
    stk = env.yard.stacks.get(stk_key)
    if stk is None:
        return False
    for d in stk.containers[:c_pos]:
        if not _wl_of(d, bad_map):
            continue
        vs_d, vt_d = d.group, d.priority
        for stk2 in env.yard.stacks.values():
            cs2 = stk2.containers
            for j, dp in enumerate(cs2):
                if is_rc(dp) or dp.group != vs_d or dp.priority <= vt_d:
                    continue
                if not _wl_of(dp, bad_map):
                    continue
                for x in cs2[:j]:
                    if (not is_rc(x)
                            and x.group == vs_c
                            and x.priority < vt_c
                            and _wl_of(x, bad_map)):
                        return True
    return False


# ================================================================ #
#  Above-target classification (used by TR §5.2.2)                #
# ================================================================ #

def classify_above_target(
    env,
    stk,
    target_pos: int,
    bad_map:    Dict[int, bool],
) -> Tuple[int, int, int, int]:
    """
    For the containers strictly above `target_pos` in stack `stk`, classify
    them into four disjoint counts:

        NBC – non-bad OCs (excluding deadlock)
        DC  – deadlock (non-bad OC participating in a 4-cycle)
        BC  – bad OCs
        RCB – RC blockers

    Note on tie-breakers (§5.2.2, "min NBC → max DC → min BC"):
    RCs are aggregated separately; they are treated similarly to bad OCs when
    breaking ties (`BC + RCB`) since both force at least one relocation.
    """
    stk_key = (stk.bay, stk.row)
    nbc = dc = bc = rcb = 0
    for pos, ca in enumerate(stk.containers[target_pos + 1:],
                             start=target_pos + 1):
        if is_rc(ca):
            rcb += 1
            continue
        if bad_map.get(id(ca), False):
            bc += 1
            continue
        # non-bad OC → check if it's a deadlock node
        if is_deadlock_container(env, ca, pos, stk_key, bad_map):
            dc += 1
        else:
            nbc += 1
    return nbc, dc, bc, rcb


# ================================================================ #
#  Candidate-stack enumeration (Def 4 aligned with 2-phase env)   #
# ================================================================ #

def topmost_retrievable_pos(env, stk) -> int:
    """0-indexed position (from bottom) of the topmost retrievable OC, or -1."""
    pos = -1
    for i, c in enumerate(stk.containers):
        if env._is_retrievable(c):
            pos = i
    return pos


def candidate_stacks(env) -> List[Tuple[int, int, int]]:
    """
    Return [(stack_action_idx, stack_key, topmost_retrievable_pos), ...]
    for every yard stack that currently exposes a retrievable OC.

    Aligning TR's "candidate containers" (Def 4) with the platform's
    2-phase env: each candidate stack maps 1-to-1 with the OC that
    ``_step_high`` would auto-target for it.
    """
    out: List[Tuple[int, int, int]] = []
    n = env._n_stacks
    for i in range(n):
        key = env._idx_to_stack(i)
        stk = env.yard.stacks.get(key)
        if stk is None or stk.is_empty:
            continue
        tp = topmost_retrievable_pos(env, stk)
        if tp < 0:
            continue
        out.append((i, key, tp))
    return out


# ================================================================ #
#  SRC (stacks containing at least one OC)                         #
# ================================================================ #

def src_stacks(env) -> List[Tuple[int, Tuple[int, int]]]:
    """
    SRC = {non-empty stacks that contain at least one OC}.
    Returns [(action_idx, key), ...] sorted by action_idx.
    """
    out: List[Tuple[int, Tuple[int, int]]] = []
    n = env._n_stacks
    for i in range(n):
        key = env._idx_to_stack(i)
        stk = env.yard.stacks.get(key)
        if stk is None:
            continue
        if stack_has_oc(stk):
            out.append((i, key))
    return out


def empty_or_rolled_stacks(env) -> List[Tuple[int, Tuple[int, int]]]:
    """Non-full stacks that are empty OR rolled (only-RC) — RC-preferred sinks."""
    out: List[Tuple[int, Tuple[int, int]]] = []
    n = env._n_stacks
    for i in range(n):
        key = env._idx_to_stack(i)
        stk = env.yard.stacks.get(key)
        if stk is None or stk.is_full:
            continue
        if stk.is_empty or stack_only_rcs(stk):
            out.append((i, key))
    return out
