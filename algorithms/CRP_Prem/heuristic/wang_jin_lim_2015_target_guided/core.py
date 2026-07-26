"""
Candidate selection, relocation, and giant moves for the Wang, Jin, Lim
(2015) target-guided heuristic (§4).

Sections implemented
---------------------
§4.1  Candidate container / candidate stack selection.
§4.2  Target-pair evaluation ``(f(c,s), fixed_height[s])`` (smallF-lowT).
§4.3  Giant move (Case 1: same-stack target; Case 2: different-stack
      target), each with its 3 / 4 ``nslot``-based sub-scenarios.
§4.4  Container relocation with the 5-priority destination rule and
      one-move-at-a-time fulfillment.
"""

from __future__ import annotations

from typing import List, Optional, Set, Tuple

from .state import (
    Move,
    Pos,
    State,
    apply_move,
    borrow_fixed_top,
    commit_fix,
    group_of,
    is_clean_stack,
    is_dummy,
    largest_dirty_group,
    num_above,
    num_empty,
    return_fixed_top,
    stack_height,
    top_is_fixed,
    unfixed_count,
)


class InfeasibleInstance(Exception):
    pass


# ================================================================== #
# §4.1  Candidate containers / stacks                                 #
# ================================================================== #

def candidate_containers(state: State, g: int) -> List[Pos]:
    """All unfixed containers with group label ``g``."""
    out: List[Pos] = []
    for s, arr in state.stacks.items():
        for t in range(state.fixed_height[s], len(arr)):
            if arr[t] == g:
                out.append((s, t))
    return out


def candidate_stacks(state: State, g: int) -> List[int]:
    """
    §4.1: stack ``s`` (not the dummy stack) such that
      (i)  uf(s) <= sum_{i != s} e(i)   [enough external room to evacuate]
      (ii) s can accept group ``g`` once its unfixed containers are evacuated.
    """
    total_empty = sum(num_empty(state, i) for i in state.stacks)
    out: List[int] = []
    for s in state.stacks:
        if is_dummy(state, s):
            continue
        if state.fixed_height[s] >= state.max_tiers:
            continue
        uf_s = unfixed_count(state, s)
        ext_empty = total_empty - num_empty(state, s)
        if uf_s > ext_empty:
            continue
        fh = state.fixed_height[s]
        if fh > 0 and state.stacks[s][fh - 1] < g:
            # Non-increasing order forbids stacking a larger label on a
            # smaller fixed one; equal labels are fine.
            continue
        out.append(s)
    out.sort(key=lambda s: (0 if unfixed_count(state, s) == 0 else 1, s))
    return out


def auto_fix_clean_prefix(state: State, g: int) -> None:
    """
    Algorithm 1 line 4: mark already-clean containers with label ``g``
    (sitting exactly at the lowest unfixed slot, in the correct order) as
    fixed without any move.

    The dummy stack (CPMPDS) is excluded: its containers must always remain
    unfixed so they stay eligible for relocation and the dummy ends empty.
    """
    for s in state.stacks:
        if is_dummy(state, s):
            continue
        arr = state.stacks[s]
        while state.fixed_height[s] < len(arr) and arr[state.fixed_height[s]] == g:
            commit_fix(state, s)


# ================================================================== #
# §4.2  Target-pair evaluation                                        #
# ================================================================== #

def f_cost(state: State, c: Pos, s: int) -> int:
    """f(c, s): rough movement cost estimate for fixing c into stack s."""
    uf_s = unfixed_count(state, s)
    if c[0] == s:
        return uf_s
    return uf_s + num_above(state, c)


def evaluate_pair(state: State, c: Pos, s: int) -> Tuple[int, int]:
    """(f(c, s), fixed_height[s]) -- smallF-lowT scheme (paper's final choice)."""
    return (f_cost(state, c, s), state.fixed_height[s])


def select_target_pair(
    state: State, containers: List[Pos], stacks: List[int],
) -> Tuple[Pos, int]:
    best: Optional[Tuple[Tuple[int, int], Pos, int]] = None
    for c in containers:
        for s in stacks:
            key = evaluate_pair(state, c, s)
            if best is None or key < best[0]:
                best = (key, c, s)
    if best is None:
        raise InfeasibleInstance("no candidate pair available")
    return best[1], best[2]


# ================================================================== #
# §4.4  Relocation with 5-priority destination rule + fulfillment      #
# ================================================================== #

def _choose_relocation_destination(
    state: State, c: Pos, target_group: int, forbidden: Set[int],
) -> Optional[int]:
    g_c = group_of(state, c)
    candidates = [
        s for s in state.stacks
        if s != c[0] and s not in forbidden and stack_height(state, s) < state.max_tiers
    ]
    if not candidates:
        return None

    # P1: clean stack that can accommodate c (top label >= g_c); smallest top label.
    p1 = []
    for s in candidates:
        if not is_clean_stack(state, s):
            continue
        arr = state.stacks[s]
        top = arr[-1] if arr else None
        if top is None or top >= g_c:
            p1.append((top if top is not None else 10 ** 9, s))
    if p1:
        p1.sort()
        return p1[0][1]

    # P2: dirty stack with ldg <= g_c; prefer largest ldg.
    p2 = []
    for s in candidates:
        if is_clean_stack(state, s):
            continue
        ldg = largest_dirty_group(state, s)
        if ldg is not None and ldg <= g_c:
            p2.append((-ldg, s))
    if p2:
        p2.sort()
        return p2[0][1]

    # P3: dirty stack with g_c < ldg < target_group; prefer smallest ldg.
    p3 = []
    for s in candidates:
        if is_clean_stack(state, s):
            continue
        ldg = largest_dirty_group(state, s)
        if ldg is not None and g_c < ldg < target_group:
            p3.append((ldg, s))
    if p3:
        p3.sort()
        return p3[0][1]

    # P4: clean stack with largest top group label (fallback, may not accommodate).
    p4 = []
    for s in candidates:
        if not is_clean_stack(state, s):
            continue
        arr = state.stacks[s]
        top = arr[-1] if arr else -1
        p4.append((-top, s))
    if p4:
        p4.sort()
        return p4[0][1]

    # P5: dirty stack with ldg == target_group.
    p5 = [
        s for s in candidates
        if not is_clean_stack(state, s) and largest_dirty_group(state, s) == target_group
    ]
    if p5:
        return sorted(p5)[0]

    return None


def relocate_container(
    state: State,
    c: Pos,
    target_group: int,
    forbidden: Set[int],
    max_fulfillments: int = 64,
) -> List[Move]:
    """
    §4.4: relocate the container currently at the top of ``c``'s stack to an
    appropriate destination, applying at most one fulfillment move per
    re-selection. Returns the sequence of applied moves.
    """
    moves: List[Move] = []
    src = c[0]

    for _ in range(max_fulfillments + 1):
        top_t = len(state.stacks[src]) - 1
        g_c = state.stacks[src][top_t]
        dest = _choose_relocation_destination(state, (src, top_t), target_group, forbidden)
        if dest is None:
            raise InfeasibleInstance(f"no relocation destination for container at stack {src}")

        dest_arr = state.stacks[dest]
        can_fulfill = is_clean_stack(state, dest) and bool(dest_arr) and dest_arr[-1] >= g_c
        if can_fulfill:
            dc = dest_arr[-1]
            best: Optional[Tuple[int, int]] = None  # (label, stack)
            for s in state.stacks:
                if s in forbidden or s in (dest, src):
                    continue
                arr = state.stacks[s]
                if not arr or is_clean_stack(state, s):
                    continue
                label = arr[-1]
                if g_c < label <= dc and (best is None or label > best[0]):
                    best = (label, s)
            if best is not None:
                _, s = best
                if apply_move(state, s, dest):
                    moves.append((s, dest))
                    continue

        if not apply_move(state, src, dest):
            raise InfeasibleInstance(f"relocation move ({src},{dest}) failed")
        moves.append((src, dest))
        return moves

    raise InfeasibleInstance("fulfillment loop did not converge")


def relocate_top_of(state: State, s: int, target_group: int, forbidden: Set[int]) -> List[Move]:
    top_t = len(state.stacks[s]) - 1
    return relocate_container(state, (s, top_t), target_group, forbidden)


def _apply(state: State, src: int, dst: int, moves: List[Move], what: str) -> None:
    if not apply_move(state, src, dst):
        raise InfeasibleInstance(f"move ({src},{dst}) failed: {what}")
    moves.append((src, dst))


# ================================================================== #
# §4.3  Giant move                                                     #
# ================================================================== #

def choose_temp_stack(state: State, exclude: Set[int]) -> int:
    """Highest non-full stack, dirty preferred, excluding ``exclude``."""
    cands = [s for s in state.stacks if s not in exclude and stack_height(state, s) < state.max_tiers]
    if not cands:
        raise InfeasibleInstance("no temporary stack available")
    cands.sort(key=lambda s: (
        -stack_height(state, s),
        0 if not is_clean_stack(state, s) else 1,
        s,
    ))
    return cands[0]


def _choose_top_container_for_borrow(state: State, stacks: List[int], ts_is_clean: bool) -> Optional[int]:
    """§4.3.1 Case-1 scenario 3: pick a top container from ``stacks`` to vacate."""
    dirty_tops = [(state.stacks[s][-1], s) for s in stacks if state.stacks[s] and not is_clean_stack(state, s)]
    clean_tops = [(state.stacks[s][-1], s) for s in stacks if state.stacks[s] and is_clean_stack(state, s)]
    if ts_is_clean:
        if dirty_tops:
            return max(dirty_tops)[1]
        if clean_tops:
            return max(clean_tops)[1]
    else:
        if dirty_tops:
            return min(dirty_tops)[1]
        if clean_tops:
            return min(clean_tops)[1]
    return None


def giant_move_case1(state: State, target: Pos, target_stack: int) -> List[Move]:
    """§4.3.1: origin of the target container is ``target_stack`` itself."""
    moves: List[Move] = []
    sn = target_stack
    target_group = group_of(state, target)

    # Step 1: relocate everything currently above the target.
    while len(state.stacks[sn]) - 1 > target[1]:
        moves.extend(relocate_top_of(state, sn, target_group, forbidden={sn}))

    ts = choose_temp_stack(state, exclude={sn})
    AS = [s for s in state.stacks if s not in (sn, ts)]
    nslot = sum(num_empty(state, s) for s in AS)
    threshold = unfixed_count(state, sn) - 1  # = uf(sn) - 1, target still counted in uf(sn)

    if nslot >= threshold:
        # Scenario 1.
        _apply(state, sn, ts, moves, "case1/s1: park target in ts")
        while unfixed_count(state, sn) > 0:
            moves.extend(relocate_top_of(state, sn, target_group, forbidden={sn, ts}))
        _apply(state, ts, sn, moves, "case1/s1: restore target")
        commit_fix(state, sn)
        return moves

    if 0 < nslot < threshold:
        # Scenario 2.
        _apply(state, sn, ts, moves, "case1/s2: park target in ts")
        for _ in range(max(0, nslot - 1)):
            if unfixed_count(state, sn) == 0:
                break
            moves.extend(relocate_top_of(state, sn, target_group, forbidden={sn, ts}))
        dest_slot = next((s for s in AS if num_empty(state, s) > 0), None)
        if dest_slot is None:
            raise InfeasibleInstance("case1/s2: no empty AS slot for target")
        _apply(state, ts, dest_slot, moves, "case1/s2: move target into last AS slot")
        while unfixed_count(state, sn) > 0:
            _apply(state, sn, ts, moves, "case1/s2: drain sn to ts")
        _apply(state, dest_slot, sn, moves, "case1/s2: restore target")
        commit_fix(state, sn)
        return moves

    # Scenario 3: nslot == 0.
    ts_clean = is_clean_stack(state, ts)
    borrow_from = _choose_top_container_for_borrow(state, AS, ts_clean)
    if borrow_from is None:
        raise InfeasibleInstance("case1/s3: no container available to borrow")

    was_top_fixed = top_is_fixed(state, borrow_from)
    if was_top_fixed:
        if not borrow_fixed_top(state, borrow_from, ts):
            raise InfeasibleInstance("case1/s3: cannot borrow fixed top")
        moves.append((borrow_from, ts))
    else:
        _apply(state, borrow_from, ts, moves, "case1/s3: borrow top container")

    _apply(state, sn, borrow_from, moves, "case1/s3: park target in freed slot")
    while unfixed_count(state, sn) > 0:
        _apply(state, sn, ts, moves, "case1/s3: drain sn to ts")
    _apply(state, borrow_from, sn, moves, "case1/s3: restore target")
    commit_fix(state, sn)

    if was_top_fixed:
        if not return_fixed_top(state, ts, borrow_from):
            raise InfeasibleInstance("case1/s3: cannot restore borrowed fixed container")
        moves.append((ts, borrow_from))
    return moves


def giant_move_case2(state: State, target: Pos, target_stack: int) -> List[Move]:
    """§4.3.2: origin of the target container differs from ``target_stack``."""
    moves: List[Move] = []
    sn = target_stack
    origin = target[0]
    target_group = group_of(state, target)

    AS = [s for s in state.stacks if s not in (sn, origin)]
    nslot = sum(num_empty(state, s) for s in AS)
    f_val = f_cost(state, target, sn)
    o_val = num_above(state, target)

    def relocate_priority_from(origin_or_sn: bool) -> bool:
        """Relocate whichever of {top(origin) above target, top(sn)} has the larger label."""
        cand: List[Tuple[int, int]] = []
        if len(state.stacks[origin]) - 1 > target[1]:
            cand.append((state.stacks[origin][-1], origin))
        if unfixed_count(state, sn) > 0:
            cand.append((state.stacks[sn][-1], sn))
        if not cand:
            return False
        cand.sort(reverse=True)
        _, s = cand[0]
        moves.extend(relocate_top_of(state, s, target_group, forbidden={sn, origin}))
        return True

    if nslot >= f_val:
        # Scenario 1.
        while relocate_priority_from(True):
            pass
        _apply(state, origin, sn, moves, "case2/s1: place target")
        commit_fix(state, sn)
        return moves

    if o_val + 1 <= nslot < f_val:
        # Scenario 2.
        budget = nslot - 1
        while budget > 0 and relocate_priority_from(True):
            budget -= 1
        dest_slot = next((s for s in AS if num_empty(state, s) > 0), None)
        if dest_slot is None:
            raise InfeasibleInstance("case2/s2: no empty AS slot for target")
        _apply(state, origin, dest_slot, moves, "case2/s2: move target into AS slot")
        while unfixed_count(state, sn) > 0:
            _apply(state, sn, origin, moves, "case2/s2: drain sn to origin")
        _apply(state, dest_slot, sn, moves, "case2/s2: restore target")
        commit_fix(state, sn)
        return moves

    if 1 <= nslot < o_val + 1:
        # Scenario 3.
        for _ in range(max(0, nslot - 1)):
            if len(state.stacks[origin]) - 1 <= target[1]:
                break
            moves.extend(relocate_top_of(state, origin, target_group, forbidden={sn, origin}))
        while len(state.stacks[origin]) - 1 > target[1]:
            _apply(state, origin, sn, moves, "case2/s3: park leftover above-target containers in sn")
        dest_slot = next((s for s in AS if num_empty(state, s) > 0), None)
        if dest_slot is None:
            raise InfeasibleInstance("case2/s3: no empty AS slot for target")
        _apply(state, origin, dest_slot, moves, "case2/s3: move target into AS slot")
        while unfixed_count(state, sn) > 0:
            _apply(state, sn, origin, moves, "case2/s3: drain sn (incl. parked) to origin")
        _apply(state, dest_slot, sn, moves, "case2/s3: restore target")
        commit_fix(state, sn)
        return moves

    # Scenario 4: nslot == 0.
    dirty_tops = [(state.stacks[s][-1], s) for s in AS if state.stacks[s] and not is_clean_stack(state, s)]
    clean_tops = [(state.stacks[s][-1], s) for s in AS if state.stacks[s] and is_clean_stack(state, s)]
    if dirty_tops:
        _, cs = min(dirty_tops)
    elif clean_tops:
        _, cs = min(clean_tops)
    else:
        raise InfeasibleInstance("case2/s4: no container to borrow")

    was_top_fixed = top_is_fixed(state, cs)
    if was_top_fixed:
        if not borrow_fixed_top(state, cs, sn):
            raise InfeasibleInstance("case2/s4: cannot borrow fixed top")
        moves.append((cs, sn))
    else:
        _apply(state, cs, sn, moves, "case2/s4: park borrowed container in sn")

    parked_count = 0
    while len(state.stacks[origin]) - 1 > target[1]:
        _apply(state, origin, sn, moves, "case2/s4: park leftover above-target containers in sn")
        parked_count += 1

    _apply(state, origin, cs, moves, "case2/s4: park target in freed slot")

    # The borrowed container sits just below the leftovers we parked on sn;
    # drain those first so we can reach it.
    for _ in range(parked_count):
        _apply(state, sn, origin, moves, "case2/s4: drain parked leftovers to origin")

    if was_top_fixed:
        if not return_fixed_top(state, sn, cs):
            raise InfeasibleInstance("case2/s4: cannot restore borrowed fixed container")
        moves.append((sn, cs))
    else:
        _apply(state, sn, origin, moves, "case2/s4: drain borrowed container to origin")

    while unfixed_count(state, sn) > 0:
        _apply(state, sn, origin, moves, "case2/s4: drain sn to origin")

    _apply(state, cs, sn, moves, "case2/s4: restore target")
    commit_fix(state, sn)
    return moves


def giant_move(state: State, target: Pos, target_stack: int) -> List[Move]:
    if target[0] == target_stack:
        return giant_move_case1(state, target, target_stack)
    return giant_move_case2(state, target, target_stack)
