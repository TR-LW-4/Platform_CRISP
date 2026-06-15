"""
Heuristic, metaheuristic and condensation algorithms for CRP-U.

Ported from Tricoire et al. (2018) C++ codebase (block-relocation-master).
Source files: brppolicy.cpp, safemoves.cpp, subsequence.cpp,
              rakesearch.cpp, pilotmethod.cpp, brpstate.cpp (condensation).

Implemented algorithms
----------------------
SM-1   (SafeMovesPolicy level 1)  – safe 1-relocates only
SM-2   (SafeMovesPolicy level 2)  – safe 1- or 2-relocates
SmSEQ-1 (SmartSubsequencePolicy level 1) – seq + SM-1
SmSEQ-2 (SmartSubsequencePolicy level 2) – seq + SM-2
RakeSearch  – BFS metaheuristic, C++ RakeSearch
PilotMethod – look-ahead metaheuristic, C++ PilotMethod

Condensation post-processors (applied after SmSEQ variants)
-----------------------------------------------------------
condense_jin     – classic (Jin et al.) condensation
condense_tricoire – improved (Tricoire 2018) condensation

All functions operate on BRPState objects (see state.py).
"""

from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional, Tuple

from .state import BRPState


# ═══════════════════════════════════════════════════════════════════════ #
#  Section 1 — forced relocation (LA heuristic)                          #
# ═══════════════════════════════════════════════════════════════════════ #

def _best_dest_la(state: BRPState, item: int) -> int:
    """
    Return best destination stack for LA-relocating *item*.

    Preference order (mirrors C++ bestDestForLaRelocate):
    1. Any stack with low > item  (no new conflict), preferring smallest low.
    2. If no such stack, any valid stack with highest low < item
       (latest possible conflict).
    """
    src = state.stack_for_item[item]
    best = -1
    for dst in range(state.W):
        if dst == src or state.height[dst] == state.H:
            continue
        if best < 0:
            best = dst
        elif state.low[dst] > item:
            low_b = state.low[best]
            low_d = state.low[dst]
            if low_b < item or (low_b > item and low_d < low_b):
                best = dst
        elif (
            state.low[dst] < item
            and state.low[best] < item
            and state.low[dst] > state.low[best]
        ):
            best = dst
    return best


def _la_relocate(state: BRPState, item: int) -> None:
    """Relocate *item* (must be the top of its stack) using the LA rule."""
    dst = _best_dest_la(state, item)
    if dst >= 0:
        state.relocate(state.stack_for_item[item], dst)


def _forced_move(state: BRPState) -> None:
    """Forced move: LA-relocate the top item blocking state.next_item."""
    src = state.stack_for_item[state.next_item]
    _la_relocate(state, state.top(src))


# ═══════════════════════════════════════════════════════════════════════ #
#  Section 2 — necessary relocates (candidate items for safe moves)      #
# ═══════════════════════════════════════════════════════════════════════ #

def _necessary_relocates(state: BRPState) -> List[Tuple[int, int]]:
    """
    List of (stack_idx, top_item) for stacks whose top item is badly placed.

    A top item is "badly placed" iff top(s) > low[s], i.e. there is a
    lower-index item somewhere below it in the same stack.
    (Equivalent to C++ necessaryRelocates.)
    """
    result: List[Tuple[int, int]] = []
    for s in range(state.W):
        if state.height[s] > 0 and state.top(s) > state.low[s]:
            result.append((s, state.top(s)))
    return result


# ═══════════════════════════════════════════════════════════════════════ #
#  Section 3 — safe 1-relocates                                          #
# ═══════════════════════════════════════════════════════════════════════ #

def _best_safe1_item(
    state: BRPState, src: int, item: int
) -> Tuple[int, int]:
    """
    Return (best_dst, best_impact) for safe-1-relocating *item* from *src*.
    Returns (-1, n+1) if no safe destination exists.
    """
    best_dst = -1
    best_impact = state.n + 1
    for dst in range(state.W):
        if dst != src and state.height[dst] < state.H:
            low_d = state.low[dst]
            if low_d >= item:
                impact = low_d - item
                if impact < best_impact:
                    best_dst, best_impact = dst, impact
    return best_dst, best_impact


def _best_safe1(
    state: BRPState, necessary: List[Tuple[int, int]]
) -> Tuple[int, int]:
    """Best safe 1-relocate across all necessary items.  Returns (src, dst)."""
    best_src = best_dst = -1
    best_impact = state.n + 1
    for src, item in necessary:
        dst, impact = _best_safe1_item(state, src, item)
        if dst >= 0 and impact < best_impact:
            best_src, best_dst, best_impact = src, dst, impact
    return best_src, best_dst


def _try_safe1(state: BRPState, necessary: List[Tuple[int, int]]) -> bool:
    """Perform best safe 1-relocate.  Returns True if one was done."""
    src, dst = _best_safe1(state, necessary)
    if src >= 0:
        state.relocate(src, dst)
        return True
    return False


# ═══════════════════════════════════════════════════════════════════════ #
#  Section 4 — safe 2-relocates                                          #
# ═══════════════════════════════════════════════════════════════════════ #

def _all_safe2(
    state: BRPState, necessary: List[Tuple[int, int]]
) -> List[Tuple[int, int]]:
    """
    Return list of (src, dst) for the first move of each qualifying safe
    2-relocate.  (One entry per intermediate stack s_to, as in C++ safe2Relocates.)
    """
    high = state.n * 4
    result: List[Tuple[int, int]] = []

    for s_to in range(state.W):
        best_diff = high
        best_src = best_dst = -1

        for s_from, item in necessary:
            if s_to == s_from or state.height[s_to] == 0:
                continue
            if state.top(s_to) >= item:     # top(s_to) must be < item
                continue
            # item3 = min of s_to except its top
            item3 = (
                min(state.stacks[s_to][:-1])
                if state.height[s_to] > 1
                else state.n + 1
            )
            if item3 < item:
                continue
            # check safe-1 for top(s_to)
            top_item = state.top(s_to)
            to_to, sr_impact = _best_safe1_item(state, s_to, top_item)
            if to_to < 0:
                continue
            diff = item3 - item + sr_impact
            if diff < best_diff:
                best_diff = diff
                best_src = s_to
                best_dst = to_to

        if best_src >= 0:
            result.append((best_src, best_dst))

    return result


def _best_safe2(
    state: BRPState, necessary: List[Tuple[int, int]]
) -> Tuple[int, int]:
    """Best single safe 2-relocate (minimum impact).  Returns (src, dst)."""
    high = state.n * 4
    best_src = best_dst = -1
    best_diff = high

    for s_to in range(state.W):
        for s_from, item in necessary:
            if s_to == s_from or state.height[s_to] == 0:
                continue
            if state.top(s_to) >= item:
                continue
            item3 = (
                min(state.stacks[s_to][:-1])
                if state.height[s_to] > 1
                else state.n + 1
            )
            if item3 < item:
                continue
            top_item = state.top(s_to)
            to_to, sr_impact = _best_safe1_item(state, s_to, top_item)
            if to_to < 0:
                continue
            diff = item3 - item + sr_impact
            if diff < best_diff:
                best_diff = diff
                best_src = s_to
                best_dst = to_to

    return best_src, best_dst


def _try_safe2(state: BRPState, necessary: List[Tuple[int, int]]) -> bool:
    src, dst = _best_safe2(state, necessary)
    if src >= 0:
        state.relocate(src, dst)
        return True
    return False


# ═══════════════════════════════════════════════════════════════════════ #
#  Section 5 — decreasing sequence helpers (SmSEQ)                       #
# ═══════════════════════════════════════════════════════════════════════ #

def _decreasing_sequence(state: BRPState) -> List[int]:
    """
    Return the decreasing sequence DS(s) starting from the top of
    next_item's stack, going toward the bottom.

    Returns items in top-to-bottom order.  Stops just before next_item.
    E.g.  stack [2,3,4,5] (bottom→top), next_item=2 → [5,4,3].
    """
    s = state.stack_for_item[state.next_item]
    stk = state.stacks[s]
    h = state.height[s]
    if h == 0 or stk[-1] == state.next_item:
        return []

    top_idx = h - 1
    bot_idx = top_idx
    i = top_idx - 1
    while i >= 0:
        if stk[i] == state.next_item:
            break
        if stk[i] <= stk[i + 1]:    # each step downward is ≤ previous
            bot_idx = i
            i -= 1
        else:
            break
    # items from top-index down to bot_idx, in top-to-bottom order
    return list(reversed(stk[bot_idx : top_idx + 1]))


def _best_subseq_fit(
    state: BRPState,
    seq: List[int],    # top-to-bottom; seq[0]=highest-index item in DS
    t: int,            # candidate target stack
    s_src: int,        # next_item's stack (source)
) -> Tuple[int, int]:
    """
    Compute (n_to_remove, cost) to fit *seq* into stack *t*.
    Translated from C++ SubsequencePolicy::bestSubsequenceFit.

    n_to_remove = items to LA-relocate out of t before placing seq.
    cost        = estimated extra moves introduced.
    """
    seq_size = len(seq)
    seq_bottom = seq[-1]          # bottom item of DS (highest priority in DS)

    stk_t = state.stacks[t]
    h_t = state.height[t]

    # free slots in all stacks except t and s_src
    n_free = sum(
        state.H - state.height[k]
        for k in range(state.W)
        if k != t and k != s_src
    )

    # find how many items (from bottom) to keep in t
    position = h_t
    while True:
        cond1 = (position + seq_size > state.H)
        cond2 = False
        if position > 0 and not cond1:
            min_below = min(stk_t[:position])
            n_remove = h_t - position
            cond2 = (seq_bottom > min_below and n_remove < n_free)
        if (cond1 or cond2) and position > 0:
            position -= 1
        else:
            break

    n_to_remove = h_t - position

    # cost of removing items from t[position:]
    cost = 0
    for i in range(n_to_remove):
        item_at = stk_t[position + i]
        is_bad = (item_at > state.low[t])
        present_cost = 0 if is_bad else 1
        future_cost = 1
        for k in range(state.W):
            if (
                k != t
                and state.low[k] >= item_at
                and state.low[k] != state.next_item
                and state.height[k] < state.H
            ):
                future_cost = 0
                break
        cost += present_cost + future_cost

    # extra cost for DS items creating conflicts in t below position
    local_min = (
        min(stk_t[:position]) if position > 0 else state.n + 1
    )
    for seq_item in seq:          # top→bottom (decreasing: seq[0] largest)
        if seq_item > local_min:
            cost += 1
        else:
            break                 # all remaining items in DS are ≤ local_min

    return n_to_remove, cost


def _make_room_for_subseq(state: BRPState) -> None:
    """
    Prepare the best target stack for the decreasing sequence by
    LA-relocating items out of it.  (C++ makeRoomForSubsequence)
    """
    s_src = state.stack_for_item[state.next_item]
    if state.top(s_src) == state.next_item:
        return

    seq = _decreasing_sequence(state)

    best_n_moves = -1
    best_cost = state.H + 1
    best_t = -1

    for t in range(state.W):
        if t == s_src:
            continue
        n_remove, cost = _best_subseq_fit(state, seq, t, s_src)
        if cost < best_cost:
            best_n_moves = n_remove
            best_cost = cost
            best_t = t

    if best_n_moves > 0 and best_t >= 0:
        for _ in range(best_n_moves):
            _la_relocate(state, state.top(best_t))


# ═══════════════════════════════════════════════════════════════════════ #
#  Section 6 — main greedy solve loop                                    #
# ═══════════════════════════════════════════════════════════════════════ #

def _solve(state: BRPState, mode: str) -> None:
    """
    Run the greedy solve loop in-place.

    mode : 'SM-1' | 'SM-2' | 'SmSEQ-1' | 'SmSEQ-2'
    """
    sm_level = 2 if mode in ("SM-2", "SmSEQ-2") else 1
    use_seq  = mode in ("SmSEQ-1", "SmSEQ-2")

    while True:
        state.auto_retrieve()
        if state.is_empty():
            break

        if use_seq:
            seq = _decreasing_sequence(state)
            if len(seq) >= 2:
                # make room in best target, then do one forced move
                _make_room_for_subseq(state)
                _forced_move(state)
                continue

        necessary = _necessary_relocates(state)
        did_vol = _try_safe1(state, necessary)
        if not did_vol and sm_level == 2:
            did_vol = _try_safe2(state, necessary)
        if not did_vol:
            _forced_move(state)


# ═══════════════════════════════════════════════════════════════════════ #
#  Section 7 — condensation (post-processing)                            #
# ═══════════════════════════════════════════════════════════════════════ #

def condense_jin(state: BRPState) -> None:
    """
    Classic (Jin et al.) condensation applied in-place to state.ops.

    Repeatedly merges pairs of relocations (s1→s2, s2→s3) of the same
    item C into a single relocation (s1→s3), provided:
      (a) C remains on top of s2 between the two,
      (b) stack s3 is not touched between the two.
    Also eliminates back-and-forth moves (s1→s2, s2→s1 → nothing).
    """
    ops = state.ops
    while _condense_jin_sub(ops):
        pass
    state.n_relocations = sum(1 for f, t in ops if f != t)


def _condense_jin_sub(ops: list) -> bool:
    changed = False
    i = 0
    while i < len(ops) - 1:
        if ops[i][0] == ops[i][1]:          # skip retrieval
            i += 1
            continue

        s1, s2 = ops[i]
        j = i + 1

        while j < len(ops):
            if ops[j][0] == ops[j][1]:      # retrieval
                j += 1
                continue

            if ops[j][0] != s2:
                j += 1
                continue

            # Potential condensation: check C stays on top and s3 untouched
            s3 = ops[j][1]
            above_c = 0
            ok = True

            for k in range(i + 1, j):
                f, t = ops[k]
                if above_c == 0 and f == s2:
                    ok = False
                    break
                if f != s2 and t == s2:
                    above_c += 1
                if f == s2:
                    above_c -= 1
                if f == s3 or t == s3:
                    ok = False
                    break

            if ok and above_c != 0:
                ok = False

            if ok:
                ops.pop(j)
                ops[i] = (s1, s3)
                s2 = s3
                changed = True
                if s1 == s3:                # back-and-forth: eliminate both
                    ops.pop(i)
                    # advance to next non-retrieval from i
                    while i < len(ops) and ops[i][0] == ops[i][1]:
                        i += 1
                    if i >= len(ops):
                        return True
                    s1, s2 = ops[i]
                    j = i + 1
                # do NOT advance j (re-check from same position)
            else:
                j += 1

        i += 1
    return changed


def condense_tricoire(state: BRPState, initial_heights: List[int]) -> None:
    """
    Improved (Tricoire 2018) condensation applied in-place to state.ops.

    Like Jin condensation but also handles the case where s3 receives/loses
    items between the two relocations, as long as capacity constraints are
    satisfied and s3 ends up in the same state.

    initial_heights : height of each stack before the first operation
                      (i.e. the starting heights of the heuristic run).
    """
    ops = state.ops
    # Build height-before-op array from scratch
    heights = _build_heights(ops, initial_heights, state.W)
    while _condense_tricoire_sub(ops, heights, state.W, state.H):
        pass
    state.n_relocations = sum(1 for f, t in ops if f != t)


def _build_heights(
    ops: list, initial_heights: List[int], W: int
) -> List[List[int]]:
    """
    heights[i] = stack heights BEFORE operation i.
    (C++ heightBeforeOp array in condenseTricoireSub)
    """
    heights: List[List[int]] = []
    curr = list(initial_heights)
    for from_s, to_s in ops:
        heights.append(list(curr))
        curr[from_s] -= 1
        if from_s != to_s:
            curr[to_s] += 1
    heights.append(list(curr))   # heights after last op
    return heights


def _condense_tricoire_sub(
    ops: list,
    heights: list,
    W: int,
    H: int,
) -> bool:
    changed = False
    i = 0
    while i < len(ops) - 1:
        if ops[i][0] == ops[i][1]:          # retrieval
            i += 1
            continue

        s1, s2 = ops[i]
        j = i + 1

        while j < len(ops):
            if ops[j][0] == ops[j][1]:      # retrieval
                j += 1
                continue

            if ops[j][0] != s2:
                j += 1
                continue

            s3 = ops[j][1]

            # s3 full at firstReloc time → skip
            if s1 != s3 and heights[i][s3] == H:
                j += 1
                continue

            # Check feasibility conditions
            above_c = 0
            above_s3 = 0
            ok = True

            for k in range(i + 1, j):
                f, t = ops[k]
                # condition 1: C is retrieved/moved while on top of s2
                if above_c == 0 and f == s2:
                    ok = False
                    break
                # condition 2: item placed on top of C
                if f != s2 and t == s2:
                    above_c += 1
                # condition 3: item removed from above C
                if f == s2:
                    above_c -= 1
                    if above_c < 0:
                        ok = False
                        break
                # condition 4/5: something goes TO s3
                if t == s3:
                    if f == s3:             # retrieval from s3
                        ok = False
                        break
                    above_s3 += 1
                    if above_s3 + heights[k][s3] >= H - 1:
                        ok = False
                        break
                # condition 6: something moves FROM s3
                if f == s3:
                    above_s3 -= 1
                    if above_s3 < 0:
                        ok = False
                        break

            if ok and (above_c != 0 or above_s3 != 0):
                ok = False

            if ok:
                # Update heights between i+1 and j-1
                for k in range(i + 1, j):
                    heights[k][s2] -= 1
                    heights[k][s3] += 1
                del heights[j]
                del ops[j]
                ops[i] = (s1, s3)
                s2 = s3
                changed = True

                if s1 == s3:                # back-and-forth: eliminate both
                    del heights[i]
                    del ops[i]
                    while i < len(ops) and ops[i][0] == ops[i][1]:
                        i += 1
                    if i >= len(ops):
                        return True
                    s1, s2 = ops[i]
                    j = i + 1
                # do NOT advance j
            else:
                j += 1

        i += 1
    return changed


# ═══════════════════════════════════════════════════════════════════════ #
#  Section 8 — fast-meta (all 4 heuristics, best wins)                   #
# ═══════════════════════════════════════════════════════════════════════ #

def fast_meta(
    partial_state: BRPState,
    condensation: str = "improved",
) -> BRPState:
    """
    Run all four heuristics from *partial_state* and return the best result.
    (C++ FastMetaPolicy::solve, which calls all four policies.)

    condensation : 'improved' (Tricoire) | 'classic' (Jin) | 'none'
    """
    best: Optional[BRPState] = None

    for mode in ("SM-1", "SM-2", "SmSEQ-1", "SmSEQ-2"):
        candidate = partial_state.copy()
        init_h = list(candidate.height)
        _solve(candidate, mode)
        if "SmSEQ" in mode and condensation != "none":
            if condensation == "improved":
                condense_tricoire(candidate, init_h)
            else:
                condense_jin(candidate)
        if best is None or candidate.n_relocations < best.n_relocations:
            best = candidate

    return best  # type: ignore[return-value]


# ═══════════════════════════════════════════════════════════════════════ #
#  Section 9 — Rake Search                                               #
# ═══════════════════════════════════════════════════════════════════════ #

def _gen_succ_moves_rake(state: BRPState) -> List[Tuple[int, int]]:
    """
    Generate successor moves for Rake Search BFS node.
    (C++ RakeSearch::genSuccMoves)

    Order: all safe-1 relocates → all safe-2 relocates → forced moves for
           the stacks of items next_item..min(next_item+4, n).
    """
    result: List[Tuple[int, int]] = []
    seen: set = set()

    necessary = _necessary_relocates(state)

    # all safe-1 relocates
    for src, item in necessary:
        for dst in range(state.W):
            if dst != src and state.height[dst] < state.H and state.low[dst] >= item:
                p = (src, dst)
                if p not in seen:
                    result.append(p)
                    seen.add(p)

    # all safe-2 relocates (first move of each 2-step)
    for p in _all_safe2(state, necessary):
        if p not in seen:
            result.append(p)
            seen.add(p)

    # forced moves: stacks containing next_item .. min(next_item+4, n)
    force_stacks: set = set()
    for i in range(state.next_item, min(state.next_item + 5, state.n + 1)):
        force_stacks.add(state.stack_for_item[i])

    for s in force_stacks:
        for t in range(state.W):
            if t != s and state.height[t] < state.H:
                p = (s, t)
                if p not in seen:
                    result.append(p)
                    seen.add(p)

    return result


def _update_queue(
    Q: list,
    start: int,
    state: BRPState,
) -> None:
    """
    Insert *state* into Q with dominance pruning among positions ≥ start.
    (C++ RakeSearch::updateQueue)
    """
    i = start
    while i < len(Q):
        if Q[i].dominates(state):
            return                             # dominated → discard
        if state.dominates(Q[i]):
            if i == len(Q) - 1:
                Q[i] = state
                return
            Q[i] = Q[-1]
            Q.pop()
        else:
            i += 1
    Q.append(state)


def rake_search(
    init_state: BRPState,
    width: int = 100,
    condensation: str = "improved",
) -> BRPState:
    """
    Rake Search metaheuristic.  (C++ RakeSearch::solve)

    1. BFS phase: expand states until |Q| ≥ width.
    2. Completion phase: apply fast_meta to each state in Q.

    Parameters
    ----------
    init_state  : starting BRPState (not modified)
    width       : BFS width threshold
    condensation: condensation mode for fast_meta completion
    """
    start = init_state.copy()
    start.auto_retrieve()
    if start.is_empty():
        return start

    Q: list = [start]

    # BFS phase
    while len(Q) < width:
        n_remaining = len(Q)
        for _ in range(n_remaining):
            cs = Q.pop(0)
            cs.auto_retrieve()
            if cs.is_empty():
                return cs
            for move in _gen_succ_moves_rake(cs):
                succ = cs.copy()
                succ.relocate(move[0], move[1])
                _update_queue(Q, n_remaining, succ)
        if not Q:
            break

    # Completion phase
    best: Optional[BRPState] = None
    for partial in Q:
        candidate = fast_meta(partial, condensation=condensation)
        if best is None or candidate.n_relocations < best.n_relocations:
            best = candidate

    return best  # type: ignore[return-value]


# ═══════════════════════════════════════════════════════════════════════ #
#  Section 10 — Pilot Method                                             #
# ═══════════════════════════════════════════════════════════════════════ #

def _gen_succ_moves_pilot(state: BRPState) -> List[Tuple[int, int]]:
    """
    Generate all valid moves for the Pilot Method node.
    (C++ PilotMethod::genSuccMoves)

    All (s, t) pairs with s ≠ last_relocated_to, height[s] > 0,
    height[t] < H.  At most one empty destination is included.
    """
    result: List[Tuple[int, int]] = []
    last = state.last_relocated_to

    for s in range(state.W):
        if s == last or state.height[s] == 0:
            continue
        to_empty_added = False
        for t in range(state.W):
            if t == s or state.height[t] == state.H:
                continue
            if state.height[t] > 0:
                result.append((s, t))
            elif not to_empty_added:
                result.append((s, t))
                to_empty_added = True

    return result


def pilot_method(
    init_state: BRPState,
    width: int = 10,
    hub_width: int = 2,
    condensation: str = "improved",
) -> BRPState:
    """
    Pilot Method metaheuristic.  (C++ PilotMethod::solve)

    For each candidate state in Q:
      - generate all valid single relocations
      - apply one SM-2 voluntary move
      - evaluate with rake_search(hub_width) as look-ahead (UB)
    Keep the *width* candidates with the best UB as new Q.

    Parameters
    ----------
    init_state  : starting BRPState (not modified)
    width       : max candidates to keep per iteration
    hub_width   : rake-search width for the look-ahead procedure
    condensation: condensation mode for the hub rake search
    """
    start = init_state.copy()
    start.auto_retrieve()
    if start.is_empty():
        return start

    # Warm-start bestKnown with the hub heuristic
    best_known = rake_search(start.copy(), hub_width, condensation)
    Q: list = [start]

    while Q:
        best_ub = best_known.n_relocations
        candidates: list = []

        for cs in Q:
            cs.auto_retrieve()
            if cs.is_empty():
                return best_known
            # Prune: if already at least as bad as best known
            if cs.n_relocations + cs.lb1 >= best_known.n_relocations:
                continue

            for move in _gen_succ_moves_pilot(cs):
                succ = cs.copy()
                succ.relocate(move[0], move[1])

                # apply one voluntary SM-2 move
                nec = _necessary_relocates(succ)
                if not _try_safe1(succ, nec):
                    _try_safe2(succ, nec)

                # evaluate with hub
                evaluated = rake_search(succ.copy(), hub_width, condensation)
                ub = evaluated.n_relocations
                if ub < best_known.n_relocations:
                    best_known = evaluated

                if ub < best_ub:
                    best_ub = ub
                    candidates = [succ]
                elif ub == best_ub:
                    candidates.append(succ)

        if not candidates:
            return best_known

        Q = candidates[:width]

    return best_known


# ═══════════════════════════════════════════════════════════════════════ #
#  Section 11 — public entry point                                       #
# ═══════════════════════════════════════════════════════════════════════ #

def solve_brp(
    init_state: BRPState,
    mode: str = "SmSEQ-2",
    width: int = 100,
    hub_width: int = 2,
    condensation: str = "improved",
) -> BRPState:
    """
    Solve a BRP instance from *init_state*.

    Parameters
    ----------
    mode : one of
        'SM-1'       – Safe 1-relocates only
        'SM-2'       – Safe 1- or 2-relocates
        'SmSEQ-1'    – Decreasing sequences + SM-1
        'SmSEQ-2'    – Decreasing sequences + SM-2  (default)
        'RakeSearch' – Rake Search BFS metaheuristic
        'PilotMethod'– Pilot Method look-ahead metaheuristic
    width : BFS width for RakeSearch / PilotMethod outer width
    hub_width : rake-search width used as hub in PilotMethod
    condensation : 'improved' (Tricoire) | 'classic' (Jin) | 'none'

    Returns
    -------
    Completed BRPState with ops log and n_relocations updated.
    """
    if mode in ("SM-1", "SM-2", "SmSEQ-1", "SmSEQ-2"):
        state = init_state.copy()
        init_h = list(state.height)
        _solve(state, mode)
        if "SmSEQ" in mode and condensation != "none":
            if condensation == "improved":
                condense_tricoire(state, init_h)
            else:
                condense_jin(state)
        return state

    if mode == "RakeSearch":
        return rake_search(init_state, width=width, condensation=condensation)

    if mode == "PilotMethod":
        return pilot_method(
            init_state,
            width=width,
            hub_width=hub_width,
            condensation=condensation,
        )

    raise ValueError(f"Unknown mode: {mode!r}")
