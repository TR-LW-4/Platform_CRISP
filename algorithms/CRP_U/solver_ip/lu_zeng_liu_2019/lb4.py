"""
LB4 — strongest lower bound for BRP (Lu, Zeng & Liu, 2019, Section III).

LB4 = LB1  +  2 × |P7-subsets|  +  |P5-subsets|  +  |P8-subsets|

Algorithms implemented
-----------------------
A5   — identify one virtual layer satisfying Property 5 (O(BS))
A5*  — variant of A5 that seeks blocks at the lowest possible positions
A7   — identify two overlapping virtual layers sharing one WP block
        satisfying Property 7 (O(BS))
A8   — identify one block subset satisfying Property 8, reduced to
        exactly S blocks via A8-s (O(B log S))
compute_lb4 — orchestrates A7 → A5/A5* → A8 to compute LB4

All functions operate on plain Python lists (snapshots of BRPState),
not on BRPState directly, so they are pure and side-effect-free.

Reference
---------
C. Lu, B. Zeng, S. Liu,
"A Study on the Block Relocation Problem: Lower Bound Derivations and
Strong Formulations",
IEEE Transactions on Automation Science and Engineering, 2019.
arXiv:1904.03347
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple


# ═══════════════════════════════════════════════════════════════════════ #
#  Section 1 — helpers                                                   #
# ═══════════════════════════════════════════════════════════════════════ #

def _lb1(stacks: List[List[int]], n: int) -> int:
    """
    LB1 = number of badly-placed items.
    Item i is badly placed iff there exists item j below it in the same
    stack with j > i (j should be retrieved after i but is below it).
    Equivalently: top(s) > low_except_top(s) is already encoded by
    checking each stack's cumulative minimum from bottom.
    """
    count = 0
    for stk in stacks:
        running_min = n + 1
        for item in stk:
            if item < running_min:
                running_min = item
            else:
                count += 1
    return count


def _low(stk: List[int], n: int) -> int:
    """Minimum item in stack (n+1 if empty)."""
    return min(stk) if stk else n + 1


def _low_except_top_k(stk: List[int], k: int, n: int) -> int:
    """
    Minimum item in stack *excluding* the top k items.
    Mirrors C++ lowestExceptTopK.
    """
    if k == 0:
        return min(stk) if stk else n + 1
    h = len(stk)
    if k >= h:
        return n + 1
    sub = stk[: h - k]
    return min(sub) if sub else n + 1


def _is_badly_placed(item: int, stk: List[int], n: int) -> bool:
    """True iff item is badly placed in its stack."""
    if not stk or stk[-1] != item:
        return False
    running_min = n + 1
    for x in stk:
        if x == item:
            break
        if x < running_min:
            running_min = x
    return item > running_min


def _stack_of(item: int, stacks: List[List[int]]) -> int:
    """Return 0-indexed stack containing item."""
    for s, stk in enumerate(stacks):
        if item in stk:
            return s
    return -1


def _items_above(item: int, stacks: List[List[int]], stack_idx: int) -> List[int]:
    """Items strictly above item in its stack (top-most first)."""
    stk = stacks[stack_idx]
    idx = stk.index(item)
    return list(reversed(stk[idx + 1 :]))


def _highest_priority_top(stacks: List[List[int]], n: int,
                           exclude: Optional[int] = None) -> int:
    """
    Highest priority (smallest index) among top items of all stacks,
    optionally excluding stack `exclude`.
    """
    result = n + 1
    for s, stk in enumerate(stacks):
        if s == exclude or not stk:
            continue
        result = min(result, stk[-1])
    return result


def _next_item(stacks: List[List[int]], n: int) -> int:
    """The item currently to be retrieved next (smallest item in the yard)."""
    result = n + 1
    for stk in stacks:
        for item in stk:
            result = min(result, item)
    return result


# ═══════════════════════════════════════════════════════════════════════ #
#  Section 2 — Property 5 check                                          #
# ═══════════════════════════════════════════════════════════════════════ #

def _check_p5(
    virtual_layer: List[int],         # one item per stack (index = stack)
    stacks: List[List[int]],
    n: int,
    W: int,
) -> bool:
    """
    Check whether a virtual layer satisfies Property 5 (Theorem 3).

    Conditions:
    (1) ∃ item below the virtual layer whose priority is higher than
        max priority in the virtual layer.
    (2) max BP-item priority in the virtual layer < min stack priority
        after removing items above the virtual layer.
    """
    if len(virtual_layer) != W:
        return False

    # Heights of each item in its stack (0 = bottom)
    heights_in_stack: List[int] = []
    for s, item in enumerate(virtual_layer):
        stk = stacks[s]
        if item not in stk:
            return False
        heights_in_stack.append(stk.index(item))

    max_vl_priority = max(virtual_layer)

    # Condition (1): exist item below the layer with higher priority
    cond1 = False
    for s, item in enumerate(virtual_layer):
        h = heights_in_stack[s]
        stk = stacks[s]
        for j in range(h):               # items below virtual_layer[s]
            if stk[j] < max_vl_priority:
                cond1 = True
                break
        if cond1:
            break
    if not cond1:
        return False

    # Condition (2): max BP priority in VL < min stack priority after
    #                removing items above VL in each stack
    stacks_trimmed = [stacks[s][: heights_in_stack[s] + 1] for s in range(W)]
    min_trimmed_priority = min(
        (_low(st, n) for st in stacks_trimmed if st),
        default=n + 1,
    )
    # BP items in virtual layer
    bp_in_vl = [
        item for s, item in enumerate(virtual_layer)
        if item > _low(stacks_trimmed[s], n)
    ]
    if not bp_in_vl:
        return False

    max_bp_vl = max(bp_in_vl)
    return max_bp_vl < min_trimmed_priority


# ═══════════════════════════════════════════════════════════════════════ #
#  Section 3 — Algorithm A5 / A5*                                        #
# ═══════════════════════════════════════════════════════════════════════ #

def _a5(
    stacks: List[List[int]],
    n: int,
    W: int,
    used: Optional[Set[Tuple[int, int]]] = None,
) -> Optional[List[int]]:
    """
    Algorithm A5: find one virtual layer satisfying P5.

    Starts from the top physical layer and tries to fix violating blocks
    by descending one tier at a time.

    Returns a list of length W (one item per stack, 0-indexed)
    or None if no virtual layer satisfying P5 exists.

    `used`: set of (stack, item) pairs already assigned to previous virtual
            layers, so we skip them.
    """
    if used is None:
        used = set()

    # Start with the top physical layer
    vl: List[Optional[int]] = []
    for s in range(W):
        stk = stacks[s]
        # find topmost item in stack s that is not used
        candidate = None
        for item in reversed(stk):
            if (s, item) not in used:
                candidate = item
                break
        vl.append(candidate)

    if any(v is None for v in vl):
        return None

    # Precompute highest priority of items strictly below each candidate
    # (for condition 1) and stack heights of each candidate
    def heights_of(vl_curr: List[Optional[int]]) -> List[int]:
        h = []
        for s, item in enumerate(vl_curr):
            if item is None:
                h.append(-1)
            else:
                h.append(stacks[s].index(item))
        return h

    max_iters = sum(len(st) for st in stacks) + 1
    for _ in range(max_iters):
        # Check P5
        vl_full: List[int] = [v for v in vl if v is not None]  # type: ignore
        if len(vl_full) == W and _check_p5(vl_full, stacks, n, W):  # type: ignore
            return vl_full

        # Try to fix the first violating position
        fixed = False
        hs = heights_of(vl)
        stacks_trimmed = [
            stacks[s][: hs[s] + 1] if hs[s] >= 0 else []
            for s in range(W)
        ]
        min_tp = min((_low(st, n) for st in stacks_trimmed if st), default=n + 1)
        max_vl = max((v for v in vl if v is not None), default=n + 1)

        for s in range(W):
            item = vl[s]
            if item is None:
                break
            # Check if item causes violation
            stk_t = stacks_trimmed[s]
            low_s = _low(stk_t, n)
            causes_viol = (item > low_s and item >= min_tp) or (item < max_vl and item == low_s)
            if causes_viol:
                # Replace with item directly below it in stack s
                h = stacks[s].index(item)
                # find next available item below
                found_below = None
                for j in range(h - 1, -1, -1):
                    if (s, stacks[s][j]) not in used:
                        found_below = stacks[s][j]
                        break
                if found_below is None:
                    return None
                vl[s] = found_below
                fixed = True
                break

        if not fixed:
            return None

    return None


def _a5_star(
    stacks: List[List[int]],
    n: int,
    W: int,
    used: Optional[Set[Tuple[int, int]]] = None,
) -> Optional[List[int]]:
    """
    Algorithm A5*: variant of A5 that seeks blocks at the LOWEST possible
    positions while ensuring P5 eligibility.

    Strategy: run A5, then try to push each item as deep as possible.
    """
    vl = _a5(stacks, n, W, used)
    if vl is None:
        return None

    if used is None:
        used = set()

    # For each position, try to replace with the deepest possible item
    for s in range(W):
        current = vl[s]
        stk = stacks[s]
        h_curr = stk.index(current)
        for j in range(0, h_curr):
            if (s, stk[j]) not in used:
                candidate = list(vl)
                candidate[s] = stk[j]
                if _check_p5(candidate, stacks, n, W):
                    vl = candidate

    return vl


# ═══════════════════════════════════════════════════════════════════════ #
#  Section 4 — Algorithm A7 (Property 7)                                 #
# ═══════════════════════════════════════════════════════════════════════ #

def _a7(
    stacks: List[List[int]],
    n: int,
    W: int,
    used: Set[Tuple[int, int]],
) -> Optional[Tuple[List[int], List[int], int]]:
    """
    Algorithm A7: find two virtual layers sharing exactly one WP block
    that together satisfy Property 7 (Theorem 4).

    For each candidate WP block `i` in the yard:
      1. Form vl1 containing i by running A5-logic (include i at its stack).
      2. Form vl2 containing i similarly.
      3. Check Property 7 conditions.

    Returns (vl1, vl2, shared_item) or None.
    """
    for s_shared in range(W):
        stk = stacks[s_shared]
        for i_idx, shared_item in enumerate(stk):
            # shared_item must be WP: it equals the stack minimum
            stack_min = min(stk)
            if shared_item != stack_min:
                continue
            if (s_shared, shared_item) in used:
                continue

            # Try building two virtual layers that both satisfy P5
            # and include shared_item at position s_shared
            vl1 = _a5_with_fixed(stacks, n, W, s_shared, shared_item, used)
            if vl1 is None:
                continue

            used_ext = used | {(s_shared, shared_item)}
            used_ext.update((s, v) for s, v in enumerate(vl1) if v != shared_item)
            vl2 = _a5_with_fixed(stacks, n, W, s_shared, shared_item, used_ext)
            if vl2 is None:
                continue

            # Property 7 condition: priority of shared WP block < min
            # priority of other S-1 stacks after removing both layers
            vl1_heights = [stacks[s].index(vl1[s]) for s in range(W)]
            vl2_heights = [stacks[s].index(vl2[s]) for s in range(W)]
            min_h = [min(vl1_heights[s], vl2_heights[s]) for s in range(W)]
            other_stacks_mins = []
            for s in range(W):
                if s == s_shared:
                    continue
                trimmed = stacks[s][: min_h[s] + 1]
                m = min(trimmed) if trimmed else n + 1
                other_stacks_mins.append(m)

            if not other_stacks_mins:
                continue
            min_others = min(other_stacks_mins)
            if shared_item < min_others:
                return (vl1, vl2, shared_item)

    return None


def _a5_with_fixed(
    stacks: List[List[int]],
    n: int,
    W: int,
    fixed_s: int,
    fixed_item: int,
    used: Set[Tuple[int, int]],
) -> Optional[List[int]]:
    """
    A5 variant that forces `fixed_item` at stack `fixed_s`.
    For other stacks, picks the topmost unused item, then checks P5.
    """
    vl: List[Optional[int]] = [None] * W
    vl[fixed_s] = fixed_item

    for s in range(W):
        if s == fixed_s:
            continue
        for item in reversed(stacks[s]):
            if (s, item) not in used and item != fixed_item:
                vl[s] = item
                break
        if vl[s] is None:
            return None

    vl_full: List[int] = vl  # type: ignore
    if _check_p5(vl_full, stacks, n, W):
        return vl_full
    return None


# ═══════════════════════════════════════════════════════════════════════ #
#  Section 5 — Algorithm A8 / A8-s (Property 8)                         #
# ═══════════════════════════════════════════════════════════════════════ #

def _a8(
    stacks: List[List[int]],
    n: int,
    W: int,
    target_item: int,
    used: Set[Tuple[int, int]],
) -> Optional[List[int]]:
    """
    Algorithm A8: find a block subset B8 satisfying Property 8 for
    `target_item`, reduced to exactly S blocks (A8-s).

    Returns a list of S items (one per stack, the barrier-chain),
    or None if property is not satisfied.

    Steps:
    1. B8_1 = items above target_item with lower priority
    2. B8_2 = highest-priority item in each other stack
    3. Verification: relocate items of B8_1 once (without height limit)
       to be WP.  Sort stacks by low ascending.
       If some item cannot be made WP → P8 satisfied.
    4. Reduction (A8-s): extract S-item barrier chain.
    """
    s_target = _stack_of(target_item, stacks)
    if s_target < 0:
        return None

    stk_t = stacks[s_target]
    idx_t = stk_t.index(target_item)

    # B8_1: items above target with priority > target (lower in retrieval)
    b8_1: List[int] = [
        stk_t[j] for j in range(idx_t + 1, len(stk_t))
        if stk_t[j] > target_item
    ]
    if not b8_1:
        return None

    # B8_2: highest priority (smallest index) from each other stack
    b8_2: List[Tuple[int, int]] = []  # (item, stack)
    for s in range(W):
        if s == s_target:
            continue
        cands = [item for item in stacks[s] if item > target_item]
        if not cands:
            continue
        b8_2.append((min(cands), s))

    # Build working copy: remove items NOT in B8_1 ∪ B8_2 ∪ {target_item}
    b8_all_items = set(b8_1) | {p for p, _ in b8_2} | {target_item}
    work_stacks = [
        [x for x in stk if x in b8_all_items]
        for stk in stacks
    ]

    # Sort other stacks ascending by their minimum (= low)
    other_stacks_sorted: List[int] = sorted(
        [s for s in range(W) if s != s_target],
        key=lambda s: (min(work_stacks[s]) if work_stacks[s] else n + 1),
    )

    # Verification: relocate each item in b8_1 once to become WP
    # (no height limit); use greedy: target the stack with highest min
    can_place = list(other_stacks_sorted)  # already sorted asc by low
    failed_item: Optional[int] = None

    for item in b8_1:
        placed = False
        for s in reversed(can_place):   # prefer higher-index = higher min
            stk_s = work_stacks[s]
            low_s = min(stk_s) if stk_s else n + 1
            if low_s >= item:           # item becomes WP on s
                work_stacks[s].append(item)
                placed = True
                break
        if not placed:
            failed_item = item
            break

    if failed_item is None:
        return None   # all items could be made WP → P8 not satisfied

    # A8-s: extract the barrier chain of exactly S items
    # Start from failed_item, walk the barrier chain
    chain: List[int] = [failed_item]

    prev_item = failed_item
    for s in other_stacks_sorted:
        stk_s = work_stacks[s]
        barrier = None
        for x in stk_s:
            if x < prev_item:          # x has higher priority → barrier
                if barrier is None or x > barrier:
                    barrier = x
        if barrier is None:
            # no barrier in this stack: take highest-priority WP item
            for x in stk_s:
                if x > target_item:
                    if barrier is None or x < barrier:
                        barrier = x
        if barrier is not None:
            chain.append(barrier)
            prev_item = barrier
        if len(chain) == W:
            break

    if len(chain) < W:
        return None

    return chain


# ═══════════════════════════════════════════════════════════════════════ #
#  Section 6 — compute_lb4                                               #
# ═══════════════════════════════════════════════════════════════════════ #

def compute_lb4(stacks: List[List[int]], n: int, W: int, H: int) -> int:
    """
    Compute LB4 for the given (stacks, n, W, H) configuration.

    LB4 = LB1
          + 2 × (number of P7 subsets found)
          + 1 × (number of P5 subsets found)
          + 1 × (number of P8 subsets found)

    The algorithm applies A7 → A5/A5* → A8, each time working on
    unpicked blocks, to maximise the total lower bound.
    """
    lb1 = _lb1(stacks, n)

    # working copy of stacks (we do NOT modify stacks; used tracks consumed)
    used: Set[Tuple[int, int]] = set()

    p7_count = 0
    p5_count = 0
    p8_count = 0

    # ── Phase 1: A7 (P7, each contributes +2) ────────────────────── #
    while True:
        r7 = _a7(stacks, n, W, used)
        if r7 is None:
            break
        vl1, vl2, shared = r7
        p7_count += 1
        # mark all items in both virtual layers as used (except shared,
        # which appears in both; mark it only once)
        for s, item in enumerate(vl1):
            used.add((s, item))
        for s, item in enumerate(vl2):
            used.add((s, item))

    # ── Phase 2: A5 / A5* (P5, each contributes +1) ──────────────── #
    while True:
        vl_a5 = _a5(stacks, n, W, used)
        vl_a5s = _a5_star(stacks, n, W, used)

        # Take the one that exists (both same contribution; prefer A5*)
        vl = None
        if vl_a5 is not None and vl_a5s is not None:
            vl = vl_a5s
        elif vl_a5 is not None:
            vl = vl_a5
        elif vl_a5s is not None:
            vl = vl_a5s

        if vl is None:
            break

        p5_count += 1
        for s, item in enumerate(vl):
            used.add((s, item))

    # ── Phase 3: A8 (P8, each contributes +1) ────────────────────── #
    # iterate over all remaining items ordered by increasing priority
    remaining_items = sorted(
        {item for s, stk in enumerate(stacks) for item in stk
         if (s, item) not in used},
    )
    found_p8: Set[int] = set()
    for target in remaining_items:
        if target in found_p8:
            continue
        chain = _a8(stacks, n, W, target, used)
        if chain is None:
            continue
        # check none of the chain items already consumed
        chain_set = set(chain)
        if chain_set & {item for _, item in used}:
            continue
        p8_count += 1
        s_target = _stack_of(target, stacks)
        for item in chain:
            s = _stack_of(item, stacks)
            used.add((s, item))
        found_p8.update(chain)

    lb4 = lb1 + 2 * p7_count + p5_count + p8_count
    return lb4


def compute_lb4_from_state(state) -> int:
    """
    Convenience wrapper: compute LB4 from a BRPState object.
    """
    return compute_lb4(state.stacks, state.n, state.W, state.H)


def compute_all_lower_bounds(stacks: List[List[int]], n: int, W: int, H: int) -> Dict[str, int]:
    """
    Compute all five lower bounds (LB1..LB4) and return as a dict.
    Useful for benchmarking and reporting.
    """
    lb1 = _lb1(stacks, n)
    lb2 = _lb2(stacks, n, W)
    lb3 = _lb3(stacks, n, W, H)
    lb4 = compute_lb4(stacks, n, W, H)
    return {"LB1": lb1, "LB2": lb2, "LB3": lb3, "LB4": lb4}


# ── LB2 (Forster & Bortfeldt 2012) ─────────────────────────────────── #

def _lb2(stacks: List[List[int]], n: int, W: int) -> int:
    """
    LB2 = LB1 + 1  iff all stacks non-empty AND
          min(top items of all stacks) > max(low of all stacks).
    """
    lb1 = _lb1(stacks, n)
    if any(not stk for stk in stacks):
        return lb1

    tops = [stk[-1] for stk in stacks]
    lows = [min(stk) for stk in stacks]

    # min top among BP tops
    min_bp_top = n + 1
    max_low = max(lows)
    for s, stk in enumerate(stacks):
        t = stk[-1]
        if t > lows[s] and t < min_bp_top:
            min_bp_top = t

    if min_bp_top > max_low:
        return lb1 + 1
    return lb1


# ── LB3 (Tricoire et al. 2018) ─────────────────────────────────────── #

def _lb3(stacks: List[List[int]], n: int, W: int, H: int) -> int:
    """
    LB3 = LB1 + k  where k is the maximum k such that
    removing the top k layers reveals that the top kth layer has
    no valid safe placement.  (Same as Tricoire 2018 LB3.)
    """
    lb1 = _lb1(stacks, n)

    # shortest stack height
    shortest = min(len(stk) for stk in stacks)
    k = 0
    while k < shortest:
        # max low of stacks after removing top k items
        max_low_trimmed = 0
        for stk in stacks:
            if k == 0:
                m = min(stk)
            else:
                sub = stk[: len(stk) - k]
                m = min(sub) if sub else n + 1
            max_low_trimmed = max(max_low_trimmed, m)

        # min top-of-layer-k item (the (k+1)-th from top) among BP items
        min_bp_top_k = n + 1
        for stk in stacks:
            top_k = stk[len(stk) - 1 - k]
            if top_k == 1:              # reached next_item sentinel
                min_bp_top_k = top_k
                break
            stk_low = min(stk)
            if top_k > stk_low and top_k < min_bp_top_k:
                min_bp_top_k = top_k

        if min_bp_top_k > max_low_trimmed:
            k += 1
        else:
            break

    return lb1 + k
