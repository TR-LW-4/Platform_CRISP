from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Sequence, Set, Tuple

Stacks = Dict[int, List[int]]  # stack_idx -> priorities bottom..top
Move = Tuple[int, int]         # (src, dst)


def clone_stacks(stacks: Stacks) -> Stacks:
    return {k: list(v) for k, v in stacks.items()}


def stack_height(stacks: Stacks, s: int) -> int:
    return len(stacks.get(s, []))


def top_priority(stacks: Stacks, s: int) -> Optional[int]:
    arr = stacks.get(s, [])
    if not arr:
        return None
    return int(arr[-1])


def is_stack_sorted(arr: Sequence[int]) -> bool:
    """
    Type-A rule:
    smaller index => higher priority; top container priority must be >= below.
    With bottom..top representation this means non-increasing values upward:
      arr[h] <= arr[h-1] for all h>=1.
    """
    for h in range(1, len(arr)):
        if arr[h] > arr[h - 1]:
            return False
    return True


def classify_rw(stacks: Stacks) -> Tuple[Set[int], Set[int], Set[int]]:
    """
    Returns (R, W, E):
      R - correctly arranged non-empty stacks
      W - wrongly arranged non-empty stacks
      E - empty stacks
    """
    R: Set[int] = set()
    W: Set[int] = set()
    E: Set[int] = set()
    for s, arr in stacks.items():
        if not arr:
            E.add(s)
        elif is_stack_sorted(arr):
            R.add(s)
        else:
            W.add(s)
    return R, W, E


def total_bad_overlaps(stacks: Stacks) -> int:
    bad = 0
    for arr in stacks.values():
        for h in range(1, len(arr)):
            x = arr[h]
            is_bad = False
            for j in range(h):
                if arr[j] < x:
                    is_bad = True
                    break
            if is_bad:
                bad += 1
    return bad


def is_yard_sorted(stacks: Stacks) -> bool:
    return total_bad_overlaps(stacks) == 0


def move_top(stacks: Stacks, src: int, dst: int, max_tiers: int) -> bool:
    if src == dst:
        return False
    if src not in stacks or dst not in stacks:
        return False
    if not stacks[src] or len(stacks[dst]) >= max_tiers:
        return False
    x = stacks[src].pop()
    stacks[dst].append(x)
    return True


def _select_source_from_w(
    stacks: Stacks,
    W: Set[int],
    want: int,
    max_tiers: int,
    *,
    fallback_delta: int = -1,
) -> Optional[int]:
    cands: List[Tuple[int, int]] = []  # (height, stack)
    for s in W:
        if stack_height(stacks, s) <= 0:
            continue
        p = top_priority(stacks, s)
        if p is None:
            continue
        if p == want:
            cands.append((stack_height(stacks, s), s))
    if cands:
        cands.sort()
        return cands[0][1]

    if fallback_delta != 0:
        want2 = want + fallback_delta
        cands2: List[Tuple[int, int]] = []
        for s in W:
            if stack_height(stacks, s) <= 0:
                continue
            p = top_priority(stacks, s)
            if p == want2:
                cands2.append((stack_height(stacks, s), s))
        if cands2:
            cands2.sort()
            return cands2[0][1]
    return None


def _fill_dest_from_w(
    stacks: Stacks,
    dest: int,
    W: Set[int],
    max_tiers: int,
    moves: List[Move],
) -> bool:
    moved = False
    while stack_height(stacks, dest) < max_tiers:
        p = top_priority(stacks, dest)
        if p is None:
            break
        src = _select_source_from_w(stacks, W, p, max_tiers, fallback_delta=-1)
        if src is None:
            break
        ok = move_top(stacks, src, dest, max_tiers)
        if not ok:
            break
        moves.append((src, dest))
        moved = True
    return moved


def _best_compatible_r_stack(stacks: Stacks, R: Set[int], src_stack: int, p: int, max_tiers: int) -> Optional[int]:
    # If there is one stack in R where i_pq - i_rc is minimum and i_pq >= i_rc.
    cands: List[Tuple[int, int, int]] = []  # (gap, height, stack)
    for s in R:
        if s == src_stack:
            continue
        if stack_height(stacks, s) >= max_tiers:
            continue
        tp = top_priority(stacks, s)
        if tp is None:
            continue
        if tp >= p:
            cands.append((tp - p, stack_height(stacks, s), s))
    if not cands:
        return None
    cands.sort()
    return cands[0][2]


def _fallback_greedy_move(stacks: Stacks, max_tiers: int, moves: List[Move]) -> bool:
    cur_bad = total_bad_overlaps(stacks)
    best: Optional[Tuple[int, int, int]] = None  # (new_bad, src, dst)
    keys = sorted(stacks.keys())
    for src in keys:
        if not stacks[src]:
            continue
        for dst in keys:
            if src == dst or stack_height(stacks, dst) >= max_tiers:
                continue
            trial = clone_stacks(stacks)
            ok = move_top(trial, src, dst, max_tiers)
            if not ok:
                continue
            nb = total_bad_overlaps(trial)
            if best is None or nb < best[0]:
                best = (nb, src, dst)
    if best is None or best[0] > cur_bad:
        return False
    _, src, dst = best
    ok = move_top(stacks, src, dst, max_tiers)
    if not ok:
        return False
    moves.append((src, dst))
    return True


def heuristic_a_type_a(
    stacks_init: Stacks,
    max_tiers: int,
    b: float,
    max_moves: int,
    rng: random.Random,
) -> Tuple[List[Move], int, bool]:
    stacks = clone_stacks(stacks_init)
    moves: List[Move] = []
    checked_low_r: Set[int] = set()
    high_threshold = int(math.ceil(max(0.0, min(1.0, b)) * max_tiers))

    for _ in range(max_moves):
        if is_yard_sorted(stacks):
            return moves, 0, True

        R, W, E = classify_rw(stacks)
        moved = False

        # Step 2: complete high R stacks.
        high_r = [s for s in R if high_threshold <= stack_height(stacks, s) < max_tiers]
        if high_r and W:
            high_r.sort(key=lambda s: (-stack_height(stacks, s), s))
            dest = high_r[0]
            moved = _fill_dest_from_w(stacks, dest, W, max_tiers, moves)
            if moved:
                checked_low_r.discard(dest)
                continue

        # Step 3: complete low R stacks.
        low_r = [s for s in R if stack_height(stacks, s) < high_threshold and s not in checked_low_r]
        for dest in low_r:
            snap = clone_stacks(stacks)
            m_before = len(moves)
            moved_local = _fill_dest_from_w(stacks, dest, W, max_tiers, moves)
            if moved_local and stack_height(stacks, dest) >= high_threshold:
                moved = True
                checked_low_r.discard(dest)
                break
            # resume as it was and mark checked (paper step 3c).
            stacks = snap
            del moves[m_before:]
            checked_low_r.add(dest)

        if moved:
            continue

        # Step 4: deconstruct low checked R stack.
        checked_exist = [s for s in checked_low_r if s in R and stack_height(stacks, s) > 0]
        if checked_exist:
            checked_exist.sort(key=lambda s: (stack_height(stacks, s), s))
            src = checked_exist[0]
            p = top_priority(stacks, src)
            if p is not None:
                dst = _best_compatible_r_stack(stacks, R, src, p, max_tiers)
                if dst is None:
                    dst = _select_source_from_w(
                        stacks=stacks,
                        W=W,
                        want=p,
                        max_tiers=max_tiers,
                        fallback_delta=-1,
                    )
                if dst is None:
                    dst = _select_source_from_w(
                        stacks=stacks,
                        W=W,
                        want=p,
                        max_tiers=max_tiers,
                        fallback_delta=+1,
                    )
                if dst is not None and src != dst:
                    ok = move_top(stacks, src, dst, max_tiers)
                    if ok:
                        moves.append((src, dst))
                        moved = True
                        if stack_height(stacks, src) == 0 or stack_height(stacks, src) >= high_threshold:
                            checked_low_r.discard(src)
        if moved:
            continue

        # Step 5: move W to empty stack.
        if E and W:
            dest = sorted(E)[0]
            # largest top index (lowest priority) first.
            cands = []
            for s in W:
                tp = top_priority(stacks, s)
                if tp is not None:
                    cands.append((-tp, stack_height(stacks, s), s))
            if cands:
                cands.sort()
                src = cands[0][2]
                ok = move_top(stacks, src, dest, max_tiers)
                if ok:
                    moves.append((src, dest))
                    moved = True
                    while stack_height(stacks, dest) < max_tiers:
                        p = top_priority(stacks, dest)
                        if p is None:
                            break
                        src2 = _select_source_from_w(
                            stacks=stacks,
                            W=W,
                            want=p,
                            max_tiers=max_tiers,
                            fallback_delta=-1,
                        )
                        if src2 is None:
                            break
                        ok2 = move_top(stacks, src2, dest, max_tiers)
                        if not ok2:
                            break
                        moves.append((src2, dest))
        if moved:
            continue

        # Fallback: avoid deadlock.
        moved = _fallback_greedy_move(stacks, max_tiers, moves)
        if not moved:
            break

    rem = total_bad_overlaps(stacks)
    return moves, rem, rem == 0
