from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

Stacks = Dict[int, List[int]]  # stack_idx -> priorities bottom..top
Move = Tuple[int, int]
ContainerRef = Tuple[int, int]  # (stack, tier_idx from bottom, 0-based)


def clone_stacks(stacks: Stacks) -> Stacks:
    return {k: list(v) for k, v in stacks.items()}


def is_stack_well_sorted(arr: Sequence[int]) -> bool:
    # Smaller index = higher priority, therefore priority must not increase upward.
    for i in range(1, len(arr)):
        if arr[i] > arr[i - 1]:
            return False
    return True


def total_bad_overlaps(stacks: Stacks) -> int:
    bad = 0
    for arr in stacks.values():
        for i in range(1, len(arr)):
            v = arr[i]
            if any(arr[j] < v for j in range(i)):
                bad += 1
    return bad


def non_located_refs(stacks: Stacks) -> List[ContainerRef]:
    refs: List[ContainerRef] = []
    for s, arr in stacks.items():
        for t in range(len(arr)):
            v = arr[t]
            if any(arr[j] < v for j in range(t)):
                refs.append((s, t))
    return refs


def non_located_with_priority(stacks: Stacks, p: int) -> List[ContainerRef]:
    refs: List[ContainerRef] = []
    for s, arr in stacks.items():
        for t in range(len(arr)):
            if arr[t] != p:
                continue
            if any(arr[j] < p for j in range(t)):
                refs.append((s, t))
    return refs


def stack_last_blocking_tier_for_priority(arr: Sequence[int], p: int) -> int:
    """
    Highest tier index whose container has *higher* priority than p (index < p).
    Returns -1 if none exists.
    """
    last = -1
    for t, v in enumerate(arr):
        if v < p:
            last = t
    return last


def f_to_make_well_located(stacks: Stacks, s: int, p: int) -> int:
    """
    Number of top containers that must be removed from stack s so p can be placed
    on s and be well-located.
    """
    arr = stacks[s]
    if not arr:
        return 0
    last = stack_last_blocking_tier_for_priority(arr, p)
    if last < 0:
        return 0
    return len(arr) - (last + 1)


def g_above_target(tier_idx: int, stack_len: int) -> int:
    return stack_len - tier_idx - 1


def choose_target_container(
    stacks: Stacks,
    p: int,
    k1: int,
    rng: random.Random,
) -> Optional[ContainerRef]:
    cand = non_located_with_priority(stacks, p)
    if not cand:
        return None
    scored = sorted(
        cand,
        key=lambda st: (g_above_target(st[1], len(stacks[st[0]])), st[0], st[1]),
    )
    topk = scored[: max(1, min(k1, len(scored)))]
    return rng.choice(topk)


def choose_destination_stack(
    stacks: Stacks,
    target: ContainerRef,
    p: int,
    k2: int,
    max_tiers: int,
    rng: random.Random,
) -> int:
    src, t = target
    g = g_above_target(t, len(stacks[src]))
    scored: List[Tuple[int, int]] = []
    for s in sorted(stacks.keys()):
        if len(stacks[s]) >= max_tiers:
            continue
        f = f_to_make_well_located(stacks, s, p)
        w = f + 1 if s == src else f + g + 1
        scored.append((w, s))
    scored.sort()
    topk = scored[: max(1, min(k2, len(scored)))]
    return rng.choice([s for _, s in topk])


def _stack_nonlocated_pmin(stacks: Stacks, s: int) -> Optional[int]:
    vals = []
    arr = stacks[s]
    for t in range(len(arr)):
        v = arr[t]
        if any(arr[j] < v for j in range(t)):
            vals.append(v)
    if not vals:
        return None
    return min(vals)


def choose_temp_stack(
    stacks: Stacks,
    forbidden: Tuple[int, ...],
    k3: int,
    max_tiers: int,
    rng: random.Random,
) -> Optional[int]:
    scored: List[Tuple[int, int, int]] = []  # (tierclass, pmin, s)
    for s in sorted(stacks.keys()):
        if s in forbidden or len(stacks[s]) >= max_tiers:
            continue
        if not stacks[s]:
            scored.append((0, 10**9, s))  # empty stack: highest preference
            continue
        pmin = _stack_nonlocated_pmin(stacks, s)
        if pmin is None:
            scored.append((1, 10**9, s))  # no non-located in stack: second preference
        else:
            scored.append((2, pmin, s))
    if not scored:
        return None
    scored.sort(key=lambda x: (x[0], x[1], x[2]))
    topk = scored[: max(1, min(k3, len(scored)))]
    return rng.choice([s for _, _, s in topk])


def move_top(stacks: Stacks, src: int, dst: int, max_tiers: int) -> bool:
    if src == dst:
        return False
    if not stacks[src] or len(stacks[dst]) >= max_tiers:
        return False
    stacks[dst].append(stacks[src].pop())
    return True


def relocate_target(
    stacks: Stacks,
    target: ContainerRef,
    dst: int,
    max_tiers: int,
    k3: int,
    rng: random.Random,
    moves: List[Move],
) -> bool:
    src, _ = target
    p = stacks[src][target[1]]

    # Re-locate target within same stack.
    if src == dst:
        tmp = choose_temp_stack(stacks, forbidden=(src,), k3=k3, max_tiers=max_tiers, rng=rng)
        if tmp is None:
            return False
        while stacks[src] and stacks[src][-1] != p:
            tm = choose_temp_stack(stacks, forbidden=(src, dst), k3=k3, max_tiers=max_tiers, rng=rng)
            if tm is None:
                return False
            if not move_top(stacks, src, tm, max_tiers):
                return False
            moves.append((src, tm))
        if not move_top(stacks, src, tmp, max_tiers):
            return False
        moves.append((src, tmp))
        while f_to_make_well_located(stacks, src, p) > 0:
            tm = choose_temp_stack(stacks, forbidden=(src, tmp), k3=k3, max_tiers=max_tiers, rng=rng)
            if tm is None:
                return False
            if not move_top(stacks, src, tm, max_tiers):
                return False
            moves.append((src, tm))
        if not move_top(stacks, tmp, src, max_tiers):
            return False
        moves.append((tmp, src))
        return True

    # Normal case src != dst.
    while True:
        src_top_is_target = bool(stacks[src]) and (stacks[src][-1] == p)
        need_remove_from_dst = f_to_make_well_located(stacks, dst, p)
        if src_top_is_target and need_remove_from_dst <= 0:
            break

        o_prio = None
        m_prio = None
        if not src_top_is_target and stacks[src]:
            o_prio = stacks[src][-1]
        if need_remove_from_dst > 0 and stacks[dst]:
            m_prio = stacks[dst][-1]
        if o_prio is None and m_prio is None:
            return False

        # Paper rule (Section 4.3): choose between top o and m.
        choose_src = False
        if o_prio is not None and m_prio is not None:
            choose_src = o_prio < m_prio
        elif o_prio is not None:
            choose_src = True

        if choose_src:
            tm = choose_temp_stack(stacks, forbidden=(src, dst), k3=k3, max_tiers=max_tiers, rng=rng)
            if tm is None or not move_top(stacks, src, tm, max_tiers):
                return False
            moves.append((src, tm))
        else:
            tm = choose_temp_stack(stacks, forbidden=(src, dst), k3=k3, max_tiers=max_tiers, rng=rng)
            if tm is None or not move_top(stacks, dst, tm, max_tiers):
                return False
            moves.append((dst, tm))

    if not move_top(stacks, src, dst, max_tiers):
        return False
    moves.append((src, dst))
    return True


def stack_filling(
    stacks: Stacks,
    touched_destinations: Sequence[int],
    max_tiers: int,
    k3: int,
    rng: random.Random,
    moves: List[Move],
) -> None:
    for dst in touched_destinations:
        if dst not in stacks or len(stacks[dst]) >= max_tiers or not stacks[dst]:
            continue
        top = stacks[dst][-1]
        while len(stacks[dst]) < max_tiers:
            candidates = []
            for s, arr in stacks.items():
                if s == dst or not arr:
                    continue
                # only top movable containers.
                v = arr[-1]
                if v >= top and any(arr[j] < v for j in range(len(arr) - 1)):
                    candidates.append((v - top, len(arr), s))
            if not candidates:
                break
            candidates.sort(key=lambda x: (x[0], x[1], x[2]))
            src = candidates[0][2]
            if not move_top(stacks, src, dst, max_tiers):
                break
            moves.append((src, dst))
            top = stacks[dst][-1]


@dataclass
class LPFHResult:
    moves: List[Move]
    remaining_bad: int
    solved: bool


def run_lpfh_once(
    stacks_init: Stacks,
    max_tiers: int,
    k1: int,
    k2: int,
    k3: int,
    use_stack_filling: bool,
    rng: random.Random,
    move_budget: int,
) -> LPFHResult:
    stacks = clone_stacks(stacks_init)
    moves: List[Move] = []

    priorities = sorted({v for arr in stacks.values() for v in arr}, reverse=True)  # lowest priority first (largest index)
    for p in priorities:
        while True:
            target = choose_target_container(stacks, p, k1=k1, rng=rng)
            if target is None:
                break
            dst = choose_destination_stack(stacks, target, p=p, k2=k2, max_tiers=max_tiers, rng=rng)
            ok = relocate_target(
                stacks=stacks,
                target=target,
                dst=dst,
                max_tiers=max_tiers,
                k3=k3,
                rng=rng,
                moves=moves,
            )
            if not ok:
                break
            if use_stack_filling:
                stack_filling(
                    stacks=stacks,
                    touched_destinations=[dst],
                    max_tiers=max_tiers,
                    k3=k3,
                    rng=rng,
                    moves=moves,
                )
            if len(moves) >= move_budget:
                rem = total_bad_overlaps(stacks)
                return LPFHResult(moves=moves[:move_budget], remaining_bad=rem, solved=(rem == 0))

    rem = total_bad_overlaps(stacks)
    return LPFHResult(moves=moves, remaining_bad=rem, solved=(rem == 0))


def run_lpfh_multistart(
    stacks_init: Stacks,
    max_tiers: int,
    k1: int,
    k2: int,
    k3: int,
    max_iterations: int,
    no_improve_limit: int,
    use_stack_filling: bool,
    rng_seed: int,
    move_budget: int,
) -> LPFHResult:
    best: Optional[LPFHResult] = None
    no_improve = 0
    rng = random.Random(rng_seed)

    for it in range(max(1, max_iterations)):
        rr = random.Random(rng.randint(0, 2**31 - 1))
        res = run_lpfh_once(
            stacks_init=stacks_init,
            max_tiers=max_tiers,
            k1=k1,
            k2=k2,
            k3=k3,
            use_stack_filling=use_stack_filling,
            rng=rr,
            move_budget=move_budget,
        )
        if best is None:
            best = res
            no_improve = 0
        else:
            if (res.remaining_bad, len(res.moves)) < (best.remaining_bad, len(best.moves)):
                best = res
                no_improve = 0
            else:
                no_improve += 1
        if no_improve >= max(1, no_improve_limit):
            break
        if best is not None and best.solved and len(best.moves) == 0:
            break
    return best if best is not None else LPFHResult(moves=[], remaining_bad=0, solved=True)
