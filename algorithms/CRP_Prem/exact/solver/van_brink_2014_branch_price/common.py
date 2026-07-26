"""
Common helpers for van Brink & van der Zwaan (2014) branch-and-price scaffold.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

Stacks = Dict[int, List[int]]  # stack_idx -> priorities bottom..top
Move = Tuple[int, int]         # (src_idx, dst_idx)


@dataclass(frozen=True)
class StackColumn:
    """
    One column for a specific stack in the master model.

    - adds/rems indexed by (priority, time)
    - cost = number of add operations (= number of moves into this stack)
    """
    stack_idx: int
    adds: Dict[Tuple[int, int], int]
    rems: Dict[Tuple[int, int], int]
    cost: int
    max_time_used: int


def clone_stacks(stacks: Stacks) -> Stacks:
    return {k: list(v) for k, v in stacks.items()}


def total_bad_overlaps(stacks: Stacks) -> int:
    total = 0
    for prios in stacks.values():
        for i in range(len(prios) - 1):
            if prios[i] < prios[i + 1]:
                total += 1
    return total


def is_yard_sorted(stacks: Stacks) -> bool:
    return total_bad_overlaps(stacks) == 0


def stack_wrongly_placed_count(prios: Sequence[int]) -> int:
    """
    Number of wrongly placed containers in a stack:
    a container is wrongly placed if there exists a lower element below with
    higher priority (smaller number), directly or transitively.
    """
    count = 0
    for i in range(len(prios)):
        p = prios[i]
        bad = False
        for j in range(i):
            if prios[j] < p:
                bad = True
                break
        if bad:
            count += 1
    return count


def lower_bound_wrongly_plus_free(stacks: Stacks) -> int:
    wrongly = [stack_wrongly_placed_count(v) for v in stacks.values()]
    return int(sum(wrongly) + min(wrongly or [0]))


def apply_move(stacks: Stacks, src: int, dst: int, max_tiers: int) -> bool:
    if src == dst:
        return False
    src_s = stacks.get(src)
    dst_s = stacks.get(dst)
    if src_s is None or dst_s is None or not src_s or len(dst_s) >= max_tiers:
        return False
    p = src_s.pop()
    dst_s.append(p)
    return True


def legal_moves(stacks: Stacks, max_tiers: int) -> List[Move]:
    moves: List[Move] = []
    keys = list(stacks.keys())
    for src in keys:
        if not stacks[src]:
            continue
        for dst in keys:
            if src == dst:
                continue
            if len(stacks[dst]) < max_tiers:
                moves.append((src, dst))
    return moves


def greedy_candidate_moves(
    stacks_init: Stacks,
    max_tiers: int,
    max_steps: int,
    rng: random.Random,
) -> List[Move]:
    stacks = clone_stacks(stacks_init)
    seq: List[Move] = []
    for _ in range(max_steps):
        if is_yard_sorted(stacks):
            break
        cands = legal_moves(stacks, max_tiers)
        if not cands:
            break

        scored: List[Tuple[int, float, Move]] = []
        for mv in cands:
            src, dst = mv
            trial = clone_stacks(stacks)
            apply_move(trial, src, dst, max_tiers)
            score = total_bad_overlaps(trial)
            scored.append((score, rng.random(), mv))
        scored.sort(key=lambda x: (x[0], x[1]))

        k = max(1, min(6, len(scored)))
        pick = scored[rng.randrange(k)][2]
        apply_move(stacks, pick[0], pick[1], max_tiers)
        seq.append(pick)
    return seq


def sequence_to_columns(
    stacks_init: Stacks,
    seq: Sequence[Move],
    max_tiers: int,
) -> Tuple[Dict[int, StackColumn], bool]:
    stacks = clone_stacks(stacks_init)
    adds: Dict[int, Dict[Tuple[int, int], int]] = {s: {} for s in stacks}
    rems: Dict[int, Dict[Tuple[int, int], int]] = {s: {} for s in stacks}
    add_count: Dict[int, int] = {s: 0 for s in stacks}
    max_time: Dict[int, int] = {s: 0 for s in stacks}

    for t, (src, dst) in enumerate(seq, start=1):
        src_s = stacks.get(src)
        dst_s = stacks.get(dst)
        if src_s is None or dst_s is None or not src_s or len(dst_s) >= max_tiers or src == dst:
            continue
        p = src_s.pop()
        dst_s.append(p)
        rems[src][(p, t)] = 1
        adds[dst][(p, t)] = 1
        add_count[dst] += 1
        max_time[src] = max(max_time[src], t)
        max_time[dst] = max(max_time[dst], t)

    sorted_ok = is_yard_sorted(stacks)
    out: Dict[int, StackColumn] = {}
    for s in stacks:
        out[s] = StackColumn(
            stack_idx=s,
            adds=adds[s],
            rems=rems[s],
            cost=int(add_count[s]),
            max_time_used=int(max_time[s]),
        )
    return out, sorted_ok

