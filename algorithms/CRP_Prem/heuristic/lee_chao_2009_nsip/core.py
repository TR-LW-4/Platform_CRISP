"""
Core helpers for Lee & Chao (2009) NS+IP pre-marshalling heuristic.

The module is algorithm-local and intentionally does not import other
algorithm packages.
"""

from __future__ import annotations

import random
from typing import Dict, List, Sequence, Tuple

Stacks = Dict[int, List[int]]           # stack_idx -> priorities bottom..top
Move = Tuple[int, int]                  # (src_idx, dst_idx)
ExecutedMove = Tuple[int, int, int]     # (priority, src_idx, dst_idx)


def clone_stacks(stacks: Stacks) -> Stacks:
    return {k: list(v) for k, v in stacks.items()}


def stack_mis_overlay_index(prios: Sequence[int]) -> int:
    """
    Lee & Chao's stack mis-overlay index:
    max depth (from top) of any container involved in a mis-overlay.
    """
    n = len(prios)
    best = 0
    for i in range(n - 1):
        lower = prios[i]
        for j in range(i + 1, n):
            upper = prios[j]
            if upper > lower:
                depth = n - 1 - i
                if depth > best:
                    best = depth
    return best


def bay_mis_overlay_index(stacks: Stacks) -> int:
    return sum(stack_mis_overlay_index(prios) for prios in stacks.values())


def apply_move(stacks: Stacks, src: int, dst: int, max_tiers: int) -> bool:
    if src == dst:
        return False
    src_s = stacks.get(src)
    dst_s = stacks.get(dst)
    if src_s is None or dst_s is None:
        return False
    if not src_s or len(dst_s) >= max_tiers:
        return False
    block = src_s.pop()
    dst_s.append(block)
    return True


def simulate_sequence(
    stacks_init: Stacks,
    seq: Sequence[Move],
    max_tiers: int,
) -> Tuple[List[Move], Stacks, List[ExecutedMove]]:
    """
    Run a sequence and drop infeasible moves ("rationalization" in paper spirit).
    Returns (feasible_sequence, final_stacks, executed_move_records).
    """
    stacks = clone_stacks(stacks_init)
    feasible: List[Move] = []
    executed: List[ExecutedMove] = []

    for src, dst in seq:
        src_s = stacks.get(src)
        dst_s = stacks.get(dst)
        if src == dst or src_s is None or dst_s is None or not src_s or len(dst_s) >= max_tiers:
            continue
        p = src_s.pop()
        dst_s.append(p)
        feasible.append((src, dst))
        executed.append((p, src, dst))

    return feasible, stacks, executed


def objective_value(
    seq: Sequence[Move],
    final_stacks: Stacks,
    w_idx: float = 1.0,
    w_len: float = 0.1,
) -> float:
    return float(w_idx * bay_mis_overlay_index(final_stacks) + w_len * len(seq))


def random_empty_stack_prefix(
    stacks_init: Stacks,
    max_tiers: int,
    rng: random.Random,
) -> List[Move]:
    """
    Minor subroutine 1: empty a random non-empty stack via random legal moves.
    """
    stacks = clone_stacks(stacks_init)
    nonempty = [k for k, v in stacks.items() if v]
    if not nonempty:
        return []
    target = rng.choice(nonempty)

    moves: List[Move] = []
    while stacks[target]:
        cands = [k for k, v in stacks.items() if k != target and len(v) < max_tiers]
        if not cands:
            break
        dst = rng.choice(cands)
        block = stacks[target].pop()
        stacks[dst].append(block)
        moves.append((target, dst))
    return moves


def neighborhood_mutation(
    seq: Sequence[Move],
    n_stacks: int,
    p_add: float,
    p_delete: float,
    p_relocate: float,
    rng: random.Random,
) -> List[Move]:
    out = list(seq)
    r = rng.random()

    # Add random move
    if r < p_add:
        src = rng.randrange(n_stacks)
        dst = rng.randrange(n_stacks - 1)
        if dst >= src:
            dst += 1
        pos = rng.randrange(len(out) + 1)
        out.insert(pos, (src, dst))
        return out

    # Delete random move
    if r < p_add + p_delete and out:
        del out[rng.randrange(len(out))]
        return out

    # Relocate random move in sequence
    if r < p_add + p_delete + p_relocate and len(out) >= 2:
        i = rng.randrange(len(out))
        mv = out.pop(i)
        j = rng.randrange(len(out) + 1)
        out.insert(j, mv)
        return out

    return out


def reduce_sequence_rule(
    seq: Sequence[Move],
    stacks_init: Stacks,
    max_tiers: int,
) -> List[Move]:
    """
    Minor subroutine 2 (simple rule):
    if same container p moves i: a->b and later j: b->c, and no move targets c
    in between, then merge into a->c and remove move j.
    """
    cur = list(seq)
    feasible, _, executed = simulate_sequence(stacks_init, cur, max_tiers)
    if not executed:
        return feasible

    i = 0
    while i < len(executed):
        p_i, a, b = executed[i]
        merged = False
        for j in range(i + 1, len(executed)):
            p_j, b2, c = executed[j]
            if p_j != p_i or b2 != b:
                continue
            blocked = any(executed[k][2] == c for k in range(i + 1, j))
            if blocked:
                continue
            feasible[i] = (a, c)
            del feasible[j]
            executed = [
                (executed[k][0], executed[k][1], executed[k][2])
                for k in range(len(executed)) if k != j
            ]
            merged = True
            break
        if not merged:
            i += 1

    out, _, _ = simulate_sequence(stacks_init, feasible, max_tiers)
    return out


def reduce_misoverlay_suffix(
    stacks_init: Stacks,
    max_tiers: int,
    max_moves: int = 64,
) -> List[Move]:
    """
    Minor subroutine 3:
    greedily append relocations that reduce bay mis-overlay index.
    """
    stacks = clone_stacks(stacks_init)
    moves: List[Move] = []

    for _ in range(max_moves):
        cur_idx = bay_mis_overlay_index(stacks)
        if cur_idx <= 0:
            break

        best: Tuple[int, int, int] | None = None  # (new_idx, src, dst)
        for src, src_s in stacks.items():
            if not src_s:
                continue
            if stack_mis_overlay_index(src_s) <= 0:
                continue
            for dst, dst_s in stacks.items():
                if src == dst or len(dst_s) >= max_tiers:
                    continue
                trial = clone_stacks(stacks)
                apply_move(trial, src, dst, max_tiers)
                nidx = bay_mis_overlay_index(trial)
                if nidx < cur_idx and (best is None or nidx < best[0]):
                    best = (nidx, src, dst)

        if best is None:
            break
        _, src, dst = best
        apply_move(stacks, src, dst, max_tiers)
        moves.append((src, dst))

    return moves

