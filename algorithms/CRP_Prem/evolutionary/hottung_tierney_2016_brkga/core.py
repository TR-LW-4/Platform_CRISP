"""
BRKGA decoder and population update for HottungTierney2016BRKGA.

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
corresponding original algorithm paper.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

Stacks = Dict[int, List[int]]  # stack_idx -> priorities bottom..top
Move = Tuple[int, int]         # (src, dst)


@dataclass
class DecodeResult:
    moves: List[Move]
    bad_overlaps: int
    fitness: float


def clone_stacks(stacks: Stacks) -> Stacks:
    return {k: list(v) for k, v in stacks.items()}


def total_bad_overlaps(stacks: Stacks) -> int:
    bad = 0
    for arr in stacks.values():
        for i in range(len(arr) - 1):
            if arr[i] < arr[i + 1]:
                bad += 1
    return bad


def is_misoverlaid_stack(arr: Sequence[int]) -> bool:
    for i in range(len(arr) - 1):
        if arr[i] < arr[i + 1]:
            return True
    return False


def top_group(arr: Sequence[int], g_max: int) -> int:
    if not arr:
        return g_max
    return int(arr[-1])


def apply_move(stacks: Stacks, mv: Move, max_tiers: int) -> bool:
    src, dst = mv
    if src == dst or src not in stacks or dst not in stacks:
        return False
    if not stacks[src] or len(stacks[dst]) >= max_tiers:
        return False
    x = stacks[src].pop()
    stacks[dst].append(x)
    return True


def find_excellent_move(stacks: Stacks, max_tiers: int, g_max: int) -> List[Move]:
    """
    Excellent move: move a misoverlaying top container into a non-misoverlaid
    stack where no new misoverlay is created.
    """
    mis_stacks = [s for s, arr in stacks.items() if is_misoverlaid_stack(arr) and arr]
    if not mis_stacks:
        return []
    recv_stacks = [
        s for s, arr in stacks.items()
        if (not is_misoverlaid_stack(arr)) and len(arr) < max_tiers
    ]
    if not recv_stacks:
        return []

    # Prefer high groups first (LPFH-style tendency from CPMP literature).
    mis_stacks.sort(key=lambda s: stacks[s][-1], reverse=True)
    for src in mis_stacks:
        p = stacks[src][-1]
        best_dst = None
        for dst in recv_stacks:
            if src == dst:
                continue
            if not stacks[dst]:
                best_dst = dst
                break
            if stacks[dst][-1] >= p:
                best_dst = dst
                break
        if best_dst is not None:
            return [(src, best_dst)]
    return []


def all_normal_moves(stacks: Stacks, max_tiers: int, g_max: int) -> List[Move]:
    out: List[Move] = []
    for src, arrs in stacks.items():
        if not arrs:
            continue
        src_mis = is_misoverlaid_stack(arrs)
        for dst, arrd in stacks.items():
            if src == dst or len(arrd) >= max_tiers:
                continue
            dst_mis = is_misoverlaid_stack(arrd)
            if dst_mis:
                continue
            p = arrs[-1]
            if arrd and arrd[-1] < p:
                continue
            # normal moves from misoverlaid stacks or small clean stacks
            if src_mis or len(arrs) <= 2:
                out.append((src, dst))
    return out


def score_normal_move(stacks: Stacks, mv: Move, chrom: Sequence[float], g_max: int) -> float:
    src, dst = mv
    p = stacks[src][-1]
    tr = top_group(stacks[dst], g_max)
    gap = tr - p
    min_group_src = min(stacks[src]) if stacks[src] else g_max
    h_src = len(stacks[src])
    h_dst = len(stacks[dst])

    # compact weighted rating (chromosome-guided)
    w_gap = chrom[0]
    w_min = chrom[1]
    w_hs = chrom[2]
    w_hd = chrom[3]
    w_clear = chrom[4]

    src_after = list(stacks[src][:-1])
    clear_bonus = 1.0 if (is_misoverlaid_stack(stacks[src]) and not is_misoverlaid_stack(src_after)) else 0.0
    return (
        w_gap * gap
        - w_min * min_group_src
        - w_hs * h_src
        + w_hd * h_dst
        + w_clear * clear_bonus
    )


def clear_stack_move_sequence(
    stacks: Stacks,
    max_tiers: int,
    chrom: Sequence[float],
    retries: int,
    rng: random.Random,
) -> List[Move]:
    n_stacks = len(stacks)
    # choose provider stack by chromosome preference + current "difficulty"
    provider_candidates = [s for s, arr in stacks.items() if arr]
    if not provider_candidates:
        return []
    provider_candidates.sort(
        key=lambda s: (
            -chrom[5 + (s % max(1, min(n_stacks, len(chrom) - 6)))],
            -len(stacks[s]),
            0 if is_misoverlaid_stack(stacks[s]) else 1,
        )
    )
    src = provider_candidates[0]

    best_seq: List[Move] = []
    best_score = float("-inf")

    for _ in range(max(1, retries)):
        trial = clone_stacks(stacks)
        seq: List[Move] = []
        score = 0.0
        # try moving out entire selected stack
        while trial[src]:
            dst_cands = [
                d for d in range(n_stacks)
                if d != src and len(trial[d]) < max_tiers
            ]
            if not dst_cands:
                break
            rng.shuffle(dst_cands)
            dst = dst_cands[0]
            p = trial[src][-1]
            if trial[dst] and trial[dst][-1] < p:
                score -= 2.0
            else:
                score += 1.0
            ok = apply_move(trial, (src, dst), max_tiers)
            if not ok:
                break
            seq.append((src, dst))
        score -= 0.1 * len(seq)
        if score > best_score and seq:
            best_score = score
            best_seq = seq
    return best_seq


def decode_brkga_solution(
    stacks_init: Stacks,
    max_tiers: int,
    chromosome: Sequence[float],
    max_steps: int,
    clear_retries: int,
    rng: random.Random,
) -> DecodeResult:
    stacks = clone_stacks(stacks_init)
    g_max = max((max(v) for v in stacks.values() if v), default=1)
    moves: List[Move] = []

    for _ in range(max_steps):
        bad = total_bad_overlaps(stacks)
        if bad <= 0:
            break

        m = find_excellent_move(stacks, max_tiers=max_tiers, g_max=g_max)
        if not m:
            normals = all_normal_moves(stacks, max_tiers=max_tiers, g_max=g_max)
            if normals:
                best = max(normals, key=lambda x: score_normal_move(stacks, x, chromosome, g_max))
                m = [best]
            else:
                m = clear_stack_move_sequence(
                    stacks,
                    max_tiers=max_tiers,
                    chrom=chromosome,
                    retries=clear_retries,
                    rng=rng,
                )
                if not m:
                    break

        for mv in m:
            if apply_move(stacks, mv, max_tiers):
                moves.append(mv)
        if len(moves) >= max_steps:
            break

    bad = total_bad_overlaps(stacks)
    # prioritize low bad_overlaps then short sequence
    fit = float(bad * 1_000_000 + len(moves))
    return DecodeResult(moves=moves, bad_overlaps=bad, fitness=fit)

