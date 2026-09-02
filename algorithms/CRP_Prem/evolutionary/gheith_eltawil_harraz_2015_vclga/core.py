"""
Variable-length chromosome operators for GheithEltawilHarraz2015VCLGA.

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
Move = Tuple[int, int]         # (src_idx, dst_idx)


@dataclass
class EvalResult:
    chromosome: List[Move]
    executed: List[Move]
    bad_overlaps: int
    moves_used: int
    fitness: int


def clone_stacks(stacks: Stacks) -> Stacks:
    return {k: list(v) for k, v in stacks.items()}


def total_bad_overlaps(stacks: Stacks) -> int:
    bad = 0
    for arr in stacks.values():
        for i in range(len(arr) - 1):
            if arr[i] < arr[i + 1]:
                bad += 1
    return bad


def all_possible_moves(n_stacks: int) -> List[Move]:
    return [(i, j) for i in range(n_stacks) for j in range(n_stacks) if i != j]


def apply_move(stacks: Stacks, mv: Move, max_tiers: int) -> bool:
    src, dst = mv
    if src == dst:
        return False
    if src not in stacks or dst not in stacks:
        return False
    if not stacks[src] or len(stacks[dst]) >= max_tiers:
        return False
    x = stacks[src].pop()
    stacks[dst].append(x)
    return True


def feasible_solution_implementation(
    stacks_init: Stacks,
    chromosome: Sequence[Move],
    max_tiers: int,
) -> EvalResult:
    stacks = clone_stacks(stacks_init)
    executed: List[Move] = []
    for mv in chromosome:
        if apply_move(stacks, mv, max_tiers):
            executed.append(mv)
    bad = total_bad_overlaps(stacks)
    used = len(executed)
    # Lexicographic objective emulation: eliminate mis-overlays first, then moves.
    fit = bad * 1_000_000 + used
    return EvalResult(
        chromosome=list(chromosome),
        executed=executed,
        bad_overlaps=bad,
        moves_used=used,
        fitness=fit,
    )


def cyclic_movements_elimination(chromosome: Sequence[Move]) -> List[Move]:
    out: List[Move] = []
    for mv in chromosome:
        if out and out[-1] == (mv[1], mv[0]):
            out.pop()
            continue
        out.append(mv)
    return out


def minimum_chromosome_length_preservation(
    chromosome: Sequence[Move],
    min_len: int,
    move_pool: Sequence[Move],
    rng: random.Random,
) -> List[Move]:
    out = list(chromosome)
    while len(out) < min_len:
        out.append(move_pool[rng.randrange(len(move_pool))])
    return out


def parent_generation(
    pop_size: int,
    move_pool: Sequence[Move],
    min_len: int,
    max_len: int,
    rng: random.Random,
) -> List[List[Move]]:
    out: List[List[Move]] = []
    for _ in range(pop_size):
        ln = rng.randint(min_len, max_len)
        c = [move_pool[rng.randrange(len(move_pool))] for _ in range(ln)]
        out.append(c)
    return out


def single_point_crossover(
    p1: Sequence[Move],
    p2: Sequence[Move],
    rng: random.Random,
) -> Tuple[List[Move], List[Move]]:
    if not p1 or not p2:
        return list(p1), list(p2)
    cut = rng.randint(1, min(len(p1), len(p2)))
    c1 = list(p1[:cut]) + list(p2[cut:])
    c2 = list(p2[:cut]) + list(p1[cut:])
    return c1, c2


def growth_mutation(
    chrom: Sequence[Move],
    move_pool: Sequence[Move],
    rng: random.Random,
) -> List[Move]:
    if not move_pool:
        return list(chrom)
    c = list(chrom)
    pos = rng.randint(0, len(c))
    c.insert(pos, move_pool[rng.randrange(len(move_pool))])
    return c


def shrink_mutation(chrom: Sequence[Move], rng: random.Random) -> List[Move]:
    if not chrom:
        return []
    c = list(chrom)
    pos = rng.randrange(len(c))
    del c[pos]
    return c


def swap_mutation(chrom: Sequence[Move], rng: random.Random) -> List[Move]:
    if len(chrom) < 2:
        return list(chrom)
    c = list(chrom)
    i = rng.randrange(len(c))
    j = rng.randrange(len(c))
    while j == i:
        j = rng.randrange(len(c))
    c[i], c[j] = c[j], c[i]
    return c


def replace_mutation(
    chrom: Sequence[Move],
    move_pool: Sequence[Move],
    rng: random.Random,
) -> List[Move]:
    if not chrom or not move_pool:
        return list(chrom)
    c = list(chrom)
    i = rng.randrange(len(c))
    c[i] = move_pool[rng.randrange(len(move_pool))]
    return c


def tournament_selection(
    evaluated: Sequence[EvalResult],
    k: int,
    rng: random.Random,
) -> EvalResult:
    kk = max(1, min(k, len(evaluated)))
    picks = [evaluated[rng.randrange(len(evaluated))] for _ in range(kk)]
    picks.sort(key=lambda e: (e.bad_overlaps, e.moves_used, e.fitness))
    return picks[0]

