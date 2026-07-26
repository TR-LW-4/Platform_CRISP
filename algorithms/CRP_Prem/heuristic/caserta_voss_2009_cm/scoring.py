"""
Caserta & Voß (2009) — Corridor Method (CM) for Pre-marshalling.

This file keeps a paper-faithful core:
  1) corridor definition/selection
  2) neighborhood generation
  3) move evaluation + elite roulette selection
  4) local search (heuristic_swap + heuristic_subsequence_building)

No extra diversification/perturbation mechanism is used.
"""

from __future__ import annotations

import random
import time
from typing import Any, Dict, List, Optional, Tuple

Stacks = Dict[Any, List[int]]


def stack_forced_relocations(prios: List[int]) -> int:
    count = 0
    for i in range(len(prios) - 1):
        if prios[i] < prios[i + 1]:
            count += 1
    return count


def all_forced_relocations(stacks: Stacks) -> Dict[Any, int]:
    return {k: stack_forced_relocations(v) for k, v in stacks.items()}


def total_forced_relocations(stacks: Stacks) -> int:
    return sum(stack_forced_relocations(v) for v in stacks.values())


def min_priority(prios: List[int], n_total: int) -> int:
    return min(prios) if prios else n_total + 1


def roulette_pick(weights: Dict[Any, float], rng: random.Random) -> Any:
    keys = list(weights.keys())
    vals = [max(0.0, weights[k]) for k in keys]
    total = sum(vals)
    if total <= 0:
        return rng.choice(keys)
    r = rng.uniform(0.0, total)
    acc = 0.0
    for k, w in zip(keys, vals):
        acc += w
        if acc >= r:
            return k
    return keys[-1]


def select_source_stack(stacks: Stacks, rng: random.Random) -> Optional[Any]:
    forced = all_forced_relocations(stacks)
    candidates = {k: v for k, v in forced.items() if v > 0}
    if not candidates:
        return None
    return roulette_pick(candidates, rng)


def classify_destinations(
    stacks: Stacks,
    src_key: Any,
    block: int,
    max_tiers: int,
    n_total: int,
) -> Tuple[List[Any], List[Any], List[Any]]:
    s0: List[Any] = []
    s1: List[Any] = []
    s2: List[Any] = []
    for key, prios in stacks.items():
        if key == src_key or len(prios) >= max_tiers:
            continue
        if not prios:
            s0.append(key)
            continue
        m = min_priority(prios, n_total)
        if m > block:
            s1.append(key)
        else:
            s2.append(key)
    return s0, s1, s2


def score_destinations(
    stacks: Stacks,
    s0: List[Any], s1: List[Any], s2: List[Any],
    n_total: int,
    w0: float, w1: float, w2: float,
) -> Dict[Any, float]:
    scores: Dict[Any, float] = {}
    for k in s0:
        scores[k] = w0 / len(s0)

    if s1:
        sum1 = sum(min_priority(stacks[k], n_total) for k in s1)
        for k in s1:
            m = min_priority(stacks[k], n_total)
            scores[k] = w1 * (sum1 / m if m > 0 else sum1)

    if s2:
        sum2 = sum(min_priority(stacks[k], n_total) for k in s2)
        for k in s2:
            m = min_priority(stacks[k], n_total)
            scores[k] = w2 * (m / sum2 if sum2 > 0 else 0.0)

    return scores


def select_corridor(scores: Dict[Any, float], delta: int, rng: random.Random) -> List[Any]:
    remaining = dict(scores)
    corridor: List[Any] = []
    for _ in range(min(delta, len(remaining))):
        if not remaining:
            break
        pick = roulette_pick(remaining, rng)
        corridor.append(pick)
        del remaining[pick]
    return corridor


def evaluate_move(stacks: Stacks, src_key: Any, dst_key: Any) -> int:
    block = stacks[src_key][-1]
    trial = {k: list(v) for k, v in stacks.items()}
    trial[src_key].pop()
    trial[dst_key].append(block)
    return total_forced_relocations(trial)


def select_move(
    stacks: Stacks,
    src_key: Any,
    corridor: List[Any],
    elite_quantile: float,
    rng: random.Random,
) -> Optional[Any]:
    if not corridor:
        return None
    scored = [(dst, evaluate_move(stacks, src_key, dst)) for dst in corridor]
    scored.sort(key=lambda x: x[1])
    n_elite = max(1, int(round(len(scored) * elite_quantile)))
    elite = scored[:n_elite]
    worst = elite[-1][1]
    weights = {dst: (worst - g + 1) for dst, g in elite}
    return roulette_pick(weights, rng)


def _build_sorted_sequence(
    stacks: Stacks,
    target: Any,
    max_tiers: int,
    first_forbidden_src: Optional[Any] = None,
) -> int:
    """Fill target stack with a descending top sequence (paper Fig. 2 idea)."""
    moves = 0
    last_placed: Optional[int] = None
    forbidden = first_forbidden_src
    while len(stacks[target]) < max_tiers:
        candidates: List[Tuple[Any, int]] = []
        for key, prios in stacks.items():
            if key == target or key == forbidden or not prios:
                continue
            top = prios[-1]
            if last_placed is None or top <= last_placed:
                candidates.append((key, top))
        forbidden = None
        if not candidates:
            break
        src, block = max(candidates, key=lambda kv: kv[1])
        stacks[src].pop()
        stacks[target].append(block)
        last_placed = block
        moves += 1
    return moves


def heuristic_swap(stacks: Stacks, max_tiers: int, n_total: int) -> int:
    """
    Local search operator from Section 3:
    - build sorted sequences on empty stacks
    - if no empty stack exists, try single-block-stack evacuation then build.
    """
    def _sig() -> Tuple[Tuple[Any, Tuple[int, ...]], ...]:
        return tuple(sorted((k, tuple(v)) for k, v in stacks.items()))

    moves = 0
    progress = True
    seen = set()
    max_rounds = max(1, n_total * max(1, len(stacks)))
    rounds = 0
    while progress and rounds < max_rounds:
        rounds += 1
        sig = _sig()
        if sig in seen:
            break
        seen.add(sig)
        progress = False

        # Case A: empty-stack build (Fig. 2)
        empty_target = next((k for k, v in stacks.items() if not v), None)
        if empty_target is not None:
            made = _build_sorted_sequence(stacks, empty_target, max_tiers)
            if made > 0:
                moves += made
                progress = True
                continue

        # Case B: single-block stack build (Fig. 3)
        for stack_k, prios_k in stacks.items():
            if len(prios_k) != 1:
                continue
            block = prios_k[0]
            # L = top blocks lower than 'block'
            l_vals = [v[-1] for kk, v in stacks.items() if kk != stack_k and v and v[-1] < block]
            if not l_vals:
                continue

            forced = all_forced_relocations(stacks)
            zero_forced_nonfull = [
                kk for kk, vv in stacks.items()
                if kk != stack_k and len(vv) < max_tiers and forced.get(kk, 0) == 0
            ]
            if not zero_forced_nonfull:
                continue

            # Prefer stack with minimum top value among zero-forced stacks (paper text).
            dst = min(
                zero_forced_nonfull,
                key=lambda kk: stacks[kk][-1] if stacks[kk] else (n_total + 1),
            )

            stacks[stack_k].pop()
            stacks[dst].append(block)
            moves += 1

            made = _build_sorted_sequence(
                stacks, target=stack_k, max_tiers=max_tiers, first_forbidden_src=dst
            )
            moves += made
            if made > 0:
                progress = True
                break
    return moves


def heuristic_subsequence_building(stacks: Stacks) -> int:
    """
    Build tight subsequences:
    move top v from src onto dst where dst currently has zero forced
    relocations and dst top is v+1.
    """
    def _sig() -> Tuple[Tuple[Any, Tuple[int, ...]], ...]:
        return tuple(sorted((k, tuple(v)) for k, v in stacks.items()))

    moves = 0
    progressed = True
    seen = set()
    while progressed:
        sig = _sig()
        if sig in seen:
            break
        seen.add(sig)
        progressed = False
        forced = all_forced_relocations(stacks)
        for src_key, src_prios in stacks.items():
            if not src_prios:
                continue
            v = src_prios[-1]
            for dst_key, dst_prios in stacks.items():
                if dst_key == src_key or not dst_prios:
                    continue
                if forced.get(dst_key, 0) == 0 and dst_prios[-1] == v + 1:
                    src_prios.pop()
                    dst_prios.append(v)
                    moves += 1
                    progressed = True
                    break
            if progressed:
                break
    return moves


def corridor_method_attempt(
    stacks_in: Stacks,
    n_total: int,
    max_tiers: int,
    delta: int,
    w0: float,
    w1: float,
    w2: float,
    elite_quantile: float,
    rng: random.Random,
    deadline: Optional[float],
) -> Tuple[Stacks, int, bool]:
    stacks = {k: list(v) for k, v in stacks_in.items()}
    moves = 0

    moves += heuristic_swap(stacks, max_tiers, n_total)
    moves += heuristic_subsequence_building(stacks)

    while total_forced_relocations(stacks) > 0:
        if deadline is not None and time.perf_counter() >= deadline:
            break

        src_key = select_source_stack(stacks, rng)
        if src_key is None:
            break
        block = stacks[src_key][-1]

        s0, s1, s2 = classify_destinations(stacks, src_key, block, max_tiers, n_total)
        if not (s0 or s1 or s2):
            break

        scores = score_destinations(stacks, s0, s1, s2, n_total, w0, w1, w2)
        corridor = select_corridor(scores, delta, rng)
        dst_key = select_move(stacks, src_key, corridor, elite_quantile, rng)
        if dst_key is None:
            break

        stacks[src_key].pop()
        stacks[dst_key].append(block)
        moves += 1

        moves += heuristic_swap(stacks, max_tiers, n_total)
        moves += heuristic_subsequence_building(stacks)

    solved = total_forced_relocations(stacks) == 0
    return stacks, moves, solved


def corridor_method_solve(
    stacks_init: Stacks,
    n_total: int,
    max_tiers: int,
    delta: int = 6,
    w0: float = 0.4,
    w1: float = 0.3,
    w2: float = 0.3,
    elite_quantile: float = 0.5,
    num_restarts: int = 10,
    time_limit: float = 20.0,
    seed: int = 0,
) -> Tuple[int, int, bool]:
    """Multi-restart driver under the paper stopping criterion."""
    rng = random.Random(seed)
    deadline = time.perf_counter() + max(0.0, time_limit)

    best_key: Optional[Tuple[int, int, int]] = None  # (solved_flag, remaining, moves)
    best_moves = 0
    best_remaining = total_forced_relocations(stacks_init)
    best_solved = False

    for _ in range(max(1, num_restarts)):
        if time.perf_counter() >= deadline:
            break
        attempt_rng = random.Random(rng.randint(0, 2**31 - 1))
        final_stacks, moves, solved = corridor_method_attempt(
            stacks_in=stacks_init,
            n_total=n_total,
            max_tiers=max_tiers,
            delta=delta,
            w0=w0,
            w1=w1,
            w2=w2,
            elite_quantile=elite_quantile,
            rng=attempt_rng,
            deadline=deadline,
        )
        remaining = total_forced_relocations(final_stacks)
        key = (0 if solved else 1, remaining, moves)
        if best_key is None or key < best_key:
            best_key = key
            best_moves = moves
            best_remaining = remaining
            best_solved = solved

    return best_moves, best_remaining, best_solved
