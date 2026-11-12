"""One corridor pass. Each blocker keeps delta stacks, then finishes with greedy."""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from .greedy import Move, greedy


def _weights(good_sum: int, bad_sum: int, n_empty: int) -> Tuple[float, float, float]:
    has_good = good_sum > 0
    has_bad = bad_sum > 0
    has_empty = n_empty > 0
    if has_good and has_bad and has_empty:
        return 0.25, 0.5, 0.25
    if has_good and has_bad:
        return 0.0, 2.0 / 3.0, 1.0 / 3.0
    if has_good and has_empty:
        return 1.0 / 3.0, 2.0 / 3.0, 0.0
    if has_good:
        return 0.0, 1.0, 0.0
    if has_bad and has_empty:
        return 0.5, 0.0, 0.5
    if has_bad:
        return 0.0, 0.0, 1.0
    if has_empty:
        return 1.0, 0.0, 0.0
    return 0.0, 0.0, 0.0


def _scores(
    stacks: Sequence[Sequence[int]],
    src: int,
    max_tiers: int,
) -> List[Tuple[int, float]]:
    container = stacks[src][-1]
    empty: List[int] = []
    good: List[Tuple[int, int]] = []
    bad: List[Tuple[int, int]] = []
    for index, stack in enumerate(stacks):
        if index == src or len(stack) >= max_tiers:
            continue
        if not stack:
            empty.append(index)
            continue
        earliest = min(stack)
        if earliest > container:
            good.append((index, earliest))
        else:
            bad.append((index, earliest))

    good_sum = sum(earliest for _index, earliest in good)
    bad_sum = sum(earliest for _index, earliest in bad)
    empty_weight, good_weight, bad_weight = _weights(good_sum, bad_sum, len(empty))
    scored: List[Tuple[int, float]] = []
    if empty:
        share = empty_weight / len(empty)
        scored.extend((index, share) for index in empty)
    if good and good_sum:
        raw = [(index, good_sum / earliest) for index, earliest in good]
        scale = sum(value for _index, value in raw)
        scored.extend((index, good_weight * value / scale) for index, value in raw)
    if bad and bad_sum:
        scored.extend(
            (index, bad_weight * earliest / bad_sum) for index, earliest in bad
        )
    return scored


def _corridor(
    stacks: Sequence[Sequence[int]],
    src: int,
    delta: int,
    max_tiers: int,
    rng: np.random.Generator,
) -> List[int]:
    scored = _scores(stacks, src, max_tiers)
    if not scored:
        return []
    if delta < 0 or delta >= len(scored):
        return [index for index, _score in scored]
    remaining = {index: score for index, score in scored}
    chosen: List[int] = []
    width = min(delta, len(remaining))
    for _ in range(width):
        indexes = list(remaining)
        weights = np.asarray([remaining[index] for index in indexes], dtype=float)
        total = float(weights.sum())
        if total <= 0.0:
            picked = int(rng.choice(indexes))
        else:
            picked = int(rng.choice(indexes, p=weights / total))
        chosen.append(picked)
        del remaining[picked]
    return chosen


def trajectory(
    stacks_init: List[List[int]],
    n_total: int,
    delta: int,
    max_tiers: int,
    best: List[Move],
    best_cost: int,
    rng: np.random.Generator,
    expired: Callable[[], bool],
) -> Tuple[List[Move], int]:
    """Walk one bay. A greedy completion can replace the incumbent mid-pass."""
    stacks = [list(stack) for stack in stacks_init]
    moves: List[Move] = []
    cost = 0

    for target in range(1, n_total + 1):
        if expired():
            break
        src = next((index for index, stack in enumerate(stacks) if target in stack), -1)
        if src < 0:
            continue
        while stacks[src] and stacks[src][-1] != target:
            if expired():
                return best, best_cost
            chosen = _corridor(stacks, src, delta, max_tiers, rng)
            if not chosen:
                return best, best_cost
            container = stacks[src][-1]
            best_dst: Optional[int] = None
            best_heur = None
            for dst in sorted(chosen):
                trial = [list(stack) for stack in stacks]
                trial[dst].append(trial[src].pop())
                suffix, heur = greedy(trial, n_total, max_tiers)
                total = cost + 1 + heur
                if total < best_cost:
                    best_cost = total
                    best = moves + [(container, src, dst)] + suffix
                if best_heur is None or heur < best_heur:
                    best_heur = heur
                    best_dst = dst
            if best_dst is None:
                return best, best_cost
            cost += 1
            if cost >= best_cost:
                return best, best_cost
            stacks[best_dst].append(stacks[src].pop())
            moves.append((container, src, best_dst))
        if stacks[src] and stacks[src][-1] == target:
            moves.append((target, src, None))
            stacks[src].pop()
    return best, best_cost
