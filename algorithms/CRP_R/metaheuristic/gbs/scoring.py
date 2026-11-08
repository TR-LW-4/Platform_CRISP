"""Completion scores for the greedy beam search."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from algorithms.CRP_R.operator.sdh import choose_destination


def sdh_cost(stacks: Sequence[Sequence[int]], n_total: int, max_tiers: int) -> int:
    """Relocation count of walking the stack-difference rule."""
    state = [list(stack) for stack in stacks]
    relocations = 0
    for target in range(1, n_total + 1):
        source = next((index for index, stack in enumerate(state) if target in stack), None)
        if source is None:
            continue
        while state[source] and state[source][-1] != target:
            destination = choose_destination(state, max_tiers)
            if destination is None:
                return n_total * max_tiers
            container = state[source].pop()
            state[destination].append(container)
            relocations += 1
        if state[source] and state[source][-1] == target:
            state[source].pop()
    return relocations


def _difference(
    blocker: int,
    source: int,
    stacks: Sequence[Sequence[int]],
    n_total: int,
    max_tiers: int,
) -> Optional[int]:
    good: List[Tuple[int, int]] = []
    bad: List[Tuple[int, int]] = []
    for destination, stack in enumerate(stacks):
        if destination == source or len(stack) >= max_tiers:
            continue
        earliest = min(stack) if stack else n_total + 1
        difference = earliest - blocker
        if difference > 0:
            good.append((difference, destination))
        else:
            bad.append((difference, destination))
    if good:
        return min(good)[1]
    if bad:
        return max(bad)[1]
    return None


def _assign(
    blockers: Sequence[int],
    source: int,
    stacks: Sequence[Sequence[int]],
    n_total: int,
    max_tiers: int,
) -> Dict[int, int]:
    assignments: Dict[int, int] = {}
    earliest = [min(stack) if stack else n_total + 1 for stack in stacks]
    minimum_tier = [float("inf")] * len(stacks)
    assigned: List[List[Tuple[int, int]]] = [[] for _ in stacks]
    heights = [len(stack) for stack in stacks]

    for tier in sorted(range(len(blockers)), key=lambda index: -blockers[index]):
        blocker = blockers[tier]
        valid = []
        for destination in range(len(stacks)):
            if destination == source or heights[destination] >= max_tiers:
                continue
            difference = earliest[destination] - blocker
            if difference > 0 and tier < minimum_tier[destination]:
                valid.append((difference, destination))
        if not valid:
            continue
        _, destination = min(valid)
        assignments[tier] = destination
        earliest[destination] = blocker
        minimum_tier[destination] = min(minimum_tier[destination], tier)
        assigned[destination].append((blocker, tier))
        heights[destination] += 1

    skipped = [tier for tier in range(len(blockers)) if tier not in assignments]
    skipped.sort(key=lambda tier: blockers[tier])
    for tier in skipped:
        blocker = blockers[tier]
        best_destination = None
        best_value = float("-inf")
        for destination in range(len(stacks)):
            if destination == source or heights[destination] >= max_tiers:
                continue
            uppers = [
                priority
                for priority, assigned_tier in assigned[destination]
                if assigned_tier < tier and priority > blocker
            ]
            lowers = [
                priority
                for priority, assigned_tier in assigned[destination]
                if assigned_tier > tier
            ] + list(stacks[destination])
            upper_min = min(uppers) if uppers else n_total + 1
            lower_min = min(lowers) if lowers else n_total + 1
            if len(uppers) <= 1:
                value = min(lower_min - blocker, blocker - upper_min)
            else:
                value = -n_total + blocker - upper_min
            if value > best_value:
                best_value = value
                best_destination = destination
        if best_destination is None:
            best_destination = _difference(blocker, source, stacks, n_total, max_tiers)
        if best_destination is None:
            continue
        assignments[tier] = best_destination
        assigned[best_destination].append((blocker, tier))
        earliest[best_destination] = min(earliest[best_destination], blocker)
        heights[best_destination] += 1
    return assignments


def vrh_cost(stacks: Sequence[Sequence[int]], n_total: int, max_tiers: int) -> int:
    """Relocation count after assigning each blocked column together."""
    state = [list(stack) for stack in stacks]
    relocations = 0
    for target in range(1, n_total + 1):
        source = next((index for index, stack in enumerate(state) if target in stack), None)
        if source is None:
            continue
        tier = state[source].index(target)
        blockers = state[source][tier + 1:]
        if not blockers:
            state[source].pop()
            continue
        assignments = _assign(blockers, source, state, n_total, max_tiers)
        for index in range(len(blockers) - 1, -1, -1):
            destination = assignments.get(index)
            if destination is None:
                destination = choose_destination(state, max_tiers)
            if destination is None:
                return n_total * max_tiers
            state[source].pop()
            state[destination].append(blockers[index])
            relocations += 1
        if state[source] and state[source][-1] == target:
            state[source].pop()
    return relocations
