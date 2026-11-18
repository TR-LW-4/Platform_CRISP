"""
------------------------------- Note -------------------------------------
The function moves builds the relocation sequence in one pass. It
assigns the containers above the current target once, relocates them
from the top of the stack downward, and repeats this until the yard is
empty. The function choose_destination returns only the destination of
the uppermost container. If choose_destination is called again after
each relocation, its sequence can differ from the complete sequence
returned by moves.
------------------------------- Reference --------------------------------
C.-J. Ting, K.-C. Wu,
"Optimizing container relocation operations at container yards with
 beam search",
Transportation Research Part E 103 (2017) 17–31.
------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from algorithms.distinct.CRP_R.operator.sdh import choose_destination as sdh_destination


class Move:
    def __init__(self, container, source, destination):
        self.container = container
        self.source = source
        self.destination = destination


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


def _column(stacks: Sequence[Sequence[int]]):
    priorities = [priority for stack in stacks for priority in stack]
    if not priorities:
        return None
    n_total = max(priorities)
    target = min(priorities)
    source = next(index for index, stack in enumerate(stacks) if target in stack)
    if stacks[source][-1] == target:
        return n_total, target, source, []
    tier = stacks[source].index(target)
    return n_total, target, source, list(stacks[source][tier + 1:])


def choose_destination(stacks, max_tiers) -> Optional[int]:
    column = _column(stacks)
    if column is None:
        return None
    n_total, _target, source, blockers = column
    if not blockers:
        return None
    assignments = _assign(blockers, source, stacks, n_total, max_tiers)
    destination = assignments.get(len(blockers) - 1)
    if destination is None:
        return sdh_destination(stacks, max_tiers)
    return destination


def moves(stacks, max_tiers) -> List[Move]:
    """One-shot VRH. Each column is assigned once, then moved from the top down."""
    state = [list(stack) for stack in stacks]
    priorities = [priority for stack in state for priority in stack]
    if not priorities:
        return []
    n_total = max(priorities)
    result: List[Move] = []

    for target in range(1, n_total + 1):
        source = next((index for index, stack in enumerate(state) if target in stack), None)
        if source is None:
            continue
        tier = state[source].index(target)
        blockers = state[source][tier + 1:]
        if not blockers:
            result.append(Move(target, (source, len(state[source])), None))
            state[source].pop()
            continue
        assignments = _assign(blockers, source, state, n_total, max_tiers)
        for index in range(len(blockers) - 1, -1, -1):
            destination = assignments.get(index)
            if destination is None:
                destination = sdh_destination(state, max_tiers)
            if destination is None:
                raise ValueError(f"no free stack for container {blockers[index]}")
            result.append(Move(
                blockers[index],
                (source, len(state[source])),
                (destination, len(state[destination]) + 1),
            ))
            state[source].pop()
            state[destination].append(blockers[index])
        if state[source] and state[source][-1] == target:
            result.append(Move(target, (source, len(state[source])), None))
            state[source].pop()
    return result


from algorithms.ALGORITHM import Algorithm


class VRH(Algorithm):

    def main(self):
        return moves(self.problem.stacks, self.problem.max_tiers)