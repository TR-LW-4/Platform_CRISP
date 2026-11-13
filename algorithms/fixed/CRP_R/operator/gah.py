"""
------------------------------- Reference --------------------------------
K.-C. Wu, C.-J. Ting,
"A beam search algorithm for minimizing reshuffle operations at
 container yards",
International Conference on Logistics and Maritime Systems (2010).
The group assignment is GAH as used by Bacci, Mattia and Ventura,
Computers & Operations Research 103 (2019) 252–264.
------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple


def _blocking(stacks: Sequence[Sequence[int]]) -> int:
    total = 0
    for stack in stacks:
        for index, item in enumerate(stack):
            if any(below < item for below in stack[:index]):
                total += 1
    return total


def _prepare(stacks: Sequence[Sequence[int]], max_tiers: int, n: int):
    width = len(stacks)
    yard = [[-1] * max_tiers for _ in range(width)]
    earliest = [n + 1] * width
    free = [max_tiers] * width
    location = [[-1, -1] for _ in range(n + 1)]
    for index, stack in enumerate(stacks):
        for slot, item in enumerate(stack):
            yard[index][slot] = item
            free[index] -= 1
            location[item][0] = index
            location[item][1] = slot
            if item < earliest[index]:
                earliest[index] = item
    return yard, earliest, free, location


def _order_descending(blockers: List[List[int]]) -> List[int]:
    order = list(range(len(blockers)))
    for left in range(len(order) - 1):
        for right in range(left + 1, len(order)):
            if blockers[order[left]][1] < blockers[order[right]][1]:
                order[left], order[right] = order[right], order[left]
    return order


def _assign_column(yard, earliest, free, location, stack, slot, n, width, height, blocking_left, first_dest):
    blockers: List[List[int]] = []
    top = height - free[stack] - 1
    top_item = -1
    for level in range(top, slot, -1):
        blockers.append([len(blockers), yard[stack][level], level, -1, -1])
        if first_dest == -1 and level == top:
            top_item = yard[stack][level]
    order = _order_descending(blockers)
    added = 0
    count = len(blockers)

    for position in range(count):
        index = order[position]
        container = blockers[index][1]
        occupied = [0] * width
        for other in range(count):
            other_index = order[other]
            dest = blockers[other_index][3]
            below = blockers[index][2] > blockers[other_index][2]
            later = blockers[index][1] < blockers[other_index][1]
            if below and later and dest != -1:
                occupied[dest] += 1
        best = n * n
        chosen = -1
        for candidate in range(width):
            if candidate != stack and free[candidate] and occupied[candidate] == 0:
                if earliest[candidate] >= container and earliest[candidate] < best:
                    best = earliest[candidate]
                    chosen = candidate
        if chosen == -1:
            continue
        earliest[chosen] = container
        blocking_left -= 1
        added += 1
        if first_dest == -1 and container == top_item:
            first_dest = chosen
        origin = blockers[index][2]
        yard[stack][origin] = -1
        free[stack] += 1
        placed = height - free[chosen]
        yard[chosen][placed] = container
        blockers[index][3] = chosen
        blockers[index][4] = placed
        location[container][0] = chosen
        location[container][1] = placed
        free[chosen] -= 1

    for position in range(count - 1, -1, -1):
        index = order[position]
        if blockers[index][3] != -1:
            continue
        occupied = [0] * width
        named = [0] * width
        for other in range(count):
            other_index = order[other]
            dest = blockers[other_index][3]
            below = blockers[index][2] > blockers[other_index][2]
            later = blockers[index][1] < blockers[other_index][1]
            if below and later and dest != -1:
                if occupied[dest] == 0:
                    named[dest] = blockers[other_index][1]
                occupied[dest] += 1
        container = blockers[index][1]
        origin = blockers[index][2]
        best_bad = -n * n
        best_good = n * n
        bad = good = -1
        yard[stack][origin] = -1
        free[stack] += 1
        for candidate in range(width):
            if candidate == stack or not free[candidate]:
                continue
            if occupied[candidate] <= 1:
                score = min(earliest[candidate] - container, container - named[candidate])
            else:
                score = -n + 1
            if score < 0:
                if score > best_bad:
                    best_bad = score
                    bad = candidate
            elif score < best_good:
                best_good = score
                good = candidate
        chosen = good if good != -1 else bad
        blockers[index][3] = chosen
        anchor_slot = -1
        anchor = -1
        for other in range(count):
            other_index = order[other]
            same = chosen == blockers[other_index][3]
            below = origin > blockers[other_index][2]
            if same and below and blockers[other_index][2] > anchor_slot and other != position:
                anchor_slot = blockers[other_index][2]
                anchor = blockers[other_index][1]
        if anchor != -1:
            old_min = n + 1
            anchor_at = location[anchor][1]
            for level in range(anchor_at):
                if old_min > yard[chosen][level]:
                    old_min = yard[chosen][level]
            level = height - free[chosen]
            while level > anchor_at:
                yard[chosen][level] = yard[chosen][level - 1]
                moved_item = yard[chosen][level]
                if moved_item > 0:
                    location[moved_item][1] = level
                    if container < moved_item <= old_min:
                        blocking_left += 1
                level -= 1
            # The anchor's slot was updated by the shift. The hole is one below that.
            blockers[index][4] = location[anchor][1] - 1
        else:
            blockers[index][4] = height - free[chosen]
        if earliest[chosen] > container:
            earliest[chosen] = container
            blocking_left -= 1
        added += 1
        if first_dest == -1 and container == top_item:
            first_dest = chosen
        placed = blockers[index][4]
        yard[chosen][placed] = container
        location[container][0] = chosen
        location[container][1] = placed
        free[chosen] -= 1
    return blocking_left, first_dest, added


def _simulate(stacks, max_tiers, best_ub, blocking) -> Tuple[int, Optional[int]]:
    present = [item for stack in stacks for item in stack]
    if not present:
        return 0, None
    n = max(present)
    width = len(stacks)
    yard, earliest, free, location = _prepare(stacks, max_tiers, n)
    blocking_left = _blocking(stacks) if blocking is None else blocking
    moved = 0
    target = min(present)
    first_dest = -1
    while target <= n and blocking_left > 0:
        if moved + blocking_left >= best_ub:
            return n * n, None if first_dest < 0 else first_dest
        stack, slot = location[target]
        if slot < max_tiers - 1 and yard[stack][slot + 1] != -1:
            blocking_left, first_dest, added = _assign_column(
                yard, earliest, free, location, stack, slot, n, width, max_tiers,
                blocking_left, first_dest,
            )
            moved += added
        yard[stack][slot] = -1
        free[stack] += 1
        earliest[stack] = n + 1
        for level in range(slot):
            if yard[stack][level] != -1 and yard[stack][level] < earliest[stack]:
                earliest[stack] = yard[stack][level]
        location[target][0] = -1
        target += 1
    return moved, None if first_dest < 0 else first_dest


def choose_destination(stacks, max_tiers) -> Optional[int]:
    """Destination GAH records for the current top blocker."""
    present = [item for stack in stacks for item in stack]
    if not present:
        return None
    target = min(present)
    source = next(index for index, stack in enumerate(stacks) if target in stack)
    if stacks[source][-1] == target:
        return None
    n = max(present)
    _cost, destination = _simulate(stacks, max_tiers, n * n, None)
    return destination


def upper_bound(stacks, max_tiers, best_ub, blocking=None) -> int:
    """GAH completion cost. Each blocked column is assigned as a group."""
    cost, _destination = _simulate(stacks, max_tiers, best_ub, blocking)
    return cost
