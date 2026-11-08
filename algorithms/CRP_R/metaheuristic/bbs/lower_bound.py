"""UBALB lower bound from Bacci, Mattia and Ventura (2019)."""

from __future__ import annotations

from typing import List, Sequence


def blocking_count(stacks: Sequence[Sequence[int]]) -> int:
    total = 0
    for stack in stacks:
        for index, item in enumerate(stack):
            if any(below < item for below in stack[:index]):
                total += 1
    return total


def _sort_by_earliest(ids: List[int], earliest: Sequence[int]) -> None:
    """Descending earliest-item order. Equal keys keep the C++ merge's right half."""
    if len(ids) <= 1:
        return
    left_buf = [0] * len(ids)
    right_buf = [0] * len(ids)

    def sort(left: int, right: int) -> None:
        if left >= right:
            return
        mid = left + (right - left) // 2
        sort(left, mid)
        sort(mid + 1, right)
        n_left = mid - left + 1
        n_right = right - mid
        for offset in range(n_left):
            left_buf[offset] = ids[left + offset]
        for offset in range(n_right):
            right_buf[offset] = ids[mid + 1 + offset]
        i = j = 0
        cursor = left
        while i < n_left and j < n_right:
            if earliest[left_buf[i]] > earliest[right_buf[j]]:
                ids[cursor] = left_buf[i]
                i += 1
            else:
                ids[cursor] = right_buf[j]
                j += 1
            cursor += 1
        while i < n_left:
            ids[cursor] = left_buf[i]
            i += 1
            cursor += 1
        while j < n_right:
            ids[cursor] = right_buf[j]
            j += 1
            cursor += 1

    sort(0, len(ids) - 1)


def ubalb(stacks: Sequence[Sequence[int]], max_tiers: int, n: int, min_item: int) -> int:
    """Extra relocations beyond the containers already known to be blocking."""
    width = len(stacks)
    height = max_tiers
    yard = [[-1] * height for _ in range(width)]
    earliest = [n + 1] * width
    free = [height] * width
    location = [[-1, -1] for _ in range(n + 1)]
    for index, stack in enumerate(stacks):
        for slot, item in enumerate(stack):
            yard[index][slot] = item
            free[index] -= 1
            location[item][0] = index
            location[item][1] = slot
            if item < earliest[index]:
                earliest[index] = item

    bound = 0
    max_earliest = n + 1
    empty_stacks = 0
    ranked = False
    heavy = 0
    usable = 0
    usable_ids = [0] * width
    slack = free[:]

    while min_item <= n and empty_stacks < height:
        if location[min_item][0] != -1:
            stack, slot = location[min_item]
            free_before = free[stack]
            phi: List[int] = []
            top = height - free_before - 1
            for level in range(top, slot, -1):
                item = yard[stack][level]
                if item < max_earliest:
                    phi.append(item)
                else:
                    bound += 1
                location[item][0] = -1
                yard[stack][level] = -1
                free[stack] += 1
            n_blocks = len(phi)
            if n_blocks > 0 and empty_stacks < n_blocks:
                phi.sort(reverse=True)
                if not ranked:
                    max_earliest = -1
                    usable = 0
                    empty_stacks = 0
                    for index in range(width):
                        has_room = (index == stack and free_before > 0) or (index != stack and free[index] > 0)
                        if has_room:
                            usable_ids[usable] = index
                            usable += 1
                            if index != stack and earliest[index] > max_earliest:
                                max_earliest = earliest[index]
                                heavy = index
                        if index != stack and free[index] == height:
                            empty_stacks += 1
                        slack[index] = free[index]
                    ranked = True
                if free[heavy] < n_blocks or phi[0] > max_earliest:
                    ids = usable_ids[:usable]
                    _sort_by_earliest(ids, earliest)
                    usable_ids[:usable] = ids
                    cursor = 0
                    if usable_ids[cursor] == stack:
                        cursor += 1
                    for offset, item in enumerate(phi):
                        if item <= earliest[usable_ids[cursor]]:
                            slack[usable_ids[cursor]] -= 1
                            if slack[usable_ids[cursor]] == 0 and offset != n_blocks - 1:
                                cursor += 1
                                if usable_ids[cursor] == stack:
                                    cursor += 1
                        else:
                            bound += 1
                    for index in range(cursor + 1):
                        slack[usable_ids[index]] = free[usable_ids[index]]
            location[yard[stack][slot]][0] = -1
            yard[stack][slot] = -1
            free[stack] += 1
            if free[stack] == height:
                empty_stacks += 1
            earliest[stack] = n + 1
            for level in range(slot):
                if yard[stack][level] != -1 and yard[stack][level] < earliest[stack]:
                    earliest[stack] = yard[stack][level]
            if earliest[stack] > max_earliest:
                max_earliest = earliest[stack]
                heavy = stack
            if ranked and free_before == 0:
                usable_ids[usable] = stack
                usable += 1
            slack[stack] = free[stack]
        min_item += 1
    return bound
