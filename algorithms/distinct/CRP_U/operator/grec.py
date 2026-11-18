"""
Gre-C relocation rule for CRP-U
------------------------------- Reference --------------------------------
R. Jovanović, M. Tuba, S. Voß,
"An efficient ant colony optimization algorithm for the blocks relocation
 problem",
European Journal of Operational Research 274 (2019) 78–90.
------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section.
--------------------------------------------------------------------------
"""

from typing import List, Optional, Tuple


def choose_move(stacks, max_tiers) -> Optional[Tuple[int, int]]:
    options = candidates(stacks, max_tiers)
    if not options:
        return None
    source, destination, _ = min(options, key=lambda option: option[2])
    return source, destination


def candidates(stacks, max_tiers, extended=True) -> List[Tuple[int, int, int]]:
    n = max((max(stack) for stack in stacks if stack), default=0)
    if n == 0:
        return []
    earliest = [min(stack) if stack else n + 1 for stack in stacks]
    target = min(earliest)
    home = earliest.index(target)
    if stacks[home][-1] == target:
        return []
    blocker = stacks[home][-1]
    open_stacks = [index for index, stack in enumerate(stacks) if len(stack) < max_tiers]
    bad = {source for source, stack in enumerate(stacks)
           if source != home and stack and stack[-1] > earliest[source]}
    unblock = [(home, destination) for destination in open_stacks if destination != home]
    improve = [(source, destination) for source in sorted(bad) for destination in open_stacks
               if destination != source and earliest[destination] > stacks[source][-1]]
    moves = unblock + improve
    blocker_fits = any(earliest[destination] > blocker for _, destination in unblock)
    # Tw needs max_n over Tn; with Tn empty no well-located relocation is beneficial.
    if extended and not improve and not blocker_fits and bad:
        largest = max(stacks[source][-1] for source in bad)
        for source, stack in enumerate(stacks):
            if source != home and source not in bad and stack and _below(stack, n) > largest:
                moves += [(source, destination) for destination in open_stacks if destination != source]
    return [(source, destination, _dife(stacks, earliest, n, source, destination))
            for source, destination in moves]


def _below(stack, n):
    return min(stack[:-1]) if len(stack) > 1 else n + 1


def _dife(stacks, earliest, n, source, destination):
    container = stacks[source][-1]
    earliest_destination = earliest[destination]
    value = (earliest_destination - container if earliest_destination > container
             else 2 * n + 1 - earliest_destination)
    if container == earliest[source]:
        value += n - (_below(stacks[source], n) - container)
    return value
