"""
Forster-Bortfeldt greedy relocation rule for CRP-U

Priority: BG relocation to a non-empty stack, BG relocation to an empty
stack, then FLG_BB relocation (Fig. 4 of the reference).

------------------------------- Reference --------------------------------
F. Forster, A. Bortfeldt,
"A tree search procedure for the container relocation problem",
Computers & Operations Research 39 (2012) 299–309.
------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section.
--------------------------------------------------------------------------
"""

from typing import Optional, Tuple


def choose_move(stacks, max_tiers) -> Optional[Tuple[int, int]]:
    present = [priority for stack in stacks for priority in stack]
    if not present:
        return None
    lowest = min(present)
    if any(stack and stack[-1] == lowest for stack in stacks):
        return None
    blockers = [source for source, stack in enumerate(stacks) if badly_placed_top(stack)]
    open_stacks = [index for index, stack in enumerate(stacks) if len(stack) < max_tiers]

    good = [(stacks[destination][-1] - stacks[source][-1], source, destination)
            for source in blockers for destination in open_stacks
            if destination != source and stacks[destination]
            and min(stacks[destination]) >= stacks[source][-1]]
    if not good:
        good = [(stacks[source][-1], source, destination)
                for source in blockers for destination in open_stacks if not stacks[destination]]
    if good:
        return min(good)[1:]

    freeing = [(min(stacks[destination]) - stacks[source][-1], source, destination)
               for source in blockers if lowest in stacks[source] for destination in open_stacks
               if destination != source and stacks[destination]
               and min(stacks[destination]) < stacks[source][-1]]
    return min(freeing)[1:] if freeing else None


def badly_placed_top(stack) -> bool:
    return len(stack) > 1 and min(stack[:-1]) < stack[-1]
