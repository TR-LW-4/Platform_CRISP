"""
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

from typing import Optional


def choose_destination(stacks, max_tiers) -> Optional[int]:
    """Destination of the current top blocker by stack-minimum difference."""
    present = [item for stack in stacks for item in stack]
    if not present:
        return None
    largest = max(present)
    target = min(present)
    source = next(index for index, stack in enumerate(stacks) if target in stack)
    blocker = stacks[source][-1]
    if blocker == target:
        return None

    good = []
    bad = []
    for destination, stack in enumerate(stacks):
        if destination == source or len(stack) >= max_tiers:
            continue
        earliest = min(stack) if stack else largest + 1
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
