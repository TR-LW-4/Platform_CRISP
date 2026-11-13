"""
------------------------------- Reference --------------------------------
C. Expósito-Izquierdo, B. Melián-Batista, J.M. Moreno-Vega,
"A domain-specific knowledge-based heuristic for the Blocks Relocation Problem",
Applied Soft Computing 14 (2014) 1–20.
------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section.
--------------------------------------------------------------------------
"""

from typing import Optional, Sequence, Tuple


def choose_move(stacks: Sequence[Sequence[int]], max_tiers: int) -> Optional[Tuple[int, int]]:
    """Return ``(source, destination)`` for the blocker above the current target."""
    priorities = [priority for stack in stacks for priority in stack]
    if not priorities:
        return None
    target = min(priorities)
    source = next(index for index, stack in enumerate(stacks) if target in stack)
    if stacks[source][-1] == target:
        return None
    blocker = stacks[source][-1]
    candidates = [
        index for index, stack in enumerate(stacks)
        if index != source and len(stack) < max_tiers
    ]
    if not candidates:
        return None

    def score(index: int) -> Tuple[int, float, int]:
        stack = stacks[index]
        blocked = sum(1 for priority in stack if priority < blocker)
        earliest = min(stack) if stack else float("inf")
        return blocked, -earliest, index

    return source, min(candidates, key=score)
