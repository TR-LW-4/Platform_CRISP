"""
------------------------------- Reference --------------------------------
L. Tang, W. Jiang, J. Liu, Y. Dong,
"Research into container reshuffling and stacking problems in container
 terminal yards",
IIE Transactions 47 (2015) 751–766.
------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section.
--------------------------------------------------------------------------
"""

def choose_destination(stacks, max_tiers, rule="H2"):
    priorities = [p for stack in stacks for p in stack]
    if not priorities:
        return None

    target = min(priorities)
    src = next(i for i, stack in enumerate(stacks) if target in stack)
    blocker = stacks[src][-1]
    if blocker == target:
        return None

    candidates = [
        i for i, stack in enumerate(stacks)
        if i != src and len(stack) < max_tiers
    ]
    if not candidates:
        return None

    initial_count = max(priorities)

    def earliest(i):
        stack = stacks[i]
        return min(stack) if stack else initial_count + 1

    def reshuffle_index(i):
        return sum(p < blocker for p in stacks[i])

    def blocking_index(i):
        placed = [*stacks[i], blocker]
        urgent_at = placed.index(min(placed))
        return len(placed) - 1 - urgent_at

    good = [i for i in candidates if earliest(i) > blocker]
    if good:
        return min(good, key=lambda i: (earliest(i), i))

    if rule == "H1":
        return min(candidates, key=lambda i: (reshuffle_index(i), -earliest(i), i))
    return min(candidates, key=lambda i: (blocking_index(i), -earliest(i), i))