"""
------------------------------- Reference --------------------------------
M. Caserta, S. Schwarze, S. Voß,
"A mathematical formulation and complexity considerations for the
 blocks relocation problem",
European Journal of Operational Research 219 (2012) 96–104.
------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section.
--------------------------------------------------------------------------
"""

def choose_destination(stacks, max_tiers):
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

    good = [i for i in candidates if earliest(i) > blocker]
    if good:
        return min(good, key=lambda i: (earliest(i), i))
    return min(candidates, key=lambda i: (-earliest(i), i))