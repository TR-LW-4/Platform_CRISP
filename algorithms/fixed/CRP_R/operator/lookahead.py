"""
------------------------------- Reference --------------------------------
M. Caserta, S. Schwarze, S. Voß,
"A New Binary Description of the Blocks Relocation Problem and Benefits
 in a Look Ahead Heuristic",
Evolutionary Computation in Combinatorial Optimization (EvoCOP 2009),
LNCS 5482, pp. 37–48, Springer, 2009.
------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section.
--------------------------------------------------------------------------
"""

def _min_max(stacks, max_tiers):
    priorities = [p for stack in stacks for p in stack]
    if not priorities:
        return None

    target = min(priorities)
    src = next(i for i, stack in enumerate(stacks) if target in stack)
    if stacks[src][-1] == target:
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


def _completion(stacks, max_tiers):
    state = [list(stack) for stack in stacks]
    count = 0
    while True:
        priorities = [p for stack in state for p in stack]
        if not priorities:
            return count
        target = min(priorities)
        src = next(i for i, stack in enumerate(state) if target in stack)
        if state[src][-1] == target:
            state[src].pop()
            continue
        dst = _min_max(state, max_tiers)
        if dst is None:
            return count
        state[dst].append(state[src].pop())
        count += 1


def choose_destination(stacks, max_tiers):
    priorities = [p for stack in stacks for p in stack]
    if not priorities:
        return None

    target = min(priorities)
    src = next(i for i, stack in enumerate(stacks) if target in stack)
    if stacks[src][-1] == target:
        return None

    candidates = [
        i for i, stack in enumerate(stacks)
        if i != src and len(stack) < max_tiers
    ]
    if not candidates:
        return None

    def score(dst):
        trial = [list(stack) for stack in stacks]
        trial[dst].append(trial[src].pop())
        return _completion(trial, max_tiers), dst

    return min(candidates, key=score)