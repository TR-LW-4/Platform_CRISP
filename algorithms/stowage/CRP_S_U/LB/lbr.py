"""
LBr for the block relocation problem with a stowage plan.
Optimal value of the relaxation with infinitely many empty yard stacks.
Section 5.2. Each container is relocated at most once.

------------------------------- Reference --------------------------------
S. Tanaka, S. Voß,
"An exact algorithm for the block relocation problem with a stowage plan",
European Journal of Operational Research 279 (2019) 767–781.
------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section.
--------------------------------------------------------------------------
"""

from __future__ import annotations


def lower_bound(stacks, heights, cache=None):
    """Optimal value of the infinite-empty-stack relaxation.

    Containers are interned to small integers inside ``cache`` so that the
    same yard can be reused across search nodes. Vessel tiers in the
    generated instances are gap-free, so removing the current minimum tier
    advances that vessel stack by one.
    """
    if cache is None:
        cache = {}
    tables = cache.get("tables")
    if tables is None:
        tables = [{}, [], []]
        cache["tables"] = tables
    mapping, vessel_of, tier_of = tables
    heights = tuple(heights)

    def intern(container):
        found = mapping.get(container)
        if found is None:
            found = len(vessel_of)
            mapping[container] = found
            vessel_of.append(container[0])
            tier_of.append(container[1])
        return found

    def canon(pieces):
        loaded = list(heights)
        for piece in pieces:
            for container in piece:
                vessel = vessel_of[container]
                if tier_of[container] < loaded[vessel]:
                    loaded[vessel] = tier_of[container]
        changed = True
        while changed:
            changed = False
            for piece in pieces:
                if not piece:
                    continue
                container = piece[-1]
                vessel = vessel_of[container]
                if tier_of[container] != loaded[vessel]:
                    continue
                piece.pop()
                loaded[vessel] += 1
                changed = True
                break
        kept = [tuple(piece) for piece in pieces if piece]
        kept.sort()
        return tuple(kept)

    def rec(state):
        saved = cache.get(state)
        if saved is not None:
            return saved
        if not state:
            cache[state] = 0
            return 0
        loaded = list(heights)
        total = 0
        for stack in state:
            total += len(stack)
            for container in stack:
                vessel = vessel_of[container]
                if tier_of[container] < loaded[vessel]:
                    loaded[vessel] = tier_of[container]
        best = total
        for index, stack in enumerate(state):
            target = None
            for position in range(len(stack) - 1, -1, -1):
                container = stack[position]
                if tier_of[container] == loaded[vessel_of[container]]:
                    target = position
                    break
            if target is None or target == len(stack) - 1:
                continue
            pieces = []
            if target:
                pieces.append(list(stack[:target]))
            for other_index, other in enumerate(state):
                if other_index != index and other:
                    pieces.append(list(other))
            for position in range(target + 1, len(stack)):
                pieces.append([stack[position]])
            best = min(best, len(stack) - target - 1 + rec(canon(pieces)))
        cache[state] = best
        return best

    return rec(canon([[intern(container) for container in stack] for stack in stacks]))
