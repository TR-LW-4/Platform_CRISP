"""
OPT(n) operator used by OPTN: the dynamic program that reoptimizes the
relocations of one container while all other moves stay unchanged.

A step is (container, source, destination); destination is None for a
retrieval. A state (stack, tier) places container n in configuration t of
S^-n, with tiers counted from 1 at the bottom. A relocation of n in
configuration t happens before step t of S^-n.

------------------------------- Reference --------------------------------
D. Feillet, S.N. Parragh, F. Tricoire,
"A local-search based heuristic for the unrestricted block relocation problem",
Computers & Operations Research 108 (2019) 44–56.
------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section.
--------------------------------------------------------------------------
"""


def opt_n(n, steps, stacks, max_tiers, relocations):
    retrieval = next(index for index, (container, _, destination) in enumerate(steps)
                     if container == n and destination is None)
    others = [step for step in steps[:retrieval] if step[0] != n]
    start, height = _remove(n, stacks, others)
    last = len(others)
    low, high = _suffix(height)
    width = len(stacks)

    costs = {start: 0}
    parents = [{}]
    for t, (_, source, destination) in enumerate(others):
        following, parent = {}, {}
        parents.append(parent)
        for (stack, tier), cost in costs.items():
            successors = []
            if _feasible(height[t], max_tiers, source, destination, stack, tier):
                successors.append((stack, tier, cost))
            if tier == height[t][stack] + 1 and cost + 1 < relocations:
                for other in _targets(t, stack, others, width):
                    if other != stack and height[t][other] < max_tiers:
                        landing = height[t][other] + 1
                        if _feasible(height[t], max_tiers, source, destination, other, landing):
                            successors.append((other, landing, cost + 1))
            for other, landing, new_cost in successors:
                if new_cost >= relocations:
                    continue
                if new_cost == relocations - 1 and landing != height[last][other] + 1:
                    continue
                if new_cost < following.get((other, landing), relocations):
                    following[(other, landing)] = new_cost
                    parent[(other, landing)] = (stack, tier)
                    if (high[t + 1][other] < max_tiers and low[t + 1][other] >= landing - 1
                            and height[last][other] == landing - 1):
                        return _rebuild(n, steps, retrieval, others, parents, t + 1, (other, landing))
        costs = following
        if not costs:
            return None

    finals = [(cost, state) for state, cost in costs.items() if state[1] == height[last][state[0]] + 1]
    if not finals:
        return None
    cost, state = min(finals, key=lambda item: item[0])
    if cost >= relocations:
        return None
    return _rebuild(n, steps, retrieval, others, parents, last, state)


def _remove(n, stacks, others):
    heights = [len(stack) for stack in stacks]
    start = None
    for index, stack in enumerate(stacks):
        if n in stack:
            start = (index, stack.index(n) + 1)
            heights[index] -= 1
    height = [list(heights)]
    for _, source, destination in others:
        heights[source] -= 1
        if destination is not None:
            heights[destination] += 1
        height.append(list(heights))
    return start, height


def _suffix(height):
    low, high = [list(height[-1])], [list(height[-1])]
    for row in reversed(height[:-1]):
        low.append([min(a, b) for a, b in zip(row, low[-1])])
        high.append([max(a, b) for a, b in zip(row, high[-1])])
    return low[::-1], high[::-1]


def _feasible(heights, max_tiers, source, destination, stack, tier):
    top = heights[stack]
    if tier > top + 1 or top >= max_tiers:
        return False
    if source == stack and tier > top:
        return False
    if destination == stack and top + 1 >= max_tiers:
        return False
    return True


# Useless evaluations: only stacks changed by the previous step can offer new
# relocations, unless n has just been uncovered in the source of that step.
def _targets(t, stack, others, width):
    if t == 0 or others[t - 1][1] == stack:
        return range(width)
    _, source, destination = others[t - 1]
    return [index for index in (source, destination) if index is not None]


def _rebuild(n, steps, retrieval, others, parents, t, state):
    final = state[0]
    moves = {}
    while t > 0:
        previous = parents[t][state]
        if previous[0] != state[0]:
            moves[t - 1] = (n, previous[0], state[0])
        state = previous
        t -= 1
    rebuilt = []
    for index, step in enumerate(others):
        if index in moves:
            rebuilt.append(moves[index])
        rebuilt.append(step)
    rebuilt.append((n, final, None))
    return rebuilt + steps[retrieval + 1:]
