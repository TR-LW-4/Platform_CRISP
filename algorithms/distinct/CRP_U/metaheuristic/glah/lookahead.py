"""
Look-ahead tree search for GLAH

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
corresponding original algorithm paper.
--------------------------------------------------------------------------
"""

import math

from algorithms.distinct.CRP_U.LB.lb2_u import lb2_u
from algorithms.distinct.CRP_U.operator import jzw

from .condense import condense

# Successors kept per category: FTBG, NTBG, FTBB, NTBB, GG, GB.
WIDTHS = (5, 5, 3, 3, 1, 1)


class Lookahead:

    def __init__(self, stacks, max_tiers, depth):
        self.stacks = [list(stack) for stack in stacks]
        self.max_tiers = max_tiers
        self.depth = depth
        # Capacity of an empty stack; the GG and GB scores depend on it.
        self.floor = max(500, max((p for stack in stacks for p in stack), default=0) + 1)
        self.path = []
        self.relocations = 0
        self.best = None
        self.best_count = math.inf
        self.visited = set()
        self.rollouts = {}

    def most_promising(self, state):
        self.visited = set()
        return self._branch(state, 0)[0]

    def rollout(self, state):
        """Steps of the JZW rule from ``state`` until the bay is empty, and
        their relocation count, or ``(None, None)`` if it gets stuck."""
        key = tuple(map(tuple, state))
        if key not in self.rollouts:
            state = [list(stack) for stack in state]
            steps = []
            retrieve(state, steps)
            while any(state):
                move = jzw.choose_move(state, self.max_tiers)
                if move is None:
                    steps = None
                    break
                source, destination = move
                state[destination].append(state[source].pop())
                steps.append(move)
                retrieve(state, steps)
            count = None if steps is None else sum(d is not None for _, d in steps)
            self.rollouts[key] = (steps, count)
        return self.rollouts[key]

    def update(self, tail):
        solution = condense(self.stacks, self.path + tail)
        count = sum(d is not None for _, d in solution)
        if count < self.best_count:
            self.best, self.best_count = solution, count

    def _branch(self, state, depth):
        key = (tuple(map(tuple, state)), depth)
        if key in self.visited:
            return None, None
        self.visited.add(key)
        if lb2_u(state, self.max_tiers) + self.relocations >= self.best_count:
            return None, None

        steps, count = self.rollout(state)
        if steps is not None:
            self.update(list(steps))
        if not any(state) or depth >= self.depth:
            return None, None if steps is None else self.relocations + count

        moves = self._candidates(state)
        if steps:
            advice = steps[0]
            moves = [advice] + [move for move in moves if move != advice]
        best_move, best_value = None, None
        for source, destination in moves:
            child = [list(stack) for stack in state]
            child[destination].append(child[source].pop())
            mark = len(self.path)
            self.path.append((source, destination))
            self.relocations += 1
            retrieve(child, self.path)
            _, value = self._branch(child, depth + 1)
            del self.path[mark:]
            self.relocations -= 1
            if value is not None and (best_value is None or value < best_value):
                best_move, best_value = (source, destination), value
        return best_move, best_value

    def _candidates(self, state):
        target = min(min(stack) for stack in state if stack)
        caps = [min(stack) if stack else self.floor for stack in state]
        groups = [[] for _ in WIDTHS]
        for source, stack in enumerate(state):
            if not stack:
                continue
            top = stack[-1]
            below = min(stack[:-1]) if len(stack) > 1 else self.floor
            empty_seen = False
            for destination, other in enumerate(state):
                if destination == source or len(other) == self.max_tiers:
                    continue
                if not other:
                    if empty_seen:
                        continue
                    empty_seen = True
                good = top <= caps[destination]
                if top == caps[source]:
                    group = 4 if good else 5
                    score = caps[destination] - below if good else -caps[destination] - below
                else:
                    group = (0 if caps[source] == target else 1) + (0 if good else 2)
                    score = caps[destination] - top if good else top - caps[destination]
                groups[group].append((score, (source, destination)))
        moves = []
        for group, width in zip(groups, WIDTHS):
            group.sort(key=lambda item: item[0])
            moves.extend(move for _, move in group[:width])
        return moves


def retrieve(state, steps):
    while any(state):
        target = min(min(stack) for stack in state if stack)
        source = next((s for s, stack in enumerate(state) if stack and stack[-1] == target), None)
        if source is None:
            return
        state[source].pop()
        steps.append((source, None))
