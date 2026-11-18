"""
Tree search with compound moves used by CMTS (Figs. 5-7 of the reference).

A move is (source, destination); destination None is a remove.

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

import time

from algorithms.distinct.CRP_U.LB.lb2_u import lb2_u
from algorithms.distinct.CRP_U.operator.fb import badly_placed_top


class Search:

    def __init__(self, max_tiers, time_limit, n_succ, cm_stop, max_flg_bb, max_gg):
        self.max_tiers = max_tiers
        self.deadline = time.monotonic() + time_limit
        self.n_succ = n_succ
        self.cm_stop = cm_stop
        self.max_flg_bb = max_flg_bb
        self.max_gg = max_gg
        self.groups = 0
        self.target = 0
        self.best = []
        self.prefix = []

    def run(self, stacks, initial):
        self.groups = max((max(stack) for stack in stacks if stack), default=0)
        self.target = self.lower_bound(stacks)
        self.best = list(initial)
        self.perform(stacks, [])
        return self.best

    def timed_out(self):
        return time.monotonic() > self.deadline

    def lower_bound(self, stacks):
        return sum(len(stack) for stack in stacks) + lb2_u(stacks, self.max_tiers)

    def perform(self, stacks, solution):
        if len(self.best) <= self.target or self.timed_out():
            return
        if not any(stacks):
            if len(solution) < len(self.best):
                self.best = solution
            return
        self.prefix = solution
        found = []
        self.compound(stacks, [], len(self.best) - len(solution) - 1, 1, found)
        ranked = [(len(moves) + bound, -len(moves), -self.clean_supply(after), moves, after)
                  for moves, after, bound in found]
        ranked.sort(key=lambda item: item[:3])
        for *_, moves, after in ranked[:self.n_succ]:
            self.perform(after, solution + moves)

    def compound(self, stacks, work, max_moves, stop_value, found):
        productive = self.productive(stacks)
        for move in productive:
            if self.timed_out():
                return
            after = apply(stacks, [move])
            if not any(after):
                if len(self.prefix) + len(work) + 1 < len(self.best):
                    self.best = self.prefix + work + [move]
                continue
            bound = self.lower_bound(after)
            if bound > max_moves:
                continue
            if stop_value < self.cm_stop:
                self.compound(after, work + [move], max_moves - 1, stop_value * len(productive), found)
            else:
                found.append((work + [move], after, bound))

    def productive(self, stacks):
        lowest = min(priority for stack in stacks for priority in stack)
        removes = [(source, None) for source, stack in enumerate(stacks) if stack and stack[-1] == lowest]
        if removes:
            return removes
        open_stacks = [index for index, stack in enumerate(stacks) if len(stack) < self.max_tiers]
        blockers = [source for source, stack in enumerate(stacks) if badly_placed_top(stack)]
        good = [(source, destination) for source in blockers for destination in open_stacks
                if destination != source and fits(stacks[destination], stacks[source][-1])]
        if good:
            return good
        freeing = sorted((stacks[source][-1] - min(stacks[destination]), source, destination)
                         for source in blockers if lowest in stacks[source] for destination in open_stacks
                         if destination != source and not fits(stacks[destination], stacks[source][-1]))
        shifts = sorted((_gain(stacks[source], stacks[destination]), source, destination)
                        for source, stack in enumerate(stacks) if stack and not badly_placed_top(stack)
                        for destination in open_stacks
                        if destination != source and fits(stacks[destination], stack[-1]))
        return ([move[1:] for move in freeing[:self.max_flg_bb]]
                + [move[1:] for move in shifts[:self.max_gg]])

    def clean_supply(self, stacks):
        supply = 0
        for stack in stacks:
            if not stack:
                supply += self.groups * self.max_tiers
            elif all(stack[index] <= min(stack[:index]) for index in range(1, len(stack))):
                supply += stack[-1] * (self.max_tiers - len(stack))
        return supply


def fits(stack, item):
    return not stack or min(stack) >= item


def apply(stacks, moves):
    after = [list(stack) for stack in stacks]
    for source, destination in moves:
        item = after[source].pop()
        if destination is not None:
            after[destination].append(item)
    return after


def _gain(source, destination):
    # Differences are non-negative as in Fig. 4; the paper leaves them undefined
    # for a bottom item or an empty destination, where they count as zero.
    item = source[-1]
    before = source[-2] - item if len(source) > 1 else 0
    after = destination[-1] - item if destination else 0
    return before - after
