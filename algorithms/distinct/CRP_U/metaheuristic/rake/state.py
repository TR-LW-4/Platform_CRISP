"""
Bay state shared by SmSEQ, RakeSearch, Pilot, and DFBBU, and the
conversion of its operation log into moves.

------------------------------- Reference --------------------------------
F. Tricoire, J. Scagnetti, A. Beham,
"New insights on the block relocation problem",
Computers & Operations Research 89 (2018) 127–139.
------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section.
--------------------------------------------------------------------------
"""


class State:
    """Stacks are listed bottom to top and priorities must be consecutive
    integers. `ops` logs every operation as (source, destination); a
    retrieval is logged as (source, source)."""

    __slots__ = ("stacks", "max_tiers", "n", "next", "where", "heights",
                 "ops", "relocations", "last_to")

    def __init__(self, stacks, max_tiers):
        self.stacks = [list(stack) for stack in stacks]
        self.max_tiers = max_tiers
        priorities = [priority for stack in self.stacks for priority in stack]
        self.n = max(priorities, default=0)
        self.next = min(priorities, default=1)
        self.where = {priority: index for index, stack in enumerate(self.stacks) for priority in stack}
        self.heights = [len(stack) for stack in self.stacks]
        self.ops = []
        self.relocations = 0
        self.last_to = -1

    def copy(self):
        other = State.__new__(State)
        other.stacks = [list(stack) for stack in self.stacks]
        other.max_tiers = self.max_tiers
        other.n = self.n
        other.next = self.next
        other.where = dict(self.where)
        other.heights = self.heights
        other.ops = list(self.ops)
        other.relocations = self.relocations
        other.last_to = self.last_to
        return other

    def empty(self):
        return not self.where

    def lows(self):
        return [min(stack) if stack else self.n + 1 for stack in self.stacks]

    def retrieve_all(self):
        count = 0
        while self.where:
            source = self.where[self.next]
            if self.stacks[source][-1] != self.next:
                break
            self.stacks[source].pop()
            del self.where[self.next]
            self.ops.append((source, source))
            self.next += 1
            self.last_to = -1
            count += 1
        return count

    def relocate(self, source, destination):
        item = self.stacks[source].pop()
        self.stacks[destination].append(item)
        self.where[item] = destination
        self.ops.append((source, destination))
        self.relocations += 1
        self.last_to = destination

    def undo(self):
        source, destination = self.ops.pop()
        if source == destination:
            self.next -= 1
            self.stacks[source].append(self.next)
            self.where[self.next] = source
        else:
            item = self.stacks[destination].pop()
            self.stacks[source].append(item)
            self.where[item] = source
            self.relocations -= 1


class Move:
    def __init__(self, container, source, destination):
        self.container = container
        self.source = source
        self.destination = destination


def as_moves(stacks, ops):
    state = [list(stack) for stack in stacks]
    moves = []
    for source, destination in ops:
        tier = len(state[source])
        container = state[source].pop()
        if source == destination:
            moves.append(Move(container, (source, tier), None))
        else:
            state[destination].append(container)
            moves.append(Move(container, (source, tier), (destination, len(state[destination]))))
    return moves
