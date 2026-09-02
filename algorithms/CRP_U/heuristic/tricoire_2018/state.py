"""
Bay-state representation for TricoireHeuristic.

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
corresponding original algorithm paper.
--------------------------------------------------------------------------
"""

from __future__ import annotations
from typing import List, Tuple


class BRPState:
    """
    Mutable BRP state.

    Attributes
    ----------
    stacks[s] : list[int]
        Items in stack s from bottom (index 0) to top (index -1).
    low[s] : int
        min item in stack s  (n+1 if empty)  — maintained incrementally.
    lb1 : int
        Count of badly-placed items (LB1).  Maintained incrementally with the
        same rules as C++ LB_ in brpstate.cpp.
    ops : list of (from_s, to_s)
        Full operation log.  Retrieval is encoded as from_s == to_s.
    n_relocations : int
        Number of relocations (ops where from_s != to_s).
    last_relocated_to : int
        Stack index of the most recent relocation destination (-1 after
        retrieval or at start).
    """

    __slots__ = (
        "W", "n", "H",
        "stacks", "height", "low",
        "stack_for_item",
        "lb1",
        "next_item",
        "ops",
        "n_relocations",
        "last_relocated_to",
    )

    def __init__(
        self,
        stacks: List[List[int]],
        n: int,
        H: int,
        W: int,
    ) -> None:
        self.W = W
        self.n = n
        self.H = H
        self.stacks: List[List[int]] = [list(s) for s in stacks]
        self.height: List[int] = [len(s) for s in stacks]
        self.low: List[int] = [n + 1] * W
        self.stack_for_item: List[int] = [0] * (n + 2)   # index 1..n
        self.lb1: int = 0
        self.next_item: int = 1
        self.ops: List[Tuple[int, int]] = []
        self.n_relocations: int = 0
        self.last_relocated_to: int = -1

        # Initialise low, stack_for_item, lb1 by re-pushing bottom→top
        for s in range(W):
            running_low = n + 1
            for item in self.stacks[s]:       # bottom → top
                self.stack_for_item[item] = s
                if item < running_low:
                    running_low = item
                    self.low[s] = running_low
                else:
                    self.lb1 += 1             # badly placed

    # ------------------------------------------------------------------ #
    # Accessors                                                            #
    # ------------------------------------------------------------------ #

    def top(self, s: int) -> int:
        return self.stacks[s][-1]

    def is_empty(self) -> bool:
        return self.next_item > self.n

    # ------------------------------------------------------------------ #
    # Core operations                                                      #
    # ------------------------------------------------------------------ #

    def relocate(self, from_s: int, to_s: int) -> None:
        """Move top item of from_s to to_s.  Updates all metadata."""
        item = self.stacks[from_s][-1]

        # --- remove from from_s ---
        self.stacks[from_s].pop()
        self.height[from_s] -= 1
        if item == self.low[from_s]:
            # item WAS the minimum — recompute low (no LB change per C++ logic)
            self.low[from_s] = (
                min(self.stacks[from_s]) if self.stacks[from_s] else self.n + 1
            )
        elif self.low[from_s] < item:
            # item was badly placed (lb1 decreases)
            self.lb1 -= 1

        # --- add to to_s ---
        self.stacks[to_s].append(item)
        self.height[to_s] += 1
        self.stack_for_item[item] = to_s
        if item < self.low[to_s]:
            self.low[to_s] = item
        else:
            self.lb1 += 1                     # becomes badly placed

        self.n_relocations += 1
        self.last_relocated_to = to_s
        self.ops.append((from_s, to_s))

    def retrieve_next(self) -> bool:
        """
        Retrieve next_item if it is on top of its stack.
        Returns True on success.
        """
        if self.next_item > self.n:
            return False
        s = self.stack_for_item[self.next_item]
        if not self.stacks[s] or self.stacks[s][-1] != self.next_item:
            return False

        item = self.stacks[s].pop()
        self.height[s] -= 1
        self.stack_for_item[item] = -1
        self.last_relocated_to = -1
        if item == self.low[s]:
            self.low[s] = min(self.stacks[s]) if self.stacks[s] else self.n + 1
        else:
            self.lb1 -= 1

        self.ops.append((s, s))               # retrieval encoded as (s, s)
        self.next_item += 1
        return True

    def auto_retrieve(self) -> None:
        """Retrieve all items that are currently on top of their stacks."""
        while self.retrieve_next():
            pass

    # ------------------------------------------------------------------ #
    # Copy / dominance                                                     #
    # ------------------------------------------------------------------ #

    def copy(self) -> "BRPState":
        obj = object.__new__(BRPState)
        obj.W = self.W
        obj.n = self.n
        obj.H = self.H
        obj.stacks = [list(st) for st in self.stacks]
        obj.height = list(self.height)
        obj.low = list(self.low)
        obj.stack_for_item = list(self.stack_for_item)
        obj.lb1 = self.lb1
        obj.next_item = self.next_item
        obj.ops = list(self.ops)
        obj.n_relocations = self.n_relocations
        obj.last_relocated_to = self.last_relocated_to
        return obj

    def dominates(self, other: "BRPState") -> bool:
        """True iff same stack configuration with ≤ relocations."""
        return (
            self.stacks == other.stacks
            and self.n_relocations <= other.n_relocations
        )

    # ------------------------------------------------------------------ #
    # Factory                                                              #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_yard(cls, initial_yard, all_keys: list, H_max: int) -> "BRPState":
        """Build a BRPState from a platform Yard object."""
        n = sum(stk.height for stk in initial_yard.stacks.values())
        stacks = [
            [c.priority for c in (initial_yard.stacks[k].containers if k in initial_yard.stacks else [])]
            for k in all_keys
        ]
        return cls(stacks, n, H_max, len(all_keys))
