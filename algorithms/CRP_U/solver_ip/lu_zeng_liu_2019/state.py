"""
Self-contained BRP state for Lu, Zeng & Liu (2019).

Manages the incremental stack representation needed by:
  - LB4 (virtual-layer derivation)
  - IS* fast heuristics (greedy completion)
  - BRP-m3 solution extraction

Items are 1-indexed (priority 1 = retrieved first).
Stacks are 0-indexed.
"""

from __future__ import annotations

from typing import List, Optional, Tuple


class BRPState:
    """
    Mutable BRP state.

    stacks[s]        : list of items bottom → top (int, 1-indexed priorities)
    low[s]           : minimum item in stack s  (n+1 if empty)
    must_move[i]     : True iff item i is badly placed (above a lower-priority item)
    lb1              : count of badly-placed items (= LB1), maintained incrementally
    next_item        : next item to be retrieved (starts at 1)
    n_relocations    : number of relocation moves performed so far
    ops              : full operation log as (from_s, to_s); retrieval = (s, s)
    """

    __slots__ = (
        "W", "n", "H",
        "stacks", "height", "low",
        "stack_for_item",
        "must_move",
        "lb1",
        "next_item",
        "n_relocations",
        "ops",
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
        self.stack_for_item: List[int] = [0] * (n + 2)
        self.must_move: List[bool] = [False] * (n + 2)
        self.lb1: int = 0
        self.next_item: int = 1
        self.n_relocations: int = 0
        self.ops: List[Tuple[int, int]] = []
        self.last_relocated_to: int = -1

        # Initialise incremental metadata
        for s in range(W):
            running_min = n + 1
            for item in self.stacks[s]:
                self.stack_for_item[item] = s
                if item < running_min:
                    running_min = item
                    self.low[s] = running_min
                else:
                    self.must_move[item] = True
                    self.lb1 += 1

    # ------------------------------------------------------------------ #
    # Accessors                                                            #
    # ------------------------------------------------------------------ #

    def top(self, s: int) -> int:
        return self.stacks[s][-1]

    def is_empty(self) -> bool:
        return self.next_item > self.n

    # ------------------------------------------------------------------ #
    # Operations                                                           #
    # ------------------------------------------------------------------ #

    def relocate(self, from_s: int, to_s: int) -> None:
        item = self.stacks[from_s][-1]

        # --- remove from from_s ---
        self.stacks[from_s].pop()
        self.height[from_s] -= 1
        if item == self.low[from_s]:
            self.low[from_s] = (
                min(self.stacks[from_s]) if self.stacks[from_s] else self.n + 1
            )
        elif self.low[from_s] < item:
            self.lb1 -= 1
            self.must_move[item] = False

        # --- add to to_s ---
        self.stacks[to_s].append(item)
        self.height[to_s] += 1
        self.stack_for_item[item] = to_s
        if item < self.low[to_s]:
            self.low[to_s] = item
            self.must_move[item] = False
        else:
            self.lb1 += 1
            self.must_move[item] = True

        self.n_relocations += 1
        self.last_relocated_to = to_s
        self.ops.append((from_s, to_s))

    def retrieve_next(self) -> bool:
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
            self.must_move[item] = False

        self.ops.append((s, s))
        self.next_item += 1
        return True

    def auto_retrieve(self) -> None:
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
        obj.must_move = list(self.must_move)
        obj.lb1 = self.lb1
        obj.next_item = self.next_item
        obj.n_relocations = self.n_relocations
        obj.ops = list(self.ops)
        obj.last_relocated_to = self.last_relocated_to
        return obj

    # ------------------------------------------------------------------ #
    # Factory                                                              #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_yard(cls, yard, all_keys: list, H_max: int) -> "BRPState":
        """Build from a platform Yard object."""
        stacks = [
            [c.priority for c in (yard.stacks[k].containers if k in yard.stacks else [])]
            for k in all_keys
        ]
        n = sum(len(s) for s in stacks)
        return cls(stacks, n, H_max, len(all_keys))

    # ------------------------------------------------------------------ #
    # Helpers for LB4                                                      #
    # ------------------------------------------------------------------ #

    def badly_placed_count(self) -> int:
        """LB1 = number of badly placed items."""
        return self.lb1

    def stacks_snapshot(self) -> List[List[int]]:
        """Snapshot of stacks (copies, safe for LB4 exploration)."""
        return [list(s) for s in self.stacks]

    def heights_snapshot(self) -> List[int]:
        return list(self.height)

    def low_snapshot(self) -> List[int]:
        return list(self.low)
