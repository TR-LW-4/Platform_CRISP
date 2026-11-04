
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .container import Container

class Stack:

    def __init__(self, bay: int, row: int, max_tiers: int):
        self.bay       = bay
        self.row       = row
        self.max_tiers = max_tiers
        self.containers: List[Container] = []

    @property
    def height(self) -> int:
        return len(self.containers)

    @property
    def is_full(self) -> bool:
        return len(self.containers) >= self.max_tiers

    @property
    def is_empty(self) -> bool:
        return len(self.containers) == 0

    @property
    def top(self) -> Optional[Container]:
        return self.containers[-1] if self.containers else None

    @property
    def available_tiers(self) -> int:
        return self.max_tiers - self.height

    def push(self, container: Container) -> None:
        if self.is_full:
            raise ValueError(
                f"Stack ({self.bay},{self.row}) is full "
                f"(max_tiers={self.max_tiers})"
            )
        self.containers.append(container)

    def pop(self) -> Container:
        if self.is_empty:
            raise ValueError(
                f"Stack ({self.bay},{self.row}) is empty"
            )
        return self.containers.pop()

    def blockers_above(self, container: Container) -> List[Container]:
        """Return containers above *container*, ordered bottom-to-top."""
        try:
            idx = self.containers.index(container)
            return list(self.containers[idx + 1:])
        except ValueError:
            return []

    def tier_of(self, container: Container) -> int:
        """Return 1-indexed tier of *container*, or -1 if not present."""
        try:
            return self.containers.index(container) + 1
        except ValueError:
            return -1

    def contains(self, container: Container) -> bool:
        return container in self.containers

    def snapshot(self) -> List[int]:
        """Container IDs from bottom to top (0 = empty slot not included)."""
        return [c.id for c in self.containers]

    def group_snapshot(self) -> List[int]:
        """Container groups from bottom to top."""
        return [c.group for c in self.containers]





@dataclass
class Move:
    kind:         str            # "relocate" | "retrieve" | "load"
    container_id: int
    src:          Tuple[int, int]           # (bay, row)
    dst:          Optional[Tuple[int, int]] = None  # None for retrieval
    from_tier:    Optional[int] = None
    to_tier:      Optional[int] = None


# ================================================================ #
#  Yard                                                              #
# ================================================================ #

class Yard:

    def __init__(
        self,
        num_bays:  int,
        num_rows:  int,
        max_tiers: int,
    ):
        self.num_bays  = num_bays
        self.num_rows  = num_rows
        self.max_tiers = max_tiers

        self.stacks: Dict[Tuple[int, int], Stack] = {}
        for b in range(1, num_bays + 1):
            for r in range(1, num_rows + 1):
                self.stacks[(b, r)] = Stack(b, r, max_tiers)

        self.total_relocations: int = 0
        self.total_retrievals:  int = 0
        self.move_history: List[Move] = []


    def place(self, bay: int, row: int, container: Container) -> None:
        """Place a container (used during episode reset, not counted as move)."""
        self.stacks[(bay, row)].push(container)

    def reset_stats(self) -> None:
        self.total_relocations = 0
        self.total_retrievals  = 0
        self.move_history.clear()

    def clear(self) -> None:
        for stack in self.stacks.values():
            stack.containers.clear()
        self.reset_stats()

