"""
Movement plan primitives for batch planning algorithms (e.g. Lee & Lee 2010).

A plan is an ordered list of Movement objects, each representing the
Lee & Lee (c:a,b) triplet: container c moved from position a to position b.
`to_pos = None` denotes retrieval to the waiting truck (outside the yard).

Public API
----------
Movement         – single crane movement (container_id, from_pos, to_pos)
RelocationPlan   – ordered list of Movements with helpers
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ================================================================ #
#  Data structures                                                   #
# ================================================================ #

@dataclass
class Movement:
    """
    One crane movement: pick container_id from from_pos, place at to_pos.
    to_pos = None  →  retrieved to truck (removed from yard).
    """
    container_id: int
    from_pos:     Tuple[int, int]            # (bay, row)
    to_pos:       Optional[Tuple[int, int]]  # None = OUT (truck)
    from_tier:    Optional[int] = None        # 1-indexed tier before pickup
    to_tier:      Optional[int] = None        # 1-indexed tier after placement

    @property
    def is_retrieval(self) -> bool:
        return self.to_pos is None

    def __repr__(self) -> str:
        dst = "OUT" if self.to_pos is None else str(self.to_pos)
        return f"Move(c={self.container_id}: {self.from_pos}→{dst})"


@dataclass
class RelocationPlan:
    """Ordered sequence of crane movements."""
    movements: List[Movement] = field(default_factory=list)

    # ---------------------------------------------------------------- #
    # Mutation                                                           #
    # ---------------------------------------------------------------- #

    def add(self, m: Movement) -> None:
        self.movements.append(m)

    def clone(self) -> "RelocationPlan":
        return RelocationPlan(movements=[copy.copy(m) for m in self.movements])

    # ---------------------------------------------------------------- #
    # Query                                                              #
    # ---------------------------------------------------------------- #

    def num_moves(self) -> int:
        return len(self.movements)

    def num_relocations(self) -> int:
        """Non-retrieval moves (intermediate repositioning moves)."""
        return sum(1 for m in self.movements if not m.is_retrieval)

    def num_retrievals(self) -> int:
        return sum(1 for m in self.movements if m.is_retrieval)

    def moves_of(self, container_id: int) -> List[Tuple[int, Movement]]:
        """Return [(step_index, move), ...] for all moves of container_id, in order."""
        return [(i, m) for i, m in enumerate(self.movements)
                if m.container_id == container_id]

    def type_b_container_ids(self) -> List[int]:
        """
        Container IDs that appear in at least one non-retrieval move
        (Lee & Lee 'Type B': relocated at least once before retrieval).
        """
        reloc_ids = {m.container_id for m in self.movements if not m.is_retrieval}
        return sorted(reloc_ids)

    def __repr__(self) -> str:
        return (f"RelocationPlan({self.num_moves()} moves, "
                f"{self.num_relocations()} relocs, "
                f"{self.num_retrievals()} retrievals)")
