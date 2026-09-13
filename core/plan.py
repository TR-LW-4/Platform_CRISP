"""
Movement plan primitives for batch planning algorithms (e.g. Lee & Lee 2010).

A plan is an ordered list of Movement objects, each representing the
Lee & Lee (c:a,b) triplet: container c moved from position a to position b.
`to_pos = None` denotes retrieval to the waiting truck (outside the yard).

Public API
----------
Movement         – single crane movement (container_id, from_pos, to_pos)
RelocationPlan   – ordered list of Movements with helpers
SimResult        – outcome of plan simulation
simulate_plan()  – execute a plan on a deep copy of a Yard, detect conflicts
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


# ================================================================ #
#  Data structures                                                   #
# ================================================================ #

class ConflictType(Enum):
    STACKING = "stacking"   # target container not on top of source stack
    OVERHIGH = "overhigh"   # destination stack would exceed max_tiers


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


# ================================================================ #
#  Simulation                                                        #
# ================================================================ #

@dataclass
class SimResult:
    feasible:        bool
    num_moves:       int
    num_relocations: int
    num_retrievals:  int
    conflicts: List[Tuple[int, ConflictType, str]]  # (step_idx, type, message)


def simulate_plan(
    yard,           # core.yard.Yard (forward ref to avoid circular import)
    plan:           RelocationPlan,
    max_tiers:      int,
) -> SimResult:
    """
    Execute *plan* on a **deep copy** of *yard* and return SimResult.

    Simulation continues past conflicts (to collect all errors), treating
    conflicting moves as no-ops so subsequent moves can still be checked.

    Parameters
    ----------
    yard      : initial Yard state (not mutated)
    plan      : the movement sequence to simulate
    max_tiers : maximum allowed stack height
    """
    from .yard import Yard  # local import avoids circular dependency

    sim: Yard = copy.deepcopy(yard)
    conflicts: List[Tuple[int, ConflictType, str]] = []
    n_reloc    = 0
    n_retrieve = 0

    for step, move in enumerate(plan.movements):
        src_stack = sim.stacks.get(move.from_pos)

        # ── stacking conflict: container not on top ───────────────── #
        stacking_ok = True
        if src_stack is None or src_stack.is_empty:
            conflicts.append((
                step, ConflictType.STACKING,
                f"Source {move.from_pos} is empty",
            ))
            stacking_ok = False
        elif src_stack.top.id != move.container_id:
            conflicts.append((
                step, ConflictType.STACKING,
                f"C{move.container_id} not on top of {move.from_pos} "
                f"(top is C{src_stack.top.id})",
            ))
            stacking_ok = False

        if not stacking_ok:
            continue  # skip this move, keep simulating

        container = src_stack.pop()

        if move.to_pos is None:
            # Retrieval – container leaves the yard
            n_retrieve += 1
        else:
            # Relocation – check over-height
            dst_stack = sim.stacks.get(move.to_pos)
            if dst_stack is None:
                conflicts.append((
                    step, ConflictType.OVERHIGH,
                    f"Destination {move.to_pos} not found",
                ))
                src_stack.push(container)  # undo pop, keep simulating
            elif dst_stack.height >= max_tiers:
                conflicts.append((
                    step, ConflictType.OVERHIGH,
                    f"Destination {move.to_pos} full ({dst_stack.height}/{max_tiers})",
                ))
                dst_stack.push(container)  # accept anyway to continue simulation
                n_reloc += 1
            else:
                dst_stack.push(container)
                n_reloc += 1

    return SimResult(
        feasible        = (len(conflicts) == 0),
        num_moves       = len(plan.movements),
        num_relocations = n_reloc,
        num_retrievals  = n_retrieve,
        conflicts       = conflicts,
    )
