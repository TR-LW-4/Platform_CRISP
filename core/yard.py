"""
Core relocation engine.

Yard  ←  the physical container yard (bays × rows × tiers)
Stack ←  one column of stacked containers

This module is algorithm-agnostic and problem-agnostic.
All stowage problems build on top of these primitives.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .container import Container


# ================================================================ #
#  Stack                                                             #
# ================================================================ #

class Stack:
    """One column of containers at position (bay, row).

    Containers are stored bottom-to-top in `self.containers`.
    `tier` is 1-indexed (tier 1 = bottom).
    """

    def __init__(self, bay: int, row: int, max_tiers: int):
        self.bay       = bay
        self.row       = row
        self.max_tiers = max_tiers
        self.containers: List[Container] = []

    # ---------------------------------------------------------------- #
    # Properties                                                         #
    # ---------------------------------------------------------------- #

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

    # ---------------------------------------------------------------- #
    # Mutation                                                           #
    # ---------------------------------------------------------------- #

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

    # ---------------------------------------------------------------- #
    # Query                                                              #
    # ---------------------------------------------------------------- #

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

    def priority_snapshot(self) -> List[int]:
        """Container priorities from bottom to top."""
        return [c.priority for c in self.containers]

    # ---------------------------------------------------------------- #
    # Sorting quality                                                    #
    # ---------------------------------------------------------------- #

    def is_sorted_by_priority(self) -> bool:
        """True if containers are in non-increasing priority from bottom to top
        (i.e. larger-priority = later-retrieved containers are on top)."""
        for i in range(len(self.containers) - 1):
            if self.containers[i].priority < self.containers[i + 1].priority:
                return False
        return True

    def num_bad_overlaps(self) -> int:
        """Count (lower, upper) pairs where lower.priority < upper.priority."""
        count = 0
        for i in range(len(self.containers) - 1):
            if self.containers[i].priority < self.containers[i + 1].priority:
                count += 1
        return count

    def __repr__(self) -> str:
        ids = [str(c.id) for c in self.containers]
        return f"Stack({self.bay},{self.row})[{','.join(ids)}]"


# ================================================================ #
#  Move record                                                       #
# ================================================================ #

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
    """
    Physical container yard: num_bays × num_rows, each column up to max_tiers high.

    Core operations
    ---------------
    relocate(src, dst)          – move top of src stack to top of dst stack
    retrieve(container, dst)    – remove container counting its blockers
    place(bay, row, container)  – initialisation helper
    """

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

    # ---------------------------------------------------------------- #
    # Initialisation helpers                                             #
    # ---------------------------------------------------------------- #

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

    # ---------------------------------------------------------------- #
    # Core operations                                                    #
    # ---------------------------------------------------------------- #

    def relocate(
        self,
        src: Tuple[int, int],
        dst: Tuple[int, int],
    ) -> Container:
        """
        Move the top container from stack *src* to stack *dst*.

        Returns the moved container.
        Raises ValueError if src is empty or dst is full.
        """
        src_stack = self.stacks[src]
        dst_stack = self.stacks[dst]

        if src_stack.is_empty:
            raise ValueError(f"Source stack {src} is empty")
        if dst_stack.is_full:
            raise ValueError(f"Destination stack {dst} is full")

        from_tier = src_stack.height
        to_tier = dst_stack.height + 1
        container = src_stack.pop()
        dst_stack.push(container)

        self.total_relocations += 1
        self.move_history.append(
            Move(
                "relocate",
                container.id,
                src,
                dst,
                from_tier=from_tier,
                to_tier=to_tier,
            )
        )
        return container

    def retrieve(
        self,
        container: Container,
        relocation_policy: Optional[
            Callable[[Container, "Yard"], Tuple[int, int]]
        ] = None,
    ) -> Tuple[int, List[Move]]:
        """
        Remove *container* from the yard, relocating any blockers first.

        Parameters
        ----------
        container          : the container to retrieve
        relocation_policy  : callable(blocker, yard) → (bay, row)
                             If None, uses greedy min-height strategy.

        Returns
        -------
        (num_relocations, relocation_moves)
        """
        stack = self._find_stack(container)
        if stack is None:
            raise ValueError(f"Container {container.id} not found in yard")

        blockers = stack.blockers_above(container)  # bottom-to-top order
        reloc_moves: List[Move] = []

        # Relocate from top of stack downwards
        for blocker in reversed(blockers):
            if relocation_policy is not None:
                dst = relocation_policy(blocker, self)
            else:
                dst = self._greedy_relocation_dst(
                    (stack.bay, stack.row)
                )
            moved = self.relocate((stack.bay, stack.row), dst)
            reloc_moves.append(self.move_history[-1])

        # Now retrieve the target (it is on top)
        from_tier = stack.height
        stack.pop()
        self.total_retrievals += 1
        self.move_history.append(
            Move(
                "retrieve",
                container.id,
                (stack.bay, stack.row),
                from_tier=from_tier,
            )
        )
        return len(blockers), reloc_moves

    # ---------------------------------------------------------------- #
    # Default relocation policy: greedy minimum height                   #
    # ---------------------------------------------------------------- #

    def _greedy_relocation_dst(
        self, avoid: Tuple[int, int]
    ) -> Tuple[int, int]:
        best: Optional[Tuple[int, int]] = None
        best_h = self.max_tiers + 1
        for pos, stack in self.stacks.items():
            if pos == avoid or stack.is_full:
                continue
            if stack.height < best_h:
                best_h = stack.height
                best = pos
        if best is None:
            raise RuntimeError("No valid relocation destination – yard is full")
        return best

    # ---------------------------------------------------------------- #
    # Query helpers                                                       #
    # ---------------------------------------------------------------- #

    def _find_stack(self, container: Container) -> Optional[Stack]:
        for stack in self.stacks.values():
            if stack.contains(container):
                return stack
        return None

    def find_container(self, container_id: int) -> Optional[Tuple[Stack, int]]:
        """Return (stack, 1-indexed-tier) or None."""
        for stack in self.stacks.values():
            for i, c in enumerate(stack.containers):
                if c.id == container_id:
                    return stack, i + 1
        return None

    def get_stack(self, bay: int, row: int) -> Stack:
        return self.stacks[(bay, row)]

    def all_stacks(self) -> List[Stack]:
        return list(self.stacks.values())

    def accessible_containers(self) -> List[Container]:
        """All containers sitting on top of their respective stacks."""
        result = []
        for stack in self.stacks.values():
            if not stack.is_empty:
                result.append(stack.top)
        return result

    def all_containers(self) -> List[Container]:
        containers = []
        for stack in self.stacks.values():
            containers.extend(stack.containers)
        return containers

    def num_containers(self) -> int:
        return sum(s.height for s in self.stacks.values())

    def is_empty(self) -> bool:
        return all(s.is_empty for s in self.stacks.values())

    def total_bad_overlaps(self) -> int:
        return sum(s.num_bad_overlaps() for s in self.stacks.values())

    def is_sorted(self) -> bool:
        """True iff no priority inversion exists anywhere in the yard."""
        return self.total_bad_overlaps() == 0

    # ---------------------------------------------------------------- #
    # State representation for observations / visualisation             #
    # ---------------------------------------------------------------- #

    def get_state_array(self) -> np.ndarray:
        """
        Returns array of shape (num_bays, num_rows, max_tiers, 5):
            [..., 0] = group + 1  (0 = empty)
            [..., 1] = priority   (0 = empty)
            [..., 2] = weight (×10, int)
            [..., 3] = size   (0=empty, 1=TEU, 2=FEU)
            [..., 4] = ctype  (0=empty, 1=STD, 2=REEF, 3=HAZ)
        """
        arr = np.zeros(
            (self.num_bays, self.num_rows, self.max_tiers, 5),
            dtype=np.int32,
        )
        for (b, r), stack in self.stacks.items():
            for tier_idx, c in enumerate(stack.containers):
                arr[b - 1, r - 1, tier_idx, 0] = c.group + 1
                arr[b - 1, r - 1, tier_idx, 1] = c.priority
                arr[b - 1, r - 1, tier_idx, 2] = int(c.weight * 10)
                arr[b - 1, r - 1, tier_idx, 3] = int(c.size)
                arr[b - 1, r - 1, tier_idx, 4] = int(c.ctype) + 1
        return arr

    def get_flat_obs(self) -> np.ndarray:
        """Flat (bay*row*tier*5,) observation vector."""
        return self.get_state_array().flatten()

    def snapshot(self) -> Dict[Tuple[int, int], List[int]]:
        """Lightweight snapshot: {(bay,row): [container_ids bottom→top]}."""
        return {
            pos: stack.snapshot()
            for pos, stack in self.stacks.items()
        }

    def group_snapshot(self) -> Dict[Tuple[int, int], List[int]]:
        """Like snapshot() but with group IDs (better for visualisation)."""
        return {
            pos: stack.group_snapshot()
            for pos, stack in self.stacks.items()
        }

    # ---------------------------------------------------------------- #
    # Deep copy (for GA evaluation)                                      #
    # ---------------------------------------------------------------- #

    def clone(self) -> "Yard":
        """Return a deep copy for simulation without affecting original."""
        return copy.deepcopy(self)

    # ---------------------------------------------------------------- #
    # Crane timing model (optional metric)                               #
    # ---------------------------------------------------------------- #

    def estimate_move_time(
        self,
        src: Tuple[int, int],
        dst: Tuple[int, int],
        crane_speed: float = 1.0,
    ) -> float:
        """
        Simple travel-time estimate: Manhattan distance between stacks,
        normalised by crane_speed (bays/second).
        """
        bay_dist = abs(src[0] - dst[0])
        row_dist = abs(src[1] - dst[1])
        return (bay_dist + row_dist) / crane_speed

    def __repr__(self) -> str:
        lines = [f"Yard({self.num_bays}B×{self.num_rows}R×{self.max_tiers}T)"]
        for (b, r), stack in sorted(self.stacks.items()):
            if not stack.is_empty:
                lines.append(f"  {stack}")
        return "\n".join(lines)
