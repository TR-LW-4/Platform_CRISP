"""
CRP-R
<restricted> <distinct> <CRP-R>
Classical Block Relocation Problem with fixed retrieval order

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP".
--------------------------------------------------------------------------
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Optional, Sequence

from core.container import Container
from core.problem import BaseProblem


class CRP_R(BaseProblem):
    """One CRP-R bay: fixed retrieval order, only blockers above the target may move."""

    name = "CRP-R"

    def __init__(self, config=None):
        super().__init__(config)
        self._place_containers()
        self.n = int(self.config.num_containers)
        self.max_tiers = int(self.config.max_tiers)
        self.stacks = [
            [int(container.priority) for container in stack.containers]
            for _, stack in sorted(self.yard.stacks.items())
        ]

    def _place_containers(self) -> None:
        """Fill the yard from a layout file, or from a random feasible layout."""
        path = _layout_path(self.config.extra)
        if path is not None:
            columns = _read_priority_columns(Path(path))
            self._place_columns(columns)
            self.config.num_containers = sum(len(column) for column in columns)
            return

        self._place_random()

    def _place_columns(self, columns: Sequence[Sequence[int]]) -> None:
        slots = sorted(self.yard.stacks)
        if len(columns) > len(slots):
            raise ValueError(
                f"layout has {len(columns)} stacks but the yard has {len(slots)}"
            )
        placed: List[Container] = []
        for column, (bay, row) in zip(columns, slots):
            for priority in column:
                container = Container(id=int(priority), priority=int(priority))
                self.yard.place(bay, row, container)
                placed.append(container)
        self.containers = placed

    def _place_random(self) -> None:
        n = int(self.config.num_containers)
        if n <= 0:
            return
        rng = random.Random(self.config.extra.get("seed", 0))
        priorities = list(range(1, n + 1))
        rng.shuffle(priorities)
        slots = sorted(self.yard.stacks)
        heights = {slot: 0 for slot in slots}
        placed: List[Container] = []
        for priority in priorities:
            room = [slot for slot, height in heights.items() if height < self.config.max_tiers]
            if not room:
                raise ValueError("not enough stack space for the containers")
            bay, row = rng.choice(room)
            container = Container(id=priority, priority=priority)
            self.yard.place(bay, row, container)
            heights[(bay, row)] += 1
            placed.append(container)
        self.containers = placed

    def target(self, stacks: Optional[Sequence[Sequence[int]]] = None) -> Optional[int]:
        """Smallest remaining priority. That container is retrieved next."""
        present = [priority for stack in (self.stacks if stacks is None else stacks) for priority in stack]
        return min(present) if present else None

    def blockers(self, stacks: Optional[Sequence[Sequence[int]]] = None) -> List[int]:
        """Containers above the current target, from the top of the stack downward."""
        yard = self.stacks if stacks is None else stacks
        current = self.target(yard)
        if current is None:
            return []
        source = next(index for index, stack in enumerate(yard) if current in stack)
        above = yard[source][yard[source].index(current) + 1 :]
        return list(reversed(above))

    def legal_destinations(self, stacks: Sequence[Sequence[int]], source: int) -> List[int]:
        """Other stacks that still have a free tier."""
        return [
            index
            for index, stack in enumerate(stacks)
            if index != source and len(stack) < self.max_tiers
        ]


def _layout_path(extra) -> Optional[str]:
    if not extra:
        return None
    path = extra.get("layout_file_path") or extra.get("caserta_dat_path")
    return str(path) if path else None


def _read_priority_columns(path: Path) -> List[List[int]]:
    """Caserta ``.dat`` and Zhu ``.txt`` both store one stack per line, bottom to top."""
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError(f"layout file is too short: {path}")
    header = lines[0].split()
    n_stacks, n_containers = int(header[0]), int(header[1])
    columns: List[List[int]] = []
    for line in lines[1 : n_stacks + 1]:
        numbers = [int(token) for token in line.split()]
        height = numbers[0]
        columns.append(numbers[1 : 1 + height])
    found = [priority for column in columns for priority in column]
    if len(found) != n_containers:
        raise ValueError(f"{path}: found {len(found)} containers, header says {n_containers}")
    return columns
