"""A bay of stacks. Each stack list runs from the bottom to the top."""

from __future__ import annotations

from typing import Dict, List, Tuple

from .container import Container


class Stack:

    def __init__(self, bay: int, row: int, max_tiers: int):
        self.bay = bay
        self.row = row
        self.max_tiers = max_tiers
        self.containers: List[Container] = []

    def push(self, container: Container) -> None:
        if len(self.containers) >= self.max_tiers:
            raise ValueError(
                f"Stack ({self.bay},{self.row}) is full (max_tiers={self.max_tiers})"
            )
        self.containers.append(container)


class Yard:

    def __init__(self, num_bays: int, num_rows: int, max_tiers: int):
        self.num_bays = num_bays
        self.num_rows = num_rows
        self.max_tiers = max_tiers
        self.stacks: Dict[Tuple[int, int], Stack] = {
            (bay, row): Stack(bay, row, max_tiers)
            for bay in range(1, num_bays + 1)
            for row in range(1, num_rows + 1)
        }

    def place(self, bay: int, row: int, container: Container) -> None:
        self.stacks[(bay, row)].push(container)

    def clear(self) -> None:
        for stack in self.stacks.values():
            stack.containers.clear()
