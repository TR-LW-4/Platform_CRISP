"""One container in a CRP layout. Retrieval priority 1 leaves first."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Container:
    id: int
    priority: int
    group: int = 0
