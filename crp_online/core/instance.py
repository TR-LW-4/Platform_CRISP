from dataclasses import dataclass
from typing import Sequence
from .bay_floor import BayFloor


@dataclass(frozen=True)
class ProblemInstance:
    """Immutable problem instance definition for the Container Relocation Problem.
    
    Attributes:
        name: Instance identifier (e.g., 'data3-3-1').
        num_stacks: Number of vertical stacks (S).
        num_containers: Total number of containers (N).
        max_stack_height: Maximum permitted stack height (H = h + 2).
        initial_height: Initial stack height (h = N / S).
        initial_layout: Initial containers per stack from floor (tier 0) to top.
    """
    name: str
    num_stacks: int
    num_containers: int
    max_stack_height: int
    initial_height: int
    initial_layout: tuple[tuple[int, ...], ...]

    def create_bay_floor(self) -> BayFloor:
        """Instantiates a fresh BayFloor matching the initial problem layout."""
        return BayFloor(self.num_stacks, self.initial_layout)

    def __repr__(self) -> str:
        return f"ProblemInstance('{self.name}', S={self.num_stacks}, N={self.num_containers}, H={self.max_stack_height})"
