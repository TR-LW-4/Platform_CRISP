from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Location:
    """Represents a discrete coordinate in the container yard.
    
    Attributes:
        bay: Bay index (0 for single-bay CRP).
        stack: Stack index (0 to S-1; -1 represents the truck position).
    """
    bay: int = 0
    stack: int = 0

    def __repr__(self) -> str:
        return f"Location(bay={self.bay}, stack={self.stack})"
