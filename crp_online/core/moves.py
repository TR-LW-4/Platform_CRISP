from dataclasses import dataclass
from .location import Location


@dataclass(frozen=True, slots=True)
class ContainerMove:
    """Record of a single crane movement operation in the container yard.
    
    Attributes:
        source: Origin location of the container.
        destination: Destination location of the container (truck if retrieval).
        container_id: Ground-truth container ID moved.
        is_retrieval: True if retrieved to truck, False if relocated within yard.
    """
    source: Location
    destination: Location
    container_id: int = 0
    is_retrieval: bool = False

    def __repr__(self) -> str:
        op = "RETRIEVE" if self.is_retrieval else "RELOCATE"
        return f"Move({op} c={self.container_id} from s={self.source.stack} to s={self.destination.stack})"
