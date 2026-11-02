from typing import Any, List, Optional, Sequence
from .location import Location
from .moves import ContainerMove
from .bay_floor import BayFloor
from .instance import ProblemInstance
from ..config.defaults import (
    GANTRY_SPEED,
    TROLLEY_SPEED,
    GANTRY_ACCELERATION,
    PICK_UP_TIME,
    TRUCK_STACK_INDEX,
)


class Environment:
    """State simulator for the Online Container Relocation Problem (Single-Bay).
    
    Maintains ground-truth bay floor state, crane kinematics, move history,
    and retrieval sequence. Enforces physical constraints (capacity H, legal moves).
    """

    def __init__(
        self,
        instance: ProblemInstance,
        store_moves: bool = True,
    ) -> None:
        self.instance: ProblemInstance = instance
        self.num_stacks: int = instance.num_stacks
        self.max_stack_height: int = instance.max_stack_height
        self.num_containers: int = instance.num_containers
        self.store_moves: bool = store_moves

        # Yard physical state
        self.bay_floor: BayFloor = instance.create_bay_floor()

        # Location mapping for O(1) container lookups
        self._container_stack: dict[int, int] = {}
        for s in range(self.num_stacks):
            for cid in self.bay_floor.get_stack_containers(s):
                self._container_stack[cid] = s

        # Crane kinematics state
        self.crane_pos: Location = Location(bay=0, stack=TRUCK_STACK_INDEX)
        self.total_execution_time: float = 0.0
        self.total_relocations: int = 0
        self.total_retrievals: int = 0
        self.current_target: int = 1
        self.move_history: List[ContainerMove] = []

    def get_stack_height(self, stack: int) -> int:
        """Returns the number of containers currently in the given stack."""
        if stack == TRUCK_STACK_INDEX:
            return 1
        return self.bay_floor.get_stack_height(stack)

    def is_stack_empty(self, stack: int) -> bool:
        """Checks if the stack contains zero containers."""
        return self.bay_floor.is_stack_empty(stack)

    def is_stack_full(self, stack: int) -> bool:
        """Checks if the stack has reached maximum capacity H = h + 2."""
        return self.bay_floor.get_stack_height(stack) >= self.max_stack_height

    def get_top_container(self, stack: int) -> int:
        """Returns the ID of the top container on the stack."""
        return self.bay_floor.get_top_container(stack)

    def get_container_at_height(self, stack: int, height: int) -> int:
        """Returns the container ID at a specific vertical tier (0 = bottom)."""
        return self.bay_floor.get_container_at_height(stack, height)

    def get_stack_containers(self, stack: int) -> tuple[int, ...]:
        """Returns all container IDs in the stack from bottom to top."""
        return self.bay_floor.get_stack_containers(stack)

    def get_current_target(self) -> int:
        """Returns the ground-truth ID of the container currently required for retrieval."""
        return self.current_target

    def get_stack_of_container(self, container_id: int) -> int:
        """Returns the stack index containing the specified container ID."""
        if container_id not in self._container_stack:
            raise KeyError(f"Container {container_id} is not in the yard!")
        return self._container_stack[container_id]

    def is_empty(self) -> bool:
        """Returns True when all containers have been retrieved."""
        return self.total_retrievals == self.num_containers

    def calculate_crane_time(self, source: Location, destination: Location) -> float:
        """Calculates exact crane travel time according to C++ ContainerYard line 244.
        
        Single-bay kinematics:
          timeToGetSource = 1 * gantryAcceleration + trolleySpeed * |src.stack - crane.stack|
          timeToGetFromSourceToDest = pickUpTime + 1 * gantryAcceleration + trolleySpeed * |dest.stack - src.stack|
          total = timeToGetSource + timeToGetFromSourceToDest
        """
        source_bay_change = 1
        gantry_acc = GANTRY_ACCELERATION
        trolley_spd = TROLLEY_SPEED
        pickup_t = PICK_UP_TIME

        time_to_get_source = (
            source_bay_change * gantry_acc
            + trolley_spd * abs(source.stack - self.crane_pos.stack)
        )

        bay_change = 1
        time_to_get_dest = (
            pickup_t
            + bay_change * gantry_acc
            + trolley_spd * abs(destination.stack - source.stack)
        )

        return time_to_get_source + time_to_get_dest

    def relocate(self, source_stack: int, dest_stack: int) -> ContainerMove:
        """Moves the top container from source_stack to dest_stack.
        
        Enforces:
          - source_stack != dest_stack
          - source_stack is not empty
          - dest_stack is not full (< max_stack_height)
        Updates crane position, execution time, and move counts.
        """
        if source_stack == dest_stack:
            raise ValueError(f"Source and destination stacks cannot be the same ({source_stack})!")
        if self.is_stack_empty(source_stack):
            raise ValueError(f"Cannot relocate from empty stack {source_stack}!")
        if self.is_stack_full(dest_stack):
            raise ValueError(f"Destination stack {dest_stack} is at capacity {self.max_stack_height}!")

        src_loc = Location(bay=0, stack=source_stack)
        dest_loc = Location(bay=0, stack=dest_stack)

        # Update kinematics
        self.total_execution_time += self.calculate_crane_time(self.crane_pos, src_loc)
        self.total_execution_time += self.calculate_crane_time(src_loc, dest_loc)
        self.crane_pos = dest_loc

        # Perform physical move
        cid = self.bay_floor.move_container(source_stack, dest_stack)
        self._container_stack[cid] = dest_stack
        self.total_relocations += 1

        move = ContainerMove(source=src_loc, destination=dest_loc, container_id=cid, is_retrieval=False)
        if self.store_moves:
            self.move_history.append(move)
        return move

    def retrieve(self, source_stack: int) -> ContainerMove:
        """Retrieves the top container from source_stack to the truck (stack -1).
        
        Enforces that the top container is the current retrieval target.
        Updates crane position to truck (stack -1), advances target ID.
        """
        if self.is_stack_empty(source_stack):
            raise ValueError(f"Cannot retrieve from empty stack {source_stack}!")

        top_cid = self.bay_floor.get_top_container(source_stack)
        if top_cid != self.current_target:
            raise ValueError(
                f"Cannot retrieve container {top_cid}: current target is {self.current_target}!"
            )

        src_loc = Location(bay=0, stack=source_stack)
        truck_loc = Location(bay=0, stack=TRUCK_STACK_INDEX)

        # Update kinematics
        self.total_execution_time += self.calculate_crane_time(src_loc, truck_loc)
        self.crane_pos = truck_loc

        # Perform physical retrieval
        cid = self.bay_floor.pop_top_container(source_stack)
        del self._container_stack[cid]
        self.total_retrievals += 1
        self.current_target += 1

        move = ContainerMove(source=src_loc, destination=truck_loc, container_id=cid, is_retrieval=True)
        if self.store_moves:
            self.move_history.append(move)
        return move

    def get_observation(
        self,
        look_ahead_level: float = 1.0,
        horizon_strategy: str = "ignore",
        use_container_groups: bool = False,
        num_groups: int = 2,
        group_size: Optional[int] = None,
        known_within_group: bool = False,
        mode: str = "cpp_compat",
    ) -> Any:
        """Returns the active ObservationState view enforcing information barriers."""
        from .observation import ObservationState
        return ObservationState(
            env=self,
            look_ahead_level=look_ahead_level,
            horizon_strategy=horizon_strategy,
            use_container_groups=use_container_groups,
            num_groups=num_groups,
            group_size=group_size,
            known_within_group=known_within_group,
            mode=mode,
        )

    def clone(self) -> "Environment":
        """Creates a deep copy of the current environment state."""
        cloned = Environment(self.instance, store_moves=self.store_moves)
        cloned.bay_floor = self.bay_floor.clone()
        cloned._container_stack = dict(self._container_stack)
        cloned.crane_pos = self.crane_pos
        cloned.total_execution_time = self.total_execution_time
        cloned.total_relocations = self.total_relocations
        cloned.total_retrievals = self.total_retrievals
        cloned.current_target = self.current_target
        cloned.move_history = list(self.move_history)
        return cloned
