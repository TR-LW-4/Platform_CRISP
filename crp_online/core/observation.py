from dataclasses import dataclass
from typing import List, Optional, Sequence
from .environment import Environment
from ..config.defaults import INT_MAX_SENTINEL, UNKNOWN_CONTAINER_SENTINEL


@dataclass(frozen=True, slots=True)
class MaskedContainer:
    """Represents a container's observable state at a decision step.
    
    Attributes:
        true_id: Ground-truth rank in 1..N (accessible ONLY for assertion/invariance checking).
        group_id: Container group identifier (1..GC).
        is_known: Whether the container's exact ID is revealed at this step.
        obs_value: The numerical proxy/value exposed to terminal calculations.
    """
    true_id: int
    group_id: int
    is_known: bool
    obs_value: float


class ObservationState:
    """View of the container yard visible to heuristics and GP terminals.
    
    Prevents information leakage beyond the specified Look-Ahead horizon or Container Group model.
    """

    def __init__(
        self,
        env: Environment,
        look_ahead_level: float = 1.0,
        horizon_strategy: str = "ignore",
        use_container_groups: bool = False,
        num_groups: int = 2,
        group_size: Optional[int] = None,
        known_within_group: bool = False,
        mode: str = "cpp_compat",
    ) -> None:
        self.num_stacks: int = env.num_stacks
        self.max_stack_height: int = env.max_stack_height
        self.num_containers: int = env.num_containers
        self.current_target: int = env.get_current_target()
        self.look_ahead_level: float = look_ahead_level
        self.horizon_strategy: str = horizon_strategy
        self.use_container_groups: bool = use_container_groups
        self.known_within_group: bool = known_within_group
        self.mode: str = mode

        # Group size and group count calculation
        if self.use_container_groups:
            if group_size is not None and group_size > 0:
                self.group_size: int = group_size
                self.num_groups: int = max(1, self.num_containers // group_size)
            elif num_groups > 0:
                self.num_groups = num_groups
                self.group_size = max(1, self.num_containers // num_groups)
            else:
                self.num_groups = 1
                self.group_size = self.num_containers
        else:
            self.num_groups = num_groups
            self.group_size = self.num_containers

        # Maximum known container ID for limited horizon matching C++ line 940
        # int(GetNextContainerId() + dynamiclevel * GetNumberOfContainers())
        self.max_known_id: int = int(
            self.current_target + self.look_ahead_level * self.num_containers
        )

        # Build masked container stacks
        self._stacks: List[List[MaskedContainer]] = []
        for s in range(self.num_stacks):
            stack_containers: List[MaskedContainer] = []
            for h in range(env.get_stack_height(s)):
                cid = env.get_container_at_height(s, h)
                masked = self._create_masked_container(cid)
                stack_containers.append(masked)
            self._stacks.append(stack_containers)

    def _get_group_id(self, container_id: int) -> int:
        """Determines the group ID (1..GC) for a given container ID."""
        step = self.group_size + 1
        return (container_id // step) + 1

    def _get_group_proxy(self, container_id: int) -> int:
        """Computes C++ group proxy value: groupId * (groupSize + 1)."""
        step = self.group_size + 1
        group_id = (container_id // step) + 1
        return group_id * step

    def _create_masked_container(self, true_id: int) -> MaskedContainer:
        """Constructs a MaskedContainer under the active visibility configuration."""
        group_id = self._get_group_id(true_id)

        # Target container is always known
        if true_id == self.current_target:
            return MaskedContainer(
                true_id=true_id,
                group_id=group_id,
                is_known=True,
                obs_value=float(true_id),
            )

        if self.use_container_groups:
            target_group = self._get_group_id(self.current_target)
            if self.known_within_group and group_id == target_group:
                return MaskedContainer(
                    true_id=true_id,
                    group_id=group_id,
                    is_known=True,
                    obs_value=float(true_id),
                )
            else:
                proxy = self._get_group_proxy(true_id)
                return MaskedContainer(
                    true_id=true_id,
                    group_id=group_id,
                    is_known=False,
                    obs_value=float(proxy),
                )
        else:
            # Limited Look-Ahead Horizon
            if true_id <= self.max_known_id:
                return MaskedContainer(
                    true_id=true_id,
                    group_id=group_id,
                    is_known=True,
                    obs_value=float(true_id),
                )
            else:
                if self.horizon_strategy == "maximum":
                    val = float(self.max_known_id + 1)
                    return MaskedContainer(
                        true_id=true_id,
                        group_id=group_id,
                        is_known=False,
                        obs_value=val,
                    )
                else:
                    # Ignore strategy
                    val = float(UNKNOWN_CONTAINER_SENTINEL) if self.mode == "cpp_compat" else float(self.max_known_id + 1)
                    return MaskedContainer(
                        true_id=true_id,
                        group_id=group_id,
                        is_known=False,
                        obs_value=val,
                    )

    def get_stack_height(self, stack: int) -> int:
        return len(self._stacks[stack])

    def is_stack_empty(self, stack: int) -> bool:
        return len(self._stacks[stack]) == 0

    def is_stack_full(self, stack: int) -> bool:
        return len(self._stacks[stack]) >= self.max_stack_height

    def get_container(self, stack: int, height: int) -> MaskedContainer:
        return self._stacks[stack][height]

    def get_container_obs_value(self, stack: int, height: int) -> float:
        return self._stacks[stack][height].obs_value

    def get_top_obs_value(self, stack: int) -> float:
        """Returns the observable value of the top container on the stack, or -1.0 if empty."""
        if not self._stacks[stack]:
            return -1.0
        return self._stacks[stack][-1].obs_value

    def get_stack_min_obs_value(self, stack: int) -> float:
        """Returns the minimum observable container value in the stack (excluding -1 sentinels).
        
        Returns INT_MAX_SENTINEL (2147483647) if stack is empty or has only unknown sentinels.
        """
        min_val = float(INT_MAX_SENTINEL)
        for mc in self._stacks[stack]:
            if mc.obs_value == UNKNOWN_CONTAINER_SENTINEL:
                continue
            if mc.obs_value < min_val:
                min_val = mc.obs_value
        return min_val

    def get_number_unknown(self, stack: int) -> int:
        """Counts containers in stack whose ID is unrevealed / unknown."""
        return sum(1 for mc in self._stacks[stack] if not mc.is_known)

    def is_well_located(self, stack: int, height: int) -> bool:
        """C++ isWellLocated (line 520): tier 0 is always true.
        For tier > 0: if current is -1, false; if current > any known container below it, false.
        """
        if height == 0:
            return True
        curr_val = self._stacks[stack][height].obs_value
        if curr_val == UNKNOWN_CONTAINER_SENTINEL:
            return False

        for h in range(height - 1, -1, -1):
            below_val = self._stacks[stack][h].obs_value
            if below_val == UNKNOWN_CONTAINER_SENTINEL:
                continue
            if curr_val > below_val:
                return False
        return True

    def get_well_located_count(self, stack: int) -> int:
        """Counts well-located containers in the stack (WL terminal)."""
        return sum(1 for h in range(len(self._stacks[stack])) if self.is_well_located(stack, h))

    def get_bad_located_count(self, stack: int) -> int:
        """Counts ill-located containers in the stack (NL terminal = SH - WL)."""
        return len(self._stacks[stack]) - self.get_well_located_count(stack)

    def calculate_average_id(self, stack: int) -> float:
        """Calculates average observable ID of containers in stack (AVG terminal)."""
        height = len(self._stacks[stack])
        if height == 0:
            return 0.0
        total = 0.0
        count = 0
        for mc in self._stacks[stack]:
            if mc.obs_value == UNKNOWN_CONTAINER_SENTINEL:
                continue
            total += mc.obs_value
            count += 1
        if count == 0:
            return 0.0
        return total / count

    def reshuffle_index(self, stack: int, container_obs_val: float) -> int:
        """Counts containers in candidate stack strictly smaller than the moving container (RI terminal)."""
        count = 0
        for mc in self._stacks[stack]:
            if mc.obs_value == UNKNOWN_CONTAINER_SENTINEL:
                continue
            if container_obs_val > mc.obs_value:
                count += 1
        return count

    def expected_reshuffle_index(self, stack: int, container_obs_val: float) -> float:
        """Expected Reshuffle Index (ERI heuristic): +1 if smaller, +0.5 if equal."""
        score = 0.0
        for mc in self._stacks[stack]:
            if mc.obs_value == UNKNOWN_CONTAINER_SENTINEL:
                continue
            if container_obs_val > mc.obs_value:
                score += 1.0
            elif container_obs_val == mc.obs_value:
                score += 0.5
        return score

    def height_of_lower_index(self, stack: int, container_obs_val: float) -> int:
        """Bottom-up tier of the first container with ID < moving container ID (DSM terminal)."""
        for h, mc in enumerate(self._stacks[stack]):
            if mc.obs_value == UNKNOWN_CONTAINER_SENTINEL:
                continue
            if container_obs_val > mc.obs_value:
                return h
        return 0

    def containers_above_target(self, target_stack: int) -> int:
        """Count of containers from target up to top on the target's stack (REM terminal)."""
        h_target = 0
        for h, mc in enumerate(self._stacks[target_stack]):
            if mc.true_id == self.current_target:
                h_target = h
                break
        return len(self._stacks[target_stack]) - h_target
