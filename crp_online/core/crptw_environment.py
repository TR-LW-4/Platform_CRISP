"""Stochastic CRPTW Environment supporting both Online (Unknown) and Batch (Known) information regimes."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
from ..io.ku_loader import KuInstance
from ..core.observation import ObservationState, MaskedContainer
from ..core.environment import Environment
from ..core.instance import ProblemInstance


@dataclass
class CRPTWContainer:
    id: int
    window: int
    original_stack: int
    original_tier: int


class CRPTWEnvironment:
    """Simulation environment for the Container Relocation Problem with Time Windows (CRPTW).
    
    Adheres strictly to the Information Barrier contract:
    - Supports Unknown (Online) and Known (Batch) within-group revelation regimes.
    - In Unknown mode, only the active target container is revealed; unretrieved containers are masked by their window ID or scaled group proxy.
    - Prevents future-information leakage by isolating the latent departure sequence.
    """

    def __init__(
        self,
        instance: KuInstance,
        variant: str = "unknown",
        encoding: str = "proxy",
        group_size: int = 2,
    ) -> None:
        self.instance: KuInstance = instance
        self.num_stacks: int = instance.num_stacks
        self.max_stack_height: int = instance.num_tiers  # H = T
        self.num_containers: int = instance.num_containers
        self.variant: str = variant.lower()
        self.encoding: str = encoding.lower()
        self.group_size: int = group_size

        if self.variant not in ("unknown", "known"):
            raise ValueError(f"Unknown variant '{self.variant}'; must be 'unknown' or 'known'")
        if self.encoding not in ("proxy", "offset"):
            raise ValueError(f"Unknown encoding '{self.encoding}'; must be 'proxy' or 'offset'")

        # Initialize containers and stacks
        self.containers: Dict[int, CRPTWContainer] = {}
        self.stacks: List[List[int]] = [[] for _ in range(self.num_stacks)]

        # Cumulative offsets for Encoding B
        window_counts: Dict[int, int] = {}
        cid = 0
        for s_idx, stk_win in enumerate(instance.stack_windows):
            for t_idx, win in enumerate(stk_win):
                c = CRPTWContainer(id=cid, window=win, original_stack=s_idx, original_tier=t_idx)
                self.containers[cid] = c
                self.stacks[s_idx].append(cid)
                window_counts[win] = window_counts.get(win, 0) + 1
                cid += 1

        self.cumulative_offsets: Dict[int, int] = {}
        cum = 1
        for win in sorted(window_counts.keys()):
            self.cumulative_offsets[win] = cum
            cum += window_counts[win]

        # Execution state
        self.total_relocations: int = 0
        self.endgame_relocations: int = 0  # relocations incurred while C_rem <= S
        self.current_target: Optional[int] = None
        self.retrieved_containers: List[int] = []

        # Latent departure sequence (precomputed per trial)
        self._departure_sequence: List[int] = []
        self._seq_idx: int = 0

    def set_departure_sequence(self, sequence: Sequence[int]) -> None:
        """Sets the predetermined departure sequence for this trial."""
        if len(sequence) != self.num_containers:
            raise ValueError(f"Sequence length {len(sequence)} != num_containers {self.num_containers}")
        self._departure_sequence = list(sequence)
        self._seq_idx = 0
        self.advance_next_target()

    def advance_next_target(self) -> None:
        """Advances to the next container in the departure sequence."""
        if self._seq_idx < len(self._departure_sequence):
            self.current_target = self._departure_sequence[self._seq_idx]
            self._seq_idx += 1
        else:
            self.current_target = None

    def get_remaining_container_count(self) -> int:
        return sum(len(stk) for stk in self.stacks)

    def is_empty(self) -> bool:
        return self.get_remaining_container_count() == 0

    def get_stack_height(self, stack: int) -> int:
        return len(self.stacks[stack])

    def is_stack_full(self, stack: int) -> bool:
        return len(self.stacks[stack]) >= self.max_stack_height

    def is_stack_empty(self, stack: int) -> bool:
        return len(self.stacks[stack]) == 0

    def get_top_container(self, stack: int) -> int:
        if not self.stacks[stack]:
            raise IndexError(f"Stack {stack} is empty")
        return self.stacks[stack][-1]

    def get_stack_of_container(self, container_id: int) -> int:
        for s_idx, stk in enumerate(self.stacks):
            if container_id in stk:
                return s_idx
        raise KeyError(f"Container {container_id} not found in yard")

    def relocate(self, source_stack: int, dest_stack: int) -> None:
        """Executes a relocation move from source_stack to dest_stack."""
        if source_stack == dest_stack:
            raise ValueError(f"Cannot relocate within same stack {source_stack}")
        if self.is_stack_empty(source_stack):
            raise IndexError(f"Source stack {source_stack} is empty")
        if self.is_stack_full(dest_stack):
            raise OverflowError(f"Destination stack {dest_stack} is full (capacity {self.max_stack_height})")

        cid = self.stacks[source_stack].pop()
        self.stacks[dest_stack].append(cid)

        self.total_relocations += 1
        if self.get_remaining_container_count() <= self.num_stacks:
            self.endgame_relocations += 1

    def retrieve(self, stack: int) -> int:
        """Retrieves the top container from stack, asserting it is the active target."""
        if self.is_stack_empty(stack):
            raise IndexError(f"Cannot retrieve from empty stack {stack}")
        cid = self.stacks[stack][-1]
        if cid != self.current_target:
            raise RuntimeError(
                f"Top container {cid} (window {self.containers[cid].window}) is not active target {self.current_target}"
            )
        self.stacks[stack].pop()
        self.retrieved_containers.append(cid)
        self.advance_next_target()
        return cid

    def get_container_obs_value(self, container_id: int) -> float:
        """Computes the observable value exposed to priority heuristics and GP terminals."""
        c = self.containers[container_id]

        # Target container is always known
        if container_id == self.current_target:
            return float(c.window) if self.encoding == "proxy" else float(self.cumulative_offsets[c.window])

        target_c = self.containers[self.current_target] if self.current_target is not None else None

        if self.variant == "known" and target_c is not None and c.window == target_c.window:
            # In Known mode, order within active window is known
            # Give exact sub-rank within the active window
            active_window_order = [
                cid for cid in self._departure_sequence
                if self.containers[cid].window == target_c.window
            ]
            rank = active_window_order.index(container_id) + 1
            if self.encoding == "proxy":
                return float(c.window * (self.group_size + 1) + rank)
            else:
                return float(self.cumulative_offsets[c.window] + rank - 1)
        else:
            # Unknown mode or later window: mask with group proxy
            if self.encoding == "proxy":
                step = self.group_size + 1
                return float(c.window * step)
            else:
                return float(self.cumulative_offsets[c.window])

    def create_mock_environment(self) -> Environment:
        """Creates a compatible standard Environment instance for GP terminal evaluation."""
        layout = tuple(tuple(self.stacks[s]) for s in range(self.num_stacks))
        inst = ProblemInstance(
            name=self.instance.name,
            num_stacks=self.num_stacks,
            num_containers=self.num_containers,
            max_stack_height=self.max_stack_height,
            initial_height=max((len(stk) for stk in self.stacks), default=0),
            initial_layout=layout,
        )
        env = Environment(inst, store_moves=False)
        return env

    def get_observation_state(self) -> ObservationState:
        """Creates an ObservationState representing the current yard under the active masking contract."""
        env = self.create_mock_environment()

        # Build custom ObservationState
        obs = ObservationState(
            env,
            look_ahead_level=1.0,
            use_container_groups=True,
            num_groups=self.instance.num_batches,
            group_size=self.group_size,
            known_within_group=(self.variant == "known"),
        )

        # Overwrite observable container values with CRPTW values
        for s in range(self.num_stacks):
            new_stack: List[MaskedContainer] = []
            for h in range(len(self.stacks[s])):
                cid = self.stacks[s][h]
                obs_val = self.get_container_obs_value(cid)
                is_known = (cid == self.current_target) or (
                    self.variant == "known"
                    and self.current_target is not None
                    and self.containers[cid].window == self.containers[self.current_target].window
                )
                mc = MaskedContainer(
                    true_id=cid,
                    group_id=self.containers[cid].window,
                    is_known=is_known,
                    obs_value=obs_val,
                )
                new_stack.append(mc)
            obs._stacks[s] = new_stack

        return obs


def precompute_departure_sequence(instance: KuInstance, rng: np.random.Generator) -> List[int]:
    """Generates a complete latent departure sequence preserving strict inter-window precedence.
    
    For each departure window w in ascending order, containers in window w are permuted uniformly at random.
    """
    window_containers: Dict[int, List[int]] = {}
    cid = 0
    for s_idx, stk_win in enumerate(instance.stack_windows):
        for win in stk_win:
            window_containers.setdefault(win, []).append(cid)
            cid += 1

    departure_sequence: List[int] = []
    for win in sorted(window_containers.keys()):
        cids = window_containers[win]
        if len(cids) > 1:
            permuted = rng.permutation(cids).tolist()
        else:
            permuted = list(cids)
        departure_sequence.extend(permuted)

    return departure_sequence
