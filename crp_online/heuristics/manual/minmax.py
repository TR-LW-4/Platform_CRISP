from typing import Optional
from ...core.environment import Environment
from ...core.observation import ObservationState
from ...config.defaults import INT_MAX_SENTINEL


def select_minmax_destination(
    obs: ObservationState,
    env: Environment,
    source_stack: int,
    moving_val: float,
) -> int:
    """MM (Min-Max - Caserta et al., 2012):
    Selects stack with minimum container ID > relocating container ID.
    If none exists, selects stack with maximum container ID <= relocating container ID.
    Faithfully reproduces C++ MinMax.cpp empty stack logic (lowestIndex = 0).
    """
    lowest_index: float = float(INT_MAX_SENTINEL)
    lowest_location: Optional[int] = None

    largest_index: float = -float(INT_MAX_SENTINEL)
    largest_location: Optional[int] = None

    for s in range(env.num_stacks):
        if s == source_stack or env.is_stack_full(s):
            continue

        s_lowest = obs.get_stack_min_obs_value(s)

        if s_lowest > moving_val:
            if lowest_index > s_lowest:
                lowest_index = s_lowest
                lowest_location = s
            elif s_lowest == float(INT_MAX_SENTINEL):
                lowest_index = 0.0
                lowest_location = s
        else:
            if largest_index < s_lowest:
                largest_index = s_lowest
                largest_location = s

    if lowest_location is not None:
        return lowest_location
    if largest_location is not None:
        return largest_location

    raise RuntimeError("No legal destination stack in Min-Max!")
