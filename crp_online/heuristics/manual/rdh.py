from typing import Optional
from ...core.environment import Environment
from ...core.observation import ObservationState
from ...config.defaults import INT_MAX_SENTINEL


def select_rdh_destination(
    obs: ObservationState,
    env: Environment,
    source_stack: int,
    moving_val: float,
) -> int:
    """RDH (Revised Difference Heuristic - Zhao & Goodchild, 2010):
    4-Tier hierarchical candidate selection directly matching C++ RDH.cpp.
    """
    x = moving_val
    smallest_y = float(INT_MAX_SENTINEL)
    selected_location: Optional[int] = None

    # Tier 1: Stacks with min(s) > X
    for s in range(env.num_stacks):
        if s == source_stack or env.is_stack_full(s):
            continue
        y = obs.get_stack_min_obs_value(s)
        if y > x:
            if y == float(INT_MAX_SENTINEL) or smallest_y > y:
                smallest_y = y
                selected_location = s

    # Tier 2: Stacks with min(s) == X
    if selected_location is None:
        for s in range(env.num_stacks):
            if s == source_stack or env.is_stack_full(s):
                continue
            y = obs.get_stack_min_obs_value(s)
            if y == x:
                if smallest_y > y:
                    smallest_y = y
                    selected_location = s

    # Tier 3: Stacks with top(s) < X
    if selected_location is None:
        largest_z = -float(INT_MAX_SENTINEL)
        for s in range(env.num_stacks):
            if s == source_stack or env.is_stack_full(s):
                continue
            z = -float(INT_MAX_SENTINEL) + 1.0 if obs.is_stack_empty(s) else obs.get_top_obs_value(s)
            if z < x:
                if largest_z < z:
                    largest_z = z
                    selected_location = s

    # Tier 4: Default fallback minimizing top(s)
    if selected_location is None:
        largest = float(INT_MAX_SENTINEL)
        for s in range(env.num_stacks):
            if s == source_stack or env.is_stack_full(s):
                continue
            z = float(INT_MAX_SENTINEL) - 1.0 if obs.is_stack_empty(s) else obs.get_top_obs_value(s)
            if largest > z:
                largest = z
                selected_location = s

    if selected_location is not None:
        return selected_location

    raise RuntimeError("No legal destination stack in RDH!")
