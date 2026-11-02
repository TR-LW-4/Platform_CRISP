from typing import Optional
from ...core.environment import Environment
from ...core.observation import ObservationState
from ...config.defaults import INT_MAX_SENTINEL


def select_tlp_destination(
    obs: ObservationState,
    env: Environment,
    source_stack: int,
    moving_val: float,
) -> int:
    """TLP (Lee & Lee, 2010): Selects stack with minimum height.
    Ties broken by lowest stack index.
    """
    best_stack: Optional[int] = None
    min_height = INT_MAX_SENTINEL

    for s in range(env.num_stacks):
        if s == source_stack or env.is_stack_full(s):
            continue
        h = env.get_stack_height(s)
        if h < min_height:
            min_height = h
            best_stack = s

    if best_stack is None:
        raise RuntimeError("No legal destination stack in TLP!")
    return best_stack
