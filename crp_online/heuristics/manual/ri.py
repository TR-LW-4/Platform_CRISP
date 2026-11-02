from typing import Optional
from ...core.environment import Environment
from ...core.observation import ObservationState
from ...config.defaults import INT_MAX_SENTINEL


def select_ri_destination(
    obs: ObservationState,
    env: Environment,
    source_stack: int,
    moving_val: float,
) -> int:
    """RI (Lee & Lee, 2010): Selects stack minimizing the reshuffle index.
    Ties broken by lowest stack index.
    """
    best_stack: Optional[int] = None
    min_ri = float(INT_MAX_SENTINEL)

    for s in range(env.num_stacks):
        if s == source_stack or env.is_stack_full(s):
            continue

        ri = float(obs.reshuffle_index(s, moving_val))
        if ri < min_ri:
            min_ri = ri
            best_stack = s

    if best_stack is None:
        raise RuntimeError("No legal destination stack in RI!")
    return best_stack
