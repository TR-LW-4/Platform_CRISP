from typing import Optional
from ...core.environment import Environment
from ...core.observation import ObservationState
from ...config.defaults import INT_MAX_SENTINEL


def select_eri_destination(
    obs: ObservationState,
    env: Environment,
    source_stack: int,
    moving_val: float,
) -> int:
    """ERI (Expected Reshuffle Index - Ku & Arthanari, 2016):
    Minimizes expected reshuffles (+1 if smaller, +0.5 if equal).
    """
    best_stack: Optional[int] = None
    min_eri: float = float(INT_MAX_SENTINEL)

    for s in range(env.num_stacks):
        if s == source_stack or env.is_stack_full(s):
            continue

        eri = obs.expected_reshuffle_index(s, moving_val)
        if eri < min_eri:
            min_eri = eri
            best_stack = s

    if best_stack is None:
        raise RuntimeError("No legal destination stack in ERI!")
    return best_stack
