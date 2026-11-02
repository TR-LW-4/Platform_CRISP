from typing import Optional
from ...core.environment import Environment
from ...core.observation import ObservationState
from ...config.defaults import INT_MAX_SENTINEL


def select_pr4_destination(
    obs: ObservationState,
    env: Environment,
    source_stack: int,
    moving_val: float,
) -> int:
    """PR4 (Zhu et al., 2012):
    Refines Min-Max with a penalty when placing on a near-full stack
    if multiple containers smaller than the candidate's min remain on source stack.
    """
    lowest_index: float = float(INT_MAX_SENTINEL)
    selected_location: Optional[int] = None

    highest_index: float = -float(INT_MAX_SENTINEL)
    selected_location_highest: Optional[int] = None

    for s in range(env.num_stacks):
        if s == source_stack or env.is_stack_full(s):
            continue

        s_lowest = obs.get_stack_min_obs_value(s)

        if s_lowest > moving_val:
            if lowest_index > s_lowest:
                lowest_index = s_lowest
                selected_location = s
            elif s_lowest == float(INT_MAX_SENTINEL):
                lowest_index = 0.0
                selected_location = s
        else:
            # Check how many containers on source stack are smaller than highest_index
            num_smaller = 0
            for h in range(obs.get_stack_height(source_stack)):
                cid = obs.get_container_obs_value(source_stack, h)
                if cid != -1 and highest_index > cid:
                    num_smaller += 1

            # Penalty: near full stack and multiple smaller remaining
            if num_smaller > 1 and (obs.max_stack_height - obs.get_stack_height(s) == 1):
                pass  # Penalized! Do not update
            elif highest_index < s_lowest:
                highest_index = s_lowest
                selected_location_highest = s

    if lowest_index < float(INT_MAX_SENTINEL) and selected_location is not None:
        return selected_location
    if selected_location_highest is not None:
        return selected_location_highest

    raise RuntimeError("No legal destination stack in PR4!")
