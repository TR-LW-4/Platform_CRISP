from typing import List, Optional
from ...core.environment import Environment
from ...core.observation import ObservationState
from ...core.moves import ContainerMove
from ...config.defaults import INT_MAX_SENTINEL


def solve_pu2(
    env: Environment,
    look_ahead_level: float = 1.0,
    horizon_strategy: str = "ignore",
    use_container_groups: bool = False,
    num_groups: int = 2,
    group_size: Optional[int] = None,
    known_within_group: bool = False,
    mode: str = "cpp_compat",
) -> List[ContainerMove]:
    """PU2 (Zhu et al., 2012):
    Unrestricted heuristic executing PR4 with preventive moves.
    Faithfully reproduces C++ PU2.cpp logic and typo side effect.
    """
    moves: List[ContainerMove] = []

    while not env.is_empty():
        target = env.get_current_target()
        source_stack = env.get_stack_of_container(target)

        while env.get_top_container(source_stack) != target:
            obs = ObservationState(
                env,
                look_ahead_level=look_ahead_level,
                horizon_strategy=horizon_strategy,
                use_container_groups=use_container_groups,
                num_groups=num_groups,
                group_size=group_size,
                known_within_group=known_within_group,
                mode=mode,
            )
            top_val = obs.get_top_obs_value(source_stack)

            lowest_index: float = float(INT_MAX_SENTINEL)
            selected_location: Optional[int] = None

            highest_index: float = -float(INT_MAX_SENTINEL)
            selected_location_highest: Optional[int] = None

            for s in range(env.num_stacks):
                if s == source_stack or env.is_stack_full(s):
                    continue

                s_lowest = obs.get_stack_min_obs_value(s)
                if s_lowest > top_val:
                    if lowest_index > s_lowest:
                        lowest_index = s_lowest
                        selected_location = s
                    elif s_lowest == float(INT_MAX_SENTINEL):
                        lowest_index = 0.0
                        selected_location = s
                else:
                    # PR4 penalty check
                    num_smaller = 0
                    for h in range(obs.get_stack_height(source_stack)):
                        cid = obs.get_container_obs_value(source_stack, h)
                        if cid != -1 and highest_index > cid:
                            num_smaller += 1

                    if num_smaller > 1 and (obs.max_stack_height - obs.get_stack_height(s) == 1):
                        pass
                    elif highest_index < s_lowest:
                        highest_index = s_lowest
                        selected_location_highest = s

            if lowest_index < float(INT_MAX_SENTINEL) and selected_location is not None:
                move = env.relocate(source_stack, selected_location)
                moves.append(move)
            else:
                # Preventive search
                selected_prev: Optional[int] = None
                container_id_prev: float = -1.0

                for s in range(env.num_stacks):
                    if s == source_stack:
                        continue

                    s_lowest = float(INT_MAX_SENTINEL)
                    second_smallest = float(INT_MAX_SENTINEL)
                    h_count = obs.get_stack_height(s)

                    for h in range(h_count):
                        cid = obs.get_container_obs_value(s, h)
                        if cid != -1 and s_lowest > cid:
                            second_smallest = s_lowest
                            s_lowest = cid

                    # Typo side effect handling:
                    if mode == "cpp_compat":
                        condition = (s_lowest == obs.get_top_obs_value(s)) and (h_count == 1)
                    else:
                        condition = (s_lowest == obs.get_top_obs_value(s)) and (second_smallest != float(INT_MAX_SENTINEL) or h_count == 1)

                    if condition:
                        if s_lowest > container_id_prev:
                            container_id_prev = s_lowest
                            selected_prev = s

                # Look for destination for selected_prev
                selected_dest1: Optional[int] = None
                lowest_index1: float = float(INT_MAX_SENTINEL)

                if selected_prev is not None:
                    for s in range(env.num_stacks):
                        if s == source_stack or s == selected_prev or env.is_stack_full(s):
                            continue

                        s_lowest = obs.get_stack_min_obs_value(s)
                        if s_lowest > container_id_prev:
                            if lowest_index1 > s_lowest:
                                lowest_index1 = s_lowest
                                selected_dest1 = s
                            elif s_lowest == float(INT_MAX_SENTINEL):
                                lowest_index1 = 0.0
                                selected_dest1 = s

                if (
                    lowest_index1 < float(INT_MAX_SENTINEL)
                    and selected_dest1 is not None
                    and selected_prev is not None
                ):
                    prev_move = env.relocate(selected_prev, selected_dest1)
                    moves.append(prev_move)
                    main_move = env.relocate(source_stack, selected_prev)
                    moves.append(main_move)
                elif selected_location_highest is not None:
                    main_move = env.relocate(source_stack, selected_location_highest)
                    moves.append(main_move)
                else:
                    raise RuntimeError("PU2: No destination found!")

        retrieval_move = env.retrieve(source_stack)
        moves.append(retrieval_move)

    return moves
