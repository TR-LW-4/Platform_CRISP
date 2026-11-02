from typing import Callable, List, Optional, Union
from ..core.environment import Environment
from ..core.observation import ObservationState
from ..core.moves import ContainerMove
from ..evaluator.prefix_evaluator import PrefixExpression
from .restricted import select_best_stack_pf, DestinationSelector
from ..config.defaults import INT_MAX_SENTINEL


def solve_unrestricted(
    env: Environment,
    rule: Union[PrefixExpression, DestinationSelector],
    look_ahead_level: float = 1.0,
    horizon_strategy: str = "ignore",
    use_container_groups: bool = False,
    num_groups: int = 2,
    known_within_group: bool = False,
    mode: str = "cpp_compat",
) -> List[ContainerMove]:
    """Executes Algorithm 2: Unrestricted Relocation Scheme with Preventive Relocations.
    
    Selects primary destination stack D using the priority function.
    Then performs preventive relocations from D if its top container is smaller
    than the relocating container, clearing D to receive the larger container well-placed.
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
                known_within_group=known_within_group,
                mode=mode,
            )
            moving_val = obs.get_top_obs_value(source_stack)

            if isinstance(rule, PrefixExpression):
                dest_stack = select_best_stack_pf(obs, env, source_stack, moving_val, rule, mode)
            else:
                dest_stack = rule(obs, env, source_stack, moving_val)

            # Preventive relocation loop on selected destination stack
            while True:
                if env.get_stack_height(dest_stack) == 0:
                    break

                sc_top = obs.get_top_obs_value(source_stack)
                dc_top = obs.get_top_obs_value(dest_stack)

                # Condition: dest top < source top
                if sc_top is not None and dc_top is not None and dc_top < sc_top:
                    best_s1: Optional[int] = None
                    lowest_index: float = float(INT_MAX_SENTINEL)

                    for s1 in range(env.num_stacks):
                        if s1 == source_stack or s1 == dest_stack or env.is_stack_full(s1):
                            continue

                        # Find lowest container index in s1
                        s1_lowest = obs.get_stack_min_obs_value(s1)

                        if s1_lowest > dc_top:
                            if mode == "cpp_compat":
                                if lowest_index > s1_lowest:
                                    lowest_index = s1_lowest
                                    best_s1 = s1
                                elif s1_lowest == float(INT_MAX_SENTINEL):
                                    # C++ empty stack quirk: sets lowest_index = 0
                                    lowest_index = 0.0
                                    best_s1 = s1
                            else:
                                if s1_lowest < lowest_index:
                                    lowest_index = s1_lowest
                                    best_s1 = s1
                                elif s1_lowest == lowest_index and best_s1 is not None:
                                    if env.get_stack_height(s1) < env.get_stack_height(best_s1):
                                        best_s1 = s1

                    if best_s1 is not None:
                        prev_move = env.relocate(dest_stack, best_s1)
                        moves.append(prev_move)
                        # Re-observe after preventive move
                        obs = ObservationState(
                            env,
                            look_ahead_level=look_ahead_level,
                            horizon_strategy=horizon_strategy,
                            use_container_groups=use_container_groups,
                            num_groups=num_groups,
                            known_within_group=known_within_group,
                            mode=mode,
                        )
                        continue

                break

            move = env.relocate(source_stack, dest_stack)
            moves.append(move)

        retrieval_move = env.retrieve(source_stack)
        moves.append(retrieval_move)

    return moves
