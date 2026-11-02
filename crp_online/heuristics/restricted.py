from typing import Callable, List, Optional, Union
from ..core.environment import Environment
from ..core.observation import ObservationState
from ..core.moves import ContainerMove
from ..evaluator.prefix_evaluator import PrefixExpression


DestinationSelector = Callable[[ObservationState, Environment, int, float], int]


def select_best_stack_pf(
    obs: ObservationState,
    env: Environment,
    source_stack: int,
    moving_val: float,
    expr: PrefixExpression,
    mode: str = "cpp_compat",
) -> int:
    """Evaluates candidate stacks using a prefix expression and returns the best stack index.
    
    Excludes source_stack and full stacks (< max_stack_height).
    Minimizes priority. Ties broken by earliest stack index (reproducing C++ line 54 behavior).
    """
    best_stack: Optional[int] = None
    best_priority: float = float("inf")

    for s in range(env.num_stacks):
        if s == source_stack or env.is_stack_full(s):
            continue

        priority = expr.evaluate(obs, env, source_stack, s, moving_val)
        if priority < best_priority:
            best_priority = priority
            best_stack = s

    if best_stack is None:
        raise RuntimeError(f"No legal destination stack available from source {source_stack}!")
    return best_stack


def solve_restricted(
    env: Environment,
    rule: Union[PrefixExpression, DestinationSelector],
    look_ahead_level: float = 1.0,
    horizon_strategy: str = "ignore",
    use_container_groups: bool = False,
    num_groups: int = 2,
    group_size: Optional[int] = None,
    known_within_group: bool = False,
    mode: str = "cpp_compat",
) -> List[ContainerMove]:
    """Executes Algorithm 1: Restricted Relocation Scheme.
    
    Relocates blocking containers above the target on its stack until target is retrieved.
    Repeats until all containers are retrieved.
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
            moving_val = obs.get_top_obs_value(source_stack)

            if isinstance(rule, PrefixExpression):
                dest_stack = select_best_stack_pf(obs, env, source_stack, moving_val, rule, mode)
            else:
                dest_stack = rule(obs, env, source_stack, moving_val)

            move = env.relocate(source_stack, dest_stack)
            moves.append(move)

        retrieval_move = env.retrieve(source_stack)
        moves.append(retrieval_move)

    return moves
