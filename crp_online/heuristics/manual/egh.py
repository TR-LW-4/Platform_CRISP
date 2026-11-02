from typing import List, Optional
from ...core.environment import Environment
from ...core.observation import ObservationState
from ...core.moves import ContainerMove
from ...config.defaults import INT_MAX_SENTINEL


def select_egh_destination(
    obs: ObservationState,
    env: Environment,
    source_stack: int,
    moving_val: float,
) -> int:
    """Selects best relocation destination using EGH tie-breaking."""
    best_eri = INT_MAX_SENTINEL
    best_location: Optional[int] = None
    best_min_s = float(INT_MAX_SENTINEL)

    for s in range(env.num_stacks):
        if s == source_stack or env.is_stack_full(s):
            continue

        min_label = float(INT_MAX_SENTINEL)
        count_eri = 0

        for h in range(obs.get_stack_height(s)):
            cid = obs.get_container_obs_value(s, h)
            if cid != -1:
                if cid < moving_val:
                    count_eri += 1
                if cid < min_label:
                    min_label = cid

        if count_eri < best_eri or (count_eri == best_eri and min_label > best_min_s):
            best_eri = count_eri
            best_location = s
            best_min_s = min_label

    if best_location is None:
        raise RuntimeError("No legal destination stack in EGH!")
    return best_location


def solve_egh(
    env: Environment,
    look_ahead_level: float = 1.0,
    horizon_strategy: str = "ignore",
    use_container_groups: bool = False,
    num_groups: int = 2,
    group_size: Optional[int] = None,
    known_within_group: bool = False,
    mode: str = "cpp_compat",
) -> List[ContainerMove]:
    """EGH (Expected Group Heuristic - Galle, 2018):
    Faithfully reproduces C++ ExpectedGroupHeuristic.cpp.
    Identifies all blocking containers above the target and sequentially relocates them
    minimizing expected reshuffles, breaking ties by largest minimum container ID (minLabel).
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
            c_val = obs.get_top_obs_value(source_stack)
            best_location = select_egh_destination(obs, env, source_stack, c_val)

            move = env.relocate(source_stack, best_location)
            moves.append(move)

        retrieval_move = env.retrieve(source_stack)
        moves.append(retrieval_move)

    return moves
