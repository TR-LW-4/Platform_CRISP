from typing import List, Optional
from ...core.environment import Environment
from ...core.observation import ObservationState
from ...config.defaults import INT_MAX_SENTINEL


def _shallowest_stack(env: Environment, candidates: List[int]) -> int:
    """Helper: chooses candidate stack with smallest height; ties to earliest index."""
    best = candidates[0]
    min_h = env.get_stack_height(best)
    for s in candidates[1:]:
        h = env.get_stack_height(s)
        if h < min_h:
            min_h = h
            best = s
    return best


def select_em_destination(
    obs: ObservationState,
    env: Environment,
    source_stack: int,
    moving_val: float,
) -> int:
    """EM (Expected Min-Max - Galle et al., 2018):
    Exact implementation of C++ ExpectedMiniMax.cpp.
    """
    top_container = moving_val
    lowest_index: float = float(INT_MAX_SENTINEL)
    rule1_candidates: List[int] = []

    # Rule 1 Candidates: minLabel > top_container
    for s in range(env.num_stacks):
        if s == source_stack or env.is_stack_full(s):
            continue

        min_label = obs.get_stack_min_obs_value(s)
        if min_label > top_container:
            if min_label < lowest_index:
                lowest_index = min_label
                rule1_candidates = [s]
            elif min_label == lowest_index:
                rule1_candidates.append(s)

    if rule1_candidates:
        return _shallowest_stack(env, rule1_candidates)

    # Rule 2 Candidates: max of minLabel
    m_val: float = -float(INT_MAX_SENTINEL)
    rule2_candidates: List[int] = []

    for s in range(env.num_stacks):
        if s == source_stack or env.is_stack_full(s):
            continue

        min_label = obs.get_stack_min_obs_value(s)
        if min_label > m_val:
            m_val = min_label
            rule2_candidates = [s]
        elif min_label == m_val:
            rule2_candidates.append(s)

    if not rule2_candidates:
        raise RuntimeError("No legal destination stack in Expected Min-Max!")

    if m_val == top_container:
        min_count_m = INT_MAX_SENTINEL
        best_with_min_m: List[int] = []

        for s in rule2_candidates:
            count_m = sum(
                1 for h in range(obs.get_stack_height(s))
                if obs.get_container_obs_value(s, h) == m_val
            )
            if count_m < min_count_m:
                min_count_m = count_m
                best_with_min_m = [s]
            elif count_m == min_count_m:
                best_with_min_m.append(s)

        return _shallowest_stack(env, best_with_min_m)
    else:
        return _shallowest_stack(env, rule2_candidates)
