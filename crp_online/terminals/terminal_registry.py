from typing import Callable, Dict
from ..core.environment import Environment
from ..core.observation import ObservationState
from ..core.location import Location


TerminalFunc = Callable[[ObservationState, Environment, int, int, float], float]


def _term_sh(obs: ObservationState, env: Environment, s_src: int, s_cand: int, c_val: float) -> float:
    return float(obs.get_stack_height(s_cand))


def _term_avg(obs: ObservationState, env: Environment, s_src: int, s_cand: int, c_val: float) -> float:
    return float(obs.calculate_average_id(s_cand))


def _term_emp(obs: ObservationState, env: Environment, s_src: int, s_cand: int, c_val: float) -> float:
    return float(obs.max_stack_height - obs.get_stack_height(s_cand))


def _term_cur(obs: ObservationState, env: Environment, s_src: int, s_cand: int, c_val: float) -> float:
    return float(c_val)


def _term_ri(obs: ObservationState, env: Environment, s_src: int, s_cand: int, c_val: float) -> float:
    return float(obs.reshuffle_index(s_cand, c_val))


def _term_diff(obs: ObservationState, env: Environment, s_src: int, s_cand: int, c_val: float) -> float:
    return float(obs.get_stack_min_obs_value(s_cand) - c_val)


def _term_dur(obs: ObservationState, env: Environment, s_src: int, s_cand: int, c_val: float) -> float:
    return float(env.calculate_crane_time(Location(bay=0, stack=s_src), Location(bay=0, stack=s_cand)))


def _term_min(obs: ObservationState, env: Environment, s_src: int, s_cand: int, c_val: float) -> float:
    return float(obs.get_stack_min_obs_value(s_cand))


def _term_rem(obs: ObservationState, env: Environment, s_src: int, s_cand: int, c_val: float) -> float:
    return float(obs.containers_above_target(s_src))


def _term_next(obs: ObservationState, env: Environment, s_src: int, s_cand: int, c_val: float) -> float:
    if obs.mode == "paper_clean":
        return 0.0
    # C++ mode (Terminals.cpp line 107): returns top container if height > 1, else 0
    if obs.get_stack_height(s_cand) > 1:
        return float(obs.get_top_obs_value(s_cand))
    return 0.0


def _term_empty(obs: ObservationState, env: Environment, s_src: int, s_cand: int, c_val: float) -> float:
    return 1.0 if obs.get_stack_height(s_cand) == 0 else 0.0


def _term_wl(obs: ObservationState, env: Environment, s_src: int, s_cand: int, c_val: float) -> float:
    return float(obs.get_well_located_count(s_cand))


def _term_nl(obs: ObservationState, env: Environment, s_src: int, s_cand: int, c_val: float) -> float:
    return float(obs.get_bad_located_count(s_cand))


def _term_dsm(obs: ObservationState, env: Environment, s_src: int, s_cand: int, c_val: float) -> float:
    return float(obs.height_of_lower_index(s_cand, c_val))


def _term_nunc(obs: ObservationState, env: Environment, s_src: int, s_cand: int, c_val: float) -> float:
    return float(obs.get_number_unknown(s_cand))


def _term_nper(obs: ObservationState, env: Environment, s_src: int, s_cand: int, c_val: float) -> float:
    h = obs.get_stack_height(s_cand)
    denom = 1.0 if h == 0 else float(h)
    return float(obs.get_number_unknown(s_cand)) / denom


def _term_nrat(obs: ObservationState, env: Environment, s_src: int, s_cand: int, c_val: float) -> float:
    unk = obs.get_number_unknown(s_cand)
    denom = 1.0 if unk == 0 else float(unk)
    known = float(obs.get_stack_height(s_cand) - unk)
    return known / denom


def _term_ngrp(obs: ObservationState, env: Environment, s_src: int, s_cand: int, c_val: float) -> float:
    return float(obs.num_groups)


def _term_grps(obs: ObservationState, env: Environment, s_src: int, s_cand: int, c_val: float) -> float:
    return float(obs.group_size)


TERMINAL_REGISTRY: Dict[str, TerminalFunc] = {
    "SH": _term_sh,
    "AVG": _term_avg,
    "EMP": _term_emp,
    "CUR": _term_cur,
    "RI": _term_ri,
    "DIFF": _term_diff,
    "DUR": _term_dur,
    "MIN": _term_min,
    "REM": _term_rem,
    "NEXT": _term_next,
    "EMPTY": _term_empty,
    "WL": _term_wl,
    "NL": _term_nl,
    "DSM": _term_dsm,
    "NUNC": _term_nunc,
    "NPER": _term_nper,
    "NRAT": _term_nrat,
    "NGRP": _term_ngrp,
    "GRPS": _term_grps,
}


def evaluate_terminal(
    name: str,
    obs: ObservationState,
    env: Environment,
    source_stack: int,
    candidate_stack: int,
    moving_val: float,
) -> float:
    """Computes the numerical value for the specified terminal identifier."""
    if name not in TERMINAL_REGISTRY:
        raise KeyError(f"Unknown terminal: '{name}'")
    return TERMINAL_REGISTRY[name](obs, env, source_stack, candidate_stack, moving_val)
