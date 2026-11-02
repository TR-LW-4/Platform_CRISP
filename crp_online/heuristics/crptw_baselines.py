"""Verified reference heuristics for the CRPTW benchmark matching Ku & Arthanari (2016) and Galle et al. (2018)."""

from typing import Callable, List, Optional, Sequence, Tuple, Union
import numpy as np
from ..core.crptw_environment import CRPTWEnvironment
from ..evaluator.prefix_evaluator import PrefixExpression


CRPTWSelector = Callable[[CRPTWEnvironment, int, int], int]


def select_eri_destination(env: CRPTWEnvironment, source_stack: int, moving_cid: int) -> int:
    """Expected Reshuffling Index (ERI) policy matching Galle et al. (2018) retrieveERI.m.
    
    Evaluates expected future reshuffles for each legal candidate stack:
    - In Unknown mode: +1.0 if window < moving_window, +0.5 if window == moving_window.
    - In Known mode: strict comparisons when both containers belong to active window.
    - Tie-breaking: Smallest ERI -> Largest stack height -> Leftmost stack.
    """
    moving_c = env.containers[moving_cid]
    moving_win = moving_c.window

    best_stack: Optional[int] = None
    best_eri = float("inf")
    best_height = -1

    for s in range(env.num_stacks):
        if s == source_stack or env.is_stack_full(s):
            continue

        eri = 0.0
        for cid in env.stacks[s]:
            c_win = env.containers[cid].window

            if env.variant == "known" and env.current_target is not None:
                target_win = env.containers[env.current_target].window
                if moving_win == target_win and c_win == target_win:
                    # Strict rank comparison in active window
                    order = env._departure_sequence
                    if order.index(cid) < order.index(moving_cid):
                        eri += 1.0
                else:
                    if c_win < moving_win:
                        eri += 1.0
                    elif c_win == moving_win:
                        eri += 0.5
            else:
                if c_win < moving_win:
                    eri += 1.0
                elif c_win == moving_win:
                    eri += 0.5

        h = env.get_stack_height(s)
        # Tie-break: lowest ERI, then higher stack, then leftmost
        if eri < best_eri:
            best_eri = eri
            best_height = h
            best_stack = s
        elif abs(eri - best_eri) < 1e-9:
            if h > best_height:
                best_eri = eri
                best_height = h
                best_stack = s

    if best_stack is None:
        raise RuntimeError(f"No legal destination stack available for moving container {moving_cid}")
    return best_stack


def select_em_destination(env: CRPTWEnvironment, source_stack: int, moving_cid: int) -> int:
    """Expected MinMax (EM) policy matching Galle et al. (2018) retrieveEM.m.
    
    Exact 4-tier decision logic:
    1. If a 'good move' is possible (exists candidate with min_window > moving_window):
       Select candidate with the smallest min_window > moving_window.
    2. If no good move is possible:
       Select candidate with the largest min_window <= moving_window.
    3. Tie-breaking tier 1: Fewest containers carrying that minimum window value.
    4. Tie-breaking tier 2: Largest stack height.
    5. Tie-breaking tier 3: Leftmost stack (earliest index).
    """
    moving_win = env.containers[moving_cid].window
    max_win = env.instance.num_batches + 1

    # Compute minimum window for each legal candidate stack
    candidates: List[dict] = []
    for s in range(env.num_stacks):
        if s == source_stack or env.is_stack_full(s):
            continue

        if env.is_stack_empty(s):
            min_w = max_win
            min_count = 0
        else:
            wins = [env.containers[cid].window for cid in env.stacks[s]]
            min_w = min(wins)
            min_count = wins.count(min_w)

        candidates.append({
            "stack": s,
            "min_w": min_w,
            "min_count": min_count,
            "height": env.get_stack_height(s),
        })

    if not candidates:
        raise RuntimeError(f"No legal destination stack available for moving container {moving_cid}")

    # Check if a 'good move' exists (min_w > moving_win)
    good_moves = [c for c in candidates if c["min_w"] > moving_win]

    if good_moves:
        # Pick stack with smallest min_w
        best_min_w = min(c["min_w"] for c in good_moves)
        filtered = [c for c in good_moves if c["min_w"] == best_min_w]
    else:
        # Pick stack with largest min_w
        best_min_w = max(c["min_w"] for c in candidates)
        filtered = [c for c in candidates if c["min_w"] == best_min_w]

    # Tie-breaking:
    # 1. Fewest containers with that minimum (min_count)
    # 2. Highest stack (descending height)
    # 3. Leftmost stack (ascending stack index)
    filtered.sort(key=lambda c: (c["min_count"], -c["height"], c["stack"]))
    return filtered[0]["stack"]


def select_leveling_destination(env: CRPTWEnvironment, source_stack: int, moving_cid: int) -> int:
    """Leveling / Shortest-Stack policy (TLP baseline).
    
    Selects the legal candidate stack with the lowest height. Ties broken by leftmost stack.
    """
    best_stack: Optional[int] = None
    best_height = float("inf")

    for s in range(env.num_stacks):
        if s == source_stack or env.is_stack_full(s):
            continue
        h = env.get_stack_height(s)
        if h < best_height:
            best_height = h
            best_stack = s

    if best_stack is None:
        raise RuntimeError(f"No legal destination stack available for moving container {moving_cid}")
    return best_stack


def select_random_destination(
    env: CRPTWEnvironment,
    source_stack: int,
    moving_cid: int,
    rng: np.random.Generator,
) -> int:
    """Uniform Random policy drawing strictly from an independent decoupled RNG stream."""
    legal_stacks = [s for s in range(env.num_stacks) if s != source_stack and not env.is_stack_full(s)]
    if not legal_stacks:
        raise RuntimeError(f"No legal destination stack available for moving container {moving_cid}")
    return int(rng.choice(legal_stacks))


def select_gp_destination(
    env: CRPTWEnvironment,
    source_stack: int,
    moving_cid: int,
    expr: PrefixExpression,
) -> int:
    """Evaluates candidate stacks using a GP PrefixExpression on the CRPTW ObservationState."""
    obs = env.get_observation_state()
    mock_env = env.create_mock_environment()
    moving_val = obs.get_top_obs_value(source_stack)

    best_stack: Optional[int] = None
    best_priority = float("inf")

    for s in range(env.num_stacks):
        if s == source_stack or env.is_stack_full(s):
            continue
        priority = expr.evaluate(obs, mock_env, source_stack, s, moving_val)
        if priority < best_priority:
            best_priority = priority
            best_stack = s

    if best_stack is None:
        raise RuntimeError(f"No legal destination stack available for moving container {moving_cid}")
    return best_stack


def solve_crptw_trial(
    env: CRPTWEnvironment,
    policy: Union[CRPTWSelector, PrefixExpression],
    rand_rng: Optional[np.random.Generator] = None,
    hybrid_leveling: bool = False,
) -> Tuple[int, int]:
    """Simulates a complete CRPTW trial until the yard is empty.
    
    Args:
        env: Prepared CRPTWEnvironment with departure sequence already set.
        policy: Destination selector callable or GP PrefixExpression.
        rand_rng: Dedicated RNG stream for the Random policy.
        hybrid_leveling: If True, switches to Leveling (L) when C_rem <= S.
        
    Returns:
        (total_relocations, endgame_relocations)
    """
    is_gp = isinstance(policy, PrefixExpression)

    while not env.is_empty():
        target = env.current_target
        if target is None:
            break
        source_stack = env.get_stack_of_container(target)

        while env.get_top_container(source_stack) != target:
            moving_cid = env.get_top_container(source_stack)

            # Check hybrid leveling trigger
            if hybrid_leveling and env.get_remaining_container_count() <= env.num_stacks:
                dest = select_leveling_destination(env, source_stack, moving_cid)
            elif is_gp:
                dest = select_gp_destination(env, source_stack, moving_cid, policy)
            elif policy == select_random_destination:
                if rand_rng is None:
                    raise ValueError("rand_rng must be provided for Random policy")
                dest = select_random_destination(env, source_stack, moving_cid, rand_rng)
            else:
                dest = policy(env, source_stack, moving_cid)

            env.relocate(source_stack, dest)

        env.retrieve(source_stack)

    return env.total_relocations, env.endgame_relocations
