"""
LA-N look-ahead planner (Petering & Hussein, EJOR 2013).

Builds a complete RelocationPlan by running the LA-N algorithm on a
deep copy of the initial yard.  No randomness; deterministic for a
given (yard, N) pair.

Key concepts
------------
c*         : current target container (lowest remaining priority)
s*         : Stack(c*) = the stack that contains c*
Low(s)     : lowest-numbered (= earliest-retrieved) container in stack s
Stacks[a]  : set of stacks that contain one of the a lowest-priority
             remaining containers
Top[Q]     : top containers of stacks in set Q
Topr[Q]    : rth highest-numbered element in Top[Q]  (r=1 = highest)

Algorithm (Steps 0-6 from the paper)
-------------------------------------
Outer loop (step 1): auto-retrieve all targets already on top.
Inner single-move decision (steps 2-6):
  2. N' = min(N, remaining);  r = 1
  3. Shrink N' while Stacks[N'] covers ALL stacks OR every
     non-Stacks[N'] stack is at max height.
  4. Consider n = Topr[Stacks[N']] (iterate r = 1, 2, …):
       · n == Top(s*)  →  forced relocation (step 6)
       · else          →  check for good cleaning move (step 5)
  5. Good cleaning move exists iff
       E = {s | Low(s) > n_priority, Height(s) < mxHeight}  is non-empty
       AND n is NOT the lowest-priority container in its own stack.
     No good move → r += 1, back to step 4.
  6. Relocate n:
       D non-empty → stack in D with lowest Low(s)   (conserve good slots)
       D empty     → non-full stack with highest Low(s) (delay re-relocation)

Reference
---------
M.E.H. Petering, M.I. Hussein, "A new mixed integer program and extended
look-ahead heuristic algorithm for the block relocation problem",
European Journal of Operational Research 231 (2013) 120–130.
"""

from __future__ import annotations

import copy
import os
from typing import Callable, Dict, List, Optional, Set, Tuple

from core.layout_trace import trace_lan
from core.plan import Movement, RelocationPlan


def _yard_compact(sim) -> str:
    """Bottom→top priorities per nonempty stack (for tracing)."""
    parts: List[str] = []
    for pos in sorted(sim.stacks.keys()):
        stk = sim.stacks[pos]
        if stk.is_empty:
            continue
        pris = [c.priority for c in stk.containers]
        parts.append(f"{pos}:{pris}")
    return " | ".join(parts) if parts else "<empty>"


# ================================================================ #
#  Stack-level helpers                                               #
# ================================================================ #

def _low(stk) -> float:
    """
    Lowest-numbered (earliest-retrieved) container priority in a stack.
    Returns +inf for empty stacks so they sort as 'best' in fallback.
    """
    if stk.is_empty:
        return float("inf")
    return float(min(c.priority for c in stk.containers))


def _good_dsts(
    sim,
    n_priority: int,
    max_tiers:  int,
    exclude:    Tuple[int, int],
) -> List[Tuple[int, int]]:
    """
    D = { s | Low(s) > n_priority  AND  Height(s) < max_tiers }

    These are stacks where container n can be placed without ever needing
    to be moved again before it is finally retrieved.
    Empty stacks are excluded (after placing n, Low(s) = n, not > n).
    """
    result = []
    for pos, stk in sim.stacks.items():
        if pos == exclude:
            continue
        if stk.height >= max_tiers or stk.is_empty:
            continue
        if _low(stk) > n_priority:
            result.append(pos)
    return result


def _choose_dst(
    sim,
    D:          List[Tuple[int, int]],
    max_tiers:  int,
    exclude:    Tuple[int, int],
) -> Tuple[int, int]:
    """
    D non-empty → stack in D with LOWEST Low(s)  (conserve good destinations)
    D empty     → non-full stack with HIGHEST Low(s) (delay next re-relocation;
                   empty stacks count as Low = +inf = best choice)
    """
    if D:
        return min(D, key=lambda p: _low(sim.stacks[p]))

    # Fallback: any non-full, non-source stack
    candidates = [
        (pos, _low(stk))
        for pos, stk in sim.stacks.items()
        if pos != exclude and stk.height < max_tiers
    ]
    if not candidates:
        raise RuntimeError(
            f"No valid relocation destination (all stacks full or source={exclude}). "
            "This should not happen with a well-formed BRP instance."
        )
    return max(candidates, key=lambda x: x[1])[0]


# ================================================================ #
#  Algorithm helpers                                                 #
# ================================================================ #

def _stacks_of_n_lowest(
    sim,
    remaining:    List[int],
    priority_map: Dict[int, object],
    n:            int,
) -> Set[Tuple[int, int]]:
    """
    Stacks[n] = set of (bay, row) positions of stacks containing
    one of the n lowest-priority remaining containers.
    """
    result: Set[Tuple[int, int]] = set()
    for pri in remaining[:n]:
        c   = priority_map[pri]
        stk = sim._find_stack(c)
        if stk is not None:
            result.add((stk.bay, stk.row))
    return result


def _auto_retrieve(
    sim,
    remaining:    List[int],
    priority_map: Dict[int, object],
    plan:         RelocationPlan,
    log_move: Optional[Callable[[str], None]] = None,
) -> None:
    """
    Step 1: retrieve every target that is already on top of its stack.
    Modifies *remaining* and *sim* in place.
    """
    while remaining:
        target = priority_map[remaining[0]]
        stk    = sim._find_stack(target)
        if stk is None or stk.top != target:
            break
        from_pos = (stk.bay, stk.row)
        stk.pop()
        plan.add(Movement(container_id=target.id, from_pos=from_pos, to_pos=None))
        remaining.pop(0)
        if log_move is not None:
            nxt = remaining[0] if remaining else None
            log_move(
                f"RETRIEVE id={target.id} priority={target.priority} "
                f"from={from_pos} next_target_pri={nxt}"
            )


def _shrink_N_prime(
    sim,
    all_stacks:  Set[Tuple[int, int]],
    stacks_Np:   Set[Tuple[int, int]],
    max_tiers:   int,
    N_prime:     int,
) -> int:
    """
    Step 3: reduce N' until placing a container is feasible.

    Conditions for reduction:
      (a) Stacks[N'] == all stacks (no room outside target group)
      (b) Every stack NOT in Stacks[N'] is full
    """
    while N_prime > 0:
        non_Np = all_stacks - stacks_Np
        if stacks_Np >= all_stacks:          # (a)
            N_prime -= 1
        elif non_Np and all(sim.stacks[s].height >= max_tiers for s in non_Np):  # (b)
            N_prime -= 1
        else:
            break
    return N_prime


# ================================================================ #
#  Main planner                                                      #
# ================================================================ #

def build_lan_plan(
    yard,
    containers: list,
    N:          int,
    max_tiers:  int,
) -> RelocationPlan:
    """
    Build a complete RelocationPlan for *yard* using the LA-N algorithm.

    Parameters
    ----------
    yard       : initial Yard state (deep-copied internally)
    containers : list of Container objects for the episode
    N          : look-ahead parameter (1 = LA-1 ≈ basic LA; S-1 = maximum)
    max_tiers  : maximum stack height (mxHeight in the paper)

    Returns
    -------
    A feasible RelocationPlan that empties the yard in priority order.
    """
    sim  = copy.deepcopy(yard)
    plan = RelocationPlan()

    priority_map: Dict[int, object] = {c.priority: c for c in containers}
    remaining: List[int]            = sorted(priority_map.keys())
    all_stacks: Set[Tuple[int, int]] = set(sim.stacks.keys())

    trace_on = bool(os.environ.get("CRISP_TRACE_LAN"))
    seq: List[int] = [0]

    def log_move(msg: str) -> None:
        if not trace_on:
            return
        seq[0] += 1
        trace_lan(f"#{seq[0]} {msg} | stacks={_yard_compact(sim)}")

    if trace_on:
        trace_lan(
            f"build_lan_plan start N={N} max_tiers={max_tiers} "
            f"n_containers={len(containers)} | {_yard_compact(sim)}"
        )

    # Safety cap: at most C*(C+1) moves before declaring failure
    max_moves = len(containers) * (len(containers) + 1)
    move_count = 0

    while remaining and move_count < max_moves:

        # ── Step 1: auto-retrieve accessible targets ──────────────── #
        _auto_retrieve(sim, remaining, priority_map, plan, log_move=log_move)
        if not remaining:
            break

        # Target and its stack
        target     = priority_map[remaining[0]]
        target_stk = sim._find_stack(target)
        if target_stk is None:
            remaining.pop(0)
            continue
        s_star = (target_stk.bay, target_stk.row)

        if trace_on:
            trace_lan(
                f"--- outer-loop target_pri={target.priority} "
                f"s*={s_star} remaining_cnt={len(remaining)}"
            )

        # ── Step 2: N' = min(N, |remaining|) ─────────────────────── #
        N_prime   = min(N, len(remaining))
        stacks_Np = _stacks_of_n_lowest(sim, remaining, priority_map, N_prime)

        # ── Step 3: shrink N' if necessary ───────────────────────── #
        N_prime   = _shrink_N_prime(sim, all_stacks, stacks_Np, max_tiers, N_prime)
        N_prime   = max(N_prime, 1)   # always keep at least 1 (target's stack)
        stacks_Np = _stacks_of_n_lowest(sim, remaining, priority_map, N_prime)

        # ── Steps 4-6: one relocation decision ───────────────────── #

        # Collect tops of Stacks[N'], sorted descending by priority
        # (r = 1 = highest numbered = last to be retrieved = safest to move)
        tops: List[Tuple[Tuple[int, int], object]] = []
        for pos in stacks_Np:
            stk = sim.stacks[pos]
            if not stk.is_empty:
                tops.append((pos, stk.top))
        tops.sort(key=lambda x: -x[1].priority)

        moved = False
        for n_pos, n in tops:
            if n_pos == s_star:
                # ── Step 4 → Step 6: forced relocation of target blocker ── #
                D   = _good_dsts(sim, n.priority, max_tiers, n_pos)
                dst = _choose_dst(sim, D, max_tiers, n_pos)
                plan.add(Movement(container_id=n.id, from_pos=n_pos, to_pos=dst))
                sim.relocate(n_pos, dst)
                log_move(
                    f"RELOC target-blocker id={n.id} priority={n.priority} "
                    f"{n_pos}->{dst}"
                )
                moved = True
                break

            else:
                # ── Step 5: check for good cleaning move ─────────────── #
                E     = _good_dsts(sim, n.priority, max_tiers, n_pos)
                low_n = min(c.priority for c in sim.stacks[n_pos].containers)

                if not E or n.priority == low_n:
                    continue   # no good cleaning move → try next r

                # ── Step 6: execute cleaning move ────────────────────── #
                dst = _choose_dst(sim, E, max_tiers, n_pos)
                plan.add(Movement(container_id=n.id, from_pos=n_pos, to_pos=dst))
                sim.relocate(n_pos, dst)
                log_move(
                    f"RELOC cleaning id={n.id} priority={n.priority} "
                    f"{n_pos}->{dst}"
                )
                moved = True
                break

        if not moved:
            # Safety fallback: should not be reached in valid instances
            # (s* is always in Stacks[N'] so its top always triggers step 6)
            n = target_stk.top
            if n is not None:
                D   = _good_dsts(sim, n.priority, max_tiers, s_star)
                dst = _choose_dst(sim, D, max_tiers, s_star)
                plan.add(Movement(container_id=n.id, from_pos=s_star, to_pos=dst))
                sim.relocate(s_star, dst)
                log_move(
                    f"RELOC fallback id={n.id} priority={n.priority} "
                    f"{s_star}->{dst}"
                )
                move_count += 1
        else:
            move_count += 1

    if trace_on:
        trace_lan(
            f"build_lan_plan done n_trace_steps={seq[0]} "
            f"plan_moves={plan.num_moves()} remaining={remaining!r}"
        )

    return plan
