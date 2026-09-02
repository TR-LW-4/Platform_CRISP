"""
IS* iterative exact driver for LuZengLiu2019BRP.

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
corresponding original algorithm paper.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

from .lb4 import compute_lb4
from .brp_m3 import solve_brp_m3r, solve_brp_m3, minmax_ub
from .state import BRPState


# ═══════════════════════════════════════════════════════════════════════ #
#  Section 1 — direct-blockage count                                     #
# ═══════════════════════════════════════════════════════════════════════ #

def _direct_blockages(stacks: List[List[int]]) -> int:
    """
    Number of direct blockages: (lower, upper) pairs where upper is
    directly on top of lower with upper > lower (badly placed).
    """
    count = 0
    for stk in stacks:
        for k in range(len(stk) - 1):
            if stk[k + 1] > stk[k]:    # upper has lower priority → blockage
                count += 1
    return count


# ═══════════════════════════════════════════════════════════════════════ #
#  Section 2 — fast heuristics                                           #
# ═══════════════════════════════════════════════════════════════════════ #

def _greedy_step(
    state: BRPState,
    criterion: str = "blockages",
) -> Tuple[int, int]:
    """
    Perform one greedy relocation step minimising *criterion*:
      'blockages' — minimise direct blockages after move
      'lb4'       — minimise LB4 after move
    Returns (from_stack, to_stack).
    """
    W, H, n = state.W, state.H, state.n
    best_from = best_to = -1
    best_val = float("inf")

    # try all valid relocations (top of each non-empty stack → each non-full stack)
    for from_s in range(W):
        if not state.stacks[from_s]:
            continue
        item = state.top(from_s)
        for to_s in range(W):
            if to_s == from_s or state.height[to_s] >= H:
                continue
            # simulate
            trial = state.copy()
            trial.relocate(from_s, to_s)
            trial.auto_retrieve()

            if criterion == "lb4":
                val = compute_lb4(trial.stacks, trial.n, trial.W, trial.H)
            else:
                val = _direct_blockages(trial.stacks)

            if val < best_val:
                best_val = val
                best_from, best_to = from_s, to_s

    return best_from, best_to


def _fast_heuristic(
    stacks: List[List[int]],
    n: int,
    W: int,
    H: int,
    criterion: str = "blockages",
) -> List[Tuple[int, int]]:
    """
    Fast heuristic: greedily perform relocations minimising *criterion*
    until the yard is empty.

    Returns list of (from_stack, to_stack) relocation actions (0-indexed).
    """
    state = BRPState(stacks, n, H, W)
    state.auto_retrieve()
    ops: List[Tuple[int, int]] = []

    while not state.is_empty():
        from_s, to_s = _greedy_step(state, criterion)
        if from_s < 0:
            break
        state.relocate(from_s, to_s)
        ops.append((from_s, to_s))
        state.auto_retrieve()

    return ops


# ═══════════════════════════════════════════════════════════════════════ #
#  Section 3 — reparation heuristic                                      #
# ═══════════════════════════════════════════════════════════════════════ #

def _check_height_feasibility(
    ops: List[Tuple[int, int]],
    stacks: List[List[int]],
    n: int,
    W: int,
    H: int,
) -> bool:
    """
    Check whether the relocation sequence (from BRP-m3R without height limit)
    violates the stack height limit at any point.
    """
    state = BRPState(stacks, n, H, W)
    state.auto_retrieve()
    for from_s, to_s in ops:
        if state.height[to_s] >= H:
            return False
        state.relocate(from_s, to_s)
        state.auto_retrieve()
        if state.is_empty():
            break
    return True


def _repair_ops(
    ops: List[Tuple[int, int]],
    stacks: List[List[int]],
    n: int,
    W: int,
    H: int,
    criterion: str = "blockages",
) -> Optional[List[Tuple[int, int]]]:
    """
    Conservative reparation: for each relocation that would violate H,
    redirect it to a lower stack.  If conservative strategy fails at any
    turn, regenerate remaining moves with the fast heuristic.

    Returns repaired ops (possibly longer) or None if reparation fails to
    reduce to the same or fewer relocations.
    """
    state = BRPState(stacks, n, H, W)
    state.auto_retrieve()
    repaired: List[Tuple[int, int]] = []

    for from_s, to_s in ops:
        if state.is_empty():
            break

        if state.height[to_s] < H:
            state.relocate(from_s, to_s)
            repaired.append((from_s, to_s))
        else:
            # Conservative: find an alternative destination
            alt = -1
            for t in range(W):
                if t != from_s and state.height[t] < H:
                    if alt < 0 or state.height[t] < state.height[alt]:
                        alt = t
            if alt >= 0:
                state.relocate(from_s, alt)
                repaired.append((from_s, alt))
            else:
                # Aggressive: regenerate from here
                tail = _fast_heuristic(
                    state.stacks, state.n, state.W, state.H, criterion
                )
                repaired.extend(tail)
                return repaired

        state.auto_retrieve()

    return repaired


# ═══════════════════════════════════════════════════════════════════════ #
#  Section 4 — ops (BRP-m3R) to (from_stack, to_stack) list             #
# ═══════════════════════════════════════════════════════════════════════ #

def _brp_m3r_ops_to_stack_actions(
    m3r_ops: Optional[List[Tuple[int, int]]],
    stacks: List[List[int]],
    n: int,
    W: int,
    H: int,
) -> Optional[List[Tuple[int, int]]]:
    """
    Convert BRP-m3R ops (block-to-block adjacency pairs) to
    (from_stack, to_stack) relocation pairs, simulating through a BRPState.

    Returns None if ops is None.
    """
    if m3r_ops is None:
        return None

    # Build stack_for_item lookup
    sfm: Dict[int, int] = {}
    for s, stk in enumerate(stacks):
        for item in stk:
            sfm[item] = s

    actions: List[Tuple[int, int]] = []
    state = BRPState(stacks, n, H, W)
    state.auto_retrieve()

    for (i, j) in m3r_ops:
        if j < 0:
            continue  # retrieval marker
        # i is placed upon j: i was lifted from its stack to j's stack
        from_s = state.stack_for_item[i] if i <= n else -1
        # j's stack
        if j == n + 1:     # floor
            to_s = -1
            for s in range(W):
                if not state.stacks[s]:
                    to_s = s
                    break
        else:
            to_s = state.stack_for_item[j] if j <= n else -1

        if from_s >= 0 and to_s >= 0 and from_s != to_s:
            try:
                if state.stacks[from_s] and state.top(from_s) == i:
                    state.relocate(from_s, to_s)
                    actions.append((from_s, to_s))
                    state.auto_retrieve()
            except Exception:
                pass

    return actions if actions else None


# ═══════════════════════════════════════════════════════════════════════ #
#  Section 5 — IS (basic)                                                #
# ═══════════════════════════════════════════════════════════════════════ #

def run_is(
    stacks: List[List[int]],
    n: int,
    W: int,
    H: int,
    time_limit_s: float = 3600.0,
    output_flag: int = 0,
) -> Dict:
    """
    Basic IS algorithm (Algorithm IS in the paper).

    Uses LB4 as initial L.
    Iterates: solve BRP-m3R → update L → repeat until convergence.

    Returns dict with: obj, optimal, ops (list of (from_s, to_s)), iterations.
    """
    import time as _time

    t_start = _time.perf_counter()

    # Compute initial lower bound
    L = compute_lb4(stacks, n, W, H)
    best_ops: Optional[List[Tuple[int, int]]] = None
    best_obj = float("inf")
    iterations = 0

    while True:
        elapsed = _time.perf_counter() - t_start
        if elapsed >= time_limit_s:
            break

        remaining_t = time_limit_s - elapsed
        result = solve_brp_m3r(
            stacks, n, W, H, L,
            time_limit_s=remaining_t,
            output_flag=output_flag,
        )

        iterations += 1

        if result["obj"] is None:
            break

        L_new = L + result["obj"]   # L + remaining blockages

        if L_new == L:
            # convergence: optimal
            ops = _brp_m3r_ops_to_stack_actions(result["ops"], stacks, n, W, H)
            best_ops = ops
            best_obj = L_new
            break

        L = L_new

        # After enough iters, fall back to full BRP-m3
        if iterations >= 20:
            elapsed = _time.perf_counter() - t_start
            remaining_t = time_limit_s - elapsed
            full_result = solve_brp_m3(
                stacks, n, W, H, L, L,
                time_limit_s=remaining_t,
                output_flag=output_flag,
            )
            if full_result["obj"] is not None:
                ops = _brp_m3r_ops_to_stack_actions(full_result["ops"], stacks, n, W, H)
                best_ops = ops
                best_obj = full_result["obj"]
            break

    return dict(
        obj=int(best_obj) if best_obj < float("inf") else None,
        optimal=(best_obj < float("inf")),
        ops=best_ops,
        iterations=iterations,
        lower_bound=L,
    )


# ═══════════════════════════════════════════════════════════════════════ #
#  Section 6 — IS* (enhanced)                                            #
# ═══════════════════════════════════════════════════════════════════════ #

def run_is_star(
    stacks: List[List[int]],
    n: int,
    W: int,
    H: int,
    time_limit_s: float = 3600.0,
    output_flag: int = 0,
) -> Dict:
    """
    Enhanced IS* algorithm (Algorithm IS* in Appendix E).

    Enhancements over IS:
    (a) Fast heuristics to generate initial solutions (warm start Gurobi).
    (b) Solve BRP-m3R without height limit first; check height feasibility.
    (c) Reparation heuristic to convert solution to height-feasible one.
    (d) If reparation fails, re-solve BRP-m3R with height limit.

    Returns dict with: obj, optimal, ops, iterations, lower_bound.
    """
    import time as _time

    t_start = _time.perf_counter()

    L = compute_lb4(stacks, n, W, H)
    best_ops: Optional[List[Tuple[int, int]]] = None
    best_obj = float("inf")
    iterations = 0

    # ── Fast heuristics warm-start ────────────────────────────────── #
    for crit in ("blockages", "lb4"):
        h_ops = _fast_heuristic(stacks, n, W, H, crit)
        h_relocs = len(h_ops)
        if h_relocs < best_obj:
            best_obj = h_relocs
            best_ops = h_ops

    if best_obj <= L:
        # Heuristic already found optimal solution
        return dict(
            obj=int(best_obj),
            optimal=True,
            ops=best_ops,
            iterations=0,
            lower_bound=L,
        )

    # ── IS* main loop ─────────────────────────────────────────────── #
    while L < best_obj:
        elapsed = _time.perf_counter() - t_start
        if elapsed >= time_limit_s:
            break

        remaining_t = time_limit_s - elapsed

        # Phase 1: solve without height limit
        result_nolim = solve_brp_m3r(
            stacks, n, W, H_nolim := n,  # no height limit
            L,
            time_limit_s=min(remaining_t, time_limit_s / 2),
            output_flag=output_flag,
        )
        iterations += 1

        if result_nolim["obj"] is not None:
            L_new = L + result_nolim["obj"]

            # Update best if heuristic found something good
            if L_new < best_obj:
                # Try to convert to height-feasible solution
                nolim_ops = _brp_m3r_ops_to_stack_actions(
                    result_nolim["ops"], stacks, n, W, n
                )
                if nolim_ops is not None:
                    feasible = _check_height_feasibility(nolim_ops, stacks, n, W, H)
                    if feasible:
                        best_obj = L_new
                        best_ops = nolim_ops
                    else:
                        repaired = _repair_ops(nolim_ops, stacks, n, W, H)
                        if repaired is not None and len(repaired) <= best_obj:
                            best_obj = len(repaired)
                            best_ops = repaired
                        else:
                            # Phase 2: re-solve with height limit
                            elapsed2 = _time.perf_counter() - t_start
                            remaining2 = time_limit_s - elapsed2
                            if remaining2 > 0:
                                result_lim = solve_brp_m3r(
                                    stacks, n, W, H, L,
                                    time_limit_s=remaining2,
                                    output_flag=output_flag,
                                )
                                iterations += 1
                                if result_lim["obj"] is not None:
                                    L_new2 = L + result_lim["obj"]
                                    lim_ops = _brp_m3r_ops_to_stack_actions(
                                        result_lim["ops"], stacks, n, W, H
                                    )
                                    if L_new2 < best_obj and lim_ops is not None:
                                        best_obj = L_new2
                                        best_ops = lim_ops
                                    L_new = L_new2

            if L_new <= L:
                # Converged
                break
            L = L_new
        else:
            break

    return dict(
        obj=int(best_obj) if best_obj < float("inf") else None,
        optimal=(best_obj < float("inf")),
        ops=best_ops,
        iterations=iterations,
        lower_bound=L,
    )
