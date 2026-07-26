"""
Regression / sanity checks for the Lee & Hsu (2007) network-flow IP
embedding. Not part of an automated test runner -- run directly with:

    cd Platform_CRISP
    python -m algorithms.CRP_Prem.exact.solver.lee_hsu_2007_network_flow_ip.test_lee_hsu

Requires Gurobi (gurobipy) with a valid license.
"""

from __future__ import annotations

from .core import (
    apply_moves,
    clone_stacks,
    count_bad_overlaps,
    is_fully_well_located,
)
from .ordering import order_segment_moves
from .solve import solve_lee_hsu


def check(name, stacks_init, max_tiers, **kwargs):
    print(f"\n=== {name} ===")
    print("initial:", stacks_init)
    res = solve_lee_hsu(stacks_init, max_tiers=max_tiers, **kwargs)
    print(f"solved={res.solved} proved_optimal={res.proved_optimal} T_used={res.T_used} "
          f"attempts={res.attempts} n_binaries={res.n_binaries} elapsed={res.elapsed_s:.2f}s "
          f"skipped={res.skipped_reason} ordering_failed={res.ordering_failed}")
    print("moves:", res.moves)
    final = apply_moves(stacks_init, res.moves, max_tiers)
    print("final:", final)
    bad = count_bad_overlaps(final)
    total_before = sum(len(v) for v in stacks_init.values())
    total_after = sum(len(v) for v in final.values())
    ok = (
        res.solved
        and bad == 0
        and is_fully_well_located(final)
        and total_before == total_after
    )
    print("independently verified OK:", ok)
    return ok


def check_ordering_cycle_break(name, stacks_init, moves, max_tiers):
    print(f"\n=== {name} (ordering.py unit test) ===")
    st = clone_stacks(stacks_init)
    total_before = sum(len(v) for v in st.values())
    ordered = order_segment_moves(st, moves, max_tiers)
    print("unordered moves in:", moves)
    print("ordered moves out:", ordered)
    print("resulting stacks:", st)
    total_after = sum(len(v) for v in st.values())
    ok = total_before == total_after and len(ordered) >= len(moves)
    print("OK:", ok)
    return ok


def run_all() -> bool:
    results = []

    # A 2-stack cycle: 0's top should go to 1, and (simultaneously, per the
    # unordered MIP solution) 1's top should go to 0. With no spare room in
    # either stack, this can only be executed by rerouting one leg through
    # a third, uninvolved stack.
    results.append(check_ordering_cycle_break(
        "2-cycle needs spare-stack reroute",
        {0: [9, 1], 1: [9, 2], 2: []},
        moves=[(0, 1), (1, 0)],
        max_tiers=2,
    ))

    # A 3-stack cycle: 0->1, 1->2, 2->0, none of which has room until the
    # cycle is broken.
    results.append(check_ordering_cycle_break(
        "3-cycle needs spare-stack reroute",
        {0: [9, 1], 1: [9, 2], 2: [9, 3], 3: []},
        moves=[(0, 1), (1, 2), (2, 0)],
        max_tiers=2,
    ))

    results.append(check(
        "already sorted (0 moves expected)",
        {0: [3, 1], 1: []}, max_tiers=3,
    ))

    results.append(check(
        "single blocking move",
        {0: [1, 2], 1: []}, max_tiers=3,
    ))

    results.append(check(
        "two independent single moves in parallel",
        {0: [1, 2], 1: [3, 4], 2: []}, max_tiers=3,
        max_moves_per_segment=3, cycle_breaking_level=3,
    ))

    results.append(check(
        "swap-like scenario needing temp stack",
        {0: [1, 3], 1: [2, 4], 2: []}, max_tiers=3,
    ))

    results.append(check(
        "3-stack fully mixed, small",
        {0: [2, 5], 1: [1, 4], 2: [3]}, max_tiers=3,
    ))

    results.append(check(
        "deeper stack requiring multiple relocations",
        {0: [4, 1, 3, 2], 1: [], 2: []}, max_tiers=4,
        t_start=4, t_max=10, per_solve_time_limit_s=20.0, overall_time_budget_s=90.0,
    ))

    results.append(check(
        "literal basic model (K=1, one move per segment)",
        {0: [1, 2], 1: [3, 4], 2: []}, max_tiers=3,
        max_moves_per_segment=1, cycle_breaking_level=0,
        t_start=2, t_max=8, t_step=1,
    ))

    print("\n=== coarsening: 6 distinct types capped to 3 (best-effort, not expected to fully solve) ===")
    res = solve_lee_hsu(
        {0: [1, 4], 1: [2, 5], 2: [3, 6]}, max_tiers=3,
        max_types=3, t_start=4, t_max=6, t_step=2, overall_time_budget_s=30.0,
    )
    print(f"n_types_coarsened_from={res.n_types_coarsened_from} solved={res.solved} "
          f"remaining_bad_overlaps={res.remaining_bad_overlaps} attempts={res.attempts}")
    # Coarsening is a documented approximation: full solve isn't guaranteed,
    # but it must complete (not hang) and report honestly against the
    # *original* (non-coarsened) priorities.
    results.append(res.n_types_coarsened_from == 6 and res.attempts > 0)
    print("OK:", results[-1])

    # --- extension: exact final layout (Eq. 27) --------------------------
    # Needs a spare stack: reversing 2 items within one stack's own slots
    # is infeasible with zero spare stacks (no 3rd parking spot).
    print("\n=== extension: exact final layout (Eq. 27) ===")
    stacks_init = {0: [1, 2], 1: [], 2: []}
    demand = {(0, 0): 2, (0, 1): 1, (1, 0): None, (2, 0): None}
    res = solve_lee_hsu(
        stacks_init, max_tiers=3,
        exact_final_layout_demand=demand,
    )
    final = apply_moves(stacks_init, res.moves, max_tiers=3)
    print("moves:", res.moves, "final:", final, "solved:", res.solved)
    demand_ok = final.get(0) == [2, 1] and not final.get(1) and not final.get(2)
    results.append(bool(res.solved and demand_ok))
    print("OK:", results[-1])

    # --- extension: one type per stack (Eq. 28) ---------------------------
    # Needs duplicate type values to be a meaningful (satisfiable) test:
    # group all type-1 containers into one stack, all type-2 into another.
    # A spare stack keeps this a realistic (non-pathological) instance --
    # like standard premarshalling benchmarks, some headroom is assumed.
    print("\n=== extension: one type per stack (Eq. 28) ===")
    stacks_init = {0: [1, 2], 1: [2, 1], 2: []}
    res = solve_lee_hsu(
        stacks_init, max_tiers=3,
        one_type_per_stack=True,
    )
    final = apply_moves(stacks_init, res.moves, max_tiers=3)
    print("moves:", res.moves, "final:", final, "solved:", res.solved)
    single_type = all(len(set(arr)) <= 1 for arr in final.values())
    results.append(bool(res.solved and is_fully_well_located(final) and single_type))
    print("OK:", results[-1])

    # --- size guard: an oversized T/type combo should be skipped, not hang
    print("\n=== size guard: tiny max_binaries forces skip ===")
    res = solve_lee_hsu(
        {0: [1, 2], 1: []}, max_tiers=3,
        max_binaries=10,
        t_start=4, t_max=6, t_step=2,
        overall_time_budget_s=10,
    )
    print("solved:", res.solved, "skipped_reason:", res.skipped_reason, "attempts:", res.attempts)
    results.append(bool(not res.solved and res.attempts == 0 and res.skipped_reason))
    print("OK:", results[-1])

    print("\n\nSUMMARY:", results, "ALL PASS" if all(results) else "SOME FAILED")
    return all(results)


if __name__ == "__main__":
    ok = run_all()
    raise SystemExit(0 if ok else 1)
