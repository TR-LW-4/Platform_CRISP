"""
Regression / sanity checks for the de Melo da Silva et al. (2018) PMPm1
embedding. Not part of an automated test runner -- run directly with:

    cd Platform_CRISP
    python -m algorithms.CRP_Prem.exact.solver.de_melo_silva_2018.test_de_melo_silva_pmp

Requires Gurobi (gurobipy) with a valid license.
"""

from __future__ import annotations

from .core import (
    apply_moves,
    best_pmp_heuristic_upper_bound,
    count_bad_overlaps,
    is_fully_well_located,
    pmp_heuristic,
)
from .solve import solve_pmp


def check(name, stacks_init, max_tiers, **kwargs):
    print(f"\n=== {name} ===")
    print("initial:", stacks_init)
    res = solve_pmp(stacks_init, max_tiers=max_tiers, **kwargs)
    print(f"solved={res.solved} proved_optimal={res.proved_optimal} "
          f"heuristic_T={res.heuristic_T} T_used={res.T_used} attempts={res.attempts} "
          f"n_binaries={res.n_binaries} elapsed={res.elapsed_s:.2f}s skipped={res.skipped_reason}")
    print("moves:", res.moves, "objective:", res.objective)
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


def check_heuristic_is_valid_ub(name, stacks_init, max_tiers):
    print(f"\n=== {name} (Section 3.3 heuristic sanity) ===")
    relocs, moves = best_pmp_heuristic_upper_bound(stacks_init, max_tiers)
    final = apply_moves(stacks_init, moves, max_tiers)
    total_before = sum(len(v) for v in stacks_init.values())
    total_after = sum(len(v) for v in final.values())
    ok = (
        is_fully_well_located(final)
        and total_before == total_after
        and relocs == len(moves)
        and relocs >= 0
    )
    print(f"relocations={relocs} moves={moves} final={final} OK={ok}")
    return ok


def run_all() -> bool:
    results = []

    # --- Section 3.3 heuristic: must always produce a valid, fully sorted
    # upper bound regardless of which stack a run starts from.
    results.append(check_heuristic_is_valid_ub(
        "heuristic on a 3-stack mixed instance",
        {0: [2, 5], 1: [1, 4], 2: [3]}, max_tiers=4,
    ))
    results.append(check_heuristic_is_valid_ub(
        "heuristic on a single deep stack",
        {0: [4, 1, 3, 2], 1: [], 2: []}, max_tiers=4,
    ))
    results.append(check_heuristic_is_valid_ub(
        "heuristic already sorted (0 relocations expected)",
        {0: [3, 1], 1: []}, max_tiers=3,
    ))
    relocs0, _ = pmp_heuristic({0: [3, 1], 1: []}, max_tiers=3)
    results.append(relocs0 == 0)
    print("already-sorted heuristic gives 0 relocations:", relocs0 == 0)

    # --- Exact model: small instances comparable to the Lee & Hsu suite ---
    results.append(check(
        "already sorted (0 moves expected)",
        {0: [3, 1], 1: []}, max_tiers=3,
    ))

    results.append(check(
        "single blocking move",
        {0: [1, 2], 1: []}, max_tiers=3,
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
        per_solve_time_limit_s=20.0, overall_time_budget_s=90.0,
    ))

    results.append(check(
        "literal basic model (no strengthening constraints)",
        {0: [1, 2], 1: [3, 4], 2: []}, max_tiers=3,
        lifo_strengthening=False, flow_strengthening=False,
    ))

    # --- extension: exact final layout (Eq. 6) ----------------------------
    print("\n=== extension: exact final layout (Eq. 6) ===")
    stacks_init = {0: [1, 2], 1: [], 2: []}
    demand = {(0, 0): 2, (0, 1): 1, (1, 0): None, (2, 0): None}
    res = solve_pmp(
        stacks_init, max_tiers=3,
        use_exact_final_layout=True, exact_final_layout_demand=demand,
    )
    final = apply_moves(stacks_init, res.moves, max_tiers=3)
    print("moves:", res.moves, "final:", final, "solved:", res.solved)
    demand_ok = final.get(0) == [2, 1] and not final.get(1) and not final.get(2)
    results.append(bool(res.solved and demand_ok))
    print("OK:", results[-1])

    # --- size guard: an oversized T/group combo should be skipped, not hang
    print("\n=== size guard: tiny max_binaries forces skip ===")
    res = solve_pmp(
        {0: [1, 2, 4], 1: [3]}, max_tiers=4,
        max_binaries=10, overall_time_budget_s=10,
    )
    print("solved:", res.solved, "skipped_reason:", res.skipped_reason, "attempts:", res.attempts)
    results.append(bool(not res.solved and res.attempts == 0 and res.skipped_reason))
    print("OK:", results[-1])

    # --- group coarsening: many distinct priorities capped to fewer groups
    print("\n=== coarsening: 6 distinct priorities capped to 3 groups ===")
    res = solve_pmp(
        {0: [1, 4], 1: [2, 5], 2: [3, 6]}, max_tiers=3,
        max_types=3, overall_time_budget_s=30.0,
    )
    print(f"n_groups_coarsened_from={res.n_groups_coarsened_from} solved={res.solved} "
          f"remaining_bad_overlaps={res.remaining_bad_overlaps} attempts={res.attempts}")
    results.append(res.n_groups_coarsened_from == 6 and res.attempts > 0)
    print("OK:", results[-1])

    print("\n\nSUMMARY:", results, "ALL PASS" if all(results) else "SOME FAILED")
    return all(results)


if __name__ == "__main__":
    ok = run_all()
    raise SystemExit(0 if ok else 1)
