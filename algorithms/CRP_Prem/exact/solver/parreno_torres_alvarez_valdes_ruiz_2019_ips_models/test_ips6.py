"""
Regression / sanity checks for the Parreno-Torres, Alvarez-Valdes & Ruiz
(2019) IPS6 embedding. Not part of an automated test runner -- run directly
with:

    cd Platform_CRISP
    python -m algorithms.CRP_Prem.exact.solver.parreno_torres_alvarez_valdes_ruiz_2019_ips_models.test_ips6

Requires Gurobi (gurobipy) with a valid license.
"""

from __future__ import annotations

from .core import (
    apply_moves,
    blocking_containers_lower_bound,
    count_bad_overlaps,
    is_fully_well_located,
)
from .solve import solve_ips6


def check(name, stacks_init, max_tiers, **kwargs):
    print(f"\n=== {name} ===")
    print("initial:", stacks_init)
    res = solve_ips6(stacks_init, max_tiers=max_tiers, **kwargs)
    print(f"solved={res.solved} proved_optimal={res.proved_optimal} "
          f"lower_bound={res.lower_bound} n_time_points_used={res.n_time_points_used} "
          f"attempts={res.attempts} n_binaries={res.n_binaries} "
          f"elapsed={res.elapsed_s:.2f}s skipped={res.skipped_reason}")
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
    return ok, res


def check_lb_valid(name, stacks_init, expected=None):
    lb = blocking_containers_lower_bound(stacks_init)
    print(f"\n=== {name}: LB={lb} ===")
    ok = lb >= 0 and (expected is None or lb == expected)
    if expected is not None:
        print(f"expected={expected} OK={ok}")
    return ok


def cross_check_against_other_pmp_models() -> bool:
    """The optimal relocation count must match this platform's other two
    independently-verified exact PMP embeddings (Lee & Hsu 2007;
    de Melo da Silva et al. 2018) on identical small instances."""
    try:
        from algorithms.CRP_Prem.exact.solver.de_melo_silva_2018.solve import solve_pmp as solve_de_melo
    except Exception as e:
        print(f"skip cross-check (de_melo_silva_2018 unavailable): {e}")
        solve_de_melo = None

    try:
        from algorithms.CRP_Prem.exact.solver.lee_hsu_2007_network_flow_ip.solve import solve_lee_hsu
    except Exception as e:
        print(f"skip cross-check (lee_hsu_2007 unavailable): {e}")
        solve_lee_hsu = None

    instances = [
        ({0: [1, 2], 1: []}, 3),
        ({0: [1, 3], 1: [2, 4], 2: []}, 3),
        ({0: [2, 5], 1: [1, 4], 2: [3]}, 3),
        ({0: [4, 1, 3, 2], 1: [], 2: []}, 4),
        # NOTE: {0: [1, 2, 4], 1: [3]} (only 2 stacks) was tried here too, but
        # an independent BFS over the true state graph shows it is a genuine
        # PMP instance with NO valid solution at all (with only 2 stacks and
        # no spare buffer, the reachable-state graph from this layout never
        # includes a sorted state) -- i.e. all three solvers correctly
        # agreeing "unsolvable" on it would not be evidence of a bug, so a
        # 3-stack (buffer-available) variant is used below instead.
        ({0: [1, 2, 4], 1: [3], 2: []}, 4),
    ]

    all_ok = True
    for stacks_init, max_tiers in instances:
        print(f"\n=== cross-check: {stacks_init} tiers={max_tiers} ===")
        res_ips6 = solve_ips6(stacks_init, max_tiers=max_tiers, overall_time_budget_s=60.0)
        n_ips6 = len(res_ips6.moves) if res_ips6.solved else None
        print(f"IPS6: solved={res_ips6.solved} moves={n_ips6}")

        n_de_melo = None
        if solve_de_melo is not None:
            res_dm = solve_de_melo(stacks_init, max_tiers=max_tiers, overall_time_budget_s=60.0)
            n_de_melo = len(res_dm.moves) if res_dm.solved else None
            print(f"de Melo da Silva: solved={res_dm.solved} moves={n_de_melo}")

        n_lee_hsu = None
        if solve_lee_hsu is not None:
            res_lh = solve_lee_hsu(stacks_init, max_tiers=max_tiers, overall_time_budget_s=60.0)
            n_lee_hsu = len(res_lh.moves) if res_lh.solved else None
            print(f"Lee-Hsu: solved={res_lh.solved} moves={n_lee_hsu}")

        vals = [v for v in (n_ips6, n_de_melo, n_lee_hsu) if v is not None]
        ok = res_ips6.solved and len(set(vals)) <= 1
        print("agreement OK:", ok, "values:", vals)
        all_ok = all_ok and ok

    return all_ok


def run_all() -> bool:
    results = []

    # --- Section 6 lower bound sanity --------------------------------------
    results.append(check_lb_valid("already sorted", {0: [3, 1], 1: []}, expected=0))
    results.append(check_lb_valid("single blocker", {0: [1, 2], 1: []}, expected=1))
    results.append(check_lb_valid("nested blockers", {0: [1, 3, 2]}, expected=2))

    # --- Exact model: small instances --------------------------------------
    ok, _ = check("already sorted (0 moves expected)", {0: [3, 1], 1: []}, max_tiers=3)
    results.append(ok)

    ok, res = check("single blocking move", {0: [1, 2], 1: []}, max_tiers=3)
    results.append(ok and len(res.moves) == 1)

    ok, _ = check("swap-like scenario needing temp stack", {0: [1, 3], 1: [2, 4], 2: []}, max_tiers=3)
    results.append(ok)

    ok, _ = check("3-stack fully mixed, small", {0: [2, 5], 1: [1, 4], 2: [3]}, max_tiers=3)
    results.append(ok)

    ok, _ = check(
        "deeper stack requiring multiple relocations",
        {0: [4, 1, 3, 2], 1: [], 2: []}, max_tiers=4,
        per_solve_time_limit_s=20.0, overall_time_budget_s=90.0,
    )
    results.append(ok)

    ok, _ = check(
        "no transitive-move breaking (plain model)",
        {0: [1, 2], 1: [3, 4], 2: []}, max_tiers=3,
        transitive_move_breaking=False, fix_last_segment_vars=False,
    )
    results.append(ok)

    ok, _ = check(
        "continuous move vars (Proposition 1)",
        {0: [1, 2], 1: []}, max_tiers=3,
        continuous_move_vars=True,
    )
    results.append(ok)

    # --- Section 8 extensions: must never break feasibility / correctness --
    ok, _ = check(
        "extension: no empty/full stacks",
        {0: [1, 2], 1: [3, 4], 2: []}, max_tiers=3,
        no_empty_or_full_stacks=True,
    )
    results.append(ok)

    ok, _ = check(
        "extension: balance kappa=1",
        {0: [1, 2], 1: [3, 4], 2: []}, max_tiers=3,
        balance_kappa=1,
    )
    results.append(ok)

    ok, _ = check(
        "extension: stability bonus",
        {0: [2, 5], 1: [1, 4], 2: [3]}, max_tiers=3,
        stability_bonus=True,
    )
    results.append(ok)

    ok, _ = check(
        "extension: same-priority bonus",
        {0: [2, 5], 1: [1, 4], 2: [3]}, max_tiers=3,
        same_priority_bonus=True,
    )
    results.append(ok)

    # --- size guard: an oversized T/priority combo should be skipped, not hang
    print("\n=== size guard: tiny max_binaries forces skip ===")
    res = solve_ips6(
        {0: [1, 2, 4], 1: [3]}, max_tiers=4,
        max_binaries=10, overall_time_budget_s=10,
    )
    print("solved:", res.solved, "skipped_reason:", res.skipped_reason, "attempts:", res.attempts)
    results.append(bool(not res.solved and res.attempts == 0 and res.skipped_reason))
    print("OK:", results[-1])

    # --- genuinely unsolvable instance: only 2 stacks, no buffer -----------
    # Independently confirmed via brute-force BFS over the true state graph
    # that {0: [1, 2, 4], 1: [3]} with max_tiers=4 has NO reachable sorted
    # state at all (a real PMP property, not a solver bug). The model must
    # report this quickly as infeasible rather than hang or mis-report a
    # bogus solution.
    print("\n=== genuinely unsolvable instance (2 stacks, no buffer) ===")
    res = solve_ips6(
        {0: [1, 2, 4], 1: [3]}, max_tiers=4,
        per_solve_time_limit_s=5.0, overall_time_budget_s=20.0, max_t_attempts=6,
    )
    print(f"solved={res.solved} attempts={res.attempts} n_time_points_used={res.n_time_points_used}")
    results.append(not res.solved)
    print("OK:", results[-1])

    # --- priority coarsening: many distinct priorities capped to fewer -----
    print("\n=== coarsening: 6 distinct priorities capped to 3 ===")
    res = solve_ips6(
        {0: [1, 4], 1: [2, 5], 2: [3, 6]}, max_tiers=3,
        max_priorities=3, overall_time_budget_s=30.0,
    )
    print(f"n_priorities_coarsened_from={res.n_priorities_coarsened_from} solved={res.solved} "
          f"remaining_bad_overlaps={res.remaining_bad_overlaps} attempts={res.attempts}")
    results.append(res.n_priorities_coarsened_from == 6 and res.attempts > 0)
    print("OK:", results[-1])

    # --- cross-check optimal move counts against the platform's other two
    # independently-verified exact PMP embeddings -----------------------
    results.append(cross_check_against_other_pmp_models())

    print("\n\nSUMMARY:", results, "ALL PASS" if all(results) else "SOME FAILED")
    return all(results)


if __name__ == "__main__":
    ok = run_all()
    raise SystemExit(0 if ok else 1)
