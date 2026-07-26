"""
Regression / sanity checks for the Tierney, Pacino & Voß (2017) A*/IDA*
embedding. Not part of an automated test runner -- run directly with:

    cd Platform_CRISP
    python -m algorithms.CRP_Prem.exact.search.tierney_pacino_voss_2017_astar_idastar.test_astar_idastar

No external solver or compiler is required -- both A* and IDA* are pure
Python graph search.
"""

from __future__ import annotations

import itertools

from .bounds import lb_direct, lb_emo
from .core import apply_move_sequence, is_sorted, mis_overlay_count
from .search import solve_astar, solve_idastar


def _verify(stacks_init, moves, max_tiers) -> bool:
    final = apply_move_sequence(stacks_init, moves, max_tiers)
    total_before = sum(len(v) for v in stacks_init.values())
    total_after = sum(len(v) for v in final.values())
    return is_sorted(final) and total_before == total_after


def check(name, stacks_init, max_tiers, backend="idastar", **kwargs):
    print(f"\n=== {name} [{backend}] ===")
    print("initial:", stacks_init)
    solve_fn = solve_astar if backend == "astar" else solve_idastar
    res = solve_fn(stacks_init, max_tiers=max_tiers, **kwargs)
    print(f"solved={res.solved} proved_optimal={res.proved_optimal} "
          f"moves={len(res.moves)} timed_out={res.timed_out} "
          f"expanded={res.expanded_nodes} elapsed={res.elapsed_s:.2f}s")
    print("moves:", res.moves)
    ok = res.solved and _verify(stacks_init, res.moves, max_tiers)
    print("independently verified OK:", ok)
    return ok, res


def check_trivial_instances() -> bool:
    print("\n=== trivial instances ===")
    results = []
    for backend in ("astar", "idastar"):
        for stacks_init, max_tiers in (
            ({}, 3),
            ({0: [], 1: []}, 3),
            ({0: [3, 1], 1: []}, 3),
        ):
            solve_fn = solve_astar if backend == "astar" else solve_idastar
            res = solve_fn(stacks_init, max_tiers=max_tiers)
            ok = res.solved and res.moves == [] and res.expanded_nodes == 0
            print(f"[{backend}] {stacks_init}: solved={res.solved} moves={res.moves} ok={ok}")
            results.append(ok)
    return all(results)


def check_unsolvable_instance() -> bool:
    print("\n=== genuinely unsolvable instance (2 stacks, no buffer) ===")
    # Independently confirmed unsolvable via brute-force BFS in the IPS6
    # embedding's own test suite: {0: [1, 2, 4], 1: [3]} with max_tiers=4.
    results = []
    for backend in ("astar", "idastar"):
        solve_fn = solve_astar if backend == "astar" else solve_idastar
        res = solve_fn({0: [1, 2, 4], 1: [3]}, max_tiers=4, time_limit_s=10.0)
        print(f"[{backend}] solved={res.solved} timed_out={res.timed_out}")
        results.append(not res.solved)
    return all(results)


def check_lower_bounds_admissible_and_ordered() -> bool:
    """The EMO bound must never exceed the true optimum, and (Section 4.3)
    should dominate the direct bound: lb_direct <= lb_emo <= optimal moves."""
    print("\n=== lower bound sanity (admissibility + domination) ===")
    instances = [
        ({0: [1, 2], 1: []}, 3),
        ({0: [1, 3], 1: [2, 4], 2: []}, 3),
        ({0: [2, 5], 1: [1, 4], 2: [3]}, 3),
        ({0: [4, 1, 3, 2], 1: [], 2: []}, 4),
    ]
    ok = True
    for stacks_init, max_tiers in instances:
        d = lb_direct(stacks_init)
        e = lb_emo(stacks_init, max_tiers)
        res = solve_idastar(stacks_init, max_tiers=max_tiers, time_limit_s=20.0)
        opt = len(res.moves) if res.solved else None
        good = (d <= e) and (opt is None or e <= opt)
        print(f"{stacks_init}: direct={d} emo={e} optimal={opt} ok={good}")
        ok = ok and good
    return ok


def check_astar_vs_idastar_agree() -> bool:
    """A* and IDA* must find the same optimal move count on every instance."""
    print("\n=== A* vs IDA* agreement ===")
    instances = [
        ({0: [1, 2], 1: []}, 3),
        ({0: [1, 3], 1: [2, 4], 2: []}, 3),
        ({0: [2, 5], 1: [1, 4], 2: [3]}, 3),
        ({0: [4, 1, 3, 2], 1: [], 2: []}, 4),
        ({0: [1, 2, 4], 1: [3], 2: []}, 4),
        ({0: [3, 1, 4], 1: [2, 5], 2: [6], 3: []}, 4),
    ]
    ok = True
    for stacks_init, max_tiers in instances:
        r_a = solve_astar(stacks_init, max_tiers=max_tiers, time_limit_s=30.0)
        r_i = solve_idastar(stacks_init, max_tiers=max_tiers, time_limit_s=30.0)
        n_a = len(r_a.moves) if r_a.solved else None
        n_i = len(r_i.moves) if r_i.solved else None
        good = r_a.solved and r_i.solved and n_a == n_i
        good = good and _verify(stacks_init, r_a.moves, max_tiers)
        good = good and _verify(stacks_init, r_i.moves, max_tiers)
        print(f"{stacks_init}: A*={n_a} IDA*={n_i} ok={good}")
        ok = ok and good
    return ok


def check_parameterizations_agree() -> bool:
    """All combinations of EMO/direct bound, unrelated/transitive rule modes
    and direction must still yield the *same* optimal move count -- the
    branching rules are pruning-only and must never change optimality."""
    print("\n=== parameterization ablation agreement (IDA*) ===")
    stacks_init, max_tiers = {0: [2, 5], 1: [1, 4], 2: [3]}, 3
    baseline = solve_idastar(stacks_init, max_tiers=max_tiers, time_limit_s=30.0)
    assert baseline.solved
    n_opt = len(baseline.moves)

    ok = True
    combos = itertools.product(
        (True, False),                     # use_emo
        ("off", "direct", "successive"),   # unrelated_mode
        ("off", "direct", "successive"),   # transitive_mode
        ("lt", "gt"),                       # direction
        (True, False),                      # empty_stack_symmetry
    )
    for use_emo, unrelated_mode, transitive_mode, direction, ess in combos:
        res = solve_idastar(
            stacks_init, max_tiers=max_tiers, time_limit_s=30.0,
            use_emo=use_emo, unrelated_mode=unrelated_mode,
            transitive_mode=transitive_mode, direction=direction,
            empty_stack_symmetry=ess,
        )
        good = res.solved and len(res.moves) == n_opt and _verify(stacks_init, res.moves, max_tiers)
        if not good:
            print(f"MISMATCH emo={use_emo} unrel={unrelated_mode} trans={transitive_mode} "
                  f"dir={direction} ess={ess}: solved={res.solved} moves={len(res.moves) if res.solved else None}")
        ok = ok and good
    print(f"all {sum(1 for _ in itertools.product((1,1),(1,1,1),(1,1,1),(1,1),(1,1)))} "
          f"parameterizations agree on optimum {n_opt}: {ok}")
    return ok


def cross_check_against_other_pmp_models() -> bool:
    """The optimal relocation count must match this platform's other
    independently-verified exact PMP embeddings (Lee & Hsu 2007; de Melo da
    Silva et al. 2018; Parreno-Torres et al. 2019's IPS6; Tanaka & Tierney
    2018's IDBB) on identical small instances."""
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

    try:
        from algorithms.CRP_Prem.exact.solver.parreno_torres_alvarez_valdes_ruiz_2019_ips_models.solve import solve_ips6
    except Exception as e:
        print(f"skip cross-check (ips6 unavailable): {e}")
        solve_ips6 = None

    try:
        from algorithms.CRP_Prem.exact.search.tanaka_tierney_2018_idbb.solve import solve_idbb
    except Exception as e:
        print(f"skip cross-check (tanaka_tierney_2018_idbb unavailable): {e}")
        solve_idbb = None

    instances = [
        ({0: [1, 2], 1: []}, 3),
        ({0: [1, 3], 1: [2, 4], 2: []}, 3),
        ({0: [2, 5], 1: [1, 4], 2: [3]}, 3),
        ({0: [4, 1, 3, 2], 1: [], 2: []}, 4),
        ({0: [1, 2, 4], 1: [3], 2: []}, 4),
    ]

    all_ok = True
    for stacks_init, max_tiers in instances:
        print(f"\n=== cross-check: {stacks_init} tiers={max_tiers} ===")
        res_ida = solve_idastar(stacks_init, max_tiers=max_tiers, time_limit_s=30.0)
        n_ida = len(res_ida.moves) if res_ida.solved else None
        print(f"IDA*: solved={res_ida.solved} moves={n_ida}")

        res_a = solve_astar(stacks_init, max_tiers=max_tiers, time_limit_s=30.0)
        n_a = len(res_a.moves) if res_a.solved else None
        print(f"A*: solved={res_a.solved} moves={n_a}")

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

        n_ips6 = None
        if solve_ips6 is not None:
            res_ips6 = solve_ips6(stacks_init, max_tiers=max_tiers, overall_time_budget_s=60.0)
            n_ips6 = len(res_ips6.moves) if res_ips6.solved else None
            print(f"IPS6: solved={res_ips6.solved} moves={n_ips6}")

        n_idbb = None
        if solve_idbb is not None:
            res_idbb = solve_idbb(stacks_init, max_tiers=max_tiers, time_limit_s=30.0)
            n_idbb = len(res_idbb.moves) if res_idbb.solved else None
            print(f"IDBB: solved={res_idbb.solved} moves={n_idbb}")

        vals = [v for v in (n_ida, n_a, n_de_melo, n_lee_hsu, n_ips6, n_idbb) if v is not None]
        ok = res_ida.solved and res_a.solved and len(set(vals)) <= 1
        print("agreement OK:", ok, "values:", vals)
        all_ok = all_ok and ok

    return all_ok


def run_all() -> bool:
    results = []

    results.append(check_trivial_instances())
    results.append(check_unsolvable_instance())
    results.append(check_lower_bounds_admissible_and_ordered())
    results.append(check_astar_vs_idastar_agree())
    results.append(check_parameterizations_agree())

    for backend in ("astar", "idastar"):
        ok, res = check("single blocking move", {0: [1, 2], 1: []}, max_tiers=3, backend=backend)
        results.append(ok and len(res.moves) == 1)

        ok, _ = check(
            "swap-like scenario needing temp stack",
            {0: [1, 3], 1: [2, 4], 2: []}, max_tiers=3, backend=backend,
        )
        results.append(ok)

        ok, _ = check(
            "duplicate priorities across stacks",
            {0: [1, 2], 1: [2, 1], 2: []}, max_tiers=3, backend=backend,
        )
        results.append(ok)

        ok, _ = check(
            "larger 4-stack mixed instance",
            {0: [3, 1, 4], 1: [2, 5], 2: [6], 3: []}, max_tiers=4, backend=backend,
            time_limit_s=30.0,
        )
        results.append(ok)

    results.append(cross_check_against_other_pmp_models())

    print("\n\nSUMMARY:", results, "ALL PASS" if all(results) else "SOME FAILED")
    return all(results)


if __name__ == "__main__":
    ok = run_all()
    raise SystemExit(0 if ok else 1)
