"""
Regression / sanity checks for the Tanaka & Tierney (2018) IDBB embedding.
Not part of an automated test runner -- run directly with:

    cd Platform_CRISP
    python -m algorithms.CRP_Prem.exact.search.tanaka_tierney_2018_idbb.test_idbb

Requires a C compiler (``gcc``/``make``) to build the vendored solver on
first use; no external MIP solver is needed for the embedding itself
(Gurobi is only used, if available, for the optional cross-checks against
this platform's other exact PMP embeddings).
"""

from __future__ import annotations

from .bridge import _encode_instance, ensure_binary_built, run_idbb
from .core import apply_moves, count_bad_overlaps, is_fully_well_located
from .solve import solve_idbb


def check(name, stacks_init, max_tiers, **kwargs):
    print(f"\n=== {name} ===")
    print("initial:", stacks_init)
    res = solve_idbb(stacks_init, max_tiers=max_tiers, **kwargs)
    print(f"solved={res.solved} proved_optimal={res.proved_optimal} "
          f"n_relocation={res.n_relocation} timed_out={res.timed_out} "
          f"elapsed={res.elapsed_s:.2f}s error={res.error}")
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
    return ok, res


def check_build() -> bool:
    print("\n=== vendored solver build ===")
    exe = ensure_binary_built()
    ok = exe.exists()
    print(f"binary at {exe}: exists={ok}")
    return ok


def check_instance_encoding_roundtrip() -> bool:
    """The C program's own parsing (bottom-to-top, 1-indexed stacks) must
    exactly match this wrapper's assumptions -- verified here by running
    the raw vendored binary and checking a hand-worked single-move case."""
    print("\n=== instance encoding: single blocking move, hand-verified ===")
    stacks = {0: [1, 2], 1: []}
    text = _encode_instance(stacks, max_tiers=3, stack_order=[0, 1])
    print("encoded instance:\n" + text)
    expected = "Tiers: 3\nStacks: 2\nContainers: 2\nStack 1: 1 2\nStack 2: \n"
    ok_encoding = text == expected
    print("encoding matches expected literal text:", ok_encoding)

    result = run_idbb(stacks, max_tiers=3, time_limit_s=5.0)
    print(f"moves={result.moves} solved={result.solved} n_relocation={result.n_relocation}")
    # Container with priority 2 sits on top of the more-urgent priority 1 in
    # stack 0 -- the only possible fix is relocating it (0 -> 1).
    ok_result = result.moves == [(0, 1)] and result.n_relocation == 1 and result.proved_optimal
    print("OK:", ok_encoding and ok_result)
    return ok_encoding and ok_result


def check_empty_and_trivial_instances() -> bool:
    print("\n=== trivial instances (no subprocess should be needed) ===")
    results = []

    res = solve_idbb({}, max_tiers=3)
    print(f"empty bay: solved={res.solved} moves={res.moves}")
    results.append(res.solved and res.moves == [])

    res = solve_idbb({0: [], 1: []}, max_tiers=3)
    print(f"all-empty stacks: solved={res.solved} moves={res.moves}")
    results.append(res.solved and res.moves == [])

    res = solve_idbb({0: [3, 1], 1: []}, max_tiers=3)
    print(f"already sorted: solved={res.solved} moves={res.moves}")
    results.append(res.solved and res.moves == [])

    return all(results)


def cross_check_against_other_pmp_models() -> bool:
    """The optimal relocation count must match this platform's other
    independently-verified exact PMP embeddings (Lee & Hsu 2007; de Melo
    da Silva et al. 2018; Parreno-Torres et al. 2019's IPS6) on identical
    small instances -- these are separate papers/implementations, so
    agreement here is strong evidence of correctness on both sides."""
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

    instances = [
        ({0: [1, 2], 1: []}, 3),
        ({0: [1, 3], 1: [2, 4], 2: []}, 3),
        ({0: [2, 5], 1: [1, 4], 2: [3]}, 3),
        ({0: [4, 1, 3, 2], 1: [], 2: []}, 4),
        # NOTE: {0: [1, 2, 4], 1: [3]} (only 2 stacks) is a genuine PMP
        # instance with NO valid solution at all (confirmed via brute-force
        # BFS in the IPS6 embedding's own test suite) -- a 3-stack
        # (buffer-available) variant is used here instead.
        ({0: [1, 2, 4], 1: [3], 2: []}, 4),
    ]

    all_ok = True
    for stacks_init, max_tiers in instances:
        print(f"\n=== cross-check: {stacks_init} tiers={max_tiers} ===")
        res_idbb = solve_idbb(stacks_init, max_tiers=max_tiers, time_limit_s=30.0)
        n_idbb = len(res_idbb.moves) if res_idbb.solved else None
        print(f"IDBB: solved={res_idbb.solved} moves={n_idbb}")

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

        vals = [v for v in (n_idbb, n_de_melo, n_lee_hsu, n_ips6) if v is not None]
        ok = res_idbb.solved and len(set(vals)) <= 1
        print("agreement OK:", ok, "values:", vals)
        all_ok = all_ok and ok

    return all_ok


def run_all() -> bool:
    results = []

    results.append(check_build())
    results.append(check_instance_encoding_roundtrip())
    results.append(check_empty_and_trivial_instances())

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
        time_limit_s=20.0,
    )
    results.append(ok)

    ok, _ = check(
        "duplicate priorities across stacks",
        {0: [1, 2], 1: [2, 1], 2: []}, max_tiers=3,
    )
    results.append(ok)

    ok, _ = check(
        "larger 4-stack mixed instance",
        {0: [3, 1, 4], 1: [2, 5], 2: [6], 3: []}, max_tiers=4,
        time_limit_s=30.0,
    )
    results.append(ok)

    # --- genuinely unsolvable instance: only 2 stacks, no buffer -----------
    # Independently confirmed (see IPS6 embedding's own test suite) via
    # brute-force BFS over the true state graph that {0: [1, 2, 4], 1: [3]}
    # with max_tiers=4 has NO reachable sorted state at all (a real PMP
    # property, not a solver bug). The vendored solver must report this
    # quickly rather than hang.
    print("\n=== genuinely unsolvable instance (2 stacks, no buffer) ===")
    res = solve_idbb({0: [1, 2, 4], 1: [3]}, max_tiers=4, time_limit_s=10.0)
    print(f"solved={res.solved} n_relocation={res.n_relocation} timed_out={res.timed_out}")
    results.append(not res.solved)
    print("OK:", results[-1])

    # --- cross-check optimal move counts against the platform's other
    # independently-verified exact PMP embeddings ---------------------------
    results.append(cross_check_against_other_pmp_models())

    print("\n\nSUMMARY:", results, "ALL PASS" if all(results) else "SOME FAILED")
    return all(results)


if __name__ == "__main__":
    ok = run_all()
    raise SystemExit(0 if ok else 1)
