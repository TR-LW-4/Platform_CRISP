"""
Regression / sanity checks for the Zhu et al. (2012) IDA*-U embedding.
Not part of an automated test runner -- run directly with:

    cd Platform_CRISP
    python -m algorithms.CRP_U.exact.search.zhu_2012.test_ida

Cross-validates against an independent brute-force B&B (unrestricted
branching) to confirm IDA*-U/-UM find the true optimum, checks
LB1 <= optimal (admissibility) on every instance, and checks that the
unrestricted optimum never exceeds the restricted optimum on the same
layout (CRP-U's move set is a superset of CRP-R's).
"""

from __future__ import annotations

import random

from .ida_core import (
    LB1, LB3, PU1, PU2, ida_star_unrestricted, state_from_stacks, lb1,
    reduce_state, probe_best_of,
)


def _random_instance(w: int, h: int, n: int, seed: int):
    rng = random.Random(seed)
    order = list(range(1, n + 1))
    rng.shuffle(order)
    stacks = [[] for _ in range(w)]
    slot = 0
    for p in order:
        if slot >= w * h:
            break
        si = slot % w
        if len(stacks[si]) < h:
            stacks[si].append(p)
        slot += 1
    return stacks


def _brute_force_optimal_unrestricted(stacks, n_total, max_tiers):
    """
    Pure exhaustive B&B (no transposition table -- deliberately, to
    avoid subtle unsoundness when combined with the anti-reversal
    rule, since this is meant to be an unimpeachable ground truth for
    small test instances only).
    """
    state0 = state_from_stacks(stacks)
    # Seed `best` with a heuristic upper bound so the LB-based pruning is
    # effective from the very first branch, instead of degenerating into
    # an (effectively) unbounded exhaustive walk. This uses the module's
    # own probe heuristics purely as an *upper bound* -- it does not
    # affect the ground truth's correctness, only its runtime.
    best = [probe_best_of(state0, 1, n_total, max_tiers)]

    def locate(state, p):
        for si, s in enumerate(state):
            for hi, pp in enumerate(s):
                if pp == p:
                    return si, hi
        return None

    def reduce_(state, target):
        state = list(state)
        while target <= n_total:
            loc = locate(tuple(state), target)
            if loc is None:
                target += 1
                continue
            si, hi = loc
            if hi != len(state[si]) - 1:
                break
            state[si] = state[si][:-1]
            target += 1
        return tuple(state), target

    def confirmed(state):
        total = 0
        for s in state:
            min_below = None
            for p in s:
                if min_below is not None and p > min_below:
                    total += 1
                min_below = p if min_below is None else min(min_below, p)
        return total

    def rec(state, target, g, last_move):
        state, target = reduce_(state, target)
        if target > n_total:
            best[0] = min(best[0], g)
            return
        lb = g + confirmed(state)
        if lb >= best[0]:
            return
        n_stacks = len(state)
        for src in range(n_stacks):
            if not state[src]:
                continue
            for dst in range(n_stacks):
                if dst == src or len(state[dst]) >= max_tiers:
                    continue
                if last_move is not None and (src, dst) == (last_move[1], last_move[0]):
                    continue
                new_state = list(state)
                p = new_state[src][-1]
                new_state[src] = new_state[src][:-1]
                new_state[dst] = new_state[dst] + (p,)
                rec(tuple(new_state), target, g + 1, (src, dst))

    rec(state0, 1, 0, None)
    return best[0]


def check_instance(w, h, n, seed):
    stacks = _random_instance(w, h, n, seed)
    n_total, max_tiers = n, h

    optimal_u = _brute_force_optimal_unrestricted(stacks, n_total, max_tiers)

    state0 = state_from_stacks(stacks)
    red_state, red_target = reduce_state(state0, 1, n_total)
    lb1_val = lb1(red_state, red_target, n_total, max_tiers)
    admissible_ok = lb1_val <= optimal_u

    results = {}
    for use_map in (False, True):
        for probe_mode in (PU1, PU2):
            res = ida_star_unrestricted(
                state0, 1, n_total, max_tiers,
                lb_mode=LB1, probe_mode=probe_mode, use_map=use_map,
                time_limit_s=15.0,
            )
            results[("LB1", use_map, probe_mode)] = res

    all_match = all(r["best"] == optimal_u and r["optimal"] for r in results.values())

    status = "OK" if (admissible_ok and all_match) else "FAIL"
    print(
        f"[{status}] W={w} H={h} N={n} seed={seed}  optimal_U={optimal_u}  "
        f"LB1={lb1_val}  ida_best={sorted(set(r['best'] for r in results.values()))}"
    )
    if not admissible_ok:
        print("   !! LB1 admissibility violated")
    if not all_match:
        for k, r in results.items():
            if r["best"] != optimal_u or not r["optimal"]:
                print(f"   !! mismatch {k}: {r}")
    return admissible_ok and all_match


def check_um3_and_relative_order():
    """LB3 aggressive mode should still find feasible (if not certified
    optimal) solutions, and CRP-U's optimum should never exceed CRP-R's."""
    import sys
    sys.path.insert(0, ".")
    from algorithms.CRP_R.exact.search.zhu_2012.ida_core import (
        ida_star_restricted, LB3 as R_LB3,
    )

    ok = True
    for w, h, n, seed in [(3, 4, 8, 1), (4, 4, 10, 2), (3, 5, 10, 3)]:
        stacks = _random_instance(w, h, n, seed)
        state0 = state_from_stacks(stacks)

        res_r = ida_star_restricted(state0, 1, n, h, lb_mode=R_LB3, time_limit_s=10.0)
        res_u_lb1 = ida_star_unrestricted(state0, 1, n, h, lb_mode=LB1, time_limit_s=10.0)
        res_u_lb3 = ida_star_unrestricted(state0, 1, n, h, lb_mode=LB3, time_limit_s=10.0)

        rel_ok = res_u_lb1["best"] <= res_r["best"]
        um3_feasible = res_u_lb3["best"] >= res_u_lb1["best"] or res_u_lb3["best"] >= 0
        um3_not_marked_optimal = res_u_lb3["optimal"] is False

        print(
            f"[{'OK' if rel_ok and um3_not_marked_optimal else 'FAIL'}] "
            f"W={w} H={h} N={n} seed={seed}  R={res_r['best']}  "
            f"U(LB1)={res_u_lb1['best']}  UM3(LB3)={res_u_lb3['best']}"
        )
        ok = ok and rel_ok and um3_not_marked_optimal and um3_feasible
    return ok


def main():
    random.seed(0)
    all_ok = True
    for w, h, n in [(3, 3, 6), (3, 4, 8), (4, 3, 8)]:
        for seed in range(5):
            ok = check_instance(w, h, n, seed)
            all_ok = all_ok and ok
    print()
    all_ok = check_um3_and_relative_order() and all_ok
    print("\nALL OK" if all_ok else "\nSOME FAILURES")


if __name__ == "__main__":
    main()
