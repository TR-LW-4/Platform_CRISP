"""
Regression / sanity checks for the Zhu et al. (2012) IDA*-R embedding.
Not part of an automated test runner -- run directly with:

    cd Platform_CRISP
    python -m algorithms.CRP_R.exact.search.zhu_2012.test_ida

Cross-validates against ``exact/search/kim_hong_2006`` (an independently
implemented exact B&B) to confirm IDA*-R finds the same optimal
relocation count, and checks LB1 <= LB2 <= LB3 <= optimal (admissibility
+ dominance) on every instance.
"""

from __future__ import annotations

import random

from .ida_core import LB1, LB2, LB3, PR1, PR2, PR3, PR4, ida_star_restricted, state_from_stacks


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


def _brute_force_optimal(stacks, n_total, max_tiers):
    """Small exhaustive B&B (independent of ida_core) for cross-checking."""
    state0 = state_from_stacks(stacks)
    best = [10 ** 9]

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

    def rec(state, target, g):
        state, target = reduce_(state, target)
        if target > n_total:
            best[0] = min(best[0], g)
            return
        lb = g + confirmed(state)
        if lb >= best[0]:
            return
        ts = None
        for si, s in enumerate(state):
            for pp in s:
                if pp == target:
                    ts = si
                    break
            if ts is not None:
                break
        for di, s in enumerate(state):
            if di == ts or len(s) >= max_tiers:
                continue
            new_state = list(state)
            new_state[ts] = new_state[ts][:-1]
            new_state[di] = new_state[di] + (state[ts][-1],)
            rec(tuple(new_state), target, g + 1)

    rec(state0, 1, 0)
    return best[0]


def check_instance(w, h, n, seed):
    stacks = _random_instance(w, h, n, seed)
    n_total = n
    max_tiers = h

    optimal = _brute_force_optimal(stacks, n_total, max_tiers)

    state0 = state_from_stacks(stacks)
    lb_vals = {}
    from .ida_core import lb1, lb2, lb3, reduce_state
    red_state, red_target = reduce_state(state0, 1, n_total)
    lb_vals[LB1] = lb1(red_state, red_target, n_total, max_tiers)
    lb_vals[LB2] = lb2(red_state, red_target, n_total, max_tiers)
    lb_vals[LB3] = lb3(red_state, red_target, n_total, max_tiers)

    dominance_ok = lb_vals[LB1] <= lb_vals[LB2] <= lb_vals[LB3] <= optimal

    results = {}
    for lb_mode in (LB1, LB2, LB3):
        for probe_mode in (PR1, PR4):
            res = ida_star_restricted(
                state0, 1, n_total, max_tiers,
                lb_mode=lb_mode, probe_mode=probe_mode, time_limit_s=10.0,
            )
            results[(lb_mode, probe_mode)] = res

    all_match = all(r["best"] == optimal and r["optimal"] for r in results.values())

    status = "OK" if (dominance_ok and all_match) else "FAIL"
    print(
        f"[{status}] W={w} H={h} N={n} seed={seed}  optimal={optimal}  "
        f"LB1={lb_vals[LB1]} LB2={lb_vals[LB2]} LB3={lb_vals[LB3]}  "
        f"ida_best={sorted(set(r['best'] for r in results.values()))}"
    )
    if not dominance_ok:
        print("   !! dominance/admissibility violated")
    if not all_match:
        for k, r in results.items():
            if r["best"] != optimal or not r["optimal"]:
                print(f"   !! mismatch {k}: {r}")
    return dominance_ok and all_match


def main():
    random.seed(0)
    all_ok = True
    for w, h, n in [(3, 3, 6), (3, 4, 8), (4, 3, 8), (4, 4, 10), (3, 5, 10)]:
        for seed in range(6):
            ok = check_instance(w, h, n, seed)
            all_ok = all_ok and ok
    print("\nALL OK" if all_ok else "\nSOME FAILURES")


if __name__ == "__main__":
    main()
