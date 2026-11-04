"""
Sanity checks for KuArthanari2016Abstraction.

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
corresponding original algorithm paper.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import random

from . import combinatorics as comb
from .abstraction_core import (
    _confirmed_relocations,
    build_pdb,
    solve,
    state_from_stacks,
)


# ================================================================ #
#  1. Combinatorics vs paper's worked example (Table 2 / Section 4.2) #
# ================================================================ #

def check_combinatorics() -> bool:
    ok = True

    def expect(label, got, want):
        nonlocal ok
        status = "OK" if got == want else "FAIL"
        if got != want:
            ok = False
        print(f"[{status}] {label}: got={got} want={want}")

    expect("P(5,8,3) [Theorem 2, original state space]",
           comb.num_stacking_configurations(5, 8, 3), 6_249_600)
    expect("A(5,8,3) [Theorem 3, abstract state space]",
           comb.num_abstract_states(5, 8, 3), 62_160)

    breakdown = comb.abstract_state_breakdown(5, 8, 3)
    expect("breakdown[3 baseslots]", breakdown.get(3), 20_160)
    expect("breakdown[4 baseslots]", breakdown.get(4), 31_920)
    expect("breakdown[5 baseslots]", breakdown.get(5), 10_080)

    expect("nHr(4,9) [Definition 3]",
           comb.num_combinations_with_repetition(4, 9), 220)
    expect("g(1) [Lemma 1, n=4 r=9 m=3]",
           comb.overloaded_profile_count(4, 9, 3, 1), 224)
    expect("g(2) [Lemma 1, n=4 r=9 m=3]",
           comb.overloaded_profile_count(4, 9, 3, 2), 24)
    expect("C(4,9,3) [Theorem 1, stacking profiles]",
           comb.num_stacking_profiles(4, 9, 3), 20)

    return ok


# ================================================================ #
#  2. PDB vs independent brute-force B&B                             #
# ================================================================ #
#
# A canonical abstract state IS already a valid concrete layout (its
# columns hold distinct priorities 1..k) -- so each PDB entry can be
# checked directly against ``_brute_force_optimal`` (Section 3, an
# independent, non-abstracted exhaustive B&B; see part 3 below), with
# no separate re-implementation needed.

def check_pdb() -> bool:
    ok = True
    for n_cols, m_tiers, depth in [(2, 3, 5), (3, 2, 5), (3, 3, 6), (4, 2, 5)]:
        levels = build_pdb(depth, n_cols, m_tiers, time_limit_s=30.0)
        max_k = max(levels.keys())
        mismatches = 0
        checked = 0
        for k in range(1, max_k + 1):
            sample = list(levels[k].keys())[:60]  # sample for speed
            for st in sample:
                pdb_val = levels[k][st]
                # Pad with the implicit empty columns the canonical form
                # dropped -- the PDB was built with the *full* n_cols
                # budget available, not just the columns a given state
                # happens to occupy.
                padded = list(st) + [()] * (n_cols - len(st))
                bf_val = _brute_force_optimal(padded, k, m_tiers)
                checked += 1
                if pdb_val != bf_val:
                    mismatches += 1
                    if mismatches <= 3:
                        print(f"   !! mismatch n={n_cols} m={m_tiers} k={k} "
                              f"state={st} pdb={pdb_val} bf={bf_val}")
        status = "OK" if mismatches == 0 else "FAIL"
        if mismatches:
            ok = False
        print(f"[{status}] PDB n_cols={n_cols} m_tiers={m_tiers} depth={max_k} "
              f"checked={checked} mismatches={mismatches}")
    return ok


# ================================================================ #
#  3. solve() vs independent exhaustive B&B (random instances)       #
# ================================================================ #

def _random_instance(w, h, n, seed):
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
    """Small exhaustive B&B (independent of abstraction_core) for cross-checking."""
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

    def rec(state, target, g):
        state, target = reduce_(state, target)
        if target > n_total:
            best[0] = min(best[0], g)
            return
        lb = g + _confirmed_relocations(state)
        if lb >= best[0]:
            return
        ts = None
        for si, s in enumerate(state):
            if target in s:
                ts = si
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


def check_search() -> bool:
    ok = True
    configs = [(3, 3, 6), (3, 4, 8), (4, 3, 8), (3, 5, 10)]
    for w, h, n in configs:
        for seed in range(5):
            stacks = _random_instance(w, h, n, seed)
            optimal = _brute_force_optimal(stacks, n, h)

            pdb_levels = build_pdb(depth=4, n_cols=w, m_tiers=h, time_limit_s=10.0)

            plain = solve(stacks, n, h, pdb_levels={0: {(): 0}}, pdb_depth=0,
                          cache_depth=0, max_cache_size=1, time_limit_s=10.0)
            abstracted = solve(stacks, n, h, pdb_levels=pdb_levels, pdb_depth=4,
                                cache_depth=30, max_cache_size=100_000, time_limit_s=10.0)

            match = (plain["best"] == optimal and plain["optimal"]
                      and abstracted["best"] == optimal and abstracted["optimal"])
            status = "OK" if match else "FAIL"
            if not match:
                ok = False
            print(
                f"[{status}] W={w} H={h} N={n} seed={seed}  optimal={optimal}  "
                f"plain={plain['best']}  abstracted={abstracted['best']}  "
                f"cache_hits={abstracted['cache_hits']}  pdb_hits={abstracted['pdb_hits']}  "
                f"nodes(plain->abstracted)={plain['nodes']}->{abstracted['nodes']}"
            )
    return ok


def main():
    random.seed(0)
    r1 = check_combinatorics()
    r2 = check_pdb()
    r3 = check_search()
    print("\nALL OK" if (r1 and r2 and r3) else "\nSOME FAILURES")


if __name__ == "__main__":
    main()
