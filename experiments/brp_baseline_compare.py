"""
BRP-Fixed baseline comparison – relocations only.

Compares four heuristics on the same random instances:
  - Greedy (depth-1 look-ahead)
  - Kim–Hong (2006) ENAR
  - LA-1   (basic look-ahead, no cleaning moves)
  - LA-2   (2-step look-ahead with cleaning moves)
  - LA-(S-1) (maximum look-ahead)
  - Lee–Lee Phase-1 (greedy retrieval plan, before move-reduction)
  - Lee–Lee Full    (Phase 1 + Phase 2 move-reduction)

Output
------
  • Formatted table in the terminal
  • Optional CSV saved to results/brp_baseline_YYYYMMDD_HHMMSS.csv

Usage
-----
cd crp_platform
conda run -n rl python experiments/brp_baseline_compare.py
conda run -n rl python experiments/brp_baseline_compare.py --bays 1 --rows 5 --tiers 6 --containers 20 --seeds 20
conda run -n rl python experiments/brp_baseline_compare.py --help
"""

from __future__ import annotations

import argparse
import copy
import csv
import os
import sys
import time
from datetime import datetime
from typing import Dict, List

import numpy as np

# ── add project root to path ──────────────────────────────────────────── #
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.base_problem import ProblemConfig
from core.objectives import lower_bound_relocations
from core.plan import simulate_plan
from problems.CRP_R import BRPFixed

from algorithms._shared.heuristic.greedy.algorithm     import GreedyHeuristic
from algorithms.CRP_R.heuristic.kim_hong.scoring   import select_action as kim_select
from algorithms.CRP_R.heuristic.lan.planner        import build_lan_plan
from algorithms.CRP_Time.heuristic.lee_lee.phase1      import phase1_greedy
from algorithms.CRP_Time.heuristic.lee_lee.phase2      import phase2_reduce_moves


# ================================================================ #
#  Per-algorithm runners (direct, no multiprocessing overhead)      #
# ================================================================ #

def run_greedy(env: BRPFixed) -> Dict:
    """Depth-1 greedy via BRPFixed.step()."""
    import copy as _copy
    obs, info = env.reset()
    done = False
    while not done:
        mask   = info.get("action_mask")
        n      = env.action_space.n
        valid  = list(range(n)) if mask is None else list(np.where(mask)[0])
        if not valid:
            break
        best_a, best_r = valid[0], -float("inf")
        for a in valid:
            try:
                ec = _copy.deepcopy(env)
                _, r, _, _, _ = ec.step(a)
                if r > best_r:
                    best_r, best_a = r, a
            except Exception:
                break
        _, _, done, _, info = env.step(best_a)
    return env.get_metrics()


def run_kim_hong(env: BRPFixed) -> Dict:
    """Kim–Hong ENAR rule via BRPFixed.step()."""
    env.reset()
    done = False
    while not done:
        info   = env._get_info()
        action = kim_select(env, info.get("action_mask"))
        _, _, done, _, _ = env.step(action)
    return env.get_metrics()


def run_lan(
    env:       BRPFixed,
    initial_yard,
    containers: list,
    N:          int,
) -> Dict:
    """LA-N via RelocationPlan + evaluate_plan()."""
    plan = build_lan_plan(initial_yard, containers, N=N,
                          max_tiers=env.config.max_tiers)
    return env.evaluate_plan(plan)


def run_lee_lee_p1(
    env:        BRPFixed,
    initial_yard,
    containers: list,
) -> Dict:
    """Lee–Lee Phase 1 only (greedy plan, no move-reduction)."""
    plan = phase1_greedy(initial_yard, containers,
                         env.config.num_containers)
    return env.evaluate_plan(plan)


def run_lee_lee_full(
    env:        BRPFixed,
    initial_yard,
    containers: list,
    max_no_improve: int = 200,
) -> Dict:
    """Lee–Lee Phase 1 + Phase 2 (with move-reduction)."""
    plan = phase1_greedy(initial_yard, containers,
                         env.config.num_containers)
    plan, _ = phase2_reduce_moves(
        plan, initial_yard, containers,
        env.config.max_tiers,
        max_no_improve=max_no_improve,
    )
    return env.evaluate_plan(plan)


# ================================================================ #
#  Main comparison                                                   #
# ================================================================ #

def run_comparison(
    num_bays:        int   = 1,
    num_rows:        int   = 4,
    max_tiers:       int   = 5,
    num_containers:  int   = 15,
    n_seeds:         int   = 10,
    lee_lee_p2_iters: int  = 200,
    save_csv:        bool  = True,
) -> None:

    S = num_bays * num_rows   # total stacks

    algo_names = [
        "Greedy",
        "Kim–Hong",
        f"LA-1",
        f"LA-2",
        f"LA-(S-1) [N={max(1, S-1)}]",
        "Lee–Lee P1",
        f"Lee–Lee P1+P2 [{lee_lee_p2_iters}it]",
    ]

    # ── Header ──────────────────────────────────────────────────── #
    print()
    print("=" * 72)
    print("  BRP-Fixed Baseline Comparison  –  Relocations only")
    print("=" * 72)
    print(f"  Config : {num_bays} bay(s) × {num_rows} rows × H={max_tiers}"
          f"  |  C={num_containers}  |  capacity={S*max_tiers}"
          f"  ({num_containers/(S*max_tiers)*100:.0f}% full)")
    print(f"  Seeds  : {n_seeds}")
    print("=" * 72)

    all_relocs: Dict[str, List[float]] = {a: [] for a in algo_names}
    all_lb:     List[float] = []
    timing:     Dict[str, float] = {a: 0.0 for a in algo_names}

    for seed in range(n_seeds):
        cfg = ProblemConfig(
            num_bays=num_bays,
            num_rows=num_rows,
            max_tiers=max_tiers,
            num_containers=num_containers,
            seed=seed,
        )

        # Shared initial state for plan-based algorithms
        env_base = BRPFixed(cfg)
        env_base.reset()
        initial_yard = copy.deepcopy(env_base.yard)
        containers   = list(env_base.containers)
        lb           = lower_bound_relocations(initial_yard)
        all_lb.append(float(lb))

        # ── Greedy ─────────────────────────────────────────────── #
        env = BRPFixed(cfg)
        t0  = time.perf_counter()
        m   = run_greedy(env)
        timing["Greedy"] += time.perf_counter() - t0
        all_relocs["Greedy"].append(m["relocations"])

        # ── Kim–Hong ───────────────────────────────────────────── #
        env = BRPFixed(cfg)
        t0  = time.perf_counter()
        m   = run_kim_hong(env)
        timing["Kim–Hong"] += time.perf_counter() - t0
        all_relocs["Kim–Hong"].append(m["relocations"])

        # ── LA-1 ───────────────────────────────────────────────── #
        env = BRPFixed(cfg)
        t0  = time.perf_counter()
        m   = run_lan(env, initial_yard, containers, N=1)
        timing["LA-1"] += time.perf_counter() - t0
        all_relocs["LA-1"].append(m["relocations"])

        # ── LA-2 ───────────────────────────────────────────────── #
        env = BRPFixed(cfg)
        t0  = time.perf_counter()
        m   = run_lan(env, initial_yard, containers, N=2)
        timing["LA-2"] += time.perf_counter() - t0
        all_relocs["LA-2"].append(m["relocations"])

        # ── LA-(S-1) ───────────────────────────────────────────── #
        N_max = max(1, S - 1)
        key_la = f"LA-(S-1) [N={N_max}]"
        env = BRPFixed(cfg)
        t0  = time.perf_counter()
        m   = run_lan(env, initial_yard, containers, N=N_max)
        timing[key_la] += time.perf_counter() - t0
        all_relocs[key_la].append(m["relocations"])

        # ── Lee–Lee P1 ─────────────────────────────────────────── #
        env = BRPFixed(cfg)
        t0  = time.perf_counter()
        m   = run_lee_lee_p1(env, initial_yard, containers)
        timing["Lee–Lee P1"] += time.perf_counter() - t0
        all_relocs["Lee–Lee P1"].append(m["relocations"])

        # ── Lee–Lee P1+P2 ──────────────────────────────────────── #
        key_ll = f"Lee–Lee P1+P2 [{lee_lee_p2_iters}it]"
        env = BRPFixed(cfg)
        t0  = time.perf_counter()
        m   = run_lee_lee_full(env, initial_yard, containers,
                               max_no_improve=lee_lee_p2_iters)
        timing[key_ll] += time.perf_counter() - t0
        all_relocs[key_ll].append(m["relocations"])

        # Progress dot
        print(f"  seed {seed+1:3d}/{n_seeds}  lb={lb:.0f}", end="\r", flush=True)

    print(" " * 60, end="\r")  # clear progress line

    # ── Results table ───────────────────────────────────────────── #
    lb_mean = float(np.mean(all_lb))
    col_w   = 30

    header = (f"{'Algorithm':<{col_w}}"
              f"{'Mean':>7}{'±Std':>7}{'Min':>7}{'Max':>7}"
              f"{'LB-ratio':>10}{'Time(s)':>9}")
    sep    = "-" * len(header)

    print()
    print(header)
    print(sep)

    rows = []
    for name in algo_names:
        vals      = all_relocs[name]
        mean_r    = float(np.mean(vals))
        std_r     = float(np.std(vals))
        min_r     = float(np.min(vals))
        max_r     = float(np.max(vals))
        lb_ratio  = mean_r / max(lb_mean, 1)
        t_total   = timing[name]
        line = (f"{name:<{col_w}}"
                f"{mean_r:7.2f}{std_r:7.2f}{min_r:7.0f}{max_r:7.0f}"
                f"{lb_ratio:10.3f}{t_total:9.3f}")
        print(line)
        rows.append({
            "algorithm":  name,
            "mean":       round(mean_r, 3),
            "std":        round(std_r,  3),
            "min":        int(min_r),
            "max":        int(max_r),
            "lb_ratio":   round(lb_ratio, 4),
            "time_s":     round(t_total,  4),
        })

    print(sep)
    print(f"{'Lower bound (avg)':<{col_w}}{lb_mean:7.2f}")
    print()

    # ── Optional CSV ────────────────────────────────────────────── #
    if save_csv:
        results_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "results",
        )
        os.makedirs(results_dir, exist_ok=True)
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(results_dir, f"brp_baseline_{ts}.csv")

        # Summary CSV
        with open(csv_path, "w", newline="") as f:
            fieldnames = ["algorithm", "mean", "std", "min", "max",
                          "lb_ratio", "time_s"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        # Per-seed CSV
        detail_path = csv_path.replace(".csv", "_detail.csv")
        with open(detail_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["seed", "lower_bound"] + algo_names)
            for i in range(n_seeds):
                row = [i, all_lb[i]] + [all_relocs[a][i] for a in algo_names]
                writer.writerow(row)

        print(f"  Results saved:")
        print(f"    Summary : {csv_path}")
        print(f"    Per-seed: {detail_path}")
        print()


# ================================================================ #
#  CLI                                                               #
# ================================================================ #

def main():
    parser = argparse.ArgumentParser(
        description="BRP-Fixed baseline comparison (relocations only)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--bays",       type=int,  default=1,   help="Number of yard bays")
    parser.add_argument("--rows",       type=int,  default=4,   help="Stacks per bay (rows)")
    parser.add_argument("--tiers",      type=int,  default=5,   help="Max stack height")
    parser.add_argument("--containers", type=int,  default=15,  help="Number of containers")
    parser.add_argument("--seeds",      type=int,  default=10,  help="Number of random seeds")
    parser.add_argument("--p2-iters",   type=int,  default=200,
                        help="Lee–Lee Phase 2 max non-improving iterations")
    parser.add_argument("--no-csv",     action="store_true",    help="Skip CSV export")
    args = parser.parse_args()

    run_comparison(
        num_bays        = args.bays,
        num_rows        = args.rows,
        max_tiers       = args.tiers,
        num_containers  = args.containers,
        n_seeds         = args.seeds,
        lee_lee_p2_iters = args.p2_iters,
        save_csv        = not args.no_csv,
    )


if __name__ == "__main__":
    main()
