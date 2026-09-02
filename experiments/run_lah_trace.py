"""
Print Caserta 2009 LAH search metrics, initial yard, and every move.

Does not change the heuristic.  After train() finishes, a fresh CRP-R
environment replays the validated move list for display only.

Usage
-----
cd /data/liuw2/Platform_CRISP
PYTHONPATH=. python experiments/run_lah_trace.py
PYTHONPATH=. python experiments/run_lah_trace.py --file benchmark/Caserta_dataset/data3-4-4.dat
PYTHONPATH=. python experiments/run_lah_trace.py --n-restarts 200 --seed 0
"""

from __future__ import annotations

import argparse
from multiprocessing import Event
from pathlib import Path
from queue import Queue
from typing import Any, List

from core.base_algorithm import AlgorithmConfig
from core.benchmarks.caserta import problem_config_for_caserta_dat
from problems.CRP_R import CRP_R
from algorithms.CRP_R.heuristic.caserta_2009_lah.algorithm import CasertaLAH

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FILE = ROOT / "benchmark" / "Caserta_dataset" / "data3-4-4.dat"


def print_yard(env: CRP_R, title: str) -> None:
    print(f"\n==== {title} ====")
    keys = sorted(env.yard.stacks)
    print("stack:", "  ".join(f"S{i + 1}" for i in range(len(keys))))
    heights = [len(env.yard.stacks[k].containers) for k in keys]
    height = max(env.config.max_tiers, max(heights) if heights else 0)
    for tier in range(height - 1, -1, -1):
        cells = []
        for key in keys:
            prios = env.yard.stacks[key].priority_snapshot()
            cells.append(f"{prios[tier]:2d}" if tier < len(prios) else " .")
        print("     ", "  ".join(cells))
    print("(bottom is lowest row; . = empty)")


def replay_moves(env: CRP_R, moves: List[dict]) -> None:
    env.reset(options={"skip_auto_retrieve": True})
    print_yard(env, "initial layout (file as loaded)")

    env._finish_reset_after_layout_loaded()
    print_yard(env, "after opening auto-retrieves")

    env.reset(options={"skip_auto_retrieve": True})
    env._finish_reset_after_layout_loaded()
    for i, move in enumerate(moves, 1):
        cid = move["container_id"]
        frm = tuple(move["from"])
        to = None if move.get("to") is None else tuple(move["to"])
        if move.get("kind") == "retrieve" or to is None:
            got = env.yard.stacks[frm].pop()
            if got.id != cid:
                raise RuntimeError(
                    f"step {i}: expected C{cid} on top of {frm}, got C{got.id}"
                )
            print_yard(env, f"{i}. retrieve C{cid} from {frm}")
        else:
            env.yard.relocate(frm, to)
            print_yard(env, f"{i}. relocate C{cid}  {frm} -> {to}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace Caserta 2009 LAH on one Caserta file.")
    parser.add_argument(
        "--file",
        type=Path,
        default=DEFAULT_FILE,
        help="Caserta .dat path (default: data3-4-4.dat)",
    )
    parser.add_argument("--n-restarts", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    path = args.file.resolve()
    if not path.is_file():
        raise SystemExit(f"layout file not found: {path}")

    cfg = problem_config_for_caserta_dat(path, {})
    print(f"algorithm: Caserta et al. (2009) LAH")
    print(f"file: {path}")
    print(f"S={cfg.num_bays}  N={cfg.num_containers}  max_tiers={cfg.max_tiers}")
    print(f"n_restarts={args.n_restarts}  seed={args.seed}")
    print("layout file:\n" + path.read_text().rstrip())

    algo = CasertaLAH(AlgorithmConfig(seed=args.seed, extra={"n_restarts": args.n_restarts}))
    queue: Queue[Any] = Queue()
    algo.train(lambda: CRP_R(cfg), queue, Event())

    records = []
    while not queue.empty():
        records.append(queue.get())
    if not records:
        raise SystemExit("LAH produced no progress records")

    final = records[-1]
    solution = algo.get_best_solution()
    moves = final.extra.get("moves") or []

    print("\nsearch progress (best relocations so far):")
    for rec in records:
        print(f"  restart={rec.step}  best={rec.metric}")

    print("\nfinal metrics:")
    for key, value in sorted(final.metrics.items()):
        print(f"  {key}: {value}")
    print("validation_errors:", final.extra.get("validation_errors"))
    print("actions:", solution)

    replay_moves(CRP_R(cfg), moves)


if __name__ == "__main__":
    main()
