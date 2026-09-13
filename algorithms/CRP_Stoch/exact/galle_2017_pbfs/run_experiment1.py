#!/usr/bin/env python3
"""
Reproduce Galle et al. Experiment 1 (batch-model exact PBFS).

Mirrors ``StochasticCRP-master/Experiments_1.m``:

    for T in 3..6:
        for S in 5..10:
            for instance in 1..30:
                read Ku/Arthanari ``crptw_instance`` txt
                bounds + (optional) 5 heuristics + PBFSA(..., error_gap=0)

``python main.py run`` / ``layout-run`` cannot do this: they generate a
random CRP-Stoch yard (or only accept Caserta/Zhu layouts), they do not
scan the 1440 ``T271014_*.txt`` / ``T281014_*.txt`` files.

Examples (run from ``Platform_CRISP``)::

    python algorithms/CRP_Stoch/exact/galle_2017_pbfs/run_experiment1.py --smoke
    python algorithms/CRP_Stoch/exact/galle_2017_pbfs/run_experiment1.py \\
        --fill-rate 0.5 --stacks 5 --tiers 3
    python algorithms/CRP_Stoch/exact/galle_2017_pbfs/run_experiment1.py \\
        --fill-rate 0.5
    python algorithms/CRP_Stoch/exact/galle_2017_pbfs/run_experiment1.py \\
        --fill-rates 0.5 0.67 --heuristics
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


def _ensure_platform_on_path() -> Path:
    platform_root = Path(__file__).resolve().parents[4]
    if str(platform_root) not in sys.path:
        sys.path.insert(0, str(platform_root))
    return platform_root


_PLATFORM_ROOT = _ensure_platform_on_path()

from core.stoch.galle_2017_source import (  # noqa: E402
    HEURISTIC_IDS,
    blocking_lower_bound,
    pbfsa,
    read_input_file,
    rolling_lower_bound,
    run_heuristic_batch,
)


DEFAULT_DATA_ROOT = Path("/data/liuw2/StochasticCRP-master")
DEFAULT_GOLD_DIR = DEFAULT_DATA_ROOT / "Results" / "Experiments_1"
DEFAULT_OUT_DIR = _PLATFORM_ROOT / "results" / "CRP-Stoch" / "galle_experiment1"
CSV_FIELDS = ["Rows", "b", "b_1", "b_2", "opt", "time", "EG", "EM", "ERI", "L", "Rand."]
FINAL_FIELDS = ["S", "T", "C", "b", "b_1", "b_2", "opt", "time", "EG", "EM", "ERI", "L", "Rand."]
OPT_COMPARE_ATOL = 1e-4


@dataclass
class InstanceResult:
    instance: int
    b: float
    b_1: float
    b_2: float
    opt: float
    time: float
    eg: float
    em: float
    eri: float
    l_heur: float
    rand: float
    nodes_expanded: int
    cache_hits: int

    def as_csv_row(self) -> dict:
        return {
            "Rows": str(self.instance),
            "b": self.b,
            "b_1": self.b_1,
            "b_2": self.b_2,
            "opt": self.opt,
            "time": self.time,
            "EG": self.eg,
            "EM": self.em,
            "ERI": self.eri,
            "L": self.l_heur,
            "Rand.": self.rand,
        }


def _utilization_tag(fill_rate: float) -> str:
    return str(int(round(100 * fill_rate)))


def size_csv_name(stacks: int, tiers: int, fill_rate: float) -> str:
    return f"{stacks:02d}S_{tiers:02d}T_{_utilization_tag(fill_rate)}Utilization.csv"


def final_csv_name(fill_rate: float) -> str:
    return f"{_utilization_tag(fill_rate)}Utilization_FinalResults.csv"


def parse_int_list(spec: str) -> List[int]:
    """Parse ``'3-6'``, ``'5,6,8'``, or ``'1-3,10'``."""
    out: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(part))
    return out


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=float))) if values else float("nan")


def _nan() -> float:
    return float("nan")


def solve_instance(
    data_root: Path,
    stacks: int,
    tiers: int,
    instance: int,
    fill_rate: float,
    *,
    time_limit_s: float,
    seed: int,
    heuristics: bool,
    n_samples: int,
) -> InstanceResult:
    bay = read_input_file(data_root, stacks, tiers, instance, fill_rate)
    b = blocking_lower_bound(bay)
    b_1 = b + rolling_lower_bound(bay, 1)
    b_2 = b + rolling_lower_bound(bay, 2)
    rng = np.random.RandomState(seed)
    obj, stats, elapsed = pbfsa(
        bay,
        lower_bound_type=1,
        error_gap=0.0,
        time_limit_s=time_limit_s,
        rng=rng,
    )
    heur = {name: _nan() for name in HEURISTIC_IDS}
    if heuristics:
        for name, hid in HEURISTIC_IDS.items():
            heur[name] = run_heuristic_batch(
                bay, hid, n_samples, rng=np.random.RandomState(seed + hid)
            )
    return InstanceResult(
        instance=instance,
        b=b,
        b_1=b_1,
        b_2=b_2,
        opt=float(obj),
        time=float(elapsed),
        eg=heur["EG"],
        em=heur["EM"],
        eri=heur["ERI"],
        l_heur=heur["L"],
        rand=heur["Rand"],
        nodes_expanded=int(stats.get("nodes_expanded", 0)),
        cache_hits=int(stats.get("cache_hits", 0)),
    )


def _write_size_csv(path: Path, rows: Sequence[InstanceResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    avg = {
        "Rows": "Average",
        "b": _mean([r.b for r in rows]),
        "b_1": _mean([r.b_1 for r in rows]),
        "b_2": _mean([r.b_2 for r in rows]),
        "opt": _mean([r.opt for r in rows]),
        "time": _mean([r.time for r in rows]),
        "EG": _mean([r.eg for r in rows]),
        "EM": _mean([r.em for r in rows]),
        "ERI": _mean([r.eri for r in rows]),
        "L": _mean([r.l_heur for r in rows]),
        "Rand.": _mean([r.rand for r in rows]),
    }
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerow(avg)
        for row in rows:
            writer.writerow(row.as_csv_row())


def _append_progress(path: Path, result: InstanceResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["instance", *CSV_FIELDS[1:], "nodes_expanded", "cache_hits"],
        )
        if write_header:
            writer.writeheader()
        payload = result.as_csv_row()
        payload["instance"] = result.instance
        payload.pop("Rows", None)
        payload["nodes_expanded"] = result.nodes_expanded
        payload["cache_hits"] = result.cache_hits
        writer.writerow(payload)


def _load_done_instances(progress_path: Path) -> set:
    if not progress_path.exists():
        return set()
    done = set()
    with progress_path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                done.add(int(row["instance"]))
            except (KeyError, ValueError):
                continue
    return done


def load_gold_opts(
    gold_dir: Path, stacks: int, tiers: int, fill_rate: float
) -> dict:
    path = gold_dir / size_csv_name(stacks, tiers, fill_rate)
    if not path.exists():
        return {}
    out = {}
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            key = str(row.get("Rows", "")).strip()
            if key in {"", "Average"}:
                continue
            try:
                out[int(key)] = float(row["opt"])
            except (KeyError, ValueError):
                continue
    return out


def _finite(x: float) -> bool:
    return math.isfinite(x)


def compare_opt(got: float, gold: float) -> str:
    if not _finite(got) and not _finite(gold):
        return "both-inf"
    if not _finite(got) or not _finite(gold):
        return "mismatch"
    return "ok" if abs(got - gold) <= OPT_COMPARE_ATOL else "mismatch"


def run_size_group(
    *,
    data_root: Path,
    out_dir: Path,
    gold_dir: Path,
    stacks: int,
    tiers: int,
    fill_rate: float,
    instances: Sequence[int],
    time_limit_s: float,
    seed: int,
    heuristics: bool,
    n_samples: int,
    resume: bool,
) -> Tuple[List[InstanceResult], int, int]:
    tag = size_csv_name(stacks, tiers, fill_rate)
    progress_path = out_dir / "progress" / f"{tag}.progress.csv"
    gold = load_gold_opts(gold_dir, stacks, tiers, fill_rate)
    if not resume and progress_path.exists():
        progress_path.unlink()
    done = _load_done_instances(progress_path) if resume else set()
    results: List[InstanceResult] = []
    n_ok = 0
    n_cmp = 0

    if resume and progress_path.exists():
        with progress_path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                inst = int(row["instance"])
                if inst not in instances:
                    continue
                results.append(
                    InstanceResult(
                        instance=inst,
                        b=float(row["b"]),
                        b_1=float(row["b_1"]),
                        b_2=float(row["b_2"]),
                        opt=float(row["opt"]),
                        time=float(row["time"]),
                        eg=float(row.get("EG", "nan")),
                        em=float(row.get("EM", "nan")),
                        eri=float(row.get("ERI", "nan")),
                        l_heur=float(row.get("L", "nan")),
                        rand=float(row.get("Rand.", "nan")),
                        nodes_expanded=int(float(row.get("nodes_expanded", 0))),
                        cache_hits=int(float(row.get("cache_hits", 0))),
                    )
                )

    for instance in instances:
        if instance in done:
            continue
        print(
            f"Solving instance {instance} of size T={tiers}, S={stacks}, "
            f"fillRate={fill_rate}",
            flush=True,
        )
        result = solve_instance(
            data_root,
            stacks,
            tiers,
            instance,
            fill_rate,
            time_limit_s=time_limit_s,
            seed=seed,
            heuristics=heuristics,
            n_samples=n_samples,
        )
        results.append(result)
        results.sort(key=lambda r: r.instance)
        _append_progress(progress_path, result)
        _write_size_csv(out_dir / tag, results)

        gold_opt = gold.get(instance)
        extra = ""
        if gold_opt is not None:
            n_cmp += 1
            status = compare_opt(result.opt, gold_opt)
            if status == "ok" or status == "both-inf":
                n_ok += 1
            extra = f"  gold_opt={gold_opt} [{status}]"
        print(
            f"  opt={result.opt}  time={result.time:.4f}s  "
            f"nodes={result.nodes_expanded}{extra}",
            flush=True,
        )

    _write_size_csv(out_dir / tag, results)
    return results, n_ok, n_cmp


def write_final_results(
    out_dir: Path,
    fill_rate: float,
    groups: Iterable[Tuple[int, int, Sequence[InstanceResult]]],
) -> None:
    path = out_dir / final_csv_name(fill_rate)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FINAL_FIELDS)
        writer.writeheader()
        for stacks, tiers, rows in groups:
            if not rows:
                continue
            writer.writerow(
                {
                    "S": stacks,
                    "T": tiers,
                    "C": int(round(stacks * tiers * fill_rate)),
                    "b": _mean([r.b for r in rows]),
                    "b_1": _mean([r.b_1 for r in rows]),
                    "b_2": _mean([r.b_2 for r in rows]),
                    "opt": _mean([r.opt for r in rows]),
                    "time": _mean([r.time for r in rows]),
                    "EG": _mean([r.eg for r in rows]),
                    "EM": _mean([r.em for r in rows]),
                    "ERI": _mean([r.eri for r in rows]),
                    "L": _mean([r.l_heur for r in rows]),
                    "Rand.": _mean([r.rand for r in rows]),
                }
            )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Galle Experiment 1: Ku/Arthanari files + exact PBFSA (error_gap=0)."
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="StochasticCRP-master root (must contain crptw_instance/).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory for reproduced CSVs.",
    )
    p.add_argument(
        "--gold-dir",
        type=Path,
        default=DEFAULT_GOLD_DIR,
        help="Official MATLAB Experiment 1 CSVs used for opt comparison.",
    )
    p.add_argument("--fill-rate", type=float, default=None, help="0.5 or 0.67.")
    p.add_argument(
        "--fill-rates",
        type=float,
        nargs="+",
        default=None,
        help="One or more fill rates (overrides --fill-rate).",
    )
    p.add_argument("--stacks", type=str, default="5-10")
    p.add_argument("--tiers", type=str, default="3-6")
    p.add_argument("--instances", type=str, default="1-30")
    p.add_argument("--time-limit", type=float, default=3600.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--heuristics",
        action="store_true",
        help="Also evaluate EG/EM/ERI/L/Rand (MATLAB used 5000 samples).",
    )
    p.add_argument("--n-samples", type=int, default=5000)
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip instances already present in progress CSVs.",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Only 5 stacks, 3 tiers, instance 1, fill 0.5 (gold opt=1.5).",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.smoke:
        fill_rates = [0.5]
        stacks_list = [5]
        tiers_list = [3]
        instances = [1]
    else:
        if args.fill_rates:
            fill_rates = list(args.fill_rates)
        elif args.fill_rate is not None:
            fill_rates = [args.fill_rate]
        else:
            fill_rates = [0.5]
        stacks_list = parse_int_list(args.stacks)
        tiers_list = parse_int_list(args.tiers)
        instances = parse_int_list(args.instances)

    data_root = args.data_root.resolve()
    out_dir = args.out_dir.resolve()
    gold_dir = args.gold_dir.resolve()
    if not (data_root / "crptw_instance").is_dir():
        print(f"ERROR: crptw_instance/ not found under {data_root}", file=sys.stderr)
        return 2

    out_dir.mkdir(parents=True, exist_ok=True)
    total_ok = 0
    total_cmp = 0
    n_jobs = 0

    for fill_rate in fill_rates:
        print(f"Experiment 1 with fillRate={fill_rate}", flush=True)
        groups: List[Tuple[int, int, List[InstanceResult]]] = []
        for tiers in tiers_list:
            for stacks in stacks_list:
                n_jobs += len(instances)
                rows, n_ok, n_cmp = run_size_group(
                    data_root=data_root,
                    out_dir=out_dir,
                    gold_dir=gold_dir,
                    stacks=stacks,
                    tiers=tiers,
                    fill_rate=fill_rate,
                    instances=instances,
                    time_limit_s=args.time_limit,
                    seed=args.seed,
                    heuristics=args.heuristics,
                    n_samples=args.n_samples,
                    resume=args.resume,
                )
                groups.append((stacks, tiers, rows))
                total_ok += n_ok
                total_cmp += n_cmp
        write_final_results(out_dir, fill_rate, groups)

    print(f"Wrote CSVs under {out_dir}", flush=True)
    if total_cmp:
        print(
            f"opt vs gold MATLAB CSV: {total_ok}/{total_cmp} matched "
            f"(atol={OPT_COMPARE_ATOL})",
            flush=True,
        )
        if total_ok != total_cmp:
            return 1
    elif n_jobs:
        print("No gold CSVs found for comparison; results were still written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
