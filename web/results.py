"""Saved-run commands: python -m web.results results|bench-summary"""

from __future__ import annotations

import argparse
from pathlib import Path


def _result_files() -> list[Path]:
    root = Path(__file__).resolve().parents[1] / "results"
    if not root.is_dir():
        return []
    files = [
        path for path in root.rglob("*.json")
        if not path.name.startswith("batch_summary_")
    ]
    return sorted(files, key=lambda path: path.stat().st_mtime, reverse=True)


def show_results(limit: int, verbose: bool) -> None:
    """Print recent runs saved under results/."""
    import json

    files = _result_files()[: max(1, limit)]
    if not files:
        print("No saved runs found under results/")
        return

    print(f"\n{'=' * 72}")
    print(f"Most recent {len(files)} run(s)")
    print(f"{'=' * 72}\n")
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        print(
            f"{data.get('timestamp', '')}  {data.get('problem', '')}  "
            f"{data.get('algorithm', '')}  →  {data.get('metrics', {})}"
        )
        print(f"    file: {path}")
        if verbose:
            config = data.get("prob_config") or {}
            layout = config.get("layout_file_path") or config.get("caserta_dat_path")
            if layout:
                print(f"    layout: {layout}")
        print()


def show_bench_summary(problem: str, algo: str, limit: int, save: bool) -> None:
    """Aggregate newest saved runs by benchmark class."""
    from urllib.parse import urlencode

    from web.backend.summary_pages import (
        save_batch_summary_sidecar,
        summarize_saved_runs,
    )

    try:
        summary = summarize_saved_runs(problem=problem, algorithm=algo, limit=limit)
    except ImportError as exc:
        raise SystemExit(f"Benchmark summary is unavailable: {exc}") from exc
    print(f"\n{problem}  ·  {algo}")
    print(f"total_runs={summary.get('total_runs')}  ungrouped={summary.get('ungrouped')}\n")
    print(f"{'Class':<12} {'n':>5}  {'metric':<14} {'mean':>10} {'std':>10}")
    print("-" * 56)
    for row in summary.get("classes") or []:
        metrics = row.get("metrics") or {}
        if not metrics:
            print(f"{row.get('label', '?'):<12} {row.get('n', 0):>5}")
            continue
        first = True
        for name, agg in metrics.items():
            label = row.get("label", "?") if first else ""
            count = row.get("n", 0) if first else ""
            print(
                f"{label:<12} {count!s:>5}  {name:<14} "
                f"{agg.get('mean', float('nan')):>10.4g} "
                f"{agg.get('std', float('nan')):>10.4g}"
            )
            first = False
    if save and summary.get("total_runs"):
        path = save_batch_summary_sidecar(
            [],
            summary,
            problem=problem,
            algorithm=algo,
        )
        if path:
            print(f"\nSaved sidecar: {path}")
    query = urlencode({"problem": problem, "algorithm": algo, "limit": limit})
    print(
        "\nAlso view in the browser while the web server is running:\n"
        f"  http://127.0.0.1:8000/bench-summary/from-results?{query}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect saved CRISP runs")
    commands = parser.add_subparsers(dest="command", required=True)

    results = commands.add_parser("results", help="List recent saved run files")
    results.add_argument("--limit", type=int, default=20)
    results.add_argument("-v", "--verbose", action="store_true")

    summary = commands.add_parser(
        "bench-summary",
        help="Mean and standard deviation by benchmark class",
    )
    summary.add_argument("--problem", default="CRP-R")
    summary.add_argument("--algo", default="Zhu2012IDAStarR")
    summary.add_argument("--limit", type=int, default=120)
    summary.add_argument("--save", action="store_true")

    args = parser.parse_args()
    if args.command == "results":
        show_results(args.limit, args.verbose)
    else:
        show_bench_summary(args.problem, args.algo, args.limit, args.save)


if __name__ == "__main__":
    main()
