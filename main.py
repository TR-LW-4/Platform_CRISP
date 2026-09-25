"""
CRP Platform – CLI entry point.

Usage
-----
# Launch the React + FastAPI Web workbench
python main.py web

# List registered problems and algorithms
python main.py list

# Quick smoke-test: run a compatible heuristic on every problem for 1 episode
python main.py test

# Run a single experiment from the command line
python main.py run --problem "CRP-R" --algo "Caserta (2012) HEUR"

# Run one benchmark layout (Caserta .dat or Zhu .txt) without the web UI — prints metrics
python main.py layout-run --problem "CRP-R" --algo "Caserta (2012) HEUR" \\
    --layout benchmark/Caserta_dataset/data3-3-1.dat

# Optimise the tier-dependent crane-time model instead of f2 (CRP-Time only)
python main.py layout-run --problem "CRP-Time" --algo "Lee–Lee (2010) Retrieval" \\
    --layout benchmark/Caserta_dataset/data5-5-1.dat --time-model f2_vertical

# Same, but print LA-N move-by-move trace on stderr (--trace-lan or export CRISP_TRACE_LAN=1)
python main.py layout-run --problem "CRP-R" --algo "LA-N Look-Ahead" \\
    --layout benchmark/Caserta_dataset/data3-3-1.dat --trace-lan

# Exact solver (Tanaka B&B on bundled ``brp_bb``)
python main.py layout-run --problem "CRP-R" --algo "Tanaka (2016) B&B" \\
    --layout benchmark/Zhu_dataset/5-8-39/06101.txt

# Show recent saved result JSON files (same as the web workbench writes under results/)
python main.py results --limit 15

# Class-wise mean±std over newest saved Caserta/Zhu runs (no npm; also /bench-summary in web)
python main.py bench-summary --problem "CRP-R" --algo "Caserta (2012) HEUR" --limit 120
"""

import sys
import os
from functools import partial
from typing import Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))


def cmd_web(host: str, port: int):
    import uvicorn
    uvicorn.run(
        "web.backend.api:app",
        host=host,
        port=port,
        reload=False,
    )


def cmd_list():
    from core.registry import get_problem_info, get_algorithm_info
    print("\n=== Problems ===")
    for p in get_problem_info():
        print(f"  [{', '.join(p['tags'])}] {p['name']}: {p['description'][:70]}")

    print("\n=== Algorithms ===")
    for a in get_algorithm_info():
        print(f"  [{a['category']}] {a['name']}: {a['description'][:70]}")


def cmd_test():
    """Quick smoke-test: one heuristic episode per registered problem (no Greedy module)."""
    from core.registry import (
        compatible_algorithms,
        get_algorithm_class,
        get_problem_class,
        list_problems,
    )
    from core.base_problem import ProblemConfig
    from core.base_algorithm import AlgorithmConfig
    from core.caserta_benchmark import DEFAULT_CASERTA_DIR, problem_config_for_caserta_dat
    import multiprocessing as mp

    _PREFERRED = ("Caserta (2012) HEUR", "Kim–Hong (2006) ENAR")

    print("\n=== Smoke Test ===")
    cfg = ProblemConfig(num_bays=3, num_rows=2, max_tiers=3, num_containers=8, num_groups=2)
    # CRP-Time rejects random layouts, so it runs on a small Caserta instance.
    time_cfg = problem_config_for_caserta_dat(DEFAULT_CASERTA_DIR / "data3-3-1.dat", {})
    for pname in list_problems():
        pcls = get_problem_class(pname)
        if pcls is None:
            continue
        names = compatible_algorithms(pname)
        heur = [
            n
            for n in names
            if getattr(get_algorithm_class(n), "category", "") == "Heuristic"
        ]
        pick = None
        for pref in _PREFERRED:
            if pref in heur:
                pick = pref
                break
        if pick is None and heur:
            pick = heur[0]
        if pick is None and names:
            pick = names[0]
        if pick is None:
            print(f"  ? {pname}: no compatible algorithm")
            continue

        acls = get_algorithm_class(pick)
        algo = acls(AlgorithmConfig(max_iterations=20))
        q = mp.Queue()
        ev = mp.Event()

        # partial() instead of a closure: Windows spawns workers and must
        # pickle this factory, which cannot reach a function-local object.
        factory = partial(pcls, config=time_cfg if pname == "CRP-Time" else cfg)

        proc = mp.Process(target=algo.train, args=(factory, q, ev), daemon=True)
        proc.start()
        proc.join(timeout=60)
        ev.set()

        records = []
        while not q.empty():
            records.append(q.get_nowait())

        if records:
            final = records[-1]
            print(
                f"  ✓ {pname} [{pick}]: metric={final.metric:.3f}  metrics={final.metrics}"
            )
        else:
            print(f"  ? {pname} [{pick}]: no records returned")


def cmd_run(problem: str, algo: str, iterations: int):
    from core.registry        import get_problem_class, get_algorithm_class
    from core.base_problem    import ProblemConfig
    from core.base_algorithm  import AlgorithmConfig
    import multiprocessing as mp

    pcls = get_problem_class(problem)
    acls = get_algorithm_class(algo)
    if pcls is None:
        print(f"Unknown problem: {problem}")
        return
    if acls is None:
        print(f"Unknown algorithm: {algo}")
        return
    if problem == "CRP-Time":
        print(
            "CRP-Time does not use random layouts; run it on a layout file with "
            'python main.py layout-run --problem "CRP-Time" --layout <file>'
        )
        return

    cfg_p = ProblemConfig()
    cfg_a = AlgorithmConfig(max_iterations=iterations, report_interval=max(1, iterations // 20))
    inst  = acls(cfg_a)
    q     = mp.Queue()
    ev    = mp.Event()

    factory = partial(pcls, config=cfg_p)

    print(f"\nRunning {algo} on {problem} for {iterations} iterations...")
    proc = mp.Process(target=inst.train, args=(factory, q, ev), daemon=True)
    proc.start()

    while proc.is_alive():
        try:
            r = q.get(timeout=1.0)
            m = r.metrics
            best_metric = m.get("best_metric", getattr(r, "best_metric", r.metric))
            print(
                f"  step={r.step:5d}/{cfg_a.max_iterations}"
                f"  metric={r.metric:.3f}  best={best_metric:.3f}"
                f"  metrics={m}"
                f"  [{r.progress*100:.1f}%]"
            )
        except Exception:
            pass

    proc.join()
    ev.set()
    print("Done.")


_BASE_ALGO_KEYS = frozenset({
    "max_iterations", "seed", "report_interval",
    "population_size", "crossover_rate", "mutation_rate",
    "tournament_size", "elite_count",
})


def _algorithm_config_from_schema(algo_cls, overrides: dict):
    """Build ``AlgorithmConfig`` from ``config_schema()`` defaults + overrides."""
    from core.base_algorithm import AlgorithmConfig

    params: dict = {}
    if hasattr(algo_cls, "config_schema"):
        for aname, spec in algo_cls.config_schema().items():
            t = spec.get("type", "int")
            if t == "int":
                params[aname] = int(spec["default"])
            elif t == "float":
                params[aname] = float(spec["default"])
            elif t == "bool":
                params[aname] = bool(spec["default"])
            else:
                params[aname] = spec["default"]
    params.update(overrides)
    base_kwargs = {k: v for k, v in params.items() if k in _BASE_ALGO_KEYS}
    cfg = AlgorithmConfig(**base_kwargs)
    for k, v in params.items():
        cfg.extra[k] = v
    return cfg


def cmd_layout_run(
    layout: str,
    problem: str,
    algo: str,
    save: bool,
    timeout: float,
    trace_lan: bool = False,
    time_model: Optional[str] = None,
):
    """
    Run *one* registered algorithm on *one* Caserta (.dat) or Zhu (.txt) layout file.
    Metrics go to stdout; optional JSON under results/.
    """
    if trace_lan:
        os.environ["CRISP_TRACE_LAN"] = "1"
        print(
            "[layout-run] LA-N planner trace enabled on stderr ([CRISP_TRACE_LAN]=1).",
            file=sys.stderr,
            flush=True,
        )

    from pathlib import Path

    from core.registry import get_algorithm_class, get_problem_class
    from core.benchmark_keys import LAYOUT_FILE_EXTRA_KEY
    from core.caserta_benchmark import problem_config_for_caserta_dat
    from core.result_store import moves_from_records, save_run
    from core.zhu_benchmark import problem_config_for_zhu_txt

    root = Path(__file__).resolve().parent
    p = (root / layout).resolve() if not Path(layout).is_absolute() else Path(layout).resolve()
    if not p.is_file():
        print(f"Layout file not found: {p}")
        return

    pcls = get_problem_class(problem)
    acls = get_algorithm_class(algo)
    if pcls is None:
        print(f"Unknown problem: {problem!r}  (try: python main.py list)")
        return
    if acls is None:
        print(f"Unknown algorithm: {algo!r}  (try: python main.py list)")
        return

    # objective_mode travels with time_model because the saved-result identity
    # is keyed off it; CRP-Time forces "crane_time" as its only mode anyway.
    prob_params = (
        {"time_model": time_model, "objective_mode": "crane_time"}
        if time_model
        else {}
    )

    suffix = p.suffix.lower()
    if suffix == ".txt":
        prob_cfg = problem_config_for_zhu_txt(p, prob_params)
        src_tag = "cli_zhu_txt"
    elif suffix == ".dat":
        prob_cfg = problem_config_for_caserta_dat(p, prob_params)
        src_tag = "cli_caserta_dat"
    else:
        print("Expected a `.dat` (Caserta) or `.txt` (Zhu) layout file.")
        return

    cfg_a = _algorithm_config_from_schema(acls, {})
    inst = acls(config=cfg_a)

    import multiprocessing as mp

    factory = partial(pcls, config=prob_cfg)

    q = mp.Queue()
    ev = mp.Event()
    proc = mp.Process(target=inst.train, args=(factory, q, ev), daemon=True)
    proc.start()
    proc.join(timeout=timeout)
    ev.set()
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=5)
        print("ERROR: train subprocess timed out or did not exit cleanly.")
        return
    if proc.exitcode != 0:
        print(
            f"ERROR: train subprocess crashed (exit code {proc.exitcode}); "
            "its traceback is on stderr. Any records it pushed describe an "
            "incomplete run, so no metrics are reported."
        )
        return

    records = []
    while not q.empty():
        records.append(q.get_nowait())

    if not records:
        print("No progress records returned (empty queue).")
        return

    final = records[-1]
    algo_cat = getattr(acls, "category", "")
    print(f"\nlayout: {p}")
    print(f"problem={problem}  algorithm={algo}")
    print(
        f"step={final.step}  metric={final.metric:.6f}  "
        f"best_metric={final.best_metric:.6f}  progress={final.progress:.4f}"
    )
    print(f"metrics: {final.metrics}")
    moves = getattr(final, "extra", None) or {}
    move_list = moves.get("moves") if isinstance(moves, dict) else None
    if move_list:
        n_rel = sum(1 for m in move_list if m.get("kind") == "relocate")
        n_ret = sum(1 for m in move_list if m.get("kind") == "retrieve")
        print(f"moves: {len(move_list)} total ({n_rel} relocate, {n_ret} retrieve)")

    if save:
        history = [
            {"step": r.step, "metric": r.metric, "metrics": r.metrics}
            for r in records
        ]
        algo_params = cfg_a.to_dict()
        prob_save = {
            **prob_cfg.extra,
            LAYOUT_FILE_EXTRA_KEY: str(p),
            "source": src_tag,
        }
        out = save_run(
            problem=problem,
            algorithm=algo,
            category=algo_cat,
            seed=abs(hash(p.name)) % 2_000_000_000,
            prob_config=prob_save,
            algo_config=algo_params,
            metrics=final.metrics,
            history=history,
            moves=moves_from_records(records),
        )
        print(f"\nSaved: {out}")


def cmd_results(limit: int, verbose: bool):
    """Print recent runs saved under results/ (newest first)."""
    from pathlib import Path

    from core.benchmark_keys import LAYOUT_FILE_EXTRA_KEY
    from core.result_store import list_saved_runs, load_run

    runs = list_saved_runs()[: max(1, limit)]
    if not runs:
        print("No saved runs found under results/")
        return

    print(f"\n{'=' * 72}")
    print(f"Most recent {len(runs)} run(s) — same JSON files as the web workbench saves")
    print(f"{'=' * 72}\n")

    for r in runs:
        fp = r["file"]
        prob = r["problem"]
        alg = r["algorithm"]
        ts = r.get("timestamp", "")
        m = r.get("metrics", {})
        primary = None
        if m:
            primary = m.get("relocations", m.get("metric", next(iter(m.values()), None)))
        line = f"{ts}  {prob}  {alg}  →  {m}"
        if primary is not None and isinstance(primary, (int, float)):
            line += f"  (primary≈{primary})"
        print(line)
        print(f"    file: {fp}")

        if verbose:
            data = load_run(Path(fp))
            pc = data.get("prob_config") or {}
            layout = pc.get(LAYOUT_FILE_EXTRA_KEY) or pc.get("caserta_dat_path")
            if layout:
                print(f"    layout: {layout}")
        print()


def cmd_bench_summary(problem: str, algo: str, limit: int, save: bool):
    """Aggregate newest saved runs by Caserta/Zhu class (mean ± std). No npm."""
    from web.backend.summary_pages import (
        save_batch_summary_sidecar,
        summarize_saved_runs,
    )

    summary = summarize_saved_runs(problem=problem, algorithm=algo, limit=limit)
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
            n = row.get("n", 0) if first else ""
            print(
                f"{label:<12} {n!s:>5}  {name:<14} "
                f"{agg.get('mean', float('nan')):>10.4g} "
                f"{agg.get('std', float('nan')):>10.4g}"
            )
            first = False
    if save and summary.get("total_runs"):
        # Sidecar without re-listing every path (optional bookkeeping).
        path = save_batch_summary_sidecar(
            [],
            summary,
            problem=problem,
            algorithm=algo,
        )
        if path:
            print(f"\nSaved sidecar: {path}")
    from urllib.parse import urlencode

    qs = urlencode({"problem": problem, "algorithm": algo, "limit": limit})
    print(
        "\nAlso view in browser while web is running:\n"
        f"  http://127.0.0.1:8000/bench-summary/from-results?{qs}\n"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CRP Platform")
    subparsers = parser.add_subparsers(dest="command")

    web_parser = subparsers.add_parser(
        "web",
        help="Launch React + FastAPI Web workbench",
    )
    web_parser.add_argument("--host", default="127.0.0.1")
    web_parser.add_argument("--port", type=int, default=8000)
    subparsers.add_parser("list", help="List registered problems and algorithms")
    subparsers.add_parser("test", help="Quick smoke-test")

    run_parser = subparsers.add_parser("run", help="Run a single experiment")
    run_parser.add_argument("--problem",    default="CRP-Stow")
    run_parser.add_argument("--algo",       default="Caserta (2012) HEUR")
    run_parser.add_argument("--iterations", type=int, default=100)

    layout_parser = subparsers.add_parser(
        "layout-run",
        help="Run one algorithm on one Caserta (.dat) or Zhu (.txt) file (no web UI)",
    )
    layout_parser.add_argument(
        "--layout",
        "-l",
        required=True,
        help="Path to .dat or .txt layout (relative to project root or absolute)",
    )
    layout_parser.add_argument("--problem", default="CRP-R")
    layout_parser.add_argument(
        "--algo",
        "-a",
        required=True,
        help='Algorithm display name (see: python main.py list), e.g. "Caserta (2012) HEUR"',
    )
    layout_parser.add_argument(
        "--time-model",
        choices=["f2", "f2_vertical", "rmgc_current"],
        help=(
            "CRP-Time crane-time model the search minimises: Voß–Schwarze f2 "
            "(default), f2vert with tier-dependent pickup/place-down, or the "
            "legacy multi-bay RMGC model. Other problems ignore it."
        ),
    )
    layout_parser.add_argument(
        "--save",
        action="store_true",
        help="Also write results/<problem>/<algo>/…json like the web workbench",
    )
    layout_parser.add_argument(
        "--timeout",
        type=float,
        default=900.0,
        help="Seconds to wait for the training subprocess (default 900)",
    )
    layout_parser.add_argument(
        "--trace-lan",
        action="store_true",
        help=(
            "Print LA-N build_lan_plan step-by-move trace to stderr "
            '(same as export CRISP_TRACE_LAN=1; only affects "LA-N Look-Ahead").'
        ),
    )

    res_parser = subparsers.add_parser(
        "results",
        help="List recent saved run JSON files under results/",
    )
    res_parser.add_argument("--limit", type=int, default=20)
    res_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help='Include layout path from prob_config (layout_file_path / legacy caserta_dat_path)',
    )

    bench_parser = subparsers.add_parser(
        "bench-summary",
        help="Class-wise mean/std over saved benchmark results (no npm)",
    )
    bench_parser.add_argument("--problem", default="CRP-R")
    bench_parser.add_argument("--algo", default="Caserta (2012) HEUR")
    bench_parser.add_argument(
        "--limit",
        type=int,
        default=120,
        help="Use the newest N result JSON files under results/<problem>/<algo>/",
    )
    bench_parser.add_argument(
        "--save",
        action="store_true",
        help="Also write results/.../batch_summary_*.json sidecar",
    )

    args = parser.parse_args()

    if args.command == "web":
        cmd_web(args.host, args.port)
    elif args.command == "list":
        cmd_list()
    elif args.command == "test":
        cmd_test()
    elif args.command == "run":
        cmd_run(args.problem, args.algo, args.iterations)
    elif args.command == "layout-run":
        cmd_layout_run(
            args.layout,
            args.problem,
            args.algo,
            args.save,
            args.timeout,
            trace_lan=args.trace_lan,
            time_model=args.time_model,
        )
    elif args.command == "results":
        cmd_results(args.limit, args.verbose)
    elif args.command == "bench-summary":
        cmd_bench_summary(args.problem, args.algo, args.limit, args.save)
    else:
        parser.print_help()
