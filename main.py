"""
CRP Platform – CLI entry point.

Usage
-----
# Launch the Streamlit GUI
python main.py gui

# List registered problems and algorithms
python main.py list

# Quick smoke-test: run a compatible heuristic on every problem for 1 episode
python main.py test

# Run a single experiment from the command line
python main.py run --problem "CRP-Stow" --algo "Genetic Algorithm" --iterations 200

# Run one benchmark layout (Caserta .dat or Zhu .txt) without the GUI — prints metrics
python main.py layout-run --problem "CRP-R" --algo "Caserta (2012) HEUR" \\
    --layout benchmark/Caserta_dataset/data3-3-1.dat

# Same, but print LA-N move-by-move trace on stderr (--trace-lan or export CRISP_TRACE_LAN=1)
python main.py layout-run --problem "CRP-R" --algo "LA-N Look-Ahead" \\
    --layout benchmark/Caserta_dataset/data3-3-1.dat --trace-lan

# Exact solver (Tanaka B&B on bundled ``brp_bb``)
python main.py layout-run --problem "CRP-R" --algo "Tanaka (2016) B&B" \\
    --layout benchmark/Zhu_dataset/5-8-39/06101.txt --seeds 1

# Show recent saved result JSON files (same as GUI Test/Experiment writes under results/)
python main.py results --limit 15
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))


def cmd_gui():
    import subprocess
    app_path = os.path.join(os.path.dirname(__file__), "gui", "app.py")
    subprocess.run(["streamlit", "run", app_path])


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
    import multiprocessing as mp

    _PREFERRED = ("Caserta (2012) HEUR", "Kim–Hong (2006) ENAR")

    print("\n=== Smoke Test ===")
    cfg = ProblemConfig(num_bays=3, num_rows=2, max_tiers=3, num_containers=8, num_groups=2)
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
        algo = acls(AlgorithmConfig(num_eval_seeds=2, max_iterations=20))
        q = mp.Queue()
        ev = mp.Event()

        def factory(_p=pcls, _c=cfg):
            return _p(config=_c)

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

    cfg_p = ProblemConfig()
    cfg_a = AlgorithmConfig(max_iterations=iterations, report_interval=max(1, iterations // 20))
    inst  = acls(cfg_a)
    q     = mp.Queue()
    ev    = mp.Event()

    def factory(_p=pcls, _c=cfg_p):
        return _p(config=_c)

    print(f"\nRunning {algo} on {problem} for {iterations} iterations...")
    proc = mp.Process(target=inst.train, args=(factory, q, ev), daemon=True)
    proc.start()

    while proc.is_alive():
        try:
            r = q.get(timeout=1.0)
            print(f"  step={r.step:5d}  metric={r.metric:.4f}  progress={r.progress*100:.1f}%")
        except Exception:
            pass

    proc.join()
    ev.set()
    print("Done.")


_BASE_ALGO_KEYS = frozenset({
    "max_iterations", "seed", "report_interval", "num_eval_seeds",
    "total_timesteps", "learning_rate", "gamma", "num_envs",
    "num_steps", "batch_size", "hidden_dim",
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
    seeds: int,
    save: bool,
    timeout: float,
    trace_lan: bool = False,
):
    """
    Run *one* registered algorithm on *one* Caserta (.dat) or Zhu (.txt) layout file.
    No Streamlit — metrics go to stdout; optional JSON under results/.
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
    from core.result_store import save_run
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

    suffix = p.suffix.lower()
    if suffix == ".txt":
        prob_cfg = problem_config_for_zhu_txt(p, {})
        src_tag = "cli_zhu_txt"
    elif suffix == ".dat":
        prob_cfg = problem_config_for_caserta_dat(p, {})
        src_tag = "cli_caserta_dat"
    else:
        print("Expected a `.dat` (Caserta) or `.txt` (Zhu) layout file.")
        return

    cfg_a = _algorithm_config_from_schema(acls, {"num_eval_seeds": max(1, seeds)})
    inst = acls(config=cfg_a)

    import multiprocessing as mp

    def factory():
        return pcls(config=prob_cfg)

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

    if save:
        history = [
            {"step": r.step, "metric": r.metric, "metrics": r.metrics}
            for r in records
        ]
        algo_params = cfg_a.to_dict()
        prob_save = {
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
    print(f"Most recent {len(runs)} run(s) — same JSON files as the GUI saves")
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CRP Platform")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("gui",  help="Launch Streamlit GUI")
    subparsers.add_parser("list", help="List registered problems and algorithms")
    subparsers.add_parser("test", help="Quick smoke-test")

    run_parser = subparsers.add_parser("run", help="Run a single experiment")
    run_parser.add_argument("--problem",    default="CRP-Stow")
    run_parser.add_argument("--algo",       default="Caserta (2012) HEUR")
    run_parser.add_argument("--iterations", type=int, default=100)

    layout_parser = subparsers.add_parser(
        "layout-run",
        help="Run one algorithm on one Caserta (.dat) or Zhu (.txt) file (no GUI)",
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
        "--seeds",
        type=int,
        default=1,
        help="num_eval_seeds (heuristics often use 1 for a single pass)",
    )
    layout_parser.add_argument(
        "--save",
        action="store_true",
        help="Also write results/<problem>/<algo>/…json like the GUI",
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

    args = parser.parse_args()

    if args.command == "gui":
        cmd_gui()
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
            args.seeds,
            args.save,
            args.timeout,
            trace_lan=args.trace_lan,
        )
    elif args.command == "results":
        cmd_results(args.limit, args.verbose)
    else:
        parser.print_help()
