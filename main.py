"""
CRP Platform – CLI entry point.

Usage
-----
# Launch the Streamlit GUI
python main.py gui

# List registered problems and algorithms
python main.py list

# Quick smoke-test: run greedy on every problem for 1 episode
python main.py test

# Run a single experiment from the command line
python main.py run --problem CSPP --algo "Genetic Algorithm" --iterations 200
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
    """Quick smoke-test: greedy on every problem."""
    from core.registry        import list_problems, get_problem_class
    from core.base_problem    import ProblemConfig
    from algorithms._shared.heuristic.greedy import GreedyHeuristic
    from core.base_algorithm  import AlgorithmConfig
    import multiprocessing as mp

    print("\n=== Smoke Test ===")
    for pname in list_problems():
        pcls = get_problem_class(pname)
        if pcls is None:
            continue
        cfg  = ProblemConfig(num_bays=3, num_rows=2, max_tiers=3, num_containers=8, num_groups=2)
        algo = GreedyHeuristic(AlgorithmConfig(num_eval_seeds=2, max_iterations=20))
        q    = mp.Queue()
        ev   = mp.Event()

        def factory(_p=pcls, _c=cfg):
            return _p(config=_c)

        proc = mp.Process(target=algo.train, args=(factory, q, ev), daemon=True)
        proc.start()
        proc.join(timeout=30)
        ev.set()

        records = []
        while not q.empty():
            records.append(q.get_nowait())

        if records:
            final = records[-1]
            print(f"  ✓ {pname}: metric={final.metric:.3f}  metrics={final.metrics}")
        else:
            print(f"  ? {pname}: no records returned")


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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CRP Platform")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("gui",  help="Launch Streamlit GUI")
    subparsers.add_parser("list", help="List registered problems and algorithms")
    subparsers.add_parser("test", help="Quick smoke-test")

    run_parser = subparsers.add_parser("run", help="Run a single experiment")
    run_parser.add_argument("--problem",    default="CSPP")
    run_parser.add_argument("--algo",       default="Greedy Heuristic")
    run_parser.add_argument("--iterations", type=int, default=100)

    args = parser.parse_args()

    if args.command == "gui":
        cmd_gui()
    elif args.command == "list":
        cmd_list()
    elif args.command == "test":
        cmd_test()
    elif args.command == "run":
        cmd_run(args.problem, args.algo, args.iterations)
    else:
        parser.print_help()
