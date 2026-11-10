"""
    CRISP - Run one algorithm on one problem.

starts the GUI of CRISP.

    python CRISP.py --problem <name> --algo <name>

runs an algorithm on a problem with the specified parameter settings.

  
All the acceptable arguments are:

    --problem       a problem
    --algo          an algorithm 
    
Examples:

    python CRISP.py 
    
displays the GUI of CRISP.

    python CRISP.py --problem CRP-R --algo CM --layout benchmark/Caserta_dataset --class 3-3 --save

runs CM on the 40 Caserta instances of class 3-3.

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section of the algorithm that was used.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace


def crisp(algorithm, problem):
    """Solve one problem with one algorithm. Returns ``(moves, cost)``."""
    return algorithm.solve(problem)


def _algorithm_config(algorithm_cls):
    from web.backend.header_schema import algorithm_config_schema

    extra = {}
    for name, spec in algorithm_config_schema(algorithm_cls).items():
        default = spec.get("default")
        kind = spec.get("type", "str")
        if kind == "int":
            extra[name] = int(default)
        elif kind == "float":
            extra[name] = float(default)
        else:
            extra[name] = default
    return SimpleNamespace(extra=extra, seed=0)


def _problem(name: str, layout: str | None, time_model: str | None):
    from core.registry import get_problem_class

    problem_cls = get_problem_class(name)
    if problem_cls is None:
        raise SystemExit(
            f"Unknown problem: {name!r}  (try: python -m core.registry)"
        )
    if not layout:
        return problem_cls()

    path = Path(layout)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    path = path.resolve()
    if not path.is_file():
        raise SystemExit(f"Layout file not found: {path}")

    params = {}
    if time_model:
        params = {"time_model": time_model, "objective_mode": "crane_time"}
    suffix = path.suffix.lower()
    if suffix == ".txt":
        from core.benchmarks.zhu import problem_config_for_zhu_txt
        config = problem_config_for_zhu_txt(path, params)
    elif suffix == ".dat":
        from core.benchmarks.caserta import problem_config_for_caserta_dat
        config = problem_config_for_caserta_dat(path, params)
    else:
        raise SystemExit("Expected a .dat (Caserta) or .txt (Zhu) layout file.")
    problem = problem_cls(config=config)
    problem.layout_path = str(path)
    return problem


def _print_result(problem_name: str, algorithm_name: str, moves, cost) -> None:
    relocations = 0
    if moves:
        relocations = sum(
            1
            for move in moves
            if (move.get("destination") if isinstance(move, dict) else move[2])
            is not None
        )
    print(f"problem={problem_name}  algorithm={algorithm_name}")
    print(f"relocations={cost if cost is not None else relocations}  moves={0 if not moves else len(moves)}")


def _save(problem_name, algorithm_name, algorithm_cls, problem, moves, cost) -> None:
    import json
    from datetime import datetime

    payload = []
    for move in moves or []:
        if isinstance(move, dict):
            payload.append(move)
            continue
        container, source, destination = move[0], move[1], move[2]
        payload.append({
            "container": container,
            "source": source,
            "destination": destination,
            "kind": "retrieve" if destination is None else "relocate",
        })
    config = getattr(problem, "config", None)
    prob_config = dict(getattr(config, "extra", {}) or {})
    layout = getattr(problem, "layout_path", None)
    if layout:
        prob_config["layout_file_path"] = layout
    folder = (
        Path(__file__).resolve().parent
        / "results"
        / problem_name.replace(" ", "_")
        / algorithm_name.replace(" ", "_")
    )
    folder.mkdir(parents=True, exist_ok=True)
    stem = Path(layout).stem if layout else "run"
    path = folder / f"{stem}.json"
    path.write_text(json.dumps({
        "problem": problem_name,
        "algorithm": algorithm_name,
        "category": getattr(algorithm_cls, "category", ""),
        "seed": 0,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "prob_config": prob_config,
        "algo_config": {},
        "metrics": {"relocations": float(cost or 0)},
        "history": [],
        "moves": payload,
    }, indent=2), encoding="utf-8")
    print(f"Saved: {path}")


def main() -> None:
    if len(sys.argv) == 1:
        from web.__main__ import main as web_main
        web_main()
        return
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem")
    parser.add_argument("--algo")
    parser.add_argument("--layout")
    parser.add_argument("--save")
    args = parser.parse_args()

    from core.registry import get_algorithm_class

    algorithm_cls = get_algorithm_class(args.algo)      
    
    problem = _problem(args.problem, args.layout)
    
    moves, cost = crisp(algorithm_cls(config=_algorithm_config(algorithm_cls)), problem)
    _print_result(args.problem, args.algo, moves, cost)
    if args.save:
        _save(args.problem, args.algo, algorithm_cls, problem, moves, cost)


if __name__ == "__main__":
    main()
