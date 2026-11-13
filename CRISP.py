"""
    CRISP   (Container Relocation integrated Stowage Platform)

    python CRISP.py --problem <name> --algo <name>

runs an algorithm on a problem with the specified parameter settings.

  
All the acceptable arguments are:

    --problem       a problem
    --algo          an algorithm 
    
Examples:

    python CRISP.py 
    
displays the GUI of CRISP.

    python CRISP.py --problem CRP-R --algo CM --benchmark caserta --size 3-3 

runs CM on the 40 Caserta instances of class 3-3.

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section of the algorithm that was used.
--------------------------------------------------------------------------
"""

import sys
import argparse

def split_call(text):
    name, sep, rest = text.partition("(")
    if not sep:
        return name, {}
    extra = {}
    for item in rest[:-1].split(","):
        if item.strip():
            key, value = item.split("=")
            extra[key.strip()] = value.strip()
    return name.strip(), extra


def main() -> None:
    if len(sys.argv) == 1:
        from web.__main__ import main as web_main
        web_main()
        return
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem")
    parser.add_argument("--algo")
    parser.add_argument("--benchmark")
    parser.add_argument("--size")
    
    args = parser.parse_args()
    
    
    
    from web.registry import get_algorithm_class, get_problem_class

    problem_name, problem_extra = split_call(args.problem)
    algo_name, algo_extra = split_call(args.algo)
    ALG = get_algorithm_class(algo_name)
    Pro = get_problem_class(problem_name)
    
    import importlib

    reader = importlib.import_module(f"benchmark.{args.benchmark}")
    for path in reader.layout_paths(args.benchmark, args.size):
        data = reader.read_layout(path)
        problem = Pro(
            num_bays=data["num_bays"],
            num_rows=data["num_rows"],
            max_tiers=data["max_tiers"],
            num_containers=data["num_containers"],
            stacks=data["stacks"],
            retrieval_order=data["retrieval_order"],
        )
        for name, value in problem_extra.items():
            if hasattr(problem, name):
                setattr(problem, name, type(getattr(problem, name))(value))
        algorithm = ALG(type("Config", (), {"extra": algo_extra})())
        result, metric = algorithm.solve(problem)

if __name__ == "__main__":
    main()
