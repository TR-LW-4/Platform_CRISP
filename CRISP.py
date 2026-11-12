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
    problem_cls = get_problem_class(args.problem)   
    
    for path in layout_paths(args.layout, args.class_name):
        data = read_layout(path)
        problem = problem_cls(
            num_bays=data["num_bays"],
            num_rows=data["num_rows"],
            max_tiers=data["max_tiers"],
            num_containers=data["num_containers"],
        )
        moves, cost = crisp(algorithm_cls(config=_algorithm_config(algorithm_cls)), problem)
    

if __name__ == "__main__":
    main()
