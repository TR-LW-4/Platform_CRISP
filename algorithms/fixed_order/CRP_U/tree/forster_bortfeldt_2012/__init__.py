"""
Forster & Bortfeldt (2012) — heuristic tree search for CRP.

Public symbols
--------------
ForsterBortfeldt2012   : platform BaseAlgorithm subclass (entry point).
FBLayout               : internal bay-state representation.
greedy_solution        : standalone greedy solver.
run_tree_search        : standalone tree search solver.
"""

from .algorithm import ForsterBortfeldt2012
from .layout import FBLayout
from .tree_search import greedy_solution, run_tree_search

__all__ = [
    "ForsterBortfeldt2012",
    "FBLayout",
    "greedy_solution",
    "run_tree_search",
]
