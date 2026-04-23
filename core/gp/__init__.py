"""
core.gp — Genetic-Programming infrastructure for the platform.

This sub-package is **platform-level infrastructure**, not tied to any
specific paper.  It provides the generic pieces every GP-based
container-relocation algorithm reuses:

* ``engine``         — Node tree, ramped-half-and-half initialisation,
                       subtree crossover / mutation, steady-state
                       3-tournament driver.
* ``terminals``      — CRP-domain terminal functions (SH, EMP, CUR, RI,
                       AVG, DIFF + DIS, DUR for multi-bay).
* ``rs_restricted``  — restricted relocation scheme whose destination
                       scoring is driven by a PF tree.

Paper-specific pieces (custom operators, extra terminals, unrestricted
RS variants, multitask drivers, …) stay in each paper's own algorithm
folder and only depend on this package.

Contribution history (for reference, not ownership):
    Ðurasević & Ðumić (2024, ASOC) — first platform integration.
    Ðurasević, Ðumić & Gil-Gala (2025, EAAI) — extended to multitask.
"""

from .engine         import (
    Node,
    init_population_rhh,
    subtree_crossover,
    subtree_mutation,
    evolve_gp,
)
from .terminals      import (
    STANDARD_TERMINALS,
    MULTIBAY_EXTRA_TERMINALS,
    build_terminal_table,
    terminal_names,
)
from .rs_restricted  import build_plan_restricted

__all__ = [
    "Node",
    "init_population_rhh",
    "subtree_crossover",
    "subtree_mutation",
    "evolve_gp",
    "STANDARD_TERMINALS",
    "MULTIBAY_EXTRA_TERMINALS",
    "build_terminal_table",
    "terminal_names",
    "build_plan_restricted",
]
