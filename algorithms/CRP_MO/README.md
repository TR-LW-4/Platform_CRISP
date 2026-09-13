# CRP-MO algorithms

Bi-objective restricted retrieval: minimise **relocations** and **crane
working time** together. Dynamics match CRP-Time / CRP-R (distinct
priorities, restricted relocation). The search returns a Pareto set, not
a scalar.

**No algorithm is registered yet.** `evolutionary/` and `heuristic/` are
scaffolds. Add a method as
`algorithms/CRP_MO/<category>/<paper_key>/algorithm.py` with
`compatible_problems = ["CRP-MO"]`. Do not register CRP-Time single-
objective heuristics here: they return one plan, not a front.

## Layout

```
algorithms/CRP_MO/
├── evolutionary/     # NSGA-II / MOEA/D variants (empty)
└── heuristic/        # Pareto heuristics (empty)
```

## Problem

See `problems/CRP_MO.py`. One episode evaluates one plan to
`(relocations, crane_time)` via `evaluate_vector`. Set indicators
(hypervolume, non-dominated count) belong to the algorithm once it
exists.
