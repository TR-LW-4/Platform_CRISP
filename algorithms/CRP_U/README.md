# CRP-U algorithms

Algorithms for the unrestricted block relocation problem are organized by
solution mechanism:

- `tree/`: tree-search methods, including exact B&B/IDA*/IDBB algorithms and
  heuristic tree searches.
- `solver_ip/`: solver-backed integer and mixed-integer programming models.
- `heuristic/`: constructive heuristics, local search, metaheuristics, and
  other non-tree heuristic methods.

## Tree search

- `forster_bortfeldt_2012/` — heuristic compound-move tree search.
- `zhu_2012/` — IDA*.
- `tanaka_mizuno_2018/` — exact branch-and-bound.
- `jin_tanaka_2023/` — exact iterative deepening branch-and-bound.

## Solver IP/MIP

- `caserta_2012_brp1/` — BRP-I.
- `petering_hussein_2013/` — BRP-III.
- `de_melo_silva_2018/` — unified BRP formulations.
- `lu_zeng_liu_2019/` — BRP-m3 and IS*.

The registry discovers algorithm classes recursively. Directory placement
describes the solution mechanism; each class's `category`,
`compatible_problems`, and solver metadata describe its runtime behavior.
