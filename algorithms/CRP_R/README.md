# CRP-R algorithms

Algorithms for the restricted block relocation problem are organized by
solution mechanism:

- `tree/`: B&B, A*, IDA*, abstraction, and other tree-search methods.
- `solver_ip/`: solver-backed integer and mixed-integer programming models.
- `heuristic/`: constructive heuristics, local search, metaheuristics, and
  other non-tree heuristic methods.

## Tree search

- `kim_hong_2006/` — branch-and-bound.
- `zhu_2012/` — IDA*.
- `exposito_2015/` — branch-and-bound.
- `ku_arthanari_2016_abstraction/` — abstraction-based exact search.
- `tanaka/` — Tanaka branch-and-bound.
- `tanaka_2018/` — enhanced Tanaka branch-and-bound.
- `tanaka_voss_2022/` — Tanaka–Voß exact tree search.

## Solver IP/MIP

- `wan_2009/` — MRIP.
- `caserta_2012_brp2/` — BRP-II.
- `tang_2015_ilp/` — ILP formulation.
- `zehendner_2015/` — BRP-II-A.
- `de_melo_silva_2018/` — restricted BRP formulations.
- `galle_2018/` — CRP-I.
- `bacci_2020/` — branch-and-cut formulation.
- `tanaka_voss_2022_ip/` — IP formulation and native solver.

The registry discovers algorithm classes recursively. Directory placement
describes the solution mechanism; each class's `category`,
`compatible_problems`, and solver metadata describe its runtime behavior.
