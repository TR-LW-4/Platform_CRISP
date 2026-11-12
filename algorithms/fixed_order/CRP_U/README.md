# CRP-U algorithms

Unrestricted block relocation with **distinct** priorities (take 1…N in
order; any stack-top may move to any non-full stack). Organised by
solution mechanism:

- `tree/`: exact B&B / IDA* / IDBB and heuristic tree search.
- `solver_ip/`: solver-backed integer / mixed-integer models.
- `heuristic/`: constructive heuristics, local search, metaheuristics.

Directory placement describes the mechanism. Each class's `category`,
`compatible_problems`, and solver metadata describe runtime behaviour.

## Tree search

- `forster_bortfeldt_2012/` — heuristic compound-move tree search.
- `zhu_2012/` — Zhu et al. (2012) IDA*-U.
- `tanaka_mizuno_2018/` — Tanaka & Mizuno (2018) unrestricted-distinct B&B.
- `jin_tanaka_2023/` — Jin & Tanaka (2023) IDB&B (distinct as a special case
  of the duplicate-priority algorithm; CRP-D has the duplicate adapter).

## Solver IP/MIP

- `caserta_2012_brp1/` — BRP-I.
- `petering_hussein_2013/` — BRP-III.
- `de_melo_silva_2018/` — unified BRP formulations.
- `lu_zeng_liu_2019/` — BRP-m3 and IS*.

## Heuristic

- `glah/` — Jin, Zhu & Lim (2015) GLAH. This is the only GLAH port;
  there is no GLAH under CRP-Time or CRP-D.
- `tricoire_2018/` — Tricoire, Scagnetti & Beham heuristics.
- `exposito_2014_dsk/` — Expósito-Izquierdo et al. domain-specific heuristic.
- `feillet_2019_ls/` — Feillet, Parragh & Tricoire local search.
- `jovanovic_2019_aco/` — ACO for uBRP.

The registry discovers algorithm classes recursively.
