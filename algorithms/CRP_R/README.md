# CRP-R algorithms

Restricted block relocation with **distinct** priorities (take 1…N in
order; only blockers above the current target may move). Organised by
solution mechanism:

- `tree/`: B&B, A*, IDA*, abstraction, beam search, and other tree search.
- `solver_ip/`: solver-backed integer / mixed-integer models (and the
  Tanaka–Voß native IP binary).
- `heuristic/`: constructive heuristics, local search, metaheuristics.

Directory placement describes the mechanism. Each class's `category`,
`compatible_problems`, and solver metadata describe runtime behaviour.
`tree/` is not the same as Exact: Bacci BBS and Ting–Wu BS are heuristic
tree searches.

## Tree search

- `kim_hong_2006/` — Kim & Hong (2006) branch-and-bound.
- `zhu_2012/` — Zhu et al. (2012) IDA*-R.
- `exposito_2015/` — Expósito-Izquierdo (2015) B&B.
- `ku_arthanari_2016_abstraction/` — Ku & Arthanari (2016) abstraction + PDB.
- `tanaka/` — Tanaka & Takii (2016) B&B (`restricted-duplicate` vendor).
  Hung on CRP-R; the duplicate-priority entry point is
  `CRP_D/exact/search/tanaka_2016_r/`.
- `tanaka_2018/` — Tanaka & Mizuno (2018) restricted-distinct B&B.
- `tanaka_voss_2022/` — Tanaka & Voß (2022) comparison B&B (not the IP).
- `bacci_2019_bbs/` — Bacci et al. (2019) bounded beam search (heuristic).
- `ting_wu_2017_bs/` — Ting & Wu (2017) beam search (heuristic).

## Solver IP/MIP

- `wan_2009/` — MRIP.
- `caserta_2012_brp2/` — BRP-II.
- `tang_2015_ilp/` — ILP formulation.
- `zehendner_2015/` — BRP-II-A.
- `de_melo_silva_2018/` — restricted BRP formulations.
- `galle_2018/` — CRP-I.
- `bacci_2020/` — branch-and-cut BC-RBRP.
- `tanaka_voss_2022_ip/` — Tanaka & Voß (2022) sequence-based IP (Algorithm 1).

## Heuristic

- `kim_hong_2006/` — ENAR; also listed for CRP-Time via `compatible_problems`.
- `caserta_2009_lah/` — look-ahead heuristic.
- `caserta_2011_cm/` — corridor method.
- `caserta_2012/` — min-priority HEUR; also listed for CRP-Time.
- `lan_2013/` — Petering & Hussein LA-N; also listed for CRP-Time.
- `tang_2015/` — H1/H2.
- `jovanovic_voss_2014/` — chain heuristic.
- `jovanovic_2019_aco/` — ACO for rBRP.

The registry discovers algorithm classes recursively.
