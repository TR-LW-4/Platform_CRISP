# CRP-Online algorithms

Algorithms for the **Online Container Relocation Problem** as implemented in
`problems/CRP_Online.py`.

## Model recap

- Yard dynamics match restricted CRP-R: distinct priorities 1...N and only
  blockers above the current target may be relocated.
- The difference from CRP-R is informational rather than physical: the policy
  sees only the current target plus an optional look-ahead horizon
  `lookahead_h`.
- `lookahead_h = 0` matches Zehendner, Feillet & Jaillet (2017).
- The primary metric is the realized number of relocations on the revealed
  trajectory, not an expectation over stochastic batch realizations.

## Layout

```
algorithms/CRP_Online/
└── heuristic/
    └── zehendner_2017_level/
        ├── policies.py         # L / R / M pick rules + Theorem 1 bound
        └── algorithm.py        # LevelingHeuristic + R + M BaseAlgorithms
```

## Registered algorithms

| Class | GUI name | Category | Paper |
|---|---|---|---|
| `LevelingHeuristic` | Zehendner et al. (2017) Leveling L | Heuristic | Zehendner, Feillet & Jaillet, *EJOR* 2017 |
| `RandomStackHeuristic` | Zehendner et al. (2017) Random R | Heuristic | (same paper, baseline) |
| `RightNeighbourHeuristic` | Zehendner et al. (2017) Right-Neighbour M | Heuristic | (same paper, baseline) |

## References

- E. Zehendner, D. Feillet, P. Jaillet. "An algorithm with performance
  guarantee for the Online Container Relocation Problem."
  *European Journal of Operational Research* 259 (2017) 48-62.
  DOI: 10.1016/j.ejor.2016.09.011
