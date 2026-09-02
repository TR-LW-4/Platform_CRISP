# CRP-Stoch algorithms

Algorithms for the **Stochastic Container Relocation Problem** (SCRP)
as implemented in `problems/CRP_Stoch.py`.

## Model recap

- Containers are partitioned into ordered **batches** `B_1, ..., B_b`
  (config knob `batch_size`, default `1`).
- Precedence: all containers in `B_k` are retrieved before any in
  `B_{k+1}`.
- Intra-batch order is revealed only when `B_{k-1}` is empty.
- The primary metric is the **expected number of relocations** `E[R]`
  over the uniform distribution on `|B_k|!` intra-batch permutations
  (Bacci, Mattia & Ventura 2022; Galle et al. 2018).
- `batch_size = 1` degenerates to deterministic CRP-R.

## Layout

Shared infrastructure lives under `core/stoch/` (same convention as
`core/stoch/policy_tree.py` and `core/stoch/batch_layout.py`); each
paper gets its own `author_year_shortname/` folder under `heuristic/`
or `exact/`, consistent with every other problem in this repository
(e.g. `algorithms/CRP_R/heuristic/caserta_2009_lah/`).

```
core/stoch/
├── policy_tree.py
├── batch_layout.py
└── galle_2017_source/            # Port of /data/liuw2/StochasticCRP-master
    ├── bay_utils.py               # readInputFile / GenerateIncompleteConfig /
    │                               # Pre_Processing / UnvielContainers*
    ├── bounds.py                  # BlockingLowerBound / RollingLowerBound /
    │                               # boundsDifference
    ├── heuristics.py              # EG / EM / ERI / L / Rand retrieval policies
    ├── astar.py                   # deterministic CRP tail solver
    └── tree_search.py             # PBFS_Online / PBFSA (exact if error_gap=0)

algorithms/CRP_Stoch/
├── heuristic/
│   ├── bacci_2022_rirh/          # Bacci, Mattia & Ventura, Soft Comp 2022
│   │   ├── rirh_policy.py        # MinMax + RIRH (Alg. 3) + EM (Alg. 1)
│   │   └── algorithm.py          # RIRH + ExpectedMinMax BaseAlgorithms
│   ├── galle_2017_eg/            # Galle et al. SCRP repository (new heuristic)
│   │   └── algorithm.py          # ExpectedGroupAssignment (EG)
│   └── ku_arthanari_2016_eri/    # Ku & Arthanari, EJOR 2016
│       └── algorithm.py          # ExpectedReshufflingIndex (ERI)
└── exact/
    └── galle_2017_pbfs/          # Galle et al. SCRP repository (exact search)
        └── algorithm.py          # PBFSBatchExact + PBFSAApprox + PBFSOnlineExact
```

## Registered algorithms

| Class | GUI name | Category | Paper |
|---|---|---|---|
| `RIRH` | Bacci et al. (2022) RIRH | Heuristic | Bacci, Mattia & Ventura, *Soft Computing* 2022 |
| `ExpectedMinMax` | Galle et al. (2018) Expected MinMax | Heuristic | Galle, Manshadi, Barnhart & Jaillet, *Transp. Sci.* 2018 |
| `ExpectedGroupAssignment` | Galle et al. Expected Group (EG) | Heuristic | Galle et al. SCRP repository (new heuristic) |
| `ExpectedReshufflingIndex` | Ku & Arthanari (2016) ERI | Heuristic | Ku & Arthanari, *EJOR* 2016 |
| `PBFSBatchExact` | Galle et al. PBFS [batch exact] | Exact | Galle et al. SCRP repository |
| `PBFSAApprox` | Galle et al. PBFSA [batch approx] | Heuristic | Galle et al. SCRP repository |
| `PBFSOnlineExact` | Galle et al. PBFS [online exact] | Exact | Galle et al. SCRP repository |

### Why no duplicate EM / L / Random entries

The source repository (`/data/liuw2/StochasticCRP-master`) also ships
`retrieveEM.m` and `retrieveL.m` / `retrieveRand.m`, but those already
have native or primary homes elsewhere in this platform
(`bacci_2022_rirh`'s `ExpectedMinMax`, and the primary
`CRP_Online/heuristic/zehendner_2017_level/` implementation for
Zehendner's online baselines). They were **not** re-embedded here as
separate "[ported]" classes to avoid duplicate GUI entries; the shared
`core.stoch.galle_2017_source.heuristics` module still contains the
ported `EM`/`L`/`Rand` retrieval functions for internal reuse (e.g. as
upper-bound heuristics inside `astar.py` / `tree_search.py`), they are
just not exposed as separate top-level algorithms.

## Backward compatibility

- All CRP-R baselines still work on `CRP-Stoch` when the algorithm's
  `compatible_problems` list contains `"CRP-Stoch"` (only `Genetic
  Algorithm` currently does).  Setting `batch_size = 1` makes the SCRP
  environment behave *exactly* as CRP-R (single realization, `E[R]`
  reduces to the deterministic reshuffle count).
- No CRP-R / CRP-U / CRP-Time / CRP-D / CRP-Stow / CRP-Prem algorithm
  file was modified.

## References

- T. Bacci, S. Mattia, P. Ventura. "The realization-independent
  reallocation heuristic for the stochastic container relocation
  problem." *Soft Computing* (2022).
  DOI: 10.1007/s00500-022-07070-3
- V. Galle, S. Borjian Boroujeni, V. H. Manshadi, C. Barnhart, P.
  Jaillet. "The Stochastic Container Relocation Problem" (source
  repository, 2017); published as Galle et al., *Transportation
  Science* 52(5) (2018).  Source repository vendored/ported from
  `/data/liuw2/StochasticCRP-master`
  (`https://github.com/vgalle/StochasticCRP`).
- D. Ku, T. S. Arthanari. "Container relocation problem with time
  windows for container departure." *European Journal of Operational
  Research* 252 (2016) 1031–1039.
