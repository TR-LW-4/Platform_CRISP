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

Batch construction lives in `problems/CRP_Stoch.py`. Each algorithm
folder is self-contained: it does not import another algorithm folder.
EG, ERI, and PBFS each keep their own copy of the Galle SCRP port
they need (`source/`). Bacci keeps the policy tree and its scorer.

```
algorithms/CRP_Stoch/
├── heuristic/
│   ├── bacci_2022_rirh/          # Bacci, Mattia & Ventura, Soft Comp 2022
│   │   ├── policy_tree.py        # Chance/Decision nodes + E[R]
│   │   ├── rirh_policy.py        # MinMax + RIRH (Alg. 3) + EM (Alg. 1)
│   │   └── algorithm.py          # RIRH + ExpectedMinMax
│   ├── galle_2017_eg/            # Galle et al. Expected Group
│   │   ├── algorithm.py
│   │   └── source/               # local copy: bay, bounds, EG runner
│   └── ku_arthanari_2016_eri/    # Ku & Arthanari ERI (Galle's port of the rule)
│       ├── algorithm.py
│       └── source/               # local copy: bay, bounds, ERI runner
└── exact/
    └── galle_2017_pbfs/          # Galle et al. PBFS / PBFSA
        ├── algorithm.py
        ├── run_experiment1.py
        └── source/               # local copy: bay, bounds, heuristics, A*, search
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
Zehendner's online baselines). They were **not** registered again as
separate classes. PBFS's own `source/heuristics.py` still contains
`EM` / `L` / `Rand` for upper bounds inside its A* / tree search and
for `run_experiment1.py --heuristics`.

## Galle Experiment 1 (Ku / Arthanari files)

`python main.py run` and `layout-run` do **not** scan the 1440
`crptw_instance/T271014_*.txt` files.  To reproduce the MATLAB loop in
`Experiments_1.m` (read each Ku file, then `PBFSA(..., errorGap=0)`):

```bash
cd /data/liuw2/Platform_CRISP
# Use an env with numpy (e.g. tsp_ha). The login `python` may not have it.
# Single instance: 5 stacks, 3 tiers, instance 1, 50% fill (gold opt=1.5)
/data/liuw2/conda/envs/tsp_ha/bin/python \
    algorithms/CRP_Stoch/exact/galle_2017_pbfs/run_experiment1.py --smoke

# One bay size (30 instances), compare opt against the official CSV
python algorithms/CRP_Stoch/exact/galle_2017_pbfs/run_experiment1.py \
    --fill-rate 0.5 --stacks 5 --tiers 3

# Full Experiment 1 at 50% fill (720 instances).  Use --fill-rates 0.5 0.67
# for all 1440.  Large bays can hit the 1-hour cap per instance.
python algorithms/CRP_Stoch/exact/galle_2017_pbfs/run_experiment1.py --fill-rate 0.5
```

Output CSVs go to `results/CRP-Stoch/galle_experiment1/` with the same
names as `/data/liuw2/StochasticCRP-master/Results/Experiments_1/`.
`--heuristics` also runs EG / EM / ERI / L / Rand (5000 samples, as in
MATLAB).  Bounds and `opt` are always computed.

## Backward compatibility

- All CRP-R baselines still work on `CRP-Stoch` when the algorithm's
  `compatible_problems` list contains `"CRP-Stoch"` (only `Genetic
  Algorithm` currently does).  Setting `batch_size = 1` makes the SCRP
  environment behave *exactly* as CRP-R (single realization, `E[R]`
  reduces to the deterministic reshuffle count).
- No CRP-R / CRP-U / CRP-Time / CRP-D / CRP-Stow algorithm
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
