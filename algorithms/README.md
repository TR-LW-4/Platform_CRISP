# Algorithms

Algorithms are organised **problem-family-first**, PlatEMO-style. Open
any problem directory and the algorithms for that family live inside it.

How the problem families relate (same axes as the root README):

```
任务 / 目标     Time | MO | Prem | Stow | Stoch
翻箱规则        R vs U                         (distinct CRP only)
优先级          R/U = distinct；D = duplicate  (D: extra restricted_relocation)
几何            num_bays                       (config, not a family)
信息揭示        Online | batch-stochastic      (separate families when user-facing)
```

Per-paper lists are in each family's README. The live GUI / CLI catalog is
`python main.py list`.

## Directory layout

```
algorithms/
├── _shared/                     # helpers only — not registered solvers
│   └── solver/                  # shared Gurobi utilities
│
├── CRP_R/                       # restricted + distinct
│   ├── tree/                    # B&B, IDA*, abstraction, beam search, …
│   ├── solver_ip/               # Gurobi IP/MIP (+ Tanaka–Voß native IP)
│   └── heuristic/
│
├── CRP_U/                       # unrestricted + distinct
│   ├── tree/
│   ├── solver_ip/
│   └── heuristic/
│
├── CRP_D/                       # duplicate priorities (Du default, Dr optional)
│   ├── exact/search/
│   ├── exact/solver/
│   └── heuristic/
│
├── CRP_Time/                    # same rules as CRP-R; crane-time objective
│   ├── heuristic/
│   └── exact/                   # empty scaffold (search/ + solver/)
│
├── CRP_MO/                      # same rules as CRP-Time; (R, T) vector
│   ├── evolutionary/            # empty scaffold
│   └── heuristic/               # empty scaffold
│
├── CRP_Online/                  # progressively revealed fixed-order CRP
│   └── heuristic/
│
├── CRP_Prem/                    # pre-marshalling
│   ├── heuristic/
│   ├── exact/search/
│   ├── exact/solver/
│   └── evolutionary/
│
├── CRP_Stow/                    # BRLP / POCRP / POCRP-RC
│   ├── heuristic/
│   └── exact/
│
└── CRP_Stoch/                   # stochastic CRP
    ├── heuristic/
    └── exact/
```

`tree/` (CRP-R / CRP-U) versus `exact/search` (CRP-D / Prem / …) are
historical folder names for the same idea: search-based methods. Runtime
behaviour comes from each class's `category`, not from the folder name.
`tree/` also holds heuristic tree searches (beam search, compound-move
search); those classes set `category = "Heuristic"`.

## Conventions

* **One physical home per algorithm** — whichever problem the paper
  originally targeted.
* **Symlinks** (`mode 120000` in git) *may* be used when the same code
  is registered under another family via `compatible_problems`. Many
  cross-family listings skip the symlink and rely on
  `compatible_problems` alone (registry still finds the class).
* **Problem-specific ports** use independent copies when the adapter or
  environment integration differs (for example Jin–Tanaka 2023 under
  both CRP-U and CRP-D).
* **`_shared/`** holds cross-algorithm helpers (currently Gurobi
  utilities). It is not a home for registered algorithms.
* **`core/gp/`** (not inside `algorithms/`) hosts the platform-level
  GP engine. Any GP paper should import from `core.gp.*`.
* **Registry** (`core.registry`) auto-discovers classes via
  `pkgutil.walk_packages`; the `name` class attribute deduplicates
  imports that come in through both a physical path and a symlink.

## Adding a new algorithm

1. Decide the primary problem (= the one the paper was written for).
2. `mkdir algorithms/<primary_problem>/<category>/<paper_key>/` and
   drop `__init__.py` + the algorithm module(s) there.
3. Declare `compatible_problems = [...]` on the algorithm class.
4. Optionally, for each additional problem in `compatible_problems`,
   add a symlink:

   ```bash
   ln -s ../../<primary_problem>/<category>/<paper_key> \
         algorithms/<other_problem>/<category>/<paper_key>
   ```
5. Restart the GUI — the registry picks it up automatically.
