# Algorithms

Algorithms are organised **problem-family-first**, PlatEMO-style.  Open
any problem directory and every algorithm applicable to that problem is
listed directly inside it (as a real folder for the paper's native
problem, or as a symlink for cross-problem support).

## Directory layout

```
algorithms/
├── _shared/                  # universal algorithms (physical home)
│   ├── evolutionary/genetic/
│   └── rl/{ppo,reinforce}/
│
├── CRP_D/                # duplicate-group stowage (CRP-D scaffold)
│   ├── heuristic/
│   │   └── glah/             ← physical (Jin 2015 GLAH — CRP-D packaging)
│   └── exact/
│       ├── tanaka/            ← physical (Tanaka 2016 CRP-D adapter)
│       └── tanaka_2018/       ← physical (Tanaka 2018 CRP-D adapter)
│
├── CRP_R/                # fixed-order yard retrieval (CRP-R problem family)
│   ├── heuristic/
│   │   ├── kim_hong/         ← physical (original paper = single-bay CRP-R)
│   │   ├── caserta/          ← physical
│   │   ├── lan/              ← physical
│   │   ├── lin_2015/         → symlink
│   │   └── durasevic_2024/   → symlink
│   ├── exact/
│   │   ├── tanaka/            ← physical (Tanaka 2016 B&B + vendor C)
│   │   └── tanaka_2018/       ← physical (Tanaka 2018 B&B + vendor C)
│   ├── evolutionary/genetic/ → symlink
│   └── rl/{ppo,reinforce}/   → symlinks
│
├── CRP_Time/                 # multi-bay CRP with crane-time objective
│   ├── heuristic/
│   │   ├── lee_lee/          ← physical (Lee & Lee 2010, COR)
│   │   ├── lin_2015/         ← physical (Lin, Lee & Lee 2015, TRC)
│   │   ├── cifuentes_riff_2020/  ← physical (Cifuentes & Riff 2020, ASOC)
│   │   ├── durasevic_2024/   ← physical (Ðurasević & Ðumić 2024, ASOC)
│   │   ├── durasevic_2025_mgp/   ← physical (Ðurasević et al. 2025, EAAI)
│   │   ├── kim_hong/         → symlink to CRP_R
│   │   ├── caserta/          → symlink
│   │   ├── lan/              → symlink
│   │   └── glah/             ← physical (CRP-Time packaging; sync with CRP_D / CRP_U)
│   ├── evolutionary/genetic/ → symlink
│   └── rl/{ppo,reinforce}/   → symlinks
│
├── CRP_U/                # free-order CRP (same env as BRP-NonFixed)
│   └── heuristic/
│       └── glah/             ← physical (Jin 2015 GLAH — CRP-U packaging)
│
├── brp_nonfixed/             # free-order BRP (GA / RL only; see also CRP_U/)
│   ├── evolutionary/genetic/ → symlink
│   └── rl/{ppo,reinforce}/   → symlinks
│
├── CRP_Prem/               # pre-marshalling (CRP-Prem)
├── CRP_Stow/               # stowage (CRP-Stow)
```

## Conventions

* **One physical home per algorithm** — whichever problem the paper
  originally targeted.  Edit-once, propagates everywhere.
* **Symlinks** (`mode 120000` in git) for every other problem in the
  algorithm's `compatible_problems`.  No file duplication, no sync
  trouble.
* **`_shared/`** holds algorithms that are not tied to any specific
  problem (GA, PPO, REINFORCE).  They appear under **every**
  problem dir via symlink.
* **`core/gp/`** (not inside `algorithms/`) hosts the platform-level
  GP engine (tree, operators, CRP terminals, restricted RS).  Any GP
  paper imports from `core.gp.*`, never from another paper's folder.
* **Registry** (`core.registry`) auto-discovers classes via
  `pkgutil.walk_packages`; the `name` class attribute deduplicates
  imports that come in through both the physical path and a symlinked
  path.

## Adding a new algorithm

1. Decide the primary problem (= the one the paper was written for).
2. `mkdir algorithms/<primary_problem>/<category>/<paper_key>/` and
   drop `__init__.py` + the algorithm module(s) there.
3. Declare `compatible_problems = [...]` on the algorithm class.
4. For each additional problem in `compatible_problems`, add a symlink:

   ```bash
   ln -s ../../<primary_problem>/<category>/<paper_key> \
         algorithms/<other_problem>/<category>/<paper_key>
   ```
5. Restart the GUI — the registry picks it up automatically.
