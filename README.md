# CRP Platform

**Container Relocation & Stowage Platform**

A research platform for container-yard relocation and stowage problems. It supports heuristics, exact methods, evolutionary algorithms, and reproducible experiments, with a live web workbench.
Each problem class keeps a Gymnasium-compatible `reset` / `step` interface for step-wise simulation, solution validation, web replay, and optional learning-based extensions.

---

## Quick start

### 1. Install dependencies

```bash
conda activate rl
pip install -r requirements.txt
```

Check the install:

```bash
conda activate rl
python -c "import fastapi, gymnasium, matplotlib; print('All OK')"
```

---

### 2. Start the web workbench

```bash
conda activate rl
cd Platform_CRISP

python main.py web
# open http://127.0.0.1:8000
```

The React UI is served by FastAPI. Algorithms run in separate Python processes. Closing the browser does not stop jobs on the current server process.

---

### 3. Command line (no browser)

```bash
conda activate rl
cd Platform_CRISP

# list registered problems and algorithms
python main.py list

# smoke-test all problems
python main.py test

# run one experiment
python main.py run --problem "CRP-R" --algo "Caserta (2012) HEUR"
```

---

## Web workbench

The workbench has Workbench, Jobs, and Compare pages. Problems and algorithms are discovered from the registry. Parameter widgets come from `config_schema()`, so a new algorithm does not require React changes. Jobs run in separate processes and support live progress, stop, convergence plots, yard-snapshot replay, and result saving.

Rebuild the frontend after source changes:

```bash
cd web/frontend
npm install
npm run build
```

---

## Problem families

The platform has eight problem classes. They are not aliases of one CRP. They differ along these axes:

```
Task / objective     Time | Prem | Stow | Stoch     ← different problems; do not mix primary metrics
Relocation rule      R vs U                         ← distinct-priority CRP only
Priorities           R/U = distinct; D = duplicate  ← CRP-D also uses restricted_relocation
Geometry             num_bays                       ← a config field, not a family
Information          Online | batch-stochastic      ← separate families
```

| Problem | Literature | Description | Primary metric |
|---------|------------|-------------|----------------|
| **CRP-R** | restricted + distinct | Retrieve in order 1…N; only blockers above the current target may move | relocations |
| **CRP-U** | unrestricted + distinct | Retrieve in order 1…N; any stack top may move to any non-full stack | relocations |
| **CRP-D** | duplicate (Du by default) | Groups are ordered; order inside a group is free. `extra["restricted_relocation"]`: `False` (default) = unrestricted Du, `True` = restricted Dr. Not a stowage problem; unrelated to CRP-Stow | relocations |
| **CRP-Time** | same rules as CRP-R, different objective | Restricted + distinct; minimise total yard-crane working time | crane_time (also report relocations) |
| **CRP-Online** | OCRP / limited look-ahead | Same dynamics as CRP-R, but future retrievals are revealed with `lookahead_h`. `lookahead_h=0` matches Zehendner 2017 | relocations |
| **CRP-Prem** | pre-marshalling | No retrievals; reshuffle the bay until every stack is internally sorted | moves |
| **CRP-Stow** | BRLP / POCRP | Retrieve according to a vessel stowage plan. `rc_ratio>0` enables **POCRP-RC** (rolled containers) | relocations |
| **CRP-Stoch** | SCRP | Ordered batches; intra-batch order is uniform-random. `batch_size=1` reduces to CRP-R. This is not the same as CRP-Online progressive revelation | expected_relocations |

There is no **BRP-NonFixed** family (free choice of which container to retrieve next).

---

## Algorithms

The live catalog is the registry, not a static table on this page:

```bash
python main.py list
```

Family directories and paper lists live in `algorithms/<family>/README.md` (`CRP_R`, `CRP_U`, `CRP_D`, `CRP_Time`, `CRP_Online`, `CRP_Prem`, `CRP_Stow`, `CRP_Stoch`).

> **Placement.** One family shares one `ProblemConfig` (`num_bays`, `num_rows`, `max_tiers`, `num_containers`, …). Put a new method in `algorithms/<primary_problem>/<category>/<paper_key>/` and list runnable families in `compatible_problems`. Keep the problem definition and objective unchanged; only the method changes.
>
> **Single-bay vs multi-bay is not two problems.** It is only `num_bays`. CRP-R and CRP-Time differ by objective (relocations vs crane time); the geometry is shared.
>
> Some CRP-R heuristics (Kim–Hong 2006 ENAR, Caserta 2012 HEUR, LA-N) also appear in the CRP-Time menu via `compatible_problems`. Their code stays in `algorithms/CRP_R/heuristic/`; there is no symlink under Time. Jin (2015) GLAH currently lives only in `algorithms/CRP_U/heuristic/glah/`.

---

## Add an algorithm

Create a `.py` file under `algorithms/`:

```python
# algorithms/CRP_R/heuristic/my_rule/algorithm.py
from core.base_algorithm import BaseAlgorithm, AlgorithmConfig
import multiprocessing as mp

class MyRule(BaseAlgorithm):
    name     = "My Rule"       # label in the GUI menu
    category = "Heuristic"     # "Exact" / "Evolutionary" / "Heuristic"
    description = "Custom heuristic"
    compatible_problems = ["CRP-R"]

    def train(self, problem_factory, result_queue, stop_event):
        env = problem_factory()
        obs, info = env.reset()

        # Algorithm logic goes here. The environment validates actions,
        # updates the yard, and computes metrics.
        for step in range(self.config.max_iterations):
            if stop_event.is_set():
                break

            action = env.action_space.sample()  # replace with your rule
            obs, reward, done, _, info = env.step(action)
            if done:
                break

            if step % self.config.report_interval == 0:
                self._push(result_queue,
                    step=step,
                    metric=-reward,
                    metrics=env.get_metrics(),
                    progress=step/self.config.max_iterations,
                    snapshot=env.get_state_snapshot(),
                )

    def get_best_solution(self):
        return self._best_solution
```

Save the file and restart the GUI. **My Rule** appears in the algorithm menu automatically.

---

## Add a problem

Create a `.py` file under `problems/`:

```python
# problems/my_problem.py
from core.base_problem import BaseProblem, ProblemConfig
import numpy as np, gymnasium as gym

class MyProblem(BaseProblem):
    name        = "My Problem"
    description = "Custom container problem"
    tags        = ["relocation", "custom"]
    metric_names = ["my_metric"]

    def _setup_spaces(self):
        n = self.config.num_bays * self.config.num_rows * self.config.max_tiers
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(n,), dtype=np.int32)
        self.action_space      = gym.spaces.Discrete(n)

    def _build_episode(self):
        # Initialise the yard and place containers in self.yard
        pass

    def reset(self, seed=None, options=None):
        self.yard.clear()
        self._build_episode()
        return self._get_obs(), {}

    def step(self, action):
        # Apply the action; return (obs, reward, terminated, truncated, info)
        return self._get_obs(), 0.0, False, False, {}

    def evaluate(self, solution):
        return {"my_metric": 0.0}

    def get_metrics(self):
        return {"my_metric": 0.0}

    def _get_obs(self):
        return self.yard.get_flat_obs()
```

---

## Layout

```
Platform_CRISP/
├── core/
│   ├── container.py      container model (size/weight/type/group/priority)
│   ├── yard.py           relocation engine (Stack + Yard)
│   ├── base_problem.py   problem base class
│   ├── base_algorithm.py algorithm base class (subprocess training)
│   └── registry.py       auto-registration
├── problems/             Gymnasium-compatible environments
├── algorithms/           algorithm library
│   ├── CRP_R/
│   ├── CRP_U/
│   ├── CRP_D/
│   ├── CRP_Time/
│   ├── CRP_Online/
│   ├── CRP_Prem/
│   ├── CRP_Stow/
│   └── CRP_Stoch/
├── visualization/        rendering for the environment and web replay
├── web/
│   ├── backend/          FastAPI catalog, jobs, and results
│   └── frontend/         React + TypeScript UI
├── main.py               CLI / web entry
└── requirements.txt      dependencies
```

---

## Collaboration

The default branch is `main`. Open a branch per problem family and send a pull request. Do not push straight to `main`.

| Branch | Scope |
|--------|--------|
| `crp-time` | `algorithms/CRP_Time/` and CRP-Time experiments |

```bash
git clone https://github.com/TR-LW-4/Platform_CRISP.git
cd Platform_CRISP
git checkout crp-time
```

Conventions:

- Put a new Time method in `algorithms/CRP_Time/<category>/<paper_key>/`
- Use the same module-header style as `algorithms/CRP_R/heuristic/caserta_2009_lah/algorithm.py` (main file) and `scoring.py` (helpers)
- Push to `origin crp-time`, then open a PR into `main`

---

## FAQ

**Q: The browser does not open after `python main.py web`?**  
A: Open `http://127.0.0.1:8000` yourself.

**Q: Training is slow?**  
A: Lower `max_iterations`, or first try a cheap heuristic such as `Caserta (2012) HEUR`.

**Q: How do I reuse earlier stowage-gym code?**  
A: `problems/CRP_Stow.py` follows the original `stowage_gym.py` logic.

**Q: How are results saved?**  
A: Finished Workbench runs write to `results/`. The Compare page can summarise and export CSV.
