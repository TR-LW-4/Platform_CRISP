# _shared

Universal / problem-agnostic algorithms.  Each is the **physical home**;
every problem directory (`brp_fixed/`, `crp_time/`, `brp_nonfixed/`,
`premarshalling/`, `cspp/`, `cspp_constrained/`) contains symlinks to
these so the user sees them from any problem view.

## Algorithms

| Folder | Type | Notes |
|---|---|---|
| `heuristic/greedy/` | depth-1 greedy | works on every problem that exposes an action mask |
| `evolutionary/genetic/` | GA | chromosome = action sequence, uniform crossover + random mutation |
| `rl/ppo/` | RL | actor-critic with clipped surrogate loss |
| `rl/reinforce/` | RL | policy gradient with greedy baseline |
