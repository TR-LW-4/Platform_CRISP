# BRP-Fixed

**Block Relocation Problem with a fixed, fully-known retrieval order.**
Containers have strict priorities `1 .. N`; the agent must retrieve
them in that order while relocating blocking containers.
**Objective:** minimise total relocations.

## Algorithms available

| Folder | Type | Paper | Origin |
|---|---|---|---|
| `heuristic/kim_hong/` | rule | Kim & Hong 2006, COR | **native** (single-bay) |
| `heuristic/caserta/` | rule | Caserta et al. 2012, EJOR | **native** (single-bay) |
| `heuristic/lan/` | look-ahead | Petering & Hussein 2013, EJOR | **native** (single-bay) |
| `heuristic/glah/` | metaheuristic | Jin, Zhu & Lim 2015, EJOR | **native** (single-bay) |
| `heuristic/lee_lee/` | multi-phase | Lee & Lee 2010, COR | symlink (native: crp_time) |
| `heuristic/lin_2015/` | rule | Lin, Lee & Lee 2015, TRC | symlink (native: crp_time) |
| `heuristic/cifuentes_riff_2020/` | GRASP | Cifuentes & Riff 2020, ASOC | symlink (native: crp_time) |
| `heuristic/durasevic_2024/` | GP | Ðurasević & Ðumić 2024, ASOC | symlink (native: crp_time) |
| `heuristic/durasevic_2025_mgp/` | Multitask GP | Ðurasević et al. 2025, EAAI | symlink (native: crp_time) |
| `heuristic/greedy/` | depth-1 greedy | — | symlink (`_shared/`) |
| `evolutionary/genetic/` | GA | — | symlink (`_shared/`) |
| `rl/ppo/` | RL | Schulman et al. 2017 | symlink (`_shared/`) |
| `rl/reinforce/` | RL | Williams 1992 | symlink (`_shared/`) |
