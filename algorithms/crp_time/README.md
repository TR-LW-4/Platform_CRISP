# CRP-Time

**Container Retrieval Problem with crane working-time objective.**
Multi-bay yard, fixed retrieval order, Lee–Lee (2010) RMGC kinematics
(γ_row = 1.2 s, γ_bay = 3.5 s, γ_acc = 40 s, γ_pd = 30 s).
**Objective:** minimise total crane working time.

## Algorithms available

| Folder | Type | Paper | Origin |
|---|---|---|---|
| `heuristic/lee_lee/` | 3-phase IP/MIP | Lee & Lee 2010, COR | **native** (multi-bay origin of CRP-Time) |
| `heuristic/lin_2015/` | priority rule | Lin, Lee & Lee 2015, TRC | **native** (multi-bay) |
| `heuristic/cifuentes_riff_2020/` | GRASP + RIL | Cifuentes & Riff 2020, ASOC | **native** (multi-bay) |
| `heuristic/durasevic_2024/` | GP hyper-heuristic | Ðurasević & Ðumić 2024, ASOC | **native** (multi-bay) |
| `heuristic/durasevic_2025_mgp/` | Multitask GP | Ðurasević, Ðumić & Gil-Gala 2025, EAAI | **native** (multi-bay) |
| `heuristic/kim_hong/` | rule | Kim & Hong 2006, COR | symlink (native: brp_fixed, single-bay origin) |
| `heuristic/caserta/` | rule | Caserta et al. 2012, EJOR | symlink (native: brp_fixed) |
| `heuristic/lan/` | look-ahead | Petering & Hussein 2013, EJOR | symlink (native: brp_fixed) |
| `heuristic/glah/` | metaheuristic | Jin, Zhu & Lim 2015, EJOR | symlink (native: brp_fixed) |
| `heuristic/greedy/` | depth-1 greedy | — | symlink (`_shared/`) |
| `evolutionary/genetic/` | GA | — | symlink (`_shared/`) |
| `rl/ppo/` | RL | Schulman et al. 2017 | symlink (`_shared/`) |
| `rl/reinforce/` | RL | Williams 1992 | symlink (`_shared/`) |

Symlinked `brp_fixed` papers are usable as degenerate baselines here
(their scoring ignores `γ_bay`/`γ_acc`, so they are generally suboptimal
on multi-bay yards but still produce feasible plans).
