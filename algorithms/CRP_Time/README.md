# CRP-Time algorithms

Container retrieval with a **crane working-time** objective. Dynamics
match CRP-R: distinct priorities 1…N, restricted relocation. Lee–Lee
(2010) RMGC kinematics (`γ_row = 1.2 s`, `γ_bay = 3.5 s`,
`γ_acc = 40 s`, `γ_pd = 30 s`).

**Objective:** minimise total crane working time. Relocations are
reported only. Bi-objective (R, T) methods belong in `algorithms/CRP_MO/`.

`exact/search/` and `exact/solver/` exist as empty scaffolds; no exact
Time solver is registered yet.

## Algorithms in this directory

| Folder | Type | Paper |
|---|---|---|
| `heuristic/lee_lee/` | 3-phase heuristic | Lee & Lee 2010, COR |
| `heuristic/lin_2015/` | priority rule | Lin, Lee & Lee 2015, TRC (also listed for CRP-R) |
| `heuristic/kim_2016/` | multi-case rule | Kim, Kim & Lee 2016, C&IE |
| `heuristic/lopez_plata_2019/` | A*-based heuristic | López-Plata et al. 2019, C&IE |
| `heuristic/unluyurt_2012/` | Difference1 | Ünlüyurt & Aydın 2012 (also listed for CRP-R) |
| `heuristic/azari_2017/` | CSUM | Azari, Eskandari & Nourmohammadi 2017 |
| `heuristic/forster_bortfeldt_2012/` | retrieval tree search | Forster & Bortfeldt 2012 |
| `heuristic/firmino_2019_rgrasp/` | reactive GRASP | Firmino et al. 2019 |
| `heuristic/jovanovic_2019_aco/` | ACO | Jovanović et al. 2019 |

There are no Cifuentes–Riff, Ðurasević GP/MGP, or GLAH folders here.

## Also listed in the CRP-Time GUI

These live under `algorithms/CRP_R/heuristic/` and declare
`compatible_problems` including `"CRP-Time"` (no symlink in this tree):

- Kim–Hong (2006) ENAR
- Caserta (2012) HEUR
- LA-N Look-Ahead (Petering & Hussein 2013)

They ignore bay-travel / acceleration costs in their scoring, so they
are degenerate baselines on multi-bay Time instances.
