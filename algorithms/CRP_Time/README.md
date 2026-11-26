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

Labels (relocation rule, scope, category, validation status) per algorithm: `docs/validation/algorithm_labels.md`.

| Folder | Type | Paper |
|---|---|---|
| `heuristic/lee_lee/` | 3-phase heuristic | Lee & Lee 2010, COR |
| `heuristic/lin_2015/` | priority rule; not registered (`registered = False`), superseded by `crp_time_u/lin_2015/` | Lin, Lee & Lee 2015, TRC |
| `heuristic/kim_2016/` | multi-case rule; not registered (`registered = False`), superseded by `crp_time_u/kim_2016/` | Kim, Kim & Lee 2016, C&IE |
| `heuristic/lopez_plata_2019/` | A*-based heuristic | López-Plata et al. 2019, C&IE |
| `heuristic/unluyurt_2012/` | Difference1 (also listed for CRP-R), Difference2 | Ünlüyurt & Aydın 2012 |
| `heuristic/azari_2017/` | CSUM | Azari, Eskandari & Nourmohammadi 2017 |
| `crp_time_r/forster_bortfeldt_2012/` | tree search, restricted; relocations as objective, crane time evaluated afterwards | Forster & Bortfeldt 2012, C&OR |
| `crp_time_u/forster_bortfeldt_2012/` | tree search, unrestricted (paper); validation package, not registered | Forster & Bortfeldt 2012, C&OR |
| `crp_time_u/lin_2015/` | priority rule on the multi-bay layer (`utils/multibay.py`); not registered | Lin, Lee & Lee 2015, TRC |
| `crp_time_u/kim_2016/` | four-case rule on the multi-bay layer (`utils/multibay.py`); not registered | Kim, Kim & Lee 2016, C&IE |
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
