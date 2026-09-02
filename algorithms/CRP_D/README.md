# CRP-D algorithms

Duplicate-priority block relocation (`problems.CRP_D`). This is **not**
stowage (that is CRP-Stow).

Containers are grouped. Groups leave in order; within a group any
accessible container may be retrieved. Relocation mode is
`extra["restricted_relocation"]`:

- `False` (default) — unrestricted **Du** (same move rule as CRP-U).
- `True` — restricted **Dr** (only the chosen target-group stack).

## Algorithms available

| Folder | Type | Paper | Notes |
|---|---|---|---|
| `exact/search/jin_tanaka_2023/` | exact IDB&B | Jin & Tanaka, EJOR 2023 | Unrestricted duplicate (Du); distinct is a special case |
| `exact/search/tanaka_2016_r/` | exact B&B | Tanaka & Takii, T-ASE 2016 | Restricted duplicate (Dr), vendor `restricted-duplicate` |
| `exact/solver/de_melo_silva_2018/` | Gurobi | de Melo da Silva et al., EJOR 2018 | Grouped BRP and r-BRP |
| `heuristic/kim_hong_2006/` | heuristic | Kim & Hong, COR 2006 | Group-aware ENAR |
| `heuristic/forster_bortfeldt_2012/` | heuristic tree | Forster & Bortfeldt, COR 2012 | |
| `heuristic/zeng_2019/` | heuristic | Zeng 2019 | H₁–H₅ |

There is no GLAH port here, and no `exact/tanaka_2018/` adapter.
The Tanaka 2016 binary also appears under `CRP_R/tree/tanaka/` (hung on
CRP-R); this folder is the duplicate-priority entry.

See `algorithms/README.md` for layout conventions.
