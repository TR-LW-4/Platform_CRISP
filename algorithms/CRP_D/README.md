# CRP-D

Duplicate-group / stowage scaffold (`problems.CRP_D`).  Algorithm listing mirrors other families.

## Algorithms available

| Folder | Type | Paper | Origin |
|---|---|---|---|
| `heuristic/glah/` | metaheuristic | Jin, Zhu & Lim 2015, EJOR | **native** (same port as `CRP_Time` / `CRP_U` GLAH — keep in sync) |
| `exact/tanaka/` | B&B (exact) | Tanaka ``restricted-duplicate-1.01`` | **native** (duplicate-oriented exact adapter) |
| `exact/tanaka_2018/` | B&B (exact) | Tanaka ``restricted-distinct-1.11`` (2018) | **native** (CRP-D adapter; metric mapped to shifters) |

See `algorithms/README.md` for layout conventions.
