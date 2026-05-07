# CRP-U

**Unrestricted BRP (uBRP)** in this platform: **fixed** retrieval order 1 → … → N (same as CRP-R), but **relocation moves** may relocate the **top container of any source stack** to any non-full destination — unlike **restricted** relocation under **CRP-R**, where only the blocker on the **target** stack may be moved.

For **free retrieval order** (choose which container to retrieve next), use **`BRP-NonFixed`** (`problems/brp_nonfixed.py`).

## Algorithms available

| Folder | Type | Paper | Origin |
|---|---|---|---|
| `heuristic/glah/` | metaheuristic | Jin, Zhu & Lim 2015, EJOR | **native** (fixed-order port — baseline only; keep in sync with `CRP_D` / `CRP_Time` copies) |

See `algorithms/README.md` for layout conventions.
