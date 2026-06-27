# CRP-R

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
| `heuristic/unluyurt_2012/` | rule | Ünlüyurt & Aydın 2012, J. Adv. Transp. | **native** (single-bay) |
| `exact/tanaka/` | B&B (exact) | Tanaka ``restricted-duplicate-1.01`` | **native** (vendor C ``brp_bb``) |
| `exact/tanaka_2018/` | B&B (exact) | Tanaka ``restricted-distinct-1.11`` (2018) | **native** (vendor C ``brp_bb``) |
| `heuristic/lin_2015/` | rule | Lin, Lee & Lee 2015, TRC | symlink (native: CRP_Time) |
| `heuristic/durasevic_2024/` | GP | Ðurasević & Ðumić 2024, ASOC | symlink (native: CRP_Time) |
| `evolutionary/genetic/` | GA | — | symlink (`_shared/`) |
