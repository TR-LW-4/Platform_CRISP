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
| `exact/search/kim_hong_2006/` | B&B (exact) | Kim & Hong 2006, COR, Section 2.1 | **native** (single-bay) |
| `exact/search/ku_arthanari_2016_abstraction/` | DFBnB + abstraction/PDB (exact) | Ku & Arthanari 2016, COR 68 | **native** (single-bay) |
| `exact/search/zhu_2012/` | IDA* (exact) | Zhu, Qin, Lim & Zhang 2012, IEEE T-ASE | **native** (single-bay) |
| `heuristic/lin_2015/` | rule | Lin, Lee & Lee 2015, TRC | symlink (native: CRP_Time) |
| `heuristic/durasevic_2024/` | GP | Ðurasević & Ðumić 2024, ASOC | symlink (native: CRP_Time) |
| `evolutionary/genetic/` | GA | — | symlink (`_shared/`) |
