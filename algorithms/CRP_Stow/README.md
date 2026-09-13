# CRP-Stow algorithms

Physical algorithm homes for the CRP-Stow (BRLP / POCRP / POCRP-RC) problem
family.

| Category | Paper key                | Algorithm | RC-aware |
|----------|--------------------------|-----------|:--------:|
| Heuristic | `jovanovic_2019/`       | Jovanović (2019) GR-C + GRASP           | no |
| Heuristic | `wang_2026_grasp/`      | **Wang, Ma, Yang & Hu (2026) GRASP for POCRP-RC** | **yes** |
| Exact     | `exact/tanaka_voss_2019/` | Tanaka & Voß (2019) B&B for BRPSP     | no |

`wang_2026_grasp/` is the paper's *main* algorithm (§5.3, GRASP) — implements
the two RC-specific rules `TR` (§5.2.2) and `RR` (§5.2.3) plus the LNS
neighbourhood correction with the paper's original poor-target criteria
(2.1, 2.2).  Only exposes ``Wang (2026) GRASP`` in the GUI; the underlying
Greedy is kept as an internal construction routine to match the user's
"one paper = one main algorithm" convention.

Requires `ProblemConfig.rc_ratio > 0` to exercise the RC heuristics.
`rc_ratio = 0.0` degenerates the problem to POCRP (Jovanović 2019 setting)
and is fully backward-compatible with all other baselines above.

See `algorithms/README.md` for the overall layout convention and
`problems/CRP_Stow.py` for the RC extension details.

The Web Workbench exposes both random layouts and the official Jovanović
BRLP benchmark under **CRP-Stow BRLP Benchmark**.  Benchmark classes follow
`Bay-A-VS-YS-YT_seed.pro`; selecting one class queues its 40 seeded
instances, while **First instance only** is available for smoke testing.
