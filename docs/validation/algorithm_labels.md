# CRP-Time algorithms: labels

Status 26 Nov 2026 (CRISP `64bea69`). One row per CRP-Time algorithm: the ones offered in the Workbench for CRP-Time, the unregistered multi-bay versions, and the two old multi-bay classes that are no longer registered. Labels shown in the Workbench are class attributes (`core/algorithm_meta.py`): `relocation_rule`, `geometry` (scope), `time_category` (category), `validation_status` with `validation_evidence`, and `fidelity` (adapted flag). An undeclared label is not shown.

**NEEDS DECISION** marks a label that cannot be set from the definitions below without a choice. These labels stay undeclared in the code. The options and their evidence are listed after the table.

## Definitions (fixed, 26 Nov 2026)

- **Relocation rule.** Restricted: only containers above the current target are relocated. Unrestricted: otherwise. Based on the paper and the code. Where they differ, both are given.
- **Category.** 1: the selection rules or the search use crane travel or time, exact or as a distance proxy (`time-in-selection`). 2: selection uses only relocation, priority or move-count information, and crane time is evaluated afterwards (`time-afterwards`).
- **Validation status.** Validated: passed its pre-registered criterion. Partially validated: implemented per paper, criterion not met. Not validated: otherwise. Every status cites the document and section where the criterion and its result are recorded. Without such a citation the status is not validated.
- **Adapted.** A deliberate change for CRP-Time (`fidelity = "adapted"`).

Document paths are relative to `docs/validation/`. `data/<folder>/README.md` is in the nested validation data repository.

## Table

| Algorithm | Paper | Module (`algorithms/…`) | Registered | Relocation rule | Scope | Category | Status | Adapted | Evidence |
|---|---|---|---|---|---|---|---|---|---|
| Azari et al. (2017) CSUM | Azari, Eskandari & Nourmohammadi (2017), Sci. Iranica E 24(1) | `CRP_Time/heuristic/azari_2017` | yes | restricted (paper p. 312–313; code) | single-bay | 1: CSUM keeps a plan by its crane time (tcw ≤ Tbest, Fig. 3; `algorithm.py:397`) | **NEEDS DECISION (S1)** | no | `azari_2017.md` §1 (rule, CSUM), §4 (Checks 1, 2, 5, 6 pass; Check 3 not done) |
| Jovanović et al. (2019) ACO (CRP-Time) | Jovanović, Tuba & Voß (2019), EJOR 274(1) | `CRP_Time/heuristic/jovanovic_2019_aco` | yes | restricted (Sec. 6, p. 85; Sec. 7.4, p. 88; code) | single-bay | 1: solution quality and the lower bound LB_f in crane time (Sec. 6, p. 85) | **NEEDS DECISION (S1)** | no | `jovanovic_2019.md` §1 (crane-time extension), §4 (Checks 1–4) |
| López-Plata et al. (2019) A∗-Based Heuristic | López-Plata, Expósito-Izquierdo & Moreno-Vega (2019), C&IE | `CRP_Time/heuristic/lopez_plata_2019` | yes | restricted (p. 437; code) | single-bay | 1: g, the greedy start and postprocessing use f2 or f2vert (paper: operating cost from travel, eq. 1–3) | partially validated | **NEEDS DECISION (A1)**; declared no | `lopez_plata_2019.md` §0 (V42 entry: V22 criteria quoted unchanged, committed before V42), §3 (V22, V42): Fig. 8 met on 3×3 and 3×4; not on 3×5 (92.50 vs 97.50 % optimal) and 4×4 (95.00 vs 100.00); §4 (conversion) |
| Ünlüyurt–Aydın (2012) Difference1 | Ünlüyurt & Aydın (2012), J. Adv. Transp. 46(4) | `CRP_Time/heuristic/unluyurt_2012/difference1.py` | yes (CRP-Time, CRP-R) | restricted (p. 380, 382; code) | single-bay | 2: three rules on retrieval orders only (Sec. 4.2, p. 386–387); see remark C1 | **NEEDS DECISION (S1)** | no | `unluyurt_2012.md` §1.2, §1.4, §3 (all cases pass), §2 O11 (paper's results not checkable) |
| Ünlüyurt–Aydın (2012) Difference2 | as above | `CRP_Time/heuristic/unluyurt_2012/difference2.py` | yes | restricted (p. 380, 382; code) | single-bay | 1: stacks on the truck-lane side first (p. 387), a distance proxy for the paper's Objective2 = A·pickups + B·horizontal distance (Eq. 2, p. 382) | **NEEDS DECISION (S1)** | no | `unluyurt_2012.md` §1.3, §1.4, §2 O2, O7, O10, §3.2 (cases checked by hand by Thom), §4.3 |
| Forster–Bortfeldt (2012) Tree Search, restricted | Forster & Bortfeldt (2012), C&OR 39, 299–309 | `CRP_Time/crp_time_r/forster_bortfeldt_2012` | yes | paper: unrestricted (p. 302); code: restricted (R1; Wei, 10 Nov 2026) | single-bay | 2: minimises moves; crane time from `finish()` (Wei, 10 Nov 2026) | not validated (hard checks only) | yes | `forster_bortfeldt_2012_restricted.md` §3 (R1), §5, §8, §10 (I1), §11; base package: `forster_bortfeldt_2012.md` §5, §7, §9 |
| Firmino et al. (2019) Reactive GRASP | da Silva Firmino, de Abreu Silva & Times (2019), J. Heuristics 25 | `CRP_Time/heuristic/firmino_2019_rgrasp` | yes | restricted (paper p. 148, checked by Thom; code: construction relocates only above the target, `algorithm.py:224`) | single-bay | 1: local search keeps a rebuilt plan only if its crane time is lower (`algorithm.py:379–392`); paper beyond its title not checked for this label | not validated (future work) | **NEEDS DECISION (A1)**; declared no | no validation log; `../benchmark/algorithms_overview.md` (known issues) |
| Lee–Lee (2010) Retrieval | Lee & Lee (2010), C&OR 37(6) | `CRP_Time/heuristic/lee_lee` | yes (CRP-Time, CRP-R) | **NEEDS DECISION (R1)**: paper unrestricted (Example 1, p. 1145); code restricted | multi-bay | 1: Phase 3 time reduction (p. 1143–1144; `phase3.py:42–96` keeps a change only if crane time drops) | not validated (parked) | yes | `lee_lee_2010.md` §1, §2, §3.3, §4 (Checks 1 and 2 fail) |
| Caserta (2012) HEUR | outside the thesis scope | `CRP_R/heuristic/caserta_2012` | yes (CRP-R, CRP-Time) | **NEEDS DECISION (R2)**; code restricted | single-bay | 2 (code): no distance or time in `scoring.py`; crane time attached afterwards | not validated | no | no CRP-Time log; `../benchmark/algorithms_overview.md` ("Also offered for CRP-Time"). Labels not set in code (L1) |
| Kim–Hong (2006) ENAR | dropped from the thesis (no paper available) | `CRP_R/heuristic/kim_hong_2006` | yes (CRP-R, CRP-Time) | **NEEDS DECISION (R2)**; code restricted | single-bay | 2 (code): no distance or time in `scoring.py` | not validated | no | as above (L1) |
| LA-N Look-Ahead | outside the thesis scope | `CRP_R/heuristic/lan_2013` | yes (CRP-R, CRP-Time) | **NEEDS DECISION (R2)**; code restricted | single-bay | 2 (code): no distance or time in `planner.py` | not validated | no | as above (L1) |
| Forster–Bortfeldt (2012), unrestricted single bay | Forster & Bortfeldt (2012), C&OR 39 | `CRP_Time/crp_time_u/forster_bortfeldt_2012` (`search.py`) | no (validation package; I5) | unrestricted (p. 302; code) | single-bay | 2: objective is the number of moves | partially validated | no | `forster_bortfeldt_2012.md` §5 (pre-registered criterion), §7 (verdict: not met), §9 (conclusion) |
| Forster–Bortfeldt (2011) multi-bay, unrestricted | Forster & Bortfeldt, Oper. Res. Proc. 2011, 257–262 | `CRP_Time/crp_time_u/forster_bortfeldt_2012/multibay.py` | no | unrestricted (C&OR search per bay, P10; relocations stay in their bay, P9) | multi-bay | 2: per-bay search on moves (P10); merge by group order (Algorithm 1, P8); crane time computed afterwards (p. 261) | validated | no | `forster_bortfeldt_2011_multibay.md` §8 (criterion, pre-registered 13 Nov 2026), §11.1 ((A) PASS, (B) PASS 20/20) |
| Forster–Bortfeldt (2011) multi-bay, restricted | as above | `CRP_Time/crp_time_r/forster_bortfeldt_2012/multibay.py` | no | restricted (R1, O4: relocations in the bay, from the stack holding the block's lowest group) | multi-bay | 2: as the unrestricted version | not validated (hard checks only) | yes | `forster_bortfeldt_2012_restricted.md` §7 (O4, "Label: adapted"), §9 (every hard check passes on the 70 Lee & Lee instances) |
| Lin, Lee & Lee (2015), multi-bay | Lin, Lee & Lee (2015), TR-C 52, 132–143 | `CRP_Time/crp_time_u/lin_2015` | no | unrestricted (Rule II moves a container k from another stack, p. 137; code: 22.4 % of relocations from a non-target stack) | multi-bay | 1 at the comparison setting (P_r, P_b) = (30, 300): SSI = ming + P_r·row + P_b·\|bay − bay(s(c))\| (Eq. (1), p. 137; `heuristic.py:90`). At (0, 0), SSI = ming, which is category 2 | validated at (P_r, P_b) = (0, 0) against Lin Tables 1–2. The comparison setting (30, 300) differs only in the Eq. (1) weights; under it the same criterion fails (18 of 30 rows) | no | `data/lin_2015_multibay/README.md`, "Validation" (L1, pre-registered 23 Nov 2026) and "Results" (S2 = (0, 0) PASS 30/30; S1 = (30, 300) FAIL) |
| Kim, Kim & Lee (2016), multi-bay | Kim, Kim & Lee (2016), C&IE 101, 352–360 | `CRP_Time/crp_time_u/kim_2016` | no | unrestricted (Cases 3–4 move tops of other critical stacks, p. 355; code: 57.1 % of relocations from a non-target stack) | multi-bay | 2: orders only, no bay or row term (K6) | partially validated | no | `data/kim_2016_multibay/README.md`, "Validation" (KM1, pre-registered 23 Nov 2026), "Results" (40/70), "Sensitivity results" (no variant passes) |
| Lin–Lee–Lee (2015) Rule (old) | Lin, Lee & Lee (2015) | `CRP_Time/heuristic/lin_2015` | no (since `64bea69`; superseded by `crp_time_u/lin_2015`) | paper: unrestricted (p. 137). Code: Rule II never moves k, so only the target's blocker is relocated | multi-bay | 1: SSI with P_r = 30, P_b = 300 (`scoring.py:120–123`) | not validated | no | no log; known deviations in the module header (Rule II never moves k; unseeded `random.choice`) |
| Kim (2016) Multi-Case Heuristic (old) | Kim, Kim & Lee (2016) | `CRP_Time/heuristic/kim_2016` | no (since `64bea69`; superseded by `crp_time_u/kim_2016`) | paper: unrestricted (p. 355). Code: only the blocker above the target (the `restricted` parameter has no effect) | multi-bay | 2: orders only | not validated | no | no log; known deviations in the module header (`restricted` has no effect; the pre-relocations of Cases 3 and 4 are missing) |

## Open decisions

### S1: validation status of Azari, Jovanović, Difference1 and Difference2

None of these four logs records a criterion that was pre-registered before the runs. Under the definition, no citation means not validated. CLAUDE.md and `../benchmark/algorithms_overview.md` call all four "validated".

- **Azari.**
  - For *validated*: the paper's worked examples are reproduced: Fig. 1 / Def. 3 (Check 1) and Fig. 4 (Check 2). The expected values come from the paper, so they were fixed before the runs. GBH and CSUM follow Figs. 2–3 (Checks 5, 6).
  - For *partially validated*: the paper's reported results (Table 2) were not checked, because the instances and the time model are not available (Check 3, `azari_2017.md` §4).
  - For *not validated*: no criterion is written in the log as pre-registered.
- **Jovanović.**
  - For *partially validated*: relocations are within ±0.10 of Table 1 on 19 of 21 sizes (10 × 6 +0.55, 10 × 10 +0.58). O_f is above the paper's error with eq. 44 as printed. O_v is far below it, unexplained (`jovanovic_2019.md` summary, §4).
  - For *not validated*: the ±0.10 band is not recorded as fixed before the runs. The conditions that were fixed in advance (§4, repair steps) concern diagnoses, not a validation criterion.
- **Difference1 and Difference2.**
  - For *validated*: every case of §3 passes. These are Fig. 6(c) and Fig. 9 (checked against the PDF by Thom), plus the constructed cases D1–D4 for Difference2 (checked by hand by Thom). The expected destinations follow from the paper's rules and the decisions of §2.
  - For *not validated*: the expected values partly depend on our own decisions (O2–O7). The paper's numerical results cannot be checked (O11). No criterion is recorded as pre-registered.

### R1: relocation rule of Lee–Lee

- *Unrestricted* (paper): Example 1 moves container 6 (6:4,2) while 6 does not block 2 (p. 1145; `lee_lee_2010.md` §1).
- *Restricted* (code): Phase 1 relocates only containers above the target, and Phases 2–3 only rewrite existing paths (`lee_lee_2010.md` §2, §3.3). CRP-Time's `validate_plan` accepts only restricted plans.

Precedent: Forster–Bortfeldt restricted declares the code's rule (restricted) and is flagged adapted. Lee–Lee is already flagged adapted.

### R2 and L1: the three CRP_R heuristics offered for CRP-Time

- **Rule.** In code all three are restricted: they are CRP-R algorithms, and CRP-Time's step is restricted. On 3×5 #1–5 all their plans were restricted (`../benchmark/algorithms_overview.md`). Their papers were not checked: they are outside the thesis scope, and Kim–Hong has no paper available. CLAUDE.md: "UI labels declare only confirmed rules".
- **L1, labels in code.** Not set. These classes belong to the CRP-R family, and their labels would also show in the CRP-R Workbench, where category and CRP-Time validation status mean nothing. Whether they stay offered for CRP-Time is an open question for Wei. Kim–Hong is dropped from the thesis.
- Options:
  - (a) set category 2 and not validated in the CRP_R classes;
  - (b) show category and status only for CRP-Time (a frontend change);
  - (c) remove CRP-Time from their `compatible_problems`.

### A1: adapted flag of López-Plata and Firmino

- *Adapted*: the objective is changed for CRP-Time.
  - López-Plata: from the paper's operating cost (eq. 1–3) to f2 or f2vert, with a corrected lower bound h, since a literal translation is not admissible (`lopez_plata_2019.md` §4).
  - Firmino: from Lee & Lee kinematics (code description) to the selected f2 or f2vert.
- *Not adapted* (declared now): the conversion is required by the unified setting, category (a). The paper's logic is kept (`lopez_plata_2019.md` §0, "keeping the paper's logic intact").
- Consequence: `fidelity = "adapted"` adds the Workbench note "published numbers are not expected to match".

Not in A1:
- Jovanović: the paper's crane-time setting equals f2 (`jovanovic_2019.md` summary). The LB_f correction is Wei's fix of an invalid bound, not a change for CRP-Time.
- Azari: the paper gives no time formulas, so the Voß & Schwarze model fills a gap (`azari_2017.md` §3.1).

### C1 (remark, not open): tie-breaking in Difference1

Difference1 is category 2 by its rules. Ties occur only between empty stacks under rule 1. They go to the lowest stack index (O3, our choice (b)). That is the stack nearest the truck lane in f2, so the tie rule can change the crane time but not the relocations (`unluyurt_2012.md` §2 O3). Other algorithms break ties by stack order in the same way (López-Plata O19; Kim multi-bay K5). Counting this as a distance proxy would move Difference1 to category 1.
