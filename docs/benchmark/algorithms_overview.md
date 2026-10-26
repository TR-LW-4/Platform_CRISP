# CRP-Time algorithms: overview

Status of the algorithms in `algorithms/CRP_Time` on 26 Oct 2026 (code `2d32e2d`). Each algorithm keeps the relocation rule and search scope of its paper. Single-bay (2D) and multi-bay (3D) papers are listed separately.

- **Relocation rule:** as stated in the paper, with its page. "To check" means not yet confirmed from the paper. The UI label (`relocation_rule` in the algorithm's metadata) is declared only for confirmed rules.
- **Geometry:** the paper's setting / the `geometry` declared in the code.
- **Implementation status:** one run per instance on Caserta 3×5 #1–5 under f2, with the schema defaults (Python 3.11.16, in-process, no results saved). The plan each algorithm reports was checked in three ways: for completeness and feasibility (only top containers moved, retrievals in order, height ≤ H, bay empty at the end); with the problem's `validate_plan`; and against the reported f2, f2vert and relocations, recomputed with the shared evaluator. These were equal wherever the plan was complete.
- **Validation status:** the per-method workflow, logged in `docs/validation/<paper>.md`.
- **Standardised** means the structure of the first three algorithms: the final plan goes through `utils.finish()` (`validate_plan`), `runtime_s` is measured by the algorithm itself (`utils.timer`), and the validated plan is saved with the result (`extra["moves"]`).

The platform's CRP-Time problem only executes restricted relocations: a relocation always moves the top container of the current target's stack. Caserta bays are loaded as S bays × 1 row, so an algorithm built for multi-bay blocks sees every stack as a separate bay.

## Single-bay (2D)

| Algorithm (UI name) | Paper | Relocation rule (source) | Geometry | Implementation status | Validation status | Known issues |
|---|---|---|---|---|---|---|
| Azari et al. (2017) CSUM | Azari, Eskandari & Nourmohammadi (2017), Scientia Iranica E 24(1) | Restricted (p. 312–313; [log](../validation/azari_2017.md) §1); declared | 2D / single-bay | Standardised. 3×5: 5/5 complete and feasible. 416 lines; `test_azari.py` | Validated ([azari_2017.md](../validation/azari_2017.md)) | The paper's wall-clock limits (GBH 3 s, CSUM 5 s) make results on large instances depend on the machine; read literally, CSUM can end slower than its own GBH start |
| Jovanović et al. (2019) ACO (CRP-Time) | Jovanović, Tuba & Voß (2019), EJOR 274(1) | Restricted: the crane-time variant is defined for the rBRP only (Sec. 6, p. 85; Sec. 7.4, p. 88); declared | 2D / single-bay | Standardised. 3×5: 5/5 complete and feasible. 759 lines; `test_jovanovic.py` | Validated ([jovanovic_2019.md](../validation/jovanovic_2019.md)) | Crane-time lower bound without the ts term, because eq. 44 as printed is not a valid bound (Wei's decision); O_v is below the paper's error, unexplained |
| López-Plata et al. (2019) A∗-Based Heuristic | López-Plata, Expósito-Izquierdo & Moreno-Vega (2019), C&IE | Restricted (p. 437; [log](../validation/lopez_plata_2019.md) §1.2); declared | 2D / single-bay | Standardised. 3×5: 5/5 complete and feasible (0.4–14.4 s). 996 lines; `test_lopez_plata.py` | Validated ([lopez_plata_2019.md](../validation/lopez_plata_2019.md)) | Step 2 grows steeply with stack height; the 600 s default limit is reached on 20 of 40 5×7 instances under f2 |
| Firmino et al. (2019) Reactive GRASP | da Silva Firmino, de Abreu Silva & Times (2019), J. Heuristics 25 | Restricted (p. 148); not yet declared | 2D / single-bay | Not standardised: scores its own plan; resets without `skip_auto_retrieve`. 3×5: 4/5. On 3-5-1 the plan omits the retrieval of container 1, which is on top at the start; `validate_plan` rejects the plan, and the reported f2 (715.2) leaves out that retrieval. 657 lines; no tests | Not started; full workflow planned for November | Missing retrievals of containers on top at the start (as Azari before `fe2676b`); the UI value of `max_iterations` never reaches the algorithm (always 500); `_best_solution` is always empty; duplicate final record |
| Ünlüyurt–Aydın (2012) Difference1 | Ünlüyurt & Aydın (2012), J. Adv. Transp. 46(4) | To check | 2D / single-bay | Not standardised: steps the problem, which records the automatic retrievals. 3×5: 5/5 complete and feasible. 208 lines; no tests | Not started; full workflow planned for November (Difference2) | Implements Difference1, not Difference2; the description quotes paper results (1.83 % average gap, 72.55 % optimal, Table VI) that have not been checked; duplicate final record |

## Multi-bay (3D)

Wei has not yet decided how multi-bay algorithms are to be handled in the unified 2D setting; validation is on hold until then.

| Algorithm (UI name) | Paper | Relocation rule (source) | Geometry | Implementation status | Validation status | Known issues |
|---|---|---|---|---|---|---|
| Forster–Bortfeldt (2012) Retrieval Tree Search | Forster & Bortfeldt (2012), Oper. Res. Proc. 2011, 257–262 | To check | 3D block, multiple bays / multi-bay | **Does not work on Caserta.** The plan is empty on 5/5 instances, yet the run is reported as completed with f2 = 0 (4 instances) or 42.0 (1 instance: one automatic retrieval). With one stack per bay, the per-bay tree search cannot relocate anything, and the empty replay is not checked. 308 lines, plus 706 imported from `algorithms/CRP_D/heuristic/forster_bortfeldt_2012`; no tests | Not started; on hold | Depends on another algorithm folder (layout and tree search from CRP_D, whose header cites Forster & Bortfeldt, C&OR 39 (2012)); duplicate final record |
| Kim (2016) Multi-Case Heuristic | Kim, Kim & Lee (2016), C&IE 101 | To check | 3D block, 1–10 bays / multi-bay | Not standardised: steps the problem. 3×5: 5/5 complete and feasible. 301 lines; no tests | Not started; on hold | The paper's pre-relocations of other critical stacks are not executed (code description); the `restricted` parameter has no effect but is shown in the UI; duplicate final record. The paper states no formal objective function (p. 352) |
| Lee–Lee (2010) Retrieval | Lee & Lee (2010), C&OR 37(6) | Not restricted (Example 1, p. 1145; [log](../validation/lee_lee_2010.md) §1) | 3D block, multiple bays / multi-bay | Own plan, reset with `skip_auto_retrieve`. 3×5: 5/5 complete and feasible; all 5 plans are restricted. 747 lines; no tests | Parked (solver-based implementation postponed). Log: specification, code mapping and deviations done; Check 1 (Fig. 3) fails on tie-breaking only | Shared lower bound counts only adjacent inversions (B1); a final record's `metric` and `metrics` can describe different plans |
| Lin–Lee–Lee (2015) Rule | Lin, Lee & Lee (2015), Transp. Res. C 52 | To check | 3D block; single- and multi-lift RMGC / multi-bay | Not standardised: steps the problem. 3×5: 5/5 complete and feasible, but **not reproducible**: three repeat runs gave different f2 on 3 of 5 instances. 353 lines; no tests | Not started; on hold | Ties broken with the unseeded global `random`; the Rule-2 pre-move is not executed as described (the problem relocates the target's blocker instead); defaults P_r = 30, P_b = 300 come from Shin et al. (TRC 2026), not from the paper; the paper's formal objective is the number of movements (p. 134); duplicate final record |

## Also offered for CRP-Time

Three algorithms in `algorithms/CRP_R` also list CRP-Time as a compatible problem. Their papers are outside the scope of the thesis. Their declared objective is relocations. On 3×5 #1–5 under f2, each gave 5/5 complete and feasible plans, all of them restricted.

| Algorithm (UI name) | Folder | Note |
|---|---|---|
| Caserta (2012) HEUR | `algorithms/CRP_R/heuristic/caserta_2012` | Open question for Wei: keep offering for CRP-Time? |
| Kim–Hong (2006) ENAR | `algorithms/CRP_R/heuristic/kim_hong_2006` | Dropped from the thesis (no paper available) |
| LA-N Look-Ahead | `algorithms/CRP_R/heuristic/lan_2013` | Open question for Wei: keep offering for CRP-Time? |

Voß & Schwarze (2019) has no algorithm in `algorithms/CRP_Time`: `exact/search` and `exact/solver` are empty. The exact optima in the comparison tables come from the A* used in the Jovanović validation.
