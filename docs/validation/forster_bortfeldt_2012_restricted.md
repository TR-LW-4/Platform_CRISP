# Forster & Bortfeldt (2012), restricted version — validation log

Paper: F. Forster, A. Bortfeldt, "A tree search procedure for the container relocation problem", *Computers & Operations Research* 39 (2012) 299–309 (`papers/2012_Forster.pdf`; printed page − 298 = PDF page), "C&OR" below. The method is unrestricted in the paper (p. 302); this log derives the restricted version.

Code: `algorithms/CRP_Time/crp_time_r/forster_bortfeldt_2012/`, derived from the validated unrestricted package `algorithms/CRP_Time/crp_time_u/forster_bortfeldt_2012/` (single bay: `docs/validation/forster_bortfeldt_2012.md`, decisions F1–F21; multi-bay: `docs/validation/forster_bortfeldt_2011_multibay.md`, decisions P1–P14; CRISP 0186485). Scripts and raw output: `docs/validation/data/forster_bortfeldt_2012_restricted/`.

Scope (supervisor's decision, task of 13 Nov 2026): two separate folders, each with its own search code; the restricted version is derived from the validated unrestricted one. The restriction is the one of CRISP's `CRP_R.step` (§2), not a reading of another paper. The unrestricted package stays byte-identical (§6).

## Summary

Status 13 Nov 2026: step 1a (copy), 1b (what `CRP_R.step` allows), 1c (decisions R1–R5, open points O1–O5), 1d (R1, R2 implemented), 1e (checks: all pass on Caserta 3-3, 3-4, 4-4; gap to the exact restricted optimum 0 on 3-3 and 3-4, mean 0.125 and max 2 on 4-4) and 1f (unrestricted package unchanged) done. Open: O1–O5 (§3.4).

## 1. Step 1a — copy (13 Nov 2026)

- fc1de7a: the tracked files of `crp_time_u/forster_bortfeldt_2012/` copied to `crp_time_r/forster_bortfeldt_2012/` without content changes, plus `crp_time_r/__init__.py` (`# Restricted CRP-Time algorithms.`, as `crp_time_u/__init__.py`), needed for unittest discovery and the registry's package walk. Git tree hash of both package folders at fc1de7a, and of `crp_time_u/forster_bortfeldt_2012` at 0186485: `5b4cf6b3c9a5f743819e5a41b27778d663c02417` (`git rev-parse <commit>:<path>`); the 9 blob hashes are the same file by file.
- ab08b04: path lines only — the usage line of `test_forster_bortfeldt.py` and of `test_multibay.py` names `crp_time_r` instead of `crp_time_u`. The package imports only relatively (`from .state import …`) and from `core`, so no import line needed a change. The 41 tests of the two copied test modules pass at ab08b04 (unchanged behaviour).

## 2. Step 1b — what `CRP_R.step` allows (13 Nov 2026)

Read from the code at ab08b04 (`problems/CRP_R.py`, `core/yard.py`, `core/benchmarks/caserta.py`); observed with `step1b_crp_r_probe.py` → `step1b_crp_r_probe.txt`, which calls `CRP_R.step` itself on Caserta layouts. CRP-Time inherits these rules unchanged: `CRP_Time.step` (`problems/CRP_Time.py:116-126`) calls `super().step(action)` and only adds metrics and rewards.

**Which container moves.** The action is a destination only (`action_space = Discrete(num_bays · num_rows)`, `CRP_R.py:73`); the source is not chosen by the algorithm.
- The target is the container with the current priority (`_get_target_container`, `CRP_R.py:389-393`: the first container whose priority equals `_current_target_priority`). The source is the stack that holds it (`CRP_R.py:213-218`, `227`).
- If the target is on top of its stack, there is nothing to relocate: it is retrieved and no relocation happens (`CRP_R.py:219-224`).
- Otherwise the top container of the target's stack is relocated (`blocker = src_stack.top`; `self.yard.relocate(src, dst)`, `CRP_R.py:234-235`; `Yard.relocate` moves the top of `src` onto `dst`, `core/yard.py:211-233`). This container is above the target, so it always blocks the target.
- Consequently no container of any other stack can be relocated, and the target itself is never relocated. `CRP_R.validate_plan` enforces the same on explicit plans: a relocation must come from the current target's stack (`CRP_R.py:496-501`) and must not move the target (`CRP_R.py:502-506`).

**To which stacks.**
- `dst = _action_to_stack(action)` (`CRP_R.py:210`): flat index → (bay, row) with `bay = a // num_rows + 1`, `row = a % num_rows + 1`, after `a % (num_bays · num_rows)` (`CRP_R.py:395-400`). Every stack of the yard is addressable (`Yard.stacks` holds all (b, r), b = 1…num_bays, r = 1…num_rows, `core/yard.py:180-183`).
- The relocation is executed unless the destination does not exist, is full (`len ≥ max_tiers`, `core/yard.py:48-49`) or equals the source (`CRP_R.py:231-232`); in those cases the step returns reward −0.5, moves nothing and still counts a step (`CRP_R.py:207`).
- No other condition: an empty destination is allowed, and the destination's contents (well or badly placed) do not matter.

**Within or across bays.** The destination may be any non-full stack of the yard other than the source, in the same bay or another bay; `step` has no bay condition. On the Caserta instances the platform maps stack i of the file to bay i + 1, row 1 (`core/benchmarks/caserta.py:198-203`; `num_bays = S`, `num_rows = 1`, `caserta.py:227-228`), so in CRISP coordinates every Caserta stack is its own bay and every relocation goes to another "bay".

**Retrievals.** Never an action: after `reset()` and after every executed or rejected relocation, all consecutive targets that are on top are retrieved (`_advance_auto_retrievals`, `CRP_R.py:174-194`, called at `CRP_R.py:113` and `241`). `reset(options={"skip_auto_retrieve": True})` keeps the opening retrievals out (`CRP_R.py:102-106`).

**Height.** `max_tiers` of the config; for Caserta `implied_tier_capacity` = n / S + 2 = T + 2 when the layout is rectangular (`caserta.py:154-161`, `229`).

Observed (`step1b_crp_r_probe.txt`, data3-3-1, file stacks bottom to top [3, 7, 1], [2, 6, 5], [8, 9, 4], max_tiers 5):
- reset retrieves container 1; the target is 2 in stack (2, 1), with 5 on top;
- `step(0)` and `step(2)` relocate 5 to (1, 1) and (3, 1) (reward −1.0, one relocation);
- `step(1)` (the source) moves nothing (reward −0.5, relocations 0, steps 1).
- With max_tiers = 3, `step(2)` onto the full stack (3, 1) moves nothing (reward −0.5).
- On data3-5-1, `reset()` retrieves container 1, which is on top at the start; `reset(skip_auto_retrieve)` does not.

Platform details seen on the way (not part of the restriction; for Wei, `docs/notes_for_wei.md`):
- `step` accepts action indices outside [0, num_bays · num_rows) through the modulo: `step(3)` and `step(-1)` on a 3-stack layout relocate to (1, 1) and (3, 1). `validate_actions` rejects them (`CRP_R.py:291-295`).
- The action mask (`_build_action_mask`, `CRP_R.py:414-422`) marks the source stack as available although `step` rejects it.

## 3. Step 1c — the restriction in Figs. 4–7 and the lower bound (13 Nov 2026)

Page numbers are the C&OR's printed pages. "Forced" means that it follows from `CRP_R.step` (§2) without a further choice; only forced points are implemented (§4). Every F-decision of the unrestricted log (F1–F21) carries over unchanged unless a decision below says otherwise.

### 3.1 Which relocations remain

Under `CRP_R.step` a relocation exists only when no remove is possible (the target is not on top). It moves the top item c of the stack d* that holds the target t, the item of the currently lowest group, to any non-full stack r ≠ d*.
- c lies above t and t < c, so c is badly placed (def. 1, p. 302): every restricted relocation is BG or BB, never GG or GB (Sec. 4.2, p. 302).
- c blocks an item of the currently lowest group, so it is FLG (p. 302; F16). Every restricted relocation is therefore FLG_BG or FLG_BB; FOG_BG, FOG_BB, GG and GB do not occur.
- Of the productive move types (FLG_BG, FOG_BG, FLG_BB, GG and removes, p. 302), FLG_BG, FLG_BB and removes remain.
- Removes: the paper's remove rule (p. 302, group order) equals `CRP_R`'s retrieval in priority order. Fig. 7 offers the removes before any relocation (p. 305) and the greedy applies all removes first (Fig. 4, p. 303), so with distinct groups the removes of the search fall where `CRP_R` retrieves automatically (`CRP_R.py:174-194`).

### 3.2 Decisions

| # | Point | Decision | Basis | Category | Status |
|---|---|---|---|---|---|
| R1 | Move set | A relocation is generated only from the stack that holds the currently lowest group, and only if that item is not on top; it moves the top item of that stack to any stack r ≠ d that is not full. Everything else in `Bay.relocations` (type flags, order of F8) unchanged. | `CRP_R.step` (§2; `CRP_R.py:213-235`); against p. 302 ("a relocation (d, r) is applicable … iff the donator is not empty and the receiver is not full") | (c) deviation from the paper's move set, required by CRP-Time's move rule | forced, implemented (§4) |
| R2 | Repeated groups | The restricted bay accepts only layouts with distinct groups (`ValueError` otherwise). With repeated groups the lowest group can lie in several stacks, and `CRP_R.step` does not say which of them may donate: it knows one target per priority (`CRP_R.py:389-393`); the Caserta reader requires priorities 1…n (`caserta.py:169-180`). The generalisation is left open (O1). | `CRP_R.step`; p. 302 (groups may repeat in the paper) | — (guard, no choice made) | forced, implemented (§4) |
| R3 | Greedy, Fig. 4 (p. 303–304) | Unchanged; the four priority groups act on the restricted move set of R1. (1) all removes; (2) BG to a non-empty stack with minimum top(r) − g (F10); (3) BG to an empty stack with the minimum moved group, now always the one item on top of d* (the first empty stack wins, F8); (4) FLG_BB with minimum g − min(r) (F9, reading B). The paper's argument that a move of (1)–(4) always exists (p. 303–304) still holds: if no BG relocation exists, the top of d* is an FLG item and any relocation of it is FLG_BB, provided some stack other than d* is not full (otherwise `CRP_R.step` cannot move either). | Fig. 4, p. 303–304; R1 | — | consequence of R1, no code change |
| R4 | Productive moves, Fig. 7 (p. 305–306) | Unchanged; acts on the restricted move set. Step 2 gives all FLG_BG relocations (there are no FOG_BG). Step 3 gives the first maxFlg_bb = 3 FLG_BB relocations sorted by g − min(r), plus GG relocations, of which there are none: Pm is at most 3 FLG_BB relocations. The GG key (F11, F12) is never evaluated. | Fig. 7, p. 305–306; R1 | — | consequence of R1, no code change |
| R5 | Lower bound (Sec. 5, p. 302–303) | Unchanged; it stays valid. Every restricted move sequence that empties L is a move sequence permitted for L in the sense of p. 302: each restricted relocation is a relocation (d, r) with d not empty and r not full, and each retrieval is a remove of the currently lowest group. The restricted solutions are therefore a subset of the solutions of the Proposition (p. 302), and a lower bound on the moves of all solutions bounds the restricted ones. So the optimality abort of Fig. 5 (F19), the pruning of Fig. 6 (F20) and the first sort key of Fig. 5 remain correct. The F20 argument (the bound drops by at most 1 per move; §2.3 of the unrestricted log) does not depend on which moves are allowed. | Proposition and proof, p. 302–303 | — | kept as validated |

### 3.3 What changes in the search without a code change

- **Compound moves, Fig. 6 (p. 305).** Unchanged. |Pm| is now at most max(S − 1, 3) for relocations (one BG relocation per receiver, or at most 3 FLG_BB) and 1 for a remove (distinct groups). cmStopValue, the product of the |Pm| along a compound move (p. 305), therefore grows more slowly than in the unrestricted search, so compound moves can become longer before they reach cmStopThreshold = 150. This is the printed rule applied to fewer moves, not a change.
- **Search, Fig. 5 (p. 304).** Unchanged: sort keys (|cm| + n′_m, length descending, clean supply descending), nSucc = 10, time limit. The clean supply (def. 2, p. 302) is a property of the layout and is computed as before.
- **The greedy and Caserta's min–max heuristic.** The paper calls its greedy "a generalisation of the min–max-heuristic described by Caserta et al. [18] that is not limited to BG moves from a stack that contains an item with lowest item group" (p. 304). Under R1 that generalisation falls away; the restricted greedy still differs from min–max as summarised on p. 301 in rule (2), which takes the top group of the target stack (F10), where min–max takes its minimum group. Observation only.

### 3.4 Open decisions (not implemented)

| # | Point | Options | Note |
|---|---|---|---|
| O1 | Repeated groups (R2) | (a) any stack that holds an item of the lowest group may donate its top; (b) one stack, chosen by a rule; (c) keep refusing such layouts. | Not reachable through `CRP_R` (one target per priority); relevant only if the package is used outside CRISP's problem classes. |
| O2 | The maxGg = 2 slots of Fig. 7 stay empty under R1 | (a) keep the parameter setting of p. 306 (current); (b) give the GG slots to FLG_BB (e.g. maxFlg_bb = 5). | (b) changes the paper's parameter setting. |
| O3 | A bound that uses the restriction | (a) keep n′_m (current, R5); (b) a restricted bound (e.g. counting the relocations a restricted solution needs beyond n_b). | n′_m is valid (R5); a tighter bound would change pruning and the optimality abort, so it would be a change to the method. |
| O4 | Multi-bay (`multibay.py`, copied unchanged) | (a) leave as copied; (b) adapt and check against `CRP_R.step` on a multi-bay yard. | Under R1 each bay search is restricted. `CRP_R.step` also allows relocations to other bays (§2); the Proceedings method relocates only inside a bay (multi-bay log, P9). Neither the merge of restricted bay solutions nor a multi-bay replay through `CRP_R` is checked in this step. |
| O5 | Unreachable code under R1 | (a) keep the GG branch of `productive_moves` with its key (F11, F12) (current); (b) remove it from the restricted copy. | Keeping them leaves the copy closest to the validated code; removing them is a clean-up, not forced. |

## 4. Step 1d — forced changes (13 Nov 2026)

Only R1 and R2 change code; R3–R5 hold without a change (§3.2).

- **f8ed8f3, R1.** `state.py`, `Bay.relocations`: one condition — a stack donates only if it holds the currently lowest group (`self.mins[d][-1] == low`) and that item is not on top (`sd[-1] != low`). The rest of the generator (receivers not full and ≠ d, the BG/BB/FLG flags, the order of F8) is unchanged. `greedy` (Fig. 4) and `productive_moves` (Fig. 7) call this generator and need no change.
  - Comments: the module docstrings of `state.py` and `search.py` say that the relocations are restricted as in `CRP_R.step` (R1) and point to this log. `search.py` called them "unrestricted (p. 302)". Both had "F1-F19", now "F1-F21" (F20, F21 were added after the docstrings).
  - Tests (`test_forster_bortfeldt.py`), changed where an expectation needed an unrestricted move; the new expectations are derived by hand in the test comments:
    - move types on Fig. 3 and small layouts: only the target stack's top moves, always FLG_BB or FLG_BG;
    - no relocation while the lowest group is on top;
    - Fig. 7 without the FOG_BG and GG moves;
    - no GG under the restriction (replaces the F12 GG-key test).
    - The Fig. 3 greedy test checks validity, the restriction and the bound instead of the unrestricted count of 13 relocations.
    - The random-move test stops when no move is left (every stack but the target's full).
    - New: `check_restricted`, a plain-list check of R1 independent of `state.Bay`. It is applied to the greedy and search plans of all Caserta 3x3 and 3x4 runs of the unit tests (at H = n and H = T + 2), and its own test shows that it rejects a relocation from another stack.
- **ba40168, R2.** `Bay.from_stacks` raises `ValueError` on repeated groups. The random-layout test draws distinct groups; a test for the refusal.
- 44 tests of the two restricted test modules pass at ba40168. `multibay.py` is unchanged (O4); its tests pass unchanged on the restricted search.

## 5. Step 1e — checks (13 Nov 2026)

Script `step1e_checks.py` → `step1e_checks.txt`, per-instance records with plans in `step1e_checks.jsonl`. Settings:
- Caserta 3-3, 3-4 and 4-4, all 40 instances each, no time limit, sequential.
- CRISP 6a10012 (algorithm code as at ba40168), rl Python 3.11.16.
- H = `max_tiers` of the platform's own config (`problem_config_for_caserta_dat`) = T + 2, asserted per instance; equal to F6 of the unrestricted log. The unlimited-height variant H = n (F7) has no CRP-Time counterpart and is not run.

**Checks, all passed on all 120 instances:**
1. **Replay through `CRP_R.step` itself.** Before every relocation of the plan, the CRP_R yard equals the plan's layout. `step(r)` executes exactly one relocation (reward −1.0) and moves the plan's item. At the end, CRP_R's move history equals the plan move by move, including its automatic retrievals at the plan's removes; CRP_R is done and the yard is empty. In addition, `CRP_R.validate_plan` on the explicit plan reports it feasible, with the same number of relocations and no errors.
2. **`plan.check_plan` (plain lists):** only top items moved, removes in group order, height ≤ H, bay empty.
3. **Lower bound:** moves ≥ the lower bound of the initial layout (`Bay.lower_bound` = `layout.lower_bound`).
4. **Exact restricted optimum:** moves ≥ the exact restricted optimum.
   - The optimum is computed with the A* of the Jovanović validation (`jovanovic_2019_checks2/jov_astar.py`; `jovanovic_2019.md` §4, "OPT column of Table 5"): restricted BRP, retrieval as soon as the target is on top.
   - Unit cost per move: its f2 model with t_s = 0 (the module constant `TS`, set by the script) and t_pp = 1, so the cost of a plan is its number of moves and the heuristic is n + n_b. The solver code is unchanged.
   - Because the solver was re-parameterised, its optimum is cross-checked against an independent breadth-first search over restricted layouts (`bfs_restricted` in the script): equal on all 120 instances.

**Gap to the exact restricted optimum** (relocations; LB rel = lower bound − n; greedy = the restricted greedy of Fig. 4; "at opt" = gap 0; "bound hit" = the search stopped at the lower bound, F19; max s = largest wall clock of a search):

| class | LB rel | optimum | search | greedy | gap mean | gap max | at opt | bound hit | max s |
|---|---|---|---|---|---|---|---|---|---|
| 3-3 | 4.000 | 5.000 | 5.000 | 5.175 | 0.000 | 0 | 40/40 | 11/40 | 0.0003 |
| 3-4 | 5.075 | 6.175 | 6.175 | 6.350 | 0.000 | 0 | 40/40 | 11/40 | 0.0019 |
| 4-4 | 7.700 | 10.200 | 10.325 | 11.000 | 0.125 | 2 | 36/40 | 2/40 | 0.0058 |

- Gap distribution (instances per gap): 3-3 all 40 at 0; 3-4 all 40 at 0; 4-4 36 at 0, 3 at 1, 1 at 2.

**Descriptive only, not a check: restricted against unrestricted.** The unrestricted search (`crp_time_u`, same H, no time limit) on the same instances, compared in relocations. Its relocations equal those of the earlier unrestricted full run (`forster_bortfeldt_2012/step3_full.jsonl`, variant t2) on 40 of 40 instances per class. "Below restricted opt" counts the instances where the unrestricted search needs fewer relocations than the exact restricted optimum, which only an unrestricted solution can do.

| class | restricted better | equal | worse | mean restricted | mean unrestricted | unrestricted below restricted opt |
|---|---|---|---|---|---|---|
| 3-3 | 1 | 38 | 1 | 5.000 | 5.025 | 1/40 |
| 3-4 | 2 | 33 | 5 | 6.175 | 6.100 | 5/40 |
| 4-4 | 2 | 27 | 11 | 10.325 | 10.000 | 9/40 |

- The restricted search is better than the unrestricted one on a few instances (1, 2, 2 per class). Both are heuristics; the restricted one searches a smaller move set and reaches the restricted optimum on 3-3 and 3-4.

## 6. Step 1f — the unrestricted package is unchanged (13 Nov 2026)

Script `step1f_unchanged.py` → `step1f_unchanged.txt`, run at CRISP 6300e44; this log's §6 commit touches only this log.
- `git diff --stat 0186485 HEAD -- algorithms/CRP_Time/crp_time_u` is empty, as are the diff against the working tree and `git status` on that folder.
- Git tree hash of `crp_time_u/forster_bortfeldt_2012`: `5b4cf6b3c9a5f743819e5a41b27778d663c02417` at 0186485 and at HEAD. Tree hash of `crp_time_u`: `d68376cd17141488c2fbd903c97fbb3c648732d7` at both.
- Every file changed since 0186485 is under `crp_time_r/` or is this log.
- The full unit suite passes at 6a10012: 206 tests, the 162 of before plus the 44 of the restricted package.
- The registry still loads 74 algorithms with no module under `algorithms/` failing to import (`step1f_registry.txt`). The restricted package defines no algorithm class, so it is not registered (not wired into CRP-Time, the UI or the CLI).

## 7. Open decisions

O1–O5 (§3.4); none is implemented. Outside this step: wiring the restricted search into CRP-Time (a `BaseAlgorithm` class, the f2/f2vert objective, `relocation_rule = "restricted"`).
