# Lee & Lee (2010) — validation log

Paper: Y. Lee, Y.-J. Lee, "A heuristic for retrieving containers from a yard", *Computers & Operations Research* 37 (2010) 1139–1147. Page numbers below are journal pages.

Code: `algorithms/CRP_Time/heuristic/lee_lee/`

## 1. Specification from the paper

### Problem setting
- Yard with multiple bays; stacks are numbered consecutively across bays (bay 1: stacks 1..R, bay 2: R+1..2R, ...). Stack 0 is the truck. One RMGC, no crane interference. (p. 1140–1141)
- Containers are numbered 1..N; smaller numbers are retrieved earlier. All containers have the same size. (p. 1140)
- A movement is a triplet (c:a,b): container c from stack a to stack b. (p. 1141)
- Two conflict types: *stacking conflict* (container is not on top of its source stack) and *over-height conflict* (destination exceeds the maximum height H). A feasible sequence retrieves all containers in order without conflicts. (p. 1141)
- Restricted or unrestricted: not stated explicitly. The Example 1 solution (p. 1145) moves container 6 (6:4,2) before container 2 is retrieved, while 6 does not block 2. The method therefore produces sequences that are not restricted to relocating blockers of the current target.

### Objective
- Overall goal: weighted sum of the number of movements and the total crane working time. (p. 1140–1141)
- Phase 3 objective (6): number of movements + W · t*, where t* is the time the last container is retrieved. W = 1.0 in all experiments. (p. 1143–1144)

### Time model (p. 1143–1145)
- Gantry 3.5 s per bay, trolley 1.2 s per container width, gantry acceleration/deceleration 40 s combined, spreader pick-up + place-down 30 s combined.
- Working time of movement m after movement n, w(n,m), includes repositioning the crane from the end position of n to the source stack of m. w(0,m) starts from the crane's initial stowed position.
- No tier-dependent vertical term.

### Container types (p. 1141–1142)
- Type B: blocks another container in the initial layout. Type NB: does not.
- NB containers are moved once (retrieval) and never get alternate paths.

### Lower bound (p. 1140)
- Sum over all stacks of the minimum number of movements per stack.

### Phase 1 — initial phase (p. 1141)
- Retrieve containers in order. If the target is on top, retrieve it. Otherwise move each blocking container to the *nearest available stack*.
- "Nearest" and "available" are not defined further. Tie-breaking is not specified.
- Worked example (Fig. 3, three stacks, bottom→top: stack 1 = 3, 5, 1; stack 2 = 4, 2; stack 3 = empty). Expected sequence: (1:1,0), (2:2,0), (5:1,2), (3:1,0), (5:2,3), (4:2,0), (5:3,0). In the move (5:2,3), stacks 1 and 3 are both empty and at distance 1; the paper picks stack 3.

### Phase 2 — movement reduction (p. 1141–1143)
- For every type B container, generate k alternate two-movement paths (one relocation, then retrieval). The value of k is not specified (the worked example uses k = 1). How the destination stack of an alternate path is chosen is not specified.
- Randomly assemble the alternate paths with the current sequence into a super-sequence.
- Build the augmented yard (each alternate path as an extra container above the original) and execute the super-sequence. Record stacking conflicts as pairwise conflicts between paths and over-height conflicts as conflict sets; continue executing as if the conflict could be tolerated.
- Solve a binary integer program (1)–(5): minimise the total number of movements, choose exactly one path per container, exclude conflicting pairs, at most H paths per over-height conflict set. Solved with CPLEX.
- Terminate when the number of movements reaches the lower bound, or after 50 consecutive iterations without improvement.

### Phase 3 — time reduction (p. 1143–1144)
- Generate alternate two-movement paths randomly for *some* type B containers (number not specified).
- Solve a MIP: constraints (2)–(5) plus timing constraints (7)–(12), objective (6).
- The number of movements cannot increase in this phase. (p. 1144)
- Terminate after 350,000 consecutive iterations without improvement or after 21,600 s; for smaller instances the criteria may be adjusted.

### Output (p. 1141)
- "The best sequence discovered in the process is used as the final output."

### Validation material in the paper
- Fig. 3: exact phase-1 sequence (see above).
- Example 1 (Fig. 6, p. 1144–1146): 2 bays × 3 stacks, H = 6, 20 containers. Layout bottom→top: stack 1 = 10, 4, 18; stack 2 = 9, 17, 8, 20; stack 3 = 5, 19, 2; stack 4 = 16, 14, 7, 13, 3, 6; stack 5 = 1, 11; stack 6 = 15, 12. Lower bound 27 movements, reached in phase 2 after 26 iterations. The final 27-movement sequence is given on p. 1145.
- Reported quality (p. 1146): about half of the instances within 110% of the lower bound on movements, about 65% within 120%. Worst cases in Table 1 (p. 1145) are well above that (e.g. R021608_0190_001: lower bound 305, 508 movements after phase 2).
- The instance set is no longer available online.

## 2. Comparison with the code

| Step | Paper (p.) | Code (file:line) | Verdict | Category |
|---|---|---|---|---|
| Stack numbering, stack 0 = truck | 1140–1141 | `core/objectives.py:202-205`; Caserta loader pins one stack per bay and `num_rows = 1` (`core/benchmarks/caserta.py:198-200`, `:228`) | equal | unified 2D setting |
| Movement triplet (c:a,b) | 1141 | `core/plan.py:34-51`; `to_pos=None` denotes the truck | equal | — |
| Conflict types (stacking, over-height) | 1141 | `core/plan.py:28-31`, detected in `core/plan.py:139-184` | equal | — |
| Tolerate a conflict and keep executing | 1142 | All three conflict branches leave the yard unchanged (`core/plan.py:158-159`, `:174`, `:175-179`); callers read only `SimResult.feasible` (`phase2.py:155`, `phase3.py:88`) | deviation | differs from paper |
| Restricted vs unrestricted | 1145 (Example 1 moves 6:4,2) | `phase1.py:76-87` relocates only containers above the current target; phases 2–3 rewrite existing paths and never add a new mover (`core/plan.py:88-94`) | deviation | differs from paper |
| Phase 1: retrieve in order, else move blockers | 1141 | `phase1.py:45-97` | equal | — |
| Phase 1: definition of "nearest" | 1141 (undefined) | `phase1.py:36` Manhattan distance over (bay, row); one bay step counts the same as one row step | not specified in paper | choice left open by paper |
| Phase 1: definition of "available" | 1141 (undefined) | `phase1.py:34` any non-full stack other than the source | not specified in paper | choice left open by paper |
| Phase 1: tie-breaking | 1141 (undefined); Fig. 3 picks stack 3 over stack 1 | `phase1.py:37-38` lexicographically smallest (bay, row) | not specified in paper | choice left open by paper |
| Phase 2: k alternate two-movement paths per type B container | 1141–1143 | `phase2.py:132-150` picks one random type B container per iteration and accepts the first improving waypoint | deviation | differs from paper |
| Phase 2: super-sequence, augmented yard, conflict sets | 1142–1143 | absent | deviation | differs from paper |
| Phase 2: BIP (1)–(5) solved with CPLEX | 1143 | absent; replaced by local search (`phase2.py:99-165`) | deviation | differs from paper |
| Phase 2: terminate at LB or 50 non-improving iterations | 1143 | LB check `phase2.py:125-126`; `max_no_improve` defaults to **200** (`algorithm.py:83`, `:218-222`) | deviation | choice left open by paper |
| Type B definition | 1141–1142 (blocks in the *initial layout*) | `core/plan.py:88-94` containers with ≥1 non-retrieval move in the *current plan* | deviation | differs from paper |
| Lower bound | 1140 | `core/objectives.py:416-426` → `core/yard.py:128-134` counts only **adjacent** priority inversions | deviation | differs from paper (bug) |
| Phase 3: alternate paths for some type B containers | 1143–1144 | `phase3.py:52-78` random container, random waypoint | deviation | differs from paper |
| Phase 3: MIP (2)–(12) with objective (6) | 1143–1144 | `phase3.py:91-98` accepts a candidate iff the scorer strictly improves; scorer is `evaluate_plan_objectives(...)["objective_value"]` (`algorithm.py:157-162`), i.e. crane time only, with no movement-count term | deviation | unified 2D setting + differs from paper |
| Phase 3: movement count may not increase | 1144 | `phase3.py:84-85` | equal | — |
| Phase 3: terminate after 350,000 non-improving iterations | 1144 | `max_no_improve` defaults to **500** (`algorithm.py:84`, `:223-227`) | deviation | choice left open by paper |
| Time model (gantry/trolley/accel/spreader, w(n,m) with repositioning) | 1143–1145 | f2 / f2vert (`core/objectives.py:261-300`). The Lee–Lee kinematics exist (`core/objectives.py:40-131`) but serve only as the `score_fn` fallback (`phase3.py:41`), which the algorithm never reaches | deviation | unified 2D setting |
| W = 1.0, weighted movement + time objective | 1140–1141, 1144 | no weighted objective; `problems/CRP_Time.py:80-94` forces `mode="crane_time"` | deviation | unified 2D setting |
| Output = best sequence found during the process | 1141 | Since `df3c265`: every phase's plan is a candidate (`algorithm.py:113`, `:136`, `:168`); the best under the selected crane-time objective is kept (`algorithm.py:193-196`, `:236-238`) and reported in the final record (`algorithm.py:182-191`). Before: only the phase-3 plan was compared (B2, §3.4) | equal (criterion adapted, see §3.1; tie-breaking see §3.2) | unified 2D setting + choice left open by paper |

## 3. Deviations and decisions

### 3.1 Unified 2D setting (required by the project)

**Time model replaced.** The search minimises Voß & Schwarze f2 or f2vert instead of the paper's RMGC working time. The Lee–Lee kinematics are still implemented and still reported as the secondary metric `crane_time_rmgc`, but they never drive the search: `algorithm.py:157-162` always supplies a `score_fn` built from the selected `ObjectiveSpec`, so the `compute_crane_time` fallback at `phase3.py:41` is dead code in this path. This was checked explicitly because a gantry-based model would be actively misleading in the 2D setting, where the Caserta loader gives every stack its own bay: each stack pair would then be charged 3.5 s of gantry travel plus 40 s of acceleration. That does not happen.

One consequence deserves a note in Methods. The paper's phase-3 objective is t\*, the time at which the *last* container is retrieved, and its w(n,m) includes repositioning the crane from the previous movement's end position. f2 and f2vert are additive per movement and contain no inter-movement repositioning term, so the platform's total is a sum of independent movement costs, not a makespan. The two objectives therefore rank sequences differently even before any implementation difference.

**Objective (6) loses its movement term.** The paper minimises movements + W·t\* with W = 1.0. CRP-Time is single-objective by design, so phase 3 minimises crane time alone. Combined with the constraint that phase 3 may not increase the movement count (`phase3.py:84-85`), the practical difference is smaller than it looks, but it is a genuine change of objective.

**"Best sequence" is judged on crane time alone.** The paper's output rule (p. 1141) keeps the best sequence found in the process, where "best" is measured by its overall objective, movements + W·t\*. Since `df3c265` the implementation keeps the plan with the lowest `objective_value`, i.e. f2 or f2vert as selected (`algorithm.py:236-238`). Movement and relocation counts play no part in the selection. This matters more than in phase 3: phase 1 typically has more movements than phases 2 and 3, and it can still be selected as the output if its crane time is lowest (§4, Check 5).

**"Nearest" is a plain Manhattan distance** over (bay, row) (`phase1.py:36`), which treats a bay step and a row step as equally expensive. With the Caserta geometry (`num_rows = 1`) this reduces to the difference of stack indices and matches the paper's stack numbering. In a genuine multi-bay yard it would not, because the paper's crane pays 3.5 s + 40 s per bay against 1.2 s per row. Harmless here; must be revisited if multi-bay instances are ever used.

### 3.2 Choices left open by the paper

**Phase 1 tie-breaking** (`phase1.py:37-38`): the nearest stack with the lexicographically smallest (bay, row) wins. The paper does not specify a rule, but its own Fig. 3 breaks the tie the other way (see §4), so this choice is visible in the worked example.

**"Available" stack** (`phase1.py:34`): any non-full stack other than the source.

**Ties when selecting the best sequence** (`algorithm.py:236-238`): when two phases produce plans with equal crane time, the plan from the earlier phase is kept (`min()` returns the first minimum). The paper does not specify a tie rule. The alternative, preferring the later phase, would in practice favour the plan with fewer movements, since phase 2 reduces movements and phase 3 may not increase them. That would amount to silently selecting on relocations, which the project rules out: relocations are always reported but never optimised, and that includes their use as a tie-breaker. Keeping the earlier phase keeps the selection purely on the selected crane-time objective.

**k, the number of alternate paths** (not specified in the paper): the implementation does not generate a set of k paths at all; it samples one container and one waypoint per iteration.

**Termination parameters**: the paper uses 50 consecutive non-improving iterations for phase 2 and 350,000 for phase 3. The defaults here are 200 and 500 (`algorithm.py:83-84`), exposed as `max_no_improve_p2` / `max_no_improve_p3`. Since neither phase solves a mathematical program, an iteration means something different in the two settings and the numbers are not comparable.

### 3.3 Differs from the paper

**Phases 2 and 3 are local search, not mathematical programming.** The paper builds a super-sequence over alternate paths, executes it on an augmented yard to harvest pairwise and set-wise conflicts, and hands the result to a BIP (phase 2) or a MIP (phase 3) solved with CPLEX. The implementation replaces both with a randomised single-container neighbourhood search that accepts the first strictly improving feasible candidate. This is stated in the class description (`algorithm.py:52-62`) and is the dominant reason the paper's Table 1/3 numbers are out of reach. It also makes the conflict machinery in `core/plan.py` vestigial: `SimResult.conflicts` is populated but no caller reads it.

**The implementation is restricted; the paper is not.** Phase 1 only ever relocates containers sitting above the current target, and phases 2 and 3 only rewrite the path of a container that is already being moved, so no container that never blocked anything is ever touched. The paper's own Example 1 solution contains (6:4,2), a move of a container that does not block the current target. The implementation therefore searches a strictly smaller solution space than the paper's method.

This is consistent with the rest of the platform — CRP-Time inherits CRP-R dynamics, and `CRP_R.validate_plan` (`problems/CRP_R.py:496-506`) rejects any relocation that does not come from the current target's stack, so an unrestricted plan would fail the platform's own validator. It is also consistent with Voß & Schwarze (2019), who use the restricted variant. **Open question for Wei:** should all CRP-Time baselines be forced into the restricted variant for comparability, accepting that Lee–Lee is then not reproduced faithfully, or should the platform admit unrestricted plans for the methods whose papers assume them?

**Type B is defined on the current plan, not the initial layout** (`core/plan.py:88-94`). For the phase-1 output the two definitions almost coincide, since a container is relocated exactly when it blocks something. After phase 2 rewrites paths, the set is recomputed on the rewritten plan.

### 3.4 Suspected bugs

**B1 — the lower bound counts only adjacent inversions.** `Stack.num_bad_overlaps` (`core/yard.py:128-134`) documents "(lower, upper) pairs where lower.priority < upper.priority" but compares only positions *i* and *i+1*. `lower_bound_relocations` (`core/objectives.py:416-426`) calls itself the standard BRP lower bound and cites Lee & Lee. Neither the docstring's definition (all inverted pairs, which would over-count) nor the implementation (adjacent pairs, which under-counts) is the standard bound, which counts each container having *any* smaller-numbered container beneath it. Measurements are in §4; the correct definition reproduces the paper's lower bound of 27 on Example 1, the implemented one under-counts by up to 40 % on Caserta instances.

Effects: the reported `lower_bound` and `lb_ratio` are wrong for every CRP-R and CRP-Time run, not just Lee–Lee, and `lb_ratio` is inflated because its denominator is too small. Phase 2's early exit (`phase2.py:125-126`) also fires too rarely, which costs time but cannot produce a wrong answer. **This is platform-wide and sits in shared code; it is Wei's call, not a CRP-Time-local fix.**

**B2 — the best sequence across phases is not kept. Fixed in `df3c265`.** The paper takes the best sequence found anywhere in the process as the output (p. 1141). Before the fix, the driver compared only the phase-3 plan, and it compared it against `self._best_metric` (`primary <= self._best_metric`). `BaseAlgorithm._push` (`core/base_algorithm.py:154-155`) had already lowered that value to the minimum of every metric pushed so far, including the phase-1 and phase-2 values. The phase-1 and phase-2 plans themselves were pushed for the progress plot and then dropped. Because phase 2 minimises movement count rather than time, it can hand phase 3 a worse plan than phase 1 produced, and phase 3 does not always recover.

Whenever phase 1 or phase 2 beat phase 3, this had three effects at once:
1. `_best_plan` and `_best_solution` were never assigned, so `get_best_plan()` and `get_best_solution()` returned `None`.
2. The final record's `metric` (and `best_metric`) held the phase-1 value.
3. The final record's `metrics` dict, which `save_run` writes to disk, held the phase-3 values. So `metric` and `metrics["objective_value"]` described two different plans, and the saved result was not the best one found.

Observed on `data6-6-1.dat` under f2vert: phase 1 = 8408.82, phase 2 = 8500.92, phase 3 = 8491.32. Before the fix, the final record showed `metric = 8408.82` next to `objective_value = 8491.32`, and both accessors returned `None`. The greedy first phase was better than the saved answer by 1 %. Before/after output is in §4, Check 5.

The fix: every phase's plan is a candidate. After each phase the best candidate so far is stored in `_best_plan` / `_best_solution`, so a run stopped early still returns the best plan up to that point. The final record carries that plan's metrics and its phase in `extra["best_phase"]`. `extra` is pushed but not saved by `save_run`; the phase can still be derived from the saved `history`. Phase 3 still starts from the phase-2 plan, as in the paper; the change affects only which plan is returned. The comparison no longer depends on `_best_metric`.

The fix also dropped the mean over seeds of the phase-3 metrics that the final record used to carry. With `n_seeds = 1` (`algorithm.py:85`) that mean equalled the phase-3 metrics, and it would not describe a single plan if multiple seeds were re-enabled.

### 3.5 Platform observations (for Wei)

**`metric` and `metrics` in a progress record can describe different plans.** `BaseAlgorithm._push` (`core/base_algorithm.py:154-160`) sets `best_metric` to the minimum over every metric pushed so far. It stores the `metric` argument as passed, but algorithms often pass `self._best_metric` as `metric` in their final record (e.g. Lee–Lee before `df3c265`), and the `metrics` dict is whatever the caller passes, usually the last plan's. Any algorithm that pushes intermediate phases or iterations and then reports its last plan, rather than its best, therefore produces a final record whose `metric` or `best_metric` belongs to a different plan than `metrics`. `cmd_layout_run` saves `final.metrics` (`main.py:345`), so the saved result follows the last plan, while the printed `metric` follows the best. This is an observation about shared code, not something fixed here; it affects every algorithm that follows this pattern, not only Lee–Lee. The Lee–Lee fix avoids it by passing the best plan's own metrics and objective value explicitly.

## 4. Behavioural checks

Reproduce with `scratchpad/leelee_validate.py` (not committed).

### Check 1 — Fig. 3: phase 1 reproduces the exact sequence — **fails, on tie-breaking only**

```
paper : (1:1,0) (2:2,0) (5:1,2) (3:1,0) (5:2,3) (4:2,0) (5:3,0)
code  : (1:1,0) (2:2,0) (5:1,2) (3:1,0) (5:2,1) (4:2,0) (5:1,0)
```

The sequences agree on every decision except the destination of container 5's second relocation, where stacks 1 and 3 are both empty and both at distance 1. The paper picks stack 3, the code picks stack 1 (`phase1.py:37-38`, lexicographically smallest). The last movement differs only as a consequence. Movement count is identical (7), so this is a tie-breaking convention, not a quality difference; but Fig. 3 cannot be reproduced exactly while the rule stands. Preferring the highest stack index on ties would reproduce the figure — on this single data point, which is not enough to infer the paper's rule.

### Check 2 — Example 1: phases 1–2 reach 27 movements — **fails**

| | movements | relocations |
|---|---|---|
| paper, lower bound, reached in phase 2 after 26 iterations | 27 | 7 |
| code, phase 1 | 40 | 20 |
| code, phase 2 (seed 0, `max_no_improve = 2000`) | 35 | 15 |
| code, phase 2 (best over seeds 0–9) | 30 | 10 |

The implementation does not reach the lower bound on the paper's own example, and phase 1 alone is nearly three times the optimal relocation count. The gap is consistent with §3.3: replacing the BIP by first-improvement local search loses the ability to reconsider several containers' paths jointly, which is exactly what the paper's formulation does.

This also puts the earlier `data5-5-1` result in perspective: 43 relocations against a true lower bound of 15 is a ratio of 2.9, in the same range as the 30/7 ≈ 2.9 relocation ratio measured here on Example 1. The implementation behaves consistently; it is simply much weaker than the paper's method.

### Check 3 — lower bound definition

| instance | code (adjacent inversions) | standard BRP bound | paper |
|---|---|---|---|
| Example 1 | 7 | 7 | 7 |
| `data3-3-1.dat` | 3 | 4 | — |
| `data5-5-1.dat` | 9 | 15 | — |
| `data6-6-1.dat` | 12 | 17 | — |
| `[1, 3, 2]` bottom→top | 1 | 2 | — |

Example 1 is a coincidence: adjacent-only counting happens to give the right answer there, which is why the defect is easy to miss. The minimal counterexample is a single stack `[1, 3, 2]`, where both 3 and 2 must be relocated but only one adjacent inversion exists.

### Check 4 — Caserta results compared with Voß & Schwarze (2019) optima — **not done**

Requires their published optima; not yet available.

### Check 5 — B2: best plan across phases, before and after `df3c265`

Reproduce with `scratchpad/b2_check.py` (not committed). It runs `train()` in-process on `data6-6-1.dat` (CRP-Time, `time_model = f2_vertical`, default parameters, seed 0), prints every progress record, and calls `get_best_plan()` / `get_best_solution()` afterwards.

Before (`f69fcdd`):
```
step=1 metric=8408.82 best_metric=8408.82 objective_value=8408.82 relocations=66 extra={'phase': 1, 'seed': 0}
step=2 metric=8500.92 best_metric=8408.82 objective_value=8500.92 relocations=65 extra={'phase': 2, 'seed': 0}
step=3 metric=8491.32 best_metric=8408.82 objective_value=8491.32 relocations=65 extra={'phase': 3, 'seed': 0}
step=3 metric=8408.82 best_metric=8408.82 objective_value=8491.32 relocations=65 extra={}
get_best_plan() is None: True
get_best_solution() is None: True
```

After (`df3c265`):
```
step=1 metric=8408.82 best_metric=8408.82 objective_value=8408.82 relocations=66 extra={'phase': 1, 'seed': 0}
step=2 metric=8500.92 best_metric=8408.82 objective_value=8500.92 relocations=65 extra={'phase': 2, 'seed': 0}
step=3 metric=8491.32 best_metric=8408.82 objective_value=8491.32 relocations=65 extra={'phase': 3, 'seed': 0}
step=3 metric=8408.82 best_metric=8408.82 objective_value=8408.82 relocations=66 extra={'best_phase': 1}
get_best_plan() is None: False
get_best_solution() is None: False
best plan re-evaluated: objective_value=8408.82 relocations=66 moves=102 len(solution)=66
```

The per-phase records are identical before and after, so the search itself is unchanged. After the fix, the final record's `metric` and `objective_value` agree, and the returned plan re-evaluates to the same value. The selected plan has one relocation more than the phase-3 plan (66 against 65). This is the expected consequence of selecting on crane time alone (§3.1).

## 5. Suggested next steps

1. ~~Fix B2 (keep the best plan across phases).~~ Done in `df3c265`; see §3.4 and §4, Check 5.
2. Raise B1 (lower bound) with Wei. It is shared code and affects every problem family that reports `lb_ratio`. Raise the `_push` observation from §3.5 at the same time.
3. Decide the restricted/unrestricted question with Wei before any cross-method comparison table is produced.
4. Record the tie-breaking rule from Check 1 in Methods as an explicit, documented choice.
