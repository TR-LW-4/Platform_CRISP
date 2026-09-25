# Jovanović, Tuba & Voß (2019) — validation log

Paper: R. Jovanovic, M. Tuba, S. Voß, "An efficient ant colony optimization algorithm for the blocks relocation problem", *European Journal of Operational Research* 274 (2019) 78–90. Page numbers below are journal pages.

Code: `algorithms/CRP_Time/heuristic/jovanovic_2019_aco/`

## 1. Specification from the paper

### Problem setting (p. 79)
- Two-dimensional bay: W stacks, maximum height H. N containers with unique due dates 1..N. Initial layout known.
- Only top containers can be accessed. Only the container with the lowest due date can be retrieved. A relocated container goes on top of another container or on the ground ("tier 0").
- Two variants: uBRP (unrestricted) and rBRP (restricted). rBRP adds constraint (RES): only containers above the current target may be relocated.
- **The crane-time variant is defined for the rBRP only** (Sec. 6, p. 85; Sec. 7.4, p. 88). This is the variant relevant for CRP-Time.

### Greedy base algorithm for the rBRP (p. 79–81)
- Algorithm 1 (p. 80): retrieve containers in due-date order; while the target is not on top, relocate the obstructing top container to the stack chosen by heuristic h; then retrieve.
- Candidate stacks R_c: all non-full stacks except the source stack (eq. 1, p. 80).
- dd(S): minimum due date in stack S; N + 1 for an empty stack (eq. 2).
- dif(c, d) = d − c if d > c, else 2N + 1 − d (eq. 3). Values 1..2N; dif ≤ N means c is well-located above d.
- MinMax(c) = argmin over S ∈ R_c of dif(c, dd(S)) (eq. 5, equivalently eq. 7). Prefers a stack where c is well-located with the smallest minimum due date; otherwise the stack with the largest minimum due date.
- Tie-breaking in the argmin: not specified.
- rBRP candidate list: T = {top(s(t))} × R_top(s(t)) (eq. 11, p. 81).
- The uBRP greedy variants (Gre-N, Gre-C; eqs. 8–24, p. 81–82) are not needed for CRP-Time.

### ACO (Sec. 5, p. 82–85)
- Pheromone matrix τ_cdnt (p. 83):
  - t: current target container; c: container being relocated (1..N)
  - d = dd*(S): dd(S) for a non-empty stack, N + i(S) for an empty stack with index i(S) (1..N+W)
  - n: number of times c has already been moved (0..MaxMoves)
- Heuristic value f(c, d) = 1 / (1 + dif_e(c, d)) (eq. 26); for the rBRP dif_e = dif.
- g(α) = f(α) · τ_c,d,m_c,t (eq. 30).
- Transition rule (eq. 31–32, p. 83): with probability q0, take argmax g(α); otherwise sample α with probability g(α) / Σ g(δ).
- Solution quality for the relocation objective: val(S) = 1 / (|S| − LB + 1) (eq. 33, p. 83). LB for the rBRP is the lower bound of Zhu et al. (2012) (p. 84); its definition is not given in the paper.
- A solution is stored as a list of 4-tuples (c, dd*(S), m_c, t) (p. 84).
- Global update (ACS, only the best-so-far solution): τ = (1 − p)·τ + p·val(S_best) for all tuples in S_best (eq. 34–35, p. 84).
- Local update after each ant, also for an aborted partial solution: τ = ϕ·τ for all tuples in S_i (eq. 36, p. 84).
- Initial pheromone τ0 = (1/W) · val(S_g), with S_g the greedy solution (eq. 37, p. 84).
- Minimum pheromone τ_min = (1/W²) · val(S_best) (eq. 38, p. 84). No maximum.
- Algorithm 2 (p. 84): per iteration, n ants each build a solution; an ant aborts as soon as |S| + LB(current bay) ≥ |S_best|; after MaxConst iterations without improvement the pheromone matrix is reinitialised; then the global update.

### Crane-time extension (Sec. 6, p. 85)
- **Only the objective changes; the heuristic stays MinMax.** The crane-time objective replaces the relocation count in the solution quality (pheromone initialisation and global update) and in the early-abort check.
- Of (fixed pick-up/place-down), per operation (eq. 40): retrieval 2·ts·s(c) + tpp; relocation 2·ts·|s(c) − S| + tpp. Retrieved containers go to stack 0.
- Ov (tier-dependent), per operation (eq. 41): retrieval 2·ts·s(c) + tr·(2·hmax − t(c) − hout); relocation 2·ts·|s(c) − S| + tr·(2·hmax − t(c) − H*_S), with H*_S the new top tier of the destination stack.
- Of(S) and Ov(S) are the sums over all operations (eq. 42–43).
- The paper notes that this crane movement is an approximation, not exact when the target changes or the last container is retrieved (p. 85).
- Lower bounds (p. 85), with NW the set of non-well-located containers:
  - LB_f = Σ_c (tpp + 2·s(c)·ts) + Σ_{c∈NW} (tpp + ts) (eq. 44)
  - LB_v = Σ_c ((2·hmax − t(c) − hout)·tr + 2·s(c)·ts) + Σ_{c∈NW} (2·tr + ts) (eq. 45)
- val_f(S) = 1 / (Of(S) − LB_f + 1), val_v(S) = 1 / (Ov(S) − LB_v + 1) (eq. 46–47). Used for the global update, the initial pheromone, and the early-abort check.

### Parameters (p. 85–86, 88)
- p = 0.1, ϕ = 0.9, q0 = 0.9, n = 10 ants.
- Stopping criterion: 5000 colony iterations for the rBRP (1000 for the uBRP).
- Stagnation: MaxConst = 100 iterations without improvement → reinitialise pheromones.
- MaxMoves = 10.
- Crane-time experiments: Hmax = T + 2; ts = 1.2 s; tpp = 30 s (Of,30) or 5 s (Of,5); tr = 7.77 s; hout = 1.5; hmax = Hmax + 1 (p. 88). The paper says the ACO needed "around 5 times" more iterations for the crane-time objectives than for the relocation count (p. 88); whether the stopping criterion was raised accordingly is not stated.

### Not specified in the paper
- Tie-breaking in MinMax and in the argmax of the transition rule.
- How τ_min is applied (after every update, or only after the global update).
- The value used when reinitialising the pheromone matrix (τ0 from the greedy solution, or recomputed from S_best).
- The Zhu et al. (2012) lower bound itself.
- The number of runs per instance and any random seeds.
- Tier numbering: Sec. 2 calls the ground "tier 0", but hout = 1.5 and hmax = Hmax + 1 are taken from Schwarze & Voß, where the ground is tier 1. The matching optimal values (see §4) suggest the Schwarze & Voß convention is used in the crane-time experiments.

### Validation material in the paper
- Instances: Caserta et al. (2009), T × S from 3 × 3 to 10 × 10, 40 instances per size (p. 85). These are the Caserta instances in CRISP.
- **Original objective (relocations, rBRP, Hmax = T + 2)**: Table 1 (p. 86), average number of relocations over 40 instances, e.g. 5 × 7: ACO 24.33, optimum 22.08.
- **Crane-time objectives (rBRP, Hmax = T + 2)**: Table 5 (p. 88), per size the average optimal value (OPT) and the average error of the ACO (E_ACO), in seconds. Of,30 and Of,5 for 3 × 3 to 5 × 4; Ov for 3 × 3 to 4 × 5. Largest average errors: 1.7 s (Of,30), 3.5 s (Of,5), 20 s (Ov) (p. 88).
- Cross-check: Table 5 gives OPT = 476.4 for Of,30 on 3 × 3, the same as the average optimal f2 for set 3–3 in Voß & Schwarze (2019, Table 1, p. 108). The two papers therefore use the same setting.

## 2. Comparison with the code

File references without a directory are to `algorithms/CRP_Time/heuristic/jovanovic_2019_aco/`. Line numbers refer to `42a27fa` unless a row names a later commit; they will be updated after the repair (§4, "Repair runs").

| Step | Paper (p.) | Code (file:line) | Verdict | Category |
|---|---|---|---|---|
| 2D bay, W stacks, Hmax; Caserta instances | 79, 85–86 | stacks read in stack-index order (`algorithm.py:165-176`); Caserta loader: one stack per bay, `max_tiers` = T + 2 = Hmax (`core/benchmarks/caserta.py:154-160`, `:227-229`) | equal | — |
| rBRP: only the container on top of the target's stack is relocated (RES) | 79; eq. 11, p. 81 | `algorithm.py:252-257` | equal | — |
| Candidate stacks R_c: non-full stacks except the source | eq. 1, p. 80 | ACO `algorithm.py:270-274`; greedy `scoring.py:245-249` | equal | — |
| dd(S) = minimum due date, N + 1 for an empty stack | eq. 2, p. 80 | greedy `scoring.py:24-31`; ACO keeps the stack minima with N + 1 for empty stacks (`algorithm.py:194`, `:327-336`, `:375-380`) | equal | — |
| dif(c, d) | eq. 3, p. 80 | `scoring.py:52-66`; ACO inline `algorithm.py:281` | equal | — |
| Greedy MinMax: argmin of dif over R_c | Alg. 1, eqs. 5, 7, p. 80 | `run_greedy_rbrp_time` (`scoring.py:200-280`) | equal | — |
| Ties in MinMax | not specified | strict `<` keeps the first stack in index order (`scoring.py:251`) | not specified in paper | choice left open by paper |
| Pheromone matrix τ_cdnt, d = 1..N + W, n = 0..MaxMoves | p. 83 | `np.full((N, N + W, MaxMoves + 1, N), τ0)` (`algorithm.py:214-217`); indices c − 1, d − 1, n, t − 1 (`:263-265`, `:284`) | equal | — |
| dd*(S) = N + i(S) for an empty stack | p. 83 | `empty_d` (`algorithm.py:195`, `:278`); greedy `scoring.py:34-45`, `:258` | equal | — |
| n = times c has already been moved | p. 83 | `min(M[c], MaxMoves)` before the move, `M[c] += 1` after (`algorithm.py:264`, `:339`) | equal | — |
| n > MaxMoves | not specified | clamped to MaxMoves (`algorithm.py:264`) | not specified in paper | choice left open by paper |
| Heuristic f(c, S) = 1 / (1 + dif(c, dd(S))) | eqs. 26–27, p. 83 | dif is computed with d = dd*(S), i.e. N + i(S) instead of N + 1 for an empty stack (`algorithm.py:278-282`) | deviation | differs from paper |
| g(α) = f(α) · τ_c,dd*(S),m_c,t | eqs. 29–30, p. 83 | `algorithm.py:284-285` | equal | — |
| Transition rule: argmax g if q < q0, else roulette on g / Σg | eqs. 31–32, p. 83 | `algorithm.py:300-313` | equal | — |
| Ties in the argmax | not specified | strict `>` keeps the first stack in index order (`algorithm.py:290`) | not specified in paper | choice left open by paper |
| Solution stored as 4-tuples (c, dd*(S), M[c], t) | p. 84 | `algorithm.py:316-317`; decoded to a plan in `_solution_to_plan` (`algorithm.py:42-91`) | equal | — |
| val(S) = 1 / (\|S\| − LB + 1), LB of the initial bay (Zhu et al. 2012 for the rBRP) | eq. 33, p. 84 | relocation objective: LB = 0, since `lb_time` returns a bound only for f2 and f2vert (`scoring.py:96-98`, since `f7a9c9a`); the NWL count is computed (`algorithm.py:185`) but not used | deviation | differs from paper (Zhu bound not implemented; accepted as a limitation of Check 1) |
| val_f, val_v with LB_f, LB_v | eqs. 44–47, p. 85 | since `f7a9c9a`: `lb_time` computes eq. 44 or 45 literally (`scoring.py:96-138`) for the initial bay (`algorithm.py:187`); val uses it (`algorithm.py:212`, `:410`, `:417`) | equal | — |
| O_f, O_v: per-operation costs, summed | eqs. 40–43, p. 85 | `movement_objective_cost` with the shared `ObjectiveSpec` (`algorithm.py:342-348`, `:366-372`), i.e. f2 and f2vert (`core/objectives.py:261-295`) | equal | — |
| Parameters ts = 1.2, tpp = 30 or 5, tr = 7.77, hout = 1.5, hmax = Hmax + 1 | p. 88 | `ObjectiveSpec` defaults ts = 1.2, tpp = 30, tr = 2.59 + 5.18, hout = 1.5 (`core/objectives.py:141-145`); h_max = `max_tiers` + 1 (`core/objectives.py:280`); tpp = 5 via `pickup_place_s` | equal | — |
| Tier numbering | p. 79 vs p. 85, 88 | 1-based: source tier = height before the pop, destination tier = height after the push (`algorithm.py:322-323`, `:364`), as in `annotate_plan_tiers` | not specified in paper | choice left open by paper |
| Local update τ = ϕτ for all tuples of S_i, also for an aborted ant | eq. 36, p. 84 | `algorithm.py:383-396`, for valid and aborted ants | equal | — |
| Global update with S_best only, after every iteration | eqs. 34–35, p. 84 | `algorithm.py:415-432` | equal | — |
| τ0 = (1/W) · val(S_g) | eq. 37, p. 84 | `tau_0 = val_greedy / n_stacks` (`algorithm.py:211-212`) | equal | — |
| τmin = (1/W²) · val(S_best) | eq. 38, p. 84 | `algorithm.py:418` | equal | — |
| How τmin is applied | not specified | as a floor in both the local and the global update (`algorithm.py:393-396`, `:429-432`); before the first global update τmin = τ0 / W (`:213`) | not specified in paper | choice left open by paper |
| Early abort: \|S\| + LB(Bay) ≥ \|S_best\|; for crane time with O and LB_f / LB_v | Alg. 2, p. 84; p. 85 | since `c87b7d5`: `time_so_far + lb_cur >= best_cost` after each relocation (`algorithm.py:377`), with LB(Bay) kept incrementally (`:199-209`, `:254-255`, `:339`, `:365-373`, `:401-405`); LB = 0 for the relocation objective | equal | — |
| val when O − LB + 1 < 1 | not specified | clamped: val = 1 / max(O − LB + 1, 1) (`algorithm.py:211`, `:409`, `:416`) | not specified in paper | choice left open by paper |
| Initial S_best | not specified (Alg. 2 uses \|S_best\| from the first ant on) | the greedy solution and its cost (`algorithm.py:179`, `:199-208`) | not specified in paper | choice left open by paper |
| New best: "Check if S is valid new best solution" | Alg. 2, p. 84 | valid and strictly lower cost (`algorithm.py:399-403`) | equal | — |
| Reinitialisation after MaxConst iterations without improvement, then global update | Alg. 2, p. 84 | `algorithm.py:405-413`, then `:415-432` | equal | — |
| Value used for the reinitialisation | not specified | val(S_best) / W (`algorithm.py:409-412`) | not specified in paper | choice left open by paper |
| p = 0.1, ϕ = 0.9, q0 = 0.9, n = 10, 5000 iterations (rBRP), MaxConst = 100, MaxMoves = 10 | p. 85–86 | defaults `rho` 0.1, `phi` 0.9, `q0` 0.9, `n_ants` 10, `n_iterations` 5000, `max_const_iter` 100, `max_moves` 10 (`algorithm.py:125-131`, schema `:511-560`) | equal | — |
| Iterations for the crane-time objectives ("around 5 times higher") | p. 88 | default 5000 for every objective; the help text suggests "10 000+" (`algorithm.py:514-522`) | not specified in paper | choice left open by paper |
| Runs per instance, random seeds | not specified | one run, `np.random.seed(cfg.seed)` (`algorithm.py:124`, `:145`) | not specified in paper | choice left open by paper |
| Objective: relocations (Table 1) or O_f / O_v (Table 5) | p. 84–85 | the search minimises the objective of the shared `ObjectiveSpec` (`algorithm.py:151`); CRP-Time offers only `crane_time` in the UI and the CLI (`problems/CRP_Time.py:144-154`, `main.py:258-261`); `objective_mode = "relocations"` can only be set programmatically | equal | unified 2D setting |
| Output: best solution found | Alg. 2 | `S_best` decoded and evaluated with the shared evaluator (`algorithm.py:459-470`) | equal | — |

Verdict: equal / deviation / not specified in paper.
Category: unified 2D setting / choice left open by paper / differs from paper (bug).

## 3. Deviations and decisions

### 3.1 Unified 2D setting

**The paper's crane-time setting is the unified setting.** O_f and O_v (eqs. 40–43) are the Schwarze & Voß cost functions, and the parameters on p. 88 are the Voß & Schwarze values, so O_f,30 = f2, O_f,5 = f2 with tpp = 5, and O_v = f2vert. The code uses the shared evaluator for both (`algorithm.py:342-348`, `:366-372`). The instances are the Caserta instances with Hmax = T + 2, as in CRISP. For Table 5, "the paper's own settings" and the crane-time-only setting therefore coincide.

**Relocation objective.** Table 1 uses the number of relocations. CRP-Time allows only crane time in the UI and the CLI, so the relocation objective can only be selected by setting `objective_mode = "relocations"` in the problem configuration from a script. The search then counts relocations: `movement_objective_cost` gives 1 per relocation and 0 per retrieval.

**Crane start position.** `crane_pos = (1, 1)` (`algorithm.py:240`) is only read by the `rmgc_current` model; f2 and f2vert do not depend on it.

### 3.2 Choices left open by the paper

**Ties.** In the greedy (`scoring.py:251`) and in the argmax of the transition rule (`algorithm.py:290`), the first stack in index order wins.

**n beyond MaxMoves.** A container moved more than MaxMoves times uses n = MaxMoves (`algorithm.py:264`).

**τmin** is applied as a floor in both updates (`algorithm.py:393-396`, `:429-432`). **Reinitialisation** uses val(S_best) / W (`algorithm.py:409-412`). **Initial S_best** is the greedy solution (`algorithm.py:179`). **New best** requires a strictly lower cost (`algorithm.py:399`).

**Clamp on val.** val = 1 / max(O − LB + 1, 1) (`algorithm.py:211`, `:409`, `:416`). With a valid lower bound O − LB + 1 ≥ 1 and the clamp never binds. With LB_f taken literally from eq. 44 it can bind (§3.6); the paper does not say what val is then. Decision (Thom): keep the clamp.

**Iterations for crane time.** The paper says the number of iterations for the crane-time objectives "was around 5 times higher" than for the relocation count (p. 88), without a stopping value. The default stays 5000; the help text's "10 000+" (`algorithm.py:519-521`) is not from the paper.

**Runs and seeds.** One run per instance with `np.random.seed(cfg.seed)` (`algorithm.py:145`); the paper does not say how many runs Tables 1 and 5 average over.

**Tier numbering.** The code uses the Schwarze & Voß convention (ground tier 1, h_max = Hmax + 1), which §1 infers from the matching optimal values.

### 3.3 Differs from the paper

**No lower bound in val and in the early abort.** The paper's quality function subtracts a lower bound of the initial bay: the Zhu et al. (2012) bound for the relocation objective (eq. 33), LB_f or LB_v for crane time (eqs. 44–47). The early abort compares the partial cost plus the lower bound of the current bay with the best cost (Alg. 2; p. 85). The code sets the bound to 0 in both places (`algorithm.py:186`, comment "safe for every selectable objective"), so val = 1 / (O + 1) and an ant is aborted only when its cost so far reaches the best cost (`algorithm.py:350-352`). The CRP-R version of this algorithm (`algorithms/CRP_R/heuristic/jovanovic_2019_aco/`) does subtract a bound, the non-well-located count.

Repair: step 1a (`f7a9c9a`) puts LB_f / LB_v of the initial bay into val. Step 1b (`c87b7d5`) adds LB(Bay) of the current bay to the early abort. For the relocation objective LB stays 0. Effect: §4, "Repair runs".

Effect, measured on data4-5-1 under f2 (scratchpad computation): LB_f = 1087.2, the greedy solution costs 1281.6 and the best solution of a 5000-iteration run 1262.4.
- Pheromone scale: with LB_f, Δτ / τ0 for that best solution is 5.54; with LB = 0 it is 5.08. On this instance the missing bound changes the ratio little.
- Early abort: LB_f is 86 % of the best cost. With the bound, an ant is aborted as soon as its cost so far plus the retrieval and relocation cost still ahead exceeds the best; without it, only once its cost so far alone does, which is near the end of the solution. The paper names early abort mainly as a way of "not wasting the information stored in the pheromone matrix" (p. 84): every tuple of the partial solution gets the local update, so ants that run on longer decay more pheromone entries.

Where the check sits is as in the paper: Alg. 2 checks only after a relocation, and so does the code. A check after a retrieval would change nothing, because a retrieval adds to O exactly the term it removes from LB(Bay).

**The heuristic uses dd* instead of dd for empty stacks.** Eq. 27 evaluates the heuristic with dd(S), which is N + 1 for every empty stack; dd*(S) = N + i(S) is only the pheromone index (eq. 29). The code uses dd* in both (`algorithm.py:278-282`), so an empty stack gets f = 1 / (1 + N + i(S) − c) instead of 1 / (N + 2 − c): the higher its index, the less attractive it looks. Under uniform pheromone the argmax is the same, because a well-located stack still beats any empty stack, an empty stack still beats any stack where c is not well-located, and among empty stacks the lowest index wins in both cases. The difference shows in the roulette choice and in the argmax once τ differs between stacks. Example: N = 35, c = 10, empty stack i = 7: f = 1/33 in the code against 1/27 in the paper.

### 3.4 Suspected bugs

**J1 — the greedy start retrieves containers out of order when there is no room.** If no destination stack exists, `run_greedy_rbrp_time` breaks out of the relocation loop without retrieving the target (`scoring.py:255-256`) and continues with the next target, which can then be retrieved before the blocked one. Scratchpad check on stacks `[[1, 2], [3, 4]]` with H = 2 (target 1 blocked, the only other stack full): the greedy returns `[(4, 1, 0, 3)]`, a relocation of container 4 during target 3, and decoding it with `_solution_to_plan` raises `KeyError: None`. This requires an empty candidate list, which cannot occur on Caserta (`azari_2017.md` §3.2), so the bug is latent there. The ACO ants handle the same case by marking the solution invalid (`algorithm.py:295-297`).

### 3.5 Notes for the standardisation (not deviations)

- The class description says the crane-time objective is computed "via platform 2-D KinematicsModel (bay × row, gantry + trolley)" (`algorithm.py:98-105`). The search uses the selected f2 or f2vert through the shared evaluator; the kinematics model is only read by `rmgc_current`.
- Dead code from the legacy RMGC model: the greedy accumulates a crane time with inline gantry/trolley kinematics (`scoring.py:94-170`, `:263-277`) that is discarded (`_legacy_greedy_cost`, `algorithm.py:179-182`), and the four kinematics parameters are read only for that (`algorithm.py:159-163`). `compute_lb_time` (`scoring.py:177-193`) and `lb_init_nwl` (`algorithm.py:185`) are unused.
- Intermediate progress records carry `lower_bound = 0.0` and `iteration` (`algorithm.py:446-449`). The `lower_bound` value is not a lower bound, and the metric set differs from the other CRP-Time algorithms.
- The seed loop and the mean over seeds (`algorithm.py:124`, `:136`, `:490-501`) are left over from multi-seed evaluation; with `n_seeds = 1` the final record describes one plan.
- The header docstring lists 3 of the 7 parameters (`algorithm.py:5-7`).
- `fidelity = "adapted"` (`algorithm.py:109`). To be revisited after §3.3.
- The base parameters `max_iterations` and `report_interval` appear in the UI but are not read; the ACO uses `n_iterations` and pushes every `n_iterations // 50` iterations (`algorithm.py:132`). Same platform point as in `azari_2017.md` §3.5.

### 3.6 Observations on the paper

**LB_f (eq. 44) is not a valid lower bound.** Eq. 44 counts, for every container, its retrieval from its current stack, plus tpp + ts for every non-well-located container. A relocated container is retrieved from its new stack, which can be closer to stack 0.

Counterexample (checked by hand by Thom): three stacks, containers 1 and 2 in stack 3 with 2 on top, stacks 1 and 2 empty.
- Plan: relocate 2 to stack 1, retrieve 1, retrieve 2. Cost by eq. 40: (4 ts + tpp) + (6 ts + tpp) + (2 ts + tpp) = 12 ts + 3 tpp.
- Eq. 44: (6 ts + tpp) + (6 ts + tpp) + (tpp + ts) = 13 ts + 3 tpp.
- With the shared evaluator (ts = 1.2, tpp = 30, Hmax = 2): plan 104.40, LB_f 105.60.

In general, for a container relocated once from stack s to stack s′ and then retrieved, the horizontal cost is 2 ts (|s − s′| + s′), which equals 2 ts · s when s′ < s. The extra ts in the second term of eq. 44 is then not paid.

LB_v (eq. 45) has the same horizontal term, but its vertical part leaves room for it. For the same container, moved from tier t to tier t′, eq. 41 exceeds the vertical part of eq. 45 by 2 tr (Hmax − t′). With tr = 7.77 s > ts, that covers the extra ts unless t′ = Hmax.

On the Caserta instances, the literal LB of the initial bay against the best solution found by the baseline (§4):

| Objective | Instances | LB above the cost of a found solution | LB / best (min, mean, max) |
|---|---|---|---|
| O_f,30 | 440 | 11 | 0.731, 0.920, 1.009 |
| O_f,5 | 440 | 13 | 0.694, 0.909, 1.032 |
| O_v | 320 | 0 | 0.670, 0.807, 1.000 |

Consequences, if implemented literally: val can be clamped (§3.2), and the early abort can stop ants that would reach a solution below LB_f. Decision (Thom): implement eq. 44 literally and measure the effect after the lower bound is added to the early abort. Measured in §4, step 1b: with the literal LB_f in the early abort, O_f gets worse; with a valid bound (eq. 44 without the ts) it hardly does. Script: `docs/validation/data/jovanovic_2019_baseline/jov_lb_check.py` (local).

## 4. Behavioural checks

The paper has no worked example. It reports averages over the 40 Caserta instances per size, which can be compared with the code (§1, "Validation material"). Running time of the current code with the default 5000 iterations, one instance, one process on the machine of `azari_2017.md` Check 5: data4-5-1 (f2) 7.7 s, data5-7-1 (relocations) 9.0 s, data10-10-1 (f2) 62.2 s.

### Baseline of the current code

All runs below use the algorithm code as of `42a27fa` (last changed in `308d016`), before any repair of §3.3–3.4. Setup:
- 5000 iterations and the paper's other parameters (§1), one run per instance with seed 0, averaged over the 40 instances per size;
- in-process runs through a scratch driver, 6 processes in parallel, nothing written to `results/`; 2520 runs in total, 82.5 min;
- raw results, summary and scripts: `docs/validation/data/jovanovic_2019_baseline/` (local, not in git).

The paper gives its averages to 0.01 (Table 1) or 0.1 s (Table 5). The code's averages are rounded half-up to the same precision before comparing. No average of the code lies below OPT at that precision. On 3 × 3 the code reaches OPT exactly for O_f,30 (476.4) and for O_v (904.0), which supports the reading in §1 and §3.1 that O_f = f2 and O_v = f2vert, with the Schwarze & Voß tier numbering.

### Check 1 — Table 1: relocations, rBRP, Hmax = T + 2 — **done (baseline)**

Limitation: the code has no Zhu et al. (2012) bound (§3.3), so this check runs with LB = 0 in val and in the early abort. A difference from Table 1 is expected for that reason alone.

| T × S | code | ACO (Table 1) | code − ACO | OPT (Table 1) |
|---|---|---|---|---|
| 3 × 3 | 5.00 | 5.00 | 0.00 | 5.00 |
| 3 × 4 | 6.18 | 6.18 | 0.00 | 6.18 |
| 3 × 5 | 7.03 | 7.02 | +0.01 | 7.02 |
| 3 × 6 | 8.40 | 8.40 | 0.00 | 8.40 |
| 3 × 7 | 9.28 | 9.28 | 0.00 | 9.28 |
| 3 × 8 | 10.65 | 10.65 | 0.00 | 10.65 |
| 4 × 4 | 10.20 | 10.20 | 0.00 | 10.20 |
| 4 × 5 | 12.95 | 12.95 | 0.00 | 12.95 |
| 4 × 6 | 14.03 | 14.02 | +0.01 | 14.00 |
| 4 × 7 | 16.13 | 16.12 | +0.01 | 16.12 |
| 5 × 4 | 15.45 | 15.42 | +0.03 | 15.28 |
| 5 × 5 | 18.98 | 18.95 | +0.03 | 18.65 |
| 5 × 6 | 22.15 | 22.15 | 0.00 | 21.95 |
| 5 × 7 | 24.30 | 24.33 | −0.03 | 22.08 |
| 5 × 8 | 27.78 | 27.73 | +0.05 | – |
| 5 × 9 | 30.48 | 30.50 | −0.02 | – |
| 5 × 10 | 33.35 | 33.40 | −0.05 | – |
| 6 × 6 | 31.15 | 31.05 | +0.10 | – |
| 6 × 10 | 45.95 | 45.93 | +0.02 | – |
| 10 × 6 | 79.78 | 79.50 | +0.28 | – |
| 10 × 10 | 114.50 | 113.45 | +1.05 | – |

Result: 19 of 21 sizes are within ±0.10 of the ACO column. The code is worse on the two largest sizes: 10 × 6 by 0.28 and 10 × 10 by 1.05. That is larger than the seed spread on 5 × 7 (below). Whether the missing Zhu bound causes it has not been tested.

### Check 2 — Table 5: O_f,30 and O_f,5 — **done (baseline)**

| T × S | O_f,30 code | OPT | code − OPT | E_ACO | O_f,5 code | OPT | code − OPT | E_ACO |
|---|---|---|---|---|---|---|---|---|
| 3 × 3 | 476.4 | 476.4 | 0.0 | 0.0 | 126.4 | 126.4 | 0.0 | 0.0 |
| 3 × 4 | 633.7 | 633.6 | +0.1 | 0.1 | 178.5 | 178.4 | +0.1 | 0.0 |
| 3 × 5 | 789.2 | 789.2 | 0.0 | 0.1 | 236.6 | 236.6 | 0.0 | 0.3 |
| 3 × 6 | 968.2 | 968.2 | 0.0 | 0.1 | 303.7 | 303.2 | +0.5 | 0.8 |
| 3 × 7 | 1137.1 | 1137.0 | +0.1 | 0.0 | 375.5 | 375.2 | +0.3 | 0.9 |
| 3 × 8 | 1331.3 | 1331.3 | 0.0 | 0.3 | 457.7 | 455.1 | +2.6 | 1.9 |
| 4 × 4 | 911.9 | 911.9 | 0.0 | 0.0 | 255.6 | 255.5 | +0.1 | 0.2 |
| 4 × 5 | 1170.2 | 1170.2 | 0.0 | 0.0 | 344.5 | 343.0 | +1.5 | 1.3 |
| 4 × 6 | 1386.0 | 1385.4 | +0.6 | 0.4 | 429.5 | 428.5 | +1.0 | 2.2 |
| 4 × 7 | 1647.8 | 1647.7 | +0.1 | 0.8 | 537.6 | 532.8 | +4.8 | 3.5 |
| 5 × 4 | 1230.9 | 1230.0 | +0.9 | 1.7 | 344.3 | 343.5 | +0.8 | 0.6 |

Result:
- O_f,30: the error of the code is at most E_ACO on 9 of 11 sizes. It is larger on 4 × 6 (+0.6 against 0.4) and on 3 × 7 (+0.1 against 0.0, at the paper's rounding precision).
- O_f,5: the error is at most E_ACO on 6 of 11 sizes. It is larger on 3 × 8 (+2.6 against 1.9), 4 × 5 (+1.5 against 1.3), 4 × 7 (+4.8 against 3.5) and 5 × 4 (+0.8 against 0.6), and on 3 × 4 (+0.1 against 0.0, at the rounding precision).

### Check 3 — Table 5: O_v — **done (baseline)**

| T × S | O_v code | OPT | code − OPT | E_ACO |
|---|---|---|---|---|
| 3 × 3 | 904.0 | 904.0 | 0.0 | 1.8 |
| 3 × 4 | 1187.7 | 1187.0 | +0.7 | 6.2 |
| 3 × 5 | 1461.2 | 1460.8 | +0.4 | 6.8 |
| 3 × 6 | 1760.2 | 1758.4 | +1.8 | 12.4 |
| 3 × 7 | 2045.7 | 2043.4 | +2.3 | 18.3 |
| 3 × 8 | 2369.4 | 2362.6 | +6.8 | 25.5 |
| 4 × 4 | 1929.9 | 1928.1 | +1.8 | 10.7 |
| 4 × 5 | 2429.2 | 2424.6 | +4.6 | 20.2 |

Result: the error of the code is below E_ACO on all 8 sizes, by a wide margin: +4.6 against 20.2 on 4 × 5 and +6.8 against 25.5 on 3 × 8. See the open observation below.

### Seed spread

Seeds 0–4 on one size per objective, 40 instances each.

| Objective, T × S | Seeds 0–4: average over 40 instances | Best of 5 per instance, averaged | Paper |
|---|---|---|---|
| Relocations, 5 × 7 | 24.30, 24.38, 24.33, 24.35, 24.33 | 24.25 | ACO 24.33 |
| O_f,30, 4 × 5 | 1170.2, 1170.3, 1170.9, 1170.3, 1170.3 | 1170.2 | OPT + E_ACO 1170.2 |
| O_v, 4 × 5 | 2429.2, 2429.6, 2429.1, 2428.5, 2429.8 | 2427.1 | OPT + E_ACO 2444.8 |

The spread of the size average across seeds is 0.08 relocations on 5 × 7, 0.7 s for O_f,30 on 4 × 5 and 1.3 s for O_v on 4 × 5. The O_v gap to the paper on 4 × 5 (15.6 s between the code's seed-0 average and OPT + E_ACO) is not explained by seed variation.

### Open observation — the code does better than the paper on O_v

With 5000 iterations the code finds better O_v solutions than the paper reports, on every size of Table 5, although the paper used "around 5 times" more iterations for the crane-time objectives (p. 88). On O_f,30 and O_f,5 the code is close to the paper. The cause is not known.

Hypothesis, to be tested after the repair: if the O_v error rises towards E_ACO once LB_v (eq. 45) is introduced, the missing early abort explains the difference.

Outcome (§4, steps 1a and 1b): the O_v error did not rise, neither with LB_v in val nor with LB_v in the early abort, although the abort now acts early. The hypothesis is refuted; the difference with the paper remains unexplained.

### Repair runs

Each repair step is run on the code of its own commit, with the same jobs, settings and driver as the baseline. Raw results and scripts: `docs/validation/data/jovanovic_2019_repair/` (local, not in git).

#### Step 1a — LB_f / LB_v in val (`f7a9c9a`), O_v

Seed 0 on the 8 sizes of Table 5, and seeds 0–4 on 4 × 5; 480 runs.

| T × S | OPT | E_ACO | baseline: code − OPT | step 1a: code − OPT |
|---|---|---|---|---|
| 3 × 3 | 904.0 | 1.8 | 0.0 | 0.0 |
| 3 × 4 | 1187.0 | 6.2 | +0.7 | +1.1 |
| 3 × 5 | 1460.8 | 6.8 | +0.4 | +0.4 |
| 3 × 6 | 1758.4 | 12.4 | +1.8 | +1.9 |
| 3 × 7 | 2043.4 | 18.3 | +2.3 | +2.3 |
| 3 × 8 | 2362.6 | 25.5 | +6.8 | +2.7 |
| 4 × 4 | 1928.1 | 10.7 | +1.8 | +2.3 |
| 4 × 5 | 2424.6 | 20.2 | +4.6 | +3.9 |

- Per instance against the baseline (seed 0, 320 instances): 11 better, 301 equal, 8 worse.
- Seeds 0–4 on 4 × 5, average over 40 instances per seed: 2428.5, 2428.5, 2429.2, 2431.1, 2429.7 (baseline 2428.5–2429.8).

Result: no measurable effect. The differences per size fall within the seed spread (4 × 5, seeds 0–4, over the baseline and steps 1a and 1b: 2428.2–2431.1). The O_v error stays far below E_ACO on every size.

#### Step 1b — LB(Bay) in the early abort (`c87b7d5`), O_v, O_f,30 and O_f,5

Same jobs as the baseline for the three crane-time objectives; 1520 runs.

O_v:

| T × S | OPT | E_ACO | baseline: code − OPT | step 1a: code − OPT | step 1b: code − OPT |
|---|---|---|---|---|---|
| 3 × 3 | 904.0 | 1.8 | 0.0 | 0.0 | 0.0 |
| 3 × 4 | 1187.0 | 6.2 | +0.7 | +1.1 | +0.7 |
| 3 × 5 | 1460.8 | 6.8 | +0.4 | +0.4 | +0.4 |
| 3 × 6 | 1758.4 | 12.4 | +1.8 | +1.9 | +1.1 |
| 3 × 7 | 2043.4 | 18.3 | +2.3 | +2.3 | +3.2 |
| 3 × 8 | 2362.6 | 25.5 | +6.8 | +2.7 | +4.4 |
| 4 × 4 | 1928.1 | 10.7 | +1.8 | +2.3 | +1.0 |
| 4 × 5 | 2424.6 | 20.2 | +4.6 | +3.9 | +4.9 |

- Per instance against the baseline (seed 0, 320 instances): 14 better, 290 equal, 16 worse.
- Seeds 0–4 on 4 × 5: 2429.5, 2428.2, 2428.7, 2429.5, 2428.9.

The abort now acts early. One 5000-iteration run per instance (50 000 ants):

| Instance, objective | Aborted ants before (`42a27fa`) | After (`c87b7d5`) | Mean target at abort, before → after | Best, before → after |
|---|---|---|---|---|
| data4-5-1, O_v | 1.3 % | 100.0 % | 19.3 → 11.1 of 20 | 2564.82 → 2564.82 |
| data3-8-1, O_v | 0.0 % | 100.0 % | – → 9.7 of 24 | 2220.90 → 2220.90 |
| data4-5-1, O_f,30 | 1.0 % | 100.0 % | 19.0 → 7.0 of 20 | 1262.40 → 1267.20 |

Nearly every ant is aborted after step 1b, partly because the check is ≥ (Alg. 2): an ant that rebuilds S_best is aborted too.

O_f,30:

| T × S | OPT | E_ACO | baseline: code − OPT | step 1b: code − OPT |
|---|---|---|---|---|
| 3 × 3 | 476.4 | 0.0 | 0.0 | +0.1 |
| 3 × 4 | 633.6 | 0.1 | +0.1 | +0.3 |
| 3 × 5 | 789.2 | 0.1 | 0.0 | +0.3 |
| 3 × 6 | 968.2 | 0.1 | 0.0 | +0.2 |
| 3 × 7 | 1137.0 | 0.0 | +0.1 | +0.8 |
| 3 × 8 | 1331.3 | 0.3 | 0.0 | +0.8 |
| 4 × 4 | 911.9 | 0.0 | 0.0 | +0.1 |
| 4 × 5 | 1170.2 | 0.0 | 0.0 | +0.8 |
| 4 × 6 | 1385.4 | 0.4 | +0.6 | +1.1 |
| 4 × 7 | 1647.7 | 0.8 | +0.1 | +2.0 |
| 5 × 4 | 1230.0 | 1.7 | +0.9 | +1.5 |

O_f,5:

| T × S | OPT | E_ACO | baseline: code − OPT | step 1b: code − OPT |
|---|---|---|---|---|
| 3 × 3 | 126.4 | 0.0 | 0.0 | +0.1 |
| 3 × 4 | 178.4 | 0.0 | +0.1 | +0.5 |
| 3 × 5 | 236.6 | 0.3 | 0.0 | +0.3 |
| 3 × 6 | 303.2 | 0.8 | +0.5 | +0.9 |
| 3 × 7 | 375.2 | 0.9 | +0.3 | +1.4 |
| 3 × 8 | 455.1 | 1.9 | +2.6 | +3.3 |
| 4 × 4 | 255.5 | 0.2 | +0.1 | +0.2 |
| 4 × 5 | 343.0 | 1.3 | +1.5 | +2.0 |
| 4 × 6 | 428.5 | 2.2 | +1.0 | +2.8 |
| 4 × 7 | 532.8 | 3.5 | +4.8 | +4.9 |
| 5 × 4 | 343.5 | 0.6 | +0.8 | +1.8 |

- Per instance against the baseline (seed 0, 440 instances each): O_f,30 4 better, 384 equal, 52 worse; O_f,5 17 better, 345 equal, 78 worse.
- Seeds 0–4 on 4 × 5, O_f,30: 1171.0, 1170.9, 1171.0, 1171.1, 1170.9 (baseline 1170.2–1170.9).
- Mean running time per run: 5.2 s for O_f,30 and O_f,5 (baseline 7.9 and 8.0 s), 7.4 s for O_v (baseline 6.9 s).
- Relation to §3.6: none of the 52 worse O_f,30 instances is among the 11 where LB_f of the initial bay exceeds a found solution; for O_f,5 it is 1 of 78 (13 such instances). On 11 of 11 and 12 of 13 of those instances step 1b still finds a solution below LB_f of the initial bay.

Result:
- O_v: no measurable effect; the differences per size fall within the seed spread (4 × 5, seeds 0–4: 2428.2–2431.1). The hypothesis in "Open observation" is refuted: the O_v error does not rise towards E_ACO, although the abort now acts early and on nearly every ant (the check is ≥, as in Alg. 2).
- O_f: with LB_f taken literally from eq. 44, the error grows on every size and exceeds E_ACO on 10 (O_f,30) and 9 (O_f,5) of 11 sizes (baseline: 2 and 5). A scratch run with a valid bound (eq. 44 without the ts per non-well-located container; code otherwise as `c87b7d5`, not committed) brings the error back to about the baseline level: per instance against the baseline, O_f,30 6 better and 16 worse (literal: 4 and 52), O_f,5 30 better and 15 worse (literal: 17 and 78); O_f,30 on 4 × 5, seeds 0–4: 1170.3–1170.5 (literal 1170.9–1171.1, baseline 1170.2–1170.9). On O_f,30 a small difference with the baseline remains, e.g. +0.5 against 0.0 on 4 × 4. The worsening comes largely from the overestimate in eq. 44 (§3.6): the early abort can stop ants that could still improve on S_best. This is faithful to the paper, not a bug in the code.

Error of the code (code − OPT), seed 0:

| T × S | O_f,30: baseline | 1b literal | 1b valid bound | E_ACO | O_f,5: baseline | 1b literal | 1b valid bound | E_ACO |
|---|---|---|---|---|---|---|---|---|
| 3 × 3 | 0.0 | +0.1 | 0.0 | 0.0 | 0.0 | +0.1 | 0.0 | 0.0 |
| 3 × 4 | +0.1 | +0.3 | 0.0 | 0.1 | +0.1 | +0.5 | +0.1 | 0.0 |
| 3 × 5 | 0.0 | +0.3 | 0.0 | 0.1 | 0.0 | +0.3 | +0.1 | 0.3 |
| 3 × 6 | 0.0 | +0.2 | +0.1 | 0.1 | +0.5 | +0.9 | +0.3 | 0.8 |
| 3 × 7 | +0.1 | +0.8 | 0.0 | 0.0 | +0.3 | +1.4 | +0.5 | 0.9 |
| 3 × 8 | 0.0 | +0.8 | +0.6 | 0.3 | +2.6 | +3.3 | +2.1 | 1.9 |
| 4 × 4 | 0.0 | +0.1 | +0.5 | 0.0 | +0.1 | +0.2 | +0.2 | 0.2 |
| 4 × 5 | 0.0 | +0.8 | +0.3 | 0.0 | +1.5 | +2.0 | +1.1 | 1.3 |
| 4 × 6 | +0.6 | +1.1 | +0.3 | 0.4 | +1.0 | +2.8 | +0.9 | 2.2 |
| 4 × 7 | +0.1 | +2.0 | +0.8 | 0.8 | +4.8 | +4.9 | +3.4 | 3.5 |
| 5 × 4 | +0.9 | +1.5 | +1.1 | 1.7 | +0.8 | +1.8 | +0.5 | 0.6 |

- Iterations. Scratch run with literal eq. 44 and 25 000 instead of 5000 iterations, O_f,5, seed 0, 3 × 8 and 4 × 7 (about 5.2× the running time per run). Interpretation fixed before the run: if the error is at or below E_ACO on both sizes, the paper's O_f results are compatible with literal eq. 44 plus about 5× as many iterations (compatible, not proven: one seed, two sizes); if not, the iterations do not explain the worsening. Result:
  - the pre-registered condition is not met on 3 × 8, by 0.3 s (+2.2 against E_ACO 1.9); the seed spread on this size was not measured;
  - the extra iterations close most of the gap: fully on 4 × 7 (+4.9 → +3.5 against E_ACO 3.5) and on 3 × 8 from +3.3 to +2.2; per instance 14 better and none worse. A full explanation is not shown;
  - on 3 × 8 (O_f,5) no version of the code reaches E_ACO at 5000 iterations, the baseline included (+2.6; valid bound +2.1), so this size discriminates poorly.

| T × S (O_f,5, seed 0) | E_ACO | baseline, 5000 | literal, 5000 | literal, 25 000 | valid bound, 5000 |
|---|---|---|---|---|---|
| 3 × 8 | 1.9 | +2.6 | +3.3 | +2.2 | +2.1 |
| 4 × 7 | 3.5 | +4.8 | +4.9 | +3.5 | +3.4 |

Raw results: `step1b_validlb_c87b7d5.jsonl` and `iters25k_c87b7d5.jsonl` in `docs/validation/data/jovanovic_2019_repair/` (local).
