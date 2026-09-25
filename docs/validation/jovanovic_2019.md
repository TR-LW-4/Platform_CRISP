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
- Crane-time experiments: Hmax = T + 2; ts = 1.2 s; tpp = 30 s (Of,30) or 5 s (Of,5); tr = 7.77 s; hout = 1.5; hmax = Hmax + 1 (p. 88). Sec. 7.4 (p. 88): "the number of iterations of the ACO algorithm in case of the objective functions based on the crane operation time was around 5 times higher than in the case when the objective function is the number of relocations." The paper gives no stopping value for these runs; the rBRP stopping criterion is 5000 iterations (p. 86).

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

File references without a directory are to `algorithms/CRP_Time/heuristic/jovanovic_2019_aco/`. Line references are to `b379031` unless a commit is given.

| Step | Paper (p.) | Code (file:line) | Verdict | Category |
|---|---|---|---|---|
| 2D bay, W stacks, Hmax; Caserta instances | 79, 85–86 | stacks read in stack-index order (`algorithm.py:167-178`); Caserta loader: one stack per bay, `max_tiers` = T + 2 = Hmax (`core/benchmarks/caserta.py:154-160`, `:227-229`) | equal | — |
| rBRP: only the container on top of the target's stack is relocated (RES) | 79; eq. 11, p. 81 | `algorithm.py:261-266` | equal | — |
| Candidate stacks R_c: non-full stacks except the source | eq. 1, p. 80 | ACO `algorithm.py:279-283`; greedy `scoring.py:150-154` | equal | — |
| dd(S) = minimum due date, N + 1 for an empty stack | eq. 2, p. 80 | greedy `scoring.py:23-30`; ACO keeps the stack minima with N + 1 for empty stacks (`algorithm.py:197`, `:337-346`, `:403-408`) | equal | — |
| dif(c, d) | eq. 3, p. 80 | `scoring.py:51-65`; ACO inline `algorithm.py:290` | equal | — |
| Greedy MinMax: argmin of dif over R_c | Alg. 1, eqs. 5, 7, p. 80 | `run_greedy_rbrp_time` (`scoring.py:121-181`) | equal | — |
| Ties in MinMax | not specified | strict `<` keeps the first stack in index order (`scoring.py:156`) | not specified in paper | choice left open by paper |
| Pheromone matrix τ_cdnt, d = 1..N + W, n = 0..MaxMoves | p. 83 | `np.full((N, N + W, MaxMoves + 1, N), τ0)` (`algorithm.py:221-224`); indices c − 1, d − 1, n, t − 1 (`:272-274`, `:293`) | equal | — |
| dd*(S) = N + i(S) for an empty stack | p. 83 | `empty_d` (`algorithm.py:198`, `:287`); greedy `scoring.py:33-44`, `:168` | equal | — |
| n = times c has already been moved | p. 83 | `min(M[c], MaxMoves)` before the move, `M[c] += 1` after (`algorithm.py:273`, `:349`) | equal | — |
| n > MaxMoves | not specified | clamped to MaxMoves in the ants (`algorithm.py:273`) and, since `802e4d1`, in the greedy start (`scoring.py:169-171`); M keeps the true count (`algorithm.py:349`, `scoring.py:175`) | not specified in paper | choice left open by paper |
| Greedy start without a destination stack (R_c empty) | not specified (Alg. 1, eq. 1, p. 80) | since `0bcfd7d`: `ValueError` (`scoring.py:160-166`); an ant in the same situation is marked invalid (`algorithm.py:304-306`) | not specified in paper | choice left open by paper |
| Heuristic f(c, S) = 1 / (1 + dif(c, dd(S))) | eqs. 26–27, p. 83 | since `4cb92dd`: dif is computed with dd(S) = `smin` (N + 1 for an empty stack) (`algorithm.py:289-291`); the pheromone index and the tuple keep dd*(S) (`:287`, `:293`, `:326`) | equal | — |
| g(α) = f(α) · τ_c,dd*(S),m_c,t | eqs. 29–30, p. 83 | `algorithm.py:293-294` | equal | — |
| Transition rule: argmax g if q < q0, else roulette on g / Σg | eqs. 31–32, p. 83 | `algorithm.py:309-322` | equal | — |
| Ties in the argmax | not specified | strict `>` keeps the first stack in index order (`algorithm.py:299`) | not specified in paper | choice left open by paper |
| Solution stored as 4-tuples (c, dd*(S), M[c], t) | p. 84 | `algorithm.py:325-326`; decoded to a plan in `_solution_to_plan` (`algorithm.py:51-100`) | equal | — |
| val(S) = 1 / (\|S\| − LB + 1), LB of the initial bay (Zhu et al. 2012 for the rBRP) | eq. 33, p. 84 | relocation objective: LB = 0 in val and in the early abort, since `lb_time` returns a bound only for f2 and f2vert (`scoring.py:72-74`, `:105-106`): `lb_init_time` = 0 (`algorithm.py:188`) in val (`:218`, `:437`, `:444`), and `use_lb` is false (`:204`), so `lb_cur` stays 0 (`:250`) and the abort (`:372`) compares \|S\| with \|S_best\| alone; the unused NWL count (`algorithm.py:185` at `42a27fa`) was removed in `4c59f06` | deviation | differs from paper: for the rBRP, p. 84 uses the bound of Zhu et al. (2012) (the number of non-well-located containers only for the uBRP); not implemented, accepted as a limitation of Check 1 |
| val_f, val_v with LB_f, LB_v | eqs. 44–47, p. 85 | since `f7a9c9a`: `lb_time` computes eq. 44 or 45 literally (`scoring.py:72-114`) for the initial bay (`algorithm.py:188`); val uses it (`algorithm.py:218`, `:437`, `:444`) | equal | — |
| O_f, O_v: per-operation costs, summed | eqs. 40–43, p. 85 | `movement_objective_cost` with the shared `ObjectiveSpec` (`algorithm.py:352-358`, `:387-393`), i.e. f2 and f2vert (`core/objectives.py:261-295`) | equal | — |
| Parameters ts = 1.2, tpp = 30 or 5, tr = 7.77, hout = 1.5, hmax = Hmax + 1 | p. 88 | `ObjectiveSpec` defaults ts = 1.2, tpp = 30, tr = 2.59 + 5.18, hout = 1.5 (`core/objectives.py:141-145`); h_max = `max_tiers` + 1 (`core/objectives.py:280`); tpp = 5 via `pickup_place_s` | equal | — |
| Tier numbering | p. 79 vs p. 85, 88 | 1-based: source tier = height before the pop, destination tier = height after the push (`algorithm.py:331-332`, `:385`), as in `annotate_plan_tiers` | not specified in paper | choice left open by paper |
| Local update τ = ϕτ for all tuples of S_i, also for an aborted ant | eq. 36, p. 84 | `algorithm.py:411-424`, for valid and aborted ants | equal | — |
| Global update with S_best only, after every iteration | eqs. 34–35, p. 84 | `algorithm.py:443-460` | equal | — |
| τ0 = (1/W) · val(S_g) | eq. 37, p. 84 | `tau_0 = val_greedy / n_stacks` (`algorithm.py:218-219`) | equal | — |
| τmin = (1/W²) · val(S_best) | eq. 38, p. 84 | `algorithm.py:446` | equal | — |
| How τmin is applied | not specified | as a floor in both the local and the global update (`algorithm.py:421-424`, `:457-460`); before the first global update τmin = τ0 / W (`:220`) | not specified in paper | choice left open by paper |
| Early abort: \|S\| + LB(Bay) ≥ \|S_best\|; for crane time with O and LB_f / LB_v | Alg. 2, p. 84; p. 85 | since `c87b7d5`: `time_so_far + lb_cur >= best_cost` after each relocation (`algorithm.py:372`), with LB(Bay) kept incrementally (`:194-204`, `:249-250`, `:334`, `:360-368`, `:396-400`); LB = 0 for the relocation objective | equal | — |
| val when O − LB + 1 < 1 | not specified | clamped: val = 1 / max(O − LB + 1, 1) (`algorithm.py:218`, `:437`, `:444`) | not specified in paper | choice left open by paper |
| Initial S_best | not specified (Alg. 2 uses \|S_best\| from the first ant on) | the greedy solution and its cost (`algorithm.py:181`, `:206-215`) | not specified in paper | choice left open by paper |
| New best: "Check if S is valid new best solution" | Alg. 2, p. 84 | valid and strictly lower cost (`algorithm.py:427-431`) | equal | — |
| Reinitialisation after MaxConst iterations without improvement, then global update | Alg. 2, p. 84 | `algorithm.py:433-441`, then `:443-460` | equal | — |
| Value used for the reinitialisation | not specified | val(S_best) / W (`algorithm.py:437-440`) | not specified in paper | choice left open by paper |
| p = 0.1, ϕ = 0.9, q0 = 0.9, n = 10, 5000 iterations (rBRP), MaxConst = 100, MaxMoves = 10 | p. 85–86 | defaults `rho` 0.1, `phi` 0.9, `q0` 0.9, `n_ants` 10, `n_iterations` 5000, `max_const_iter` 100, `max_moves` 10 (`algorithm.py:134-140`, schema `:536-585`) | equal | — |
| Iterations for the crane-time objectives ("around 5 times higher") | p. 88 | default 5000 for every objective; since `b379031` the help text paraphrases Sec. 7.4 (`algorithm.py:539-546`) | not specified in paper | choice left open by paper |
| Runs per instance, random seeds | not specified | one run, `np.random.seed(cfg.seed)` (`algorithm.py:133`, `:154`) | not specified in paper | choice left open by paper |
| Objective: relocations (Table 1) or O_f / O_v (Table 5) | p. 84–85 | the search minimises the objective of the shared `ObjectiveSpec` (`algorithm.py:160`); CRP-Time offers only `crane_time` in the UI and the CLI (`problems/CRP_Time.py:144-154`, `main.py:258-261`); `objective_mode = "relocations"` can only be set programmatically | equal | unified 2D setting |
| Output: best solution found | Alg. 2 | `S_best` decoded and evaluated with the shared evaluator; since `24b5fc1` also returned by `get_best_solution()` (`algorithm.py:497-504`) | equal | — |

Verdict: equal / deviation / not specified in paper.
Category: unified 2D setting / choice left open by paper / differs from paper (bug).

## 3. Deviations and decisions

### 3.1 Unified 2D setting

**The paper's crane-time setting is the unified setting.** O_f and O_v (eqs. 40–43) are the Schwarze & Voß cost functions, and the parameters on p. 88 are the Voß & Schwarze values, so O_f,30 = f2, O_f,5 = f2 with tpp = 5, and O_v = f2vert. The code uses the shared evaluator for both (`algorithm.py:352-358`, `:387-393`). The instances are the Caserta instances with Hmax = T + 2, as in CRISP. For Table 5, "the paper's own settings" and the crane-time-only setting therefore coincide.

**Relocation objective.** Table 1 uses the number of relocations. CRP-Time allows only crane time in the UI and the CLI, so the relocation objective can only be selected by setting `objective_mode = "relocations"` in the problem configuration from a script. The search then counts relocations: `movement_objective_cost` gives 1 per relocation and 0 per retrieval.

**Crane start position.** `crane_pos = (1, 1)` (`algorithm.py:247`) is only read by the `rmgc_current` model; f2 and f2vert do not depend on it.

### 3.2 Choices left open by the paper

**Ties.** In the greedy (`scoring.py:156`) and in the argmax of the transition rule (`algorithm.py:299`), the first stack in index order wins.

**n beyond MaxMoves.** A container moved more than MaxMoves times uses n = MaxMoves: in the ants (`algorithm.py:273`) and, since `802e4d1`, in the greedy start (`scoring.py:169-171`). M keeps the true count in both.

**Greedy start without a free stack.** Alg. 1 (p. 80) has no case for an empty R_c (eq. 1), and without S_g the initial pheromone (eq. 37) is undefined. Decision (Thom): stop with an error (`scoring.py:160-166`, since `0bcfd7d`). An ant in the same situation is marked invalid (`algorithm.py:304-306`).

**τmin** is applied as a floor in both updates (`algorithm.py:421-424`, `:457-460`). **Reinitialisation** uses val(S_best) / W (`algorithm.py:437-440`). **Initial S_best** is the greedy solution (`algorithm.py:181`). **New best** requires a strictly lower cost (`algorithm.py:427`).

**Clamp on val.** val = 1 / max(O − LB + 1, 1) (`algorithm.py:218`, `:437`, `:444`). With a valid lower bound O − LB + 1 ≥ 1 and the clamp never binds. With LB_f taken literally from eq. 44 it can bind (§3.6); the paper does not say what val is then. Decision (Thom): keep the clamp.

**Iterations for crane time.** Sec. 7.4 says the number of iterations for the crane-time objectives "was around 5 times higher" than for the relocation count (p. 88), without a stopping value. The default stays 5000 for every objective; this choice will be put to Wei. Since `b379031` the help text paraphrases Sec. 7.4 instead of suggesting "10 000+", which was not from the paper.
- O_v: with 5000 iterations the code already does better than the paper (§4), so fewer iterations cannot explain that difference.
- O_f: with literal eq. 44 and 5000 iterations the code does worse than the paper (§4, step 1b). With 25 000 iterations (O_f,5, seed 0, about 5.2× the running time) the gap closes on 4 × 7 (+3.5 against E_ACO 3.5) and mostly on 3 × 8 (+2.2 against 1.9). The condition fixed before that run is not met on 3 × 8 (seed spread not measured), and 3 × 8 discriminates poorly: no version of the code reaches E_ACO there at 5000 iterations. A full explanation by the number of iterations is not shown.

**Runs and seeds.** One run per instance with `np.random.seed(cfg.seed)` (`algorithm.py:154`); the paper does not say how many runs Tables 1 and 5 average over.

**Tier numbering.** The code uses the Schwarze & Voß convention (ground tier 1, h_max = Hmax + 1), which §1 infers from the matching optimal values.

### 3.3 Differs from the paper

**No lower bound in val and in the early abort.** The paper's quality function subtracts a lower bound of the initial bay: the Zhu et al. (2012) bound for the relocation objective (eq. 33), LB_f or LB_v for crane time (eqs. 44–47). The early abort compares the partial cost plus the lower bound of the current bay with the best cost (Alg. 2; p. 85). The code sets the bound to 0 in both places (`algorithm.py:186` at `42a27fa`, comment "safe for every selectable objective"), so val = 1 / (O + 1) and an ant is aborted only when its cost so far reaches the best cost (`algorithm.py:350-352` at `42a27fa`). The CRP-R version of this algorithm (`algorithms/CRP_R/heuristic/jovanovic_2019_aco/`) does subtract a bound, the non-well-located count.

Repair: step 1a (`f7a9c9a`) puts LB_f / LB_v of the initial bay into val. Step 1b (`c87b7d5`) adds LB(Bay) of the current bay to the early abort. For the relocation objective LB stays 0. Effect: §4, "Repair runs".

Effect, measured on data4-5-1 under f2 (scratchpad computation): LB_f = 1087.2, the greedy solution costs 1281.6 and the best solution of a 5000-iteration run 1262.4.
- Pheromone scale: with LB_f, Δτ / τ0 for that best solution is 5.54; with LB = 0 it is 5.08. On this instance the missing bound changes the ratio little.
- Early abort: LB_f is 86 % of the best cost. With the bound, an ant is aborted as soon as its cost so far plus the retrieval and relocation cost still ahead exceeds the best; without it, only once its cost so far alone does, which is near the end of the solution. The paper names early abort mainly as a way of "not wasting the information stored in the pheromone matrix" (p. 84): every tuple of the partial solution gets the local update, so ants that run on longer decay more pheromone entries.

Where the check sits is as in the paper: Alg. 2 checks only after a relocation, and so does the code. A check after a retrieval would change nothing, because a retrieval adds to O exactly the term it removes from LB(Bay).

**The heuristic uses dd* instead of dd for empty stacks.** Eq. 27 evaluates the heuristic with dd(S), which is N + 1 for every empty stack; dd*(S) = N + i(S) is only the pheromone index (eq. 29). The code uses dd* in both (`algorithm.py:278-282` at `42a27fa`), so an empty stack gets f = 1 / (1 + N + i(S) − c) instead of 1 / (N + 2 − c): the higher its index, the less attractive it looks. Under uniform pheromone the argmax is the same, because a well-located stack still beats any empty stack, an empty stack still beats any stack where c is not well-located, and among empty stacks the lowest index wins in both cases. The difference shows in the roulette choice and in the argmax once τ differs between stacks. Example: N = 35, c = 10, empty stack i = 7: f = 1/33 in the code against 1/27 in the paper.

Repair: step 2 (`4cb92dd`) evaluates the heuristic with dd(S); the pheromone index and the tuple in S keep dd*(S). Effect: §4, step 2.

### 3.4 Suspected bugs

**J1 — the greedy start retrieves containers out of order when there is no room.** If no destination stack exists, `run_greedy_rbrp_time` breaks out of the relocation loop without retrieving the target (`scoring.py:255-256` at `42a27fa`) and continues with the next target, which can then be retrieved before the blocked one. Scratchpad check on stacks `[[1, 2], [3, 4]]` with H = 2 (target 1 blocked, the only other stack full): the greedy returns `[(4, 1, 0, 3)]`, a relocation of container 4 during target 3, and decoding it with `_solution_to_plan` raises `KeyError: None`. This requires an empty candidate list, which cannot occur on Caserta (`azari_2017.md` §3.2), so the bug is latent there. The ACO ants handle the same case by marking the solution invalid (`algorithm.py:304-306`).

Step 3 found a second consequence: on a random UI layout (1 bay, 2 rows, 2 tiers, 4 containers, seed 4; stacks `[[1, 4], [2, 3]]`) the greedy skips both blocked targets without a relocation, the decoder builds a plan that retrieves container 1 from under container 4, and the run ends normally with objective 134.40. Fixed in `0bcfd7d`: the greedy stops with a `ValueError` (§3.2).

**J2 — the greedy start does not cap n at MaxMoves.** The ants record n = min(M[c], MaxMoves) (`algorithm.py:273`), but the greedy recorded M[c] itself (`scoring.py:310` at `4cb92dd`). If the greedy solution is still S_best at a global update and has n > MaxMoves, the update indexes past the pheromone matrix (`algorithm.py:456-457`). On Caserta the greedy's largest n is 5 (MaxMoves = 10), so the bug is latent there. Scratch check with `max_moves = 1` on data4-4-7 at `4cb92dd`: `IndexError: index 2 is out of bounds for axis 2 with size 2`. Fixed in `802e4d1`: the greedy tuple uses min(n, MaxMoves) and M keeps the true count, as for the ants.

**J3 — `get_best_solution()` always returns None.** Every progress record goes through `BaseAlgorithm._push`, which lowers `_best_metric` to the pushed metric (`core/base_algorithm.py:154-155`), and the progress pushes carry `best_cost` (`algorithm.py:474-477`). At the end, `_best_solution` is set only if `best_cost < self._best_metric` (`algorithm.py:505` at `802e4d1`), which is then never true, so `_best_solution` stays None. Seen in all 336 runs of the step 3 bit-identical comparison. Nothing in the platform calls `get_best_solution()` (only the docstring in `core/base_algorithm.py:79` names it), so the bug is latent. Fixed in `24b5fc1`: the final check uses `<=` (`algorithm.py:497`).

### 3.5 Notes for the standardisation (not deviations)

- The class description says the crane-time objective is computed "via platform 2-D KinematicsModel (bay × row, gantry + trolley)" (`algorithm.py:98-105` at `42a27fa`). The search uses the selected f2 or f2vert through the shared evaluator; the kinematics model is only read by `rmgc_current`. Corrected in `b379031` (`algorithm.py:107-114`).
- Dead code from the legacy RMGC model: the greedy accumulates a crane time with inline gantry/trolley kinematics (at `42a27fa`: `scoring.py:94-170`, `:263-277`) that is discarded (`_legacy_greedy_cost`, `algorithm.py:179-182` at `42a27fa`), and the four kinematics parameters are read only for that (`algorithm.py:159-163` at `42a27fa`). `compute_lb_time` (`scoring.py:177-193` at `42a27fa`) and `lb_init_nwl` (`algorithm.py:185` at `42a27fa`) are unused. Removed in `4c59f06`; the cost of S_g for eq. 37 still comes from the shared evaluator (`algorithm.py:209-218`).
- Intermediate progress records carry `lower_bound = 0.0` and `iteration` (`algorithm.py:446-449` at `42a27fa`). The `lower_bound` value is not a lower bound, and the metric set differs from the other CRP-Time algorithms. Removed in `b379031`; the records now carry the same metrics as Azari.
- The seed loop and the mean over seeds (`algorithm.py:133`, `:145`, `:515-526`) are left over from multi-seed evaluation; with `n_seeds = 1` the final record describes one plan. Kept: Azari has the same structure; to be decided for the three algorithms together.
- The header docstring lists 3 of the 7 parameters (`algorithm.py:5-7` at `42a27fa`). Lists all seven parameters since `b379031`.
- `fidelity`: `"faithful"` since `6081770` (`algorithm.py:118`); decision (Thom). The label `"adapted"` was set in `308d016` (Wei Liu, 14 Sept 2026, "Sync local platform work including CRP-MO scaffold and comparison metadata"), which added geometry, objectives and fidelity metadata to 83 algorithm files in one commit; no reason for this algorithm is recorded. At that point the description called the method an "rBRP-ACO adapted for CRP-Time" with crane time "via platform 2-D KinematicsModel", and the code had the deviations of §3.3 (no lower bound, dd* in the heuristic) and the bugs J1–J3. After the repair, `faithful` fits the platform's definition, "how close the implementation is to the published method" (`core/algorithm_meta.py:18-19`):
  1. The code follows Alg. 1 and 2 and eqs. 1–3, 26–38 and 44–47, with eq. 44 taken literally; the deviations of §3.3 and the bugs J1–J3 are repaired.
  2. The unified 2D setting is the paper's own crane-time setting: f2 and f2vert are eqs. 40 and 41 in the tier numbering of Sec. 2 (equal on all 47 772 moves checked), and the exact rBRP optima equal the OPT column of Table 5 on all 9 complete sizes (§4, "Checks on the O_v difference").
  3. The choices the paper leaves open are recorded in §3.2.
  4. The only remaining deviation is the missing Zhu et al. (2012) bound for the rBRP. It affects only the relocation objective (Check 1), which CRP-Time does not offer in the UI or the CLI; the crane-time objectives use the paper's bounds.
  5. Table 5 is not reproduced: O_f is worse, largely through the literal eq. 44, and O_v is better, unexplained (§4, step 5). The label concerns the method, not the numbers; both differences are recorded in this log. `adapted` would present the method as a platform adaptation, which it is not.
  6. This matches Azari (`azari_2017.md` §3.5), which is `faithful` with the same kind of remaining differences; `adapted` is used on the platform for replaced methods, such as Lee–Lee's local search instead of a MIP.

  The open questions to Wei do not change the label: the number of iterations is a choice the paper leaves open, and eq. 44 is taken literally. If a valid lower bound (eq. 44 without ts) is chosen instead, the label stays `faithful` and the correction is named in the description, because the method is not replaced (proposed; depends on Wei's answer on eq. 44). With `faithful` the UI shows no fidelity note, so the differences from the paper are only in this log. Checked via the API catalog and the Workbench on the current frontend source (Vite dev server; the local build in `web/frontend/dist` from 3 Sept predates the fidelity labels): the adapted tag and note no longer appear for Jovanović, while Lee–Lee (`adapted`) still shows them (screenshots `web_ui_fidelity_*.png`, local).
- The base parameters `max_iterations` and `report_interval` appear in the UI but are not read; the ACO uses `n_iterations` and pushes every `n_iterations // 50` iterations (`algorithm.py:141`). Same platform point as in `azari_2017.md` §3.5.
- The shared evaluator (`core/objectives.py:356-381`) takes the tiers from the plan and does not check that the plan is feasible; the J1 plan in §3.4 was scored without error. Platform point for Wei.

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

### Check 1 — Table 1: relocations, rBRP, Hmax = T + 2 — **done (baseline; after the repair: Step 5)**

Limitation: for the rBRP, eq. 33 and the early abort of Alg. 2 use the lower bound of Zhu et al. (2012) (p. 84: "In case of the rBRP we use the lower bound proposed by Zhu et al. (2012). In case of the uBRP we use the total number of containers that are not well-located in the bay."). The code has no Zhu bound, so LB = 0 for the relocation objective, both in val (`algorithm.py:188`, `:218`, `:437`, `:444`) and in the early abort (`use_lb` false at `:204`; abort at `:372`); the baseline at `42a27fa` did the same. Decision (Thom): accepted as a limitation. This may cause a difference from Table 1.

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

### Check 2 — Table 5: O_f,30 and O_f,5 — **done (baseline; after the repair: Step 5)**

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

### Check 3 — Table 5: O_v — **done (baseline; after the repair: Step 5)**

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

#### Checks on the O_v difference

Scripts and results: `docs/validation/data/jovanovic_2019_checks2/` (local, not in git).

**Local update on aborted ants — equal.** Alg. 2 and the text on p. 84 apply the local update rule to S also when its construction was aborted. The code (`c87b7d5`) adds the tuple to S before the relocation is applied and before the abort check (`algorithm.py:331`; check `:377-378`), then leaves the relocation loop (`:266-267`), skips the retrieval (`:381-382`) and the remaining targets (`:258-259`), and applies the local update to all of S without a condition on `valid` (`:416-429`). An ant without a destination stack (`:310`) is treated the same way.

**OPT column of Table 5 — equal to the exact rBRP optimum under f2 and f2vert.** Exact optima by A* over bay states (restricted BRP, Hmax = T + 2), with an admissible bound: retrieval of every container from its current position plus, per non-well-located container, tpp (f2) or 2 tr (f2vert). Checked against a full dynamic program on the 80 instances of 3 × 3 and 3 × 4 (largest difference 5·10⁻¹³). Averages over 40 instances:

| T × S | O_f,30: exact | Table 5 | O_f,5: exact | Table 5 | O_v: exact | Table 5 |
|---|---|---|---|---|---|---|
| 3 × 3 | 476.4 | 476.4 | 126.4 | 126.4 | 904.0 | 904.0 |
| 3 × 4 | 633.6 | 633.6 | 178.4 | 178.4 | 1187.0 | 1187.0 |
| 3 × 5 | 789.2 | 789.2 | 236.6 | 236.6 | 1460.8 | 1460.8 |
| 3 × 6 | 968.2 | 968.2 | 303.2 | 303.2 | 1758.4 | 1758.4 |
| 3 × 7 | 1137.0 | 1137.0 | 375.2 | 375.2 | 2043.4 | 2043.4 |
| 4 × 4 | 911.9 | 911.9 | 255.5 | 255.5 | 1928.1 | 1928.1 |
| 4 × 5 | 1170.2 | 1170.2 | 343.0 | 343.0 | 2424.6 | 2424.6 |
| 4 × 6 | 1385.4 | 1385.4 | 428.5 | 428.5 | – | – |
| 5 × 4 | 1230.0 | 1230.0 | 343.5 | 343.5 | – | – |

On every complete size the average optimum equals the OPT column, and no baseline ACO result lies below the exact optimum. 3 × 8 and 4 × 7 were not completed (A* did not finish on all 40 instances; Table 5 reports averages only, so partial rows are not compared).

**Eq. 41 — equal to f2vert.** Eq. 41 implemented literally with the tier numbering of Sec. 2 (p. 79): tier 0 is the ground and containers occupy tiers 1..H (H_S = 0 for an empty stack, eq. 2; H_S < H for a non-full stack, eq. 1). t(c) is the tier of c and H*_S the tier where c lands; hmax = Hmax + 1, hout = 1.5, tr = 7.77, ts = 1.2. On 1550 rBRP plans (the MinMax greedy and 30 random plans for each of 10 instances of 3 × 3, 3 × 5, 3 × 8, 4 × 4 and 4 × 5), eq. 41 and the shared f2vert agree on all 47 772 moves.

Conclusion: where checked, the code follows the paper (local update on aborted ants), the objective is the paper's (eq. 41 = f2vert), and the optimum it is compared with is the same (OPT column = exact optimum under f2vert). Two hypotheses for the O_v difference are refuted: the missing early abort (steps 1a and 1b) and a difference in objective convention. The difference on O_v remains unexplained.

#### Step 2 — dd instead of dd* in the heuristic (`4cb92dd`), O_v, O_f,30 and O_f,5

Only f(c, S) changes (eqs. 26–28): every empty stack now gets dd = N + 1. The pheromone index (eq. 29) and the tuple in S (Alg. 2) keep dd*(S) = N + i(S). A scratch control on an instrumented copy of `4cb92dd` (data3-8-1 O_v, data4-7-1 O_f,5, data4-5-3 O_f,30, data6-6-2 O_f,30; 200 iterations each) confirmed that f uses dd(S) at all 509 340 candidate evaluations (40 655 of them empty stacks) and that the pheromone index and all 110 996 tuples use dd*(S). The greedy start is unchanged (`scoring.py` already used dd).

Fixed before the runs:
- E1: under uniform τ the argmax is unchanged, since by eq. 3 a well-located stack still beats an empty one, an empty stack still beats one where c is not well-located, and a tie between empty stacks goes to the lowest index as before. Only the argmax branch is unchanged: in the roulette branch the probabilities change, so the same random draw can select another stack. The ACO run is therefore not expected to be identical for the same seed; only the greedy start is bit-identical.
- E2: between two empty stacks, only τ decides.
- E3: no measurable effect or a small worsening on O_f and O_v, most likely on wide bays.
- Hypothesis H-dd: dd* favoured empty stacks with a low index, close to the I/O point, which suits crane time. If the O_v error rises measurably after step 2, especially on 3 × 6 to 3 × 8, dd* is a candidate explanation for part of the O_v difference with the paper; at most a part (3 × 8: E_ACO 25.5 against +4.4). If it does not rise measurably, this explanation is refuted too.
- Control group: on sizes where few choices are affected (O_f on 3 × 3, 3 × 4, 4 × 4, 5 × 4; stated in advance as ≤ 1 %, measured 0.3–1.2 %, see the count below), hardly any per-instance differences are expected. Many differences there would point to something other than the intended change.
- Criterion: sign test (two-sided, seed 0, all sizes) and seed ranges (seeds 0–4) per measured size, per objective. An effect is convincing only if the test and the seed ranges agree, or if the same sign recurs on several objectives. A single p < 0.05 on one of the three objectives counts as a weak signal.

Where an effect can occur: share of the choices ants actually make (code `c87b7d5`, one 200-iteration run per instance, 40 instances) with at least two empty stacks among the candidates:

| T × S | O_f,30: choices | ≥ 2 empty | O_f,5: choices | ≥ 2 empty | O_v: choices | ≥ 2 empty |
|---|---|---|---|---|---|---|
| 3 × 3 | 273 870 | 1.0 % | 277 898 | 1.1 % | 399 770 | 4.3 % |
| 3 × 4 | 333 120 | 1.1 % | 326 228 | 1.2 % | 490 945 | 9.9 % |
| 3 × 5 | 311 826 | 4.2 % | 341 322 | 6.1 % | 553 633 | 18.8 % |
| 3 × 6 | 418 048 | 6.9 % | 429 415 | 10.9 % | 667 753 | 22.3 % |
| 3 × 7 | 409 525 | 8.4 % | 409 689 | 11.2 % | 740 968 | 33.5 % |
| 3 × 8 | 506 736 | 10.1 % | 530 335 | 15.2 % | 845 633 | 32.4 % |
| 4 × 4 | 594 401 | 0.3 % | 600 267 | 0.6 % | 806 887 | 3.5 % |
| 4 × 5 | 735 470 | 2.4 % | 687 102 | 2.2 % | 1 016 142 | 8.3 % |
| 4 × 6 | 800 321 | 3.0 % | 804 117 | 3.5 % | – | – |
| 4 × 7 | 890 966 | 5.0 % | 910 992 | 6.6 % | – | – |
| 5 × 4 | 926 428 | 0.3 % | 933 967 | 0.4 % | – | – |

Runs: same jobs as step 1b, plus seeds 1–4 on 3 × 8 (O_v) and on 4 × 7 and 3 × 8 (O_f,5), for both `c87b7d5` and `4cb92dd`. Error of the code (code − OPT), seed 0; per instance against step 1b.

O_v:

| T × S | E_ACO | step 1b | step 2 | per instance: better / equal / worse |
|---|---|---|---|---|
| 3 × 3 | 1.8 | 0.0 | 0.0 | 0 / 40 / 0 |
| 3 × 4 | 6.2 | +0.7 | +0.7 | 0 / 40 / 0 |
| 3 × 5 | 6.8 | +0.4 | +0.4 | 0 / 40 / 0 |
| 3 × 6 | 12.4 | +1.1 | +2.1 | 4 / 33 / 3 |
| 3 × 7 | 18.3 | +3.2 | +3.4 | 2 / 35 / 3 |
| 3 × 8 | 25.5 | +4.4 | +5.6 | 4 / 30 / 6 |
| 4 × 4 | 10.7 | +1.0 | +2.3 | 2 / 34 / 4 |
| 4 × 5 | 20.2 | +4.9 | +4.1 | 2 / 37 / 1 |

O_f,30:

| T × S | E_ACO | step 1b | step 2 | per instance: better / equal / worse |
|---|---|---|---|---|
| 3 × 3 (control) | 0.0 | +0.1 | +0.1 | 0 / 40 / 0 |
| 3 × 4 (control) | 0.1 | +0.3 | +0.3 | 0 / 40 / 0 |
| 3 × 5 | 0.1 | +0.3 | +0.3 | 0 / 40 / 0 |
| 3 × 6 | 0.1 | +0.2 | +0.2 | 1 / 38 / 1 |
| 3 × 7 | 0.0 | +0.8 | +0.7 | 1 / 39 / 0 |
| 3 × 8 | 0.3 | +0.8 | +1.5 | 0 / 35 / 5 |
| 4 × 4 (control) | 0.0 | +0.1 | +0.6 | 0 / 39 / 1 |
| 4 × 5 | 0.0 | +0.8 | +1.2 | 0 / 37 / 3 |
| 4 × 6 | 0.4 | +1.1 | +1.2 | 0 / 39 / 1 |
| 4 × 7 | 0.8 | +2.0 | +1.6 | 6 / 29 / 5 |
| 5 × 4 (control) | 1.7 | +1.5 | +1.9 | 1 / 36 / 3 |

O_f,5:

| T × S | E_ACO | step 1b | step 2 | per instance: better / equal / worse |
|---|---|---|---|---|
| 3 × 3 (control) | 0.0 | +0.1 | +0.1 | 0 / 40 / 0 |
| 3 × 4 (control) | 0.0 | +0.5 | +0.3 | 1 / 39 / 0 |
| 3 × 5 | 0.3 | +0.3 | +0.5 | 1 / 38 / 1 |
| 3 × 6 | 0.8 | +0.9 | +1.0 | 2 / 34 / 4 |
| 3 × 7 | 0.9 | +1.4 | +2.0 | 3 / 32 / 5 |
| 3 × 8 | 1.9 | +3.3 | +3.4 | 9 / 22 / 9 |
| 4 × 4 (control) | 0.2 | +0.2 | +0.2 | 0 / 39 / 1 |
| 4 × 5 | 1.3 | +2.0 | +2.2 | 4 / 33 / 3 |
| 4 × 6 | 2.2 | +2.8 | +2.8 | 5 / 30 / 5 |
| 4 × 7 | 3.5 | +4.9 | +4.3 | 7 / 25 / 8 |
| 5 × 4 (control) | 0.6 | +1.8 | +1.6 | 4 / 36 / 0 |

| Objective | Sign test (better / worse, p) | Seed ranges, step 1b → step 2 |
|---|---|---|
| O_v | 14 / 17, p = 0.72 | 3 × 8: 2365.40–2369.38 → 2366.93–2370.33; 4 × 5: 2428.16–2429.50 → 2428.61–2430.37 |
| O_f,30 | 9 / 19, p = 0.087 | 4 × 5: 1170.90–1171.14 → 1170.90–1171.38 |
| O_f,5 | 36 / 36, p = 1 | 3 × 8: 458.44–458.90 → 458.31–459.38; 4 × 7: 537.40–538.48 → 537.14–538.23 |

Result:
- No measurable effect on any of the three objectives: no sign test reaches p < 0.05, and the seed ranges overlap on every measured size.
- The direction on O_f,30 (9 better, 19 worse, p = 0.087) and slightly on O_v (14 / 17) fits E3 (a small worsening), but stays below the threshold fixed in advance; O_f,5 is neutral (36 / 36).
- H-dd is refuted: the O_v error does not rise measurably, also not on 3 × 6 to 3 × 8, where 22–34 % of the choices are affected. Even a real shift of the order seen on 3 × 8 (the seed ranges move by about 1) would be small against the gap with the paper (E_ACO 25.5 against +4.4), so dd* could explain at most a negligible part.
- Control group: 5 of 160 (O_f,30) and 6 of 160 (O_f,5) instances differ, mostly on 5 × 4, consistent with the expectation of hardly any differences.

Scripts and results: `docs/validation/data/jovanovic_2019_step2/` (local, not in git).

#### Step 3 — J1 and J2 in the greedy start (`0bcfd7d`, `802e4d1`)

Paper basis: Alg. 1 (p. 80) has no case for an empty R_c (eq. 1, p. 80), while Alg. 2 and eq. 37 (p. 84) need S_g; n = 0, …, MaxMoves (p. 83), MaxMoves = 10 (p. 86).

Count before the change (instrumented scratch copy of `4cb92dd`; all 21 sizes × 40 instances; ants: one 200-iteration run per instance under the relocation objective, O_f,30, O_f,5 and O_v):

| T × S | Greedy: starts without a solution | Greedy: largest n | Ants: choices | No candidate | M[c] > MaxMoves | Largest M[c] |
|---|---|---|---|---|---|---|
| 3 × 3 | 0 / 40 | 2 | 1 351 158 | 0 | 0 | 3 |
| 3 × 4 | 0 / 40 | 2 | 1 643 889 | 0 | 0 | 4 |
| 3 × 5 | 0 / 40 | 2 | 1 766 721 | 0 | 0 | 3 |
| 3 × 6 | 0 / 40 | 1 | 2 165 650 | 0 | 0 | 3 |
| 3 × 7 | 0 / 40 | 1 | 2 302 924 | 0 | 0 | 4 |
| 3 × 8 | 0 / 40 | 1 | 2 660 494 | 0 | 0 | 3 |
| 4 × 4 | 0 / 40 | 3 | 2 822 093 | 0 | 0 | 5 |
| 4 × 5 | 0 / 40 | 2 | 3 483 192 | 0 | 0 | 4 |
| 4 × 6 | 0 / 40 | 2 | 3 819 600 | 0 | 0 | 4 |
| 4 × 7 | 0 / 40 | 2 | 4 345 172 | 0 | 0 | 4 |
| 5 × 4 | 0 / 40 | 3 | 4 333 558 | 0 | 0 | 5 |
| 5 × 5 | 0 / 40 | 3 | 5 415 399 | 0 | 0 | 6 |
| 5 × 6 | 0 / 40 | 3 | 6 294 516 | 0 | 0 | 5 |
| 5 × 7 | 0 / 40 | 2 | 6 636 924 | 0 | 0 | 4 |
| 5 × 8 | 0 / 40 | 2 | 7 621 175 | 0 | 0 | 4 |
| 5 × 9 | 0 / 40 | 2 | 8 474 836 | 0 | 0 | 4 |
| 5 × 10 | 0 / 40 | 2 | 9 218 377 | 0 | 0 | 4 |
| 6 × 6 | 0 / 40 | 3 | 9 130 408 | 0 | 0 | 6 |
| 6 × 10 | 0 / 40 | 3 | 13 098 788 | 0 | 0 | 5 |
| 10 × 6 | 0 / 40 | 5 | 25 442 141 | 0 | 0 | 8 |
| 10 × 10 | 0 / 40 | 5 | 37 339 528 | 0 | 0 | 8 |

- Greedy: 0 of 840 starts without a solution, and no tuple with n > MaxMoves (largest n: 5). The greedy is deterministic and does not depend on the objective, so this count is complete.
- Ants: 159 366 543 choices, none without a candidate stack and none with M[c] > MaxMoves (largest M[c]: 8); this holds for 200-iteration runs.

Expectation fixed before the change: neither branch is reached on the benchmark set, so the behaviour there does not change, and with the same seed the results are bit-identical.

| Check | `4cb92dd` | `802e4d1` |
|---|---|---|
| J1, stacks `[[1, 2], [3, 4]]`, H = 2 | `KeyError: None` when decoding | `ValueError` (container 2 above target 1) |
| J1, random layout seed 4, stacks `[[1, 4], [2, 3]]` | no error; infeasible plan, objective 134.40 | `ValueError` (container 4 above target 1) |
| J2, data4-4-7, `max_moves = 1` | `IndexError` in the global update | runs; all greedy tuples n ≤ 1 |

- Web UI: the random layout above, submitted as a job (`POST /api/jobs`), failed within a second with 0 records. The Workbench shows the status FAILED and a "Run error" banner whose first line is the `ValueError` message, followed by the traceback (screenshot `web_ui_j1.png`, local). Since `cc2c7ad`, CRP-Time no longer accepts random layouts, so this reproduction can no longer be run through the UI or the CLI.
- Bit-identical: `4cb92dd` against `802e4d1`, seed 0, 5000 iterations, instances 1 and 2 of all 21 sizes under the four objectives (168 runs per revision). Objective, relocations, f2 and f2vert are identical in all 168. The best solution itself could not be compared: `get_best_solution()` returns None for this algorithm (J3, §3.4).

Result: J1 and J2 are fixed; on the benchmark set the behaviour is unchanged, as expected.

Scripts and results: `docs/validation/data/jovanovic_2019_step3/` (local, not in git).

#### Step 4 — J3 and cleanup (`24b5fc1`, `4c59f06`, `b379031`)

- 4a (`24b5fc1`): J3 fixed; the final check against `_best_metric` uses `<=` (`algorithm.py:497`).
- 4b (`4c59f06`): dead legacy code removed from `scoring.py` and `algorithm.py`: the greedy's own RMGC crane-time accumulation and its four kinematics inputs, `compute_lb_time`, `compute_lb`, `lb_init_nwl`. A repo-wide search (including `web/` and `tests/`) found no other use; `compute_lb` and `_TRUCK_POS` also exist in the CRP-R/CRP-U versions and in `core/objectives.py` as separate definitions, which are untouched. `compute_lb_time` (n_nwl · spreader_s, unused) is not the `lb_time` of step 1a. The cost of S_g for eq. 37 comes from the shared evaluator (`algorithm.py:209-218`), not from the greedy's own time.
- 4c (`b379031`): description, header docstring (all seven parameters), `n_iterations` help (paraphrase of Sec. 7.4), and the progress records without `lower_bound` and `iteration`. Neither key is read by the frontend or the backend.

Expectation fixed before the changes: no change in behaviour; bit-identical results.

- `802e4d1` → `24b5fc1`, seed 0, 5000 iterations, instances 1 and 2 of all 21 sizes under the four objectives (168 runs per revision): objective, relocations, f2 and f2vert identical in all 168; `get_best_solution()` returns a solution in 168 of 168 (802e4d1: in none).
- `24b5fc1` → `b379031`, same subset: identical in objective, relocations, f2, f2vert and best solution in all 168.
- Feasibility (scratch `jov_feasibility.py`): all 336 stored solutions, replayed as rBRP plans, are feasible (only top containers moved, retrievals in due-date order, height ≤ H, bay empty at the end), and their recomputed relocations, f2 and f2vert equal the reported values (largest deviation 0).
- API: the catalog shows the new description; the configuration schema equals that of `802e4d1` except `n_iterations.help`.
- Web UI: a default random CRP-Time layout (f2) completed in 3.1 s with 52 records, none carrying `lower_bound` or `iteration`; the Workbench shows COMPLETED, 52 updates and the convergence chart, with no error banner and no JavaScript exceptions (screenshot `web_ui_4c.png`, local). Caserta instances cannot be run for CRP-Time through the UI (platform point), so data3-5-1 was not used. Update 25 Sept: Caserta is now enabled for CRP-Time (`06e2436`). A web UI job on data3-5-1 (f2) completes with 52 records, f2 757.20, f2vert 1428.675 and 6 relocations, with no error banner or JavaScript errors (screenshot `ui_jovanovic.png`, local). `main.py layout-run` on the same instance gives the same values (since `eb1f1e1`; before that fix the CLI timed out on this run because it joined the subprocess before draining the queue).

Result: J3 is fixed; the cleanup does not change the behaviour.

Scripts and results: `docs/validation/data/jovanovic_2019_step4/` (local, not in git).

#### Step 5 — full run after the repair (`b379031`)

Same jobs, settings and driver as the baseline: 2520 runs, 5000 iterations, eq. 44 taken literally, LB = 0 for the relocation objective (Check 1); 74.1 min (baseline 82.5 min). The runner also stores `get_best_solution()` per run.

Expectation fixed before the runs: O_f,30 and O_f,5 worse than the baseline because of the literal eq. 44 (step 1b); O_v without a measurable difference (steps 1a, 1b, 2); relocations about equal. Criterion as in step 2: sign test (two-sided, seed 0, all sizes) and seed ranges where measured.

Feasibility: all 2520 stored solutions, replayed as rBRP plans, are feasible, and their recomputed relocations, f2, f2vert and objective equal the reported values (largest deviation 0). Reproducibility: the 1520 runs for O_f,30, O_f,5 and O_v are identical to those of step 2 (`4cb92dd`), as expected after the bit-identical steps 3 and 4.

Relocations (Table 1), seed 0; per instance against the baseline:

| T × S | ACO (Table 1) | baseline − ACO | step 5 − ACO | better / equal / worse |
|---|---|---|---|---|
| 3 × 3 | 5.00 | 0.00 | 0.00 | 0 / 40 / 0 |
| 3 × 4 | 6.18 | 0.00 | 0.00 | 0 / 40 / 0 |
| 3 × 5 | 7.02 | +0.01 | +0.01 | 0 / 40 / 0 |
| 3 × 6 | 8.40 | 0.00 | 0.00 | 0 / 40 / 0 |
| 3 × 7 | 9.28 | 0.00 | 0.00 | 0 / 40 / 0 |
| 3 × 8 | 10.65 | 0.00 | 0.00 | 0 / 40 / 0 |
| 4 × 4 | 10.20 | 0.00 | 0.00 | 0 / 40 / 0 |
| 4 × 5 | 12.95 | 0.00 | 0.00 | 0 / 40 / 0 |
| 4 × 6 | 14.02 | +0.01 | +0.01 | 0 / 40 / 0 |
| 4 × 7 | 16.12 | +0.01 | +0.01 | 0 / 40 / 0 |
| 5 × 4 | 15.42 | +0.03 | +0.03 | 0 / 40 / 0 |
| 5 × 5 | 18.95 | +0.03 | 0.00 | 1 / 39 / 0 |
| 5 × 6 | 22.15 | 0.00 | 0.00 | 0 / 40 / 0 |
| 5 × 7 | 24.33 | −0.03 | 0.00 | 0 / 39 / 1 |
| 5 × 8 | 27.73 | +0.05 | +0.05 | 0 / 40 / 0 |
| 5 × 9 | 30.50 | −0.02 | −0.02 | 0 / 40 / 0 |
| 5 × 10 | 33.40 | −0.05 | −0.05 | 0 / 40 / 0 |
| 6 × 6 | 31.05 | +0.10 | +0.03 | 2 / 37 / 1 |
| 6 × 10 | 45.93 | +0.02 | +0.10 | 2 / 34 / 4 |
| 10 × 6 | 79.50 | +0.28 | +0.55 | 14 / 11 / 15 |
| 10 × 10 | 113.45 | +1.05 | +0.58 | 20 / 3 / 17 |

O_f,30 (Table 5), seed 0:

| T × S | E_ACO | baseline − OPT | step 5 − OPT | better / equal / worse |
|---|---|---|---|---|
| 3 × 3 | 0.0 | 0.0 | +0.1 | 0 / 39 / 1 |
| 3 × 4 | 0.1 | +0.1 | +0.3 | 0 / 38 / 2 |
| 3 × 5 | 0.1 | 0.0 | +0.3 | 0 / 37 / 3 |
| 3 × 6 | 0.1 | 0.0 | +0.2 | 0 / 38 / 2 |
| 3 × 7 | 0.0 | +0.1 | +0.7 | 1 / 33 / 6 |
| 3 × 8 | 0.3 | 0.0 | +1.5 | 0 / 31 / 9 |
| 4 × 4 | 0.0 | 0.0 | +0.6 | 0 / 38 / 2 |
| 4 × 5 | 0.0 | 0.0 | +1.2 | 0 / 30 / 10 |
| 4 × 6 | 0.4 | +0.6 | +1.2 | 2 / 31 / 7 |
| 4 × 7 | 0.8 | +0.1 | +1.6 | 1 / 28 / 11 |
| 5 × 4 | 1.7 | +0.9 | +1.9 | 1 / 32 / 7 |

O_f,5 (Table 5), seed 0:

| T × S | E_ACO | baseline − OPT | step 5 − OPT | better / equal / worse |
|---|---|---|---|---|
| 3 × 3 | 0.0 | 0.0 | +0.1 | 0 / 39 / 1 |
| 3 × 4 | 0.0 | +0.1 | +0.3 | 1 / 36 / 3 |
| 3 × 5 | 0.3 | 0.0 | +0.5 | 0 / 35 / 5 |
| 3 × 6 | 0.8 | +0.5 | +1.0 | 1 / 33 / 6 |
| 3 × 7 | 0.9 | +0.3 | +2.0 | 1 / 29 / 10 |
| 3 × 8 | 1.9 | +2.6 | +3.4 | 5 / 22 / 13 |
| 4 × 4 | 0.2 | +0.1 | +0.2 | 0 / 39 / 1 |
| 4 × 5 | 1.3 | +1.5 | +2.2 | 3 / 29 / 8 |
| 4 × 6 | 2.2 | +1.0 | +2.8 | 2 / 24 / 14 |
| 4 × 7 | 3.5 | +4.8 | +4.3 | 8 / 22 / 10 |
| 5 × 4 | 0.6 | +0.8 | +1.6 | 1 / 31 / 8 |

O_v (Table 5), seed 0:

| T × S | E_ACO | baseline − OPT | step 5 − OPT | better / equal / worse |
|---|---|---|---|---|
| 3 × 3 | 1.8 | 0.0 | 0.0 | 0 / 40 / 0 |
| 3 × 4 | 6.2 | +0.7 | +0.7 | 0 / 40 / 0 |
| 3 × 5 | 6.8 | +0.4 | +0.4 | 0 / 40 / 0 |
| 3 × 6 | 12.4 | +1.8 | +2.1 | 4 / 33 / 3 |
| 3 × 7 | 18.3 | +2.3 | +3.4 | 1 / 35 / 4 |
| 3 × 8 | 25.5 | +6.8 | +5.6 | 4 / 31 / 5 |
| 4 × 4 | 10.7 | +1.8 | +2.3 | 0 / 38 / 2 |
| 4 × 5 | 20.2 | +4.6 | +4.1 | 4 / 33 / 3 |

| Objective | Sign test vs baseline (better / worse, p) | Seed ranges, baseline → step 5 | Expected | Outcome |
|---|---|---|---|---|
| Relocations | 39 / 38, p = 1 | 5 × 7: 24.30–24.38 → 24.30–24.40 | about equal | as expected |
| O_f,30 | 5 / 60, p = 4.9·10⁻¹³ | 4 × 5: 1170.18–1170.90 → 1170.90–1171.38 | worse | as expected |
| O_f,5 | 22 / 79, p = 1.0·10⁻⁸ | — | worse | as expected |
| O_v | 13 / 17, p = 0.59 | 4 × 5: 2428.55–2429.80 → 2428.61–2430.37 | no measurable difference | as expected |

Mean running time per run, baseline → step 5: relocations 17.4 → 17.8 s, O_f,30 8.3 → 5.2 s, O_f,5 8.0 → 4.8 s, O_v 7.9 → 7.6 s.

Result: every objective behaves as fixed in advance, so there is no new deviation to investigate. Against the paper: relocations within ±0.10 of Table 1 on 19 of 21 sizes (10 × 6 +0.55, 10 × 10 +0.58); O_f above E_ACO on 11 (O_f,30) and 10 (O_f,5) of 11 sizes, largely through the literal eq. 44 (§3.6, step 1b); O_v far below E_ACO on all sizes, unexplained ("Checks on the O_v difference").

Scripts and results: `docs/validation/data/jovanovic_2019_step5/` (local, not in git).
