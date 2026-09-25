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

| Step | Paper (p.) | Code (file:line) | Verdict | Category |
|---|---|---|---|---|
| | | | | |

Verdict: equal / deviation / not specified in paper.
Category: unified 2D setting / choice left open by paper / differs from paper (bug).

## 3. Deviations and decisions

## 4. Behavioural checks
- [ ] Original objective: average relocations on 5 × 7 (40 instances, rBRP, Hmax = T + 2) compared with Table 1 (ACO 24.33)
- [ ] Crane time: average Of,30 on one size from Table 5 (e.g. 4 × 5: OPT 1170.2, E_ACO 0.0) compared with OPT + E_ACO
- [ ] Crane time: average Ov on one size from Table 5 (e.g. 4 × 5: OPT 2424.6, E_ACO 20.2) compared with OPT + E_ACO