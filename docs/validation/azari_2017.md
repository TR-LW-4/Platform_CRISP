# Azari, Eskandari & Nourmohammadi (2017) — validation log

Paper: E. Azari, H. Eskandari, A. Nourmohammadi, "Decreasing the crane working time in retrieving the containers from a bay", *Scientia Iranica, Transactions E: Industrial Engineering* 24(1) (2017) 309–318. Page numbers below are journal pages.

Code: `algorithms/CRP_Time/heuristic/azari_2017/`

## 1. Specification from the paper

### Problem setting (Sec. 2–3, p. 311)
- A single bay with W stacks and H tiers per stack; N is the number of containers in the initial configuration (p. 311).
- All containers have the same size. Each container has a number that gives its retrieval priority; a lower number means a higher priority (p. 311).
- One RMGC moves one container at a time and can only access containers on top of a stack (p. 311).
- The bay is one-sided: the truck lane lies next to stack 1 (Fig. 1, p. 311). At the start, the crane is above the truck lane (p. 311).
- Assumptions (p. 311): only top containers are accessible; only the container with the highest priority can be retrieved; relocations stay within the bay; stack capacity may not be exceeded; no containers enter the bay during retrieval; one container at a time.
- Present container (PC): the container with the highest priority; present stack (PS): the stack that holds it (p. 311).
- **Relocation rule: restricted.** Good containers are never relocated (p. 312, below Def. 2), relocations only originate from the PS (p. 312, end of Sec. 4.2), and the container relocated is always q, the top container of the PS (Table 1, p. 312; p. 313). Only containers above the PC are therefore moved.

### Objective (p. 309–311)
- Minimise the total working time of the crane, T_CW (p. 311). The number of movements (NMOV) is reported as a secondary measure (Tables 2–4).

### Time model (p. 311)
- The paper follows the procedure of Lee & Lee (2010) [6]: a relocation from s1 to s2 consists of a trolley movement to s1, a spreader movement to pick up the container, and the same movements to s2. Times are deterministic and proportional to the distance the crane moves (p. 311).
- The paper gives no formulas and no parameter values. For the Lee & Lee model, see `lee_lee_2010.md`.

### Notation (Table 1, p. 312)
- PS, PC; w = stack number of the PS; q = container on top of the PS; S = all non-full stacks except the PS; Nm = current number of movements; LB = lower bound on the number of movements to empty the current bay; UB = Nm + LB; tcw = current crane working time; Tbest = crane working time of the best solution found; t_rem = time to remove the PC; t_rel = time to relocate q to its destination.

### Definitions (Sec. 3, p. 311–312)
- Good / bad container (Def. 1, p. 311): a container is bad if it blocks a container with a higher priority (a lower number); otherwise it is good.
- BB / BG relocation (Def. 2, p. 312): a Bad-Bad (BB) relocation moves a bad container to a stack where it is still bad; a Bad-Good (BG) relocation moves a bad container to a stack where it becomes good.
- Best BB / best BG stack (Def. 3, p. 312): the best BB stack for q is the stack s ∈ S where relocating q is a BB relocation and whose lowest number is the highest. The best BG stack for q is the stack s ∈ S where relocating q is a BG relocation and whose lowest number is the lowest.
- The lowest number of an empty stack is N + 1 (p. 312).
- If more than one empty stack qualifies, the one with the lowest stack number is chosen (p. 312).
- Worked example (Fig. 1, p. 311–312): layout bottom→top: stack 1 = 12, 13; stack 2 = 3, 6, 5; stack 3 = 9, 2, 8, 1; stack 4 = 10, 7, 11, 4. After retrieving container 1, the PC is 2 in stack 3; q = 8 and S = {1, 2, 4}. Best BB stack: 4. Best BG stack: 1.

### Lower bound (Sec. 4.1, p. 312)
- Lower bound on the number of movements (relocations plus retrievals) needed to empty the current bay: 2x + y, with x the number of bad containers and y the number of good containers (Def. 1). Taken from Kim & Hong (2006).
- A bad container blocks a container with a higher priority anywhere below it, not only directly below it.

### GBH — Good-Bad Heuristic (Sec. 4.2, p. 312–313, pseudocode Fig. 2)
- Minimises the number of movements (NMOV). Recursive depth-first search with backtracking, at most two branches per node.
- Initialisation: UB = H × N, Nm = 0 (p. 313).
- Each call first checks the time limit; if it is reached, the search stops and the best solution found so far is reported (p. 313).
- PC not blocked and the last container: retrieve it, set UB = (length of the current solution) − 1, store the solution as the best so far, and go back one level to search other branches (p. 313).
- PC not blocked and not the last container: retrieve it, call GBH recursively, then undo the retrieval (p. 313).
- PC blocked: q is the top container of the PS. At most two destination stacks are tried, in this order (p. 313):
  - if a BG relocation exists for q in S: first destination stack (FDS) = best BG stack, second destination stack (SDS) = best BB stack;
  - otherwise: FDS = best BB stack, SDS = best BB stack in S \ {FDS}.
- If neither FDS nor SDS exists, or the time limit is reached, the search stops (p. 313).
- For each of FDS and SDS: move q there. If Nm + LB ≤ UB, continue by calling GBH recursively; otherwise prune the branch. In both cases the move is undone afterwards (p. 313).

### CSUM — Constant Summation (Sec. 4.3, p. 313–315, pseudocode Fig. 3)
- First solve the instance with GBH under a time limit T; its solution is the initial solution (p. 313).
- Split the containers (Fig. 3, p. 314): a container goes to set A if it has at least one BB relocation in the initial solution, otherwise to set B. The text (p. 313) phrases this as "relocated more than once in the initial solution". Under the restricted rule these are equivalent: after a BG relocation a container is good and is never moved again, so it is relocated more than once only if at least one of its relocations was BB.
- Initialisation (Fig. 3, p. 314): Nm = 0; NMOV = number of movements in the initial solution; tcw = 0; Tbest = ∞.
- Each call first checks the time limit (Fig. 3, p. 314).
- PC not blocked (Fig. 3, p. 314): retrieve the PC, add t_rem to tcw, Nm + 1.
  - If it was the last container and tcw ≤ Tbest: set NMOV = Nm and Tbest = tcw, and store the solution.
  - Otherwise call CSUM recursively.
  - Afterwards undo the retrieval: tcw − t_rem, Nm − 1.
- PC blocked, q = top container of the PS, Q = empty ordered list of destination stacks (Fig. 3, p. 314):
  - q ∈ A: same branching as GBH. If a BG relocation exists for q: add the best BG stack, then the best BB stack to Q. Otherwise: add the best and second-best BB stacks to Q.
  - q ∈ B:
    - G1 = stacks with a lower stack number than the PS (between the truck lane and the PS); G2 = stacks with a higher stack number than the PS.
    - s = best BG stack for q in G1. If there is none, bound = ∞; otherwise bound = lowest number in s, and s is added to Q.
    - For each stack s ∈ G2 in increasing order: if relocating q to s is a BG relocation and m, the lowest number in s, is lower than bound, add s to the bottom of Q and update bound (see "Not specified").
- The time limit is checked again, then for each stack s in Q, in order (Fig. 3, p. 314; p. 315):
  - add t_rel (the time to relocate q to s) to tcw, and Nm + 1;
  - if Nm ≤ NMOV: move q to s, call CSUM recursively, and undo the move;
  - undo the bookkeeping: tcw − t_rel, Nm − 1.
- Consequence: branches are pruned on the **number of movements**, not on time. NMOV is the move count of the best solution so far, so CSUM only looks for time improvements at an equal or smaller number of movements. A solution with more movements but a lower crane time is never explored.
- Worked example (Fig. 4, p. 315): q = 10, W = 8, w = 3; lowest numbers per stack: 14, 13, PS, 12, 9, 15, 11, N + 1 (stack 8 is empty). BG stacks for q: 1, 2, 4, 6, 7, 8 (stack 5 is BB). G1 = {1, 2}: s = 2, bound = 13, Q = [2]. G2: stack 4 (12 < 13) added, bound = 12; stack 5 skipped (BB); stack 6 (15) skipped; stack 7 (11 < 12) added, bound = 11; stack 8 (N + 1) skipped. Result: Q = [2, 4, 7], as stated in the paper.

### Parameters
- Initial upper bound for GBH: UB = H × N (p. 313).
- Tie-breaking among empty stacks: lowest stack number (p. 312).
- Time limits (p. 316):
  - Table 2 instances: "the allotted time to address each category of the instances was one second".
  - Lee & Lee instances: 3 s for GBH to find the initial solution, then at most 5 s for CSUM.
- No randomness: the algorithm is deterministic. Because the search is cut off by wall-clock time limits, results depend on the speed of the machine.

### Not specified in the paper
- The time model: no formulas and no parameter values (p. 311). The T_CW values in Table 2 are around 300, those in Table 3 around 10,000 s (p. 315–316). This suggests that each comparison uses the time model of its own source paper (Ünlüyurt & Aydın 2012 for Table 2, Lee & Lee 2010 for Table 3); this is inferred, not stated.
- Def. 3 (p. 312) literally compares the lowest number of s with the lowest numbers in "all other stacks of the set S". Read literally, this cannot always hold: the stack with the highest minimum in S may be a BG stack, and the stack with the lowest minimum may be a BB stack. The worked example (Fig. 1) shows that the comparison is made only among the stacks where the relocation is BB (for the best BB stack) or BG (for the best BG stack). This reading is inferred from the example.
- S: Table 1 (p. 312) defines it as all non-full stacks except the PS; p. 313 says all stacks except the PS. Full stacks cannot receive a container, so this is taken to mean the Table 1 definition.
- Tie-breaking for non-empty stacks with the same lowest number: not specified.
- CSUM, set B, update of bound: Fig. 3 (p. 314) literally says `m := bound`, which leaves bound unchanged; `bound := m` is almost certainly intended. The Fig. 4 example gives Q = [2, 4, 7] under both readings, so it does not settle this.
- CSUM, set B, selection condition: the text (p. 314–315) says stack m must have a lower minimum than every earlier stack n. Read literally, this includes BB stacks (stack 5, minimum 9, in Fig. 4), and then stack 7 (minimum 11) would not be selected. The example therefore implies that only BG stacks are compared.
- CSUM, set B: if no BG stack exists in G1 or G2, Q is empty and the branch has no children. The paper does not mention a BB fallback for set-B containers.
- CSUM pruning: the text (p. 315) says Nm ≤ UB, the pseudocode (Fig. 3) says Nm ≤ NMOV. CSUM has no UB of its own, so the pseudocode is taken as intended.
- CSUM accepts a new best solution when tcw ≤ Tbest (Fig. 3), so a later solution with equal time replaces the earlier one.
- Whether the one-second limit for Table 2 applies per instance or per category of 40 instances (p. 316).

### Validation material in the paper
- Worked examples: Fig. 1 with Def. 3 (best BB stack 4, best BG stack 1; p. 311–312) and Fig. 4 (Q = [2, 4, 7]; p. 315). Both can be used as unit tests without the original instances.
- Table 2 (p. 315): average NMOV and T_CW of CSUM per instance type (40 instances each) on the Ünlüyurt & Aydın (2012) set of 640 instances, "accessible upon request" (p. 315). Also reports the optimality gap to their B&B.
- Table 3 (p. 316): NMOV and T_CW (s) of CSUM on the 14 Lee & Lee (2010) instances. The download link given (p. 315) is the one from Lee & Lee; that set is no longer available online.
- Table 4 (p. 316): comparison with Forster & Bortfeldt (2012) on the Lee & Lee instances.
- None of the paper's instances are Caserta instances. A check under the paper's own settings requires the Ünlüyurt & Aydın set and their time model.

## 2. Comparison with the code

Points to check in particular:
- Which lower bound is used for pruning: the Kim & Hong bound as defined above, or the shared CRISP bound (which counts only adjacent inversions, see B1 in `lee_lee_2010.md`).
- The set A / set B criterion.
- The set-B update of bound (`m := bound` or `bound := m`) and which stacks are compared.
- Pruning on Nm ≤ NMOV, and NMOV updated on every new best solution.
- Tie-breaking.
- Which time model supplies t_rem and t_rel (f2 / f2vert through the shared evaluator, or something else).
- How the time limits are set and exposed as parameters.
- Which value CRISP uses for H (for Caserta: H′ + 2).

File references without a directory are to `algorithms/CRP_Time/heuristic/azari_2017/algorithm.py`.

| Step | Paper (p.) | Code (file:line) | Verdict | Category |
|---|---|---|---|---|
| Single bay, W stacks, H tiers | 311 | Caserta loader: one stack per bay, `num_rows = 1` (`core/benchmarks/caserta.py:227-228`); stacks read bottom→top (`algorithm.py:516-520`) | equal | — |
| Truck lane next to stack 1 | 311 | truck at stack index 0, retrieval travel grows with the stack index (`core/objectives.py:202-205`, `:261-265`); G1 = stacks with a lower index than the PS (`algorithm.py:264-267`) | equal | — |
| Crane starts above the truck lane | 311 | `crane_pos = (1, 1)` (`algorithm.py:112`, `:297`); only `rmgc_current` reads it; f2/f2vert do not depend on the crane position (`core/objectives.py:303-316`) | deviation | unified 2D setting |
| Priorities unique; lower number = earlier | 311 | `targets = sorted(...)` (`algorithm.py:78-80`); Caserta priorities are a permutation of 1..N (`core/benchmarks/caserta.py:164-180`) | equal | — |
| Only top containers accessible; one move at a time; capacity H | 311 | moves pop/push the top (`algorithm.py:160-198`); destination must satisfy `len(st) < max_tiers` (`:200-204`, `:266`, `:277`) | equal | — |
| Restricted: only q, the top of the PS, is relocated | 312–313 | q = top of the stack that holds the target (`algorithm.py:354`, `:438`) | equal | — |
| Objective: minimise T_CW; NMOV secondary | 309–311 | search minimises f2 or f2vert through `movement_objective_cost` (`algorithm.py:417-419`, `:460-462`); movements reported as `steps` (`:543`) and `total_moves` by the shared evaluator | equal (time model adapted) | unified 2D setting |
| Time model for t_rem and t_rel | 311 (no formulas) | f2: t_rem = 2·ts·s + tpp, t_rel = 2·ts·\|s1 − s2\| + tpp (`core/objectives.py:261-270`); f2vert (`:278-295`); tiers passed as in `annotate_plan_tiers` (`algorithm.py:340`, `:362-363`) | not specified in paper | unified 2D setting |
| Def. 1: good / bad container | 311 | `_is_bg` / `_is_bb` compare q with the stack minimum (`algorithm.py:132-136`); the LB's bad test compares with every container below (`:148-153`) | equal | — |
| Def. 2: BB / BG relocation | 312 | `algorithm.py:132-136` | equal | — |
| Def. 3: best BB / best BG stack, compared only among BB resp. BG stacks | 312 (reading inferred from Fig. 1) | `_best_bb_stack` / `_best_bg_stack` filter on BB/BG first, then take the max/min stack minimum (`algorithm.py:206-226`) | equal | choice left open by paper |
| Lowest number of an empty stack = N + 1 | 312 | `empty_lowest = max(targets) + 1` (`algorithm.py:81`, `:129-130`) | equal | — |
| Several empty stacks: lowest stack number | 312 | BG key `(lowest, index)` (`algorithm.py:215`, `:270`) | equal | — |
| Ties between non-empty stacks | not specified | cannot occur: priorities are unique, and a BB stack is never empty (`algorithm.py:226`, `:248`) | not specified in paper | choice left open by paper |
| S = non-full stacks except the PS | 312 (Table 1) vs 313 | `_true_eligible_dsts` (`algorithm.py:200-204`) | equal (Table 1 reading) | choice left open by paper |
| Lower bound 2x + y (Kim & Hong) | 312 | `_lower_bound_moves_2x_plus_y` (`algorithm.py:144-158`): a container is bad if any container below it has a lower number; this is not the shared B1 bound. Used for pruning in GBH (`:368`) since `6cf6b3c`; the CSUM branch that uses it when `mode == "relocations"` (`:398-403`) is never reached in CRP-Time (`problems/CRP_Time.py:144-154`) | equal | — |
| GBH: recursive DFS with backtracking, UB = H × N, LB pruning, own time limit | 312–313, Fig. 2 | `_run_gbh` / `_gbh_dfs` (`algorithm.py:312-371`): UB starts at H × N (`:316`); each complete solution is stored and sets UB = \|solution\| − 1 (`:346-347`); a relocation is followed only if Nm + LB ≤ UB (`:368`); time checks `:327`, `:356`. Since `6cf6b3c`; before, a single greedy pass (§3.3) | equal | — |
| GBH branching: FDS / SDS | 313 | `_gbh_candidates` (`algorithm.py:228-251`) | equal | — |
| Neither FDS nor SDS exists: "the search stops" | 313 | GBH and CSUM: only that node returns and siblings are still explored (`algorithm.py:355`, `:446-447`) | not specified in paper | choice left open by paper |
| No complete GBH solution within T | not specified | `solve()` raises a `RuntimeError` (`algorithm.py:93-98`) | not specified in paper | choice left open by paper |
| CSUM initial solution: GBH under time limit T | 313 | the best GBH solution found within `gbh_time_limit_s` (`algorithm.py:93-99`) | equal | — |
| Set A: at least one BB relocation / relocated more than once | 313, Fig. 3 | relocated more than once in the initial solution (`algorithm.py:101-104`) | equal | — |
| Initialisation: NMOV = GBH moves, Tbest = ∞ | Fig. 3 | `best_time` starts at the crane time of the GBH solution, which becomes a candidate output (`algorithm.py:99`, `:295-310`); NMOV is not used | deviation | differs from paper |
| Time limit checked on each call and before branching | Fig. 3 | `algorithm.py:382-383`, `:449-450` | equal | — |
| PC not blocked: retrieve, tcw + t_rem, Nm + 1, recurse, undo | Fig. 3 | `algorithm.py:410-436` | equal | — |
| New best when the last container is retrieved and tcw ≤ Tbest | Fig. 3 | only a strictly lower time reaches the leaf (`algorithm.py:385-386`, `:421-422`, `:464`); leaf acceptance `:388-396`. The equal-time, fewer-moves branch (`:389-391`) is unreachable | deviation | differs from paper |
| Pruning on Nm ≤ NMOV; NMOV = Nm at each new best | Fig. 3 (p. 315 says UB) | no movement-count pruning in crane-time mode; a branch is cut once tcw + t ≥ best time (`algorithm.py:385-386`, `:421-422`, `:464`); `best_total_moves` is stored (`:394`) but never used for pruning | deviation | differs from paper |
| q ∈ A: GBH branching | Fig. 3 | `algorithm.py:439-440` | equal | — |
| q ∈ B, G1: best BG stack s, bound = its minimum | Fig. 3 | `algorithm.py:264-272` | equal | — |
| q ∈ B, G2: which stacks are compared, update of bound | Fig. 3, Fig. 4 | in increasing order, stack j is added if it is non-full, the relocation is BG and its minimum is below bound; then bound := that minimum (`algorithm.py:274-282`). Before `ac344eb` every earlier stack was compared (A1, §3.4) | equal | choice left open by paper |
| q ∈ B and Q empty | not specified | falls back to GBH branching (`algorithm.py:441-444`) | not specified in paper | choice left open by paper |
| Cap on the number of set-B branches | not in the paper | `dedup[: max_branches_b]`, default 6, exposed in the UI (`algorithm.py:293`, `:532`, `:590-594`) | deviation | differs from paper |
| For each s in Q, in order: t_rel, move, recurse, undo | Fig. 3 | `algorithm.py:449-475` | equal | — |
| Time limits: 1 s (Table 2); 3 s GBH + 5 s CSUM (Lee & Lee instances) | 316 | since `6cf6b3c` two budgets, `gbh_time_limit_s` (default 3 s) and `csum_time_limit_s` (default 5 s), each clamped to ≥ 0.1 s; each phase sets its own deadline (`algorithm.py:73-74`, `:106`, `:315`, `:530-531`, `:580-589`) | not specified in paper (for Caserta) | choice left open by paper |
| H | 313 (UB = H × N) | `max_tiers` = H′ + 2 for Caserta (`core/benchmarks/caserta.py:154-160`, `:229`); used for S, for h_max in f2vert (`core/objectives.py:280`) and for the initial GBH bound UB = H × N (`algorithm.py:316`) | equal | unified 2D setting |
| Deterministic | 316 | no random choices in the solver; the seed is set on the environment only (`algorithm.py:512`) | equal | — |
| Output: best solution found | 313 | `best_moves`, re-evaluated with the shared evaluator (`algorithm.py:534-542`) | equal | — |

Verdict: equal / deviation / not specified in paper.
Category: unified 2D setting / choice left open by paper / differs from paper (bug).

## 3. Deviations and decisions

### 3.1 Unified 2D setting

**Time model.** The paper gives no formulas and follows Lee & Lee (2010), whose procedure includes moving the trolley to the source stack of each movement. The code takes t_rem and t_rel from Voß & Schwarze f2 or f2vert (`core/objectives.py:261-295`). In f2, a retrieval from stack s costs 2·ts·s + tpp and a relocation from s1 to s2 costs 2·ts·|s1 − s2| + tpp. Each movement's cost is independent of the previous one: there is no repositioning term, as already noted in `lee_lee_2010.md` §3.1. The search logic is unchanged, but the times it compares differ from those in the paper's model.

**Crane start position.** The paper starts the crane above the truck lane. The code starts it at `(1, 1)`, i.e. stack 1 (`algorithm.py:112`, `:297`), but only `rmgc_current` reads the position. Under f2 and f2vert the start position has no effect.

**Geometry.** Azari et al. already work in a single bay with the truck lane next to stack 1. That matches the CRISP convention (truck at stack index 0), so no 3D → 2D conversion is needed. For Caserta, H = H′ + 2 (`core/benchmarks/caserta.py:154-160`). H determines which stacks are full, and f2vert uses it for h_max.

### 3.2 Choices left open by the paper

**Def. 3.** The code compares the best BB stack only with BB stacks and the best BG stack only with BG stacks (`algorithm.py:206-226`). This is the reading inferred from Fig. 1, and Check 1 (§4) passes.

**S** is the Table 1 definition: non-full stacks except the PS (`algorithm.py:200-204`).

**Tie-breaking.** Only empty stacks can tie, since priorities are unique and a BB stack is never empty. Among empty stacks the lowest stack number wins, as the paper prescribes (`algorithm.py:215`, `:270`).

**Set-B container without any BG destination.** The code falls back to GBH branching (`algorithm.py:441-444`), so the container gets the best and second-best BB stacks. The paper does not mention this case. Without the fallback the branch would have no children.

**"If neither FDS nor SDS exists, the search stops."** I am not sure whether the paper means the whole search or only the current branch. Both GBH and CSUM stop only the current branch. On Caserta the case cannot occur: S is empty only if all W − 1 other stacks are full, which requires (W − 1)(H′ + 2) ≤ W·H′ − 1, i.e. H′ ≥ 2W − 1. The Caserta class closest to that is data10-6 (H′ = 10, W = 6), and it does not meet it.

**Time limit.** Since `6cf6b3c`, GBH and CSUM have separate limits, `gbh_time_limit_s` (default 3 s) and `csum_time_limit_s` (default 5 s). These are the values the paper uses for the Lee & Lee instances (p. 316), and the only per-instance limits it states unambiguously; the 1 s for Table 2 may be per instance or per category (§1). Before, one 5 s budget covered both phases. The benchmark duration is decided later, with the design of the experiments. As in the paper, results depend on machine speed.

**No complete GBH solution within T.** The paper does not say what happens then. CSUM needs the GBH solution for NMOV and set A, so `solve()` raises a `RuntimeError` with a clear message instead of continuing (`algorithm.py:93-98`). On Caserta this does not happen: S is never empty (above), and GBH found a complete solution on all 840 instances (§4, Check 5).

### 3.3 Differs from the paper

**GBH was a single greedy pass. Fixed in `6cf6b3c`.** The paper's GBH is a depth-first search with backtracking that minimises the number of movements within time limit T, pruning on Nm + LB ≤ UB. The code used to follow the FDS at every node until all containers were retrieved: the first branch of the paper's GBH, without backtracking. GBH now follows Sec. 4.2 and Fig. 2 (`algorithm.py:312-371`). UB starts at H × N (`:316`). Each complete solution is stored as the best so far and sets UB = |solution| − 1 (`:346-347`). After each relocation, the branch is continued only if Nm + LB ≤ UB, with the Kim & Hong bound (`:368`). The search stops at the GBH time limit. Relocations and movements are used internally here, as in the paper, which is allowed (Decisions from Wei).
- Nm + LB never decreases along a path: every move adds 1 to Nm, while a BG relocation lowers LB by 1, a BB relocation leaves it unchanged, and a retrieval lowers it by 1. Pruning after relocations only, as the paper does, is therefore sufficient, and once a solution with LB₀ movements is found the rest of the tree is pruned at once.
- The first complete solution is the old greedy pass, because UB = H × N never prunes it. GBH therefore never ends with more movements than before. Check 5 (§4) confirms this on all 840 Caserta instances, and confirms on the three smallest classes that the pruning loses nothing compared with the full FDS/SDS tree.
- Set A is now derived from the best GBH solution (`algorithm.py:101-104`) instead of the greedy one.
- The shared B1 bound is not used, so B1 does not affect Azari.

**CSUM prunes on time instead of movement count.** The paper prunes a branch unless Nm ≤ NMOV, where NMOV is the movement count of the best solution so far. CSUM therefore only looks for faster solutions with at most that many movements. The code has no movement-count condition: it cuts a branch as soon as its accumulated time reaches the best time (`algorithm.py:385-386`, `:421-422`, `:464`). Because f2 and f2vert costs are non-negative, this time cut only removes branches that cannot give a strictly lower time. Dropping Nm ≤ NMOV, however, widens the search: the code can return a solution with more movements than the initial solution if it is faster, which the paper's CSUM never does. Within the same time budget, the two searches also spend their time in different parts of the tree. The code does not document whether this change was meant as an adaptation to the crane-time-only setting.

**Tbest starts at the GBH solution, not at ∞.** The GBH solution's crane time becomes the initial best time (`algorithm.py:99`). The code's output is therefore never slower than the GBH solution, whereas in the paper the output is the best CSUM leaf.

**Ties.** Fig. 3 accepts a new best when tcw ≤ Tbest, so a later solution with the same time replaces the earlier one. In the code, a branch that reaches the best time exactly is cut before the leaf, so the first solution with a given time is kept. The code also contains an "equal time, fewer moves" rule (`algorithm.py:304-306`, `:389-391`). It is unreachable at present: `_dfs` never reaches a leaf with equal time, and `_accept_solution_if_better` is called only once, while the best time is still ∞. If it ever became reachable, it would break time ties on the movement count. An algorithm may use relocations or movements internally where its paper does so (Wei's decision, late Sept), but Fig. 3 does not break ties this way: it accepts any solution with tcw ≤ Tbest, so the later one wins. Movements enter CSUM only through the pruning Nm ≤ NMOV.

**Cap on set-B branches.** `max_branches_b` (default 6, exposed in the UI) truncates Q (`algorithm.py:293`). The paper has no such cap. Q holds at most W − 1 stacks, so the cap can only bind for W ≥ 8 (Caserta classes data3-8, data5-8, data5-9, data5-10, data6-10, data10-10).

### 3.4 Suspected bugs

**A1 — set-B branching compared against every earlier stack, not only BG stacks. Fixed in `ac344eb`.** Before the fix (`algorithm.py:269-270` at `af4e547`):

```python
prev = [i for i in range(0, m) if i != src]
if all(lows[m] < lows[i] for i in prev):
```

A G2 stack m was added only if its minimum was lower than the minimum of every earlier stack except the PS. That included BB stacks, full stacks and G1 stacks other than s. This is the literal reading of the text (p. 314–315). As §1 notes, Fig. 4 contradicts it: the example only compares BG stacks, via bound. On Fig. 4, BB stack 5 (minimum 9) blocked stack 7 (minimum 11), and the code returned Q = [2, 4] instead of [2, 4, 7]. In general, any BB or full stack with a low minimum hid every later G2 stack. Set-B containers then got fewer branches than in the paper, and possibly none, in which case the GBH fallback from §3.2 applied.

The fix follows the pseudocode of Fig. 3 (`algorithm.py:264-282`). s is the best BG stack in G1, and bound starts at its minimum, or ∞ if there is none. Each G2 stack, in increasing order, is added to the bottom of Q if the relocation is BG and its minimum m is below bound; then bound := m. Fig. 3 literally reads `m := bound`. Following §1 ("Not specified"), `bound := m` is taken as intended, and a comment in the code says so. Fig. 4 does not distinguish `bound := m` from a bound that stays fixed at the minimum of s: both give Q = [2, 4, 7]. The fix therefore follows the pseudocode, and the example only confirms the outcome (§4, Check 2).

**A2 — crash when no complete solution is found. Partly resolved in `6cf6b3c`.** Before, if neither GBH nor CSUM completed a solution, `best_total_moves` stayed `inf` and `int(self.best_total_moves)` (`algorithm.py:122`) raised. Scratchpad check on stacks `[[1, 2], [3, 4]]` with H = 2 (PC 1 blocked, the only other stack full):

```
No complete solution (edge case, not from the paper)
  raised OverflowError: cannot convert float infinity to integer
```

This requires GBH to fail, which only happens when S is empty. §3.2 shows that this cannot occur on Caserta, so the bug is latent there.

Since `6cf6b3c`, a GBH run without a complete solution raises a `RuntimeError` with a clear message (`algorithm.py:93-98`, §3.2). The same layout now gives:

```
No complete solution (edge case, not from the paper)
  raised RuntimeError: GBH found no complete solution within gbh_time_limit_s = 1.0 s
```

`int(self.best_total_moves)` (`algorithm.py:122`) is still there. As long as Tbest starts at the GBH solution, CSUM always has a solution and the line is safe. If Tbest is changed to start at ∞ (§3.3), A2 has to be revisited in that step.

### 3.5 Notes for the standardisation (not deviations)

- `fidelity = "faithful"` (`algorithm.py:493`) does not match §3.3 and §3.4.
- The seed loop and the mean over seeds (`algorithm.py:504-571`) are left over from multi-seed evaluation. With `n_seeds = 1`, the final record's `metric` and `metrics` describe the same plan, so the `_push` issue from `lee_lee_2010.md` §3.5 does not arise here.
- `max_branches_b` is a UI parameter that does not exist in the paper (header docstring line 7, schema `algorithm.py:590-594`).
- The `mode == "relocations"` branch in CSUM (`algorithm.py:398-403`) is unused in CRP-Time. Since `6cf6b3c`, the lower bound itself is used by GBH.
- The algorithm builds its `ObjectiveSpec` from the config (`algorithm.py:523`) and does not force `mode = "crane_time"` the way `CRP_Time._objective_metrics` does (`problems/CRP_Time.py:80-83`). That is safe today, because the UI only offers `crane_time` (`problems/CRP_Time.py:144-154`) and the CLI sets it (`main.py:258-261`).
- **Platform point for Wei: saved results do not depend on the algorithm parameters.** A saved run's path is built from the problem, the algorithm name, the seed and the problem configuration (`core/results/store.py:165-180`); for CRP-Time the configuration part covers only the objective and time-model settings (`_objective_result_suffix`, `core/results/store.py:66-97`). The algorithm configuration is stored inside the file but plays no part in its name, so two runs of the same algorithm with different parameters (e.g. other time limits) on the same instance and objective overwrite each other: `save_run` writes to that path with mode `"w"` (`core/results/store.py:266-268`).
- **Platform point for Wei: the web backend accepts CRP-Time only on random layouts.** `validate_source_for_problem` (`web/backend/benchmarks.py:280-301`), called when a job is started (`web/backend/api.py:230`), rejects the Caserta and Zhu sources for every problem outside `BENCH_PROBLEMS = {"CRP-R", "CRP-U"}` (`web/backend/benchmarks.py:37`, `:288-289`). A CRP-Time job on Caserta therefore fails with HTTP 422, "caserta benchmark requires CRP-R or CRP-U"; the same rule is on `origin/main`. Caserta runs of CRP-Time currently work only through the CLI (`main.py layout-run`). This blocks the standardisation requirement that each algorithm runs correctly through the web UI on the Caserta instances.

## 4. Behavioural checks

Reproduce with `scratchpad/azari_validate.py` (not committed). It calls the solver's branching helpers directly on the two layouts; stack k in the paper is index k − 1 in the code.

### Check 1 — Fig. 1 / Def. 3: best BB stack 4, best BG stack 1 — **passes**

Layout of Fig. 1 after container 1 is retrieved; H = 5, so that stack 4 (four containers) is not full and S = {1, 2, 4} as in the paper.

```
Fig. 1 / Def. 3
  q = 8, S = [1, 2, 4]
  best BB stack: 4   (paper: 4)
  best BG stack: 1   (paper: 1)
  GBH order [FDS, SDS]: [1, 4]   (paper: [1, 4])
  -> PASS
```

The paper does not state the FDS/SDS order for this example; [1, 4] follows from Sec. 4.2 (a BG relocation exists, so FDS = best BG, SDS = best BB).

### Check 2 — Fig. 4: set-B branching gives Q = [2, 4, 7] — **passes** (since `ac344eb`)

Each stack holds only its minimum (14, 13, PS, 12, 9, 15, 11, empty); the PS holds 1 below q = 10. Only the minima enter the set-B rule, so the other containers of Fig. 4 are not needed.

After `ac344eb`:
```
Fig. 4
  q = 10, PS = 3, lowest per stack = [14, 13, 'PS', 12, 9, 15, 11, 16]
    stack 1 (G1, BG, min 14): -
    stack 2 (G1, BG, min 13): in Q
    stack 4 (G2, BG, min 12): in Q
    stack 5 (G2, BB, min 9): -
    stack 6 (G2, BG, min 15): -
    stack 7 (G2, BG, min 11): in Q
    stack 8 (G2, BG, min 16): -
  Q = [2, 4, 7]   (paper: [2, 4, 7])
  -> PASS
```

Before (`af4e547`), same script:
```
    stack 7 (G2, BG, min 11): -
  Q = [2, 4]   (paper: [2, 4, 7])
  -> FAIL
```

Before the fix, stack 7 was rejected only because of stack 5, which is BB and is not compared in the paper's example (A1, §3.4). As noted there, this example does not distinguish `bound := m` from a fixed bound, so it confirms the outcome, not the update rule. (The empty stack shows 16 = max priority + 1, because this reduced layout has no containers 2–8; the value plays no role.)

### Check 3 — under the paper's own settings (Table 2) — **not done**

Requires the Ünlüyurt & Aydın (2012) instances and their time model; neither is available.

### Check 4 — under the crane-time-only setting (f2 / f2vert) — **not done**

### Check 5 — GBH as depth-first search (since `6cf6b3c`) — **passes**

Reproduce with `scratchpad/azari_gbh_check.py` (not committed). It runs GBH alone on all 840 Caserta instances with `gbh_time_limit_s = 3` and compares it with the greedy pass of `3510021`. Machine: 16 logical processors (AMD64 Family 25 Model 117), Windows 11, conda env `rl`, 6 instances in parallel.

Columns: *first=greedy*: the first complete GBH solution equals the old greedy solution, move for move. *LB0<=GBH<=greedy*: the GBH result lies between the lower bound of the initial layout and the greedy result. *UB strict*: each new GBH solution has strictly fewer movements than the previous one. *valid*: every GBH solution replays as a restricted-BRP solution (top-only moves, retrieval order, capacity, relocations only from the current target's stack, tiers). *exhausted*: GBH searched its whole tree before T. *greedy*, *GBH*, *LB0*: mean number of movements. *=LB0*: GBH reached LB₀. *max s*: longest GBH run. *brute=GBH*: on the three smallest classes, the minimum over the full FDS/SDS tree without pruning and without time limit equals the GBH result (instances where GBH exhausted its tree).

```
class        n first=greedy LB0<=GBH<=greedy UB strict valid exhausted  greedy     GBH    LB0  =LB0  max s brute=GBH
data3-3     40        40/40            40/40     40/40 40/40     40/40   14.07   14.00  12.70  7/40   0.00     40/40
data3-4     40        40/40            40/40     40/40 40/40     40/40   18.30   18.23  16.77  9/40   0.00     40/40
data3-5     40        40/40            40/40     40/40 40/40     40/40   22.05   22.02  20.77 10/40   0.00         -
data3-6     40        40/40            40/40     40/40 40/40     40/40   26.45   26.43  25.20 11/40   0.02         -
data3-7     40        40/40            40/40     40/40 40/40     40/40   30.32   30.30  29.18 13/40   0.00         -
data3-8     40        40/40            40/40     40/40 40/40     40/40   34.73   34.65  33.25  5/40   0.02         -
data4-4     40        40/40            40/40     40/40 40/40     40/40   26.98   26.27  23.30  1/40   0.02     40/40
data4-5     40        40/40            40/40     40/40 40/40     40/40   33.55   32.98  30.18  2/40   0.02         -
data4-6     40        40/40            40/40     40/40 40/40     40/40   38.67   38.05  35.08  1/40   0.02         -
data4-7     40        40/40            40/40     40/40 40/40     40/40   44.90   44.17  41.40  0/40   0.02         -
data5-4     40        40/40            40/40     40/40 40/40     40/40   36.75   35.52  30.60  0/40   0.03         -
data5-5     40        40/40            40/40     40/40 40/40     40/40   46.23   44.17  38.15  0/40   0.23         -
data5-6     40        40/40            40/40     40/40 40/40     40/40   54.25   52.40  46.80  0/40   0.39         -
data5-7     40        40/40            40/40     40/40 40/40     40/40   61.33   59.67  54.12  0/40   0.84         -
data5-8     40        40/40            40/40     40/40 40/40     40/40   69.60   67.88  62.20  0/40   0.80         -
data5-9     40        40/40            40/40     40/40 40/40     40/40   77.35   75.83  70.05  0/40   1.77         -
data5-10    40        40/40            40/40     40/40 40/40     39/40   85.50   83.53  77.72  0/40   3.00         -
data6-6     40        40/40            40/40     40/40 40/40     28/40   71.90   68.20  57.45  0/40   3.00         -
data6-10    40        40/40            40/40     40/40 40/40     17/40  109.85  107.47  96.08  0/40   3.00         -
data10-6    40        40/40            40/40     40/40 40/40      0/40  161.25  156.85 102.25  0/40   3.00         -
data10-10   40        40/40            40/40     40/40 40/40      0/40  239.28  236.75 170.53  0/40   3.00         -

hit the GBH limit: 116 instances; max overshoot 0.0 ms
```

- All checks hold on all 840 instances; the brute-force comparison holds on all 120 instances of data3-3, data3-4 and data4-4.
- GBH searched its whole tree within 3 s on 724 of 840 instances; the 116 others (mostly data6-6, data6-10, data10-6, data10-10) stopped at the limit, and overshot it by less than 0.1 ms. GBH reached LB₀ on 59 instances, all in the classes with 3 or 4 tiers.
- The worked examples still pass (Check 1, Check 2), and the edge case from A2 now raises the `RuntimeError` (§3.4).

**Effect on the full run (descriptive, not a check).** `scratchpad/azari_compare.py` runs the complete algorithm before (`3510021`: greedy GBH, one 5 s budget) and after (`6cf6b3c`: GBH 3 s, CSUM 5 s) on instances 1–10 of every class, under f2 and f2vert. The old greedy pass took milliseconds, so CSUM had about 5 s in both versions; the differences come from the GBH solution, which sets the initial best time and set A. Columns: mean objective old/new, relative difference, number of instances where the new objective is lower/higher, mean relocations old/new.

```
time model f2
class        n    obj old    obj new  diff % new<old new>old  rel old  rel new
data3-3     10     455.28     455.28    0.00       0       0     4.40     4.40
data3-4     10     637.80     637.80    0.00       0       0     6.30     6.30
data3-5     10     805.32     805.32    0.00       0       0     7.30     7.30
data3-6     10     978.48     978.48    0.00       0       0     8.60     8.60
data3-7     10    1147.20    1147.20    0.00       0       0     9.40     9.40
data3-8     10    1328.04    1328.04    0.00       0       0    10.30    10.30
data4-4     10     869.52     869.52    0.00       0       0     9.00     9.00
data4-5     10    1156.80    1156.80    0.00       0       0    12.40    12.40
data4-6     10    1396.56    1391.52   -0.36       1       0    14.20    14.00
data4-7     10    1672.92    1673.88    0.06       0       1    16.50    16.50
data5-4     10    1258.08    1230.12   -2.22       5       1    16.00    15.10
data5-5     10    1606.20    1569.12   -2.31       5       0    19.90    18.60
data5-6     10    1936.80    1912.44   -1.26       6       0    22.80    21.70
data5-7     10    2226.72    2190.96   -1.61       2       0    24.20    23.20
data5-8     10    2667.72    2638.32   -1.10       6       0    29.10    28.20
data5-9     10    3081.72    3044.28   -1.21       5       1    33.10    32.30
data5-10    10    3446.04    3420.48   -0.74       5       0    36.10    34.40
data6-6     10    2680.32    2544.48   -5.07       9       0    37.20    32.80
data6-10    10    4484.40    4413.24   -1.59       7       0    50.60    48.50
data10-6    10    5970.24    5857.68   -1.89       9       0   103.60   100.20
data10-10   10    9663.72    9577.20   -0.90       7       0   137.10   134.60

time model f2_vertical
class        n    obj old    obj new  diff % new<old new>old  rel old  rel new
data3-3     10     865.50     865.50    0.00       0       0     4.40     4.40
data3-4     10    1204.82    1204.82    0.00       0       0     6.30     6.30
data3-5     10    1526.14    1526.14    0.00       0       0     7.40     7.40
data3-6     10    1802.65    1802.65    0.00       0       0     8.60     8.60
data3-7     10    2095.91    2095.91    0.00       0       0     9.50     9.50
data3-8     10    2432.00    2432.00    0.00       0       0    10.30    10.30
data4-4     10    1886.56    1888.48    0.10       0       1     9.10     9.00
data4-5     10    2491.07    2484.74   -0.25       1       0    12.40    12.50
data4-6     10    2955.28    2936.15   -0.65       1       0    14.30    14.10
data4-7     10    3468.48    3470.99    0.07       0       1    16.70    16.60
data5-4     10    3011.39    2938.94   -2.41       5       1    16.30    15.30
data5-5     10    3850.50    3739.06   -2.89       6       0    20.30    18.90
data5-6     10    4525.73    4445.40   -1.77       4       1    23.60    22.10
data5-7     10    5139.39    5026.77   -2.19       3       0    24.20    23.10
data5-8     10    6138.98    6036.90   -1.66       6       0    29.70    28.50
data5-9     10    7016.91    6885.41   -1.87       4       1    33.20    31.90
data5-10    10    7774.51    7612.85   -2.08       6       1    36.50    34.50
data6-6     10    6996.82    6500.56   -7.09       9       0    37.50    32.80
data6-10    10   11082.82   10805.80   -2.50       9       0    50.40    48.10
data10-6    10   21282.43   20726.65   -2.61       9       0   104.10   100.20
data10-10   10   32534.15   32066.75   -1.44       7       0   136.80   134.60
```

On this sample the new version has a lower crane time on 67 of 210 instances under f2 and 70 under f2vert, and a higher one on 3 and 6. On the 3-tier classes the results are identical. The relocation count is reported only; it is not optimised. This is a sample of 10 instances per class, not a benchmark.

**End-to-end.**
- CLI, `main.py layout-run --problem "CRP-Time" --algo "Azari–Eskandari–Nourmohammadi (2017) CSUM" --layout benchmark/Caserta_dataset/data5-5-1.dat`: with `--time-model f2`, objective 1725.60, 23 relocations, 48 movements; with `--time-model f2_vertical`, objective 4135.64 for the same plan. Both exit with code 0.
- Web backend, through the API the web UI uses (`scratchpad/azari_web_check.py`, FastAPI TestClient, results written to the scratchpad instead of `results/`): the catalog shows `gbh_time_limit_s` (default 3.0) and `csum_time_limit_s` (default 5.0). A CRP-Time job on Caserta is rejected with HTTP 422 (platform point, §3.5). Jobs on a random layout complete under f2 and f2vert.
