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
| Single bay, W stacks, H tiers | 311 | Caserta loader: one stack per bay, `num_rows = 1` (`core/benchmarks/caserta.py:227-228`); stacks read bottom→top (`algorithm.py:488-492`) | equal | — |
| Truck lane next to stack 1 | 311 | truck at stack index 0, retrieval travel grows with the stack index (`core/objectives.py:202-205`, `:261-265`); G1 = stacks with a lower index than the PS (`algorithm.py:256-259`) | equal | — |
| Crane starts above the truck lane | 311 | `crane_pos = (1, 1)` (`algorithm.py:104`, `:286`); only `rmgc_current` reads it; f2/f2vert do not depend on the crane position (`core/objectives.py:303-316`) | deviation | unified 2D setting |
| Priorities unique; lower number = earlier | 311 | `targets = sorted(...)` (`algorithm.py:75-77`); Caserta priorities are a permutation of 1..N (`core/benchmarks/caserta.py:164-180`) | equal | — |
| Only top containers accessible; one move at a time; capacity H | 311 | moves pop/push the top (`algorithm.py:152-190`); destination must satisfy `len(st) < max_tiers` (`:192-196`, `:258`, `:265`) | equal | — |
| Restricted: only q, the top of the PS, is relocated | 312–313 | q = top of the stack that holds the target (`algorithm.py:327`, `:410`) | equal | — |
| Objective: minimise T_CW; NMOV secondary | 309–311 | search minimises f2 or f2vert through `movement_objective_cost` (`algorithm.py:389-391`, `:432-434`); movements reported as `steps` (`:514`) and `total_moves` by the shared evaluator | equal (time model adapted) | unified 2D setting |
| Time model for t_rem and t_rel | 311 (no formulas) | f2: t_rem = 2·ts·s + tpp, t_rel = 2·ts·\|s1 − s2\| + tpp (`core/objectives.py:261-270`); f2vert (`:278-295`); tiers passed as in `annotate_plan_tiers` (`algorithm.py:317`, `:337-338`) | not specified in paper | unified 2D setting |
| Def. 1: good / bad container | 311 | `_is_bg` / `_is_bb` compare q with the stack minimum (`algorithm.py:124-128`); the LB's bad test compares with every container below (`:140-145`) | equal | — |
| Def. 2: BB / BG relocation | 312 | `algorithm.py:124-128` | equal | — |
| Def. 3: best BB / best BG stack, compared only among BB resp. BG stacks | 312 (reading inferred from Fig. 1) | `_best_bb_stack` / `_best_bg_stack` filter on BB/BG first, then take the max/min stack minimum (`algorithm.py:198-218`) | equal | choice left open by paper |
| Lowest number of an empty stack = N + 1 | 312 | `empty_lowest = max(targets) + 1` (`algorithm.py:78`, `:121-122`) | equal | — |
| Several empty stacks: lowest stack number | 312 | BG key `(lowest, index)` (`algorithm.py:207`, `:261`) | equal | — |
| Ties between non-empty stacks | not specified | cannot occur: priorities are unique, and a BB stack is never empty (`algorithm.py:218`, `:240`) | not specified in paper | choice left open by paper |
| S = non-full stacks except the PS | 312 (Table 1) vs 313 | `_true_eligible_dsts` (`algorithm.py:192-196`) | equal (Table 1 reading) | choice left open by paper |
| Lower bound 2x + y (Kim & Hong) | 312 | `_lower_bound_moves_2x_plus_y` (`algorithm.py:136-150`): a container is bad if any container below it has a lower number; this is not the shared B1 bound. Used only when `mode == "relocations"` (`:370-375`), which CRP-Time never allows (`problems/CRP_Time.py:144-154`) | equal (definition), unused | — |
| GBH: recursive DFS with backtracking, UB = H × N, LB pruning, own time limit | 312–313, Fig. 2 | `_run_gbh_initial` is one greedy pass that always takes the FDS (`algorithm.py:301-343`): no backtracking, no UB, no LB, no time check | deviation | differs from paper |
| GBH branching: FDS / SDS | 313 | `_gbh_candidates` (`algorithm.py:220-243`) | equal | — |
| Neither FDS nor SDS exists: "the search stops" | 313 | GBH returns an incomplete solution (`algorithm.py:328-330`); in CSUM only that node returns and siblings are still explored (`:418-419`) | not specified in paper | choice left open by paper |
| CSUM initial solution: GBH under time limit T | 313 | the greedy GBH pass (row above) | deviation | differs from paper |
| Set A: at least one BB relocation / relocated more than once | 313, Fig. 3 | relocated more than once in the initial solution (`algorithm.py:94-97`) | equal | — |
| Initialisation: NMOV = GBH moves, Tbest = ∞ | Fig. 3 | `best_time` starts at the crane time of the GBH solution, which becomes a candidate output (`algorithm.py:90-92`, `:284-299`); NMOV is not used | deviation | differs from paper |
| Time limit checked on each call and before branching | Fig. 3 | `algorithm.py:354-355`, `:421-422` | equal | — |
| PC not blocked: retrieve, tcw + t_rem, Nm + 1, recurse, undo | Fig. 3 | `algorithm.py:382-408` | equal | — |
| New best when the last container is retrieved and tcw ≤ Tbest | Fig. 3 | only a strictly lower time reaches the leaf (`algorithm.py:357-358`, `:393-394`, `:436`); leaf acceptance `:360-368`. The equal-time, fewer-moves branch (`:361-363`) is unreachable | deviation | differs from paper |
| Pruning on Nm ≤ NMOV; NMOV = Nm at each new best | Fig. 3 (p. 315 says UB) | no movement-count pruning in crane-time mode; a branch is cut once tcw + t ≥ best time (`algorithm.py:357-358`, `:393-394`, `:436`); `best_total_moves` is stored (`:366`) but never used for pruning | deviation | differs from paper |
| q ∈ A: GBH branching | Fig. 3 | `algorithm.py:411-412` | equal | — |
| q ∈ B, G1: best BG stack s, bound = its minimum | Fig. 3 | `algorithm.py:256-262` | equal | — |
| q ∈ B, G2: which stacks are compared, update of bound | Fig. 3, Fig. 4 | stack m is added if it is BG, non-full, and its minimum is below the minimum of *every* earlier stack except the PS, including BB stacks, full stacks and G1 stacks other than s (`algorithm.py:264-271`). There is no bound variable | deviation | differs from paper (suspected bug A1) |
| q ∈ B and Q empty | not specified | falls back to GBH branching (`algorithm.py:413-416`) | not specified in paper | choice left open by paper |
| Cap on the number of set-B branches | not in the paper | `dedup[: max_branches_b]`, default 6, exposed in the UI (`algorithm.py:282`, `:503`, `:556-560`) | deviation | differs from paper |
| For each s in Q, in order: t_rel, move, recurse, undo | Fig. 3 | `algorithm.py:421-447` | equal | — |
| Time limits: 1 s (Table 2); 3 s GBH + 5 s CSUM (Lee & Lee instances) | 316 | one budget `time_limit_s` for GBH and CSUM together, default 5 s, clamped to ≥ 0.1 s (`algorithm.py:71`, `:73`, `:502`, `:551-555`) | not specified in paper (for Caserta) | choice left open by paper |
| H | 313 (UB = H × N) | `max_tiers` = H′ + 2 for Caserta (`core/benchmarks/caserta.py:154-160`, `:229`); used for S and for h_max in f2vert (`core/objectives.py:280`). UB = H × N is not used | equal | unified 2D setting |
| Deterministic | 316 | no random choices in the solver; the seed is set on the environment only (`algorithm.py:484`) | equal | — |
| Output: best solution found | 313 | `best_moves`, re-evaluated with the shared evaluator (`algorithm.py:505-513`) | equal | — |

Verdict: equal / deviation / not specified in paper.
Category: unified 2D setting / choice left open by paper / differs from paper (bug).

## 3. Deviations and decisions

### 3.1 Unified 2D setting

**Time model.** The paper gives no formulas and follows Lee & Lee (2010), whose procedure includes moving the trolley to the source stack of each movement. The code takes t_rem and t_rel from Voß & Schwarze f2 or f2vert (`core/objectives.py:261-295`). In f2, a retrieval from stack s costs 2·ts·s + tpp and a relocation from s1 to s2 costs 2·ts·|s1 − s2| + tpp. Each movement's cost is independent of the previous one: there is no repositioning term, as already noted in `lee_lee_2010.md` §3.1. The search logic is unchanged, but the times it compares differ from those in the paper's model.

**Crane start position.** The paper starts the crane above the truck lane. The code starts it at `(1, 1)`, i.e. stack 1 (`algorithm.py:104`, `:286`), but only `rmgc_current` reads the position. Under f2 and f2vert the start position has no effect.

**Geometry.** Azari et al. already work in a single bay with the truck lane next to stack 1. That matches the CRISP convention (truck at stack index 0), so no 3D → 2D conversion is needed. For Caserta, H = H′ + 2 (`core/benchmarks/caserta.py:154-160`). H determines which stacks are full, and f2vert uses it for h_max.

### 3.2 Choices left open by the paper

**Def. 3.** The code compares the best BB stack only with BB stacks and the best BG stack only with BG stacks (`algorithm.py:198-218`). This is the reading inferred from Fig. 1, and Check 1 (§4) passes.

**S** is the Table 1 definition: non-full stacks except the PS (`algorithm.py:192-196`).

**Tie-breaking.** Only empty stacks can tie, since priorities are unique and a BB stack is never empty. Among empty stacks the lowest stack number wins, as the paper prescribes (`algorithm.py:207`, `:261`).

**Set-B container without any BG destination.** The code falls back to GBH branching (`algorithm.py:413-416`), so the container gets the best and second-best BB stacks. The paper does not mention this case. Without the fallback the branch would have no children.

**"If neither FDS nor SDS exists, the search stops."** I am not sure whether the paper means the whole search or only the current branch. The code stops the current branch. On Caserta the case cannot occur: S is empty only if all W − 1 other stacks are full, which requires (W − 1)(H′ + 2) ≤ W·H′ − 1, i.e. H′ ≥ 2W − 1. The Caserta class closest to that is data10-6 (H′ = 10, W = 6), and it does not meet it.

**Time limit.** There is one budget, `time_limit_s` (default 5 s), for GBH and CSUM together. The paper uses 1 s for its Table 2 instances and 3 s + 5 s for the Lee & Lee instances. Because GBH is a single greedy pass here (§3.3), almost the whole budget goes to CSUM. As in the paper, results depend on machine speed.

### 3.3 Differs from the paper

**GBH is a single greedy pass.** The paper's GBH is a depth-first search with backtracking that minimises the number of movements within time limit T, pruning on Nm + LB ≤ UB. The code follows the FDS at every node until all containers are retrieved (`algorithm.py:301-343`). That is the first branch the paper's GBH explores (assuming UB = H × N does not prune it), but the search stops there. Consequences:
- The initial solution can have more movements than the paper's.
- Set A is derived from this solution (`algorithm.py:94-97`), so the A/B split can differ from the paper's.
- The Kim & Hong lower bound is implemented correctly, but no CRP-Time code path uses it. It is only reached when `mode == "relocations"` (`algorithm.py:370-375`), and CRP-Time offers only `crane_time`. The shared B1 bound is not used here either, so B1 does not affect Azari.

**CSUM prunes on time instead of movement count.** The paper prunes a branch unless Nm ≤ NMOV, where NMOV is the movement count of the best solution so far. CSUM therefore only looks for faster solutions with at most that many movements. The code has no movement-count condition: it cuts a branch as soon as its accumulated time reaches the best time (`algorithm.py:357-358`, `:393-394`, `:436`). Because f2 and f2vert costs are non-negative, this time cut only removes branches that cannot give a strictly lower time. Dropping Nm ≤ NMOV, however, widens the search: the code can return a solution with more movements than the initial solution if it is faster, which the paper's CSUM never does. Within the same time budget, the two searches also spend their time in different parts of the tree. The code does not document whether this change was meant as an adaptation to the crane-time-only setting.

**Tbest starts at the GBH solution, not at ∞.** The GBH solution's crane time becomes the initial best time (`algorithm.py:90-92`). The code's output is therefore never slower than the GBH solution, whereas in the paper the output is the best CSUM leaf.

**Ties.** Fig. 3 accepts a new best when tcw ≤ Tbest, so a later solution with the same time replaces the earlier one. In the code, a branch that reaches the best time exactly is cut before the leaf, so the first solution with a given time is kept. The code also contains an "equal time, fewer moves" rule (`algorithm.py:293-295`, `:361-363`). It is unreachable at present: `_dfs` never reaches a leaf with equal time, and `_accept_solution_if_better` is called only once, while the best time is still ∞. If it ever became reachable, it would break time ties on the movement count. An algorithm may use relocations or movements internally where its paper does so (Wei's decision, late Sept), but Fig. 3 does not break ties this way: it accepts any solution with tcw ≤ Tbest, so the later one wins. Movements enter CSUM only through the pruning Nm ≤ NMOV.

**Cap on set-B branches.** `max_branches_b` (default 6, exposed in the UI) truncates Q (`algorithm.py:282`). The paper has no such cap. Q holds at most W − 1 stacks, so the cap can only bind for W ≥ 8 (Caserta classes data3-8, data5-8, data5-9, data5-10, data6-10, data10-10).

### 3.4 Suspected bugs

**A1 — set-B branching compares against every earlier stack, not only BG stacks. Fails Fig. 4.** `algorithm.py:269-270`:

```python
prev = [i for i in range(0, m) if i != src]
if all(lows[m] < lows[i] for i in prev):
```

A G2 stack m is added only if its minimum is lower than the minimum of every earlier stack except the PS. This includes BB stacks, full stacks and G1 stacks other than s. This is the literal reading of the text (p. 314–315). As §1 notes, Fig. 4 contradicts it: the example only compares BG stacks, via bound. On Fig. 4, BB stack 5 (minimum 9) blocks stack 7 (minimum 11), and the code returns Q = [2, 4] instead of [2, 4, 7] (§4, Check 2). In general, any BB or full stack with a low minimum hides every later G2 stack. Set-B containers then get fewer branches than in the paper, and possibly none, in which case the GBH fallback from §3.2 applies. Not fixed here; the code is unchanged.

**A2 — crash when no complete solution is found.** If neither GBH nor CSUM completes a solution, `best_total_moves` stays `inf` and `int(self.best_total_moves)` (`algorithm.py:114`) raises. Scratchpad check on stacks `[[1, 2], [3, 4]]` with H = 2 (PC 1 blocked, the only other stack full):

```
No complete solution (edge case, not from the paper)
  raised OverflowError: cannot convert float infinity to integer
```

This requires GBH to fail, which only happens when S is empty. §3.2 shows that this cannot occur on Caserta, so the bug is latent there.

### 3.5 Notes for the standardisation (not deviations)

- `fidelity = "faithful"` (`algorithm.py:465`) does not match §3.3 and §3.4.
- The seed loop and the mean over seeds (`algorithm.py:476-542`) are left over from multi-seed evaluation. With `n_seeds = 1`, the final record's `metric` and `metrics` describe the same plan, so the `_push` issue from `lee_lee_2010.md` §3.5 does not arise here. The `time_limit_s` help text still says "each seed's" (`algorithm.py:554`).
- `max_branches_b` is a UI parameter that does not exist in the paper (header docstring line 6, schema `:556-560`).
- The `mode == "relocations"` branch and the lower bound (`algorithm.py:136-150`, `:370-375`) are unused in CRP-Time.
- The algorithm builds its `ObjectiveSpec` from the config (`algorithm.py:495`) and does not force `mode = "crane_time"` the way `CRP_Time._objective_metrics` does (`problems/CRP_Time.py:80-83`). That is safe today, because the UI only offers `crane_time` (`problems/CRP_Time.py:144-154`) and the CLI sets it (`main.py:258-261`).

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

### Check 2 — Fig. 4: set-B branching gives Q = [2, 4, 7] — **fails**

Each stack holds only its minimum (14, 13, PS, 12, 9, 15, 11, empty); the PS holds 1 below q = 10. Only the minima enter the set-B rule, so the other containers of Fig. 4 are not needed.

```
Fig. 4
  q = 10, PS = 3, lowest per stack = [14, 13, 'PS', 12, 9, 15, 11, 16]
    stack 4 (BG, min 12): blocked by earlier stacks []
    stack 5 (BB, min 9): blocked by earlier stacks []
    stack 6 (BG, min 15): blocked by earlier stacks [1, 2, 4, 5]
    stack 7 (BG, min 11): blocked by earlier stacks [5]
    stack 8 (BG, min 16): blocked by earlier stacks [1, 2, 4, 5, 6, 7]
  Q = [2, 4]   (paper: [2, 4, 7])
  -> FAIL
```

Stack 7 is rejected only because of stack 5, which is BB and would not be compared in the paper's example. See A1 (§3.4). (The empty stack shows 16 = max priority + 1, because this reduced layout has no containers 2–8; the value plays no role.)

### Check 3 — under the paper's own settings (Table 2) — **not done**

Requires the Ünlüyurt & Aydın (2012) instances and their time model; neither is available.

### Check 4 — under the crane-time-only setting (f2 / f2vert) — **not done**
