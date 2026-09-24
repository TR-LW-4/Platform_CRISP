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

| Step | Paper (p.) | Code (file:line) | Verdict | Category |
|---|---|---|---|---|
| | | | | |

Verdict: equal / deviation / not specified in paper.
Category: unified 2D setting / choice left open by paper / differs from paper (bug).

## 3. Deviations and decisions

### 3.1 Unified 2D setting

### 3.2 Choices left open by the paper

### 3.3 Differs from the paper

## 4. Behavioural checks
- [ ] Fig. 1 / Def. 3: best BB stack 4, best BG stack 1
- [ ] Fig. 4: CSUM set-B branching gives Q = [2, 4, 7]
- [ ] Under the paper's own settings: Table 2 averages (requires the Ünlüyurt & Aydın instances and time model)
- [ ] Under the crane-time-only setting (f2 / f2vert):