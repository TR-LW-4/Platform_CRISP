# Validation log — López-Plata et al. (2019)

**Paper:** I. López-Plata, C. Expósito-Izquierdo, E. Lalla-Ruiz, B. Melián-Batista, J. M. Moreno-Vega (2019). *Minimizing the operating cost of block retrieval operations in stacking facilities.* Computers & Industrial Engineering 136, 436–452.
**Code:** `algorithms/CRP_Time/heuristic/lopez_plata_2019/`
**Relocation rule:** restricted
**Status:** §1 (specification) complete. §2, §3 and §5 filled in (code comparison, 29 Sept 2026); §4 still open. Rebuild in 7 steps; steps 1 (eq. 1–3) and 2 (complete plans) done.

Page numbers refer to the journal pagination (436–452). "Eq." refers to the paper's equation numbers.

---

## §0 Goal and ground rules

- The algorithm must be faithful to the paper's algorithmic logic. Reproducing the paper's numerical results is not required. Every assumption and deviation is documented here.
- Order of work:
  1. Validate in the paper's own setting (operating-cost objective, eq. 1–3, paper parameters).
  2. Convert to the CRP-Time setting with crane time as the only objective (f2 / f2vert, Voß & Schwarze 2019), keeping the paper's logic intact.
- Relocations and moves may be used internally wherever the paper uses them (e.g. β is counted in moves, κ in retrievals).
- Relocation count is always reported as a secondary metric.
- The algorithm keeps the paper's relocation rule (restricted). Results are not compared with unrestricted algorithms, and the UI shows this property.

---

## §1 Specification from the paper

### 1.1 Scope: what CRISP implements

The paper contributes three things: an optimization model (Sec. 4), an exact A\* algorithm (Sec. 5) and a heuristic based on A\* (Sec. 6). **CRISP implements only the heuristic from Sec. 6.** That heuristic runs a depth-limited version of the Sec. 5 A\* internally (step 1), so Sec. 5 is part of the specification. The model from Sec. 4 is not.

According to the current docstring, the only parameters are `delta = 3` and `num_expansions = 5`. Their mapping to the paper's parameters is still to be confirmed. By value they match τ = 3 and µ = 5, but that is a hypothesis (see C01).

### 1.2 Problem setting (p. 437–438)

- Blocks $C = \{1,\dots,n_C\}$, stacks $S = \{1,\dots,n_S\}$, tiers $T = \{1,\dots,n_T\}$. $s(c)$ and $t(c)$ are the stack and tier of block $c$, and $n(s)$ is the number of blocks in stack $s$ (p. 437).
- $\Upsilon = \{s \in S \mid n(s) < n_T\}$ is the set of non-full stacks (p. 437).
- $r(c)$ is the retrieval order. $c^*$ is the next block to retrieve, and $c^+$ is the block on top of stack $s(c^*)$ (p. 437).
- **Retrieval point:** a dummy slot to the left of the facility, stack 0, tier 1 (p. 437). Stack 1 is therefore the stack closest to the retrieval point.
- **Restricted:** "The relocation movements in the BRP-OC are associated with those blocks placed above the next to retrieve" (p. 437). Only $c^+$ may be relocated, and never within its own stack (see also constraints 15–18, p. 441).
- **Well-located / non-located:** a block is well-located if it is on the ground or above blocks with a later retrieval order; otherwise it is non-located (p. 437).
- **Solution:** $\phi = ((s_0,s'_0),(s_1,s'_1),\dots,(s_{|\phi|},s'_{|\phi|}))$, where each element is one crane move from stack $s_i$ to $s'_i$. The dummy move $(s_0,s'_0)=(0,0)$ encodes the crane's starting position at the retrieval point (p. 437–438). For a retrieval, $s'_i = 0$.
- **Partial solution** $\phi'$: the first $|\phi'|$ moves of a solution $\phi$, in the same order (p. 438).
- At least $n_T - 1$ empty slots are required for feasibility (p. 437).

### 1.3 Objective: operating cost (p. 437–438, 445)

Empty crane move from slot $(s_1,t_1)$ to $(s_2,t_2)$, eq. 1 (p. 437):

$$m(s_1,t_1,s_2,t_2) = (n_T+\epsilon-t_1)\,\eta_1 + d(s_1,s_2)\,\gamma_1 + (n_T+\epsilon-t_2)\,\eta_1$$

Loaded crane move, eq. 2 (p. 437):

$$m'(s_1,t_1,s_2,t_2) = (n_T+\epsilon-t_1)\,\eta_2 + d(s_1,s_2)\,\gamma_2 + (n_T+\epsilon-t_2)\,\eta_2$$

Before moving between stacks, the crane rises to $n_T + \epsilon$, where $\epsilon$ is a security margin (p. 437).

Weight definitions (p. 437):
- $\eta_1$: vertical, crane unloaded; $\eta_2$: vertical, crane loaded
- $\gamma_1$: horizontal, crane unloaded; $\gamma_2$: horizontal, crane loaded

Objective, eq. 3 (p. 438):

$$\min_{\phi\in\chi} f(\phi) = \sum_{i=1}^{|\phi|} m\big(s'_{i-1}, n(s'_{i-1}), s_i, n(s_i)\big) + m'\big(s_i, n(s_i), s'_i, n(s'_i)\big)$$

Every move therefore costs an **empty trip** (from where the previous block was dropped to the pickup location) plus a **loaded trip** (pickup to drop-off). The empty crane movement is counted explicitly, so this is **not** f2 or f2vert.

The sum runs from $i = 1$, so the dummy move carries no cost of its own. It only sets the starting position for the first empty trip.

**Values in the experiments (p. 445):** $\epsilon = 2$; 2D, so $d(s_1,s_2) = |s_1 - s_2|$. Printed weights: $\eta_2 = 1,\ \gamma_2 = 1,\ \eta_1 = 2,\ \gamma_1 = 2$. The sentence around them, however, says a move "costs twice when the crane is loaded … than when it is unloaded". Under the definitions on p. 437 (index 1 = unloaded, index 2 = loaded), the printed values mean the opposite. The Fig. 2 example (p. 438) prints the same values: $\eta_1 = 2,\ \gamma_1 = 2,\ \eta_2 = 1,\ \gamma_2 = 1$. **Contradiction — see O01.**

**Verification of eq. 3 against Fig. 2 (p. 438):** the instance has $n_C = 13$, $n_S = 6$, $n_T = 4$, $\epsilon = 2$. Stacks from bottom to top:
- stack 1: 9, 1, 8
- stack 2: 11, 3, 2
- stack 3: 5, 4
- stack 4: 13, 6
- stack 5: 12, 10, 7
- stack 6: empty

The paper reports two solutions:
- minimizing moves: relocate 8 from stack 1 to 6, then retrieve 1–13; cost 531
- minimizing cost: relocate 8 from stack 1 to 2, retrieve 1, relocate 8 from stack 2 to 1, then retrieve 2–13; cost 513

Recomputing with eq. 3 reproduces both 531 (14 moves) and 513 (15 moves), under the following reading:
- the crane starts at (0, 1);
- a retrieval drops the block at (0, 1);
- the pickup tier is the stack height before the move;
- the drop tier is the stack height after the move.

In this example the totals are identical under both weight readings. This is structural, not a coincidence: under the restricted rule, with the start and every retrieval at (0, 1), the unweighted empty and loaded sums are equal, vertically ($V$, the terms $n_T+\epsilon-t$) and horizontally ($H$, the distances). Vertically, the empty and loaded terms differ only by (start tier) − (drop tier of the last move), and the last move is a retrieval, so this is 1 − 1 = 0. Horizontally, every relocation from $s$ is followed by another move from $s$, so its empty return trip equals its loaded trip; each target is reached from the retrieval point and returned to it from the same stack. Hence $f = (\eta_1+\eta_2)V + (\gamma_1+\gamma_2)H$ for every restricted plan (see V02 in §3 for the numerical check). Hand check, min-moves solution: loaded $H$ = 48 = empty $H$; loaded $V$ = 129 = empty $V$; with the printed weights $f = 3 \cdot 129 + 3 \cdot 48 = 531$. Fig. 2 therefore cannot settle O01, and no restricted instance can. It is, however, a usable unit test for the objective (see V01).

### 1.4 Instances (p. 445)

- Caserta et al. (2011a): 760 instances in 19 groups, i.e. 40 per group (derived as 760/19). The stacks are initially completely full: $n_C = n_T \times n_S$.
- **Two empty tiers** are added on top "to avoid deadlock scenarios" (p. 445).
- Table 1 (p. 445), written as $n_T \times n_S$ ($n_C$):

| Type | Groups |
|---|---|
| Small | 3×3 (9), 3×4 (12), 3×5 (15), 3×6 (18), 4×4 (16), 4×5 (20) |
| Medium | 3×7 (21), 3×8 (24), 4×6 (24), 4×7 (28), 5×4 (20), 5×5 (25), 5×6 (30) |
| Large | 5×7 (35), 5×8 (40), 5×9 (45), 5×10 (50), 6×10 (60), 6×6 (36) |

The range is 3×3 up to **6×10** (60 blocks). An earlier version of this log said "up to 6×6"; that was wrong, because 6×6 is merely the last row of the table.

Table 3 (p. 448) additionally reports a 10×10 row that is not in Table 1 (see O14).

Implementation details (p. 445): Java SE 8, Intel Core i7-4500U 2.4 GHz, 8 GB RAM, CPLEX 12.8 for the model.

### 1.5 Search tree and Algorithm 1 (A\*) (p. 441–443)

**Search tree (p. 441):**
- A node is a configuration of the stacking facility, and a branch is a move (a relocation or a retrieval).
- The root is the initial configuration. A leaf is a configuration in which all blocks have been retrieved.
- The depth of a node equals the number of moves from the root, retrievals included.
- $g(n)$ = the operating cost of the moves from the root to $n$, i.e. $f$ of the partial solution (p. 441).
- $h(n)$ = an estimate of the minimum remaining operating cost (Sec. 5.1).
- A node is only expanded if $g(n) + h(n) < ub$. The initial $ub$ is an input "provided … by the heuristic" (p. 441).
- A retrieval gives exactly one child; a relocation gives at most $|\Upsilon| - 1$ children (p. 441).

**Algorithm 1 (p. 441–442), in words:**
- Lines 1–3: the open list $\mathcal{N} = \{r\}$ and $best$ = empty.
- Lines 4–5: while $\mathcal{N}$ is not empty, take $n = first(\mathcal{N})$.
- **Case (i)** (line 6, $c^* = n_C$, the last block): create the retrieval child $n'$ (line 7). If $g(n') < ub$, then set $ub \leftarrow g(n')$ and $best \leftarrow n'$ (lines 8–10).
- **Case (ii)** (line 13, $c^* = c^+$, so $c^*$ is on top): create the retrieval child and put it **at the beginning** of $\mathcal{N}$ (lines 14–15).
- **Case (iii)** (line 16, $c^*$ is blocked):
  - line 17: $\Pi(n)$ = the children obtained by relocating $c^+$ to each $s \in \Upsilon \setminus \{s(c^+)\}$;
  - line 18: $\Omega(n) = \{n' \in \Pi(n) \mid g(n')+h(n') < ub\}$, sorted by $g(n')$;
  - line 19: put $\Omega(n)$ **at the beginning** of $\mathcal{N}$.
- Line 23: return the moves from $r$ to $best$.

**Interpretation:** because children are always inserted at the front, this is depth-first branch-and-bound with an admissible bound, not best-first A\* with a priority queue. The name "A\*" refers to the admissible $h$. The sort on $g$ in line 18 makes the cheapest child the next one explored.

**What the pseudocode leaves open:**
- $first(\mathcal{N})$ must also remove $n$ from $\mathcal{N}$. This is not written but is required for termination.
- Pruning happens only when relocation children are added (line 18). The retrieval child (lines 14–15) is never tested, and nodes already in $\mathcal{N}$ are not re-tested after $ub$ drops.
- The tie-breaking order for equal $g$ is not specified.

**Estimate $h(n)$ (Sec. 5.1, p. 442):** $h(n)$ is the sum over all blocks still present:

- Well-located block, eq. 21:
  $$EstRet(c) = m(s(c), n_T+\epsilon, s(c), t(c)) + m'(s(c), t(c), 0, 1)$$
  That is: the empty crane descends from above $s(c)$ to the block, then carries it to the retrieval point.
- Non-located block, eq. 22:
  $$EstRel(c) = m(s(c), n_T+\epsilon, s(c), t(c)) + m'(s(c), t(c), s(c)-1, n_T) + EstRet(c)$$
  That is: one relocation to the top of the adjacent stack (towards the retrieval point), plus the retrieval. Exception: if $s(c) = 1$, the relocation goes to $s(c)+1$.

Admissibility argument (p. 442–443):
- the crane is assumed to already be above $s(c)$;
- only one relocation is counted per non-located block;
- that relocation goes to the nearest stack even when this is not feasible.

Hence $h(n) \le g'(n)$, where $g'(n)$ is the true minimum remaining cost.

### 1.6 Heuristic, overview (Sec. 6, Algorithm 2, p. 443)

Three steps (p. 443):
1. **Promising partial solutions:** depth-limited A\* to depth β, followed by selection of up to λ nodes (see O06).
2. **Completing:** Expansion, repeated κ retrievals at a time until every block is retrieved.
3. **Postprocessing:** remove temporary moves.

**Algorithm 2 (p. 443):**
- line 1: $\phi_{best} \leftarrow ()$
- line 2: $\mathcal{N} \leftarrow$ GetPartialSolutions(β, λ) — **executed once**
- lines 3–14: for $i = 0$ to $\rho$:
  - line 4: $\mathcal{N}' \leftarrow \mathcal{N}$
  - lines 5–7: while some $n \in \mathcal{N}'$ still has blocks to retrieve, $\mathcal{N}' \leftarrow$ Expansion($\mathcal{N}'$, κ)
  - lines 8–13: for each $n \in \mathcal{N}'$, if $f(\phi_n) < f(\phi_{best})$ then $\phi_{best} \leftarrow \phi_n$
- line 15: $\phi_{best} \leftarrow$ PostProcessing($\phi_{best}$) — **executed once**
- line 16: return $\phi_{best}$

**Randomness:** the only stochastic component is the roulette wheel selection in Expansion (Sec. 6.2). The paper calls the heuristic "semi-random" and gives that as the reason for ρ (p. 443–444). Step 1 and step 3 are deterministic, given a fixed tie-breaking rule.

Note: the pseudocode reads "for (i = 0 to ρ)", which is literally ρ + 1 iterations. The text says "repeated ρ times" (p. 443). See O15.

### 1.7 Step 1 — promising partial solutions (Sec. 6.1, p. 443–444)

- **β = the search depth of step 1, in moves** (relocations and retrievals), and therefore the length of every partial solution in $\mathcal{N}_\beta$. Sec. 7.2 describes β as "the number of movements of the partial solutions … used with the exact technique" (p. 446). In Algorithm 2 it reads "number of movements to calculate" (p. 443), which does *not* mean the number of remaining moves.
- A branch is only explored to depth β "if exists the possibility to improve the best solution obtained until that moment" (p. 443), i.e. the $g+h<ub$ test. Step 1 is therefore a modified Algorithm 1: nodes at depth β are collected in $\mathcal{N}_\beta$ instead of being expanded further. This modification is not in the pseudocode.
- $\mathcal{N}_\beta$ contains only the nodes at depth β, not intermediate nodes and not pruned nodes (p. 443).
- **The ub in step 1 is not specified in Sec. 6.1.** It must be finite, because with $ub = \infty$ nothing is pruned and the whole exponential $\mathcal{N}_\beta$ is enumerated. The most plausible reading is $f(\phi)$ of the greedy solution that eq. 29 also uses (see O04).
- **Line 9 in step 1:** this can only execute if a complete solution fits within β moves. Since every solution has at least $n_C$ moves, this effectively never happens (see O05).
- **Selection** (p. 443): $\mathcal{N} = \mathcal{N}_g \cup \mathcal{N}_q$, where
  - $\mathcal{N}_g$ = the λ nodes of $\mathcal{N}_\beta$ with the **lowest** $g(\cdot)$ — backward-looking: what did the first β moves cost?
  - $\mathcal{N}_q$ = the λ nodes of $\mathcal{N}_\beta$ with the **lowest** $q(\cdot)$ — forward-looking: how difficult is the remaining configuration?

  Taking the union gives step 2 both cheap and well-structured starting points. Note that $\mathcal{N}_g$ ranks on $g$, not on $g+h$.

**Quality function $q(n)$** (p. 443–444), summed over every block $c$ still in the facility:

- Non-located, eq. 23: $q_b(c) = 1$ if $c$ is non-located, otherwise 0.
- Compactness, eq. 24:
  $$q_c(c) = \begin{cases} |r(c) - \min\{r(c') : s(c') = s(c)\}|, & n(s(c)) > 1 \\ r(c) - r(c^*), & \text{otherwise} \end{cases}$$
  The text says "if the stack is empty"; the formula means "c is alone in its stack". See O10.
- Distance to the retrieval point, eq. 25: $q_r(c) = d(s(c), 0)$. This uses the stack only, not the tier.
- Total, eq. 26:
  $$q(n) = \sum_c q_b(c) + \frac{q_c(c)}{n_C} + \frac{q_r(c)}{maxDist}$$
  with $maxDist = \max_{s \in S} d(s,0)$ (p. 444). **Eq. 26 has no mq weights**; those appear only in eq. 28.

### 1.8 Step 2 — completing partial solutions (Sec. 6.2, p. 444)

**Expansion:** starting from a partial solution, build a subtree with **branching factor at most τ**, until κ additional blocks have been retrieved.
- τ is a *width* ("maximum degree of the nodes"), not a depth. The depth is variable and depends on how many relocations are needed.
- A retrieval node has one child anyway, so τ only restricts relocation nodes.
- A retrieval is performed only if the block is on top of its stack (p. 444).
- The result is a set of up to $\tau^R$ sequences (with R relocations), each of which retrieves κ blocks.
- One of the **µ best** sequences is selected by **roulette wheel selection**, and becomes the new partial solution. Expansion is repeated until the solution is complete.

**Relocation evaluation.** Eq. 27 (p. 444):

$$q_m((s,s')) = m'(s, n(s), s', n(s')+1) + m(s', n(s')+1, s, n(s)-1)$$

This is the loaded trip to the new top of $s'$ plus the empty trip back to the new top of $s$. Because of the restricted rule, the next move always happens on $s$ again: either another block above $c^*$, or $c^*$ itself. Here $n(\cdot)$ is evaluated before the move.

$q_m$ is normalized by $maxMov$ = the maximum cost over all possible relocations of this block.

Eq. 28 (p. 444):

$$\alpha((s,s'),c) = \frac{q_m(s,s')}{maxMov}\,mq_m + q_b(c)\,mq_b + \frac{q_c(c)}{n_C}\,mq_c + \frac{q_r(c)}{maxDist}\,mq_r$$

- **Lower α is better:** every term is a penalty.
- Eq. 28 uses only $q_b$, $q_c$ and $q_r$ of **the relocated block c**, not the sum over the whole node as in eq. 26. The text speaks of "the quality measures of the node once applied the movement", which is ambiguous (see O11).
- Fig. 4 (p. 444) illustrates $\alpha((1,3),5)$.

### 1.9 Step 3 — postprocessing (Sec. 6.3, p. 444–445)

- Input: $\phi_{best}$ after the ρ loop. It runs once (Alg. 2, line 15).
- Loop (p. 444):
  1. analyse every move;
  2. if part of the solution can be improved **while keeping feasibility**, apply the change;
  3. if something improved, start again from step 1; otherwise stop.
- The criterion is a lower objective function value (p. 444). An unchanged number of moves is not a stated condition. It does hold for the fix as described, but as a consequence.
- The postprocessing handles **only** temporary moves: "In this paper, the postprocessing is focused on the removal of temporary movements" (p. 444).
- **Temporary move** (Sec. 6.3.1, p. 444): a relocation for which
  - (i) the relocated block is moved again "only a few movements after" — **no number is given**, not in 6.3 and not in 7.5 (see O12);
  - (ii) no other block is relocated on top of it between its two moves.
- **Fix:** relocate the block in its first move to another stack "that implies a lower increase on the objective function value". Condition: "this stack cannot be completed in the sequence of movements between the two movements of the block". What "completed" means is ambiguous (see O13).
  - The criterion is lower cost, which is often, but not always, a closer stack. Eq. 1–2 also count vertical distance, so the height of the target stack matters.
  - The origin stack of the block's second move changes as well, so the saving must be computed over both moves.
- **Differing definition in Sec. 6 (p. 443):** step 3 there describes a temporary move as a block that later "come back … to its original stack", and says such moves are "ruled out". Sec. 6.3.1 does not require a return to the original stack, and there the fix is a different target stack, not a removal (see O12).
- **Fig. 5 (p. 445) does not match the text as written:**
  - Original (red): 5 goes from stack 1 to 3 (onto 4); retrieve 1, 2, 3; 5 goes from stack 3 to 1; retrieve 4.
  - Fix (blue): 5 goes from stack 1 to **2, onto block 2**. After retrieving 1, block 2 is blocked by 5, so 5 has to move from stack 2 to 1 *before* retrieving 2.
  - The fix therefore only works if the second move of 5 is **moved earlier in the sequence**. The number of moves stays the same, and both trips of 5 are shorter.
  - The text only speaks of a different target stack for the first move, not of reordering. With the original order kept, the fix is infeasible (see O13).
  - In this example the solution stays restricted: 5 lies above the next block to retrieve both times.

### 1.10 Parameters

| Symbol | Meaning | Value | Source |
|---|---|---|---|
| $\eta_1,\gamma_1$ | vertical / horizontal weight, unloaded | printed: 2, 2 (contradiction, O01) | p. 437, 445 |
| $\eta_2,\gamma_2$ | vertical / horizontal weight, loaded | printed: 1, 1 (contradiction, O01) | p. 437, 445 |
| $\epsilon$ | security margin | 2 | p. 445 |
| κ | retrievals per expansion | 2 | p. 446 (7.2) |
| µ | size of the restricted candidate list (roulette) | 5 | p. 446 (7.2) |
| $mq_m, mq_b, mq_c, mq_r$ | weights in eq. 28 | 1, 2, 1, 0.5 | p. 446 (7.2) |
| β | search depth of step 1 (moves) | eq. 29–30 | p. 446 (7.2) |
| τ | maximum branching factor in Expansion | 3 | p. 446 (7.3) |
| ρ | number of executions of step 2 | 300 | p. 447–448 (7.4) |
| θ | percentage in eq. 29 | 40 % | p. 447–448 (7.4) |
| λ | maximum number of nodes from step 1 | 500 | p. 447–448 (7.4) |

Eq. 29 (p. 446):

$$\beta = \mathrm{floor}\big(|\phi| \times (\theta/100) \times p(n_C)\big)$$

Here $|\phi|$ is the number of moves (relocations and retrievals) of a greedy solution of the instance. The only greedy the paper describes is the one in Sec. 7.1 (p. 445), which "uses exclusively the cost to move a block to determine where it has to be relocated". The same greedy supplies the initial $ub$ of the exact A\*.

Eq. 30 as printed (p. 446):

$$p(n_C) = \begin{cases} 0.5, & n_C \le 9 \\ 1.5 - \frac{n_C-9}{25-9}, & 9 < n_C < 25 \\ 1.5, & n_C \ge 25 \end{cases}$$

As printed, $p$ jumps at both boundaries: $p = 0.5$ at 9, ≈ 1.44 at 10, ≈ 0.56 at 24, and 1.5 at 25. With θ = 40 this gives β = floor(0.2·|φ|) for $n_C \le 9$ and floor(0.6·|φ|) for $n_C \ge 25$. See O02.

Tested ranges (p. 446–447):
- τ ∈ {1, …, 7} (Fig. 7)
- ρ ∈ {100, 200, 300, 400, 500}
- θ ∈ {10, 20, 30, 40, 50} % in the text of Sec. 7.4; Table 4 (p. 448) also has θ = 0 (O16)
- λ ∈ {50, 100, 250, 500}

The text of Sec. 7.4 lists five θ values, but Table 4 (p. 448) shows six (θ = 0, 10, …, 50), so "5 × 6 × 4 = 120" combinations is consistent (see O16).

Chosen combination: Table 2, first row (p. 447), and the text on p. 448: ρ = 300, θ = 40, λ = 500. This row reaches 97.48 % optimal solutions on the instances where A\* found the optimum (3×3 up to 4×5).

### 1.11 Reference results (orientation only, not a target)

Table 3 (p. 448), averages with the chosen parameters. The time is in seconds on the paper's hardware (Java).

| $n_T\times n_S$ | Avg time (s) | Avg f |
|---|---|---|
| 3×3 | 0.15 | 497.40 |
| 3×4 | 1.45 | 666.00 |
| 3×5 | 4.33 | 834.75 |
| 3×6 | 105.34 | 1022.85 |
| 3×7 | 42.61 | 1210.80 |
| 3×8 | 16.25 | 1415.70 |
| 4×4 | 5.42 | 1040.70 |
| 4×5 | 31.82 | 1333.20 |
| 4×6 | 38.26 | 1587.90 |
| 4×7 | 14.78 | 1903.35 |
| 5×4 | 7.91 | 1510.50 |
| 5×5 | 27.89 | 1921.05 |
| 5×6 | 20.61 | 2357.55 |
| 5×7 | 28.40 | 2828.70 |
| 5×8 | 54.69 | 3267.30 |
| 5×9 | 89.19 | 3747.75 |
| 5×10 | 128.83 | 4226.85 |
| 6×6 | 82.26 | 3356.70 |
| 6×10 | 523.77 | 5780.55 |
| 10×10 | 4920.12 | 14709.75 |
| *Average (as printed)* | *3072.04* | *2760.97* |

The printed average time of 3072.04 is inconsistent with the rows: the mean of the 20 rows is 307.20, so it looks like a factor-10 typo. The average f does match the mean of the 20 rows.

Because of O01 (the weights) and O03 ($n_T$ including or excluding the empty tiers), these values are **not directly comparable** until it is clear which convention the paper used.

### 1.12 Open points and possible errors in the paper

| ID | Point | Paper | What the code should reveal |
|---|---|---|---|
| O01 | Weights: printed values ($\eta_1=\gamma_1=2$ unloaded, $\eta_2=\gamma_2=1$ loaded) contradict "costs twice when … loaded". Fig. 2 cannot settle it (same totals under both readings). The earlier version of this log used "unloaded = 1, loaded = 2" (following the sentence) and the symbols α/β instead of η/γ. | p. 437, 438, 445 | Which weights the code uses, and whether this is documented **Resolved in step 3 (Thom, checked on the page images): (b), loaded heavier.** Table 5 (p. 450) lists η₁, η₂, γ₁, γ₂ with time weights 3, 6, 1, 1.125 (speeds 90, 45, 270, 240 m/min) and energy weights 1, 7.5, 1.61, 2.175, so index 2 (loaded) is heavier; Sec. 7.7 (p. 449): "the chosen weights are generic, assuming a double of effort by the crane when it is loaded". Only the numbers printed at Fig. 2 (p. 438) and on p. 445 say the opposite. Default since step 3: η₁ = γ₁ = 1, η₂ = γ₂ = 2. $f$ of a restricted plan does not depend on it (V02), so V01 stays 531 / 513; $h$ does (V08). |
| O02 | Eq. 30 is discontinuous as printed. Possible typo: $0.5 + \frac{n_C-9}{16}$ (continuous: 0.5 at 9, 1.5 at 25). Hypothesis, not established. | p. 446 | Which version the code implements Step 3a: implemented as printed (provisional, Thom); V09 compares the three readings. Step 3b (Thom): still literal, as one replaceable function (`measures.p`), to be decided after V11. Circumstantial evidence for (c), not proof: under (a) and (b), step 1 on $n_C \ge 25$ searches deeper than $n_C$ (V09), while Fig. 6 (p. 446–447) reports 18 964.99 s (3×6) and 88 145.73 s (4×5) for the same DFS with the greedy ub and no depth limit, and Table 3 (p. 448) reports 27.89 s in total for 5×5 and 523.77 s for 6×10. **Decided (Thom, after V11): (c), a documented correction of a presumed misprint, not the printed formula:** $p = 1.5$ for $n_C \le 9$, $1.5 - (n_C-9)/16$ for $9 < n_C < 25$, $0.5$ for $n_C \ge 25$. Evidence: (1) the printed ramp runs from 1.5 at $n_C$ = 9 to 0.5 at $n_C$ = 25, so (c) makes eq. 30 continuous without changing the ramp; Appendix A calls $p$ a "penalization function". (2) Table 3 (p. 448): the time falls from 3×6 (105.34 s) to 3×7 (42.61 s) and 3×8 (16.25 s) although the instances grow, while under (b) $p$ rises there; from 4×6 (38.26 s) to 4×7 (14.78 s) the time halves, while under (a) $p$ rises from 0.56 to 1.5 there and under (c) it falls to 0.5. (3) V11 shows the same pattern: a peak at 3×6/4×5, mostly finishing at 3×7, 3×8 and 4×6 under (a)/(c) while (b) times out there, and at 4×7 and 5×5 only (c) partly finishes. Caveat: the Table 3 times include step 2, so this is circumstantial evidence, not proof. `measures.p` stays one replaceable function; T5 fails on the printed version. |
| O03 | Is $n_T$ in eq. 1–2 the original number of tiers or including the 2 added empty tiers? This affects the cost values ($n_T+\epsilon$). | p. 437, 445 | Which $n_T$ the cost function uses |
| O04 | Initial ub of the depth-limited A\* in step 1 is not specified in 6.1. Assumption: $f$ of the greedy solution. | p. 441, 443, 445 | Where the ub in step 1 comes from; is it finite? **Decided in step 3b (Thom): (a), ub = f(greedy)** (p. 441, 445). |
| O05 | Edge case: if a complete solution fits within β moves (small instances), step 1 reaches leaves and line 9 can update ub. Behaviour not described. | p. 441–443 | How the code handles leaves in step 1 **Decided in step 3b (Thom): (a), Alg. 1 lines 8–10 literally** (update ub and best). A leaf counts at any depth ≤ β, not only at depth β; the best leaf of step 1 is also a candidate for $\phi_{best}$ in Alg. 2, because a complete node has $g = f$ and may fall outside $\mathcal{N}_g$ (assumption, L15). V09: leaves are reachable within β on $n_C \ge 25$ under (a)/(b). |
| O06 | λ: "the λ nodes" per subset (so $\lvert \mathcal{N}\rvert  \le 2\lambda$) vs. "maximum number of nodes obtained in step 1" (so $\lvert \mathcal{N}\rvert  \le \lambda$; the split between $\mathcal{N}_g$ and $\mathcal{N}_q$ is then unknown). | p. 443 | Which reading the code uses Step 3: Appendix A defines λ as the "maximum number of most promising nodes" and Alg. 2 as the "number of open nodes obtained on step 1"; both support $\lvert\mathcal{N}\rvert \le \lambda$ in total, which leaves the split between $\mathcal{N}_g$ and $\mathcal{N}_q$ open. Sec. 6.1 ("the λ nodes" per subset) supports the other reading. The contradiction stays; not decided. **Decided in step 3b (Thom): (a), λ per subset, $\lvert\mathcal{N}\rvert \le 2\lambda$.** Reason: the λ bullet of Alg. 2 says "the use of this parameter is detailed in Section 6.1", and (a) is the only reading that needs no invented split rule. Counter-evidence kept: Appendix A and the Alg. 2 wording (above) support $\lvert\mathcal{N}\rvert \le \lambda$ in total. |
| O07 | Tie-breaking for equal $g$ (Alg. 1, line 18) and for equal $g$/$q$ in the selection; behaviour when $\lvert \mathcal{N}_\beta\rvert  < \lambda$. | p. 441, 443 | Tie-breaking rules; what happens with a small $\mathcal{N}_\beta$ **Decided in step 3b (Thom):** stable sorts; equal $g$ at line 18 keeps the generation order (stack index ascending); equal $g$ or $q$ in the selection keeps the order of discovery in $\mathcal{N}_\beta$; if $\lvert\mathcal{N}_\beta\rvert < \lambda$, all nodes are taken. |
| O08 | Which τ children per relocation node in Expansion. Assumption: the τ target stacks with the lowest α (eq. 28). | p. 444 | How the τ children are chosen |
| O09 | (a) The measure used to rank sequences for "the µ best" is not given (sum of α? $g$ of the sequence?). (b) The roulette probability computation is not given (low = good, so inversion is needed: 1/x, max − x, rank-based…). | p. 444 | Ranking measure and probability formula |
| O10 | Eq. 24: the text says "if the stack is empty", the formula means $n(s(c)) = 1$. | p. 443 | Follow the formula? **Decided in step 3 (Thom): (a), the formula.** |
| O11 | Eq. 28 uses $q_b$, $q_c$, $q_r$ of block c only; the text speaks of "the quality measures of the node". | p. 444 | Block-level or node-level? **Resolved in step 3 by Fig. 4 (p. 444, Thom):** block 5 is drawn on the destination stack 3 (grey, $q_b$), the $q_c$ bracket runs between 5 and 4 in stack 3, and the $q_r$ line runs from the retrieval point to stack 3. So $q_b$, $q_c$ and $q_r$ in eq. 28 are those of $c$ on $s'$ after the move (block level). For step 4. |
| O12 | Temporary move: "only a few movements after" has no number. Sec. 6 (p. 443) additionally requires a return to the original stack and speaks of "ruled out"; Sec. 6.3.1 does neither. | p. 443, 444 | Window size; which definition |
| O13 | Fix: meaning of "cannot be completed" (full vs. not touched). Fig. 5 is only feasible if the block's second move is moved earlier. Is reordering allowed? | p. 444–445 | Allowed stacks; whether the code reorders moves; does the solution stay restricted? |
| O14 | Table 3 includes 10×10, which is not in Table 1. | p. 445, 448 | — (documentation only) |
| O15 | Alg. 2: "for (i = 0 to ρ)" is ρ + 1 iterations; the text says ρ times. | p. 443 | Number of iterations in the code |
| O16 | Corrected in step 3: Table 4 (p. 448) shows θ = 0, 10, …, 50 (six values), so "5 × 6 × 4 = 120" combinations is consistent; only the text of Sec. 7.4 lists five θ values. The earlier version of this row was wrong. | p. 447–448 | — (documentation only) |
| O17 | Alg. 1: the retrieval child (lines 14–15) is not tested against ub; nodes in $\mathcal{N}$ are not re-tested after ub drops. In step 1 this can affect which nodes end up in $\mathcal{N}_\beta$. | p. 441–442 | At which point the code prunes **Decided in step 3b (Thom): (i), prune only at line 18** (relocation children); the retrieval child is not tested and nodes are not re-tested when popped. |
| O18 | Minor: the Fig. 1 caption says $n_S = 4$, while the figure shows 6 stacks. | p. 438 | — |
| O19 | Added in step 3. The greedy of Sec. 7.1: which cost "to move a block" decides the destination, how ties are broken, whether it is restricted, and what \|φ\| counts. Eq. 2 and Appendix A call $m'$ the cost of moving a block; Sec. 7.1 says the greedy uses "exclusively the cost to move a block"; Sec. 7.2 takes \|φ\| from "a greedy algorithm". Nothing on ties or on restricted. | p. 437, 445, 446, App. A | **Decided (Thom, step 3):** R1, $m'$ of the relocation only (the first term of eq. 27, drop tier $n(s')+1$). Tentative: ties to the lowest stack index (paper silent); restricted; \|φ\| = relocations + all retrievals including those at the start, excluding the dummy move. Step 3a: with weights where $\eta_1/\gamma_1 = \eta_2/\gamma_2$ (both O01 readings), R1 and R2 (q_m of eq. 27) rank the destinations identically, because both reduce to $(H - t')$ against \|s − s′\| in the same ratio; they differ only for other weights, e.g. the Table 5 time weights. `test_greedy_criterion` pins R1 with asymmetric weights. |
| O20 | Added in step 3. What if $\mathcal{N}_\beta$ is empty (every branch pruned before depth β)? | — (not described) | Open; options for step 3b: the greedy solution as $\phi_{best}$, the root as the only partial solution, or an error. **Step 3b, provisional (Thom):** $\phi_{best}$ is initialised with the greedy solution (p. 441 defines ub as "the objective function value of the best solution found by the algorithm until that moment"; with ub = f(greedy) the greedy is that solution). How often $\mathcal{N}_\beta$ is empty: V11. (L16) |
| O21 | Added in step 3. Well-located: must every block below $c$ have a later retrieval order, or only the block directly below? | p. 437 | **Resolved by p. 437 (Thom):** "above blocks with later retrieval order", non-located = "above another block with earlier retrieval order", and "every non-located block gives rise to at least one relocation": every block below $c$ must have a later retrieval order. The old code checked the blocks above (C10). |
| O22 | Added in step 3. Scope of the sum in eq. 26: all three terms per block, or only $q_b$? | p. 444 | **Resolved (Thom, PDF):** all three terms per block, $q(n) = \sum_c \big(q_b(c) + q_c(c)/n_C + q_r(c)/maxDist\big)$. |

---

## §2 Comparison with the code

Legend: ✅ matches · ⚠️ deviation, documented and acceptable · ❌ deviation, to be fixed · ❓ unclear

| ID | Component | Paper | Check | Code (file:line) | Status | Note |
|---|---|---|---|---|---|---|
| C01 | Parameter mapping | §1.10 | Map `delta`, `num_expansions` and all other parameters to κ, µ, τ, ρ, θ, λ, mq, ε, weights; check default values | `algorithm.py:5-6, 52-56, 264-265, 372, 494-519, 535-545` | ❌ | Mapping by role in the code, not by name. The hypothesis in §1.1 (delta = τ, num_expansions = µ) is refuted. `delta` (3) = depth limit of step 1, i.e. the role of β, but counted in relocations and fixed instead of eq. 29. `max_partial_nodes` (50) = λ (paper 500). `num_expansions` (5) = number of repetitions of step 2, i.e. the role of ρ (paper 300). `mu` (2, UI label "μ") plays the role of κ, but counts loop steps rather than retrievals (C12); the paper's µ (RCL size, 5) does not exist. `branching` (3, argument of `_expand_partial`) = τ by value, but it is never read and is not in the UI. `alpha1, beta1, alpha2, beta2` = η1, γ1, η2, γ2 with defaults 1, 1, 2, 2 (O01, see C02); they still carry the old α/β names. `security_margin` = ε = 2 ✅ (not in the UI). `w_move, w_b, w_c, w_r` = mq_m, mq_b, mq_c, mq_r with 1, 1, 1, 0.5 against the paper's 1, 2, 1, 0.5; `w_c` and `w_r` are never used (C13), and none of them is in the UI. θ and µ are missing. The docstring header lists only `delta` and `num_expansions`. **Step 1:** `CostModel` (operating_cost.py) has the paper's names `eta1, gamma1, eta2, gamma2, epsilon` with the printed values as defaults (O01, suggestion still to be confirmed by Thom); the UI schema is unchanged until step 7. **Step 3:** the `CostModel` default is now (b), η₁ = γ₁ = 1, η₂ = γ₂ = 2 (O01 resolved). **Step 3b:** `delta` is no longer read (β comes from eq. 29); θ is read from `theta` (default 40, not yet in the UI); λ = `max_partial_nodes` with code default 500, while the UI schema still says 50. UI changes in step 7. |
| C02 | Cost function | eq. 1–3 | Empty + loaded trip per move; start at (0, 1); retrieval to (0, 1); tiers per §1.3; weights (O01); $n_T$ (O03) | `algorithm.py:39-74, 333-336, 413-417, 630-643, 715-728` | ❌ | (1) `unloaded_cost`/`loaded_cost` compute $(n_T+\epsilon+t)$ instead of $(n_T+\epsilon-t)$ (lines 60, 64); the docstring (44-48) prints $(n_T+t)$, without ε. (2) The search never evaluates eq. 3: $g$ adds up only $m'$ of relocations. Empty trips are missing, retrievals cost 0, and the start at (0, 1) is not modelled. (3) The reported `operating_cost` (643) is that sum over the step-2 part only (726), so the step-1 moves are missing too. V01: 765 / 807 instead of 531 / 513; the code's own accounting gives 42 / 78, so the ranking of the two Fig. 2 solutions is reversed (§3). **O01:** the code uses the sentence reading (unloaded 1, loaded 2) without documenting it; the unloaded weights never influence a decision, because `unloaded_cost` is never called. **O03:** $n_T$ = `env.config.max_tiers` (562, 689), i.e. **including** the 2 added tiers. `retrieval_cost` (66-68) uses distance 0 to the retrieval point, but is never called. **Step 1:** eq. 1–3 are implemented in `operating_cost.py` (`CostModel.m`, `CostModel.m_loaded`, `Bay.apply`, `f`), with the start at (0, 1), retrievals to (0, 1), pickup tier = height before, drop tier = height after, and $n_T$ as an argument (O03 → including the 2 added tiers). `f` accepts partial solutions (g(n), p. 441) and raises on impossible moves. The search still uses the old `OperatingCostModel` until steps 3–4 replace its callers, so the status stays ❌. |
| C03 | Restricted | p. 437 | Only $c^+$ is relocated, never within its own stack; retrieval only when $c^*$ is on top | `algorithm.py:322-347, 389-420`; `problems/CRP_R.py:395-440` | ✅ | Step 1 and step 2 relocate only the top of the stack of $c^*$, to every non-full stack except the source ($\Upsilon \setminus \{s(c^+)\}$); step 2 retrieves first when $c^*$ is on top (394-396); the env enforces the same rule on replay. Latent: the step-1 root is not auto-retrieved, so if $c^*$ were on top at the root, step 1 would relocate $c^*$ itself (322-339). In `train()` this cannot happen, because the state is built after `env.reset()` has retrieved those blocks (557, 608). **Step 2:** the latent root issue is gone: `train()` reads the layout after `env.reset(skip_auto_retrieve=True)` and `_retrieve_on_top` retrieves $c^*$ while it is on top before the search starts, so the root never has $c^*$ on top. |
| C04 | Instances | p. 445 | Caserta, +2 empty tiers, $d = \lvert s_1-s_2\rvert $, retrieval point stack 0 | `core/benchmarks/caserta.py:154-161, 205-238`; `algorithm.py:77-79, 566-570` | ⚠️ | Caserta via the shared loader; max_tiers = N/S + 2 ✅; $d = \lvert s_1-s_2\rvert$ ✅; retrieval point to the left of the first stack ✅. The code's stack indices are 0-based (flat index = bay − 1), so the paper's stack 1 is code stack 0 and coincides with the retrieval point "stack 0" in `stack_distance(s, 0)`. This only affects $q_r$ and maxDist (C10) and `retrieval_cost` (unused), because no cost term uses a distance to the retrieval point (C02). |
| C05 | $h(n)$ | eq. 21–22 | EstRet / EstRel incl. the exception $s(c)=1$; sum over remaining blocks | `measures.py` (`h`), used in `step1.py:44` | ✅ | Not implemented: there is no EstRet or EstRel. As "h", step 1 uses $q(n)$ (C10), a dimensionless quality score (at most about 3) added to a cost $g$. It is not an estimate of the remaining cost and does not bound it; the comment "admissible enough" (304) is incorrect. **Step 3a:** `measures.h` implements eq. 21–22 with the well-located test of O21 (every block below later), the relocation to tier $n_T$ of $s(c)-1$, or $s(c)+1$ when $s(c)=1$. That exception has no effect on the cost: $m'$ only uses \|s₁ − s₂\|, and from stack 1 both stack 0 and stack 2 are at distance 1. T1 (V08): no state with $h$ above the exact remaining optimum under either weight reading. Not used by the search yet (step 3b), so the status stays ❌. **Step 3b:** used by Algorithm 1 for the pruning test $g + h < ub$ (line 18). Fixed in 2b76a04. |
| C06 | Tree search | Alg. 1 | Children only via $c^+$; sort on $g$; pruning $g+h<ub$ (O17); DFS vs. best-first | `step1.py` (`algorithm1`) | ✅ | Best-first with a priority queue on $g+q$ (`heapq`, 309-354), instead of depth-first with children inserted at the front, sorted on $g$. There is no ub and no $g+h<ub$ test. **O17:** the code prunes nowhere. Instead: (a) once the open list holds more than 4·`max_partial_nodes` entries, the loop over destinations is left with `break` (356-357), so the remaining children are dropped in stack-index order (bias towards low indices); (b) the search stops once more than 2·`max_partial_nodes` nodes have been collected (359-360). Retrieval children: blocks that become retrievable are retrieved inside the relocation child (341-347), which is equivalent to the one-child retrieval branch. Ties: insertion order (heap counter, 353). **Step 3b:** Algorithm 1 as depth-first branch-and-bound: first(𝒩) is popped; case (i) (last block) updates ub and best when $g < ub$; case (ii) puts the retrieval child at the front without a test; case (iii) keeps the relocation children with $g + h < ub$, sorted on $g$ (stable, generation order), cheapest explored next (O17 (i), O07). Without a depth limit it is exact on Fig. 2 and all 40 Caserta 3×3 instances (V10). Fixed in 2b76a04. **Optimisation (after 461c362):** `algorithm1` is now a recursion on one `Bay` with apply/undo (`Bay.undo`), and h is updated per move with the cached per-block term `measures.block_term` (a relocation changes only the moved block's term, a retrieval removes c*'s term; the blocks below keep their well-located status). The children of a node are still generated, tested with the ub of that moment and sorted before any is explored, so the order and the pruning are those of Algorithm 1. V12: $\mathcal{N}_\beta$ (nodes, order, g), best leaf and final ub identical to 2b76a04 on all 41 V11 instances that finished under (c). The incremental h adds the same terms in a different order than `measures.h`; with integer-valued weights (both O01 readings) the values are exact, with other weights a rounding difference could in principle flip a tie in $g + h < ub$. |
| C07 | Greedy | p. 445 | Greedy on relocation cost; delivers $\lvert \phi\rvert $ (in moves) and ub | `measures.py` (`greedy`), used in `algorithm.py` `train` | ✅ | Not implemented. Nothing computes the greedy solution, its length or a ub. **Step 3a:** `measures.greedy` (Sec. 7.1, O19 reading R1: lowest $m'$ with drop tier $n(s')+1$, ties to the lowest stack index, restricted); returns the moves and their cost, i.e. \|φ\| and ub. Not used by the search yet (step 3b), so the status stays ❌. **Step 3b:** gives ub (O04) and \|φ\| for β; ties tentative (O19). Fixed in 2b76a04. |
| C08 | β | eq. 29–30 | floor; $\lvert \phi\rvert $ in moves; which $p(n_C)$ (O02) | `measures.py` (`p`, `beta`), used in `algorithm.py` `train` | ⚠️ | β is the fixed parameter `delta` (default 3, UI range 1–10), and depth increases only per relocation (354), so it counts relocations, not moves. Eq. 29–30 are not implemented, so O02 is not resolved by the code. **Step 3a:** `measures.p` (eq. 30 as printed, provisional, O02) and `measures.beta` (eq. 29, floor, exact fractions, \|φ\| in moves, θ = 40 by default). V09 tabulates the three readings of eq. 30. Not used by the search yet (step 3b), so the status stays ❌. **Step 3b:** β from eq. 29 with θ = 40 (`theta`, not yet in the UI) is now the depth of step 1; eq. 30 is literal and provisional until O02 is decided (V11). **O02 decided: (c)**, eq. 30 with the end values swapped, as a documented correction of a presumed misprint (L19); the status stays ⚠️ because it is a documented deviation from the printed formula. |
| C09 | Step 1 | Sec. 6.1 | Depth limit β; $\mathcal{N}_\beta$ = depth-β nodes only; ub (O04); leaves (O05) | `step1.py` (`algorithm1` with `beta`), `algorithm.py` `train` | ✅ | Collected: nodes at depth `delta` **and** nodes in which every block has been retrieved (323-326). **Search nodes store only the state, not the moves from the root**, so the relocations of step 1 cannot be put into the plan; the comment at 586-589 acknowledges this. The step-2 destinations are then replayed from the initial state instead of from the partial state: this is the cause of the truncated plans (210 of 210 earlier; here 15 of 15 runs = 5 instances × 3 seeds, §3). **O04:** no ub at all; growth is limited by the caps in C06 instead. **O05:** complete solutions found within the depth limit are treated as partial solutions and their moves are lost: on Fig. 2 every collected node is complete and the returned plan is empty with cost 0 (V01b). Step 1 runs 1 + `num_expansions` times instead of once (575 and 704 via 618). **Step 2:** `_SearchState` now carries φ′ (`phi`, all moves from the root, retrievals included), so the step-1 prefix and complete nodes are kept (O05 (b); the ub part waits for step 3). Depth, ub and the selection are unchanged, so the status stays ❌. **Step 3b:** step 1 runs once from the real initial configuration; depth = moves incl. retrievals; a node at depth β is collected when popped; a leaf at any depth ≤ β updates ub/best (O05); ub = f(greedy) (O04). Its β is provisional (C08). Fixed in 2b76a04. |
| C10 | $q(n)$ | eq. 23–26 | Three terms, normalization $n_C$ and $maxDist$, no mq weights; eq. 24 (O10) | `measures.py` (`q`), used in `step1.py` (`select`) | ✅ | (1) $q_b$: `is_well_located` checks the blocks **above** $c$ instead of below (132-134), so the test is inverted. Stack [5, 1] (bottom → top): the code marks 5 non-located (paper: well-located); stack [2, 6]: the code marks 6 well-located (paper: non-located). (2) $q_c$: the first branch of eq. 24 is used for every block (243-246); a block alone in its stack gets 0 instead of $r(c)-r(c^*)$. **O10:** the code follows the literal text ("if the stack is empty" never holds for a stack that contains $c$, so the second case never applies), not the formula. (3) $q_r$: 0-based stack, so $d = s-1$, and maxDist = $n_S-1$ instead of $n_S$ (230, 249). (4) Each of the three sums is also divided by the number of remaining blocks (252-256), which makes them averages; eq. 26 uses sums. $n_C$ = largest remaining priority (229), which always equals $n_C$ ✅. No mq weights ✅ (259). **Step 3a:** `measures.q` implements eq. 23–26: the well-located test of O21, eq. 24 per the formula (O10 (a): a block alone in its stack gets $r(c) - r(c^*)$), $q_r = s$ with the paper's stack numbering, maxDist = $n_S$, sums over the blocks (O22), $n_C$ = the number of blocks of the instance, no mq weights. The old `_quality_node` is still used by the search. Not used by the search yet (step 3b), so the status stays ❌. **Step 3b:** used in the selection of $\mathcal{N}_q$; `_quality_node` is removed (the old step-2 score now uses `measures.q` too). Fixed in 2b76a04. |
| C11 | Selection | Sec. 6.1 | $\mathcal{N}_g$ lowest $g$ (not $g+h$), $\mathcal{N}_q$ lowest $q$, union; λ reading (O06); ties (O07) | `step1.py` (`select`) | ✅ | One list: the `max_partial_nodes` nodes with the lowest $g+q$. No $\mathcal{N}_g$, no $\mathcal{N}_q$, no union. **O06:** the code follows neither reading ($\lvert\mathcal{N}\rvert \le \lambda$, but with one ranking). **O07:** ties by insertion order (heap counter, stable sort); if $\mathcal{N}_\beta$ is smaller than λ, all of it is returned; if it is empty, the initial state is used (576-577). **Step 3b:** $\mathcal{N}_g$ = λ nodes with the lowest $g$, $\mathcal{N}_q$ = λ nodes with the lowest $q$, union in that order without duplicates (O06 (a), O07); checked by hand in `test_select_by_hand`. Fixed in 2b76a04. |
| C12 | Expansion | Sec. 6.2 | Branching ≤ τ (O08); stop at κ retrievals; repeat until complete | `algorithm.py:371-430, 594-601, 717-722` | ❌ | No subtree: the single relocation with the lowest score is taken (409-411, comment "we take best for reproducibility"); `branching` is never read. **O08:** not implemented. Stop: after `mu` loop steps (385), where a step is one retrieval, or one relocation plus the retrievals it enables, so not after κ retrievals. Repeated until complete, but the loop also stops as soon as a round contains no relocation (`if not seq: break`, 600-601, 721-722), which would leave blocks behind. Not observed on the 4 instances checked (§3, V03): step 2 always ran to the end. **Step 3b (throwaway until step 4):** the old expansion works on `Bay` nodes; its score uses `measures.q` instead of the removed `_quality_node`, and `_construct_solution` first retrieves $c^*$ while it is on top, because Algorithm 1 can collect nodes with $c^*$ on top and the old loop then stopped after a round without relocation, leaving an incomplete φ (caught by `validate_plan`). |
| C13 | α | eq. 27–28 | $q_m$ incl. empty return trip; $maxMov$; block-level terms (O11); weights | `algorithm.py:262-289` | ❌ | $q_m$ = the loaded trip only (274-278), without the empty return trip of eq. 27, and with the sign error from C02. Tiers: pickup = height before the move, drop = height + 1 ✅. maxMov = `loaded_cost(nT, nT, nT, nS − 1)` (286), a constant, not the maximum over the possible relocations of this block. Quality term = $q(n)$ of the whole resulting node (281-283). **O11:** node level, with the normalization from C10, weighted as a whole with `w_b` = 1; mq_b = 2, mq_c and mq_r are not applied (`w_c` and `w_r` are unused, 289). Lower is better ✅ (ascending sort, 409). |
| C14 | Selection of sequences | Sec. 6.2 | µ best (measure, O09a); roulette (formula, O09b); seed / reproducibility | `algorithm.py:409-411, 553-554` | ❌ | No µ best and no roulette wheel: step 2 is deterministic. The seed is set (`np.random.seed`, 554), but nothing random is drawn, so every seed gives the same solution (V04). **O09a/b:** not implemented. |
| C15 | Main loop | Alg. 2 | Step 1 once; ρ iterations (O15) starting from the same $\mathcal{N}$; best by $f$ | `algorithm.py:574-632, 674-730` | ❌ | (1) The loop in `train()` (582) calls `_construct_solution` (618) on every iteration, which runs step 1 again (704) and has its own `num_expansions` loop (711): step 1 runs ρ + 1 times, step 2 ρ² times. Everything is deterministic, so every outer iteration returns the same solution. (2) Each iteration completes **one** partial solution, round robin (`partials[i % len]`, 584, 712), instead of all of $\mathcal{N}$; with 5 iterations only the first 5 partial solutions are ever completed. (3) The best is chosen on the step-2 relocation cost (726), not on $f(\phi)$ of the complete solution. (4) Dead code: 584-604 (`collected_dests`, `cleaned` unused) and 609-627 (the `env2` replay, whose `actions` and `total_cost` are unused). **O15:** `range(num_exp)` = ρ iterations (the text's reading), nested as in (1). **Step 2:** `_construct_solution` now returns φ instead of env actions; the selection is still on the old step-2 relocation cost, and the dead code in `train()` (first pass) stays until step 5. The env replays of the old code are gone. **Step 3b:** step 1 now runs once (Alg. 2 line 2); $\phi_{best}$ starts as the greedy solution (O20) and the best step-1 leaf is a candidate (O05); the final choice among greedy, leaf and expansion result is on $f$ (Alg. 2 line 10, ties keep the earlier one). The selection among the expansions is still on the old step-2 cost and the loop is still nested, so the status stays ❌. |
| C16 | Postprocessing | Sec. 6.3 | Temporary moves only; window (O12); allowed stacks (O13); reordering; repeat until no improvement; feasibility and restricted maintained | `algorithm.py:437-452` | ❌ | Not implemented: `_remove_temporary_moves` returns its input unchanged (comment: "safe no-op for now"), while the description claims step 3 (478). **O12/O13:** not addressed. |
| C17 | Output / metrics | §0 | Objective value, number of relocations (secondary), time; UI shows "restricted" | `algorithm.py:483-486, 634-668`; `core/algorithm_meta.py` | ❌ | The metrics come from replaying the plan on the env (objective_value under the selected time model, relocations, crane_time_f2, crane_time_vertical, plus `operating_cost` in `extra`) ✅. But the plan is incomplete on every instance run (C09), so the metrics cover only part of it while the job reports completed; on Fig. 2 every metric is 0. A second final record repeats the first (665-668; mean over the `n_seeds` = 1 leftover). `objectives = ["crane_time"]` (485) is not true of the current search (it uses its own cost, C02), and `fidelity = "faithful"` (486) is not true either. Relocation rule in the UI: the platform has no metadata field for it. CRP-Time only allows restricted moves (`CRP_R.step`), so the rule follows from the problem but is not shown per algorithm (platform point). **Step 2:** the plan is φ_best itself, converted by `_plan_from_phi` and checked by `validate_plan` on a fresh env (an infeasible or incomplete plan raises `RuntimeError`); the metrics are those of the validated plan, and `operating_cost` = f(φ_best) with $n_T$ = `max_tiers` and the printed weights. The duplicate record, the metadata and the relocation-rule display remain, so the status stays ❌. |
| C18 | Code style | §0 | No unused UI parameters/metrics; comments correct; objective clearly named | whole file | ❌ | Unused UI parameters: `max_iterations` and `report_interval` (base schema), and `seed` has no effect (C14). Unused code: `_build_initial_search_state` (193-206), `clone_after_forced_move` (159-178), `non_located_blocks` (137-142), `retrieval_cost` (66-68), `crane_to_pickup_unloaded` (70-74), the `cost_model` argument of `_quality_node`, the `n_seeds` loop, and the dead code from C15. Incorrect docs and comments: the author list is missing E. Lalla-Ruiz and B. Melián-Batista (9-10); the docstring cost formula is wrong (44-48); "α/β weights" (476); the description claims step 3 (478); "admissible enough" (304); "paper uses a composite; we keep simple" (258); "the paper mixes them; we use a weighted sum" (289); the UI label "Blocks per expansion (μ)" (512) conflicts with the paper's µ; "advances the next mu targets" (374-375). There is no "3-bay" reference in this folder. **Step 3b:** removed `_SearchState` (incl. `clone_after_forced_move`, `non_located_blocks`), `_build_initial_search_state`, `_quality_node`, `_limited_a_star` and the dead first pass in `train`. |

---

## §3 Validation in the paper's setting

| ID | Test | Expected | Result |
|---|---|---|---|
| V01 | Cost function on the Fig. 2 instance (§1.3), $n_T = 4$, $\epsilon = 2$, no extra tiers | min-moves solution: 531 (14 moves); min-cost solution: 513 (15 moves) | ❌ The code has no eq. 3 evaluator, so eq. 3 was assembled from the code's `unloaded_cost`/`loaded_cost` (stack distance \|s₁ − s₂\|, retrieval point stack 0, start (0, 1)). Result: **765** (min-moves) and **807** (min-cost), under both weight readings. The code's own accounting (sum of `loaded_cost` over the relocations, as in step 1 and step 2) gives 21 / 39 with the printed weights and **42 / 78** with the code's defaults. Both accountings reverse the ranking: the paper's cheaper solution (513) comes out more expensive. A reference implementation of eq. 1–3 as in §1.3 gives 531 / 513 under both readings, which confirms the reading in §1.3. **Step 1:** `f` in `operating_cost.py` gives 531 (14 moves) and 513 (15 moves) (`test_lopez_plata.py`, `test_fig2`). |
| V01b | The whole algorithm on the Fig. 2 instance (`max_tiers` = 4; default parameters; seeds 0, 1, 7) | A complete plan; f ≥ 513 | ❌ Empty plan, `operating_cost` = 0, f2 = 0, 0 of 13 blocks retrieved. With `delta` = 3, every node collected in step 1 is already complete, and the moves that led there are not kept (C09, O05). **Step 2:** complete plan, accepted by `validate_plan`. In this run the old search returned the min-moves plan (relocate 8 from stack 1 to 6) with `operating_cost` = 531 and f2 = 535.2. That choice is an accident, not evidence about the search: on Fig. 2 every node collected in step 1 is already complete, so each gets old step-2 cost 0 and the first one wins. f2 was recomputed in the test with the f2 formula per move (retrieval from stack s: 2·ts·s + tpp; relocation s → s′: 2·ts·\|s − s′\| + tpp; ts = 1.2 s, tpp = 30 s; `_f2_formula` in `test_lopez_plata.py`), independently of the platform evaluator, on the executed plan; it was not checked by hand. Since 3954b71 the tests no longer assert which plan is chosen, only completeness, `validate_plan`, `operating_cost` = f(φ) on a fresh `Bay`, and f2 = the formula (`test_fig2_complete`, `test_fig2_stack_mapping`). |
| V02 | Small hand-made instance where one weight reading gives a different f (settles whether the code follows O01 consistently) | — | **Not possible with restricted plans.** Claim (Thom, 29 Sept 2026): under the restricted rule, with the start and every retrieval at (0, 1), the unweighted empty and loaded sums are equal, vertically ($V$, the terms $n_T+\epsilon-t$) and horizontally ($H$, the distances), so $f = (\eta_1+\eta_2)V + (\gamma_1+\gamma_2)H$, and O01 cannot change $f$ for any restricted plan. Test (scratch script, eq. 1–3 under the §1.3 reading): 15 000 random restricted plans (random destination per blocker), 1000 each on Caserta shapes 3×3, 3×8, 4×4, 4×7, 5×4, 5×10, 6×6, 6×10, 10×6 and 10×10 (full stacks, capacity $H'+2$) and 5000 on random shapes (2–8 stacks, 2–7 tiers, at least $n_T-1$ empty slots), each with random weights in [0.1, 5]; per plan it checks the equality of both sums, $f$ against the closed form, and $f$ with the weights swapped. **0 counterexamples.** Why: vertically the empty and loaded terms differ only by (start tier) − (drop tier of the last move) = 1 − 1; horizontally, every relocation from $s$ is followed by a move from $s$ again, and each target is reached from, and retrieved to, the retrieval point from the same stack. Control: random *unrestricted* plans do give $E_h 
e L_h$ (5937 of 20 000), so the check can fail. Consequence: O01 cannot be decided through $f$; it still matters for $h$ (eq. 21–22) and $q_m$ (eq. 27). The equality in Fig. 2 (§1.3, "happen to be equal") is structural. What remains for V02: see the step-1 proposal (a unit test of $m$ and $m'$ with asymmetric weights, and a V02 on $h$ and $q_m$ in the steps that implement them). |
e L_v$, 0 with $E_h 
e L_h$. As a control, 20 000 unrestricted plans: 0 with $E_v 
e L_v$, 5937 with $E_h 
e L_h$ (smallest example: stacks [3], [], [1, 2], [], [], $n_T$ = 3, moves (1,2), (3,2), (3,0), (2,0), (2,0): $E_h$ = 7, $L_h$ = 9), but such plans are not allowed here. Consequence: O01 cannot be decided through $f$. It does matter for the search, through $h$ (eq. 21–22: $m$ has only vertical terms there, $m'$ goes to the retrieval point) and $q_m$ (eq. 27). V02 therefore moves to the step that implements those, as a test on $h$ and $q_m$ rather than on $f$. The equality in Fig. 2 (§1.3) is structural, not a coincidence. |
| V03 | A few representative Caserta instances (e.g. one per type) with the paper's parameters | Feasible and restricted; order of magnitude compared with Table 3 (orientation only) | ❌ The paper's parameters cannot be set (θ, µ, τ, mq and the λ/ρ reading are missing, C01), so this ran with the defaults, f2, seeds 0, 1 and 7, on data3-3-1, data4-5-1, data5-7-1 and data6-10-1 (12 runs; the three seeds give identical results, V04). Blocks retrieved: 2/9, 5/20, 11/35, 11/60; none of the plans is complete. `operating_cost` 126, 1016, 1956, 4802, against f2 over the executed part: 132.0, 568.8, 1377.6, 2365.2. Consistent with the earlier measurement on 210 instances (`docs/validation/data/caserta_crp_time/subset_lopez_plata.jsonl`: 210 of 210 incomplete). The relocations that were executed are restricted (enforced by the env). A comparison with Table 3 is not possible. Diagnosis (C09): the partial states from step 1 already have 1–3 blocks retrieved (data3-3-1: 1 of the 8 left after reset; data4-5-1: 3 of 20; data6-10-1: 1 of 60), and step 2 completes each of them internally (0 blocks left). The destinations it produces, however, are replayed on the env from the initial state, without the step-1 moves. **Step 2:** see V07; plans are now complete. A comparison with Table 3 still needs steps 3–7. |
| V04 | Reproducibility: same seed gives the same solution | Identical | ✅ trivially, but for the wrong reason: seeds 0, 1 and 7 give identical plans on all five instances (Fig. 2 and the four above), because there is no random component (C14). The paper's roulette wheel, and with it the seed dependence, is missing. |
| V05 | Postprocessing never increases f and preserves feasibility and restrictedness | f_after ≤ f_before | Trivially true: postprocessing is a no-op (C16). Nothing is tested. |
| V06 | O03 orientation: exact restricted optimum of eq. 3 on the 40 Caserta 3×3 instances (capacity 3 + 2 = 5 in both runs; only $n_T$ inside eq. 1–2 varies; ε = 2; printed weights, which do not affect f, V02) | Table 3 (p. 448), 3×3: average 497.40 (the heuristic's average, orientation only) | The loader has 40 instances (data3-3-1 … data3-3-40). Average optimum with $n_T$ = 5 (including the 2 added tiers): **497.40** (min 351, max 585). With $n_T$ = 3: **329.10** (min 243, max 381). Scratch Dijkstra over (configuration, crane position), 29 Sept 2026. The first number equals the printed Table 3 value; this is orientation, not a decision on its own, because Table 3 reports the heuristic's average, not the optimum. |
| V07 | Step 2: completeness and feasibility on the 210 instances of subset_lopez_plata (Caserta 1–10 per class, 21 classes, f2, default parameters) | Every plan complete and feasible; reported f2, f2vert and `operating_cost` equal to independent recomputations | **210 of 210** pass (29 Sept 2026, `docs/validation/data/lopez_plata_2019/step2_complete.py`, output `step2_complete.jsonl`). Per instance: `get_best_solution()` replayed with `validate_actions` (the env's own auto-retrieval) feasible; the executed plan accepted by `check_plan` and `validate_plan`; all blocks retrieved; f2 and f2vert recomputed per move with `movement_f2` / `movement_f2_vertical` equal to the reported values; `operating_cost` equal to `f` on the executed moves. No run raised. 48 of the 210 instances (70 containers) have containers on top at the start; their retrievals are in φ. The early break in step 2 (`if not seq: break`, C12) did not occur. Total run time 292 s, at most 20 s (10×10). |
| V08 | T1: $h$ (eq. 21–22) ≤ exact remaining optimum in every reachable state (configuration + crane position), both O01 readings | No state with $h > V$ (admissibility claim, p. 442–443) | 29 Sept 2026, `docs/validation/data/lopez_plata_2019/h_bound.py` and `h_slack.py` (local); the unit test `test_h_lower_bound` covers Fig. 2 and Caserta 3×3. Exact remaining optimum by memoised recursion over the restricted state graph (acyclic). **0 violating states under both readings**: Fig. 2 (67 states), Caserta 3×3 (40 instances, 5 817 states), random full 3×4 (20, 184 985), 3×5 (10, 5 125 899), 4×4 (3, 1 036 742); capacity $H'+2$, Fig. 2 $n_T$ = 4. Thom's suspicion (a non-located block relocated to a higher tier, while EstRet prices the retrieval at its original slot) was checked separately on the 14 379 states of Fig. 2, Caserta 3×3 and random 3×4 in which an optimal next move puts the block on a higher tier: smallest slack $V - h$ = 31 under (b) and 62 under (a); $h/V$ at most 0.862 (b) and 0.663 (a). The check can fail: $1.3\,h$ violates in 1 899 states of Fig. 2 + Caserta 3×3, and $h$ with the old (inverted) well-located test also violates. The only states with $h = V$ are the empty end states. |
| V09 | O02: $p(n_C)$ and β (eq. 29, θ = 40, \|φ\| from the greedy, R1) for every Table 1 size under (a) as printed, (b) $0.5 + (n_C-9)/16$, (c) the printed ramp with the end values swapped | — (orientation for O02) | 29 Sept 2026, `beta_table.py` (local), all 40 Caserta instances per size, β averaged per instance. (a) and (b) agree for $n_C \le 9$ and $n_C \ge 25$; between them (b)/(a) ranges from 0.51 (3×4) to 2.65 (3×8). (c) differs from (a) only outside the ramp: 3.24× larger on 3×3, 0.33× on every size with $n_C \ge 25$. Under (a) and (b), β exceeds $n_C$ on every size with $n_C \ge 25$ (e.g. 5×5: β ≈ 41.8, $n_C$ = 25; 6×10: β ≈ 121.0, $n_C$ = 60; average \|φ\| 70.3 and 202.3), so complete solutions can lie within depth β there (O05), and step 1 becomes very deep. Under (c), β stays below $n_C$ on those sizes. |
| V10 | Step 3b: Algorithm 1 without a depth limit, ub = f(greedy): min(f(best), ub) against the exact optimum, both weight readings | Equal on every instance; Caserta 3×3 average 497.40 (V06, Table 3) | Equal on Fig. 2 (513) and on all 40 Caserta 3×3 instances; average **497.40** under both readings (`test_algorithm1_exact`). Mutation checks (29 Sept 2026, scratch): the test fails when the pruning overestimates (2h, 1.3h), when leaves do not lower ub, and when a destination stack is skipped; `test_n_beta_reference` also catches testing the retrieval child (O17 (iii)), re-testing at pop (O17 (ii)), the most expensive child first, depth counted in relocations, and leaves collected in $\mathcal{N}_\beta$. Not caught by any test: accepting a leaf on $g \le ub$ instead of $g < ub$ (it only differs when a leaf costs exactly ub, and the value found is the same). |
| V11 | Step 3b: step-1 runtime per Table 1 size under the three readings of eq. 30 (O02), θ = 40, weights (b) | — (input for O02 and the step-1 limit) | 29 Sept 2026, `step1_runtime.py` (local, output `step1_runtime.jsonl`): Caserta instances 1–5 per size, Algorithm 1 to depth β in a separate process with a 60 s wall-clock guard; after the first time-out the rest of that size and reading is skipped; runs with equal β are shared between readings. Python 3, one core. **$\mathcal{N}_\beta$ was never empty and no leaf was reached** in any of the 100 completed runs (leaves only become reachable at depths where the runs time out). Per size, with (a) = (c) unless listed: 3×3 (a/b: β 2–3, \|𝒩_β\| ≤ 4; c: β 6–9, ≤ 32; all 0.00 s); 3×4 (a: β 8–11, ≤ 954, ≤ 0.02 s; b: β 4–6, ≤ 57); 3×5 (a: β 11–15, ≤ 441 469, max 8.8 s; b: β 8–12, ≤ 26 004, max 0.6 s); **3×6: time-out under all three** (β 19 / 22 / 19); 4×4 (a: β 11–18, ≤ 67 433, max 1.5 s; b: β 9–16, max 0.3 s); **4×5: time-out under all three** (β 16 / 24 / 16); 3×7 (a: 4 of 5, β 9–13, max 1.5 s, 5th times out at β 13; b: 9.5 s at β 16, time-out at β 17); 3×8 (a: 5 of 5, β 9–11, ≤ 161 938, max 4.5 s; b: time-out at β 26); 4×6 (a: 4 of 5, max 4.4 s, time-out at β 14; b: time-out at β 30); 4×7 (a/b: time-out at β 37; c: 41.3 s at β 12 with \|𝒩_β\| = 1 019 195, time-out at β 14); 5×4 (a: 5 of 5, β 13–20, ≤ 311 821, max 9.4 s; b: time-out at β 26); 5×5 (a/b: time-out at β 39; c: 9.3 s and 0.03 s at β 13 and 12, time-out at β 17); **5×6, 5×7, 5×8, 5×9, 5×10, 6×6, 6×10: time-out under all three** (a/b β 51–118; c β 17–39). So with the literal eq. 30 step 1 does not finish within 60 s from 3×6 upwards on at least one instance, and on every size with $n_C \ge 25$; (c) finishes on more medium instances but also times out from 5×6 upwards. For comparison only (Java, other hardware, averaged over instances not stated here): Table 4 (p. 448) gives an average $T_{A^*}$ of 283.63 s at θ = 40. |
| V12 | Optimised `algorithm1` against the step-3b version (2b76a04) | Identical $\mathcal{N}_\beta$ (same nodes, same order, same g), best leaf and final ub | **Identical on 41 of 41** V11 instances that finished under (c) (29 Sept 2026, `step1_equivalence.py`, local; the old version is loaded from git). Speed-up 1.4–2.7× in these runs (e.g. 4×7-1: 30.3 s → 13.4 s; 5×5-1: 7.4 s → 3.4 s); with a very large $\mathcal{N}_\beta$, copying the collected nodes now dominates. Unit tests (exactness 497.40 / 513, $\mathcal{N}_\beta$ against brute force and the recursive transcription, determinism) pass unchanged. |
| V13 | Step-1 runtime after the optimisation, O02 (c), instances 1–2 per size, 300 s guard; peak memory of the worker | — (input for the step-1 limit) | 29 Sept 2026, `step1_runtime.py` (local, output `step1_runtime_c300.jsonl`). A first attempt without a memory limit reached a working set of 24.5 GB on 3×6-1 (the machine has 31.7 GB), so a memory limit of 8 GB was added next to the 300 s guard. **Every stop was the memory limit, after 57–84 s; none reached 300 s.** Finished: 3×3, 3×4 (≤ 0.01 s, < 40 MB); 3×5 (0.05 s and 2.3 s, \|𝒩_β\| 204 787, 251 MB); 4×4 (0.8 s); 4×5 (20.7 s with \|𝒩_β\| = 2 069 496 and 2.25 GB; 0.6 s); 3×7, 3×8, 4×6 (≤ 1.7 s, ≤ 244 MB); 4×7-1 (12.1 s, \|𝒩_β\| = 1 019 195, 1.31 GB); 5×4 (≤ 3.2 s); 5×5 (2.6 s and 0.01 s). Stopped at 8 GB: 3×6-1 (β 19), 4×7-2 (β 14), and the first instance of 5×6, 5×7, 5×8, 5×9, 5×10, 6×6, 6×10 (β 17–39). About 1.1 KB per collected node (2.07 million nodes ≈ 2.25 GB), so the stops correspond to roughly 7 million collected nodes. 𝒩_β was never empty and no leaf was reached in any finished run. |

Note on V01: under both weight readings the totals in this example are the same (531 / 513), so this test checks the structure of eq. 3 but not O01. Hence V02.

Run count: V01b, V03 and V04 are the same 15 `train()` runs: 5 instances (Fig. 2, data3-3-1, data4-5-1, data5-7-1, data6-10-1) × seeds 0, 1, 7. All 15 plans are incomplete.

How the runs were made (29 Sept 2026; nothing in the algorithm changed; scratch scripts, not in the repo): V01 imports `OperatingCostModel` and `stack_distance` and applies them move by move to both Fig. 2 solutions. V01b/V03/V04 call `LopezPlata2019Heuristic.train()` with the schema defaults plus `seed`, on `CRP_Time` with `time_model = f2` (Fig. 2 as a `.dat` file with `max_tiers` set to 4, because `implied_tier_capacity` gives 3 for 13 blocks on 6 stacks), and replay `get_best_solution()` on a fresh env. The diagnosis in V03 calls `_limited_a_star` and `_expand_partial` directly.

---

## §4 Conversion to crane time only (f2 / f2vert) (to be filled in)

Every place where the objective enters the logic has to switch to f2 / f2vert (Voß & Schwarze 2019, as implemented in CRISP). Moves and relocations remain wherever the paper uses them.

| Location | Paper | Uses operating cost? | Conversion | Deviation to document |
|---|---|---|---|---|
| $g(n)$ | p. 441 | yes (eq. 3) | crane time of the partial solution | |
| $h(n)$ | eq. 21–22 | yes (m, m′) | lower bound for crane time; check admissibility again | If $h$ is not admissible, step 1 prunes more aggressively |
| Initial ub / greedy | p. 445 | yes | greedy on crane time; $f$ = crane time | |
| β via $\lvert \phi\rvert $ | eq. 29 | indirectly (the greedy chooses on cost) | $\lvert \phi\rvert $ stays in moves | $\lvert \phi\rvert $ can change because the greedy chooses differently |
| $\mathcal{N}_g$ | p. 443 | yes ($g$) | as $g(n)$ | |
| $q(n)$ | eq. 23–26 | no ($q_r$ is distance) | unchanged | |
| $q_m$, $maxMov$ | eq. 27–28 | yes | crane time of the relocation plus the return trip | |
| µ ranking | p. 444 | depends on O09a | | |
| $f(\phi) < f(\phi_{best})$ | Alg. 2, line 10 | yes | crane time | |
| Postprocessing criterion | p. 444 | yes | crane time | |

---

## §5 Deviations and assumptions

Labels (validation workflow, step 3): (a) required by the unified 2D setting, (b) a choice the paper leaves open, (c) a real deviation from the paper. "No reason given" means the code states none.

| ID | Deviation / assumption | Reason | Impact | Relates to |
|---|---|---|---|---|
| L1 | The step-1 moves are not kept: search nodes store only the state, and the plan consists of the step-2 destinations alone, replayed from the initial state. Complete solutions found within the depth limit are treated as partial solutions and lost. | (c). The comment at 586-589 calls it "simplified". | **Every plan is incomplete** (210 of 210 earlier; here 15 of 15 runs = 5 instances × 3 seeds, §3). The metrics cover only the executed part while the job reports completed; on Fig. 2 the plan is empty and f = 0. | C09, C15, C17; O05 **Resolved in step 2**: `_SearchState` carries φ′, the plan is φ_best, and complete nodes keep their moves (O05 (b)); V07: 210 of 210 complete and feasible. Fixed in 7b6e41f. |
| L2 | Cost function: $(n_T+\epsilon+t)$ instead of $(n_T+\epsilon-t)$; the search and `operating_cost` count only the loaded trip of relocations (no empty trips, no retrievals, no start at (0, 1)); `operating_cost` also excludes the step-1 part. | (c), no reason given. | Every cost-driven decision (order in step 1, choice in step 2, choice of the best solution) uses a different cost. On Fig. 2 the ranking of the two paper solutions is reversed (V01). The reported `operating_cost` is not the paper's $f$ of the returned plan. | C02, C13, C15; O01 Step 1: a correct eq. 1–3 now exists in `operating_cost.py`; the old model stays in use until steps 3–4. **Step 2, partly resolved:** the reported `operating_cost` is now f(φ_best) (eq. 3, $n_T$ = `max_tiers`, printed weights). The decisions (step-1 order, step-2 choice, choice of the best solution) still use the old cost; that is steps 3–5. |
| L3 | Weights default to unloaded 1, loaded 2 (the sentence on p. 445), not the printed 2 / 1; undocumented; still named α/β. | (b) given O01, but the choice is not documented. | Scales $m'$ only; the unloaded weights are never used (L2). | C01, C02; O01 Step 1: the new `CostModel` defaults to the printed values (2, 2, 1, 1) with the paper's names; this is a suggestion that Thom still has to confirm. `f` does not depend on it (V02); step 3 checks whether $h$ is a lower bound under both readings. The old UI defaults (1, 1, 2, 2) remain until step 7. **Step 3:** O01 resolved in favour of (b) (Table 5, Sec. 7.7); the `CostModel` default is now η₁ = γ₁ = 1, η₂ = γ₂ = 2. The old UI defaults (1, 1, 2, 2) happen to match; the old names α/β remain until step 7. |
| L4 | $n_T$ in the cost includes the 2 added tiers (`max_tiers`). | (b): the paper does not say which $n_T$ it means. | Shifts every vertical term by 2·η; affects comparisons with Table 3 only. | C02; O03 Decision (Thom, 29 Sept 2026): (a), $n_T$ including the 2 added tiers, passed to `f` as an argument; it is also the stack capacity, as the paper uses one symbol for both (p. 437). V06: with this choice the exact 3×3 optimum averages 497.40, the Table 3 value. V06 gives the exact optimum, whereas Table 3 gives the heuristic's average, so the match is supporting evidence for O03 (a), not proof. |
| L5 | No $h$ (eq. 21–22), no ub, no greedy, no β by eq. 29–30. Step 1 is a best-first search on $g+q$ with a fixed depth `delta` in relocations, and with caps that cut off children by stack index and stop the search early. | (c), no reason given. | Step 1 selects differently from the paper, and the set of partial solutions depends on the caps and on stack order instead of on the bound. | C05–C09; O02, O04, O17 **Step 3a:** $h$, the greedy (O19 R1) and β (eq. 29–30 as printed) now exist as pure functions in `measures.py`, with tests; the search does not use them until step 3b. **Step 3b:** resolved: Algorithm 1 with $h$, ub = f(greedy) and β by eq. 29 is the new step 1; only eq. 30 is provisional (O02). Fixed in 2b76a04. |
| L6 | Selection: one list of `max_partial_nodes` nodes with the lowest $g+q$ instead of $\mathcal{N}_g \cup \mathcal{N}_q$. Ties by insertion order. | (c) for the list; (b) for the ties. | The forward-looking set $\mathcal{N}_q$ is missing as a separate source of starting points. | C11; O06, O07 **Step 3b:** resolved (O06 (a), O07). Fixed in 2b76a04. |
| L7 | $q(n)$: the well-located test is inverted (it checks the blocks above instead of below); $q_c$ gives a block alone in its stack 0; $q_r$ and maxDist are off by one (0-based stacks); each term is averaged over the remaining blocks. | (c), no reason given. | $q$ ranks configurations differently in step 1 and in step 2; $q_b$ even penalizes well-located blocks under blocks with an earlier retrieval order. | C10, C04; O10 **Step 3a:** a corrected $q$ exists in `measures.py` (O21, O10 (a), O22, paper stack numbering); the old `_quality_node` is still used by the search until step 3b. **Step 3b:** resolved; the corrected $q$ is used in the selection and in the old step-2 score. Fixed in 2b76a04. |
| L8 | Expansion: no subtree with branching τ, no µ best, no roulette wheel; the single relocation with the lowest score is taken. It stops after `mu` loop steps instead of κ retrievals, and breaks off when a round has no relocation. | (c). Comment at 410: "here we take best for reproducibility". | Step 2 is a deterministic greedy; ρ repetitions add nothing and the seed has no effect (V04). The early break is a latent source of incomplete plans (not observed). | C12, C14; O08, O09 |
| L9 | α (eq. 27–28): $q_m$ without the empty return trip; constant maxMov; $q(n)$ of the whole node as the quality term, weighted with `w_b` = 1 (paper mq_b = 2); `w_c` and `w_r` unused. | (c), no reason given. Comment at 289: "the paper mixes them; we use a weighted sum". O11 is (b), but the code's choice is not documented. | The destination choice in step 2 differs from eq. 28. | C13; O11 |
| L10 | Main loop: step 1 runs 1 + ρ times, step 2 ρ² times, nested; each iteration completes one partial solution (round robin) instead of all of $\mathcal{N}$; the best is chosen on the step-2 relocation cost instead of $f(\phi)$. | (c), no reason given. | With the defaults only 5 partial solutions are ever completed, and every iteration gives the same result. | C15; O15 **Step 3b:** partly: step 1 runs once; the rest is step 5. |
| L11 | Postprocessing is a no-op. | (c). Comment: "safe no-op for now". | Step 3 is missing, although the description claims it. | C16; O12, O13 |
| L12 | Parameters: `delta` = 3 plays the role of β (paper: eq. 29), `max_partial_nodes` = 50 of λ (500), `num_expansions` = 5 of ρ (300), `mu` = 2 of κ (named after µ); τ, µ, θ, mq are not configurable. | (c), no reason given. | The paper's settings cannot be reproduced through the UI. | C01 |
| L13 | Metadata and UI: `objectives = ["crane_time"]` and `fidelity = "faithful"` do not describe the current code; unused UI parameters (`max_iterations`, `report_interval`, `seed`); a duplicate final record; the relocation rule is not shown (platform). | (c) for the algorithm; the relocation rule is a platform point. | The UI suggests an optimisation and a fidelity that do not hold. | C17, C18 |
| L14 | **Assumption:** when $c^*$ is on top of its stack, it must be retrieved and may not be relocated; `Bay._check` refuses such a relocation (step 1). | (b). §1.2 alone ("only $c^+$ may be relocated", p. 437) does not settle it, because then $c^+ = c^*$. The assumption is based on Alg. 1 case (ii), line 13 (p. 441–442), which creates only the retrieval child when $c^* = c^+$. The model is more permissive: constraint (17) (p. 441, checked by Thom) only forbids relocating blocks below $c^*$ and does not exclude relocating $c^*$ itself. The heuristic follows Alg. 1 lines 13–15 (only a retrieval child when $c^*$ is on top), so L14 stays, with Alg. 1 as its basis. | Such a relocation can never be part of a plan that `f` accepts; the search never generates it. | C03 |
| L15 | **Assumption (O05):** a leaf reached in step 1 at any depth ≤ β updates ub and best (Alg. 1 lines 8–10); the best such leaf is a candidate for $\phi_{best}$ next to the expansion results. | (b), Thom. A complete node has $g = f$ and may fall outside $\mathcal{N}_g$; Alg. 2 does not say what happens to it. | Only relevant when β reaches a leaf (V09: $n_C \ge 25$ under O02 (a)/(b)). | C09, C15; O05 |
| L16 | **Provisional (O20):** $\phi_{best}$ is initialised with the greedy solution instead of the empty solution of Alg. 2 line 1. | (b), Thom: p. 441 defines ub as the value of the best solution found so far; with ub = f(greedy) that solution is the greedy. | When $\mathcal{N}_\beta$ is empty the greedy is returned; otherwise it only matters if nothing beats it. Frequency in V11. | C15; O20 |
| L17 | **Throwaway until steps 4, 5 and 7:** the old expansion runs on `Bay` nodes with `measures.q` in its score and first retrieves $c^*$ while it is on top; the selection among expansion results is on the old cost, the final choice on $f$; `theta` and λ = 500 are code defaults while the UI still shows `delta` and λ = 50. | Keeps the old step 2 working on the new step-1 nodes. | The old expansion behaves slightly differently from step 2 (its q term changed); none of this is the paper's step 2. | C12, C15, C01 |
| L18 | **No limit on step 1 (decision (iii)):** Algorithm 1 runs to depth β without a node or time limit. | (b)/(c): the paper gives none. | With the provisional literal eq. 30, large instances may run very long, also through the UI; V11 measures it per size. | C09; O02 **After the optimisation (V13):** memory, not time, is the binding limit: every stopped run hit 8 GB within 57–84 s, because all of $\mathcal{N}_\beta$ is kept (about 1.1 KB per node) before the selection. Undecided; options in the step-3b report. |
| L19 | **Documented correction of a presumed misprint (O02 (c)):** eq. 30 is used with the end values swapped, $p = 1.5$ for $n_C \le 9$ and $0.5$ for $n_C \ge 25$; the ramp in between is as printed. | (c). The printed version jumps at both ends; (c) makes it continuous without changing the ramp, and the runtime pattern of Table 3 (p. 448) and V11 fits (c) better than (a) or (b). Circumstantial evidence, not proof (Thom). | β = floor(0.6·\|φ\|) for $n_C \le 9$ (was 0.2) and floor(0.2·\|φ\|) for $n_C \ge 25$ (was 0.6); unchanged in between. | C08; O02 |

---

## §6 Log

- **Round 1** — Scope (heuristic only), restricted, objective eq. 1–2, instances, and the parameters from Sec. 7 established.
- **Round 2** — Sec. 5 (search tree, Algorithm 1, $h$) and Sec. 6.1–6.3 worked through. Corrections to round 1:
  - the weight symbols are η/γ, not α/β (α is the evaluation function in eq. 28);
  - the printed weight values contradict the text (O01);
  - Table 1 runs up to 6×10, not 6×6.

  Eq. 3 verified against Fig. 2 (531 / 513). Open points O01–O18 recorded; the code comparison (§2) is next.
- **Round 3 (29 Sept 2026)** — Code comparison (§2), behavioural checks (§3) and deviations (§5). Checked at HEAD 755acd2; `algorithm.py` last changed in 308d016. Main findings:
  - the step-1 moves are not kept, so every plan is incomplete (L1);
  - the cost function has a sign error and counts only the loaded trip of relocations (L2), so V01 fails and reverses the ranking;
  - $h$, ub, the greedy, eq. 29–30, the $\mathcal{N}_g \cup \mathcal{N}_q$ selection, the τ subtree, the roulette wheel and the postprocessing are missing (L5–L11);
  - the well-located test is inverted (L7).

  The docstring hypothesis in §1.1 (delta = τ, num_expansions = µ) is refuted: by role, delta ≈ β and num_expansions ≈ ρ (C01). O01 → sentence reading, undocumented; O03 → including the 2 empty tiers; O06, O10, O11, O15 → see C11, C10, C13, C15; O02, O04, O05, O08, O09, O12, O13, O17 are not resolved by the code, because the corresponding part is missing.
- **Step 1 (29 Sept 2026, 4a789f6)** — eq. 1–3 as `f` over a list of moves, incremental per move (`Bay.apply`: the crane stays at the last drop), start at (0, 1), $n_T$ as an argument, impossible moves raise. New files `operating_cost.py` and `test_lopez_plata.py`; the rest of the algorithm is unchanged. The tests give V01 = 531 / 513. O01: printed values as the default (suggestion). O03: (a). V06 added.

- **Step 2 (29 Sept 2026, 7b6e41f)** — the search state carries φ′ (throwaway list in `_SearchState`, `Bay` from step 3), the layout is read after `reset(skip_auto_retrieve=True)` and the starting retrievals are in φ, `_plan_from_phi` turns φ_best into the platform plan and env actions, `validate_plan` runs once on a fresh env (an infeasible plan raises `RuntimeError`), and `operating_cost` = f(φ_best). Old decision logic and cost unchanged. Tests: 4 new unit tests (Fig. 2 complete and stack mapping, starting retrievals on data3-3-5, a truncated φ raises), V07 210 of 210.
- **Step 3, O01 (29 Sept 2026, 5c5122a)** — O01 resolved in favour of (b) (Table 5, Sec. 7.7); `CostModel` default η₁ = γ₁ = 1, η₂ = γ₂ = 2; all tests pass the weights explicitly. §1.12: O01, O06, O11 and O16 updated, O19–O22 added; §1.2 page of constraints 15–18 corrected (p. 441); L3 and L14 updated.
- **Step 3a (29 Sept 2026, 4b87f98)** — pure functions in `measures.py`: `h` (eq. 21–22), `q` (eq. 23–26), `greedy` (Sec. 7.1, R1), `p`/`beta` (eq. 29–30 as printed); the search code is unchanged. Tests T1–T5 (8 new unit tests) plus mutation checks; V08 (h ≤ exact remaining optimum, 0 violations in about 6.4 million states, both readings) and V09 (β under three readings of eq. 30) added.
- **Step 3b (29 Sept 2026, 2b76a04)** — new step 1: `step1.py` with Algorithm 1 as depth-first branch-and-bound (O17 (i), O07) to depth β and the selection $\mathcal{N}_g \cup \mathcal{N}_q$ (O06 (a)); `Bay` is the search node (φ′, g); ub = f(greedy) (O04); leaves update ub/best and the best leaf is a candidate (O05, L15); $\phi_{best}$ starts as the greedy (O20, L16); the final choice is on f. Removed `_SearchState`, `_limited_a_star`, `_quality_node` and the dead first pass; the old expansion runs on `Bay` nodes until step 4 (L17). No step-1 limit (L18). Tests: 5 new (exactness, brute force, recursive reference, selection by hand, determinism) plus mutation checks; V10 (exact on Fig. 2 and Caserta 3×3, average 497.40) and V11 (step-1 runtime per size and O02 reading) added.
- **O02 (29 Sept 2026, 461c362)** — O02 decided as (c), a documented correction of a presumed misprint (L19); `measures.p` changed, T5 updated so that the printed version fails; 3b hash filled in.
- **Step-1 optimisation (29 Sept 2026)** — `algorithm1` as a recursion with apply/undo and an incremental h (`measures.block_term`, `Bay.undo`); same order and pruning. V12: identical to 2b76a04 on 41 instances; V13: re-measured under (c) with 300 s and 8 GB limits, memory is the binding limit.
