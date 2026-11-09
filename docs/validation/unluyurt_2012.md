# Ünlüyurt & Aydın (2012) — validation log

Paper: T. Ünlüyurt, C. Aydın, "Improved rehandling strategies for the container retrieval process", *Journal of Advanced Transportation* 46 (2012) 378–393. DOI 10.1002/atr.1193. Page numbers below are journal pages.

Code: `algorithms/CRP_Time/heuristic/unluyurt_2012/`

File:line references without a commit of their own point to the code at `f08d64a`, which implements Difference1 only. The new code (Difference1 and Difference2) is `4c3b3d6`; scripts and raw output of §3–§4 are in `docs/validation/data/unluyurt_2012/`.

## Summary

Status (9 Nov 2026): implemented in `4c3b3d6`. Difference1 and Difference2 are fixed-rule heuristics (Sec. 4.2, p. 386–387): each relocation is decided by three rules on the retrieval orders in the candidate stacks, and Difference2 adds a preference for stacks on the truck-lane side. The old code implemented Difference1 with a deviation: rules 2 and 3 used the earliest-retrieved container of a stack instead of its top container, so rule 3 was never reached (§2, O5). Both heuristics now follow the decisions in §2 and pass all cases of §3. On the 200 campaign instances the corrected Difference1 changes the plan on 151 instances and needs 2.2 % more relocations and 1.0 % more f2 time in total than the old code (§4.1). Difference2 is worse than Difference1 under f2 (+0.45 %) and f2vert (+1.14 %) but better under the paper's own Objective2 (−1.36 %): it saves horizontal distance at the cost of extra relocations, a trade that pays off on these instances only while tpp/ts is below about 15.8 (§4.3). The paper's 8000 instances are not public, so its numerical results cannot be checked (§2, O11). Fig. 6(c) and Fig. 9 were checked against the PDF by Thom; the Fig. 3 layout is taken from the text on p. 382; Fig. 2 was not checked.

## 1. Specification from the paper

### 1.1 Scope: what CRISP implements
- The paper proposes a branch and bound algorithm (Sec. 3, p. 381–384), EAR1/EAR2 after Kim & Hong (2006) (p. 384–386), Greedy1/Greedy2 (Sec. 4.1, p. 386) and Difference1/Difference2 (Sec. 4.2, p. 386–387).
- CRISP implements Difference1 (for Objective1 in the paper) and Difference2 (for Objective2). The other methods are out of scope.

### 1.2 Problem setting
- A single bay (p. 381: "we assume that there is a single bay").
- Containers are retrieved one by one in a predetermined sequence of individual containers: the setting of Kim & Hong [13], whose variant with groups of containers is a separate version (p. 379); "retrieve the containers in a bay one by one in a predetermined sequence" (p. 380); the numbers on the containers are the order of retrieval (p. 381).
- The truck lane is on the left of the bay (p. 384). This is stated once, for the example of Fig. 6(c); Difference2 (p. 387) refers to the truck lane without repeating the side.
- **Relocation rule: restricted.** The paper does not use the term. Only containers above the container to be retrieved are relocated: "If a container is not accessible, then the containers above it should be rehandled/relocated to other available positions in the bay" (p. 380). The branch and bound children only move the containers above container 1 (containers 4 and 5, p. 382). The setting is that of Kim & Hong [13] (p. 379).

### 1.3 Objectives
- **Objective1:** the number of relocations, as in [13] (p. 381; Eq. 1, p. 381).
- The Objective1 values in Table I (p. 390) include the retrievals: "even if every container is in order and there are no relocations, we still have to execute as many rehandles as the initial number of containers" (p. 391).
- **Objective2:** A × (number of crane pickups) + B × (horizontal distance travelled) (Eq. 2, p. 382; handle(a_k) = crane pickups, p. 382). A = 5, B = 1 (p. 388).
  - p. 381 describes Objective2 with "the number of relocations"; Eq. 2 (p. 382) uses pickups. Pickups include retrievals: "Realized handle refers to the number of retrievals and rehandles" (p. 384).
- **Horizontal distance:**
  - A retrieval from stack i costs 2i units. Source: the example on p. 384: "the crane should move six units to retrieve container 5 and come back to the original position"; 2 units for stack 1, 4 for stack 2, 6 for stack 3, total 26. The lane is 1 unit from stack 1, adjacent stacks are 1 unit apart, and every move is a round trip.
  - A relocation from stack l to stack i costs 2|l − i| units: the term 2B|l − i| in Eq. 4 (p. 385). Eq. 4 is the expected cost of EAR2, not a general cost definition.
  - Both agree with the examples: Fig. 3 (p. 381; layout [<1,4,5><3,2><6>], p. 382): moving 5 and 4 onto container 2 and retrieving container 1 takes three pickups and 2·1 + 2·1 + 2·1 = 6 units. Fig. 6 (p. 382, 384): moving container 5 from stack 1 to stack 3 costs 4 units, container 4 costs 2 units.

### 1.4 Difference heuristic (Sec. 4.2, p. 386–387)
When container X is to be relocated, the destination is chosen by the following rules, in order (p. 386):
1. If there are stacks whose container that will be retrieved first is retrieved later than X, choose among them the stack whose earliest-retrieved container Y is retrieved earliest. "By minimizing the difference in the orders of retrieval, we minimize the number of containers ordered between X and Y" (p. 386). Such a relocation causes no additional cost (p. 386).
2. Otherwise, among the stacks whose container Z "that is accessible (meaning at the top of the stack)" has an order number smaller than X, minimise the difference between X and Z (p. 386).
3. Otherwise, "minimize the difference between the order numbers of container X and the container where X will be relocated on" (p. 386).
- "The empty stacks are assumed to contain the highest ordered container" (p. 387).
- Figure 9 (p. 387) illustrates one decision per rule (§3).

**Difference2** (p. 387): "we distinguish between two types of stacks. The first kind is the stacks between the stack of container to be rehandled and truck lane, and it has higher priority than the other kind. If the available position is not found in the first kind of stacks, then the stacks away from the truck lane (the second type) are searched. This algorithm for Objective2, unlike other algorithms for Objective2, does not consider the constants A and B, but the priority of handling is kept higher than the horizontal distance traveled."

### 1.5 Not specified in the paper
- Tie-breaking between stacks that are equal under a rule (p. 386–387).
- Whether "the highest ordered container" for an empty stack is container N or a value above N (p. 387).
- What happens when no stack can receive X.
- Difference2: whether the lane-side preference comes before the three rules or within each rule level (p. 387; §2, O2).
- Difference2: whether distance matters within one kind of stack.

### 1.6 Instances and reference results
- 8000 random instances (p. 387): balanced or unbalanced layouts, 3–7 stacks, stack heights 4–7, intensity 55–75 % of the slots occupied, containers distributed uniformly; 40 instances per combination, 200 combinations (p. 392). Not public. Azari et al. (2017, p. 315) state that a set of 640 of these instances is "accessible upon request" (`azari_2017.md` §1).
- Reference results (orientation only, not a target): Table I (p. 390) gives average Diff1 and Diff2 values for eight instance types; Table VI (p. 392) gives 72.55 % optimal and an average gap of 1.83 % for Difference1 under Objective1.
- p. 392 reports Objective1 on 10 single-bay Lee & Lee (2010) instances only as a relative gap of 7–12 % to Lee & Lee, without separating the heuristics.

## 2. Decisions

Decided by Thom on 9 Nov 2026. Categories as in the validation workflow: (a) required by the unified 2D setting, (b) a choice the paper leaves open, (c) a real deviation from the paper.

| # | Point | Decision | Basis | Category |
|---|---|---|---|---|
| O1 | Relocation rule | Restricted; the class declares `relocation_rule = "restricted"`. The existing class declares none (`algorithm.py:125-145`), i.e. `None` in `core/algorithm_meta.py:23-27`. | Decision, p. 380, 382 (§1.2) | — |
| O2 | Difference2 ordering | Per rule level: rule 1 lane side, rule 1 far side, rule 2 lane side, rule 2 far side, rule 3 lane side, rule 3 far side. | Assumption, from "the priority of handling is kept higher than the horizontal distance traveled" (p. 387) | (b) |
| O3 | Ties | The lowest stack index wins. | Assumption; the paper is silent (p. 386–387) | (b) |
| O4 | Empty stack | Counts as container N + 1. | Assumption (p. 387) | (b) |
| O5 | Rules 2 and 3 | Use the top container of the candidate stack; rule 1 uses the earliest-retrieved container. | Decision, p. 386 | fixes (c) in the existing code |
| O6 | No destination available | Raise an explicit error. | Decision; the paper is silent | (b) |
| O7 | Lane side | The stacks with a lower index than the source stack. | Decision, p. 384 and the f2 model | (a) |
| O8 | Fixed rule under f2 / f2vert | Accepted as fixed-rule baselines. | Decision | (a) |
| O9 | Problems | CRP-Time only. | Decision | — |
| O10 | Within one side and rule level | The rule's own minimisation, then O3. | Assumption; the paper is silent (p. 387) | (b) |
| O11 | Paper's instances | Not requested from the authors; recorded as a limitation. | Decision | — |

### O2 — Difference2 ordering
- Two readings of p. 387:
  - (a) apply all three rules to the lane-side stacks, and search the far side only if no lane-side stack has a free slot. This follows "If the available position is not found in the first kind of stacks"; the paper uses "available position" for a free slot elsewhere ("relocated to other available positions in the bay", p. 380).
  - (b) per rule level, lane side before far side. This follows the last sentence: "handling" is the number of pickups (handle(a_k), p. 382; "A: The handling coefficient", p. 385), and rule 1 is the only rule that causes no additional rehandle (p. 386). Under (a), a rule-3 placement on the lane side would beat a rule-1 placement on the far side, so distance would come before handling.
- Chosen: (b). Reading (a) will be run in a scratch analysis on all Caserta instances (3×5 to 6×10), reporting on how many instances the plans differ and the f2 / f2vert difference.

### O3 — Ties
- Priorities are distinct, so with O4 a tie can only occur between empty stacks under rule 1.
- The lowest stack index is the stack nearest the truck lane in f2 (O7). For Difference1 the choice does not change the number of relocations, since the rules treat identical empty stacks alike; it can change the crane time.
- The existing code does the same: `min` keeps the first minimum in `all_keys` order (`algorithm.py:61-67`, `:76`).

### O4 — Empty stack = N + 1
- Containers are numbered 1…N and retrieved in that order, so container N is in the bay at every decision. "The highest ordered container" (p. 387) therefore reads as N, whether it means the highest in the bay or in the instance.
- Read literally as N, an empty stack cannot receive container N under rule 1 (N is not retrieved later than N) or rule 2 (N is not smaller than N), so N goes onto a stack whose top it blocks, although an empty stack is free (case E4, §3). That contradicts the idea of the heuristic: "to avoid further relocations, as much as possible, for each realized relocation" (p. 386). N + 1 makes the empty stack a rule-1 stack for every X.
- The existing code uses N + 1 (`algorithm.py:32-34`, `:72`).

### O5 — Rules 2 and 3 on the top container
- Rule 2 names the container "that is accessible (meaning at the top of the stack)"; rule 3 the container "where X will be relocated on" (p. 386). Both are the top container of the candidate stack. Rule 1 uses "the container that will be retrieved first in that stack" (p. 386).
- At rule 3 every candidate's top is retrieved later than X (rules 1 and 2 found nothing, and priorities are distinct), so the absolute and the signed difference are the same.
- The existing code uses the earliest-retrieved container of the stack in all three rules (`tp` = `_top_priority` = the stack minimum, `algorithm.py:32-34`, `:57`). Rule 1 (`:59-61`) agrees with the paper. Rule 2 (`:63-65`) selects on the stack minimum instead of the top. Rule 3 (`:67`) is unreachable: when rule 1 finds nothing, every candidate stack has a minimum below X, so the code's rule 2 always applies. Cases E2 and E3 (§3) show the difference. The description (`algorithm.py:133-135`) states the code's reading.

### O6 — No destination available
- The paper does not say what happens when no stack other than the source has a free slot.
- The new code raises an error. The existing code calls `env.step(0)` (`algorithm.py:109-112`); that step changes nothing, so the loop would not end.
- On Caserta this cannot occur: a bay has W·H − N = 2W free slots, and the source stack holds at least two containers, so at most H′ = N/W of its slots are free; another stack has room while H′ < 2W, which holds for 3×5 to 6×10.

### O7 — Lane side
- f2 charges a retrieval from stack i with 2·ts·i + tpp, with i = (bay − 1)·num_rows + row (`algorithms/CRP_Time/utils/crane_time.py:126-141`). For Caserta, bay = the stack's line in the file and row = 1 (`core/benchmarks/caserta.py:198-203`, `:227-228`), so i is the stack's line number.
- Stack 1 is therefore nearest the truck, as in the paper (lane on the left, retrieval from stack i costs 2i, p. 384). The lane side of source stack l is the stacks 1…l − 1; for l = 1 there is none, and Difference2 then decides as Difference1.
- A relocation costs 2|l − i| in the paper and 2·ts·|l − i| + tpp in f2, the same in both directions. The lane-side preference pays off in the later retrieval of X, which is cheaper from a stack nearer the lane.

### O8 — Fixed rules under f2 and f2vert
- Neither heuristic reads the parameters of the time model: Difference1 decides on retrieval orders only, Difference2 also on stack positions. The plan is the same under f2, f2vert and the practice preset; only the reported main objective differs.
- They are therefore reported as fixed-rule baselines (`objectives = ["fixed-rule"]`), not as methods that optimise f2 or f2vert.

### O9 — CRP-Time only
- The existing class is listed for CRP-R and CRP-Time (`algorithm.py:142`). The new classes are CRP-Time only, like the three validated algorithms; the CRP-R menu loses Difference1.

### O10 — Within one side and rule level
- Within the lane side or the far side at one rule level, the stack is chosen by the rule's own minimisation (rule 1: earliest Y; rule 2: smallest X − Z; rule 3: smallest difference to the top), then by O3. Distance within a side is not used.

### O11 — No instances of the paper
- The paper's instances are not public (§1.6) and are not requested. The numerical results in Tables I–VI cannot be checked; the behavioural checks are the worked examples (Fig. 3, Fig. 6, Fig. 9) and the cases in §3.

### Objective2
- No metric in the result files. Objective2 is f2 with tpp = A and ts = B: a retrieval from stack i costs 2·B·i + A and a relocation 2·B·|l − i| + A (§1.3; `crane_time.py:132-141`). It is used in the tests only, with A = 5 and B = 1 (p. 388):
  - Fig. 3 (p. 381–382): layout [<1,4,5><3,2><6>]; relocate 5 and then 4 from stack 1 to stack 2, retrieve 1 from stack 1: 3A + 6B = 21.
  - Fig. 6(c) (p. 384): layout [<1><3,2,4><6,5>], checked against the PDF by Thom (container 5 moved to stack 3 and container 4 to stack 2, p. 382, 384). The horizontal part of the six retrievals is 2 + 3·4 + 2·6 = 26 units, so the six retrievals cost 6A + 26B = 56. The relocations to reach it cost 2A + 6B (4 + 2 units, p. 384).

### Existing code, other points (at `f08d64a`)
- It does not use `read_bay`, `Stopwatch`, `check_objective`, `finish()` or `saved_moves`; `runtime_s` is the runner's wall clock, and every time model, including `rmgc_current`, is accepted.
- It calls `env.reset()` without `skip_auto_retrieve` (`algorithm.py:78`) but does not score its own plan: the metrics come from the plan the problem records (`algorithm.py:118`; `problems/CRP_Time.py:87-96`), which includes the retrievals at reset. A layout with container 1 on top at the start gave a complete plan that `validate_plan` accepts with the same f2 (scratch check, 9 Nov 2026). The A3 problem therefore does not apply.
- `steps` is the problem's step counter, not the number of moves (`finish()` sets the number of moves).
- Only core import: `core.base_algorithm` (`algorithm.py:25`), an allowed exception. It reads problem internals (`_action_to_stack`, `yard.stacks`, `_current_target_priority`).
- The description quotes Table VI (p. 392) correctly, but these are the paper's results.
- No tests.

## 3. Test cases

Expected destination of the first relocation under §2: rule 1 on the earliest-retrieved container, rules 2 and 3 on the top container, empty stack = N + 1, ties to the lowest stack index. Stacks are listed left to right (stack 1 nearest the truck lane), containers bottom to top. Container 1 is the next target and X is the top container of its stack. Figure 9 was transcribed from the PDF by Thom; the E cases are constructed. "Existing code" is the destination chosen by Difference1 at `f08d64a` (scratch run, 9 Nov 2026).

### 3.1 Difference1

| Case | H | Stacks | X | Expected | Rule | Existing code | Tests |
|---|---|---|---|---|---|---|---|
| Fig. 9 A | 4 | [1,4,2] [3,5] [6] | 2 | S2 | 1 | S2 | as in the figure |
| Fig. 9 B | 4 | [1,6,5] [3,2] [4] | 5 | S3 | 2 | S3 | as in the figure |
| Fig. 9 C | 4 | [1,7,4] [3,5] [2,6] | 4 | S2 | 3 | S2 | as in the figure |
| E1 | 4 | [1,2,3] [4,7] [6,5] | 3 | S2 | 1 | S2 | rule 1 uses the earliest-retrieved container (4 < 5), not the top (7 > 5) |
| E2 | 4 | [1,5] [2,4] [3] | 5 | S2 | 2 | **S3** | rule 2 on the top: 5 − 4 < 5 − 3 (the stack minima 2 and 3 would give S3) |
| E3 | 4 | [1,4] [3,6] [2,5] | 4 | S3 | 3 | **S2** | rule 3 on the top: 5 − 4 < 6 − 4 (the stack minima would give S2) |
| E4 | 4 | [1,5] [2,3] [4] [] | 5 | S4 | 1 | S4 | O4: X = N; with the empty stack as N it would be S3 (rule 2) |
| E5 | 4 | [1,3] [2,4] [5] [] | 3 | S3 | 1 | S3 | O4: 5 < 6 = N + 1; with the empty stack as N, S3 and S4 would tie |
| E6 | 4 | [1,3] [2,4] [] [] | 3 | S3 | 1 | S3 | O3: tie between two empty stacks |
| E7 | 3 | [1,4,3] [5,6,7] [2] | 3 | S3 | 2 | S3 | full S2 is excluded; otherwise rule 1 would choose S2 |

In all these cases the source is stack 1, so it has no lane side, and Difference2 chooses the same stacks as Difference1.

### 3.2 Difference2 (checked by hand by Thom, 9 Nov 2026)

Constructed cases with the source not in stack 1. Expected under O2 reading (b); reading (a) and Difference1 are given for comparison.

| Case | H | Stacks | Source | X | Difference2 (b) | Reading (a) | Difference1 | Tests |
|---|---|---|---|---|---|---|---|---|
| D1 | 4 | [3,2] [4] [1,5] [6] | S3 | 5 | S4 (rule 1, far side) | S2 (rule 2, lane side) | S4 | rule 1 far side before rule 2 lane side |
| D2 | 4 | [6] [1,4] [5] [3,2] | S2 | 4 | S1 (rule 1, lane side) | S1 | S3 | lane side before far side at rule 1, against rule 1's own minimisation |
| D3 | 4 | [2,6] [1,4] [5,3] | S2 | 4 | S3 (rule 2, far side) | S1 (rule 3, lane side) | S3 | rule 2 far side before rule 3 lane side |
| D4 | 4 | [2,6] [1,4] [3,5] | S2 | 4 | S1 (rule 3, lane side) | S1 | S3 | lane side before far side at rule 3 |

### 3.3 Objective2 (f2 with tpp = 5, ts = 1)
- Fig. 3: 3A + 6B = 21 (§2, Objective2).
- Fig. 6(c) (layout checked against the PDF by Thom): six retrievals 6A + 26B = 56; the two relocations before it 2A + 6B = 16.

### 3.4 Tests
`algorithms/CRP_Time/heuristic/unluyurt_2012/test_unluyurt.py` (`4c3b3d6`), 8 tests, all passing:
- the first relocation of every case in §3.1 and §3.2, for both selectors;
- O6: a `RuntimeError` when no other stack has room, directly and through the problem;
- a complete plan through the problem on data3-5-1 (container 1 on top at the start): N retrievals in order, `steps` = number of moves, objective = f2;
- the same plan under f2 and f2vert on data4-6-1 (O8);
- Objective2 as f2 with tpp = 5 and ts = 1 on the examples of §3.3.

## 4. Behavioural checks

The paper's own instances are not available (O11), so there is no check against its results. The checks below run on the 200 instances of the campaign classes 3x5, 3x7, 4x6, 5x7 and 6x10 (40 each), 9 Nov 2026, rl env Python 3.11.16. Every plan in §4.1–4.3 passes `check_plan` (`docs/validation/data/jovanovic_2019_step3/jov_feasibility.py`) and the problem's `validate_plan`; 52 of the 200 instances start with container 1 on top. Scripts: `compare.py` (§4.1–4.2) and `obj2.py` (§4.3) with their output in `docs/validation/data/unluyurt_2012/`. Default parameters: f2 with ts = 1.2 s per stack and tpp = 30 s per move (tpp/ts = 25); f2vert with the Voß & Schwarze values; Objective2 with A = 5 and B = 1 (A/B = 5, p. 388).

### 4.1 Old (`f08d64a`) vs new Difference1

The plan differs on 151 of 200 instances (3x5: 19, 3x7: 20, 4x6: 33, 5x7: 39, 6x10: 40). Totals, old → new; "new lower / higher" counts the instances where the new plan is better / worse.

| Class | Relocations | f2 | f2vert |
|---|---|---|---|
| 3x5 | 282 → 285 | 32 047.2 → 32 142.0 | 60 225.8 → 60 059.7 |
| 3x7 | 373 → 378 | 46 815.6 → 47 080.8 | 85 110.8 → 85 303.7 |
| 4x6 | 587 → 598 | 57 498.0 → 57 914.4 | 121 515.2 → 121 011.1 |
| 5x7 | 1053 → 1059 | 93 226.8 → 93 574.8 | 217 647.5 → 215 049.4 |
| 6x10 | 1994 → 2065 | 180 060.0 → 182 890.8 | 443 701.9 → 441 667.7 |
| All | 4289 → 4385 (+2.2 %) | 409 647.6 → 413 602.8 (+0.97 %) | 928 201.3 → 923 091.6 (−0.55 %) |
| New lower / higher | 29 / 57 | 57 / 84 | 84 / 66 |

The correction (O5) is not an improvement in the numbers: the faithful Difference1 needs slightly more relocations and more f2 time than the old code, and slightly less f2vert time. The difference grows with the class (6x10: +3.6 % relocations, +1.6 % f2).

### 4.2 Difference2: reading (a) vs reading (b)

The plan differs on 167 of 200 instances (3x5: 27, 3x7: 29, 4x6: 32, 5x7: 39, 6x10: 40). Totals, reading (a) → reading (b), the implemented one (O2); "(b) lower / higher" counts the instances where (b) is better / worse.

| Class | Relocations | f2 | f2vert |
|---|---|---|---|
| 3x5 | 332 → 288 | 33 398.4 → 32 078.4 | 62 563.4 → 60 605.4 |
| 3x7 | 434 → 378 | 48 175.2 → 46 480.8 | 86 878.2 → 85 061.2 |
| 4x6 | 713 → 611 | 61 431.6 → 58 045.2 | 126 035.6 → 121 482.3 |
| 5x7 | 1292 → 1108 | 100 622.4 → 94 430.4 | 227 212.6 → 218 164.6 |
| 6x10 | 2567 → 2168 | 198 286.8 → 184 430.4 | 470 628.4 → 448 337.9 |
| All | 5338 → 4553 (−14.7 %) | 441 914.4 → 415 465.2 (−6.0 %) | 973 318.2 → 933 651.4 (−4.1 %) |
| (b) lower / higher | 157 / 2 | 159 / 7 | 149 / 17 |

Reading (a), which puts the lane side before all three rules, is clearly worse on all three measures. It can place a container on the lane side under rule 2 or 3 where the far side offers a rule-1 stack (case D1, §3.2); how often that happens was not counted. This supports the choice of (b) but does not decide what the authors meant.

### 4.3 Difference1 vs Difference2 under f2, f2vert and Objective2

Both new implementations; Difference2 is reading (b). Moves = relocations + N, so the difference in moves equals the difference in relocations; horizontal units are in stack distances (a retrieval from stack i counts 2i, a relocation from l to i counts 2|l − i|, §1.3). Objective2 = 5·moves + 1·units and f2 = 30·moves + 1.2·units; both were recomputed this way for every plan and agree with the evaluator.

Totals over 200 instances; "D2 lower / higher" counts the instances where Difference2 is better / worse.

| | Difference1 | Difference2 | Difference | D2 lower / higher |
|---|---|---|---|---|
| Relocations | 4385 | 4553 | +168 (+3.83 %) | 19 / 69 |
| Horizontal units | 80 044 | 77 396 | −2648 (−3.31 %) | 134 / 40 |
| f2 | 413 602.8 | 415 465.2 | +1862.4 (+0.45 %) | 101 / 76 |
| f2vert | 923 091.6 | 933 651.4 | +10 559.8 (+1.14 %) | 74 / 108 |
| Objective2 | 132 969 | 131 161 | −1808 (−1.36 %) | 127 / 50 |

Per class, Difference2 − Difference1 in % of Difference1 (negative: Difference2 better).

| Class | Relocations | Units | f2 | f2vert | Objective2 |
|---|---|---|---|---|---|
| 3x5 | +1.05 | −2.75 | −0.20 | +0.91 | −1.24 |
| 3x7 | 0.00 | −5.69 | −1.27 | −0.28 | −3.36 |
| 4x6 | +2.17 | −2.32 | +0.23 | +0.39 | −0.88 |
| 5x7 | +4.63 | −3.10 | +0.91 | +1.45 | −0.93 |
| 6x10 | +4.99 | −3.17 | +0.84 | +1.51 | −1.23 |

**Decomposition.** Difference2 does what p. 387 describes: it relocates towards the truck lane more often (2842 of 4553 relocations, against 2263 of 4385 for Difference1) and saves 2648 horizontal units, at the cost of 168 extra relocations. Under Objective2 the saving outweighs the cost (5·168 − 2648 = −1808); under f2 it does not (30·168 − 1.2·2648 = +1862.4). In total over these instances Difference2 is better whenever tpp/ts < 2648/168 ≈ 15.8. Objective2 has A/B = 5, below the break-even; f2 has tpp/ts = 25, above it. f2vert charges every move a tier-dependent vertical time instead of tpp, which makes an extra relocation even more expensive, and Difference2 is worse there by the largest margin.

**Limits of this result.**
- The break-even is an aggregate over these 200 Caserta instances, not a property of the heuristics: per instance the trade differs, and under f2 Difference2 still wins on 101 of the 200 instances (and in total on 3x5 and 3x7). The total turns against Difference2 in the larger classes, where its extra relocations grow fastest.
- The comparison cannot be checked against the paper: the paper reports Difference1 only under Objective1 and Difference2 only under Objective2 (Table I, p. 390; Tables III, V and VI, p. 391–392), and never compares the two under the same objective.
- It is a time-model effect, not an implementation error: under the paper's own objective Difference2 is better than Difference1, as its design intends.

### 4.4 Workbench (9 Nov 2026, `4c3b3d6`)

Both classes run through the Workbench on Caserta 3x5 #1 under f2 (`main.py web --port 8010` with the rl env's Python, headless Chrome via `docs/validation/data/caserta_crp_time/ui_jobs.py`; specs, screenshots and the two result files in `docs/validation/data/unluyurt_2012/ui/`).
- The catalog lists both only for CRP-Time, with `relocation_rule` restricted, `objectives` fixed-rule and no parameters. The Workbench shows the tags "Single-bay (2D) | Fixed rule | Restricted relocations" and marks both in the algorithm list with ⚠, the platform's mark for a method that does not optimise the selected objective (`core/algorithm_meta.py`, `assess_compatibility`), as for any fixed rule under crane time.
- Both jobs end "completed" without errors. The saved results equal a direct run on data3-5-1: Difference1 6 relocations, f2 766.8, f2vert 1469.355, 21 moves; Difference2 7 relocations, f2 792.0, f2vert 1526.715, 22 moves. The saved moves pass `check_plan`.
- results/ was backed up before the check; afterwards the two new result folders were removed, and results/ is identical to the backup.
