# Forster & Bortfeldt (2012) — validation log

Paper: F. Forster, A. Bortfeldt, "A tree search procedure for the container relocation problem", *Computers & Operations Research* 39 (2012) 299–309. DOI 10.1016/j.cor.2011.04.004. Page numbers below are the journal's printed pages (PDF page = printed page − 298). Not to be confused with the multi-bay retrieval paper in Oper. Res. Proc. 2011, 257–262, which `algorithms/CRP_Time/heuristic/forster_bortfeldt_2012/` cites (step 7 of the thesis plan).

Code: `algorithms/CRP_Time/heuristic/forster_bortfeldt_2012_tree/` (standalone; imports nothing from `algorithms/CRP_D` or any other algorithm folder). Scripts and raw output: `docs/validation/data/forster_bortfeldt_2012/`.

Scope set by Wei (10 Nov 2026): the C&OR tree search, standalone in CRP-Time, validated first unrestricted with the number of relocations as the objective, in its own layout simulation outside `CRP_R.step`. Integration with f2/f2vert comes later.

## Summary

Status 10 Nov 2026: **step 2 done; step 3 waits for the decisions F8–F19** (§2.2), which fix the readings of Figs. 4–7 that the step-3 results depend on.
- Step 1 (§3): the Fig. 3 bound is 25 (pass). The CVS labels map to the Caserta files by the stack-count column of Table 2 (F3; the item-count key failed on a misprint). The mean bound is above nrel − LB gap of Table 3 on all sizes; this is a reported discrepancy, not a stop criterion (F5).
- Step 2 (§4): greedy (Fig. 4), productive moves (Fig. 7), compound moves (Fig. 6) and search (Fig. 5) implemented from the paper; 19 unit tests pass. On all 80 Caserta 3x3 and 3x4 instances, at H = n and H = T + 2, every plan is valid, none is below the bound, none is worse than the greedy, and at H = T + 2 none is below the exact optimum (IDA*).
- Open, decisive for step 3: F9. Fig. 4 as printed makes the greedy relocate a blocker onto the stack whose lowest item is retrieved next, against the paper's own rationale (p. 304). On 10–10 the greedy then needs 1171 relocations on average (H = n), against 146 with the reading that follows the rationale (§4.3). The greedy is the search's first upper bound.

## 1. Specification from the paper

### 1.1 Problem (Sec. 1, p. 299; Sec. 4.1, p. 301–302)
- S stacks, maximum stack height H, n items in G groups (n ≤ SH). A move is a relocation (top item of one stack to the top of another) or a remove (top item leaves the bay). A remove is feasible only if the item's group is not greater than that of any remaining item (p. 299, p. 302).
- Objective: a move sequence of minimum length that empties the layout (p. 299). Every complete sequence has exactly n removes, so comparing relocations nrel is equivalent (p. 306).
- **Relocation rule: unrestricted.** A relocation (d, r) is applicable iff the donator d is not empty and the receiver r is not full (p. 302); any top item may be moved, not only those above the lowest group.
- Layout as an H × S matrix L(h, s) with 0 for an empty slot; slots fill from the bottom (p. 301–302).

### 1.2 Definitions (Sec. 4.1, p. 302; Table 1, p. 302)
- **Badly placed** (def. 1): the item in slot (h, s) of group g with h > 1 and at least one lower slot h′ < h holding a group L(h′, s) < g. Otherwise well placed. A stack is clean if all its items are well placed, otherwise dirty.
- **Clean supply slot** of group g (def. 2): slot (h, s) of a non-empty clean stack s whose highest item has group g, lying above that item. "In addition, an empty stack provides H clean supply slots of group G."
- **Clean supply** Sc(L) = 1·sc(1) + 2·sc(2) + … + G·sc(G) (Table 1).
- n(L): number of items; n_b: number of badly placed items; g_top(s): "group of topmost item in stack s, if s is not empty and topmost item is badly placed, n(L) + 1 otherwise"; g_min(s): "minimum group of all items in stack s, if s is not empty, 0 otherwise" (Table 1).

### 1.3 Move classification (Sec. 4.2, p. 302)
- A relocation is BG, BB, GG or GB by the moved item's placement status before and after the move.
- BG and BB are further FLG (the moved item blocks an item of the currently lowest group) or FOG (it does not).
- Six relocation types: FLG_BG, FOG_BG, FLG_BB, FOG_BB, GG, GB. **Productive** move types: FLG_BG, FOG_BG, FLG_BB, GG and removes.

### 1.4 Lower bound (Sec. 5, p. 302–303)
- n′_REM = n(L); n′_BG = n_b; n′_non-BG = 1 iff (a) no item can be removed, (b) there is no empty stack, (c) min{g_top(s)} > max{g_min(s)}; else 0. n′_m = n′_REM + n′_BG + n′_non-BG (Proposition, p. 302).
- Worked example, Fig. 3 (p. 303): S = 4, H = 6, n = G = 16; n′_REM = 16, n′_BG = 8, "min{11, 14, 15, 6} = 6 > 4 = max{2, 3, 1, 4}", n′_non-BG = 1, bound 25.

### 1.5 Tree search (Sec. 6, p. 303–306; Figs. 4–7)
- Greedy initial solution (Sec. 6.1, Fig. 4, p. 303–304): per step, (1) apply all removes; else (2) the BG relocation to a non-empty stack with minimum group difference between the moved item and the top item of the target stack; else (3) the BG relocation to an empty stack with the minimum moved group; else (4) the FLG_BB relocation with maximum difference between the moved item and "the item with the lowest item group in the target stack" (Fig. 4) — the text says "the topmost item of the target stack" (p. 304).
- perform_compound_moves (Sec. 6.2, Fig. 5, p. 304): abort if nm(s*) = n′_m(L_Init) or the time limit is exceeded; at a leaf update s*; otherwise generate compound moves with maxMoves = nm(s*) − nm(s) − 1 and cmStopValue = 1, sort by (moves in the compound move + n′_m of the resulting layout) ascending, then by number of moves descending, then by clean supply of the resulting layout descending; keep the first nSucc and recurse on each (depth first).
- determine_compound_moves (Sec. 6.3, Fig. 6, p. 305): for each productive move pm of L: if L ⊕ pm is empty, update s* and skip; skip if n′_m(L ⊕ pm) > maxMoves; append pm to the working compound move; if cmStopValue < cmStopThreshold recurse with (maxMoves − 1, cmStopValue · |Pm|), otherwise add the compound move to Cm.
- determine_productive_moves (Sec. 6.4, Fig. 7, p. 305–306): all removes if any; else all BG relocations if any; else the first maxFlg_bb FLG_BB relocations sorted ascending by (moved group − lowest group in the target stack) plus the first maxGg GG relocations sorted ascending by (diffBefore − diffAfter).

### 1.6 Parameters and benchmark (Sec. 7, p. 306–307)
- nSucc = 10, cmStopThreshold = 150, maxFlg_bb = 3, maxGg = 2, time limit 60 s (p. 306). Java, JVM 1.6, Core 2 Duo P7350 2 GHz (p. 306).
- CVS test cases of Caserta et al.: 40 instances each, one item per group, "the stacks are not constrained in height", rectangular initial layouts, "labelled CVS number of stacks × items per stack" (p. 306); "CVS 3–4 refers to 40 test instances with layouts that consist of 3 stacks, each containing 4 items" (p. 307). Table 2 (p. 306) lists stacks, groups, items and mean badly placed items per test case.
- Table 3 (p. 307): mean nrel, time, "LB gap" and "nrel − msh". LB gap is **absolute**: "the average absolute deviation (in relocations) between the solution of the tree search heuristic and the lower bound from Section 5 over all instances of a test case" (p. 307).
- Height-limited variant: "a maximum stack height that is assumed to be 2 slots higher than the height of the stacks in the initial layout. Taking the CVS 3–4 test case for example, the maximum stack height is set to 6" (p. 307); results in column "nrel − msh".

## 2. Decisions

### 2.1 Step 1 (F1–F7)

F1 and F4 are readings of the sources. F2, F3, F5, F6 and F7 were decided by Thom on 10 Nov 2026, after step 1 and before any step-2 or step-3 result. Categories as in the validation workflow: (a) required by the unified 2D setting, (b) a choice the paper leaves open, (c) a real deviation from the paper.

| # | Point | Decision | Basis | Category |
|---|---|---|---|---|
| F1 | Caserta file format | `data<T>-<S>-<k>.dat`: first line S n, then one line per stack: height, then groups bottom to top. | `benchmark/Caserta_dataset/Data Format.docx`; Table 2's stack counts (p. 306) | — |
| F2 | g_top | Table 1 as printed, including n(L) + 1 after removes (`layout.py`, `g_top`). | Table 1 (p. 302); Fig. 3 (p. 303) differs | — |
| F3 | CVS label → Caserta file | CVS T–S → `data<T>-<S>` (S stacks of T items), fixed on the stack-count column of Table 2. Deviation from the pre-registered key (item count). | Table 2 (p. 306); §3.2 | deviation from the protocol |
| F4 | LB gap | Absolute; mean bound = nrel − LB gap. | p. 307 | — |
| F5 | Step 1c | Downgraded from a stop criterion to a reported discrepancy. The bound is not tuned to Table 3. | Tables 2, 3, 5 (p. 306–308); §3.3 | — |
| F6 | H for "nrel − msh" | H = T + 2, T the initial stack height of the file. | p. 307 | — |
| F7 | H under unlimited height | H = n (items of the instance) for the clean supply; with H = n no stack is ever full. | Thom; p. 302, p. 306 | (a)/(b) |

#### F2 — g_top
- Table 1 (p. 302) defines g_top(s) by the *top* item: its group if it is badly placed, n(L) + 1 otherwise. In the Fig. 3 example (p. 303) the list is {11, 14, 15, 6}; stack 1 is 12, 8, 11, 2 (bottom to top), whose top item 2 is well placed, so Table 1 gives 17. 11 is the topmost *badly placed* item of stack 1, which is not on top.
- The proof of (iii) (p. 302–303) argues about "badly placed item[s] in topmost position", which matches Table 1.
- The Fig. 3 reading gives a min g_top no larger than Table 1's, so it is a weaker but valid bound. On Fig. 3 both readings give n′_non-BG = 1 (min 6 > 4 either way). On the 840 Caserta initial layouts they differ on 6 instances (one or two each in 5–4, 6–6, 6–10, 10–6 and 10–10).
- After removes, groups above n(L) + 1 can remain (e.g. 12 items with groups 5…16); the printed n(L) + 1 then no longer exceeds every group and the bound is weaker than "larger than every group" would give, never invalid. Kept as printed (decision).

#### F3 — CVS label → Caserta file
- p. 306: "labelled CVS number of stacks × items per stack"; p. 307: CVS 3–4 = 3 stacks × 4 items (12 items).
- Table 2 (p. 306): CVS 3–4 has 4 stacks and 16 items, 3–5 5 stacks and 25 items, 3–6 6 stacks and 36 items; every other row has items = stacks × the first number of the label (e.g. CVS 10–6: 6 stacks, 60 items, so 10 items per stack, against the text).
- Caserta files: `data<T>-<S>` holds S stacks of T items (F1); data3-4: 4 stacks, 12 items.
- Decision: CVS T–S → `data<T>-<S>`. The pre-registered key (item count) failed because the item column of Table 2 for 3–4, 3–5 and 3–6 (and the group column) lists S² instead of 3·S, a misprint pattern. The mapping is fixed on the stack-count column (21/21 rows). The mean number of badly placed items is the independent check: 19 of 21 within 0.005; 3–8 (9.25 against 9.30) and 5–9 (25.05 against 25.10) are off by 0.05, i.e. 2 items over 40 instances, cause unknown, reported as is (§3.2).
- The text's definition of the label (p. 306–307) is reversed with respect to Table 2 and the files.

#### F5 — step 1c is a reported discrepancy, not a stop criterion
- (1) The implied mean non-BG term, nrel − LB gap − mean badly placed items (Tables 2–3), is negative for 5–9 (24.85 − 25.10 = −0.25), which is impossible since n′_non-BG ∈ {0, 1}, and ≈ 0 within rounding for 5–10, 6–10 and 10–10.
- (2) Table 5 (p. 308), variant V1: omitting the non-BG term changes the relocations by 0.1 %, so the discrepancy in §3.3 has little bearing on the search.
- The bound is not tuned to match Table 3.

#### F6 — maximum height for the msh variant
- Rule (p. 307): "a maximum stack height that is assumed to be 2 slots higher than the height of the stacks in the initial layout". Decision: H = T + 2 with T the stack height of the file (F1), e.g. 5 for data3-4.
- The example on p. 307 ("CVS 3–4 … the maximum stack height is set to 6") follows the reversed label definition of the text (3 stacks × 4 items, F3).

#### F7 — height under the unlimited variant
- A clean supply slot lies above the highest item of a clean stack, up to H, and an empty stack provides H slots (def. 2, p. 302). The CVS runs compared with nrel have unconstrained height (p. 306), so the count is unbounded; the third sort key of Fig. 5 (p. 304) needs a finite value.
- Decision: H = n, the number of items of the instance. A stack can never hold more than n items, so H = n is equivalent to unlimited height for feasibility.

### 2.2 Step 2 (F8–F19)

Proposed by Claude on 10 Nov 2026; **none is confirmed by Thom yet**. The implementation follows the "proposed" column; "OPEN" marks the points where the paper contradicts itself or where the reading changes the results.

| # | Point | Proposed | Basis | Category |
|---|---|---|---|---|
| F8 | Enumeration and ties in the greedy | Relocations by donator, then receiver, ascending; the first move with the best key wins. Removes by stack. | Fig. 4 (p. 303) silent | (b) |
| F9 | Greedy priority (4), BB_FLG | **OPEN.** Implemented as printed in Fig. 4: maximum of g − lowest group of the target. | Fig. 4 (p. 303) vs. text and rationale (p. 304) | (b) |
| F10 | Greedy priority (2), BG to a non-empty stack | Minimum of top group of the target − g (≥ 0). | Fig. 4, p. 304 ("packed" stacks) | (b) |
| F11 | GG sort in Fig. 7 | **OPEN.** Differences signed as g − other: key = (g − below) − (g − top of target), ascending. | Fig. 7 (p. 306), p. 306 text | (b) |
| F12 | GG with no item below g, or an empty target | That item counts as group G. | def. 2 (p. 302) | (b) |
| F13 | G | Largest group of the initial layout (n for Caserta). | Table 1 (p. 302) | (b) |
| F14 | Pruning in Fig. 6 | As printed: skip pm if n′_m(L′) > maxMoves. | Fig. 6, p. 305 | — |
| F15 | Time limit | Wall clock (`time.perf_counter`) from the start of the run, including bound and greedy; checked only on entry of perform_compound_moves. | Fig. 5 (p. 304); p. 306 | (b) |
| F16 | FLG | The donator holds an item of the currently lowest group below the moved item. | p. 302 | — |
| F17 | Ties in the sorts of Figs. 5 and 7 | Stable: generation order. | Figs. 5, 7 silent | (b) |
| F18 | Several removes at once | Greedy applies all of Rm (Fig. 4); Fig. 7 offers all removes as alternatives. Only with repeated groups (never on Caserta). | Figs. 4, 7 | — |
| F19 | Optimality abort | Stop when nm(s*) equals n′_m(L_Init), the bound of the initial layout, in moves. | Fig. 5 (p. 304) | — |

#### F9 — greedy priority (4)
- Fig. 4 (p. 303): "determine bb_flgBest ∈ Bb_flg so that the item group difference between the moved item and the item with the lowest item group in the target stack is maximum".
- Text (p. 304): "the relocation with maximum item group difference between the moved item and the topmost item of the target stack is chosen. This way, it is tried to avoid an early re-relocation of the moved item that would likely be again of type BB."
- The moved item g blocks the lowest item of the target, min(r) < g, and must be relocated again when min(r) is retrieved. The maximum of g − min(r) chooses the target whose lowest item comes out *earliest*, so the re-relocation is as early as possible, against the stated rationale. Fig. 7 (p. 306) sorts the same quantity, g − min(r), *ascending*, with the rationale "helps to avoid multiple BB relocations of the same items".
- Readings: (A) Fig. 4 as printed, max(g − min(r)) — implemented; (B) max(min(r) − g), i.e. the minimum of g − min(r), consistent with the rationale and with Fig. 7; (C) the text: max |g − top(r)|.
- Greedy alone over all 840 instances (§4.3): on 10–10, (A) 1171.28, (B) 145.70, (C) 217.00 relocations (H = n). For comparison, the Kim & Hong heuristic in Table 3 (p. 307) needs 178.60. The paper reports no greedy results.
- Effect on the search (smoke run, §4.4): with (A), 10–10 #2 at H = 12 did not improve on the greedy plan (484 relocations) within 20 s.

#### F11 — GG sort
- Fig. 7 (p. 306): "sort Gg ascending by the value of (diffBefore − diffAfter), where diffBefore is the item group difference between the moved item m and the item in the slot directly below m, and diffAfter is the item group difference between m and the topmost item of the target stack". Text (p. 306): "the sorting of GG moves strengthens this ability, since it prefers layouts with higher clean supply".
- In a GG move, g ≤ below and g ≤ top(r). With absolute differences the key is below − top(r), and ascending order prefers a small `below` and a large top(r): the donator's new top is low and the target's top drops from far above g, which *lowers* the clean supply (Table 1). With signed differences g − other the key is top(r) − below, and ascending order prefers the higher clean supply that the text states.
- Proposed: signed (implemented). Equivalent to absolute differences sorted descending.

#### F12 — missing reference item in the GG key
- A GG move of the only item of a stack has no item below m; a GG move to an empty stack has no top item. Def. 2 (p. 302): "an empty stack provides H clean supply slots of group G". Proposed: the missing item counts as group G, so the stack that becomes empty, or the empty target, acts like a stack topped by group G.

#### F14 — pruning bound in Fig. 6
- Fig. 6 and p. 305 skip pm if "n′_m(L′) … is greater than maxMoves", with maxMoves the moves from L that still improve s*. After pm, at most maxMoves − 1 moves remain from L′, so the printed test prunes one level later than it could. Implemented as printed; it cannot cut off an improvement.

#### F15 — time limit
- p. 306: "a time limit of 60 s"; Fig. 5 checks it on entry only. The run's clock starts before the bound and the greedy. An overrun is at most one call of determine_compound_moves (bounded by cmStopThreshold); the step-3 runtimes report it.

## 3. Step 1 — deterministic checks (10 Nov 2026)

Script `docs/validation/data/forster_bortfeldt_2012/step1_checks.py` (rl Python 3.11), output `step1_output.txt`. Tolerances fixed before the run: 1a exactly 25; 1b within half a unit of the last printed digit (0.005); 1c within one unit (0.01).

### 3.1 Fig. 3 lower bound (1a)
(n′_REM, n′_BG, n′_non-BG) = (16, 8, 1), bound 25. **Pass.**

### 3.2 CVS label → file class by item count (1b)
| CVS (Table 2) | items (Table 2) | file classes with that item count | same-label file: items | Table 2 n_b | same-label file n_b | item-count candidate n_b |
|---|---|---|---|---|---|---|
| 3–3 | 9 | 3-3 | 9 | 3.70 | 3.700 | — |
| 3–4 | 16 | 4-4 | **12** | 4.78 | 4.775 | 4-4: 7.300 |
| 3–5 | 25 | 5-5 | **15** | 5.78 | 5.775 | 5-5: 13.150 |
| 3–6 | 36 | 6-6 | **18** | 7.20 | 7.200 | 6-6: 21.450 |
| 3–7 | 21 | 3-7 | 21 | 8.18 | 8.175 | — |
| 3–8 | 24 | 3-8, 4-6 | 24 | 9.30 | **9.250** | 4-6: 11.075 |
| 4–4 | 16 | 4-4 | 16 | 7.30 | 7.300 | — |
| 4–5 | 20 | 4-5, 5-4 | 20 | 10.18 | 10.175 | 5-4: 10.600 |
| 4–6 | 24 | 3-8, 4-6 | 24 | 11.08 | 11.075 | 3-8: 9.250 |
| 4–7 | 28 | 4-7 | 28 | 13.40 | 13.400 | — |
| 5–4 | 20 | 4-5, 5-4 | 20 | 10.60 | 10.600 | 4-5: 10.175 |
| 5–5 | 25 | 5-5 | 25 | 13.15 | 13.150 | — |
| 5–6 | 30 | 5-6 | 30 | 16.80 | 16.800 | — |
| 5–7 | 35 | 5-7 | 35 | 19.13 | 19.125 | — |
| 5–8 | 40 | 5-8 | 40 | 22.20 | 22.200 | — |
| 5–9 | 45 | 5-9 | 45 | 25.10 | **25.050** | — |
| 5–10 | 50 | 5-10 | 50 | 27.73 | 27.725 | — |
| 6–6 | 36 | 6-6 | 36 | 21.45 | 21.450 | — |
| 6–10 | 60 | 6-10, 10-6 | 60 | 36.08 | 36.075 | 10-6: 42.250 |
| 10–6 | 60 | 6-10, 10-6 | 60 | 42.25 | 42.250 | 6-10: 36.075 |
| 10–10 | 100 | 10-10 | 100 | 70.53 | 70.525 | — |

- By item count alone there is no bijection: rows 3–4, 3–5 and 3–6 can only map to 4-4, 5-5 and 6-6, which are already the rows 4–4, 5–5 and 6–6, and the file classes 3-4, 3-5 and 3-6 (12, 15, 18 items) match no row. **Fail** under the rule fixed in advance; resolved by decision F3 (stack-count column).
- The independent check rejects the item-count candidates for those three rows: their n_b (7.30, 13.15, 21.45) are far from 4.78, 5.78 and 7.20.
- The same-label mapping (adopted as F3) CVS T–S → `data<T>-<S>` matches the stack count of Table 2 in all 21 rows and the mean n_b in 19 of 21 rows within 0.005. The exceptions are 3–8 (9.25 against 9.30) and 5–9 (25.05 against 25.10), both 0.05 = 2 items over 40 instances. Nine matches sit exactly on the 0.005 boundary (3–4, 3–5, 3–7, 4–5, 4–6, 5–7, 5–10, 6–10, 10–10; e.g. 4.775 printed as 4.78).
- In 18 rows the item count in Table 2 equals the stack count × the label's first number; in rows 3–4 to 3–6 it equals the stack count squared.

### 3.3 Mean lower bound vs. nrel − LB gap (1c)
LB gap is absolute (p. 307, F4), so the paper's mean bound (relocation part) is nrel − LB gap. File classes by F3.

| CVS | nrel | LB gap | nrel − gap | ours: n′_BG + n′_non-BG | diff | ours: n_b |
|---|---|---|---|---|---|---|
| 3–3 | 4.95 | 1.10 | 3.85 | 4.000 | +0.150 | 3.700 |
| 3–4 | 6.05 | 1.20 | 4.85 | 5.075 | +0.225 | 4.775 |
| 3–5 | 6.85 | 1.05 | 5.80 | 5.950 | +0.150 | 5.775 |
| 3–6 | 8.28 | 1.03 | 7.25 | 7.400 | +0.150 | 7.200 |
| 3–7 | 9.20 | 1.00 | 8.20 | 8.325 | +0.125 | 8.175 |
| 3–8 | 10.45 | 1.18 | 9.27 | 9.375 | +0.105 | 9.250 |
| 4–4 | 9.90 | 2.43 | 7.47 | 7.700 | +0.230 | 7.300 |
| 4–5 | 12.63 | 2.43 | 10.20 | 10.425 | +0.225 | 10.175 |
| 4–6 | 13.70 | 2.53 | 11.17 | 11.325 | +0.155 | 11.075 |
| 4–7 | 15.78 | 2.35 | 13.43 | 13.475 | +0.045 | 13.400 |
| 5–4 | 15.00 | 4.25 | 10.75 | 11.000 | +0.250 | 10.600 |
| 5–5 | 18.63 | 5.35 | 13.28 | 13.525 | +0.245 | 13.150 |
| 5–6 | 21.83 | 4.98 | 16.85 | 17.025 | +0.175 | 16.800 |
| 5–7 | 23.58 | 4.43 | 19.15 | 19.225 | +0.075 | 19.125 |
| 5–8 | 26.73 | 4.53 | 22.20 | 22.325 | +0.125 | 22.200 |
| 5–9 | 29.45 | 4.60 | 24.85 | 25.175 | +0.325 | 25.050 |
| 5–10 | 31.85 | 4.13 | 27.72 | 27.800 | +0.080 | 27.725 |
| 6–6 | 29.68 | 7.93 | 21.75 | 21.825 | +0.075 | 21.450 |
| 6–10 | 43.60 | 7.53 | 36.07 | 36.175 | +0.105 | 36.075 |
| 10–6 | 75.65 | 33.35 | 42.30 | 42.725 | +0.425 | 42.250 |
| 10–10 | 116.90 | 46.38 | 70.52 | 70.775 | +0.255 | 70.525 |

- **Fail** on all 21 sizes under the pre-registered tolerance; ours is higher everywhere. Downgraded to a reported discrepancy by F5. n′_BG = n_b in every layout, so our n_b part reproduces Table 2 (§3.2); the excess is in n′_non-BG: ours fires on 3–19 of 40 instances per size, the paper's figures imply 0–12.
- Table 3 is inconsistent with the Proposition for CVS 5–9: nrel − gap = 24.85 is below the mean n_b, both Table 2's 25.10 and the files' 25.05. A bound that includes n′_BG = n_b cannot have a smaller mean. For 3–8 the same holds against Table 2 (9.27 < 9.30), though not against the files (9.25). For 5–10, 6–10 and 10–10, nrel − gap is 0.005 below the files' n_b (within rounding). So 1c as specified cannot pass for 5–9 with any implementation of the Proposition (F5).
- Diagnostic (scratch, not part of the code): three readings of g_top, per size the number of the 40 initial layouts with n′_non-BG = 1:

| CVS | implied by paper: 40·(nrel − gap) − Σn_b | Table 1 (ours) | Fig. 3 example (topmost badly placed item) | top item regardless of placement |
|---|---|---|---|---|
| 3–3 | 6.0 | 12 | 12 | 5 |
| 3–4 | 3.0 | 12 | 12 | 1 |
| 3–5 | 1.0 | 7 | 7 | 0 |
| 3–6 | 2.0 | 8 | 8 | 2 |
| 3–7 | 1.0 | 6 | 6 | 0 |
| 3–8 | 0.8 | 5 | 5 | 0 |
| 4–4 | 6.8 | 16 | 16 | 3 |
| 4–5 | 1.0 | 10 | 10 | 3 |
| 4–6 | 3.8 | 10 | 10 | 0 |
| 4–7 | 1.2 | 3 | 3 | 0 |
| 5–4 | 6.0 | 16 | 14 | 6 |
| 5–5 | 5.2 | 15 | 15 | 8 |
| 5–6 | 2.0 | 9 | 9 | 4 |
| 5–7 | 1.0 | 4 | 4 | 2 |
| 5–8 | 0.0 | 5 | 5 | 1 |
| 5–9 | −8.0 | 5 | 5 | 1 |
| 5–10 | −0.2 | 3 | 3 | 0 |
| 6–6 | 12.0 | 15 | 14 | 2 |
| 6–10 | −0.2 | 4 | 3 | 0 |
| 10–6 | 2.0 | 19 | 18 | 9 |
| 10–10 | −0.2 | 10 | 9 | 5 |

  The implied counts carry a rounding error of up to ±0.4 (two printed values with 0.005 each, × 40). No reading reproduces them. Both alternative readings give a value of min g_top no larger than Table 1's, so they are valid but weaker bounds; the third is not what Table 1 or the proof (p. 302–303) states and is listed only because it comes closest on several sizes.

## 4. Step 2 — greedy and search on small instances (10 Nov 2026)

Code: `state.py` (incremental layout: running minima and badly placed counts per stack), `search.py` (Figs. 4–7), `plan.py` (independent plan check on plain lists). Tests: `test_forster_bortfeldt.py`, 19 tests, 0.65 s:

    python -m unittest algorithms.CRP_Time.heuristic.forster_bortfeldt_2012_tree.test_forster_bortfeldt

### 4.1 Unit tests
- **State vs. definitions:** on 300 random layouts (2–5 stacks, H 3–6, repeated groups allowed) and up to 40 random moves each, n_b, the bound and the clean supply of `state.Bay` equal the plain definitions of `layout.py` (bound) and of def. 2 / Table 1 slot by slot (clean supply).
- **Bound:** Fig. 3 gives (16, 8, 1) = 25. On all 40 Caserta 3x3 instances at H = 5 the bound (with n′_non-BG) is at most the exact optimum. The exact optimum comes from an IDA* in the test file that uses plain tuples and only the Kim–Hong bound n + n_b, so it does not depend on the code under test.
- **Move types (Sec. 4.2):** Fig. 3: 15 from stack 3 to stack 1 is FLG_BB, 6 from stack 4 to stack 1 FOG_BB, 2 from stack 1 to stack 2 GG, to stack 3 GB; FLG_BG and FOG_BG (to a non-empty and an empty stack) on small layouts; a full stack is no receiver.
- **Greedy (Fig. 4), one case per priority group:** (1) removes; (2) BG to a non-empty stack with the smallest difference, and the tie 5→6 vs. 7→8 resolved to the first (F8); (3) BG to an empty stack with the smallest group; (4) FLG_BB: layout [1,9][2,8][3,7] gives 9 → stack 2 (9 − 2 = 7) under F9 (A); reading (B) and the text would give stack 3. Fig. 3: greedy plan valid, 26 relocations.
- **Productive moves (Fig. 7):** only removes when one exists; all BG relocations when one exists; layout [1,9][7,3][6,4][8,5][2,10] gives the three FLG_BB moves of 9 with the smallest 9 − min(r) (4, 5, 6; 7 cut) and the two GG moves with the smallest signed key (3→4: −3, 3→5: −2; 4→5: −1 cut, F11); the single-item GG key uses G (F12).
- **Plan check:** an empty plan, a non-top item, a remove out of group order, a full receiver and a non-empty end are rejected.
- **Search:** see §4.2; with a clock that advances 1 s per call and a 0.5 s limit the search returns the greedy plan without expanding a node; on 3x4 #2 the greedy reaches the bound and the search stops at once (Fig. 5).

### 4.2 Caserta 3x3 and 3x4, no time limit
Script `docs/validation/data/forster_bortfeldt_2012/step2_small.py`, output `step2_small.txt`. Means over 40 instances, in relocations; "=exact" and "=LB" count instances where the search reaches the exact optimum or the bound.

| class | H | LB | greedy | search | exact | =exact | =LB | nodes |
|---|---|---|---|---|---|---|---|---|
| 3-3 | n | 4.00 | 6.40 | 4.95 | – | – | 12 | 3.6 |
| 3-3 | T + 2 | 4.00 | 6.50 | 5.03 | 4.97 | 39 | 11 | 3.9 |
| 3-4 | n | 5.08 | 8.68 | 6.08 | – | – | 12 | 7.7 |
| 3-4 | T + 2 | 5.08 | 8.65 | 6.10 | 6.03 | 37 | 12 | 7.6 |

Every plan passes `check_plan`; no search plan is below the bound, above the greedy, or below the exact optimum. These means come from the code under the proposed F8–F19 without a time limit; they are not step-3 results.

### 4.3 Greedy readings of priority (4) (F9)
Script `step2_greedy_readings.py`, output `step2_greedy_readings.txt`: greedy alone (no search) on all 840 instances, mean relocations; every plan checked. (A) Fig. 4 as printed, (B) minimum of g − min(r), (C) text p. 304, maximum |g − top(r)|.

| CVS | A, H = n | B, H = n | C, H = n | A, T + 2 | B, T + 2 | C, T + 2 |
|---|---|---|---|---|---|---|
| 3–3 | 6.40 | 5.58 | 6.10 | 6.50 | 5.70 | 6.17 |
| 3–8 | 17.40 | 11.62 | 15.25 | 16.93 | 11.68 | 15.18 |
| 4–7 | 34.42 | 18.20 | 23.90 | 31.38 | 18.65 | 23.75 |
| 5–10 | 99.47 | 37.67 | 53.15 | 82.65 | 39.55 | 55.73 |
| 6–10 | 191.18 | 53.85 | 76.00 | 137.07 | 57.42 | 77.78 |
| 10–6 | 495.20 | 101.88 | 152.12 | 280.23 | 115.62 | 152.55 |
| 10–10 | 1171.28 | 145.70 | 217.00 | 486.43 | 160.32 | 226.88 |

All 21 sizes are in the output file; (B) is the lowest on every size and both variants. The Fig. 3 trace under (A): the first two moves put 15 and 7 onto stack 1, whose 2 is the next item after 1, so both are moved again after the first remove.

### 4.4 Smoke run (development check, before the tests)
Greedy + search with a 20 s limit on the Fig. 3 layout and on #1–2 of 3-3, 3-4, 4-4, 5-5, 6-6 and 10-10 at H = n and H = T + 2: all plans valid. 6-6 #2 (H = n) and all four 10-10 runs hit the limit; 10-10 #2 at H = 12 stayed at the greedy plan (484 relocations, 3063 nodes), the other three 10-10 runs reached 120–133.

## 5. Step 3 protocol (pre-registered)

Fixed before any step-3 result (task of 10 Nov 2026 and Thom's additions of 10 Nov 2026):
- Smoke run first: 10 instances per size, runtimes reported before the full run.
- Full run: all 840 Caserta instances (F3), two variants: H = n (F7), compared with nrel of Table 3; H = T + 2 (F6), compared with nrel − msh. Parameters of p. 306, time limit 60 s (F15).
- Criterion: per size, the mean nrel is within max(0.10; 1 %) of Table 3, for every size except CVS 10–6 and 10–10, which are reported only. CVS 5–9 stays in the criterion; if it is the only failing size, the verdict is "fail on 5–9", with a note that this row of Table 3 is internally inconsistent (§3.3); it is not excluded.
- Report per size: mean nrel, difference with the paper, mean runtime, the number of instances that hit the 60 s limit, and how many ended by the optimality abort (bound reached), by the limit, or by exhausting the tree.
- If a size fails: rerun it without a time limit, and rerun it with the Fig. 3 reading of g_top; both are diagnostics only, the verdict stays with the main 60 s run.
