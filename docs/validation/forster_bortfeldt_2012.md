# Forster & Bortfeldt (2012) — validation log

Paper: F. Forster, A. Bortfeldt, "A tree search procedure for the container relocation problem", *Computers & Operations Research* 39 (2012) 299–309. DOI 10.1016/j.cor.2011.04.004. Page numbers below are the journal's printed pages (PDF page = printed page − 298). Not to be confused with the multi-bay retrieval paper in Oper. Res. Proc. 2011, 257–262, which `algorithms/CRP_Time/heuristic/forster_bortfeldt_2012/` cites (step 7 of the thesis plan).

Code: `algorithms/CRP_Time/heuristic/forster_bortfeldt_2012_tree/` (standalone; imports nothing from `algorithms/CRP_D` or any other algorithm folder). Scripts and raw output: `docs/validation/data/forster_bortfeldt_2012/`.

Scope set by Wei (10 Nov 2026): the C&OR tree search, standalone in CRP-Time, validated first unrestricted with the number of relocations as the objective, in its own layout simulation outside `CRP_R.step`. Integration with f2/f2vert comes later.

## Summary

Status 10 Nov 2026: **stopped after step 1** (deterministic checks), as the task prescribes when 1a, 1b or 1c does not match.
- 1a: the lower bound of the Fig. 3 layout is 25 = 16 + 8 + 1. Pass.
- 1b: no bijective mapping from CVS labels to Caserta file classes exists by item count: Table 2 lists 16, 25 and 36 items for CVS 3–4, 3–5 and 3–6, while the file classes data3-4, data3-5 and data3-6 hold 12, 15 and 18 items, and no row lists 12, 15 or 18. Fail (§3.2).
- 1c: the mean lower bound (relocation part) is above nrel − LB gap of Table 3 on all 21 sizes, by +0.045 to +0.425. For CVS 5–9, Table 3 implies a mean bound below the mean number of badly placed items, which the Proposition rules out. Fail (§3.3).
No search code is written yet. Steps 2–4 wait for the decisions F3, F5 and F6 (§2).

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

Proposed by Claude on 10 Nov 2026; none is confirmed by Thom yet. Categories as in the validation workflow: (a) required by the unified 2D setting, (b) a choice the paper leaves open, (c) a real deviation from the paper.

| # | Point | Decision | Basis | Category |
|---|---|---|---|---|
| F1 | Caserta file format | `data<T>-<S>-<k>.dat`: first line S n, then one line per stack: height, then groups bottom to top. | `benchmark/Caserta_dataset/Data Format.docx`; Table 2's stack counts (p. 306) | — |
| F2 | g_top of a stack whose top item is well placed | Table 1 as printed: n(L) + 1 (`layout.py`, `g_top`). | Table 1 (p. 302); Fig. 3 example (p. 303) differs | (b) |
| F3 | CVS label → Caserta file | **Open.** The rule "by item count only" finds no mapping (§3.2). | Table 2 (p. 306), text p. 306–307 | — |
| F4 | LB gap | Absolute; mean bound = nrel − LB gap. | p. 307 | — |
| F5 | Step 1c | **Open.** As specified it cannot pass for CVS 5–9 (§3.3). | Tables 2–3 (p. 306–307) | — |
| F6 | H for "nrel − msh" | **Open.** T + 2 per file gives 5 for data3-4; p. 307 gives 6 for CVS 3–4. | p. 307 | — |
| F7 | Clean supply under unlimited height | **Open, needed for step 2.** | p. 302, p. 306 | — |

### F2 — g_top
- Table 1 (p. 302) defines g_top(s) by the *top* item: its group if it is badly placed, n(L) + 1 otherwise. In the Fig. 3 example (p. 303) the list is {11, 14, 15, 6}; stack 1 is 12, 8, 11, 2 (bottom to top), whose top item 2 is well placed, so Table 1 gives 17. 11 is the topmost *badly placed* item of stack 1, which is not on top.
- The proof of (iii) (p. 302–303) argues about "badly placed item[s] in topmost position", which matches Table 1.
- On Fig. 3 both readings give n′_non-BG = 1 (min 6 > 4 either way), so 1a cannot tell them apart. On the 840 Caserta initial layouts they differ on 6 instances (one or two each in 5–4, 6–6, 6–10, 10–6 and 10–10).
- Open sub-point for step 2: n(L) + 1 is meant to exceed every group, which holds when the groups are 1…n(L) (every Caserta initial layout). After removes, groups above n(L) + 1 remain (e.g. 12 items with groups 5…16). The literal value then weakens the bound (never invalidates it). Taking "larger than every group in L" instead would be category (b); not decided.

### F3 — CVS label → Caserta file
- p. 306: "labelled CVS number of stacks × items per stack"; p. 307: CVS 3–4 = 3 stacks × 4 items (12 items).
- Table 2 (p. 306): CVS 3–4 has 4 stacks and 16 items, 3–5 5 stacks and 25 items, 3–6 6 stacks and 36 items; every other row has items = stacks × the first number of the label (e.g. CVS 10–6: 6 stacks, 60 items, so 10 items per stack, against the text).
- Caserta files: `data<T>-<S>` holds S stacks of T items (F1); data3-4: 4 stacks, 12 items.
- See §3.2 for the numbers. Not decided.

### F6 — maximum height for the msh variant
- "2 slots higher than the height of the stacks in the initial layout", example CVS 3–4 → 6 (p. 307). With data3-4 (3 items per stack, F1) the rule gives 5; the example fits the text's 3 stacks × 4 items. Needed for step 3.

### F7 — clean supply under unlimited height
- A clean supply slot lies above the highest item of a clean stack, up to H, and an empty stack provides H slots (def. 2, p. 302). The CVS runs that are compared with nrel have unconstrained height (p. 306), so the count is unbounded. The third sort key of Fig. 5 (p. 304) needs a finite value. To be decided before step 2 (e.g. a bound of n(L) as the effective height); the paper does not say what was used.

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

- By item count alone there is no bijection: rows 3–4, 3–5 and 3–6 can only map to 4-4, 5-5 and 6-6, which are already the rows 4–4, 5–5 and 6–6, and the file classes 3-4, 3-5 and 3-6 (12, 15, 18 items) match no row. **Fail** under the rule fixed in advance.
- The independent check rejects the item-count candidates for those three rows: their n_b (7.30, 13.15, 21.45) are far from 4.78, 5.78 and 7.20.
- For information only (not a mapping established by the rule): the same-label mapping CVS T–S → `data<T>-<S>` matches the stack count of Table 2 in all 21 rows and the mean n_b in 19 of 21 rows within 0.005. The exceptions are 3–8 (9.25 against 9.30) and 5–9 (25.05 against 25.10), both 0.05 = 2 items over 40 instances. Seven matches sit exactly on the 0.005 boundary (e.g. 4.775 printed as 4.78).
- In 18 rows the item count in Table 2 equals the stack count × the label's first number; in rows 3–4 to 3–6 it equals the stack count squared.

### 3.3 Mean lower bound vs. nrel − LB gap (1c)
LB gap is absolute (p. 307, F4), so the paper's mean bound (relocation part) is nrel − LB gap. Same-label file classes (F3 open).

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

- **Fail** on all 21 sizes; ours is higher everywhere. n′_BG = n_b in every layout, so our n_b part reproduces Table 2 (§3.2); the excess is in n′_non-BG: ours fires on 3–19 of 40 instances per size, the paper's figures imply 0–12.
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
