# Forster & Bortfeldt (2012) — validation log

Paper: F. Forster, A. Bortfeldt, "A tree search procedure for the container relocation problem", *Computers & Operations Research* 39 (2012) 299–309. DOI 10.1016/j.cor.2011.04.004. Page numbers below are the journal's printed pages (PDF page = printed page − 298). Not to be confused with the multi-bay retrieval paper in Oper. Res. Proc. 2011, 257–262, which `algorithms/CRP_Time/heuristic/forster_bortfeldt_2012/` cites (step 7 of the thesis plan).

Code: `algorithms/CRP_Time/heuristic/forster_bortfeldt_2012_tree/` (standalone; imports nothing from `algorithms/CRP_D` or any other algorithm folder). Scripts and raw output: `docs/validation/data/forster_bortfeldt_2012/`.

Scope set by Wei (10 Nov 2026): the C&OR tree search, standalone in CRP-Time, validated first unrestricted with the number of relocations as the objective, in its own layout simulation outside `CRP_R.step`. Integration with f2/f2vert comes later.

## Summary

Status 10 Nov 2026: **step 2 done with Thom's decisions F8–F19; step 3 smoke run and profiling done; F20 and F21 decided by Thom; full run (§7) at the F20 commit.**
- Step 1 (§3): the Fig. 3 bound is 25 (pass). The CVS labels map to the Caserta files by the stack-count column of Table 2 (F3; the item-count key failed on a misprint). The mean bound is above nrel − LB gap of Table 3 on all sizes; this is a reported discrepancy, not a stop criterion (F5).
- Step 2 (§4): greedy (Fig. 4), productive moves (Fig. 7), compound moves (Fig. 6) and search (Fig. 5) implemented from the paper; 20 unit tests pass (the 20th for F20). On all 80 Caserta 3x3 and 3x4 instances, at H = n and H = T + 2, every plan is valid, none is below the bound, none is worse than the greedy, and at H = T + 2 none is below the exact optimum (IDA*).
- Deviation from the printed figure: greedy priority (4) takes the *minimum* of g − lowest group of the target (F9, reading B), where Fig. 4 and p. 304 print "maximum"; decided by Thom after seeing the greedy results of §4.3.
- Step 3 (§5–6): smoke run, instances 1–10 per size, both height variants, sequential, 420 valid plans; 81 runs hit the 60 s limit, all on sizes 5-8 and larger; maximum wall clock 60.07 s; the full run would take about 6.4 h. Mean nrel is not compared with Table 3 before the full run.
- Profiling (§6.1): no deviation from Figs. 5–7 as printed. Almost all work on the wide bays goes to nodes that can no longer improve s*, a consequence of the pruning test as printed in Fig. 6 (F14), which is one move weaker than its stated purpose (p. 305). Adopted by Thom as F20, an implementation optimisation with the same plans without a time limit; duplicate compound moves stay (F21). The rest is a per-node cost of 0.75–2.7 ms in Python.

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

Proposed by Claude and decided by Thom on 10 Nov 2026, after step 2. The decisions on F9 and F11 were taken after the greedy results under readings A, B and C (§4.3) had been seen.

| # | Point | Decision | Basis | Category |
|---|---|---|---|---|
| F8 | Enumeration and ties in the greedy | Relocations by donator, then receiver, ascending; the first move with the best key wins. Removes by stack. | Fig. 4 (p. 303) silent | (b) |
| F9 | Greedy priority (4), BB_FLG | Reading (B): the FLG_BB relocation with the **minimum** of g − lowest group of the target. | Rationale p. 304; Fig. 7 (p. 306); Fig. 4 and p. 304 print "maximum" | (c) |
| F10 | Greedy priority (2), BG to a non-empty stack | Minimum of top group of the target − g (≥ 0). | Fig. 4, p. 304 ("packed" stacks) | (b) |
| F11 | GG sort in Fig. 7 | Differences signed as moved − other: key = (g − below) − (g − top of target), ascending. | Fig. 7 (p. 306), p. 306 text | (b) |
| F12 | GG with no item below g, or an empty target | That item counts as group G. | def. 2 (p. 302) | (b) |
| F13 | G | Largest group of the initial layout (n for Caserta). | Table 1 (p. 302) | (b) |
| F14 | Pruning in Fig. 6 | As printed: skip pm if n′_m(L′) > maxMoves. **Superseded by F20.** | Fig. 6, p. 305 | — |
| F15 | Time limit | Wall clock (`time.perf_counter`) from the start of the run, including bound and greedy; checked only on entry of perform_compound_moves. The maximum wall clock per instance is reported per size. | Fig. 5 (p. 304); p. 306 | (b) |
| F16 | FLG | The donator holds an item of the currently lowest group below the moved item. | p. 302 | — |
| F17 | Ties in the sorts of Figs. 5 and 7 | Stable: generation order. | Figs. 5, 7 silent | (b) |
| F18 | Several removes at once | Greedy applies all of Rm (Fig. 4); Fig. 7 offers all removes as alternatives. Only with repeated groups (never on Caserta). | Figs. 4, 7 | — |
| F19 | Optimality abort | Stop when nm(s*) equals n′_m(L_Init), the bound of the initial layout, in moves. | Fig. 5 (p. 304) | — |

### 2.3 After the profiling (F20–F21)

Proposed by Claude after the profiling of §6.1 and decided by Thom on 10 Nov 2026, before the full step-3 run. The decision was based on runtimes and on plan identity; the smoke-run relocations were recorded in `step3_smoke.jsonl` but neither summarised nor compared with Table 3.

| # | Point | Printed | Decision | Basis | Category |
|---|---|---|---|---|---|
| F20 | Pruning test of Fig. 6 (supersedes F14) | Skip pm if n′_m(L′) > maxMoves. | Skip pm if n′_m(L′) > maxMoves − 1, the move budget from L′. | Fig. 6 and p. 305: maxMoves is "the maximum number of (single) moves to an empty layout starting with current layout L that still ensures an improvement"; pm is skipped if it "would result in a compound move that can no longer lead to a new best solution" | implementation optimisation: same plans without a time limit, not an interpretation change |
| F21 | Duplicate layouts in Cm | Kept (Cm is a list, Figs. 5–6). | Keep, as printed. | Figs. 5–6; the paper is silent on duplicates | — (as printed) |

#### F20 — pruning test of Fig. 6
**Decision (Thom, 10 Nov 2026): adopt the tight test, skip pm if n′_m(L′) > maxMoves − 1, as an implementation optimisation, not an interpretation change.** Argument:
1. maxMoves is defined on p. 305 as the move budget from L ("the maximum number of (single) moves to an empty layout starting with current layout L that still ensures an improvement"), so after pm the budget for L′ is maxMoves − 1. Both Fig. 6 and the sentence on p. 305 print "greater than maxMoves".
2. The bound drops by at most 1 per move: a remove or a BG relocation implies that the non-BG term was 0 before the move; FLG_BB and GG relocations leave n and n_b unchanged. So any compound move that passes the printed test but fails the tight one, and every extension of it, has first sort key |cm| + n′_m = nm(s*) − nm(s); every compound move that can still improve s* has at most nm(s*) − nm(s) − 1. These moves therefore always sort after all useful ones (Fig. 5, first criterion) and never take a useful branch's nSucc slot.
3. Hence, without a time limit, the explored useful branches and the returned s* are identical under both tests; the 339 of 339 identical plans below are consistent with this. The change only affects which runs reach the 60 s limit.
4. Secondary: the paper's runtimes (Table 3, p. 307: 5-8 < 0.1 s, 5-10 0.2 s) are inconsistent with the printed test (5-8 #1: 7.9 s; 5-10 #1: more than 60 s in Python).

Implementation and evidence:
- `search.py` `_determine`: `if nxt.lower_bound() > max_moves - 1: continue`.
- Unit test `test_tight_pruning_returns_the_plan_of_the_printed_test`: printed and tight tests return identical plans without a time limit on all 3-3 and 3-4 instances, at H = n and H = T + 2 (the printed test is obtained by passing maxMoves + 1 to the first-level call).
- Smoke runs that ended before 60 s, at 3210477 (`step3_f14_check.py`, no time limit): 339 of 339 plans identical; total wall clock 883.6 s printed against 93.5 s tight (944 677 against 117 966 nodes; longest run 60.14 s against 14.92 s).
- Profiling (§6.1): on 5-8 #1 and 5-10 #1 at H = n, 97.2 % and 99.5 % of the determine_compound_moves calls lie in nodes that cannot improve s* on arrival; with the tight test, 5-8 #1 ends in 0.13 s instead of 7.9 s and 5-10 #1 searches its tree to the end in 0.19 s instead of hitting the 60 s limit (instrumented times).
- Step-2 table (§4.2) at the F20 commit: identical except the mean node counts (`step2_small_f20.txt`).

#### F21 — duplicate compound moves
- The same moves in another order often give the same layout: 31–65 % of Cm before the cut on the three profiled instances, and 278–5406 duplicates among the kept nSucc. Fig. 6 collects compound moves in a list and the paper does not mention removing duplicates, so they are kept. With the tight test (F20), 13–96 duplicates remain among the kept entries.
- **Decision (Thom, 10 Nov 2026): keep duplicates in Cm, as printed.** Removing them would change which branches fill the nSucc slots and could change the result.

#### F9 — greedy priority (4)
- Fig. 4 (p. 303): "determine bb_flgBest ∈ Bb_flg so that the item group difference between the moved item and the item with the lowest item group in the target stack is maximum".
- Text (p. 304): "the relocation with maximum item group difference between the moved item and the topmost item of the target stack is chosen. This way, it is tried to avoid an early re-relocation of the moved item that would likely be again of type BB."
- The moved item g blocks the lowest item of the target, min(r) < g, and must be relocated again when min(r) is retrieved. The maximum of g − min(r) chooses the target whose lowest item comes out *earliest*, so the re-relocation is as early as possible, against the stated rationale. Fig. 7 (p. 306) sorts the same quantity, g − min(r), *ascending*, with the rationale "helps to avoid multiple BB relocations of the same items".
- Readings: (A) Fig. 4 as printed, max(g − min(r)); (B) max(min(r) − g), i.e. the minimum of g − min(r), consistent with the rationale and with Fig. 7; (C) the text: max |g − top(r)|.
- Greedy alone over all 840 instances (§4.3): on 10–10, (A) 1171.28, (B) 145.70, (C) 217.00 relocations (H = n). For comparison, the Kim & Hong heuristic in Table 3 (p. 307) needs 178.60. The paper reports no greedy results.
- Effect on the search (development smoke run, §4.4): with (A), 10–10 #2 at H = 12 did not improve on the greedy plan (484 relocations) within 20 s.
- **Decision (Thom, 10 Nov 2026): reading (B)**, a deviation from the printed figure: Fig. 4 and the sentence on p. 304 both print "maximum". Textual basis: (1) the stated rationale on p. 304, "avoid an early re-relocation" — the maximum targets the stack whose lowest item is retrieved soonest; (2) Fig. 7 (p. 306) sorts the same quantity (lowest group in the target) ascending; (3) Fig. 4 and Fig. 7 both use the lowest group in the target, hence (B) over the text's top item (C). Secondary, empirical: under (A) the greedy needs about 1171 relocations on 10–10 (H = n), about 6.6× Kim & Hong in Table 3 (178.60). The decision was taken after the greedy results under A, B and C had been seen (§4.3).
- Implemented in `search.py`, `greedy` (key g − min(r), smallest wins, ties by F8). If a size fails in step 3, it is rerun under (C) as a diagnostic only.

#### F11 — GG sort
- Fig. 7 (p. 306): "sort Gg ascending by the value of (diffBefore − diffAfter), where diffBefore is the item group difference between the moved item m and the item in the slot directly below m, and diffAfter is the item group difference between m and the topmost item of the target stack". Text (p. 306): "the sorting of GG moves strengthens this ability, since it prefers layouts with higher clean supply".
- In a GG move, g ≤ below and g ≤ top(r). With absolute differences the key is below − top(r), and ascending order prefers a small `below` and a large top(r): the donator's new top is low and the target's top drops from far above g, which *lowers* the clean supply (Table 1). With signed differences g − other the key is top(r) − below, and ascending order prefers the higher clean supply that the text states.
- Decision (Thom, 10 Nov 2026): signed differences, moved minus other (implemented). Equivalent to absolute differences sorted descending.
- The same convention (moved − other) makes Fig. 4 read as (A) (maximum of g − min(r)), which F9 rejects. Figs. 4 and 7 therefore genuinely conflict: no single sign convention makes both figures agree with their stated rationales.

#### F12 — missing reference item in the GG key
- A GG move of the only item of a stack has no item below m; a GG move to an empty stack has no top item. Def. 2 (p. 302): "an empty stack provides H clean supply slots of group G". Decision: the missing item counts as group G, so the stack that becomes empty, or the empty target, acts like a stack topped by group G.

#### F14 — pruning bound in Fig. 6
- Fig. 6 and p. 305 skip pm if "n′_m(L′) … is greater than maxMoves", with maxMoves the moves from L that still improve s*. After pm, at most maxMoves − 1 moves remain from L′, so the printed test prunes one level later than it could. Implemented as printed at first; it cannot cut off an improvement. **Superseded by F20** (Thom, 10 Nov 2026), after the profiling of §6.1 showed the cost of the printed test.

#### F15 — time limit
- p. 306: "a time limit of 60 s"; Fig. 5 checks it on entry only. The run's clock starts before the bound and the greedy. An overrun is at most one call of determine_compound_moves (bounded by cmStopThreshold); the step-3 runtimes report it: per size the maximum wall clock per instance (Thom).

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
- **Greedy (Fig. 4), one case per priority group:** (1) removes; (2) BG to a non-empty stack with the smallest difference, and the tie 5→6 vs. 7→8 resolved to the first (F8); (3) BG to an empty stack with the smallest group; (4) FLG_BB under F9 (B): layout [1,9][2,8][3,7] gives 9 → stack 3 (9 − 3 = 6 < 9 − 2 = 7), where (A) gives stack 2; layout [1,9][2,6][4,8] gives 9 → stack 3 (9 − 4 = 5), where the text (C) gives stack 2 (|9 − 6| = 3). Fig. 3: greedy plan valid, 13 relocations (26 under A).
- **Productive moves (Fig. 7):** only removes when one exists; all BG relocations when one exists; layout [1,9][7,3][6,4][8,5][2,10] gives the three FLG_BB moves of 9 with the smallest 9 − min(r) (4, 5, 6; 7 cut) and the two GG moves with the smallest signed key (3→4: −3, 3→5: −2; 4→5: −1 cut, F11); the single-item GG key uses G (F12).
- **Plan check:** an empty plan, a non-top item, a remove out of group order, a full receiver and a non-empty end are rejected.
- **Search:** see §4.2; with a clock that advances 1 s per call and a 0.5 s limit the search returns the greedy plan without expanding a node; on 3x4 #2 the greedy reaches the bound and the search stops at once (Fig. 5).

### 4.2 Caserta 3x3 and 3x4, no time limit
Script `docs/validation/data/forster_bortfeldt_2012/step2_small.py`, output `step2_small.txt`. Means over 40 instances, in relocations; "=exact" and "=LB" count instances where the search reaches the exact optimum or the bound.

| class | H | LB | greedy | search | exact | =exact | =LB | nodes |
|---|---|---|---|---|---|---|---|---|
| 3-3 | n | 4.00 | 5.58 | 4.95 | – | – | 12 | 3.5 |
| 3-3 | T + 2 | 4.00 | 5.70 | 5.03 | 4.97 | 39 | 11 | 3.8 |
| 3-4 | n | 5.08 | 6.95 | 6.08 | – | – | 12 | 7.3 |
| 3-4 | T + 2 | 5.08 | 6.97 | 6.10 | 6.03 | 37 | 12 | 7.0 |

Greedy reading (B) (F9), the code of the step-2 commit. Every plan passes `check_plan`; no search plan is below the bound, above the greedy, or below the exact optimum. No time limit; not step-3 results.

At the F20 commit (`step2_small_f20.txt`) every column is identical except the mean node counts: 1.7, 1.6, 4.5, 4.2.

The first version of this table used reading (A) (output kept as `step2_small_readingA.txt`): greedy 6.40, 6.50, 8.68, 8.65 in the row order above; search, exact, =exact and =LB identical; nodes 3.6, 3.9, 7.7, 7.6.

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
Under greedy reading (A). Greedy + search with a 20 s limit on the Fig. 3 layout and on #1–2 of 3-3, 3-4, 4-4, 5-5, 6-6 and 10-10 at H = n and H = T + 2: all plans valid. 6-6 #2 (H = n) and all four 10-10 runs hit the limit; 10-10 #2 at H = 12 stayed at the greedy plan (484 relocations, 3063 nodes), the other three 10-10 runs reached 120–133.

## 5. Step 3 protocol (pre-registered)

Fixed before any step-3 result (task of 10 Nov 2026 and Thom's additions of 10 Nov 2026):
- Runs are sequential (Thom), one instance at a time, on an otherwise idle machine.
- Smoke run first: instances 1–10 per size, both height variants; runtimes and time-limit hits reported, then stop before the full run (Thom).
- Full run at the F20 commit (Thom, 10 Nov 2026; the smoke run and the profiling, §6, were at 3210477 with the printed test): all 840 Caserta instances (F3), two variants: H = n (F7), compared with nrel of Table 3; H = T + 2 (F6), compared with nrel − msh. Parameters of p. 306, time limit 60 s (F15).
- Criterion: per size, the mean nrel is within max(0.10; 1 %) of Table 3, for every size except CVS 10–6 and 10–10, which are reported only. CVS 5–9 stays in the criterion; if it is the only failing size, the verdict is "fail on 5–9", with a note that this row of Table 3 is internally inconsistent (§3.3); it is not excluded.
- Report per size: mean nrel, difference with the paper, mean and maximum wall clock per instance (F15), the number of instances that hit the 60 s limit, and how many ended by the optimality abort (bound reached), by the limit, or by exhausting the tree.
- If a size fails: rerun it without a time limit, with the Fig. 3 reading of g_top, and with greedy reading (C) (F9); all diagnostics only, the verdict stays with the main 60 s run.
- The diagnostic rerun "without time limit" is capped at 600 s per instance (Thom, 10 Nov 2026); the report states how many instances hit the cap.

## 6. Step 3 smoke run (10 Nov 2026)

Instances 1–10 of every size, H = n and H = T + 2, 420 runs, sequential, time limit 60 s, CRISP 3210477, rl Python 3.11.16. Every plan passed `check_plan`. Two idle VS Code python processes were present during the run; nothing else ran. Data: `docs/validation/data/forster_bortfeldt_2012/step3_smoke.jsonl`, summary `step3_smoke_summary.txt` (`step3_summary.py` without `--paper`). Mean nrel is not compared with Table 3 here: the criterion (§5) applies to all 40 instances per size.

Wall clock per instance in seconds. "limit" = stopped by the 60 s limit, "optimal" = stopped at the lower bound, "exhaust" = the tree was searched to the end (Fig. 5, nSucc and cmStopThreshold) without reaching the bound.

| CVS | H = n mean | max | limit | optimal | exhaust | H = T + 2 mean | max | limit | optimal | exhaust |
|---|---|---|---|---|---|---|---|---|---|---|
| 3-3 | 0.00 | 0.00 | 0 | 4 | 6 | 0.00 | 0.00 | 0 | 4 | 6 |
| 3-4 | 0.00 | 0.01 | 0 | 3 | 7 | 0.00 | 0.01 | 0 | 3 | 7 |
| 3-5 | 0.01 | 0.02 | 0 | 4 | 6 | 0.01 | 0.01 | 0 | 4 | 6 |
| 3-6 | 0.02 | 0.08 | 0 | 3 | 7 | 0.02 | 0.07 | 0 | 2 | 8 |
| 3-7 | 0.03 | 0.10 | 0 | 2 | 8 | 0.03 | 0.11 | 0 | 2 | 8 |
| 3-8 | 0.05 | 0.24 | 0 | 1 | 9 | 0.07 | 0.27 | 0 | 1 | 9 |
| 4-4 | 0.02 | 0.12 | 0 | 1 | 9 | 0.01 | 0.04 | 0 | 1 | 9 |
| 4-5 | 0.02 | 0.06 | 0 | 2 | 8 | 0.02 | 0.04 | 0 | 2 | 8 |
| 4-6 | 0.10 | 0.56 | 0 | 2 | 8 | 0.09 | 0.47 | 0 | 2 | 8 |
| 4-7 | 0.48 | 2.49 | 0 | 0 | 10 | 0.46 | 2.02 | 0 | 0 | 10 |
| 5-4 | 0.04 | 0.14 | 0 | 0 | 10 | 0.04 | 0.12 | 0 | 0 | 10 |
| 5-5 | 0.38 | 2.01 | 0 | 0 | 10 | 0.17 | 0.60 | 0 | 0 | 10 |
| 5-6 | 0.63 | 2.29 | 0 | 0 | 10 | 0.83 | 3.34 | 0 | 0 | 10 |
| 5-7 | 2.97 | 15.97 | 0 | 1 | 9 | 2.98 | 15.15 | 0 | 0 | 10 |
| 5-8 | 16.46 | 60.01 | 1 | 0 | 9 | 13.46 | 60.00 | 1 | 0 | 9 |
| 5-9 | 32.54 | 60.05 | 3 | 0 | 7 | 27.39 | 60.01 | 2 | 0 | 8 |
| 5-10 | 53.22 | 60.03 | 8 | 0 | 2 | 51.27 | 60.03 | 7 | 0 | 3 |
| 6-6 | 6.40 | 34.95 | 0 | 0 | 10 | 4.34 | 11.17 | 0 | 0 | 10 |
| 6-10 | 60.01 | 60.02 | 10 | 0 | 0 | 59.93 | 60.02 | 9 | 0 | 1 |
| 10-6 | 60.03 | 60.05 | 10 | 0 | 0 | 60.02 | 60.06 | 10 | 0 | 0 |
| 10-10 | 60.03 | 60.05 | 10 | 0 | 0 | 60.04 | 60.07 | 10 | 0 | 0 |

- Time-limit hits: 42 of 210 runs at H = n, 39 of 210 at H = T + 2, all on sizes 5-8 and larger. All 10-6 and 10-10 runs, and all but one 6-10 run, end by the limit.
- Overshoot (F15): the largest wall clock is 60.07 s, so the check on entry of perform_compound_moves overshoots the limit by at most 0.07 s.
- Total wall clock 0.82 h (H = n) + 0.78 h (H = T + 2) = 1.60 h. At this rate the full run (40 instances per size) takes about 6.4 h sequentially.
- Machine dependence: the 81 runs that hit the limit depend on the machine's speed (one-machine rule); the others end by the bound or by exhausting the tree and are deterministic.
- Status: stopped before the full run (Thom).

### 6.1 Profiling before the full run (10 Nov 2026)

Question (Thom): the slowdown against the times of Table 3 (p. 307) is not constant, about 16× on 6-6 (0.4 s → 6.40 s) but over 160× on 5-8 and over 300× on 5-9, concentrated on bays with 8 or more stacks; a Python-versus-Java factor would be roughly constant. Runtime and work only, no relocation numbers.

Setup: `step3_profile.py`, H = n, 60 s, package as at 3210477. productive_moves, Bay.apply, Bay.lower_bound and Bay.clean_supply are timed, and the search classes are subclassed to count calls and to inspect Cm before the nSucc cut. The time for the statistics is taken off the search clock. The timers make the search about 1.5× slower (5-8 #1: 7.90 s against 5.39 s in the smoke run, with the same 10 576 nodes). Output: `step3_profile.txt`; F20 diagnostic: `step3_profile_f14tight.txt`.

| | 6-6 #1 | 5-8 #1 | 5-10 #1 |
|---|---|---|---|
| stacks | 6 | 8 | 10 |
| ended | tree exhausted | tree exhausted | 60 s limit |
| search clock (s) | 1.05 | 7.90 | 60.01 |
| calls of perform_compound_moves (nodes) | 1 266 | 10 576 | 21 965 (21 916 past the abort check) |
| calls of determine_compound_moves per node | 31.5 | 80.5 | 224.8 |
| nodes that cannot improve s* on arrival, n′_m(L) > nm(s*) − nm(s) − 1 | 888 | 10 533 | 21 905 |
| share of determine calls in those nodes | 3.6 % | 97.2 % | 99.5 % |
| \|Cm\| before the cut, mean / max | 21.9 / 2 277 | 17.3 / 2 304 | 89.5 / 6 720 |
| nodes with \|Cm\| = 0 | 1 126 | 9 505 | 19 718 |
| nodes with \|Cm\| > nSucc | 114 | 1 034 | 2 196 |
| compound-move length, mean / max | 7.82 / 26 | 12.01 / 24 | 18.29 / 26 |
| duplicate layouts in Cm (F21) | 31.4 % | 40.9 % | 64.8 % |
| duplicates among the kept nSucc | 278 | 4 439 | 5 406 |
| \|Pm\| per determine call: remove / BG / FLG_BB / GG (after cuts) | 0.26 / 0.58 / 1.07 / 0.66 | 0.89 / 0.35 / 0.06 / 0.04 | 0.89 / 0.55 / 0.00 / 0.00 |
| \|Pm\| at the first level of a node | 0.23 / 1.01 / 0.22 / 0.15 | 0.93 / 0.15 / 0.06 / 0.04 | 0.58 / 2.22 / 0.06 / 0.04 |
| largest cmStopValue | 625 | 1 584 | 1 820 |
| time per node (µs): move generation / apply / lower bound / clean supply / other | 293 / 116 / 264 / 19 / 134 | 288 / 131 / 110 / 21 / 198 | 1 074 / 461 / 404 / 132 / 667 |
| time per node (µs), total | 826 | 747 | 2 738 |
| last improvement of s* (search clock, s) | 0.26 | 1.82 | 0.18 |

- cmStopValue: Fig. 6 (p. 305) compares the value passed in with cmStopThreshold and passes cmStopValue · |Pm| to the recursive call, so the check comes before the multiplication by the current |Pm|. `search.py` `_determine` does the same. A compound move stops at the first layout whose incoming value is at least 150, which can be up to 149 · |Pm|, hence values up to 1 820. On Caserta a remove has |Pm| = 1 and does not raise the value, hence compound moves of up to 26 moves.
- Nodes that cannot improve s*: on 5-8 #1 and 5-10 #1 almost every node is one. These nodes are created by the pruning test of Fig. 6 as printed (F20). Pm there consists of removes and BG relocations, each lowering the bound by exactly one, so the printed test never cuts these compound moves. On 6-6 #1, FLG_BB and GG relocations are frequent; they do not lower the bound and are cut, so the dead nodes cost only 3.6 % of the work. This matches the concentration on wide bays.
- Diagnostic with the tight test (F20, `--f14-tight`, not the implementation): 6-6 #1 0.45 s, 413 nodes; 5-8 #1 0.13 s, 101 nodes; 5-10 #1 0.19 s, 111 nodes; all three trees searched to the end. Checked on the 339 smoke runs that ended before 60 s (`step3_f14_check.py`, no time limit): 339 of 339 plans identical; total wall clock 883.6 s printed against 93.5 s tight (944 677 against 117 966 nodes; longest run 60.14 s against 14.92 s).
- Verdict. (1) No deviation from Figs. 5–7 as printed: maxMoves (Fig. 5), the pruning test and the recursion with maxMoves − 1 (Fig. 6), the order of the cmStopValue check, and Cm as a list are all as printed. (2) Open interpretation: the pruning test, one move weaker than the purpose stated on p. 305; recorded as F20, not changed. Without a time limit it does not change the plan, but it causes the non-constant slowdown on bays with 8 or more stacks. (3) Constant-factor cost: 0.75–2.7 ms per node (instrumented), with move generation the largest share; with the tight test 1.1–1.8 ms per node. Duplicate compound moves (F21) are as printed and cost little once F20 is applied.
- Decision (Thom, 10 Nov 2026): F20 adopted as an implementation optimisation, F21 kept as printed (§2.3).
