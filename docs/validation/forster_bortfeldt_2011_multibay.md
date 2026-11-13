# Forster & Bortfeldt (2011) multi-bay retrieval — validation log

Paper: F. Forster, A. Bortfeldt, "A tree search heuristic for the container retrieval problem", in: D. Klatte et al. (eds.), *Operations Research Proceedings 2011*, Springer 2012, pp. 257–262, DOI 10.1007/978-3-642-29210-1_41 (`papers/2011_Forster.pdf`; PDF page = printed page − 256). Referred to below as "the Proceedings paper". Companion source for the instances and the crane-time parameters: Y. Lee, Y.-J. Lee, "A heuristic for retrieving containers from a yard", *Computers & Operations Research* 37 (2010) 1139–1147 (`papers/2010_Lee.pdf`; PDF page = printed page − 1138), "Lee & Lee".

Code: `algorithms/CRP_Time/crp_time_u/forster_bortfeldt_2012/` — the validated single-bay C&OR tree search (`state.py`, `search.py`, `plan.py`, `layout.py`; log `docs/validation/forster_bortfeldt_2012.md`, decisions F1–F21), unchanged, plus the multi-bay part in the same package: `crane_time.py` (move-time evaluator of p. 258) and `multibay.py` (merge of Algorithm 1, block plan check). Instance reader: `core/benchmarks/lee_lee.py` (parse only; not wired into problems/, the web backend, the UI or the CLI). Instances: `benchmark/lee_lee/` (gitignored). Scripts and raw output: `docs/validation/data/forster_bortfeldt_2011_multibay/`.

Scope (Wei, 26 Oct 2026; task of 13 Nov 2026): the CRP-Time version of Forster–Bortfeldt follows the Proceedings paper — the C&OR search applied per bay, the per-bay solutions merged into one block solution (§3.2, Algorithm 1, p. 261), crane move time computed afterwards. Unrestricted version only. The crane-time model of p. 258 is a separate evaluator inside the package, used for this validation only; CRP-Time's existing RMGC model (`core/objectives.py`) is compared with it (§4.4) but not used.

## Summary

Status 13 Nov 2026: **step 1 (deterministic checks) done.** Decided by Thom after step 1, before any search run: P5–P7, P10, P12 (groups ranked per bay), P13 (G = containers in the bay), P14 (T1, 60 s per bay), and the validation criterion (§8, pre-registered).
- 1a (§3): 71 instance files read; R 10 test cases × 5, U 10 × 2, plus Example 1. Every file matches its name (bays, 16 stacks, maximum height, containers); every container in slot 2 or higher of a U instance is badly placed; one group per container in every file. C + n_b equals Lee & Lee's lower bound L on all 71 files. Inside the Proceedings paper's Table 1 the Lee & Lee CT column does not recompute from Lee & Lee's tables (CMT and M do).
- 1b (§4): `crane_time.py` implements p. 258 with the footnote parameters; 6 hand-computed tests pass. **Lee & Lee's Example 1 (2346 s) is not reproduced:** 2243.2 s under the proposed readings, and none of 720 combinations of readings gives 2346 s (closest 2283.2 s). The one-bay times of both papers (79.3–93.9 s per move) exceed the 68.4 s that a truck inside the bay allows, so a remove crosses a bay in both papers (P6). Core's RMGC model differs in three points (max of the axes, t_ad per trip, start (1, 1)): 2855.3 s on Example 1.
- 1c (§5): Algorithm 1 implemented as printed (`multibay.py`); 9 tests on a hand-made 3-bay block and Example 1 pass: removes in group order, every bay keeps its own move order, block empty; the hand-timed merged plan (911.0 s) agrees.
- 1d (§6): the mean summed per-bay bound is ≤ M of Table 1 for both methods on all 20 test cases. Tree-search gaps 0.5–82.2 moves (0.40–9.07 %), Lee & Lee gaps 3.0–336.0 (2.22–37.09 %).
- 1e (§7): Table 1's CT stays at 8.2–11.9 s on R for 1 to 10 bays (1.19–10.10 s per bay); three options T1–T3 with upper bounds 5.13 h, 0.19 h and 1.17 h for all 70 instances. Not chosen.

## 1. Specification from the papers

### 1.1 Problem (Proceedings paper, Sec. 2, p. 258–259)
- Stacks in a row form a bay, adjacent bays a block; a gantry crane serves the block; the frame height limits the stack height (p. 258).
- A crane move picks up the topmost container of any stack in the block and either puts it on another stack below the maximum height or moves it "to a dedicated spot where a vehicle is waiting" (a remove) (p. 258).
- Each container belongs to a group g ∈ ℕ; a container can only be removed once all containers of lower groups have been removed (p. 258).
- Objective: a sequence of crane moves that removes every container, observes the group order and "minimizes the time necessary to execute the moves" (p. 259).
- Relocations between bays are allowed by the problem (p. 258: "any stack in the block … another stack"), but the method does not use them: each bay is solved on its own (§3.2, p. 260); taking them into account is named as future work (p. 262).

### 1.2 Crane move time (p. 258; footnote 1, p. 261)
- "The time necessary to execute a crane move is proportional to the number of bays b and stacks s the crane has to cross between his last position, the starting stack and the target stack, with time factors t_b and t_s respectively. The acceleration and deceleration time t_ad of the crane as well as the pickup and place-down time t_pp can be considered constants. Hence, the total time for a crane move is calculated as b·t_b + s·t_s + ad·t_ad + t_pp, with ad = 1 if b ≥ 1 and ad = 0 otherwise." (p. 258)
- Parameters, "given by Lee and Lee": t_b 3.5 s, t_s 1.2 s, t_ad 40 s, t_pp 30 s (footnote 1, p. 261).
- Left open by the Proceedings paper: the crane's initial position, where the vehicle for removed containers stands, and how b and s are counted over "last position, starting stack, target stack".
- Lee & Lee: gantry (bay direction), trolley (row direction) and spreader (tier direction) (p. 1141). After lifting, the crane "can lower the container onto a truck waiting at one end of the crane, or place the container on the top of another stack" (p. 1140). Stacks are numbered consecutively over the bays (bay 1: 1…S, bay 2: S+1…2S); stack 0 is the truck (p. 1141). The working time w(n, m) of movement m after movement n is "the time needed for the gantry to move from its final position upon completing movement n to the bay stack a belongs to, position the trolley at stack a, pick up the container, move the gantry to the bay of stack b, position its trolley at stack b, and put down its cargo"; w(0, m) starts "from its initial stowed position" (p. 1143). Gantry 3.5 s per bay, trolley 1.2 s per container width, acceleration and deceleration time loss of the gantry 40 s combined, pick up and place down 30 s combined (p. 1144–1145).
- Worked example: Lee & Lee Example 1 (Fig. 6, p. 1144): 2 bays × 3 stacks, H = 6, 20 containers, lower bound 27; the 27-move sequence on p. 1145 "can be completed in 2346 s by an RMGC" (p. 1146).

### 1.3 Per-bay search and merge (Sec. 3.2, p. 260–261; Algorithm 1, p. 261)
- "For each of the bays, we can apply the tree search algorithm for the container relocation problem in order to get a sequence of crane moves emptying the respective bay. … these solutions could be computed in parallel, as they are completely independent of each other." (p. 260) The tree search is the C&OR one ([5]; Sec. 3.1, p. 259–260), objective the number of moves.
- Merge (p. 260 and Algorithm 1, p. 261): start with an empty block solution; while the block is not empty: lowestGroupBlock ← the lowest group in the block; for all bays b: if b is not empty and its lowest group equals lowestGroupBlock, take moves from the front of b's solution, execute them and append them to the block solution, up to and including the first remove.
- Crane move times are calculated for the merged solution ("calculated crane move times based on the crane time parameters given by Lee and Lee", p. 261).

### 1.4 Benchmark (Sec. 4, p. 260–262)
- Lee & Lee's test cases: set R, 10 test cases × 5 instances; set U, 10 test cases × 2 instances; 16 stacks per bay; test cases differ in bays, maximum stack height and containers (p. 260). U: "all containers residing in the second slot or higher within a stack are initially badly placed"; R: groups randomly assigned (p. 261).
- Table 1 (p. 262): per test case, tree search and Lee & Lee: CMT (average crane move time, s), M (average number of moves), CT (average computation time, s), and the differences in %; a "Weighted avg." row. Text: mean moves −8.7 %, mean crane move time −15.7 %, 8.8 s per instance on average (p. 261).
- Hardware, language and the C&OR parameter setting are not restated. No time limit is stated (§7).

## 2. Decisions

Categories as in the validation workflow: (a) required by the unified 2D setting, (b) a choice the paper leaves open, (c) a real deviation from the paper. "Reading" = what the sources say, no choice involved; "proposed" = Claude's proposal, not yet decided by Thom; "decided" = decided by Thom on 13 Nov 2026, after step 1 and before any search run.

| # | Point | Decision | Basis | Category | Status |
|---|---|---|---|---|---|
| P1 | File format of the Lee & Lee instances | Header `name B S H C X`; then one line per bay and stack, `b s k` and k pairs (id, group), bottom to top. X kept as `n_groups` (equals C and the number of distinct groups in every file). | Files; orientation fixed by Fig. 6 (p. 1144, Lee & Lee) and by L = C + n_b (Tables 1, 3) on all 71 files (§3) | — | reading |
| P2 | Test case ↔ files | Test case `R011606-0070` (Table 1, p. 262) = files `R011606_0070_001`…`_005`; `R020306_0020_001` is Lee & Lee's Example 1, not part of the sets. | p. 260; Lee & Lee Tables 1, 3; §3 | — | reading |
| P3 | Groups | One container per group (group = id in every file); retrieval order by increasing id. | Lee & Lee p. 1140; files | — | reading |
| P4 | Acceleration term | ad = 1 once per move if b ≥ 1, also when both the empty and the loaded trip cross bays. | p. 258 (formula as printed); Lee & Lee describe two gantry trips per move (p. 1143) | — | reading (as printed) |
| P5 | Counting b and s | Positions (bay, row), bays 1…B, rows 1…S inside a bay. b = \|bay(last) − bay(src)\| + \|bay(src) − bay(dst)\|, s the same over rows; gantry and trolley terms added. | p. 258 ("between his last position, the starting stack and the target stack"); Lee & Lee p. 1143 (w(n, m): gantry from the final position of movement n to the bay of stack a, trolley to stack a, pick up, gantry to the bay of stack b, trolley to stack b, put down), p. 1141 (stack numbering), p. 1144–1145 (3.5 s per bay, 1.2 s per container width) | (b) | decided |
| P6 | Where removed containers go | A fixed truck position (bay 0, row 0): one bay before bay 1, beside row 1. Every remove crosses at least one bay. | p. 258 silent ("a dedicated spot"); Lee & Lee p. 1141 (stack 0); one-bay times of both papers (§4.2); for row 0: "a truck waiting at one end of the crane" (Lee & Lee p. 1140); equals core's truck position | (b) | decided |
| P7 | Initial crane position | The truck position (0, 0). | p. 258 silent; Lee & Lee p. 1143 ("initial stowed position", not located) | (b) | decided |
| P8 | Merge | Algorithm 1 as printed: lowestGroupBlock is fixed for one pass over the bays; the bays are visited in index order; every bay holding that group gives its moves up to and including its next remove (one remove per bay and pass). A bay solution that ends before that remove, or has moves left at the end, is an error. | Algorithm 1 and Sec. 3.2 (p. 260–261) | — | reading |
| P9 | Relocations between bays | None: each bay is solved on its own, so every relocation of the block solution stays in its bay. The crane still travels between bays (crane_time.py). | p. 260 ("completely independent"), p. 262 (inter-bay relocations as future work) | — | reading |
| P10 | Search per bay | The validated C&OR search unchanged (F1–F21), objective the number of moves, H of the test case, unrestricted; parameters of C&OR p. 306 (nSucc 10, cmStopThreshold 150, maxFlg_bb 3, maxGg 2). | p. 259–260 (refers to [5] for the algorithm; no parameters restated) | (b) for the parameters | decided |
| P11 | Order of the bay searches | One after another, in bay index order. Under P14 (each bay its own 60 s) the order changes only the wall clock, not the plans. | p. 260 (parallel possible), p. 262 (parallel named as an improvement) | — | no effect under P14 |
| P12 | Group labels inside a bay | Before the bay search, the groups of the bay are renumbered 1…n, order preserved (dense ranks; with one container per group, P3, rank = position in the sorted groups). The bay solution is mapped back to the block's groups for the merge (Algorithm 1 uses block groups). Not chosen: the block's groups as in the file. Removal order and badly placed items are the same under both; the C&OR keys that use group values differ (greedy F9/F10, GG key F11/F12, clean supply, and g_top = n(L) + 1 of the bound, F2); on the initial layouts the bound differs on 4 of 70 instances by 1 move (§6). | C&OR defines an instance with groups 1…G (p. 299, Sec. 4.1); the Proceedings paper applies that algorithm to each bay (p. 260); it is the form the code was validated on | (b) | decided |
| P13 | G inside a bay (F13) | G = the number of containers in the bay (follows from P12; equals the largest rank, since every group holds one container, P3). | C&OR Table 1, def. 2 (p. 302); P12 | (b) | decided |
| P14 | Time limit | T1: 60 s per bay, the C&OR setting (p. 306) applied to each bay as an independent CRP instance (Proceedings p. 260), wall clock including the greedy (F15). Alternative reading not chosen: a budget of about 10 s per instance shared by its bays (T2, from the CT column of Table 1, §7.1), because that value is inferred, not stated, and with Python being much slower than the paper's Java it would give far less search per bay than the paper had (§7.2: 365–1673 nodes per bay on 8 and 10 bays). | C&OR p. 306; Proceedings p. 260; §7 | (b) | decided |

## 3. Step 1a — instances and reader (13 Nov 2026)

Source: `C:\Users\thomt\Downloads\Lee_instances.zip` (SHA-256 `9f8556f19d31f73bac7bcda50a36ae6e270a352020882895173fd99cfebdf9bd`), folders `Individual, random` and `Individual, upside down` extracted to `benchmark/lee_lee/` (71 .txt files; the `__MACOSX` metadata was skipped). Reader `core/benchmarks/lee_lee.py`; tests `core/benchmarks/test_lee_lee.py` (9 tests; the 5 dataset tests skip when `benchmark/lee_lee` is missing — checked by hiding the folder). Script `step1a_instances.py`, output `step1a_output.txt`; paper tables transcribed in `paper_tables.py`.

### 3.1 Checks
- **Names and headers:** for all 71 files, header name = file name and header (B, S, H, C) = the fields of the name; ids are 1…C. **Pass.**
- **Set sizes (p. 260):** R 10 test cases × 5 instances, U 10 × 2; every test case has 16 stacks per bay; the 20 test cases are exactly the rows of Table 1 (p. 262), and the 70 instances exactly the rows of Lee & Lee's Tables 1 and 3. **Pass.**
- **U property (p. 261):** in every U instance every container in slot 2 or higher is badly placed (e.g. U011606_0070_001: 55 of 55); for R the counts per instance are in `step1a_output.txt`. **Pass.**
- **Lee & Lee's lower bound** (p. 1140: containers plus the minimum relocations per stack): C + n_b equals L of Tables 1 and 3 on all 70 instances, and 27 for Example 1 (p. 1145). **Pass.** This also confirms the bottom-to-top orientation (P1).
- **Example 1:** `R020306_0020_001` (2 bays × 3 stacks, H = 6, 20 containers) equals Fig. 6 stack by stack (stacks 1–3 bay 1, 4–6 bay 2, bottom to top: [10, 4, 18], [9, 17, 8, 20], [5, 19, 2], [16, 14, 7, 13, 3, 6], [1, 11], [15, 12]). **Yes, it is Example 1.**

### 3.2 Per test case

| test case | inst | B | S | H | C | max stack height | groups per instance | occupancy C/(B·S·H) |
|---|---|---|---|---|---|---|---|---|
| R011606-0070 | 5 | 1 | 16 | 6 | 70 | 6 | 70 | 0.729 |
| R011608-0090 | 5 | 1 | 16 | 8 | 90 | 8 | 90 | 0.703 |
| R021606-0140 | 5 | 2 | 16 | 6 | 140 | 6 | 140 | 0.729 |
| R021608-0190 | 5 | 2 | 16 | 8 | 190 | 8 | 190 | 0.742 |
| R041606-0280 | 5 | 4 | 16 | 6 | 280 | 6 | 280 | 0.729 |
| R041608-0380 | 5 | 4 | 16 | 8 | 380 | 8 | 380 | 0.742 |
| R061606-0430 | 5 | 6 | 16 | 6 | 430 | 6 | 430 | 0.747 |
| R061608-0570 | 5 | 6 | 16 | 8 | 570 | 8 | 570 | 0.742 |
| R081606-0570 | 5 | 8 | 16 | 6 | 570 | 6 | 570 | 0.742 |
| R101606-0720 | 5 | 10 | 16 | 6 | 720 | 6 | 720 | 0.750 |
| U011606-0070 | 2 | 1 | 16 | 6 | 70 | 6 | 70 | 0.729 |
| U011608-0090 | 2 | 1 | 16 | 8 | 90 | 8 | 90 | 0.703 |
| U021606-0140 | 2 | 2 | 16 | 6 | 140 | 6 | 140 | 0.729 |
| U021608-0190 | 2 | 2 | 16 | 8 | 190 | 8 | 190 | 0.742 |
| U041606-0280 | 2 | 4 | 16 | 6 | 280 | 6 | 280 | 0.729 |
| U041608-0380 | 2 | 4 | 16 | 8 | 380 | 8 | 380 | 0.742 |
| U061606-0430 | 2 | 6 | 16 | 6 | 430 | 6 | 430 | 0.747 |
| U061608-0570 | 2 | 6 | 16 | 8 | 570 | 8 | 570 | 0.742 |
| U081606-0570 | 2 | 8 | 16 | 6 | 570 | 6 | 570 | 0.742 |
| U101606-0720 | 2 | 10 | 16 | 6 | 720 | 6 | 720 | 0.750 |

- Every instance has as many groups as containers (P3); in every test case some stack reaches H. Per instance (n_b, empty stacks): `step1a_output.txt`.
- The occupancy matches Lee & Lee's "approximately 75% full" (p. 1145).

### 3.3 The transcribed tables
- The Lee & Lee columns CMT and M of Table 1 (p. 262) equal the means of Lee & Lee's phase-three T and M (Tables 1 and 3) on all 20 test cases (within 0.05). The tree-search columns and these two Lee & Lee columns of the "Weighted avg." row recompute with instance weights (5 per R test case, 2 per U test case): 55184.72, 579.24, 8.78, 65530.96, 634.21. The transcription is consistent.
- **Discrepancy inside Table 1, Lee & Lee CT:** only R011606-0070 (8204.3) equals the mean phase-three CPU time of Lee & Lee; the other 19 values do not equal the mean phase-three, phase-two or summed CPU time of their test case. Three of them equal the mean phase-three CPU of *another* test case (R041608-0380 ← R011608-0090, U011606-0070 ← R021606-0140, U041608-0380 ← R021608-0190). The column's weighted average is 15815.03, not the printed 19276.6. Not used anywhere in this validation.
- The printed CMT difference of the weighted averages, −15.7 % (also in the abstract and on p. 261), recomputes to −15.79 % from the printed averages.

## 4. Step 1b — crane-time evaluator (13 Nov 2026)

Code: `crane_time.py` (`move_time`, `plan_time`, `TimeParams`; P4–P7). Tests: `test_multibay.py`, class `TestMoveTime` (6 tests, hand-computed times: within a bay with and without an empty trip, across bays (ad = 1, also with both trips across bays and one ad), removes to the truck, a chained three-move plan, and a remove with an in-bay truck passed as a parameter). Scripts: `step1b_readings.py` / `step1b_readings_grid.py` (readings against Example 1), `step1b_evaluator.py` (evaluator, one-bay bound, RMGC comparison); outputs `step1b_*.txt`.

### 4.1 Worked example: Lee & Lee Example 1
- Lee & Lee has one worked example with a time: Example 1 (Fig. 6, p. 1144), its 27-move sequence (p. 1145) and "can be completed in 2346 s" (p. 1146). `R020306_0020_001` is this instance (§3.1). Replayed on the file, the sequence is feasible (only top containers, removes in group order, H = 6 respected, block empty at the end): 27 moves, 7 relocations of which 3 between bays, 20 removes.
- **Not reproduced.** Under the proposed readings (P4–P7) the sequence takes **2243.2 s (−102.8 s)**; the per-move breakdown is in `step1b_evaluator.txt`.
- None of 720 combinations of the open readings gives 2346 s within 0.5 s: truck in the source bay or at bay 0, 1, 2 or 3, truck row 0, 1, 3 or 4; start at the first source, at the truck, (1, 0), (1, 1), (0, 0) or (0, 1); ad per move or per gantry trip; axes added or simultaneous (max per trip); with or without a return to the start. The closest is 2283.2 s (−62.8 s; truck at bay 0, start in bay 1, ad per move, axes added, return to the start after the last move). Readings that keep the truck in the source bay give 1403.5–1537.7 s (`step1b_readings.txt`).
- The cause of the gap is unknown; no further readings were tried, so as not to fit the model to one number.
- Which sequence was timed (Thom, 13 Nov 2026). Lee & Lee print the sequence after "The estimated lower bound … is 27, which is reached in the second phase. The resulting sequence is …" (p. 1145) and give the time after the phase counts: "the movement reduction phase took 26 iterations, and the time reduction phase took 3392 iterations. … The retrieval work in this case can be completed in 2346 s" (p. 1146). Phase three cannot increase the number of moves (p. 1144) and reduces the working time (p. 1143–1144). So the printed 27-move sequence may be the phase-2 result and 2346 s the time after phase 3; then the correct model must give **at least** 2346 s for the printed sequence, not exactly 2346 s. Either way Example 1 does not identify the model. The evaluator is not tuned further.
- Under the "at least" reading (`readings.py`, output `readings_selection.txt`, 13 Nov 2026, before any search run): of the 732 readings of the two step-1b grids (720 + the 12 with start (1, S + 1)), 194 give at least 2346 s, all with t_ad per gantry trip and the truck outside the block (bay 0 or B + 1): 2794.3–3003.2 s. Every reading with one t_ad per move, as printed on p. 258 (P4), gives at most 2283.2 s, the default model 2243.2 s. So under that reading the printed formula would be too cheap for the printed sequence; under the exact reading no grid reading matches. These 194 readings (`readings_ge2346.json`) give the descriptive CMT range of §8.

### 4.2 One-bay instances: removes cross a bay in both papers
- If a remove needs no gantry travel (truck at row 0 or 17 of the same 16-stack bay), every move of a one-bay plan takes at most t_pp + t_s·(16 + 16) = 68.4 s.
- Reported times per move on one-bay test cases are all higher: Lee & Lee T/M 87.3–93.9 s (14 instances, Tables 1 and 3); Table 1 of the Proceedings paper, CMT/M of the tree search 88.28 (R011606), 84.24 (R011608), 80.62 (U011606), 79.31 s (U011608), and of Lee & Lee 88.52–91.27 s.
- So in both papers a remove crosses at least one bay (or the truck lane lies at least 4.55 container widths beyond the outer stack, 7.88 for Lee & Lee's own T/M). This is the basis of P6. Under P6 every move that starts or ends at the truck pays the 40 s acceleration.

### 4.3 Hand-computed unit tests
| move (last → src → dst) | b | s | ad | time (s) |
|---|---|---|---|---|
| (1,3) → (1,3) → (1,7) | 0 | 4 | 0 | 34.8 |
| (2,1) → (2,5) → (2,2) | 0 | 7 | 0 | 38.4 |
| (1,2) → (1,2) → (3,5) | 2 | 3 | 1 | 80.6 |
| (1,1) → (2,4) → (2,6) | 1 | 5 | 1 | 79.5 |
| (1,1) → (3,1) → (2,1) | 3 | 0 | 1 | 80.5 |
| remove: (2,3) → (2,3) → truck (0,0) | 2 | 3 | 1 | 80.6 |
| remove: truck (0,0) → (1,1) → truck | 2 | 2 | 1 | 79.4 |
| plan from (0,0): (1,2)→(1,4); remove (1,2); remove (1,4) | 1, 1, 2 | 4, 4, 8 | 1, 1, 1 | 78.3 + 78.3 + 86.6 = 243.2 |

### 4.4 Comparison with CRP-Time's RMGC model (`core/objectives.py`)
Not used for the validation. `KinematicsModel` / `compute_crane_time` (core/objectives.py:42–131, 469–490) against p. 258 with P4–P7:

| | p. 258 (`crane_time.py`) | core RMGC |
|---|---|---|
| gantry and trolley | added: b·t_b + s·t_s over both trips | simultaneous: max(gantry, trolley) per trip |
| acceleration t_ad | once per move if b ≥ 1 | once per trip with a bay change, so up to twice per move |
| pick-up and place-down | t_pp once per move | spreader_s once per move (same) |
| parameters | 3.5 / 1.2 / 40 / 30 | 3.5 / 1.2 / 40 / 30 (same) |
| truck | (0, 0) (P6, proposed) | `_TRUCK_POS` = (0, 0) |
| initial position | the truck (P7, proposed) | (1, 1) by default |
| positions on Caserta | — | `apply_caserta_file_to_yard` puts stack k at (bay k, row 1) (core/benchmarks/caserta.py), so on a single Caserta bay every stack change counts as a gantry move |

On Example 1: core 2855.3 s (initial (1, 1)) and 2858.8 s (initial (0, 0)); `crane_time.py` 2243.2 s (start (0, 0)) and 2238.5 s (start (1, 1)). One difference at a time, start and truck (0, 0): p. 258 2243.2 s; t_ad per gantry trip instead of per move 2963.2 s; max of the axes per trip with t_ad per trip 2858.8 s (= core). Neither model gives 2346 s.

## 5. Step 1c — merge (Algorithm 1) (13 Nov 2026)

Code: `multibay.py` — `merge` (Algorithm 1, P8), `check_block_plan` (independent replay on plain lists: only top items move, a remove takes an item of the block's lowest group, no stack above H, block empty at the end; moves between bays allowed so that Lee & Lee's Example 1 can be checked), `to_positions` (block moves → (bay, row) from 1 for `crane_time.py`). It imports only `state.Move` and `crane_time.Position` from the package. Tests: `test_multibay.py`, class `TestMerge` (9 tests).

Hand-made block: 3 bays × 2 stacks, H = 3, groups 1–8; bay 0 = [3, 6], [1]; bay 1 = [2, 5], [4]; bay 2 = [7], [8] (bottom to top). Given bay solutions (each accepted by the validated `plan.check_plan`): bay 0 remove 1, 6 → stack 1, remove 3, remove 6; bay 1 5 → stack 1, remove 2, 5 → stack 0, remove 4, remove 5; bay 2 remove 7, remove 8.
- The merged sequence equals the one written out by hand from Algorithm 1 (11 moves: bay 0 for group 1; bay 1 relocation + remove for group 2; bay 0 relocation + remove for group 3; bay 1 relocation + remove for group 4; then 5, 6, 7, 8). **Pass.**
- Removes come in group order 1…8, `check_block_plan` accepts the sequence (3 relocations) and the block is empty. **Pass.**
- The subsequence of every bay equals that bay's solution, and no relocation leaves its bay (P9). **Pass.**
- Algorithm 1 with repeated groups: with group 1 in two bays, both bays give one remove in the same pass, in bay order; a bay holding two items of group 1 gives one remove per pass. **Pass** (as printed; does not occur in the Lee & Lee instances, P3).
- Errors: a bay solution that ends before its remove, a non-top move, moves left after the block is empty, and a missing bay solution raise. `check_block_plan` rejects a remove out of block group order, a full stack, a non-empty end and a non-top move. **Pass.**
- Lee & Lee's Example 1 sequence (with its 3 relocations between bays) passes `check_block_plan` on Fig. 6 with 7 relocations; its time under P4–P7 is 2243.2 s (§4.1, regression value).
- Merged plan timed by hand (positions from 1, start and truck (0, 0)): 81.8 + 79.4 + 79.4 + 75.9 + 75.9 + 80.6 + 80.6 + 86.4 + 81.8 + 93.4 + 95.8 = 911.0 s; `plan_time(to_positions(merge(...)))` gives the same. **Pass.**

Other checks (`step1c_tests_registry.txt`): the 20 validated tests of `test_forster_bortfeldt.py`, the 15 of `test_multibay.py` and the 9 of `core/benchmarks/test_lee_lee.py` pass (44); the registry still loads 74 algorithms and no module under `algorithms/` fails to import; `state.py`, `search.py`, `plan.py` and `layout.py` are unchanged (`git diff` against 8a5252a is empty).

## 6. Step 1d — lower bound per test case (13 Nov 2026)

Script `step1d_lower_bound.py`, output `step1d_output.txt`. Per instance the bound on moves is the sum over the bays of the validated `layout.lower_bound` (n′_REM + n′_BG + n′_non-BG, C&OR p. 302) of the initial bay layout; per test case the mean over its instances. Computed under both group labellings of P12; n′_REM + n′_BG is the same under both and equals Lee & Lee's L on every instance. Gap = M (Table 1, p. 262) − mean bound, with the file labelling.

| test case | bays | mean bound (file) | mean bound (ranks) | M tree search | gap | gap % | M Lee & Lee | gap | gap % | bays with n′_non-BG = 1 (file / ranks) |
|---|---|---|---|---|---|---|---|---|---|---|
| R011606-0070 | 1 | 104.4 | 104.4 | 108.0 | 3.6 | 3.45 | 125.4 | 21.0 | 20.11 | 0 / 0 |
| R011608-0090 | 1 | 142.0 | 142.0 | 148.4 | 6.4 | 4.51 | 191.4 | 49.4 | 34.79 | 0 / 0 |
| R021606-0140 | 2 | 209.0 | 209.0 | 215.6 | 6.6 | 3.16 | 230.2 | 21.2 | 10.14 | 0 / 0 |
| R021608-0190 | 2 | 305.8 | 305.8 | 324.6 | 18.8 | 6.15 | 367.8 | 62.0 | 20.27 | 0 / 0 |
| R041606-0280 | 4 | 426.8 | 426.8 | 439.8 | 13.0 | 3.05 | 454.2 | 27.4 | 6.42 | 0 / 0 |
| R041608-0380 | 4 | 610.6 | 610.6 | 657.8 | 47.2 | 7.73 | 769.2 | 158.6 | 25.97 | 0 / 0 |
| R061606-0430 | 6 | 655.6 | 655.6 | 684.2 | 28.6 | 4.36 | 709.8 | 54.2 | 8.27 | 0 / 0 |
| R061608-0570 | 6 | 906.0 | 906.2 | 988.2 | 82.2 | 9.07 | 1242.0 | 336.0 | 37.09 | 0 / 1 |
| R081606-0570 | 8 | 875.6 | 875.8 | 910.2 | 34.6 | 3.95 | 945.4 | 69.8 | 7.97 | 0 / 1 |
| R101606-0720 | 10 | 1092.0 | 1092.0 | 1134.0 | 42.0 | 3.85 | 1169.2 | 77.2 | 7.07 | 0 / 0 |
| U011606-0070 | 1 | 124.5 | 124.5 | 125.0 | 0.5 | 0.40 | 127.5 | 3.0 | 2.41 | 0 / 0 |
| U011608-0090 | 1 | 164.0 | 164.0 | 167.0 | 3.0 | 1.83 | 177.5 | 13.5 | 8.23 | 0 / 0 |
| U021606-0140 | 2 | 248.0 | 248.5 | 251.5 | 3.5 | 1.41 | 257.0 | 9.0 | 3.63 | 0 / 1 |
| U021608-0190 | 2 | 348.0 | 348.0 | 353.5 | 5.5 | 1.58 | 400.5 | 52.5 | 15.09 | 0 / 0 |
| U041606-0280 | 4 | 496.5 | 496.5 | 501.0 | 4.5 | 0.91 | 507.5 | 11.0 | 2.22 | 0 / 0 |
| U041608-0380 | 4 | 697.0 | 697.0 | 706.0 | 9.0 | 1.29 | 800.0 | 103.0 | 14.78 | 2 / 2 |
| U061606-0430 | 6 | 764.5 | 764.5 | 772.5 | 8.0 | 1.05 | 793.0 | 28.5 | 3.73 | 1 / 1 |
| U061608-0570 | 6 | 1047.5 | 1048.0 | 1059.5 | 12.0 | 1.15 | 1234.0 | 186.5 | 17.80 | 7 / 8 |
| U081606-0570 | 8 | 1014.0 | 1014.0 | 1019.5 | 5.5 | 0.54 | 1044.0 | 30.0 | 2.96 | 1 / 1 |
| U101606-0720 | 10 | 1282.0 | 1282.0 | 1291.0 | 9.0 | 0.70 | 1345.0 | 63.0 | 4.91 | 2 / 2 |

- **Pass:** on all 20 test cases the mean bound is ≤ M of the tree search and ≤ M of Lee & Lee, under both labellings. The smallest gap is U011606-0070 (0.5 moves: the tree search is on average half a move above the bound).
- The labellings differ on 4 instances (R061608_0570_001, R081606_0570_001, U021606_0140_001, U061608_0570_001), each by one n′_non-BG term; ranks give the higher bound there. With the file's groups, g_top = n(L) + 1 of a bay can lie below the bay's groups, which weakens the non-BG term but never invalidates it (F2).
- Lee & Lee's sequences may relocate between bays, so the per-bay non-BG terms are not a proven bound for their M; n′_REM + n′_BG = L is. Lee & Lee's M is above the larger bound anyway.

## 7. Step 1e — time limit: evidence and options (13 Nov 2026)

Script `step1e_time_limit.py`, output `step1e_output.txt`. The Proceedings paper states no time limit, no hardware and no parameter setting.

### 7.1 Evidence
- **Table 1 CT against bays and containers** (p. 262; CT per bay and per container computed):

| test case | B | C | CT (s) | CT/B (s) | test case | B | C | CT (s) | CT/B (s) |
|---|---|---|---|---|---|---|---|---|---|
| R011606-0070 | 1 | 70 | 8.2 | 8.20 | U011606-0070 | 1 | 70 | 1.2 | 1.20 |
| R011608-0090 | 1 | 90 | 10.1 | 10.10 | U011608-0090 | 1 | 90 | 10.0 | 10.00 |
| R021606-0140 | 2 | 140 | 9.1 | 4.55 | U021606-0140 | 2 | 140 | 2.9 | 1.45 |
| R021608-0190 | 2 | 190 | 10.1 | 5.05 | U021608-0190 | 2 | 190 | 4.5 | 2.25 |
| R041606-0280 | 4 | 280 | 9.5 | 2.38 | U041606-0280 | 4 | 280 | 6.1 | 1.52 |
| R041608-0380 | 4 | 380 | 10.8 | 2.70 | U041608-0380 | 4 | 380 | 5.3 | 1.32 |
| R061606-0430 | 6 | 430 | 10.2 | 1.70 | U061606-0430 | 6 | 430 | 5.6 | 0.93 |
| R061608-0570 | 6 | 570 | 11.9 | 1.98 | U061608-0570 | 6 | 570 | 4.6 | 0.77 |
| R081606-0570 | 8 | 570 | 10.7 | 1.34 | U081606-0570 | 8 | 570 | 3.8 | 0.47 |
| R101606-0720 | 10 | 720 | 11.9 | 1.19 | U101606-0720 | 10 | 720 | 6.9 | 0.69 |

  - R: CT 8.2–11.9 s for 1 to 10 bays, while every bay holds 70–95 containers (C/B), so the per-bay problem has about the same size in every test case. CT per bay falls from 8.2–10.1 s (one bay) to 1.19 s (ten bays). Pearson r(CT, B) = 0.75 over the 10 R test cases, but the totals stay close: one bay 8.2 and 10.1 s, two to ten bays 9.1–11.9 s.
  - U: CT 1.2–10.0 s with no relation to the number of bays (r = 0.13). U011606 (1.2 s) is the test case where the tree search ends 0.5 moves above the bound (§6), consistent with searches that stop at the bound (Fig. 5 of C&OR).
  - Instance-weighted mean CT 8.78 s (printed 8.8 s, p. 261).
  - Reading of these numbers (not stated in the paper): a fixed limit per bay would make CT grow roughly in proportion to B on R; the observed totals near 10 s for every B fit a budget of about 10 s per instance shared by its bays, with the excess above 10 s (up to 1.9 s) from overruns and overhead. The paper gives neither the value nor how it is shared.
- **C&OR setting** (C&OR 39, p. 306): 60 s per single-bay instance; Java, Core 2 Duo P7350 2 GHz. The Proceedings paper refers to [5] for the algorithm but restates none of its settings.
- **Remark on parallel bays:** the bay solutions "could be computed in parallel, as they are completely independent of each other" (p. 260); as future work, "calculating the single bay results in parallel, giving each individual tree search instance more time to explore the solution space" (p. 261–262). This implies that in the reported runs the bays were computed one after another and that each bay search got less time than it would have in a parallel run, i.e. a time budget shared by the bays of an instance.
- **Our Python C&OR implementation** (log forster_bortfeldt_2012.md §6–7; rows of `step3_full_summary.txt`): at 60 s per instance, all 40 instances of 10-6 and 10-10 and 15 (H = n) or 17 (H = T + 2) of 40 of 6-10 end by the limit; 0.75–2.7 ms per node. A Lee & Lee bay (16 stacks, 70–95 containers) is wider than any Caserta size, so most R bays are expected to end by the limit under every option (expectation, not measured).

### 7.2 Options (not chosen)
70 instances, 308 bay searches. Upper bounds of the total search time, sequential, if every bay runs to its limit:

| option | limit | per bay (B = 1, 2, 4, 6, 8, 10) | per instance ≤ | total ≤ |
|---|---|---|---|---|
| T1 | 60 s per bay (the C&OR value, applied per bay) | 60 s | 600 s | 18 480 s = 5.13 h |
| T2 | 10 s per instance, split equally over its bays | 10, 5, 2.5, 1.67, 1.25, 1 s | 10 s | 700 s = 0.19 h |
| T3 | 60 s per instance (the C&OR value), split equally over its bays | 60, 30, 15, 10, 7.5, 6 s | 60 s | 4 200 s = 1.17 h |

- T1 follows the C&OR setting literally per bay; its totals (up to 600 s per instance) are far above Table 1's CT.
- T2 matches the order of Table 1's CT (§7.1), but 10 s is inferred from Table 1, not stated; in Python it leaves 1–1.25 s per bay on 8 and 10 bays, i.e. 457–1673 nodes per bay on 8 bays and 365–1339 on 10 bays at 0.747–2.738 ms per node (C&OR log §6.1), including the greedy (F15).
- T3 keeps the C&OR value as a per-instance budget, consistent with a shared budget (p. 261–262), but with 60 s per instance where Table 1 reports 8.2–11.9 s on R.
- Sub-questions for T2/T3: equal split, or unused time of a bay that ends early passed on to the next bay (P11); the limit includes the greedy (F15).
- Expected actual runtime: close to the upper bound on R (wide bays, see above); lower on U where searches may end at the bound. Not measured: no search was run in step 1.

### 7.3 Decision (Thom, 13 Nov 2026)
**T1: 60 s per bay** (P14), each bay searched as an independent CRP instance with the C&OR setting, one bay after another (P11). The per-instance budget suggested by the CT column (§7.1) is recorded as an alternative reading that was not chosen: it is inferred, not stated, and with Python much slower than the paper's Java it would give far less search per bay than the paper had.

## 8. Validation criterion (pre-registered, 13 Nov 2026, before any search run)

Set by Thom after step 1. Runs: all 70 instances of R and U, P5–P7, P10, P12–P14 (T1). Per instance: M = number of moves of the merged block plan; CMT = `crane_time.plan_time` of that plan under P4–P7 (the default model). Per test case: means over its instances. Every block plan must pass `check_block_plan` (only top containers moved, removes in group order, height ≤ H, block empty) and keep its relocations inside their bay; a failing plan is reported as such.

- **(A)** The weighted mean M over all 70 instances (R and U, per instance as in Table 1, i.e. the instance-weighted average of §3.3) is within 2 % of 579.2 (Table 1, "Weighted avg.", tree search).
- **(B)** In each of the 20 test cases, our mean M and our mean CMT (under P5–P7) are both below Lee & Lee's values in Table 1 (p. 262).
- **Descriptive only:** CMT against the Forster–Bortfeldt column of Table 1, under the default model, plus the min–max over the 194 alternative readings of `readings_ge2346.json` (§4.1).
- **Report per test case:** mean M, mean CMT, the differences with both columns of Table 1 (tree search and Lee & Lee), the number of bays that hit the 60 s limit, and the total runtime.

## 9. Step 2 — the multi-bay algorithm on small blocks (13 Nov 2026)

Code (0fb3815): `multibay.solve_bay` (one bay as an independent CRP instance: groups ranked 1…n (P12), G = containers in the bay (P13), the validated `TreeSearch` with the parameters of C&OR p. 306 and its own 60 s limit (P10, P14); its moves mapped back to block groups) and `multibay.solve_block` (the bays one after another (P11), then `merge`). Tests: `test_multibay.py`, class `TestSolveBlock` (6 tests, without a time limit, so deterministic). Script `step2_blocks.py` at T1, output `step2_output.txt`. All 50 tests of the package and the reader pass; the registry still loads 74 algorithms, no import failures.

Checked on every block: the block plan passes `check_block_plan` (only top containers moved, removes in group order, height ≤ H, block empty), every relocation stays in its bay (P9), M ≥ the summed per-bay lower bound (ranked groups), and the moves of each bay equal those of a standalone single-bay run of `TreeSearch` on that bay (same layout, H, G and parameters). **Pass** on all three blocks.

| block | bays, H, containers | M | relocations | summed bound | per bay: moves / bound / exact in-bay optimum / how the search ended |
|---|---|---|---|---|---|
| Example 1 (R020306_0020_001) | 2, 6, 20 | 30 | 10 | 27 | bay 1: 17 / 14 / 17 / tree exhausted (greedy 17); bay 2: 13 / 13 / 13 / bound reached |
| hand-made A | 2, 4, 12 | 17 | 5 | 17 | 8 / 8 / 8 / bound; 9 / 9 / 9 / bound (greedy 10) |
| hand-made B (one empty stack) | 2, 3, 6 | 9 | 3 | 9 | 6 / 6 / 6 / bound; 3 / 3 / 3 / bound |

- **Example 1: M = 30 against Lee & Lee's 27** (p. 1145, equal to their lower bound). Lee & Lee relocate between bays (3 of their 7 relocations), Forster & Bortfeldt do not (P9). The exact in-bay optimum of each bay (IDA* of `test_forster_bortfeldt.exact_moves`, independent of the search code) is 17 and 13, so 30 is the minimum for any method that relocates only inside a bay; the search finds it in both bays.
- The exact in-bay optimum comes from a diagnostic with the IDA* of the validated test file; it is not part of the method.

## 10. Step 3 — smoke run, runtime only (13 Nov 2026)

`solve_block` at T1 (60 s per bay, wall clock including the greedy) on R011606_0070_001, R101606_0720_001, R011608_0090_001 and R061608_0570_001, sequential, CRISP 1abe282 (algorithm code as at 0fb3815), rl Python 3.11.16. Two idle `python3.13` processes (VS Code) were present; nothing else ran. Script `step3_smoke.py`, data `step3_smoke.jsonl` / `.log`, summary `step3_summary.py` → `step3_smoke_summary.txt`. M and crane time were not recorded (Thom); the only result check is plan validity: all 4 block plans pass `check_block_plan` with every relocation inside its bay.

| instance | bays | bays at the limit | per bay (s; L = limit, E = tree exhausted) | instance wall clock (s) |
|---|---|---|---|---|
| R011606_0070_001 | 1 | 1 | 60.01 L | 60.01 |
| R101606_0720_001 | 10 | 2 | 2.31 E, 1.97 E, 60.02 L, 12.15 E, 0.33 E, 35.76 E, 60.06 L, 2.02 E, 7.18 E, 2.09 E | 183.91 |
| R011608_0090_001 | 1 | 1 | 60.00 L | 60.00 |
| R061608_0570_001 | 6 | 6 | 60.06, 60.06, 60.06, 60.01, 60.01, 60.01, all L | 360.23 |

- 10 of 18 bays hit the limit; no bay reached its lower bound. Largest bay wall clock 60.06 s (overrun ≤ 0.06 s, F15). Smoke run total 664.1 s.
- H = 6 bays: mean 22.17 s over 11 bays (3 at the limit); H = 8 bays: 60.03 s over 7 bays (all at the limit).
- **Projected total of the full run** (70 instances, 308 bays: R H = 6 155, R H = 8 65, U H = 6 62, U H = 8 26): mean per bay by H × bays with that H = **10 274 s ≈ 2.85 h**; upper bound 308 × 60.06 s = 18 499 s ≈ 5.14 h. The sample has no U instance; U bays may end earlier at the bound (§7.1), so the projection is likely on the high side for U.
