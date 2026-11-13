# Forster & Bortfeldt (2011) multi-bay retrieval — validation log

Paper: F. Forster, A. Bortfeldt, "A tree search heuristic for the container retrieval problem", in: D. Klatte et al. (eds.), *Operations Research Proceedings 2011*, Springer 2012, pp. 257–262, DOI 10.1007/978-3-642-29210-1_41 (`papers/2011_Forster.pdf`; PDF page = printed page − 256). Referred to below as "the Proceedings paper". Companion source for the instances and the crane-time parameters: Y. Lee, Y.-J. Lee, "A heuristic for retrieving containers from a yard", *Computers & Operations Research* 37 (2010) 1139–1147 (`papers/2010_Lee.pdf`; PDF page = printed page − 1138), "Lee & Lee".

Code: `algorithms/CRP_Time/crp_time_u/forster_bortfeldt_2012/` — the validated single-bay C&OR tree search (`state.py`, `search.py`, `plan.py`, `layout.py`; log `docs/validation/forster_bortfeldt_2012.md`, decisions F1–F21), unchanged, plus the multi-bay part in the same package: `crane_time.py` (move-time evaluator of p. 258) and `multibay.py` (merge of Algorithm 1, block plan check). Instance reader: `core/benchmarks/lee_lee.py` (parse only; not wired into problems/, the web backend, the UI or the CLI). Instances: `benchmark/lee_lee/` (gitignored). Scripts and raw output: `docs/validation/data/forster_bortfeldt_2011_multibay/`.

Scope (Wei, 26 Oct 2026; task of 13 Nov 2026): the CRP-Time version of Forster–Bortfeldt follows the Proceedings paper — the C&OR search applied per bay, the per-bay solutions merged into one block solution (§3.2, Algorithm 1, p. 261), crane move time computed afterwards. Unrestricted version only. The crane-time model of p. 258 is a separate evaluator inside the package, used for this validation only; CRP-Time's existing RMGC model (`core/objectives.py`) is compared with it (§4.4) but not used.

## Summary

Status 13 Nov 2026: **step 1 (deterministic checks) done; no tree search has been run.** Open for Thom before any search run: the time limit (§7, options T1–T3), the validation criterion, and the proposals P6–P11 (§2).
- 1a (§3): 71 instance files read; R 10 test cases × 5, U 10 × 2, plus Example 1. Every file matches its name (bays, 16 stacks, maximum height, containers); every container in slot 2 or higher of a U instance is badly placed; one group per container in every file. C + n_b equals Lee & Lee's lower bound L on all 71 files. Inside the Proceedings paper's Table 1 the Lee & Lee CT column does not recompute from Lee & Lee's tables (CMT and M do).
- 1b (§4): see §4.
- 1c (§5): see §5.
- 1d (§6): see §6.
- 1e (§7): see §7.

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

Categories as in the validation workflow: (a) required by the unified 2D setting, (b) a choice the paper leaves open, (c) a real deviation from the paper. "Reading" = what the sources say, no choice involved; "proposed" = Claude's proposal, not yet decided by Thom.

| # | Point | Decision | Basis | Category | Status |
|---|---|---|---|---|---|
| P1 | File format of the Lee & Lee instances | Header `name B S H C X`; then one line per bay and stack, `b s k` and k pairs (id, group), bottom to top. X kept as `n_groups` (equals C and the number of distinct groups in every file). | Files; orientation fixed by Fig. 6 (p. 1144, Lee & Lee) and by L = C + n_b (Tables 1, 3) on all 71 files (§3) | — | reading |
| P2 | Test case ↔ files | Test case `R011606-0070` (Table 1, p. 262) = files `R011606_0070_001`…`_005`; `R020306_0020_001` is Lee & Lee's Example 1, not part of the sets. | p. 260; Lee & Lee Tables 1, 3; §3 | — | reading |
| P3 | Groups | One container per group (group = id in every file); retrieval order by increasing id. | Lee & Lee p. 1140; files | — | reading |

## 3. Step 1a — instances and reader (13 Nov 2026)

Source: `C:\Users\thomt\Downloads\Lee_instances.zip` (SHA-256 `9f8556f19d31f73bac7bcda50a36ae6e270a352020882895173fd99cfebdf9bd`), folders `Individual, random` and `Individual, upside down` extracted to `benchmark/lee_lee/` (71 .txt files; the `__MACOSX` metadata was skipped). Reader `core/benchmarks/lee_lee.py`; tests `core/benchmarks/test_lee_lee.py` (9 tests; the 5 dataset tests skip when `benchmark/lee_lee` is missing — checked by hiding the folder). Script `step1a_instances.py`, output `step1a_output.txt`; paper tables transcribed in `paper_tables.py`.

### 3.1 Checks
- **Names and headers:** for all 71 files, header name = file name and header (B, S, H, C) = the fields of the name; ids are 1…C. **Pass.**
- **Set sizes (p. 260):** R 10 test cases × 5 instances, U 10 × 2; every test case has 16 stacks per bay; the 20 test cases are exactly the rows of Table 1 (p. 262), and the 70 instances exactly the rows of Lee & Lee's Tables 1 and 3. **Pass.**
- **U property (p. 261):** in every U instance every container in slot 2 or higher is badly placed (e.g. U011606_0070_001: 55 of 55). In R the share is 30/54 to 387/560. **Pass.**
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
