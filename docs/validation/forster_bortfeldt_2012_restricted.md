# Forster & Bortfeldt (2012), restricted version — validation log

Paper: F. Forster, A. Bortfeldt, "A tree search procedure for the container relocation problem", *Computers & Operations Research* 39 (2012) 299–309 (`papers/2012_Forster.pdf`; printed page − 298 = PDF page), "C&OR" below. The method is unrestricted in the paper (p. 302); this log derives the restricted version.

Code: `algorithms/CRP_Time/crp_time_r/forster_bortfeldt_2012/`, derived from the validated unrestricted package `algorithms/CRP_Time/crp_time_u/forster_bortfeldt_2012/` (single bay: `docs/validation/forster_bortfeldt_2012.md`, decisions F1–F21; multi-bay: `docs/validation/forster_bortfeldt_2011_multibay.md`, decisions P1–P14; CRISP 0186485). Scripts and raw output: `docs/validation/data/forster_bortfeldt_2012_restricted/`.

Scope (supervisor's decision, task of 13 Nov 2026): two separate folders, each with its own search code; the restricted version is derived from the validated unrestricted one. The restriction is the one of CRISP's `CRP_R.step` (§2), not a reading of another paper. The unrestricted package stays byte-identical (§6).

## Summary

Status 13 Nov 2026: step 1a (copy) and 1b (what `CRP_R.step` allows) done.

## 1. Step 1a — copy (13 Nov 2026)

- fc1de7a: the tracked files of `crp_time_u/forster_bortfeldt_2012/` copied to `crp_time_r/forster_bortfeldt_2012/` without content changes, plus `crp_time_r/__init__.py` (`# Restricted CRP-Time algorithms.`, as `crp_time_u/__init__.py`), needed for unittest discovery and the registry's package walk. Git tree hash of both package folders at fc1de7a, and of `crp_time_u/forster_bortfeldt_2012` at 0186485: `5b4cf6b3c9a5f743819e5a41b27778d663c02417` (`git rev-parse <commit>:<path>`); the 9 blob hashes are the same file by file.
- ab08b04: path lines only — the usage line of `test_forster_bortfeldt.py` and of `test_multibay.py` names `crp_time_r` instead of `crp_time_u`. The package imports only relatively (`from .state import …`) and from `core`, so no import line needed a change. The 41 tests of the two copied test modules pass at ab08b04 (unchanged behaviour).

## 2. Step 1b — what `CRP_R.step` allows (13 Nov 2026)

Read from the code at ab08b04 (`problems/CRP_R.py`, `core/yard.py`, `core/benchmarks/caserta.py`); observed with `step1b_crp_r_probe.py` → `step1b_crp_r_probe.txt`, which calls `CRP_R.step` itself on Caserta layouts. CRP-Time inherits these rules unchanged: `CRP_Time.step` (`problems/CRP_Time.py:116-126`) calls `super().step(action)` and only adds metrics and rewards.

**Which container moves.** The action is a destination only (`action_space = Discrete(num_bays · num_rows)`, `CRP_R.py:73`); the source is not chosen by the algorithm.
- The target is the container with the current priority (`_get_target_container`, `CRP_R.py:389-393`: the first container whose priority equals `_current_target_priority`). The source is the stack that holds it (`CRP_R.py:213-218`, `227`).
- If the target is on top of its stack, there is nothing to relocate: it is retrieved and no relocation happens (`CRP_R.py:219-224`).
- Otherwise the top container of the target's stack is relocated (`blocker = src_stack.top`; `self.yard.relocate(src, dst)`, `CRP_R.py:234-235`; `Yard.relocate` moves the top of `src` onto `dst`, `core/yard.py:211-233`). This container is above the target, so it always blocks the target.
- Consequently no container of any other stack can be relocated, and the target itself is never relocated. `CRP_R.validate_plan` enforces the same on explicit plans: a relocation must come from the current target's stack (`CRP_R.py:496-501`) and must not move the target (`CRP_R.py:502-506`).

**To which stacks.**
- `dst = _action_to_stack(action)` (`CRP_R.py:210`): flat index → (bay, row) with `bay = a // num_rows + 1`, `row = a % num_rows + 1`, after `a % (num_bays · num_rows)` (`CRP_R.py:395-400`). Every stack of the yard is addressable (`Yard.stacks` holds all (b, r), b = 1…num_bays, r = 1…num_rows, `core/yard.py:180-183`).
- The relocation is executed unless the destination does not exist, is full (`len ≥ max_tiers`, `core/yard.py:48-49`) or equals the source (`CRP_R.py:231-232`); in those cases the step returns reward −0.5, moves nothing and still counts a step (`CRP_R.py:207`).
- No other condition: an empty destination is allowed, and the destination's contents (well or badly placed) do not matter.

**Within or across bays.** The destination may be any non-full stack of the yard other than the source, in the same bay or another bay; `step` has no bay condition. On the Caserta instances the platform maps stack i of the file to bay i + 1, row 1 (`core/benchmarks/caserta.py:198-203`; `num_bays = S`, `num_rows = 1`, `caserta.py:227-228`), so in CRISP coordinates every Caserta stack is its own bay and every relocation goes to another "bay".

**Retrievals.** Never an action: after `reset()` and after every executed or rejected relocation, all consecutive targets that are on top are retrieved (`_advance_auto_retrievals`, `CRP_R.py:174-194`, called at `CRP_R.py:113` and `241`). `reset(options={"skip_auto_retrieve": True})` keeps the opening retrievals out (`CRP_R.py:102-106`).

**Height.** `max_tiers` of the config; for Caserta `implied_tier_capacity` = n / S + 2 = T + 2 when the layout is rectangular (`caserta.py:154-161`, `229`).

Observed (`step1b_crp_r_probe.txt`, data3-3-1, file stacks bottom to top [3, 7, 1], [2, 6, 5], [8, 9, 4], max_tiers 5):
- reset retrieves container 1; the target is 2 in stack (2, 1), with 5 on top;
- `step(0)` and `step(2)` relocate 5 to (1, 1) and (3, 1) (reward −1.0, one relocation);
- `step(1)` (the source) moves nothing (reward −0.5, relocations 0, steps 1).
- With max_tiers = 3, `step(2)` onto the full stack (3, 1) moves nothing (reward −0.5).
- On data3-5-1, `reset()` retrieves container 1, which is on top at the start; `reset(skip_auto_retrieve)` does not.

Platform details seen on the way (not part of the restriction; for Wei, `docs/notes_for_wei.md`):
- `step` accepts action indices outside [0, num_bays · num_rows) through the modulo: `step(3)` and `step(-1)` on a 3-stack layout relocate to (1, 1) and (3, 1). `validate_actions` rejects them (`CRP_R.py:291-295`).
- The action mask (`_build_action_mask`, `CRP_R.py:414-422`) marks the source stack as available although `step` rejects it.
