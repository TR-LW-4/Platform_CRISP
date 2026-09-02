"""
Policy-tree builders for RIRH and Expected MinMax.

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
corresponding original algorithm paper.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import copy
from itertools import permutations
from typing import Dict, List, Optional, Sequence, Tuple

from core.plan import Movement
from core.stoch import ChanceNode, DecisionNode


# ────────────────────────────────────────────────────────────────────
#  Shared primitives
# ────────────────────────────────────────────────────────────────────


Stacks = Dict[Tuple[int, int], List[int]]
StackKey = Tuple[int, int]


def _min_priority_of(stack: Sequence[int], n_containers: int) -> int:
    """σ_c: smallest priority in the stack (or N+1 for empty stack)."""
    return min(stack) if stack else n_containers + 1


def _clone(stacks: Stacks) -> Stacks:
    return {k: list(v) for k, v in stacks.items()}


def _locate(stacks: Stacks, priority: int) -> Optional[StackKey]:
    for key, prios in stacks.items():
        if priority in prios:
            return key
    return None


# ────────────────────────────────────────────────────────────────────
#  MinMax pick (Galle et al. 2018 / Bacci 2022 Section 2)
# ────────────────────────────────────────────────────────────────────


def minmax_pick_stack(
    stacks: Stacks,
    max_tiers: int,
    src_key: StackKey,
    blocker_priority: int,
    n_containers: int,
    candidate_keys: Optional[Sequence[StackKey]] = None,
) -> Optional[StackKey]:
    """
    Pick a destination stack for ``blocker_priority`` using MinMax.

    Selection rule:
      1. Among candidate stacks (excluding ``src_key`` and full stacks),
         choose the one with the SMALLEST ``σ_c > blocker_priority``.
         Rationale: relocated container will *not* create a new blocker
         there, and the tightest such stack leaves the roomier ones for
         later blockers.
      2. If no stack satisfies ``σ_c > blocker_priority`` (unavoidable
         re-relocation), pick the one with the LARGEST ``σ_c`` — this
         delays the eventual re-relocation of ``blocker_priority`` as
         long as possible.
    """
    cands: List[Tuple[StackKey, List[int]]] = []
    keys = candidate_keys if candidate_keys is not None else list(stacks.keys())
    for k in keys:
        if k == src_key:
            continue
        s = stacks.get(k)
        if s is None or len(s) >= max_tiers:
            continue
        cands.append((k, s))

    if not cands:
        return None

    good = [
        (k, s) for k, s in cands if _min_priority_of(s, n_containers) > blocker_priority
    ]
    if good:
        # smallest σ_c > blocker
        good.sort(key=lambda t: _min_priority_of(t[1], n_containers))
        return good[0][0]

    # all stacks worse than blocker → pick largest σ_c
    cands.sort(key=lambda t: -_min_priority_of(t[1], n_containers))
    return cands[0][0]


# ────────────────────────────────────────────────────────────────────
#  EM (Expected MinMax) — full per-realization enumeration
# ────────────────────────────────────────────────────────────────────


def em_retrieve_batch(
    stacks: Stacks,
    batch: Sequence[int],
    realization: Sequence[int],
    max_tiers: int,
    n_containers: int,
    batch_idx: int,
) -> Optional[DecisionNode]:
    """
    Execute one batch under a specific realization using MinMax.  Returns
    a ``DecisionNode`` whose moves list contains every relocation and
    retrieval for this batch, in temporal order.

    The input ``stacks`` dict is **mutated in-place** to reflect the yard
    state after the batch is retrieved.  Callers who need to preserve
    state should pass a clone.
    """
    moves: List[Movement] = []
    for target in realization:
        src = _locate(stacks, target)
        if src is None:
            return None  # infeasible: container missing from yard
        # Relocate every blocker above ``target``, top-first
        while stacks[src] and stacks[src][-1] != target:
            blocker = stacks[src][-1]
            dst = minmax_pick_stack(
                stacks,
                max_tiers=max_tiers,
                src_key=src,
                blocker_priority=blocker,
                n_containers=n_containers,
            )
            if dst is None:
                return None  # no feasible destination
            stacks[src].pop()
            stacks[dst].append(blocker)
            moves.append(Movement(blocker, src, dst))
        # Retrieve ``target`` (now on top of ``src``)
        stacks[src].pop()
        moves.append(Movement(target, src, None))

    return DecisionNode(
        batch_idx=batch_idx,
        realization=tuple(realization),
        moves=moves,
        next_chance=None,   # set by the outer driver
    )


# ────────────────────────────────────────────────────────────────────
#  RIRH — Algorithm 3 in Bacci 2022
# ────────────────────────────────────────────────────────────────────


def _batch_members_topdown(stacks: Stacks, batch: Sequence[int]) -> List[Tuple[int, StackKey, int]]:
    """
    Return the members of ``batch`` present in ``stacks`` in the
    "leftmost stack, topmost slot" order used by Algorithm 3
    (i.e. iterate stacks in the natural key order; within a stack take
    the batch members from the top down).
    """
    batch_set = set(batch)
    out: List[Tuple[int, StackKey, int]] = []
    for key in sorted(stacks.keys()):
        s = stacks[key]
        # top-down within the stack
        for h in range(len(s) - 1, -1, -1):
            p = s[h]
            if p in batch_set:
                out.append((p, key, h))
    return out


def rirh_try_batch(
    stacks: Stacks,
    batch: Sequence[int],
    max_tiers: int,
    n_containers: int,
    batch_idx: int,
) -> Optional[DecisionNode]:
    """
    Try Algorithm 3 (Bacci 2022, page 5): assign a *disjoint* set of
    destination columns for the blockers above each batch member.  This
    yields a set of moves that is valid regardless of the intra-batch
    retrieval order.

    Returns
    -------
    DecisionNode whose moves are realization-independent, or ``None`` if
    the procedure fails (in which case the caller must fall back to EM).

    Implementation notes
    --------------------
    - ``R(i)`` (blockers above ``i``) is captured from the INITIAL stacks
      passed by the caller.  This matches the paper's definition
      ("blockers above i in the initial configuration").
    - The "``i`` sits above another batch member" check considers the
      **entire** column below ``i`` in the initial stack (not just the
      immediately underneath slot), matching the paper's semantics.
    - Once container ``i`` has been retrieved, the stack it just left is
      re-inserted into ``Θ`` **only if** the stack has no remaining
      batch member and has spare capacity.  This mirrors the Fig 4
      example in Bacci 2022 where the blockers of a lower batch member
      end up in a column that was just emptied by a higher batch
      member's retrieval.
    - ``Set Θ = Θ \\ C`` (rule r2) is applied strictly, so blockers of
      distinct batch members never share a destination column.
    """
    # --- Θ initialisation (from the incoming state) -------------------
    bset = set(batch)
    theta: List[StackKey] = []
    for k, s in stacks.items():
        if len(s) >= max_tiers:
            continue
        if any(p in bset for p in s):
            continue
        theta.append(k)

    initial = _clone(stacks)         # frozen initial layout for R(i)
    working = _clone(stacks)          # yard state that evolves
    all_moves: List[Movement] = []

    for (i, i_key, i_tier) in _batch_members_topdown(initial, batch):
        C: List[StackKey] = []
        # R(i) from the INITIAL layout: blockers strictly above i in
        # the initial stack, bottom-to-top (later processed top-first).
        R_i: List[int] = list(initial[i_key][i_tier + 1 :])
        # If i sits ABOVE any other batch member in the same stack
        # (not necessarily immediately) → in some realization that
        # member is retrieved first, so i itself needs to be moved.
        below_in_stack = initial[i_key][:i_tier]
        if any(p in bset for p in below_in_stack):
            R_i.append(i)

        # Relocate blockers top-first
        for k in reversed(R_i):
            if not theta:
                return None  # Θ exhausted → procedure fails
            dst = minmax_pick_stack(
                working,
                max_tiers=max_tiers,
                src_key=i_key,
                blocker_priority=k,
                n_containers=n_containers,
                candidate_keys=theta,
            )
            if dst is None:
                return None
            # Locate k in the CURRENT working yard (may have moved).
            k_src = _locate(working, k)
            if k_src is None:
                return None
            working[k_src].pop()
            working[dst].append(k)
            all_moves.append(Movement(k, k_src, dst))
            C.append(dst)
            if len(working[dst]) >= max_tiers:
                theta = [t for t in theta if t != dst]

        # Retrieve i (may be at its original spot or at a c ∈ Θ if it
        # was appended to R_i above).
        i_now = _locate(working, i)
        if i_now is None or working[i_now][-1] != i:
            return None
        working[i_now].pop()
        all_moves.append(Movement(i, i_now, None))

        # Θ ← Θ \ C  (rule r2)
        theta = [t for t in theta if t not in C]

        # Re-admit i's stack to Θ if it has capacity AND no remaining
        # batch member.  This lets later batch members' blockers use
        # the newly-freed column (Bacci 2022 Fig 4 example).
        remaining = working[i_now]
        if (
            i_now not in theta
            and len(remaining) < max_tiers
            and not any(p in bset for p in remaining)
        ):
            theta.append(i_now)

    # Propagate the resulting yard state back to the caller for the
    # NEXT batch.
    for k in stacks.keys():
        stacks[k] = working[k]

    canonical = tuple(sorted(batch))
    return DecisionNode(
        batch_idx=batch_idx,
        realization=canonical,
        moves=all_moves,
        next_chance=None,
    )


# ────────────────────────────────────────────────────────────────────
#  Full policy-tree builders (Algorithm 2 in Bacci 2022, Algorithm 1 in
#  Galle et al. 2018 for the EM baseline).
# ────────────────────────────────────────────────────────────────────


def build_em_policy_tree(
    initial_stacks: Stacks,
    batches: Sequence[Sequence[int]],
    max_tiers: int,
    n_containers: int,
) -> ChanceNode:
    """
    Galle et al. 2018 Algorithm 1: enumerate every realization at every
    batch boundary and drive MinMax on each branch.  |leaves| = |Ω_B|.
    """
    root = ChanceNode(batch_idx=0)

    def _grow(cn: ChanceNode, stacks: Stacks, depth: int) -> None:
        if depth >= len(batches):
            return
        b = list(batches[depth])
        for perm in permutations(b):
            local = _clone(stacks)
            dn = em_retrieve_batch(
                local,
                batch=b,
                realization=perm,
                max_tiers=max_tiers,
                n_containers=n_containers,
                batch_idx=depth,
            )
            if dn is None:
                # Infeasible — skip this permutation.
                continue
            cn.add_outcome(perm, dn)
            if depth + 1 < len(batches):
                dn.next_chance = ChanceNode(batch_idx=depth + 1)
                _grow(dn.next_chance, local, depth + 1)

    _grow(root, _clone(initial_stacks), 0)
    return root


def build_rirh_policy_tree(
    initial_stacks: Stacks,
    batches: Sequence[Sequence[int]],
    max_tiers: int,
    n_containers: int,
) -> ChanceNode:
    """
    Bacci et al. 2022 Algorithm 2: try the realization-independent
    procedure per batch; when it fails, fall back to per-realization
    EM branching.  Shared DecisionNodes account for the E[R] reduction.
    """
    root = ChanceNode(batch_idx=0)

    def _grow(cn: ChanceNode, stacks: Stacks, depth: int) -> None:
        if depth >= len(batches):
            return
        b = list(batches[depth])
        # Try realization-independent
        working_ri = _clone(stacks)
        dn_ri = rirh_try_batch(
            working_ri,
            batch=b,
            max_tiers=max_tiers,
            n_containers=n_containers,
            batch_idx=depth,
        )
        if dn_ri is not None:
            # Attach dn_ri to every permutation of b (shared object)
            for perm in permutations(b):
                cn.add_outcome(perm, dn_ri)
            if depth + 1 < len(batches):
                dn_ri.next_chance = ChanceNode(batch_idx=depth + 1)
                _grow(dn_ri.next_chance, working_ri, depth + 1)
            return

        # Fallback: per-realization MinMax (like EM).
        for perm in permutations(b):
            local = _clone(stacks)
            dn = em_retrieve_batch(
                local,
                batch=b,
                realization=perm,
                max_tiers=max_tiers,
                n_containers=n_containers,
                batch_idx=depth,
            )
            if dn is None:
                continue
            cn.add_outcome(perm, dn)
            if depth + 1 < len(batches):
                dn.next_chance = ChanceNode(batch_idx=depth + 1)
                _grow(dn.next_chance, local, depth + 1)

    _grow(root, _clone(initial_stacks), 0)
    return root
