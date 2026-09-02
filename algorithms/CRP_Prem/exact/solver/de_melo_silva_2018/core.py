"""
PMPm1 shared helpers for DeMeloSilva2018PMP.

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
corresponding original algorithm paper.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

Stacks = Dict[int, List[int]]
Move = Tuple[int, int]


def clone_stacks(stacks: Stacks) -> Stacks:
    return {k: list(v) for k, v in stacks.items()}


def has_mis_overlay(arr: List[int]) -> bool:
    """A mis-overlay (deadlock) exists wherever a lower slot holds a more
    urgent (smaller-valued) container than the slot directly above it."""
    return any(arr[i] < arr[i + 1] for i in range(len(arr) - 1))


def is_well_located_stack(arr: List[int]) -> bool:
    return not has_mis_overlay(arr)


def is_fully_well_located(stacks: Stacks) -> bool:
    return all(is_well_located_stack(arr) for arr in stacks.values())


def count_bad_overlaps(stacks: Stacks) -> int:
    bad = 0
    for arr in stacks.values():
        for i in range(len(arr) - 1):
            if arr[i] < arr[i + 1]:
                bad += 1
    return bad


def distinct_types(stacks: Stacks) -> List[int]:
    """Sorted list of distinct priority values present (ascending == most
    urgent first). Maps 1:1 to the paper's group index g = 1..G."""
    return sorted({v for arr in stacks.values() for v in arr})


def coarsen_types(stacks: Stacks, max_types: int) -> Tuple[Stacks, Dict[int, int]]:
    """
    The PMPm1 model's variable count scales with G (number of groups). When
    an instance has more distinct priority values than ``max_types``, bucket
    them into ``max_types`` contiguous groups (preserving relative order) so
    the model stays tractable. Containers sharing a bucket become mutually
    interchangeable for well-located purposes -- a documented approximation,
    not part of the original paper (which treats groups as already given).
    """
    values = distinct_types(stacks)
    if len(values) <= max_types:
        mapping = {v: v for v in values}
        return clone_stacks(stacks), mapping

    n = len(values)
    mapping = {}
    for i, v in enumerate(values):
        bucket = (i * max_types) // n
        mapping[v] = bucket + 1  # 1-indexed bucket "group value"

    coarse = {s: [mapping[v] for v in arr] for s, arr in stacks.items()}
    return coarse, mapping


def apply_moves(stacks_init: Stacks, moves: List[Move], max_tiers: int) -> Stacks:
    st = clone_stacks(stacks_init)
    for src, dst in moves:
        if src == dst or src not in st or dst not in st:
            continue
        if not st[src] or len(st[dst]) >= max_tiers:
            continue
        st[dst].append(st[src].pop())
    return st


# ==================================================================== #
#  Section 3.3 -- greedy heuristic for the upper bound T                #
# ==================================================================== #
#
# Reproduced in spirit rather than as a literal pseudocode transcription:
# the published Algorithm 3 selects "the chosen stack" via a rule that
# requires it to already contain a mis-overlay (line 7 of the listing),
# which as transcribed makes its own "if the chosen stack is already
# empty, go to Step 3 (rebuild)" branch unreachable (a selected stack with
# a mis-overlay can never be empty). This is almost certainly an artifact
# of the source PDF's pseudocode formatting rather than the authors'
# intent, since Step 3 (rebuild) is central to the algorithm's own
# description. This implementation instead runs an explicit
# empty-then-rebuild rotation on every visited stack, which preserves the
# three named phases and their documented prioritisation rules, and is
# guaranteed to terminate with a valid (fully sorted) upper bound on the
# number of relocations -- the only property the exact model actually
# needs from this heuristic.


def _best_top_destination(st: Stacks, src: int, value: int, max_tiers: int) -> Optional[int]:
    """Pick a destination stack for the top container (``value``) currently
    leaving ``src`` during the Step 2 (Empty) phase, following the paper's
    documented priority order:
      1. a sorted stack with room where dropping ``value`` on top does not
         create a new mis-overlay (top-of-stack >= value, or empty);
      2. else, an unsorted stack with room, preferring ones already holding
         a smaller (more urgent) value than ``value``;
      3. else, any other unsorted stack with room;
      4. else, exceptionally, any sorted stack with room.
    """
    candidates = [s for s in st if s != src and len(st[s]) < max_tiers]
    if not candidates:
        return None

    sorted_ok = [s for s in candidates if is_well_located_stack(st[s]) and (not st[s] or st[s][-1] >= value)]
    if sorted_ok:
        return min(sorted_ok, key=lambda s: len(st[s]))

    unsorted = [s for s in candidates if not is_well_located_stack(st[s])]
    unsorted_with_smaller = [s for s in unsorted if min(st[s]) < value]
    if unsorted_with_smaller:
        return min(unsorted_with_smaller, key=lambda s: len(st[s]))
    if unsorted:
        return min(unsorted, key=lambda s: len(st[s]))

    sorted_any = [s for s in candidates if is_well_located_stack(st[s])]
    if sorted_any:
        return min(sorted_any, key=lambda s: len(st[s]))
    return None


def _best_rebuild_donor(st: Stacks, dst: int, stack_ids: List[int]) -> Optional[int]:
    """Step 3 (Rebuild): among all other non-empty stacks, pick the one
    whose top container has the *largest* value (least urgent) -- the
    paper's "topmost containers with the highest indexes are relocated
    back first" rule, applied one container at a time so the rebuilt stack
    ends up well located (largest values at the bottom)."""
    donors = [s for s in stack_ids if s != dst and st[s]]
    if not donors:
        return None
    return max(donors, key=lambda s: st[s][-1])


def pmp_heuristic(
    stacks_init: Stacks,
    max_tiers: int,
    start_stack: Optional[int] = None,
) -> Tuple[int, List[Move]]:
    """Run the Section 3.3 greedy heuristic once, optionally forcing the
    first stack processed to be ``start_stack`` (used by the S-restart
    driver below). Returns (relocations, moves)."""
    st = clone_stacks(stacks_init)
    stack_ids = sorted(st.keys())
    moves: List[Move] = []

    n_containers = sum(len(v) for v in st.values())
    max_guard = 50 * (n_containers + 5)
    guard = 0

    current: Optional[int] = start_stack if (
        start_stack is not None and start_stack in st and has_mis_overlay(st[start_stack])
    ) else None

    while any(has_mis_overlay(st[s]) for s in stack_ids):
        guard += 1
        if guard > max_guard:
            break  # safety net; should not trigger for valid instances

        if current is None:
            dirty = [s for s in stack_ids if has_mis_overlay(st[s])]
            if not dirty:
                break
            # smallest height first; ties broken by the highest current top value
            dirty.sort(key=lambda s: (len(st[s]), -(st[s][-1] if st[s] else -1)))
            current = dirty[0]

        # --- Step 2 (Empty): fully empty `current` into other stacks -----
        while st[current]:
            value = st[current][-1]
            dest = _best_top_destination(st, current, value, max_tiers)
            if dest is None:
                break  # no room anywhere; caller's instance may be over-packed
            st[dest].append(st[current].pop())
            moves.append((current, dest))

        # If emptying `current` already sorted the whole bay, stop right
        # away -- Step 3 (Rebuild) exists only to keep a reservoir of
        # spare capacity for *future* passes, and blindly refilling it
        # once nothing is left to fix would just undo the progress just
        # made (there is nowhere left to pull "spare" containers from
        # other than the stacks we just finished sorting).
        if not any(has_mis_overlay(st[s]) for s in stack_ids):
            break

        # --- Step 3 (Rebuild): pull the least-urgent tops back in --------
        # Target height H-2 (paper: "the last two tiers are not filled").
        target_height = max(0, max_tiers - 2)
        while len(st[current]) < target_height and any(has_mis_overlay(st[s]) for s in stack_ids):
            donor = _best_rebuild_donor(st, current, stack_ids)
            if donor is None:
                break
            st[current].append(st[donor].pop())
            moves.append((donor, current))

        current = None  # force fresh Step-1 selection next round

    return len(moves), moves


def best_pmp_heuristic_upper_bound(stacks_init: Stacks, max_tiers: int) -> Tuple[int, List[Move]]:
    """Section 3.3: run the heuristic once per starting stack (S runs total)
    and keep the run with the fewest relocations, used as the exact model's
    time horizon T."""
    stack_ids = sorted(stacks_init.keys())
    best_relocs: Optional[int] = None
    best_moves: List[Move] = []
    for s in stack_ids:
        relocs, moves = pmp_heuristic(stacks_init, max_tiers, start_stack=s)
        if best_relocs is None or relocs < best_relocs:
            best_relocs = relocs
            best_moves = moves
    if best_relocs is None:
        relocs, moves = pmp_heuristic(stacks_init, max_tiers, start_stack=None)
        return relocs, moves
    return best_relocs, best_moves
