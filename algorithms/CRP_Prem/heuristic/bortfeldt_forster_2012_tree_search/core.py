"""
Layout representation, transitive well-/badly-placed predicates, clean supply
and move classification for the Bortfeldt & Forster (2012) tree search.

All definitions follow the paper strictly (Section 4, Table 1). In particular
the badly-placed predicate is *transitive*: an item is badly placed if the
item immediately below it has a smaller group index **or** if any item below
it is itself badly placed. Any item above a badly-placed item is therefore
also badly placed.

Stack representation
--------------------
``Stacks[s]`` is a list of group indices bottom -> top. A smaller group index
means higher priority (i.e. loaded earlier / should sit on top in a final
layout). The final layout condition (paper §1) is:

    for each stack s and each pair of tiers h > h': arr[h] <= arr[h']

Equivalently, ``arr`` is non-increasing bottom -> top.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

Stacks = Dict[int, List[int]]  # stack_idx -> group indices bottom..top
Move = Tuple[int, int]         # (donator, receiver)


# ---------------------------------------------------------------- #
# Cloning / basic accessors                                         #
# ---------------------------------------------------------------- #

def clone_stacks(stacks: Stacks) -> Stacks:
    return {k: list(v) for k, v in stacks.items()}


def stack_height(stacks: Stacks, s: int) -> int:
    return len(stacks.get(s, []))


def top_group(stacks: Stacks, s: int) -> Optional[int]:
    arr = stacks.get(s, [])
    if not arr:
        return None
    return int(arr[-1])


def infer_num_groups(stacks: Stacks) -> int:
    """
    BF §1 defines G as the number of container groups. When groups are not
    supplied explicitly, take G = max value present. This is safe because the
    only place ``G`` is *semantically* needed is:
      - the clean-supply of empty stacks (paper §4 def (3), only group G),
      - the ``dg`` filtering rule for empty receivers (normal-move rule 1
        uses ``g(receiver_top) = G`` for an empty receiver).
    """
    g_max = 0
    for arr in stacks.values():
        for x in arr:
            if x > g_max:
                g_max = int(x)
    return max(1, g_max)


def num_empty_stacks(stacks: Stacks) -> int:
    return sum(1 for arr in stacks.values() if not arr)


# ---------------------------------------------------------------- #
# Transitive badly-placed predicate                                  #
# ---------------------------------------------------------------- #

def first_bad_tier(arr: List[int]) -> int:
    """
    Return the 0-based tier index of the lowest badly-placed item in ``arr``.
    Returns ``len(arr)`` when the stack is clean.

    Uses the paper's transitive definition: once an item at height ``h`` is
    badly placed, so is every item stored above it.
    """
    if len(arr) < 2:
        return len(arr)
    min_below = arr[0]
    for h in range(1, len(arr)):
        if arr[h] > min_below:
            return h
        if arr[h] < min_below:
            min_below = arr[h]
    return len(arr)


def is_clean_stack(stacks: Stacks, s: int) -> bool:
    arr = stacks.get(s, [])
    return first_bad_tier(arr) == len(arr)


def bad_placed_count(stacks: Stacks) -> int:
    """Total number of badly placed items nb (BF §4, transitive)."""
    total = 0
    for arr in stacks.values():
        total += len(arr) - first_bad_tier(arr)
    return total


def bad_placed_per_stack(stacks: Stacks) -> Dict[int, int]:
    """Per-stack nb(s)."""
    return {s: len(arr) - first_bad_tier(arr) for s, arr in stacks.items()}


def is_final_layout(stacks: Stacks) -> bool:
    """A layout is final iff no stack has a badly placed item."""
    return bad_placed_count(stacks) == 0


def highest_well_placed_group(stacks: Stacks, s: int) -> Optional[int]:
    """
    BF §4 def (2): 'the highest well placed item in s'. Returns its group
    index; ``None`` for empty stacks.
    """
    arr = stacks.get(s, [])
    if not arr:
        return None
    hb = first_bad_tier(arr)
    if hb == 0:
        # Bottom item is (defensively) treated as always well placed;
        # first_bad_tier() returns >= 1 for non-empty arrays, so this branch
        # is only reached if the array is empty (already handled above).
        return None
    return int(arr[hb - 1])


# ---------------------------------------------------------------- #
# Potential supply and clean supply (BF §4 Table 1)                 #
# ---------------------------------------------------------------- #

def potential_supply_slots_g(stacks: Stacks, g: int, max_tiers: int) -> int:
    """
    sp(g): potential supply slots of group g.

    A slot (h, s) is a potential supply slot of g iff:
      (i) stack s is non-empty and its highest well-placed item has group g;
      (ii) the slot lies above the highest well-placed item.

    Every tier from ``first_bad_tier(arr)`` up to ``max_tiers`` (exclusive)
    contributes one such slot, regardless of whether it currently holds a
    badly-placed item.
    """
    total = 0
    for arr in stacks.values():
        if not arr:
            continue
        hb = first_bad_tier(arr)
        if hb == 0:
            continue
        if int(arr[hb - 1]) != g:
            continue
        total += max(0, max_tiers - hb)
    return total


def cumulative_potential_supply(stacks: Stacks, g: int, num_groups: int, max_tiers: int) -> int:
    """
    Sp(g) = sp(g) + sp(g+1) + ... + sp(G) + H * ns_empty  (BF §4 Table 1).
    """
    inner = sum(potential_supply_slots_g(stacks, gg, max_tiers) for gg in range(g, num_groups + 1))
    return inner + max_tiers * num_empty_stacks(stacks)


def demand_group(stacks: Stacks, g: int) -> int:
    """d(g): number of badly placed items of group g."""
    d = 0
    for arr in stacks.values():
        hb = first_bad_tier(arr)
        for h in range(hb, len(arr)):
            if arr[h] == g:
                d += 1
    return d


def cumulative_demand(stacks: Stacks, g: int, num_groups: int) -> int:
    """D(g) = d(g) + d(g+1) + ... + d(G)."""
    return sum(demand_group(stacks, gg) for gg in range(g, num_groups + 1))


def clean_supply_slots_g(stacks: Stacks, g: int, max_tiers: int, num_groups: int) -> int:
    """
    sc(g): clean supply slots of group g.
      - potential supply slots of g in *clean* stacks, plus
      - H per empty stack, all counted only for g = G.
    """
    total = 0
    for arr in stacks.values():
        if not arr:
            if g == num_groups:
                total += max_tiers
            continue
        if first_bad_tier(arr) != len(arr):
            continue
        top_g = int(arr[-1])
        if top_g != g:
            continue
        total += max(0, max_tiers - len(arr))
    return total


def clean_supply_weighted(stacks: Stacks, max_tiers: int, num_groups: int) -> int:
    """
    Sc(L) = sum_{g=1..G} 10^(g-1) * sc(g)  (BF §4 Table 1).

    Second-priority sort key when ranking compound moves. Higher is better,
    since higher-group clean supply enables more downstream BG moves.
    """
    total = 0
    for g in range(1, num_groups + 1):
        sc = clean_supply_slots_g(stacks, g, max_tiers, num_groups)
        if sc:
            total += (10 ** (g - 1)) * sc
    return total


# ---------------------------------------------------------------- #
# Move classification (BF §4 end)                                    #
# ---------------------------------------------------------------- #

def move_type(stacks: Stacks, move: Move) -> str:
    """
    Classify an applicable move as one of BG, BB, GG, GB.

    - was_bad: the moved item is badly placed *before* the move (i.e. its
               source-stack tier index is at or above the first bad tier).
    - will_bad: the item will be badly placed *after* the move. This is
               determined transitively: if the destination already contains
               a badly-placed item, the new top is bad. Otherwise it is bad
               iff its group exceeds ``min(dst_arr)``.
    """
    src, dst = move
    src_arr = stacks[src]
    dst_arr = stacks[dst]
    if not src_arr:
        raise ValueError("move classification called on empty donator")
    item = int(src_arr[-1])
    was_bad = (len(src_arr) - 1) >= first_bad_tier(src_arr)
    if not dst_arr:
        will_bad = False
    elif first_bad_tier(dst_arr) < len(dst_arr):
        will_bad = True
    else:
        will_bad = item > min(dst_arr)
    if was_bad and not will_bad:
        return "BG"
    if was_bad and will_bad:
        return "BB"
    if not was_bad and not will_bad:
        return "GG"
    return "GB"


def is_non_bg(move_type_str: str) -> bool:
    return move_type_str in ("BB", "GG", "GB")


# ---------------------------------------------------------------- #
# Move application                                                   #
# ---------------------------------------------------------------- #

def apply_move(stacks: Stacks, move: Move, max_tiers: int) -> bool:
    """Apply an (src, dst) move in place. Returns True on success."""
    src, dst = move
    if src == dst:
        return False
    if src not in stacks or dst not in stacks:
        return False
    if not stacks[src] or len(stacks[dst]) >= max_tiers:
        return False
    x = stacks[src].pop()
    stacks[dst].append(x)
    return True


def undo_move(stacks: Stacks, move: Move) -> None:
    """Reverse a previously applied (src, dst) move."""
    src, dst = move
    stacks[src].append(stacks[dst].pop())


def apply_move_sequence(stacks: Stacks, moves: List[Move], max_tiers: int) -> bool:
    """Apply a sequence of moves in place. Returns True iff all succeeded."""
    for mv in moves:
        if not apply_move(stacks, mv, max_tiers):
            return False
    return True


def legal_move_targets(
    stacks: Stacks, donator: int, max_tiers: int, forbidden_moves: Optional[set] = None,
) -> List[int]:
    """Return destination stack ids for which (donator, dst) is applicable."""
    if not stacks.get(donator):
        return []
    out: List[int] = []
    for dst in stacks:
        if dst == donator:
            continue
        if len(stacks[dst]) >= max_tiers:
            continue
        if forbidden_moves is not None and (donator, dst) in forbidden_moves:
            continue
        out.append(dst)
    return out


def inverse(move: Move) -> Move:
    return (move[1], move[0])
