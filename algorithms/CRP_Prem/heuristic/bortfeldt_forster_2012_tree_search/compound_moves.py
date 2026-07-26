"""
Normal and extra compound-move generation (Bortfeldt & Forster 2012, §6.1-6.2).

A compound move is a permitted sequence of moves that share the same donator
stack and belong to a single move-type family:

  * Normal compound move: only BG moves. Extended until no further BG move
    from the donator is possible (i.e. max length).
  * Extra compound move : only non-BG moves (BB, GG, GB). Every non-empty
    permitted prefix is admissible.

Filtering rules
---------------
Normal (§6.1):
  (N1) Discard moves whose ``dg = g(receiver_top) - g(item)`` is not minimum;
       empty receiver contributes ``g(receiver_top) = G``.
  (N2) Keep only moves with the highest receiver slot (``prmax``).
  (N3) Pick the first survivor.

Extra (§6.2):
  (E1) If any GG move exists, discard all BB / GB. Otherwise, if any move has
       a dirty receiver, discard moves with a clean receiver.
  (E2) Compute ``dg`` with empty receiver contributing ``0`` (**note the
       difference vs. normal**). If any move has ``dg <= 0``, discard moves
       with ``dg > 0``.
  (E3) Keep only moves with the highest receiver slot.
  (E4) Pick the first survivor.

Sorting (§6.1, §6.2)
--------------------
Normal: (num_moves desc, clean_supply desc).
Extra : (num_moves asc,  clean_supply desc). The compound move with the
        globally maximum clean supply is guaranteed to be among the top
        ``n_succ`` returned.

Acceptance
----------
A compound move ``cm`` from partial solution ``s_imported`` on layout ``L`` is
accepted only when

    nm(s_imported) + nm(cm) + lb_moves(L_after) < target

where ``target`` is ``nm(s*)`` if a best solution ``s*`` already exists, else
``round(pub * lb_moves(L_init))`` (paper §6.1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Set, Tuple

from .core import (
    Move,
    Stacks,
    apply_move,
    clean_supply_weighted,
    clone_stacks,
    is_clean_stack,
    is_non_bg,
    move_type,
    stack_height,
    top_group,
    undo_move,
)


LowerBoundFn = Callable[[Stacks, int, int], int]


@dataclass
class CompoundMove:
    """
    Result of generating a compound move.

    Attributes
    ----------
    moves         : the ordered list of (src, dst) moves
    result        : the layout after all moves are applied (fresh copy)
    clean_supply  : Sc(L_after), used as a secondary sort key
    lb_after      : lower bound on remaining moves from ``result``
    kind          : "normal" or "extra"
    """
    moves: List[Move]
    result: Stacks
    clean_supply: int
    lb_after: int
    kind: str = "normal"
    donator: int = -1


# ---------------------------------------------------------------- #
# Filtering rules                                                    #
# ---------------------------------------------------------------- #

def _apply_normal_filters(
    candidates: List[Move],
    stacks: Stacks,
    num_groups: int,
) -> List[Move]:
    """Filtering rules N1 + N2 for normal compound moves (§6.1)."""
    if not candidates:
        return candidates

    def dg_of(mv: Move) -> int:
        src, dst = mv
        item = stacks[src][-1]
        dst_arr = stacks[dst]
        top_g = num_groups if not dst_arr else int(dst_arr[-1])
        return top_g - item

    min_dg = min(dg_of(mv) for mv in candidates)
    filtered = [mv for mv in candidates if dg_of(mv) == min_dg]

    def rslot(mv: Move) -> int:
        # 1-based receiver slot index (paper §4 uses 1..H).
        return len(stacks[mv[1]]) + 1

    prmax = max(rslot(mv) for mv in filtered)
    filtered = [mv for mv in filtered if rslot(mv) == prmax]
    return filtered


def _apply_extra_filters(
    candidates: List[Tuple[Move, str]],
    stacks: Stacks,
) -> List[Move]:
    """Filtering rules E1 + E2 + E3 for extra compound moves (§6.2)."""
    if not candidates:
        return []

    types_present = {t for _, t in candidates}
    if "GG" in types_present:
        pool: List[Move] = [mv for mv, t in candidates if t == "GG"]
    else:
        # If any dirty receiver, drop clean-receiver moves.
        dirty_pool = [mv for mv, _ in candidates if not is_clean_stack(stacks, mv[1])]
        if dirty_pool:
            pool = dirty_pool
        else:
            pool = [mv for mv, _ in candidates]

    def dg_of(mv: Move) -> int:
        src, dst = mv
        item = stacks[src][-1]
        dst_arr = stacks[dst]
        # BF §6.2 rule E2: empty receiver contributes g(receiver_top) = 0.
        top_g = 0 if not dst_arr else int(dst_arr[-1])
        return top_g - item

    dgs = [dg_of(mv) for mv in pool]
    if any(d <= 0 for d in dgs):
        pool = [mv for mv, d in zip(pool, dgs) if d <= 0]

    def rslot(mv: Move) -> int:
        return len(stacks[mv[1]]) + 1

    prmax = max(rslot(mv) for mv in pool)
    pool = [mv for mv in pool if rslot(mv) == prmax]
    return pool


# ---------------------------------------------------------------- #
# Generation of a single compound move from one donator             #
# ---------------------------------------------------------------- #

def _build_normal_from_donator(
    stacks: Stacks,
    donator: int,
    max_tiers: int,
    num_groups: int,
    forbidden_inverses: Set[Move],
    use_filtering: bool,
    truncate_to_single_move: bool,
) -> Optional[List[Move]]:
    """
    Build a max-length normal (BG-only) compound move from ``donator``.
    Returns None when no BG move is possible.
    ``truncate_to_single_move``: paper §7.5 V2 ablation, keep only the first move.
    """
    local = clone_stacks(stacks)
    compound: List[Move] = []
    local_forbidden = set(forbidden_inverses)

    while True:
        if not local[donator]:
            break
        cands: List[Move] = []
        for dst in local:
            if dst == donator:
                continue
            if len(local[dst]) >= max_tiers:
                continue
            mv = (donator, dst)
            if mv in local_forbidden:
                continue
            if move_type(local, mv) != "BG":
                continue
            cands.append(mv)
        if not cands:
            break
        if use_filtering:
            cands = _apply_normal_filters(cands, local, num_groups)
        m_best = cands[0]
        apply_move(local, m_best, max_tiers)
        compound.append(m_best)
        local_forbidden.add((m_best[1], m_best[0]))
        if truncate_to_single_move:
            break
    return compound if compound else None


def _iter_extra_from_donator(
    stacks: Stacks,
    donator: int,
    max_tiers: int,
    forbidden_inverses: Set[Move],
    use_filtering: bool,
    truncate_to_single_move: bool,
) -> List[List[Move]]:
    """
    Yield every non-empty prefix of a non-BG-only compound move from
    ``donator``. Returns them as a list of move-lists (each independently
    applicable to ``stacks``).
    """
    local = clone_stacks(stacks)
    compound: List[Move] = []
    local_forbidden = set(forbidden_inverses)
    prefixes: List[List[Move]] = []

    while True:
        if not local[donator]:
            break
        typed_cands: List[Tuple[Move, str]] = []
        for dst in local:
            if dst == donator:
                continue
            if len(local[dst]) >= max_tiers:
                continue
            mv = (donator, dst)
            if mv in local_forbidden:
                continue
            t = move_type(local, mv)
            if not is_non_bg(t):
                continue
            typed_cands.append((mv, t))
        if not typed_cands:
            break
        if use_filtering:
            pool = _apply_extra_filters(typed_cands, local)
        else:
            pool = [mv for mv, _ in typed_cands]
        if not pool:
            break
        m_best = pool[0]
        apply_move(local, m_best, max_tiers)
        compound.append(m_best)
        prefixes.append(list(compound))
        local_forbidden.add((m_best[1], m_best[0]))
        if truncate_to_single_move:
            break
    return prefixes


# ---------------------------------------------------------------- #
# Public generators                                                  #
# ---------------------------------------------------------------- #

def _acceptance_target(best_move_count: int, pub: float, lb_init: int) -> int:
    """
    ``nm(s*)`` if a best solution exists, else round(pub * lb_init) (§6.1).
    """
    if best_move_count < 10**9:
        return best_move_count
    return int(round(pub * max(0, lb_init)))


def generate_normal_compound_moves(
    stacks: Stacks,
    imported_moves: List[Move],
    *,
    max_tiers: int,
    num_groups: int,
    n_succ: int,
    pub: float,
    best_move_count: int,
    lb_init: int,
    lb_fn: LowerBoundFn,
    use_filtering: bool = True,
    truncate_to_single_move: bool = False,
    sort_output: bool = True,
) -> List[CompoundMove]:
    """
    Generate at most ``n_succ`` normal compound moves for the current layout.

    One compound move per donator stack (any stack containing at least one
    badly-placed item). Accepted only when the acceptance criterion holds.
    """
    accepted: List[CompoundMove] = []
    forbidden = {(m[1], m[0]) for m in imported_moves}
    target = _acceptance_target(best_move_count, pub, lb_init)

    for donator in sorted(stacks.keys()):
        moves = _build_normal_from_donator(
            stacks=stacks,
            donator=donator,
            max_tiers=max_tiers,
            num_groups=num_groups,
            forbidden_inverses=forbidden,
            use_filtering=use_filtering,
            truncate_to_single_move=truncate_to_single_move,
        )
        if not moves:
            continue
        result = clone_stacks(stacks)
        ok = True
        for mv in moves:
            if not apply_move(result, mv, max_tiers):
                ok = False
                break
        if not ok:
            continue
        lb_after = lb_fn(result, num_groups, max_tiers)
        if len(imported_moves) + len(moves) + lb_after >= target:
            continue
        cs = clean_supply_weighted(result, max_tiers, num_groups)
        accepted.append(CompoundMove(
            moves=moves,
            result=result,
            clean_supply=cs,
            lb_after=lb_after,
            kind="normal",
            donator=donator,
        ))

    if sort_output:
        accepted.sort(key=lambda cm: (-len(cm.moves), -cm.clean_supply, cm.donator))
    return accepted[:n_succ]


def generate_extra_compound_moves(
    stacks: Stacks,
    imported_moves: List[Move],
    *,
    max_tiers: int,
    num_groups: int,
    n_succ: int,
    pub: float,
    best_move_count: int,
    lb_init: int,
    lb_fn: LowerBoundFn,
    use_filtering: bool = True,
    truncate_to_single_move: bool = False,
    sort_output: bool = True,
) -> List[CompoundMove]:
    """
    Generate at most ``n_succ`` extra compound moves for the current layout.

    Every non-empty prefix of a permitted non-BG sequence per donator is a
    candidate; only those with positive clean supply and satisfying the
    acceptance criterion are accepted. The globally max clean-supply move is
    guaranteed to be in the returned top-n_succ list (§6.2).
    """
    accepted: List[CompoundMove] = []
    forbidden = {(m[1], m[0]) for m in imported_moves}
    target = _acceptance_target(best_move_count, pub, lb_init)

    for donator in sorted(stacks.keys()):
        prefixes = _iter_extra_from_donator(
            stacks=stacks,
            donator=donator,
            max_tiers=max_tiers,
            forbidden_inverses=forbidden,
            use_filtering=use_filtering,
            truncate_to_single_move=truncate_to_single_move,
        )
        for prefix in prefixes:
            result = clone_stacks(stacks)
            ok = True
            for mv in prefix:
                if not apply_move(result, mv, max_tiers):
                    ok = False
                    break
            if not ok:
                continue
            cs = clean_supply_weighted(result, max_tiers, num_groups)
            if cs <= 0:
                continue
            lb_after = lb_fn(result, num_groups, max_tiers)
            if len(imported_moves) + len(prefix) + lb_after >= target:
                continue
            accepted.append(CompoundMove(
                moves=list(prefix),
                result=result,
                clean_supply=cs,
                lb_after=lb_after,
                kind="extra",
                donator=donator,
            ))

    if sort_output:
        accepted.sort(key=lambda cm: (len(cm.moves), -cm.clean_supply, cm.donator))
        if len(accepted) > n_succ:
            top = accepted[:n_succ]
            max_cs = max(cm.clean_supply for cm in accepted)
            if not any(cm.clean_supply == max_cs for cm in top):
                # Replace the worst-ranked entry with the max-clean-supply cm.
                max_cs_cm = max(accepted, key=lambda cm: cm.clean_supply)
                top[-1] = max_cs_cm
                top.sort(key=lambda cm: (len(cm.moves), -cm.clean_supply, cm.donator))
            return top
    return accepted[:n_succ]
