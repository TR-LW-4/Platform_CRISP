"""
Problem-specific terminal functions for Ðurasević & Ðumić (2024) GP.

Scope
-----
Phase-1 deployment — multi-bay CRP with **distinct due dates**, restricted
relocation scheme.  Only the "standard 6" terminals from Table 1 and the
two multi-bay terminals (DIS, DUR) are implemented here.

The terminals for the **duplicate / container-groups** variant
(NRC, NCUR, NSC, HC, DIFFI, CH, NCH) are intentionally left out — they
live in a future ``CRP-Groups`` problem class and can be dropped into a
separate ``terminals_groups.py`` when that problem is added.

Context dict
------------
Each terminal function takes a dict ``ctx`` with keys:

    sim       : current (simulated) Yard state
    src       : (bay, row) of the source stack
    dst       : (bay, row) of the candidate destination stack
    blocker   : Container to be relocated (its .priority is CUR)
    kin       : KinematicsModel (for DUR)
    max_tiers : yard capacity per stack

All terminals return ``float``.  Helpers treat empty stacks sensibly.
"""

from __future__ import annotations

from typing import Callable, Dict, List

from core.objectives import KinematicsModel


# ================================================================ #
#  Standard 6 terminals (applicable to every CRP variant)            #
# ================================================================ #

def term_SH(ctx: dict) -> float:
    """Height of the candidate destination stack."""
    return float(ctx["sim"].stacks[ctx["dst"]].height)


def term_EMP(ctx: dict) -> float:
    """Remaining capacity of the candidate destination stack."""
    return float(ctx["max_tiers"] - ctx["sim"].stacks[ctx["dst"]].height)


def term_CUR(ctx: dict) -> float:
    """Retrieval order (= priority / due-date) of the blocker to relocate."""
    return float(ctx["blocker"].priority)


def term_RI(ctx: dict) -> float:
    """
    Reshuffle index of the destination stack: number of containers
    already in ``dst`` whose priority is smaller than the blocker's
    (i.e. will force a re-relocation of the blocker later).
    """
    p = int(ctx["blocker"].priority)
    return float(sum(
        1 for c in ctx["sim"].stacks[ctx["dst"]].containers
        if c.priority < p
    ))


def term_AVG(ctx: dict) -> float:
    """Average priority of containers currently in ``dst`` (0 if empty)."""
    stk = ctx["sim"].stacks[ctx["dst"]]
    if stk.is_empty:
        return 0.0
    return float(sum(c.priority for c in stk.containers) / stk.height)


def term_DIFF(ctx: dict) -> float:
    """
    ``blocker.priority − min_priority(dst)`` when ``dst`` is non-empty.
    Measures how 'badly' the blocker fits — positive value means there
    is already a smaller-priority (earlier-retrieved) container beneath.
    Returns 0.0 on empty stacks.
    """
    stk = ctx["sim"].stacks[ctx["dst"]]
    if stk.is_empty:
        return 0.0
    return float(ctx["blocker"].priority - min(
        c.priority for c in stk.containers
    ))


# ================================================================ #
#  Multi-bay terminals (paper Table 1, multi-bay section)            #
# ================================================================ #

def term_DIS(ctx: dict) -> float:
    """Euclidean distance between source and destination stacks."""
    sb, sr = ctx["src"]
    db, dr = ctx["dst"]
    return float(((sb - db) ** 2 + (sr - dr) ** 2) ** 0.5)


def term_DUR(ctx: dict) -> float:
    """
    RMGC carry-time from ``src`` to ``dst`` under the Lee-&-Lee
    kinematic model (γ_row, γ_bay, γ_acc — spreader time omitted since
    it is a constant offset for every candidate).
    """
    kin: KinematicsModel = ctx["kin"]
    return float(kin.travel_time(ctx["src"], ctx["dst"]))


# ================================================================ #
#  Registry                                                           #
# ================================================================ #

#: Standard 6 — used by every CRP variant in the paper.
STANDARD_TERMINALS: Dict[str, Callable[[dict], float]] = {
    "SH":   term_SH,
    "EMP":  term_EMP,
    "CUR":  term_CUR,
    "RI":   term_RI,
    "AVG":  term_AVG,
    "DIFF": term_DIFF,
}

#: Extra two for multi-bay with crane-time objective.
MULTIBAY_EXTRA_TERMINALS: Dict[str, Callable[[dict], float]] = {
    "DIS": term_DIS,
    "DUR": term_DUR,
}


def build_terminal_table(use_multibay_extras: bool) -> Dict[str, Callable]:
    """
    Compose the dict of terminal-name → evaluator function.

    Parameters
    ----------
    use_multibay_extras : when True, DIS and DUR are appended — turn
                          this on whenever the GP is optimising
                          crane_time or running on a multi-bay yard.
    """
    table = dict(STANDARD_TERMINALS)
    if use_multibay_extras:
        table.update(MULTIBAY_EXTRA_TERMINALS)
    return table


def terminal_names(use_multibay_extras: bool) -> List[str]:
    """Sorted list of terminal names exposed to GP."""
    return sorted(build_terminal_table(use_multibay_extras).keys())
