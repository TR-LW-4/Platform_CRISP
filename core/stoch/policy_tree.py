"""
Policy tree for the Stochastic Container Relocation Problem (SCRP).

The SCRP calls for a **commitment-style** decision policy:  when a batch
B_k is being processed, the yard-crane must commit to a full sequence of
relocations *without* knowing the intra-batch retrieval order of the
following batches.  Only when the last container of B_k leaves the yard
is the retrieval order of B_{k+1} revealed and the algorithm may branch
its future decisions on that realization.

This is naturally represented by a bipartite tree:

    ChanceNode  (uncertainty)  ──►  DecisionNode  (committed moves)
        ▲                                    │
        │                                    ▼
        └──────────  next batch  ────────────┘

Notation
--------
- Root of a policy = ``ChanceNode`` (batch index 0)
- Each ChanceNode has one child ``DecisionNode`` per realization it
  branches on.  Two realizations that lead to *identical* decisions
  (as with Bacci 2022 realization-independent moves) share the SAME
  ``DecisionNode`` object.
- Each ``DecisionNode`` holds a list of ``Movement`` objects
  (relocations + one retrieval per container of the current batch)
  and a link to the next ChanceNode (or ``None`` for terminal).

Expected cost
-------------
Under the uniform intra-batch distribution assumed by Bacci 2022 and
Galle et al. 2018, the expected number of relocations is:

    E[R]  =  (1 / |Ω_B|)  *  Σ_{ω ∈ Ω_B}  R(ω)

where R(ω) is the number of relocations performed along the unique path
from the root to a leaf induced by realization ω.  The recursive
computation implemented here traverses the tree in O(|nodes|) time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from core.plan import Movement


# ────────────────────────────────────────────────────────────────────
#  Node types
# ────────────────────────────────────────────────────────────────────


@dataclass
class DecisionNode:
    """
    A committed sequence of moves for one batch under one realization
    prefix.  Every SCRP heuristic that respects the identical-order-
    identical-moves rule produces DecisionNodes shared across multiple
    realizations of the current batch.
    """
    batch_idx: int
    realization: Tuple[int, ...]         # priorities in retrieval order
    moves: List[Movement] = field(default_factory=list)
    next_chance: Optional["ChanceNode"] = None

    # --- basic queries ------------------------------------------------- #

    def num_relocations(self) -> int:
        """Non-retrieval moves in this node only."""
        return sum(1 for m in self.moves if not m.is_retrieval)

    def num_retrievals(self) -> int:
        return sum(1 for m in self.moves if m.is_retrieval)


@dataclass
class ChanceNode:
    """
    Batch-boundary branching point.  ``outcomes`` maps each realization
    (as an *ordered* tuple of priorities) to the DecisionNode that will
    be executed under it.  Multiple realizations may share the same
    DecisionNode instance — that is precisely what Bacci 2022 exploits.
    """
    batch_idx: int
    outcomes: Dict[Tuple[int, ...], DecisionNode] = field(default_factory=dict)

    def add_outcome(self, realization: Tuple[int, ...], node: DecisionNode) -> None:
        self.outcomes[tuple(realization)] = node

    def num_outcomes(self) -> int:
        return len(self.outcomes)


# Type alias used by the callers (env.evaluate_policy(tree))
PolicyTree = ChanceNode


# ────────────────────────────────────────────────────────────────────
#  Expected-cost evaluation
# ────────────────────────────────────────────────────────────────────


def expected_relocations(root: Optional[ChanceNode]) -> float:
    """
    Recursively compute E[R] = expected number of relocations under a
    policy tree with the uniform intra-batch distribution.

    Returns 0.0 for an empty / None tree (matches the convention that
    an already-empty yard requires 0 relocations).
    """
    if root is None or not root.outcomes:
        return 0.0

    cache: Dict[int, float] = {}

    def _decision_cost(dn: DecisionNode) -> float:
        # id-based caching handles shared DecisionNode instances.
        key = id(dn)
        if key in cache:
            return cache[key]
        c = float(dn.num_relocations()) + _chance_cost(dn.next_chance)
        cache[key] = c
        return c

    def _chance_cost(cn: Optional[ChanceNode]) -> float:
        if cn is None or not cn.outcomes:
            return 0.0
        total = 0.0
        for dn in cn.outcomes.values():
            total += _decision_cost(dn)
        return total / len(cn.outcomes)

    return _chance_cost(root)


def total_relocations_along(root: Optional[ChanceNode],
                            realization: List[int]) -> int:
    """
    Simulate a single realization ω = (ω_1, ω_2, ...) — a concatenation
    of intra-batch orders — through the policy tree and return the
    number of relocations actually performed.

    Used by tests / plotting; the average of many such calls converges
    to :func:`expected_relocations`.
    """
    if root is None or not root.outcomes:
        return 0

    cursor = 0
    node = root
    n_reloc = 0
    while node is not None:
        # Determine current batch size from stored outcome keys.
        any_key = next(iter(node.outcomes))
        b_size = len(any_key)
        omega_k = tuple(realization[cursor : cursor + b_size])
        cursor += b_size
        dn = node.outcomes.get(omega_k)
        if dn is None:
            # No branch prepared for this realization — infeasible policy.
            return n_reloc
        n_reloc += dn.num_relocations()
        node = dn.next_chance
    return n_reloc


# ────────────────────────────────────────────────────────────────────
#  Utility: pretty-print a policy tree (for debugging)
# ────────────────────────────────────────────────────────────────────


def describe(root: Optional[ChanceNode], max_depth: int = 5) -> str:
    """Human-readable summary of the tree structure (top ``max_depth`` levels)."""
    if root is None:
        return "<empty PolicyTree>"

    lines: List[str] = []

    def _walk_chance(cn: ChanceNode, depth: int, prefix: str) -> None:
        if depth > max_depth:
            lines.append(f"{prefix}⋯ (truncated)")
            return
        lines.append(
            f"{prefix}Chance(batch={cn.batch_idx}, outcomes={cn.num_outcomes()})"
        )
        seen: Dict[int, str] = {}
        for i, (real, dn) in enumerate(cn.outcomes.items()):
            key = id(dn)
            if key in seen:
                lines.append(f"{prefix}  ω{i}={real} → same as {seen[key]}")
                continue
            seen[key] = f"ω{i}"
            _walk_decision(dn, depth + 1, prefix + "  ")

    def _walk_decision(dn: DecisionNode, depth: int, prefix: str) -> None:
        lines.append(
            f"{prefix}Decision(batch={dn.batch_idx}, order={dn.realization}, "
            f"reloc={dn.num_relocations()}, retr={dn.num_retrievals()})"
        )
        if dn.next_chance is not None:
            _walk_chance(dn.next_chance, depth + 1, prefix + "  ")

    _walk_chance(root, 0, "")
    return "\n".join(lines)
