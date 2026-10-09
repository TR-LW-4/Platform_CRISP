"""
Step 3 of López-Plata et al. (2019): postprocessing of phi_best by repairing temporary
movements (Sec. 6.3, p. 444-445; Algorithm 2, line 15). Moves are (s, s'), s' = 0 for a
retrieval; stacks are numbered 1..n_S.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from .bay import Bay, Move, f
from .operating_cost import CostModel


def moved_blocks(stacks: Sequence[Sequence[int]], phi: List[Move], cost: CostModel) -> List[int]:
    """The block taken by each move of phi (the top of its source stack at that moment)."""
    bay, blocks = Bay(stacks, cost), []
    for s, s2 in phi:
        blocks.append(bay.stacks[s - 1][-1])
        bay.apply((s, s2))
    return blocks


def temporary_moves(stacks: Sequence[Sequence[int]], phi: List[Move], cost: CostModel) -> List[Tuple[int, int]]:
    """Pairs (i, j): move i relocates a block b to stack d, the next move of b is the relocation j,
    and no move between them places a block on d, i.e. on top of b (Sec. 6.3.1, O12 (a): no window)."""
    blocks, pairs = moved_blocks(stacks, phi, cost), []
    for i, (_, d) in enumerate(phi):
        if d == 0:
            continue
        j = next((k for k in range(i + 1, len(phi)) if blocks[k] == blocks[i]), None)
        if j is not None and phi[j][1] != 0 and all(phi[k][1] != d for k in range(i + 1, j)):
            pairs.append((i, j))
    return pairs


def refit(stacks: Sequence[Sequence[int]], phi: List[Move], cost: CostModel, i: int, j: int,
          target: int) -> Optional[Tuple[List[Move], float]]:
    """Move i of the temporary pair (i, j) goes to target instead (O13 (d)). The block b is then
    relocated to its original second destination at the first moment it is c+, the block on top
    of s(c*) (restricted rule); the move j is dropped and every other move keeps its order.
    Returns the new moves and their f, or None if a move becomes impossible or if b is not forced
    to move at or before the time of its original second move."""
    b, second = moved_blocks(stacks, phi, cost)[i], phi[j][1]
    bay = Bay(stacks, cost)
    out = list(phi[:i]) + [(phi[i][0], target)]
    try:
        for move in phi[:i]:
            bay.apply(move)
        bay.apply((phi[i][0], target))
        moved = False
        for k in range(i + 1, len(phi)):
            if k == j:
                continue
            if not moved and bay.target is not None:
                s = bay.target_stack()
                if bay.stacks[s - 1][-1] == b:            # b = c+: it must be relocated now
                    bay.apply((s, second))
                    out.append((s, second))
                    moved = True
            if not moved and k > j:
                return None
            bay.apply(phi[k])
            out.append(phi[k])
    except ValueError:
        return None
    if not moved or len(out) != len(phi):
        return None
    return out, bay.g


def improve_once(stacks: Sequence[Sequence[int]], phi: List[Move], cost: CostModel,
                 f_phi: float) -> Optional[Tuple[List[Move], float]]:
    """The first temporary movement (in order) that some other target improves strictly, repaired
    with the target of lowest f (ties to the lowest stack index); None if there is none."""
    for i, j in temporary_moves(stacks, phi, cost):
        best = None
        for target in range(1, len(stacks) + 1):
            if target in phi[i]:
                continue
            fitted = refit(stacks, phi, cost, i, j, target)
            if fitted and fitted[1] < f_phi and (best is None or fitted[1] < best[1]):
                best = fitted
        if best:
            return best
    return None


def postprocess(stacks: Sequence[Sequence[int]], phi: List[Move], cost: CostModel) -> Tuple[List[Move], float]:
    """Sec. 6.3 (p. 444): repair the first improvable temporary movement, then start again from the
    beginning; stop when no temporary movement can be improved."""
    phi, f_phi = list(phi), f(stacks, phi, cost)
    while True:
        better = improve_once(stacks, phi, cost, f_phi)
        if better is None:
            return phi, f_phi
        phi, f_phi = better
