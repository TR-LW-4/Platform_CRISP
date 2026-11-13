"""
GLAH
<2015> <heuristic> <unrestricted> <single-bay> <CRP-U>
Greedy look-ahead heuristic
D --- 3 --- Look-ahead depth
n_ftbg --- 5 --- Fast BG successors

------------------------------- Reference --------------------------------
B. Jin, W. Zhu, A. Lim,
"Solving the container relocation problem by an improved greedy look-ahead
 heuristic",
European Journal of Operational Research 240 (2015) 837–847.
------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section.
--------------------------------------------------------------------------
"""

from algorithms.ALGORITHM import Algorithm
from algorithms.fixed.CRP_U.move import Move

from .evaluate import evaluation_heuristic
from .layout import GlahLayout, GlahState, lower_bound
from .lookahead import Lookahead


class GLAH(Algorithm):

    def main(self):
        depth = self.parameter("D", 3, int)
        limits = [
            self.parameter(name, default, int)
            for name, default in (
                ("n_ftbg", 5),
                ("n_nfbg", 5),
                ("n_ftbb", 3),
                ("n_nfbb", 3),
                ("n_gg", 1),
                ("n_gb", 1),
            )
        ]
        layout = GlahLayout.build_from_stacks(self.problem.stacks, self.problem.max_tiers)
        opening = GlahState(layout.copy())
        evaluation_heuristic(opening)
        search = Lookahead(depth, *limits)
        state = GlahState(layout.copy())
        state.best_ops = opening.best_ops
        state.best_reloc = opening.best_reloc
        state.try_retrievals()
        while not state.is_empty():
            if lower_bound(state.inst) + state.reloc_count >= state.best_reloc:
                break
            operation = search.most_promising_relocation(state)
            if operation is None:
                break
            state.go_one_step(operation)
            state.try_retrievals()
        operations = state.best_ops or opening.best_ops or state.ops
        return _ops_to_moves(operations)


def _ops_to_moves(operations):
    moves = []
    for operation in operations or []:
        destination = None if operation.to_s == 0 else operation.to_s - 1
        moves.append(Move(operation.gc.priority, operation.from_s - 1, destination))
    return moves
