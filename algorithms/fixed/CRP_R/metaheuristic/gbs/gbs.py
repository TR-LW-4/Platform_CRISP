"""
Greedy beam search for CRP-R
<2017> <metaheuristic> <restricted> <single-bay> <CRP-R>
beam_width --- 5 --- Nodes retained at each level
operator --- vrh --- Completion rule: vrh or sdh

------------------------------- Reference --------------------------------
C.-J. Ting, K.-C. Wu,
"Optimizing container relocation operations at container yards with
 beam search",
Transportation Research Part E 103 (2017) 17–31.
------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from algorithms.ALGORITHM import Algorithm

from .scoring import sdh_cost, vrh_cost
from .search import _Node, _children, _copy, _retrieve, _select


class GBS(Algorithm):

    def main(self):
        stacks = _copy(self.problem.stacks)
        n_total = self.problem.num_containers
        max_tiers = self.problem.max_tiers
        width = max(1, self.parameter("beam_width", 5, int))
        evaluator = self.parameter("operator", "vrh", str)
        score = vrh_cost if evaluator.upper() == "VRH" else sdh_cost
        def show(moves):
            state = _copy(self.problem.stacks)
            for step in moves:
                print(state)
                state[step.source[0]].pop()
                if step.destination is not None:
                    state[step.destination[0]].append(step.container)
            print(state)
        root_moves = []
        _retrieve(stacks, root_moves)
        if not any(stacks):
            show(root_moves)
            return root_moves

        beam = [_Node(stacks, 0, root_moves)]
        while beam:
            children = []
            for parent_index, parent in enumerate(beam):
                for child, _destination in _children(parent, max_tiers):
                    if not any(child.stacks):
                        show(child.moves)
                        return child.moves
                    children.append((
                        child,
                        float(score(child.stacks, n_total, max_tiers)),
                        parent_index,
                    ))
            if not children:
                break
            beam = _select(children, width, True)
        show(beam[0].moves)
        return beam[0].moves