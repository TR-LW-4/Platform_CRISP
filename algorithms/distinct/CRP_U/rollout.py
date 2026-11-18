"""
Rollout of one relocation rule for CRP-U
<heuristic> <unrestricted> <single-bay> <CRP-U>
operator --- dsk --- Relocation rule from the operator folder

------------------------------- Reference --------------------------------
Cite the paper listed in the Reference section of the chosen operator.
------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section.
--------------------------------------------------------------------------
"""

import importlib

from algorithms.ALGORITHM import Algorithm


class Move:
    def __init__(self, container, source, destination):
        self.container = container
        self.source = source
        self.destination = destination


class RolloutU(Algorithm):

    def main(self):
        name = self.parameter("operator", "dsk", str)
        module = importlib.import_module("algorithms.distinct.CRP_U.operator." + name)
        choose = module.choose_move
        state = [list(stack) for stack in self.problem.stacks]
        height = self.problem.max_tiers
        limit = sum(len(stack) for stack in state) * height * max(len(state), 1)
        moves = []

        while True:
            print(state)
            present = [priority for stack in state for priority in stack]
            if not present:
                return moves
            target = min(present)
            home = next(index for index, stack in enumerate(state) if target in stack)

            if state[home][-1] == target:
                moves.append(Move(target, (home, len(state[home])), None))
                state[home].pop()
                continue

            chosen = choose(state, height)
            if chosen is None or len(moves) > limit:
                raise ValueError(f"operator cannot retrieve container {target}")
            source, destination = chosen
            if source == destination or not state[source] or len(state[destination]) >= height:
                raise ValueError(f"operator returned an illegal relocation {chosen}")
            container = state[source][-1]
            moves.append(Move(
                container,
                (source, len(state[source])),
                (destination, len(state[destination]) + 1),
            ))
            state[source].pop()
            state[destination].append(container)