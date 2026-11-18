"""
OPT(n) local search for CRP-U
<2019> <metaheuristic> <unrestricted> <single-bay> <CRP-U>
init --- sm2 --- Starting solution from the operator folder: dsk, jzw, lan, sm1, or sm2

------------------------------- Reference --------------------------------
D. Feillet, S.N. Parragh, F. Tricoire,
"A local-search based heuristic for the unrestricted block relocation problem",
Computers & Operations Research 108 (2019) 44–56.
------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section.
--------------------------------------------------------------------------
"""

import importlib

from algorithms.ALGORITHM import Algorithm

from .dp import opt_n


class Move:
    def __init__(self, container, source, destination):
        self.container = container
        self.source = source
        self.destination = destination


class OPTN(Algorithm):

    def main(self):
        name = self.parameter("init", "sm2", str)
        choose = importlib.import_module("algorithms.distinct.CRP_U.operator." + name).choose_move
        stacks = [list(stack) for stack in self.problem.stacks]
        max_tiers = self.problem.max_tiers
        steps = _rollout(stacks, max_tiers, choose)
        lower = _lower_bounds(stacks)
        relocations = {container: 0 for container in lower}
        for container, _, destination in steps:
            if destination is not None:
                relocations[container] += 1
        improvement = True
        while improvement:
            improvement = False
            for container in sorted(lower):
                if relocations[container] > lower[container]:
                    updated = opt_n(container, steps, stacks, max_tiers, relocations[container])
                    if updated is not None:
                        steps = updated
                        relocations[container] = sum(1 for moved, _, destination in steps
                                                     if moved == container and destination is not None)
                        improvement = True
        return _as_moves(stacks, steps)


def _rollout(stacks, max_tiers, choose):
    state = [list(stack) for stack in stacks]
    steps = []
    limit = sum(len(stack) for stack in state) * max_tiers * max(len(state), 1)
    while any(state):
        target = min(priority for stack in state for priority in stack)
        home = next(index for index, stack in enumerate(state) if target in stack)
        if state[home][-1] == target:
            steps.append((state[home].pop(), home, None))
            continue
        chosen = choose(state, max_tiers)
        if chosen is None or len(steps) > limit:
            raise ValueError(f"operator cannot retrieve container {target}")
        source, destination = chosen
        container = state[source].pop()
        state[destination].append(container)
        steps.append((container, source, destination))
    return steps


def _lower_bounds(stacks):
    return {container: int(index > 0 and container > min(stack[:index]))
            for stack in stacks for index, container in enumerate(stack)}


def _as_moves(stacks, steps):
    state = [list(stack) for stack in stacks]
    moves = []
    for container, source, destination in steps:
        tier = len(state[source])
        state[source].pop()
        if destination is None:
            moves.append(Move(container, (source, tier), None))
        else:
            state[destination].append(container)
            moves.append(Move(container, (source, tier), (destination, len(state[destination]))))
    return moves
