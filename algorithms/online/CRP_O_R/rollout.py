"""
Rollout of one relocation rule for CRP-O-R.
<heuristic> <restricted> <single-bay> <CRP-O-R>
operator --- minmax --- Relocation rule: tlp, ri, minmax, pr4, rdh, eri, em, or egh

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
from problems.MOVE import Move


def solve(problem, operator):
    choose = importlib.import_module(
        "algorithms.online.CRP_O_R.operator." + operator
    ).choose_destination
    state = [list(stack) for stack in problem.stacks]
    height = problem.max_tiers
    moves = []
    limit = sum(len(stack) for stack in state) * height * max(len(state), 1) + 1
    while any(state):
        if len(moves) > limit:
            raise ValueError("relocation rule did not finish")
        remaining = {container for stack in state for container in stack}
        target = problem.targets(remaining)[0]
        source = next(index for index, stack in enumerate(state) if target in stack)
        if state[source][-1] == target:
            moves.append(Move(target, (source, len(state[source])), None))
            state[source].pop()
            continue

        def rank(container):
            return problem.score(container, target)

        destination = choose(state, height, rank)
        if destination is None:
            raise ValueError("no legal relocation")
        container = state[source].pop()
        state[destination].append(container)
        moves.append(Move(
            container,
            (source, len(state[source]) + 1),
            (destination, len(state[destination])),
        ))
    return moves


class RolloutOR(Algorithm):
    compatible_problems = ["CRP-O-R"]

    def main(self):
        operator = self.parameter("operator", "minmax", str)
        return solve(self.problem, operator)