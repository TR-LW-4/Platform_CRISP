"""
Rollout of one relocation rule for CRP-O-U.
<heuristic> <unrestricted> <single-bay> <CRP-O-U>
operator --- pu1 --- Relocation rule: pu1 or pu2

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
        "algorithms.online.CRP_O_U.operator." + operator
    ).choose_moves
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

        steps = choose(state, height, rank)
        if not steps:
            raise ValueError("no legal relocation")
        for move_source, destination in steps:
            if not state[move_source] or len(state[destination]) >= height:
                raise ValueError(f"illegal relocation {(move_source, destination)}")
            container = state[move_source].pop()
            state[destination].append(container)
            moves.append(Move(
                container,
                (move_source, len(state[move_source]) + 1),
                (destination, len(state[destination])),
            ))
    return moves


class RolloutOU(Algorithm):
    compatible_problems = ["CRP-O-U"]

    def main(self):
        operator = self.parameter("operator", "pu1", str)
        return solve(self.problem, operator)
