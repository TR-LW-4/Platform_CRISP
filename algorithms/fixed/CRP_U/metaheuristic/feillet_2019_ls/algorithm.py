"""
Feillet
<2019> <heuristic> <unrestricted> <single-bay> <CRP-U>
OPT(n) dynamic-programming local search for uBRP
max_iter --- 0 --- Outer iterations (0 = until convergence)
init --- greedy --- Initial solution: greedy or jzw

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

from algorithms.ALGORITHM import Algorithm
from algorithms.fixed.CRP_U.move import Move
from algorithms.fixed.CRP_U.operator import jzw
from algorithms.fixed.CRP_U.rollout import rollout

from .opt_operator import opt_n
from .solution import Step, build_greedy


class Feillet(Algorithm):

    def main(self):
        max_iter = self.parameter("max_iter", 0, int)
        init = self.parameter("init", "greedy", str)
        stacks = [list(stack) for stack in self.problem.stacks]
        width = len(stacks)
        height = self.problem.max_tiers
        total = self.problem.num_containers
        indexed = [None] + stacks
        if init == "jzw":
            steps = _to_steps(rollout(stacks, height, jzw.choose_move))
        else:
            steps = build_greedy(indexed, width, height, total)
        iteration = 0
        while True:
            improved = False
            for container in range(1, total + 1):
                updated = opt_n(container, steps, indexed, width, height, total)
                if updated is not None:
                    steps = updated
                    improved = True
            iteration += 1
            if not improved or (max_iter > 0 and iteration >= max_iter):
                return _to_moves(steps)


def _to_steps(moves):
    return [
        Step(
            move.container,
            move.from_stack + 1,
            0 if move.to_bay is None else move.to_bay + 1,
        )
        for move in moves
    ]


def _to_moves(steps):
    return [
        Move(
            step.container,
            step.src - 1,
            None if step.dst == 0 else step.dst - 1,
        )
        for step in steps
    ]
