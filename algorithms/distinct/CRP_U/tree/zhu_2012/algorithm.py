"""
Iterative deepening A* for CRP-U
<2012> <exact> <unrestricted> <single-bay> <CRP-U>
time_limit --- 300 --- Wall-clock limit (s); the best solution found is returned on timeout
lb_mode --- LB1 --- Lower bound: LB1 (exact) or LB3 (not proven admissible for CRP-U)
probe_mode --- PU2 --- Probe heuristic at frontier nodes: PU1 or PU2
use_visited_map --- true --- Transposition table per iteration (IDA*-UM)

------------------------------- Reference --------------------------------
W. Zhu, H. Qin, A. Lim, H. Zhang,
"Iterative deepening A* algorithms for the container relocation problem",
IEEE Transactions on Automation Science and Engineering 9 (2012) 710–722.
------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section.
--------------------------------------------------------------------------
"""

from algorithms.ALGORITHM import Algorithm

from .ida_core import LB1, LB3, PU1, PU2, ida_star_unrestricted, state_from_stacks


class Move:
    def __init__(self, container, source, destination):
        self.container = container
        self.source = source
        self.destination = destination


class IDAU(Algorithm):

    def main(self):
        time_limit = self.parameter("time_limit", 300.0, float)
        lb_mode = self.parameter("lb_mode", LB1, str)
        probe_mode = self.parameter("probe_mode", PU2, str)
        use_map = self.parameter("use_visited_map", "true", str).lower() in ("1", "true", "yes")
        if lb_mode not in (LB1, LB3):
            lb_mode = LB1
        if probe_mode not in (PU1, PU2):
            probe_mode = PU2
        stacks = [list(stack) for stack in self.problem.stacks]
        result = ida_star_unrestricted(
            state_from_stacks(stacks),
            1,
            self.problem.num_containers,
            self.problem.max_tiers,
            lb_mode=lb_mode,
            probe_mode=probe_mode,
            use_map=use_map,
            time_limit_s=time_limit,
        )
        return _as_moves(self.problem.stacks, result["path"])


def _as_moves(stacks, relocations):
    state = [list(stack) for stack in stacks]
    moves = []
    for source, destination in relocations:
        _retrieve(state, moves)
        container = state[source][-1]
        moves.append(Move(
            container,
            (source, len(state[source])),
            (destination, len(state[destination]) + 1),
        ))
        state[source].pop()
        state[destination].append(container)
    _retrieve(state, moves)
    return moves


def _retrieve(state, moves):
    while any(state):
        target = min(priority for stack in state for priority in stack)
        home = next((index for index, stack in enumerate(state) if stack and stack[-1] == target), None)
        if home is None:
            return
        moves.append(Move(target, (home, len(state[home])), None))
        state[home].pop()
