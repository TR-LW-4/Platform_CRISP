from problems.MOVE import Move
from problems.PROBLEM import BaseProblem


def replay(problem, relocations):
    state = [list(stack) for stack in problem.stacks]
    loaded = [0] * len(problem.vessel_heights)
    moves = []

    def retrieve():
        progressed = True
        while progressed:
            progressed = False
            for index, stack in enumerate(state):
                if not stack:
                    continue
                vessel_stack, vessel_tier = stack[-1]
                if vessel_tier != loaded[vessel_stack]:
                    continue
                container = stack.pop()
                loaded[vessel_stack] += 1
                moves.append(Move(container, (index, len(stack) + 1), None))
                progressed = True
                break

    retrieve()
    height = problem.max_tiers
    for source, destination in relocations:
        if source == destination or not state[source] or len(state[destination]) >= height:
            raise ValueError(f"illegal relocation {(source, destination)}")
        container = state[source].pop()
        state[destination].append(container)
        moves.append(Move(
            container,
            (source, len(state[source]) + 1),
            (destination, len(state[destination])),
        ))
        retrieve()
    if any(state):
        raise ValueError("relocations do not empty the bay")
    return moves


class CRP_Stow(BaseProblem):
    name = "CRP-Stow"

    def __init__(self, **data):
        self.vessel_heights = data.pop("vessel_heights")
        super().__init__(**data)
        return replay(self, relocations)