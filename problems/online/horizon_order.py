"""Online retrieval with a limited horizon: one fixed order, revealed up to alpha * N."""

from problems.MOVE import Move
from problems.PROBLEM import BaseProblem


def targets(remaining, data):
    for box_id in data:
        if box_id in remaining:
            return (box_id,)
    return ()


def visible(container, next_target, alpha, num_containers):
    last = int(next_target + alpha * num_containers)
    return container if container <= last else last + 1


def replay(problem, relocations):
    """Turn relocations [(source, destination), ...] into the full move sequence.
    A container is retrieved as soon as it is the next target and on top."""
    state = [list(stack) for stack in problem.stacks]
    moves = []

    def retrieve():
        while True:
            wanted = problem.targets({c for stack in state for c in stack})
            top = next((s for s, stack in enumerate(state) if stack and stack[-1] in wanted), None)
            if top is None:
                return
            moves.append(Move(state[top].pop(), (top, len(state[top]) + 1), None))

    retrieve()
    for source, destination in relocations:
        if source == destination or not state[source] or len(state[destination]) >= problem.max_tiers:
            raise ValueError(f"illegal relocation {(source, destination)}")
        container = state[source].pop()
        state[destination].append(container)
        moves.append(Move(container, (source, len(state[source]) + 1),
                        (destination, len(state[destination]))))
        retrieve()
    if any(state):
        raise ValueError("relocations do not empty the bay")
    return moves


class CRP_O_R(BaseProblem):
    name = "CRP-O-R"
    alpha = 1.0

    def targets(self, remaining):
        return targets(remaining, self.retrieval_order)

    def score(self, container, target):
        return visible(container, target, self.alpha, self.num_containers)

    def replay(self, relocations):
        return replay(self, relocations)


class CRP_O_U(BaseProblem):
    name = "CRP-O-U"
    alpha = 1.0

    def targets(self, remaining):
        return targets(remaining, self.retrieval_order)

    def score(self, container, target):
        return visible(container, target, self.alpha, self.num_containers)

    def replay(self, relocations):
        return replay(self, relocations)
