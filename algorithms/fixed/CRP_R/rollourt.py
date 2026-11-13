import importlib

from algorithms.ALGORITHM import Algorithm


class Move:
    def __init__(self, container, source, destination):
        self.container = container
        self.source = source
        self.destination = destination


class Rollout(Algorithm):

    def main(self):
        name = self.parameter("operator", "min_max", str)
        module = importlib.import_module("algorithms.fixed.CRP_R.operator." + name)
        choose = module.choose_destination
        state = [list(stack) for stack in self.problem.stacks]
        height = self.problem.max_tiers
        moves = []
        
        while True:
            present = [priority for stack in state for priority in stack]
            print(state)
            if not present:
                return moves
            target = min(present)
            source = next(index for index, stack in enumerate(state) if target in stack)
            
            if state[source][-1] == target:
                moves.append(Move(target, (source, len(state[source])), None))
                state[source].pop()
                continue
            
            
            destination = choose(state, height)
            container = state[source][-1]
            moves.append(Move(
                container,
                (source, len(state[source])),
                (destination, len(state[destination]) + 1),
            ))
            state[source].pop()
            state[destination].append(container)
            