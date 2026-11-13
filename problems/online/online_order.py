from problems.PROBLEM import BaseProblem

def targets(remaining, data):
    for box_id in data:
        if box_id in remaining:
            return (box_id,)
    return ()

def visible(container, next_target, alpha, num_containers):
    last = int(next_target + alpha * num_containers)
    return container if container <= last else last + 1


class CRP_O_R(BaseProblem):
    name = "CRP-O-R"
    rule = "R"
    order = "online"
    alpha = 1.0

    def targets(self, remaining):
        return targets(remaining, self.retrieval_order)

    def visible(self, container, next_target):
        return visible(container, next_target, self.alpha, self.num_containers)


class CRP_O_U(BaseProblem):
    name = "CRP-O-U"
    rule = "U"
    order = "online"
    alpha = 1.0

    def targets(self, remaining):
        return targets(remaining, self.retrieval_order)

    def visible(self, container, next_target):
        return visible(container, next_target, self.alpha, self.num_containers)