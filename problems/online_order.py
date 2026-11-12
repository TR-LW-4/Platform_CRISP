

from problems.PROBLEM import BaseProblem

class CRP_O_R(BaseProblem):
    name = "CRP-O-R"
    rule = "R"
    order = "online"

    def targets(self, remaining):
        return targets(remaining, self.retrieval_order)


class CRP_O_U(BaseProblem):
    name = "CRP-O-U"
    rule = "U"
    order = "online"

    def targets(self, remaining):
        return targets(remaining, self.retrieval_order)