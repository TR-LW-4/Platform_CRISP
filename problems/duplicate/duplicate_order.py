

class CRP_D_R(BaseProblem):
    name = "CRP-D-R"
    rule = "R"
    order = "duplicate"

    def targets(self, remaining):
        return targets(remaining, self.retrieval_order)


class CRP_D_U(BaseProblem):
    name = "CRP-D-U"
    rule = "U"
    order = "duplicate"

    def targets(self, remaining):
        return targets(remaining, self.retrieval_order)