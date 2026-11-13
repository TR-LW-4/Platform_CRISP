"""Retrieve priority groups in order, with a choice of box within a group."""

from __future__ import annotations

from collections.abc import Sequence, Set

from problems.PROBLEM import BaseProblem

def targets(remaining, data):
    for box_id in data:
        if box_id in remaining:
            return (box_id,)
    return ()


class CRP_R(BaseProblem):
    name = "CRP-R"
    rule = "R"
    order = "fixed"

    def targets(self, remaining):
        return targets(remaining, self.retrieval_order)


class CRP_U(BaseProblem):
    name = "CRP-U"
    rule = "U"
    order = "fixed"

    def targets(self, remaining):
        return targets(remaining, self.retrieval_order)