"""Lower bounds that share one calling shape.

Every bound is ``lower_bound(stacks, heights, cache=None) -> int``. ``lb2c``
and ``lb2c4c`` ignore ``heights`` and ``cache``. ``lbr`` uses both. Pass a
yard that has already had every currently retrievable container removed.
"""

from algorithms.stowage.CRP_S_U.LB.lb2c import lower_bound as lb2c
from algorithms.stowage.CRP_S_U.LB.lb2c4c import lower_bound as lb2c4c
from algorithms.stowage.CRP_S_U.LB.lbr import lower_bound as lbr


BOUNDS = {
    "lb2c": lb2c,
    "lb2c4c": lb2c4c,
    "lbr": lbr,
}


def lower_bound(name, stacks, heights, cache=None):
    found = BOUNDS.get(name)
    if found is None:
        raise ValueError(f"unknown lower bound {name}")
    return found(stacks, heights, cache)
