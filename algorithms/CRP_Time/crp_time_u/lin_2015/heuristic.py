"""
Lin, Lee & Lee (2015), single-lift heuristic (Sec. 3.2, pp. 136-138), on the multi-bay layer.

D.-Y. Lin, Y.-J. Lee, Y. Lee, "The container retrieval problem with respect to relocation",
Transportation Research Part C 52 (2015) 132-143.

Paper text (page numbers as printed):
- Type B (p. 135): "the containers in the initial layout [divide] into two types: those that do not block other
  containers (type A) and those that do (type B)."
- Notation (p. 136): h(s) containers in stack s; ming(s) the smallest group index in s; g(c) the group of c.
  "a virtual container is placed at the bottom of every stack. The ID number of the virtual container is n + 1, and
  the group index of the container is G + 1. These virtual containers are never moved or retrieved." So an empty
  stack has ming = G + 1.
- Target (p. 136): "If the selected container c is located at the top of its stack, it can be retrieved directly.
  Otherwise, let d = top(s(c)) ... Suppose that there exists at least one stack s, where h(s) < H and ming(s) > g(d)
  ... We select an ideal stack t for this container, based on the rule presented below (Rule I) ... if there is no
  such stack ... we select another stack u based on Rule III".
- Rule I (p. 136-137): "From among all of the stacks s in S \\ {s(c)} satisfying h(s) < H and ming(s) > g(d), select a
  stack t with the smallest stack selection index", SSI(s) = ming(s) + Pr row(s) + Pb |bay(s) - bay(s(c))| (Eq. (1),
  p. 137).
- Rule II (p. 137): "If stack t has available space for more than one container, the heuristic goes one step further
  and determines whether it should perform a pre-relocation, which moves another container k into this stack, before
  moving container d into it." "Find a container k that satisfies the following four conditions. If there is more
  than one such container, select the container with the largest identification number. 1. Container k is of type B,
  which blocks the way of another container. 2. g(k) > g(d). 3. Container k is located at the top of s(k).
  4. ming(t) - 5 < g(k) < ming(t)."
- Rule III (p. 138): "Among all stacks other than s(c) that are not full, select stack u which ming(u) is largest, for
  the relocation of container d."

Readings and assumptions (the paper does not say otherwise):
- A1 Groups: one container per group (MultiBayLayout requires it; true for every Lee & Lee instance), so g(c) is the
  container's rank in the retrieval order, G = n, and the within-group target rule of p. 136 never applies.
- A2 Type B (condition 1) is read as "blocks another container now": some container below k in s(k) has a smaller
  group. The present tense of condition 1 and the rationale ("Because container k blocks the access to another
  container, at least one relocation of container k in the future ... is unavoidable", p. 137) support it; the
  definition on p. 135 is stated for the initial layout.
- A3 Rule II is repeated (Fig. 3, p. 137, the flowchart, as read by Thom): after "Move k into t" the arrow returns to
  "If stack t has more than one empty slot?", so Rule II is retried with the updated ming(t) until t has at most one
  empty slot or no k qualifies; only then "Move d into t". The worked example moves one k (p. 138).
- A4 "Available space for more than one container": h(t) <= H - 2.
- A5 Ties in Rule I and Rule III are not addressed (the paper's "Ties are broken arbitrarily" on p. 136 concerns the
  target within a group): a choice among the tied stacks with random.Random(seed), stacks sorted by (bay, row).
- A6 Single spreader only (Sec. 3.2); Rule IV (multiple spreaders, Sec. 3.3) is not part of this heuristic.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

from core.plan import Movement, RelocationPlan

from algorithms.CRP_Time.utils.multibay import MultiBayLayout, Position, YardState, check_multibay_plan


def solve(layout: MultiBayLayout, P_r: float = 30.0, P_b: float = 300.0, seed: int = 0) -> RelocationPlan:
    """Working plan of the heuristic; checked with check_multibay_plan(restricted=False) before it is returned."""
    rng = random.Random(seed)
    state = YardState(layout)
    group: Dict[int, int] = {c: n for n, c in enumerate(layout.order, 1)}
    virtual = len(layout.order) + 1           # G + 1 (A1)
    H = layout.max_height
    plan = RelocationPlan()

    def ming(s: Position) -> int:
        return min((group[x] for x in state.stacks[s]), default=virtual)

    def blocks(k: int) -> bool:               # A2
        stack = state.stacks[state.where[k]]
        return any(group[x] < group[k] for x in stack[:stack.index(k)])

    def choose(candidates: List[Position], key) -> Position:
        best = min(key(s) for s in candidates)
        return rng.choice(sorted(s for s in candidates if key(s) == best))   # A5

    def relocate(src: Position, dst: Position) -> None:
        plan.add(Movement(state.move(src, dst), src, dst))

    for c in layout.order:
        while True:
            sc = state.where[c]
            if state.top(sc) == c:
                state.retrieve(sc)
                plan.add(Movement(c, sc, None))
                break
            d = state.top(sc)
            others = [s for s in state.stacks if s != sc and len(state.stacks[s]) < H]
            ideal = [s for s in others if ming(s) > group[d]]
            if ideal:                          # Rule I
                t = choose(ideal, lambda s: ming(s) + P_r * s[1] + P_b * abs(s[0] - sc[0]))
                while (k := _rule_ii(state, group, ming, blocks, d, t, H)) is not None:   # A3, Fig. 3
                    relocate(state.where[k], t)
                relocate(sc, t)
            elif others:                       # Rule III
                relocate(sc, choose(others, lambda s: -ming(s)))
            else:
                raise ValueError(f"{layout.name}: no stack can take container {d}")
    check_multibay_plan(plan, layout, restricted=False)
    return plan


def _rule_ii(state: YardState, group, ming, blocks, d: int, t: Position, H: int) -> Optional[int]:
    """Rule II (p. 137): the container k to move into t before d, or None."""
    if H - len(state.stacks[t]) < 2:           # A4
        return None
    mt = ming(t)
    tops = [state.top(s) for s in state.stacks if s != t and state.stacks[s]]
    ks = [k for k in tops if blocks(k) and group[k] > group[d] and mt - 5 < group[k] < mt]
    return max(ks, default=None)               # largest identification number
