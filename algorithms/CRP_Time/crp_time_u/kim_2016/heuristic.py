"""
Kim, Kim & Lee (2016), four-case heuristic (Sec. 4, pp. 353-355; Appendix A, p. 357), on the multi-bay layer.

Y. Kim, T. Kim, H. Lee, "Heuristic algorithm for retrieving containers",
Computers & Industrial Engineering 101 (2016) 352-360.

Paper text (page numbers as printed):
- Definitions (p. 353): "Ideal stack: Empty stack or a stack for which the specified order of containers is in
  top-bottom ascending order." "Unideal stack: A stack for which the specified order of containers is not in
  top-bottom ascending order." "Critical stack: Unideal stack in which a container exists that has to be retrieved
  prior to containers above it." "TN(Top Number): Specified order of the container located at the top of a stack."
  "TNC(Top Number of Critical stack): TN of critical stack." "cmin: Container which has minimum order in a yard."
  "tnccmin: TNC of the stack where cmin is located." "Candidate stack: The stack of which TN is in
  A = {x | x in TNIdeal and x > tnccmin}." "When a stack is empty, we assume the TN of the stack to equal big M" (p. 354).
- Case 1 (p. 354): cmin on top: "cmin can be retrieved directly".
- Case 2 (p. 354): no ideal stack: "a target container is moved to the stack of which the minimum order is the largest
  among all other stacks."
- Case 3 (p. 355): "the candidate stack ... which has the smallest TN is selected for relocation. After stack 1 is
  selected, check whether it is possible to shift back rehandlings of the top container in other critical stacks
  without additional rehandling. This inspection proceeds by searching the elements of TNCCritical greater than
  tnccmin and less than TN of the selected stack. After that, the containers with each searched element as an order are
  moved to the selected stack in descending order. After the containers are moved, move the top container of the stack
  containing cmin to the selected stack. When it finishes, return to identify the circumstance of the container yard".
- Case 4 (p. 355): "the ideal stack, which has the largest TN, is selected for relocation, after which TNCCritical
  contains fewer search elements than TN of the selected stack."
- Appendix A (p. 357): "for (i = 1; i < total number of containers; i++) { while (cmin is in yard) { if (cmin is on top)
  { // Case 1 Make cmin move out of yard to the vehicle for shipping } else { if (at least an ideal stack exists) { if
  (at least an candidate stack exists) { // Case 3 The stack which has minimum TN among candidate stacks is selected
  while (critical stacks of which TNCs are between TNstackcmin and TNselected) { Move top containers of them to the
  selected stack in descending order } Move top container of stackcmin to the selected stack } else { // Case 4 The
  stack which has maximum TN among ideal stacks is selected while (critical stacks of which TNCs are less than
  TNselected) { Move top containers of them to the selected stack in descending order } Move top container of
  stackcmin to the selected stack } } else { // Case 2 Move top container of stackcmin to the stack which has the
  biggest minimum movement priority } } }"

Where text and pseudocode differ, the pseudocode is followed:
- P1 Case 3: the text searches the critical stacks once and then moves them; the pseudocode is a while loop, so the
  condition is re-evaluated after every move (a source stack whose new top qualifies can give another container).
- P2 Case 4: the text sentence is garbled; the pseudocode moves critical tops with TNC < TNselected (no lower bound).
- P3 The pseudocode's for loop runs i = 1 .. N - 1, which would leave the last container; every container is
  retrieved (the text: "retrieve all containers", p. 353).

Readings and assumptions (neither text nor pseudocode says otherwise):
- K1 Order: one container per order number (MultiBayLayout requires unique retrieval orders); order = retrieval rank.
- K2 Critical: some container below the top has a smaller order than every container above it. Equivalent to "the
  top's order is larger than the order of some container below it". Fig. 2 (p. 354, the layout rebuilt from the
  move sequence on p. 353) fits: stack 6 = [5, 13, 1] is unideal but not critical, stacks 1-3 are critical with TNC
  6, 11, 9 as stated.
- K3 TNselected in the while loops is the current TN of the selected stack (its top after the last move). With moves
  in descending order this keeps the selected stack ideal in Case 3, as the text intends ("the number of ideal stacks
  either increases or remains unchanged", p. 355).
- K4 Capacity is not addressed: an ideal stack can be selected only if it is not full; a critical top is moved into
  the selected stack only while it has at least two empty slots, so the top of stackcmin still fits; Case 2 takes the
  non-full stacks other than stackcmin (none: ValueError).
- K5 Ties: only empty candidate stacks tie (TN = big M) in Case 3; the paper does not say how to break them: a choice
  with random.Random(seed), stacks sorted by (bay, row). (Case 2 and Case 4 cannot tie: orders are unique and an empty
  stack is always a candidate.)
- K6 Bays: the rules use orders only; bays and rows play no part (the paper states no distance term).
"""

from __future__ import annotations

import random
from typing import Dict, List

from core.plan import Movement, RelocationPlan

from algorithms.CRP_Time.utils.multibay import MultiBayLayout, Position, YardState, check_multibay_plan


def is_ideal(orders: List[int]) -> bool:
    """Empty, or orders in top-bottom ascending order (p. 353); orders are given bottom to top."""
    return all(a > b for a, b in zip(orders, orders[1:]))


def is_critical(orders: List[int]) -> bool:
    """K2: a container below the top whose order is smaller than that of every container above it (p. 353)."""
    return any(all(x > orders[i] for x in orders[i + 1:]) for i in range(len(orders) - 1))


def solve(layout: MultiBayLayout, seed: int = 0) -> RelocationPlan:
    """Working plan of the heuristic; checked with check_multibay_plan(restricted=False) before it is returned."""
    rng = random.Random(seed)
    state = YardState(layout)
    order: Dict[int, int] = {c: n for n, c in enumerate(layout.order, 1)}
    big_m = len(layout.order) + 1
    H = layout.max_height
    plan = RelocationPlan()

    def orders(s: Position) -> List[int]:
        return [order[c] for c in state.stacks[s]]

    def tn(s: Position) -> int:
        return order[state.stacks[s][-1]] if state.stacks[s] else big_m

    def ideal(s: Position) -> bool:
        return is_ideal(orders(s))

    def critical(s: Position) -> bool:
        return is_critical(orders(s))

    def relocate(src: Position, dst: Position) -> None:
        plan.add(Movement(state.move(src, dst), src, dst))

    def pre_moves(sel: Position, sc: Position, low: int) -> None:
        """The while loops of Cases 3 and 4 (p. 357): tops of critical stacks with low < TNC < TNselected, largest
        first, re-evaluated after every move (P1, K3), while sel keeps room for the top of stackcmin (K4)."""
        while H - len(state.stacks[sel]) >= 2:
            qual = [s for s in state.stacks if s not in (sel, sc) and state.stacks[s] and critical(s)
                    and low < tn(s) < tn(sel)]
            if not qual:
                return
            relocate(max(qual, key=tn), sel)

    for c in layout.order:
        while c in state.where:
            sc = state.where[c]
            if state.top(sc) == c:                       # Case 1
                state.retrieve(sc)
                plan.add(Movement(c, sc, None))
                continue
            tncc = tn(sc)
            ideals = [s for s in state.stacks if s != sc and len(state.stacks[s]) < H and ideal(s)]   # K4
            if ideals:
                cands = [s for s in ideals if tn(s) > tncc]
                if cands:                                # Case 3
                    best = min(tn(s) for s in cands)
                    sel = rng.choice(sorted(s for s in cands if tn(s) == best))                    # K5
                    pre_moves(sel, sc, tncc)
                else:                                    # Case 4
                    sel = max(ideals, key=tn)
                    pre_moves(sel, sc, 0)
                relocate(sc, sel)
            else:                                        # Case 2
                others = [s for s in state.stacks if s != sc and len(state.stacks[s]) < H]
                if not others:
                    raise ValueError(f"{layout.name}: no stack can take the top of {sc}")
                relocate(sc, max(others, key=lambda s: min(orders(s), default=big_m)))
    check_multibay_plan(plan, layout, restricted=False)
    return plan
