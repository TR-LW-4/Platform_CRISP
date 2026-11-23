"""
Multi-bay layer for CRP-Time: a Lee & Lee (2010) layout of bays x rows, a yard state, and a strict plan replay.

Independent of the platform's problem classes (which hold one bay on Caserta and validate restricted plans only).
Positions are (bay, row), both from 1; plans are core.plan.RelocationPlan (Movement.to_pos None = retrieval).

layout     MultiBayLayout, load_lee_lee
state      YardState (move, retrieve)
check      check_multibay_plan(plan, layout, restricted) -> ReplaySummary, PlanError
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

from core.benchmarks.lee_lee import parse_lee_lee
from core.plan import RelocationPlan

Position = Tuple[int, int]


@dataclass(frozen=True)
class MultiBayLayout:
    """stacks[(bay, row)]: container ids bottom to top; order: container ids in retrieval order."""
    name: str
    n_bays: int
    n_rows: int
    max_height: int
    stacks: Dict[Position, Tuple[int, ...]]
    order: Tuple[int, ...]

    def orders(self) -> Tuple[Tuple[Tuple[int, ...], ...], ...]:
        """Retrieval orders (1..N) as layout[bay][row] from 0, bottom to top (the form lb_shin takes)."""
        rank = {c: n for n, c in enumerate(self.order, 1)}
        return tuple(tuple(tuple(rank[c] for c in self.stacks[(b, r)]) for r in range(1, self.n_rows + 1))
                     for b in range(1, self.n_bays + 1))


def load_lee_lee(path: Path) -> MultiBayLayout:
    """A Lee & Lee file; containers are retrieved by increasing group, which must be unique (one per group, P3)."""
    inst = parse_lee_lee(path)
    stacks: Dict[Position, Tuple[int, ...]] = {}
    group: Dict[int, int] = {}
    for b in range(inst.n_bays):
        for s in range(inst.n_stacks):
            stacks[(b + 1, s + 1)] = inst.ids[b][s]
            group.update(zip(inst.ids[b][s], inst.groups[b][s]))
    if len(set(group.values())) != len(group):
        raise ValueError(f"{inst.name}: several containers share a group; no single retrieval order")
    return MultiBayLayout(inst.name, inst.n_bays, inst.n_stacks, inst.max_height, stacks,
                          tuple(sorted(group, key=group.get)))


class YardState:
    """Mutable yard of a layout; only the top container of a stack can move."""

    def __init__(self, layout: MultiBayLayout):
        self.max_height = layout.max_height
        self.stacks: Dict[Position, List[int]] = {p: list(s) for p, s in layout.stacks.items()}
        self.where: Dict[int, Position] = {c: p for p, s in layout.stacks.items() for c in s}
        if any(len(s) > self.max_height for s in self.stacks.values()):
            raise ValueError("initial layout exceeds the maximum height")

    def top(self, pos: Position) -> Optional[int]:
        stack = self.stacks.get(pos)
        return stack[-1] if stack else None

    def move(self, src: Position, dst: Position) -> int:
        if dst not in self.stacks:
            raise ValueError(f"destination {dst} does not exist")
        if dst == src:
            raise ValueError(f"destination equals source {src}")
        if len(self.stacks[dst]) >= self.max_height:
            raise ValueError(f"destination {dst} is full (H = {self.max_height})")
        c = self.retrieve(src)
        self.stacks[dst].append(c)
        self.where[c] = dst
        return c

    def retrieve(self, src: Position) -> int:
        if not self.stacks.get(src):
            raise ValueError(f"source {src} is empty or does not exist")
        c = self.stacks[src].pop()
        del self.where[c]
        return c

    def empty(self) -> bool:
        return not self.where


class PlanError(ValueError):
    def __init__(self, step: int, message: str):
        super().__init__(f"move {step}: {message}")
        self.step = step


class ReplaySummary(NamedTuple):
    relocations: int
    retrievals: int
    non_target_relocations: int   # relocations not from the stack of the current target


def check_multibay_plan(plan: RelocationPlan, layout: MultiBayLayout, restricted: bool = False) -> ReplaySummary:
    """Replay plan on layout; raise PlanError at the first violation of: only top containers move, height <= H
    after every move, retrievals in retrieval order, the current target is never relocated (restricted: only
    containers above it move; Shin's model retrieves a target on top at once, env.py clear()), yard empty at the end,
    and (restricted) every relocation from the stack of the current target."""
    state = YardState(layout)
    nxt = 0
    relocations = non_target = 0
    for i, m in enumerate(plan.movements):
        if nxt == len(layout.order):
            raise PlanError(i, "move after the yard is empty")
        if state.top(m.from_pos) != m.container_id:
            raise PlanError(i, f"container {m.container_id} is not on top of {m.from_pos}")
        target_pos = state.where[layout.order[nxt]]
        if m.is_retrieval:
            if m.container_id != layout.order[nxt]:
                raise PlanError(i, f"retrieves {m.container_id}, next in order is {layout.order[nxt]}")
            state.retrieve(m.from_pos)
            nxt += 1
            continue
        if m.container_id == layout.order[nxt]:
            raise PlanError(i, f"relocates the current target {m.container_id}")
        if m.from_pos != target_pos:
            if restricted:
                raise PlanError(i, f"relocation from {m.from_pos}, not from the target's stack {target_pos}")
            non_target += 1
        try:
            state.move(m.from_pos, m.to_pos)
        except ValueError as e:
            raise PlanError(i, str(e)) from None
        relocations += 1
    if not state.empty():
        raise PlanError(len(plan.movements), f"plan ends with {len(state.where)} containers in the yard")
    return ReplaySummary(relocations, nxt, non_target)
