"""Stack quality used by the unrestricted one-step operators."""

from typing import List, Optional, Sequence


class View:
    def __init__(self, stacks: Sequence[Sequence[int]]):
        priorities = [priority for stack in stacks for priority in stack]
        self.n_stacks = len(stacks)
        self.n_blocks = len(priorities)
        self.n_bad = 0
        self.h = [len(stack) for stack in stacks]
        cap = (max(priorities) + 1) if priorities else 1
        self.p: List[List[int]] = []
        self.q: List[List[int]] = []
        self.b: List[List[int]] = []
        for stack in stacks:
            priority = [cap] + list(stack)
            quality = [0] * len(priority)
            badness = [0] * len(priority)
            quality[0] = cap
            for tier in range(1, len(priority)):
                if priority[tier] <= quality[tier - 1]:
                    quality[tier] = priority[tier]
                    badness[tier] = 0
                else:
                    quality[tier] = quality[tier - 1]
                    badness[tier] = badness[tier - 1] + 1
                if badness[tier] > 0:
                    self.n_bad += 1
            self.p.append(priority)
            self.q.append(quality)
            self.b.append(badness)
        self.list = sorted(
            range(self.n_stacks),
            key=lambda stack: (self.q[stack][self.h[stack]], self.b[stack][self.h[stack]]),
        )
        self.rank = [0] * self.n_stacks
        for index, stack in enumerate(self.list):
            self.rank[stack] = index

    def top_q(self, stack: int) -> int:
        return self.q[stack][self.h[stack]]

    def top_b(self, stack: int) -> int:
        return self.b[stack][self.h[stack]]

    def top_p(self, stack: int) -> int:
        return self.p[stack][self.h[stack]]


def view(stacks: Sequence[Sequence[int]]) -> Optional[View]:
    if not stacks or not any(stacks):
        return None
    return View(stacks)
