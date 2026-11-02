from typing import List, Sequence


class BayFloor:
    """Low-level container storage array representing vertical stacks.
    
    Each stack is a dynamic array (list) of integer container IDs, where:
      - index 0 represents tier 0 (the yard floor / bottom tier)
      - index (len - 1) represents the top container on the stack
    """

    def __init__(self, num_stacks: int, initial_layout: Sequence[Sequence[int]] | None = None) -> None:
        self.num_stacks = num_stacks
        if initial_layout is not None:
            self._stacks: List[List[int]] = [list(stack) for stack in initial_layout]
            assert len(self._stacks) == num_stacks, f"Expected {num_stacks} stacks, got {len(self._stacks)}"
        else:
            self._stacks = [[] for _ in range(num_stacks)]

    def get_stack_height(self, stack: int) -> int:
        return len(self._stacks[stack])

    def is_stack_empty(self, stack: int) -> bool:
        return len(self._stacks[stack]) == 0

    def get_top_container(self, stack: int) -> int:
        if not self._stacks[stack]:
            raise IndexError(f"Stack {stack} is empty!")
        return self._stacks[stack][-1]

    def get_container_at_height(self, stack: int, height: int) -> int:
        return self._stacks[stack][height]

    def get_stack_containers(self, stack: int) -> tuple[int, ...]:
        return tuple(self._stacks[stack])

    def pop_top_container(self, stack: int) -> int:
        if not self._stacks[stack]:
            raise IndexError(f"Cannot pop from empty stack {stack}!")
        return self._stacks[stack].pop()

    def push_container(self, stack: int, container_id: int) -> None:
        self._stacks[stack].append(container_id)

    def move_container(self, source_stack: int, dest_stack: int) -> int:
        cid = self.pop_top_container(source_stack)
        self.push_container(dest_stack, cid)
        return cid

    def clone(self) -> "BayFloor":
        return BayFloor(self.num_stacks, self._stacks)

    def __repr__(self) -> str:
        lines = [f"Stack {s}: {self._stacks[s]}" for s in range(self.num_stacks)]
        return "BayFloor(\n  " + "\n  ".join(lines) + "\n)"
