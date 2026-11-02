import copy
from typing import List, Optional, Tuple


class GPNode:
    """Node in a Genetic Programming prefix expression tree."""

    __slots__ = ("value", "children")

    def __init__(self, value: str, children: Optional[List["GPNode"]] = None) -> None:
        self.value: str = value
        self.children: List["GPNode"] = children if children is not None else []

    def is_terminal(self) -> bool:
        """Returns True if this node is a leaf (terminal), False if it is a function."""
        return len(self.children) == 0

    def depth(self) -> int:
        """Returns the maximum depth of the tree rooted at this node (leaf has depth 0)."""
        if self.is_terminal():
            return 0
        return 1 + max(c.depth() for c in self.children)

    def size(self) -> int:
        """Returns the total number of nodes in the subtree rooted at this node."""
        return 1 + sum(c.size() for c in self.children)

    def to_prefix_str(self) -> str:
        """Serializes tree to Polish prefix notation string."""
        if self.is_terminal():
            return self.value
        child_strs = " ".join(c.to_prefix_str() for c in self.children)
        return f"{self.value} {child_strs}"

    def clone(self) -> "GPNode":
        """Returns an independent deep copy of the tree."""
        return GPNode(self.value, [c.clone() for c in self.children])

    def get_all_nodes(self) -> List["GPNode"]:
        """Returns all nodes in the tree in preorder traversal."""
        nodes = [self]
        for c in self.children:
            nodes.extend(c.get_all_nodes())
        return nodes

    def get_subtrees_with_parents(
        self,
    ) -> List[Tuple["GPNode", Optional["GPNode"], int]]:
        """Returns a list of (node, parent, child_index) tuples for all nodes in the tree."""
        result: List[Tuple["GPNode", Optional["GPNode"], int]] = [(self, None, -1)]
        queue: List["GPNode"] = [self]

        while queue:
            curr = queue.pop(0)
            for idx, child in enumerate(curr.children):
                result.append((child, curr, idx))
                queue.append(child)

        return result

    def __repr__(self) -> str:
        return self.to_prefix_str()
