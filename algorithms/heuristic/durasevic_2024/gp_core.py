"""
Genetic Programming core — tree representation + standard operators.

Reference
---------
Ðurasević, M.; Ðumić, M. (2024) "Designing relocation rules with genetic
programming for the container relocation problem with multiple bays and
container groups", Applied Soft Computing 150: 111104.

Design
------
GP individual = arithmetic expression tree that maps a context
(yard state + candidate destination) to a scalar **score**.  The smaller
the score, the more attractive the destination.

Function set  : { +, −, ×, ÷ (protected) }                (paper §5)
Terminal set  : problem-specific, supplied by the caller             ↑
Initialisation: ramped half-and-half                                 ↑
Selection     : 3-tournament (keep 2 best, drop worst)               ↑
Variation     : subtree crossover + subtree mutation (both standard) ↑
Replacement   : steady-state — child replaces loser of the tournament ↑

This module is deliberately problem-agnostic: it knows nothing about
container yards.  The caller supplies:
  * a list of ``terminal_names`` (strings)
  * a ``fitness_fn(tree) -> float``  (lower = better)
so the same engine can later be reused for other GP papers in the
platform (Jin 2022, Ðurasević 2025, …).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple


# ================================================================ #
#  Node                                                              #
# ================================================================ #

_OPS = ("+", "-", "*", "/")


@dataclass
class Node:
    """
    One node of a GP expression tree.

    * Internal node:  ``op`` in {+, −, ×, ÷}, ``children`` has 2 Nodes.
    * Terminal leaf (variable):  ``terminal`` is a string name.
    * Terminal leaf (constant):  ``const`` is a float.
    """
    op:       Optional[str]       = None
    children: List["Node"]        = field(default_factory=list)
    terminal: Optional[str]       = None
    const:    Optional[float]     = None

    # ---------------------------------------------------------------- #

    def is_leaf(self) -> bool:
        return self.op is None

    def copy(self) -> "Node":
        if self.is_leaf():
            return Node(terminal=self.terminal, const=self.const)
        return Node(
            op       = self.op,
            children = [c.copy() for c in self.children],
        )

    # ---------------------------------------------------------------- #
    # Evaluation                                                        #
    # ---------------------------------------------------------------- #

    def evaluate(self, ctx: dict) -> float:
        """
        Evaluate the tree given a context dict holding terminal values.

        The caller is responsible for populating ``ctx`` with every key
        that may appear as a terminal in the tree.
        """
        if self.is_leaf():
            if self.terminal is not None:
                return float(ctx.get(self.terminal, 0.0))
            return float(self.const if self.const is not None else 0.0)

        left  = self.children[0].evaluate(ctx)
        right = self.children[1].evaluate(ctx)

        op = self.op
        if op == "+":
            return left + right
        if op == "-":
            return left - right
        if op == "*":
            return left * right
        if op == "/":
            # Protected division (paper §5): returns 1 near-zero divisor
            return left / right if abs(right) > 1e-9 else 1.0
        raise ValueError(f"Unknown op: {op}")

    # ---------------------------------------------------------------- #
    # Diagnostics                                                       #
    # ---------------------------------------------------------------- #

    def depth(self) -> int:
        if self.is_leaf():
            return 0
        return 1 + max(c.depth() for c in self.children)

    def size(self) -> int:
        if self.is_leaf():
            return 1
        return 1 + sum(c.size() for c in self.children)

    def __repr__(self) -> str:
        if self.is_leaf():
            if self.terminal is not None:
                return self.terminal
            return f"{self.const:.2f}"
        return f"({self.children[0]} {self.op} {self.children[1]})"


# ================================================================ #
#  Initialisation                                                    #
# ================================================================ #

def _random_terminal(
    terminal_names: List[str],
    rng:            random.Random,
    const_prob:     float = 0.2,
    const_low:      float = 0.0,
    const_high:     float = 10.0,
) -> Node:
    if rng.random() < const_prob:
        return Node(const=rng.uniform(const_low, const_high))
    return Node(terminal=rng.choice(terminal_names))


def _grow_tree(
    max_depth:      int,
    terminal_names: List[str],
    rng:            random.Random,
    full:           bool = False,
) -> Node:
    """
    Grow a tree with either the 'full' method (only pick internal nodes
    until max_depth, then only terminals) or the 'grow' method
    (probabilistic mix).
    """
    if max_depth <= 0:
        return _random_terminal(terminal_names, rng)

    # For 'grow' method, sometimes pick a terminal even before max_depth
    if not full and rng.random() < 0.25:
        return _random_terminal(terminal_names, rng)

    op = rng.choice(_OPS)
    return Node(
        op       = op,
        children = [
            _grow_tree(max_depth - 1, terminal_names, rng, full=full),
            _grow_tree(max_depth - 1, terminal_names, rng, full=full),
        ],
    )


def init_population_rhh(
    pop_size:       int,
    max_depth:      int,
    terminal_names: List[str],
    rng:            random.Random,
) -> List[Node]:
    """
    Ramped half-and-half (paper §6): half the individuals use the
    'full' method, half use 'grow', spread across depths 2..max_depth.
    """
    pop: List[Node] = []
    depths = list(range(2, max_depth + 1))
    if not depths:
        depths = [max_depth]
    per_depth = max(1, pop_size // len(depths))

    for d in depths:
        for _ in range(per_depth // 2):
            pop.append(_grow_tree(d, terminal_names, rng, full=True))
            pop.append(_grow_tree(d, terminal_names, rng, full=False))

    while len(pop) < pop_size:
        pop.append(
            _grow_tree(max_depth, terminal_names, rng, full=False)
        )
    return pop[:pop_size]


# ================================================================ #
#  Traversal helpers                                                 #
# ================================================================ #

def _collect_nodes(tree: Node) -> List[Tuple[Node, Optional[Node], int]]:
    """
    Return a flat list ``[(node, parent, index_in_parent), …]`` for
    convenient uniform random picking (used by subtree crossover /
    mutation).
    """
    result: List[Tuple[Node, Optional[Node], int]] = []

    def walk(node: Node, parent: Optional[Node], idx: int) -> None:
        result.append((node, parent, idx))
        for i, c in enumerate(node.children):
            walk(c, node, i)

    walk(tree, None, 0)
    return result


# ================================================================ #
#  Subtree crossover and mutation                                    #
# ================================================================ #

def subtree_crossover(
    parent_a:   Node,
    parent_b:   Node,
    max_depth:  int,
    rng:        random.Random,
) -> Node:
    """
    Classic subtree crossover: pick a random node in each parent,
    splice the second parent's subtree into a clone of the first.
    The resulting tree is depth-clamped at ``max_depth + 2`` to avoid
    runaway bloat (the paper also uses a tree-depth cap).
    """
    child        = parent_a.copy()
    nodes_child  = _collect_nodes(child)
    nodes_donor  = _collect_nodes(parent_b)

    cx_child_node, cx_child_parent, cx_child_idx = rng.choice(nodes_child)
    cx_donor_node, _, _                          = rng.choice(nodes_donor)

    donor_copy = cx_donor_node.copy()

    if cx_child_parent is None:
        child = donor_copy
    else:
        cx_child_parent.children[cx_child_idx] = donor_copy

    if child.depth() > max_depth + 2:
        # Bloat guard — fall back to a fresh random tree
        return _grow_tree(max_depth, _get_terminals_of(parent_a), rng)
    return child


def subtree_mutation(
    tree:           Node,
    max_depth:      int,
    terminal_names: List[str],
    rng:            random.Random,
) -> Node:
    """
    Replace a random subtree of ``tree`` with a newly grown subtree.
    """
    child       = tree.copy()
    nodes_child = _collect_nodes(child)

    cx_node, cx_parent, cx_idx = rng.choice(nodes_child)
    new_subtree = _grow_tree(
        max_depth=max(2, max_depth - (cx_node.depth() or 0)),
        terminal_names=terminal_names,
        rng=rng,
        full=False,
    )

    if cx_parent is None:
        return new_subtree
    cx_parent.children[cx_idx] = new_subtree
    return child


# ================================================================ #
#  Best-effort terminal recovery (for bloat-guard fallback)          #
# ================================================================ #

def _get_terminals_of(tree: Node) -> List[str]:
    names = set()

    def walk(n: Node) -> None:
        if n.is_leaf():
            if n.terminal is not None:
                names.add(n.terminal)
        else:
            for c in n.children:
                walk(c)

    walk(tree)
    return sorted(names) if names else ["SH"]


# ================================================================ #
#  Steady-state 3-tournament driver                                  #
# ================================================================ #

def evolve_gp(
    fitness_fn:     Callable[[Node], float],
    terminal_names: List[str],
    pop_size:       int,
    max_depth:      int,
    max_evals:      int,
    mutation_prob:  float,
    rng:            random.Random,
    report_cb:      Optional[Callable[[int, float, Node], None]] = None,
    stop_flag:      Optional[Callable[[], bool]] = None,
) -> Tuple[Node, float]:
    """
    Run steady-state GP (paper Alg. 1) and return the best tree + fitness.

    Parameters
    ----------
    fitness_fn    : callable(tree) → float    (lower is better)
    terminal_names: names of leaves GP may use
    pop_size      : population size
    max_depth     : tree-depth cap
    max_evals     : total fitness evaluations budget (initial pop + offspring)
    mutation_prob : probability of applying subtree mutation to each child
    report_cb     : optional progress callback (evals_done, best_fit, best_tree)
    stop_flag     : optional callable returning True to early-stop
    """
    pop      = init_population_rhh(pop_size, max_depth, terminal_names, rng)
    fitness  = [fitness_fn(t) for t in pop]
    evals    = len(pop)

    # Track best
    best_idx = min(range(len(pop)), key=lambda i: fitness[i])
    best_tree = pop[best_idx].copy()
    best_fit  = fitness[best_idx]

    if report_cb is not None:
        report_cb(evals, best_fit, best_tree)

    while evals < max_evals:
        if stop_flag is not None and stop_flag():
            break

        # 3-tournament
        i1, i2, i3 = rng.sample(range(len(pop)), 3)
        triples    = sorted([i1, i2, i3], key=lambda i: fitness[i])
        p1_idx, p2_idx, loser_idx = triples[0], triples[1], triples[2]

        # Crossover
        child = subtree_crossover(
            pop[p1_idx], pop[p2_idx], max_depth, rng,
        )
        # Mutation
        if rng.random() < mutation_prob:
            child = subtree_mutation(child, max_depth, terminal_names, rng)

        child_fit = fitness_fn(child)
        evals    += 1

        # Steady-state replacement
        pop[loser_idx]     = child
        fitness[loser_idx] = child_fit

        # Track best
        if child_fit < best_fit:
            best_fit  = child_fit
            best_tree = child.copy()
            if report_cb is not None:
                report_cb(evals, best_fit, best_tree)

    return best_tree, best_fit
