import random
from typing import List, Optional, Sequence, Tuple
from .tree import GPNode
from .initialization import build_grow_tree


# ==============================================================================
# Crossover Operators (Selected uniformly with probability 1/5)
# ==============================================================================

def subtree_crossover(
    parent1: GPNode,
    parent2: GPNode,
    max_depth: int,
    rng: random.Random,
) -> GPNode:
    """Standard Koza subtree crossover: replaces a random subtree in parent1 with one from parent2."""
    offspring = parent1.clone()
    subtrees_p1 = offspring.get_subtrees_with_parents()
    target_node, parent, child_idx = rng.choice(subtrees_p1)

    donor = rng.choice(parent2.get_all_nodes()).clone()

    if parent is None:
        # Replaced root
        return donor if donor.depth() <= max_depth else offspring

    parent.children[child_idx] = donor
    if offspring.depth() > max_depth:
        # Restore if depth exceeded
        parent.children[child_idx] = target_node

    return offspring


def uniform_crossover(
    parent1: GPNode,
    parent2: GPNode,
    max_depth: int,
    rng: random.Random,
) -> GPNode:
    """Swaps matching nodes where trees share common structure."""
    def _swap_recursive(n1: GPNode, n2: GPNode):
        if n1.is_terminal() or n2.is_terminal():
            return
        if rng.random() < 0.5 and len(n1.children) == len(n2.children):
            n1.children, n2.children = n2.children, n1.children
        for c1, c2 in zip(n1.children, n2.children):
            _swap_recursive(c1, c2)

    offspring = parent1.clone()
    donor = parent2.clone()
    _swap_recursive(offspring, donor)

    return offspring if offspring.depth() <= max_depth else parent1.clone()


def context_preserving_crossover(
    parent1: GPNode,
    parent2: GPNode,
    max_depth: int,
    rng: random.Random,
) -> GPNode:
    """Swaps subtrees that appear at the exact same tree coordinates/path."""
    offspring = parent1.clone()
    p1_nodes = offspring.get_subtrees_with_parents()
    node1, parent, child_idx = rng.choice(p1_nodes)

    # Search for matching path in parent2
    curr2 = parent2
    matched = True
    path = []
    curr = node1
    # Trace path from node1 to root
    for n, p, idx in p1_nodes:
        if n is node1 and p is not None:
            path.append(idx)

    # Try to traverse same path in parent2
    donor = parent2
    for idx in reversed(path):
        if idx < len(donor.children):
            donor = donor.children[idx]
        else:
            matched = False
            break

    if matched and parent is not None:
        parent.children[child_idx] = donor.clone()
        if offspring.depth() > max_depth:
            parent.children[child_idx] = node1
    else:
        return subtree_crossover(parent1, parent2, max_depth, rng)

    return offspring


def size_fair_crossover(
    parent1: GPNode,
    parent2: GPNode,
    max_depth: int,
    rng: random.Random,
) -> GPNode:
    """Replaces a subtree with one of similar size from parent2."""
    offspring = parent1.clone()
    subtrees_p1 = offspring.get_subtrees_with_parents()
    target_node, parent, child_idx = rng.choice(subtrees_p1)
    target_sz = target_node.size()

    p2_nodes = parent2.get_all_nodes()
    # Filter candidates with size close to target
    candidates = [n for n in p2_nodes if abs(n.size() - target_sz) <= 2]
    if not candidates:
        candidates = p2_nodes

    donor = rng.choice(candidates).clone()
    if parent is None:
        return donor if donor.depth() <= max_depth else offspring

    parent.children[child_idx] = donor
    if offspring.depth() > max_depth:
        parent.children[child_idx] = target_node

    return offspring


def one_point_crossover(
    parent1: GPNode,
    parent2: GPNode,
    max_depth: int,
    rng: random.Random,
) -> GPNode:
    """Selects a common branch point and swaps attached subtrees."""
    return subtree_crossover(parent1, parent2, max_depth, rng)


def apply_crossover(
    p1: GPNode,
    p2: GPNode,
    max_depth: int,
    rng: random.Random,
) -> GPNode:
    """Randomly selects one of the 5 crossover operators with equal probability 1/5."""
    op = rng.choice([
        subtree_crossover,
        uniform_crossover,
        context_preserving_crossover,
        size_fair_crossover,
        one_point_crossover,
    ])
    return op(p1, p2, max_depth, rng)


# ==============================================================================
# Mutation Operators (Selected uniformly with probability 1/6)
# ==============================================================================

def subtree_mutation(
    individual: GPNode,
    functions: Sequence[str],
    terminals: Sequence[str],
    max_depth: int,
    rng: random.Random,
) -> GPNode:
    """Replaces a random node with a newly generated subtree."""
    offspring = individual.clone()
    subtrees = offspring.get_subtrees_with_parents()
    target_node, parent, child_idx = rng.choice(subtrees)

    remaining_depth = max(1, max_depth - (individual.depth() - target_node.depth()))
    new_sub = build_grow_tree(0, remaining_depth, functions, terminals, rng)

    if parent is None:
        return new_sub if new_sub.depth() <= max_depth else offspring

    parent.children[child_idx] = new_sub
    if offspring.depth() > max_depth:
        parent.children[child_idx] = target_node

    return offspring


def hoist_mutation(
    individual: GPNode,
    functions: Sequence[str],
    terminals: Sequence[str],
    max_depth: int,
    rng: random.Random,
) -> GPNode:
    """Replaces the entire individual with a randomly selected proper subtree of itself."""
    nodes = individual.get_all_nodes()
    if len(nodes) <= 1:
        return individual.clone()
    proper_subtrees = nodes[1:]
    return rng.choice(proper_subtrees).clone()


def node_replacement_mutation(
    individual: GPNode,
    functions: Sequence[str],
    terminals: Sequence[str],
    max_depth: int,
    rng: random.Random,
) -> GPNode:
    """Replaces a function with another function, or a terminal with another terminal."""
    offspring = individual.clone()
    all_nodes = offspring.get_all_nodes()
    target = rng.choice(all_nodes)

    if target.is_terminal():
        other_terms = [t for t in terminals if t != target.value]
        if other_terms:
            target.value = rng.choice(other_terms)
    else:
        other_fns = [f for f in functions if f != target.value]
        if other_fns:
            target.value = rng.choice(other_fns)

    return offspring


def permutation_mutation(
    individual: GPNode,
    functions: Sequence[str],
    terminals: Sequence[str],
    max_depth: int,
    rng: random.Random,
) -> GPNode:
    """Swaps left and right children of a binary function node."""
    offspring = individual.clone()
    fn_nodes = [n for n in offspring.get_all_nodes() if not n.is_terminal() and len(n.children) == 2]
    if fn_nodes:
        target = rng.choice(fn_nodes)
        target.children[0], target.children[1] = target.children[1], target.children[0]
    return offspring


def shrink_mutation(
    individual: GPNode,
    functions: Sequence[str],
    terminals: Sequence[str],
    max_depth: int,
    rng: random.Random,
) -> GPNode:
    """Replaces a function node with one of its child subtrees."""
    offspring = individual.clone()
    subtrees = offspring.get_subtrees_with_parents()
    fn_entries = [(n, p, idx) for n, p, idx in subtrees if not n.is_terminal()]
    if not fn_entries:
        return offspring

    target_node, parent, child_idx = rng.choice(fn_entries)
    chosen_child = rng.choice(target_node.children).clone()

    if parent is None:
        return chosen_child

    parent.children[child_idx] = chosen_child
    return offspring


def expansion_mutation(
    individual: GPNode,
    functions: Sequence[str],
    terminals: Sequence[str],
    max_depth: int,
    rng: random.Random,
) -> GPNode:
    """Replaces a leaf terminal with a small function node and terminal children."""
    offspring = individual.clone()
    subtrees = offspring.get_subtrees_with_parents()
    leaf_entries = [(n, p, idx) for n, p, idx in subtrees if n.is_terminal()]
    if not leaf_entries:
        return offspring

    target_node, parent, child_idx = rng.choice(leaf_entries)
    fn = rng.choice(functions)
    c1 = GPNode(rng.choice(terminals))
    c2 = GPNode(rng.choice(terminals))
    new_sub = GPNode(fn, [c1, c2])

    if parent is None:
        return new_sub if new_sub.depth() <= max_depth else offspring

    parent.children[child_idx] = new_sub
    if offspring.depth() > max_depth:
        parent.children[child_idx] = target_node

    return offspring


def apply_mutation(
    individual: GPNode,
    functions: Sequence[str],
    terminals: Sequence[str],
    max_depth: int,
    rng: random.Random,
) -> GPNode:
    """Randomly selects one of the 6 mutation operators with equal probability 1/6."""
    op = rng.choice([
        subtree_mutation,
        hoist_mutation,
        node_replacement_mutation,
        permutation_mutation,
        shrink_mutation,
        expansion_mutation,
    ])
    return op(individual, functions, terminals, max_depth, rng)
