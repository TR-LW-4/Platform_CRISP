import random
from typing import List, Sequence
from .tree import GPNode


def build_full_tree(
    curr_depth: int,
    max_depth: int,
    functions: Sequence[str],
    terminals: Sequence[str],
    rng: random.Random,
) -> GPNode:
    """Recursively builds a tree where all branches extend to max_depth."""
    if curr_depth >= max_depth:
        term = rng.choice(terminals)
        return GPNode(term)

    # Function node (binary arithmetic operators)
    fn = rng.choice(functions)
    left = build_full_tree(curr_depth + 1, max_depth, functions, terminals, rng)
    right = build_full_tree(curr_depth + 1, max_depth, functions, terminals, rng)
    return GPNode(fn, [left, right])


def build_grow_tree(
    curr_depth: int,
    max_depth: int,
    functions: Sequence[str],
    terminals: Sequence[str],
    rng: random.Random,
) -> GPNode:
    """Recursively builds a tree where branches terminate randomly or at max_depth."""
    if curr_depth >= max_depth:
        term = rng.choice(terminals)
        return GPNode(term)

    # If at root, force function to avoid trivial 1-terminal tree
    if curr_depth == 0:
        fn = rng.choice(functions)
        left = build_grow_tree(curr_depth + 1, max_depth, functions, terminals, rng)
        right = build_grow_tree(curr_depth + 1, max_depth, functions, terminals, rng)
        return GPNode(fn, [left, right])

    all_symbols = list(functions) + list(terminals)
    chosen = rng.choice(all_symbols)
    if chosen in terminals:
        return GPNode(chosen)

    left = build_grow_tree(curr_depth + 1, max_depth, functions, terminals, rng)
    right = build_grow_tree(curr_depth + 1, max_depth, functions, terminals, rng)
    return GPNode(chosen, [left, right])


def ramped_half_and_half(
    pop_size: int,
    min_depth: int,
    max_depth: int,
    functions: Sequence[str],
    terminals: Sequence[str],
    rng: random.Random,
) -> List[GPNode]:
    """Generates an initial population using Ramped Half-and-Half initialization."""
    population: List[GPNode] = []
    depths = list(range(min_depth, max_depth + 1))
    trees_per_depth = pop_size // len(depths)

    for depth in depths:
        half = trees_per_depth // 2
        # Full method
        for _ in range(half):
            population.append(build_full_tree(0, depth, functions, terminals, rng))
        # Grow method
        for _ in range(half):
            population.append(build_grow_tree(0, depth, functions, terminals, rng))

    # Fill remainder if pop_size is not perfectly divisible
    while len(population) < pop_size:
        depth = rng.choice(depths)
        method = rng.choice(["full", "grow"])
        if method == "full":
            population.append(build_full_tree(0, depth, functions, terminals, rng))
        else:
            population.append(build_grow_tree(0, depth, functions, terminals, rng))

    return population
