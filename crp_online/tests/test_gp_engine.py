import random
from pathlib import Path
import pytest
from crp_online.gp.tree import GPNode
from crp_online.gp.initialization import ramped_half_and_half, build_full_tree, build_grow_tree
from crp_online.gp.operators import apply_crossover, apply_mutation
from crp_online.gp.engine import SteadyStateGP, Individual
from crp_online.io.caserta_loader import parse_caserta_file


DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "caserta"


def test_gp_node_primitives():
    # Tree: (+ (- SH EMP) RI)
    n1 = GPNode("SH")
    n2 = GPNode("EMP")
    n_sub = GPNode("-", [n1, n2])
    n_ri = GPNode("RI")
    root = GPNode("+", [n_sub, n_ri])

    assert root.depth() == 2
    assert root.size() == 5
    assert root.to_prefix_str() == "+ - SH EMP RI"

    cloned = root.clone()
    assert cloned.to_prefix_str() == root.to_prefix_str()
    # Ensure clone is independent
    cloned.children[1].value = "DIFF"
    assert root.children[1].value == "RI"


def test_ramped_half_and_half():
    rng = random.Random(42)
    pop = ramped_half_and_half(
        pop_size=50,
        min_depth=1,
        max_depth=4,
        functions=["+", "-", "*", "/"],
        terminals=["SH", "AVG", "EMP", "CUR", "RI", "DIFF"],
        rng=rng,
    )
    assert len(pop) == 50
    for ind in pop:
        assert ind.depth() <= 4
        assert not ind.is_terminal()  # Root is function


def test_gp_crossover_and_mutation():
    rng = random.Random(123)
    p1 = build_full_tree(0, 3, ["+", "-", "*", "/"], ["SH", "DIFF"], rng)
    p2 = build_full_tree(0, 3, ["+", "-", "*", "/"], ["AVG", "RI"], rng)

    child = apply_crossover(p1, p2, max_depth=5, rng=rng)
    assert child.depth() <= 5

    mutated = apply_mutation(child, ["+", "-", "*", "/"], ["SH", "DIFF", "AVG", "RI"], max_depth=5, rng=rng)
    assert mutated.depth() <= 5


def test_steady_state_gp_training():
    # Small fast training session on 2 small instances
    inst1 = parse_caserta_file(DATA_DIR / "data3-3-1.dat")
    inst2 = parse_caserta_file(DATA_DIR / "data3-3-2.dat")
    training_set = [inst1, inst2]

    gp = SteadyStateGP(
        pop_size=30,
        max_evals=100,
        tournament_size=3,
        crossover_rate=1.0,
        mutation_rate=0.3,
        max_tree_depth=4,
        functions=["+", "-", "*", "/"],
        terminals=["SH", "AVG", "EMP", "CUR", "RI", "DIFF"],
        scheme="restricted",
        seed=42,
    )

    best_rule, logs = gp.train(training_set, log_interval=50)
    assert best_rule is not None
    assert best_rule.fitness < float("inf")
    assert len(logs) >= 2
    assert logs[-1].best_fitness <= logs[0].best_fitness
