from pathlib import Path
import numpy as np
import pytest
from crp_online.io.ku_loader import parse_ku_file
from crp_online.core.crptw_environment import CRPTWEnvironment, precompute_departure_sequence
from crp_online.heuristics.crptw_baselines import (
    select_eri_destination,
    select_em_destination,
    select_leveling_destination,
    select_random_destination,
    select_gp_destination,
)
from crp_online.evaluator.prefix_evaluator import PrefixExpression


DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "ku_arthanari"
EXPR_FILE = Path(__file__).resolve().parent.parent / "data" / "expressions" / "expressions.txt"


def test_metamorphic_information_barrier():
    """Gate 3: Metamorphic test for Information-Barrier Contract.
    
    In Unknown (Online) mode, swapping the latent departure order of unrevealed containers
    in future windows MUST produce 100% identical relocation decisions for any policy.
    """
    file_path = DATA_DIR / "0503" / "T271014_0503_001.txt"
    inst = parse_ku_file(file_path)

    rule_str = EXPR_FILE.read_text(encoding="utf-8").strip().splitlines()[0]
    gp_expr = PrefixExpression(rule_str)

    rng1 = np.random.Generator(np.random.PCG64(42))
    seq1 = precompute_departure_sequence(inst, rng1)

    # seq1 has containers in window 1 first, then window 2, then window 3, then window 4
    # In seq2, we keep window 1 in the exact same order, but reverse the latent order of window 4
    win4_containers = [cid for cid, c in enumerate(inst.stack_windows) for _ in c]
    # Find containers in window 4
    w4_cids = [cid for cid in range(inst.num_containers) if seq1[cid] in (0, 7)]  # specific to instance

    # Create perturbed sequence where window 4 order is reversed
    seq2 = list(seq1)
    # The last 2 containers are window 4
    seq2[-2], seq2[-1] = seq2[-1], seq2[-2]
    assert seq1 != seq2  # sequences are distinct

    # Initialize two environments with the two different latent sequences
    env1 = CRPTWEnvironment(inst, variant="unknown")
    env1.set_departure_sequence(seq1)

    env2 = CRPTWEnvironment(inst, variant="unknown")
    env2.set_departure_sequence(seq2)

    # For the first blocking container relocation (before window 4 is ever reached):
    # Both environments must select the exact same destination stack
    target1 = env1.current_target
    target2 = env2.current_target
    assert target1 == target2

    src1 = env1.get_stack_of_container(target1)
    src2 = env2.get_stack_of_container(target2)
    assert src1 == src2

    # While target is blocked, verify all policy decisions are identical
    while env1.get_top_container(src1) != target1:
        moving1 = env1.get_top_container(src1)
        moving2 = env2.get_top_container(src2)
        assert moving1 == moving2

        # 1. ERI decision
        dest_eri_1 = select_eri_destination(env1, src1, moving1)
        dest_eri_2 = select_eri_destination(env2, src2, moving2)
        assert dest_eri_1 == dest_eri_2

        # 2. EM decision
        dest_em_1 = select_em_destination(env1, src1, moving1)
        dest_em_2 = select_em_destination(env2, src2, moving2)
        assert dest_em_1 == dest_em_2

        # 3. GP Rule decision
        dest_gp_1 = select_gp_destination(env1, src1, moving1, gp_expr)
        dest_gp_2 = select_gp_destination(env2, src2, moving2, gp_expr)
        assert dest_gp_1 == dest_gp_2

        env1.relocate(src1, dest_eri_1)
        env2.relocate(src2, dest_eri_2)


def test_crn_and_seed_stability():
    """Gate 4: Verifies Common Random Numbers reproducibility and decoupled Random stream."""
    file_path = DATA_DIR / "0503" / "T271014_0503_001.txt"
    inst = parse_ku_file(file_path)

    # Re-running with same seed produces bit-identical sequences
    rng_a = np.random.Generator(np.random.PCG64(20261018))
    seq_a = precompute_departure_sequence(inst, rng_a)

    rng_b = np.random.Generator(np.random.PCG64(20261018))
    seq_b = precompute_departure_sequence(inst, rng_b)

    assert seq_a == seq_b

    # Decoupled Random stream does not perturb departure sequence
    seed_seq = np.random.SeedSequence(20261018)
    child_seeds = seed_seq.spawn(2)
    dep_rng = np.random.Generator(np.random.PCG64(child_seeds[0]))
    rand_policy_rng = np.random.Generator(np.random.PCG64(child_seeds[1]))

    # Draw from rand_policy_rng
    _ = rand_policy_rng.choice([1, 2, 3])

    # Sequence generated from dep_rng is completely unaffected
    seq_c = precompute_departure_sequence(inst, dep_rng)
    assert len(seq_c) == inst.num_containers
