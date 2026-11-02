from pathlib import Path
import numpy as np
import pytest
from crp_online.io.ku_loader import parse_ku_file
from crp_online.core.crptw_environment import CRPTWEnvironment, precompute_departure_sequence
from crp_online.heuristics.crptw_baselines import (
    select_eri_destination,
    select_em_destination,
    select_leveling_destination,
    solve_crptw_trial,
)


DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "ku_arthanari"


def test_golden_micro_instance_analytical_parity():
    """Gate 2: Verifies theoretical 1.5000 moves on T271014_0503_001.txt across 5,000 Monte Carlo draws.
    
    Theoretical derivation (bottom-to-top layout):
    - Stack 1: [3]
    - Stack 2: [1, 2] -> window 2 container blocks window 1 container -> 1 deterministic relocation.
    - Stack 3: [1] -> top container retrieved with 0 relocations.
    - Stack 4: [3]
    - Stack 5: [4, 4, 1] -> top window 1 retrieved with 0 relocations.
    - Remaining window 4 containers in Stack 5 block each other with probability 0.5 -> 0.5 expected relocations.
    - Grand total expectation: exactly 1.5000 moves, std = 0.5000.
    """
    file_path = DATA_DIR / "0503" / "T271014_0503_001.txt"
    inst = parse_ku_file(file_path)

    master_seed = 20261018
    num_trials = 5000
    seed_seq = np.random.SeedSequence(master_seed)
    trial_seeds = seed_seq.generate_state(num_trials)

    eri_results = []
    em_results = []
    l_results = []

    for trial_idx in range(num_trials):
        rng = np.random.Generator(np.random.PCG64(trial_seeds[trial_idx]))
        seq = precompute_departure_sequence(inst, rng)

        # 1. Leveling
        env_l = CRPTWEnvironment(inst, variant="unknown")
        env_l.set_departure_sequence(seq)
        reloc_l, _ = solve_crptw_trial(env_l, select_leveling_destination)
        l_results.append(reloc_l)

        # 2. ERI
        env_eri = CRPTWEnvironment(inst, variant="unknown")
        env_eri.set_departure_sequence(seq)
        reloc_eri, _ = solve_crptw_trial(env_eri, select_eri_destination)
        eri_results.append(reloc_eri)

        # 3. EM
        env_em = CRPTWEnvironment(inst, variant="unknown")
        env_em.set_departure_sequence(seq)
        reloc_em, _ = solve_crptw_trial(env_em, select_em_destination)
        em_results.append(reloc_em)

    mean_l = np.mean(l_results)
    mean_eri = np.mean(eri_results)
    mean_em = np.mean(em_results)

    std_l = np.std(l_results)
    std_eri = np.std(eri_results)
    std_em = np.std(em_results)

    print(f"\nGolden Micro Results (5,000 trials):")
    print(f"Leveling: mean={mean_l:.4f}, std={std_l:.4f}")
    print(f"ERI:      mean={mean_eri:.4f}, std={std_eri:.4f}")
    print(f"EM:       mean={mean_em:.4f}, std={std_em:.4f}")

    # Standard error for N=5000 with sigma=0.5 is 0.5 / sqrt(5000) = 0.00707
    # 3 * SE is ~0.0212
    assert abs(mean_l - 1.5000) < 0.025
    assert abs(mean_eri - 1.5000) < 0.025
    assert abs(mean_em - 1.5000) < 0.025

    assert abs(std_l - 0.5000) < 0.025
    assert abs(std_eri - 0.5000) < 0.025
    assert abs(std_em - 0.5000) < 0.025

    # Invariant lower bound floor test
    b0 = inst.get_initial_blocking_bound()
    se = std_l / np.sqrt(num_trials)
    assert mean_l >= b0 - 3 * se
    assert mean_eri >= b0 - 3 * se
    assert mean_em >= b0 - 3 * se
