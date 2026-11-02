"""
Comprehensive Test Suite for CRP-Online (2026 Paper) and CRISP Platform Integration.

Covers all core operational scenarios:
1. Standard Research Benchmarks across all 9 registered 2026 algorithms.
2. Synthetic Procedural Generation across diverse yard geometries and seeds.
3. Edge & Boundary Cases (already-sorted yard, single container, deep inversions).
4. Gymnasium API Contract Compliance (5-tuple output, space bounds, action masking).
5. Illegal Action Enforcement & Capacity Penalty Verification (+10,000 penalty).
6. Multiprocessing Progress Streaming & React Frontend Snapshot Format.
7. Custom Genetic Programming Expressions & Rule Index Catalogs.
8. Metamorphic Information-Barrier Verification (unknown container permutation).
"""

from __future__ import annotations

import multiprocessing as mp
import numpy as np
from pathlib import Path
import sys
import pytest

# Ensure repository root is on sys.path
REPO_ROOT = str(Path(__file__).resolve().parents[2])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core.base_algorithm import AlgorithmConfig, ProgressRecord
from core.base_problem import ProblemConfig
from problems.CRP_Online_2026 import CRP_Online_2026

from algorithms.CRP_Online_2026.algorithm import (
    GPOnline2026Algorithm,
    TLP2026Algorithm,
    RI2026Algorithm,
    MM2026Algorithm,
    PR42026Algorithm,
    RDH2026Algorithm,
    ERI2026Algorithm,
    EM2026Algorithm,
    EGH2026Algorithm,
)

# Reference benchmark file
BENCHMARK_FILE = Path(__file__).resolve().parents[2] / "benchmark" / "Caserta_dataset" / "data3-3-1.dat"
KU_BENCHMARK_FILE = Path(__file__).resolve().parents[2] / "crp_online" / "data" / "ku_arthanari" / "0503" / "T271014_0503_001.txt"
ALL_2026_ALGORITHMS = [
    GPOnline2026Algorithm,
    TLP2026Algorithm,
    RI2026Algorithm,
    MM2026Algorithm,
    PR42026Algorithm,
    RDH2026Algorithm,
    ERI2026Algorithm,
    EM2026Algorithm,
    EGH2026Algorithm,
]


# ==============================================================================
# Scenario 1: Standard Research Benchmark Instances across all 9 Algorithms
# ==============================================================================

@pytest.mark.parametrize("algo_cls", ALL_2026_ALGORITHMS)
def test_caserta_benchmark_all_algorithms(algo_cls):
    """Verifies that all 9 algorithms successfully solve a standard Caserta layout."""
    assert BENCHMARK_FILE.is_file(), f"Benchmark file not found at {BENCHMARK_FILE}"

    cfg = ProblemConfig(
        extra={"layout_file_path": str(BENCHMARK_FILE)}
    )
    env = CRP_Online_2026(config=cfg)
    algo = algo_cls(AlgorithmConfig(report_interval=1))

    q: mp.Queue = mp.Queue()
    ev = mp.Event()

    algo.train(lambda: env, q, ev)

    records = []
    while not q.empty():
        records.append(q.get())

    assert len(records) >= 2, "Must emit at least initial and final records"
    initial = records[0]
    final = records[-1]

    # Initial record check
    assert initial.step == 0
    assert initial.metric == 0.0

    # Final completion check
    assert final.progress == 1.0
    assert final.metrics["retrievals"] == 9.0, "All 9 containers in data3-3-1 must be retrieved"
    assert 0 < final.metric < 1000.0, f"Relocations {final.metric} must be positive and non-penalized"
    assert final.best_metric == final.metric


@pytest.mark.parametrize("algo_cls", ALL_2026_ALGORITHMS)
def test_ku_benchmark_all_algorithms(algo_cls):
    """Verifies that all 9 algorithms successfully solve a standard Ku & Arthanari layout."""
    assert KU_BENCHMARK_FILE.is_file(), f"Ku benchmark file not found at {KU_BENCHMARK_FILE}"

    cfg = ProblemConfig(
        extra={"layout_file_path": str(KU_BENCHMARK_FILE)}
    )
    env = CRP_Online_2026(config=cfg)
    algo = algo_cls(AlgorithmConfig(report_interval=1))

    q: mp.Queue = mp.Queue()
    ev = mp.Event()

    algo.train(lambda: env, q, ev)

    records = []
    while not q.empty():
        records.append(q.get())

    assert len(records) >= 2, "Must emit at least initial and final records"
    initial = records[0]
    final = records[-1]

    # Initial record check
    assert initial.step == 0
    assert initial.metric == 0.0

    # Final completion check
    assert final.progress == 1.0
    assert final.metrics["retrievals"] == 8.0, "All 8 containers in Ku 0503 layout must be retrieved"
    assert 0 <= final.metric < 1000.0, f"Relocations {final.metric} must be non-negative and non-penalized"
    assert final.best_metric == final.metric


# ==============================================================================
# Scenario 2: Synthetic Procedural Matrix (Diverse Geometries & Seeds)
# ==============================================================================

@pytest.mark.parametrize("stacks,containers,seed", [
    (3, 6, 0),
    (3, 9, 1),
    (3, 12, 5),
    (3, 15, 5),   # Previously vulnerable to capacity starvation
    (4, 12, 42),
    (5, 15, 99),
    (5, 25, 7),
    (6, 18, 123),
])
def test_synthetic_procedural_matrix(stacks, containers, seed):
    """Verifies that procedural layouts across different geometries never deadlock."""
    cfg = ProblemConfig(
        num_rows=stacks,
        num_containers=containers,
        max_tiers=3,
        seed=seed,
    )
    env = CRP_Online_2026(config=cfg)
    algo = GPOnline2026Algorithm(AlgorithmConfig(report_interval=5))

    q: mp.Queue = mp.Queue()
    ev = mp.Event()
    algo.train(lambda: env, q, ev)

    records = []
    while not q.empty():
        records.append(q.get())

    assert len(records) >= 1
    final = records[-1]

    assert final.metrics["retrievals"] == float(containers), f"All {containers} containers must be retrieved"
    assert final.metric < 10000.0, f"Run deadlocked with penalty: {final.metrics}"


# ==============================================================================
# Scenario 3: Boundary & Edge Cases
# ==============================================================================

def test_edge_case_already_sorted():
    """An already sorted yard must require exactly 0 relocations."""
    # Stacks: 0: [3, 2, 1], 1: [6, 5, 4], 2: []
    # Targets are retrieved in order 1 -> 2 -> 3 -> 4 -> 5 -> 6 with zero blockers!
    class SortedCRP(CRP_Online_2026):
        def _build_episode(self):
            super()._build_episode()
            from crp_online.core.instance import ProblemInstance
            import crp_online.core.environment as isolated_env
            inst = ProblemInstance(
                name="sorted",
                num_stacks=3,
                num_containers=6,
                max_stack_height=5,
                initial_height=3,
                initial_layout=((3, 2, 1), (6, 5, 4), ()),
            )
            self._sim = isolated_env.Environment(inst, store_moves=True)

    env = SortedCRP()
    obs, info = env.reset()

    # The auto_retrieve loop in reset() should automatically retrieve all containers
    assert env._sim.is_empty(), "All containers should be retrieved without any relocation"
    metrics = env.get_metrics()
    assert metrics["relocations"] == 0.0
    assert metrics["retrievals"] == 6.0


def test_edge_case_single_container():
    """A yard with a single container must complete in 0 relocations and 1 retrieval."""
    cfg = ProblemConfig(num_rows=3, num_containers=1, max_tiers=3, seed=0)
    env = CRP_Online_2026(config=cfg)
    obs, info = env.reset()

    assert env._sim.is_empty()
    metrics = env.get_metrics()
    assert metrics["relocations"] == 0.0
    assert metrics["retrievals"] == 1.0


def test_edge_case_deep_inversion():
    """Target 1 placed under 4 blocking containers; must clear all blockers and retrieve 1."""
    class DeepInversionCRP(CRP_Online_2026):
        def _build_episode(self):
            super()._build_episode()
            from crp_online.core.instance import ProblemInstance
            import crp_online.core.environment as isolated_env
            inst = ProblemInstance(
                name="deep_inversion",
                num_stacks=3,
                num_containers=5,
                max_stack_height=8,
                initial_height=5,
                initial_layout=((1, 5, 4, 3, 2), (), ()),
            )
            self._sim = isolated_env.Environment(inst, store_moves=True)

    env = DeepInversionCRP()
    algo = TLP2026Algorithm()

    q: mp.Queue = mp.Queue()
    ev = mp.Event()
    algo.train(lambda: env, q, ev)

    records = [q.get() for _ in range(q.qsize())]
    final = records[-1]
    assert final.metrics["retrievals"] == 5.0
    assert final.metrics["relocations"] == 6.0  # 4 initial relocations + 2 cascade relocations


# ==============================================================================
# Scenario 4: Gymnasium API Output Contract & Observation Bounds
# ==============================================================================

def test_gymnasium_compliance():
    """Validates the standard Gymnasium API 5-tuple contract, types, and bounds."""
    cfg = ProblemConfig(num_rows=4, num_containers=8, max_tiers=4, seed=42)
    env = CRP_Online_2026(config=cfg)

    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.dtype == np.float64
    assert "action_mask" in info
    assert info["action_mask"].shape == (4,)
    assert info["action_mask"].dtype == bool

    # Observation space bounds check
    assert env.observation_space.contains(obs), "Obs grid must conform to Box space bounds"

    # Action mask consistency: target stack must NOT be a valid destination
    target = env._sim.get_current_target()
    src = env._sim.get_stack_of_container(target)
    assert info["action_mask"][src] is False or info["action_mask"][src] == 0

    # Pick a valid destination according to mask
    valid_dests = np.where(info["action_mask"])[0]
    assert len(valid_dests) > 0
    action = int(valid_dests[0])

    step_obs, reward, terminated, truncated, metrics = env.step(action)
    assert isinstance(step_obs, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(metrics, dict)
    assert "relocations" in metrics
    assert "retrievals" in metrics
    assert "crane_time_s" in metrics


# ==============================================================================
# Scenario 5: Action Masking & Illegal Action Penalty (+10,000)
# ==============================================================================

@pytest.mark.parametrize("illegal_action", [
    -1,   # Negative out-of-bounds
    999,  # High out-of-bounds
])
def test_illegal_out_of_bounds_action(illegal_action):
    """Out-of-bounds action must trigger +10,000 penalty and terminate immediately."""
    cfg = ProblemConfig(num_rows=3, num_containers=6, max_tiers=3, seed=0)
    env = CRP_Online_2026(config=cfg)
    env.reset()

    obs, reward, terminated, truncated, metrics = env.step(illegal_action)
    assert terminated is True
    assert reward == -10000.0
    assert metrics["relocations"] >= 10000.0


def test_illegal_source_stack_destination():
    """Moving a container onto its own source stack must trigger +10,000 penalty."""
    cfg = ProblemConfig(num_rows=3, num_containers=6, max_tiers=3, seed=0)
    env = CRP_Online_2026(config=cfg)
    env.reset()

    target = env._sim.get_current_target()
    src = env._sim.get_stack_of_container(target)

    obs, reward, terminated, truncated, metrics = env.step(src)
    assert terminated is True
    assert reward == -10000.0
    assert metrics["relocations"] >= 10000.0


# ==============================================================================
# Scenario 6: CRISP Multiprocessing Integration & React Snapshot Format
# ==============================================================================

class _TestProblemFactory:
    def __init__(self, cfg):
        self.cfg = cfg
    def __call__(self):
        return CRP_Online_2026(config=self.cfg)


def test_multiprocessing_progress_and_react_snapshot():
    """Verifies that progress records stream over multiprocessing queues and format snapshots."""
    cfg = ProblemConfig(num_rows=3, num_containers=6, max_tiers=3, seed=10)
    algo = GPOnline2026Algorithm(AlgorithmConfig(report_interval=1))

    q: mp.Queue = mp.Queue()
    ev = mp.Event()

    factory = _TestProblemFactory(cfg)
    # Run in a separate multiprocessing process like CRISP JobManager
    proc = mp.Process(target=algo.train, args=(factory, q, ev))
    proc.start()
    proc.join(timeout=10)
    assert not proc.is_alive()

    records = []
    while not q.empty():
        records.append(q.get())

    assert len(records) > 0
    first = records[0]
    assert first.step == 0
    assert first.metric == 0.0

    last = records[-1]
    assert last.progress == 1.0

    # Verify React visualizer snapshot format: {(bay, row): [containers]}
    snapshot = last.yard_snapshot
    assert snapshot is not None
    assert "yard" in snapshot
    assert "metrics" in snapshot
    assert "history" in snapshot

    # Yard keys must be (bay, row) tuples
    for pos_key, stack_containers in snapshot["yard"].items():
        assert isinstance(pos_key, tuple)
        assert len(pos_key) == 2
        assert isinstance(stack_containers, list)


# ==============================================================================
# Scenario 7: Custom GP Expression Evaluation & Catalogs
# ==============================================================================

def test_custom_gp_expression_string():
    """Verifies evaluating a custom user prefix expression passed via extra."""
    custom_expr = "/ / - DIFF RI / / SH WL / DSM DUR + CUR DUR"
    cfg_algo = AlgorithmConfig(extra={"expression": custom_expr})
    algo = GPOnline2026Algorithm(config=cfg_algo)

    cfg_prob = ProblemConfig(num_rows=3, num_containers=6, max_tiers=3, seed=2)
    env = CRP_Online_2026(config=cfg_prob)

    q: mp.Queue = mp.Queue()
    ev = mp.Event()
    algo.train(lambda: env, q, ev)

    records = [q.get() for _ in range(q.qsize())]
    final = records[-1]
    assert final.metrics["retrievals"] == 6.0
    assert final.metric < 1000.0


def test_rule_index_catalog_switching():
    """Verifies that different rule indices from expressions.txt compile and run."""
    for rule_idx in [0, 1, 2]:
        algo = GPOnline2026Algorithm(config=AlgorithmConfig(extra={"rule_index": rule_idx}))
        assert algo._compiled_expr is not None


# ==============================================================================
# Scenario 8: Information Barrier Metamorphic Invariance
# ==============================================================================

def test_information_barrier_metamorphic_invariance():
    """
    Metamorphic test: permuting the true IDs of unknown containers beyond lookahead
    must produce strictly identical decisions by the heuristic.
    """
    from crp_online.core.instance import ProblemInstance
    from crp_online.core.environment import Environment
    from crp_online.core.observation import ObservationState
    from crp_online.heuristics.manual import select_minmax_destination

    # 3 stacks: Stack 0 has target 1 at bottom, blocked by container 10.
    # Stack 1 has unknown container 8.
    # Stack 2 has unknown container 9.
    # Lookahead horizon: 0 (only target 1 is known; 8, 9, 10 are unknown).
    layout_A = ((10, 1), (8,), (9,))
    layout_B = ((10, 1), (9,), (8,))  # 8 and 9 swapped

    inst_A = ProblemInstance("A", 3, 4, 4, 2, layout_A)
    inst_B = ProblemInstance("B", 3, 4, 4, 2, layout_B)

    env_A = Environment(inst_A, store_moves=False)
    env_B = Environment(inst_B, store_moves=False)

    obs_A = ObservationState(env_A, look_ahead_level=0.0)
    obs_B = ObservationState(env_B, look_ahead_level=0.0)

    # Decisions must be identical regardless of permutation of unknown containers
    dest_A = select_minmax_destination(obs_A, env_A, source_stack=0, moving_val=float(env_A.get_top_container(0)))
    dest_B = select_minmax_destination(obs_B, env_B, source_stack=0, moving_val=float(env_B.get_top_container(0)))

    assert dest_A == dest_B, f"Decision leaked hidden information: {dest_A} != {dest_B}"
