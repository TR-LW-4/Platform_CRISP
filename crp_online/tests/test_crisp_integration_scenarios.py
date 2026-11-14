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
    PU12026Algorithm,
    PU22026Algorithm,
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
    PU12026Algorithm,
    PU22026Algorithm,
]


def _drain_queue(q: mp.Queue) -> List[ProgressRecord]:
    """Reliably drains multiprocessing queue across all OS platforms."""
    records: List[ProgressRecord] = []
    while True:
        try:
            r = q.get(timeout=0.5)
            records.append(r)
            if r.progress >= 1.0:
                break
        except Exception:
            break
    return records


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

    records = _drain_queue(q)

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

    records = _drain_queue(q)

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

    records = _drain_queue(q)

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

    records = _drain_queue(q)

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


# ==============================================================================
# Scenario 9: Rolling Horizon Progression & Offline Equivalence (α = 1.0)
# ==============================================================================

def test_rolling_horizon_offline_equivalence():
    """When lookahead α = 1.0, all containers are known and equal to true IDs (offline problem)."""
    cfg = ProblemConfig(
        extra={
            "info_structure": "rolling_horizon",
            "lookahead_horizon": 1.0,
            "layout_file_path": str(BENCHMARK_FILE),
        }
    )
    env = CRP_Online_2026(config=cfg)
    grid, _ = env.reset()

    # In data3-3-1, 9 containers total: all containers in the observation must have obs_value == true_id
    obs = env.get_observation()
    for s in range(env._sim.num_stacks):
        for h in range(obs.get_stack_height(s)):
            container = obs.get_container(s, h)
            assert container.is_known, "All containers must be known when α=1.0"
            assert container.obs_value == float(container.true_id)


def test_rolling_horizon_reveals_as_targets_retrieved():
    """Tests that advancing targets in rolling horizon reveals previously hidden containers."""
    cfg = ProblemConfig(
        extra={
            "info_structure": "rolling_horizon",
            "lookahead_horizon": 0.0,
            "layout_file_path": str(BENCHMARK_FILE),
        }
    )
    env = CRP_Online_2026(config=cfg)
    grid, _ = env.reset()

    # In data3-3-1, container 1 was at the top, so _auto_retrieve() immediately retrieved it!
    # Target is now 2. At α = 0.0, only the active target (2) is known; all other containers are unknown.
    obs_initial = env.get_observation()
    active_target = obs_initial.current_target
    assert active_target == 2
    for s in range(env._sim.num_stacks):
        for h in range(obs_initial.get_stack_height(s)):
            container = obs_initial.get_container(s, h)
            if container.true_id == active_target:
                assert container.is_known
            else:
                assert not container.is_known


# ==============================================================================
# Scenario 10: Container Groups (Normal vs Relaxed, Group Count vs Group Size)
# ==============================================================================

def test_container_groups_normal_vs_relaxed():
    """Verifies that normal groups mask all non-targets, while relaxed reveals current group."""
    cfg_normal = ProblemConfig(
        extra={
            "info_structure": "container_groups",
            "group_mode": "normal",
            "group_by": "group_count",
            "group_count": 2,
            "layout_file_path": str(BENCHMARK_FILE),
        }
    )
    env_normal = CRP_Online_2026(config=cfg_normal)
    env_normal.reset()
    obs_norm = env_normal.get_observation()

    # Normal mode: only target 1 is known
    known_norm = sum(
        1 for s in range(env_normal._sim.num_stacks)
        for h in range(obs_norm.get_stack_height(s))
        if obs_norm.get_container(s, h).is_known
    )
    assert known_norm == 1, "Normal group mode should only have target container known"

    cfg_relaxed = ProblemConfig(
        extra={
            "info_structure": "container_groups",
            "group_mode": "relaxed",
            "group_by": "group_count",
            "group_count": 2,
            "layout_file_path": str(BENCHMARK_FILE),
        }
    )
    env_relaxed = CRP_Online_2026(config=cfg_relaxed)
    env_relaxed.reset()
    obs_rel = env_relaxed.get_observation()

    # Relaxed mode: all containers belonging to target's group are known
    known_rel = sum(
        1 for s in range(env_relaxed._sim.num_stacks)
        for h in range(obs_rel.get_stack_height(s))
        if obs_rel.get_container(s, h).is_known
    )
    assert known_rel > 1, f"Relaxed group mode must reveal all containers in target's group, got {known_rel}"


# ==============================================================================
# Scenario 11: Restricted vs Unrestricted Relocation Schemes
# ==============================================================================

def test_restricted_vs_unrestricted_simulation():
    """Verifies that relocation_scheme correctly configures environment execution."""
    cfg_rest = ProblemConfig(
        extra={
            "relocation_scheme": "restricted",
            "layout_file_path": str(BENCHMARK_FILE),
        }
    )
    env_rest = CRP_Online_2026(config=cfg_rest)
    assert env_rest.relocation_scheme == "restricted"

    cfg_unrest = ProblemConfig(
        extra={
            "relocation_scheme": "unrestricted",
            "layout_file_path": str(BENCHMARK_FILE),
        }
    )
    env_unrest = CRP_Online_2026(config=cfg_unrest)
    assert env_unrest.relocation_scheme == "unrestricted"


# ==============================================================================
# Scenario 12: Clean Parameter Schema Verification
# ==============================================================================

def test_clean_config_schema_no_crane_bleed():
    """Verifies that config_schema() excludes non-relevant crane/attribute keys."""
    schema = CRP_Online_2026.config_schema()

    # Must contain paper-specific parameters
    assert "relocation_scheme" in schema
    assert "info_structure" in schema
    assert "lookahead_horizon" in schema
    assert "unknown_treatment" in schema
    assert "group_mode" in schema
    assert "group_by" in schema
    assert "group_count" in schema
    assert "group_size" in schema

    # Must NOT contain irrelevant crane or multi-bay yard toggles
    forbidden = ["num_cranes", "crane_speed", "enable_weight", "enable_size", "enable_type", "num_bays"]
    for k in forbidden:
        assert k not in schema, f"Forbidden key {k} must not appear in CRP_Online_2026 config schema!"

