from pathlib import Path
import pytest
from crp_online.io.caserta_loader import parse_caserta_file
from crp_online.core.environment import Environment
from crp_online.core.observation import ObservationState


DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "caserta"


def test_full_information_observation():
    inst = parse_caserta_file(DATA_DIR / "data3-3-1.dat")  # N=9, S=3, h=3
    env = Environment(inst)
    obs = ObservationState(env, look_ahead_level=1.0)

    # All containers should be known under LA=1.0
    for s in range(env.num_stacks):
        assert obs.get_number_unknown(s) == 0
        for h in range(obs.get_stack_height(s)):
            mc = obs.get_container(s, h)
            assert mc.is_known
            assert mc.obs_value == float(mc.true_id)


def test_limited_horizon_observation():
    inst = parse_caserta_file(DATA_DIR / "data3-3-1.dat")  # N=9, target=1
    env = Environment(inst)
    # alpha = 0.5 => max_known = 1 + int(0.5 * 9) = 1 + 4 = 5
    obs = ObservationState(env, look_ahead_level=0.5, horizon_strategy="ignore")

    for s in range(env.num_stacks):
        for h in range(obs.get_stack_height(s)):
            mc = obs.get_container(s, h)
            if mc.true_id <= 5:
                assert mc.is_known
                assert mc.obs_value == float(mc.true_id)
            else:
                assert not mc.is_known
                assert mc.obs_value == -1.0


def test_container_groups_monotonicity():
    inst = parse_caserta_file(DATA_DIR / "data4-4-1.dat")  # N=16, S=4, h=4
    env = Environment(inst)
    # GC = 4 groups => group size = 16 / 4 = 4
    obs = ObservationState(
        env,
        use_container_groups=True,
        num_groups=4,
        known_within_group=False,
    )

    # Verify group proxy step = GS + 1 = 5
    # Group 1 (1..4) -> 5
    # Group 2 (5..8) -> 10
    # Group 3 (9..12) -> 15
    # Group 4 (13..16) -> 20
    # Strict monotonicity: G1 < G2 < G3 < G4 proxies
    for s in range(env.num_stacks):
        for h in range(obs.get_stack_height(s)):
            mc = obs.get_container(s, h)
            if mc.true_id == 1:
                assert mc.is_known
                assert mc.obs_value == 1.0
            else:
                assert not mc.is_known
                expected_proxy = ((mc.true_id // 5) + 1) * 5
                assert mc.obs_value == float(expected_proxy)
