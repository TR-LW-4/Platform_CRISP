import pytest
from crp_online.core.instance import ProblemInstance
from crp_online.core.environment import Environment
from crp_online.core.observation import ObservationState
from crp_online.evaluator.prefix_evaluator import PrefixExpression
from crp_online.terminals.terminal_registry import evaluate_terminal


def test_figure_5_golden_walkthrough():
    """Verifies the golden walkthrough from Figure 5 (Page 21) of Đurasević et al. (2026).
    
    Bay: S = 5 stacks, H = 4 capacity.
    Layout (bottom to top):
      Stack 0: [2, 11, 7]
      Stack 1: [1, 3, 9] (target is 1, relocating container is 9)
      Stack 2: [6, 5]
      Stack 3: [8, 4, 10]
      Stack 4: [12, 13]
    """
    layout = (
        (2, 11, 7),
        (1, 3, 9),
        (6, 5),
        (8, 4, 10),
        (12, 13),
    )
    inst = ProblemInstance(
        name="figure_5_walkthrough",
        num_stacks=5,
        num_containers=13,
        max_stack_height=4,
        initial_height=3,
        initial_layout=layout,
    )
    env = Environment(inst)
    obs = ObservationState(env, look_ahead_level=1.0)

    # Relocating container is 9 from Stack 1
    source_stack = 1
    relocating_val = 9.0

    # Stack 0: AVG = (2 + 11 + 7) / 3 = 6.6667, RI = 2 (2, 7 < 9), EMP = 4 - 3 = 1
    assert abs(evaluate_terminal("AVG", obs, env, source_stack, 0, relocating_val) - 20.0 / 3.0) < 1e-6
    assert evaluate_terminal("RI", obs, env, source_stack, 0, relocating_val) == 2.0
    assert evaluate_terminal("EMP", obs, env, source_stack, 0, relocating_val) == 1.0

    # Stack 2: AVG = (6 + 5) / 2 = 5.50, RI = 2 (6, 5 < 9), EMP = 4 - 2 = 2
    assert abs(evaluate_terminal("AVG", obs, env, source_stack, 2, relocating_val) - 5.50) < 1e-6
    assert evaluate_terminal("RI", obs, env, source_stack, 2, relocating_val) == 2.0
    assert evaluate_terminal("EMP", obs, env, source_stack, 2, relocating_val) == 2.0

    # Stack 3: AVG = (8 + 4 + 10) / 3 = 7.3333, RI = 2 (8, 4 < 9), EMP = 4 - 3 = 1
    assert abs(evaluate_terminal("AVG", obs, env, source_stack, 3, relocating_val) - 22.0 / 3.0) < 1e-6
    assert evaluate_terminal("RI", obs, env, source_stack, 3, relocating_val) == 2.0
    assert evaluate_terminal("EMP", obs, env, source_stack, 3, relocating_val) == 1.0

    # Stack 4: AVG = (12 + 13) / 2 = 12.50, RI = 0, EMP = 4 - 2 = 2
    assert abs(evaluate_terminal("AVG", obs, env, source_stack, 4, relocating_val) - 12.50) < 1e-6
    assert evaluate_terminal("RI", obs, env, source_stack, 4, relocating_val) == 0.0
    assert evaluate_terminal("EMP", obs, env, source_stack, 4, relocating_val) == 2.0

    # Priority function from Figure 5: (+ (- AVG EMP) RI)
    # Stack 0: 6.67 - 1 + 2 = 7.67
    # Stack 2: 5.50 - 2 + 2 = 5.50
    # Stack 3: 7.33 - 1 + 2 = 8.33
    # Stack 4: 12.50 - 2 + 0 = 10.50
    expr = PrefixExpression("+ - AVG EMP RI")
    p0 = expr.evaluate(obs, env, source_stack, 0, relocating_val)
    p2 = expr.evaluate(obs, env, source_stack, 2, relocating_val)
    p3 = expr.evaluate(obs, env, source_stack, 3, relocating_val)
    p4 = expr.evaluate(obs, env, source_stack, 4, relocating_val)

    assert abs(p0 - 7.6666667) < 1e-5
    assert abs(p2 - 5.5000000) < 1e-5
    assert abs(p3 - 8.3333333) < 1e-5
    assert abs(p4 - 10.5000000) < 1e-5

    # Optimal candidate is Stack 2 with priority 5.50
    priorities = {0: p0, 2: p2, 3: p3, 4: p4}
    best_stack = min(priorities, key=priorities.get)
    assert best_stack == 2


def test_protected_division():
    inst = ProblemInstance("test", 3, 3, 5, 1, ((1,), (2,), (3,)))
    env = Environment(inst)
    obs = ObservationState(env)

    # Division by zero returns 1.0
    expr = PrefixExpression("/ 5.0 0.0")
    res = expr.evaluate(obs, env, 0, 1, 1.0)
    assert res == 1.0

    # Normal division
    expr2 = PrefixExpression("/ 10.0 2.0")
    assert expr2.evaluate(obs, env, 0, 1, 1.0) == 5.0
