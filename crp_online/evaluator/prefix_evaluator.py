import math
from typing import List, Sequence, Union
from ..core.environment import Environment
from ..core.observation import ObservationState
from ..terminals.terminal_registry import evaluate_terminal, TERMINAL_REGISTRY
from ..config.defaults import DIVISION_EPSILON


class PrefixExpression:
    """Compiled prefix (Polish notation) expression tree for container relocation priority."""

    def __init__(self, expression_str: str) -> None:
        self.raw_expression: str = expression_str.strip()
        self.tokens: List[str] = self.raw_expression.split()
        if not self.tokens:
            raise ValueError("Cannot parse empty expression!")

    def evaluate(
        self,
        obs: ObservationState,
        env: Environment,
        source_stack: int,
        candidate_stack: int,
        moving_val: float,
    ) -> float:
        """Evaluates the prefix expression for a specific candidate destination stack."""
        pos = [0]
        result = self._eval_recursive(pos, obs, env, source_stack, candidate_stack, moving_val)
        return result

    def _eval_recursive(
        self,
        pos: List[int],
        obs: ObservationState,
        env: Environment,
        s_src: int,
        s_cand: int,
        moving_val: float,
    ) -> float:
        if pos[0] >= len(self.tokens):
            return 0.0

        tok = self.tokens[pos[0]]
        pos[0] += 1

        # Binary Operators
        if tok == "+":
            left = self._eval_recursive(pos, obs, env, s_src, s_cand, moving_val)
            right = self._eval_recursive(pos, obs, env, s_src, s_cand, moving_val)
            return left + right

        if tok == "-":
            left = self._eval_recursive(pos, obs, env, s_src, s_cand, moving_val)
            right = self._eval_recursive(pos, obs, env, s_src, s_cand, moving_val)
            return left - right

        if tok == "*":
            left = self._eval_recursive(pos, obs, env, s_src, s_cand, moving_val)
            right = self._eval_recursive(pos, obs, env, s_src, s_cand, moving_val)
            return left * right

        if tok == "/":
            left = self._eval_recursive(pos, obs, env, s_src, s_cand, moving_val)
            right = self._eval_recursive(pos, obs, env, s_src, s_cand, moving_val)
            if abs(right) < DIVISION_EPSILON:
                return 1.0  # Protected division contract
            return left / right

        # Extended functions (supported for general GP compatibility)
        if tok == "min":
            left = self._eval_recursive(pos, obs, env, s_src, s_cand, moving_val)
            right = self._eval_recursive(pos, obs, env, s_src, s_cand, moving_val)
            return min(left, right)

        if tok == "max":
            left = self._eval_recursive(pos, obs, env, s_src, s_cand, moving_val)
            right = self._eval_recursive(pos, obs, env, s_src, s_cand, moving_val)
            return max(left, right)

        if tok == "avg":
            left = self._eval_recursive(pos, obs, env, s_src, s_cand, moving_val)
            right = self._eval_recursive(pos, obs, env, s_src, s_cand, moving_val)
            return (left + right) / 2.0

        if tok == "pos":
            val = self._eval_recursive(pos, obs, env, s_src, s_cand, moving_val)
            return max(0.0, val)

        if tok == "neg":
            val = self._eval_recursive(pos, obs, env, s_src, s_cand, moving_val)
            return -val

        if tok == "sqrt":
            val = self._eval_recursive(pos, obs, env, s_src, s_cand, moving_val)
            return math.sqrt(val) if val > 0.0 else 0.0

        # Terminals
        if tok in TERMINAL_REGISTRY:
            return evaluate_terminal(tok, obs, env, s_src, s_cand, moving_val)

        # Numeric constants (e.g. "c_1.5", "1.0", "5")
        if tok.startswith("c_"):
            return float(tok[2:])

        try:
            return float(tok)
        except ValueError:
            raise ValueError(f"Unrecognized expression token: '{tok}'")

    def __repr__(self) -> str:
        return f"PrefixExpression('{self.raw_expression}')"
