"""
REINFORCE with Greedy Baseline (Policy Gradient).

Adapted from the existing spp-main_CSPP/cleanrl/train_RF.py.

Architecture
------------
PolicyNetwork: MLP → softmax over valid actions (action mask applied)
Baseline     : running mean of episode returns (greedy baseline)

Updates the policy with REINFORCE gradient:
  ∇ log π(a|s) · (G_t − baseline)
"""

from __future__ import annotations

import multiprocessing as mp
from typing import Callable, Dict, List, Optional

import numpy as np

from core.base_algorithm import BaseAlgorithm, AlgorithmConfig


class REINFORCE(BaseAlgorithm):

    name        = "REINFORCE"
    category    = "RL"
    description = ("REINFORCE policy gradient with greedy baseline. "
                   "Works with any problem that exposes an action mask.")
    compatible_problems: List[str] = []

    def __init__(self, config: Optional[AlgorithmConfig] = None):
        super().__init__(config)

    # ---------------------------------------------------------------- #
    # Train                                                              #
    # ---------------------------------------------------------------- #

    def train(
        self,
        problem_factory: Callable,
        result_queue:    mp.Queue,
        stop_event:      mp.Event,
    ) -> None:
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
        except ImportError:
            raise RuntimeError("PyTorch is required for RL algorithms. pip install torch")

        cfg = self.config
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        # ── Build environment ────────────────────────────────────── #
        env = problem_factory()
        obs, info = env.reset()
        obs_dim   = int(np.prod(obs.shape))
        n_actions = env.action_space.n

        # ── Policy network ───────────────────────────────────────── #
        hidden = cfg.hidden_dim

        class PolicyNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(obs_dim, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, n_actions),
                )

            def forward(self, x, mask=None):
                logits = self.net(x)
                if mask is not None:
                    logits = logits + (mask.float() - 1) * 1e9  # mask invalid
                return torch.distributions.Categorical(logits=logits)

        policy    = PolicyNet()
        optimizer = optim.Adam(policy.parameters(), lr=cfg.learning_rate)

        # ── Greedy baseline ──────────────────────────────────────── #
        baseline        = 0.0
        baseline_alpha  = 0.05
        # Support both schema key names
        num_iterations  = cfg.extra.get("num_episodes", cfg.max_iterations)
        cfg.report_interval = cfg.extra.get("report_every", cfg.report_interval)
        best_return     = float("-inf")
        best_solution:  Optional[List[int]] = None

        for iteration in range(1, num_iterations + 1):
            if stop_event.is_set():
                break

            # ── Collect one episode ──────────────────────────────── #
            obs_t, info_t = env.reset()
            done   = False
            log_probs: List[torch.Tensor] = []
            rewards:   List[float]        = []
            actions:   List[int]          = []

            while not done:
                obs_tensor = torch.FloatTensor(obs_t).unsqueeze(0)
                mask_np    = info_t.get("action_mask", None)
                mask_tensor = (
                    torch.BoolTensor(mask_np).unsqueeze(0)
                    if mask_np is not None else None
                )

                dist   = policy(obs_tensor, mask=mask_tensor)
                action = dist.sample()
                lp     = dist.log_prob(action)

                obs_t, r, done, _, info_t = env.step(int(action.item()))
                log_probs.append(lp)
                rewards.append(r)
                actions.append(int(action.item()))

            # ── Compute returns ──────────────────────────────────── #
            G     = 0.0
            returns: List[float] = []
            for r in reversed(rewards):
                G = r + cfg.gamma * G
                returns.insert(0, G)
            episode_return = returns[0] if returns else 0.0

            # Update greedy baseline
            baseline += baseline_alpha * (episode_return - baseline)
            if episode_return > best_return:
                best_return    = episode_return
                best_solution  = actions[:]
                self._best_solution = best_solution

            # ── Policy gradient update ───────────────────────────── #
            loss = torch.tensor(0.0, requires_grad=True)
            for lp, G_t in zip(log_probs, returns):
                loss = loss - lp * (G_t - baseline)
            loss = loss / max(len(log_probs), 1)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()

            # ── Report ───────────────────────────────────────────── #
            if iteration % cfg.report_interval == 0 or iteration == num_iterations:
                metrics = env.get_metrics()
                metrics["episode_return"] = float(episode_return)
                metrics["baseline"]       = float(baseline)
                metrics["loss"]           = float(loss.item())

                self._push(
                    result_queue,
                    step     = iteration,
                    metric   = -episode_return,
                    metrics  = metrics,
                    progress = iteration / num_iterations,
                    snapshot = env.get_state_snapshot(),
                )

        # Final policy save (in-memory)
        self._policy = policy

    def get_best_solution(self) -> Optional[List[int]]:
        return self._best_solution

    # Human-readable label shown in the GUI progress bar
    step_label = "Episode"

    @classmethod
    def config_schema(cls) -> Dict:
        return {
            "num_episodes":  {"type": "int",   "default": 500,   "min": 10,   "max": 10_000,
                              "label": "Episodes (num_episodes)",
                              "help": "Total number of full episodes to train."},
            "report_every":  {"type": "int",   "default": 10,    "min": 1,    "max": 200,
                              "label": "Report every N episodes",
                              "help": "How often to push progress to the GUI."},
            "learning_rate": {"type": "float", "default": 3e-4,  "min": 1e-5, "max": 1e-2,
                              "label": "Learning rate"},
            "gamma":         {"type": "float", "default": 0.99,  "min": 0.8,  "max": 1.0,
                              "label": "Discount factor γ"},
            "hidden_dim":    {"type": "int",   "default": 128,   "min": 32,   "max": 512,
                              "label": "Hidden layer size"},
            "seed":          {"type": "int",   "default": 0,     "min": 0,    "max": 9999,
                              "label": "Random seed"},
        }
