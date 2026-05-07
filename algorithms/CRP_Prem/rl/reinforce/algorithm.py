"""
REINFORCE with greedy baseline (policy gradient) for container relocation.

Network architecture is defined in network.py (PolicyNet).
"""

from __future__ import annotations

import multiprocessing as mp
from typing import Callable, Dict, List, Optional

import numpy as np

from core.base_algorithm import BaseAlgorithm, AlgorithmConfig


class REINFORCE(BaseAlgorithm):

    name        = "REINFORCE"
    category    = "RL"
    description = ("[general]  "
                   "REINFORCE policy gradient with greedy baseline. "
                   "Works with any problem that exposes an action mask.")
    compatible_problems = [
        "CRP-R",
        "CRP-Time",
        "BRP-NonFixed",
        "CRP-Prem",
        "CRP-Stow",
        "CRP-Stoch",
        "CRP-U",
        "CRP-D",
    ]

    def __init__(self, config: Optional[AlgorithmConfig] = None):
        super().__init__(config)

    def train(
        self,
        problem_factory: Callable,
        result_queue:    mp.Queue,
        stop_event:      mp.Event,
    ) -> None:
        try:
            import torch
            import torch.optim as optim
        except ImportError:
            raise RuntimeError("PyTorch is required for RL algorithms. pip install torch")

        from .network import PolicyNet

        cfg = self.config
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        env = problem_factory()
        obs, info = env.reset()
        obs_dim   = int(np.prod(obs.shape))
        n_actions = env.action_space.n

        policy    = PolicyNet(obs_dim, n_actions, cfg.hidden_dim)
        optimizer = optim.Adam(policy.parameters(), lr=cfg.learning_rate)

        baseline        = 0.0
        baseline_alpha  = 0.05
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
            log_probs: List = []
            rewards:   List[float] = []
            actions:   List[int]   = []

            while not done:
                import torch
                obs_tensor  = torch.FloatTensor(obs_t).unsqueeze(0)
                mask_np     = info_t.get("action_mask", None)
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
            import torch
            G       = 0.0
            returns: List[float] = []
            for r in reversed(rewards):
                G = r + cfg.gamma * G
                returns.insert(0, G)
            episode_return = returns[0] if returns else 0.0

            baseline += baseline_alpha * (episode_return - baseline)
            if episode_return > best_return:
                best_return         = episode_return
                best_solution       = actions[:]
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

        self._policy = policy

    def get_best_solution(self) -> Optional[List[int]]:
        return self._best_solution

    step_label = "Episode"

    @classmethod
    def config_schema(cls) -> Dict:
        return {
            "num_episodes":  {"type": "int",   "default": 500,   "min": 10,   "max": 10_000,
                              "label": "Episodes"},
            "report_every":  {"type": "int",   "default": 10,    "min": 1,    "max": 200,
                              "label": "Report every N episodes"},
            "learning_rate": {"type": "float", "default": 3e-4,  "min": 1e-5, "max": 1e-2,
                              "label": "Learning rate"},
            "gamma":         {"type": "float", "default": 0.99,  "min": 0.8,  "max": 1.0,
                              "label": "Discount factor γ"},
            "hidden_dim":    {"type": "int",   "default": 128,   "min": 32,   "max": 512,
                              "label": "Hidden layer size"},
            "seed":          {"type": "int",   "default": 0,     "min": 0,    "max": 9999,
                              "label": "Random seed"},
        }
