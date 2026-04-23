"""
PPO Actor-Critic network.

Defined as a standalone module so it can be imported, instantiated,
and tested independently of the training loop.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    """
    Shared-trunk actor-critic for discrete action spaces.

    Architecture
    ------------
    shared → Linear(obs_dim, hidden) → ReLU → Linear(hidden, hidden) → ReLU
    actor  → Linear(hidden, n_act)   (logits)
    critic → Linear(hidden, 1)       (state value)
    """

    def __init__(self, obs_dim: int, n_act: int, hidden: int = 128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
        )
        self.actor  = nn.Linear(hidden, n_act)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor):
        h = self.shared(x)
        return self.actor(h), self.critic(h)

    def get_dist(
        self,
        x:    torch.Tensor,
        mask: torch.Tensor | None = None,
    ):
        """
        Return (Categorical distribution, value estimate).

        Parameters
        ----------
        x    : observation tensor  [..., obs_dim]
        mask : boolean action mask [..., n_act]; invalid actions are masked out
        """
        logits, value = self(x)
        if mask is not None:
            logits = logits + (mask.float() - 1) * 1e9
        dist = torch.distributions.Categorical(logits=logits)
        return dist, value.squeeze(-1)
