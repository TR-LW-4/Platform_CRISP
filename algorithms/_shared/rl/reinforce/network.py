"""
REINFORCE policy network.

Defined as a standalone module so it can be imported, modified,
and tested independently of the training loop.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PolicyNet(nn.Module):
    """
    MLP policy network for discrete action spaces with action masking.

    Architecture
    ------------
    Linear(obs_dim, hidden) → ReLU → Linear(hidden, hidden) → ReLU
    → Linear(hidden, n_actions) → Categorical distribution
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(
        self,
        x:    torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.distributions.Categorical:
        """
        Parameters
        ----------
        x    : observation tensor  [..., obs_dim]
        mask : boolean action mask [..., n_actions]; invalid actions masked out

        Returns
        -------
        Categorical distribution over valid actions
        """
        logits = self.net(x)
        if mask is not None:
            logits = logits + (mask.float() - 1) * 1e9
        return torch.distributions.Categorical(logits=logits)
