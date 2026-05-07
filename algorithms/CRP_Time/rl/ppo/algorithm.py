"""
PPO (Proximal Policy Optimization) for container relocation problems.

Based on CleanRL's PPO implementation, adapted to the platform interface.
Network architecture is defined in network.py (ActorCritic).
"""

from __future__ import annotations

import multiprocessing as mp
from typing import Callable, Dict, List, Optional

import numpy as np

from core.base_algorithm import BaseAlgorithm, AlgorithmConfig


class PPO(BaseAlgorithm):

    name        = "PPO"
    category    = "RL"
    description = ("[general]  "
                   "Proximal Policy Optimization (clipped objective). "
                   "Actor-critic architecture with GAE advantage estimation.")
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
            import torch.nn as nn
            import torch.optim as optim
        except ImportError:
            raise RuntimeError("PyTorch is required. pip install torch")

        from .network import ActorCritic

        cfg = self.config
        torch.manual_seed(cfg.seed)

        env     = problem_factory()
        obs, _  = env.reset()
        obs_dim = int(np.prod(obs.shape))
        n_act   = env.action_space.n
        hidden  = cfg.hidden_dim

        ac        = ActorCritic(obs_dim, n_act, hidden)
        optimizer = optim.Adam(ac.parameters(), lr=cfg.learning_rate, eps=1e-5)

        num_steps    = cfg.num_steps
        num_iters    = cfg.extra.get("num_updates", cfg.max_iterations)
        cfg.report_interval = cfg.extra.get("report_every", cfg.report_interval)
        gamma        = cfg.gamma
        gae_lambda   = cfg.extra.get("gae_lambda", 0.95)
        clip_coef    = cfg.extra.get("clip_coef", 0.2)
        vf_coef      = cfg.extra.get("vf_coef", 0.5)
        ent_coef     = cfg.extra.get("ent_coef", 0.01)
        n_epochs     = cfg.extra.get("n_epochs", 4)
        minibatch    = cfg.batch_size

        obs_buf  = torch.zeros(num_steps, obs_dim)
        act_buf  = torch.zeros(num_steps, dtype=torch.long)
        rew_buf  = torch.zeros(num_steps)
        done_buf = torch.zeros(num_steps)
        logp_buf = torch.zeros(num_steps)
        val_buf  = torch.zeros(num_steps)
        mask_buf = torch.zeros(num_steps, n_act, dtype=torch.bool)

        global_step = 0
        obs_t, info = env.reset()
        done_t      = False
        best_ret    = float("-inf")
        ep_return   = 0.0
        ep_returns: List[float] = []

        for iteration in range(1, num_iters + 1):
            if stop_event.is_set():
                break

            # ── Rollout ──────────────────────────────────────────── #
            for step in range(num_steps):
                global_step += 1
                ot  = torch.FloatTensor(obs_t).unsqueeze(0)
                mk  = info.get("action_mask", None)
                mkt = torch.BoolTensor(mk).unsqueeze(0) if mk is not None else None

                with torch.no_grad():
                    dist, val = ac.get_dist(ot, mkt)
                    act = dist.sample()
                    lp  = dist.log_prob(act)

                obs_buf[step]  = ot.squeeze(0)
                act_buf[step]  = act.item()
                rew_buf[step]  = 0.0
                done_buf[step] = float(done_t)
                logp_buf[step] = lp.item()
                val_buf[step]  = val.item()
                if mk is not None:
                    mask_buf[step] = torch.BoolTensor(mk)

                obs_t, r, done_t, _, info = env.step(int(act.item()))
                rew_buf[step] = r
                ep_return    += r
                if done_t:
                    ep_returns.append(ep_return)
                    if ep_return > best_ret:
                        best_ret = ep_return
                    ep_return = 0.0
                    obs_t, info = env.reset()

            # ── GAE ──────────────────────────────────────────────── #
            with torch.no_grad():
                ot_next = torch.FloatTensor(obs_t).unsqueeze(0)
                _, next_val = ac.get_dist(ot_next)
                next_val = next_val.item()

            adv_buf  = torch.zeros(num_steps)
            last_gae = 0.0
            for t in reversed(range(num_steps)):
                nv       = next_val if t == num_steps - 1 else val_buf[t + 1].item()
                nd       = 1.0 - done_buf[t].item()
                delta    = rew_buf[t].item() + gamma * nv * nd - val_buf[t].item()
                last_gae = delta + gamma * gae_lambda * nd * last_gae
                adv_buf[t] = last_gae
            ret_buf = adv_buf + val_buf

            # ── PPO update ───────────────────────────────────────── #
            idx = np.arange(num_steps)
            for _ in range(n_epochs):
                np.random.shuffle(idx)
                for start in range(0, num_steps, minibatch):
                    mb      = idx[start: start + minibatch]
                    o_mb    = obs_buf[mb]
                    a_mb    = act_buf[mb]
                    lp_mb   = logp_buf[mb]
                    adv_mb  = adv_buf[mb]
                    ret_mb  = ret_buf[mb]
                    mk_mb   = mask_buf[mb]

                    adv_mb = (adv_mb - adv_mb.mean()) / (adv_mb.std() + 1e-8)

                    dist_mb, val_mb = ac.get_dist(o_mb, mk_mb)
                    new_lp   = dist_mb.log_prob(a_mb)
                    entropy  = dist_mb.entropy().mean()
                    ratio    = (new_lp - lp_mb).exp()

                    pg_loss = torch.max(
                        -adv_mb * ratio,
                        -adv_mb * ratio.clamp(1 - clip_coef, 1 + clip_coef),
                    ).mean()
                    vf_loss = 0.5 * (val_mb - ret_mb).pow(2).mean()
                    loss    = pg_loss + vf_coef * vf_loss - ent_coef * entropy

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(ac.parameters(), 0.5)
                    optimizer.step()

            # ── Report ───────────────────────────────────────────── #
            if iteration % cfg.report_interval == 0:
                mean_ret = float(np.mean(ep_returns[-10:])) if ep_returns else 0.0
                self._push(
                    result_queue,
                    step     = global_step,
                    metric   = -mean_ret,
                    metrics  = {
                        "mean_return": mean_ret,
                        "best_return": best_ret,
                        "policy_loss": float(pg_loss.item()),
                        "value_loss":  float(vf_loss.item()),
                    },
                    progress = iteration / num_iters,
                    snapshot = env.get_state_snapshot(),
                )

        self._ac = ac

    def get_best_solution(self) -> Optional[List[int]]:
        return self._best_solution

    step_label = "Update"

    @classmethod
    def config_schema(cls) -> Dict:
        return {
            "num_updates":   {"type": "int",   "default": 500,    "min": 10,   "max": 10_000,
                              "label": "Policy updates"},
            "report_every":  {"type": "int",   "default": 10,     "min": 1,    "max": 100,
                              "label": "Report every N updates"},
            "num_steps":     {"type": "int",   "default": 256,    "min": 32,   "max": 2048,
                              "label": "Steps per rollout"},
            "learning_rate": {"type": "float", "default": 2.5e-4, "min": 1e-5, "max": 1e-2,
                              "label": "Learning rate"},
            "gamma":         {"type": "float", "default": 0.99,   "min": 0.8,  "max": 1.0,
                              "label": "Discount factor γ"},
            "batch_size":    {"type": "int",   "default": 64,     "min": 16,   "max": 512,
                              "label": "Minibatch size"},
            "hidden_dim":    {"type": "int",   "default": 128,    "min": 32,   "max": 512,
                              "label": "Hidden layer size"},
            "seed":          {"type": "int",   "default": 0,      "min": 0,    "max": 9999,
                              "label": "Random seed"},
        }
