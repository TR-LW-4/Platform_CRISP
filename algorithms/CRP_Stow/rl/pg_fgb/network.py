"""
Transformer-Encoder + Pointer-Decoder policy network for CRP-Stow.

Adapted from /data/liuw2/CRP_stowage-main/cleanrl/policy_network.py.

Architecture
------------
Encoder: multi-layer Transformer over yard-slot nodes (each node = 5 features).
Decoder: cross-attention query → pointer scores over yard slots.

Input
-----
node_features : (B, N, 5)  where N = n_yard_slots + 1
    The last node carries the current vessel-slot context.
    The first n_yard_slots nodes represent individual yard slots.

action_masks  : (B, A)  True = forbidden action
    A = n_yard_slots  (pointer selects over yard slots only, not vessel node).

The network normalises each feature dimension by feature_scale before encoding,
so raw integer yard-state values (bay, row, tier, occupied, group) are fed
directly from the platform observation without any other pre-processing.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class _MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.scale     = math.sqrt(self.head_dim)
        self.dropout   = nn.Dropout(dropout)
        self.q_proj    = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj    = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj    = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj  = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, query, key, value, mask=None):
        B, T_q, _ = query.shape
        T_k = key.shape[1]

        def split(x):
            return x.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        Q = split(self.q_proj(query))
        K = split(self.k_proj(key))
        V = split(self.v_proj(value))

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask, float("-inf"))
        attn = self.dropout(F.softmax(scores, dim=-1))
        out  = torch.matmul(attn, V)
        out  = out.transpose(1, 2).contiguous().view(B, T_q, self.embed_dim)
        return self.out_proj(out)


class _TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_dim, dropout=0.0):
        super().__init__()
        self.self_attn = _MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ffn   = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim), nn.ReLU(), nn.Linear(ffn_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.norm1(x + self.drop(self.self_attn(x, x, x, mask)))
        x = self.norm2(x + self.drop(self.ffn(x)))
        return x


class _TransformerEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim, num_layers, num_heads, ffn_dim=None, dropout=0.0):
        super().__init__()
        ffn_dim = ffn_dim or 4 * embed_dim
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.input_norm = nn.LayerNorm(embed_dim)
        self.layers = nn.ModuleList([
            _TransformerEncoderLayer(embed_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        h = self.input_norm(self.input_proj(x))
        for layer in self.layers:
            h = layer(h, mask)
        return h


class _PointerDecoder(nn.Module):
    def __init__(self, embed_dim, num_heads, clip_constant=10.0, dropout=0.0):
        super().__init__()
        self.embed_dim     = embed_dim
        self.clip_constant = clip_constant
        self.cross_attn    = _MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm          = nn.LayerNorm(embed_dim)
        self.pointer_k     = nn.Linear(embed_dim, embed_dim, bias=False)
        self.pointer_q     = nn.Linear(embed_dim, embed_dim, bias=False)
        self.start_embedding = nn.Parameter(torch.randn(1, 1, embed_dim))

    def _pointer_scores(self, query, encoder_output, action_mask):
        """
        query         : (B, 1, D)
        encoder_output: (B, N, D)  — N = n_yard_slots + 1 (vessel node last)
        action_mask   : (B, A)     — A = n_yard_slots; True = forbidden
        Returns log_softmax over A yard-slot actions.
        """
        n_actions = action_mask.shape[-1]
        # Select first n_actions nodes (yard slots) as pointer targets
        act_emb = encoder_output[:, :n_actions, :]      # (B, A, D)
        scale   = math.sqrt(self.embed_dim)
        q       = self.pointer_q(query)                  # (B, 1, D)
        k       = self.pointer_k(act_emb)               # (B, A, D)
        scores  = torch.bmm(q, k.transpose(1, 2)).squeeze(1) / scale   # (B, A)
        scores  = self.clip_constant * torch.tanh(scores)
        scores  = scores.masked_fill(action_mask.bool(), float("-inf"))
        return F.log_softmax(scores, dim=-1)

    def forward(self, encoder_output, action_mask, greedy=False):
        B, _, D = encoder_output.shape
        query   = self.start_embedding.expand(B, 1, D)
        context = self.cross_attn(query, encoder_output, encoder_output)
        context = self.norm(query + context)
        log_probs = self._pointer_scores(context, encoder_output, action_mask)
        if greedy:
            action = log_probs.argmax(dim=-1)
        else:
            action = Categorical(logits=log_probs).sample()
        log_prob = log_probs.gather(1, action.unsqueeze(1)).squeeze(1)
        return action, log_prob


class StowagePolicyNetwork(nn.Module):
    """
    Transformer + Pointer policy network for CRP-Stow.

    Parameters
    ----------
    embed_dim      : Transformer hidden dimension.
    num_enc_layers : Number of encoder layers.
    num_heads      : Multi-head attention heads.
    ffn_dim        : Feed-forward inner dimension (default 4 × embed_dim).
    clip_constant  : Pointer score clipping range.
    feature_scale  : Per-feature normalisation divisors (length-5 tensor).
    """

    INPUT_DIM = 5   # [bay, row, tier/height, occupied/has_target, group]

    def __init__(
        self,
        embed_dim:      int   = 128,
        num_enc_layers: int   = 2,
        num_heads:      int   = 4,
        ffn_dim:        int   = 256,
        clip_constant:  float = 10.0,
        dropout:        float = 0.0,
        feature_scale:  "torch.Tensor | None" = None,
    ):
        super().__init__()
        if feature_scale is None:
            feature_scale = torch.ones(self.INPUT_DIM)
        self.register_buffer("feature_scale", feature_scale.float())

        self.encoder = _TransformerEncoder(
            self.INPUT_DIM, embed_dim, num_enc_layers, num_heads, ffn_dim, dropout
        )
        self.decoder = _PointerDecoder(embed_dim, num_heads, clip_constant, dropout)

    def _reshape(self, flat_obs: "torch.Tensor") -> "torch.Tensor":
        """
        Accept flat observation from the platform:
            (B, (n_yard_slots + 1) * 5) or (B, n_yard_slots + 1, 5)
        Returns (B, n_yard_slots + 1, 5) normalised.
        """
        if flat_obs.dim() == 2:
            usable = (flat_obs.shape[-1] // self.INPUT_DIM) * self.INPUT_DIM
            flat_obs = flat_obs[:, :usable]
            flat_obs = flat_obs.view(flat_obs.shape[0], -1, self.INPUT_DIM)
        return flat_obs.float() / self.feature_scale

    def encode(self, flat_obs: "torch.Tensor") -> "torch.Tensor":
        """Return (B, N, D) encoder output."""
        return self.encoder(self._reshape(flat_obs))

    def forward(
        self,
        flat_obs:      "torch.Tensor",
        action_mask:   "torch.Tensor",      # (B, A)  True = forbidden
        greedy:        bool = False,
        encoder_output: "torch.Tensor | None" = None,
    ):
        if encoder_output is None:
            encoder_output = self.encode(flat_obs)
        return self.decoder(encoder_output, action_mask, greedy=greedy)

    def evaluate_actions(
        self,
        flat_obs:    "torch.Tensor",
        action_mask: "torch.Tensor",
        actions:     "torch.Tensor",
    ):
        """
        Re-evaluate log-probs and entropy for a batch of (obs, action) pairs.
        Used for the REINFORCE gradient update.
        Returns (log_prob, entropy) both shape (B,).
        """
        encoder_output = self.encode(flat_obs)
        B, _, D = encoder_output.shape
        query   = self.decoder.start_embedding.expand(B, 1, D)
        context = self.decoder.cross_attn(query, encoder_output, encoder_output)
        context = self.decoder.norm(query + context)
        log_probs = self.decoder._pointer_scores(context, encoder_output, action_mask)

        selected_lp = log_probs.gather(1, actions.long().unsqueeze(1)).squeeze(1)
        probs       = log_probs.exp()
        lp_stable   = log_probs.clamp(min=torch.finfo(log_probs.dtype).min)
        entropy     = -(probs * lp_stable).sum(dim=-1)
        return selected_lp, entropy
