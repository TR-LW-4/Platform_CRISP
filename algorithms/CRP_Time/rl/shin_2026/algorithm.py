"""
Shin et al. (2026) Scale-Diverse DRL for the Container Retrieval Problem.

Reference
---------
Woo-Jin Shin, Inguk Choi, Sang-Hyun Cho, Hyun-Jung Kim.
"Learning to Retrieve Containers: A Scale-Diverse Deep Reinforcement
Learning Approach for the Container Retrieval Problem."
Transportation Research Part C: Emerging Technologies, 183 (2026) 105496.
https://doi.org/10.1016/j.trc.2025.105496

Official code: https://github.com/operagang/CRP_RL

Role on the platform
--------------------
Native multi-bay CRP-Time algorithm.  Uses the official pretrained model
(offline, epoch 100) for inference.  Training (scale-diverse REINFORCE
with normalized return) is NOT re-implemented here; use the upstream repo
for training.  Inference is near-instantaneous (<1 s) for realistic
instances.

Model architecture
------------------
* Intra-stack encoder: LSTM over tier dimension
* Inter-stack encoder: multi-head attention (3 layers, 8 heads, d=128)
  + bay-level aggregation
* Decoder: cross-attention + scaled dot-product pointer

Kinematics (must match CRP-Time defaults)
-----------------------------------------
  t_pd  = 30 s   (pickup + set-down)
  t_acc = 40 s   (gantry acc/dec when switching bays)
  t_bay =  3.5 s (gantry per bay)
  t_row =  1.2 s (trolley per row)
"""

from __future__ import annotations

import argparse
import copy
import multiprocessing as mp
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from core.base_algorithm import BaseAlgorithm, AlgorithmConfig
from core.objectives import KinematicsModel, compute_crane_time, lower_bound_relocations
from core.plan import Movement, RelocationPlan

# Model lives in the same package
_PACKAGE_DIR = Path(__file__).parent


def _default_model_path() -> str:
    return str(_PACKAGE_DIR / "pretrained" / "shin2026_offline.pt")


# ================================================================ #
#  Helpers: convert platform yard → Shin-2026 tensor & back        #
# ================================================================ #

def _yard_to_tensor(env):
    """
    Convert the platform's Yard state to a (1, B, R, T) float tensor
    suitable for the Shin-2026 model.

    Container priority values are stored as floats; empty slots = 0.
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is required for Shin2026DRL.")

    b, r, t = env.config.num_bays, env.config.num_rows, env.config.max_tiers
    x = torch.zeros(1, b, r, t, dtype=torch.float32)
    for (bay, row), stk in env.yard.stacks.items():
        for tier_idx, c in enumerate(stk.containers):
            x[0, bay - 1, row - 1, tier_idx] = float(c.priority)
    return x




# ================================================================ #
#  Main algorithm class                                              #
# ================================================================ #

class Shin2026DRL(BaseAlgorithm):

    name                = "Shin et al. (2026) Scale-Diverse DRL"
    category            = "RL"
    description         = (
        "[native multi-bay]  "
        "Scale-diverse DRL for CRP-Time (Shin, Choi, Cho & Kim, TRC 2026). "
        "LSTM intra-stack + multi-head attention inter-stack encoder-decoder; "
        "trained on 35–70-container instances with normalized-return "
        "REINFORCE.  Runs pretrained model in inference mode. "
        "Outperforms all classical heuristics on the Lee & Lee benchmark."
    )
    compatible_problems = ["CRP-Time"]
    step_label          = "Seed"

    def __init__(self, config: Optional[AlgorithmConfig] = None):
        super().__init__(config)
        self._best_wt: float = float("inf")

    # ---------------------------------------------------------------- #
    # Train (= inference-only run over eval seeds)                      #
    # ---------------------------------------------------------------- #

    def train(
        self,
        problem_factory: Callable,
        result_queue:    mp.Queue,
        stop_event:      mp.Event,
    ) -> None:
        try:
            import torch
        except ImportError:
            self._push(result_queue, step=0, metric=float("inf"),
                       metrics={"error": -1.0}, progress=1.0,
                       extra={"error": "PyTorch not found"})
            return

        cfg          = self.config
        rng_seed     = cfg.seed
        n_seeds      = max(1, cfg.num_eval_seeds)
        model_path   = cfg.extra.get("model_path", _default_model_path())
        device_str   = cfg.extra.get("device", "cpu")
        device       = torch.device(device_str)

        # ── Build a minimal args namespace for the Shin-2026 model ── #
        args = argparse.Namespace(
            device       = device,
            embed_dim    = int(cfg.extra.get("embed_dim",    128)),
            n_encode_layers = int(cfg.extra.get("n_encode_layers", 3)),
            n_heads      = int(cfg.extra.get("n_heads",      8)),
            ff_hidden    = int(cfg.extra.get("ff_hidden",    512)),
            tanh_c       = float(cfg.extra.get("tanh_c",    10.0)),
            lstm         = bool(cfg.extra.get("lstm",        True)),
            bay_embedding= bool(cfg.extra.get("bay_embedding", True)),
            online       = bool(cfg.extra.get("online",      False)),
            online_known_num = cfg.extra.get("online_known_num", None),
            pomo_size    = 1,
        )

        # ── Load model ───────────────────────────────────────────── #
        from .model.model import Model as Shin2026Model
        model = Shin2026Model(args).to(device)
        if os.path.exists(model_path):
            state = torch.load(model_path, map_location=device)
            model.load_state_dict(state)
            model.eval()
            model.decoder.set_sampler("greedy")
        else:
            self._push(result_queue, step=0, metric=float("inf"),
                       metrics={"error": -1.0}, progress=1.0,
                       extra={"error": f"Model not found: {model_path}"})
            return

        # ── Inference over eval seeds ─────────────────────────────── #
        all_metrics: List[Dict] = []

        for seed_idx in range(n_seeds):
            if stop_event.is_set():
                break

            env = problem_factory()
            env.config.seed = rng_seed + seed_idx
            env.reset()

            kin = KinematicsModel.from_config_extra(env.config.extra)
            lb  = lower_bound_relocations(env.yard)

            # Convert platform yard state → Shin-2026 tensor
            x = _yard_to_tensor(env).to(device)

            with torch.no_grad():
                wt_tensor, _ = model(x, max_retrievals=None)
            crane_time = float(wt_tensor.mean().item())

            # Shin model also tracks relocations internally via its Env
            # We expose only the crane_time as primary (matches paper metric)
            metrics = {
                "crane_time":  crane_time,
                "time":        crane_time,
                "lower_bound": float(lb),
                # relocations not directly available from this inference path
                "relocations": float("nan"),
            }
            all_metrics.append(metrics)

            if crane_time < self._best_wt:
                self._best_wt  = crane_time
                self._best_metric = crane_time
                self._best_solution = []   # DRL has no action-index list

            self._push(
                result_queue,
                step     = seed_idx + 1,
                metric   = crane_time,
                metrics  = metrics,
                progress = (seed_idx + 1) / n_seeds,
                snapshot = env.get_state_snapshot(),
                extra    = {"seed": seed_idx, "model_path": model_path},
            )

        if all_metrics:
            valid = [m["crane_time"] for m in all_metrics if not np.isnan(m["crane_time"])]
            agg = {k: float(np.mean([m[k] for m in all_metrics
                                     if not np.isnan(m.get(k, float("nan")))]))
                   for k in all_metrics[0]}
            self._push(
                result_queue,
                step     = n_seeds,
                metric   = self._best_metric,
                metrics  = agg,
                progress = 1.0,
            )

    def get_best_solution(self) -> Optional[List[int]]:
        return self._best_solution

    # ---------------------------------------------------------------- #
    # Config schema                                                      #
    # ---------------------------------------------------------------- #

    @classmethod
    def config_schema(cls) -> Dict:
        base = super().config_schema()
        base.update({
            "num_eval_seeds": {
                "type": "int", "default": 3, "min": 1, "max": 20,
                "label": "Evaluation seeds",
                "help": "Number of random yards to evaluate the pretrained model on.",
            },
            "model_path": {
                "type": "str",
                "default": str(Path(__file__).parent / "pretrained" / "shin2026_offline.pt"),
                "label": "Pretrained model path (.pt)",
                "help": "Path to the pretrained PyTorch model weights.",
            },
            "device": {
                "type": "str", "default": "cpu",
                "label": "Inference device",
                "help": "'cpu' or 'cuda:0'.",
            },
            "embed_dim": {
                "type": "int", "default": 128, "min": 32, "max": 512,
                "label": "Embedding dimension",
            },
            "n_encode_layers": {
                "type": "int", "default": 3, "min": 1, "max": 8,
                "label": "Encoder layers",
            },
        })
        return base
