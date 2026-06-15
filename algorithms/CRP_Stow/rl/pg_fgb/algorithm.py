"""
REINFORCE with Frozen Greedy Baseline (PG-FGB) for CRP-Stow / CRP-D.

Adapted from /data/liuw2/CRP_stowage-main/cleanrl/train_RF.py.

Algorithm
---------
Each iteration:
  1. Collect ``episodes_per_iter`` full episodes with the *stochastic* policy.
  2. Collect the same number of episodes with the *frozen greedy* baseline
     (same initial seeds → matched pairs).
  3. Advantage = policy_return − baseline_return (per episode).
  4. REINFORCE update:  loss = −mean(log_π(a|s) × adv) − ent_coef × H(π).
  5. Paired t-test: if the stochastic policy is significantly better than the
     greedy baseline (p < p_value_threshold / 2, one-sided), copy the policy
     as the new baseline (progressive baseline update).

Primary metric: mean ``shifters`` over the last ``eval_episodes`` greedy
rollouts.  Fewer shifters = better.

Network
-------
Transformer-Encoder + Pointer-Decoder (see network.py).
Input: flat yard observation of shape ((n_stacks + 1) × 5).
Action: yard stack index (Discrete(n_stacks)).

Generalisation support
----------------------
The network is size-agnostic: the Transformer handles variable-length
sequences and the Pointer Decoder output size adapts to the action space
at run time.  To enable cross-size generalisation, feature_scale is fixed
to large constants (configurable via ``extra["feature_scale"]``) rather
than being derived from a single training instance.

dup_dataset training
--------------------
Set ``extra["dup_train_root"]`` to the benchmark root directory (e.g.
``Platform_CRISP/benchmark/dup_dataset``).  The training loop will
discover all .txt files whose stack count matches the environment's
``num_bays`` and cycle through them randomly, one per episode.

If the environment was originally configured for a different stack count,
set ``extra["dup_num_stacks"]`` to the desired S.  The environment will
be automatically reconfigured to (S stacks × 1 row) on the first
iteration.  All dup files with that stack count are collected across all
alpha sub-folders.
"""

from __future__ import annotations

import copy
import multiprocessing as mp
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from core.base_algorithm import BaseAlgorithm, AlgorithmConfig


# ================================================================ #
#  Feature scale (fixed constants for size-agnostic generalisation) #
# ================================================================ #

# Divisors for the 5 yard-node features: [bay, row, height, has_target, group].
# These values should cover any realistic CRP-D/CRP-Stow instance.
# Overridable via config.extra["feature_scale"] = [b, r, t, 1.0, g].
_DEFAULT_FEATURE_SCALE = [30.0, 30.0, 15.0, 1.0, 50.0]


def _get_feature_scale(cfg_extra: Dict, device) -> "torch.Tensor":
    """
    Return a 5-element feature-scale tensor.

    Accepts either a list/tuple of 5 floats or a comma-separated string
    like "30,30,15,1,50".  Falls back to _DEFAULT_FEATURE_SCALE when
    the value is absent, None, or an empty string.
    """
    import torch
    raw = cfg_extra.get("feature_scale", None)
    if not raw:
        scale = _DEFAULT_FEATURE_SCALE
    elif isinstance(raw, str):
        scale = [float(x.strip()) for x in raw.split(",") if x.strip()]
    else:
        scale = list(raw)
    if len(scale) != 5:
        raise ValueError(
            f"extra['feature_scale'] must be length 5, got {len(scale)}: {scale}"
        )
    return torch.tensor(scale, dtype=torch.float32, device=device)


# ================================================================ #
#  dup_dataset file scanning and env reconfiguration               #
# ================================================================ #

def _scan_dup_files(root: str, target_S: int) -> List[Tuple[int, int, int, Path]]:
    """
    Recursively scan *root* for ZhuDup .txt files whose stack count == target_S.

    Returns a list of (max_tiers_in_file, S, N, path) tuples.
    Searches all sub-directories (alpha=* / H-S-N / *.txt).
    """
    from core.zhu_dup_benchmark import parse_zhu_dup_folder_name

    results: List[Tuple[int, int, int, Path]] = []
    root_path = Path(root)
    if not root_path.is_dir():
        return results

    for alpha_dir in sorted(root_path.iterdir()):
        if not alpha_dir.is_dir():
            continue
        for size_dir in sorted(alpha_dir.iterdir()):
            if not size_dir.is_dir():
                continue
            parsed = parse_zhu_dup_folder_name(size_dir.name)
            if parsed is None:
                continue
            H, S, N = parsed   # (max_tiers, num_stacks, num_containers)
            if S != target_S:
                continue
            for f in sorted(size_dir.glob("*.txt")):
                results.append((H, S, N, f))

    return results


def _reconfigure_env_for_dup(env, S: int, max_tiers: int) -> None:
    """
    Reconfigure *env* in-place so it can load ZhuDup files with *S* stacks.

    ZhuDup files use a single-row layout: num_bays = S, num_rows = 1.
    This function updates the Yard object, the cached _n_stacks attribute,
    and the Gym observation / action spaces accordingly.
    """
    from core.yard import Yard

    env.config.num_bays  = S
    env.config.num_rows  = 1
    env.config.max_tiers = max_tiers

    # Rebuild the Yard with the correct dimensions.
    env.yard = Yard(S, 1, max_tiers)

    # Re-derive observation and action spaces.
    env._setup_spaces()


def _prepare_dup_episode(env, H: int, S: int, N: int, path: Path) -> None:
    """
    Update env config to load a specific dup file on the next reset().

    Sets layout_file_path, num_containers (required by apply_dup_file_to_yard),
    and expands yard stack capacity if the file is taller than the current yard.
    """
    env.config.extra["layout_file_path"] = str(path)
    env.config.num_containers = N

    # Expand stack max_tiers if the file stacks taller than the current yard.
    if H > env.config.max_tiers:
        env.config.max_tiers = H
        for stk in env.yard.stacks.values():
            stk.max_tiers = H


# ================================================================ #
#  .pro benchmark file scanning and env preparation                 #
# ================================================================ #

def _scan_pro_files(
    root: str,
    size_filters: Optional[List[str]] = None,
    seed_indices: Optional[List[int]] = None,
) -> List[Path]:
    """
    Scan *root* (non-recursively) for Jovanović BRLP benchmark .pro files.

    Parameters
    ----------
    root         : directory to scan (e.g. ``.../crp_stow/Data/Gen``).
    size_filters : if non-empty, only include files whose name contains
                   at least one of the filter strings.
                   Example: ``["Bay-3-", "Bay-5-3-"]`` keeps only small sizes.
    seed_indices : if non-empty, only include files whose name ends with
                   ``_<idx>.pro`` where ``idx`` is in the list.
                   Example: ``list(range(32))`` selects seeds 0–31 for training.

    Returns a sorted list of matching Path objects.
    """
    root_path = Path(root)
    if not root_path.is_dir():
        return []

    files = sorted(root_path.glob("*.pro"))

    if size_filters:
        files = [f for f in files if any(pat in f.name for pat in size_filters)]

    if seed_indices is not None and len(seed_indices) > 0:
        seed_set = set(seed_indices)
        import re as _re
        _seed_pat = _re.compile(r'_(\d+)\.pro$')
        filtered = []
        for f in files:
            m = _seed_pat.search(f.name)
            if m and int(m.group(1)) in seed_set:
                filtered.append(f)
        files = filtered

    return files


def _prepare_pro_episode(env, path: Path) -> None:
    """
    Configure *env* to load a specific Jovanović .pro benchmark file on the
    next ``reset()``.

    ``CRP_Stow._load_pro_file()`` updates all env dimensions (YS, YT, VS,
    H_v) directly from the file contents, so no additional reconfiguration is
    needed here — setting ``layout_file_path`` is sufficient.
    """
    env.config.extra["layout_file_path"] = str(path)


# ================================================================ #
#  Episode runners                                                  #
# ================================================================ #

def _greedy_high_override(env, mask_raw) -> Optional[int]:
    """
    Greedy HIGH pre-check: if we are in HIGH phase and any allowed action
    points to a stack whose TOP is already a retrievable container (cdd==0),
    return that action index immediately (free retrieval, no relocation needed).
    Returns None if no such action exists (model must decide).
    """
    if env._mode != "high":
        return None
    for i in range(len(mask_raw)):
        if not mask_raw[i]:
            continue
        key = env._idx_to_stack(i)
        stk = env.yard.stacks.get(key)
        if stk and not stk.is_empty and env._is_retrievable(stk.top):
            return i
    return None


def _run_episode_stochastic(
    env, agent, device, greedy_high: bool = True
) -> Tuple[List, List, List, float]:
    """
    Run one complete episode with the stochastic policy.
    Returns (obs_list, action_list, mask_list, episode_return).
    episode_return = −total_shifters (negative for minimisation).

    greedy_high: if True, any HIGH step where the target group is already on
                 top of some stack is handled by a free deterministic retrieval
                 (bypasses the model; those steps are excluded from training
                 data because they carry no learning signal).
    """
    import torch
    obs_list, act_list, mask_list = [], [], []
    obs, info = env.reset()
    done = False

    while not done:
        mask_raw = info.get("action_mask", np.ones(env.action_space.n, dtype=bool))

        # ── Greedy HIGH override ─────────────────────────────── #
        if greedy_high:
            free_action = _greedy_high_override(env, mask_raw)
            if free_action is not None:
                obs, reward, terminated, truncated, info = env.step(free_action)
                done = terminated or truncated
                continue   # no model involved → skip training data collection

        # ── Model decision (HIGH-blocked or LOW) ─────────────── #
        forbidden = torch.tensor(
            ~mask_raw.astype(bool), dtype=torch.bool, device=device
        ).unsqueeze(0)
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            enc_out     = agent.encode(obs_t)
            action_t, _ = agent(obs_t, forbidden, greedy=False, encoder_output=enc_out)
        action = int(action_t.item())

        obs_list.append(obs.copy())
        act_list.append(action)
        mask_list.append(mask_raw.copy())

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    ep_return = -float(env.get_metrics().get("relocations", env.get_metrics().get("shifters", 0.0)))
    return obs_list, act_list, mask_list, ep_return


def _run_episode_greedy(env, agent, device, greedy_high: bool = True) -> float:
    """
    Run one complete episode with the greedy (deterministic) policy.
    Returns −total_shifters.

    greedy_high: same as in _run_episode_stochastic.
    """
    import torch
    obs, info = env.reset()
    done = False

    while not done:
        mask_raw = info.get("action_mask", np.ones(env.action_space.n, dtype=bool))

        # ── Greedy HIGH override ─────────────────────────────── #
        if greedy_high:
            free_action = _greedy_high_override(env, mask_raw)
            if free_action is not None:
                obs, reward, terminated, truncated, info = env.step(free_action)
                done = terminated or truncated
                continue

        # ── Model decision ───────────────────────────────────── #
        forbidden = torch.tensor(
            ~mask_raw.astype(bool), dtype=torch.bool, device=device
        ).unsqueeze(0)
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            enc_out     = agent.encode(obs_t)
            action_t, _ = agent(obs_t, forbidden, greedy=True, encoder_output=enc_out)
        obs, reward, terminated, truncated, info = env.step(int(action_t.item()))
        done = terminated or truncated

    m = env.get_metrics()
    return -float(m.get("relocations", m.get("shifters", 0.0)))


# ================================================================ #
#  Algorithm class                                                   #
# ================================================================ #

class PgFgbStow(BaseAlgorithm):

    name     = "PG-FGB"
    category = "RL"
    description = (
        "Policy Gradient with Frozen Greedy Baseline (PG-FGB) for CRP-Stow / CRP-D. "
        "Uses a Transformer-Encoder + Pointer-Decoder policy network. "
        "The network is size-agnostic: train on one instance size and "
        "evaluate on any other. "
        "Advantage = policy_return − frozen_baseline_return (matched seeds). "
        "Baseline updated progressively via one-sided paired t-test. "
        "Set extra['dup_train_root'] to cycle through dup_dataset files "
        "during training for better diversity."
    )
    compatible_problems = ["CRP-Stow", "CRP-D"]
    step_label          = "Iteration"

    def __init__(self, config: Optional[AlgorithmConfig] = None):
        super().__init__(config)
        self._policy = None

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
            import torch.optim as optim
            import scipy.stats
        except ImportError as e:
            raise RuntimeError(f"Missing dependency: {e}.  pip install torch scipy")

        from .network import StowagePolicyNetwork

        cfg = self.config
        device = torch.device(
            "cuda" if torch.cuda.is_available() and cfg.extra.get("use_cuda", True)
            else "cpu"
        )
        torch.manual_seed(cfg.seed)
        rng = random.Random(cfg.seed)

        # ── Hyper-params ────────────────────────────────────────── #
        num_iters      = cfg.extra.get("num_iterations",    500)
        eps_per_iter   = cfg.extra.get("episodes_per_iter",   8)
        eval_episodes  = cfg.extra.get("eval_episodes",        8)
        lr             = cfg.extra.get("learning_rate",    2.5e-4)
        ent_coef       = cfg.extra.get("ent_coef",          0.01)
        p_threshold    = cfg.extra.get("p_value_threshold",  0.4)
        embed_dim      = cfg.extra.get("embed_dim",          128)
        num_enc_layers = cfg.extra.get("num_enc_layers",       2)
        num_heads      = cfg.extra.get("num_heads",             4)
        ffn_dim        = cfg.extra.get("ffn_dim",             256)
        clip_const     = cfg.extra.get("clip_constant",      10.0)
        report_every   = cfg.extra.get("report_every",         10)
        anneal_lr      = cfg.extra.get("anneal_lr",           True)
        greedy_high    = cfg.extra.get("greedy_high",          True)

        # ── Build env ───────────────────────────────────────────── #
        env = problem_factory()
        env.config.seed = cfg.seed

        # ── dup_dataset cycling setup ────────────────────────────── #
        dup_train_root = cfg.extra.get("dup_train_root", None)
        dup_files: List[Tuple[int, int, int, Path]] = []   # (H, S, N, path)

        if dup_train_root:
            # Determine target stack count.
            # 0 or unset → auto-detect from env config.
            target_S = cfg.extra.get("dup_num_stacks", 0) or (
                env.config.num_bays * env.config.num_rows
            )
            dup_files = _scan_dup_files(dup_train_root, int(target_S))

            if not dup_files:
                raise RuntimeError(
                    f"No dup .txt files with S={target_S} found under "
                    f"'{dup_train_root}'.  Check dup_num_stacks."
                )

            # Reconfigure env once for the dup layout (S bays × 1 row).
            max_tiers_all = max(h for h, _, _, _ in dup_files)
            _reconfigure_env_for_dup(env, int(target_S), max_tiers_all)

        # ── .pro benchmark file cycling setup ───────────────────────── #
        pro_train_root   = cfg.extra.get("pro_train_root",   None)
        pro_size_filters = cfg.extra.get("pro_size_filters", [])
        pro_train_seeds  = cfg.extra.get("pro_train_seeds",  None)  # None = all seeds
        pro_files: List[Path] = []

        if pro_train_root:
            pro_files = _scan_pro_files(
                pro_train_root,
                size_filters=pro_size_filters if pro_size_filters else None,
                seed_indices=pro_train_seeds,
            )
            if not pro_files:
                raise RuntimeError(
                    f"No .pro files found under '{pro_train_root}' "
                    f"with size_filters={pro_size_filters}, "
                    f"seed_indices={pro_train_seeds}.  "
                    f"Check pro_train_root / pro_size_filters / pro_train_seeds."
                )
            print(f"[PG-FGB] .pro training pool: {len(pro_files)} files "
                  f"(filters={pro_size_filters or 'none'}, "
                  f"seeds={'all' if pro_train_seeds is None else len(pro_train_seeds)})")

        # ── Build policy (size-agnostic feature scale) ───────────── #
        if pro_files:
            # Warm-up: load the first .pro file so spaces are initialised.
            _prepare_pro_episode(env, pro_files[0])
        env.reset()   # warm-up reset so spaces are finalised

        feature_scale = _get_feature_scale(cfg.extra, device)
        policy = StowagePolicyNetwork(
            embed_dim      = embed_dim,
            num_enc_layers = num_enc_layers,
            num_heads      = num_heads,
            ffn_dim        = ffn_dim,
            clip_constant  = clip_const,
            feature_scale  = feature_scale,
        ).to(device)
        policy.train()

        baseline = copy.deepcopy(policy).to(device)
        baseline.eval()

        optimizer = optim.Adam(policy.parameters(), lr=lr, eps=1e-5)

        best_greedy_shifters = float("inf")
        self._policy         = policy
        baseline_updates     = 0

        for iteration in range(1, num_iters + 1):
            if stop_event.is_set():
                break

            # ── LR annealing ────────────────────────────────────── #
            if anneal_lr:
                frac = 1.0 - (iteration - 1.0) / num_iters
                for pg in optimizer.param_groups:
                    pg["lr"] = frac * lr

            # ── Collect episodes ─────────────────────────────────── #
            policy.train()
            all_obs:        List = []
            all_acts:       List = []
            all_masks:      List = []
            step_advantages: List[float] = []
            policy_returns:  List[float] = []
            baseline_returns: List[float] = []

            for ep in range(eps_per_iter):
                seed = cfg.seed * 100_000 + iteration * 1000 + ep

                # ── Select instance (pro file / dup file / random) ── #
                if pro_files:
                    path = rng.choice(pro_files)
                    _prepare_pro_episode(env, path)
                elif dup_files:
                    H, S, N, path = rng.choice(dup_files)
                    _prepare_dup_episode(env, H, S, N, path)
                else:
                    env.config.extra.pop("layout_file_path", None)

                # ── Stochastic rollout ── #
                env.config.seed = seed
                obs_ep, act_ep, mask_ep, ret_pol = _run_episode_stochastic(
                    env, policy, device, greedy_high=greedy_high
                )

                # ── Greedy baseline with same seed / instance ── #
                env.config.seed = seed
                ret_bl = _run_episode_greedy(env, baseline, device, greedy_high=greedy_high)

                # Episode advantage (shared across all steps of the episode)
                adv = ret_pol - ret_bl
                policy_returns.append(ret_pol)
                baseline_returns.append(ret_bl)

                all_obs.extend(obs_ep)
                all_acts.extend(act_ep)
                all_masks.extend(mask_ep)
                step_advantages.extend([adv] * len(obs_ep))

            if not all_obs:
                continue

            # ── REINFORCE update ─────────────────────────────────── #
            obs_t  = torch.tensor(
                np.array(all_obs), dtype=torch.float32, device=device
            )
            acts_t = torch.tensor(all_acts, dtype=torch.long, device=device)
            # mask convention: True = forbidden (invert platform's valid mask)
            mask_t = torch.tensor(
                np.array([~m.astype(bool) for m in all_masks]),
                dtype=torch.bool, device=device,
            )
            adv_t = torch.tensor(
                step_advantages, dtype=torch.float32, device=device
            )
            if adv_t.std() > 1e-8:
                adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

            log_probs, entropy = policy.evaluate_actions(obs_t, mask_t, acts_t)
            pg_loss = -(log_probs * adv_t.detach()).mean() - ent_coef * entropy.mean()

            optimizer.zero_grad()
            pg_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()

            # ── Progressive baseline update (paired t-test) ──────── #
            if len(policy_returns) > 1:
                t_stat, p_val = scipy.stats.ttest_rel(
                    policy_returns, baseline_returns
                )
                if t_stat > 0 and p_val / 2 < p_threshold:
                    baseline = copy.deepcopy(policy).to(device)
                    baseline.eval()
                    baseline_updates += 1

            # ── Greedy evaluation & reporting ────────────────────── #
            if iteration % report_every == 0 or iteration == num_iters:
                policy.eval()
                greedy_shifters = []
                for ev in range(eval_episodes):
                    if pro_files:
                        path = rng.choice(pro_files)
                        _prepare_pro_episode(env, path)
                    elif dup_files:
                        H, S, N, path = rng.choice(dup_files)
                        _prepare_dup_episode(env, H, S, N, path)
                    env.config.seed = cfg.seed * 100_000 + iteration * 1000 + ev + 90000
                    gs = -_run_episode_greedy(env, policy, device, greedy_high=greedy_high)
                    greedy_shifters.append(gs)
                policy.train()

                mean_sh = float(np.mean(greedy_shifters))
                if mean_sh < best_greedy_shifters:
                    best_greedy_shifters = mean_sh

                self._push(
                    result_queue,
                    step     = iteration,
                    metric   = mean_sh,
                    metrics  = {
                        "shifters":         mean_sh,
                        "best_shifters":    best_greedy_shifters,
                        "policy_loss":      float(pg_loss.item()),
                        "entropy":          float(entropy.mean().item()),
                        "baseline_updates": float(baseline_updates),
                        "mean_advantage":   float(np.mean([
                            r - b for r, b in zip(policy_returns, baseline_returns)
                        ])),
                        "dup_pool_size":    float(len(dup_files)),
                    },
                    progress = iteration / num_iters,
                    snapshot = env.get_state_snapshot(),
                )

        self._policy = policy

        # ── Save trained policy ──────────────────────────────────── #
        save_path = cfg.extra.get("save_path", None)
        if save_path is None:
            _algo_dir = Path(__file__).parent / "trained_models"
            _algo_dir.mkdir(exist_ok=True)
            save_path = str(_algo_dir / "pg_fgb_policy.pt")
        import torch as _torch
        _torch.save({
            "policy_state_dict": policy.state_dict(),
            "embed_dim":      embed_dim,
            "num_enc_layers": num_enc_layers,
            "num_heads":      num_heads,
            "ffn_dim":        ffn_dim,
            "clip_constant":  clip_const,
            "feature_scale":  policy.feature_scale.cpu().tolist(),
            "best_shifters":  best_greedy_shifters,
            "num_bays":       cfg.extra.get("dup_num_stacks", None) or env.config.num_bays,
            "num_rows":       env.config.num_rows,
            "max_tiers":      env.config.max_tiers,
            "num_containers": env.config.num_containers,
            "num_groups":     env.config.num_groups,
            "vessel_bays":    env.config.vessel_bays,
            "vessel_rows":    env.config.vessel_rows,
            "vessel_tiers":   env.config.vessel_tiers,
        }, save_path)
        print(f"[PG-FGB] Policy saved → {save_path}  (best greedy shifters={best_greedy_shifters:.3f})")

    def get_best_solution(self) -> Optional[List]:
        return None

    # ---------------------------------------------------------------- #
    # Config schema                                                      #
    # ---------------------------------------------------------------- #

    @classmethod
    def config_schema(cls) -> Dict:
        return {
            # ── Training ─────────────────────────────────────────── #
            "num_iterations":    {"type": "int",   "default": 500,    "min": 10,   "max": 5000,
                                  "label": "Training iterations"},
            "episodes_per_iter": {"type": "int",   "default": 8,      "min": 1,    "max": 64,
                                  "label": "Episodes per iteration"},
            "eval_episodes":     {"type": "int",   "default": 8,      "min": 1,    "max": 32,
                                  "label": "Greedy eval episodes"},
            "report_every":      {"type": "int",   "default": 10,     "min": 1,    "max": 100,
                                  "label": "Report every N iterations"},
            # ── Optimiser ────────────────────────────────────────── #
            "learning_rate":     {"type": "float", "default": 2.5e-4, "min": 1e-6, "max": 1e-2,
                                  "label": "Learning rate"},
            "ent_coef":          {"type": "float", "default": 0.01,   "min": 0.0,  "max": 0.5,
                                  "label": "Entropy coefficient"},
            "p_value_threshold": {"type": "float", "default": 0.4,    "min": 0.05, "max": 1.0,
                                  "label": "Baseline t-test p-value"},
            "anneal_lr":         {"type": "bool",  "default": True,
                                  "label": "Anneal learning rate"},
            # ── Network ──────────────────────────────────────────── #
            "embed_dim":         {"type": "int",   "default": 128,    "min": 32,   "max": 512,
                                  "label": "Embedding dimension"},
            "num_enc_layers":    {"type": "int",   "default": 2,      "min": 1,    "max": 6,
                                  "label": "Encoder layers"},
            "num_heads":         {"type": "int",   "default": 4,      "min": 1,    "max": 16,
                                  "label": "Attention heads"},
            "ffn_dim":           {"type": "int",   "default": 256,    "min": 64,   "max": 1024,
                                  "label": "FFN hidden dimension"},
            # ── Generalisation / pro benchmark training ──────────── #
            "pro_train_root":    {"type": "str",   "default": "",
                                  "label": ".pro benchmark root dir (e.g. crp_stow/Data/Gen)"},
            "pro_size_filters":  {"type": "str",   "default": "",
                                  "label": "Size filter substrings, comma-separated (empty = all)"},
            "pro_train_seeds":   {"type": "str",   "default": "",
                                  "label": "Seed indices for training, comma-separated (empty = all)"},
            # ── Generalisation / dup training ────────────────────── #
            "dup_train_root":    {"type": "str",   "default": "",
                                  "label": "dup_dataset root dir (empty = random layout)"},
            "dup_num_stacks":    {"type": "int",   "default": 0,      "min": 0,    "max": 20,
                                  "label": "Stacks (S) for dup files (0 = auto from env)"},
            "feature_scale":     {"type": "str",   "default": "",
                                  "label": "Feature scale override '30,30,15,1,50' (empty = default)"},
            # ── Misc ─────────────────────────────────────────────── #
            "seed":              {"type": "int",   "default": 0,      "min": 0,    "max": 9999,
                                  "label": "Random seed"},
            "use_cuda":          {"type": "bool",  "default": True,
                                  "label": "Use CUDA if available"},
        }
