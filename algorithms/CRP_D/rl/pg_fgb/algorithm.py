"""
REINFORCE with Frozen Greedy Baseline (PG-FGB) for CRP-D.

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

5 input features per node: [stack_id(bay), row(=1 for yard / slot-idx for vessel),
                             height, has_target, group]

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

# Default divisors for CRP-D: [stack_id, row/slot-progress, height, has_target, group].
# stack_id: 1..10, slot-row: 1..80, height: 0..10, has_target: 0/1, group: 1..10.
_DEFAULT_FEATURE_SCALE = [10.0, 80.0, 10.0, 1.0, 10.0]


def _get_feature_scale(cfg_extra: Dict, device) -> "torch.Tensor":
    """
    Return a 5-element feature-scale tensor.

    Accepts either a list/tuple of 5 floats or a comma-separated string
    like "10,80,10,1,10".  Falls back to _DEFAULT_FEATURE_SCALE when
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

def _scan_dup_files(
    root: str,
    target_S: int,
    target_H: Optional[int] = None,
    max_files_per_folder: int = 0,
) -> List[Tuple[int, int, int, Path]]:
    """
    Recursively scan *root* for ZhuDup .txt files whose stack count == target_S.

    Parameters
    ----------
    target_S            : filter by this S (stack count).
    target_H            : if given, only include folders with this H (max tiers).
    max_files_per_folder: if > 0, take at most this many files per size_dir.

    Returns a list of (max_tiers_in_file, S, N, path) tuples.
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
            if target_H is not None and H != target_H:
                continue
            files = sorted(size_dir.glob("*.txt"))
            if max_files_per_folder > 0:
                files = files[:max_files_per_folder]
            for f in files:
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

    env.yard      = Yard(S, 1, max_tiers)
    env._n_stacks = S

    env._setup_spaces()


def _prepare_dup_episode(env, H: int, S: int, N: int, path: Path) -> None:
    """
    Update env config to load a specific dup file on the next reset().

    Sets layout_file_path, num_containers, and max_tiers to exactly H (the
    max-tier value encoded in the folder name, i.e. the paper's T parameter).
    This ensures each episode uses the correct height limit regardless of
    what was used in previous episodes.
    """
    env.config.extra["layout_file_path"] = str(path)
    env.config.num_containers = N

    # Always sync max_tiers to the file's H (expand OR shrink).
    env.config.max_tiers = H
    for stk in env.yard.stacks.values():
        stk.max_tiers = H


# ================================================================ #
#  Fixed evaluation set builder                                     #
# ================================================================ #

def _build_fixed_eval_set(
    dup_root: str,
    alpha: float,
    tiers: int,
    stacks: List[int],
    files_per_combo: int,
) -> List[Tuple[int, int, int, Path]]:
    """
    Build a deterministic fixed evaluation list for training monitoring.

    Scans dup_root/alpha=<alpha>/<H-S-N>/*.txt where H==tiers and S in stacks.
    Returns sorted list of (H, S, N, path) tuples.
    files_per_combo: how many files to take from each (S, N) folder.
    """
    from core.zhu_dup_benchmark import parse_zhu_dup_folder_name

    alpha_dir = Path(dup_root) / f"alpha={alpha:.1f}"
    if not alpha_dir.is_dir():
        return []

    stack_set = set(stacks)
    result: List[Tuple[int, int, int, Path]] = []
    for size_dir in sorted(alpha_dir.iterdir()):
        if not size_dir.is_dir():
            continue
        parsed = parse_zhu_dup_folder_name(size_dir.name)
        if parsed is None:
            continue
        h, s, n = parsed
        if h != tiers or s not in stack_set:
            continue
        files = sorted(size_dir.glob("*.txt"))[:files_per_combo]
        for f in files:
            result.append((h, s, n, f))

    result.sort(key=lambda x: (x[1], x[2], x[3].name))
    return result


def _make_env_for_fixed_eval(H: int, S: int, N: int, path: Path):
    """Create a fresh CRP_D environment for a fixed eval instance."""
    from core.base_problem import ProblemConfig
    from core.zhu_dup_benchmark import parse_zhu_dup_file
    from problems.CRP_D import CRP_D

    stacks, n, stack_groups = parse_zhu_dup_file(path)
    groups = int(max((g for col in stack_groups for g in col if g > 0), default=3))
    cfg = ProblemConfig(
        num_bays=S,
        num_rows=1,
        max_tiers=H,           # Use folder H = paper's T exactly
        num_containers=N,
        num_groups=groups,
        vessel_bays=1,
        vessel_rows=max(80, N + 10),
        vessel_tiers=1,
        seed=0,
    )
    cfg.extra["layout_file_path"] = str(path.resolve())
    return CRP_D(config=cfg)


# ================================================================ #
#  Episode runners                                                  #
# ================================================================ #

def _greedy_high_override(env, mask_raw) -> Optional[int]:
    """
    Greedy HIGH pre-check: if we are in HIGH phase and any allowed action
    points to a stack whose TOP is already the target group, return that
    action index immediately (free retrieval, no shifter needed).
    Returns None if no such action exists (model must decide).
    """
    if env._mode != "high" or env._current_slot is None:
        return None
    target_grp = int(env._vessel_state[env._current_slot, 4])
    for i in range(len(mask_raw)):
        if not mask_raw[i]:
            continue
        key = env._action_to_stack(i)
        stk = env.yard.stacks.get(key)
        if stk and not stk.is_empty and stk.top.group == target_grp:
            return i
    return None


def _compute_discounted_returns(rewards: List[float], gamma: float) -> List[float]:
    """
    Compute per-step discounted return G_t = r_t + gamma*r_{t+1} + ...
    from a list of per-step rewards.  γ=1.0 gives plain sum-to-go.
    """
    G, returns = 0.0, []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns


def _run_episode_stochastic(
    env, agent, device, greedy_high: bool = True
) -> Tuple[List, List, List, List[float], float]:
    """
    Run one complete episode with the stochastic policy.
    Returns (obs_list, action_list, mask_list, reward_list, episode_return).
    reward_list contains the per-step reward for each model-decision step.
    episode_return = −total_shifters (negative for minimisation).
    """
    import torch
    obs_list, act_list, mask_list, rew_list = [], [], [], []
    obs, info = env.reset()
    done = False

    while not done:
        mask_raw = info.get("action_mask", np.ones(env.action_space.n, dtype=bool))
        mask_bool = mask_raw.astype(bool)

        # Safety guard: avoid feeding all-forbidden masks into the pointer
        # decoder (log_softmax(-inf, ..., -inf) -> NaN).
        if not np.any(mask_bool):
            break

        if greedy_high:
            free_action = _greedy_high_override(env, mask_bool)
            if free_action is not None:
                obs, reward, terminated, truncated, info = env.step(free_action)
                done = terminated or truncated
                continue

        forbidden = torch.tensor(
            ~mask_bool, dtype=torch.bool, device=device
        ).unsqueeze(0)
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            enc_out     = agent.encode(obs_t)
            action_t, _ = agent(obs_t, forbidden, greedy=False, encoder_output=enc_out)
        action = int(action_t.item())

        obs_list.append(obs.copy())
        act_list.append(action)
        mask_list.append(mask_bool.copy())

        obs, reward, terminated, truncated, info = env.step(action)
        rew_list.append(float(reward))
        done = terminated or truncated

    ep_return = -float(env.get_metrics().get("shifters", 0.0))
    return obs_list, act_list, mask_list, rew_list, ep_return


def _run_episode_greedy(env, agent, device, greedy_high: bool = True) -> float:
    """
    Run one complete episode with the greedy (deterministic) policy.
    Returns −total_shifters.
    """
    import torch
    obs, info = env.reset()
    done = False

    while not done:
        mask_raw = info.get("action_mask", np.ones(env.action_space.n, dtype=bool))
        mask_bool = mask_raw.astype(bool)

        # Same safety guard as stochastic rollout.
        if not np.any(mask_bool):
            break

        if greedy_high:
            free_action = _greedy_high_override(env, mask_bool)
            if free_action is not None:
                obs, reward, terminated, truncated, info = env.step(free_action)
                done = terminated or truncated
                continue

        forbidden = torch.tensor(
            ~mask_bool, dtype=torch.bool, device=device
        ).unsqueeze(0)
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            enc_out     = agent.encode(obs_t)
            action_t, _ = agent(obs_t, forbidden, greedy=True, encoder_output=enc_out)
        obs, reward, terminated, truncated, info = env.step(int(action_t.item()))
        done = terminated or truncated

    return -float(env.get_metrics().get("shifters", 0.0))


# ================================================================ #
#  Algorithm class                                                   #
# ================================================================ #

class PgFgbD(BaseAlgorithm):
    """PG-FGB algorithm specialised for CRP-D with dup_dataset."""

    name     = "PG-FGB-D"
    category = "RL"
    description = (
        "Policy Gradient with Frozen Greedy Baseline (PG-FGB) for CRP-D. "
        "Uses a Transformer-Encoder + Pointer-Decoder policy network. "
        "The network is size-agnostic: train on one instance size and "
        "evaluate on any other. "
        "Advantage = policy_return − frozen_baseline_return (matched seeds). "
        "Baseline updated progressively via one-sided paired t-test. "
        "Set extra['dup_train_root'] to cycle through dup_dataset files "
        "during training."
    )
    compatible_problems = ["CRP-D"]
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

        from .network import CrpdPolicyNetwork

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
        p_threshold    = cfg.extra.get("p_value_threshold", 0.05)  # was 0.4; 0.05 = statistically stable
        gamma          = cfg.extra.get("gamma",              1.0)   # discount for step-level returns
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
        dup_files: List[Tuple[int, int, int, Path]] = []

        if dup_train_root:
            # Support single int or list of ints for S
            raw_stacks = cfg.extra.get("dup_num_stacks_list", None) or \
                         cfg.extra.get("dup_num_stacks", 0) or \
                         (env.config.num_bays * env.config.num_rows)
            if isinstance(raw_stacks, int):
                stacks_list = [raw_stacks]
            else:
                stacks_list = list(raw_stacks)

            for s_val in stacks_list:
                files_s = _scan_dup_files(dup_train_root, int(s_val))
                dup_files.extend(files_s)

            if not dup_files:
                raise RuntimeError(
                    f"No dup .txt files with S in {stacks_list} found under "
                    f"'{dup_train_root}'.  Check dup_num_stacks_list."
                )

            max_tiers_all = max(h for h, _, _, _ in dup_files)
            # Initialise env with the first S in the list
            _reconfigure_env_for_dup(env, int(stacks_list[0]), max_tiers_all)

        # Track which S the env is currently configured for
        current_env_S: int = env.config.num_bays

        # Build a per-S index for iteration-level S selection.
        # This prevents inhomogeneous obs shapes within one gradient update batch:
        # all episodes in the same iteration must use the SAME S.
        dup_files_by_s: dict = {}
        for h, s, n, p in dup_files:
            dup_files_by_s.setdefault(s, []).append((h, s, n, p))
        available_s_values: list = sorted(dup_files_by_s.keys())

        # ── Build policy ─────────────────────────────────────────── #
        env.reset()   # warm-up reset so spaces are finalised

        feature_scale = _get_feature_scale(cfg.extra, device)
        policy = CrpdPolicyNetwork(
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
            all_obs:         List = []
            all_acts:        List = []
            all_masks:       List = []
            step_advantages: List[float] = []
            policy_returns:  List[float] = []
            baseline_returns: List[float] = []

            # Pick a single S for this entire iteration so that all obs in
            # the batch have the same flat length (n_stacks+1)*5.
            if available_s_values:
                iter_S = rng.choice(available_s_values)
                iter_files = dup_files_by_s[iter_S]
                if iter_S != current_env_S:
                    max_tiers_all = max(h for h, _, _, _ in dup_files)
                    _reconfigure_env_for_dup(env, iter_S, max_tiers_all)
                    current_env_S = iter_S
            else:
                iter_files = dup_files

            for ep in range(eps_per_iter):
                seed = cfg.seed * 100_000 + iteration * 1000 + ep

                if iter_files:
                    H, S, N, path = rng.choice(iter_files)
                    _prepare_dup_episode(env, H, S, N, path)
                else:
                    env.config.extra.pop("layout_file_path", None)

                env.config.seed = seed
                obs_ep, act_ep, mask_ep, rew_ep, ret_pol = _run_episode_stochastic(
                    env, policy, device, greedy_high=greedy_high
                )

                env.config.seed = seed
                ret_bl = _run_episode_greedy(env, baseline, device, greedy_high=greedy_high)

                policy_returns.append(ret_pol)
                baseline_returns.append(ret_bl)

                # Step-level discounted returns minus baseline.
                # G_t = r_t + γ*r_{t+1} + ... provides a finer credit signal
                # than assigning the same episode-level advantage to all steps.
                if rew_ep:
                    step_returns = _compute_discounted_returns(rew_ep, gamma)
                    # Normalise baseline to per-step scale for subtraction.
                    per_step_bl = ret_bl / max(len(rew_ep), 1)
                    adv_steps = [g - per_step_bl for g in step_returns]
                else:
                    adv_steps = []

                all_obs.extend(obs_ep)
                all_acts.extend(act_ep)
                all_masks.extend(mask_ep)
                step_advantages.extend(adv_steps)

            if not all_obs:
                continue

            # ── REINFORCE update ─────────────────────────────────── #
            obs_t  = torch.tensor(
                np.array(all_obs), dtype=torch.float32, device=device
            )
            acts_t = torch.tensor(all_acts, dtype=torch.long, device=device)
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

                fixed_eval_set = cfg.extra.get("fixed_eval_set", None)

                if fixed_eval_set:
                    # ── Fixed deterministic eval ──────────────────── #
                    greedy_shifters_s6: List[float] = []
                    greedy_shifters_ood: List[float] = []
                    for H_e, S_e, N_e, path_e in fixed_eval_set:
                        eval_env = _make_env_for_fixed_eval(H_e, S_e, N_e, path_e)
                        eval_env.config.seed = cfg.seed
                        gs = -_run_episode_greedy(eval_env, policy, device,
                                                  greedy_high=greedy_high)
                        # "S6" = smallest S in training list; "OOD" = larger S
                        if S_e == int(stacks_list[0]):
                            greedy_shifters_s6.append(gs)
                        else:
                            greedy_shifters_ood.append(gs)
                    greedy_shifters = greedy_shifters_s6 + greedy_shifters_ood
                    eval_s6_mean = float(np.mean(greedy_shifters_s6)) if greedy_shifters_s6 else float("nan")
                    eval_ood_mean = float(np.mean(greedy_shifters_ood)) if greedy_shifters_ood else float("nan")
                else:
                    # ── Random eval (original behaviour) ─────────── #
                    greedy_shifters = []
                    for ev in range(eval_episodes):
                        if dup_files:
                            H, S, N, path = rng.choice(dup_files)
                            _prepare_dup_episode(env, H, S, N, path)
                        env.config.seed = cfg.seed * 100_000 + iteration * 1000 + ev + 90000
                        gs = -_run_episode_greedy(env, policy, device, greedy_high=greedy_high)
                        greedy_shifters.append(gs)
                    eval_s6_mean = float(np.mean(greedy_shifters))
                    eval_ood_mean = float("nan")

                policy.train()

                mean_sh = float(np.mean(greedy_shifters))
                if mean_sh < best_greedy_shifters:
                    best_greedy_shifters = mean_sh

                metrics_dict = {
                    "shifters":         mean_sh,
                    "best_shifters":    best_greedy_shifters,
                    "policy_loss":      float(pg_loss.item()),
                    "entropy":          float(entropy.mean().item()),
                    "baseline_updates": float(baseline_updates),
                    "mean_advantage":   float(np.mean([
                        r - b for r, b in zip(policy_returns, baseline_returns)
                    ])),
                    "dup_pool_size":    float(len(dup_files)),
                    "shifters_s6":      eval_s6_mean,
                    "shifters_ood":     eval_ood_mean,
                }
                self._push(
                    result_queue,
                    step     = iteration,
                    metric   = mean_sh,
                    metrics  = metrics_dict,
                    progress = iteration / num_iters,
                    snapshot = env.get_state_snapshot(),
                )

        self._policy = policy

        # ── Save trained policy ──────────────────────────────────── #
        save_path = cfg.extra.get("save_path", None)
        if save_path is None:
            _algo_dir = Path(__file__).parent / "trained_models"
            _algo_dir.mkdir(exist_ok=True)
            save_path = str(_algo_dir / "pg_fgb_crpd.pt")
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
        print(f"[PG-FGB-D] Policy saved → {save_path}  "
              f"(best greedy shifters={best_greedy_shifters:.3f})")

    def get_best_solution(self) -> Optional[List]:
        return None

    # ---------------------------------------------------------------- #
    # Config schema                                                      #
    # ---------------------------------------------------------------- #

    @classmethod
    def config_schema(cls) -> Dict:
        return {
            "num_iterations":    {"type": "int",   "default": 1000,   "min": 10,   "max": 5000,
                                  "label": "Training iterations"},
            "episodes_per_iter": {"type": "int",   "default": 16,     "min": 1,    "max": 64,
                                  "label": "Episodes per iteration"},
            "eval_episodes":     {"type": "int",   "default": 16,     "min": 1,    "max": 32,
                                  "label": "Greedy eval episodes"},
            "report_every":      {"type": "int",   "default": 20,     "min": 1,    "max": 100,
                                  "label": "Report every N iterations"},
            "learning_rate":     {"type": "float", "default": 2.5e-4, "min": 1e-6, "max": 1e-2,
                                  "label": "Learning rate"},
            "ent_coef":          {"type": "float", "default": 0.01,   "min": 0.0,  "max": 0.5,
                                  "label": "Entropy coefficient"},
            "p_value_threshold": {"type": "float", "default": 0.4,    "min": 0.05, "max": 1.0,
                                  "label": "Baseline t-test p-value"},
            "anneal_lr":         {"type": "bool",  "default": True,
                                  "label": "Anneal learning rate"},
            "embed_dim":         {"type": "int",   "default": 128,    "min": 32,   "max": 512,
                                  "label": "Embedding dimension"},
            "num_enc_layers":    {"type": "int",   "default": 2,      "min": 1,    "max": 6,
                                  "label": "Encoder layers"},
            "num_heads":         {"type": "int",   "default": 4,      "min": 1,    "max": 16,
                                  "label": "Attention heads"},
            "ffn_dim":           {"type": "int",   "default": 256,    "min": 64,   "max": 1024,
                                  "label": "FFN hidden dimension"},
            "dup_train_root":    {"type": "str",   "default": "",
                                  "label": "dup_dataset root dir (empty = random layout)"},
            "dup_num_stacks":    {"type": "int",   "default": 0,      "min": 0,    "max": 20,
                                  "label": "Stacks (S) for dup files (0 = auto from env)"},
            "feature_scale":     {"type": "str",   "default": "",
                                  "label": "Feature scale '10,80,10,1,10' (empty = CRP-D default)"},
            "seed":              {"type": "int",   "default": 0,      "min": 0,    "max": 9999,
                                  "label": "Random seed"},
            "use_cuda":          {"type": "bool",  "default": True,
                                  "label": "Use CUDA if available"},
        }
