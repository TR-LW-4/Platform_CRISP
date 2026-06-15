"""
CRP-Stow 算法对比脚本
======================
在相同配置（4bay × 5row × 5tier，80箱，8组，sequential 分组）下，
对固定 seed 列表跑以下算法并输出对比表格：

    1. Ji (2015) Nearest
    2. Ji (2015) Optimization
    3. PG-FGB (RL)

用法（从 Platform_CRISP 根目录）：
    CUDA_VISIBLE_DEVICES="" PYTHONPATH=/data/liuw2/Platform_CRISP \\
        python algorithms/CRP_Stow/rl/pg_fgb/compare.py
"""

from __future__ import annotations

import os
import sys
import pathlib

# ── 自动把项目根加到 PYTHONPATH ─────────────────────────────────── #
_ALGO_DIR = pathlib.Path(__file__).parent
_ROOT     = _ALGO_DIR.parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ════════════════════════════════════════════════════════════════════
#  CONFIG — 修改这里调整对比配置
# ════════════════════════════════════════════════════════════════════

YARD = dict(
    num_bays       = 4,
    num_rows       = 1,          # BRLP: 1-D yard
    max_tiers      = 5,
    num_containers = 20,
    num_groups     = 4,          # = number of vessel stacks (VS)
)

# 对比用的 seed 列表（每个 seed = 一个随机实例）
TEST_SEEDS = list(range(20))      # seed 0~19，共 20 个实例

# RL 模型路径
MODEL_PATH = str(_ALGO_DIR / "trained_models" / "pg_fgb_policy.pt")

# greedy-high override（与训练时保持一致）
GREEDY_HIGH = True


# ════════════════════════════════════════════════════════════════════
#  工具函数
# ════════════════════════════════════════════════════════════════════

def _make_env(seed: int):
    from core.base_problem import ProblemConfig
    from problems.CRP_Stow import CRP_Stow
    cfg = ProblemConfig(**YARD, seed=seed)
    return CRP_Stow(config=cfg)


def _run_ji_episode(env, ji_instance) -> int:
    """运行一局 Ji (2015)，返回 relocations。"""
    from algorithms.CRP_Stow.heuristic.ji_2015.algorithm import _run_episode
    _, metrics = _run_episode(env, ji_instance._select)
    return int(metrics.get("relocations", metrics.get("shifters", 0)))




def _run_rl_episode(env, policy) -> int:
    """运行一局 RL greedy 推断，返回 shifters。"""
    import torch
    import numpy as np

    obs, info = env.reset()
    done = False

    while not done:
        mask_raw = info.get("action_mask", np.ones(env.action_space.n, dtype=bool))

        action = None
        # greedy-high override: retrieve retrievable top containers immediately
        if GREEDY_HIGH and env._mode == "high":
            for i in range(env.action_space.n):
                if not mask_raw[i]:
                    continue
                key = env._idx_to_stack(i)
                stk = env.yard.stacks.get(key)
                if stk and not stk.is_empty and env._is_retrievable(stk.top):
                    action = i
                    break

        if action is None:
            forbidden = torch.tensor(~mask_raw.astype(bool), dtype=torch.bool).unsqueeze(0)
            obs_t     = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                enc_out     = policy.encode(obs_t)
                action_t, _ = policy(obs_t, forbidden, greedy=True, encoder_output=enc_out)
            action = int(action_t.item())

        obs, _, term, trunc, info = env.step(action)
        done = term or trunc

    m = env.get_metrics()
    return int(m.get("relocations", m.get("shifters", 0)))


# ════════════════════════════════════════════════════════════════════
#  主函数
# ════════════════════════════════════════════════════════════════════

def main():
    import numpy as np

    from algorithms.CRP_Stow.heuristic.ji_2015.algorithm import (
        JiNearest, JiOptimization,
    )

    # ── 加载 RL 模型 ─────────────────────────────────────────── #
    rl_available = os.path.isfile(MODEL_PATH)
    policy = None
    if rl_available:
        import torch
        from algorithms.CRP_Stow.rl.pg_fgb.network import StowagePolicyNetwork
        ckpt = torch.load(MODEL_PATH, map_location="cpu")
        feature_scale = torch.tensor(ckpt["feature_scale"], dtype=torch.float32)
        policy = StowagePolicyNetwork(
            embed_dim      = ckpt["embed_dim"],
            num_enc_layers = ckpt["num_enc_layers"],
            num_heads      = ckpt["num_heads"],
            ffn_dim        = ckpt["ffn_dim"],
            clip_constant  = ckpt["clip_constant"],
            feature_scale  = feature_scale,
        )
        policy.load_state_dict(ckpt["policy_state_dict"])
        policy.eval()
        best_s = ckpt.get("best_shifters", float("nan"))
        print(f"[✓] 加载 RL 模型: {MODEL_PATH}")
        print(f"    训练时 best greedy shifters = {best_s:.3f}")
    else:
        print(f"[!] RL 模型未找到: {MODEL_PATH}  （将只对比启发式）")

    # ── 打印配置摘要 ──────────────────────────────────────────── #
    print()
    print("=" * 70)
    print("  CRP-Stow (BRLP) 算法对比  (Jovanović 2019)")
    print("=" * 70)
    print(f"  堆场: {YARD['num_bays']} stacks × max {YARD['max_tiers']} tiers (BRLP 1-D)")
    print(f"  箱子: {YARD['num_containers']} 个  VS={YARD['num_groups']} vessel stacks")
    print(f"  实例: {len(TEST_SEEDS)} 个  (seed {TEST_SEEDS[0]}~{TEST_SEEDS[-1]})")
    print("=" * 70)

    # ── 算法列表 (名字, 运行函数) ─────────────────────────────── #
    ji_instances = {
        "Ji Near":  JiNearest(),
        "Ji Optim": JiOptimization(),
    }

    ALGOS: list[tuple[str, callable]] = [
        ("Ji Near",  lambda env, inst=ji_instances["Ji Near"]:  _run_ji_episode(env, inst)),
        ("Ji Optim", lambda env, inst=ji_instances["Ji Optim"]: _run_ji_episode(env, inst)),
    ]
    if rl_available:
        ALGOS.append(("PG-FGB", lambda env, p=policy: _run_rl_episode(env, p)))

    algo_names = [name for name, _ in ALGOS]
    col_w = 10

    # ── 表头 ─────────────────────────────────────────────────── #
    header = f"{'Seed':>6} | " + " | ".join(f"{n:>{col_w}}" for n in algo_names)
    sep    = "-" * len(header)
    print(f"\n{header}")
    print(sep)

    # ── 逐 seed 对比 ──────────────────────────────────────────── #
    results: dict[str, list[int]] = {name: [] for name in algo_names}

    for seed in TEST_SEEDS:
        row_vals = []
        for name, run_fn in ALGOS:
            env = _make_env(seed)
            try:
                s = run_fn(env)
            except Exception as exc:
                s = -1
                print(f"  [WARN] {name} seed={seed}: {exc}")
            results[name].append(s)
            row_vals.append(s)
        row = f"{seed:>6} | " + " | ".join(f"{v:>{col_w}d}" for v in row_vals)
        print(row)

    print(sep)

    # ── 汇总统计 ──────────────────────────────────────────────── #
    means = [float(np.mean(results[n])) for n in algo_names]
    stds  = [float(np.std(results[n]))  for n in algo_names]
    mins  = [float(np.min(results[n]))  for n in algo_names]
    maxs  = [float(np.max(results[n]))  for n in algo_names]

    def stat_row(label, vals):
        return f"{label:>6} | " + " | ".join(f"{v:>{col_w}.2f}" for v in vals)

    print(stat_row("Mean", means))
    print(stat_row("Std",  stds))
    print(stat_row("Min",  mins))
    print(stat_row("Max",  maxs))
    print(sep)

    # ── 改善率（相对 Greedy）─────────────────────────────────── #
    greedy_mean = means[0]
    print(f"\n  相对 Greedy (mean={greedy_mean:.2f}) 的改善率：")
    for name, mean in zip(algo_names[1:], means[1:]):
        pct  = (greedy_mean - mean) / max(greedy_mean, 1e-9) * 100
        mark = "↓ 更好" if pct >= 0 else "↑ 更差"
        print(f"    {name:>10s}: {pct:+6.1f}%  {mark}  (mean={mean:.2f})")

    # ── 最优 ─────────────────────────────────────────────────── #
    best_idx = int(np.argmin(means))
    print(f"\n  [★] 最优算法: {algo_names[best_idx]}  (mean shifters = {means[best_idx]:.2f})")
    print()


if __name__ == "__main__":
    main()
