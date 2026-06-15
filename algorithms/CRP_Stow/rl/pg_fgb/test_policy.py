"""
PG-FGB 训练后模型的逐步测试 / 批量基准评估脚本

用法（在 Platform_CRISP 根目录下执行）：
    python -m algorithms.CRP_Stow.rl.pg_fgb.test_policy [选项]

模式 1 — 随机实例逐步演示（默认）：
    python -m algorithms.CRP_Stow.rl.pg_fgb.test_policy --seed 42 --pause

模式 2 — 单个 .pro 文件逐步演示：
    python -m algorithms.CRP_Stow.rl.pg_fgb.test_policy \\
        --pro-file /path/to/crp_stow/Data/Gen/Bay-5-5-8-6_0.pro --pause

模式 3 — 批量评估整个目录（按规格汇总）：
    python -m algorithms.CRP_Stow.rl.pg_fgb.test_policy \\
        --pro-dir /path/to/crp_stow/Data/Gen \\
        --size-filters "Bay-3-,Bay-5-3-,Bay-5-5-" \\
        --test-seeds "32,33,34,35,36,37,38,39"

常用选项：
    --model       trained_models/pg_fgb_policy.pt  模型路径
    --seed        42       随机实例种子
    --pause                逐步交互模式
    --greedy-high          顶部即目标时跳过模型直接取走
    --pro-file    PATH     使用指定 .pro 文件（单文件模式）
    --pro-dir     DIR      批量评估目录下所有 .pro 文件
    --size-filters STR     规格过滤，逗号分隔子串
    --test-seeds  STR      测试 seed 编号，逗号分隔
"""
from __future__ import annotations

import argparse
import sys
import os
import re
import collections
from pathlib import Path

# ── 路径设置 ───────────────────────────────────────────────── #
_ALGO_DIR = Path(os.path.abspath(__file__)).parent          # .../pg_fgb/
_ROOT     = _ALGO_DIR.parents[4]                            # Platform_CRISP/
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_DEFAULT_MODEL = str(_ALGO_DIR / "trained_models" / "pg_fgb_policy.pt")


# ────────────────────────────────────────────────────────────────
#  打印辅助
# ────────────────────────────────────────────────────────────────

def _print_yard(env, title: str = "") -> None:
    """打印堆场当前状态（每垛一列，底→顶）。"""
    cfg      = env.config
    n_stacks = cfg.num_bays

    if title:
        print(f"\n{'─'*60}")
        print(f"  {title}")
        print(f"{'─'*60}")

    # 列头
    header = "  Tier | " + " | ".join(f"S{i+1:02d}" for i in range(n_stacks))
    print(header)
    print("  " + "-" * (len(header) - 2))

    max_h = cfg.max_tiers
    for tier in range(max_h, 0, -1):
        cells = []
        for i in range(n_stacks):
            key  = env._idx_to_stack(i)
            stk  = env.yard.stacks.get(key)
            cont = stk.containers[tier - 1] if (stk and tier <= stk.height) else None
            if cont is None:
                cells.append("----")
            else:
                vs, vt = cont.group, cont.priority
                tag    = chr(ord('A') + vs) + str(vt)   # e.g. "A0", "B3"
                # Mark retrievable containers
                mark   = "*" if env._is_retrievable(cont) else " "
                cells.append(f"{mark}{tag:3s}")
        tier_lbl = f"T{tier:2d}"
        print(f"  {tier_lbl} | " + " | ".join(f"{c:4s}" for c in cells))

    # 堆高
    heights = " ".join(
        f"S{i+1}:{env.yard.stacks.get(env._idx_to_stack(i), type('', (), {'height': 0})()).height if hasattr(env.yard.stacks.get(env._idx_to_stack(i), None), 'height') else 0}"
        for i in range(n_stacks)
    )
    print(f"         heights: " + "  ".join(
        f"S{i+1}:{env.yard.stacks[env._idx_to_stack(i)].height}"
        if env._idx_to_stack(i) in env.yard.stacks else f"S{i+1}:0"
        for i in range(n_stacks)
    ))
    print()


def _print_vessel_state(env) -> None:
    """打印船舱装载进度和当前目标信息。"""
    loaded  = env._vessel_loaded
    max_t   = env._vessel_max_tier
    vs_info = "  ".join(
        f"{chr(ord('A')+j)}({loaded[j]}/{max_t[j]})"
        for j in range(len(loaded))
    )
    print(f"  船舱进度: {vs_info}")
    if env._mode == "low" and env._target_container is not None:
        tc = env._target_container
        print(f"  LOW 目标: {chr(ord('A')+tc.group)}{tc.priority}  "
              f"源垛=S{env._source_stack_idx+1}")
    elif env._mode == "high":
        print(f"  HIGH 阶段：选择含可取容器的垛")


# ────────────────────────────────────────────────────────────────
#  单实例逐步演示
# ────────────────────────────────────────────────────────────────

def run_single(
    model_path: str,
    seed:       int,
    pause:      bool,
    greedy_high: bool,
    pro_file:   str | None = None,
    num_bays:   int = 8,
    max_tiers:  int = 6,
    num_containers: int = 29,
    num_groups: int = 5,
) -> None:
    """单实例（随机或 .pro 文件）逐步贪心推演，可暂停交互。"""
    import torch
    import numpy as np

    from core.base_problem import ProblemConfig
    from problems.CRP_Stow import CRP_Stow
    from algorithms.CRP_Stow.rl.pg_fgb.network import StowagePolicyNetwork
    from algorithms.CRP_Stow.rl.pg_fgb.algorithm import _greedy_high_override

    if not os.path.isfile(model_path):
        print(f"[ERROR] 模型文件不存在: {model_path}")
        sys.exit(1)

    ckpt = torch.load(model_path, map_location="cpu")
    print(f"[INFO] 加载模型: {model_path}")
    print(f"       训练最优 greedy relocations = {ckpt.get('best_shifters', '?')}")

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

    cfg = ProblemConfig(
        num_bays       = ckpt.get("num_bays",       num_bays),
        num_rows       = 1,
        max_tiers      = ckpt.get("max_tiers",      max_tiers),
        num_containers = ckpt.get("num_containers", num_containers),
        num_groups     = ckpt.get("num_groups",     num_groups),
        seed           = seed,
    )
    if pro_file:
        cfg.extra["layout_file_path"] = pro_file
        print(f"[INFO] 使用 .pro 文件: {pro_file}")

    env = CRP_Stow(config=cfg)
    obs, info = env.reset(seed=seed)

    print(f"\n{'═'*62}")
    print(f"  TEST  seed={seed}  mode={'pro-file' if pro_file else 'random'}")
    print(f"  堆场: {env.config.num_bays} 垛 × {env.config.max_tiers} 层")
    print(f"  箱子: N={sum(not s.is_empty and 1 or 0 for s in env.yard.stacks.values()) * 0 + sum(len(s.containers) for s in env.yard.stacks.values())}  船舱垛 VS={env.config.num_groups}")
    print(f"{'═'*62}")
    _print_yard(env, "初始堆场（* = 可取出）")
    _print_vessel_state(env)

    if pause:
        input("\n  [Enter] 开始逐步执行...")

    step_num = 0
    done     = False
    while not done:
        mask_raw = info.get("action_mask", np.ones(env.action_space.n, dtype=bool))

        # ── Greedy HIGH override ─────────────────────────────── #
        action = None
        if greedy_high:
            free_action = _greedy_high_override(env, mask_raw)
            if free_action is not None:
                action = free_action

        # ── 模型决策 ─────────────────────────────────────────── #
        if action is None:
            forbidden = torch.tensor(~mask_raw.astype(bool), dtype=torch.bool).unsqueeze(0)
            obs_t     = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                enc_out     = policy.encode(obs_t)
                action_t, _ = policy(obs_t, forbidden, greedy=True, encoder_output=enc_out)
            action = int(action_t.item())

        phase    = env._mode
        src_idx  = env._source_stack_idx
        dst_key  = env._idx_to_stack(action)
        src_key  = env._idx_to_stack(src_idx) if src_idx is not None else None

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step_num += 1

        # ── 描述本步 ─────────────────────────────────────────── #
        if phase == "high":
            stk  = env.yard.stacks.get(dst_key)
            desc = f"HIGH → S{action+1}  (选取源垛)"
        else:
            desc = (f"LOW ★RELOCATE★  "
                    f"S{src_idx+1 if src_idx is not None else '?'} → S{action+1}  "
                    f"reward={reward:+.0f}")

        print(f"\n  Step {step_num:3d}  │  {desc}")

        m = env.get_metrics()
        print(f"  累计搬运={m['relocations']:.0f}  "
              f"已装船={m['retrieved']:.0f}  "
              f"利用率={m['vessel_utilisation']*100:.1f}%")

        if not done:
            _print_vessel_state(env)

        if pause and not done:
            cmd = input("\n  [Enter] 下一步 / [q] 退出: ").strip().lower()
            if cmd == "q":
                print("  用户中断。")
                break

    final = env.get_metrics()
    print(f"\n{'═'*62}")
    print(f"  完成  共 {step_num} 步")
    print(f"  总搬运次数 (relocations) = {final['relocations']:.0f}")
    print(f"  已装船箱数               = {final['retrieved']:.0f}")
    print(f"  装船利用率               = {final['vessel_utilisation']*100:.1f}%")
    print(f"{'═'*62}")


# ────────────────────────────────────────────────────────────────
#  批量基准评估
# ────────────────────────────────────────────────────────────────

def run_batch(
    model_path:   str,
    pro_dir:      str,
    size_filters: list[str],
    test_seeds:   list[int] | None,
    greedy_high:  bool = True,
    seed:         int  = 0,
) -> None:
    """
    对 pro_dir 下（过滤后的）所有 .pro 文件批量贪心推演，
    按规格分组打印平均搬运次数。
    """
    import torch
    import numpy as np

    from core.base_problem import ProblemConfig
    from problems.CRP_Stow import CRP_Stow
    from algorithms.CRP_Stow.rl.pg_fgb.network import StowagePolicyNetwork
    from algorithms.CRP_Stow.rl.pg_fgb.algorithm import (
        _scan_pro_files, _prepare_pro_episode,
        _run_episode_greedy,
    )

    if not os.path.isfile(model_path):
        print(f"[ERROR] 模型文件不存在: {model_path}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(model_path, map_location=device)

    feature_scale = torch.tensor(ckpt["feature_scale"], dtype=torch.float32, device=device)
    policy = StowagePolicyNetwork(
        embed_dim      = ckpt["embed_dim"],
        num_enc_layers = ckpt["num_enc_layers"],
        num_heads      = ckpt["num_heads"],
        ffn_dim        = ckpt["ffn_dim"],
        clip_constant  = ckpt["clip_constant"],
        feature_scale  = feature_scale,
    ).to(device)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()

    print(f"[批量评估] 模型: {model_path}")
    print(f"           训练最优 relocations = {ckpt.get('best_shifters', '?')}")

    files = _scan_pro_files(
        pro_dir,
        size_filters=size_filters if size_filters else None,
        seed_indices=test_seeds,
    )
    if not files:
        print(f"[批量评估] 未找到匹配文件  "
              f"(dir={pro_dir}, filters={size_filters}, seeds={test_seeds})")
        return

    print(f"[批量评估] 共 {len(files)} 个文件\n")

    cfg = ProblemConfig(seed=seed, num_bays=1, num_rows=1, max_tiers=6,
                        num_containers=1, num_groups=1)
    env = CRP_Stow(config=cfg)

    pat_size  = re.compile(r'^(.+)_\d+\.pro$')
    results   = collections.defaultdict(list)

    for i, pro_path in enumerate(files, 1):
        m        = pat_size.match(pro_path.name)
        size_key = m.group(1) if m else pro_path.stem
        _prepare_pro_episode(env, pro_path)
        env.config.seed = seed
        relocs = -_run_episode_greedy(env, policy, device, greedy_high=greedy_high)
        results[size_key].append(relocs)
        if i % 50 == 0 or i == len(files):
            print(f"  进度: {i}/{len(files)}")

    print("\n" + "=" * 72)
    print(f"  PG-FGB 批量评估  (greedy_high={greedy_high})")
    print("=" * 72)
    print(f"  {'规格':<30}  {'N':>5}  {'平均':>10}  {'最小':>8}  {'最大':>8}")
    print("  " + "-" * 68)

    all_vals: list[float] = []
    for size_key in sorted(results):
        vals = results[size_key]
        all_vals.extend(vals)
        print(f"  {size_key:<30}  {len(vals):>5}  "
              f"{np.mean(vals):>10.3f}  {np.min(vals):>8.0f}  {np.max(vals):>8.0f}")

    print("  " + "-" * 68)
    if all_vals:
        print(f"  {'合计':<30}  {len(all_vals):>5}  "
              f"{np.mean(all_vals):>10.3f}  "
              f"{np.min(all_vals):>8.0f}  {np.max(all_vals):>8.0f}")
    print("=" * 72)


# ────────────────────────────────────────────────────────────────
#  CLI 入口
# ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PG-FGB 模型测试（逐步 / 批量）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 随机实例逐步演示
  python -m algorithms.CRP_Stow.rl.pg_fgb.test_policy --seed 42 --pause

  # 指定 .pro 文件逐步演示
  python -m algorithms.CRP_Stow.rl.pg_fgb.test_policy \\
      --pro-file /data/liuw2/Platform_CRISP/benchmark/crp_stow/Data/Gen/Bay-5-5-8-6_0.pro --pause

  # 批量评估（小规格全部测试 seed）
  python -m algorithms.CRP_Stow.rl.pg_fgb.test_policy \\
      --pro-dir /data/liuw2/Platform_CRISP/benchmark/crp_stow/Data/Gen \\
      --size-filters "Bay-3-,Bay-5-3-,Bay-5-5-" \\
      --test-seeds "32,33,34,35,36,37,38,39"

  # 批量评估（全部规格，测试种子 32-39）
  python -m algorithms.CRP_Stow.rl.pg_fgb.test_policy \\
      --pro-dir /data/liuw2/Platform_CRISP/benchmark/crp_stow/Data/Gen \\
      --test-seeds "32,33,34,35,36,37,38,39"
""")
    parser.add_argument("--model",  default=_DEFAULT_MODEL,
                        help=f"模型文件路径（默认: {_DEFAULT_MODEL}）")
    parser.add_argument("--seed",   type=int, default=42,
                        help="随机实例种子（默认: 42）")
    parser.add_argument("--pause",  action="store_true",
                        help="逐步交互模式（每步按 Enter）")
    parser.add_argument("--greedy-high", action="store_true", dest="greedy_high",
                        help="顶部即目标时跳过模型直接取走（推荐）")
    # .pro 文件相关
    parser.add_argument("--pro-file", default=None, dest="pro_file",
                        help="单个 .pro 文件路径（逐步演示）")
    parser.add_argument("--pro-dir",  default=None, dest="pro_dir",
                        help="批量评估：.pro 文件目录")
    parser.add_argument("--size-filters", default="", dest="size_filters",
                        help="规格过滤子串，逗号分隔（如 'Bay-3-,Bay-5-3-'）")
    parser.add_argument("--test-seeds", default="", dest="test_seeds",
                        help="测试 seed 编号，逗号分隔（如 '32,33,34,35,36,37,38,39'）")
    # 随机实例参数（pro-file 不指定时使用）
    parser.add_argument("--bays",       type=int, default=8,  help="堆场垛数")
    parser.add_argument("--tiers",      type=int, default=6,  help="最大层高")
    parser.add_argument("--containers", type=int, default=29, help="箱子总数")
    parser.add_argument("--groups",     type=int, default=5,  help="船舱垛数")

    args = parser.parse_args()

    # 解析过滤器
    size_filters = [s.strip() for s in args.size_filters.split(",") if s.strip()]
    test_seeds   = ([int(s.strip()) for s in args.test_seeds.split(",") if s.strip()]
                    if args.test_seeds else None)

    if args.pro_dir:
        # 批量评估模式
        run_batch(
            model_path   = args.model,
            pro_dir      = args.pro_dir,
            size_filters = size_filters,
            test_seeds   = test_seeds,
            greedy_high  = args.greedy_high,
            seed         = args.seed,
        )
    else:
        # 单实例逐步演示模式
        run_single(
            model_path     = args.model,
            seed           = args.seed,
            pause          = args.pause,
            greedy_high    = args.greedy_high,
            pro_file       = args.pro_file,
            num_bays       = args.bays,
            max_tiers      = args.tiers,
            num_containers = args.containers,
            num_groups     = args.groups,
        )
