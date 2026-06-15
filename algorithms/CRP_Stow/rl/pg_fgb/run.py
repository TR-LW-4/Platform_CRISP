"""
CRP-Stow × PG-FGB 独立训练脚本
================================
直接运行（从 Platform_CRISP 根目录）：

    python -m algorithms.CRP_Stow.rl.pg_fgb.run

所有超参数都在本文件的 CONFIG 区域修改，无需触碰 algorithm.py。
训练好的模型保存到本脚本所在目录的 trained_models/ 子目录：
    algorithms/CRP_Stow/rl/pg_fgb/trained_models/pg_fgb_policy.pt

数据模式
--------
1. 随机生成（默认）   — BENCHMARK['train_root'] = ""
   训练时每轮随机生成 BRLP 实例，适合快速验证网络结构。

2. .pro 基准文件训练  — BENCHMARK['train_root'] = "/path/to/crp_stow/Data/Gen"
   从 Jovanović (2019) 数据集的 .pro 文件循环采样训练实例，
   与论文中的基准测试（Tanaka & Voß 2019）完全匹配。
   训练集/测试集按 seed 编号分割（默认 0-31 训练，32-39 测试）。

3. 训练后基准评估     — BENCHMARK['test_root'] = "/path/to/crp_stow/Data/Gen"
   训练结束后，对 test_root 下所有（或过滤后的）.pro 文件运行贪心推演，
   按实例规格分组报告平均搬运次数，与 Jovanović / Tanaka 论文结果对比。
"""

import sys
import os
import pathlib
import multiprocessing as mp

# ── 项目根路径 ───────────────────────────────────────────────── #
_ALGO_DIR = pathlib.Path(__file__).parent          # .../pg_fgb/
_ROOT     = _ALGO_DIR.parents[4]                   # Platform_CRISP/
sys.path.insert(0, str(_ROOT))

# ════════════════════════════════════════════════════════════════
#  CONFIG — 在这里修改所有参数
# ════════════════════════════════════════════════════════════════

# ── 堆场 / 问题参数（仅在随机生成模式下使用）───────────────── #
YARD = dict(
    num_bays       = 8,     # 堆场垛数 (YS)
    num_rows       = 1,     # 固定为 1（BRLP 一维堆场）
    max_tiers      = 6,     # 最大层高 (YT)
    num_containers = 29,    # 堆场总箱数 N
    num_groups     = 5,     # 船舱垛数 (VS)
)

# ── 基准数据集配置 ───────────────────────────────────────────── #
# 设置 train_root / test_root 为 .pro 文件所在目录（留空 = 随机生成）
BENCHMARK = dict(
    # 训练数据根目录（留空则使用上面 YARD 参数随机生成）
    train_root   = "/data/liuw2/Platform_CRISP/benchmark/crp_stow/Data/Gen",

    # 测试/评估数据根目录（留空则跳过训练后基准评估）
    test_root    = "/data/liuw2/Platform_CRISP/benchmark/crp_stow/Data/Gen",

    # 规格过滤器：只使用文件名含这些字符串的 .pro 文件（空列表 = 全部使用）
    # 例如：["Bay-3-", "Bay-5-3-", "Bay-5-5-"] 只用小规模实例
    # 推荐训练时先用小实例，再换大实例精调
    size_filters = [
        "Bay-3-",    # VS=3 系列（最小：N=10-46）
        "Bay-5-3-",  # VS=5, small YS
        "Bay-5-5-",  # VS=5, medium YS
    ],

    # 训练集 seed 编号（文件名末尾 _N.pro 中的 N）
    # 默认取 0-31（每个规格 40 个实例中的 80%）
    train_seeds  = list(range(32)),    # 0..31 → 训练

    # 测试集 seed 编号
    test_seeds   = list(range(32, 40)),  # 32..39 → 测试
)

# ── 训练超参数 ───────────────────────────────────────────────── #
TRAIN = dict(
    num_iterations    = 500,    # 总迭代轮数
    episodes_per_iter = 8,      # 每轮收集的 episode 数
    eval_episodes     = 8,      # 每轮 greedy 评估的 episode 数
    learning_rate     = 2.5e-4, # Adam 学习率
    ent_coef          = 0.01,   # 熵正则化系数
    p_value_threshold = 0.4,    # baseline 更新的 t-test p-value 阈值
    anneal_lr         = True,   # 是否线性衰减学习率
    greedy_high       = True,   # 顶部即目标时直接取走，不经过模型
    report_every      = 10,     # 每隔多少轮打印一次训练进度
)

# ── 网络结构 ─────────────────────────────────────────────────── #
NETWORK = dict(
    embed_dim      = 128,
    num_enc_layers = 2,
    num_heads      = 4,
    ffn_dim        = 256,
    clip_constant  = 10.0,
)

# ── 随机种子 ─────────────────────────────────────────────────── #
SEED = 0


# ════════════════════════════════════════════════════════════════
#  训练后基准评估
# ════════════════════════════════════════════════════════════════

def evaluate_benchmark(
    model_path: str,
    test_root:  str,
    size_filters: list,
    test_seeds:   list,
    greedy_high:  bool = True,
    device_str:   str  = "auto",
) -> None:
    """
    用训练好的模型对 .pro 基准文件进行批量贪心推演，
    按规格（YS-YT-VS 组合）分组报告平均搬运次数。

    输出格式与 Tanaka & Voß (2019) Table 5/6 对应。
    """
    import re
    import collections
    import torch
    import numpy as np

    from core.base_problem import ProblemConfig
    from problems.CRP_Stow import CRP_Stow
    from algorithms.CRP_Stow.rl.pg_fgb.network import StowagePolicyNetwork
    from algorithms.CRP_Stow.rl.pg_fgb.algorithm import (
        _scan_pro_files, _prepare_pro_episode,
        _run_episode_greedy, _get_feature_scale,
    )

    if not pathlib.Path(model_path).exists():
        print(f"[评估] 模型文件不存在: {model_path}，跳过基准评估。")
        return

    # ── 加载模型 ─────────────────────────────────────────────── #
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    ckpt = torch.load(model_path, map_location=device)
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
    print(f"\n[评估] 已加载模型: {model_path}")
    print(f"       训练最优 greedy relocations = {ckpt.get('best_shifters', '?'):.3f}")

    # ── 扫描测试文件 ─────────────────────────────────────────── #
    test_files = _scan_pro_files(
        test_root,
        size_filters=size_filters if size_filters else None,
        seed_indices=test_seeds if test_seeds else None,
    )
    if not test_files:
        print(f"[评估] 未找到测试文件（root={test_root}, filters={size_filters}, "
              f"seeds={test_seeds}），跳过。")
        return

    print(f"[评估] 共 {len(test_files)} 个测试实例  "
          f"(filters={size_filters or 'none'}, seeds={test_seeds or 'all'})")

    # ── 构建环境 ─────────────────────────────────────────────── #
    cfg = ProblemConfig(seed=SEED, num_bays=1, num_rows=1, max_tiers=6,
                        num_containers=1, num_groups=1)
    env = CRP_Stow(config=cfg)

    # ── 逐文件推演 ───────────────────────────────────────────── #
    # 按规格名分组：文件名格式 Bay-{VS}-{X}-{YS}-{YT}_{seed}.pro
    # 规格 key = 去掉 _{seed}.pro 部分
    pat_size = re.compile(r'^(.+)_\d+\.pro$')
    results  = collections.defaultdict(list)   # size_key → [relocs, ...]

    for pro_path in test_files:
        m = pat_size.match(pro_path.name)
        size_key = m.group(1) if m else pro_path.stem

        _prepare_pro_episode(env, pro_path)
        env.config.seed = SEED   # 固定 seed（.pro 文件决定布局，seed 不影响初始状态）

        relocs = -_run_episode_greedy(env, policy, device, greedy_high=greedy_high)
        results[size_key].append(relocs)

    # ── 汇总输出 ─────────────────────────────────────────────── #
    print("\n" + "=" * 72)
    print(f"  PG-FGB 基准评估结果  (greedy_high={greedy_high})")
    print("=" * 72)
    print(f"  {'规格':<28}  {'实例数':>6}  {'平均搬运':>10}  {'最小':>8}  {'最大':>8}")
    print("  " + "-" * 68)

    all_relocs = []
    for size_key in sorted(results):
        vals = results[size_key]
        all_relocs.extend(vals)
        print(f"  {size_key:<28}  {len(vals):>6}  "
              f"{np.mean(vals):>10.3f}  {np.min(vals):>8.0f}  {np.max(vals):>8.0f}")

    print("  " + "-" * 68)
    if all_relocs:
        print(f"  {'合计':28}  {len(all_relocs):>6}  "
              f"{np.mean(all_relocs):>10.3f}  "
              f"{np.min(all_relocs):>8.0f}  {np.max(all_relocs):>8.0f}")
    print("=" * 72)


# ════════════════════════════════════════════════════════════════
#  训练入口
# ════════════════════════════════════════════════════════════════

def main():
    from core.base_problem   import ProblemConfig
    from core.base_algorithm import AlgorithmConfig
    from problems.CRP_Stow   import CRP_Stow
    from algorithms.CRP_Stow.rl.pg_fgb.algorithm import PgFgbStow

    # ── 模型保存路径 ─────────────────────────────────────────── #
    save_dir  = _ALGO_DIR / "trained_models"
    save_dir.mkdir(exist_ok=True)
    save_path = str(save_dir / "pg_fgb_policy.pt")

    train_root   = BENCHMARK.get("train_root",   "")
    test_root    = BENCHMARK.get("test_root",    "")
    size_filters = BENCHMARK.get("size_filters", [])
    train_seeds  = BENCHMARK.get("train_seeds",  None)
    test_seeds   = BENCHMARK.get("test_seeds",   None)

    use_pro_train = bool(train_root)

    # ── 问题配置（随机模式才使用 YARD 参数）────────────────────── #
    cfg_p = ProblemConfig(seed=SEED, **YARD)

    # ── 算法配置 ─────────────────────────────────────────────── #
    n_iters = TRAIN["num_iterations"]
    cfg_a   = AlgorithmConfig(
        max_iterations  = n_iters,
        report_interval = max(1, n_iters // TRAIN["report_every"]),
        seed            = SEED,
    )
    cfg_a.extra.update({
        "num_iterations":    n_iters,
        "save_path":         save_path,
        "report_every":      TRAIN["report_every"],
        **{k: v for k, v in TRAIN.items()
           if k not in ("num_iterations", "report_every")},
        **NETWORK,
        # .pro 基准训练参数
        "pro_train_root":    train_root,
        "pro_size_filters":  size_filters,
        "pro_train_seeds":   train_seeds,
    })

    # ── 打印配置摘要 ─────────────────────────────────────────── #
    print("=" * 72)
    print("  CRP-Stow × PG-FGB 训练")
    print("=" * 72)
    if use_pro_train:
        print(f"  数据来源 : .pro 基准文件  root={train_root}")
        print(f"  规格过滤 : {size_filters or '全部'}")
        print(f"  训练种子 : {train_seeds if train_seeds else '全部'}")
    else:
        print(f"  数据来源 : 随机生成  "
              f"{YARD['num_bays']}垛 × {YARD['max_tiers']}层  "
              f"N={YARD['num_containers']}  VS={YARD['num_groups']}")
    print(f"  迭代次数 : {n_iters}  lr={TRAIN['learning_rate']}  "
          f"greedy_high={TRAIN['greedy_high']}")
    print(f"  模型路径 : {save_path}")
    if test_root:
        print(f"  训练后将对 test_root 下 .pro 文件运行基准评估")
        print(f"  测试种子 : {test_seeds if test_seeds else '全部'}")
    print("=" * 72)

    inst = PgFgbStow(cfg_a)
    q    = mp.Queue()
    ev   = mp.Event()

    def factory():
        return CRP_Stow(config=cfg_p)

    proc = mp.Process(target=inst.train, args=(factory, q, ev), daemon=True)
    proc.start()

    while proc.is_alive():
        try:
            r = q.get(timeout=1.0)
            m = r.metrics
            print(
                f"  step={r.step:5d}/{n_iters}"
                f"  relocations={m.get('shifters', r.metric):.2f}"
                f"  best={m.get('best_shifters', float('inf')):.2f}"
                f"  loss={m.get('policy_loss', float('nan')):.4f}"
                f"  entropy={m.get('entropy', float('nan')):.4f}"
                f"  adv={m.get('mean_advantage', float('nan')):+.4f}"
                f"  bl_upd={int(m.get('baseline_updates', 0))}"
                f"  [{r.progress*100:.1f}%]"
            )
        except Exception:
            pass

    proc.join()
    ev.set()

    if pathlib.Path(save_path).exists():
        print(f"\n[✓] 模型已保存 → {save_path}")
    else:
        print("\n[!] 模型未能保存，请检查训练过程")

    # ── 训练后基准评估 ────────────────────────────────────────── #
    if test_root and pathlib.Path(save_path).exists():
        evaluate_benchmark(
            model_path   = save_path,
            test_root    = test_root,
            size_filters = size_filters,
            test_seeds   = test_seeds,
            greedy_high  = TRAIN["greedy_high"],
        )

    print("Done.")


if __name__ == "__main__":
    main()
