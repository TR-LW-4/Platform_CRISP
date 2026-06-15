"""
CRP-D × PG-FGB 训练脚本（实验一：dup_dataset）
===============================================
实验定位
--------
CRP-D（Duplicate-group CRP）：集装箱只有 stack 和 tier 位置信息，
不涉及真实船舱 bay / row 坐标。观测特征为：
    [stack_id,  row(=1固定),  height,  has_target,  top_group]
与 CRP-Stow（实验二）的区别：去掉了 bay × row × tier 船舱空间约束，
只用顺序组号（Group 1 → Group 2 → ...）确定取箱顺序。

数据来源
--------
ZhuDup benchmark（dup_dataset），文件夹命名规则：G-S-N
    G = 组数，S = stack 数，N = 总箱子数
路径：Platform_CRISP/benchmark/dup_dataset/

直接运行（从 Platform_CRISP 根目录）：

  # 正式训练（S=6，1000 轮）：
    python -m algorithms.CRP_D.rl.pg_fgb.run

  # 快速验证（S=3，100 轮，几分钟内看到收敛曲线）：
    python -m algorithms.CRP_D.rl.pg_fgb.run --quick

训练好的模型保存到：
    algorithms/CRP_D/rl/pg_fgb/trained_models/pg_fgb_crpd.pt
    algorithms/CRP_D/rl/pg_fgb/trained_models/pg_fgb_crpd_quick.pt  (quick 模式)
"""

import sys
import os
import pathlib
import multiprocessing as mp
import argparse

# ── 项目根路径 ───────────────────────────────────────────────── #
_ALGO_DIR = pathlib.Path(__file__).parent          # .../pg_fgb/
_ROOT     = _ALGO_DIR.parents[3]                   # Platform_CRISP/
sys.path.insert(0, str(_ROOT))

# ════════════════════════════════════════════════════════════════
#  QUICK_TEST 开关
#  运行 --quick 时使用 S=3 的小实例、100 轮，几分钟内看到收敛效果。
#  正式训练去掉 --quick 即可切回 S=6 / 1000 轮。
# ════════════════════════════════════════════════════════════════
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--quick", action="store_true",
                     help="快速验证模式：S=3, 100轮, 少 episode, 短时间看收敛")
_args, _ = _parser.parse_known_args()
QUICK_TEST = _args.quick

# ════════════════════════════════════════════════════════════════
#  CONFIG — 在这里修改所有参数
# ════════════════════════════════════════════════════════════════

# ── dup_dataset 路径 ──────────────────────────────────────────── #
DUP_ROOT = str(_ROOT / "benchmark" / "dup_dataset")

# ── 训练数据集筛选 ────────────────────────────────────────────── #
# quick 模式：只用 S=6；正式训练：混合 S=6~10 提高泛化能力
DUP_STACKS_LIST = [6] if QUICK_TEST else [6, 7, 8, 9, 10]

# 只使用 T=6 的文件（和论文测试集一致，减少噪声；None=不限制）
DUP_TIERS = 6 if not QUICK_TEST else None

# 每个文件夹最多取多少文件（控制训练集总量）
# T=6, S=6~10, 4 alphas: 5S×6C×4α = 120 folders
# 40 files/folder → 120×40 = 4800 ≈ 5000；None 或 0 = 全取
DUP_MAX_FILES_PER_FOLDER = 40 if not QUICK_TEST else 0

# 兼容旧接口：取第一个 S 作为环境初始值
DUP_STACKS = DUP_STACKS_LIST[0]

# ── 堆场参数（初始值，训练时由 dup 文件自动覆盖）────────────── #
YARD = dict(
    num_bays       = DUP_STACKS,
    num_rows       = 1,
    max_tiers      = 10,   # 训练时由 dup 文件自动覆盖为实际 H
    num_containers = 30,
    num_groups     = 3,
)

# ── 船位参数 ─────────────────────────────────────────────────── #
VESSEL = dict(
    vessel_bays  = 1,
    vessel_rows  = 80,
    vessel_tiers = 1,
)

# ── 特征归一化（CRP-D 专用）──────────────────────────────────── #
# 5 维节点特征：[stack_id(bay), row(=1固定), height, has_target, group]
FEATURE_SCALE = "10,80,10,1,10"

# ── 训练超参数 ───────────────────────────────────────────────── #
if QUICK_TEST:
    TRAIN = dict(
        num_iterations    = 100,    # 快速验证：100 轮（约 2-5 分钟）
        episodes_per_iter = 8,      # 少一点 episode 加速
        eval_episodes     = 8,
        learning_rate     = 3e-4,   # 小实例可以稍高一点 lr
        ent_coef          = 0.02,   # 稍高熵鼓励探索
        p_value_threshold = 0.05,   # 统计显著性阈值（稳定）
        gamma             = 1.0,    # 折扣因子
        anneal_lr         = True,
        greedy_high       = True,
        report_every      = 10,     # 每 10 轮打印
    )
else:
    TRAIN = dict(
        num_iterations    = 1000,
        episodes_per_iter = 16,
        eval_episodes     = 16,
        learning_rate     = 2.5e-4,
        ent_coef          = 0.01,
        p_value_threshold = 0.05,   # 改为 0.05（原 0.4 太激进）
        gamma             = 1.0,    # 折扣因子（1.0=无折扣，用 sum-to-go）
        anneal_lr         = True,
        greedy_high       = True,
        report_every      = 20,
    )

# ── 网络结构 ─────────────────────────────────────────────────── #
NETWORK = dict(
    embed_dim      = 64 if QUICK_TEST else 128,   # quick 时缩小网络加速
    num_enc_layers = 2,
    num_heads      = 4,
    ffn_dim        = 128 if QUICK_TEST else 256,
    clip_constant  = 10.0,
)

# ── 固定评估集（训练时用，替代随机采样）──────────────────────── #
# T=6, α=0.8, S=6~10；每个(S, N)文件夹取 FILES_PER_COMBO 个文件
# quick 模式时关闭（避免 OOD 评估干扰快速验证）
FIXED_EVAL = dict(
    enabled        = not QUICK_TEST,      # 正式训练开启，quick 模式关闭
    dup_root       = DUP_ROOT,
    alpha          = 0.8,                 # 测试最难的 alpha
    tiers          = 6,                   # T=6（和论文对比组一致）
    stacks         = [6, 7, 8, 9, 10],    # 所有 S 值
    files_per_combo = 2,                  # 每个(S,N)子文件夹取2个文件
                                          # 共 5S×6C×2 = 60 个实例，评估约 17s
)

# ── 随机种子 ─────────────────────────────────────────────────── #
SEED = 0

# ════════════════════════════════════════════════════════════════
#  训练入口
# ════════════════════════════════════════════════════════════════

def main():
    from core.base_problem   import ProblemConfig
    from core.base_algorithm import AlgorithmConfig
    from problems.CRP_D      import CRP_D
    from algorithms.CRP_D.rl.pg_fgb.algorithm import PgFgbD

    # ── 模型保存路径 ─────────────────────────────────────────── #
    save_dir = _ALGO_DIR / "trained_models"
    save_dir.mkdir(exist_ok=True)
    model_name = "pg_fgb_crpd_quick.pt" if QUICK_TEST else "pg_fgb_crpd.pt"
    save_path = str(save_dir / model_name)

    # ── 问题配置（初始值；训练时由 dup 文件覆盖）────────────── #
    cfg_p = ProblemConfig(seed=SEED, **YARD, **VESSEL)

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
        # ── dup_dataset 相关 ── #
        "dup_train_root":       DUP_ROOT,
        "dup_num_stacks_list":  DUP_STACKS_LIST,   # mixed-S training
        "dup_num_stacks":       DUP_STACKS,         # kept for backward compat
        # ── 特征归一化 ── #
        "feature_scale":     FEATURE_SCALE,
        **{k: v for k, v in TRAIN.items()
           if k not in ("num_iterations", "report_every")},
        **NETWORK,
    })

    inst = PgFgbD(cfg_a)
    q    = mp.Queue()
    ev   = mp.Event()

    def factory():
        return CRP_D(config=cfg_p)

    # ── 扫描并统计可用 dup 文件（所有训练 S 值，按 T 和文件数过滤）─ #
    from algorithms.CRP_D.rl.pg_fgb.algorithm import _scan_dup_files, _build_fixed_eval_set
    dup_files = []
    for s_val in DUP_STACKS_LIST:
        dup_files.extend(_scan_dup_files(
            DUP_ROOT, s_val,
            target_H            = DUP_TIERS,
            max_files_per_folder= DUP_MAX_FILES_PER_FOLDER,
        ))

    # ── 构建固定评估集 ────────────────────────────────────────── #
    fixed_eval_set = []
    if FIXED_EVAL["enabled"]:
        fixed_eval_set = _build_fixed_eval_set(
            dup_root       = FIXED_EVAL["dup_root"],
            alpha          = FIXED_EVAL["alpha"],
            tiers          = FIXED_EVAL["tiers"],
            stacks         = FIXED_EVAL["stacks"],
            files_per_combo= FIXED_EVAL["files_per_combo"],
        )
        cfg_a.extra["fixed_eval_set"] = fixed_eval_set

    # ── 打印配置摘要 ─────────────────────────────────────────── #
    mode_tag = "【快速验证 QUICK_TEST】" if QUICK_TEST else "【正式训练】"
    print("=" * 66)
    print(f"  CRP-D × PG-FGB 训练（实验一：dup_dataset）{mode_tag}")
    print("=" * 66)
    print(f"  dup_dataset 根目录  : {DUP_ROOT}")
    print(f"  训练 stack 数 (S)   : {DUP_STACKS_LIST}")
    print(f"  训练 tier  数 (T)   : {DUP_TIERS if DUP_TIERS else '全部'}")
    print(f"  每文件夹最多文件数  : {DUP_MAX_FILES_PER_FOLDER if DUP_MAX_FILES_PER_FOLDER else '全取'}")
    print(f"  可用 dup 文件数      : {len(dup_files)}")
    if dup_files:
        h_vals = sorted(set(h for h, _, _, _ in dup_files))
        n_vals = sorted(set(n for _, _, n, _ in dup_files))
        g_sets = sorted(set(
            len(set()) for h, s, n, p in dup_files
        ))
        print(f"  文件 max_tiers 范围 : {min(h_vals)}..{max(h_vals)}")
        print(f"  文件 N（箱子数）范围: {min(n_vals)}..{max(n_vals)}")
    print(f"  特征归一化 (5维)    : [{FEATURE_SCALE}]")
    print(f"  船位格数             : {VESSEL['vessel_bays']}×"
          f"{VESSEL['vessel_rows']}×{VESSEL['vessel_tiers']} = "
          f"{VESSEL['vessel_bays']*VESSEL['vessel_rows']*VESSEL['vessel_tiers']}")
    print(f"  迭代: {n_iters}  eps/iter={TRAIN['episodes_per_iter']}"
          f"  lr={TRAIN['learning_rate']}  greedy_high={TRAIN['greedy_high']}")
    if fixed_eval_set:
        s_vals = sorted(set(s for _, s, _, _ in fixed_eval_set))
        print(f"  固定评估集         : T=6, α=0.8, S={s_vals}, {len(fixed_eval_set)} 个实例")
    else:
        print(f"  评估方式           : 随机采样（{TRAIN['eval_episodes']} 个/轮）")
    print(f"  模型将保存到: {save_path}")
    print("=" * 66)

    if not dup_files:
        print("[ERROR] 未找到任何 dup 文件，请检查 DUP_ROOT 和 DUP_STACKS 配置。")
        return

    proc = mp.Process(target=inst.train, args=(factory, q, ev), daemon=True)
    proc.start()

    while proc.is_alive():
        try:
            r = q.get(timeout=1.0)
            m = r.metrics
            s6  = m.get('shifters_s6',  float('nan'))
            ood = m.get('shifters_ood', float('nan'))
            line = (
                f"  step={r.step:5d}/{n_iters}"
                f"  shifters={m.get('shifters', r.metric):.2f}"
                f"  best={m.get('best_shifters', float('inf')):.2f}"
            )
            if fixed_eval_set:
                line += f"  S6={s6:.2f}" if s6 == s6 else ""
                line += f"  OOD={ood:.2f}" if ood == ood else ""
            line += (
                f"  loss={m.get('policy_loss', float('nan')):.4f}"
                f"  entropy={m.get('entropy', float('nan')):.4f}"
                f"  adv={m.get('mean_advantage', float('nan')):+.4f}"
                f"  bl_upd={int(m.get('baseline_updates', 0))}"
                f"  [{r.progress*100:.1f}%]"
            )
            print(line)
        except Exception:
            pass

    proc.join()
    ev.set()

    if pathlib.Path(save_path).exists():
        print(f"\n[✓] 模型已保存 → {save_path}")
    else:
        print("\n[!] 模型未能保存，请检查训练过程")
    print("Done.")


if __name__ == "__main__":
    main()
