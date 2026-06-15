"""
CRP-Stow (BRLP) 启发式算法逐步测试脚本
=========================================
Jovanović 2019 BRLP 问题接口对应版本。

用法（在 Platform_CRISP 根目录下执行）：
    python -m algorithms.CRP_Stow.heuristic.test_heuristic --algo JiNearest --pause

支持的算法 (--algo)：
    JiNearest           Ji (2015) 最近栈策略
    JiOptimization      Ji (2015) 优化策略（避免二次搬箱）
    JovanovicGreedy     Jovanović (2019) 贪心 MinW4CB+MinMax4CB（步进模式）

常用选项：
    --algo        JiNearest  算法名称（见上方列表）
    --seed        42         随机 seed
    --pause                  每步按 Enter 继续（交互模式）
    --bays        4          堆场 yard stack 数（YS）
    --tiers       5          堆场最大层数（YT）
    --containers  20         堆场总箱子数（N）
    --groups      4          船舱 vessel stack 数（VS）
    --pro                    可选：Jovanović .pro 文件路径

示例：
    # 交互式逐步看 JiNearest
    python -m algorithms.CRP_Stow.heuristic.test_heuristic --algo JiNearest --pause

    # 用 .pro 文件
    python -m algorithms.CRP_Stow.heuristic.test_heuristic \\
        --algo JiOptimization \\
        --pro /data/liuw2/Platform_CRISP/benchmark/crp_stow/Data/Gen/Bay-3-3-3-6_0.pro \\
        --pause
"""
from __future__ import annotations

import argparse
import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "../../../"))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


_ALGO_NAMES = [
    "JiNearest", "JiOptimization",
    "JovanovicGreedy",
]


def _make_selector(algo_name: str):
    """Return selector_fn(env) -> int."""
    if algo_name == "JiNearest":
        from algorithms.CRP_Stow.heuristic.ji_2015.algorithm import JiNearest
        return JiNearest()._select

    if algo_name == "JiOptimization":
        from algorithms.CRP_Stow.heuristic.ji_2015.algorithm import JiOptimization
        return JiOptimization()._select

    if algo_name == "JovanovicGreedy":
        from algorithms.CRP_Stow.heuristic.jovanovic_2019.algorithm import (
            _select_high_MinW4CB, _select_low_MinMax4CB, _compute_well_located,
        )
        _wl_cache: list = [{}]

        def selector(env) -> int:
            if env._mode == "high":
                _wl_cache[0] = _compute_well_located(env)
                return _select_high_MinW4CB(env, _wl_cache[0])
            src_key = env._idx_to_stack(env._source_stack_idx)
            stk = env.yard.stacks.get(src_key)
            c = stk.top if (stk and not stk.is_empty) else None
            N = len(env.containers)
            a = _select_low_MinMax4CB(env, src_key, c, _wl_cache[0], N)
            return max(0, a)

        return selector

    raise ValueError(f"未知算法: {algo_name!r}。可选: {', '.join(_ALGO_NAMES)}")


# ────────────────────────────────────────────────────────────────────
#  打印辅助
# ────────────────────────────────────────────────────────────────────

def _container_token(c) -> str:
    """Format a container as 'A_0', 'B_2', etc."""
    return f"{chr(ord('A') + c.group)}_{c.priority}"


def _print_yard(env, title: str = "", highlight: set | None = None) -> None:
    """显示堆场（1-D BRLP: num_rows=1）。"""
    cfg   = env.config
    n     = env._n_stacks
    max_h = cfg.max_tiers
    hl    = highlight or set()

    if title:
        print(f"\n{'─'*60}")
        print(f"  {title}")
        print(f"{'─'*60}")

    col_w = 6
    header = "  " + " | ".join(
        f"{'★' if i in hl else ' '}S{i:<2d}" for i in range(n)
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for tier in range(max_h, 0, -1):
        cells = []
        for i in range(n):
            key = env._idx_to_stack(i)
            stk = env.yard.stacks.get(key)
            groups = stk.containers if stk else []
            idx = tier - 1
            if idx < len(groups):
                c   = groups[idx]
                tok = _container_token(c)
                if env._is_retrievable(c):
                    tok = f"[{tok}]"
                cells.append(f"{tok:^{col_w}}")
            else:
                cells.append(f"{'----':^{col_w}}")
        print(f"  T{tier:<2d} | " + " | ".join(cells))

    def _stk_h(i):
        s = env.yard.stacks.get(env._idx_to_stack(i))
        return len(s.containers) if s else 0

    heights = "  ".join(f"S{i}:{_stk_h(i)}" for i in range(n))
    print(f"         heights: {heights}")


def _print_brlp_state(env) -> None:
    """显示 vessel 装载进度和当前阶段。"""
    VS      = env.config.num_groups
    loaded  = env._vessel_loaded
    max_t   = env._vessel_max_tier
    phase   = env._mode.upper()

    vessel_str = "  ".join(
        f"{chr(ord('A')+j)}:{loaded[j]}/{max_t[j]}"
        for j in range(min(VS, len(loaded)))
    )
    print(f"  Vessel 装载: {vessel_str}")

    if env._mode == "low" and env._target_container is not None:
        tgt = env._target_container
        print(f"  Phase: {phase}  ← 正在清障以取出 [{_container_token(tgt)}]  "
              f"src_stack=S{env._source_stack_idx}")
    else:
        # Retrievable containers
        retrievable = [
            (i, c) for i in range(env._n_stacks)
            for c in (env.yard.stacks.get(env._idx_to_stack(i)) or type('', (), {'containers': []})()).containers
            if env._is_retrievable(c)
        ]
        if retrievable:
            r_str = "  ".join(f"S{i}[{_container_token(c)}]" for i, c in retrievable)
            print(f"  Phase: {phase}  可立即取出: {r_str}")
        else:
            print(f"  Phase: {phase}  (无可直接取出的箱子)")


# ────────────────────────────────────────────────────────────────────
#  主测试逻辑
# ────────────────────────────────────────────────────────────────────

def run_test(
    algo_name:      str,
    seed:           int,
    pause:          bool,
    num_bays:       int,
    max_tiers:      int,
    num_containers: int,
    num_groups:     int,
    pro_path:       str | None,
) -> None:
    from core.base_problem import ProblemConfig
    from problems.CRP_Stow import CRP_Stow

    try:
        selector = _make_selector(algo_name)
    except ValueError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    print(f"[INFO] 算法: {algo_name}")

    cfg = ProblemConfig(
        num_bays       = num_bays,
        num_rows       = 1,
        max_tiers      = max_tiers,
        num_containers = num_containers,
        num_groups     = num_groups,
        seed           = seed,
    )
    if pro_path:
        cfg.extra["layout_file_path"] = pro_path
        print(f"[INFO] 加载 .pro 文件: {pro_path}")

    env = CRP_Stow(config=cfg)
    env.reset(seed=seed)

    YS = env._n_stacks
    VS = env.config.num_groups

    print(f"\n{'═'*62}")
    print(f"  TEST [{algo_name}]: seed={seed}")
    print(f"  堆场: {YS} stacks × max {env.config.max_tiers} tiers")
    print(f"  箱子: {sum(not stk.is_empty for stk in env.yard.stacks.values())} 垛 / "
          f"总 {sum(stk.height for stk in env.yard.stacks.values())} 个")
    print(f"  船舱: {VS} vessel stacks  "
          f"总容量={sum(env._vessel_max_tier)}")
    print(f"{'═'*62}")

    _print_yard(env, "初始堆场状态（[] = 可立即取出）")
    _print_brlp_state(env)

    if pause:
        input("\n  [Enter] 开始逐步执行...")

    step_num = 0
    done     = False

    while not done:
        phase      = env._mode
        src_idx    = env._source_stack_idx
        target_c   = env._target_container

        mask = env._build_action_mask()
        if not mask.any():
            print("  [WARN] 动作掩码全为 False，无法继续。")
            break

        action    = selector(env)
        dst_key   = env._idx_to_stack(action)
        dst_stk   = env.yard.stacks.get(dst_key)
        dst_top   = _container_token(dst_stk.top) if (dst_stk and not dst_stk.is_empty) else "空"

        _, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        step_num += 1

        if phase == "high":
            action_desc = (
                f"HIGH  → 选垛S{action} (顶=[{dst_top}])"
            )
        else:
            src_c_str = _container_token(target_c) if target_c else "?"
            action_desc = (
                f"LOW   ★RELOCATE★  垛S{src_idx}顶 → 垛S{action}  "
                f"(目标={src_c_str})  reward={reward:+.0f}"
            )

        hl = {action}
        if phase == "low" and src_idx is not None:
            hl.add(src_idx)

        print(f"\n{'━'*60}")
        print(f"  Step {step_num:3d}  │  {action_desc}")
        print(f"{'━'*60}")
        _print_yard(env, f"Step {step_num} 后的堆场", highlight=hl)
        _print_brlp_state(env)

        metrics = env.get_metrics()
        print(f"  累计 relocations={metrics['relocations']:.0f}  "
              f"已装船={metrics['retrieved']:.0f}  "
              f"装船率={metrics['vessel_utilisation']*100:.1f}%")

        if pause and not done:
            cmd = input("\n  [Enter] 下一步 / [q] 退出: ").strip().lower()
            if cmd == "q":
                print("  用户中断。")
                break

    final = env.get_metrics()
    print(f"\n{'═'*62}")
    print(f"  [{algo_name}] 测试完成  共 {step_num} 步")
    print(f"  总 relocations = {final['relocations']:.0f}")
    print(f"  装船容器数     = {final['retrieved']:.0f} / {sum(env._vessel_max_tier)}")
    print(f"  装船利用率     = {final['vessel_utilisation']*100:.1f}%")
    print(f"{'═'*62}")


# ────────────────────────────────────────────────────────────────────
#  CLI 入口
# ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CRP-Stow (BRLP) 启发式算法逐步测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--algo",       default="JovanovicGreedy",
                        choices=_ALGO_NAMES,
                        help=f"算法名称（默认: JiNearest）")
    parser.add_argument("--seed",       type=int,  default=42,
                        help="随机 seed（默认: 42）")
    parser.add_argument("--pause",      action="store_true",
                        help="每步按 Enter 继续")
    parser.add_argument("--bays",       type=int,  default=4,
                        help="堆场 stack 数 YS（默认: 4）")
    parser.add_argument("--tiers",      type=int,  default=5,
                        help="堆场最大层数 YT（默认: 5）")
    parser.add_argument("--containers", type=int,  default=20,
                        help="堆场总箱子数 N（默认: 20）")
    parser.add_argument("--groups",     type=int,  default=4,
                        help="船舱 vessel stack 数 VS（默认: 4）")
    parser.add_argument("--pro",        type=str,  default=None,
                        help="可选：Jovanović .pro 文件路径（会覆盖 bays/tiers/groups）")
    args = parser.parse_args()

    run_test(
        algo_name      = args.algo,
        seed           = args.seed,
        pause          = args.pause,
        num_bays       = args.bays,
        max_tiers      = args.tiers,
        num_containers = args.containers,
        num_groups     = args.groups,
        pro_path       = args.pro,
    )
