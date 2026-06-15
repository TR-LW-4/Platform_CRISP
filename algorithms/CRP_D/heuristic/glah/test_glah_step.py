"""
GLAH 逐步可视化脚本（CRP-D）

用法（从 Platform_CRISP 根目录）：
    python -m algorithms.CRP_D.heuristic.glah.test_glah_step \
        --file benchmark/dup_dataset/alpha=0.2/3-6-15/00001.txt \
        --depth 3 \
        --pause

每一步会打印：
  - 操作类型（RELOCATE / RETRIEVE）
  - 从哪个 stack 搬到哪个 stack
  - 搬的是哪个组（priority）
  - 当前堆场状态
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

_ROOT = Path(__file__).parents[5]   # Platform_CRISP/
sys.path.insert(0, str(_ROOT))


# ────────────────────────────────────────────────────────────────
#  堆场打印
# ────────────────────────────────────────────────────────────────

def _print_glah_yard(inst, title: str = "", highlight: set = None) -> None:
    """打印 GlahLayout 的当前堆场状态（单列格式，从上到下）。"""
    hl = highlight or set()
    col_w = 6

    if title:
        print(f"\n{'─'*60}")
        print(f"  {title}")
        print(f"{'─'*60}")

    # 列头
    header = ["  Tier |"]
    for s in range(1, inst.S + 1):
        mark = "★" if s in hl else " "
        header.append(f"{mark}S{s:2d}  ")
    print("".join(header))
    print("  " + "-" * (8 + inst.S * (col_w + 1)))

    # 从高到低逐层打印
    max_h = inst.H
    for tier in range(max_h, 0, -1):
        cells = [f"  T{tier:2d}  |"]
        for s in range(1, inst.S + 1):
            if tier <= inst.height[s] and inst.bay[s][tier] is not None:
                g = inst.bay[s][tier].priority
                is_top = (tier == inst.height[s])
                txt = f"[G{g}]" if is_top else f" G{g} "
            else:
                txt = " --- "
            cells.append(f" {txt:<{col_w-1}}")
        print("".join(cells))

    heights = "  ".join(f"S{s}:{inst.height[s]}" for s in range(1, inst.S + 1))
    print(f"  heights: {heights}")
    print()


# ────────────────────────────────────────────────────────────────
#  主逻辑
# ────────────────────────────────────────────────────────────────

def run_glah_step(dup_file: str, depth: int, pause: bool) -> None:
    from core.zhu_dup_benchmark import parse_zhu_dup_file
    from core.base_problem import ProblemConfig
    from problems.CRP_D import CRP_D

    from algorithms.CRP_D.heuristic.glah.layout import (
        GlahLayout, GlahState, GlahOp, lower_bound, ops_to_relocation_plan
    )
    from algorithms.CRP_D.heuristic.glah.evaluate import evaluation_heuristic
    from algorithms.CRP_D.heuristic.glah.lookahead import Lookahead

    path = Path(dup_file)
    stacks, n, stack_groups = parse_zhu_dup_file(path)
    max_tiers = int(max((len(col) for col in stack_groups), default=5))
    groups = int(max((g for col in stack_groups for g in col if g > 0), default=3))

    cfg = ProblemConfig(
        num_bays=stacks, num_rows=1, max_tiers=max_tiers,
        num_containers=n, num_groups=groups,
        vessel_bays=1, vessel_rows=max(80, n + 10), vessel_tiers=1, seed=0,
    )
    cfg.extra["layout_file_path"] = str(path.resolve())
    env = CRP_D(config=cfg)
    env.reset()

    initial_yard = copy.deepcopy(env.yard)
    containers   = list(env.containers)
    glah_layout  = GlahLayout.build_from_yard(initial_yard, containers)

    print("=" * 62)
    print(f"  GLAH 逐步测试   文件: {path}")
    print(f"  S={stacks}  N={n}  G={groups}  max_tiers={max_tiers}  depth={depth}")
    print("=" * 62)

    # ── Phase 1：evaluation heuristic ────────────────────────── #
    state1 = GlahState(glah_layout.copy())
    _print_glah_yard(state1.inst, "初始堆场")
    print(f"  下界 LB = {lower_bound(state1.inst)}")

    print("\n[Phase 1] evaluation_heuristic 开始……")
    if pause:
        input("  [Enter] 继续...")

    # 逐步版 evaluation_heuristic（手动展开，打印每一步）
    _run_eval_heuristic_verbose(state1, pause)

    print(f"\n[Phase 1] 完成  best_reloc = {state1.best_reloc}  "
          f"shifters = {state1.reloc_count}")

    # ── Phase 2：look-ahead ──────────────────────────────────── #
    la = Lookahead(depth, 5, 5, 3, 3, 1, 1)
    state2 = GlahState(glah_layout.copy())
    state2.best_ops   = state1.best_ops
    state2.best_reloc = state1.best_reloc
    state2.try_retrievals()

    print(f"\n[Phase 2] look-ahead (depth={depth}) 开始……")
    step2 = 0
    while not state2.is_empty():
        lb_now = lower_bound(state2.inst)
        if lb_now + state2.reloc_count >= state2.best_reloc:
            print(f"  [GLAH] LB 剪枝: lb={lb_now}  reloc={state2.reloc_count}"
                  f"  best={state2.best_reloc} → 退出")
            break
        op = la.most_promising_relocation(state2)
        if op is None:
            print("  [GLAH] 无候选操作 → 退出")
            break
        step2 += 1
        kind = "RELOCATE" if op.is_relocation else "RETRIEVE"
        hl   = {op.from_s, op.to_s} if op.to_s else {op.from_s}
        print(f"\n{'━'*60}")
        print(f"  Phase2 Step {step2}  │  {kind}  S{op.from_s} → "
              f"{'S'+str(op.to_s) if op.to_s else 'OUT'}  "
              f"G{op.gc.priority}  累计reloc={state2.reloc_count}")
        print(f"{'━'*60}")
        state2.go_one_step(op)
        state2.try_retrievals()
        _print_glah_yard(state2.inst, f"Phase2 Step {step2} 后", highlight=hl)
        if pause:
            cmd = input("  [Enter] 下一步 / [q] 退出: ").strip().lower()
            if cmd == "q":
                break

    best_reloc = state2.best_reloc if state2.best_ops else state1.best_reloc
    print(f"\n{'═'*62}")
    print(f"  GLAH 完成  最终最优 shifters = {best_reloc}")
    print(f"{'═'*62}")


def _run_eval_heuristic_verbose(state: "GlahState", pause: bool) -> None:
    """展开版 evaluation_heuristic，每步打印堆场。"""
    from algorithms.CRP_D.heuristic.glah.layout import (
        GlahOp, _get_nearest_target, lower_bound
    )
    from algorithms.CRP_D.heuristic.glah.evaluate import (
        _find_support_stack_min_capacity, _find_stack_max_capacity, _gap_utilize
    )

    state.try_retrievals()
    inst  = state.inst
    step  = 0

    while not inst.is_empty():
        ut = _get_nearest_target(inst)
        if ut is None:
            break

        ut_uid   = ut.uid
        ut_tier  = inst.at_tier[ut_uid]
        ut_stack = inst.at_stack[ut_uid]
        target_g = ut.priority

        while ut_tier < inst.height[ut_stack]:
            c = inst.top_container(ut_stack)
            if c is None:
                break

            p_stack = _find_support_stack_min_capacity(inst, c.priority, [ut_stack])

            if p_stack == -1:
                cv = None; largest_cv = 0; v_stack = -1; found_p = -1
                for i in range(1, inst.S + 1):
                    if i == ut_stack:
                        continue
                    itop = inst.top_container(i)
                    if itop is None:
                        continue
                    if (inst.support_capacity_except_top(i) >= itop.priority and
                            inst.support_capacity_except_top(i) >= c.priority):
                        j = _find_support_stack_min_capacity(inst, itop.priority, [ut_stack, i])
                        if j != -1 and itop.priority > largest_cv:
                            largest_cv = itop.priority; found_p = i; v_stack = j; cv = itop
                if found_p != -1:
                    step += 1
                    print(f"\n{'━'*60}")
                    print(f"  Step {step}  │  RELOCATE(vacate)  "
                          f"S{found_p}→S{v_stack}  G{cv.priority}")
                    print(f"{'━'*60}")
                    state.go_one_step(GlahOp(cv, found_p, v_stack))
                    _print_glah_yard(inst, f"Step {step} 后 (vacate G{cv.priority})",
                                     highlight={found_p, v_stack})
                    if pause:
                        cmd = input("  [Enter] / [q]: ").strip().lower()
                        if cmd == "q":
                            return
                    p_stack = found_p
                else:
                    p_stack = _find_stack_max_capacity(inst, [ut_stack])
                    if p_stack == -1:
                        break

            _gap_utilize(state, ut_stack, p_stack, c)

            step += 1
            print(f"\n{'━'*60}")
            print(f"  Step {step}  │  RELOCATE  "
                  f"S{ut_stack}→S{p_stack}  G{c.priority}  "
                  f"(为取 G{target_g} 搬走阻挡)  累计reloc={state.reloc_count+1}")
            print(f"{'━'*60}")
            state.go_one_step(GlahOp(c, ut_stack, p_stack))
            _print_glah_yard(inst, f"Step {step} 后", highlight={ut_stack, p_stack})
            if pause:
                cmd = input("  [Enter] / [q]: ").strip().lower()
                if cmd == "q":
                    return

        before = state.reloc_count
        state.try_retrievals()
        after  = state.reloc_count
        if after == before:
            _print_glah_yard(inst, f"Step {step} 后 RETRIEVE G{target_g}",
                             highlight={ut_stack})
            step += 1

    state.update_best()


# ────────────────────────────────────────────────────────────────
#  CLI
# ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GLAH 逐步可视化")
    parser.add_argument("--file",  required=True,  help="dup_dataset .txt 文件路径")
    parser.add_argument("--depth", type=int, default=3, help="look-ahead 深度（默认 3）")
    parser.add_argument("--pause", action="store_true", help="每步按 Enter 继续")
    args = parser.parse_args()
    run_glah_step(args.file, args.depth, args.pause)
