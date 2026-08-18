# CRP Platform

**Container Relocation & Stowage Platform**

一个专为集装箱堆场管理问题设计的研究平台，类似 PlatEMO，支持启发式、精确算法、进化算法和可复现实验，带实时可视化界面。
各问题类保留 Gymnasium-compatible 的 `reset` / `step` 环境接口，用于逐步仿真、解验证、Web 回放和未来可选的学习型方法扩展。

---

## 快速开始

### 第一步：安装依赖

```bash
conda activate rl
pip install -r requirements.txt
```

验证安装：
```bash
conda activate rl
python -c "import fastapi, gymnasium, matplotlib; print('All OK')"
```

---

### 第二步：启动 Web 工作台

```bash
conda activate rl
cd Platform_CRISP

python main.py web
# 浏览器访问 http://127.0.0.1:8000
```

React 界面由 FastAPI 直接托管；算法在独立 Python 进程中运行。关闭浏览器不会
中断当前服务器进程中的任务。

---

### 第三步：命令行使用（不需要浏览器）

```bash
conda activate rl
cd Platform_CRISP

# 查看所有已注册的问题和算法
python main.py list

# 快速测试（验证所有问题能跑）
python main.py test

# 运行单个实验
python main.py run --problem "CRP-Stow" --algo "Genetic Algorithm" --iterations 200
```

---

## Web 工作台使用说明

Web 工作台提供 Workbench、Jobs 和 Compare 页面。问题与算法由现有 registry 自动
发现，参数控件由 `config_schema()` 自动生成；新增算法无需修改 React。任务
在独立进程中执行，支持实时进度、停止、收敛曲线、堆场快照回放和结果保存。

修改前端源码后可重新构建：

```bash
cd web/frontend
npm install
npm run build
```

---

## 问题类型说明

| 问题名 | 说明 | 主要指标 |
|--------|------|----------|
| **CRP-R** | 固定顺序取箱，必须按优先级 1→2→…→N 取出 | relocations |
| **CRP-Time** | 与 CRP-R 规则相同，主目标为场桥总作业时间（秒） | time / crane_time（兼看 relocations） |
| **BRP-NonFixed** | 自由选择取箱顺序，优化总搬移次数 | relocations |
| **CRP-Prem** | 开船前重排堆场，使所有栈有序 | moves |
| **CRP-Stow** | 堆场→船舶配载，考虑分组约束；可选 `rc_ratio>0` 开启 **POCRP-RC**（Rolled Container） | shifters / relocations |
| **CRP-Stoch** | 随机 CRP（SCRP：批次化、批内均匀随机、揭示时承诺）；`batch_size=1` 时等价 CRP-R | expected_relocations / relocations |
| **CRP-U** | 无约束翻箱（uBRP）：取箱顺序固定 1→…→N，翻箱可从**任意栈顶**搬到**任意未满栈** | relocations |
| **CRP-D** | 重复箱组配载（当前与 CRP-Stow 同构，可扩展生成） | shifters |

---

## 算法说明

| 算法 | 类型 | 适用问题 | 说明 |
|------|------|---------|------|
| **Genetic Algorithm** | 进化算法 | 全部 | 染色体=动作序列，均匀交叉+随机变异 |
| **Kim–Hong (2006) ENAR** | 启发式 | CRP-R, CRP-Time | Kim & Hong 2006, COR 33 – 经典单贝翻箱规则 |
| **Caserta (2012) HEUR** | 启发式 | CRP-R, CRP-Time | Caserta, Schwarze & Voß 2012, EJOR – min-priority 目的栈规则 |
| **Jin (2015) GLAH (CRP-Time)** | 启发式 | CRP-Time, CRP-R | Jin, Zhu & Lim 2015, EJOR — 同名算法按问题分包：`CRP_Time/heuristic/glah` |
| **Jin (2015) GLAH (CRP-D)** | 启发式 | CRP-D | 同上代码基线（CRP-D / stowage scaffold） |
| **Jin (2015) GLAH (CRP-U)** | 启发式 | CRP-U, BRP-NonFixed | 同名 GLAH 端口的**基线**；CRP-U 现为 uBRP 动作空间，与 2015 **受限**翻箱过程不完全一致；``BRP-NonFixed`` 仍为自由取顺。路径 ``algorithms/CRP_U/heuristic/glah`` |
| **LA-N Look-Ahead** | 启发式 | CRP-R, CRP-Time | Petering & Hussein 2013, EJOR – N 步 look-ahead + cleaning moves |
| **Tanaka (2016) B&B** | 精确 / B&B | **CRP-R** | Tanaka ``restricted-duplicate-1.01`` – 受限翻箱 + 允许重复 priority；封装 vendor ``brp_bb``；指标含 ``optimal_proven`` |
| **Tanaka (2016) B&B [CRP-D]** | 精确 / B&B | **CRP-D** | CRP-D 下的 duplicate 命名适配器；复用同一 ``brp_bb`` 后端，主指标映射为 ``shifters`` |
| **Tanaka (2018) B&B** | 精确 / B&B | **CRP-R** | Tanaka ``restricted-distinct-1.11``（2018 修订）— 对 priority 做秩压缩，因此同时兼容 duplicate / non-duplicate；封装 vendor ``brp_bb`` |
| **Lee–Lee (2010) Retrieval** | 启发式 | CRP-Time | Lee & Lee 2010, COR – 三阶段启发式（含起重机时间）；实现位于 `algorithms/CRP_Time/heuristic/lee_lee` |
| **Lin–Lee–Lee (2015) Rule** | 启发式 | **CRP-Time**, CRP-R | Lin, Lee & Lee 2015, TRC – 位置优先级规则：`P_b·(not-well-placed) + P_r·severity + carry-time`；默认 `P_r=30, P_b=300`（Shin 2026 参数）|
| **López-Plata et al. (2019) Operating-Cost Heuristic** | 启发式 (A*+expansion) | **CRP-Time**, CRP-R | López-Plata, Expósito-Izquierdo & Moreno-Vega 2019, C&IE – 最小化作业成本（非仅移动次数）；有限深度 A* 找 promising partial + expansion + 去 temporary move 后处理；支持可配置线性成本权重。|
| **Cifuentes–Riff (2020) G-CREM** | 启发式 / GRASP | **CRP-Time** | Cifuentes & Riff 2020, ASOC – 多贝 GRASP：构造用 myopic `H−N_j` + size-adaptive RCL，局部搜索以 RIL 修复（Wu & Ting 2010）；目标 `α·moves + β·crane_time`，默认 `α=0.3, β=0.005, k=3` |
| **Ðurasević–Ðumić (2024) GP** | 启发式 / GP hyper-heuristic | **CRP-Time**, CRP-R | Ðurasević & Ðumić 2024, ASOC – 用 Genetic Programming **自动进化** Priority Function（表达式树），配合 restricted RS；terminals = `SH/EMP/CUR/RI/AVG/DIFF (+ DIS/DUR)`；**第一阶段部署：多贝 distinct + restricted RS**（unrestricted 与 container-groups 留给未来 `CRP-Groups` 问题类）|
| **Ðurasević–Ðumić–Gil-Gala (2025) MGP** | 启发式 / Multitask GP | **CRP-Time** | Ðurasević, Ðumić & Gil-Gala 2025, EAAI – **多任务 GP**：同时进化 `S_P` 个子种群（每个子种群对应一个 CRP 任务），通过 cross-subpop crossover / ring-topology migration 共享知识；scenarios = `max_tiers` / `objective`（论文 Table 7 红利）/ `layout` / `load`；复用 2024 GP 的 `gp_core / terminals / rs_restricted` |
| **Wang (2026) GRASP** | 启发式 / GRASP | **CRP-Stow** (`rc_ratio>0` = POCRP‑RC) | Wang, Ma, Yang & Hu 2026, C&OR 191:107434 – 论文主算法。构造阶段的 `TR`（min NBC → max DC → min BC+RCB → 优先制造 empty/rolled sink）与 RC‑感知的 `RR`（Task 1 for RCs / Task 2 for OCs），加 LNS+ 四条 poor‑move 判据（CMM/CMB 借用 Jovanović 2019；poor‑target 2.1/2.2 为本文原创）。要求 `rc_ratio>0` 才能真正展开 RC 逻辑，`rc_ratio=0` 时相当于纯 POCRP。 |

> **Baseline 部署约定**：同一问题族共享一套 `ProblemConfig` 变量（`num_bays, num_rows, max_tiers, num_containers` 等）与统一指标（`crane_time` / `relocations` / `lb_ratio`）。新增论文的算法仅放入 `algorithms/<category>/<paper_key>/`，通过 `compatible_problems` 声明挂到哪些问题上，**问题定义与目标保持不动**，变的只有方法。
>
> **「单贝 / 多贝」不是两个问题**，只是 `num_bays` 参数的取值。`CRP-R` 与 `CRP-Time` 的区别在 **目标函数**（最少翻箱 vs 最短场桥时间），几何都通用。因此所有固定顺序启发式同时挂到两个问题下，描述前缀用如下 tag 标注其 **原文适用范围**，不影响可运行性：
>
> - `[native multi-bay]` — 原文本身就是多贝 CRP（Lee–Lee 2010 / Lin 2015 / G-CREM 2020）；`CRP-Time` 的推荐主力。
> - `[single-bay origin]` — 原文只做单贝受限翻箱（Kim–Hong 2006 / Caserta 2012 / LA-N 2013 / GLAH 2015）；在 `CRP-Time`（多贝）上可跑，作为 **退化 baseline** 用，时间目标通常次优。
> - `[general]` — 问题无关的通用求解器（GA）。

---

## 如何添加自己的算法

在 `algorithms/` 下任意子目录新建 `.py` 文件：

```python
# algorithms/CRP_R/heuristic/my_rule/algorithm.py
from core.base_algorithm import BaseAlgorithm, AlgorithmConfig
import multiprocessing as mp

class MyRule(BaseAlgorithm):
    name     = "My Rule"       # GUI 下拉菜单中显示的名字
    category = "Heuristic"     # "Exact" / "Evolutionary" / "Heuristic"
    description = "自定义启发式算法"
    compatible_problems = ["CRP-R"]

    def train(self, problem_factory, result_queue, stop_event):
        env = problem_factory()
        obs, info = env.reset()

        # ↓ 在这里写你的算法逻辑。问题环境负责验证动作、更新堆场和计算指标。
        for step in range(self.config.max_iterations):
            if stop_event.is_set():
                break

            action = env.action_space.sample()  # 替换成你的规则
            obs, reward, done, _, info = env.step(action)
            if done:
                break
            
            # 每隔 report_interval 步，推送进度给 GUI
            if step % self.config.report_interval == 0:
                self._push(result_queue,
                    step=step,
                    metric=-reward,
                    metrics=env.get_metrics(),
                    progress=step/self.config.max_iterations,
                    snapshot=env.get_state_snapshot(),
                )

    def get_best_solution(self):
        return self._best_solution
```

**保存文件，重启 GUI** → "My Rule" 自动出现在算法下拉菜单中。

---

## 如何添加自己的问题

在 `problems/` 下新建 `.py` 文件：

```python
# problems/my_problem.py
from core.base_problem import BaseProblem, ProblemConfig
import numpy as np, gymnasium as gym

class MyProblem(BaseProblem):
    name        = "My Problem"
    description = "自定义集装箱问题"
    tags        = ["relocation", "custom"]
    metric_names = ["my_metric"]

    def _setup_spaces(self):
        n = self.config.num_bays * self.config.num_rows * self.config.max_tiers
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(n,), dtype=np.int32)
        self.action_space      = gym.spaces.Discrete(n)

    def _build_episode(self):
        # 初始化堆场，往 self.yard 里放集装箱
        pass

    def reset(self, seed=None, options=None):
        self.yard.clear()
        self._build_episode()
        return self._get_obs(), {}

    def step(self, action):
        # 执行动作，返回 (obs, reward, terminated, truncated, info)
        return self._get_obs(), 0.0, False, False, {}

    def evaluate(self, solution):
        return {"my_metric": 0.0}

    def get_metrics(self):
        return {"my_metric": 0.0}

    def _get_obs(self):
        return self.yard.get_flat_obs()
```

---

## 项目结构

```
crp_platform/
├── core/
│   ├── container.py      集装箱数据模型（size/weight/type/group/priority）
│   ├── yard.py           底层 relocation 引擎（Stack + Yard）
│   ├── base_problem.py   问题抽象基类
│   ├── base_algorithm.py 算法抽象基类（子进程训练框架）
│   └── registry.py       自动注册系统
├── problems/             问题环境：Gymnasium-compatible 仿真 / 验证接口
├── algorithms/           算法库（可无限扩展）
│   ├── CRP_R/
│   ├── CRP_U/
│   ├── CRP_D/
│   ├── CRP_Time/
│   ├── CRP_Stow/
│   └── CRP_Prem/
├── visualization/        可视化渲染（供环境 / Web 回放）
├── web/
│   ├── backend/          FastAPI catalog、任务与结果 API
│   └── frontend/         React + TypeScript 界面
├── main.py               CLI / Web 入口
└── requirements.txt      依赖列表
```

---

## 常见问题

**Q: `python main.py web` 后浏览器没有自动打开？**  
A: 手动在浏览器访问 `http://127.0.0.1:8000`

**Q: 训练很慢？**  
A: 减小 `max_iterations`，或用 `Caserta (2012) HEUR` 等轻量启发式先测试

**Q: 如何用你现有的 spp-main 代码？**  
A: `problems/CRP_Stow.py` 已经基于你的原始 `stowage_gym.py` 逻辑重构，逻辑完全一致

**Q: 如何保存训练结果？**  
A: Workbench 运行结束后会写入 `results/`；Compare 页可汇总与导出 CSV
