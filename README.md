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
python main.py run --problem "CRP-R" --algo "Caserta (2012) HEUR"
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

平台有七个问题类。它们不是同一 CRP 的七种别名，而是按下面四根轴切开的：

```
任务 / 目标     Time | Prem | Stow | Stoch     ← 不同问题，不要横比主指标
翻箱规则        R vs U                         ← 只用于 distinct CRP
优先级          R/U = distinct；D = duplicate  ← D 内再用 restricted_relocation
几何            num_bays                       ← 配置项，不是问题类
```

| 问题名 | 文献位置 | 说明 | 主指标 |
|--------|----------|------|--------|
| **CRP-R** | restricted + distinct | 取顺 1…N；只能搬当前目标上方的阻塞箱 | relocations |
| **CRP-U** | unrestricted + distinct | 取顺 1…N；任意栈顶都可搬到任意未满栈 | relocations |
| **CRP-D** | duplicate（默认 Du） | 组间有序、组内任意。`extra["restricted_relocation"]`：`False`（默认）= 无限制 Du，`True` = 受限 Dr。**不是**配载问题，与 CRP-Stow 无关 | relocations |
| **CRP-Time** | 同 CRP-R，换目标 | 规则与 CRP-R 相同（受限 + distinct），主目标为场桥总作业时间 | crane_time（兼看 relocations） |
| **CRP-Online** | OCRP / limited look-ahead | 动力学同 CRP-R，但未来取箱信息按 `lookahead_h` 逐步揭示；`lookahead_h=0` 对应 Zehendner 2017 | relocations |
| **CRP-Prem** | pre-marshalling | 不取箱，把堆场重排到每栈有序 | moves |
| **CRP-Stow** | BRLP / POCRP | 按船舶配载计划从堆场取箱；`rc_ratio>0` 开启 **POCRP-RC**（Rolled Container） | relocations |
| **CRP-Stoch** | SCRP | 批次揭示、批内均匀随机；`batch_size=1` 时退回 CRP-R。与 **CRP-Online** 的逐步揭示语义区分开 | expected_relocations |

没有 **BRP-NonFixed**（自由选择取箱顺序）这一类。

---

## 算法说明

完整名单以 registry 为准，不要以本页表格为准：

```bash
python main.py list
```

各族目录和论文入口写在 `algorithms/<族>/README.md`（`CRP_R`、`CRP_U`、`CRP_D`、`CRP_Time`、`CRP_Online`、`CRP_Prem`、`CRP_Stow`、`CRP_Stoch`）。

> **部署约定**：同一问题族共享一套 `ProblemConfig`（`num_bays, num_rows, max_tiers, num_containers` 等）。新算法放进 `algorithms/<primary_problem>/<category>/<paper_key>/`，用 `compatible_problems` 声明可跑哪些问题；**问题定义与目标保持不动**，变的只有方法。
>
> **「单贝 / 多贝」不是两个问题**，只是 `num_bays`。CRP-R 与 CRP-Time 的差别在目标（翻箱次数 vs 场桥时间），几何通用。
>
> 若干 CRP-R 启发式（Kim–Hong 2006 ENAR、Caserta 2012 HEUR、LA-N）通过 `compatible_problems` 也可在 CRP-Time 下拉框出现，当作退化 baseline；它们的目录仍在 `algorithms/CRP_R/heuristic/`，Time 目录里没有 symlink。Jin (2015) GLAH 目前只在 `algorithms/CRP_U/heuristic/glah/`。

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
│   ├── CRP_Prem/
│   ├── CRP_Stow/
│   └── CRP_Stoch/
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
