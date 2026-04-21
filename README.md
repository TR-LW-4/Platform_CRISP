# CRP Platform

**Container Relocation & Stowage Platform**

一个专为集装箱堆场管理问题设计的研究平台，类似 PlatEMO，支持进化算法和强化学习，带实时可视化界面。

---

## 快速开始

### 第一步：安装依赖

```bash
conda activate rl
pip install streamlit imageio
```

验证安装：
```bash
conda activate rl
python -c "import streamlit, matplotlib, torch, gymnasium; print('All OK')"
```

---

### 第二步：启动 GUI 界面

```bash
conda activate rl
cd /home/liuw2/data1/crp_platform
streamlit run gui/app.py
```

浏览器会自动打开（或手动访问 `http://localhost:8501`）。

---

### 第三步：命令行使用（不需要 GUI）

```bash
conda activate rl
cd /home/liuw2/data1/crp_platform

# 查看所有已注册的问题和算法
python main.py list

# 快速测试（验证所有问题能跑）
python main.py test

# 运行单个实验
python main.py run --problem CSPP --algo "Genetic Algorithm" --iterations 200
python main.py run --problem "BRP-Fixed" --algo REINFORCE --iterations 500
```

---

## GUI 界面使用说明

打开 `http://localhost:8501` 后，界面分 4 个 Tab：

### 🔬 Test Tab（测试单个算法）

1. **左侧边栏**选择问题（Problem）和算法（Algorithm）
2. **左侧面板**调整参数：
   - 问题参数：堆场大小（num_bays/num_rows/max_tiers）、集装箱数量、组数、吊机数量等
   - 算法参数：迭代次数、学习率（RL）、种群大小（GA）等
3. 点击 **▶ Start** 开始训练
4. 右侧实时显示：
   - 堆场状态彩色图（每种颜色代表一个 group/目的港）
   - 收敛曲线（reward / fitness 随训练步数变化）
   - 实时指标（shifters、relocations 等）
5. 点击 **⏹ Stop** 随时停止

### 📊 Experiment Tab（批量对比实验）

1. 选择多个问题 + 多个算法
2. 设置随机种子数量
3. 点击 **🚀 Run Experiment**
4. 自动运行所有组合，输出对比表格

### 📈 Compare Tab（结果对比）

- 运行 Experiment 后，这里显示柱状图对比
- 可选择不同指标（shifters / relocations / time 等）

### ℹ️ About Tab

- 平台介绍和使用说明

---

## 问题类型说明

| 问题名 | 说明 | 主要指标 |
|--------|------|----------|
| **BRP-Fixed** | 固定顺序取箱，必须按优先级 1→2→…→N 取出 | relocations |
| **BRP-NonFixed** | 自由选择取箱顺序，优化总搬移次数 | relocations |
| **Pre-Marshalling** | 开船前重排堆场，使所有栈有序 | moves |
| **CSPP** | 堆场→船舶配载，考虑分组约束（你的核心问题） | shifters |
| **CSPP-Constrained** | CSPP + 重量限制 + 冷藏箱约束 + 危险品约束 | shifters + violations |

---

## 算法说明

| 算法 | 类型 | 说明 |
|------|------|------|
| **REINFORCE** | 强化学习 | 策略梯度 + 贪心基线，适合所有问题 |
| **PPO** | 强化学习 | Actor-Critic + GAE，更稳定 |
| **Genetic Algorithm** | 进化算法 | 染色体=动作序列，均匀交叉+随机变异 |
| **Greedy Heuristic** | 启发式 | 深度1贪心基线，用于对比 |

---

## 如何添加自己的算法

在 `algorithms/` 下任意子目录新建 `.py` 文件：

```python
# algorithms/rl/my_dqn.py
from core.base_algorithm import BaseAlgorithm, AlgorithmConfig
import multiprocessing as mp

class MyDQN(BaseAlgorithm):
    name     = "My DQN"        # GUI 下拉菜单中显示的名字
    category = "RL"            # "RL" / "Evolutionary" / "Heuristic"
    description = "自定义 DQN 算法"

    def train(self, problem_factory, result_queue, stop_event):
        env = problem_factory()
        obs, info = env.reset()
        
        # ↓ 在这里写你的算法逻辑
        for step in range(self.config.max_iterations):
            if stop_event.is_set():
                break
            
            action = env.action_space.sample()  # 替换成你的策略
            obs, reward, done, _, info = env.step(action)
            if done:
                obs, info = env.reset()
            
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

**保存文件，重启 GUI** → "My DQN" 自动出现在算法下拉菜单中。

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
├── problems/             5 个问题（可无限扩展）
├── algorithms/           4 个算法（可无限扩展）
│   ├── rl/               强化学习
│   ├── evolutionary/     进化算法
│   └── heuristic/        启发式
├── visualization/        可视化渲染
├── gui/app.py            Streamlit 界面
├── main.py               CLI 入口
└── requirements.txt      依赖列表
```

---

## 常见问题

**Q: streamlit 启动后浏览器没有自动打开？**  
A: 手动在浏览器访问 `http://localhost:8501`

**Q: 训练很慢？**  
A: 减小 `max_iterations`，或用 `Greedy Heuristic` 先测试

**Q: 如何用你现有的 spp-main 代码？**  
A: `problems/cspp.py` 已经基于你的原始 `stowage_gym.py` 逻辑重构，逻辑完全一致

**Q: 如何保存训练结果？**  
A: Experiment Tab 的结果会显示在表格中，后续可加 CSV 导出功能
