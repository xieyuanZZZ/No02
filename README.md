# First-Hit Risk-Guided Diffusion for 2D Multi-Robot Trajectory Planning
> **第二版方案执行报告（S0–S6）**：[docs/revised/06_final_report.md](docs/revised/06_final_report.md)

> **审计与修复报告（第 1 轮）**：[docs/audit_fix/05_round1_report.md](docs/audit_fix/05_round1_report.md)

> **Online First-Hit v2 阶段一（MPC 审计与可信基准）**：
> [docs/Online_FirstHit_v2_阶段一_MPC审计与可信基准.md](docs/Online_FirstHit_v2_阶段一_MPC审计与可信基准.md)

> **Online First-Hit v2 阶段二（动力学风险 M3 与冻结盲测）**：
> [docs/Online_FirstHit_v2_阶段二_动力学风险建模与盲测.md](docs/Online_FirstHit_v2_阶段二_动力学风险建模与盲测.md)

# 可微首次碰撞风险引导的二维多机器人扩散轨迹规划

[中文](#中文) | [English](#english)

---

## 中文

从零实现的独立研究工程：在一个**自行训练的单机器人二维条件扩散模型**之上，用
"**随机执行进度的首次碰撞概率传播**"构造**可微风险引导**，让多机器人在随机暂停下生成
更安全、又不过度牺牲完成时间的时空轨迹。

### 核心链条

```
静态地图 + N 组起终点
        -> 条件 Diffusion（自行训练，1D U-Net）预测 N 条几何轨迹 Q
        -> 可微随机执行风险模块
             · 进度状态转移概率 a_{i,m}(t)
             · 联合进度概率传播 F_t(m,n)
             · 首次碰撞吸收
        -> ∇_Q R_soft 与 ∇_w R_soft
        -> 修正当前干净轨迹预测 x̂0 与等待 w
        -> DDIM 继续去噪 -> 最终 Q + w + 安全复核
```

### 目录

```
configs/          全部实验参数的唯一来源（YAML）
src/fhd/          核心实现
  geometry/       连续几何、线段/相对运动最小距离、SDF
  maps/           4 类地图生成器与手工场景
  data/           A* 路径搜索、数据集构建与预处理
  execution/      随机执行进度模型与联合 Monte Carlo 模拟器
  risk/           首次碰撞概率 DP（不依赖 diffusion）
  diffusion/      DDPM 调度、条件 1D U-Net、DDIM、引导 DDIM
  optim/          不依赖 Diffusion 的 Q/w 投影梯度优化
  evaluation/     基线、指标、统计、可视化、报告
scripts/          可执行入口（见 REPRODUCE.md）
tests/            R1–R10 等正确性测试
docs/stages/Sx/   每个阶段的报告、指标、图（纳入 Git）
results/Sx/       每次运行的完整记录（不入 Git）
```

### 快速开始

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe torch --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv\Scripts\python.exe -e ".[dev]"

.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe scripts/run_s0.py
.\.venv\Scripts\python.exe scripts/verify_risk.py
```

可视化图集见 [docs/VISUALIZATIONS.md](docs/VISUALIZATIONS.md)（含场景仿真、时空图、动画 GIF）。

完整流程见 [REPRODUCE.md](REPRODUCE.md)，方法细节见 [METHOD.md](METHOD.md)，
实验结果见 [RESULTS.md](RESULTS.md) 与 [docs/final_report.md](docs/final_report.md)。

### 重要声明

- 软碰撞代理值（`R_soft`）**不是**真实碰撞概率，只用于优化；
- `R_hard`（hard gate + hard collision）在层 A 假设下才是精确首次碰撞概率；
- 两两风险之和**不是**全队"至少碰撞一次"的概率，全队碰撞率由联合 Monte Carlo 估计；
- 本项目**不提供**任何 MPC / 连续动力学闭环下的安全保证（S6 为独立后续工作）。

---

## English

An independent, from-scratch research project: starting from a **self-trained single-robot 2D
conditional diffusion model**, we build a **differentiable first-hit collision risk** by propagating
joint progress probabilities under **random execution pauses**, and use its gradient to guide DDIM
sampling for multiple robots — trading safety against makespan without retraining the diffusion
network.

### Pipeline

```
static map + N (start, goal) pairs
        -> conditional diffusion (trained from scratch, 1D U-Net) -> N geometric paths Q
        -> differentiable random-execution risk module
             - progress transition probabilities a_{i,m}(t)
             - joint progress propagation F_t(m,n)
             - first-hit absorption
        -> gradients dL/dQ and dL/dw
        -> correct the current clean prediction x̂0 and the waiting plan w
        -> DDIM continues -> final Q + w + independent safety re-check
```

### Honest scope

- `R_soft` is a **differentiable surrogate**, never a calibrated collision probability;
- `R_hard` (hard gate + hard collision) is an exact first-hit probability **only under the discrete
  progress execution model** of layer A;
- pairwise risk sums are **not** the team collision probability; team-level rates come from joint
  Monte Carlo;
- no closed-loop MPC/dynamics safety claim is made anywhere in this repository.

### Quick start

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe torch --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv\Scripts\python.exe -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest tests -q
```

See [REPRODUCE.md](REPRODUCE.md) for the full pipeline and [METHOD.md](METHOD.md) for details.
