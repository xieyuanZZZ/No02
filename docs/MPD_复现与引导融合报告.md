# MPD 官方复现与引导融合报告

**日期**：2026-11-03
**官方仓库**：<https://github.com/joaoamcarvalho/mpd-splines-public>（**不是**已废弃的 `jacarvalho/mpd-public`）
**本地目录**：`D:\mpd_official`（独立于 `D:\paper2`，不覆盖 No02 任何代码与结果）
**固定版本**：commit `3676cbf371b0e2a98663d8cef666f671e004e3da`（2025-08-21）

---

## 0. 当前状态（先看这里）

| 项 | 状态 |
|---|---|
| 官方仓库 + 子模块 | ✅ 已克隆并锁定 commit |
| WSL2 Ubuntu 22.04 + GPU | ✅ RTX 4070 / driver 610.88 |
| Python 3.8.19 + torch 2.0.0+cu118 | ✅ 与官方 `environment.yml` 一致 |
| `torch_robotics` / `torchkin` / `experiment_launcher` | ✅ 已安装（torchkin 为纯 Python，无需 C++ 构建） |
| 官方 `EnvSimple2D` | ✅ **构建成功**（纯 PyTorch SDF 网格，0.18 s） |
| `mpd.parametric_trajectory.trajectory_bspline` | ✅ import 通过 |
| `mpd.inference.cost_guides` | ✅ import 通过 |
| `mpd.inference.inference` | ✅ import 通过 |
| Isaac Gym Preview 4 | ❌ **无法获取**（NVIDIA 登录墙）→ import-only shim |
| `pb_ompl`（OMPL 绑定） | ❌ 未编译（py3.8 无 conda 预编译包）→ import-only shim |
| 官方 `data_public.tar.gz`（22.5 GB） | ⚠️ **Google Drive 配额耗尽**（需等最长 24 h） |
| 官方 2D **数据集** | ✅ 已从已下载的 3.5 GB 前缀中抽出（79 MB hdf5，20000 条轨迹） |
| 官方 2D **预训练权重** | ❌ 在压缩包后段的 `data_trained_models/`，前缀够不到 |
| 官方 2D 预训练模型推理 | ⏳ 等权重 |
| 三场景迁移测试 | ⏳ 已写好适配器，等权重 |
| 进度风险引导接入 | ◐ 已实现并通过符号自检，等权重 |

**一句话**：**目前还不能宣称"复现成功"**——官方推理尚未真正跑起来，唯一卡点是预训练权重；
环境、2D 环境适配、引导接口与符号约定都已跑通并验证。

---

## 1. 环境搭建过程（含踩坑记录）

### 1.1 克隆与版本固定

```bash
git clone --recurse-submodules https://github.com/joaoamcarvalho/mpd-splines-public.git D:\mpd_official
```

**问题 1：子模块用 SSH URL，克隆失败。**
`.gitmodules` 里 `pybullet_ompl` 和 `theseus` 写的是 `git@github.com:...`，
在没有 GitHub SSH key 的机器上会 `Host key verification failed`。
**解决**：把 `.gitmodules` 的两个 URL 改成 `https://github.com/...` 后 `git submodule sync && git submodule update --init --recursive`。
这是本地改动，已记录（不改变官方代码内容）。

子模块锁定版本：
```
deps/experiment_launcher  23dbe6125472c6d815d520efa9687c566a8d0135
deps/pybullet_ompl        9a345ac30e7469f1ce0139fc211d9922829f2bbc
deps/theseus              c8583de41824613fb135ee9bad0e930ded6be404
```

### 1.2 WSL2 + Conda

本机是 Windows 11，但**已装 WSL2 Ubuntu 22.04**，正好满足官方要求的系统版本。
GPU 在 WSL 里可见：`nvidia-smi` → RTX 4070 / 610.88 / 12282 MiB。

**问题 2：WSL 里没有 pip，`apt-get install` 卡死（无交互终端 / 源不可达）。**
**解决**：放弃 apt，直接装 Miniconda（自带 pip），`$HOME/miniconda3`。

**问题 3：`conda create` 被 Anaconda 频道 ToS 拦住**（新版 conda 要求接受 `defaults` 条款）。
**解决**：`conda create -y -n mpd -c conda-forge --override-channels python=3.8.19`。

**问题 4：用 `wsl -lc "... &"` 启动的后台进程会在会话结束时被杀。**
表现为 `nohup` 的日志文件根本不存在。
**解决**：所有长任务必须作为**阻塞式前台命令**放进后台作业运行，不能在 WSL 内部再 `&`。

### 1.3 依赖安装

按官方 `environment.yml` 的最小可用集安装（Python 3.8.19）：

```
torch==2.0.0+cu118  torchvision==0.15.1+cu118
einops==0.8.0  h5py==3.11.0  hydra-core==1.3.2  omegaconf==2.3.0
dotmap  scikit-learn  matplotlib  scipy  tqdm  pyyaml  termcolor
torchdiffeq==0.2.4  torch-kmeans==0.2.0  torch-summary==1.4.5  torchlie==0.1.0  trimesh
urdf-parser-py  lxml  wandb  pybullet  colour  ghalton  sobol_seq  vendi_score
pip install -e mpd/torch_robotics
pip install -e deps/theseus/torchkin        # 纯 python，无需 CUDA 编译
pip install -e deps/experiment_launcher
```

**问题 5：官方 `environment.yml` 里 `Hydra==2.5` 不存在**（PyPI 最高 1.3.x）。
**解决**：装 `hydra-core==1.3.2`。

**问题 6：逐个冒出的缺失模块**：`urdf_parser_py`、`lxml`、`wandb`、`colour`、`ghalton`、
`sobol_seq`、`vendi_score`。这些都不在官方 `environment.yml` 的 pip 段里（属于隐式依赖），
用循环脚本逐个补齐。

### 1.4 两个无法获得的依赖（关键）

#### E1：Isaac Gym Preview 4

| 尝试 | 结果 |
|---|---|
| `https://developer.nvidia.com/isaac-gym/download` | HTTP 200，但是登录页 |
| 猜测的直链 `developer.download.nvidia.com/assets/robotics/isaac-gym/IsaacGym_Preview_4_Package.tar.gz` | **HTTP 404** |
| PyPI | 不存在该包 |

Isaac Gym Preview 4 需要 NVIDIA 开发者账号并接受许可协议，**本机无法合法获取**。

**但它对这个任务是不是必需的？实测结论：2D 路径上不需要。**

证据：
1. 全仓库 `isaacgym.` 属性调用只有 3 处，`trajectory_bspline.py` / `trajectory_waypoints.py`
   里只有 `import isaacgym` 一行，**没有任何使用**；
2. 2D 环境的碰撞检测由 `torch_robotics` 的纯 PyTorch SDF 网格完成：
   实测 `EnvSimple2D()` 构建成功，耗时 0.18 s；
3. `mpd.inference.inference` 本体不 import isaacgym（只有 `scripts/inference/inference.py`
   这个入口脚本会 import，因为它还要构造 Isaac Gym 可视化环境）。

**处理**：`wsl_setup/shims/isaacgym/__init__.py` 提供只含 `__version__` 的 shim，
任何真实属性访问都会 `RuntimeError`，避免静默假跑。

#### E2：pb_ompl

`mpd/inference/inference.py` 顶部有 `from pb_ompl.pb_ompl import add_box, fit_bspline_to_path`，
`scripts/generate_data/generate_trajectories.py` 还要 `PbOMPLRobot, PbOMPL, add_sphere, add_box`。

真实 `pb_ompl` 需要编译 OMPL：官方 `setup.sh` 里要 `cmake` + `make update_bindings` + 5 轮 `make -j 32`。
conda-forge 的 `ompl` 只有 py3.13–3.15 的绑定，没有 py3.8。

**但它对 MPD 主路径是不是必需？不是。**

证据：`add_box` 只在 `add_start_goal_marker()`（pybullet 可视化）里用；
`fit_bspline_to_path` 只在 rrtconnect 基线分支里用；`PbOMPL` 只用于数据生成和 OMPL 基线。

**处理**：`wsl_setup/shims/pb_ompl/` 提供占位实现，**调用即 `NotImplementedError`**，
保证 rrtconnect 基线不会被悄悄替换成假实现。

> ⚠️ **所有基于本环境的实验结果都必须标注**：
> 「官方代码 + import-only shim（isaacgym / pb_ompl 未被调用）」。
> 这不是完整的官方环境，不能用来说明「官方 pipeline 在 Isaac Gym 下可运行」。

---

## 2. 配置核查：官方 2D 配置默认确实没启用 MPD

`scripts/inference/cfgs/config_EnvSimple2D-RobotPointMass2D_00.yaml` 原文：

```yaml
model_selection: 'bspline'
planner_alg: 'diffusion_prior'        # ← 官方默认，只有先验，没有代价引导
diffusion_sampling_method: 'ddim'
```

必须改成 `planner_alg: 'mpd'` 才会启用 MPD 的代价引导（这与任务描述完全一致，已核实）。

其它与本任务相关的关键项：

| 键 | 值 | 含义 |
|---|---|---|
| `n_trajectory_samples` | 100 | **候选数（batch）**，不是机器人数 |
| `ddim.ddim_sampling_timesteps` | 15 | DDIM 步数 |
| `ddim.n_guide_steps` | 4 | 每个去噪步内做几次引导 |
| `ddim.guide_lr` | 1.0 | 引导步长 |
| `ddim.max_perturb_x` | 0.15 | 相对扩散均值的最大偏移（信赖域） |
| `ddim.t_start_guide_steps_fraction` | 0.3 | 从 30% 的时间步开始引导 |
| `num_T_pts` | 128 | 轨迹时间采样点数 |
| `obstacle_cutoff_margin_extra` | 0.02 | SDF 代价的截断裕度（2D 用） |
| `costs.CostTaskSpaceCollisionObjects.weight` | 10.0 | 官方 SDF 碰撞代价权重 |

---

## 3. 官方引导接口的精确语义（源码审计）

这是第二阶段的接口契约，逐条从源码核对，不靠猜。

### 3.1 引导在 DDIM 里的调用点

`mpd/models/diffusion_models/diffusion_model_base.py::ddim_sample_loop(...)` 接收：

```python
guide=None, guide_lr=0.05, n_guide_steps=1, max_perturb_x=0.1,
clip_grad=False, clip_grad_rule="value", max_grad_norm=1.0, max_grad_value=1.0,
t_start_guide=torch.inf, scale_grad_by_one_minus_alpha=False,
tt_start_guide_steps_fraction 由上层映射到 t_start_guide
```

### 3.2 guide 函数的签名与返回值

`mpd/models/diffusion_models/sample_functions.py::guide_gradient_steps`：

```python
grad_guide = guide(x_opt, context_d=context_d)      # 只传一个张量，返回同形状张量
grad_guide_clipped = clip_grad_fn(grad_guide)
x_opt.grad = -1.0 * grad_guide_clipped              # ← 注意负号
opt.step()                                          # SGD: x ← x - lr * grad = x + lr * grad_guide
x_delta = torch.clip(x_opt - x_start, -max_perturb_x, max_perturb_x)
x_opt = x_start + x_delta                           # 信赖域：限制相对扩散均值的偏移
x_opt = apply_hard_conditioning(x_opt, hard_conds)  # 重新钉住起终点
```

**结论（正负号约定）**：
`guide` 返回的必须是 **`-∂L/∂x`**（"要最大化的方向"）。
官方 `CostGuideManagerParametricTrajectory.__call__` 末尾正是这样做的：

```python
# -1 because the denoising gradient methods expect an objective function to maximize,
# but we want to minimize the cost
grad_costs_all_wrt_cp_normalized = -1.0 * grad_costs_all_wrt_cp_normalized
```

**这正是任务里警告的陷阱**：如果自定义 guide 直接返回 `∂R/∂P`，符号反了，
风险越大反而越靠近碰撞。

### 3.3 变量的归一化

`CostGuideManager.__call__` 的入参是 **归一化的控制点**
`control_points_normalized: (batch_size, n_control_points, q_dim)`，
内部先 `dataset.unnormalize_control_points(...)` 再算代价与梯度；
返回的梯度也是相对**归一化**控制点的。

所以我们的进度风险代价必须：
```
cp_norm → unnormalize → cp → B-spline → Q(t) → 风险 → 反传 → 对 cp_norm 的梯度 → 取负号
```

### 3.4 batch 的语义（对应任务 §9 的警告）

`GaussianDiffusionModel.run_inference` 开头：

```python
for k, v in hard_conds.items():
    hard_conds[k] = einops.repeat(v, "... -> b ...", b=n_samples)
for k, v in context_d.items():
    context_d[k] = einops.repeat(v, "... -> b ...", b=n_samples)
```

**所有 B 个样本共享同一组起终点条件**，所以 `n_samples` 是"同一条任务的多条候选"，
**不是"多个机器人"**。任务描述的警告属实，已核实。

---

## 4. 第二阶段设计：多机器人同步采样 + 进度风险引导

> 按任务要求，多机器人接入涉及较大改动，先给设计与接口定义，再实施。

### 4.1 张量布局

```
B : 候选数（TaskSuccess@8 对齐时 B=8）
N : 机器人数
K : B-spline 控制点数（官方 2D 为 18）
D : 状态维（2D 点质量机器人 D=2）

控制点张量：  cp  [B*N, K, D]       # 展平后喂给官方 diffusion model
索引约定：    idx = b * N + n        # 固定，所有接口共用
```

**为什么这样排**：官方 model 的 batch 维是第一维，且 `hard_conds` 会被 `repeat` 成 B 份；
我们把 `B*N` 当成 batch，等价于"每个机器人 B 条候选"，
再用 `view(B, N, K, D)` 还原成"一组机器人的联合样本"。

### 4.2 同步去噪的要求

不能"先各自规划再算风险"。必须：

1. 把 N 个机器人的起终点拼成 `B*N` 条条件（每条 = 该机器人的 start/goal）；
2. **一次** DDIM 循环，得到 `cp [B*N, K, D]`；
3. 在**每个引导步**内，把 `cp` reshape 成 `[B, N, K, D]`，
   对每个 b 联合计算首次碰撞风险 `R_b`（N 个机器人一起）；
4. 对 `R = mean_b R_b` 反传到 `cp`，**同时**更新所有机器人的控制点；
5. 继续下一个 DDIM 步。

关键点：同一个 DDIM 步内所有机器人共享同一个噪声水平 `t`，
这样"同步"才有意义（等价于同时执行）。

### 4.3 引导函数接口

```python
def first_hit_guide(cp_norm: Tensor[B*N, K, D], context_d=None) -> Tensor[B*N, K, D]:
    """官方 guide 契约：返回 -∂L/∂cp_norm"""
    cp = dataset.unnormalize_control_points(cp_norm)       # 官方数据集的反归一化
    q = bspline.evaluate(cp, n_T_pts)                      # [B*N, T, D]  B-spline → 稠密轨迹
    q = q.view(B, N, T, D)
    paths_m = to_meters(q)                                 # 官方归一化坐标 → 米
    risk = risk_model(paths_m, wait=0, pause_p, T)         # 我们的首次碰撞 DP
    loss = lam_R * risk["pair_sum_risk"] + lam_T * risk["finish_cost"]
    g = torch.autograd.grad(loss, cp_norm)[0]
    return -g                                              # ← 官方约定的负号
```

### 4.4 与官方 SDF 引导的叠加

官方 `CostGuideManager` 已经算了 `-∂C_SDF/∂cp_norm`。两种做法：

- **方案 A（推荐）**：写一个组合 guide，`return sdf_guide(cp) + w_R * first_hit_guide(cp)`，
  直接替换 `guide=` 参数。改动最小，全部在官方接口内。
- **方案 B**：给 `CostGuideManager` 增加一个 cost 项（加进 `costs:` 配置）。
  更"官方"，但要动 `cost_guides.py` 的注册逻辑，改动大。

先做 A；如果 A 稳定，再考虑 B。

### 4.5 等待变量 w

第一轮 `w=0`，只验证进度风险梯度能否改变几何。
第二轮把 w 作为**扩散之外**的辅助变量，用同一个风险的 `∂L/∂w` 做投影梯度更新
（与我们 S6 的做法一致，已被证明是唯一显著有效的通道）。

---

## 5. 待办（按依赖顺序）

1. ⏳ 等 `data_public.tar.gz` 下完 → 解出 2D 预训练模型与数据集 → 记录 SHA256；
2. ⏳ 跑官方 2D 原始案例（`diffusion_prior`），确认能加载、能采样、能画图、能计时；
3. ⏳ 改 `planner_alg: 'mpd'`，确认 SDF 引导生效（对比 A/B 的最小障碍物净空）；
4. ⏳ 把官方 2D 的障碍物字段替换成我们 S1 的三类场景，跑 A/B 对照；
5. ⏳ 实现 `first_hit_guide`（§4.3），做**符号自检**（风险上升时 guide 值必须下降）；
6. ⏳ 多机器人同步采样（§4.2），先 2 机器人，再 4/6/9；
7. ⏳ 五方法对照 A/B/C/D/E，统计安全完成率、碰撞率、完成时间、推理开销。

---

## 6. 已确认的事实清单（可直接引用）

1. 官方 2D 配置默认 `planner_alg: 'diffusion_prior'`，**不是 MPD**（已读原文）。
2. guide 必须返回 `-∂L/∂x`；官方在 `CostGuideManager` 里显式取负（已读源码）。
3. `n_samples` 是候选数不是机器人数，`run_inference` 会 `einops.repeat` 同一份条件（已读源码）。
4. 引导作用在**归一化控制点**上，内部反归一化（已读源码）。
5. 引导后有 `max_perturb_x` 信赖域裁剪 + 重新施加起终点硬约束（已读源码）。
6. 官方 `EnvSimple2D` 是纯 PyTorch，不依赖 Isaac Gym（已实测构建成功）。
7. 官方 2D 模型路径依赖 `${HOME}/Projects/...`，需按实际解压位置改配置（待下载完成后处理）。

---

## 7. 已完成的代码级验证（不依赖预训练权重）

### 7.1 官方 B-spline + 我们的首碰风险 + 官方 Guide 契约：打通

新增 `D:\mpd_official\mpd_extension\first_hit_guide.py`：

```
cp_norm [B*N, K, D]
   └─ dataset.unnormalize_control_points   (保持计算图)
      └─ ParametricTrajectoryBspline.get_q_trajectory  → q [B*N, 128, D]
         └─ reshape [B, N, T, D]（B=候选，N=机器人）
            └─ FirstHitRisk(q, w=0, pause_p)  → R_soft, J_finish
               └─ loss = λ_R·R + λ_T·J
                  └─ autograd.grad(loss, cp_norm) → 取负 → 返回
```

配置 `FirstHitGuideConfig(lam_R, lam_T, n_agents, batch_candidates, start, goal, clip_value)`；
端点是 B-spline 的硬约束，必须与扩散条件用同一组 `start/goal`，已显式传入。

另有 `SumGuide([官方SDF_guide, first_hit_guide], weights)`，用于"官方 SDF + 进度风险"联合引导。

### 7.2 符号自检：通过

用不饱和的"最小间距"替代目标（同一符号约定），实测：

```
|g| = 1.61
L(cp)          = 1.013517e-01
L(cp + ε·guide)= 9.621563e-02   ← 下降
L(cp - ε·guide)= 1.066212e-01   ← 上升
```

即 `guide = -∂L/∂cp_norm` 的方向约定**正确**，不会出现"风险越大反而更靠近碰撞"。

### 7.3 新发现：首碰软风险只在很窄的间距带里有梯度

对两条平行、间距 `sep` 的直线，用我们真实的 `FirstHitRisk`（`d_safe=0.35`，软核 ε=0.02）：

| 间距 [m] | pair_sum_risk | ‖∂R/∂cp‖ |
|---:|---:|---:|
| 0.00 | 1.000000 | 2.7e-82 |
| 0.15 | 1.000000 | 2.5e-57 |
| 0.30 | 1.000000 | 1.9e-22 |
| 0.34 | 1.000000 | 2.8e-14 |
| 0.38 | 1.000000 | 1.7e-07 |
| **0.50** | **0.254147** | **3.5e+00** |
| 0.80 | 0.000002 | 3.0e-05 |

**解读**：
- 间距 ≤0.38 m 时，首个碰撞概率已经饱和到 1，梯度是 1e-14 以下——**引导对"几乎必然碰撞"的样本无能为力**；
- 间距 ≥0.6 m 时风险约为 0，也没有梯度；
- 只有在 **0.45–0.55 m** 这条窄带里，梯度量级才到 1，引导才真正起作用。

这解释了此前 No02 项目里"几何引导只推动几毫米"的一部分原因：
不是优化器不行，而是**大部分样本落在饱和区，梯度本身就消失了**。

**对第二阶段的影响**：
1. 必须有"梯度有效性"判据：若 `‖guide‖` 低于阈值，就跳过引导并记录（不能假装引导生效）；
2. 可能需要给软核加一个"距离项"（如 `-d` 的线性项）作为饱和区的兜底梯度；
3. `max_perturb_x` 的信赖域要配合这个窄带设计——步子太大直接跨过可行带。

---

## 8. 复现脚本清单（都在 `D:\mpd_official\wsl_setup\`）

| 脚本 | 作用 |
|---|---|
| `install_env.sh` | 建 conda env + 装 torch 2.0.0+cu118 与全部 python 依赖 |
| `download_curl.sh` | 用 `drive.usercontent` 直链断点续传下载 22.5 GB 官方数据 |
| `check_imports.sh` | 逐项定位 import 缺口 |
| `check_shim.sh` | 在 shim 下检查官方模块能否 import |
| `fix_imports_loop.sh` | 循环补齐缺失模块 |
| `run_guide_test.sh` | 跑首碰引导的接口 + 符号 + 饱和测试 |

`shims/` 下是 `isaacgym` 与 `pb_ompl` 两个 import-only 占位实现（调用即报错）。

---

## 9. 数据下载的实际情况（2026-11-03 记录）

| 尝试 | 结果 |
|---|---|
| `gdown --fuzzy` | 首次成功解析并开始下载 22.5 GB，随后报 `Too many users have viewed or downloaded this file` |
| `gdown --continue` | 同一配额错误，**无法续传** |
| `curl` 到 `drive.usercontent.google.com` | 可下载，但**不支持 Range**（`curl: (33) HTTP server doesn't seem to support byte ranges`），断流后无法续传 |
| 已下载前缀 | **3.44 GB / 22.5 GB**（约 15%） |

Google 的提示原文：

> Too many users have viewed or downloaded this file recently. ...
> it may take up to 24 hours to be able to view or download the file.

**已从 3.44 GB 前缀中成功抽取的部分（gzip 流式特性）**：

```
data_public/data_trajectories/EnvSimple2D-RobotPointMass2D-joint_joint-one-RRTConnect/
    dataset_merged_doubled.hdf5                     79 MB   (20000 条轨迹, 256 点/条, task_id)
    dataset_merged_doubled_reload-...-bspline....pickle   6.3 MB
    dataset_merged_doubled_reload-...-waypoints....pickle 6.3 MB
    args.yaml                                       638 B
```

`args.yaml` 关键信息：`bspline_degree=5`、`bspline_num_control_points=30`、
`num_trajectories_generated=10000`、`interpolate_num=256`、`planner=RRTConnect`、`robot_id=RobotPointMass2D`。

**缺的是 `data_public/data_trained_models/` 段**——按 tar 顺序它在 `data_trajectories/` 之后，
需要后续约 19 GB 才能到达。

---

## 10. 拿到 tarball 之后的一键流程

把 `data_public.tar.gz` 放到 `D:\mpd_official\` 或任意路径后，依次执行：

```bash
# 1) 解压到 WSL
mkdir -p ~/mpd_data && cd ~/mpd_data
tar -xzf /mnt/d/mpd_official/data_public.tar.gz
ln -sfn ~/mpd_data/data_public/data_trajectories   /mnt/d/mpd_official/data_trajectories
ln -sfn ~/mpd_data/data_public/data_trained_models /mnt/d/mpd_official/data_trained_models

# 2) 记录权重 SHA256（交付要求）
find /mnt/d/mpd_official/data_trained_models -name '*.pt' -exec sha256sum {} \; > weights_sha256.txt

# 3) 改官方 2D 配置里的三处路径（指向实际解压位置）并启用 MPD
#    scripts/inference/cfgs/config_EnvSimple2D-RobotPointMass2D_00.yaml
#    model_dir_ddpm_bspline / model_dir_cvae_bspline / model_dir_ddpm_waypoints
#    planner_alg: 'diffusion_prior'  ->  'mpd'

# 4) 跑官方原始 2D 案例（A: prior, B: mpd）
bash /mnt/d/mpd_official/wsl_setup/run_official_2d.sh prior
bash /mnt/d/mpd_official/wsl_setup/run_official_2d.sh mpd
```

`run_official_2d.sh` 尚待编写（要等权重就位后按官方 `scripts/inference/inference.py` 的 CLI 参数确定）。

---

## 11. 当前阻塞点（需要你决定）

1. **预训练权重**：Google Drive 配额耗尽，需等最长 24 小时，或你直接把 `data_public.tar.gz`
   （或只把 `data_public/data_trained_models/` 目录打成小包）放到 `D:\mpd_official\`。
2. **Isaac Gym**：官方 tarball 需要 NVIDIA 开发者账号。我已证明 2D 路径不需要它的功能，
   只用 shim 满足 import；如果你要求 100% 官方环境，需要你提供 tarball。

---

## 12. 关于 `D:\第二篇文章` / `~/delay_robust_mmd` 的核查（2026-11-03）

按要求检查了这个目录，结论：**它是另一篇论文（MMD 延迟鲁棒）的工程，不含 MPD 的二维点质量机器人权重。**

### 12.1 里面实际有什么

```
D:\第二篇文章\
  .venv_mpc\            Linux 布局的 venv（只有 numpy/scipy/matplotlib/clarabel，无 torch）
  baselines\external\   Continuous-CBS、SMART
  collision_safety\ learning_windows\ reports\   该论文自己的实验与检查点
```

WSL 侧对应 `~/delay_robust_mmd/`：

```
delay_robust_mmd\mmd\                 MMD 官方仓库（yoraish/mmpd 派生）
    data_trained_models\              5 个场景，全部是 RobotPlanarDisk：
        EnvEmpty2D-RobotPlanarDisk
        EnvEmptyNoWait2D-RobotPlanarDisk
        EnvConveyor2D-RobotPlanarDisk
        EnvDropRegion2D-RobotPlanarDisk
        EnvHighways2D-RobotPlanarDisk
    data_trajectories\                与上面 5 个场景配套
    deps\torch_robotics\              与 MPD 同一套 torch_robotics
```

### 12.2 为什么不能直接拿来做 MPD 基线

| 项 | MPD（我们要的） | MMD（这个仓库） |
|---|---|---|
| 环境 | `EnvSimple2D`（0.6 m 圆 + 矩形障碍） | `EnvEmpty2D` / `EnvHighways2D` 等 |
| 机器人 | `RobotPointMass2D`（纯质点，D=2） | `RobotPlanarDisk`（平面圆盘连杆） |
| 轨迹表示 | **B-spline 控制点**（18/30 个） | 逐路点（无 `parametric_trajectory` 模块） |
| 扩散步数 | 1000（DDPM），DDIM 15 步采样 | 25 步（`n_diffusion_steps: 25`） |
| 用途 | **单机器人**规划器 + 代价引导 | **多机器人** CBS 协调 |

全盘搜索确认：
```
find ~ -name env_simple_2d.py        -> 无
find ~ -name trajectory_bspline.py   -> 无
find ~ -iname "*EnvSimple2D*"        -> 只有我抽出来的数据集
```
`data_trained_models.tar.xz` 里也只有上述 5 个 `RobotPlanarDisk` 场景。

### 12.3 它能提供什么（有价值的副产品）

1. **同一套 `torch_robotics`**（已在 mpd 环境里装好，版本一致）；
2. **MMD 的五组预训练模型**——如果后续要把 MMD 也作为对照基线，可以直接用；
3. 证明这台机器已经跑通过 MMD/MPD 家族的环境（conda env `mmd` 存在）。

### 12.4 仍缺的东西

MPD `EnvSimple2D-RobotPointMass2D` 的预训练权重（在 `data_public.tar.gz` 的
`data_trained_models/` 段，位于压缩包后约 19 GB 处，Drive 配额耗尽拿不到）。

**替代方案**：用官方 MPD 代码 + 我已抽出的官方 EnvSimple2D 数据集（20000 条轨迹），
在这台 4070 上自己训一个 point-mass 模型。这是"官方代码 + 官方数据 + 自训权重"，
必须明确标注**不是**官方预训练权重，不能称为完整复现。
