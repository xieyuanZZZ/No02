# MPD 几何引导：阶段 0 冻结清单

**冻结日期**：2026-11-04
**分支**：`experiment/w2plus-delay-robustness`
**提交**：`799f3ff`
**冻结标签**：`baseline/pre-mpd-geometry-guidance`
**新开发分支**：`experiment/mpd-firsthit-geometry`

状态：工作区干净，本地与远程一致（`799f3ff`），无未提交修改。
本清单完成后才开始几何引导开发；旧结果目录一律不动。

---

## 1. Git 状态（冻结时）

```
branch  : experiment/w2plus-delay-robustness
HEAD    : 799f3ff4690f5411787dbf029880413d1ed46ccc
origin  : 799f3ff4690f5411787dbf029880413d1ed46ccc   （一致）
tag     : baseline/pre-mpd-geometry-guidance -> 799f3ff
```

近期提交：

| 提交 | 内容 |
|---|---|
| `799f3ff` | 高延迟研究：p=0.2 残余主因是优化概率不匹配 |
| `d9e7de0` | W2+ 公平对照：整数搜索显著优于 W3 梯度法 |
| `f9a8f2f` | 实验记录：环境、种子、权重 SHA256 |
| `cf13520` | N=4：First-Hit 等待优化显著优于枚举基线 |
| `b8545fc` | 阶段三：等待优化 W0–W3（N=2） |
| `815fbf7` | 阶段一+二：冻结 24 个 N=2 联合任务 |

---

## 2. 环境

| 项 | 值 |
|---|---|
| 主环境 | Python 3.12.13 / torch 2.11.0+cu128 / RTX 4070 12 GB |
| MPD 侧 | WSL2 Ubuntu 22.04 / Python 3.8.19 / torch 2.0.0+cu118 |
| 官方 MPD | commit `3676cbf371b0e2a98663d8cef666f671e004e3da` |
| 权重 SHA256 | `6632bbb494353e14505ea6ab0f600a5cf240279a1e4dc964608a66ae6aa013d0` |
| 权重文件 | `ema_model_current_state_dict.pth`（4.23 MB） |

---

## 3. 冻结数据文件（SHA256 前 32 位）

| 文件 | SHA256 |
|---|---|
| `data/mpd_delay_plan_bank/plan_bank.json`（N=2，24 任务） | `3c0c4303fceb8a16598ed0101570cb2d` |
| `data/mpd_delay_plan_bank/plan_bank_trajs.npz` | `50b287d67c59139eb72862660b9769c1` |
| `data/mpd_delay_plan_bank/plan_bank_n4.json`（N=4，20 任务） | `b1fdc0a83b746bac93c364306697d151` |
| `data/mpd_delay_plan_bank/plan_bank_n4_trajs.npz` | `c89193ad4f06a5d52f22a04d296aedf1` |
| `data/mpd_delay_plan_bank/task_freeze.json`（N=2 分组） | `5be2485bf8f890130c8b65dc09c66e32` |
| `data/mpd_delay_plan_bank/task_freeze_n4.json`（N=4 分组） | `64d0ba518a167343f71554a1937509fd` |

---

## 4. 冻结结果（SHA256 前 32 位）

| 结果 | SHA256 |
|---|---|
| `results/MPD_S3_wait/final/summary.json`（N=2 W0–W3） | `b037b3a0e6100994ebab9ed7a8d1b508` |
| `results/MPD_N4_wait/final/summary.json`（N=4 W0–W3） | `ea7a2ffb68fab8f6f7d49cc281c80450` |
| `results/MPD_W2Plus/final/summary.json`（W2+ 对照） | `29e75f6ec734093b823766022eb10b69` |
| `results/MPD_HighDelay/final/summary.json`（高延迟） | `9e04558de83cb7a012d4289c7c986f46` |

**以下目录后续只读，禁止覆盖**：`results/MPD_S3_wait/`、`results/MPD_N4_wait/`、
`results/MPD_S2_delay/`、`results/MPD_W2Plus/`、`results/MPD_HighDelay/`。

---

## 5. 基线指标（供几何引导对照）

### N=4，p=0.1（20 任务，3 种子 × 300 MC）

| 方法 | 碰撞率 | 安全完成率 | makespan | 新增等待 |
|---|---:|---:|---:|---:|
| G0/W0 原冻结轨迹 + 原始等待 | 55.81% | 44.19% | 43.7 | 0 |
| W2+@800（最强纯等待，充分搜索） | **4.45%** | **95.55%** | 47.2 | 10.50 |
| W3（可微梯度等待） | 6.63% | 93.37% | 46.9 | 9.55 |

### N=4 高延迟（p=0.2）

| 配置 | 碰撞率 |
|---|---:|
| W0 | 71.58% |
| W3（opt@0.1） | 37.71% |
| W3（opt@0.2） | **20.88%** |
| W2+（opt@0.1） | 36.77% |

**几何引导要打败的目标**：p=0.1 的 W2+@800（4.45%）；p=0.2 的 opt@0.2 W3（20.88%）。

---

## 6. 冻结的自定义代码模块

| 模块 | 作用 |
|---|---|
| `src/fhd/risk/` | First-Hit DP、多对风险、目标函数 |
| `src/fhd/execution/` | 随机暂停执行模拟器、进度传播 |
| `src/fhd/optim/wait_policies.py` | W0/W1/W2/W2+/W3 等待策略 |
| `mpd_extension/first_hit_guide.py` | 首碰风险 Guide（官方契约实现，已符号自检） |
| `mpd_extension/run_mpd_2d.py` | 官方 2D 环境驱动 |
| `mpd_extension/simple_scenes_adapter.py` | No02 场景 → 官方环境适配 |
| `scripts/mpd_s1_build_plan_bank.py` | 轨迹库构建与连续几何审计 |
| `scripts/mpd_w2plus_comparison.py` | W2+ 公平对照 |
| `scripts/mpd_high_delay_study.py` | 高延迟三实验 |

---

## 7. 测试基线

冻结时全量测试：**106 passed / 0 failed**（`pytest tests -q`）。

---

## 8. 恢复方式

```powershell
git checkout baseline/pre-mpd-geometry-guidance   # 恢复冻结状态
.\.venv\Scripts\python.exe -m pytest tests -q     # 106 passed
```

几何引导的全部新产物写到 `results/MPD_Geometry/`，与旧目录隔离。
