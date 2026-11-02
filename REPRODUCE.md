# REPRODUCE：从零复现全部实验

> 所有命令在项目根目录 `D:\paper2` 执行，使用项目内虚拟环境 `.venv`。
> 每个脚本都会创建**新的运行目录** `results/<阶段>/<时间戳>_<名字>_<id>/`，绝不覆盖旧结果。

---

## 0. 环境

```powershell
# Python 3.12（由 uv 管理）
uv python install 3.12
uv venv --python 3.12 .venv

# PyTorch（CUDA 12.8 轮子；无 GPU 时去掉 --index-url 用 CPU 轮子）
uv pip install --python .venv\Scripts\python.exe torch --index-url https://download.pytorch.org/whl/cu128

# 其余依赖 + 以可编辑模式安装本项目
uv pip install --python .venv\Scripts\python.exe -e ".[dev]"

# 验证
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

实测环境（写入每次运行目录的 `system_info.json`）：

| 项 | 值 |
|---|---|
| Python | 3.12.13 |
| PyTorch | 2.11.0+cu128 |
| GPU | NVIDIA GeForce RTX 4070 (12 GB)，驱动 610.88 |
| 操作系统 | Windows |

> **注意**：S1/S2/S4/S5 的风险 DP 在 **CPU + float64** 上运行更合适（张量小、96 步串行，
> GPU 的 kernel 启动开销反而更大；S1 报告里记录了实测对比）。

---

## 1. 全量单元测试

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

覆盖 R1–R10：概率守恒、确定性极限、首次碰撞只计一次、交叉换位、终点占据、等待行为、
DP vs Monte Carlo 二项检验、梯度有限差分、软硬差异、数值稳定性。

---

## 2. S0：环境与几何基础

```powershell
.\.venv\Scripts\python.exe scripts/run_s0.py
```

产出：4 类地图（几何 + SDF）、7 个手工场景、一次带碰撞标记的执行轨迹、
`results/S0_environment/<run>/report.md`，并同步到 `docs/stages/S0/`。

---

## 3. S1：首次碰撞风险 DP 正确性

```powershell
.\.venv\Scripts\python.exe scripts/verify_risk.py
```

产出：R1–R10 测试结果、20 个案例 × 10,000 次 MC 的二项检验、软硬温度 sweep、
soft/hard 进度分布偏差、存活概率曲线、联合进度网格、梯度有限差分误差、计算开销。

---

## 4. S2：不使用 Diffusion 的风险梯度优化

```powershell
.\.venv\Scripts\python.exe scripts/optimize_crossing.py --steps 250 --mc-samples 5000
```

产出：交叉 / 追尾 / 窄门口三个场景的 A1/A2/A3 结果、硬风险—完成时间 Pareto、
等待安全性检查、梯度饱和诊断。

---

## 5. S3：数据集与 Diffusion 训练

```powershell
# 5.1 生成数据（先用 medium 规模验证流程，再按需扩到 full）
.\.venv\Scripts\python.exe scripts/generate_dataset.py --config configs/diffusion_train.yaml --split-size medium

# 5.2 smoke test（600 步，检查 loss/梯度/采样是否正常）
.\.venv\Scripts\python.exe scripts/train_diffusion.py --config configs/diffusion_train.yaml --smoke

# 5.3 正式训练（可加 --time-budget 限制单次时长；支持 --resume 续训）
.\.venv\Scripts\python.exe scripts/train_diffusion.py --config configs/diffusion_train.yaml --steps 40000

# 5.4 单机器人生成质量评估
.\.venv\Scripts\python.exe scripts/sample_single.py --config configs/diffusion_train.yaml --n-tasks 500
```

断点续训：

```powershell
.\.venv\Scripts\python.exe scripts/train_diffusion.py --config configs/diffusion_train.yaml `
    --run-dir results/S3_diffusion_training/<run_id> --resume results/S3_diffusion_training/<run_id>/checkpoints/latest.pt --steps 40000
```

---

## 6. S4：推理期风险引导

```powershell
.\.venv\Scripts\python.exe scripts/run_guided_demo.py --config configs/guidance.yaml
```

自动做四项验证：零引导严格等价、模型参数冻结、端点保持、硬风险/静态/速度独立复核。

---

## 7. S5：对照实验

```powershell
# 预实验（1/5 规模）
.\.venv\Scripts\python.exe scripts/run_benchmarks.py --config configs/benchmark.yaml

# 指定方法/规模
.\.venv\Scripts\python.exe scripts/run_benchmarks.py --config configs/benchmark.yaml `
    --methods B0,B1,B2,B3,A1,A2,A3 --tasks-per-cell 5
```

---

## 8. 汇总报告

```powershell
.\.venv\Scripts\python.exe scripts/make_report.py
```

生成 `docs/final_report.md` 与根目录 `RESULTS.md`。

---

## 9. 结果目录约定

```
results/<阶段>/<run_id>/
    config_snapshot.yaml     实际运行配置
    system_info.json         环境信息
    git_commit.txt
    stage_status.json        状态、耗时、测试结果
    metrics.csv              全部核心指标
    tables/*.csv             曲线原始数值（不允许只有 PNG）
    plots/*.png
    report.md                中文阶段报告
    cases.jsonl / plans/     逐任务记录（S5）
docs/stages/Sx/              上面关键文件的副本，纳入 Git
docs/PROJECT_PROGRESS.md     自动汇总的阶段进度
```

---

## 10. 常见问题

| 现象 | 处理 |
|---|---|
| `torch.cuda.is_available()` 为 False | 检查是否装了 cu 轮子；CPU 也能跑 S0–S2 |
| 训练显存不足 | 训练脚本会自动把 batch 减半并启用梯度累积，并在报告中记录实际配置 |
| 规划很慢 | 风险 DP 是 96 步串行，N=6 约 100–150 ms/次；用 `--robot-counts 2` 缩小规模 |
| 数据集生成慢 | A* + 连续几何复核约 0.3 s/条路径；先用 `--split-size smoke` 验证流程 |
