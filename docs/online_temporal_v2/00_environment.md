# 00 在线时序评估 v2：环境、基线与保护项

**日期**：2026-11-05  
**仓库**：`https://github.com/xieyuanZZZ/No02`  
**审查基线**：`experiment/online-temporal-coordination` @ `f5f7f737db2ae7a644734d93b703a5fd4239ec29`  
**本轮分支**：`experiment/online-temporal-evaluation-v2`  
**远端核对**：`git fetch origin` 后远端 HEAD 仍为 `f5f7f73`；本地新建分支未覆盖任何已有提交。

**环境**：Python 3.12.13，numpy 2.5.4，scipy 1.18.1，torch 2.11.0+cu128，OSQP 1.1.3，
Windows 16 逻辑核。  
**工作区**：开工时干净（`git status` 无修改）。

**保留项（只读）**：

- `results/Online_Temporal/03_ab_comparison/` 旧三方法结果；
- `docs/online_temporal/03_ablation_results.md` 旧报告；
- `data/mpd_delay_plan_bank/` 冻结任务、路径、基线等待；
- `configs/base.yaml` 的 MPC/碰撞/执行参数。

**本轮新增结果目录**：

- `results/Online_Temporal_v2/01_pilot/`：8 个 N4 任务、5 方法、4 个 p、30 seeds；
- `results/Online_Temporal_v2/02_smoke/`：快速机制检查；
- `results/Online_Temporal_v2/03_full_n4/`：若全量算力足够则写入，未完成会在报告中明确标注；
- `docs/online_temporal_v2/`：修复、复测、组合、规模化报告。
