# 24 大规模测试准备

## 当前冻结状态

- 方法：`Ours-W2plus-relaxed-v1`
- 冻结 Tag：`freeze-ours-w2relaxed-v1`
- 冻结提交：`d9b603a1abb024a4176bac3025104ce4758a30d6`
- 数据：`results/Continuous_Baseline_Comparison/credible_v1/frozen_v1/`
- 规模：4 任务 × 3 p × 5 seeds × 3 方法 = 180 条

## 是否可以直接大规模？

**部分可以，但仍需先扩大任务集。**

当前 180 条已经覆盖：

- p=0/0.1/0.2；
- 每条件 5 seeds；
- Ours / ORCA / MPC-CBF；
- 开阔和静态障碍两类场景。

但论文门槛还需要：

- 9 个左右任务，而不是 4 个；
- p=0 也应有 20 seeds（或至少 5 seeds，与延迟条件一致）；
- 统一评价器重算全部原始轨迹；
- 95% 置信区间；
- ORCA/MPC-CBF 的障碍结果单独说明局限性。

## 大规模协议（冻结版）

建议：

| 项 | 值 |
|---|---|
| 任务 | 9 个（A/B 各 4–5 个） |
| p | 0、0.1、0.2（可加 0.05） |
| seeds | 每条件 20 |
| 方法 | Ours、ORCA、MPC-CBF |
| 记录数 | 9×3×20×3 = 1620 |

命令形式：

```powershell
python scripts/continuous_baselines/run_credible_pilot.py `
  --manifest results/Continuous_Baseline_Comparison/credible_v1/task_manifest.json `
  --ps 0,0.1,0.2 `
  --n-deterministic 20 `
  --n-delay 20 `
  --out results/Continuous_Baseline_Comparison/credible_v1/large_v1
```

脚本现在会读取已有 `per_run.jsonl` 并跳过相同
`(method, task_id, p, seed)`，因此可断点续跑；每条立即追加。

## 启动前仍缺

1. 任务集从 4 扩到 9，且先冻结 manifest；
2. 统一评价器对全部 NPZ 重算；
3. ORCA 障碍结果标注为“未稳定”，不能把静态场景当公平排名；
4. MPC-CBF 硬障碍约束不可行问题需在论文中说明；
5. 生成表 1–3 和图 1–4。

结论：**冻结已完成，180 条小规模主表可用；真正大规模测试还差 5 个任务和正式评价/图表流程，不能把 180 条说成论文最终表。**
