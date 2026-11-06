# 00 旧离散基线清理与冻结保护报告

**日期**：2026-11-06  
**新分支**：`experiment/continuous-baselines-v1`  
**起点**：`cc5a4ae1bd925908c08db9b345fae61c1672526f`（旧外部基线分支）  
**冻结版本**：`7019d54208c0892a1132a9fddd69af421ef8aa38`  
**冻结 Tag**：`freeze-online-temporal-evaluation-v2-20261105`

---

## 1. 清理决定

Switchable-Edge Search、WinkTPG、STPG 已完成官方示例复现，但其离散网格/步进模型
与 No02 连续 32 航点 + MPC 执行器不匹配。本轮停止适配，不再投入时间；
旧基线不再作为主对比方法。

## 2. 清理内容

从新分支工作区移除：

- `docs/external_baselines/01_reproduction.md`
- `docs/external_baselines/02_comparison_protocol.md`
- `docs/external_baselines/03_pilot_results.md`
- `results/External_Baseline_Comparison/` 下旧原生复现结果

**未删除**：

- `docs/external_baselines/00_freeze_report.md`
- `docs/external_baselines/SHA256_MANIFEST.txt`
- `docs/external_baselines/FREEZE_INFO.txt`
- 冻结 Tag、冻结提交、原 `results/Online_Temporal*`、模型/数据/算法代码

## 3. 历史备份

旧文档/结果归档：

```text
D:\No02_archive\old_external_baselines_cc5a4ae\
  docs_external_baselines\
  results_External_Baseline_Comparison\
  ARCHIVE_INFO.txt
  SHA256_MANIFEST.txt
```

旧第三方仓库/编译目录归档：

```text
D:\No02_archive\old_external_repos_cc5a4ae\
  Switchable-Edge-Search\
  WinkTPG-IEEE-TASE\
  STPG\
```

归档前旧仓库固定提交：

- Switchable：`c60c6b52b9563655b0241b473ba146cbcf25b4b2`
- WinkTPG：`dd96946f225e206de35b041320786eab43210658`
- STPG：`25fb931eff03f1cce23a22a68ab42b7533f85ab3`

旧 Git 提交 `cc5a4ae` 仍保留在远端分支
`experiment/external-baseline-comparison-v1`，不会被强推或改写。

## 4. 冻结保护检查

- 冻结分支 `experiment/online-temporal-evaluation-v2` 仍为 `7019d54`；
- 冻结 Tag 仍为 `bbfc44d8...`，peeled commit `7019d54`；
- 旧备份 `D:\No02_freezes\No02_online_temporal_v2_20261105` 不变；
- 从 `7019d54` 到本清理提交，`src/fhd`、`configs`、`data`、`results/Online_Temporal*` 无改动；
- 没有修改 First-Hit、离线整数搜索、在线联合协调、Diffusion、MPC 执行器、参数或旧结果。

## 5. 新工作目录

```text
docs/continuous_baselines/
scripts/continuous_baselines/
src/fhd/adapters/
configs/continuous_baselines/
results/Continuous_Baseline_Comparison/
  native_reproduction/
  pilot/
  common_mpc/
  figures/
```

后续只允许新增适配器和外部基线结果；我们的算法保持冻结。
