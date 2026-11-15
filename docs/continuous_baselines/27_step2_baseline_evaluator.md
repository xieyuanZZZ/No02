# 27 第二步：基线与评价器修复

## 完成内容

### ORCA

- 完成判定改为实际位置 + 终端速度：
  - 终点误差 <0.25 m；
  - 终端速度 <0.25 m/s；
- 不再只看 C++ runner 的 `target>=h`；
- 新增 `elapsed_s` 总运行时间；
- 端点保持逻辑保留。

效果：ORCA 最大终点误差从约 **0.42 m** 降到 **0.019 m**。

### MPC-CBF

- 静态障碍硬约束改为**带 slack 的软约束**；
- slack 进入目标函数，不再轻易让 IPOPT 直接不可行；
- B1 p=0 单例：
  - hard CBF：63 次求解失败；
  - soft CBF：1 次求解失败；
- 但最小间距仍约 **0.348 m**，略低于 0.35 m，不能算完全解决。

### 统一评价器

- 保留机器人间连续子步、静态障碍、真实到达；
- 新增 2 个回归测试：
  - 最后一个动力学区间碰撞；
  - 时间/位置/速度长度不一致；
- 当前 `test_common_evaluator.py + test_wait_policies.py` 共 **11 passed**。

## 可视化

![Step 2 修复前后](D:/No02_external_baseline_v1/results/Continuous_Baseline_Comparison/credible_v2/figures/step2_fixes.png)

## 结论

- ORCA 的到点判定和计时已经修好；
- MPC-CBF 的不可行问题大幅减少，但仍有边界碰撞；
- 评价器测试通过；
- ORCA 静态障碍的动态碰撞仍未完全解决；
- 因此**第二步还没有达到“可以放心跑 1440 条”的状态**，但已经具备继续验证的基础。

## 尚未完成

- ORCA 静态场景仍有机器人碰撞；
- MPC soft 约束仍有 0.348 m 边界违例；
- 还没有用统一评价器对 8 个 v2 任务做 smoke test；
- 1440 条正式实验未启动。

下一步：在 credible_v2 上跑 1 个任务 × p=0/0.1 的 smoke test，确认 ORCA/MPC 的失败率后再启动 1440 条。
