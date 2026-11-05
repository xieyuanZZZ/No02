# 03 小规模公平对照状态

**状态**：**未启动统一 MPC 对照**。这不是运行失败，而是适配条件未满足。

## 已完成

- 三种外部方法均已在官方输入上编译、运行并保存原始输出：
  - Switchable-Edge Search：官方 `example/path.txt`；
  - WinkTPG：官方 `example_path.txt`，`-m 1` 与 `-m 0`；
  - STPG：官方 `data/benchmark/test_PBS2_delay_p01` 实例。
- 版本、许可证、构建命令、日志和输出在
  `results/External_Baseline_Comparison/native_reproduction/`。

## 未启动原因

统一对照要求“原方法的通行顺序/等待/速度规划”进入 No02 的连续 MPC 执行器。当前上游程序：

- Switchable：只输出聚合统计和每 step 计数，没有逐机器人切换/等待动作；
- WinkTPG：只输出聚合 CSV，没有逐机器人 action/speed profile；
- STPG：输入是离散有向网格 + situation JSON，No02 没有等价生成器。

如果现在直接拿它们的原生 `solution cost`/`runtime` 与 No02 的碰撞率比较，会把
**不同执行模型**混在一起；如果只把路径四舍五入到网格，又会改变冲突拓扑和切换语义。
因此本阶段不产生“外部基线在统一 MPC 下优于/劣于我们”的数字。

## 下一步最小工作

1. 给 WinkTPG 加**非侵入式导出**：在 `driver.cpp` 每个 `exec_actions` 轮次后 dump
   `agent, loc, orient, start_t, cost`；保留原生输出作为单独对照；
2. 给 Switchable 加导出：输出 replanned ADG 的每 agent 状态时间，或至少每 step 的
   switchable edge 选择；
3. 为 STPG 写 situation 生成器（从 No02 暂停事件映射到离散 step），并验证
   switchable-edge 语义；
4. 写 `src/fhd/adapters/` 的 `stop_at`/`speed_profile` 转换，跑 2 任务 × 3 seeds smoke；
5. 通过后才进入 50-seed 主实验。

## 明确禁止

- 不得用“论文简化重写”冒充原方法；
- 不得把原生统计和 No02 碰撞率混合平均；
- 不得按外部方法的成绩筛选场景或删除超时样本。
