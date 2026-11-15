# 28 credible_v3：12 场景与逐次随机扰动（B4/B6/B8 重设计）

## 场景

- 4 个开阔交叉：A1–A4；
- 8 个静态障碍：
  - B1/B2：单块；
  - B3：双块（障碍外移，净空 **0.60 m**）；
  - B4：两块的**四折线 Z 形**路径；
  - B5：中央单柱；
  - B6：双块 + **斜线/折线绕行**；
  - B7：三块；
  - B8：L 形障碍 + **四条嵌套 L 形路线**。

B4/B6/B8 的路径不再使用水平/垂直同构形式；它们只允许在少数交叉点相遇，避免长距离重合。

## 总览

![credible_v3 场景](D:/No02_external_baseline_v1/results/Continuous_Baseline_Comparison/credible_v3/figures/credible_v3_scenes.png)

## 每次实验扰动

`src/fhd/adapters/scene_variation.py`：

- 起点/终点 ±0.20 m；
- 20% 概率交换某台机器人的起点和终点；
- 障碍中心 ±0.12 m，半宽/半高 ±0.08 m；
- 每次变化后 A* 重新生成 64 点路径；
- 新增**路径重合过滤**：任一对路径若有超过 12/64 个采样点相距 <0.35 m，则该变体拒绝；
- 起点/终点、障碍净空、路径重合都通过才接受。

## 扰动示例

![B8 基础与扰动](D:/No02_external_baseline_v1/results/Continuous_Baseline_Comparison/credible_v3/figures/credible_v3_variants.png)

## 验证

- 12 任务 × 3 seed = 36 个变体；
- 36/36 成功生成；
- B3 净空 0.60 m；
- B4 净空 0.40 m；
- B6 净空 0.50 m；
- B8 净空 0.50 m。

## 冻结信息

- 任务哈希：`13F30AF5...`
- 验证哈希：`B04ABCA5...`
- 计划：12 × 3 p × 20 seeds × 3 方法 = **2160 条**
- 状态：**场景/扰动器已冻结；尚未启动 2160 条随机化实验**

## 注意

- B4/B6/B8 的显式路径与随机 A* 变体不能混用；
- 变体必须由固定 seed 表生成，不能按结果挑选；
- B8 的嵌套 L 路线只在少数点交叉，不再整段重合。
