# 28 credible_v3：12 场景与逐次随机扰动

## 场景

- 4 个开阔交叉；
- 8 个静态障碍任务：
  - B1/B2：单块；
  - B3：双块（B3 障碍已外移，净空从 0.20 m 提升到 **0.60 m**）；
  - B4：错位双块；
  - B5：中央单柱；
  - B6：双块绕行；
  - B7：三块障碍；
  - B8：L 形障碍。
- B6/B7/B8 的起终点不能由直线直接连接，参考路径由 A* 绕行生成。

## 总览

![credible_v3 场景](D:/No02_external_baseline_v1/results/Continuous_Baseline_Comparison/credible_v3/figures/credible_v3_scenes.png)

## 每次实验的随机扰动

新增 `src/fhd/adapters/scene_variation.py`：

- 起点/终点位置 jitter：约 ±0.20 m；
- 20% 概率交换某台机器人的起点和终点；
- 障碍物中心 jitter：约 ±0.12 m；
- 障碍物半宽/半高 jitter：约 ±0.08 m；
- 每次扰动后重新用 A* 生成 64 点参考路径；
- 路径必须满足起点/终点间距和障碍净空，否则重试。

## 扰动示例

![B8 基础与三个扰动变体](D:/No02_external_baseline_v1/results/Continuous_Baseline_Comparison/credible_v3/figures/credible_v3_variants.png)

## 生成验证

- 12 个任务 × 3 个 seed = 36 个扰动场景；
- 36 个全部成功生成合法路径；
- 失败 0。

## 配置

- 任务哈希：`4EC9B68B...`
- 验证哈希：`0B72B52F...`
- 计划：12 × 3 p × 20 seeds × 3 方法 = **2160 条**
- 状态：**场景和扰动器已冻结；尚未启动 2160 条随机化实验**

## 注意

- B3 净空 0.60 m，已明显高于 0.20 m；
- B8 等任务的路径可能贴近障碍但满足 0.20 m 以上净空；
- 随机扰动后的路径由 A* 重新生成，不能与固定 manifest 路径混用；
- 正式实验必须固定随机种子表，不能根据结果挑选变体。
