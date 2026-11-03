# 数据集生成报告

- 规模：large
- 输出目录：`D:\paper2\data\dataset_large`

## train

```json
{
  "tasks_attempted": 10411,
  "tasks_reachable": 10000,
  "vectors_generated": 19212,
  "discarded_unreachable": 1443,
  "discarded_too_close": 0,
  "discarded_speed": 167,
  "discarded_static": 0,
  "discarded_length": 0,
  "diversity_mean_m": 0.1484607508287305,
  "n_maps_used": 100,
  "n_paths": 19212,
  "elapsed_s": 3799.6
}
```

- 丢弃率：15.46%（不可达 1443，超速 167，静态碰撞 0）

## val

```json
{
  "tasks_attempted": 519,
  "tasks_reachable": 500,
  "vectors_generated": 950,
  "discarded_unreachable": 70,
  "discarded_too_close": 0,
  "discarded_speed": 18,
  "discarded_static": 0,
  "discarded_length": 0,
  "diversity_mean_m": 0.13608649282868,
  "n_maps_used": 20,
  "n_paths": 950,
  "elapsed_s": 174.2
}
```

- 丢弃率：16.96%（不可达 70，超速 18，静态碰撞 0）

## test

```json
{
  "tasks_attempted": 514,
  "tasks_reachable": 500,
  "vectors_generated": 958,
  "discarded_unreachable": 61,
  "discarded_too_close": 0,
  "discarded_speed": 9,
  "discarded_static": 0,
  "discarded_length": 0,
  "diversity_mean_m": 0.11968414501628749,
  "n_maps_used": 20,
  "n_paths": 958,
  "elapsed_s": 171.0
}
```

- 丢弃率：13.62%（不可达 61，超速 9，静态碰撞 0）

## 结论

数据已按训练/验证/测试互不重叠的地图种子生成，全部路径通过连续几何碰撞与逐段速度上限复核。
