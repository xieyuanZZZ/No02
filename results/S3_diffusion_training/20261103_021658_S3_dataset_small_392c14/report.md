# 数据集生成报告

- 规模：small
- 输出目录：`D:\paper2\data\dataset`

## train

```json
{
  "tasks_attempted": 126,
  "tasks_reachable": 125,
  "vectors_generated": 244,
  "discarded_unreachable": 6,
  "discarded_too_close": 0,
  "discarded_speed": 2,
  "discarded_static": 0,
  "discarded_length": 0,
  "diversity_mean_m": 0.06477344933974846,
  "n_maps_used": 5,
  "n_paths": 244,
  "elapsed_s": 45.2
}
```

- 丢弃率：6.35%（不可达 6，超速 2，静态碰撞 0）

## val

```json
{
  "tasks_attempted": 53,
  "tasks_reachable": 50,
  "vectors_generated": 97,
  "discarded_unreachable": 9,
  "discarded_too_close": 0,
  "discarded_speed": 0,
  "discarded_static": 0,
  "discarded_length": 0,
  "diversity_mean_m": 0.14470642879203105,
  "n_maps_used": 5,
  "n_paths": 97,
  "elapsed_s": 18.0
}
```

- 丢弃率：16.98%（不可达 9，超速 0，静态碰撞 0）

## test

```json
{
  "tasks_attempted": 63,
  "tasks_reachable": 60,
  "vectors_generated": 117,
  "discarded_unreachable": 8,
  "discarded_too_close": 0,
  "discarded_speed": 1,
  "discarded_static": 0,
  "discarded_length": 0,
  "diversity_mean_m": 0.14603588413417523,
  "n_maps_used": 6,
  "n_paths": 117,
  "elapsed_s": 17.5
}
```

- 丢弃率：14.29%（不可达 8，超速 1，静态碰撞 0）

## 结论

数据已按训练/验证/测试互不重叠的地图种子生成，全部路径通过连续几何碰撞与逐段速度上限复核。
