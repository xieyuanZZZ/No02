# 数据集生成报告

- 规模：medium
- 输出目录：`D:\paper2\data\dataset`

## train

```json
{
  "tasks_attempted": 2604,
  "tasks_reachable": 2500,
  "vectors_generated": 4819,
  "discarded_unreachable": 341,
  "discarded_too_close": 0,
  "discarded_speed": 48,
  "discarded_static": 0,
  "discarded_length": 0,
  "diversity_mean_m": 0.1442186523293872,
  "n_maps_used": 50,
  "n_paths": 4819,
  "elapsed_s": 947.9
}
```

- 丢弃率：14.94%（不可达 341，超速 48，静态碰撞 0）

## val

```json
{
  "tasks_attempted": 261,
  "tasks_reachable": 250,
  "vectors_generated": 473,
  "discarded_unreachable": 33,
  "discarded_too_close": 0,
  "discarded_speed": 16,
  "discarded_static": 0,
  "discarded_length": 0,
  "diversity_mean_m": 0.15049934583570365,
  "n_maps_used": 10,
  "n_paths": 473,
  "elapsed_s": 89.8
}
```

- 丢弃率：18.77%（不可达 33，超速 16，静态碰撞 0）

## test

```json
{
  "tasks_attempted": 256,
  "tasks_reachable": 250,
  "vectors_generated": 489,
  "discarded_unreachable": 20,
  "discarded_too_close": 0,
  "discarded_speed": 3,
  "discarded_static": 0,
  "discarded_length": 0,
  "diversity_mean_m": 0.11678241684433598,
  "n_maps_used": 10,
  "n_paths": 489,
  "elapsed_s": 81.4
}
```

- 丢弃率：8.98%（不可达 20，超速 3，静态碰撞 0）

## 结论

数据已按训练/验证/测试互不重叠的地图种子生成，全部路径通过连续几何碰撞与逐段速度上限复核。
