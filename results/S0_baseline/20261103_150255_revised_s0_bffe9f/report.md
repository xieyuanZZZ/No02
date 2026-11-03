# S0 旧版本冻结报告（第二版方案）

- 冻结时间：2026-11-03T15:03:11+08:00
- 分支：`experiment/revised-scene-risk-guidance`  HEAD：`ea842416331801e7f7710d4dc7f7d52fb3cc47c7`
- 工作区干净：**False**
- 已有 tag：baseline_before_fix

## 环境

| 项 | 值 |
|---|---|
| Python | 3.12.13 |
| PyTorch | 2.11.0+cu128 |
| CUDA | 12.8 |
| GPU | NVIDIA GeForce RTX 4070 |

## 修复前测试状态

- 通过 79 / 失败 0 / 错误 0 / 跳过 0

## 已冻结的资产

- 实验运行：**53** 个
- 模型检查点：**154** 个（含 SHA256）
- 数据集文件：**8** 个（含 SHA256）
- 配置：**7** 份（YAML 原文 + 解析后的完整配置）

## 实际生效配置的核对（YAML vs 训练时）

| checkpoint | 步数 | map_conditioning | map_res | batch | 训练步数设置 |
|---|---:|---|---:|---:|---:|
| `results\S3_diffusion_training\20261103_023715_s3_train_59ba46\checkpoints\best.pt` | 500 | None | None | 96 | 40000 |
| `results\S3_diffusion_training\20261103_023715_s3_train_59ba46\checkpoints\latest.pt` | 600 | None | None | 96 | 40000 |
| `results\S3_diffusion_training\20261103_023715_s3_train_59ba46\checkpoints\step_000300.pt` | 300 | None | None | 96 | 40000 |
| `results\S3_diffusion_training\20261103_023715_s3_train_59ba46\checkpoints\step_000600.pt` | 600 | None | None | 96 | 40000 |
| `results\S3_diffusion_training\20261103_023738_s3_train_a1df62\checkpoints\best.pt` | 26500 | None | None | 96 | 40000 |
| `results\S3_diffusion_training\20261103_023738_s3_train_a1df62\checkpoints\latest.pt` | 40000 | None | None | 96 | 40000 |
| `results\S3_diffusion_training\20261103_023738_s3_train_a1df62\checkpoints\step_002000.pt` | 2000 | None | None | 96 | 40000 |
| `results\S3_diffusion_training\20261103_023738_s3_train_a1df62\checkpoints\step_004000.pt` | 4000 | None | None | 96 | 40000 |
| `results\S3_diffusion_training\20261103_023738_s3_train_a1df62\checkpoints\step_006000.pt` | 6000 | None | None | 96 | 40000 |
| `results\S3_diffusion_training\20261103_023738_s3_train_a1df62\checkpoints\step_008000.pt` | 8000 | None | None | 96 | 40000 |
| `results\S3_diffusion_training\20261103_023738_s3_train_a1df62\checkpoints\step_010000.pt` | 10000 | None | None | 96 | 40000 |
| `results\S3_diffusion_training\20261103_023738_s3_train_a1df62\checkpoints\step_012000.pt` | 12000 | None | None | 96 | 40000 |
| `results\S3_diffusion_training\20261103_023738_s3_train_a1df62\checkpoints\step_014000.pt` | 14000 | None | None | 96 | 40000 |
| `results\S3_diffusion_training\20261103_023738_s3_train_a1df62\checkpoints\step_016000.pt` | 16000 | None | None | 96 | 40000 |
| `results\S3_diffusion_training\20261103_023738_s3_train_a1df62\checkpoints\step_018000.pt` | 18000 | None | None | 96 | 40000 |
| `results\S3_diffusion_training\20261103_023738_s3_train_a1df62\checkpoints\step_020000.pt` | 20000 | None | None | 96 | 40000 |
| `results\S3_diffusion_training\20261103_023738_s3_train_a1df62\checkpoints\step_022000.pt` | 22000 | None | None | 96 | 40000 |
| `results\S3_diffusion_training\20261103_023738_s3_train_a1df62\checkpoints\step_024000.pt` | 24000 | None | None | 96 | 40000 |
| `results\S3_diffusion_training\20261103_023738_s3_train_a1df62\checkpoints\step_026000.pt` | 26000 | None | None | 96 | 40000 |
| `results\S3_diffusion_training\20261103_023738_s3_train_a1df62\checkpoints\step_028000.pt` | 28000 | None | None | 96 | 40000 |
| `results\S3_diffusion_training\20261103_023738_s3_train_a1df62\checkpoints\step_030000.pt` | 30000 | None | None | 96 | 40000 |
| `results\S3_diffusion_training\20261103_023738_s3_train_a1df62\checkpoints\step_032000.pt` | 32000 | None | None | 96 | 40000 |
| `results\S3_diffusion_training\20261103_023738_s3_train_a1df62\checkpoints\step_034000.pt` | 34000 | None | None | 96 | 40000 |
| `results\S3_diffusion_training\20261103_023738_s3_train_a1df62\checkpoints\step_036000.pt` | 36000 | None | None | 96 | 40000 |
| `results\S3_diffusion_training\20261103_023738_s3_train_a1df62\checkpoints\step_038000.pt` | 38000 | None | None | 96 | 40000 |
| `results\S3_diffusion_training\20261103_023738_s3_train_a1df62\checkpoints\step_040000.pt` | 40000 | None | None | 96 | 40000 |
| `results\S3_diffusion_training\20261103_025625_s3_train_b9efe1\checkpoints\best.pt` | 6500 | None | None | 96 | 40000 |
| `results\S3_diffusion_training\20261103_025625_s3_train_b9efe1\checkpoints\latest.pt` | 8000 | None | None | 96 | 40000 |
| `results\S3_diffusion_training\20261103_025625_s3_train_b9efe1\checkpoints\step_002000.pt` | 2000 | None | None | 96 | 40000 |
| `results\S3_diffusion_training\20261103_025625_s3_train_b9efe1\checkpoints\step_004000.pt` | 4000 | None | None | 96 | 40000 |
| `results\S3_diffusion_training\20261103_025625_s3_train_b9efe1\checkpoints\step_006000.pt` | 6000 | None | None | 96 | 40000 |
| `results\S3_diffusion_training\20261103_025625_s3_train_b9efe1\checkpoints\step_008000.pt` | 8000 | None | None | 96 | 40000 |
| `results\S3_diffusion_training\20261103_030603_s3_train_383287\checkpoints\best.pt` | 22500 | None | None | 96 | 40000 |
| `results\S3_diffusion_training\20261103_030603_s3_train_383287\checkpoints\latest.pt` | 40000 | None | None | 96 | 40000 |
| `results\S3_diffusion_training\20261103_030603_s3_train_383287\checkpoints\step_002000.pt` | 2000 | None | None | 96 | 40000 |
| `results\S3_diffusion_training\20261103_030603_s3_train_383287\checkpoints\step_004000.pt` | 4000 | None | None | 96 | 40000 |
| `results\S3_diffusion_training\20261103_030603_s3_train_383287\checkpoints\step_006000.pt` | 6000 | None | None | 96 | 40000 |
| `results\S3_diffusion_training\20261103_030603_s3_train_383287\checkpoints\step_008000.pt` | 8000 | None | None | 96 | 40000 |
| `results\S3_diffusion_training\20261103_030603_s3_train_383287\checkpoints\step_010000.pt` | 10000 | None | None | 96 | 40000 |
| `results\S3_diffusion_training\20261103_030603_s3_train_383287\checkpoints\step_012000.pt` | 12000 | None | None | 96 | 40000 |
| `results\S3_diffusion_training\20261103_030603_s3_train_383287\checkpoints\step_014000.pt` | 14000 | None | None | 96 | 40000 |
| `results\S3_diffusion_training\20261103_030603_s3_train_383287\checkpoints\step_016000.pt` | 16000 | None | None | 96 | 40000 |
| `results\S3_diffusion_training\20261103_030603_s3_train_383287\checkpoints\step_018000.pt` | 18000 | None | None | 96 | 40000 |
| `results\S3_diffusion_training\20261103_030603_s3_train_383287\checkpoints\step_020000.pt` | 20000 | None | None | 96 | 40000 |
| `results\S3_diffusion_training\20261103_030603_s3_train_383287\checkpoints\step_022000.pt` | 22000 | None | None | 96 | 40000 |
| `results\S3_diffusion_training\20261103_030603_s3_train_383287\checkpoints\step_024000.pt` | 24000 | None | None | 96 | 40000 |
| `results\S3_diffusion_training\20261103_030603_s3_train_383287\checkpoints\step_026000.pt` | 26000 | None | None | 96 | 40000 |
| `results\S3_diffusion_training\20261103_030603_s3_train_383287\checkpoints\step_028000.pt` | 28000 | None | None | 96 | 40000 |
| `results\S3_diffusion_training\20261103_030603_s3_train_383287\checkpoints\step_030000.pt` | 30000 | None | None | 96 | 40000 |
| `results\S3_diffusion_training\20261103_030603_s3_train_383287\checkpoints\step_032000.pt` | 32000 | None | None | 96 | 40000 |
| `results\S3_diffusion_training\20261103_030603_s3_train_383287\checkpoints\step_034000.pt` | 34000 | None | None | 96 | 40000 |
| `results\S3_diffusion_training\20261103_030603_s3_train_383287\checkpoints\step_036000.pt` | 36000 | None | None | 96 | 40000 |
| `results\S3_diffusion_training\20261103_030603_s3_train_383287\checkpoints\step_038000.pt` | 38000 | None | None | 96 | 40000 |
| `results\S3_diffusion_training\20261103_030603_s3_train_383287\checkpoints\step_040000.pt` | 40000 | None | None | 96 | 40000 |
| `results\S3_diffusion_training\20261103_040239_s3_train_40fcfd\checkpoints\best.pt` | 28000 | None | None | 96 | 40000 |
| `results\S3_diffusion_training\20261103_040239_s3_train_40fcfd\checkpoints\latest.pt` | 50000 | None | None | 96 | 40000 |
| `results\S3_diffusion_training\20261103_040239_s3_train_40fcfd\checkpoints\step_002000.pt` | 2000 | None | None | 96 | 40000 |
| `results\S3_diffusion_training\20261103_040239_s3_train_40fcfd\checkpoints\step_004000.pt` | 4000 | None | None | 96 | 40000 |
| `results\S3_diffusion_training\20261103_040239_s3_train_40fcfd\checkpoints\step_006000.pt` | 6000 | None | None | 96 | 40000 |
| `results\S3_diffusion_training\20261103_040239_s3_train_40fcfd\checkpoints\step_008000.pt` | 8000 | None | None | 96 | 40000 |
| `results\S3_diffusion_training\20261103_040239_s3_train_40fcfd\checkpoints\step_010000.pt` | 10000 | None | None | 96 | 40000 |
| `results\S3_diffusion_training\20261103_040239_s3_train_40fcfd\checkpoints\step_012000.pt` | 12000 | None | None | 96 | 40000 |
| `results\S3_diffusion_training\20261103_040239_s3_train_40fcfd\checkpoints\step_014000.pt` | 14000 | None | None | 96 | 40000 |
| `results\S3_diffusion_training\20261103_040239_s3_train_40fcfd\checkpoints\step_016000.pt` | 16000 | None | None | 96 | 40000 |
| `results\S3_diffusion_training\20261103_040239_s3_train_40fcfd\checkpoints\step_018000.pt` | 18000 | None | None | 96 | 40000 |
| `results\S3_diffusion_training\20261103_040239_s3_train_40fcfd\checkpoints\step_020000.pt` | 20000 | None | None | 96 | 40000 |
| `results\S3_diffusion_training\20261103_040239_s3_train_40fcfd\checkpoints\step_022000.pt` | 22000 | None | None | 96 | 40000 |
| `results\S3_diffusion_training\20261103_040239_s3_train_40fcfd\checkpoints\step_024000.pt` | 24000 | None | None | 96 | 40000 |
| `results\S3_diffusion_training\20261103_040239_s3_train_40fcfd\checkpoints\step_026000.pt` | 26000 | None | None | 96 | 40000 |
| `results\S3_diffusion_training\20261103_040239_s3_train_40fcfd\checkpoints\step_028000.pt` | 28000 | None | None | 96 | 40000 |
| `results\S3_diffusion_training\20261103_040239_s3_train_40fcfd\checkpoints\step_030000.pt` | 30000 | None | None | 96 | 40000 |
| `results\S3_diffusion_training\20261103_040239_s3_train_40fcfd\checkpoints\step_032000.pt` | 32000 | None | None | 96 | 40000 |
| `results\S3_diffusion_training\20261103_040239_s3_train_40fcfd\checkpoints\step_034000.pt` | 34000 | None | None | 96 | 40000 |
| `results\S3_diffusion_training\20261103_040239_s3_train_40fcfd\checkpoints\step_036000.pt` | 36000 | None | None | 96 | 40000 |
| `results\S3_diffusion_training\20261103_040239_s3_train_40fcfd\checkpoints\step_038000.pt` | 38000 | None | None | 96 | 40000 |
| `results\S3_diffusion_training\20261103_040239_s3_train_40fcfd\checkpoints\step_040000.pt` | 40000 | None | None | 96 | 40000 |
| `results\S3_diffusion_training\20261103_040239_s3_train_40fcfd\checkpoints\step_042000.pt` | 42000 | None | None | 96 | 40000 |
| `results\S3_diffusion_training\20261103_040239_s3_train_40fcfd\checkpoints\step_044000.pt` | 44000 | None | None | 96 | 40000 |
| `results\S3_diffusion_training\20261103_040239_s3_train_40fcfd\checkpoints\step_046000.pt` | 46000 | None | None | 96 | 40000 |
| `results\S3_diffusion_training\20261103_040239_s3_train_40fcfd\checkpoints\step_048000.pt` | 48000 | None | None | 96 | 40000 |
| `results\S3_diffusion_training\20261103_040239_s3_train_40fcfd\checkpoints\step_050000.pt` | 50000 | None | None | 96 | 40000 |
| `results\S3_diffusion_training\20261103_042234_s3_train_4b11c4\checkpoints\best.pt` | 37500 | None | 16 | 96 | 40000 |
| `results\S3_diffusion_training\20261103_042234_s3_train_4b11c4\checkpoints\latest.pt` | 50000 | None | 16 | 96 | 40000 |
| `results\S3_diffusion_training\20261103_042234_s3_train_4b11c4\checkpoints\step_002000.pt` | 2000 | None | 16 | 96 | 40000 |
| `results\S3_diffusion_training\20261103_042234_s3_train_4b11c4\checkpoints\step_004000.pt` | 4000 | None | 16 | 96 | 40000 |
| `results\S3_diffusion_training\20261103_042234_s3_train_4b11c4\checkpoints\step_006000.pt` | 6000 | None | 16 | 96 | 40000 |
| `results\S3_diffusion_training\20261103_042234_s3_train_4b11c4\checkpoints\step_008000.pt` | 8000 | None | 16 | 96 | 40000 |
| `results\S3_diffusion_training\20261103_042234_s3_train_4b11c4\checkpoints\step_010000.pt` | 10000 | None | 16 | 96 | 40000 |
| `results\S3_diffusion_training\20261103_042234_s3_train_4b11c4\checkpoints\step_012000.pt` | 12000 | None | 16 | 96 | 40000 |
| `results\S3_diffusion_training\20261103_042234_s3_train_4b11c4\checkpoints\step_014000.pt` | 14000 | None | 16 | 96 | 40000 |
| `results\S3_diffusion_training\20261103_042234_s3_train_4b11c4\checkpoints\step_016000.pt` | 16000 | None | 16 | 96 | 40000 |
| `results\S3_diffusion_training\20261103_042234_s3_train_4b11c4\checkpoints\step_018000.pt` | 18000 | None | 16 | 96 | 40000 |
| `results\S3_diffusion_training\20261103_042234_s3_train_4b11c4\checkpoints\step_020000.pt` | 20000 | None | 16 | 96 | 40000 |
| `results\S3_diffusion_training\20261103_042234_s3_train_4b11c4\checkpoints\step_022000.pt` | 22000 | None | 16 | 96 | 40000 |
| `results\S3_diffusion_training\20261103_042234_s3_train_4b11c4\checkpoints\step_024000.pt` | 24000 | None | 16 | 96 | 40000 |
| `results\S3_diffusion_training\20261103_042234_s3_train_4b11c4\checkpoints\step_026000.pt` | 26000 | None | 16 | 96 | 40000 |
| `results\S3_diffusion_training\20261103_042234_s3_train_4b11c4\checkpoints\step_028000.pt` | 28000 | None | 16 | 96 | 40000 |
| `results\S3_diffusion_training\20261103_042234_s3_train_4b11c4\checkpoints\step_030000.pt` | 30000 | None | 16 | 96 | 40000 |
| `results\S3_diffusion_training\20261103_042234_s3_train_4b11c4\checkpoints\step_032000.pt` | 32000 | None | 16 | 96 | 40000 |
| `results\S3_diffusion_training\20261103_042234_s3_train_4b11c4\checkpoints\step_034000.pt` | 34000 | None | 16 | 96 | 40000 |
| `results\S3_diffusion_training\20261103_042234_s3_train_4b11c4\checkpoints\step_036000.pt` | 36000 | None | 16 | 96 | 40000 |
| `results\S3_diffusion_training\20261103_042234_s3_train_4b11c4\checkpoints\step_038000.pt` | 38000 | None | 16 | 96 | 40000 |
| `results\S3_diffusion_training\20261103_042234_s3_train_4b11c4\checkpoints\step_040000.pt` | 40000 | None | 16 | 96 | 40000 |
| `results\S3_diffusion_training\20261103_042234_s3_train_4b11c4\checkpoints\step_042000.pt` | 42000 | None | 16 | 96 | 40000 |
| `results\S3_diffusion_training\20261103_042234_s3_train_4b11c4\checkpoints\step_044000.pt` | 44000 | None | 16 | 96 | 40000 |
| `results\S3_diffusion_training\20261103_042234_s3_train_4b11c4\checkpoints\step_046000.pt` | 46000 | None | 16 | 96 | 40000 |
| `results\S3_diffusion_training\20261103_042234_s3_train_4b11c4\checkpoints\step_048000.pt` | 48000 | None | 16 | 96 | 40000 |
| `results\S3_diffusion_training\20261103_042234_s3_train_4b11c4\checkpoints\step_050000.pt` | 50000 | None | 16 | 96 | 40000 |
| `results\S3_diffusion_training\20261103_123147_s3_train_7e8f85\checkpoints\best.pt` | 6500 | index | 16 | 96 | 8000 |
| `results\S3_diffusion_training\20261103_123147_s3_train_7e8f85\checkpoints\latest.pt` | 8000 | index | 16 | 96 | 8000 |
| `results\S3_diffusion_training\20261103_123147_s3_train_7e8f85\checkpoints\step_002000.pt` | 2000 | index | 16 | 96 | 8000 |
| `results\S3_diffusion_training\20261103_123147_s3_train_7e8f85\checkpoints\step_004000.pt` | 4000 | index | 16 | 96 | 8000 |
| `results\S3_diffusion_training\20261103_123147_s3_train_7e8f85\checkpoints\step_006000.pt` | 6000 | index | 16 | 96 | 8000 |
| `results\S3_diffusion_training\20261103_123147_s3_train_7e8f85\checkpoints\step_008000.pt` | 8000 | index | 16 | 96 | 8000 |
| `results\S3_diffusion_training\20261103_123435_s3_train_8860ac\checkpoints\best.pt` | 6500 | grid_sample | 16 | 96 | 8000 |
| `results\S3_diffusion_training\20261103_123435_s3_train_8860ac\checkpoints\latest.pt` | 8000 | grid_sample | 16 | 96 | 8000 |
| `results\S3_diffusion_training\20261103_123435_s3_train_8860ac\checkpoints\step_002000.pt` | 2000 | grid_sample | 16 | 96 | 8000 |
| `results\S3_diffusion_training\20261103_123435_s3_train_8860ac\checkpoints\step_004000.pt` | 4000 | grid_sample | 16 | 96 | 8000 |
| `results\S3_diffusion_training\20261103_123435_s3_train_8860ac\checkpoints\step_006000.pt` | 6000 | grid_sample | 16 | 96 | 8000 |
| `results\S3_diffusion_training\20261103_123435_s3_train_8860ac\checkpoints\step_008000.pt` | 8000 | grid_sample | 16 | 96 | 8000 |
| `results\S3_diffusion_training\20261103_123911_s3_train_9925d5\checkpoints\best.pt` | 10500 | index | 16 | 96 | 8000 |
| `results\S3_diffusion_training\20261103_123911_s3_train_9925d5\checkpoints\latest.pt` | 30000 | index | 16 | 96 | 8000 |
| `results\S3_diffusion_training\20261103_123911_s3_train_9925d5\checkpoints\step_002000.pt` | 2000 | index | 16 | 96 | 8000 |
| `results\S3_diffusion_training\20261103_123911_s3_train_9925d5\checkpoints\step_004000.pt` | 4000 | index | 16 | 96 | 8000 |
| `results\S3_diffusion_training\20261103_123911_s3_train_9925d5\checkpoints\step_006000.pt` | 6000 | index | 16 | 96 | 8000 |
| `results\S3_diffusion_training\20261103_123911_s3_train_9925d5\checkpoints\step_008000.pt` | 8000 | index | 16 | 96 | 8000 |
| `results\S3_diffusion_training\20261103_123911_s3_train_9925d5\checkpoints\step_010000.pt` | 10000 | index | 16 | 96 | 8000 |
| `results\S3_diffusion_training\20261103_123911_s3_train_9925d5\checkpoints\step_012000.pt` | 12000 | index | 16 | 96 | 8000 |
| `results\S3_diffusion_training\20261103_123911_s3_train_9925d5\checkpoints\step_014000.pt` | 14000 | index | 16 | 96 | 8000 |
| `results\S3_diffusion_training\20261103_123911_s3_train_9925d5\checkpoints\step_016000.pt` | 16000 | index | 16 | 96 | 8000 |
| `results\S3_diffusion_training\20261103_123911_s3_train_9925d5\checkpoints\step_018000.pt` | 18000 | index | 16 | 96 | 8000 |
| `results\S3_diffusion_training\20261103_123911_s3_train_9925d5\checkpoints\step_020000.pt` | 20000 | index | 16 | 96 | 8000 |
| `results\S3_diffusion_training\20261103_123911_s3_train_9925d5\checkpoints\step_022000.pt` | 22000 | index | 16 | 96 | 8000 |
| `results\S3_diffusion_training\20261103_123911_s3_train_9925d5\checkpoints\step_024000.pt` | 24000 | index | 16 | 96 | 8000 |
| `results\S3_diffusion_training\20261103_123911_s3_train_9925d5\checkpoints\step_026000.pt` | 26000 | index | 16 | 96 | 8000 |
| `results\S3_diffusion_training\20261103_123911_s3_train_9925d5\checkpoints\step_028000.pt` | 28000 | index | 16 | 96 | 8000 |
| `results\S3_diffusion_training\20261103_123911_s3_train_9925d5\checkpoints\step_030000.pt` | 30000 | index | 16 | 96 | 8000 |
| `results\S3_diffusion_training\20261103_124832_s3_train_d6b02d\checkpoints\best.pt` | 26500 | grid_sample | 16 | 96 | 8000 |
| `results\S3_diffusion_training\20261103_124832_s3_train_d6b02d\checkpoints\latest.pt` | 30000 | grid_sample | 16 | 96 | 8000 |
| `results\S3_diffusion_training\20261103_124832_s3_train_d6b02d\checkpoints\step_002000.pt` | 2000 | grid_sample | 16 | 96 | 8000 |
| `results\S3_diffusion_training\20261103_124832_s3_train_d6b02d\checkpoints\step_004000.pt` | 4000 | grid_sample | 16 | 96 | 8000 |
| `results\S3_diffusion_training\20261103_124832_s3_train_d6b02d\checkpoints\step_006000.pt` | 6000 | grid_sample | 16 | 96 | 8000 |
| `results\S3_diffusion_training\20261103_124832_s3_train_d6b02d\checkpoints\step_008000.pt` | 8000 | grid_sample | 16 | 96 | 8000 |
| `results\S3_diffusion_training\20261103_124832_s3_train_d6b02d\checkpoints\step_010000.pt` | 10000 | grid_sample | 16 | 96 | 8000 |
| `results\S3_diffusion_training\20261103_124832_s3_train_d6b02d\checkpoints\step_012000.pt` | 12000 | grid_sample | 16 | 96 | 8000 |
| `results\S3_diffusion_training\20261103_124832_s3_train_d6b02d\checkpoints\step_014000.pt` | 14000 | grid_sample | 16 | 96 | 8000 |
| `results\S3_diffusion_training\20261103_124832_s3_train_d6b02d\checkpoints\step_016000.pt` | 16000 | grid_sample | 16 | 96 | 8000 |
| `results\S3_diffusion_training\20261103_124832_s3_train_d6b02d\checkpoints\step_018000.pt` | 18000 | grid_sample | 16 | 96 | 8000 |
| `results\S3_diffusion_training\20261103_124832_s3_train_d6b02d\checkpoints\step_020000.pt` | 20000 | grid_sample | 16 | 96 | 8000 |
| `results\S3_diffusion_training\20261103_124832_s3_train_d6b02d\checkpoints\step_022000.pt` | 22000 | grid_sample | 16 | 96 | 8000 |
| `results\S3_diffusion_training\20261103_124832_s3_train_d6b02d\checkpoints\step_024000.pt` | 24000 | grid_sample | 16 | 96 | 8000 |
| `results\S3_diffusion_training\20261103_124832_s3_train_d6b02d\checkpoints\step_026000.pt` | 26000 | grid_sample | 16 | 96 | 8000 |
| `results\S3_diffusion_training\20261103_124832_s3_train_d6b02d\checkpoints\step_028000.pt` | 28000 | grid_sample | 16 | 96 | 8000 |
| `results\S3_diffusion_training\20261103_124832_s3_train_d6b02d\checkpoints\step_030000.pt` | 30000 | grid_sample | 16 | 96 | 8000 |

## 配置解析差异（extends 展开后）

| 配置文件 | 解析后关键项 |
|---|---|
| `audit_arch_grid.yaml` | map_conditioning=grid_sample, map_res=16, training_steps=8000, batch=96, grid=64, benchmark.methods=- |
| `audit_arch_index.yaml` | map_conditioning=index, map_res=16, training_steps=8000, batch=96, grid=64, benchmark.methods=- |
| `base.yaml` | map_conditioning=-, map_res=-, training_steps=40000, batch=96, grid=64, benchmark.methods=- |
| `benchmark.yaml` | map_conditioning=-, map_res=16, training_steps=40000, batch=96, grid=64, benchmark.methods=7 |
| `diffusion_train.yaml` | map_conditioning=-, map_res=16, training_steps=40000, batch=96, grid=64, benchmark.methods=- |
| `guidance.yaml` | map_conditioning=-, map_res=16, training_steps=40000, batch=96, grid=64, benchmark.methods=7 |
| `risk_only.yaml` | map_conditioning=-, map_res=-, training_steps=40000, batch=96, grid=64, benchmark.methods=- |

> 注：`benchmark.methods` 这类项由 YAML 继承得到，实际运行时可能被 CLI 覆盖；本表只能证明「YAML 解析后」的值，CLI 覆盖需在运行目录的 manifest 里记录（本轮已在新脚本中强制写入）。
