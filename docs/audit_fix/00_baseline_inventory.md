# 00 基线清单（修复前冻结）

冻结时间：2026-11-03T12:15:48+08:00

## 1. 代码状态

- 分支：`fix/audit-foundation`
- HEAD：`5abfeb84c8459b11add7ee57ed5ffb902c15d965`
- 工作区有未提交改动：**True**
- 已有 tag：（无）
- 远程：origin	https://github.com/xieyuanZZZ/No02.git (fetch); origin	https://github.com/xieyuanZZZ/No02.git (push)

## 2. 运行环境

| 项 | 值 |
|---|---|
| Python | 3.12.13 |
| PyTorch | 2.11.0+cu128 |
| CUDA | 12.8 |
| GPU | NVIDIA GeForce RTX 4070 |
| 平台 | Windows-11-10.0.26200-SP0 |

## 3. 修复前测试状态

- 通过 **69**、失败 0、错误 0、跳过 0
- 退出码：0
- 明细：`tables/pytest_baseline.txt`

## 4. 实验运行记录（不可覆盖）

共 **37** 次运行。

| 阶段 | run_id | 状态 | 文件数 | 体积 MB |
|---|---|---|---:|---:|
| S0_environment | `20261103_015710_s0_baseline_f24a86` | failed | 26 | 0.53 |
| S0_environment | `20261103_020239_s0_baseline_96f8bf` | unknown | 10 | 0.12 |
| S0_environment | `20261103_020245_s0_baseline_fa5a14` | unknown | 21 | 0.38 |
| S0_environment | `20261103_020310_s0_baseline_7ff859` | passed | 30 | 0.54 |
| S1_risk_dp | `20261103_020414_s1_verify_58057e` | unknown | 15 | 0.25 |
| S1_risk_dp | `20261103_020438_s1_verify_ffec72` | passed | 18 | 0.26 |
| S1_risk_dp | `20261103_020513_s1_verify_f1e3d9` | passed | 17 | 0.26 |
| S2_gradient_optimization | `20261103_020636_s2_optimize_84e8c4` | failed | 28 | 0.61 |
| S2_gradient_optimization | `20261103_020832_s2_optimize_33ab38` | passed | 28 | 0.65 |
| S2_gradient_optimization | `20261103_021022_s2_optimize_a5df8d` | passed | 30 | 0.66 |
| S2_gradient_optimization | `20261103_021319_s2_optimize_b67a27` | passed | 30 | 0.66 |
| S3_diffusion_training | `20261103_021658_S3_dataset_small_392c14` | completed | 6 | 0.01 |
| S3_diffusion_training | `20261103_021831_S3_dataset_medium_ef0564` | completed | 6 | 0.01 |
| S3_diffusion_training | `20261103_023715_s3_train_59ba46` | completed | 16 | 136.97 |
| S3_diffusion_training | `20261103_023738_s3_train_a1df62` | completed | 72 | 754.88 |
| S3_diffusion_training | `20261103_024957_s3_sample_a102bb` | completed | 9 | 0.57 |
| S3_diffusion_training | `20261103_025111_s3_sample_843cdc` | completed | 9 | 0.26 |
| S3_diffusion_training | `20261103_025204_S3_dataset_large_dec1b3` | completed | 6 | 0.01 |
| S3_diffusion_training | `20261103_025625_s3_train_b9efe1` | completed | 24 | 204.32 |
| S3_diffusion_training | `20261103_025919_s3_sample_452838` | completed | 9 | 0.32 |
| S3_diffusion_training | `20261103_030603_s3_train_383287` | completed | 72 | 749.93 |
| S3_diffusion_training | `20261103_032008_s3_sample_0b5ace` | completed | 9 | 0.36 |
| S3_diffusion_training | `20261103_040239_s3_train_40fcfd` | completed | 87 | 920.43 |
| S3_diffusion_training | `20261103_041756_s3_sample_61b8e3` | completed | 9 | 0.57 |
| S3_diffusion_training | `20261103_041902_s3_sample_7bb2d5` | completed | 9 | 0.37 |
| S3_diffusion_training | `20261103_042118_s3_sample_bed93d` | completed | 9 | 0.36 |
| S3_diffusion_training | `20261103_042234_s3_train_4b11c4` | completed | 87 | 920.43 |
| S3_diffusion_training | `20261103_043800_s3_sample_402b45` | completed | 9 | 0.37 |
| S4_guided_diffusion | `20261103_025209_s4_demo_680089` | unknown | 2 | 0.0 |
| S4_guided_diffusion | `20261103_025215_s4_demo_d90c77` | passed | 11 | 0.19 |
| S4_guided_diffusion | `20261103_025347_s4_demo_9056a1` | passed | 11 | 0.2 |
| S4_guided_diffusion | `20261103_032125_s4_demo_2fb7e4` | passed | 11 | 0.21 |
| S5_evaluation | `20261103_025411_s5_benchmark_a349cc` | unknown | 2 | 0.0 |
| S5_evaluation | `20261103_030006_s5_benchmark_afed08` | completed | 14 | 0.15 |
| S5_evaluation | `20261103_032138_s5_benchmark_984312` | completed | 14 | 0.52 |
| S5_evaluation | `20261103_033237_s5_benchmark_8cf9f0` | unknown | 2 | 0.0 |
| S5_evaluation | `20261103_033410_s5_benchmark_a98f98` | completed | 14 | 0.52 |

## 5. 模型检查点（本机，未入库）

| 路径 | 步数 | best_val | MB | SHA256（前 16） |
|---|---:|---:|---:|---|
| `results\S3_diffusion_training\20261103_023715_s3_train_59ba46\checkpoints\best.pt` | 500 | 0.02656 | 34.14 | `c1b5729d0171ceaa` |
| `results\S3_diffusion_training\20261103_023715_s3_train_59ba46\checkpoints\latest.pt` | 600 | 0.02656 | 34.16 | `b3a699ab0d827f06` |
| `results\S3_diffusion_training\20261103_023715_s3_train_59ba46\checkpoints\step_000300.pt` | 300 | 0.03909 | 34.17 | `1fd0fcecadd75117` |
| `results\S3_diffusion_training\20261103_023715_s3_train_59ba46\checkpoints\step_000600.pt` | 600 | 0.02656 | 34.17 | `21bc9f3ab240c456` |
| `results\S3_diffusion_training\20261103_023738_s3_train_a1df62\checkpoints\best.pt` | 26500 | 0.01267 | 34.14 | `35af9a6089d77199` |
| `results\S3_diffusion_training\20261103_023738_s3_train_a1df62\checkpoints\latest.pt` | 40000 | 0.01267 | 34.16 | `9a4aa01394687b21` |
| `results\S3_diffusion_training\20261103_023738_s3_train_a1df62\checkpoints\step_002000.pt` | 2000 | 0.01661 | 34.17 | `595ebda825cfa8b4` |
| `results\S3_diffusion_training\20261103_023738_s3_train_a1df62\checkpoints\step_004000.pt` | 4000 | 0.01452 | 34.17 | `58aea7cba41fd5be` |
| `results\S3_diffusion_training\20261103_023738_s3_train_a1df62\checkpoints\step_006000.pt` | 6000 | 0.01273 | 34.17 | `b8c8bcfba83222d8` |
| `results\S3_diffusion_training\20261103_023738_s3_train_a1df62\checkpoints\step_008000.pt` | 8000 | 0.01273 | 34.17 | `fd75e59c230735ec` |
| `results\S3_diffusion_training\20261103_023738_s3_train_a1df62\checkpoints\step_010000.pt` | 10000 | 0.01273 | 34.17 | `d03f9ae422a0e01f` |
| `results\S3_diffusion_training\20261103_023738_s3_train_a1df62\checkpoints\step_012000.pt` | 12000 | 0.01273 | 34.17 | `8671e62b53d08312` |
| `results\S3_diffusion_training\20261103_023738_s3_train_a1df62\checkpoints\step_014000.pt` | 14000 | 0.01273 | 34.17 | `8ce9e7b44ff558fa` |
| `results\S3_diffusion_training\20261103_023738_s3_train_a1df62\checkpoints\step_016000.pt` | 16000 | 0.01273 | 34.17 | `10e55a3682ae9fac` |
| `results\S3_diffusion_training\20261103_023738_s3_train_a1df62\checkpoints\step_018000.pt` | 18000 | 0.01273 | 34.17 | `980adad80cc60356` |
| `results\S3_diffusion_training\20261103_023738_s3_train_a1df62\checkpoints\step_020000.pt` | 20000 | 0.01273 | 34.17 | `c2ac1f0e2476c31c` |
| `results\S3_diffusion_training\20261103_023738_s3_train_a1df62\checkpoints\step_022000.pt` | 22000 | 0.01273 | 34.17 | `6a19c628e0c4e227` |
| `results\S3_diffusion_training\20261103_023738_s3_train_a1df62\checkpoints\step_024000.pt` | 24000 | 0.01273 | 34.17 | `e509a31e3f377e8d` |
| `results\S3_diffusion_training\20261103_023738_s3_train_a1df62\checkpoints\step_026000.pt` | 26000 | 0.01273 | 34.17 | `a8a640fcfde6d597` |
| `results\S3_diffusion_training\20261103_023738_s3_train_a1df62\checkpoints\step_028000.pt` | 28000 | 0.01267 | 34.17 | `221233f5ad0d86ee` |
| `results\S3_diffusion_training\20261103_023738_s3_train_a1df62\checkpoints\step_030000.pt` | 30000 | 0.01267 | 34.17 | `e62846a659ec32cb` |
| `results\S3_diffusion_training\20261103_023738_s3_train_a1df62\checkpoints\step_032000.pt` | 32000 | 0.01267 | 34.17 | `031edcdb22342d9b` |
| `results\S3_diffusion_training\20261103_023738_s3_train_a1df62\checkpoints\step_034000.pt` | 34000 | 0.01267 | 34.17 | `bea64a7d922ffc3a` |
| `results\S3_diffusion_training\20261103_023738_s3_train_a1df62\checkpoints\step_036000.pt` | 36000 | 0.01267 | 34.17 | `d78067d84cb8390e` |
| `results\S3_diffusion_training\20261103_023738_s3_train_a1df62\checkpoints\step_038000.pt` | 38000 | 0.01267 | 34.17 | `cba46eff857adf98` |
| `results\S3_diffusion_training\20261103_023738_s3_train_a1df62\checkpoints\step_040000.pt` | 40000 | 0.01267 | 34.17 | `a797ab2188d96417` |
| `results\S3_diffusion_training\20261103_025625_s3_train_b9efe1\checkpoints\best.pt` | 6500 | 0.01150 | 33.92 | `1ac8efbde8f5333d` |
| `results\S3_diffusion_training\20261103_025625_s3_train_b9efe1\checkpoints\latest.pt` | 8000 | 0.01150 | 33.93 | `7d011344351cbe42` |
| `results\S3_diffusion_training\20261103_025625_s3_train_b9efe1\checkpoints\step_002000.pt` | 2000 | 0.01960 | 33.94 | `5155efce33c14104` |
| `results\S3_diffusion_training\20261103_025625_s3_train_b9efe1\checkpoints\step_004000.pt` | 4000 | 0.01502 | 33.94 | `720b23b8c6acfe4e` |
| `results\S3_diffusion_training\20261103_025625_s3_train_b9efe1\checkpoints\step_006000.pt` | 6000 | 0.01315 | 33.94 | `ca3b7fd81f14d024` |
| `results\S3_diffusion_training\20261103_025625_s3_train_b9efe1\checkpoints\step_008000.pt` | 8000 | 0.01150 | 33.94 | `17e76fa0de57913e` |
| `results\S3_diffusion_training\20261103_030603_s3_train_383287\checkpoints\best.pt` | 22500 | 0.01066 | 33.92 | `fec307d22d54b020` |
| `results\S3_diffusion_training\20261103_030603_s3_train_383287\checkpoints\latest.pt` | 40000 | 0.01066 | 33.93 | `03d9b144dfb4f2fe` |
| `results\S3_diffusion_training\20261103_030603_s3_train_383287\checkpoints\step_002000.pt` | 2000 | 0.01951 | 33.94 | `204b75eaae201756` |
| `results\S3_diffusion_training\20261103_030603_s3_train_383287\checkpoints\step_004000.pt` | 4000 | 0.01495 | 33.94 | `a790c799a41f3096` |
| `results\S3_diffusion_training\20261103_030603_s3_train_383287\checkpoints\step_006000.pt` | 6000 | 0.01316 | 33.94 | `04762102262e865d` |
| `results\S3_diffusion_training\20261103_030603_s3_train_383287\checkpoints\step_008000.pt` | 8000 | 0.01149 | 33.94 | `b434867f86020345` |
| `results\S3_diffusion_training\20261103_030603_s3_train_383287\checkpoints\step_010000.pt` | 10000 | 0.01149 | 33.94 | `74dc432fa1660360` |
| `results\S3_diffusion_training\20261103_030603_s3_train_383287\checkpoints\step_012000.pt` | 12000 | 0.01084 | 33.94 | `1bf9cbfea846f570` |
| `results\S3_diffusion_training\20261103_030603_s3_train_383287\checkpoints\step_014000.pt` | 14000 | 0.01084 | 33.94 | `c19d882554b8d494` |
| `results\S3_diffusion_training\20261103_030603_s3_train_383287\checkpoints\step_016000.pt` | 16000 | 0.01084 | 33.94 | `f43469d9b235215a` |
| `results\S3_diffusion_training\20261103_030603_s3_train_383287\checkpoints\step_018000.pt` | 18000 | 0.01084 | 33.94 | `7dc48f39d2a9fe37` |
| `results\S3_diffusion_training\20261103_030603_s3_train_383287\checkpoints\step_020000.pt` | 20000 | 0.01084 | 33.94 | `b0bedd874b54c6bc` |
| `results\S3_diffusion_training\20261103_030603_s3_train_383287\checkpoints\step_022000.pt` | 22000 | 0.01084 | 33.94 | `7c5f1f451f8f74c1` |
| `results\S3_diffusion_training\20261103_030603_s3_train_383287\checkpoints\step_024000.pt` | 24000 | 0.01066 | 33.94 | `871114350c24f86c` |
| `results\S3_diffusion_training\20261103_030603_s3_train_383287\checkpoints\step_026000.pt` | 26000 | 0.01066 | 33.94 | `041e6b8fbf0bb61c` |
| `results\S3_diffusion_training\20261103_030603_s3_train_383287\checkpoints\step_028000.pt` | 28000 | 0.01066 | 33.94 | `580209fc8aaa032e` |
| `results\S3_diffusion_training\20261103_030603_s3_train_383287\checkpoints\step_030000.pt` | 30000 | 0.01066 | 33.94 | `75156cd346775f2b` |
| `results\S3_diffusion_training\20261103_030603_s3_train_383287\checkpoints\step_032000.pt` | 32000 | 0.01066 | 33.94 | `1af33220824cd7bd` |
| `results\S3_diffusion_training\20261103_030603_s3_train_383287\checkpoints\step_034000.pt` | 34000 | 0.01066 | 33.94 | `8c73cd405bd1c1c3` |
| `results\S3_diffusion_training\20261103_030603_s3_train_383287\checkpoints\step_036000.pt` | 36000 | 0.01066 | 33.94 | `10f10823aca4dcce` |
| `results\S3_diffusion_training\20261103_030603_s3_train_383287\checkpoints\step_038000.pt` | 38000 | 0.01066 | 33.94 | `c3290e2cdac2ce78` |
| `results\S3_diffusion_training\20261103_030603_s3_train_383287\checkpoints\step_040000.pt` | 40000 | 0.01066 | 33.94 | `20658102b3b77f07` |
| `results\S3_diffusion_training\20261103_040239_s3_train_40fcfd\checkpoints\best.pt` | 28000 | 0.00875 | 33.92 | `1635fe02907f0e56` |
| `results\S3_diffusion_training\20261103_040239_s3_train_40fcfd\checkpoints\latest.pt` | 50000 | 0.00875 | 33.93 | `f3ee23c874fb0f60` |
| `results\S3_diffusion_training\20261103_040239_s3_train_40fcfd\checkpoints\step_002000.pt` | 2000 | 0.01674 | 33.94 | `bb66a629419655b4` |
| `results\S3_diffusion_training\20261103_040239_s3_train_40fcfd\checkpoints\step_004000.pt` | 4000 | 0.01466 | 33.94 | `c2feb53d251096ca` |
| `results\S3_diffusion_training\20261103_040239_s3_train_40fcfd\checkpoints\step_006000.pt` | 6000 | 0.01336 | 33.94 | `44d45b3d01bb8302` |
| `results\S3_diffusion_training\20261103_040239_s3_train_40fcfd\checkpoints\step_008000.pt` | 8000 | 0.01292 | 33.94 | `c63b5fcf6f9a476a` |
| `results\S3_diffusion_training\20261103_040239_s3_train_40fcfd\checkpoints\step_010000.pt` | 10000 | 0.01115 | 33.94 | `0d920f1c3f705199` |
| `results\S3_diffusion_training\20261103_040239_s3_train_40fcfd\checkpoints\step_012000.pt` | 12000 | 0.01115 | 33.94 | `bc4595c236b912e0` |
| `results\S3_diffusion_training\20261103_040239_s3_train_40fcfd\checkpoints\step_014000.pt` | 14000 | 0.00999 | 33.94 | `6383a3a32d176613` |
| `results\S3_diffusion_training\20261103_040239_s3_train_40fcfd\checkpoints\step_016000.pt` | 16000 | 0.00942 | 33.94 | `108f288afd49f57d` |
| `results\S3_diffusion_training\20261103_040239_s3_train_40fcfd\checkpoints\step_018000.pt` | 18000 | 0.00942 | 33.94 | `1d87a90f8ffa7daf` |
| `results\S3_diffusion_training\20261103_040239_s3_train_40fcfd\checkpoints\step_020000.pt` | 20000 | 0.00942 | 33.94 | `e1b4adfee5d67df5` |
| `results\S3_diffusion_training\20261103_040239_s3_train_40fcfd\checkpoints\step_022000.pt` | 22000 | 0.00942 | 33.94 | `a9dc7618cd3b5cf4` |
| `results\S3_diffusion_training\20261103_040239_s3_train_40fcfd\checkpoints\step_024000.pt` | 24000 | 0.00942 | 33.94 | `70e2dd15da68f6d9` |
| `results\S3_diffusion_training\20261103_040239_s3_train_40fcfd\checkpoints\step_026000.pt` | 26000 | 0.00942 | 33.94 | `24e06b62a57692f8` |
| `results\S3_diffusion_training\20261103_040239_s3_train_40fcfd\checkpoints\step_028000.pt` | 28000 | 0.00875 | 33.94 | `198165c82b073714` |
| `results\S3_diffusion_training\20261103_040239_s3_train_40fcfd\checkpoints\step_030000.pt` | 30000 | 0.00875 | 33.94 | `63d28660268ec034` |
| `results\S3_diffusion_training\20261103_040239_s3_train_40fcfd\checkpoints\step_032000.pt` | 32000 | 0.00875 | 33.94 | `3cb6501b71e2ed21` |
| `results\S3_diffusion_training\20261103_040239_s3_train_40fcfd\checkpoints\step_034000.pt` | 34000 | 0.00875 | 33.94 | `a3639184cd30913b` |
| `results\S3_diffusion_training\20261103_040239_s3_train_40fcfd\checkpoints\step_036000.pt` | 36000 | 0.00875 | 33.94 | `6c509274b0487c17` |
| `results\S3_diffusion_training\20261103_040239_s3_train_40fcfd\checkpoints\step_038000.pt` | 38000 | 0.00875 | 33.94 | `172e8294fc9ed71b` |
| `results\S3_diffusion_training\20261103_040239_s3_train_40fcfd\checkpoints\step_040000.pt` | 40000 | 0.00875 | 33.94 | `6784e67c430821c3` |
| `results\S3_diffusion_training\20261103_040239_s3_train_40fcfd\checkpoints\step_042000.pt` | 42000 | 0.00875 | 33.94 | `dcebdd8bd82fe84e` |
| `results\S3_diffusion_training\20261103_040239_s3_train_40fcfd\checkpoints\step_044000.pt` | 44000 | 0.00875 | 33.94 | `1aa69958194f5bd0` |
| `results\S3_diffusion_training\20261103_040239_s3_train_40fcfd\checkpoints\step_046000.pt` | 46000 | 0.00875 | 33.94 | `686bc453fb992108` |
| `results\S3_diffusion_training\20261103_040239_s3_train_40fcfd\checkpoints\step_048000.pt` | 48000 | 0.00875 | 33.94 | `592ccf21473a2527` |
| `results\S3_diffusion_training\20261103_040239_s3_train_40fcfd\checkpoints\step_050000.pt` | 50000 | 0.00875 | 33.94 | `d76397b58bc88016` |
| `results\S3_diffusion_training\20261103_042234_s3_train_4b11c4\checkpoints\best.pt` | 37500 | 0.00763 | 33.92 | `d97c58f2c1143d2e` |
| `results\S3_diffusion_training\20261103_042234_s3_train_4b11c4\checkpoints\latest.pt` | 50000 | 0.00763 | 33.93 | `33911c60dfe5f759` |
| `results\S3_diffusion_training\20261103_042234_s3_train_4b11c4\checkpoints\step_002000.pt` | 2000 | 0.01670 | 33.94 | `c221128afb5e630a` |
| `results\S3_diffusion_training\20261103_042234_s3_train_4b11c4\checkpoints\step_004000.pt` | 4000 | 0.01448 | 33.94 | `042ce8d61835ae1b` |
| `results\S3_diffusion_training\20261103_042234_s3_train_4b11c4\checkpoints\step_006000.pt` | 6000 | 0.01331 | 33.94 | `7969bd9ec0a1e4aa` |
| `results\S3_diffusion_training\20261103_042234_s3_train_4b11c4\checkpoints\step_008000.pt` | 8000 | 0.01285 | 33.94 | `1a0cb6b3302d2210` |
| `results\S3_diffusion_training\20261103_042234_s3_train_4b11c4\checkpoints\step_010000.pt` | 10000 | 0.01107 | 33.94 | `d4e30847c68942dc` |
| `results\S3_diffusion_training\20261103_042234_s3_train_4b11c4\checkpoints\step_012000.pt` | 12000 | 0.01107 | 33.94 | `6a4cc0b7ccc5deee` |
| `results\S3_diffusion_training\20261103_042234_s3_train_4b11c4\checkpoints\step_014000.pt` | 14000 | 0.00991 | 33.94 | `67d1cc4dc794ce95` |
| `results\S3_diffusion_training\20261103_042234_s3_train_4b11c4\checkpoints\step_016000.pt` | 16000 | 0.00927 | 33.94 | `0dcf0a62bce8d4ff` |
| `results\S3_diffusion_training\20261103_042234_s3_train_4b11c4\checkpoints\step_018000.pt` | 18000 | 0.00927 | 33.94 | `73b65ad1065184a4` |
| `results\S3_diffusion_training\20261103_042234_s3_train_4b11c4\checkpoints\step_020000.pt` | 20000 | 0.00927 | 33.94 | `c1c2aa1cda3ed0a9` |
| `results\S3_diffusion_training\20261103_042234_s3_train_4b11c4\checkpoints\step_022000.pt` | 22000 | 0.00927 | 33.94 | `06b106ba8826d175` |
| `results\S3_diffusion_training\20261103_042234_s3_train_4b11c4\checkpoints\step_024000.pt` | 24000 | 0.00927 | 33.94 | `dca479e447fe8f87` |
| `results\S3_diffusion_training\20261103_042234_s3_train_4b11c4\checkpoints\step_026000.pt` | 26000 | 0.00927 | 33.94 | `64c62216f88944fa` |
| `results\S3_diffusion_training\20261103_042234_s3_train_4b11c4\checkpoints\step_028000.pt` | 28000 | 0.00814 | 33.94 | `d2fe1ba9f3f05152` |
| `results\S3_diffusion_training\20261103_042234_s3_train_4b11c4\checkpoints\step_030000.pt` | 30000 | 0.00814 | 33.94 | `9125899a4ea852cd` |
| `results\S3_diffusion_training\20261103_042234_s3_train_4b11c4\checkpoints\step_032000.pt` | 32000 | 0.00814 | 33.94 | `f4e06c7b5a2493ac` |
| `results\S3_diffusion_training\20261103_042234_s3_train_4b11c4\checkpoints\step_034000.pt` | 34000 | 0.00814 | 33.94 | `3892e7d52cda1ba3` |
| `results\S3_diffusion_training\20261103_042234_s3_train_4b11c4\checkpoints\step_036000.pt` | 36000 | 0.00814 | 33.94 | `20b13691e5a0dc56` |
| `results\S3_diffusion_training\20261103_042234_s3_train_4b11c4\checkpoints\step_038000.pt` | 38000 | 0.00763 | 33.94 | `8a6db696e075b333` |
| `results\S3_diffusion_training\20261103_042234_s3_train_4b11c4\checkpoints\step_040000.pt` | 40000 | 0.00763 | 33.94 | `6cdf05eef80f342d` |
| `results\S3_diffusion_training\20261103_042234_s3_train_4b11c4\checkpoints\step_042000.pt` | 42000 | 0.00763 | 33.94 | `f183c28c2da16282` |
| `results\S3_diffusion_training\20261103_042234_s3_train_4b11c4\checkpoints\step_044000.pt` | 44000 | 0.00763 | 33.94 | `6f38310cc6a9b24a` |
| `results\S3_diffusion_training\20261103_042234_s3_train_4b11c4\checkpoints\step_046000.pt` | 46000 | 0.00763 | 33.94 | `c21ebbe5ac2d7d77` |
| `results\S3_diffusion_training\20261103_042234_s3_train_4b11c4\checkpoints\step_048000.pt` | 48000 | 0.00763 | 33.94 | `08ce6f566d737949` |
| `results\S3_diffusion_training\20261103_042234_s3_train_4b11c4\checkpoints\step_050000.pt` | 50000 | 0.00763 | 33.94 | `6b8cf3f9af145d39` |

## 6. 数据与配置哈希

| 文件 | KB | SHA256（前 16） |
|---|---:|---|
| `configs\base.yaml` | 3.1 | `8e7664c5c5a1a1b6` |
| `configs\benchmark.yaml` | 0.5 | `3a8795bc073627de` |
| `configs\diffusion_train.yaml` | 1.1 | `38ca3fcd7a040084` |
| `configs\guidance.yaml` | 0.8 | `858675c40667c278` |
| `configs\risk_only.yaml` | 0.8 | `d586f930c9b9d594` |
| `data\dataset\maps_test.json` | 3.9 | `b4da9537ab4264c2` |
| `data\dataset\maps_train.json` | 20.3 | `0a6093997675283c` |
| `data\dataset\maps_val.json` | 3.9 | `bd63fe8643a80c75` |
| `data\dataset\meta.json` | 5.0 | `0c7daeb359cf754b` |
| `data\dataset\traj_test.npz` | 114.2 | `ea553d76fa231e30` |
| `data\dataset\traj_train.npz` | 1125.4 | `3231a7da5ae7e8ac` |
| `data\dataset\traj_val.npz` | 115.7 | `958af5fbf3bc8bed` |
| `data\dataset_large\maps_test.json` | 8.6 | `70f4b931044ec59e` |
| `data\dataset_large\maps_train.json` | 41.2 | `d4e058f18681e846` |
| `data\dataset_large\maps_val.json` | 8.5 | `8de7ef10138c4db7` |
| `data\dataset_large\meta.json` | 5.0 | `3f38f624a9267ec0` |
| `data\dataset_large\traj_test.npz` | 225.4 | `5950f00c360e98b5` |
| `data\dataset_large\traj_train.npz` | 4499.1 | `c89dc0505d7d09b2` |
| `data\dataset_large\traj_val.npz` | 230.9 | `ecf1bbb42d9aed37` |
| `pyproject.toml` | 0.8 | `129e6181e5def399` |
| `requirements.txt` | 0.5 | `abd3c664acb20da8` |
| `uv.lock` | 80.5 | `f22c874363e9c052` |

## 7. 关键基线数字（修复前，保留原值）

| 项 | 值 | 出处 |
|---|---|---|
| S1 DP vs MC 最大绝对误差 | **0.00673775** | `results/S1_risk_dp/*/metrics.csv` |
| S1 梯度有限差分最大相对误差 | 3.53e-7 | 同上 |
| S2 交叉场景硬风险 | 0.1788 → 2.15e-6（A3） | `results/S2_gradient_optimization/*/tables/pareto_points.csv` |
| S2 窄门口硬风险 | 1.0 → 1.0（饱和） | 同上 |
| S3 主模型单机成功率 | 59.4%（489 条测试集，200 任务×6） | run `032008_s3_sample_0b5ace`，ckpt `383287` |
| S3 归档报告成功率 | 60.75%（被 16×16 模型覆盖） | `docs/stages/S3/report.md`，ckpt `4b11c4` |
| S5 规划有效率 | 11/36（7 个方法完全相同） | `results/S5_evaluation/*/tables/plans.csv` |
| S5 p=0.1 安全完成率 | B0 18.24% / B2 19.44% / B3 25.65% / A2 22.31% / A3 22.59% | 同上 |
| S5 A3−B3 配对差（跨 4 个 p 汇总） | −0.0366 [−0.0963, −0.0002] | `aggregate.json` |

> 这些数字与目录一律作为 `baseline_before_fix` 保留，后续修复不得覆盖。
