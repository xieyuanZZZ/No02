# 01 开源基线官方复现报告

**日期**：2026-11-06  
**范围**：只做官方示例/官方数据原样复现；不做“按论文重写的简化版”。  
**第三方位置**：`D:\No02_external_repos\`（仓库外，不提交进 No02）。

---

## 1. 版本、许可证、环境

| 方法 | 上游 | 固定 Commit | 许可证 | 官方示例状态 |
|---|---|---|---|---|
| Switchable-Edge Search | `YinggggFeng/Switchable-Edge-Search` | `c60c6b52b9563655b0241b473ba146cbcf25b4b2` | **无 LICENSE/COPYING 文件**；代码不随 No02 分发 | 成功 |
| WinkTPG | `JingtianYan/WinkTPG-IEEE-TASE` | `dd96946f225e206de35b041320786eab43210658` | **无 LICENSE/COPYING 文件**；代码不随 No02 分发 | 成功 |
| STPG | `DiligentPanda/STPG` | `25fb931eff03f1cce23a22a68ab42b7533f85ab3` | MIT | 成功（官方 benchmark 实例） |
| STPG 子模块 PBS | `DiligentPanda/PBS` | `f1e08e104c0cb575d6f9ed8b26344c0b982f422d` | 随上游 | 已初始化 |
| STPG 子模块 pybind11 | `pybind/pybind11` | `19a6b9f4efb569129c878b7f8db09132248fbaa1` | 随上游 | 已初始化 |

**环境**：

```text
Windows 11 + WSL2 Ubuntu 22.04
g++ 11.4.0
cmake 3.22.1
Boost 1.74
SFML 2.5.1
Python 3.10.12 + python3-dev
```

**安装记录**：WSL 的 apt 需要经宿主代理；实际使用
`apt-get -o Acquire::http::Proxy=http://127.0.0.1:7897` 安装
`cmake libboost-system-dev libboost-program-options-dev libboost-filesystem-dev libsfml-dev python3-dev`。
STPG 的 `.gitmodules` 使用 SSH URL，本环境改为 HTTPS 后初始化；未改写上游历史。

---

## 2. Switchable-Edge Search

**构建**（上游 README 的 g++ 命令；本环境不依赖 CMake）：

```bash
g++ -std=c++17 -O2 \
  src/graph/graph.cpp src/ADG/generate_ADG.cpp src/ADG/ADG_utilities.cpp \
  src/Algorithm/Astar.cpp src/Algorithm/simulator.cpp src/simulate.cpp -o a.out
```

**官方示例**：

```bash
./a.out example/path.txt 10 10 20 \
  stats_graph.csv stats_exec.csv locations.txt delay_setup.txt
```

**实际输出**：

```text
stats_graph.csv:
22948,24734,999,895,959,855,146,85,205,922,413,4327,350,5444,1,11211,0,228

stats_exec.csv:
4584319,4586105,999,895,959,855,26337,17751,34921,922,413,1660898,77505,0,17523,2552232,0,58603
```

原始输出已保存到
`results/External_Baseline_Comparison/native_reproduction/switchable_edge/`
（`stats_*.csv`、`locations.txt`、`delay_setup.txt`、`out.txt`、`metadata.json`）。

**注意**：上游参数 `p=10` 是 `[0,1000]` 的整数概率（约 1%），delay 长度是离散 step；
一次运行在第一次 delay 后即退出并输出统计，不提供逐机器人执行轨迹。

---

## 3. WinkTPG

**构建**：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
```

**官方示例**：

```bash
./build/tpg -p example_path.txt -m 1 -o output.csv
./build/tpg -p example_path.txt -m 0 -o output_tpg.csv
```

**实际输出**：

```text
WinkTPG (-m 1):
success,runtime,solution cost,min-solution cost,#type-2 edges,...
1,0.004529,1090.99,1005.06,462,1100,1140,423,7,0.002982,0.000177,7.8e-05,20,10,0,0.01,0.05,speed_solver,example_path.txt,40

TPG (-m 0):
1,0.001695,1223.75,1005.06,462,1100,1140,183,0,0.000523,0,6.7e-05,20,10,0,0.01,0.05,speed_solver,example_path.txt,40
```

原始输出、CMake/编译日志已保存到
`results/External_Baseline_Comparison/native_reproduction/winktpg/`。

---

## 4. STPG

**构建**：

```bash
git submodule sync && git submodule update --init --recursive
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
```

**官方 benchmark 实例**（不是简化示例）：

```text
path: data/benchmark/test_PBS2_delay_p01/path/map_lak303d_ins_10_an_41.path
sit:  data/benchmark/test_PBS2_delay_p01/sit/map_lak303d_ins_10_an_41_sit_0.json
```

**运行**：

```bash
./build/simulate -p <path> -s <sit> -t 30 -a search \
  -o stat.csv -n newpath.txt -b default -g all -h wcg_greedy -e true -i true
```

**实际输出**：

```text
original cost: 11902
optimized cost: 11688
search time: 0.519586 s
total time: 0.527858 s
expanded node count: 288
added node count: 565
```

`stat.csv` 实际内容是 JSON（上游扩展名如此），已原样保存到
`results/External_Baseline_Comparison/native_reproduction/stpg/`。

---

## 5. 官方复现结论与适配阻塞

**结论**：三种方法都在官方输入/示例上真实编译、运行并产生了非空输出；
没有用“按论文重写的简化版”替代。

**但本轮不能直接进入统一 MPC 对照**，原因不是环境，而是输入/输出语义：

### Switchable-Edge Search

- 输入是**离散网格状态序列**（每个状态一个整数时间），不是连续 32 航点；
- 官方程序输出的是聚合统计和每 step 的计数，**没有导出每台机器人的让行/切换动作**；
- 它的时间单位是离散 step，delay 也是整数 step；No02 是 0.4 s 的连续控制周期、
  随机暂停发生在任意 slot，且 MPC 有制动距离；
- 若直接把连续路径四舍五入到网格，会改变冲突拓扑和切换语义；若只取统计，
  又无法在 No02 的 MPC 里执行。因此不能把本次官方统计当成“统一 MPC 对照结果”。

### WinkTPG

- 输入是**带朝向的网格路径**，内部求解运动学可行速度剖面；
- 官方输出只有聚合 CSV，**没有逐机器人 action/speed profile 导出**；
- 其原生执行器与 No02 的平面双积分 MPC 不同；要公平比较必须保留原生结果，
  另加非侵入式导出 hook，再把速度/动作映射到 32 航点，不能只拿 `solution cost` 比较。

### STPG

- 输入是离散有向网格路径 + situation JSON；No02 没有对应的 situation 生成器；
- 其时间/延迟模型是步数，不是连续 0.4 s 暂停；
- 需要先做离散化映射并证明切换边语义不被破坏，才能进入统一执行器。

**因此阶段 1 的交付是“官方复现成功 + 适配阻塞已定位”，不是“外部基线已经接入 No02”。**
在没有完成语义保持的动作导出/映射前，不产生统一 MPC 对比数字，也不把原生统计
与 No02 的碰撞率混在一起。
