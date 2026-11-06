# 01 四个连续空间基线：官方复现报告

**日期**：2026-11-06  
**分支**：`experiment/continuous-baselines-v1`  
**第三方位置**：`D:\No02_external_repos\`（仓库外，不提交进 No02）  
**结果**：`results/Continuous_Baseline_Comparison/native_reproduction/`

---

## 1. 版本、许可证、状态

| 方法 | 上游 Commit | 许可证 | 官方复现 | 备注 |
|---|---|---|---|---|
| Coordination ORU | `7332cdeee220129c44aa3318e4812966fc85f261` | GPL-3.0 | **成功（构建兼容补丁）** | Gradle/jcenter 依赖失效，移除 RViz 可视化依赖后运行官方三机器人示例 |
| ORCA / RVO2 | `6b4eb00a93a9f3abf723235ff006d6aad3e2f212` | Apache-2.0 | **成功** | 官方 `Circle` 示例 |
| MPC-CBF | `62bdf68c0fc35254677e595f34fcf53c82872aca` | MIT | **成功（兼容补丁）** | 2 机器人差速模型；官方 intersection 场景 |
| MMD | `abe00192a8be1b0222a4eab1ac59f60e7e648df7` | MIT | **成功** | 本地 `/home/xuan/delay_robust_mmd_guidance` 的官方 `mmd/`，3 机器人 Empty 示例 |

环境：WSL2 Ubuntu 22.04；g++ 11.4；OpenJDK 11；Gradle 6.4.1；CMake 3.31.6；
Python 3.12（MPC-CBF venv）/ conda `mmd`（MMD）；CUDA RTX 4070。

---

## 2. Coordination ORU

**问题**：Gradle 6.4.1 依赖的 `jcenter` 已关闭，`org.ros.rosjava_bootstrap` 传递依赖
`org.apache.commons:com.springsource.*` 在 Maven Central/JitPack/rosjava repo 都不存在，
原始 `./gradlew build` 失败（`co_build.log`）。

**兼容补丁（只改构建/非核心）**：

- 移除 RViz/rosjava 可视化依赖；
- 加 `commons-lang:commons-lang:2.6`（核心 `TrajectoryEnvelopeCoordinator` 使用
  `ArrayUtils`，原先由 rosjava 传递依赖提供）；
- 临时只保留官方三机器人示例的 test 源文件，避免其它 RViz 示例编译失败；
- headless AWT + 安装 `libawt_xawt` 运行。

**官方示例**：

```bash
./gradlew_lf run -Pdemo=TestTrajectoryEnvelopeCoordinatorThreeRobots
```

**实际结果**：1,776 行日志，三机器人 placed、missions dispatched、critical sections/
dependencies 更新、tracking 与 reparking 均发生；无 Java 异常。官方示例本身是三个
`while(true)` mission-dispatch 线程，120 s 超时是主动停止，不是崩溃。

输出：`co_build4.log`、`co_run2.log`。

---

## 3. ORCA / RVO2

**构建**：RVO2 要求 CMake ≥3.26；系统 3.22 不够，下载 CMake 3.31.6 后：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
```

**官方示例**：

```bash
./build/examples/Circle
```

**实际结果**：`rvo_circle.out` 共 3,547 行 time + 每 agent 位置，动态 agent 围绕圆周；
无崩溃。原始 RVO2 使用自己的 preferred velocity/agent radius，尚未接入 No02。

---

## 4. MPC-CBF

**环境**：独立 venv（不污染 No02）；`numpy<2`、`scipy==1.13.1`、`pandas<3`，
因为上游 `np.cross` 不支持 NumPy 2D 向量、SciPy 1.18 要求 NumPy 2。

**兼容补丁**：只改 `main.py` 的 Matplotlib 动画行（`set_data([x],[y])`），
控制器/模型/约束未改。

**官方场景**：`main.py`，`scenario=intersection`，2 个差速驱动机器人，IPOPT 求解。

**实际结果**：完整运行，`EXIT: Optimal Solution Found`，生成 `agents_animation.mp4`；
`official_stdout.txt` 保存全部求解日志。

**限制**：这是 2 机器人差速 MPC-CBF，不是 4 机器人双积分器实现；不能直接套在
No02 MPC 外面，也不能声称“同一控制器”。若进入统一对比，需要保留其 CBF 约束结构，
但替换为统一双积分动力学并重新验证，不能静默改核心。

---

## 5. MMD

**环境**：conda `mmd`，需 `LD_LIBRARY_PATH=.../nvrtc_fix`（上游 CUDA 依赖）。  
**官方示例**：

```bash
cd mmd/scripts/inference
python inference_multi_agent.py
```

**实际结果**：

- 3 agents，`EnvEmptyNoWait2D`，XECBS + MPDEnsemble；
- `success=True`，`num_collisions_in_solution=0`；
- planning time **9.61 s**，data adherence **0.901**；
- 生成 GIF/PNG/results.pkl。

**限制**：这是 MMD 自己的 3 机器人环境；实验 B 要比较相同地图/起终点，不能直接
复用该示例。MMD 作为完整规划系统，必须与固定路径在线协调分开统计。

---

## 6. 结论

- 四个基线都拿到了**真实官方运行结果**，没有用简化重写冒充；
- Coordination ORU 和 MPC-CBF 的上游依赖/模型与本机环境存在实际兼容问题，补丁已记录并限定在构建/可视化；
- RVO2/MPC-CBF 的原生控制器尚未接入 No02；MMD 尚未在 No02 地图上生成路径；
- 因此本轮**没有伪造统一实验数字**，下一步必须先做接口适配（S2）和小规模验证（S3）。
