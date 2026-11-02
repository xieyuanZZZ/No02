"""fhd：可微首次碰撞风险引导的二维多机器人扩散轨迹规划。

模块分层（文档 §11）：
    geometry   —— 基础几何、连续碰撞、SDF
    maps       —— 地图生成与场景
    data       —— A* 路径、数据集
    execution  —— 随机执行进度模型与 Monte Carlo 模拟器
    risk       —— 首次碰撞风险动态规划（不依赖 diffusion）
    diffusion  —— 条件扩散模型与 DDIM
    evaluation —— 基线、指标、统计、可视化
    utils      —— 配置、种子、日志、运行环境

依赖方向约束：risk 不得依赖 diffusion；只有 diffusion.guided_ddim 反向调用 risk。
"""

__version__ = "0.1.0"
