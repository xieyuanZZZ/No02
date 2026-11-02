"""risk 子包：首次碰撞风险动态规划（不依赖 diffusion，可在纯 CPU 上跑通）。

对外主入口：multi_pair.FirstHitRisk
"""

from fhd.risk.multi_pair import FirstHitRisk

__all__ = ["FirstHitRisk"]
