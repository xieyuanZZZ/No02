"""地图条件下的 1D U-Net 去噪网络（文档 §6.3）。

输入：
    当前噪声轨迹 X_l : [B,H,2]（已归一化到 [-1,1]）
    扩散步 l        : [B]
    地图占据栅格 M  : [B,1,64,64]
    起点/终点 S,G   : [B,2] 各
输出：
    预测噪声 ε_θ  : [B,H,2]

结构保持“小型 1D U-Net + 条件 FiLM”，不引入 Transformer 或额外预测头。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
        """dtype 缺省 float32；父模块传入 x.dtype 以便支持 float64 / AMP。"""
        dtype = dtype or torch.float32
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=device, dtype=dtype) / (half - 1)
        )
        args = t.to(dtype).unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class FiLMResBlock(nn.Module):
    """带 FiLM 条件调制的残差块（1D）。"""

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, groups: int = 8):
        super().__init__()
        groups_in = math.gcd(groups, in_ch)
        groups_out = math.gcd(groups, out_ch)
        self.norm1 = nn.GroupNorm(groups_in, in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, 5, padding=2)
        self.norm2 = nn.GroupNorm(groups_out, out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 5, padding=2)
        self.film = nn.Linear(cond_dim, out_ch * 2)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.film(cond).chunk(2, dim=-1)
        h = self.norm2(h)
        h = F.silu(h) * (1.0 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)
        h = self.conv2(h)
        return h + self.skip(x)


class MapEncoder(nn.Module):
    """小型 CNN：占据栅格 -> 空间特征 [B,C,out_res,out_res] + 全局嵌入 [B,embed_dim]。

    **空间特征必须保留二维结构**：后续用 `grid_sample` 按每个路点的
    归一化二维坐标去查询局部地图特征。绝不能把二维特征压成 (1,H)——
    那会把 y 方向的信息整体丢掉，使路点位置与障碍位置失去对应关系。

    分辨率（S2 消融）：
      * `in_res`  —— 输入栅格边长（64 或 128，**从连续几何重新栅格化**，不是插值放大）
      * `out_res` —— 输出特征图边长（16/32/64）
      * 下采样次数 = log2(in_res / out_res)，全部用 stride-2 卷积，不用池化
    结构固定为「两个全分辨率卷积 + n 次下采样」，保证不同分辨率的对比只改变
    空间尺度，不改变通道预算与深度。
    """

    def __init__(self, out_dim: int = 128, spatial_channels: int = 16, in_channels: int = 1,
                 map_res: int = 8, in_res: int = 64, hidden: tuple[int, int] = (16, 32)):
        super().__init__()
        self.spatial_channels = spatial_channels
        self.map_res = int(map_res)
        self.in_res = int(in_res)
        if self.in_res % self.map_res != 0:
            raise ValueError(f"in_res={self.in_res} 不能被 map_res={self.map_res} 整除")
        ratio = self.in_res // self.map_res
        n_down = int(round(math.log2(ratio)))
        if 2 ** n_down != ratio:
            raise ValueError(f"in_res/map_res 必须是 2 的幂，实际 {ratio}")
        h0, h1 = hidden
        layers = [
            nn.Conv2d(in_channels, h0, 5, padding=2),
            nn.SiLU(),
            nn.Conv2d(h0, h1, 5, padding=2),
            nn.SiLU(),
        ]
        for _ in range(n_down):
            layers += [nn.Conv2d(h1, h1, 3, stride=2, padding=1), nn.SiLU()]
        layers += [nn.Conv2d(h1, spatial_channels, 3, padding=1), nn.SiLU()]
        self.body = nn.Sequential(*layers)
        self.embed = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(spatial_channels, out_dim),
        )

    def forward(self, grid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.body(grid)
        return feat, self.embed(feat)


def sample_map_features(feature_map: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """按路点的归一化二维坐标，从地图特征图里查询局部特征。

    参数
    ----
    feature_map : [B, C, Hf, Wf]，行索引对应 **y**、列索引对应 **x**
                  （与 `WorldMap.occupancy` / `raster_sdf` 的约定一致：
                   row i 的中心 y=(i+0.5)*H/G，col j 的中心 x=(j+0.5)*W/G）
    points      : [B, H, 2]，归一化世界坐标 ∈ [-1, 1]，即 x_norm=2x/W-1, y_norm=2y/H-1

    返回 [B, C, H]。

    坐标约定（这是最容易出错的地方，由 `tests/test_map_conditioning.py` 锁定）：
      * `F.grid_sample` 的 grid[..., 0] 是 **x**（对应 Wf/列），grid[..., 1] 是 **y**（对应 Hf/行）；
      * `align_corners=True` 时，-1 ↔ 第一列/第一行，+1 ↔ 最后一列/最后一行；
      * 我们的栅格第 0 行就是 y 最小处（origin='lower'），因此 y_norm 与 grid y 同向，
        **不需要翻转**；若将来把栅格改成 origin='upper'，这里必须同步翻转；
      * 越界路点用 `padding_mode="border"`（贴边取值），避免露出的轨迹点拿到 0 特征。
    """
    grid = points.unsqueeze(2)  # [B,H,1,2] = (x, y)
    feat = F.grid_sample(feature_map, grid, mode="bilinear", align_corners=True,
                         padding_mode="border")  # [B,C,H,1]
    return feat.squeeze(-1)  # [B,C,H]


class ConditionalUNet1D(nn.Module):
    """条件 1D U-Net：通道 64/128/256，两层下采样（H=32 -> 16 -> 8）。"""

    def __init__(
        self,
        in_channels: int = 2,
        base_channels: int = 64,
        mults: tuple[int, int, int] = (1, 2, 4),
        cond_dim: int = 128,
        map_feature_dim: int = 128,
        grid_size: int = 64,
        map_spatial_channels: int = 16,
        map_res: int = 8,
        map_conditioning: str = "grid_sample",
    ):
        super().__init__()
        ch1, ch2, ch3 = [base_channels * m for m in mults]
        self.cond_dim = cond_dim
        self.map_spatial_channels = map_spatial_channels
        if map_conditioning not in ("grid_sample", "index"):
            raise ValueError(f"未知 map_conditioning: {map_conditioning}")
        self.map_conditioning = map_conditioning

        self.time_emb = SinusoidalPosEmb(cond_dim)
        self.time_mlp = nn.Sequential(nn.Linear(cond_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        self.map_encoder = MapEncoder(map_feature_dim, spatial_channels=map_spatial_channels,
                                      in_channels=1, map_res=map_res, in_res=grid_size)
        self.map_proj = nn.Linear(map_feature_dim, cond_dim)
        self.cond_mlp = nn.Sequential(
            nn.Linear(4, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim)
        )

        self.in_conv = nn.Conv1d(in_channels + map_spatial_channels, ch1, 5, padding=2)

        self.down1 = FiLMResBlock(ch1, ch1, cond_dim)
        self.down2 = FiLMResBlock(ch1, ch2, cond_dim)
        self.down3 = FiLMResBlock(ch2, ch3, cond_dim)

        self.pool1 = nn.Conv1d(ch1, ch1, 3, stride=2, padding=1)
        self.pool2 = nn.Conv1d(ch2, ch2, 3, stride=2, padding=1)

        self.mid = FiLMResBlock(ch3, ch3, cond_dim)

        self.up2 = FiLMResBlock(ch3 + ch2, ch2, cond_dim)
        self.up1 = FiLMResBlock(ch2 + ch1, ch1, cond_dim)

        self.out_norm = nn.GroupNorm(math.gcd(8, ch1), ch1)
        self.out_conv = nn.Conv1d(ch1, in_channels, 5, padding=2)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        start: torch.Tensor,
        goal: torch.Tensor,
        grid: torch.Tensor,
    ) -> torch.Tensor:
        """x [B,H,2] -> eps [B,H,2]"""
        map_spatial, map_embed = self.map_encoder(grid)
        cond = (
            self.time_mlp(self.time_emb(t, dtype=x.dtype))
            + self.map_proj(map_embed)
            + self.cond_mlp(torch.cat([start, goal], dim=-1))
        )

        traj = x.transpose(1, 2)  # [B,2,H]
        if self.map_conditioning == "grid_sample":
            # 逐路点二维位置查询：x 是当前（带噪）轨迹，训练与采样完全一致，不使用真实 x0
            map_feat = sample_map_features(map_spatial, x)  # [B,C,H]
        else:
            # 旧实现（有 bug）：把 (Hf,Wf) 压成 (1,H)，丢掉 y 方向且与路点坐标无关
            map_feat = F.interpolate(
                map_spatial, size=(1, traj.shape[-1]), mode="nearest"
            ).squeeze(2)  # [B,C,H]
        h = torch.cat([traj, map_feat], dim=1)  # [B,2+C,H]
        h0 = self.in_conv(h)             # [B,ch1,H]
        d1 = self.down1(h0, cond)        # [B,ch1,H]
        p1 = self.pool1(d1)              # [B,ch1,H/2]
        d2 = self.down2(p1, cond)        # [B,ch2,H/2]
        p2 = self.pool2(d2)              # [B,ch2,H/4]
        d3 = self.down3(p2, cond)        # [B,ch3,H/4]
        m = self.mid(d3, cond)

        u2 = F.interpolate(m, size=d2.shape[-1], mode="nearest")
        u2 = self.up2(torch.cat([u2, d2], dim=1), cond)
        u1 = F.interpolate(u2, size=d1.shape[-1], mode="nearest")
        u1 = self.up1(torch.cat([u1, d1], dim=1), cond)

        out = self.out_conv(F.silu(self.out_norm(u1)))
        return out.transpose(1, 2)  # [B,H,2]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class EMA:
    """指数滑动平均（只作用于浮点参数/缓冲）。"""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {
            k: v.detach().clone() for k, v in model.state_dict().items() if v.dtype.is_floating_point
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict({**model.state_dict(), **self.shadow}, strict=False)

    def state_dict(self) -> dict:
        return self.shadow

    def load_state_dict(self, state: dict) -> None:
        self.shadow = {k: v.clone() for k, v in state.items()}
