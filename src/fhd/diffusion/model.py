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

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=device, dtype=torch.float32) / (half - 1)
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
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
    """小型 CNN：64×64 占据栅格 -> map_feature_dim 维向量。"""

    def __init__(self, out_dim: int = 128, in_channels: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, 5, stride=2, padding=2),   # 32
            nn.SiLU(),
            nn.Conv2d(16, 32, 5, stride=2, padding=2),            # 16
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),            # 8
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, out_dim),
        )

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        return self.net(grid)


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
    ):
        super().__init__()
        ch1, ch2, ch3 = [base_channels * m for m in mults]
        self.cond_dim = cond_dim

        self.time_emb = SinusoidalPosEmb(cond_dim)
        self.time_mlp = nn.Sequential(nn.Linear(cond_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        self.map_encoder = MapEncoder(map_feature_dim, in_channels=1)
        self.map_proj = nn.Linear(map_feature_dim, cond_dim)
        self.cond_mlp = nn.Sequential(
            nn.Linear(4, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim)
        )

        self.in_conv = nn.Conv1d(in_channels, ch1, 5, padding=2)

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
        cond = (
            self.time_mlp(self.time_emb(t))
            + self.map_proj(self.map_encoder(grid))
            + self.cond_mlp(torch.cat([start, goal], dim=-1))
        )

        h = x.transpose(1, 2)  # [B,2,H]
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
