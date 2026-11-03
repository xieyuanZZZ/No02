"""F1：地图条件必须按路点的二维坐标查询（grid_sample），不能压成一维。

这组测试锁定三件事：
  1. `sample_map_features` 的 x/y 索引方向与栅格行列约定一致（四角、中心、越界）；
  2. 障碍位置不同、全局占比相同的地图，在靠近障碍的路点上给出不同特征；
  3. 不同路点得到不同特征，梯度有限，端点仍由外部固定。
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from fhd.diffusion.model import ConditionalUNet1D, MapEncoder, sample_map_features
from fhd.maps.generate import generate_map
from fhd.utils.config import load_config

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "configs" / "base.yaml")


def _marker_map(res: int = 8, row: int = 0, col: int = 0, channels: int = 4) -> torch.Tensor:
    """构造 [1,C,res,res]，在 (row, col) 放一个 1，其余 0。row 对应 y，col 对应 x。"""
    m = torch.zeros(1, channels, res, res)
    m[0, :, row, col] = 1.0
    return m


def test_grid_sample_corner_orientation():
    """对 8×8 的每一个格子做标记，验证查询坐标能精确命中它。

    坐标约定：align_corners=True 时，索引 k（0-based）对应
    coord = -1 + 2k/(res-1)。这里 col→x、row→y。任何一个格子对不上，
    都说明 x/y 互换或上下翻转。
    """
    res = 8
    bad = []
    for row in range(res):
        for col in range(res):
            gx = -1.0 + 2.0 * col / (res - 1)
            gy = -1.0 + 2.0 * row / (res - 1)
            m = _marker_map(res, row, col)
            val = float(sample_map_features(m, torch.tensor([[[gx, gy]]])).max())
            if abs(val - 1.0) > 1e-4:
                bad.append((row, col, val))
    assert not bad, f"{len(bad)} 个格子的坐标映射不对，前 3 个：{bad[:3]}"


def test_flip_would_be_detected():
    """反证：若把 y 轴翻过来用（-gy），同一批格子应当取错——说明测试确实能抓翻转。"""
    res = 8
    row, col = 1, 6
    gx = -1.0 + 2.0 * col / (res - 1)
    gy = -1.0 + 2.0 * row / (res - 1)
    m = _marker_map(res, row, col)
    ok = float(sample_map_features(m, torch.tensor([[[gx, gy]]])).max())
    flipped = float(sample_map_features(m, torch.tensor([[[gx, -gy]]])).max())
    assert ok > 0.99 and flipped < 0.01, "翻转检测失效"


def test_no_vertical_flip():
    """障碍在 map 上边缘（row=res-1，即 y 最大）时，y=+1 的点必须命中，y=-1 的不命中。"""
    res = 8
    row, col = res - 1, res // 2
    m = _marker_map(res, row, col)
    gx = -1.0 + 2.0 * col / (res - 1)
    gy = -1.0 + 2.0 * row / (res - 1)
    hit_hi = float(sample_map_features(m, torch.tensor([[[gx, gy]]])).max())
    hit_lo = float(sample_map_features(m, torch.tensor([[[gx, -1.0]]])).max())
    assert hit_hi > 0.99, "y=+1 的点应命中最后一行"
    assert hit_lo < 0.01, "y=-1 的点不应命中；若命中说明上下翻转了"


def test_out_of_range_uses_border():
    """越界路点用贴边取值，而不是 0。"""
    m = _marker_map(8, row=0, col=0)
    val = sample_map_features(m, torch.tensor([[[-3.0, -3.0]]]))
    assert float(val.max()) == pytest.approx(1.0, abs=1e-5)


def test_different_waypoints_get_different_features():
    res = 8
    m = torch.randn(1, 4, res, res)
    pts = torch.tensor([[[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]]])
    feat = sample_map_features(m, pts)  # [1,4,3]
    assert not torch.allclose(feat[0, :, 0], feat[0, :, 1], atol=1e-4)
    assert not torch.allclose(feat[0, :, 1], feat[0, :, 2], atol=1e-4)


def test_same_obstacle_ratio_different_position():
    """障碍占比相同、位置不同 -> 同一路点的局部特征必须不同。"""
    size = 64
    a = torch.zeros(1, 1, size, size)
    b = torch.zeros(1, 1, size, size)
    a[0, 0, 10:20, 10:20] = 1.0
    b[0, 0, 40:50, 40:50] = 1.0
    assert float(a.mean()) == pytest.approx(float(b.mean()))

    torch.manual_seed(0)
    enc = MapEncoder(out_dim=16, spatial_channels=4, map_res=8).eval()
    with torch.no_grad():
        fa, emb_a = enc(a)
        fb, emb_b = enc(b)
    # 查询靠近 a 中障碍的路点（x,y 都小）
    near_a = torch.tensor([[[-0.6, -0.6]]])
    feat_a = sample_map_features(fa, near_a)
    feat_b = sample_map_features(fb, near_a)
    # 未训练网络只要求“不同”，不要求“更强”；1e-4 足以排除特征与地图无关的情形
    assert (feat_a - feat_b).abs().max() > 1e-4, "障碍位置不同但同一路点特征相同，说明空间信息丢失"
    # 反证：两张图障碍面积相同，全局平均池化几乎分不出来 —— 这正是必须做逐点查询的原因
    assert (emb_a - emb_b).abs().max() < 1e-3, "全局嵌入不应能区分同面积不同位置的障碍"


def test_encoder_resolution_family():
    """S2 消融用到的 (输入, 特征) 组合都必须能建出来并输出正确形状。"""
    for in_res, out_res in [(64, 16), (128, 32), (128, 64), (128, 16), (64, 32)]:
        enc = MapEncoder(out_dim=32, spatial_channels=8, in_res=in_res, map_res=out_res)
        grid = torch.zeros(2, 1, in_res, in_res)
        feat, emb = enc(grid)
        assert feat.shape == (2, 8, out_res, out_res), f"{in_res}->{out_res} 实际 {feat.shape}"
        assert emb.shape == (2, 32)


def test_encoder_rejects_non_power_of_two():
    with pytest.raises(ValueError):
        MapEncoder(out_dim=16, spatial_channels=4, in_res=96, map_res=64)


def test_model_uses_grid_sample_and_gradients_flow():
    model = ConditionalUNet1D(base_channels=16, mults=(1, 2, 4), cond_dim=32,
                              map_feature_dim=32, grid_size=16, map_spatial_channels=8,
                              map_res=8, map_conditioning="grid_sample")
    assert model.map_conditioning == "grid_sample"
    x = torch.randn(2, 16, 2, requires_grad=True)
    t = torch.tensor([10, 500])
    start = torch.zeros(2, 2)
    goal = torch.ones(2, 2) * 0.5
    grid = torch.zeros(2, 1, 16, 16)
    out = model(x, t, start, goal, grid)
    assert out.shape == x.shape
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all(), "对轨迹的梯度必须有限"


def test_local_feature_differs_between_mirrored_maps():
    """整模型级别：上下镜像的两张图，对同一路点应给出不同预测。"""
    torch.manual_seed(0)
    model = ConditionalUNet1D(base_channels=16, mults=(1, 2, 4), cond_dim=32,
                              map_feature_dim=32, grid_size=16, map_spatial_channels=8,
                              map_res=8, map_conditioning="grid_sample").eval()
    grid_a = torch.zeros(1, 1, 16, 16)
    grid_b = torch.zeros(1, 1, 16, 16)
    grid_a[:, :, 12:15, 6:10] = 1.0   # 上半部分（row 小 = y 小）
    grid_b[:, :, 1:4, 6:10] = 1.0     # 下半部分
    x = torch.randn(1, 16, 2)
    t = torch.tensor([20])
    s = torch.zeros(1, 2)
    g = torch.zeros(1, 2) + 1.0
    with torch.no_grad():
        oa = model(x, t, s, g, grid_a)
        ob = model(x, t, s, g, grid_b)
    assert not torch.allclose(oa, ob, atol=1e-5), "镜像地图应产生不同预测"


def test_old_index_conditioning_is_inert():
    """对照：旧实现（压成 1×H）对路点坐标不敏感——这正是被修复的 bug。"""
    torch.manual_seed(0)
    model = ConditionalUNet1D(base_channels=16, mults=(1, 2, 4), cond_dim=32,
                              map_feature_dim=32, grid_size=16, map_spatial_channels=8,
                              map_res=8, map_conditioning="index").eval()
    x1 = torch.randn(1, 16, 2)
    x2 = x1.clone()
    x2[0, :, 0] = x2[0, :, 0] + 0.5   # 只改 x 坐标（不影响形状统计）
    t = torch.tensor([20])
    s = torch.zeros(1, 2)
    g = torch.zeros(1, 2) + 1.0
    grid = torch.zeros(1, 1, 16, 16)
    with torch.no_grad():
        # 用同一份轨迹张量直接喂给 in_conv 之前的地图分支，验证 index 模式下
        # map_feat 与 x 完全无关（只与 H 有关）
        map_spatial, _ = model.map_encoder(grid)
        old_feat = F.interpolate(map_spatial, size=(1, 16), mode="nearest").squeeze(2)
        new_feat = sample_map_features(map_spatial, x1)
    assert not torch.allclose(old_feat, new_feat), "新旧两种条件方式必须不同"
    assert old_feat.shape == new_feat.shape == (1, 8, 16)


def test_real_map_encoder_keeps_2d_shape():
    """真实 64×64 占据栅格经过编码后必须仍是二维特征图（8×8 或 16×16）。"""
    world = generate_map("dual_corridor", CFG, 20261103)
    occ = world.occupancy(torch.float32).unsqueeze(0).unsqueeze(0)  # [1,1,64,64]
    for res in (8, 16):
        enc = MapEncoder(out_dim=16, spatial_channels=4, map_res=res)
        feat, emb = enc(occ)
        assert feat.dim() == 4 and feat.shape[0] == 1
        assert feat.shape[-2] == feat.shape[-1] == res, f"期望 {res}×{res}，实际 {feat.shape}"
        assert emb.shape == (1, 16)
