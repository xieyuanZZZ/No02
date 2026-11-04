"""多机器人同步 DDIM 采样（官方 MPD 扩散模型，不重训、不改网络）。

官方 `run_inference(n_samples=B)` 会把**同一组**起终点复制 B 份——
B 是候选数，不是机器人数。本模块把批次组织成 `B*N`：

    index = b * N + i        （b=候选、i=机器人）

每个批次元素有**自己**的 `qs = [start_i, goal_i]`，因此 N 台机器人
共享同一去噪时间步、各自条件独立、梯度通过联合风险耦合。

坐标约定：
  * 官方 2D 世界 = [-1, 1]²，对应 No02 的 10 m × 10 m；
  * 米制 = (q_official + 1) * 5。
"""

from __future__ import annotations

import torch


def official_to_meters(q: torch.Tensor) -> torch.Tensor:
    """官方 [-1,1] -> 米制 [0,10]。"""
    return (q + 1.0) * 5.0


def meters_to_official(q: torch.Tensor) -> torch.Tensor:
    return q / 5.0 - 1.0


def resample_arc_length_diff(q: torch.Tensor, h: int) -> torch.Tensor:
    """**可微**的等弧长重采样：[-1, T, 2] -> [-1, h, 2]。

    `torch.searchsorted` 得到的索引是离散量（不可导，符合预期），
    但插值权重与端点取值都可微，因此梯度能传回控制点。
    """
    seg = torch.linalg.vector_norm(q[..., 1:, :] - q[..., :-1, :], dim=-1)  # [...,T-1]
    zero = torch.zeros_like(seg[..., :1])
    s = torch.cat([zero, seg.cumsum(dim=-1)], dim=-1)                      # [...,T]
    total = s[..., -1:]
    target = torch.linspace(0.0, 1.0, h, dtype=q.dtype, device=q.device) * total
    idx = torch.searchsorted(s.contiguous(), target.contiguous()).clamp(1, q.shape[-2] - 1)
    s_prev = torch.gather(s, -1, (idx - 1).clamp(min=0))
    s_next = torch.gather(s, -1, idx.clamp(max=s.shape[-1] - 1))
    q_prev = torch.gather(q, -2, (idx - 1).clamp(min=0).unsqueeze(-1).expand(*idx.shape, q.shape[-1]))
    q_next = torch.gather(q, -2, idx.clamp(max=q.shape[-2] - 1).unsqueeze(-1).expand(*idx.shape, q.shape[-1]))
    w = ((target - s_prev) / (s_next - s_prev).clamp(min=1e-9)).unsqueeze(-1)
    return q_prev + w * (q_next - q_prev)


class MultiRobotSampler:
    """在官方 MPD 扩散模型上做 N 机器人同步 DDIM。"""

    def __init__(self, planner, n_agents: int, tensor_args: dict):
        self.planner = planner
        self.model = planner.model
        self.dataset = planner.dataset
        self.bspline = planner.planning_task.parametric_trajectory
        self.n = int(n_agents)
        self.tensor_args = tensor_args
        self.sample_kwargs = dict(planner.sample_fn_kwargs)
        self.horizon = int(self.dataset.n_learnable_control_points)

    # ------------------------------------------------------------------ #
    def build_context(self, starts: torch.Tensor, goals: torch.Tensor, n_samples: int,
                      offset: int = 0, total: int | None = None) -> dict:
        """starts/goals: [N,2]（官方坐标）-> 批次 B*N 的 context_d。

        必须走官方的 `create_data_sample_normalized`：context 模型吃的是
        **归一化**的 `qs_normalized`（4 维 = 起点 xy + 终点 xy）。
        """
        n = starts.shape[0]
        dev = starts.device
        ee = torch.eye(4, dtype=torch.float32, device=dev)[:3, :]
        qs_norm, qs_off = [], []
        for i in range(n):
            s_i = starts[i].to(device=dev, dtype=torch.float32)
            g_i = goals[i].to(device=dev, dtype=torch.float32)
            sample = self.dataset.create_data_sample_normalized(
                s_i, g_i, ee_pose_goal=ee)
            qs_norm.append(sample["qs_normalized"].reshape(1, -1))
            qs_off.append(torch.cat([s_i, g_i]).reshape(1, -1))
        qs_n = torch.cat(qs_norm, dim=0)                               # [N,4] 归一化
        qs_o = torch.cat(qs_off, dim=0)                                # [N,4] 官方坐标
        idx = torch.arange(n_samples, device=dev).repeat_interleave(n)
        return {"qs_normalized": qs_n[idx], "qs": qs_o[idx]}, {}

    # ------------------------------------------------------------------ #
    def control_points_to_paths(self, cp_norm: torch.Tensor, starts: torch.Tensor,
                                goals: torch.Tensor, n_samples: int, h: int = 32,
                                dense: int = 128):
        """[B*N,K,2] 归一化控制点 -> 稠密轨迹与执行航点（均为米制）。

        返回 dict：
          cp_m   : [B*N,K,2] 米制控制点
          dense_m: [B*N,dense,2] 米制稠密轨迹（可微）
          paths_m: [B*N,h,2] 米制执行航点（可微，等弧长重采样）
        """
        n = self.n
        b = cp_norm.shape[0] // n
        cp = self.dataset.unnormalize_control_points(cp_norm)
        # 官方约定：q_pos_start / q_pos_goal 形状为 [B, D]
        start_b = starts.repeat(b, 1)
        goal_b = goals.repeat(b, 1)
        q = self.bspline.get_q_trajectory(
            cp, start_b, goal_b, get_type=("pos",), get_time_representation=False)["pos"]
        if dense is not None and q.shape[1] != dense:
            q = resample_arc_length_diff(q, dense)
        dense_m = official_to_meters(q)
        paths_m = resample_arc_length_diff(dense_m, h)
        return {"cp_m": official_to_meters(cp), "dense_m": dense_m, "paths_m": paths_m,
                "q_official": q}

    # ------------------------------------------------------------------ #
    def sample(self, starts: torch.Tensor, goals: torch.Tensor, n_samples: int,
               guide=None, x_init: torch.Tensor | None = None, extra_kwargs: dict | None = None,
               chunk: int | None = None):
        """同步 DDIM 采样。返回 (cp_norm [B*N,K,2], chain, meta)。

        `chunk`：每批最多跑多少个候选。官方模型的 conv1d 在某些批大小下会触发
        cuDNN "unable to find an engine"，分块后批大小固定为 `chunk*N`。
        候选之间本来独立，分块不改变数学结果。
        """
        if chunk is None:
            chunk = n_samples
        kwargs = dict(self.sample_kwargs)
        if extra_kwargs:
            kwargs.update(extra_kwargs)

        cps, chains, ctxs = [], [], []
        for start in range(0, n_samples, chunk):
            nb = min(chunk, n_samples - start)
            context_d, hard_conds = self.build_context(
                starts, goals, nb, offset=start, total=n_samples)
            ctx = {k: v.to(self.tensor_args["device"]) for k, v in context_d.items()}
            out = self.model.conditional_sample(
                hard_conds,
                context_d=ctx,
                batch_size=nb * self.n,
                horizon=self.horizon,
                return_chain=True,
                guide=guide,
                **kwargs,
            )
            chain = out[0] if isinstance(out, (tuple, list)) else out
            cp = chain[-1] if chain.dim() == 4 else chain
            cps.append(cp)
            chains.append(chain)
            ctxs.append(ctx)
        return torch.cat(cps, dim=0), chains, {"context_d": ctxs[0] if len(ctxs) == 1 else ctxs}
