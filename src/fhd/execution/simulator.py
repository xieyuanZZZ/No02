"""联合随机执行 Monte Carlo 模拟器（层 A）。

规则（文档 §3.1、§3.2、§2.3）：
  * 每个执行时隙，机器人在“沿当前路径段前进一格”与“原地不动”之间选择；
  * 硬门槛：仅当 t >= tau_{i,z} 时允许前进；随后以概率 1-p 真的前进；
  * 一个时隙内的运动是线性插值，碰撞按**同步连续运动段**判定（文档 §4.1）；
  * 到达 H-1 后吸收，但仍占据终点坐标参与碰撞。

本模块刻意**自带**一份相对运动最小距离实现，与 risk 模块的 DP 核分开，
避免两处共用同一段错误逻辑而表面上互相印证（批复 §31）；二者在
`tests/test_collision_continuous.py` 中用暴力加密采样交叉验证。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from fhd.geometry.primitives import safe_sqrt


@dataclass
class ExecutionBatch:
    """一批（K 条）随机执行的结果。"""

    n_samples: int
    time_steps: int
    collision: torch.Tensor           # [K] bool：是否发生过任一碰撞
    first_collision_step: torch.Tensor  # [K] long：首次碰撞时隙，未碰撞为 T
    first_collision_pair: torch.Tensor  # [K,2] long：首次碰撞的机器人对，未碰撞为 -1
    pair_collision: torch.Tensor      # [K,P] bool：每对是否碰撞
    pair_min_distance: torch.Tensor   # [K,P]：每对全程最小中心距离
    completion_steps: torch.Tensor    # [K,N] long：到达终点的时隙，未到达为 -1
    makespan_steps: torch.Tensor      # [K] long：最后到达时隙，未全部到达为 T
    all_arrived: torch.Tensor         # [K] bool
    traces: dict | None = None

    # --- 统计量 ---------------------------------------------------------
    def collision_rate(self) -> float:
        return float(self.collision.to(torch.float64).mean())

    def pair_collision_rate(self) -> torch.Tensor:
        return self.pair_collision.to(torch.float64).mean(dim=0)

    def mean_makespan_steps(self) -> float:
        return float(self.makespan_steps.to(torch.float64).mean())

    def completion_rate(self) -> float:
        return float(self.all_arrived.to(torch.float64).mean())

    def summary(self, dt_s: float, d_safe: float) -> dict:
        ms = self.makespan_steps.to(torch.float64)
        comp = self.completion_steps.to(torch.float64).clamp(min=0)
        finished = (self.completion_steps >= 0).to(torch.float64)
        return {
            "n_samples": self.n_samples,
            "collision_rate": self.collision_rate(),
            "pair_collision_rate": [round(float(x), 6) for x in self.pair_collision_rate()],
            "completion_rate": self.completion_rate(),
            "mean_makespan_steps": float(ms.mean()),
            "mean_makespan_s": float(ms.mean() * dt_s),
            "std_makespan_s": float(ms.std(unbiased=False) * dt_s),
            "unfinished_agent_rate": float(1.0 - finished.mean()),
            "mean_completion_time_s": [round(float(x), 6) for x in (comp * finished).mean(0) * dt_s],
            "d_safe_m": d_safe,
        }


def _pair_index(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    ii, jj = [], []
    for i in range(n):
        for j in range(i + 1, n):
            ii.append(i)
            jj.append(j)
    return torch.tensor(ii, dtype=torch.long), torch.tensor(jj, dtype=torch.long)


def _relative_min_distance(r0: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """独立实现的相对运动最小距离（投影形式，与 DP 核实现路径不同）。"""
    vv = (v * v).sum(-1, keepdim=True)
    # 用最小二乘投影求最优 u，而不是直接写闭式 clip
    proj = -(r0 * v).sum(-1, keepdim=True) / torch.clamp(vv, min=1e-18)
    u = torch.minimum(torch.maximum(proj, torch.zeros_like(proj)), torch.ones_like(proj))
    u = torch.where(vv > 1e-18, u, torch.zeros_like(u))
    diff = r0 + u * v
    return safe_sqrt((diff * diff).sum(-1))


def dense_relative_min_distance(r0: torch.Tensor, v: torch.Tensor, n_samples: int = 2001) -> torch.Tensor:
    """暴力加密采样版相对运动最小距离，仅用于独立验证（不用于主流程）。"""
    u = torch.linspace(0.0, 1.0, n_samples, dtype=r0.dtype, device=r0.device)
    pts = r0.unsqueeze(-2) + u.view(*([1] * (r0.dim() - 1)), n_samples, 1) * v.unsqueeze(-2)
    return safe_sqrt((pts * pts).sum(-1)).min(dim=-1).values


def simulate_execution(
    paths: torch.Tensor,
    wait: torch.Tensor,
    pause_p: torch.Tensor,
    d_safe: float,
    time_steps: int,
    dt_s: float,
    n_samples: int = 1000,
    chunk_size: int = 2000,
    generator: torch.Generator | None = None,
    record_traces: bool = False,
    record_max_samples: int = 64,
) -> ExecutionBatch:
    """运行 n_samples 条独立随机执行。

    参数
    ----
    paths   : [N,H,2] 几何路径（float64 优先）
    wait    : [N,H-1] 非负等待预算（时隙）
    pause_p : [N] 每次可用前进时的暂停概率
    d_safe  : 安全阈值（米）
    time_steps : T
    """
    from fhd.execution.progress import release_times

    if paths.dim() != 3:
        raise ValueError("paths 必须是 [N,H,2]")
    n, h, _ = paths.shape
    if wait.shape != (n, h - 1):
        raise ValueError(f"wait 形状应为 {(n, h - 1)}，实际 {tuple(wait.shape)}")
    dtype, device = paths.dtype, paths.device
    wait = wait.to(dtype)
    pause_p = pause_p.to(dtype).reshape(n)

    tau = release_times(wait)  # [N,H-1]
    tau_t = tau.transpose(0, 1).contiguous()  # [H-1,N]
    ii, jj = _pair_index(n)
    ii = ii.to(device)
    jj = jj.to(device)
    n_pairs = ii.numel()

    collisions, first_steps, first_pairs = [], [], []
    pair_collisions, pair_min = [], []
    completion, makespans, all_arrived = [], [], []
    trace_positions, trace_progress, trace_collision = [], [], []
    n_traced = 0

    flat_paths = paths.reshape(n * h, 2)
    agent_offset = torch.arange(n, device=device).reshape(1, n) * h
    agent_idx = torch.arange(n, device=device).reshape(1, n)

    def gather_positions(zz: torch.Tensor) -> torch.Tensor:
        idx = agent_offset + zz  # [k,n]
        return torch.gather(flat_paths.unsqueeze(0).expand(zz.shape[0], -1, -1), 1,
                            idx.unsqueeze(-1).expand(-1, -1, 2))

    done = 0
    while done < n_samples:
        k = min(chunk_size, n_samples - done)
        z = torch.zeros(k, n, dtype=torch.long, device=device)
        pos_t = gather_positions(z)

        positions = torch.empty(k, n, time_steps + 1, 2, dtype=dtype, device=device)
        progress = (
            torch.empty(k, n, time_steps + 1, dtype=torch.long, device=device)
            if record_traces
            else None
        )
        positions[:, :, 0, :] = pos_t
        if progress is not None:
            progress[:, :, 0] = z

        first_c = torch.full((k,), time_steps, dtype=torch.long, device=device)
        first_pair_step = torch.full((k, n_pairs), time_steps, dtype=torch.long, device=device)
        first_pair_idx = torch.full((k, 2), -1, dtype=torch.long, device=device)
        min_pair = torch.full((k, n_pairs), float("inf"), dtype=dtype, device=device)
        comp = torch.full((k, n), -1, dtype=torch.long, device=device)
        arrived = torch.zeros(k, n, dtype=torch.bool, device=device)

        for t in range(time_steps):
            z_gather = z.clamp(max=h - 2)
            tau_z = tau_t[z_gather, agent_idx]  # [K,N]，按当前进度取对应放行门槛
            allowed = t >= tau_z
            rand = torch.rand(k, n, generator=generator, device=device)
            move = (z < h - 1) & allowed & (rand >= pause_p.unsqueeze(0))
            z_new = torch.where(move, z + 1, z)

            pos_new = gather_positions(z_new)

            positions[:, :, t + 1, :] = pos_new
            if progress is not None:
                progress[:, :, t + 1] = z_new

            r0 = pos_t[:, ii, :] - pos_t[:, jj, :]
            v = (pos_new[:, ii, :] - pos_t[:, ii, :]) - (pos_new[:, jj, :] - pos_t[:, jj, :])
            d = _relative_min_distance(r0, v)  # [K,P]
            min_pair = torch.minimum(min_pair, d)

            hit = d < d_safe
            any_hit = hit.any(dim=-1)
            new_hit = any_hit & (first_c == time_steps)
            first_c = torch.where(new_hit, torch.full_like(first_c, t), first_c)
            hit_idx = hit.to(torch.long).argmax(dim=-1)
            first_pair_idx = torch.where(
                new_hit.unsqueeze(-1),
                torch.stack([ii[hit_idx], jj[hit_idx]], dim=-1),
                first_pair_idx,
            )
            first_pair_step = torch.where(
                hit & (first_pair_step == time_steps), torch.full_like(first_pair_step, t), first_pair_step
            )

            newly = (z_new == h - 1) & (~arrived)
            comp = torch.where(newly, torch.full_like(comp, t + 1), comp)
            arrived = arrived | newly

            pos_t = pos_new
            z = z_new

        made = comp.max(dim=-1).values
        made = torch.where(arrived.all(dim=-1), made, torch.full_like(made, time_steps))

        collisions.append(first_c < time_steps)
        first_steps.append(first_c)
        first_pairs.append(first_pair_idx)
        pair_collisions.append(first_pair_step < time_steps)
        pair_min.append(min_pair)
        completion.append(comp)
        makespans.append(made)
        all_arrived.append(arrived.all(dim=-1))

        if record_traces and n_traced < record_max_samples and progress is not None:
            take = min(record_max_samples - n_traced, k)
            trace_positions.append(positions[:take].clone())
            trace_progress.append(progress[:take].clone())
            trace_collision.append(first_c[:take].clone())
            n_traced += take

        done += k

    out = ExecutionBatch(
        n_samples=n_samples,
        time_steps=time_steps,
        collision=torch.cat(collisions),
        first_collision_step=torch.cat(first_steps),
        first_collision_pair=torch.cat(first_pairs),
        pair_collision=torch.cat(pair_collisions),
        pair_min_distance=torch.cat(pair_min),
        completion_steps=torch.cat(completion),
        makespan_steps=torch.cat(makespans),
        all_arrived=torch.cat(all_arrived),
    )
    if trace_positions:
        out.traces = {
            "positions": torch.cat(trace_positions),
            "progress": torch.cat(trace_progress),
            "first_collision_step": torch.cat(trace_collision),
            "dt_s": dt_s,
            "d_safe": d_safe,
        }
    return out


def simulate_deterministic(
    paths: torch.Tensor,
    wait: torch.Tensor,
    pause_p: torch.Tensor,
    d_safe: float,
    time_steps: int,
    dt_s: float,
) -> dict:
    """p=0 的确定性执行：唯一一条轨线，用于 R2 的一致性检查。"""
    batch = simulate_execution(
        paths=paths,
        wait=wait,
        pause_p=torch.zeros_like(pause_p),
        d_safe=d_safe,
        time_steps=time_steps,
        dt_s=dt_s,
        n_samples=4,
        chunk_size=4,
        record_traces=True,
    )
    return {
        "collision": bool(batch.collision[0]),
        "first_collision_step": int(batch.first_collision_step[0]),
        "completion_steps": batch.completion_steps[0].tolist(),
        "makespan_steps": int(batch.makespan_steps[0]),
        "positions": batch.traces["positions"][0] if batch.traces else None,
    }
