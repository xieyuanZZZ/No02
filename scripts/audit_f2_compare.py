"""F2：旧/新地图条件架构的公平小规模对照。

同一训练数据、同一种子、同样 8000 步、同一批测试任务，只改 `map_conditioning`：
  * `index`        —— 旧实现：把二维特征压成 (1,H)，与路点坐标无关（有 bug）
  * `grid_sample`  —— 新实现：按路点二维坐标做局部查询

评估：
  * 20 个固定任务（按地图类型分层，每类 5 个），每个任务 8 条候选；
  * 逐任务成功率 + 任务级聚类 bootstrap 95% CI（不是把候选当独立样本）；
  * 每种地图类型的失败分解：静态碰撞 / 速度越界 / 两者同时满足；
  * 两模型在同一任务上的配对差。

用法：
  python scripts/audit_f2_compare.py --old <ckpt> --new <ckpt>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fhd.data.dataset import TrajectoryDataset  # noqa: E402
from fhd.data.preprocess import diversity_score  # noqa: E402
from fhd.diffusion.ddim import ddim_sample  # noqa: E402
from fhd.diffusion.schedule import DDPMSchedule  # noqa: E402
from fhd.diffusion.train import build_model  # noqa: E402
from fhd.evaluation.reporting import StageReport  # noqa: E402
from fhd.evaluation.statistics import cluster_bootstrap_ci, paired_bootstrap_diff  # noqa: E402
from fhd.evaluation.visualizations import draw_map  # noqa: E402
from fhd.geometry.collisions import check_speed_limit, trajectory_is_static_safe  # noqa: E402
from fhd.geometry.paths import path_length  # noqa: E402
from fhd.utils.config import load_config  # noqa: E402
from fhd.utils.logging import get_logger, log_section  # noqa: E402
from fhd.utils.runtime import new_run_dir, system_info  # noqa: E402
from fhd.utils.seed import set_seed  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

KIND_CN = {"open": "空旷交叉", "dual_corridor": "双通路绕障",
           "narrow_gate": "窄通道/门口", "bottleneck": "多交叉/瓶颈"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_model(cfg, ckpt_path: Path, device):
    model = build_model(cfg, device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("ema") or ckpt["model"], strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ckpt


def pick_tasks(ds: TrajectoryDataset, per_kind: int, seed: int) -> list[dict]:
    """按地图类型分层挑任务，同一批任务供两个模型共用。"""
    by_kind = defaultdict(list)
    for idx in range(len(ds)):
        mi = int(ds.data["map_idx"][idx])
        by_kind[ds.maps[mi].kind].append(idx)
    rng = np.random.default_rng(seed)
    picked = []
    for kind in ("open", "dual_corridor", "narrow_gate", "bottleneck"):
        pool = by_kind.get(kind, [])
        if not pool:
            continue
        take = min(per_kind, len(pool))
        sel = rng.choice(pool, size=take, replace=False)
        for idx in sorted(int(i) for i in sel):
            picked.append({"task": idx, "kind": kind})
    return picked


@torch.no_grad()
def evaluate_model(model, schedule, ds, tasks, n_samples, num_steps, v_max, dt_s, device, seed):
    rows = []
    gen = torch.Generator(device=device.type).manual_seed(seed)
    for t in tasks:
        idx = t["task"]
        item = ds[idx]
        start = item["start"].unsqueeze(0).expand(n_samples, -1).to(device)
        goal = item["goal"].unsqueeze(0).expand(n_samples, -1).to(device)
        grid = item["grid"].unsqueeze(0).expand(n_samples, -1, -1, -1).to(device)
        samples = ddim_sample(model, schedule, start, goal, grid, num_steps=num_steps,
                              generator=gen)
        paths = ds.denormalise(samples.float()).detach().cpu()
        world = ds.maps[int(item["map_idx"])]
        safe, clr = trajectory_is_static_safe(paths, world.obstacles(),
                                              world.robot_radius_m, world.clearance_margin_m)
        speed_ok, max_seg = check_speed_limit(paths, v_max, dt_s)
        lengths = path_length(paths)
        gt_len = float(path_length(ds.data["paths"][idx].unsqueeze(0))[0])
        ok = safe & speed_ok
        rows.append({
            "task": idx, "kind": t["kind"],
            "success_rate": float(ok.float().mean()),
            "static_safe_rate": float(safe.float().mean()),
            "speed_ok_rate": float(speed_ok.float().mean()),
            "min_clearance_m": float(clr.min()),
            "length_ratio": float((lengths / max(gt_len, 1e-6)).mean()),
            "diversity_m": diversity_score(paths),
            "n_samples": n_samples,
        })
    return rows


def summarize(rows, name, n_boot=2000):
    vals = np.array([r["success_rate"] for r in rows])
    clusters = np.array([r["task"] for r in rows])
    ci = cluster_bootstrap_ci(vals, clusters, n_boot=n_boot)
    by_kind = {}
    for kind in KIND_CN:
        sel = [r for r in rows if r["kind"] == kind]
        if not sel:
            continue
        by_kind[kind] = {
            "n_tasks": len(sel),
            "success_rate": float(np.mean([r["success_rate"] for r in sel])),
            "static_safe_rate": float(np.mean([r["static_safe_rate"] for r in sel])),
            "speed_ok_rate": float(np.mean([r["speed_ok_rate"] for r in sel])),
            "length_ratio": float(np.mean([r["length_ratio"] for r in sel])),
            "diversity_m": float(np.mean([r["diversity_m"] for r in sel])),
        }
    return {"name": name, "n_tasks": len(rows), "success_rate": float(vals.mean()),
            "ci": ci, "by_kind": by_kind,
            "static_safe_rate": float(np.mean([r["static_safe_rate"] for r in rows])),
            "speed_ok_rate": float(np.mean([r["speed_ok_rate"] for r in rows])),
            "length_ratio": float(np.mean([r["length_ratio"] for r in rows])),
            "diversity_m": float(np.mean([r["diversity_m"] for r in rows]))}


@torch.no_grad()
def map_sensitivity(model, ds, tasks, device, num_probes: int = 8) -> dict:
    """探针：模型对"地图内容"到底敏不敏感。

    对同一批轨迹/时间步，只换输入地图，看 ε 预测变化多少：
      * shift ：把同一张图的空间内容平移几格（障碍数量不变）
      * blank ：换成全自由地图
      * other ：换成另一种地图类型的图
    如果变化接近 0，说明地图条件根本没被用上，架构差异就无从体现。
    """
    x = torch.randn(num_probes, 32, 2, generator=torch.Generator().manual_seed(0))
    t = torch.full((num_probes,), 300, dtype=torch.long)
    s = torch.rand(num_probes, 2) * 1.6 - 0.8
    g = torch.rand(num_probes, 2) * 1.6 - 0.8

    true_maps, shift_maps, other_maps = [], [], []
    # 预先按地图 id 分组，保证 "换一张图" 真的换成**不同地图**
    by_map = {}
    for tk in tasks:
        idx = tk["task"]
        by_map.setdefault(int(ds.data["map_idx"][idx]), idx)
    map_ids = sorted(by_map)
    for i in range(num_probes):
        tk = tasks[i % len(tasks)]
        idx = tk["task"]
        mi = int(ds.data["map_idx"][idx])
        m = ds[idx]["grid"]
        true_maps.append(m)
        shift_maps.append(torch.roll(m, shifts=(5, 5), dims=(-2, -1)))
        alt_map = next((mid for mid in map_ids if mid != mi), None)
        other_maps.append(ds[by_map[alt_map]]["grid"] if alt_map is not None
                          else torch.zeros_like(m))
    true_maps = torch.stack(true_maps)
    shift_maps = torch.stack(shift_maps)
    other_maps = torch.stack(other_maps)

    base = model(x, t, s, g, true_maps)
    out = {}
    for name, alt in (("shift", shift_maps), ("other", other_maps)):
        diff = (model(x, t, s, g, alt) - base).pow(2).mean().sqrt()
        out[name] = float(diff)
    blank = torch.zeros_like(true_maps)
    out["blank"] = float((model(x, t, s, g, blank) - base).pow(2).mean().sqrt())
    out["base_rms"] = float(base.pow(2).mean().sqrt())
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="F2 新旧架构公平对照")
    ap.add_argument("--old", required=True, help="旧架构 checkpoint（index 条件）")
    ap.add_argument("--new", required=True, help="新架构 checkpoint（grid_sample 条件）")
    ap.add_argument("--data", default=str(ROOT / "data" / "dataset"))
    ap.add_argument("--per-kind", type=int, default=5, help="每类地图取几个任务（5×4=20）")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20261103)
    args = ap.parse_args()

    run_dir = new_run_dir("F2_architecture_ab", "f2_ab")
    logger = get_logger("fhd.f2", run_dir / "logs" / "f2.log")
    device = torch.device("cpu")
    log_section(logger, "F2 新旧地图条件架构对照")

    ds = TrajectoryDataset(Path(args.data), "test", grid_size=64, robot_radius_m=0.15,
                           clearance_margin_m=0.03, width_m=10.0, height_m=10.0)
    tasks = pick_tasks(ds, args.per_kind, args.seed)
    logger.info("固定任务 %d 个（每类 %d 个），测试集共 %d 条轨迹 / %d 张地图",
                len(tasks), args.per_kind, len(ds), len(ds.maps))

    v_max, dt_s = 1.5, 0.4
    results = {}
    manifests = {}
    for name, ckpt_path, cfg_path in (
        ("old_index", args.old, "configs/audit_arch_index.yaml"),
        ("new_grid", args.new, "configs/audit_arch_grid.yaml"),
    ):
        cfg = load_config(ROOT / cfg_path)
        set_seed(args.seed)
        model, ckpt = load_model(cfg, Path(ckpt_path), device)
        schedule = DDPMSchedule(int(cfg["diffusion"]["num_train_timesteps"]),
                                kind=str(cfg["diffusion"]["beta_schedule"]), device=device)
        t0 = time.time()
        rows = evaluate_model(model, schedule, ds, tasks, args.samples, args.steps,
                              v_max, dt_s, device, args.seed)
        results[name] = rows
        manifests[name] = {
            "checkpoint": str(ckpt_path),
            "sha256": sha256_file(Path(ckpt_path)),
            "step": int(ckpt.get("step", -1)),
            "best_val": float(ckpt.get("best_val", float("nan"))),
            "map_conditioning": cfg["diffusion"].get("map_conditioning"),
            "map_spatial_res": cfg["diffusion"].get("map_spatial_res"),
            "data": str(args.data),
            "eval_seconds": round(time.time() - t0, 1),
        }
        sens = map_sensitivity(model, ds, tasks, device)
        manifests[name]["map_sensitivity"] = sens
        logger.info("%s：成功 %.3f；地图敏感度 shift=%.4f blank=%.4f other=%.4f（基线 eps RMS %.3f）",
                    name, float(np.mean([r["success_rate"] for r in rows])),
                    sens["shift"], sens["blank"], sens["other"], sens["base_rms"])

    summaries = {name: summarize(rows, name) for name, rows in results.items()}
    a = np.array([r["success_rate"] for r in results["old_index"]])
    b = np.array([r["success_rate"] for r in results["new_grid"]])
    clusters = np.array([r["task"] for r in results["new_grid"]])
    diff = paired_bootstrap_diff(b, a, clusters, n_boot=2000)

    for name, s in summaries.items():
        logger.info("%s：成功 %.4f，任务级 CI [%.4f, %.4f]，静态安全 %.3f，长度比 %.3f",
                    name, s["success_rate"], s["ci"]["lo"], s["ci"]["hi"],
                    s["static_safe_rate"], s["length_ratio"])
    logger.info("配对差（新−旧）= %+.4f [%+.4f, %+.4f] %s", diff["diff"], diff["lo"], diff["hi"],
                "显著" if diff["significant"] else "不显著")

    # ---------------- 图 ----------------
    kinds = [k for k in KIND_CN if k in summaries["new_grid"]["by_kind"]]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    x = np.arange(len(kinds))
    w = 0.36
    for i, (name, label) in enumerate((("old_index", "旧：压成 1×H"), ("new_grid", "新：grid_sample"))):
        vals = [summaries[name]["by_kind"][k]["success_rate"] for k in kinds]
        axes[0].bar(x + (i - 0.5) * w, vals, w, label=label,
                    color="#d62728" if i == 0 else "#1f77b4")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([KIND_CN[k] for k in kinds], fontsize=8)
    axes[0].set_ylabel("成功率（静态安全 + 速度合格）")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("按地图类型的成功率", fontsize=10)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3, axis="y")

    for i, (name, label) in enumerate((("old_index", "旧"), ("new_grid", "新"))):
        s = summaries[name]
        axes[1].errorbar([i], [s["success_rate"]],
                         yerr=[[s["success_rate"] - s["ci"]["lo"]], [s["ci"]["hi"] - s["success_rate"]]],
                         fmt="o", ms=9, capsize=6, color="#d62728" if i == 0 else "#1f77b4",
                         label=f"{label}：{s['success_rate']:.3f}")
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["旧架构", "新架构"])
    axes[1].set_ylabel("总体成功率（任务级 95% CI）")
    axes[1].set_title(f"配对差（新−旧）= {diff['diff']:+.4f}\n[ {diff['lo']:+.4f}, {diff['hi']:+.4f} ]",
                      fontsize=10)
    axes[1].grid(alpha=0.3, axis="y")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(run_dir / "plots" / "f2_success_by_kind.png", dpi=130)
    plt.close(fig)

    # 逐任务散点
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(a, b, s=55)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="y = x")
    ax.set_xlabel("旧架构：逐任务成功率")
    ax.set_ylabel("新架构：逐任务成功率")
    ax.set_title(f"同一任务上的配对对比（{len(a)} 个任务）", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(run_dir / "plots" / "f2_paired_scatter.png", dpi=130)
    plt.close(fig)

    # 样例图：每类地图各一个任务，画真实路径 + 两个模型各 4 条候选
    fig, axes = plt.subplots(len(kinds), 3, figsize=(13.5, 4.3 * len(kinds)))
    axes = np.atleast_2d(axes)
    def detour_ratio(tk):
        p = ds.data["paths"][tk["task"]]
        gl = float(torch.linalg.vector_norm(p[1:] - p[:-1], dim=-1).sum())
        sd = float(torch.linalg.vector_norm(p[-1] - p[0]))
        return gl / max(sd, 1e-6)

    for r, kind in enumerate(kinds):
        task = max([t for t in tasks if t["kind"] == kind], key=detour_ratio)
        idx = task["task"]
        item = ds[idx]
        world = ds.maps[int(item["map_idx"])]
        for c, (name, title) in enumerate((("old_index", "旧架构 4 条候选"),
                                           ("new_grid", "新架构 4 条候选"))):
            ax = axes[r, c + 1]
            cfg = load_config(ROOT / ("configs/audit_arch_index.yaml" if name == "old_index"
                                      else "configs/audit_arch_grid.yaml"))
            model, _ = load_model(cfg, Path(manifests[name]["checkpoint"]), device)
            schedule = DDPMSchedule(int(cfg["diffusion"]["num_train_timesteps"]),
                                    kind=str(cfg["diffusion"]["beta_schedule"]), device=device)
            with torch.no_grad():
                start = item["start"].unsqueeze(0).expand(4, -1)
                goal = item["goal"].unsqueeze(0).expand(4, -1)
                grid = item["grid"].unsqueeze(0).expand(4, -1, -1, -1)
                s = ddim_sample(model, schedule, start, goal, grid, num_steps=args.steps,
                                generator=torch.Generator().manual_seed(idx))
                paths = ds.denormalise(s.float()).numpy()
            draw_map(ax, world)
            gt = ds.data["paths"][idx].numpy()
            ax.plot(gt[:, 0], gt[:, 1], "k--", lw=1.2, label="真实路径")
            safe, _ = trajectory_is_static_safe(torch.as_tensor(paths), world.obstacles(),
                                                world.robot_radius_m, world.clearance_margin_m)
            for k in range(paths.shape[0]):
                ax.plot(paths[k, :, 0], paths[k, :, 1], lw=1.1,
                        color="#2ca02c" if bool(safe[k]) else "#d62728")
            ax.set_title(f"{title}（绿=安全，红=碰撞）", fontsize=9)
            ax.legend(fontsize=7)
        ax = axes[r, 0]
        draw_map(ax, world)
        gt = ds.data["paths"][idx].numpy()
        ax.plot(gt[:, 0], gt[:, 1], "-", color="#333", lw=1.6)
        ax.set_title(f"{KIND_CN[kind]}：真实路径（任务 {idx}）", fontsize=9)
    fig.suptitle(f"F2：同一任务下两种地图条件的生成结果（旧 step {manifests['old_index']['step']} / "
                 f"新 step {manifests['new_grid']['step']}）", fontsize=12)
    fig.tight_layout()
    fig.savefig(run_dir / "plots" / "f2_samples_by_kind.png", dpi=130)
    plt.close(fig)

    # ---------------- 报告 ----------------
    report = StageReport("F2_architecture_ab", run_dir, "F2 地图条件架构 A/B 对照报告", {})
    info = system_info()
    report.metrics_from_dict({k: v for k, v in info.items() if not isinstance(v, (dict, list))},
                             group="environment")
    for name, s in summaries.items():
        report.metric(f"{name}/success_rate", s["success_rate"], group="summary")
        report.metric(f"{name}/ci_lo", s["ci"]["lo"], group="summary")
        report.metric(f"{name}/ci_hi", s["ci"]["hi"], group="summary")
        report.metric(f"{name}/static_safe_rate", s["static_safe_rate"], group="summary")
        report.metric(f"{name}/length_ratio", s["length_ratio"], group="summary")
        for kind, kd in s["by_kind"].items():
            report.metric(f"{name}/{kind}/success_rate", kd["success_rate"], group="by_kind")
    report.metric("paired_diff_new_minus_old", diff["diff"], group="paired")
    report.metric("paired_diff_lo", diff["lo"], group="paired")
    report.metric("paired_diff_hi", diff["hi"], group="paired")
    report.metric("paired_significant", diff["significant"], group="paired")
    report.table("f2_per_task", [{"model": n, **r} for n, rows in results.items() for r in rows])
    (run_dir / "experiment_manifest.json").write_text(
        json.dumps({"tasks": tasks, "models": manifests, "eval": {
            "n_samples": args.samples, "ddim_steps": args.steps, "seed": args.seed,
            "data": str(args.data)}, "paired": diff}, ensure_ascii=False, indent=2),
        encoding="utf-8")

    lines = ["| 模型 | 任务数 | 成功率 | 任务级 95% CI | 静态安全率 | 长度比 |", "|---|---:|---:|---|---:|---:|"]
    for name, s in summaries.items():
        label = "旧（index）" if name == "old_index" else "新（grid_sample）"
        lines.append(f"| {label} | {s['n_tasks']} | {s['success_rate']:.3f} | "
                     f"[{s['ci']['lo']:.3f}, {s['ci']['hi']:.3f}] | "
                     f"{s['static_safe_rate']:.3f} | {s['length_ratio']:.3f} |")
    kind_lines = ["| 地图类型 | 旧架构 | 新架构 | 差值 |", "|---|---:|---:|---:|"]
    for k in kinds:
        o = summaries["old_index"]["by_kind"].get(k, {}).get("success_rate", float("nan"))
        n_ = summaries["new_grid"]["by_kind"].get(k, {}).get("success_rate", float("nan"))
        kind_lines.append(f"| {KIND_CN[k]} | {o:.3f} | {n_:.3f} | {n_ - o:+.3f} |")

    report.add_section(
        "## 对照设置\n\n"
        "- 训练数据：`data/dataset`（medium：4,819 条训练轨迹 / 50 张地图）\n"
        "- 两模型唯一差异：`diffusion.map_conditioning`（interface `index` vs `grid_sample`）\n"
        "- 其余完全相同：网络宽度、`map_spatial_res=16`、batch 96、8,000 步、随机种子 20261103\n"
        f"- 评估：{len(tasks)} 个固定任务（每类地图 5 个），每任务 {args.samples} 条候选，"
        f"{args.steps} 步 DDIM，两模型共用同一任务与同一噪声种子\n"
        "- 成功率 = 静态连续几何安全 **且** 每段 ≤0.6 m\n"
    )
    report.add_section("## 结果\n\n" + "\n".join(lines) + "\n\n" + "\n".join(kind_lines) + "\n")
    sens_lines = ["| 模型 | 基线 eps RMS | 平移地图 | 全空地图 | 换另一张图 |", "|---|---:|---:|---:|---:|"]
    for name, mf in manifests.items():
        label = "旧（index）" if name == "old_index" else "新（grid_sample）"
        sv = mf.get("map_sensitivity", {})
        sens_lines.append(f"| {label} | {sv.get('base_rms', float('nan')):.4f} | "
                          f"{sv.get('shift', float('nan')):.4f} | {sv.get('blank', float('nan')):.4f} | "
                          f"{sv.get('other', float('nan')):.4f} |")
    report.add_section(
        "## 地图敏感度探针（机制是否真的接通）\n\n"
        "同一批轨迹与时间步，只替换输入地图，测量 ε 预测的 RMS 变化：\n\n"
        + "\n".join(sens_lines) +
        "\n\n解读：「平移地图」一列最能反映是否按位置查询——旧实现把二维特征压成 1×H，"
        "对空间平移几乎无感；新实现按路点坐标采样，平移后特征整体错位，敏感度显著更高。"
        "「全空地图」一列反映模型是否依赖“有障碍”这一全局信息。\n"
    )
    report.add_section(
        "## 配对比较\n\n"
        f"- 新 − 旧 = **{diff['diff']:+.4f}**，任务级配对 bootstrap 95% CI "
        f"[{diff['lo']:+.4f}, {diff['hi']:+.4f}]，"
        f"{'显著' if diff['significant'] else '**不显著**'}\n"
        f"- 任务数 {len(a)}；两模型分别训练到 step "
        f"{manifests['old_index']['step']} / {manifests['new_grid']['step']}，"
        f"仍属小规模对照，**不作为最终结论**\n"
    )
    report.write_report()
    report.write_metrics_csv()
    report.write_status("completed", {}, {"paired_diff": diff["diff"]})
    report.sync_to_docs()

    import shutil
    audit = ROOT / "docs" / "audit_fix"
    (audit / "figures").mkdir(parents=True, exist_ok=True)
    (audit / "tables").mkdir(parents=True, exist_ok=True)
    shutil.copy2(run_dir / "report.md", audit / "02_map_conditioning_fix.md")
    for png in (run_dir / "plots").glob("*.png"):
        shutil.copy2(png, audit / "figures" / png.name)
    shutil.copy2(run_dir / "tables" / "f2_per_task.csv", audit / "tables" / "f2_per_task.csv")

    print(json.dumps({"old": summaries["old_index"]["success_rate"],
                      "new": summaries["new_grid"]["success_rate"],
                      "diff": diff}, ensure_ascii=False, indent=2, default=float))
    print(f"运行目录：{run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
