"""第二版方案 · S2：地图分辨率 / CNN 特征尺度消融。

五组配置（只改地图输入分辨率与 CNN 输出特征分辨率，其余完全一致）：

| 编号 | 地图输入 | CNN 特征 |
|---|---:|---:|
| A | 64×64   | 16×16 |
| B | 128×128 | 32×32 |
| C | 128×128 | 64×64 |
| D | 128×128 | 16×16 |
| E | 64×64   | 32×32 |

第一轮先跑 A/B/C（快速筛选）；只有当 B 或 C 明显更好时才跑 D/E 以区分
「输入分辨率」与「特征空间尺度」两个因素。

评估：同一份 S1 测试集（按地图划分）、同一批任务与噪声，8 条候选/任务，
按地图类型分别统计有效率。

用法：
  python scripts/revised_s2_ablation.py --data data/revised_s1 --configs A,B,C --steps 8000
  python scripts/revised_s2_ablation.py --data data/revised_s1 --configs D,E --eval-only
"""

from __future__ import annotations

import argparse
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
from fhd.diffusion.ddim import ddim_sample  # noqa: E402
from fhd.diffusion.schedule import DDPMSchedule  # noqa: E402
from fhd.diffusion.train import Trainer, build_model  # noqa: E402
from fhd.evaluation.reporting import StageReport  # noqa: E402
from fhd.evaluation.statistics import cluster_bootstrap_ci  # noqa: E402
from fhd.evaluation.visualizations import draw_map  # noqa: E402
from fhd.geometry.collisions import check_speed_limit, trajectory_is_static_safe  # noqa: E402
from fhd.geometry.paths import path_length  # noqa: E402
from fhd.maps.simple_scenes import KIND_LABELS  # noqa: E402
from fhd.utils.config import load_config  # noqa: E402
from fhd.utils.logging import get_logger, log_section  # noqa: E402
from fhd.utils.runtime import new_run_dir, system_info  # noqa: E402
from fhd.utils.seed import resolve_device, set_seed  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ABLATIONS = {
    "A": ("s2_A_64_16.yaml", 64, 16),
    "B": ("s2_B_128_32.yaml", 128, 32),
    "C": ("s2_C_128_64.yaml", 128, 64),
    "D": ("s2_D_128_16.yaml", 128, 16),
    "E": ("s2_E_64_32.yaml", 64, 32),
}


def pick_tasks(ds, per_kind: int, seed: int = 7) -> list[dict]:
    by_kind = defaultdict(list)
    for idx in range(len(ds)):
        mi = int(ds.data["map_idx"][idx])
        by_kind[ds.maps[mi].kind].append(idx)
    rng = np.random.default_rng(seed)
    picked = []
    for kind in KIND_LABELS:
        pool = by_kind.get(kind, [])
        if not pool:
            continue
        sel = rng.choice(pool, size=min(per_kind, len(pool)), replace=False)
        for idx in sorted(int(i) for i in sel):
            picked.append({"task": idx, "kind": kind})
    return picked


@torch.no_grad()
def evaluate(model, schedule, ds, tasks, n_samples, steps, v_max, dt_s, device, seed):
    gen = torch.Generator(device=device.type).manual_seed(seed)
    rows = []
    for t in tasks:
        idx = t["task"]
        item = ds[idx]
        start = item["start"].unsqueeze(0).expand(n_samples, -1).to(device)
        goal = item["goal"].unsqueeze(0).expand(n_samples, -1).to(device)
        grid = item["grid"].unsqueeze(0).expand(n_samples, -1, -1, -1).to(device)
        samples = ddim_sample(model, schedule, start, goal, grid, num_steps=steps,
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
            "task_success_at_8": float(bool(ok.any())),
            "static_safe_rate": float(safe.float().mean()),
            "speed_ok_rate": float(speed_ok.float().mean()),
            "min_clearance_m": float(clr.min()),
            "length_ratio": float((lengths / max(gt_len, 1e-6)).mean()),
            "n_samples": n_samples,
        })
    return rows


def summarize(rows, tag):
    vals = np.array([r["success_rate"] for r in rows])
    clusters = np.array([r["task"] for r in rows])
    ci = cluster_bootstrap_ci(vals, clusters, n_boot=2000)
    by_kind = {}
    for kind in KIND_LABELS:
        sel = [r for r in rows if r["kind"] == kind]
        if not sel:
            continue
        by_kind[kind] = {
            "n_tasks": len(sel),
            "success_rate": float(np.mean([r["success_rate"] for r in sel])),
            "task_success_at_8": float(np.mean([r["task_success_at_8"] for r in sel])),
            "static_safe_rate": float(np.mean([r["static_safe_rate"] for r in sel])),
            "speed_ok_rate": float(np.mean([r["speed_ok_rate"] for r in sel])),
            "min_clearance_m": float(np.mean([r["min_clearance_m"] for r in sel])),
            "length_ratio": float(np.mean([r["length_ratio"] for r in sel])),
        }
    return {"tag": tag, "n_tasks": len(rows), "success_rate": float(vals.mean()),
            "task_success_at_8": float(np.mean([r["task_success_at_8"] for r in rows])),
            "ci": ci,
            "static_safe_rate": float(np.mean([r["static_safe_rate"] for r in rows])),
            "speed_ok_rate": float(np.mean([r["speed_ok_rate"] for r in rows])),
            "length_ratio": float(np.mean([r["length_ratio"] for r in rows])),
            "by_kind": by_kind}


def main() -> int:
    ap = argparse.ArgumentParser(description="S2 地图分辨率消融")
    ap.add_argument("--data", default=str(ROOT / "data" / "revised_s1"))
    ap.add_argument("--configs", default="A,B,C")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--per-kind", type=int, default=20, help="每类地图取多少任务评估")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--ckpt", action="append", default=[], help="tag=path 形式，eval-only 用")
    args = ap.parse_args()

    device = resolve_device("auto")
    run_dir = new_run_dir("S2_map_resolution", "s2_ablation")
    logger = get_logger("fhd.s2r", run_dir / "logs" / "s2.log")
    log_section(logger, "S2 地图分辨率 / 特征尺度消融")
    logger.info("数据=%s  配置=%s  训练步数=%d  设备=%s", args.data, args.configs, args.steps, device)

    tags = [x.strip().upper() for x in args.configs.split(",") if x.strip()]
    results = {}
    ckpt_map = {}
    for tag in tags:
        yml, in_res, feat_res = ABLATIONS[tag]
        cfg = load_config(ROOT / "configs" / yml)
        set_seed(int(cfg["seed"]))
        sub = run_dir / f"variant_{tag}"
        sub.mkdir(parents=True, exist_ok=True)
        if args.eval_only:
            given = dict(x.split("=", 1) for x in args.ckpt)
            ckpt_path = Path(given[tag])
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model = build_model(cfg, device)
            model.load_state_dict(ckpt.get("ema") or ckpt["model"], strict=True)
            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)
            train_info = {"step": int(ckpt.get("step", -1)),
                          "best_val": float(ckpt.get("best_val", float("nan")))}
        else:
            logger.info("[%s] 开始训练：输入 %d，特征 %d", tag, in_res, feat_res)
            trainer = Trainer(cfg, sub, Path(args.data), device, resume=None)
            t0 = time.time()
            train_info = trainer.train(args.steps, eval_every=1000, checkpoint_every=2000,
                                       sample_every=4000, log_every=50)
            train_info["wall_time_s"] = round(time.time() - t0, 1)
            model = trainer.model
            ckpt_path = sub / "checkpoints" / "best.pt"
            logger.info("[%s] 训练完成：step=%s best_val=%.5f", tag, train_info["step"],
                        train_info["best_val"])
        ckpt_map[tag] = str(ckpt_path)

        schedule = DDPMSchedule(int(cfg["diffusion"]["num_train_timesteps"]),
                                kind=str(cfg["diffusion"]["beta_schedule"]), device=device)
        ds = TrajectoryDataset(Path(args.data), "test",
                               grid_size=int(cfg["diffusion"]["map_input_res"]),
                               robot_radius_m=float(cfg["world"]["robot_radius_m"]),
                               clearance_margin_m=float(cfg["world"]["clearance_margin_m"]),
                               width_m=float(cfg["world"]["width_m"]),
                               height_m=float(cfg["world"]["height_m"]))
        if tag == tags[0]:
            tasks = pick_tasks(ds, args.per_kind)
            logger.info("评估任务 %d 个（每类 %d）", len(tasks), args.per_kind)
        rows = evaluate(model, schedule, ds, tasks, args.samples,
                        int(cfg["diffusion"]["ddim_sampling_steps"]),
                        float(cfg["trajectory"]["max_speed_mps"]),
                        float(cfg["trajectory"]["dt_s"]), device, int(cfg["seed"]))
        s = summarize(rows, tag)
        s["in_res"], s["feat_res"] = in_res, feat_res
        s["train"] = train_info
        s["ckpt"] = str(ckpt_path)
        results[tag] = {"summary": s, "rows": rows}
        logger.info("[%s] 成功率 %.3f（TaskSuccess@8 %.3f），静态 %.3f，长度比 %.3f",
                    tag, s["success_rate"], s["task_success_at_8"], s["static_safe_rate"],
                    s["length_ratio"])

    # ---------------- 报告 ----------------
    report = StageReport("S2_map_resolution", run_dir, "S2 地图分辨率消融报告", {})
    info = system_info()
    report.metrics_from_dict({k: v for k, v in info.items() if not isinstance(v, (dict, list))},
                             group="environment")
    for tag, r in results.items():
        s = r["summary"]
        report.metric(f"{tag}/in_res", s["in_res"], group="ablation")
        report.metric(f"{tag}/feat_res", s["feat_res"], group="ablation")
        report.metric(f"{tag}/success_rate", s["success_rate"], group="ablation")
        report.metric(f"{tag}/success_ci_lo", s["ci"]["lo"], group="ablation")
        report.metric(f"{tag}/success_ci_hi", s["ci"]["hi"], group="ablation")
        report.metric(f"{tag}/task_success_at_8", s["task_success_at_8"], group="ablation")
        report.metric(f"{tag}/static_safe_rate", s["static_safe_rate"], group="ablation")
        report.metric(f"{tag}/best_val", s["train"].get("best_val", float("nan")), group="ablation")
        for kind, kd in s["by_kind"].items():
            report.metric(f"{tag}/{kind}/success_rate", kd["success_rate"], group="by_kind")
    report.table("s2_per_task", [{"tag": tag, **r} for tag, v in results.items() for r in v["rows"]])
    (run_dir / "manifest.json").write_text(json.dumps({
        "data": args.data, "steps": args.steps, "samples": args.samples,
        "per_kind": args.per_kind, "checkpoints": ckpt_map,
        "results": {k: v["summary"] for k, v in results.items()},
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["| 编号 | 地图输入 | CNN 特征 | 训练步数 | 有效率 | 95% CI | TaskSuccess@8 | 静态安全率 | 长度比 |",
             "|---|---:|---:|---:|---:|---|---:|---:|---:|"]
    order = sorted(results, key=lambda k: results[k]["summary"]["success_rate"], reverse=True)
    for tag in order:
        s = results[tag]["summary"]
        lines.append(f"| {tag} | {s['in_res']} | {s['feat_res']} | {s['train'].get('step', '-')} | "
                     f"{s['success_rate']:.3f} | [{s['ci']['lo']:.3f}, {s['ci']['hi']:.3f}] | "
                     f"{s['task_success_at_8']:.3f} | {s['static_safe_rate']:.3f} | "
                     f"{s['length_ratio']:.3f} |")
    kind_lines = ["| 编号 | " + " | ".join(KIND_LABELS[k] for k in KIND_LABELS) + " |",
                  "|---|" + "---:|" * len(KIND_LABELS)]
    for tag in order:
        s = results[tag]["summary"]
        vals = [f"{s['by_kind'].get(k, {}).get('success_rate', float('nan')):.3f}"
                for k in KIND_LABELS]
        kind_lines.append(f"| {tag} | " + " | ".join(vals) + " |")

    report.add_section(
        "## 设置\n\n"
        f"- 数据：`{args.data}`（S1 重建的三类简单场景，按地图划分 train/val/test）\n"
        f"- 每组地图输入 `in_res` 从**连续几何重新栅格化**（不是把 64 插值成 128）\n"
        f"- 网络结构固定为「两个全分辨率卷积 + n 次 stride-2 下采样」，通道预算相同\n"
        f"- 每组训练 {args.steps} 步，同种子、同 batch；评估用同一批任务与同一噪声\n"
        "- 成功率 = 静态连续几何安全 **且** 每段 ≤0.6 m\n"
    )
    report.add_section("## 总体结果\n\n" + "\n".join(lines) + "\n")
    report.add_section("## 按地图类型\n\n" + "\n".join(kind_lines) + "\n")

    best = order[0]
    report.add_section(
        "## 结论\n\n"
        f"- 最优配置：**{best}**（{results[best]['summary']['in_res']}→"
        f"{results[best]['summary']['feat_res']}），有效率 "
        f"{results[best]['summary']['success_rate']:.3f}\n"
        f"- 这是 {args.steps} 步的快速筛选结果，**不作为最终结论**；"
        "选定架构后由 S3 用完整预算重训\n"
    )
    report.write_report()
    report.write_metrics_csv()
    report.write_status("completed", {}, {"best": best})
    report.sync_to_docs()

    import shutil
    docs = ROOT / "docs" / "revised"
    (docs / "figures").mkdir(parents=True, exist_ok=True)
    shutil.copy2(run_dir / "report.md", docs / "02_S2_resolution.md")
    for png in (run_dir / "plots").glob("*.png"):
        shutil.copy2(png, docs / "figures" / png.name)
    print(json.dumps({k: round(v["summary"]["success_rate"], 4) for k, v in results.items()},
                     ensure_ascii=False))
    print(f"运行目录：{run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
