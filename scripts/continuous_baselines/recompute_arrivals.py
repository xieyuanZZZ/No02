"""Recompute completion at the first actual all-arrived state from saved NPZ."""
import csv, json, sys
from pathlib import Path
import numpy as np

src = Path(sys.argv[1])
rows = [json.loads(x) for x in (src / "per_run.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
for r in rows:
    f = src / "raw" / f"{r['method']}_{r['task_id']}_p{r['p']}_s{r['seed']}.npz"
    if not f.exists():
        r["completion_s"] = None
        continue
    d = np.load(f)
    t, pos, vel, paths = d["times"], d["positions"], d["velocities"], d["paths"]
    ok = (np.linalg.norm(pos - paths[:, -1], axis=2) < 0.25).all(axis=1)
    ok &= (np.linalg.norm(vel, axis=2) < 0.25).all(axis=1)
    idx = np.where(ok)[0]
    r["completion_s"] = float(t[idx[0]]) if len(idx) else float(t[-1])
    r["all_arrived"] = bool(len(idx))
with (src / "per_run.csv").open("w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=sorted({k for r in rows for k in r}))
    w.writeheader()
    w.writerows(rows)
s = {"stage": "ab_5seed_actual_arrival", "by": {}}
for m in ("ours_frozen", "orca_rvo2", "mpc_cbf"):
    for sc in ("open", "static"):
        sel = [r for r in rows if r["method"] == m and r["scene"] == sc]
        def rate(k):
            v = [bool(r[k]) for r in sel if r.get(k) is not None]
            return float(np.mean(v)) if v else None
        s["by"][f"{m}|{sc}"] = {
            "n": len(sel),
            "safe_rate": rate("safe_completion"),
            "collision": rate("collision"),
            "static_collision": rate("static_collision"),
            "completion_s": float(np.mean([r["completion_s"] for r in sel
                                           if r["completion_s"] is not None])) if sel else None,
        }
(src / "summary.json").write_text(json.dumps(s, indent=2), encoding="utf-8")
print(json.dumps(s, indent=1))
