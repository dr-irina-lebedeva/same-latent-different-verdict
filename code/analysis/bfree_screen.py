"""Held-out test of the generator-free screen on B-Free (not used when the screen was fixed). No GPU.
The line is fitted on the same 10 original cells as in the paper; B-Free cells are predicted from the drop in
the flag rate of VAE-reconstructed real photos. Usage: python bfree_screen.py <v3_run_dir> <v4_run_dir>"""
import sys, numpy as np, pandas as pd
from pathlib import Path
v3, v4 = Path(sys.argv[1]), Path(sys.argv[2])
OLD = ["npr", "univfd", "probe_clip", "probe_dino", "aeroblade"]; ALTS = ["ftmse", "taesd", "cons"]; HALF = 150
parts = [pd.read_csv(v3 / f) for f in ["scores.csv", "aeroblade_scores.csv", "extra_scores.csv", "v3_base_scores.csv", "v3_new_scores.csv"] if (v3 / f).exists()]
parts.append(pd.read_csv(v4 / "bfree_scores_v4.csv"))
df = pd.concat(parts); df = df[df.group.isin(["real", "real_enc", "gen", "calib"])].drop_duplicates(["group", "decoder", "idx", "detector"], keep="last")
A = {}
for (det, g, d), x in df.groupby(["detector", "group", "decoder"]):
    x = x.sort_values("idx")
    if det.startswith("probe_") and g != "calib": x = x[x.idx >= HALF]
    A[(det, g, d)] = x.score.values
out = []
for q in [0.99, 0.95]:
    rows = []
    for det in OLD + ["bfree"]:
        t = np.quantile(A[(det, "calib", "none")], q)
        for alt in ALTS:
            v = {g: 100 * (np.mean(A[(det, g, "orig")] > t) - np.mean(A[(det, g, alt)] > t)) for g in ["real_enc", "gen"]}
            rows.append(dict(det=det, alt=alt, x=v["real_enc"], y=v["gen"], old=det in OLD and alt != "cons"))
    c = pd.DataFrame(rows); o = c[c.old]; b, a0 = np.polyfit(o.x, o.y, 1)
    n = c[c.det == "bfree"].copy(); n["pred"] = a0 + b * n.x; n["q"] = q
    out.append(n)
    print(f"FPR {100*(1-q):.0f}%: line y = {a0:.2f} + {b:.3f} x")
    print(n[["alt", "x", "y", "pred"]].round(1).to_string(index=False))
    print(f"  MAE screen {np.mean(np.abs(n.pred - n.y)):.1f} vs no-change {np.mean(np.abs(n.y)):.1f}; large-loss (>=20) flags: {int((n.pred >= 20).sum())}, true large: {int((n.y >= 20).sum())}")
pd.concat(out).to_csv(v4 / "bfree_screen.csv", index=False)
