"""Supplement v33: round-trip flag rates (encoder + decoder, no denoising) at the 1% and 5% calibrated thresholds,
with 95% percentile bootstrap intervals (2,000 replicates; calibration photographs and evaluation sources resampled
independently; threshold recomputed in every replicate). Reads saved scores only.
Usage: python v33_roundtrip_ci.py <v3_run_dir> <v4_run_dir>"""
import sys, numpy as np, pandas as pd
from pathlib import Path
run, v4 = Path(sys.argv[1]), Path(sys.argv[2])
parts = [pd.read_csv(run / f) for f in ["scores.csv", "aeroblade_scores.csv", "extra_scores.csv", "v3_base_scores.csv", "v3_new_scores.csv"]]
parts.append(pd.read_csv(v4 / "bfree_scores_v4.csv"))
df = pd.concat(parts); df = df.drop_duplicates(["group", "decoder", "idx", "detector"], keep="last")
def S(det, g, d):
    x = df[(df.detector == det) & (df.group == g) & (df.decoder == d)].sort_values("idx")
    if det.startswith("probe_") and g != "calib": x = x[x.idx >= 150]
    return x.score.values
rng = np.random.default_rng(0)
NAME = {"aeroblade": "AEROBLADE", "npr": "NPR", "univfd": "UnivFD", "probe_clip": "CLIP probe", "probe_dino": "DINO probe", "commfor": "Comm.-For.", "bfree": "B-Free"}
rows = []
for det in NAME:
    c = S(det, "calib", "none"); u = S(det, "real", "none")
    for q, lab in ((0.99, "1%"), (0.95, "5%")):
        out = [NAME[det], lab]
        t = np.quantile(c, q); out.append(f"{np.mean(u > t):.2f}")
        for d in ["orig", "ftmse", "taesd", "cons"]:
            r = S(det, "real_enc", d)
            est = np.mean(r > t); bs = []
            for _ in range(2000):
                tb = np.quantile(rng.choice(c, len(c)), q); bs.append(np.mean(rng.choice(r, len(r)) > tb))
            lo, hi = np.percentile(bs, [2.5, 97.5])
            out.append(f"{100*est:.0f} [{100*lo:.0f}, {100*hi:.0f}]")
        rows.append(out)
pd.DataFrame(rows, columns=["detector", "FPR", "untouched", "orig", "ftmse", "taesd", "cons"]).to_csv(sys.stdout, index=False)
