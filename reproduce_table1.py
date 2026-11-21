"""Reproduce Table 1 of the paper from the stored per-image scores.

Decoder swap on identical SD1.5 latents (N=300): AUROC, TPR at fixed thresholds
for a nominal 1% and 5% FPR (99th / 95th percentile of 1,000 independent
calibration photographs), TPR drop per alternative decoder with 95% bootstrap
intervals at 1% FPR, flip rate and all-decoder TPR.

Usage:  python reproduce_table1.py [--scores scores] [--boot 2000]
Needs only numpy and pandas; runs on a CPU in under a minute.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DECODERS = ["orig", "ftmse", "taesd", "cons"]
DETECTORS = [  # (name in the paper, detector id, file with the main scores)
    ("AEROBLADE (single AE)", "aeroblade", "sd15_main_n300/scores_all.csv"),
    ("NPR", "npr", "sd15_main_n300/scores_all.csv"),
    ("UnivFD", "univfd", "sd15_main_n300/scores_all.csv"),
    ("CLIP probe", "probe_clip", "sd15_main_n300/scores_all.csv"),
    ("DINO probe", "probe_dino", "sd15_main_n300/scores_all.csv"),
    ("Comm.-For.", "commfor", "sd15_main_n300/v3_new_scores.csv"),
    ("B-Free", "bfree", "sd15_main_n300_v4/bfree_scores_v4.csv"),
]
# extra files: calibration photographs and Consistency Decoder scores
EXTRA = ["sd15_main_n300/aeroblade_scores.csv", "sd15_main_n300/extra_scores.csv",
         "sd15_main_n300/v3_base_scores.csv"]
HALF = 150  # linear probes are trained on sources 0-149 and evaluated on 150-299


def load(root):
    files = sorted({f for _, _, f in DETECTORS} | set(EXTRA))
    df = pd.concat([pd.read_csv(root / f)[["group", "decoder", "idx", "detector", "score"]]
                    for f in files])
    return df.drop_duplicates(["group", "decoder", "idx", "detector"])


def get(df, det, group, dec):
    x = df[(df.detector == det) & (df.group == group) & (df.decoder == dec)]
    if det.startswith("probe") and group != "calib":
        x = x[x.idx >= HALF]
    return x.sort_values("idx").score.to_numpy()


def auroc(neg, pos):
    s = np.concatenate([neg, pos])
    r = pd.Series(s).rank().to_numpy()
    return (r[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(neg) * len(pos))


def drop_ci(cal, base, alt, q, n, seed=0):
    """TPR(base) - TPR(alt) at the q-quantile of cal; resample calibration photographs and sources."""
    r = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        t = np.quantile(cal[r.integers(0, len(cal), len(cal))], q)
        k = r.integers(0, len(base), len(base))
        out.append(np.mean(base[k] > t) - np.mean(alt[k] > t))
    return np.percentile(out, 2.5), np.percentile(out, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default=str(Path(__file__).parent / "scores"))
    ap.add_argument("--boot", type=int, default=2000, help="bootstrap replicates (0: no intervals)")
    a = ap.parse_args()
    df = load(Path(a.scores))
    rows = []
    for name, det, _ in DETECTORS:
        cal, real = get(df, det, "calib", "none"), get(df, det, "real", "none")
        gen = {d: get(df, det, "gen", d) for d in DECODERS}
        assert len(cal) == 1000 and all(len(g) == len(real) for g in gen.values()), det
        row = {"detector": name, "AUROC orig": f"{auroc(real, gen['orig']):.3f}"}
        for q, lab in [(0.99, "1%"), (0.95, "5%")]:
            t = np.quantile(cal, q)
            tp = {d: np.mean(g > t) for d, g in gen.items()}
            row[f"TPR orig {lab}"] = f"{tp['orig']:.2f}"
            for d in DECODERS[1:]:
                cell = f"{100 * (tp['orig'] - tp[d]):.0f}"
                if lab == "1%" and a.boot:
                    lo, hi = drop_ci(cal, gen["orig"], gen[d], q, a.boot)
                    cell += f" [{100 * lo:.0f},{100 * hi:.0f}]"
                row[f"drop {d} {lab}"] = cell
            if lab == "1%":
                dec = np.stack([g > t for g in gen.values()])
                row["flip"] = f"{100 * np.mean(dec.any(0) & ~dec.all(0)):.0f}%"
                row["all-dec."] = f"{np.mean(dec.all(0)):.2f}"
                row["FPR real 1%"] = f"{np.mean(real > t):.2f}"
        rows.append(row)
    out = pd.DataFrame(rows).set_index("detector")
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print("Table 1: decoder swap on identical SD1.5 latents (drops in TPR points; N=300, probes 150 held-out sources)\n")
    print(out.to_string())


if __name__ == "__main__":
    main()
