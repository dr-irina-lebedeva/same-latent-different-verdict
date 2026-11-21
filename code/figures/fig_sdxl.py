"""SDXL LatentLock figure (supplement): TPR at a nominal 1% FPR per decoder, 95% bootstrap CIs.
Reads only the saved run scores. Usage: python fig_sdxl.py <sdxl_run_dir> <out_dir>
Threshold: 99th percentile of the 1,000 fresh calibration photographs (group 'calib'), as in dod_sdxl.py.
CIs: 2,000 bootstrap replicates; sources and calibration photographs resampled independently, threshold recomputed."""
import sys, json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
run, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(run / "sdxl_scores.csv")
def S(det, g, d):
    return df[(df.detector == det) & (df.group == g) & (df.decoder == d)].sort_values("idx").score.values
DECS = ["orig", "fp16fix", "taesdxl"]
COL = {"orig": "#2a78d6", "fp16fix": "#eb6834", "taesdxl": "#1baf7a"}
HAT = {"orig": "", "fp16fix": "////", "taesdxl": "...."}
NAME = {"orig": "original SDXL VAE", "fp16fix": "FP16-fix (near-identity control)", "taesdxl": "TAESDXL (distilled)"}
DETS = [("aero_sdxl", "AEROBLADE\n(SDXL AE)"), ("npr", "NPR"), ("probe_clip", "CLIP probe\n(SD1.5-trained)"),
        ("commfor", "Comm.-For."), ("bfree", "B-Free"), ("univfd", "UnivFD")]
rng = np.random.default_rng(0); B = 2000
rows = []
for det, _ in DETS:
    cal = S(det, "calib", "none"); G = np.stack([S(det, "gen", d) for d in DECS])  # 3 x 300, paired by source
    t = np.quantile(cal, 0.99); est = (G > t).mean(1)
    bs = np.empty((B, 3))
    for b in range(B):
        tb = np.quantile(cal[rng.integers(0, len(cal), len(cal))], 0.99)
        bs[b] = (G[:, rng.integers(0, G.shape[1], G.shape[1])] > tb).mean(1)
    lo, hi = np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0)
    for j, d in enumerate(DECS):
        rows.append(dict(detector=det, decoder=d, tpr=est[j], lo=lo[j], hi=hi[j], n_gen=G.shape[1], n_cal=len(cal)))
R = pd.DataFrame(rows); R.to_csv(out / "fig_sdxl_values.csv", index=False)
plt.rcParams.update({"font.size": 8.0, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "pdf.fonttype": 42})
fig, a = plt.subplots(figsize=(6.875, 2.3)); w = 0.26
for i, (det, lab) in enumerate(DETS):
    expl = det == "univfd"; x0 = i + (0.45 if expl else 0)
    for j, d in enumerate(DECS):
        r = R[(R.detector == det) & (R.decoder == d)].iloc[0]
        x = x0 + (j - 1) * w
        a.bar(x, r.tpr, w * 0.92, color=COL[d], hatch=HAT[d], edgecolor="white", linewidth=0.4, alpha=0.35 if expl else 1.0,
              label=NAME[d] if i == 0 else None)
        a.errorbar(x, r.tpr, yerr=[[r.tpr - r.lo], [r.hi - r.tpr]], fmt="none", ecolor="#52514e", elinewidth=0.7, capsize=1.6,
                   alpha=0.5 if expl else 1.0)
a.axvline(len(DETS) - 1 - 0.5 + 0.22, color="#a3a29c", lw=0.7, ls="--")
a.text(len(DETS) - 1 + 0.45, 1.13, "exploratory:\nTPR(orig) < 0.5", ha="center", va="bottom", fontsize=6.8, color="#52514e")
a.text(2.0, 1.13, "usable detectors (TPR(orig) $\\geq$ 0.5; pre-specified rule)", ha="center", va="bottom", fontsize=6.8, color="#52514e")
xt = [i + (0.45 if d == "univfd" else 0) for i, (d, _) in enumerate(DETS)]
a.set_xticks(xt); a.set_xticklabels([l for _, l in DETS]); a.get_xticklabels()[-1].set_color("#8a8984")
a.set_ylim(0, 1.12); a.set_yticks([0, .25, .5, .75, 1]); a.set_ylabel("TPR at nominal 1% FPR")
a.yaxis.grid(True, color="#e6e5e0", lw=0.5); a.set_axisbelow(True)
a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.26), ncol=3, frameon=False, fontsize=7.2, handlelength=1.6)
fig.tight_layout(pad=0.3)
fig.savefig(out / "fig_sdxl.pdf", metadata={"Creator": None, "Producer": None, "CreationDate": None})
fig.savefig(out / "fig_sdxl.png", dpi=300)
print(R.round(3).to_string(index=False))
