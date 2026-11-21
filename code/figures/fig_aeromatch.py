"""Matched-autoencoder AEROBLADE heatmap: TPR at nominal 1% FPR (from report_v3.md, v3 scores).
Usage: python fig_aeromatch.py <run_dir> <out_dir> [e2_scores.csv]  (v19: adds the Consistency-autoencoder column from E2)"""
import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
run, out = Path(sys.argv[1]), Path(sys.argv[2])
df = pd.concat([pd.read_csv(run / f) for f in ["aeroblade_scores.csv", "v3_base_scores.csv", "v3_new_scores.csv"]])
if len(sys.argv) > 3:
    df = pd.concat([df, pd.read_csv(sys.argv[3])[["group", "decoder", "idx", "detector", "score"]]])
df = df.drop_duplicates(["group", "decoder", "idx", "detector"], keep="last")
S = lambda det, g, d: df[(df.detector == det) & (df.group == g) & (df.decoder == d)].sort_values("idx").score.values
AE = ["aero_orig", "aero_ftmse", "aero_taesd"]
def scores(ae, g, d):
    if ae != "min": return S(ae, g, d)
    return np.max(np.stack([S(a, g, d) for a in AE]), 0)  # scores are -LPIPS: max = min distance
GEN = ["orig", "ftmse", "taesd", "cons"]; COLS = AE + ["aero_cons", "min"]
M = np.zeros((4, 5))
for j, ae in enumerate(COLS):
    t = np.quantile(scores(ae, "calib", "none"), 0.99)
    for i, d in enumerate(GEN):
        M[i, j] = np.mean(scores(ae, "gen", d) > t)
print(np.round(M, 2))
plt.rcParams.update({"font.size": 7.5, "font.family": "serif"})
cmap = LinearSegmentedColormap.from_list("b", ["#f4f8fd", "#9cc3ef", "#2a78d6", "#0d3f7f"])
fig, ax = plt.subplots(figsize=(3.25, 2.05))
ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
for i in range(4):
    for j in range(5):
        ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=7.5,
                color="white" if M[i, j] > 0.55 else "#0b0b0b", fontweight="bold" if i == j else "normal")
for k in range(4):
    ax.add_patch(plt.Rectangle((k - 0.5, k - 0.5), 1, 1, fill=False, ec="#eb6834", lw=1.8, zorder=5, clip_on=False))
ax.set_xticks(range(5)); ax.set_xticklabels(["orig", "ft-MSE", "TAESD", "Cons.", "best of 3"])
ax.set_yticks(range(4)); ax.set_yticklabels(["orig", "ft-MSE", "TAESD", "Consistency"])
ax.set_xlabel("AEROBLADE reconstruction autoencoder"); ax.set_ylabel("generating decoder")
ax.xaxis.set_label_position("top"); ax.xaxis.tick_top()
for s in ax.spines.values(): s.set_visible(False)
ax.set_xticks(np.arange(-.5, 5), minor=True); ax.set_yticks(np.arange(-.5, 4), minor=True)
ax.grid(which="minor", color="white", lw=2, zorder=1); ax.tick_params(which="both", length=0)
fig.tight_layout(pad=0.2)
fig.savefig(out / "fig_aeromatch.pdf"); fig.savefig(out / "fig_aeromatch.png", dpi=200)
