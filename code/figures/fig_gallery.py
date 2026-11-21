"""Qualitative gallery from the saved lossless PNGs of the add-on run (bitwise identical to the scored images):
two SD1.5 latents, a 112x112 crop at native resolution for each decoder, and each detector's decision at its
nominal-1%-FPR threshold. Usage: python fig_gallery.py <v3_run_dir> <v4_run_dir> <png_dir> <out_dir>"""
import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
run, v4, png, out = map(Path, sys.argv[1:5])
IDX = [int(v) for v in sys.argv[5].split(",")] if len(sys.argv) > 5 else [3, 10]; C = 112
NAME_OUT = sys.argv[6] if len(sys.argv) > 6 else "fig_gallery"
DECS = ["orig", "ftmse", "taesd", "cons"]; NAMES = ["original VAE", "ft-MSE", "TAESD", "Consistency"]
df = pd.concat([pd.read_csv(run / f) for f in ["scores.csv", "aeroblade_scores.csv", "extra_scores.csv", "v3_base_scores.csv", "v3_new_scores.csv"]]
               + [pd.read_csv(v4 / "bfree_scores_v4.csv")]).drop_duplicates(["group", "decoder", "idx", "detector"], keep="last")
DETS = [("aeroblade", "A"), ("npr", "N"), ("univfd", "U"), ("commfor", "C"), ("bfree", "B")]
thr = {d: np.quantile(df[(df.detector == d) & (df.group == "calib")].score.values, 0.99) for d, _ in DETS}
def flag(det, d, i):
    return float(df[(df.detector == det) & (df.group == "gen") & (df.decoder == d) & (df.idx == i)].score.iloc[0]) > thr[det]
def crop_box(im):  # most textured 112x112 window of the original-VAE image (grey-level variance of the Laplacian)
    g = np.asarray(im.convert("L"), float); L = np.abs(4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:])
    best, bx = -1, (0, 0)
    for y in range(40, 448 - C - 40, 16):
        for x in range(40, 448 - C - 40, 16):
            v = L[y:y + C, x:x + C].mean()
            if v > best: best, bx = v, (x, y)
    return bx
plt.rcParams.update({"font.size": 7, "font.family": "serif"})
fig = plt.figure(figsize=(3.3, 0.83 * len(IDX)))
gs = fig.add_gridspec(2 * len(IDX), 5, height_ratios=[1, 0.19] * len(IDX), hspace=0.04, wspace=0.05)
for r, i in enumerate(IDX):
    ims = {d: Image.open(png / "gen" / d / f"{i:04d}.png").convert("RGB") for d in DECS}
    x0, y0 = crop_box(ims["orig"])
    ax = fig.add_subplot(gs[2 * r, 0]); ax.imshow(ims["orig"]); ax.set_xticks([]); ax.set_yticks([])
    ax.add_patch(plt.Rectangle((x0, y0), C, C, fill=False, ec="#eb6834", lw=0.9))
    if r == 0: ax.set_title("latent (orig.)", fontsize=6.5, pad=2)
    for c, d in enumerate(DECS):
        ax = fig.add_subplot(gs[2 * r, c + 1]); ax.imshow(np.asarray(ims[d])[y0:y0 + C, x0:x0 + C], interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0: ax.set_title(NAMES[c], fontsize=6.5, pad=2)
        tx = fig.add_subplot(gs[2 * r + 1, c + 1]); tx.axis("off"); tx.set_xlim(0, 5); tx.set_ylim(0, 1)
        for k, (det, lab) in enumerate(DETS):
            ok = flag(det, d, i)
            tx.add_patch(plt.Rectangle((k + 0.06, 0.08), 0.88, 0.84, fc="#d6e6f8" if ok else "#c8481c", ec="none"))
            tx.text(k + 0.5, 0.48, lab, ha="center", va="center", fontsize=5.6, color="#0b0b0b" if ok else "white",
                    fontweight="normal" if ok else "bold")
    tx = fig.add_subplot(gs[2 * r + 1, 0]); tx.axis("off")
    tx.text(0.5, 0.48, "flagged / " + r"$\bf{missed}$", ha="center", va="center", fontsize=5.2, color="#52514e")
fig.subplots_adjust(left=0.005, right=0.995, top=1 - 0.15 / (0.83 * len(IDX)), bottom=0.01)
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / f"{NAME_OUT}.pdf", dpi=600); fig.savefig(out / f"{NAME_OUT}.png", dpi=300)
