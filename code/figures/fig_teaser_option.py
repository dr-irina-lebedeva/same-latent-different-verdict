"""Fig. 1 (v23 on: source 6, zoom box 80 16): same layout as fig_teaser.py for any run index.
Usage: python fig_teaser_option.py <run_dir> <v4_run_dir> <png_dir with {idx:04d}_{dec}.png> <out_dir> <idx> [x0 y0] (zoom box on the 448 px tile, size 112)"""
import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
run, v4, png, out = map(Path, sys.argv[1:5]); IDX = int(sys.argv[5]); out.mkdir(parents=True, exist_ok=True)
DECS = ["orig", "ftmse", "taesd", "cons"]; NAMES = ["original VAE", "ft-MSE", "TAESD", "Consistency Dec."]
tiles = [np.asarray(Image.open(png / f"{IDX:04d}_{d}.png" if (png / f"{IDX:04d}_{d}.png").exists() else png / "gen" / d / f"{IDX:04d}.png").convert("RGB")) for d in DECS]
B = 112
if len(sys.argv) > 7: x0, y0 = int(sys.argv[6]), int(sys.argv[7])
else:  # most detailed 112 px region of the original decoding (local gradient energy)
    g = tiles[0].mean(2); e = np.abs(np.diff(g, axis=0))[:, :-1] + np.abs(np.diff(g, axis=1))[:-1, :]
    best = max(((e[y:y + B, x:x + B].sum(), x, y) for y in range(0, 448 - B, 16) for x in range(0, 448 - B, 16) if not (x < 190 and y > 250)))
    _, x0, y0 = best
box = (x0, y0, x0 + B, y0 + B)
df = pd.concat([pd.read_csv(run / f) for f in ["scores.csv", "aeroblade_scores.csv", "extra_scores.csv", "v3_base_scores.csv", "v3_new_scores.csv"]] + [pd.read_csv(v4 / "bfree_scores_v4.csv")])
df = df.drop_duplicates(["group", "decoder", "idx", "detector"], keep="last")
DETS = [("aeroblade", "AEROBLADE"), ("npr", "NPR"), ("univfd", "UnivFD"), ("commfor", "Comm.-For."), ("bfree", "B-Free")]
dec = {}
for det, _ in DETS:
    t = np.quantile(df[(df.detector == det) & (df.group == "calib")].score.values, 0.99)
    for d in DECS:
        dec[det, d] = float(df[(df.detector == det) & (df.group == "gen") & (df.decoder == d) & (df.idx == IDX)].score.iloc[0]) > t
plt.rcParams.update({"font.size": 7.5, "font.family": "serif"})
fig = plt.figure(figsize=(6.875, 2.9))
gs = fig.add_gridspec(2, 4, height_ratios=[1, 0.5], hspace=0.04, wspace=0.06)
for c in range(4):
    ax = fig.add_subplot(gs[0, c]); ax.imshow(tiles[c]); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(NAMES[c], fontsize=7.5, pad=2)
    ax.add_patch(plt.Rectangle((box[0], box[1]), B, B, fill=False, ec="#eb6834", lw=0.9))
    ins = ax.inset_axes([0.0, 0.0, 0.42, 0.42]); ins.imshow(tiles[c][box[1]:box[3], box[0]:box[2]], interpolation="nearest"); ins.set_xticks([]); ins.set_yticks([])
    for s in ins.spines.values(): s.set_edgecolor("#eb6834"); s.set_linewidth(0.9)
    tx = fig.add_subplot(gs[1, c]); tx.axis("off")
    for r, (det, lab) in enumerate(DETS):
        y = 0.9 - r * (0.82 / (len(DETS) - 1)); ok = dec[det, DECS[c]]
        tx.text(0.12, y, lab, fontsize=7, color="#52514e", va="center")
        tx.text(0.88, y, ("flagged" if ok else "MISSED"), fontsize=7, va="center", ha="right", color="#0b0b0b", fontweight="bold" if not ok else "normal",
                bbox=dict(boxstyle="round,pad=0.15", fc="#fde7dd" if not ok else "#e3eefb", ec="none"))
fig.text(0.5, 0.995, "Same SD1.5 latent $z$, four public decoders  →  detector decisions at a nominal 1% FPR threshold", ha="center", va="top", fontsize=8)
fig.subplots_adjust(left=0.005, right=0.995, top=0.885, bottom=0.0)
fig.savefig(out / f"fig_teaser_idx{IDX:04d}.pdf", dpi=300); fig.savefig(out / f"fig_teaser_idx{IDX:04d}.png", dpi=200)
print(IDX, box, {f"{k[0]}/{k[1]}": int(v) for k, v in dec.items()})
