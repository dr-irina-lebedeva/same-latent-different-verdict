"""Main figure: (a) TPR at 1% FPR after a decoder swap, (b) ReconScreen prediction vs true TPR loss.
Reads existing scores only. Usage: python fig_story.py <run_dir> <out_dir> [<v4_run_dir> for B-Free]"""
import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
run, out = Path(sys.argv[1]), Path(sys.argv[2])
parts = [pd.read_csv(run / f) for f in ["scores.csv", "aeroblade_scores.csv", "extra_scores.csv", "v3_base_scores.csv", "v3_new_scores.csv"]]
v4 = Path(sys.argv[3]) if len(sys.argv) > 3 else None
if v4 is not None:
    parts.append(pd.read_csv(v4 / "bfree_scores_v4.csv"))
df = pd.concat(parts); df = df[df.group.isin(["real", "real_enc", "gen", "calib"])].drop_duplicates(["group", "decoder", "idx", "detector"], keep="last")
def S(det, g, d):
    x = df[(df.detector == det) & (df.group == g) & (df.decoder == d)].sort_values("idx")
    if det.startswith("probe_") and g != "calib": x = x[x.idx >= 150]
    return x.score.values
plt.rcParams.update({"font.size": 8.2, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e"})
COL = {"orig": "#2a78d6", "ftmse": "#eb6834", "taesd": "#1baf7a", "cons": "#6250d6"}
HAT = {"orig": "", "ftmse": "////", "taesd": "....", "cons": "xxxx"}
NAME = {"orig": "original VAE", "ftmse": "ft-MSE", "taesd": "TAESD", "cons": "Consistency Dec."}
DETS = [("aeroblade", "AERO-\nBLADE"), ("npr", "NPR"), ("univfd", "Univ-\nFD"), ("probe_clip", "CLIP\nprobe"), ("commfor", "Comm.-\nFor.")] + ([("bfree", "B-Free")] if v4 is not None else [])
fig, (a, b) = plt.subplots(1, 2, figsize=(6.875, 2.5), gridspec_kw=dict(width_ratios=[1.0, 1.0]))
w = 0.2
for i, (det, lab) in enumerate(DETS):
    t = np.quantile(S(det, "calib", "none"), 0.99)
    for j, d in enumerate(["orig", "ftmse", "taesd", "cons"]):
        v = np.mean(S(det, "gen", d) > t)
        a.bar(i + (j - 1.5) * w, v, w * 0.9, color=COL[d], hatch=HAT[d], edgecolor="white", linewidth=0.4,
              label=NAME[d] if i == 0 else None)
a.set_xticks(range(len(DETS))); a.set_xticklabels([l for _, l in DETS]); a.set_ylim(0, 1.05)
a.set_ylabel("TPR at 1% FPR"); a.yaxis.grid(True, color="#e6e5e0", lw=0.5); a.set_axisbelow(True)
a.legend(ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.22), handlelength=1.2, columnspacing=0.8, fontsize=7.5)
a.set_title("(a) Same latents, only the decoder changes", loc="left", fontsize=8.8, pad=22)
# (b) diagnostic
OLD = ["npr", "univfd", "probe_clip", "probe_dino", "aeroblade"]
cells = []
for det in OLD + ["commfor", "aero_ftmse", "aero_taesd"]:
    t = np.quantile(S(det, "calib", "none"), 0.99)
    for d in ["ftmse", "taesd", "cons"]:
        x = 100 * (np.mean(S(det, "real_enc", "orig") > t) - np.mean(S(det, "real_enc", d) > t))
        y = 100 * (np.mean(S(det, "gen", "orig") > t) - np.mean(S(det, "gen", d) > t))
        cells.append((det, d, x, y, det in OLD and d != "cons"))
c = pd.DataFrame(cells, columns=["det", "alt", "x", "y", "old"])
o = c[c.old]; k, a0 = np.polyfit(o.x, o.y, 1)
xs = np.linspace(-100, 100, 10)
b.axhspan(20, 100, color="#f3f2ee", zorder=0); b.axvspan(-100, 100, ymin=0, ymax=0, alpha=0)
b.plot(xs, a0 + k * xs, color="#52514e", lw=1, zorder=1, label="line fit (10 cells)")
b.scatter(o.x, o.y, s=14, facecolor="white", edgecolor="#52514e", lw=0.9, zorder=2, label="fit cells")
n = c[~c.old]
b.scatter(n.x, n.y, s=16, color="#2a78d6", edgecolor="white", lw=0.5, zorder=3, label="held-out")
FA = []   # false alarms: predicted >= 20 pts, true < 20 pts
for _, r in n.iterrows():
    if a0 + k * r.x >= 20 and r.y < 20: FA.append((r.det, r.x, r.y, a0 + k * r.x))
if v4 is not None:   # B-Free: evaluated after the screen was fixed (second held-out detector)
    tb = np.quantile(S("bfree", "calib", "none"), 0.99); bf = []
    for d in ["ftmse", "taesd", "cons"]:
        x = 100 * (np.mean(S("bfree", "real_enc", "orig") > tb) - np.mean(S("bfree", "real_enc", d) > tb))
        y = 100 * (np.mean(S("bfree", "gen", "orig") > tb) - np.mean(S("bfree", "gen", d) > tb))
        bf.append((d, x, y))
        if a0 + k * x >= 20 and y < 20: FA.append(("bfree", x, y, a0 + k * x))
    b.scatter([r[1] for r in bf], [r[2] for r in bf], s=26, marker="D", color="#6250d6", edgecolor="white", lw=0.5,
              zorder=5, label="B-Free (later)")
NM = {"commfor": "Comm.-For.", "probe_clip": "CLIP probe", "bfree": "B-Free"}
for det, x, y, p in FA:
    b.scatter([x], [y], s=50, facecolor="none", edgecolor="#eb6834", lw=1.1, zorder=6)
# zoomed inset so the three false alarms are separately readable
ax = b.inset_axes([0.50, 0.03, 0.48, 0.38])
ax.plot(xs, a0 + k * xs, color="#52514e", lw=0.8)
ax.scatter(n.x, n.y, s=14, color="#2a78d6", edgecolor="white", lw=0.4)
if v4 is not None: ax.scatter([r[1] for r in bf], [r[2] for r in bf], s=20, marker="D", color="#6250d6", edgecolor="white", lw=0.4, zorder=5)
ORDER = {"probe_clip": 1, "commfor": 2, "bfree": 3}
NUMOFF = {"probe_clip": (-2.0, 5.0), "commfor": (-2.0, -5.0), "bfree": (2.0, 5.0)}
key = []
for det, x, y, p in sorted(FA, key=lambda r: ORDER[r[0]]):
    ax.scatter([x], [y], s=46, facecolor="none", edgecolor="#eb6834", lw=1.0, zorder=6)
    ax.text(x + NUMOFF[det][0], y + NUMOFF[det][1], str(ORDER[det]), fontsize=8, fontweight="bold", color="#eb6834", ha="center", va="center")
    key.append(f"{ORDER[det]}  {NM[det]}/Cons.: pred. {p:.0f}, true {y:.0f}")
print("KEY", key)
ax.set_xlim(12, 42); ax.set_ylim(-14, 26); ax.set_xticks([]); ax.set_yticks([])
ax.set_facecolor("#fbfaf7"); [sp.set_visible(True) for sp in ax.spines.values()]; [sp.set_color("#eb6834") for sp in ax.spines.values()]
b.indicate_inset([12, -14, 30, 40], ax, edgecolor="#eb6834", lw=0.6, alpha=0.9)
b.text(-95, 92, "large loss (≥20 pts)", fontsize=7, color="#52514e")
b.set_xlim(-100, 100); b.set_ylim(-80, 100)
b.set_xlabel("flag-rate drop on reconstructed real photos (pts)"); b.set_ylabel("TPR drop on generated (pts)")
b.legend(frameon=False, loc="upper left", fontsize=7, handlelength=1.0, bbox_to_anchor=(0, 0.92), labelspacing=0.25, handletextpad=0.4)
b.set_title("(b) ReconScreen, 1% FPR", loc="left", fontsize=8.8, pad=22)
fig.tight_layout(pad=0.3, w_pad=1.2)
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "fig_story.pdf"); fig.savefig(out / "fig_story.png", dpi=200)
c.assign(pred=a0 + k * c.x).to_csv(out / "fig_story_cells_1pct.csv", index=False)
print(c.assign(pred=a0 + k * c.x).round(1).to_string())
