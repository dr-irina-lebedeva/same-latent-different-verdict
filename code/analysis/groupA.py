"""Group A analyses from saved per-image scores (no GPU). SD1.5 main run, N=300 (probes: 150 held-out sources).
Thresholds: 99th/95th percentile of the 1,000 independent calibration photos. CIs: 2,000 bootstrap draws that resample
calibration photos and source ids independently (paired: one source id keeps all its decoder outcomes)."""
import sys, numpy as np, pandas as pd
from pathlib import Path
# usage: python groupA.py scores/sd15_main_n300 scores/sd15_main_n300_v4 [out_dir]
run = Path(sys.argv[1]); v4 = Path(sys.argv[2])
out = Path(sys.argv[3]) if len(sys.argv) > 3 else Path(".")
df = pd.concat([pd.read_csv(run / f) for f in ["scores.csv", "aeroblade_scores.csv", "extra_scores.csv", "v3_base_scores.csv", "v3_new_scores.csv"]] + [pd.read_csv(v4 / "bfree_scores_v4.csv")])
df = df[df.group.isin(["real", "real_enc", "gen", "calib"])].drop_duplicates(["group", "decoder", "idx", "detector"], keep="last")
D = ["orig", "ftmse", "taesd", "cons"]
DETS = ["aeroblade", "npr", "univfd", "probe_clip", "probe_dino", "commfor", "bfree"]
def S(det, g, d):
    x = df[(df.detector == det) & (df.group == g) & (df.decoder == d)].set_index("idx").score.sort_index()
    if det.startswith("probe_") and g != "calib": x = x[x.index >= 150]
    return x
B = 2000; rng = np.random.default_rng(0)
def ci(v): return np.percentile(v, [2.5, 97.5])
rows = []
for det in DETS:
    cal = S(det, "calib", "none").values
    G = pd.concat({d: S(det, "gen", d) for d in D}, axis=1).dropna()
    R = pd.concat({d: S(det, "real_enc", d) for d in D}, axis=1).dropna()
    U = S(det, "real", "none")
    ids = G.index.values; assert (R.index.values == ids).all() and (U.index.values == ids).all(), det
    g, r, u = G.values, R.values, U.values
    for q in [0.99, 0.95]:
        def stats(cal_, g_, r_, u_):
            t = np.quantile(cal_, q); dg = g_ > t
            return np.r_[(r_ > t).mean(0), (u_ > t).mean(), dg.mean(0), dg.all(1).mean(), (dg.any(1) & ~dg.all(1)).mean(), (dg[:, 0] != dg[:, 3]).mean()]
        est = stats(cal, g, r, u)
        bs = np.array([stats(cal[rng.integers(0, len(cal), len(cal))], *(lambda i: (g[i], r[i], u[i]))(rng.integers(0, len(ids), len(ids)))) for _ in range(B)])
        lo, hi = np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0)
        names = [f"rt_{d}" for d in D] + ["real_untouched"] + [f"tpr_{d}" for d in D] + ["worst_tpr", "flip_rate", "flip_orig_cons"]
        for n_, e, l, h in zip(names, est, lo, hi): rows.append((det, f"{int(round((1-q)*100))}%", n_, e, l, h, len(ids)))
res = pd.DataFrame(rows, columns=["detector", "fpr", "metric", "est", "lo", "hi", "n"])
res.to_csv(out / "groupA_roundtrip_flip.csv", index=False)
# ---- AEROBLADE AE ensembles
A = ["aeroblade", "aero_ftmse", "aero_taesd"]
chk = (S("aeroblade", "gen", "orig") - S("aero_orig", "gen", "orig")).abs().max() if "aero_orig" in set(df.detector) else None
cal = pd.concat({a: S(a, "calib", "none") for a in A}, axis=1).dropna()
Gs = {d: pd.concat({a: S(a, "gen", d) for a in A}, axis=1).dropna() for d in D}
Rs = {d: pd.concat({a: S(a, "real_enc", d) for a in A}, axis=1).dropna() for d in D}
mu, sd = cal.mean(), cal.std()
def ens(X, kind): return X.max(1) if kind == "raw" else ((X - mu) / sd).max(1)
erows = []
for kind in ["raw", "z"]:
    for q in [0.99, 0.95]:
        def st(ci_, gi_):
            t = np.quantile(ens(cal.iloc[ci_], kind), q)
            return np.r_[[ (ens(Gs[d].iloc[gi_], kind) > t).mean() for d in D], [(ens(Rs[d].iloc[gi_], kind) > t).mean() for d in D], t]
        allc, allg = np.arange(len(cal)), np.arange(300)
        est = st(allc, allg)
        bs = np.array([st(rng.integers(0, len(cal), len(cal)), rng.integers(0, 300, 300)) for _ in range(500)])
        lo, hi = np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0)
        for n_, e, l, h in zip([f"tpr_{d}" for d in D] + [f"rt_{d}" for d in D] + ["threshold"], est, lo, hi):
            erows.append((kind, f"{int(round((1-q)*100))}%", n_, e, l, h))
E = pd.DataFrame(erows, columns=["ensemble", "fpr", "metric", "est", "lo", "hi"]); E.to_csv(out / "groupA_ae_ensemble.csv", index=False)
# argmin AE for calibration photos and TAESD images, both ensembles
for kind in ["raw", "z"]:
    Z = cal if kind == "raw" else (cal - mu) / sd
    Zt = Gs["taesd"] if kind == "raw" else (Gs["taesd"] - mu) / sd
    print(kind, "calib argmax AE:", Z.idxmax(1).value_counts().to_dict(), "| TAESD gen argmax AE:", Zt.idxmax(1).value_counts().to_dict())
print("calib mean/sd per AE:", mu.round(4).to_dict(), sd.round(4).to_dict(), "aero_orig==aeroblade maxdiff:", chk)
pd.set_option("display.width", 200)
print(res[res.fpr == "1%"].pivot_table(index="detector", columns="metric", values="est").round(3).to_string())
print(E.round(3).to_string())
