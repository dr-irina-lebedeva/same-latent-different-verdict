"""Re-analysis of the real-photo vulnerability diagnostic from EXISTING v3 scores (no new images, no GPU).
Usage: python diag_reanalysis.py <run_dir> <out.md>
Question: does the generator-free screen identify LARGE operational TPR losses (not only mean shifts)?
Reports precision/recall with Wilson CIs, bootstrap CIs over sources+calibration photos, threshold sensitivity,
false alarms, a post-hoc leave-one-group-out over all 24 cells, and Community-Forensics score margins."""
import sys, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

run, out = Path(sys.argv[1]), Path(sys.argv[2])
OLD = ["npr", "univfd", "probe_clip", "probe_dino", "aeroblade"]
DETS = OLD + ["commfor", "aero_ftmse", "aero_taesd"]
FAM = {"aeroblade": "aeroblade", "aero_ftmse": "aeroblade", "aero_taesd": "aeroblade"}
ALTS = ["ftmse", "taesd", "cons"]
parts = [pd.read_csv(run / f) for f in ["scores.csv", "aeroblade_scores.csv", "extra_scores.csv",
                                         "v3_base_scores.csv", "v3_new_scores.csv"] if (run / f).exists()]
df = pd.concat(parts); df = df[df.group.isin(["real", "real_enc", "gen", "calib"])]
df = df.drop_duplicates(["group", "decoder", "idx", "detector"], keep="last")
HALF = 150
A = {}  # (det, group, decoder) -> array indexed by position of sorted idx
for (det, g, d), x in df.groupby(["detector", "group", "decoder"]):
    x = x.sort_values("idx")
    if det.startswith("probe_") and g != "calib":
        x = x[x.idx >= HALF]
    A[(det, g, d)] = x.score.values

def cells_for(q, src_idx=None, cal_idx=None):
    rows = []
    for det in DETS:
        cal = A.get((det, "calib", "none"))
        if cal is None: continue
        c = cal if cal_idx is None else cal[cal_idx[: len(cal)] % len(cal)]
        t = np.quantile(c, q)
        for alt in ALTS:
            v = {}
            for g in ["real_enc", "gen"]:
                o, x = A.get((det, g, "orig")), A.get((det, g, alt))
                if o is None or x is None or len(o) != len(x): v = None; break
                k = np.arange(len(o)) if src_idx is None else src_idx[: len(o)] % len(o)
                v[g] = 100 * (np.mean(o[k] > t) - np.mean(x[k] > t))
            if v is None: continue
            rows.append(dict(det=det, alt=alt, x=v["real_enc"], y=v["gen"], old=det in OLD and alt != "cons"))
    return pd.DataFrame(rows)

def fit_predict(c):
    o = c[c.old]; b, a0 = np.polyfit(o.x, o.y, 1)
    n = c[~c.old].copy(); n["p"] = a0 + b * n.x; return n, (a0, b)

def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n; cen = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0, cen - h), min(1, cen + h))

def pr(y, p, thr):
    big, flag = y >= thr, p >= thr
    tp = int((big & flag).sum()); return tp, int(flag.sum()), int(big.sum())

L = []; P = L.append
P("# Diagnostic re-analysis from existing v3 scores (14 Nov 2026)\n")
P("No new images or GPU. Unit of analysis = detector-decoder cell. 'Large loss' = TPR drop on generated images "
  ">= threshold (pts) at a fixed threshold calibrated on 1,000 independent photos. The screen predicts the TPR drop "
  "from the drop in the flag rate of VAE-reconstructed real photos, via a line fit on the 10 original cells only "
  "(planned), then applied to the 14 unseen cells.\n")
rng = np.random.default_rng(0)
summary = {}
for q in [0.99, 0.95]:
    lab = f"{int(round(100*(1-q)))}% FPR"
    c = cells_for(q); n, (a0, b) = fit_predict(c)
    P(f"## {lab}\n\nFit (10 original cells): y = {a0:+.1f} + {b:.2f} x.\n")
    P("| detector | decoder | real-photo flag drop | predicted TPR drop | true TPR drop | flagged (>=20) | true large (>=20) |")
    P("|---|---|---|---|---|---|---|")
    for _, r in n.sort_values("y", ascending=False).iterrows():
        P(f"| {r.det} | {r.alt} | {r.x:+.0f} | {r.p:+.0f} | {r.y:+.0f} | {'FLAG' if r.p >= 20 else ''} | {'yes' if r.y >= 20 else ''} |")
    P("")
    P("| large-loss threshold | true large | flagged | TP | recall [95% Wilson] | precision [95% Wilson] | false alarms |")
    P("|---|---|---|---|---|---|---|")
    for thr in [10, 15, 20, 30]:
        tp, nf, nb = pr(n.y.values, n.p.values, thr)
        fa = ", ".join(f"{r.det}/{r.alt} (pred {r.p:.0f}, true {r.y:.0f})" for _, r in n[(n.p >= thr) & (n.y < thr)].iterrows()) or "none"
        rl, rh = wilson(tp, nb); pl, ph = wilson(tp, nf)
        P(f"| {thr} | {nb} | {nf} | {tp} | {tp}/{nb} [{rl:.2f}, {rh:.2f}] | {tp}/{nf} [{pl:.2f}, {ph:.2f}] | {fa} |")
    mae, mz = np.mean(np.abs(n.p - n.y)), np.mean(np.abs(n.y))
    # bootstrap over evaluation sources AND calibration photos; refit each time
    B = []
    for _ in range(2000):
        si, ci = rng.integers(0, 300, 300), rng.integers(0, 1000, 1000)
        cb = cells_for(q, si, ci)
        nb_, _ = fit_predict(cb)
        e, z = np.mean(np.abs(nb_.p - nb_.y)), np.mean(np.abs(nb_.y))
        tp, nf, nbig = pr(nb_.y.values, nb_.p.values, 20)
        B.append((e, z, 1 - e / z, tp / nbig if nbig else np.nan, tp / nf if nf else np.nan, spearmanr(nb_.p, nb_.y)[0]))
    B = np.array(B); lo, hi = np.nanpercentile(B, 2.5, 0), np.nanpercentile(B, 97.5, 0)
    rho = spearmanr(n.p, n.y)[0]
    P(f"\nUnseen-cell MAE {mae:.1f} pts [{lo[0]:.1f}, {hi[0]:.1f}] vs zero-change {mz:.1f} [{lo[1]:.1f}, {hi[1]:.1f}]; "
      f"error reduction {1-mae/mz:.0%} [{lo[2]:.0%}, {hi[2]:.0%}]; Spearman {rho:.2f} [{lo[5]:.2f}, {hi[5]:.2f}]. "
      f"Bootstrap (2,000) resamples evaluation sources and calibration photos and refits the line; it captures "
      f"sampling noise, NOT the choice of cells. Bootstrap recall@20 [{lo[3]:.2f}, {hi[3]:.2f}], precision@20 [{lo[4]:.2f}, {hi[4]:.2f}].\n")
    summary[lab] = dict(mae=mae, mz=mz, red=1 - mae / mz, ci=(lo[2], hi[2]), rho=rho)
    # post-hoc: leave-one-group-out over ALL 24 cells
    P(f"**Post-hoc (not planned in advance): leave-one-group-out over all {len(c)} cells.**\n")
    for name, key in [("detector family (AEROBLADE variants grouped)", lambda r: FAM.get(r.det, r.det)),
                      ("decoder", lambda r: r.alt)]:
        c2 = c.copy(); c2["g"] = c2.apply(key, axis=1); preds = []
        for g in c2.g.unique():
            tr, te = c2[c2.g != g], c2[c2.g == g]
            bb, aa = np.polyfit(tr.x, tr.y, 1); preds += list(zip(te.det, te.alt, aa + bb * te.x, te.y))
        pp = pd.DataFrame(preds, columns=["det", "alt", "p", "y"])
        e, z = np.mean(np.abs(pp.p - pp.y)), np.mean(np.abs(pp.y))
        tp, nf, nb = pr(pp.y.values, pp.p.values, 20); rl, rh = wilson(tp, nb); pl, ph = wilson(tp, nf)
        fa = ", ".join(f"{r.det}/{r.alt}" for _, r in pp[(pp.p >= 20) & (pp.y < 20)].iterrows()) or "none"
        miss = ", ".join(f"{r.det}/{r.alt} (pred {r.p:.0f}, true {r.y:.0f})" for _, r in pp[(pp.p < 20) & (pp.y >= 20)].iterrows()) or "none"
        P(f"- Leave one {name} out: MAE {e:.1f} vs zero {z:.1f} pts ({1-e/z:.0%} lower); Spearman {spearmanr(pp.p, pp.y)[0]:.2f}; "
          f"large losses (>=20): recall {tp}/{nb} [{rl:.2f}, {rh:.2f}], precision {tp}/{nf} [{pl:.2f}, {ph:.2f}]; "
          f"false alarms: {fa}; misses: {miss}.")
    P("")

# Community-Forensics margins (ceiling check from existing scores)
P("## Community-Forensics: how far are generated images from the threshold?\n")
cal = A[("commfor", "calib", "none")]; real = A[("commfor", "real", "none")]; g0 = A[("commfor", "gen", "orig")]
sd = np.std(np.concatenate([real, g0]))
P("| decoder | min gen score margin above tau@1% (sd) | 1st pct margin (sd) | median margin (sd) | mean shift vs orig (sd) | shift needed to drop TPR@1% below 0.90 (sd) |")
P("|---|---|---|---|---|---|")
for q in [0.99]:
    t = np.quantile(cal, q)
    for d in ["orig", "ftmse", "taesd", "cons"]:
        g = A[("commfor", "gen", d)]; m = (g - t) / sd
        P(f"| {d} | {m.min():.2f} | {np.percentile(m,1):.2f} | {np.median(m):.2f} | {np.mean(g-g0)/sd:+.2f} | {np.percentile(m,10):.2f} |")
re = {d: np.mean(A[("commfor", "real_enc", d)] > np.quantile(cal, 0.99)) for d in ["orig", "ftmse", "taesd", "cons"]}
P(f"\nReconstructed-real flag rate at tau@1%: " + ", ".join(f"{k} {v:.2f}" for k, v in re.items()) + ".\n")
P("Interpretation: a uniform downward shift of this size in sd units would be needed before any decoder costs "
  "Community-Forensics 10 points of TPR at 1% FPR; the largest observed decoder shift (Consistency) is the 'mean shift' "
  "column. This quantifies the ceiling; it is not a test on harder images.\n")
out.write_text("\n".join(L)); print("\n".join(L))
