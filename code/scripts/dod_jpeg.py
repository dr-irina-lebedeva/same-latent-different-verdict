"""A1: does decoder-dependent detector reliability survive JPEG? (+ full LPIPS-matching check for M4 v2)

Stages
  quality   LPIPS matching error of the bandswap2 lp_* controls on ALL latents (not only the first 20)
            -> bandswap2_lpmatch.csv  (needs dod_bandswap2.py done; skipped otherwise)
  make      JPEG at quality 95 / 85 / 75 of: untouched reals, gen and real_enc under every decoder,
            and the first --calib-n independent calibration photos
  score     same detectors as the main run (UnivFD, NPR, CLIP/DINO probes, AEROBLADE)
  report    report_jpeg.md: AUROC, TPR at 5% FPR with (a) the clean-calibrated threshold and
            (b) a threshold re-calibrated on JPEG'd independent photos, decoder swing, paired
            decoder effect after JPEG

    python dod_jpeg.py --run runs/n100 2>&1 | tee jpeg.log
(~1-1.5 h on an M4; --no-aeroblade saves about half)
"""
import argparse, io, json, warnings
from pathlib import Path

import numpy as np
from PIL import Image

import dod_mac_pilot as M
import dod_bandswap as B1

warnings.filterwarnings("ignore", message=".*encountered in matmul")
log = M.log
QS = [95, 85, 75]
DECS = ["orig", "ftmse", "taesd"]
GROUPS = ["gen", "real_enc"]


def jpeg(img, q):
    buf = io.BytesIO(); img.convert("RGB").save(buf, "JPEG", quality=q)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def sources(a):
    """(group, decoder, idx, source path) before JPEG."""
    out = [("real", "none", i, a.run / "data" / "real" / f"{i:04d}.png") for i in range(a.n_eff)]
    for g in GROUPS:
        for d in DECS:
            out += [(g, d, i, a.run / "png" / g / d / f"{i:04d}.png") for i in range(a.n_eff)]
    out += [("calib", "none", k, a.run / "data" / "calib" / f"{k:04d}.png") for k in range(a.nc)]
    return out


def images(a):
    return [(f"jpg{q}_{g}", d, i, a.run / "png" / "jpeg" / str(q) / g / d / f"{i:04d}.png")
            for q in QS for g, d, i, _ in sources(a)]


# ------------------------------------------------------------------ quality
def stage_quality(a):
    import pandas as pd
    import dod_bandswap2 as B2
    root = a.run / "png" / "bandswap2"
    if not root.exists():
        log("quality: no bandswap2 images, skipped"); return
    fp = a.run / "bandswap2_lpmatch.csv"
    if fp.exists():
        log("quality: already done"); return
    lp = B2.Lp(a.device, a.fake_models)
    rows = []
    P = a.run / "png"
    for g in B2.GROUPS:
        for alt in B2.ALTS:
            for i in range(a.n_eff):
                A = np.asarray(Image.open(P / g / "orig" / f"{i:04d}.png"))
                Bx = np.asarray(Image.open(P / g / alt / f"{i:04d}.png"))
                t = lp(A, Bx)
                for k in ["blur", "noise", "jpeg"]:
                    x = np.asarray(Image.open(B2.out_dir(a, g, alt, f"lp_{k}") / f"{i:04d}.png"))
                    rows.append(dict(group=g, alt=alt, control=k, idx=i, target=t, achieved=lp(A, x)))
            log(f"quality: {g}/{alt} done")
    del lp; M.free(a.device)
    pd.DataFrame(rows).to_csv(fp, index=False)


# ------------------------------------------------------------------ make
def stage_make(a):
    made = 0
    for q in QS:
        for g, d, i, src in sources(a):
            out = a.run / "png" / "jpeg" / str(q) / g / d / f"{i:04d}.png"
            if out.exists():
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            jpeg(Image.open(src), q).save(out)   # PNG of the decoded JPEG (lossless from here on)
            made += 1
        log(f"jpeg make: q={q} done")
    log(f"jpeg make: {made} new images")


# ------------------------------------------------------------------ score
def stage_score(a):
    import pandas as pd
    fp = a.run / "jpeg_scores.csv"
    imgs = images(a)
    if fp.exists() and len(pd.read_csv(fp)) >= len(imgs) * 4:
        log("jpeg score: already done"); return
    saved = B1.images
    B1.images = lambda _a: imgs          # reuse the band-swap scorer on the JPEG image list
    old = a.run / "bandswap_scores.csv"
    try:
        if old.exists():
            old.rename(a.run / "bandswap_scores.keep.csv")
        B1.stage_score(a)
        (a.run / "bandswap_scores.csv").rename(fp)
    finally:
        if (a.run / "bandswap_scores.keep.csv").exists():
            (a.run / "bandswap_scores.keep.csv").rename(old)
        B1.images = saved


# ------------------------------------------------------------------ report
def stage_report(a):
    import pandas as pd
    parts = [pd.read_csv(a.run / "scores.csv")]
    for f in ["aeroblade_scores.csv", "extra_scores.csv"]:
        if (a.run / f).exists():
            parts.append(pd.read_csv(a.run / f))
    clean = pd.concat(parts)
    J = pd.read_csv(a.run / "jpeg_scores.csv")
    half = a.n_eff // 2
    L = []; P = lambda *s: L.append(" ".join(str(x) for x in s))
    P("# JPEG robustness of decoder-dependent detection (A1)\n")
    P(f"N = {a.n_eff} latents (probes: {a.n_eff - half} held-out). Every image (reals, generated, "
      f"calibration photos) is JPEG-compressed at the same quality. TPR at 5% FPR: 'clean thr' = threshold "
      f"from 1,000 clean independent photos (a detector deployed without knowing about JPEG); 'jpeg thr' = "
      f"threshold re-set on {a.nc} JPEG'd independent photos. swing = max - min TPR over decoders. "
      "Δ = paired decoder effect vs. orig on the same latents, in clean sd units [95% CI].\n")
    lm = a.run / "bandswap2_lpmatch.csv"
    if lm.exists():
        q = pd.read_csv(lm)
        q["err"] = q.achieved - q.target
        q["rel"] = q.err.abs() / q.target.clip(lower=1e-6)
        P(f"## LPIPS matching of the M4 v2 controls (all {a.n_eff} latents)\n")
        P("| control | target mean | achieved mean | mean err | median abs err | 90th pct abs err | median rel err |")
        P("|---|---|---|---|---|---|---|")
        for k, r in q.groupby("control"):
            P(f"| {k} | {r.target.mean():.3f} | {r.achieved.mean():.3f} | {r.err.mean():+.3f} | "
              f"{r.err.abs().median():.3f} | {r.err.abs().quantile(0.9):.3f} | {100 * r.rel.median():.0f}% |")
        P("\nControls use a discrete severity grid, so they cannot hit the target exactly; a negative mean "
          "error means they are slightly milder than the decoder swap.\n")
    for det in sorted(set(J.detector)):
        def S(df, g, dec, held=True):
            x = df[(df.detector == det) & (df.group == g) & (df.decoder == dec)]
            if det.startswith("probe_") and g != "calib" and not g.endswith("_calib") and held:
                x = x[x.idx >= half]
            return x.sort_values("idx").score.values
        cal_clean = S(clean, "calib", "none")
        if len(cal_clean) == 0:
            continue
        sd = np.std(np.concatenate([S(clean, "real", "none"), S(clean, "gen", "orig")])) + 1e-9
        t_clean = np.quantile(cal_clean, 0.95)
        P(f"## `{det}`\n")
        P("| JPEG q | AUROC orig | AUROC ftmse | AUROC taesd | TPR orig (clean thr) | ftmse | taesd | swing | "
          "FPR real | TPR orig (jpeg thr) | ftmse | taesd | swing | FPR real | Δ ftmse | Δ taesd |")
        P("|---|" + "---|" * 15)
        for q in [None] + QS:
            src = clean if q is None else J
            pre = "" if q is None else f"jpg{q}_"
            real = S(src, pre + "real", "none")
            gens = {d: S(src, pre + "gen", d) for d in DECS}
            au = [M.auroc(real, gens[d]) for d in DECS]
            tc = [float(np.mean(gens[d] > t_clean)) for d in DECS]
            if q is None:
                t_j = t_clean
            else:
                t_j = np.quantile(S(J, pre + "calib", "none"), 0.95)
            tj = [float(np.mean(gens[d] > t_j)) for d in DECS]
            dl = [M.boot_ci((gens[d] - gens["orig"]) / sd) for d in ["ftmse", "taesd"]]
            P(f"| {'none' if q is None else q} | " + " | ".join(f"{x:.3f}" for x in au) + " | "
              + " | ".join(f"{x:.2f}" for x in tc) + f" | {max(tc) - min(tc):.2f} | {np.mean(real > t_clean):.2f} | "
              + " | ".join(f"{x:.2f}" for x in tj) + f" | {max(tj) - min(tj):.2f} | {np.mean(real > t_j):.2f} | "
              + " | ".join(f"{m:+.2f} [{lo:+.2f}, {hi:+.2f}]" for m, lo, hi in dl) + " |")
        P("")
    P("How to read: if Δ keeps its sign and the swing stays large after JPEG, decoder-dependent reliability "
      "survives ordinary sharing. 'FPR real' far above 0.05 means JPEG alone breaks the clean threshold, "
      "which is a known effect and not about decoders; the 'jpeg thr' columns remove it.\n")
    (a.run / "report_jpeg.md").write_text("\n".join(L))
    log(f"report: wrote {a.run / 'report_jpeg.md'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=Path("cache"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--calib-n", type=int, default=300, help="JPEG'd calibration photos per quality")
    ap.add_argument("--stage", default="all", choices=["all", "quality", "make", "score", "report"])
    ap.add_argument("--no-aeroblade", action="store_true")
    ap.add_argument("--fake-models", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    a.device = M.pick_device(a.device)
    a.out = a.run
    meta = json.loads((a.run / "data" / "meta.json").read_text())
    a.n = a.n_eff = len(meta)
    cm = a.run / "data" / "calib_meta.json"
    if not cm.exists():
        raise SystemExit("STOP: run dod_extras.py (calib stage) on this run first.")
    a.nc = min(a.calib_n, len(json.loads(cm.read_text())))
    a.res = Image.open(a.run / "data" / "real" / "0000.png").size[0]
    M.SD_ID = "fake" if a.fake_models else M.resolve_sd(None)
    log(f"jpeg: run={a.run} n={a.n_eff} calib={a.nc} device={a.device}")
    for s in (["quality", "make", "score", "report"] if a.stage == "all" else [a.stage]):
        globals()[f"stage_{s}"](a)


if __name__ == "__main__":
    main()
