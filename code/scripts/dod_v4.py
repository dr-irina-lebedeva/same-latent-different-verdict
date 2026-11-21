"""v4 add-on to the frozen v3 run (same photos, seeds, models). Plan fixed in the run README before the run.

Stages (run in this order by run_v4.sh)
  gate     reproduction gate: compare this run with the frozen v3 files in ./frozen
           (photo ids, UnivFD/NPR scores, per-image PSNR, Community-Forensics on Consistency-Decoder
           images, model revisions) -> gate.json
  quality  P1: Consistency Decoder reconstruction quality on ALL photos, same metrics as Table 1
           (PSNR, LPIPS-AlexNet, HF ratio) + SSIM for every decoder -> quality_v4.csv
  cfjpeg   P1b: Community-Forensics under JPEG q=95/85/75 (same JPEG code as dod_jpeg.py), thresholds
           re-calibrated on the first 300 JPEG'd independent photos, clean thresholds from all 1,000
           -> cf_scores_v4.csv
  bfree    P2 (optional): B-Free (Guillaro et al., CVPR 2025) with its official code, weights and
           preprocessing; skipped unless the official demo reproduces -> bfree_scores_v4.csv
  report   report_v4.md

    python dod_v4.py --run runs/n300 --device cuda
"""
import argparse, hashlib, io, json, math, os, shutil, subprocess, sys, urllib.request, zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import dod_mac_pilot as M
import dod_v3 as V3
import dod_jpeg as J

log = M.log
DECS = ["orig", "ftmse", "taesd", "cons"]
QS = J.QS                                  # [95, 85, 75], identical to the other detectors
FROZEN = Path(__file__).resolve().parent / "frozen"
BF_REPO = "https://github.com/grip-unina/B-Free.git"
BF_WEIGHTS = "https://www.grip.unina.it/download/prog/B-Free/weights/BFREE_dino2reg4.zip"
BF_MD5 = "f3f53fa647848b16cf81c913f148a198"
BF_MODEL = "BFREE_dino2reg4"


def png(a, g, d, i):
    if g == "real":
        return a.run / "data" / "real" / f"{i:04d}.png"
    if g == "calib":
        return a.run / "data" / "calib" / f"{i:04d}.png"
    return a.run / "png" / g / d / f"{i:04d}.png"


def clean_images(a):
    out = [("real", "none", i) for i in range(a.n)]
    out += [(g, d, i) for g in ["gen", "real_enc"] for d in DECS for i in range(a.n)]
    out += [("calib", "none", k) for k in range(a.nc_all)]
    return out


# ------------------------------------------------------------------ gate
def _sd(fz, det):
    x = fz[(fz.detector == det) & (((fz.group == "real") & (fz.decoder == "none")) |
                                   ((fz.group == "gen") & (fz.decoder == "orig")))]
    return float(x.score.std())


def _cmp(new, old, sd):
    """median |diff| in s.d. units and Spearman rho over the common indices"""
    import pandas as pd
    m = pd.merge(new, old, on="idx", suffixes=("_n", "_o"))
    if len(m) < 3:
        return dict(n=len(m), med_abs_sd=float("nan"), rho=float("nan"))
    d = (m.score_n - m.score_o).abs() / sd
    return dict(n=len(m), med_abs_sd=float(d.median()), max_abs_sd=float(d.max()),
                rho=float(m.score_n.rank().corr(m.score_o.rank())))


def stage_gate(a):
    import pandas as pd
    G = {}
    # R0 photos: same COCO ids, same order
    meta = json.loads((a.run / "data" / "meta.json").read_text())
    exp = json.loads((FROZEN / "expected_main_coco_ids.json").read_text())["coco_ids"]
    got = [m["coco_id"] for m in meta]
    G["R0_photo_ids"] = dict(n=len(got), same_order=int(sum(x == y for x, y in zip(got, exp))),
                             ok=got == exp[:len(got)])
    # R1/R3 UnivFD + NPR on untouched photos (R1) and generated originals (R3)
    fz = pd.read_csv(FROZEN / "scores.csv"); nw = pd.read_csv(a.run / "scores.csv")
    for key, g, d in [("R1_real_photos", "real", "none"), ("R3_generated_orig", "gen", "orig"),
                      ("R3_generated_taesd", "gen", "taesd")]:
        G[key] = {}
        for det in ["univfd", "npr"]:
            sel = lambda x: x[(x.detector == det) & (x.group == g) & (x.decoder == d)][["idx", "score"]]
            G[key][det] = _cmp(sel(nw), sel(fz), _sd(fz, det))
    # R2 per-image PSNR of the three conventional decoders on encoded photos
    qf = pd.read_csv(FROZEN / "per_image_quality.csv")
    G["R2_psnr_real_enc"] = {}
    for d in ["orig", "ftmse", "taesd"]:
        o = qf[(qf.group == "real_enc") & (qf.decoder == d)].set_index("idx").psnr_photo
        diffs = [abs(M.psnr(png(a, "real_enc", d, i), png(a, "real", "none", i)) - o[i])
                 for i in range(a.n) if i in o.index]
        G["R2_psnr_real_enc"][d] = dict(n=len(diffs), med_abs_db=float(np.median(diffs)),
                                        max_abs_db=float(np.max(diffs)))
    # R4 Community-Forensics on Consistency-Decoder images (first 50) and on generated originals
    cf = V3.CommFor(a)
    fv = pd.read_csv(FROZEN / "v3_new_scores.csv"); fv = fv[fv.detector == "commfor"]
    sd = _sd(fv, "commfor")
    G["R4_commfor"] = {}
    for g, d in [("gen", "orig"), ("gen", "cons"), ("real_enc", "cons")]:
        k = min(50, a.n)
        new = pd.DataFrame([dict(idx=i, score=cf(Image.open(png(a, g, d, i)))) for i in range(k)])
        old = fv[(fv.group == g) & (fv.decoder == d)][["idx", "score"]]
        G["R4_commfor"][f"{g}/{d}"] = _cmp(new, old, sd)
    del cf; M.free(a.device)
    # R5 model revisions
    def revs(p):
        out = {}
        for line in Path(p).read_text().splitlines():
            parts = line.split()
            if len(parts) == 2 and "/" in parts[0] and not parts[0].startswith("cache"):
                out[parts[0]] = parts[1]
        return out
    old, new = revs(FROZEN / "model_revisions.txt"), revs(a.run / "model_revisions_v4.txt")
    G["R5_model_revisions"] = {k: dict(frozen=v, now=new.get(k), same=new.get(k) == v) for k, v in old.items()
                               if k in new}
    # verdicts (thresholds fixed in README before the run)
    def ok(r, tol):
        v = [x for x in r.values() if x["n"] >= 3]
        rho_min = 0.99 if all(x["n"] >= 20 for x in v) else 0.9   # smoke run: only 4 images
        return bool(v) and all(x["med_abs_sd"] <= tol and x["rho"] >= rho_min for x in v)
    # Tolerances allow float noise from a different GPU (e.g. A40 vs A6000): identical inputs give
    # differences far below these limits; a wrong photo, latent scale or model revision gives >> 0.1 s.d.
    photos_ok = G["R0_photo_ids"]["ok"] and ok(G["R1_real_photos"], 0.01)
    enc_ok = all(v["med_abs_db"] <= 0.05 for v in G["R2_psnr_real_enc"].values())
    gen_ok = ok(G["R3_generated_orig"], 0.05) and ok(G["R3_generated_taesd"], 0.05)
    # diffusion sampling can amplify GPU float noise over 30 steps: "near" = same images up to fine detail
    gen_near = ok(G["R3_generated_orig"], 0.25) and ok(G["R3_generated_taesd"], 0.25)
    cons_enc_ok = ok({k: v for k, v in G["R4_commfor"].items() if k == "real_enc/cons"}, 0.05)
    cons_gen_ok = ok({k: v for k, v in G["R4_commfor"].items() if k == "gen/cons"}, 0.05 if gen_ok else 0.25)
    revs_ok = all(v["same"] for v in G["R5_model_revisions"].values())
    G["verdict"] = dict(photos_identical=photos_ok, encoded_reconstructions_identical=enc_ok,
                        consistency_decoder_on_photos_reproduced=cons_enc_ok,
                        generation_reproduced=gen_ok, generation_near=gen_near,
                        consistency_decoder_on_generated_reproduced=cons_gen_ok, model_revisions_same=revs_ok,
                        quality_comparable=photos_ok and enc_ok and cons_enc_ok,
                        same_images_as_paper=photos_ok and enc_ok and gen_ok and cons_enc_ok and cons_gen_ok,
                        near_same_images=photos_ok and enc_ok and gen_near and cons_enc_ok and cons_gen_ok)
    (a.run / "gate.json").write_text(json.dumps(G, indent=1))
    log("gate:", json.dumps(G["verdict"]))


# ------------------------------------------------------------------ quality
def ssim(p, r):
    try:
        from skimage.metrics import structural_similarity
    except Exception:
        return float("nan")
    x = np.asarray(Image.open(p).convert("RGB")); y = np.asarray(Image.open(r).convert("RGB"))
    return float(structural_similarity(x, y, channel_axis=2, data_range=255))


def stage_quality(a):
    import pandas as pd
    if a.fake_models:                                  # offline plumbing test only
        lp = lambda x, y: torch.zeros(1)
    else:
        import lpips
        lp = lpips.LPIPS(net="alex", verbose=False)   # same network as Table 1 (M.quality_table)
    rows = []
    with torch.no_grad():
        for d in DECS:
            for i in range(a.n):
                p, r = png(a, "real_enc", d, i), png(a, "real", "none", i)
                rows.append(dict(decoder=d, idx=i, psnr=M.psnr(p, r),
                                 lpips=float(lp(M.to_tensor(Image.open(p)), M.to_tensor(Image.open(r)))),
                                 hf_ratio=M.hf_ratio(p) / max(M.hf_ratio(r), 1e-12), ssim=ssim(p, r)))
            log(f"quality: {d} done")
    pd.DataFrame(rows).to_csv(a.run / "quality_v4.csv", index=False)


# ------------------------------------------------------------------ Community-Forensics x JPEG
def stage_cfjpeg(a):
    import pandas as pd
    fp = a.run / "cf_scores_v4.csv"
    if fp.exists():
        log("cfjpeg: already done"); return
    cf = V3.CommFor(a)
    rows = []
    base = clean_images(a)
    for j, (g, d, i) in enumerate(base):
        rows.append([g, d, i, cf(Image.open(png(a, g, d, i)))])
        if j % 1000 == 0:
            log(f"cf clean {j}/{len(base)}")
    jp = [(g, d, i) for g, d, i in base if g != "calib" or i < a.nc_jpeg]
    for q in QS:
        for g, d, i in jp:
            # identical to dod_jpeg.py: PIL JPEG at quality q, decoded back to RGB; nothing else changes
            rows.append([f"jpg{q}_{g}", d, i, cf(J.jpeg(Image.open(png(a, g, d, i)), q))])
        log(f"cf jpeg q={q} done")
    del cf; M.free(a.device)
    pd.DataFrame(rows, columns=["group", "decoder", "idx", "score"]).assign(detector="commfor") \
        .to_csv(fp, index=False)


# ------------------------------------------------------------------ B-Free
def _md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def stage_bfree(a):
    import pandas as pd
    st = a.run / "bfree_status.json"
    def stop(msg):
        st.write_text(json.dumps(dict(ran=False, reason=msg), indent=1)); log("bfree: SKIPPED -", msg)
    root = Path("bfree_repo")
    try:
        if not root.exists():
            subprocess.run(["git", "clone", "--depth", "1", BF_REPO, str(root)], check=True)
        code = root / "code"
        wz = code / "weights" / "BFREE_dino2reg4.zip"
        if not (code / "weights" / BF_MODEL / "config.yaml").exists():
            wz.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(BF_WEIGHTS, wz)
            if _md5(wz) != BF_MD5:
                return stop(f"weights md5 {_md5(wz)} != published {BF_MD5}")
            zipfile.ZipFile(wz).extractall(code / "weights")
        commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        # 1) official demo must reproduce the published demo scores
        demo_out = (a.run / "bfree_demo_out.csv").resolve()
        r = subprocess.run([sys.executable, "main_bfree.py", "-i", "demo_images/metainfo.csv", "-o", str(demo_out),
                            "-d", "cuda:0" if a.device == "cuda" else a.device], cwd=code, capture_output=True, text=True)
        if r.returncode != 0:
            return stop("official demo failed: " + r.stderr[-800:])
        ref = pd.read_csv(code / "demo_images" / "results.csv"); got = pd.read_csv(demo_out)
        m = pd.merge(ref[["filename", BF_MODEL]], got[["filename", BF_MODEL]], on="filename", suffixes=("_ref", "_got"))
        dmax = float((m[BF_MODEL + "_ref"] - m[BF_MODEL + "_got"]).abs().max())
        if len(m) < len(ref) or not dmax <= 0.05:
            return stop(f"demo does not reproduce the published scores (n={len(m)}/{len(ref)}, max |diff|={dmax:.4f})")
        # 2) our images, unchanged files, official inference (normalisation only, whole image)
        imgs = clean_images(a)
        lst = pd.DataFrame([dict(filename=os.path.relpath(png(a, g, d, i), a.run), group=g, decoder=d, idx=i)
                            for g, d, i in imgs])
        lst[["filename"]].to_csv(a.run / "bfree_in.csv", index=False)
        out = (a.run / "bfree_raw.csv").resolve()
        r = subprocess.run([sys.executable, "main_bfree.py", "-i", str((a.run / "bfree_in.csv").resolve()), "-o", str(out),
                            "-d", "cuda:0" if a.device == "cuda" else a.device], cwd=code, capture_output=True, text=True)
        if r.returncode != 0:
            return stop("scoring failed: " + r.stderr[-800:])
        sc = pd.read_csv(out)
        lst["score"] = sc[BF_MODEL].values; lst["detector"] = "bfree"
        lst.drop(columns="filename").to_csv(a.run / "bfree_scores_v4.csv", index=False)
        st.write_text(json.dumps(dict(ran=True, repo_commit=commit, weights_md5=BF_MD5, demo_max_abs_diff=dmax,
                                      preprocessing="official main_bfree.py: ToTensor + config norm, whole "
                                                    "448x448 image, no crop/resize; score = fake logit"), indent=1))
        log("bfree: scored", len(lst))
    except Exception as e:
        return stop(f"{type(e).__name__}: {e}")


# ------------------------------------------------------------------ report
def margins(gen, t, sd):
    return float((gen.min() - t) / sd), int(np.sum((gen - t) / sd < 0.5))


def detector_section(P, name, D, sd_src, a, jpeg=True):
    """D: DataFrame group/decoder/idx/score for one detector."""
    def S(g, d):
        return D[(D.group == g) & (D.decoder == d)].sort_values("idx").score.values
    cal = S("calib", "none"); real = S("real", "none")
    sd = float(np.std(np.concatenate([real, S("gen", "orig")]))) + 1e-9
    P(f"## {name}\n")
    P(f"s.d. unit = {sd:.4g}. Thresholds: 99th / 95th percentile of {len(cal)} clean independent photos "
      "(nominal 1% / 5% FPR). Drop = TPR(orig) - TPR(alt), CI resamples calibration photos and sources.\n")
    P("| q | decoder | AUROC | TPR@1% | drop@1% [CI] | TPR@5% | drop@5% [CI] | caught->missed @1% | "
      "lowest gen score above 1% thr (s.d.) | gen within 0.5 s.d. | Δ vs orig (s.d.) [CI] | actual FPR @1% / 5% |")
    P("|---|" + "---|" * 11)
    conds = [(None, "")] + ([(q, f"jpg{q}_") for q in QS] if jpeg else [])
    for q, pre in conds:
        r_ = S(pre + "real", "none")
        if len(r_) == 0:
            continue
        c_ = cal if q is None else S(pre + "calib", "none")   # recalibrated on JPEG'd photos
        g0 = S(pre + "gen", "orig")
        t1, t5 = np.quantile(c_, 0.99), np.quantile(c_, 0.95)
        for d in DECS:
            gd = S(pre + "gen", d)
            if len(gd) != len(g0):
                continue
            o1 = V3.tpr_drop_ci(c_, g0, gd, 0.99); o5 = V3.tpr_drop_ci(c_, g0, gd, 0.95)
            mg, near = margins(gd, t1, sd)
            dl = M.boot_ci((gd - g0) / sd) if d != "orig" else (0, 0, 0)
            P(f"| {'clean' if q is None else q} | {d} | {M.auroc(r_, gd):.3f} | {o1[1]:.2f} | "
              f"{100*o1[2]:+.0f} [{100*o1[3]:+.0f}, {100*o1[4]:+.0f}] | {o5[1]:.2f} | "
              f"{100*o5[2]:+.0f} [{100*o5[3]:+.0f}, {100*o5[4]:+.0f}] | {o1[5]}/{int(np.sum(g0 > t1))} | "
              f"{mg:+.2f} | {near} | {dl[0]:+.2f} [{dl[1]:+.2f}, {dl[2]:+.2f}] | "
              f"{np.mean(r_ > t1):.2f} / {np.mean(r_ > t5):.2f} |")
    P("")


def stage_report(a):
    import pandas as pd
    L = []; P = lambda *s: L.append(" ".join(str(x) for x in s))
    P("# v4 report (plan fixed before the run)\n")
    gp = a.run / "gate.json"
    if gp.exists():
        G = json.loads(gp.read_text())
        P("## Reproduction gate\n"); P("```"); P(json.dumps(G, indent=1)); P("```\n")
    qp = a.run / "quality_v4.csv"
    if qp.exists():
        q = pd.read_csv(qp)
        P("## Reconstruction quality on all photos (Table 1 metrics)\n")
        P("| decoder | n | PSNR (dB) | LPIPS (Alex) | HF ratio | SSIM | PSNR >= 20 dB (gate G2) |")
        P("|---|---|---|---|---|---|---|")
        for d, r in q.groupby("decoder", sort=False):
            P(f"| {d} | {len(r)} | {r.psnr.mean():.2f} | {r.lpips.mean():.3f} | {r.hf_ratio.mean():.2f} | "
              f"{r.ssim.mean():.3f} | {'yes' if r.psnr.mean() >= 20 else 'NO'} |")
        c = q[q.decoder == "cons"].psnr
        P(f"\nConsistency Decoder PSNR first 20 photos: {c.iloc[:20].mean():.2f} dB (frozen value "
          f"{json.loads((FROZEN / 'cons_check.json').read_text())['psnr_real_enc_first20']:.2f}); "
          f"all: {c.mean():.2f} dB, bootstrap CI {M.boot_ci(c.values)[1]:.2f}-{M.boot_ci(c.values)[2]:.2f}.\n")
    cp = a.run / "cf_scores_v4.csv"
    if cp.exists():
        detector_section(P, "Community-Forensics: clean and JPEG (thresholds re-calibrated on JPEG'd photos)",
                         pd.read_csv(cp), None, a)
        # clean-threshold variant (deployed detector unaware of JPEG): TPR@5% only
        D = pd.read_csv(cp); cal = D[(D.group == "calib")].score.values
        P("Clean thresholds applied to JPEG'd images (TPR@1% / TPR@5%; FPR on JPEG'd eval photos):\n")
        P("| q | orig | ftmse | taesd | cons | FPR@1% / 5% |"); P("|---|---|---|---|---|---|")
        t1, t5 = np.quantile(cal, 0.99), np.quantile(cal, 0.95)
        for qq in QS:
            S = lambda g, d: D[(D.group == f"jpg{qq}_{g}") & (D.decoder == d)].score.values
            P(f"| {qq} | " + " | ".join(f"{np.mean(S('gen', d) > t1):.2f} / {np.mean(S('gen', d) > t5):.2f}" for d in DECS)
              + f" | {np.mean(S('real', 'none') > t1):.2f} / {np.mean(S('real', 'none') > t5):.2f} |")
        P("")
    bs = a.run / "bfree_status.json"
    if bs.exists():
        s = json.loads(bs.read_text()); P("## B-Free\n"); P("```"); P(json.dumps(s, indent=1)); P("```\n")
        if s.get("ran"):
            D = pd.read_csv(a.run / "bfree_scores_v4.csv")
            real = D[D.group == "real"].score.values; g0 = D[(D.group == "gen") & (D.decoder == "orig")].score.values
            cal = D[D.group == "calib"].score.values
            au = M.auroc(real, g0); t5 = np.mean(g0 > np.quantile(cal, 0.95))
            P(f"Acceptance gate G1 (same as Community-Forensics): AUROC {au:.3f} >= 0.80 and TPR@5% {t5:.2f} >= 0.50 "
              f"-> {'PASS' if au >= 0.8 and t5 >= 0.5 else 'FAIL: no decoder claims for B-Free'}\n")
            detector_section(P, "B-Free (clean only)", D, None, a, jpeg=False)
    (a.run / "report_v4.md").write_text("\n".join(L))
    log(f"report: wrote {a.run / 'report_v4.md'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=Path("cache"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--calib-jpeg", type=int, default=300, help="JPEG'd calibration photos (as dod_jpeg.py)")
    ap.add_argument("--stage", default="all", choices=["all", "gate", "quality", "cfjpeg", "bfree", "report"])
    ap.add_argument("--no-bfree", action="store_true")
    ap.add_argument("--fake-models", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    a.device = M.pick_device(a.device); a.out = a.run
    a.n = a.n_eff = len(json.loads((a.run / "data" / "meta.json").read_text()))
    a.nc_all = len(json.loads((a.run / "data" / "calib_meta.json").read_text()))
    a.nc_jpeg = min(a.calib_jpeg, a.nc_all)
    a.res = Image.open(a.run / "data" / "real" / "0000.png").size[0]
    log(f"v4: run={a.run} n={a.n} calib={a.nc_all} (jpeg {a.nc_jpeg}) device={a.device}")
    st = ["gate", "quality", "cfjpeg"] + ([] if a.no_bfree else ["bfree"]) + ["report"]
    for s in (st if a.stage == "all" else [a.stage]):
        globals()[f"stage_{s}"](a)


if __name__ == "__main__":
    main()
