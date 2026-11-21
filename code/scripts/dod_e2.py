"""Experiment E2: AEROBLADE with the Consistency Decoder's own autoencoder (SD1.5 encoder + Consistency Decoder),
on the unchanged images of the main run. No new generation.

  check   re-score 20 generated (original VAE) and 20 calibration images with the frozen AEROBLADE (SD1.5 AE) and
          compare with frozen/v3_new_scores.csv (proves the images and the LPIPS pipeline are the paper's)
  score   aero_cons: distance between an image and its reconstruction E -> Consistency Decoder (seed 7000+k per image,
          fixed), LPIPS-VGG layer 2, score = -distance; on gen x 4 decoders, the untouched real photos and the
          1,000 calibration photos; plus a seed check (second seed 8000+k) on 50 gen/cons and 50 calibration images
  bfjpeg  E5: B-Free under JPEG q=95/85/75 (same JPEG function as every other detector), official code and weights;
          gate first: the official demo must reproduce the published scores and 40 clean images must reproduce the
          paper's B-Free scores (frozen/bfree_scores_v4.csv)
  report  report_e2.md + e2_summary.json: TPR at nominal 1%/5% FPR with 95% bootstrap CIs (sources + calibration),
          actual FPR on the 300 untouched photos, for aero_cons, the frozen aero_orig/ftmse/taesd, and best-of-4

    python dod_e2.py --run runs/n300 --device cuda
"""
import argparse, json, math, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from PIL import Image
import dod_mac_pilot as M
import dod_v3 as V3
import dod_jpeg as J
import dod_v4 as V4
import subprocess, os, zipfile, urllib.request

DECS = ["orig", "ftmse", "taesd", "cons"]
log = M.log


class AeroCons:
    def __init__(self, a):
        import lpips
        self.dev, self.fake = a.device, a.fake_models
        self.enc = M.load_vae("orig", self.dev, a.fake_models)          # SD1.5 encoder, fp32 (as in AEROBLADE)
        self.dec, _ = V3.load_cons(a)                                     # Consistency Decoder (fp16 on cuda)
        if a.fake_models:
            self.lp = lambda x, r, retPerLayer: (None, [None, ((x - r) ** 2).mean(1, keepdim=True)])
        else:
            self.lp = lpips.LPIPS(net="vgg", verbose=False).to(self.dev).eval()

    @torch.no_grad()
    def __call__(self, img, seed):
        x = M.to_tensor(img).to(self.dev)
        z = self.enc.encode(x).latent_dist.mean                          # unscaled latent
        r = V3.cons_decode(self.dec, self.fake, z * 0.18215, seed)       # cons_decode takes the scaled latent
        _, layers = self.lp(x, r.to(x.dtype), retPerLayer=True)
        return -float(layers[1].mean().cpu())


def images(a):
    R = a.run
    out = [("gen", d, i, R / "png" / "gen" / d / f"{i:04d}.png") for d in DECS for i in range(a.n)]
    out += [("real", "none", i, R / "data" / "real" / f"{i:04d}.png") for i in range(a.n)]
    out += [("calib", "none", k, R / "data" / "calib" / f"{k:04d}.png") for k in range(a.calib)]
    miss = [p for *_, p in out if not p.exists()]
    if miss:
        sys.exit(f"STOP: {len(miss)} images missing, e.g. {miss[0]}. Unpack images_v4_full.tgz or regenerate (README).")
    return out


SEED0 = {"gen": 7000, "real": 17000, "calib": 27000}   # + 100*decoder index + k: one fixed seed per image


def seed_of(g, d, i, base=0):
    return base + SEED0[g] + (DECS.index(d) * 1000 if g == "gen" else 0) + i


def stage_check(a):
    fr = pd.read_csv(Path(__file__).parent / "frozen" / "v3_new_scores.csv")
    fr = fr[fr.detector == "aero_orig"]
    det = V3.AeroAE(a, "orig"); rows = []
    # calibration photos are compared only on a full run (a smoke regeneration selects a different calibration set)
    for g, d in [("gen", "orig")] + ([("calib", "none")] if a.calib >= 1000 else []):
        for i in range(min(20, a.n if g == "gen" else a.calib)):
            p = a.run / ("png/gen/orig" if g == "gen" else "data/calib") / f"{i:04d}.png"
            v = det(Image.open(p)); f = float(fr[(fr.group == g) & (fr.decoder == d) & (fr.idx == i)].score.iloc[0])
            rows.append((g, i, v, f))
    del det; M.free(a.device)
    c = pd.DataFrame(rows, columns=["group", "idx", "now", "frozen"])
    sd = fr[fr.group == "calib"].score.std()
    med = float(np.median(np.abs(c.now - c.frozen)) / sd); rho = float(c[["now", "frozen"]].corr("spearman").iloc[0, 1])
    ok = bool(med <= 0.01 and rho >= 0.99) or a.fake_models
    res = dict(n=len(c), median_abs_diff_sd=med, spearman=rho, max_abs_diff=float(np.abs(c.now - c.frozen).max()), passed=ok)
    # compatibility of the reconstruction model with the latent space: SD1.5 encoder -> Consistency Decoder on the
    # first 20 untouched evaluation photos must give the PSNR the paper reports for this path (23.2 dB, 1 dB tolerance)
    if not a.fake_models:
        det = AeroCons(a); ps = []
        for i in range(min(20, a.n)):
            img = Image.open(a.run / "data" / "real" / f"{i:04d}.png").convert("RGB")
            x = M.to_tensor(img).to(a.device)
            with torch.no_grad():
                r = V3.cons_decode(det.dec, False, det.enc.encode(x).latent_dist.mean * 0.18215, seed_of("real", "none", i))
            ps.append(float(10 * np.log10(4.0 / torch.mean((r.float() - x) ** 2).item())))
        del det; M.free(a.device)
        ref = json.loads((Path(__file__).parent / "frozen" / "cons_check.json").read_text())["psnr_real_enc_first20"]
        res["compat_psnr_first20"] = float(np.mean(ps)); res["compat_psnr_reference"] = ref
        res["compat_passed"] = bool(np.mean(ps) >= 20 and abs(np.mean(ps) - ref) <= 1.0)
        ok = ok and res["compat_passed"]
    (a.run / "e2_check.json").write_text(json.dumps(res, indent=1)); log(f"check: {res}")
    if not ok:
        sys.exit("STOP: the images or the AEROBLADE pipeline differ from the paper's (e2_check.json). Send v4 log.")


def stage_score(a):
    fp = a.run / "e2_scores.csv"
    imgs = images(a)
    rows = pd.read_csv(fp).values.tolist() if fp.exists() else []
    done = {(r[0], r[1], int(r[2]), r[3]) for r in rows}
    det = AeroCons(a)
    seed_set = [("gen", "cons", i) for i in range(min(50, a.n))] + [("calib", "none", k) for k in range(min(50, a.calib))]
    todo = [(g, d, i, p, "aero_cons", 0) for g, d, i, p in imgs]
    todo += [(g, d, i, (a.run / ("png/gen/cons" if g == "gen" else "data/calib") / f"{i:04d}.png"), "aero_cons_s2", 1000000)
             for g, d, i in seed_set]
    todo = [t for t in todo if (t[0], t[1], t[2], t[4]) not in done]
    for j, (g, d, i, p, name, base) in enumerate(todo):
        v = det(Image.open(p), seed_of(g, d, i, base)); assert math.isfinite(v)
        rows.append([g, d, i, name, v])
        if j % 200 == 0:
            log(f"score {j}/{len(todo)}")
            pd.DataFrame(rows, columns=["group", "decoder", "idx", "detector", "score"]).to_csv(fp, index=False)
    pd.DataFrame(rows, columns=["group", "decoder", "idx", "detector", "score"]).to_csv(fp, index=False)
    log(f"score: {len(rows)} scores in {fp}")


def stage_bfjpeg(a):
    """E5. JPEG'd copies are written as lossless PNGs of the decoded JPEG pixels (identical input to scoring the decoded
    image in memory, as for Community-Forensics), then scored by the official main_bfree.py."""
    fp = a.run / "e5_bfree_jpeg.csv"
    if fp.exists():
        log("bfjpeg: already done"); return
    st = {}
    if a.fake_models:
        rows = [[f"jpg{q}_{g}", d, i, float(np.random.default_rng(i).normal())] for q in J.QS for g, d, i in
                [("gen", d, i) for d in DECS for i in range(a.n)] + [("real", "none", i) for i in range(a.n)] +
                [("calib", "none", k) for k in range(min(300, a.calib))]]
        pd.DataFrame(rows, columns=["group", "decoder", "idx", "score"]).assign(detector="bfree").to_csv(fp, index=False)
        (a.run / "e5_status.json").write_text(json.dumps(dict(fake=True))); return
    root = Path("bfree_repo").resolve(); code = root / "code"
    if not root.exists():
        subprocess.run(["git", "clone", "--depth", "1", V4.BF_REPO, str(root)], check=True)
    if not (code / "weights" / V4.BF_MODEL / "config.yaml").exists():
        wz = code / "weights" / "BFREE_dino2reg4.zip"; wz.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(V4.BF_WEIGHTS, wz)
        if V4._md5(wz) != V4.BF_MD5:
            sys.exit("STOP: B-Free weights checksum differs from the published one")
        zipfile.ZipFile(wz).extractall(code / "weights")
    st["repo_commit"] = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dev = "cuda:0" if a.device == "cuda" else a.device
    def run_bf(lst, name):
        inp = (a.run / f"{name}_in.csv").resolve(); out = (a.run / f"{name}_out.csv").resolve()
        pd.DataFrame(dict(filename=[os.path.relpath(p, a.run) for p in lst])).to_csv(inp, index=False)
        r = subprocess.run([sys.executable, "main_bfree.py", "-i", str(inp), "-o", str(out), "-d", dev], cwd=code,
                           capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit("STOP: B-Free inference failed: " + r.stderr[-800:])
        return pd.read_csv(out)[V4.BF_MODEL].values
    # gate 1: official demo
    demo = (a.run / "e5_demo_out.csv").resolve()
    r = subprocess.run([sys.executable, "main_bfree.py", "-i", "demo_images/metainfo.csv", "-o", str(demo), "-d", dev],
                       cwd=code, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("STOP: B-Free demo failed: " + r.stderr[-800:])
    ref = pd.read_csv(code / "demo_images" / "results.csv"); got = pd.read_csv(demo)
    m = pd.merge(ref[["filename", V4.BF_MODEL]], got[["filename", V4.BF_MODEL]], on="filename")
    st["demo_max_abs_diff"] = float((m.iloc[:, 1] - m.iloc[:, 2]).abs().max())
    # gate 2: 20 generated (original VAE) + 20 calibration images reproduce the paper's B-Free scores
    fr = pd.read_csv(Path(__file__).parent / "frozen" / "bfree_scores_v4.csv")
    chk = [("gen", "orig", i) for i in range(min(20, a.n))] + ([("calib", "none", k) for k in range(20)] if a.calib >= 1000 else [])
    now = run_bf([V4.png(a, g, d, i) for g, d, i in chk], "e5_check")
    old = np.array([float(fr[(fr.group == g) & (fr.decoder == d) & (fr.idx == i)].score.iloc[0]) for g, d, i in chk])
    sd = fr[fr.group == "calib"].score.std()
    st["repro_median_abs_diff_sd"] = float(np.median(np.abs(now - old)) / sd)
    st["passed"] = bool(st["demo_max_abs_diff"] <= 0.05 and st["repro_median_abs_diff_sd"] <= 0.01)
    (a.run / "e5_status.json").write_text(json.dumps(st, indent=1)); log(f"bfjpeg gate: {st}")
    if not st["passed"]:
        sys.exit("STOP: B-Free does not reproduce (e5_status.json)")
    items = [("gen", d, i) for d in DECS for i in range(a.n)] + [("real", "none", i) for i in range(a.n)] + \
            [("calib", "none", k) for k in range(min(300, a.calib))]
    rows = []
    for q in J.QS:
        paths = []
        for g, d, i in items:
            o = a.run / "jpg" / str(q) / g / d / f"{i:04d}.png"; o.parent.mkdir(parents=True, exist_ok=True)
            if not o.exists():
                J.jpeg(Image.open(V4.png(a, g, d, i)), q).save(o)
            paths.append(o)
        sc = run_bf(paths, f"e5_q{q}")
        rows += [[f"jpg{q}_{g}", d, i, float(v)] for (g, d, i), v in zip(items, sc)]
        log(f"bfjpeg q={q} done")
    pd.DataFrame(rows, columns=["group", "decoder", "idx", "score"]).assign(detector="bfree").to_csv(fp, index=False)


def report_bfjpeg(a, L, summ):
    fp = a.run / "e5_bfree_jpeg.csv"
    if not fp.exists():
        return
    j = pd.read_csv(fp); c = pd.read_csv(Path(__file__).parent / "frozen" / "bfree_scores_v4.csv")
    C = lambda df, g, d: df[(df.group == g) & (df.decoder == d)].sort_values("idx").score.values
    clean_cal = C(c, "calib", "none")[:a.calib]
    L += ["\n## E5: B-Free under JPEG (thresholds recalibrated on the 300 JPEG'd calibration photos)\n",
          "| q | " + " | ".join(f"TPR@1% {d}" for d in DECS) + " | max drop vs orig [CI] | FPR (recal. 5%) | FPR (clean 5% thr.) |",
          "|---" * 8 + "|"]
    worst = 0; worst_ci = (0, 0)
    for q in J.QS:
        cal = C(j, f"jpg{q}_calib", "none"); real = C(j, f"jpg{q}_real", "none"); base = C(j, f"jpg{q}_gen", "orig")
        cells = []
        for d in DECS:
            tpr, lo, hi, _ = boot_tpr(cal, C(j, f"jpg{q}_gen", d), 0.99); cells.append(f"{tpr:.3f} [{lo:.2f},{hi:.2f}]")
            summ[f"e5|q{q}|{d}"] = tpr
        drops = []
        for d in DECS[1:]:
            b0, b1, dr, lo, hi, _ = V3.tpr_drop_ci(cal, base, C(j, f"jpg{q}_gen", d), 0.99); drops.append((dr, lo, hi, d))
        dr = max(drops); worst = max(worst, dr[0]); worst_ci = (dr[1], dr[2]) if dr[0] == worst else worst_ci
        f_rec = float(np.mean(real > np.quantile(cal, 0.95))); f_cl = float(np.mean(real > np.quantile(clean_cal, 0.95)))
        L.append(f"| {q} | " + " | ".join(cells) + f" | {100*dr[0]:.0f} [{100*dr[1]:.0f},{100*dr[2]:.0f}] ({dr[3]}) | {f_rec:.3f} | {f_cl:.3f} |")
        summ[f"e5|q{q}|fpr_recal5"] = f_rec; summ[f"e5|q{q}|fpr_clean5"] = f_cl
    tprs = [v for k, v in summ.items() if k.startswith("e5|q") and k.split("|")[2] in DECS]
    orig_min = min(summ[f"e5|q{q}|orig"] for q in J.QS)
    if min(tprs) >= 0.95 and worst <= 0.05:
        v = "ROBUST (TPR >= 0.95 everywhere, decoder swing <= 5 points)"
    elif worst >= 0.10 and worst_ci[0] > 0:
        v = "DECODER-DEPENDENT under JPEG (drop >= 10 points, CI excludes 0)"
    elif orig_min <= 0.80 and worst <= 0.05:
        v = "JPEG-FRAGILE overall (orig TPR drops >= 20 points, no decoder dependence)"
    else:
        v = "OTHER (supplement table only)"
    summ["e5_verdict"] = v
    L.append(f"\n**Planned reading (E5):** {v}.\n")


def boot_tpr(cal, gen, q, n=2000, seed=0):
    r = np.random.default_rng(seed); t = np.quantile(cal, q); out = []
    for _ in range(n):
        tt = np.quantile(cal[r.integers(0, len(cal), len(cal))], q)
        out.append(np.mean(gen[r.integers(0, len(gen), len(gen))] > tt))
    return float(np.mean(gen > t)), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)), t


def stage_report(a):
    new = pd.read_csv(a.run / "e2_scores.csv")
    fr = pd.read_csv(Path(__file__).parent / "frozen" / "v3_new_scores.csv")
    fr = fr[fr.detector.isin(["aero_orig", "aero_ftmse", "aero_taesd"]) & fr.group.isin(["gen", "real", "calib"])]
    fr = fr[((fr.group == "calib") & (fr.idx < a.calib)) | ((fr.group != "calib") & (fr.idx < a.n))]
    df = pd.concat([fr, new[new.detector == "aero_cons"]])
    w = df.pivot_table(index=["group", "decoder", "idx"], columns="detector", values="score")
    if {"aero_orig", "aero_ftmse", "aero_taesd", "aero_cons"} <= set(w.columns):
        b4 = w[["aero_orig", "aero_ftmse", "aero_taesd", "aero_cons"]].max(axis=1).reset_index().rename(columns={0: "score"})
        b4["detector"] = "aero_best4"; df = pd.concat([df, b4])
    S = lambda det, g, d: df[(df.detector == det) & (df.group == g) & (df.decoder == d)].sort_values("idx").score.values
    L = ["# E2: AEROBLADE with the Consistency Decoder's autoencoder\n",
         f"N = {a.n} generated latents per decoder, {a.calib} calibration photographs, thresholds at their 99th / 95th percentile. "
         "TPR [95% CI over sources and calibration photos]; FPR = actual flag rate on the untouched evaluation photos.\n"]
    summ = {}
    for q, lab in [(0.99, "1%"), (0.95, "5%")]:
        L += [f"\n## nominal {lab} FPR\n", "| reconstruction AE | " + " | ".join(DECS) + " | actual FPR |", "|---" * 6 + "|"]
        for det in ["aero_orig", "aero_ftmse", "aero_taesd", "aero_cons", "aero_best4"]:
            cal = S(det, "calib", "none")
            if len(cal) == 0: continue
            cells = []
            for d in DECS:
                tpr, lo, hi, t = boot_tpr(cal, S(det, "gen", d), q); cells.append(f"{tpr:.2f} [{lo:.2f}, {hi:.2f}]")
                summ[f"{det}|{d}|{lab}"] = tpr
            fpr = float(np.mean(S(det, "real", "none") > np.quantile(cal, q)))
            L.append(f"| {det} | " + " | ".join(cells) + f" | {fpr:.3f} |"); summ[f"{det}|FPR|{lab}"] = fpr
    s2 = new[new.detector == "aero_cons_s2"]
    if len(s2):
        m = s2.merge(new[new.detector == "aero_cons"], on=["group", "decoder", "idx"])
        sd = S("aero_cons", "calib", "none").std()
        rho = float(m[["score_x", "score_y"]].corr("spearman").iloc[0, 1])
        L.append(f"\nSeed check ({len(m)} images, second seed): median |score diff| = "
                 f"{np.median(np.abs(m.score_x - m.score_y)) / sd:.3f} s.d., Spearman {rho:.3f}.\n")
        summ["seed_median_abs_diff_sd"] = float(np.median(np.abs(m.score_x - m.score_y)) / sd); summ["seed_spearman"] = rho
    report_bfjpeg(a, L, summ)
    t = summ.get("aero_cons|cons|1%")
    if t is not None:
        verdict = ("FULL recovery (>= 0.80)" if t >= 0.80 else "PARTIAL recovery (0.30-0.80)" if t >= 0.30 else
                   "NO recovery (< 0.30) with this reconstruction method")
        L.append(f"\n**Planned reading:** matched AE on Consistency-Decoder images, TPR at 1% FPR = {t:.2f}: {verdict}.\n")
        summ["verdict"] = verdict
    (a.run / "report_e2.md").write_text("\n".join(L)); (a.run / "e2_summary.json").write_text(json.dumps(summ, indent=1))
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True); ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, default=300); ap.add_argument("--calib", type=int, default=1000)
    ap.add_argument("--stage", default="all", choices=["all", "check", "score", "bfjpeg", "report"])
    ap.add_argument("--fake-models", action="store_true")
    a = ap.parse_args(); a.n_eff = a.n; M.SD_ID = "fake" if a.fake_models else M.resolve_sd(None)
    for s in (["check", "score", "bfjpeg", "report"] if a.stage == "all" else [a.stage]):
        globals()[f"stage_{s}"](a)


if __name__ == "__main__":
    main()
