"""Add-on for the same pod (after run_e2.sh): features for E4/E10, and a control for the real-photo round-trip.
No new generation. Every image is the exact image scored in the paper (runs/n300 from images_v4_full.tgz).

  feats   CLIP ViT-L/14 image embeddings and DINOv2-base CLS features (the paper's probe/UnivFD features, same
          preprocessing) for: the 300 untouched evaluation photos, their VAE round-trips (4 decoders), the 1,200
          generated images (4 decoders) and the 1,000 calibration photos -> feats_clip.npy, feats_dino.npy,
          feats_index.csv. Gate: UnivFD recomputed from these features must reproduce the paper's UnivFD scores
          (median |diff| <= 0.01 s.d., Spearman >= 0.99).
  rtctrl  Control for the round-trip result: remove JPEG traces from the 300 untouched photos WITHOUT a VAE and
          score them with every detector at the paper's thresholds.
            shift4   reflect-pad 4 px top/left, crop back to the original size (JPEG 8x8 grid misaligned)
            downup2  bicubic downscale x0.5 and upscale back (removes high frequencies, incl. JPEG traces)
            noise2   additive Gaussian noise, sigma = 2/255, fixed seed per image (masks quantisation traces)
          Detectors: AEROBLADE (SD1.5 AE), NPR, UnivFD, CLIP/DINO probes (refit exactly as in the paper),
          Community-Forensics, B-Free (official code; same gates as E5). -> rtctrl_scores.csv, rtctrl_feats_*.npy
  report  report_e4.md: flag rates at nominal 1%/5% FPR for untouched, controls, and the VAE round-trips.

    python dod_e4.py --run runs/n300 --device cuda
"""
import argparse, json, math, os, subprocess, sys, urllib.request, zipfile
from pathlib import Path
import numpy as np, pandas as pd, torch
from PIL import Image, ImageOps
import dod_mac_pilot as M
import dod_v3 as V3
import dod_v4 as V4

DECS = ["orig", "ftmse", "taesd", "cons"]
CTRL = ["shift4", "downup2", "noise2"]
FROZEN = Path(__file__).resolve().parent / "frozen"
log = M.log


def index(a):
    R = a.run; rows = [("real", "none", i, R / "data" / "real" / f"{i:04d}.png") for i in range(a.n)]
    rows += [("real_enc", d, i, R / "png" / "real_enc" / d / f"{i:04d}.png") for d in DECS for i in range(a.n)]
    rows += [("gen", d, i, R / "png" / "gen" / d / f"{i:04d}.png") for d in DECS for i in range(a.n)]
    rows += [("calib", "none", k, R / "data" / "calib" / f"{k:04d}.png") for k in range(a.calib)]
    miss = [p for *_, p in rows if not p.exists()]
    if miss:
        sys.exit(f"STOP: {len(miss)} images missing, e.g. {miss[0]}")
    return rows


def univfd_head(a):
    fcp = a.cache / "univfd_fc_weights.pth"
    if not fcp.exists():
        a.cache.mkdir(parents=True, exist_ok=True); urllib.request.urlretrieve(M.UNIVFD_FC_URL, fcp)
    w = torch.load(fcp, map_location="cpu")
    return w["weight"].double().numpy().reshape(-1), float(w["bias"].reshape(-1)[0]), fcp


def extract(a, paths_or_imgs, name):
    cls = M.ClipFeat if name == "clip" else M.DinoFeat
    fx = cls(a.device, a.fake_models)
    F = np.stack([fx(p if isinstance(p, Image.Image) else Image.open(p)) for p in paths_or_imgs]).astype(np.float32)
    del fx; M.free(a.device)
    assert np.isfinite(F).all(), f"{name} features not finite"
    return F


def stage_feats(a):
    rows = index(a); R = a.run
    pd.DataFrame([(g, d, i, os.path.relpath(p, R)) for g, d, i, p in rows],
                 columns=["group", "decoder", "idx", "path"]).to_csv(R / "feats_index.csv", index=False)
    for name in ["clip", "dino"]:
        fp = R / f"feats_{name}.npy"
        if fp.exists() and len(np.load(fp)) == len(rows):
            log(f"feats {name}: already done"); continue
        np.save(fp, extract(a, [p for *_, p in rows], name)); log(f"feats {name}: {len(rows)} images")
    if a.fake_models:
        (R / "e4_feats_check.json").write_text(json.dumps(dict(fake=True, passed=True))); return
    wv, b, fcp = univfd_head(a)
    s = np.load(R / "feats_clip.npy").astype(np.float64) @ wv + b
    fr = pd.read_csv(FROZEN / "main_scores_e4.csv"); fr = fr[fr.detector == "univfd"]
    key = {(g, d, int(i)): v for g, d, i, v in fr[["group", "decoder", "idx", "score"]].itertuples(index=False)}
    pairs = np.array([(v, key[(g, d, i)]) for (g, d, i, _), v in zip(rows, s) if (g, d, i) in key])
    sd = fr[fr.group == "calib"].score.std()
    med = float(np.median(np.abs(pairs[:, 0] - pairs[:, 1])) / sd)
    rho = float(pd.DataFrame(pairs).corr("spearman").iloc[0, 1])
    res = dict(n=len(pairs), median_abs_diff_sd=med, spearman=rho, passed=bool(med <= 0.01 and rho >= 0.99))
    (R / "e4_feats_check.json").write_text(json.dumps(res, indent=1)); log(f"feats gate: {res}")
    import shutil; shutil.copy(fcp, R / "univfd_fc_weights.pth")
    if not res["passed"]:
        sys.exit("STOP: features do not reproduce the paper's UnivFD scores (e4_feats_check.json)")


def control(img, kind, i):
    img = img.convert("RGB"); W, H = img.size
    if kind == "shift4":
        x = np.asarray(img); x = np.pad(x, ((4, 0), (4, 0), (0, 0)), mode="reflect")[:H, :W]
        return Image.fromarray(x)
    if kind == "downup2":
        return img.resize((W // 2, H // 2), Image.BICUBIC).resize((W, H), Image.BICUBIC)
    if kind == "noise2":
        x = np.asarray(img).astype(np.float64) + np.random.default_rng(90000 + i).normal(0, 2.0, (H, W, 3))
        return Image.fromarray(np.clip(np.round(x), 0, 255).astype(np.uint8))
    raise ValueError(kind)


def bf_setup(a):
    root = Path("bfree_repo").resolve(); code = root / "code"
    if not root.exists():
        subprocess.run(["git", "clone", "--depth", "1", V4.BF_REPO, str(root)], check=True)
    if not (code / "weights" / V4.BF_MODEL / "config.yaml").exists():
        wz = code / "weights" / "BFREE_dino2reg4.zip"; wz.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(V4.BF_WEIGHTS, wz)
        if V4._md5(wz) != V4.BF_MD5:
            sys.exit("STOP: B-Free weights checksum differs from the published one")
        zipfile.ZipFile(wz).extractall(code / "weights")
    return code


def run_bf(a, code, paths, name):
    dev = "cuda:0" if a.device == "cuda" else a.device
    inp = (a.run / f"{name}_in.csv").resolve(); out = (a.run / f"{name}_out.csv").resolve()
    pd.DataFrame(dict(filename=[os.path.relpath(p, a.run) for p in paths])).to_csv(inp, index=False)
    r = subprocess.run([sys.executable, "main_bfree.py", "-i", str(inp), "-o", str(out), "-d", dev], cwd=code,
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("STOP: B-Free inference failed: " + r.stderr[-800:])
    return pd.read_csv(out)[V4.BF_MODEL].values


def stage_rtctrl(a):
    R = a.run; fp = R / "rtctrl_scores.csv"
    if fp.exists():
        log("rtctrl: already done"); return
    items = []
    for kind in CTRL:
        for i in range(a.n):
            o = R / "rtctrl" / kind / f"{i:04d}.png"; o.parent.mkdir(parents=True, exist_ok=True)
            if not o.exists():
                control(Image.open(R / "data" / "real" / f"{i:04d}.png"), kind, i).save(o)
            items.append((kind, i, o))
    rows = []
    # features: UnivFD + probes (refit exactly as in the paper: k < N/2, untouched real vs gen/orig)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    idx = pd.read_csv(R / "feats_index.csv"); half = a.n // 2
    tr = idx[(idx.idx < half) & ((idx.group == "real") | ((idx.group == "gen") & (idx.decoder == "orig")))]
    for name in ["clip", "dino"]:
        Fc = extract(a, [p for *_, p in items], name); np.save(R / f"rtctrl_feats_{name}.npy", Fc)
        Fm = np.load(R / f"feats_{name}.npy").astype(np.float64)
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=5000)).fit(
            Fm[tr.index.values], (tr.group == "gen").astype(int).values)
        rows += [(f"ctrl_{k}", "none", i, f"probe_{name}", float(v)) for (k, i, _), v in zip(items, clf.decision_function(Fc.astype(np.float64)))]
        if name == "clip" and not a.fake_models:
            wv, b, _ = univfd_head(a)
            rows += [(f"ctrl_{k}", "none", i, "univfd", float(v)) for (k, i, _), v in zip(items, Fc.astype(np.float64) @ wv + b)]
    log("rtctrl: probes/UnivFD done")
    if a.fake_models:
        rows += [(f"ctrl_{k}", "none", i, d, float(np.random.default_rng(i).normal())) for k, i, _ in items
                 for d in ["aeroblade", "npr", "commfor", "bfree"]]
    else:
        det = V3.AeroAE(a, "orig")
        rows += [(f"ctrl_{k}", "none", i, "aeroblade", det(Image.open(p))) for k, i, p in items]; del det; M.free(a.device)
        npr = M.load_npr(a)
        if npr is not None:
            rows += [(f"ctrl_{k}", "none", i, "npr", npr(Image.open(p))) for k, i, p in items]; M.free(a.device)
        cf = V3.CommFor(a)
        rows += [(f"ctrl_{k}", "none", i, "commfor", cf(Image.open(p))) for k, i, p in items]; del cf; M.free(a.device)
        log("rtctrl: AEROBLADE/NPR/CF done")
        code = bf_setup(a)
        fr = pd.read_csv(FROZEN / "bfree_scores_v4.csv")
        chk = [("real", "none", i) for i in range(20)]
        now = run_bf(a, code, [V4.png(a, g, d, i) for g, d, i in chk], "e4_bfcheck")
        old = np.array([float(fr[(fr.group == g) & (fr.decoder == d) & (fr.idx == i)].score.iloc[0]) for g, d, i in chk])
        med = float(np.median(np.abs(now - old)) / fr[fr.group == "calib"].score.std())
        (R / "e4_bfree_check.json").write_text(json.dumps(dict(repro_median_abs_diff_sd=med, passed=med <= 0.01)))
        if med > 0.01:
            sys.exit("STOP: B-Free does not reproduce the paper's scores on 20 untouched photos")
        sc = run_bf(a, code, [p for *_, p in items], "e4_rtctrl")
        rows += [(f"ctrl_{k}", "none", i, "bfree", float(v)) for (k, i, _), v in zip(items, sc)]
    df = pd.DataFrame(rows, columns=["group", "decoder", "idx", "detector", "score"])
    assert np.isfinite(df.score).all()
    df.to_csv(fp, index=False); log(f"rtctrl: {len(df)} scores")


def stage_report(a):
    R = a.run; c = pd.read_csv(R / "rtctrl_scores.csv")
    fr = pd.concat([pd.read_csv(FROZEN / "main_scores_e4.csv"), pd.read_csv(FROZEN / "bfree_scores_v4.csv")])
    L = ["# E4/E10 features and round-trip control (add-on)\n",
         f"Feature gate: {json.loads((R / 'e4_feats_check.json').read_text())}\n",
         "Flag rate on the 300 untouched evaluation photos after each treatment, at the paper's thresholds "
         "(1,000 independent calibration photos).\n",
         "| detector | FPR | untouched | " + " | ".join(CTRL) + " | VAE round-trip (orig) |", "|---" * 7 + "|"]
    for det in ["aeroblade", "npr", "univfd", "probe_clip", "probe_dino", "commfor", "bfree"]:
        cal = fr[(fr.detector == det) & (fr.group == "calib")].score.values
        if not len(cal) or det not in set(c.detector):
            continue
        for q in [0.99, 0.95]:
            t = np.quantile(cal, q); sel = lambda df, g, d="none": df[(df.detector == det) & (df.group == g) & (df.decoder == d)].score.values
            cells = [np.mean(sel(fr, "real") > t)] + [np.mean(sel(c, f"ctrl_{k}") > t) for k in CTRL] + [np.mean(sel(fr, "real_enc", "orig") > t)]
            L.append(f"| {det} | {100*(1-q):.0f}% | " + " | ".join(f"{v:.3f}" for v in cells) + " |")
    (R / "report_e4.md").write_text("\n".join(L)); print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True); ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, default=300); ap.add_argument("--calib", type=int, default=1000)
    ap.add_argument("--cache", type=Path, default=Path("cache"))
    ap.add_argument("--stage", default="all", choices=["all", "feats", "rtctrl", "report"])
    ap.add_argument("--fake-models", action="store_true")
    a = ap.parse_args(); a.n_eff = a.n; M.SD_ID = "fake" if a.fake_models else M.resolve_sd(None)
    for s in (["feats", "rtctrl", "report"] if a.stage == "all" else [a.stage]):
        globals()[f"stage_{s}"](a)


if __name__ == "__main__":
    main()
