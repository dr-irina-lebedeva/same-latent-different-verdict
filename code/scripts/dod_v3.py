"""v3 additions on a finished SD1.5 run (after dod_mac_pilot.py and dod_extras.py calib/aeroblade/score):

  cons      decode gen + real_enc latents with OpenAI's Consistency Decoder (diffusion decoder,
            same SD latent space)  -> png/<group>/cons
  base      score the cons images with the run's detectors (UnivFD, NPR, CLIP/DINO probes, AEROBLADE)
  new       Community-Forensics (CVPR 2025, OwensLab/commfor-model-384, MIT) on every image, and
            AEROBLADE with a MATCHED autoencoder (SD1.5 VAE / ft-MSE VAE / TAESD, + min over all)
  seeds     Consistency Decoder is stochastic: re-decode 50 held-out gen latents with 2 more seeds
            and score them (seed variance vs decoder effect)
  clipsim   semantic check: CLIP cosine between D_alt(z) and D_orig(z) on the same latent
  report    report_v3.md: TPR at calibrated 1% / 5% FPR with paired CIs that also resample the
            calibration photos, caught->missed counts, matched-AE AEROBLADE table, CLIP similarity,
            planned validation gates (modern-detector baseline, Consistency Decoder scaling +
            seed stability) and an out-of-sample test of the real-photo diagnostic

    python dod_v3.py --run runs/n300 --device cuda 2>&1 | tee v3.log
"""
import argparse, json, math, sys, warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import dod_mac_pilot as M
import dod_bandswap as B1

warnings.filterwarnings("ignore", message=".*encountered in matmul")
log = M.log
DECS = ["orig", "ftmse", "taesd", "cons"]
CONS_GROUPS = ["gen", "real_enc"]
AES = ["orig", "ftmse", "taesd"]
SEEDS = {"cons_s1": 4000, "cons_s2": 5000}   # extra decoder seeds (main cons images use 3000 + i)
N_SEED = 50
OLD_DETS = ["npr", "univfd", "probe_clip", "probe_dino", "aeroblade"]
CF_REPO = "OwensLab/commfor-model-384"
CF_TIMM = "vit_small_patch16_384.augreg_in21k_ft_in1k"


# ------------------------------------------------------------------ consistency decoder
def load_cons(a):
    if a.fake_models:
        torch.manual_seed(2)
        return M.load_vae("orig", a.device, True), True
    from diffusers import ConsistencyDecoderVAE
    dt = torch.float16 if a.device == "cuda" else torch.float32
    return ConsistencyDecoderVAE.from_pretrained("openai/consistency-decoder", torch_dtype=dt).to(a.device).eval(), False


@torch.no_grad()
def cons_decode(vae, fake, z, seed):
    if fake:
        return M.decode(vae, "orig", z)
    g = torch.Generator("cpu").manual_seed(seed)
    # ConsistencyDecoderVAE.decode multiplies by scaling_factor itself -> pass the unscaled latent
    x = vae.decode((z / 0.18215).to(vae.dtype), generator=g).sample
    return x.float().clamp(-1, 1)


def stage_cons(a):
    L, P = a.run / "latents", a.run / "png"
    todo = [g for g in CONS_GROUPS if not ((P / g / "cons").exists()
            and len(list((P / g / "cons").glob("*.png"))) == a.n_eff)]
    if not todo:
        log("cons: already done")
    else:
        vae, fake = load_cons(a)
        for g in todo:
            zs = torch.load(L / f"{g}.pt")
            o = P / g / "cons"; o.mkdir(parents=True, exist_ok=True)
            for i in range(len(zs)):
                M.to_pil(cons_decode(vae, fake, zs[i:i + 1].to(a.device), 3000 + i)).save(o / f"{i:04d}.png")
                if i % 100 == 0:
                    log(f"cons {g} {i}/{len(zs)}")
        del vae; M.free(a.device)
    # visual check: rows = first 4 gen + first 4 real_enc latents, columns = orig | ftmse | taesd | cons
    P0 = a.run / "png"; k = min(4, a.n_eff)
    tiles = [[Image.open(P0 / g / d / f"{i:04d}.png").convert("RGB").resize((224, 224)) for d in DECS]
             for g in CONS_GROUPS for i in range(k)]
    grid = Image.new("RGB", (224 * len(DECS), 224 * len(tiles)), "white")
    for r, row in enumerate(tiles):
        for c, t in enumerate(row):
            grid.paste(t, (224 * c, 224 * r))
    grid.save(a.run / "cons_grid.png"); log(f"cons: wrote {a.run / 'cons_grid.png'} (columns: {', '.join(DECS)})")
    import diffusers, platform
    vers = dict(torch=torch.__version__, diffusers=diffusers.__version__, python=platform.python_version(),
                cons_model="openai/consistency-decoder", cons_seeds="3000+i (main), 4000+i, 5000+i (seed check)",
                commfor=CF_REPO, commfor_timm=CF_TIMM, device=a.device)
    try:
        import timm; vers["timm"] = timm.__version__
    except Exception:
        pass
    (a.run / "v3_versions.json").write_text(json.dumps(vers, indent=1))
    if not a.fake_models:
        ps = [M.psnr(P / "real_enc" / "cons" / f"{i:04d}.png", a.run / "data" / "real" / f"{i:04d}.png")
              for i in range(min(20, a.n_eff))]
        log(f"sanity PSNR cons (first {len(ps)} photos): {np.mean(ps):.1f} dB")
        (a.run / "cons_check.json").write_text(json.dumps({"psnr_real_enc_first20": float(np.mean(ps))}))
        if np.mean(ps) < 20:
            sys.exit("STOP: consistency decoder gives too low PSNR on real photos (<20 dB): latent scaling "
                     "is wrong. Send me the log.")


def cons_images(a):
    return [(g, "cons", i, a.run / "png" / g / "cons" / f"{i:04d}.png") for g in CONS_GROUPS for i in range(a.n_eff)]


def calib_images(a):
    cm = a.run / "data" / "calib_meta.json"
    nc = len(json.loads(cm.read_text()))
    return [("calib", "none", k, a.run / "data" / "calib" / f"{k:04d}.png") for k in range(nc)]


def stage_base(a):
    """UnivFD, NPR, probes, AEROBLADE(SD1.5 AE) on the cons images, via the band-swap scorer."""
    import pandas as pd
    fp = a.run / "v3_base_scores.csv"
    imgs = cons_images(a)
    if fp.exists() and len(pd.read_csv(fp)) >= len(imgs) * 4:
        log("base: already done"); return
    saved = B1.images
    B1.images = lambda _a: imgs
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


# ------------------------------------------------------------------ new detectors
class CommFor:
    """Community-Forensics ViT-S/16-384 (Park & Owens, CVPR 2025). Official test transform:
    Resize(440) -> CenterCrop(384) -> ImageNet norm; output = one logit (fake = 1)."""
    def __init__(self, a):
        import timm
        from torchvision import transforms as T
        self.dev = a.device
        self.m = timm.create_model(CF_TIMM, pretrained=False, num_classes=1)
        if not a.fake_models:
            from huggingface_hub import hf_hub_download
            try:
                from safetensors.torch import load_file
                sd = load_file(hf_hub_download(CF_REPO, "model.safetensors"))
            except Exception:
                sd = torch.load(hf_hub_download(CF_REPO, "pytorch_model.bin"), map_location="cpu")
            sd = {(k[4:] if k.startswith("vit.") else k): v for k, v in sd.items()}
            missing, unexpected = self.m.load_state_dict(sd, strict=False)
            if missing or unexpected:
                raise RuntimeError(f"Community-Forensics weights do not match: missing {missing[:5]}, "
                                   f"unexpected {unexpected[:5]}")
            log(f"commfor: loaded {len(sd)} tensors from {CF_REPO}")
        self.m = self.m.to(self.dev).eval()
        self.tf = T.Compose([T.Resize(440), T.CenterCrop(384), T.ToTensor(),
                             T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    @torch.no_grad()
    def __call__(self, img):
        x = self.tf(img.convert("RGB"))[None].to(self.dev)
        return float(self.m(x).flatten()[0].cpu())


class AeroAE:
    """AEROBLADE with a chosen reconstruction autoencoder (LPIPS-VGG layer 2, score = -distance)."""
    def __init__(self, a, ae):
        import lpips
        from diffusers import AutoencoderKL, AutoencoderTiny
        self.dev, self.ae = a.device, ae
        self.vae = M.load_vae(ae, self.dev, a.fake_models)
        if a.fake_models:
            self.lp = lambda x, r, retPerLayer: (None, [None, ((x - r) ** 2).mean(1, keepdim=True)])
        else:
            self.lp = lpips.LPIPS(net="vgg", verbose=False).to(self.dev).eval()

    @torch.no_grad()
    def __call__(self, img):
        x = M.to_tensor(img).to(self.dev)
        if self.ae == "taesd":
            r = self.vae.decode(self.vae.encode(x).latents).sample
        else:
            r = self.vae.decode(self.vae.encode(x).latent_dist.mean).sample
        r = r.float().clamp(-1, 1)
        _, layers = self.lp(x, r, retPerLayer=True)
        return -float(layers[1].mean().cpu())


def new_images(a):
    main = [x for x in M.all_images(a) if x[0] != "i2i"]
    return main + cons_images(a) + calib_images(a)


def stage_new(a):
    import pandas as pd
    fp = a.run / "v3_new_scores.csv"
    imgs = new_images(a)
    have = pd.read_csv(fp) if fp.exists() else pd.DataFrame(columns=["group", "decoder", "idx", "detector", "score"])
    rows = have.values.tolist()
    dets = [("commfor", lambda: CommFor(a))] + [(f"aero_{ae}", (lambda ae=ae: AeroAE(a, ae))) for ae in AES]
    for name, make in dets:
        if (have.detector == name).sum() >= len(imgs):
            log(f"new: {name} already done"); continue
        try:
            det = make()
        except Exception as e:
            log(f"new: {name} could not load ({e}); skipped"); continue
        for j, (g, d, i, p) in enumerate(imgs):
            v = det(Image.open(p)); assert math.isfinite(v)
            rows.append([g, d, i, name, v])
            if j % 1000 == 0:
                log(f"new {name} {j}/{len(imgs)}")
        del det; M.free(a.device)
        pd.DataFrame(rows, columns=["group", "decoder", "idx", "detector", "score"]).to_csv(fp, index=False)
        log(f"new: {name} done")


def seed_images(a):
    half = a.n_eff // 2   # held-out half, so the probes are not scored on their training latents
    idx = range(half, min(a.n_eff, half + N_SEED))
    return [("gen", d, i, a.run / "png" / "gen" / d / f"{i:04d}.png") for d in SEEDS for i in idx]


def stage_seeds(a):
    import pandas as pd
    imgs = seed_images(a)
    if not all(p.exists() for *_, p in imgs):
        vae, fake = load_cons(a)
        zs = torch.load(a.run / "latents" / "gen.pt")
        for g, d, i, p in imgs:
            p.parent.mkdir(parents=True, exist_ok=True)
            # fake mode: add seeded noise so the variance code path is exercised
            x = cons_decode(vae, fake, zs[i:i + 1].to(a.device), SEEDS[d] + i)
            if fake:
                x = (x + 0.02 * torch.randn(x.shape, generator=torch.Generator().manual_seed(SEEDS[d] + i)).to(x.device)).clamp(-1, 1)
            M.to_pil(x).save(p)
        del vae; M.free(a.device)
        log(f"seeds: decoded {len(imgs)} images")
    fp = a.run / "v3_seed_scores.csv"
    if fp.exists():
        log("seeds: already scored"); return
    saved = B1.images
    B1.images = lambda _a: imgs
    old = a.run / "bandswap_scores.csv"
    try:
        if old.exists():
            old.rename(a.run / "bandswap_scores.keep.csv")
        B1.stage_score(a)
        df = pd.read_csv(a.run / "bandswap_scores.csv"); (a.run / "bandswap_scores.csv").unlink()
    finally:
        if (a.run / "bandswap_scores.keep.csv").exists():
            (a.run / "bandswap_scores.keep.csv").rename(old)
        B1.images = saved
    rows = df.values.tolist()
    for name, make in [("commfor", lambda: CommFor(a)), ("aero_orig", lambda: AeroAE(a, "orig"))]:
        try:
            det = make()
        except Exception as e:
            log(f"seeds: {name} could not load ({e}); skipped"); continue
        rows += [[g, d, i, name, det(Image.open(p))] for g, d, i, p in imgs]
        del det; M.free(a.device)
    pd.DataFrame(rows, columns=["group", "decoder", "idx", "detector", "score"]).to_csv(fp, index=False)
    log(f"seeds: scored {len(rows)} rows")


def stage_clipsim(a):
    import pandas as pd
    fp = a.run / "v3_clipsim.csv"
    if fp.exists():
        log("clipsim: already done"); return
    fx = M.ClipFeat(a.device, a.fake_models)
    P = a.run / "png"
    rows = []
    for g in CONS_GROUPS:
        for i in range(a.n_eff):
            e = {d: fx(Image.open(P / g / d / f"{i:04d}.png")) for d in DECS}
            e = {d: v / (np.linalg.norm(v) + 1e-12) for d, v in e.items()}
            ph = None
            if g == "real_enc":
                v = fx(Image.open(a.run / "data" / "real" / f"{i:04d}.png")); ph = v / (np.linalg.norm(v) + 1e-12)
            for d in DECS[1:]:
                rows.append(dict(group=g, idx=i, decoder=d, cos_vs_orig=float(e[d] @ e["orig"]),
                                 cos_vs_photo=float(e[d] @ ph) if ph is not None else np.nan,
                                 cos_orig_vs_photo=float(e["orig"] @ ph) if ph is not None else np.nan))
    del fx; M.free(a.device)
    pd.DataFrame(rows).to_csv(fp, index=False)
    # reference: CLIP cosine between two DIFFERENT photos (content change), for scale
    log("clipsim: done")


# ------------------------------------------------------------------ report
def load_all(a):
    import pandas as pd
    parts = []
    for f in ["scores.csv", "aeroblade_scores.csv", "extra_scores.csv", "v3_base_scores.csv", "v3_new_scores.csv",
              "v3_seed_scores.csv"]:
        if (a.run / f).exists():
            parts.append(pd.read_csv(a.run / f))
    df = pd.concat(parts)
    df = df[df.group.isin(["real", "real_enc", "gen", "calib"])]
    return df.drop_duplicates(["group", "decoder", "idx", "detector"], keep="last")


def S(df, det, g, d, half=None):
    x = df[(df.detector == det) & (df.group == g) & (df.decoder == d)]
    if half is not None and g != "calib":
        x = x[x.idx >= half]
    return x.sort_values("idx").score.values


def tpr_drop_ci(cal, base, alt, q, n=2000, seed=0):
    """paired TPR(base) - TPR(alt) at the (1-q) quantile of cal; resample cal AND sources."""
    r = np.random.default_rng(seed); t = np.quantile(cal, q)
    out = []
    for _ in range(n):
        tt = np.quantile(cal[r.integers(0, len(cal), len(cal))], q)
        k = r.integers(0, len(base), len(base))
        out.append(np.mean(base[k] > tt) - np.mean(alt[k] > tt))
    return np.mean(base > t), np.mean(alt > t), np.mean(base > t) - np.mean(alt > t), \
        np.percentile(out, 2.5), np.percentile(out, 97.5), int(np.sum((base > t) & ~(alt > t)))


def stage_report(a):
    import pandas as pd
    df = load_all(a)
    half = a.n_eff // 2
    L = []; P = lambda *s: L.append(" ".join(str(x) for x in s))
    P("# v3: Consistency Decoder, Community-Forensics, matched-autoencoder AEROBLADE\n")
    P(f"SD1.5, N = {a.n_eff} sources (probes: {a.n_eff - half} held-out). Thresholds: 1%/5% FPR on the "
      "independent calibration photos. TPR drop CIs resample BOTH the calibration photos (threshold "
      "uncertainty) and the evaluation sources. 'caught->missed' = sources detected with the original "
      "decoder but missed with the alternative on the identical latent.\n")
    cc = a.run / "cons_check.json"
    if cc.exists():
        P(f"Consistency decoder sanity: PSNR on real-photo round-trip (first 20) = "
          f"{json.loads(cc.read_text())['psnr_real_enc_first20']:.1f} dB.\n")
    if {"aero_orig", "aeroblade"} <= set(df.detector):
        x, y = S(df, "aero_orig", "gen", "orig"), S(df, "aeroblade", "gen", "orig")
        if len(x) == len(y) and len(x):
            P(f"Consistency: aero_orig vs the run's AEROBLADE on gen/orig, max |diff| = {np.abs(x - y).max():.2e} (should be ~0).\n")
    dets = [d for d in ["commfor", "aero_orig", "aero_ftmse", "aero_taesd", "aero_min", "npr", "univfd",
                        "probe_clip", "probe_dino"] if d in set(df.detector) or d == "aero_min"]
    # aero_min: AEROBLADE over several AEs = min distance = max score (as in the original paper)
    aer = [f"aero_{ae}" for ae in AES]
    if all(x in set(df.detector) for x in aer):
        w = df[df.detector.isin(aer)].pivot_table(index=["group", "decoder", "idx"], columns="detector", values="score")
        w = w.dropna()
        m = w.max(axis=1).reset_index().rename(columns={0: "score"}); m["detector"] = "aero_min"
        df = pd.concat([df, m[["group", "decoder", "idx", "detector", "score"]]])
    else:
        dets = [d for d in dets if d != "aero_min"]
    P("## Detection after a decoder swap (generated latents)\n")
    P("| detector | AUROC orig | ftmse | taesd | cons | Δ cons (sd) [CI] | q | TPR orig | ftmse | taesd | cons | "
      "drop ftmse [CI] | drop taesd [CI] | drop cons [CI] | caught→missed (ft/tae/cons) | actual FPR (eval photos) |")
    P("|---|" + "---|" * 15)
    for det in dets:
        h = half if det.startswith("probe_") else None
        real, cal = S(df, det, "real", "none", h), S(df, det, "calib", "none")
        g = {d: S(df, det, "gen", d, h) for d in DECS}
        if len(cal) == 0 or len(g["orig"]) == 0:
            continue
        au = [M.auroc(real, g[d]) if len(g[d]) else np.nan for d in DECS]
        sd = np.std(np.concatenate([real, g["orig"]])) + 1e-9
        dc = M.boot_ci((g["cons"] - g["orig"]) / sd) if len(g["cons"]) == len(g["orig"]) else (np.nan,) * 3
        for q in [0.99, 0.95]:
            cells, tp, flips = [], [np.mean(g["orig"] > np.quantile(cal, q))], []
            for d in DECS[1:]:
                if len(g[d]) != len(g["orig"]):
                    cells.append("—"); tp.append(np.nan); flips.append("—"); continue
                b, al, dr, lo, hi, fl = tpr_drop_ci(cal, g["orig"], g[d], q)
                nc = int(np.sum(g["orig"] > np.quantile(cal, q)))
                tp.append(al); cells.append(f"{100 * dr:+.0f} [{100 * lo:+.0f}, {100 * hi:+.0f}]")
                flips.append(f"{fl}/{nc} ({100 * fl / max(nc, 1):.0f}%)")
            fpr = np.mean(real > np.quantile(cal, q))
            P(f"| {det} | " + " | ".join(f"{x:.3f}" for x in au) + f" | {dc[0]:+.2f} [{dc[1]:+.2f}, {dc[2]:+.2f}] | "
              f"{int(round(100 * (1 - q)))}% | " + " | ".join(f"{x:.2f}" for x in tp) + " | " + " | ".join(cells)
              + " | " + " · ".join(flips) + f" | {fpr:.2f} |")
    P("\n'drop' = TPR(orig) - TPR(alt), percentage points; negative = the alternative decoder is detected MORE often. "
      "caught→missed = k/m (%): of the m generated images detected with the original decoder, k are missed with "
      "the alternative decoder on the identical latent. 'actual FPR' = rate on the untouched evaluation photos "
      "at the threshold set on the independent calibration photos.\n")
    # matched-AE table
    if "aero_min" in dets:
        P("## AEROBLADE: generating decoder x reconstruction autoencoder\n")
        P("Each AE configuration has its own threshold from the calibration photos. 'min' = minimum LPIPS "
          "distance over the three AEs (the multi-AE form of the original AEROBLADE). Diagonal = matched.\n")
        for q in [0.99, 0.95]:
            P(f"**TPR at {int(round(100 * (1 - q)))}% FPR** (AUROC in brackets)\n")
            P("| generating decoder | AE orig | AE ftmse | AE taesd | min over AEs |")
            P("|---|---|---|---|---|")
            for d in DECS:
                cells = []
                for det in aer + ["aero_min"]:
                    real, cal, gg = S(df, det, "real", "none"), S(df, det, "calib", "none"), S(df, det, "gen", d)
                    if len(gg) == 0 or len(cal) == 0:
                        cells.append("—"); continue
                    cells.append(f"{np.mean(gg > np.quantile(cal, q)):.2f} ({M.auroc(real, gg):.3f})")
                P(f"| {d} | " + " | ".join(cells) + " |")
            P("")
            P("FPR on the untouched evaluation photos: " + ", ".join(
                f"{det} {np.mean(S(df, det, 'real', 'none') > np.quantile(S(df, det, 'calib', 'none'), q)):.2f}"
                for det in aer + ["aero_min"]) + "\n")
    cs = a.run / "v3_clipsim.csv"
    if cs.exists():
        c = pd.read_csv(cs)
        P("## Semantic check: CLIP ViT-L/14 cosine similarity on the same latent\n")
        P("| latents | decoder | cos(D_alt, D_orig) mean | 5th pct | min | cos(D_alt, photo) | cos(D_orig, photo) |")
        P("|---|---|---|---|---|---|---|")
        for (g, d), r in c.groupby(["group", "decoder"]):
            P(f"| {g} | {d} | {r.cos_vs_orig.mean():.3f} | {r.cos_vs_orig.quantile(0.05):.3f} | {r.cos_vs_orig.min():.3f} | "
              f"{r.cos_vs_photo.mean():.3f} | {r.cos_orig_vs_photo.mean():.3f} |")
        P("\nValues near 1 mean the decoders show the same content; cos(D, photo) for real_enc shows how "
          "close each decoding stays to the original photograph.\n")
    report_gates(a, df, dets, half, P)
    report_diagnostic(a, df, half, P)
    (a.run / "report_v3.md").write_text("\n".join(L))
    log(f"report: wrote {a.run / 'report_v3.md'}")


def d_cell(df, det, grp, alt, half):
    """paired mean shift (alt - orig) on the same latents, in sd units of [real, gen/orig]."""
    h = half if det.startswith("probe_") else None
    real, o, x = S(df, det, "real", "none", h), S(df, det, grp, "orig", h), S(df, det, grp, alt, h)
    if len(o) == 0 or len(o) != len(x) or len(real) == 0:
        return np.nan
    sd = np.std(np.concatenate([real, S(df, det, "gen", "orig", h)])) + 1e-9
    return float(np.mean(x - o) / sd)


def report_gates(a, df, dets, half, P):
    P("## Planned validation gates\n")
    P("Fixed before the results were seen. (G1) A modern detector is used for decoder claims only if its "
      "baseline on original-decoder images is meaningful: AUROC >= 0.80 and TPR >= 0.50 at the calibrated 5% "
      "FPR. (G2) The Consistency Decoder is reported only if the real-photo round-trip PSNR >= 20 dB "
      "(correct latent scaling). (G3) Its decoder effect is reported only if it is larger than its own "
      "seed noise: the range of the mean shift over 3 decoder seeds is < 50% of |mean shift| (or the shift "
      "is < 0.10 sd, i.e. reported as 'no effect'). Cons results are averages over a stochastic decoder, "
      "never a single deterministic decoding.\n")
    P("| gate | detector / check | value | pass |")
    P("|---|---|---|---|")
    for det in [d for d in ["commfor"] if d in set(df.detector)]:
        real, cal, g = S(df, det, "real", "none"), S(df, det, "calib", "none"), S(df, det, "gen", "orig")
        au, tp = M.auroc(real, g), float(np.mean(g > np.quantile(cal, 0.95)))
        P(f"| G1 | {det} | AUROC {au:.3f}, TPR@5% {tp:.2f} | {'yes' if au >= 0.8 and tp >= 0.5 else 'NO'} |")
    cc = a.run / "cons_check.json"
    if cc.exists():
        v = json.loads(cc.read_text())["psnr_real_enc_first20"]
        P(f"| G2 | cons PSNR real round-trip | {v:.1f} dB | {'yes' if v >= 20 else 'NO'} |")
    seed_rows = []
    for det in sorted(set(df[df.decoder.isin(list(SEEDS))].detector)):
        h = half if det.startswith("probe_") else None
        real = S(df, det, "real", "none", h)
        sd = np.std(np.concatenate([real, S(df, det, "gen", "orig", h)])) + 1e-9
        x = df[(df.detector == det) & (df.group == "gen")]
        w = x[x.decoder.isin(["orig", "cons"] + list(SEEDS))].pivot_table(index="idx", columns="decoder", values="score").dropna()
        if len(w) < 5 or not all(c in w for c in ["orig", "cons"] + list(SEEDS)):
            continue
        shifts = [float(((w[c] - w["orig"]) / sd).mean()) for c in ["cons"] + list(SEEDS)]
        per_img = float((w[["cons"] + list(SEEDS)].std(axis=1, ddof=1) / sd).mean())
        m, rng = float(np.mean(shifts)), max(shifts) - min(shifts)
        ok = abs(m) < 0.10 or rng < 0.5 * abs(m)
        seed_rows.append((det, len(w), shifts, per_img, m, rng, ok))
        P(f"| G3 | {det} (n={len(w)}) | shift per seed " + ", ".join(f"{v:+.2f}" for v in shifts)
          + f"; range {rng:.2f}; per-image seed sd {per_img:.2f} | "
          + ("yes (no effect)" if abs(m) < 0.10 else "yes" if ok else "NO") + " |")
    P("\nPer-image seed sd = how much ONE image's score moves when only the decoder's noise seed changes "
      "(sd units). It can be large even when the mean shift is stable; then single-image cons decisions "
      "are partly random and only rates (TPR, mean shift) should be quoted.\n")


def report_diagnostic(a, df, half, P):
    """Out-of-sample test of: shift on encoded REAL photos predicts shift on GENERATED images."""
    dets_all = [d for d in OLD_DETS + ["commfor", "aero_ftmse", "aero_taesd"] if d in set(df.detector)]
    cells = []
    for det in dets_all:
        for alt in ["ftmse", "taesd", "cons"]:
            x, y = d_cell(df, det, "real_enc", alt, half), d_cell(df, det, "gen", alt, half)
            if np.isfinite(x) and np.isfinite(y):
                cells.append(dict(det=det, alt=alt, x=x, y=y,
                                  old=det in OLD_DETS and alt in ["ftmse", "taesd"]))
    old = [c for c in cells if c["old"]]
    new = [c for c in cells if not c["old"]]
    P("## Out-of-sample test of the real-photo diagnostic\n")
    if len(old) < 3:
        P("Not enough old cells to fit.\n"); return
    X = np.array([c["x"] for c in old]); Y = np.array([c["y"] for c in old])
    b, a0 = np.polyfit(X, Y, 1)
    P(f"Fit on the {len(old)} cells used in the paper ({len(set(c['det'] for c in old))} detectors x ft-MSE/TAESD): "
      f"gen shift = {a0:+.2f} + {b:.2f} x real-photo shift, Pearson r = {np.corrcoef(X, Y)[0, 1]:.2f}. "
      "Then predict cells that were NOT in the fit: a new detector (Community-Forensics), a held-out decoder "
      "(Consistency Decoder) and matched-AE AEROBLADE variants. Sign check only where |gen shift| >= 0.05.\n")
    # LODO within old cells
    lo = []
    for det in sorted(set(c["det"] for c in old)):
        tr = [c for c in old if c["det"] != det]; te = [c for c in old if c["det"] == det]
        bb, aa = np.polyfit([c["x"] for c in tr], [c["y"] for c in tr], 1)
        lo += [(aa + bb * c["x"], c["y"]) for c in te]
    lo = np.array(lo)
    lo_sign = [np.sign(p) == np.sign(y) for p, y in lo if abs(y) >= 0.05]
    P(f"Leave-one-detector-out on the old cells: MAE {np.mean(np.abs(lo[:, 0] - lo[:, 1])):.2f} sd, "
      f"sign correct {sum(lo_sign)}/{len(lo_sign)}.\n")
    # leave-one-decoder-out on the old cells
    for dd in ["ftmse", "taesd"]:
        tr = [c for c in old if c["alt"] != dd]; te = [c for c in old if c["alt"] == dd]
        if len(tr) >= 3 and te:
            bb, aa = np.polyfit([c["x"] for c in tr], [c["y"] for c in tr], 1)
            e = [abs(aa + bb * c["x"] - c["y"]) for c in te]
            sg = [np.sign(aa + bb * c["x"]) == np.sign(c["y"]) for c in te if abs(c["y"]) >= 0.05]
            P(f"Leave-out decoder {dd}: MAE {np.mean(e):.2f} sd, sign correct {sum(sg)}/{len(sg)}.\n")
    P("| detector | decoder | real-photo shift | gen shift | predicted | error | sign ok |")
    P("|---|---|---|---|---|---|---|")
    errs, sg = [], []
    for c in new:
        p = a0 + b * c["x"]; errs.append(abs(p - c["y"]))
        ok = "—" if abs(c["y"]) < 0.05 else ("yes" if np.sign(p) == np.sign(c["y"]) else "NO")
        if ok != "—":
            sg.append(ok == "yes")
        P(f"| {c['det']} | {c['alt']} | {c['x']:+.2f} | {c['y']:+.2f} | {p:+.2f} | {abs(p - c['y']):.2f} | {ok} |")
    if errs:
        z = [abs(c["y"]) for c in new]
        maj = np.sign(np.sum(np.sign(Y))) or 1.0
        mj = [np.sign(c["y"]) == maj for c in new if abs(c["y"]) >= 0.05]
        P(f"\nNew cells: MAE {np.mean(errs):.2f} sd vs {np.mean(z):.2f} sd for the zero-change baseline "
          f"(predict 'no effect'); skill = 1 - MAE/MAE_zero = {1 - np.mean(errs) / max(np.mean(z), 1e-9):+.2f}; "
          f"sign correct {sum(sg)}/{len(sg)} vs {sum(mj)}/{len(mj)} for the majority-sign baseline "
          f"(always predict the most common sign of the fit cells, {'+' if maj > 0 else '-'}).\n")
        P("Planned pass needs ALL of: (i) more correct signs than the majority-sign baseline, "
          "(ii) MAE below the zero-change baseline, (iii) the TPR version below flags the large operational "
          "losses (recall >= 2/3 of unseen cells with TPR drop >= 20 pts, at 1% or 5% FPR).\n")
        P(f"Old cells, LODO: MAE {np.mean(np.abs(lo[:, 0] - lo[:, 1])):.2f} sd vs zero baseline "
          f"{np.mean(np.abs(lo[:, 1])):.2f} sd.\n")
    report_tpr_diagnostic(a, df, half, P)


def tpr_cell(df, det, grp, alt, half, q):
    """TPR(orig) - TPR(alt) on the same latents at the calibrated (1-q) threshold (percentage points)."""
    h = half if det.startswith("probe_") else None
    cal = S(df, det, "calib", "none")
    o, x = S(df, det, grp, "orig", h), S(df, det, grp, alt, h)
    if len(cal) == 0 or len(o) == 0 or len(o) != len(x):
        return np.nan
    t = np.quantile(cal, q)
    return float(100 * (np.mean(o > t) - np.mean(x > t)))


def report_tpr_diagnostic(a, df, half, P):
    """Does the real-photo test predict OPERATIONAL loss? x = change in the rate at which VAE-reconstructed
    real photos are flagged; y = TPR drop on generated images; same calibrated threshold."""
    dets_all = [d for d in OLD_DETS + ["commfor", "aero_ftmse", "aero_taesd"] if d in set(df.detector)]
    P("## Does the real-photo test predict TPR loss?\n")
    P("x = drop in the flag rate of VAE-reconstructed real photos (orig decoder minus alternative), "
      "y = TPR drop on generated images, both at the same threshold from the independent calibration "
      "photos, in percentage points. Fit on the old cells only, then predict the unseen cells. "
      "Zero-change baseline = predict 0 drop.\n")
    for q in [0.99, 0.95]:
        cells = []
        for det in dets_all:
            for alt in ["ftmse", "taesd", "cons"]:
                x, y = tpr_cell(df, det, "real_enc", alt, half, q), tpr_cell(df, det, "gen", alt, half, q)
                if np.isfinite(x) and np.isfinite(y):
                    cells.append(dict(det=det, alt=alt, x=x, y=y, old=det in OLD_DETS and alt in ["ftmse", "taesd"]))
        old = [c for c in cells if c["old"]]; new = [c for c in cells if not c["old"]]
        P(f"**{int(round(100 * (1 - q)))}% FPR**\n")
        if len(old) < 3 or np.ptp([c["x"] for c in old]) == 0:
            P("Not enough variation in the old cells to fit.\n"); continue
        X = np.array([c["x"] for c in old]); Y = np.array([c["y"] for c in old])
        b, a0 = np.polyfit(X, Y, 1)
        r = np.corrcoef(X, Y)[0, 1] if np.ptp(Y) > 0 else np.nan
        P(f"Old cells: y = {a0:+.1f} + {b:.2f} x, Pearson r = {r:.2f} (n = {len(old)}).\n")
        P("| detector | decoder | real-photo flag drop | gen TPR drop | predicted | error | sign ok |")
        P("|---|---|---|---|---|---|---|")
        errs, z, sg = [], [], []
        for c in new:
            pr = a0 + b * c["x"]; errs.append(abs(pr - c["y"])); z.append(abs(c["y"]))
            ok = "—" if abs(c["y"]) < 5 else ("yes" if np.sign(pr) == np.sign(c["y"]) else "NO")
            if ok != "—":
                sg.append(ok == "yes")
            P(f"| {c['det']} | {c['alt']} | {c['x']:+.0f} | {c['y']:+.0f} | {pr:+.0f} | {abs(pr - c['y']):.0f} | {ok} |")
        if errs:
            maj = np.sign(np.sum(np.sign(Y))) or 1.0
            mj = [np.sign(c["y"]) == maj for c in new if abs(c["y"]) >= 5]
            P(f"\nUnseen cells: MAE {np.mean(errs):.1f} pts vs {np.mean(z):.1f} pts for the zero-change baseline; "
              f"sign correct {sum(sg)}/{len(sg)} vs {sum(mj)}/{len(mj)} for the majority-sign baseline "
              "(cells with |TPR drop| >= 5 pts).\n")
            pr = np.array([a0 + b * c["x"] for c in new]); yy = np.array([c["y"] for c in new])
            big, pbig = yy >= 20, pr >= 20
            P(f"Large losses (TPR drop >= 20 pts): {int(big.sum())} unseen cells; diagnostic flags {int(pbig.sum())}; "
              f"caught {int((big & pbig).sum())}/{int(big.sum())} (recall), "
              f"{int((big & pbig).sum())}/{int(pbig.sum())} of flags are real (precision). "
              "Flagged = predicted drop >= 20 pts.\n")
            if len(new) >= 4 and np.ptp(yy) > 0 and np.ptp(pr) > 0:
                from scipy.stats import spearmanr
                P(f"Rank agreement on unseen cells: Spearman {spearmanr(pr, yy)[0]:.2f}.\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=Path("cache"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--stage", default="all", choices=["all", "cons", "base", "new", "seeds", "clipsim", "report"])
    ap.add_argument("--no-aeroblade", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--fake-models", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    a.device = M.pick_device(a.device)
    a.out = a.run
    meta = json.loads((a.run / "data" / "meta.json").read_text())
    a.n = a.n_eff = len(meta)
    if not (a.run / "data" / "calib_meta.json").exists():
        raise SystemExit("STOP: run dod_extras.py (calib, aeroblade, score) on this run first.")
    a.res = Image.open(a.run / "data" / "real" / "0000.png").size[0]
    M.SD_ID = "fake" if a.fake_models else M.resolve_sd(None)
    log(f"v3: run={a.run} n={a.n_eff} device={a.device}")
    for s in (["cons", "base", "new", "seeds", "clipsim", "report"] if a.stage == "all" else [a.stage]):
        globals()[f"stage_{s}"](a)


if __name__ == "__main__":
    main()
