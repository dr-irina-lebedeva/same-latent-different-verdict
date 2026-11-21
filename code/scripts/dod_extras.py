#!/usr/bin/env python3
"""
Extra experiments on top of a finished dod_mac_pilot.py run folder (does not change it).

  calib      download M *other* COCO photos (not in the run) -> independent real calibration set
  aeroblade  AEROBLADE score (SD1.5 VAE reconstruction, LPIPS-VGG layer 2) for every image
  flux       real photos -> FLUX.1 VAE encoder -> decode with FLUX VAE and with TAEF1 (same latent)
  score      score calib + flux images with UnivFD / NPR / CLIP & DINO probes (+ AEROBLADE)
  report     report_extras.md: calibrated thresholds (5% and 1% FPR from the calib set),
             AEROBLADE in the main tables, and the FLUX decoder-swap check

Example:
  python dod_extras.py --run runs/n100 --calib 1000 --stage all
"""
import argparse, io, json, math, random, sys, urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dod_mac_pilot as M  # noqa: E402

FLUX_VAE_ID = "black-forest-labs/FLUX.1-schnell"   # Apache-2.0; VAE in subfolder "vae"
TAEF1_ID = "madebyollin/taef1"
FLUX_DECODERS = ["fluxvae", "taef1"]
log = M.log


# ------------------------------------------------------------------ calib
def stage_calib(a):
    d = a.run / "data" / "calib"; d.mkdir(parents=True, exist_ok=True)
    meta_p = a.run / "data" / "calib_meta.json"
    have = json.loads(meta_p.read_text()) if meta_p.exists() else []
    if len(have) >= a.calib:
        log("calib: already done"); return
    main = {m["coco_id"] for m in json.loads((a.run / "data" / "meta.json").read_text())}
    if a.fake_models:
        rng = np.random.default_rng(1)
        for k in range(a.calib):
            x = rng.normal(size=(a.res // 16, a.res // 16, 3))
            Image.fromarray(((x - x.min()) / np.ptp(x) * 255).astype(np.uint8)).resize(
                (a.res, a.res), Image.BICUBIC).save(d / f"{k:04d}.png")
        meta_p.write_text(json.dumps([{"k": k} for k in range(a.calib)])); return
    ann = json.loads((a.cache / "captions_val2017.json").read_text())
    ids = sorted(im["id"] for im in ann["images"]
                 if min(im["width"], im["height"]) >= a.res and im["id"] not in main)
    random.Random(1).shuffle(ids)          # different seed from the main sample
    meta = have
    done = {m["coco_id"] for m in meta}
    for iid in ids:
        if len(meta) >= a.calib:
            break
        if iid in done:
            continue
        try:
            raw = urllib.request.urlopen(M.COCO_IMG_URL.format(iid), timeout=30).read()
            im = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as e:
            log("skip", iid, e); continue
        w, h = im.size
        l, t = (w - a.res) // 2, (h - a.res) // 2
        k = len(meta)
        im.crop((l, t, l + a.res, t + a.res)).save(d / f"{k:04d}.png")
        meta.append({"k": k, "coco_id": iid, "crop_box": [l, t, l + a.res, t + a.res]})
        if k % 100 == 0:
            log(f"calib {k}/{a.calib}"); meta_p.write_text(json.dumps(meta))
    meta_p.write_text(json.dumps(meta, indent=1))
    log(f"calib: {len(meta)} independent real photos (no overlap with the run)")


# ------------------------------------------------------------------ flux
def load_flux(name, dev, fake=False):
    from diffusers import AutoencoderKL, AutoencoderTiny
    if fake:
        if name == "taef1":
            return AutoencoderTiny(latent_channels=16, encoder_block_out_channels=(8,) * 4,
                                   decoder_block_out_channels=(8,) * 4, num_encoder_blocks=(1,) * 4,
                                   num_decoder_blocks=(1,) * 4).to(dev).eval()
        return AutoencoderKL(block_out_channels=(8,) * 4, layers_per_block=1,
                             down_block_types=("DownEncoderBlock2D",) * 4,
                             up_block_types=("UpDecoderBlock2D",) * 4, latent_channels=16,
                             norm_num_groups=4, scaling_factor=0.3611, shift_factor=0.1159).to(dev).eval()
    if name == "fluxvae":
        return AutoencoderKL.from_pretrained(FLUX_VAE_ID, subfolder="vae").to(dev).eval()
    return AutoencoderTiny.from_pretrained(TAEF1_ID).to(dev).eval()


@torch.no_grad()
def stage_flux(a):
    """Same convention as the FLUX pipeline: z = (mean - shift) * scale is what the transformer sees;
    AutoencoderKL decodes z / scale + shift; TAEF1 (AutoencoderTiny) decodes z directly."""
    out = a.run / "png" / "flux_real"
    n = a.n_eff
    if all(len(list((out / d).glob("*.png"))) == n for d in FLUX_DECODERS if (out / d).exists()) \
            and all((out / d).exists() for d in FLUX_DECODERS):
        log("flux: already done"); return
    dev = a.device
    vae = load_flux("fluxvae", dev, a.fake_models)
    sc, sh = vae.config.scaling_factor, (vae.config.shift_factor or 0.0)
    zs = []
    for k in range(n):
        x = M.to_tensor(Image.open(a.run / "data" / "real" / f"{k:04d}.png")).to(dev)
        zs.append(((vae.encode(x).latent_dist.mean - sh) * sc).cpu())
    zs = torch.cat(zs)
    torch.save(zs, a.run / "latents" / "flux_real_enc.pt")
    (out / "fluxvae").mkdir(parents=True, exist_ok=True)
    for k in range(n):
        M.to_pil(vae.decode(zs[k:k + 1].to(dev) / sc + sh).sample.float().clamp(-1, 1)).save(
            out / "fluxvae" / f"{k:04d}.png")
    del vae; M.free(dev)
    tv = load_flux("taef1", dev, a.fake_models)
    # scale check: TAEF1's own encoder should land on the same latent scale as (mean - shift) * scale
    xs = torch.cat([M.to_tensor(Image.open(a.run / "data" / "real" / f"{k:04d}.png"))
                    for k in range(min(4, n))]).to(dev)
    zt = tv.encode(xs).latents.float().cpu()
    ratio = float(zt.std() / zs[:len(zt)].std())
    corr = float(np.corrcoef(zt.flatten(), zs[:len(zt)].flatten())[0, 1])
    log(f"TAEF1 scale check: std ratio {ratio:.2f} (want 0.7-1.4), correlation {corr:.2f} (want > 0.8)")
    if not a.fake_models and not (0.7 < ratio < 1.4):
        sys.exit("STOP: TAEF1 latent scale does not match FLUX latents. Send me the log.")
    (out / "taef1").mkdir(parents=True, exist_ok=True)
    for k in range(n):
        M.to_pil(tv.decode(zs[k:k + 1].to(dev)).sample.float().clamp(-1, 1)).save(
            out / "taef1" / f"{k:04d}.png")
    del tv; M.free(dev)
    for d in FLUX_DECODERS:
        ps = [M.psnr(out / d / f"{k:04d}.png", a.run / "data" / "real" / f"{k:04d}.png") for k in range(n)]
        log(f"flux sanity PSNR {d}: {np.mean(ps):.1f} dB")
    (a.run / "flux_check.json").write_text(json.dumps({"taef1_std_ratio": ratio, "taef1_corr": corr}))


# ------------------------------------------------------------------ aeroblade
class Aeroblade:
    """AEROBLADE (Ricker et al., CVPR 2024): distance between an image and its reconstruction by the
    generator's autoencoder, LPIPS-VGG 2nd layer. Low distance = fake. Score = -distance (higher =
    more fake). We use the SD1.5 autoencoder (the generator's own AE)."""
    def __init__(self, a):
        import lpips
        self.dev = a.device
        self.vae = M.load_vae("orig", self.dev, a.fake_models)
        self.lp = lpips.LPIPS(net="vgg", verbose=False).to(self.dev).eval()

    @torch.no_grad()
    def __call__(self, img):
        x = M.to_tensor(img).to(self.dev)
        z = self.vae.encode(x).latent_dist.mean
        r = self.vae.decode(z).sample.clamp(-1, 1)
        _, layers = self.lp(x, r, retPerLayer=True)
        return -float(layers[1].mean().cpu())


def stage_aeroblade(a):
    import pandas as pd
    fp = a.run / "aeroblade_scores.csv"
    paths = image_list(a)
    if fp.exists() and len(pd.read_csv(fp)) == len(paths):
        log("aeroblade: already done"); return
    if a.fake_models:
        rows = [(g, d, i, "aeroblade", float(np.random.default_rng(i).normal())) for g, d, i, _ in paths]
    else:
        det = Aeroblade(a)
        rows = []
        for j, (g, d, i, p) in enumerate(paths):
            v = det(Image.open(p)); assert math.isfinite(v)
            rows.append((g, d, i, "aeroblade", v))
            if j % 200 == 0:
                log(f"aeroblade {j}/{len(paths)}")
        del det; M.free(a.device)
    pd.DataFrame(rows, columns=["group", "decoder", "idx", "detector", "score"]).to_csv(fp, index=False)
    log(f"aeroblade: {len(rows)} images scored")


def image_list(a):
    """main run images + calib reals + flux reconstructions (whatever exists)"""
    out = M.all_images(a)
    cm = a.run / "data" / "calib_meta.json"
    if cm.exists():
        nc = len(json.loads(cm.read_text()))
        out += [("calib", "none", k, a.run / "data" / "calib" / f"{k:04d}.png") for k in range(nc)]
    for d in FLUX_DECODERS:
        p = a.run / "png" / "flux_real" / d
        if p.exists():
            out += [("flux_real", d, k, p / f"{k:04d}.png") for k in range(a.n_eff)]
    return out


# ------------------------------------------------------------------ score extra images
def stage_score(a):
    """Score calib + flux images with the same detectors. Probes are refit exactly as in the main
    run (same features, same split, same settings), then applied to the new images."""
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    main = M.all_images(a)
    extra = [x for x in image_list(a) if x[0] in ("calib", "flux_real")]
    if not extra:
        log("score: no extra images (run calib / flux first)"); return
    rows, dev = [], a.device
    for name, cls in [("clip", M.ClipFeat), ("dino", M.DinoFeat)]:
        Fm = np.load(a.run / f"feat_{name}.npy").astype(np.float64)
        assert len(Fm) == len(main), "main-run features do not match; rerun the main --stage score"
        fx = cls(dev, a.fake_models)
        Fe = np.stack([fx(Image.open(p)) for *_, p in extra]).astype(np.float64)
        del fx; M.free(dev)
        assert np.isfinite(Fe).all()
        if name == "clip" and not a.fake_models:
            w = torch.load(a.cache / "univfd_fc_weights.pth", map_location="cpu")
            s = Fe @ w["weight"].double().numpy().reshape(-1) + float(w["bias"].reshape(-1)[0])
            rows += [(g, d, i, "univfd", float(v)) for (g, d, i, _), v in zip(extra, s)]
        half = a.n_eff // 2
        tr = [j for j, (g, d, i, _) in enumerate(main)
              if i < half and ((g == "real") or (g == "gen" and d == "orig"))]
        y = [0 if main[j][0] == "real" else 1 for j in tr]
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=5000)).fit(Fm[tr], y)
        s = clf.decision_function(Fe)
        rows += [(g, d, i, f"probe_{name}", float(v)) for (g, d, i, _), v in zip(extra, s)]
    npr = None if a.fake_models else M.load_npr(a)
    if npr is not None:
        rows += [(g, d, i, "npr", npr(Image.open(p))) for g, d, i, p in extra]
    df = pd.DataFrame(rows, columns=["group", "decoder", "idx", "detector", "score"])
    assert np.isfinite(df.score).all()
    df.to_csv(a.run / "extra_scores.csv", index=False)
    log(f"score: {len(df)} extra rows")


# ------------------------------------------------------------------ report
def stage_report(a):
    import pandas as pd
    main = pd.read_csv(a.run / "scores.csv")
    parts = [main]
    if (a.run / "aeroblade_scores.csv").exists():
        parts.append(pd.read_csv(a.run / "aeroblade_scores.csv"))
    allm = pd.concat(parts)
    # 1) full main report incl. AEROBLADE (main-run images only)
    allm[allm.group.isin(["real", "real_enc", "gen", "i2i"])].to_csv(a.run / "scores_all.csv", index=False)
    a.out = a.run; a.scores_file = "scores_all.csv"; a.report_file = "report_with_aeroblade.md"
    a.steps, a.strength = a.steps, a.strength
    M.stage_report(a)
    ex = pd.read_csv(a.run / "extra_scores.csv") if (a.run / "extra_scores.csv").exists() else None
    if ex is not None:
        allm = pd.concat([allm, ex])
    half = a.n_eff // 2
    L = []; P = lambda *s: L.append(" ".join(str(x) for x in s))
    P("# Extra results\n")
    # 2) calibrated thresholds from the independent calib set
    P("## Thresholds calibrated on independent real photos\n")
    if not (allm.group == "calib").any():
        P("(no calib set; run --stage calib and score)\n")
    else:
        P("| detector | n calib | FPR target | FPR on run reals | TPR gen orig | TPR gen ftmse | TPR gen taesd "
          "| FPR real-enc orig | ftmse | taesd | FPR flux fluxvae | taef1 |")
        P("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for det in sorted(allm.detector.unique()):
            d = allm[allm.detector == det]
            if det.startswith("probe_"):
                d = d[(d.group == "calib") | (d.idx >= half)]
            S = lambda g, dec: d[(d.group == g) & (d.decoder == dec)].score.values
            cal = S("calib", "none")
            if len(cal) == 0:
                continue
            for fpr in [0.05, 0.01]:
                t = np.quantile(cal, 1 - fpr)
                rate = lambda v: f"{np.mean(v > t):.2f}" if len(v) else "—"
                P(f"| {det} | {len(cal)} | {fpr:.0%} | {rate(S('real', 'none'))} | {rate(S('gen', 'orig'))} | "
                  f"{rate(S('gen', 'ftmse'))} | {rate(S('gen', 'taesd'))} | {rate(S('real_enc', 'orig'))} | "
                  f"{rate(S('real_enc', 'ftmse'))} | {rate(S('real_enc', 'taesd'))} | "
                  f"{rate(S('flux_real', 'fluxvae'))} | {rate(S('flux_real', 'taef1'))} |")
        P("\n'FPR on run reals' should be close to the target if calibration transfers. "
          "real-enc / flux columns: share of VAE-reconstructed REAL photos called fake.\n")
    # 3) FLUX decoder swap on real photos
    P("## FLUX latent space: FLUX VAE vs TAEF1 on the same real-photo latents\n")
    if not (allm.group == "flux_real").any():
        P("(FLUX step not part of this run folder)\n")
    else:
        P("| detector | AUROC fluxvae-recon vs real | AUROC taef1-recon vs real | paired Δ taef1 - fluxvae (sd) [CI] |")
        P("|---|---|---|---|")
        for det in sorted(allm.detector.unique()):
            d = allm[allm.detector == det]
            if det.startswith("probe_"):
                d = d[d.idx >= half]
            S = lambda g, dec: d[(d.group == g) & (d.decoder == dec)].sort_values("idx").score.values
            real, f0, f1 = S("real", "none"), S("flux_real", "fluxvae"), S("flux_real", "taef1")
            if len(f0) == 0 or len(real) == 0:
                continue
            sd = np.std(np.concatenate([real, S("gen", "orig")])) + 1e-9
            m = M.boot_ci((f1 - f0) / sd)
            P(f"| {det} | {M.auroc(real, f0):.3f} | {M.auroc(real, f1):.3f} | {m[0]:+.2f} [{m[1]:+.2f}, {m[2]:+.2f}] |")
        P("\nNo FLUX generation here (needs a cloud GPU). This shows whether swapping FLUX's decoder changes "
          "how 'fake' VAE-reconstructed real photos look, i.e. decoder dependence in a modern latent space.\n")
    (a.run / "report_extras.md").write_text("\n".join(L))
    print("\n".join(L))
    log(f"wrote {a.run / 'report_with_aeroblade.md'} and {a.run / 'report_extras.md'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True, help="finished dod_mac_pilot.py run folder")
    ap.add_argument("--calib", type=int, default=1000)
    ap.add_argument("--cache", type=Path, default=Path("cache"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--stage", default="all", choices=["all", "calib", "flux", "aeroblade", "score", "report"])
    ap.add_argument("--fake-models", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    a.device = M.pick_device(a.device)
    a.out = a.run
    meta = json.loads((a.run / "data" / "meta.json").read_text())
    a.n_eff = a.n = len(meta)
    a.res = Image.open(a.run / "data" / "real" / "0000.png").size[0]
    gs = a.run / "latents" / "gen_settings.json"
    g = json.loads(gs.read_text()) if gs.exists() else {}
    a.steps, a.strength = g.get("steps", "?"), g.get("img2img_strength", "?")
    if not a.fake_models:
        M.SD_ID = M.resolve_sd(None)
    log(f"extras: run={a.run} n={a.n_eff} res={a.res} device={a.device}")
    st = ["calib", "flux", "aeroblade", "score", "report"] if a.stage == "all" else [a.stage]
    for s in st:
        globals()[f"stage_{s}"](a)


if __name__ == "__main__":
    main()
