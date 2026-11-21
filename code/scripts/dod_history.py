"""Decoder x denoising-history sweep on a finished SD1.5 run (e.g. runs/n100).

For the same real photos it makes img2img latents at strength 0.3 and 0.7 (0.5 is reused from
the run), decodes every strength with ALL three decoders (orig, ft-MSE, TAESD), scores them with
the same detectors, and reports, per detector and per decoder:

    D(s)  = mean paired score change (alt - orig) at strength s, in sd units
    I(s)  = D(s) - D(0)   (s = 0 is real_enc: no denoising)      [95% bootstrap CI over photos]

If I(s) grows with s, denoising history changes how a decoder swap moves the detector.

Usage (after dod_extras.py has finished):
    python dod_history.py --run runs/n100 2>&1 | tee history.log
Stages: latents, decode, score, report (default all; each one resumes).
"""
import argparse, json, math, sys, warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import dod_mac_pilot as M
import dod_extras as X

warnings.filterwarnings("ignore", message=".*encountered in matmul")
log = M.log
NEW = [0.3, 0.7]                    # new strengths; 0.5 comes from the run
DECS = ["orig", "ftmse", "taesd"]
ALTS = ["ftmse", "taesd"]


def sname(s):
    return f"s{int(round(s * 100)):02d}"


def stage_latents(a):
    meta = json.loads((a.run / "data" / "meta.json").read_text())
    L = a.run / "latents"
    todo = [s for s in NEW if not (L / f"i2i_{sname(s)}.pt").exists()]
    if not todo:
        log("history latents: already done"); return
    if a.fake_models:
        base = torch.load(L / "real_enc.pt"); g = torch.Generator().manual_seed(0)
        for s in todo:
            torch.save((1 - s) * base + s * torch.randn(base.shape, generator=g) * base.std(),
                       L / f"i2i_{sname(s)}.pt")
        log("history latents: fake done"); return
    from diffusers import StableDiffusionImg2ImgPipeline
    p2 = StableDiffusionImg2ImgPipeline.from_pretrained(
        M.SD_ID, torch_dtype=torch.float32, safety_checker=None,
        requires_safety_checker=False).to(a.device)
    p2.enable_attention_slicing(); p2.set_progress_bar_config(disable=True)
    for s in todo:
        zs = []
        for i, m in enumerate(meta):
            g = torch.Generator("cpu").manual_seed(2000 + i)          # same seeds as the run
            init = Image.open(a.run / "data" / "real" / f"{m['k']:04d}.png")
            z = p2(m["caption"], image=init, strength=s, num_inference_steps=a.steps,
                   guidance_scale=7.5, generator=g, output_type="latent").images
            zs.append(z.float().cpu())
            if i % 10 == 0:
                log(f"i2i strength {s}: {i + 1}/{len(meta)}")
        torch.save(torch.cat(zs), L / f"i2i_{sname(s)}.pt")
    del p2; M.free(a.device); log("history latents: done")


def groups(a):
    """history group -> latent file. s50 reuses the run's i2i latents."""
    L = a.run / "latents"
    g = {"h_s50": L / "i2i.pt"}
    for s in NEW:
        g[f"h_{sname(s)}"] = L / f"i2i_{sname(s)}.pt"
    return g


@torch.no_grad()
def stage_decode(a):
    P = a.run / "png"
    for dec in DECS:
        jobs = [(g, f) for g, f in groups(a).items()
                if not ((P / g / dec).exists() and len(list((P / g / dec).glob("*.png"))) == a.n_eff)]
        if not jobs:
            continue
        vae = M.load_vae(dec, a.device, a.fake_models)
        for g, f in jobs:
            zs = torch.load(f); o = P / g / dec; o.mkdir(parents=True, exist_ok=True)
            for i in range(len(zs)):
                M.to_pil(M.decode(vae, dec, zs[i:i + 1].to(a.device))).save(o / f"{i:04d}.png")
        del vae; M.free(a.device); log(f"history decode: {dec} done")
    # image check
    bad = 0
    for g in groups(a):
        for dec in DECS:
            for i in range(a.n_eff):
                x = np.asarray(Image.open(P / g / dec / f"{i:04d}.png"), dtype=np.float32)
                if x.std() < 3 or np.mean((x <= 1) | (x >= 254)) > 0.5:
                    bad += 1
    log(f"history image check: {bad} suspicious images")


def images(a):
    return [(g, d, i, a.run / "png" / g / d / f"{i:04d}.png")
            for g in groups(a) for d in DECS for i in range(a.n_eff)]


def stage_score(a):
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    fp = a.run / "history_scores.csv"
    imgs = images(a)
    if fp.exists() and len(pd.read_csv(fp)) >= len(imgs) * 4:
        log("history score: already done"); return
    main = M.all_images(a)
    rows, dev = [], a.device
    for name, cls in [("clip", M.ClipFeat), ("dino", M.DinoFeat)]:
        Fm = np.load(a.run / f"feat_{name}.npy").astype(np.float64)
        assert len(Fm) == len(main), "main-run features do not match the run"
        fx = cls(dev, a.fake_models)
        Fe = np.stack([fx(Image.open(p)) for *_, p in imgs]).astype(np.float64)
        del fx; M.free(dev)
        assert np.isfinite(Fe).all()
        if name == "clip" and not a.fake_models:
            w = torch.load(a.cache / "univfd_fc_weights.pth", map_location="cpu")
            s = Fe @ w["weight"].double().numpy().reshape(-1) + float(w["bias"].reshape(-1)[0])
            rows += [(g, d, i, "univfd", float(v)) for (g, d, i, _), v in zip(imgs, s)]
        half = a.n_eff // 2          # probes refit exactly as in the main run
        tr = [j for j, (g, d, i, _) in enumerate(main)
              if i < half and ((g == "real") or (g == "gen" and d == "orig"))]
        y = [0 if main[j][0] == "real" else 1 for j in tr]
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=5000)).fit(Fm[tr], y)
        rows += [(g, d, i, f"probe_{name}", float(v)) for (g, d, i, _), v in zip(imgs, clf.decision_function(Fe))]
        log(f"history score: {name} done")
    npr = None if a.fake_models else M.load_npr(a)
    if npr is not None:
        rows += [(g, d, i, "npr", npr(Image.open(p))) for g, d, i, p in imgs]
        M.free(dev); log("history score: npr done")
    if not a.no_aeroblade:
        if a.fake_models:
            rows += [(g, d, i, "aeroblade", float(np.random.default_rng(i).normal())) for g, d, i, _ in imgs]
        else:
            det = X.Aeroblade(a)
            for j, (g, d, i, p) in enumerate(imgs):
                rows.append((g, d, i, "aeroblade", det(Image.open(p))))
                if j % 200 == 0:
                    log(f"history aeroblade {j}/{len(imgs)}")
            del det; M.free(dev)
    df = pd.DataFrame(rows, columns=["group", "decoder", "idx", "detector", "score"])
    assert np.isfinite(df.score).all()
    df.to_csv(fp, index=False); log(f"history score: {len(df)} rows")


def stage_report(a):
    import pandas as pd
    parts = [pd.read_csv(a.run / "scores.csv"), pd.read_csv(a.run / "history_scores.csv")]
    if (a.run / "aeroblade_scores.csv").exists():
        parts.append(pd.read_csv(a.run / "aeroblade_scores.csv"))
    df = pd.concat(parts)
    half = a.n_eff // 2
    L = []; P = lambda *s: L.append(" ".join(str(x) for x in s))
    P("# Decoder x denoising-history sweep\n")
    P(f"N = {a.n_eff} photos (probes: {a.n_eff - half} held-out photos). Strength 0 = real_enc "
      "(VAE-encoded photo, no denoising); 0.3 / 0.5 / 0.7 = img2img of the same photo, same seeds. "
      "D = mean paired score change (alt decoder - orig) in sd units; I = D(s) - D(0). "
      "[95% bootstrap CI over photos]. sd = std of untouched real + gen/orig scores, as in report.md.\n")
    # consistency: s50 orig decoded here must match the run's i2i/orig scores
    for det in sorted(df.detector.unique()):
        d = df[df.detector == det]
        if det.startswith("probe_"):
            d = d[d.idx >= half]
        S = lambda g, dec: d[(d.group == g) & (d.decoder == dec)].sort_values("idx").score.values
        real, gen = S("real", "none"), S("gen", "orig")
        if len(real) == 0 or len(S("h_s50", "orig")) == 0:
            continue
        sd = np.std(np.concatenate([real, gen])) + 1e-9
        base = M.auroc(real, gen)
        P(f"## `{det}` (baseline AUROC {base:.3f})\n")
        old = S("i2i", "orig")
        if len(old) == len(S("h_s50", "orig")):
            P(f"Consistency: max |score difference| s50/orig here vs the run's i2i/orig = "
              f"{np.max(np.abs(old - S('h_s50', 'orig'))):.2e} (should be ~0)\n")
        P("| decoder | strength | AUROC vs real (orig -> alt) | D [CI] | I = D(s) - D(0) [CI] |")
        P("|---|---|---|---|---|")
        for alt in ALTS:
            d0 = (S("real_enc", alt) - S("real_enc", "orig")) / sd
            m0 = M.boot_ci(d0)
            P(f"| {alt} | 0 | {M.auroc(real, S('real_enc', 'orig')):.3f} -> {M.auroc(real, S('real_enc', alt)):.3f} "
              f"| {m0[0]:+.2f} [{m0[1]:+.2f}, {m0[2]:+.2f}] | — |")
            for s in [0.3, 0.5, 0.7]:
                g = f"h_{sname(s)}"
                ds = (S(g, alt) - S(g, "orig")) / sd
                m, i_ = M.boot_ci(ds), M.boot_ci(ds - d0)
                P(f"| {alt} | {s} | {M.auroc(real, S(g, 'orig')):.3f} -> {M.auroc(real, S(g, alt)):.3f} "
                  f"| {m[0]:+.2f} [{m[1]:+.2f}, {m[2]:+.2f}] | **{i_[0]:+.2f}** [{i_[1]:+.2f}, {i_[2]:+.2f}] |")
        P("")
    # joint trend test: per photo, slope of D_k(s) over s in {0, .3, .5, .7}; mean slope vs 0
    P("## Joint trend test across strengths (Holm-corrected)\n")
    P("Per photo k: least-squares slope of the paired decoder effect D_k(s) over s = 0, 0.3, 0.5, 0.7. "
      "Test: mean slope = 0, sign-flip permutation over photos (10,000), Holm correction over all "
      "detector x decoder tests. Slope unit: sd per unit strength.\n")
    P("| detector | decoder | n | mean slope [95% CI] | p | p (Holm) |")
    P("|---|---|---|---|---|---|")
    tests, ss = [], np.array([0.0, 0.3, 0.5, 0.7])
    rng = np.random.default_rng(0)
    for det in sorted(df.detector.unique()):
        d = df[df.detector == det]
        if det.startswith("probe_"):
            d = d[d.idx >= half]
        S = lambda g, dec: d[(d.group == g) & (d.decoder == dec)].sort_values("idx").score.values
        real, gen = S("real", "none"), S("gen", "orig")
        if len(S("h_s30", "orig")) == 0:
            continue
        sd = np.std(np.concatenate([real, gen])) + 1e-9
        for alt in ALTS:
            D = np.stack([(S(g, alt) - S(g, "orig")) / sd
                          for g in ["real_enc", "h_s30", "h_s50", "h_s70"]], 1)   # [k, 4]
            sc = ss - ss.mean()
            slope = (D - D.mean(1, keepdims=True)) @ sc / (sc @ sc)
            m = slope.mean()
            flips = rng.choice([-1.0, 1.0], size=(10000, len(slope)))
            null = (flips * slope).mean(1)
            pv = (np.sum(np.abs(null) >= abs(m)) + 1) / (len(null) + 1)
            ci = M.boot_ci(slope)
            tests.append([det, alt, len(slope), m, ci, pv])
    order = np.argsort([t[5] for t in tests]); k = len(tests); run = 0.0
    holm = [None] * k
    for r, j in enumerate(order):
        run = max(run, min(1.0, (k - r) * tests[j][5])); holm[j] = run
    for t, h in zip(tests, holm):
        P(f"| {t[0]} | {t[1]} | {t[2]} | {t[3]:+.2f} [{t[4][1]:+.2f}, {t[4][2]:+.2f}] | {t[5]:.4f} | "
          f"{h:.4f}{' *' if h < 0.05 else ''} |")
    P("\n(* Holm-corrected p < 0.05. A linear slope summarizes a trend; it does not test for "
      "non-monotonic patterns.)\n")
    P("How to read: if I moves away from 0 as strength grows (0.3 -> 0.5 -> 0.7) for a detector, "
      "the decoder effect depends on how much denoising the latent went through. Exploratory: many "
      "comparisons, and higher strength also changes image content.\n")
    (a.run / "report_history.md").write_text("\n".join(L))
    log(f"report: wrote {a.run / 'report_history.md'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=Path("cache"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--stage", default="all", choices=["all", "latents", "decode", "score", "report"])
    ap.add_argument("--no-aeroblade", action="store_true", help="skip AEROBLADE (saves ~20 min)")
    ap.add_argument("--fake-models", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    a.device = M.pick_device(a.device)
    a.out = a.run
    meta = json.loads((a.run / "data" / "meta.json").read_text())
    a.n = a.n_eff = len(meta)
    a.res = Image.open(a.run / "data" / "real" / "0000.png").size[0]
    gs = a.run / "latents" / "gen_settings.json"
    a.steps = json.loads(gs.read_text()).get("steps", 30) if gs.exists() else 30
    M.SD_ID = "fake" if a.fake_models else M.resolve_sd(None)
    log(f"history: run={a.run} n={a.n_eff} steps={a.steps} device={a.device}")
    st = ["latents", "decode", "score", "report"] if a.stage == "all" else [a.stage]
    for s in st:
        globals()[f"stage_{s}"](a)


if __name__ == "__main__":
    main()
