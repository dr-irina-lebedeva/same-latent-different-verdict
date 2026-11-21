"""M4 band-swap + processing controls: which frequency band carries each detector's decoder effect,
and can ordinary processing (blur / noise / JPEG at matched severity) or spectrum matching reproduce it?

For the same latent, A = original-decoder image and B = swapped-decoder image (ft-MSE or TAESD).
Hybrid H_band = A with ONE radial frequency band replaced by B's (per colour channel, FFT).
If moving only band b reproduces most of f(B) - f(A), the detector reads the decoder in band b.

Bands (cycles/pixel, radius of the 2-D frequency): low < 0.08, mid 0.08-0.25, high >= 0.25.

    python dod_bandswap.py --run runs/n100 2>&1 | tee bandswap.log
Stages: make, score, report (default all; each resumes). Output: <run>/report_bandswap.md
"""
import argparse, json, sys, warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import dod_mac_pilot as M
import dod_extras as X

warnings.filterwarnings("ignore", message=".*encountered in matmul")
log = M.log
BANDS = {"low": (0.0, 0.08), "mid": (0.08, 0.25), "high": (0.25, 1.0)}
GROUPS = ["gen", "real_enc"]
ALTS = ["ftmse", "taesd"]


def band_mask(h, w, lo, hi):
    fy = np.fft.fftfreq(h)[:, None]; fx = np.fft.fftfreq(w)[None, :]
    r = np.sqrt(fx ** 2 + fy ** 2)
    return ((r >= lo) & (r < hi)).astype(np.float64)


def hybrid(a, b, mask):
    out = np.empty_like(a)
    for c in range(3):
        Fa, Fb = np.fft.fft2(a[..., c]), np.fft.fft2(b[..., c])
        out[..., c] = np.real(np.fft.ifft2(Fa * (1 - mask) + Fb * mask))
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


CONTROLS = ["specmatch", "blur", "noise", "jpeg"]   # applied to A (original decoder), per alt decoder


def radial_bins(h, w, nb=64):
    fy = np.fft.fftfreq(h)[:, None]; fx = np.fft.fftfreq(w)[None, :]
    r = np.sqrt(fx ** 2 + fy ** 2)
    return np.minimum((r / r.max() * nb).astype(int), nb - 1)


def spec_match(a, b, bins):
    """A's phase and fine structure, with B's radially averaged amplitude profile (per channel)."""
    out = np.empty_like(a); nb = bins.max() + 1
    for c in range(3):
        Fa, Fb = np.fft.fft2(a[..., c]), np.fft.fft2(b[..., c])
        pa = np.bincount(bins.ravel(), np.abs(Fa).ravel() ** 2, nb)
        pb = np.bincount(bins.ravel(), np.abs(Fb).ravel() ** 2, nb)
        g = np.sqrt(pb / np.maximum(pa, 1e-12))[bins]
        g[0, 0] = 1.0
        out[..., c] = np.real(np.fft.ifft2(Fa * g))
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def match_control(kind, A, B):
    """Ordinary processing of A with severity chosen so its pixel MSE to A matches that of B."""
    import io
    from PIL import ImageFilter
    target = np.mean((A - B) ** 2)
    def mse(x): return np.mean((A - x.astype(np.float64)) ** 2)
    if kind == "noise":
        rng = np.random.default_rng(0)
        return np.clip(np.round(A + rng.normal(0, np.sqrt(target), A.shape)), 0, 255).astype(np.uint8)
    cands = []
    if kind == "blur":
        for sig in [0.3, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0]:
            cands.append(np.asarray(Image.fromarray(A.astype(np.uint8)).filter(ImageFilter.GaussianBlur(sig))))
    if kind == "jpeg":
        for q in [95, 90, 80, 70, 50, 30, 15]:
            buf = io.BytesIO(); Image.fromarray(A.astype(np.uint8)).save(buf, "JPEG", quality=q)
            cands.append(np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB")))
    return min(cands, key=lambda x: abs(mse(x) - target))


def out_dir(a, g, alt, band):
    return a.run / "png" / "bandswap" / g / alt / band


def stage_make(a):
    P = a.run / "png"
    masks = {}
    for g in GROUPS:
        for alt in ALTS:
            for band, (lo, hi) in BANDS.items():
                o = out_dir(a, g, alt, band)
                if o.exists() and len(list(o.glob("*.png"))) == a.n_eff:
                    continue
                o.mkdir(parents=True, exist_ok=True)
                for i in range(a.n_eff):
                    A = np.asarray(Image.open(P / g / "orig" / f"{i:04d}.png"), dtype=np.float64)
                    B = np.asarray(Image.open(P / g / alt / f"{i:04d}.png"), dtype=np.float64)
                    key = (A.shape[0], A.shape[1], band)
                    if key not in masks:
                        masks[key] = band_mask(A.shape[0], A.shape[1], lo, hi)
                    Image.fromarray(hybrid(A, B, masks[key])).save(o / f"{i:04d}.png")
                log(f"bandswap make: {g}/{alt}/{band} done")
    for g in GROUPS:
        for alt in ALTS:
            for kind in CONTROLS:
                o = out_dir(a, g, alt, kind)
                if o.exists() and len(list(o.glob("*.png"))) == a.n_eff:
                    continue
                o.mkdir(parents=True, exist_ok=True)
                for i in range(a.n_eff):
                    A = np.asarray(Image.open(P / g / "orig" / f"{i:04d}.png"), dtype=np.float64)
                    B = np.asarray(Image.open(P / g / alt / f"{i:04d}.png"), dtype=np.float64)
                    x = spec_match(A, B, radial_bins(*A.shape[:2])) if kind == "specmatch" \
                        else match_control(kind, A, B)
                    Image.fromarray(x).save(o / f"{i:04d}.png")
                log(f"bandswap make: control {g}/{alt}/{kind} done")
    # sanity: all three bands together must give back B (up to rounding)
    A = np.asarray(Image.open(P / "gen" / "orig" / "0000.png"), dtype=np.float64)
    B = np.asarray(Image.open(P / "gen" / "taesd" / "0000.png"), dtype=np.float64)
    full = sum(band_mask(*A.shape[:2], lo, hi) for lo, hi in BANDS.values())
    err = np.abs(hybrid(A, B, full).astype(float) - B).max()
    log(f"bandswap sanity: all bands swapped vs B, max |diff| = {err:.1f} (should be <= 1)")
    if err > 1.5:
        sys.exit("STOP: band masks do not cover the spectrum. Send me the log.")


def images(a):
    return [(f"bs_{g}_{band}", alt, i, out_dir(a, g, alt, band) / f"{i:04d}.png")
            for g in GROUPS for alt in ALTS for band in list(BANDS) + CONTROLS for i in range(a.n_eff)]


def stage_score(a):
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    fp = a.run / "bandswap_scores.csv"
    imgs = images(a)
    if fp.exists() and len(pd.read_csv(fp)) >= len(imgs) * 4:
        log("bandswap score: already done"); return
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
        half = a.n_eff // 2
        tr = [j for j, (g, d, i, _) in enumerate(main)
              if i < half and ((g == "real") or (g == "gen" and d == "orig"))]
        y = [0 if main[j][0] == "real" else 1 for j in tr]
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=5000)).fit(Fm[tr], y)
        rows += [(g, d, i, f"probe_{name}", float(v)) for (g, d, i, _), v in zip(imgs, clf.decision_function(Fe))]
        log(f"bandswap score: {name} done")
    npr = None if a.fake_models else M.load_npr(a)
    if npr is not None:
        rows += [(g, d, i, "npr", npr(Image.open(p))) for g, d, i, p in imgs]
        M.free(dev); log("bandswap score: npr done")
    if not a.no_aeroblade:
        if a.fake_models:
            rows += [(g, d, i, "aeroblade", float(np.random.default_rng(i + len(g)).normal()))
                     for g, d, i, _ in imgs]
        else:
            det = X.Aeroblade(a)
            for j, (g, d, i, p) in enumerate(imgs):
                rows.append((g, d, i, "aeroblade", det(Image.open(p))))
                if j % 300 == 0:
                    log(f"bandswap aeroblade {j}/{len(imgs)}")
            del det; M.free(dev)
    df = pd.DataFrame(rows, columns=["group", "decoder", "idx", "detector", "score"])
    assert np.isfinite(df.score).all()
    df.to_csv(fp, index=False); log(f"bandswap score: {len(df)} rows")


def stage_report(a):
    import pandas as pd
    parts = [pd.read_csv(a.run / "scores.csv"), pd.read_csv(a.run / "bandswap_scores.csv")]
    if (a.run / "aeroblade_scores.csv").exists():
        parts.append(pd.read_csv(a.run / "aeroblade_scores.csv"))
    df = pd.concat(parts)
    half = a.n_eff // 2
    L = []; P = lambda *s: L.append(" ".join(str(x) for x in s))
    P("# Band-swap test (M4)\n")
    P(f"N = {a.n_eff} latents (probes: {a.n_eff - half} held-out). For each latent: A = original decoder, "
      "B = swapped decoder, H_band = A with one radial frequency band taken from B "
      f"(cycles/pixel: {', '.join(f'{k} {v[0]}-{v[1]}' for k, v in BANDS.items())}). "
      "Effect = mean paired score change vs A, sd units [95% CI]. 'share' = band effect / full B effect.\n")
    for det in sorted(df.detector.unique()):
        d = df[df.detector == det]
        if det.startswith("probe_"):
            d = d[d.idx >= half]
        S = lambda g, dec: d[(d.group == g) & (d.decoder == dec)].sort_values("idx").score.values
        real, gen = S("real", "none"), S("gen", "orig")
        if len(S("bs_gen_low", "taesd")) == 0:
            continue
        sd = np.std(np.concatenate([real, gen])) + 1e-9
        P(f"## `{det}`\n")
        P("| latents | decoder | full swap B | low band | mid band | high band | low+mid+high |")
        P("|---|---|---|---|---|---|---|")
        for g in GROUPS:
            for alt in ALTS:
                A = S(g, "orig"); full = (S(g, alt) - A) / sd
                fb = M.boot_ci(full)
                cells, tot = [], 0.0
                for band in BANDS:
                    e = (S(f"bs_{g}_{band}", alt) - A) / sd
                    m = M.boot_ci(e); tot += m[0]
                    share = m[0] / fb[0] if abs(fb[0]) > 0.05 else float("nan")
                    cells.append(f"{m[0]:+.2f} [{m[1]:+.2f}, {m[2]:+.2f}] ({share:.0%})"
                                 if np.isfinite(share) else f"{m[0]:+.2f} [{m[1]:+.2f}, {m[2]:+.2f}]")
                P(f"| {g} | {alt} | {fb[0]:+.2f} [{fb[1]:+.2f}, {fb[2]:+.2f}] | " + " | ".join(cells)
                  + f" | {tot:+.2f} |")
        P("\n**Controls** (applied to A; blur / noise / JPEG severity matched per image to the pixel MSE "
          "between A and B; specmatch = A with B's radial amplitude spectrum). Does ordinary processing, "
          "or B's global spectrum alone, reproduce the decoder effect?\n")
        P("| latents | decoder | full swap B | " + " | ".join(CONTROLS) + " |")
        P("|---|---|---|" + "---|" * len(CONTROLS))
        for g in GROUPS:
            for alt in ALTS:
                A = S(g, "orig"); fb = M.boot_ci((S(g, alt) - A) / sd)
                cells = []
                for kind in CONTROLS:
                    m = M.boot_ci((S(f"bs_{g}_{kind}", alt) - A) / sd)
                    share = m[0] / fb[0] if abs(fb[0]) > 0.05 else float("nan")
                    cells.append(f"{m[0]:+.2f} [{m[1]:+.2f}, {m[2]:+.2f}]" +
                                 (f" ({share:.0%})" if np.isfinite(share) else ""))
                P(f"| {g} | {alt} | {fb[0]:+.2f} [{fb[1]:+.2f}, {fb[2]:+.2f}] | " + " | ".join(cells) + " |")
        P("")
    P("How to read: a band whose share is near 100% carries the detector's decoder effect. "
      "'low+mid+high' close to 'full swap' means the bands act additively; a large gap means the "
      "detector responds to combinations of bands. Shares are unstable when the full effect is near 0.\n")
    (a.run / "report_bandswap.md").write_text("\n".join(L))
    log(f"report: wrote {a.run / 'report_bandswap.md'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=Path("cache"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--stage", default="all", choices=["all", "make", "score", "report"])
    ap.add_argument("--no-aeroblade", action="store_true", help="skip AEROBLADE (saves ~20 min)")
    ap.add_argument("--fake-models", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    a.device = M.pick_device(a.device)
    a.out = a.run
    meta = json.loads((a.run / "data" / "meta.json").read_text())
    a.n = a.n_eff = len(meta)
    a.res = Image.open(a.run / "data" / "real" / "0000.png").size[0]
    M.SD_ID = "fake" if a.fake_models else M.resolve_sd(None)
    log(f"bandswap: run={a.run} n={a.n_eff} device={a.device}")
    st = ["make", "score", "report"] if a.stage == "all" else [a.stage]
    for s in st:
        globals()[f"stage_{s}"](a)


if __name__ == "__main__":
    main()
