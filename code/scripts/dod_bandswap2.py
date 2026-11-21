"""M4 v2: validation of the band-swap intervention (after dod_bandswap.py).

Adds, for the same latents (A = original decoder, B = swapped decoder):
  soft_<band>   A with band from B, SOFT (raised-cosine) band edges -> no ringing from hard masks
  comp_<band>   complement: B with band taken back from A  ("B without B's band")
                effect of B's band = f(B) - f(comp_band)
  lp_<kind>     blur / noise / JPEG on A, severity matched to LPIPS(A, B) instead of pixel MSE
Also reports image quality of every hybrid (PSNR and LPIPS vs A and vs B), and writes
bandswap2_grid.png for a visual check.

    python dod_bandswap2.py --run runs/n100 2>&1 | tee bandswap2.log
Output: <run>/report_bandswap2.md   (~1.5-2 h on an M4; --no-aeroblade saves ~1 h)
"""
import argparse, io, json, sys, warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter

import dod_mac_pilot as M
import dod_bandswap as B1

warnings.filterwarnings("ignore", message=".*encountered in matmul")
log = M.log
BANDS = B1.BANDS
GROUPS, ALTS = B1.GROUPS, B1.ALTS
EDGE = 0.02          # half-width of the raised-cosine transition, cycles/pixel
VARIANTS = [f"soft_{b}" for b in BANDS] + [f"comp_{b}" for b in BANDS] + ["lp_blur", "lp_noise", "lp_jpeg"]


def soft_mask(h, w, lo, hi):
    fy = np.fft.fftfreq(h)[:, None]; fx = np.fft.fftfreq(w)[None, :]
    r = np.sqrt(fx ** 2 + fy ** 2)
    def step(x, c):   # 0 below c-EDGE, 1 above c+EDGE, raised cosine between
        t = np.clip((x - (c - EDGE)) / (2 * EDGE), 0, 1)
        return 0.5 - 0.5 * np.cos(np.pi * t)
    up = step(r, lo) if lo > 0 else np.ones_like(r)
    down = 1 - step(r, hi) if hi < 1 else np.ones_like(r)
    return up * down


def out_dir(a, g, alt, v):
    return a.run / "png" / "bandswap2" / g / alt / v


class Lp:
    def __init__(self, dev, fake=False):
        self.fake = fake
        if fake:
            return
        import lpips
        self.dev = dev
        self.m = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()

    @torch.no_grad()
    def __call__(self, x, y):
        if self.fake:   # test only: MSE stand-in
            return float(np.mean((np.asarray(x, np.float64) - np.asarray(y, np.float64)) ** 2) / 1e4)
        t = lambda z: torch.from_numpy(np.asarray(z, np.float32) / 127.5 - 1).permute(2, 0, 1)[None].to(self.dev)
        return float(self.m(t(x), t(y)).mean().cpu())


def lp_control(kind, A, Bimg, lp, target):
    Au = A.astype(np.uint8)
    if kind == "blur":
        cands = [np.asarray(Image.fromarray(Au).filter(ImageFilter.GaussianBlur(s)))
                 for s in [0.3, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0]]
    elif kind == "noise":
        rng = np.random.default_rng(0)
        cands = [np.clip(np.round(A + rng.normal(0, s, A.shape)), 0, 255).astype(np.uint8)
                 for s in [1, 2, 3, 5, 8, 12, 18]]
    else:
        cands = []
        for q in [95, 90, 80, 70, 50, 30, 15]:
            buf = io.BytesIO(); Image.fromarray(Au).save(buf, "JPEG", quality=q)
            cands.append(np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB")))
    return min(cands, key=lambda x: abs(lp(Au, x) - target))


def stage_make(a):
    P = a.run / "png"
    lp = Lp(a.device, a.fake_models)
    masks = {}
    qual = []
    for g in GROUPS:
        for alt in ALTS:
            todo = [v for v in VARIANTS if not (out_dir(a, g, alt, v).exists()
                    and len(list(out_dir(a, g, alt, v).glob("*.png"))) == a.n_eff)]
            for v in todo:
                out_dir(a, g, alt, v).mkdir(parents=True, exist_ok=True)
            for i in range(a.n_eff):
                A = np.asarray(Image.open(P / g / "orig" / f"{i:04d}.png"), dtype=np.float64)
                Bx = np.asarray(Image.open(P / g / alt / f"{i:04d}.png"), dtype=np.float64)
                target = lp(A.astype(np.uint8), Bx.astype(np.uint8))
                for v in VARIANTS:
                    p = out_dir(a, g, alt, v) / f"{i:04d}.png"
                    if v in todo:
                        kind, name = v.split("_", 1)
                        if kind in ("soft", "comp"):
                            key = (A.shape[0], A.shape[1], name)
                            if key not in masks:
                                masks[key] = soft_mask(A.shape[0], A.shape[1], *BANDS[name])
                            x = B1.hybrid(A, Bx, masks[key]) if kind == "soft" else B1.hybrid(Bx, A, masks[key])
                        else:
                            x = lp_control(name, A, Bx, lp, target)
                        Image.fromarray(x).save(p)
                    if i < 20:   # image quality of hybrids on the first 20 latents
                        x = np.asarray(Image.open(p), dtype=np.float64)
                        qual.append(dict(group=g, alt=alt, variant=v,
                                         psnr_A=10 * np.log10(255 ** 2 / max(np.mean((x - A) ** 2), 1e-9)),
                                         psnr_B=10 * np.log10(255 ** 2 / max(np.mean((x - Bx) ** 2), 1e-9)),
                                         lpips_A=lp(A.astype(np.uint8), x.astype(np.uint8)),
                                         lpips_B=lp(Bx.astype(np.uint8), x.astype(np.uint8)),
                                         lpips_AB=target))
            log(f"bandswap2 make: {g}/{alt} done")
    import pandas as pd
    pd.DataFrame(qual).to_csv(a.run / "bandswap2_quality.csv", index=False)
    # sanity: soft bands must still sum to the full spectrum
    full = sum(soft_mask(a.res, a.res, *v) for v in BANDS.values())
    log(f"bandswap2 sanity: soft masks sum to 1 within {np.abs(full - 1).max():.2e}")
    if np.abs(full - 1).max() > 1e-6:
        sys.exit("STOP: soft masks do not partition the spectrum. Send me the log.")
    # grid for a visual check: rows = 4 latents, cols = A, B, soft bands, LPIPS controls
    cols = ["orig", "alt"] + [f"soft_{b}" for b in BANDS] + ["lp_blur", "lp_noise", "lp_jpeg"]
    tiles = []
    for i in range(min(4, a.n_eff)):
        row = []
        for c in cols:
            p = (P / "gen" / "orig" / f"{i:04d}.png") if c == "orig" else \
                (P / "gen" / "taesd" / f"{i:04d}.png") if c == "alt" else out_dir(a, "gen", "taesd", c) / f"{i:04d}.png"
            row.append(np.asarray(Image.open(p).resize((224, 224))))
        tiles.append(np.concatenate(row, 1))
    Image.fromarray(np.concatenate(tiles, 0)).save(a.run / "bandswap2_grid.png")
    (a.run / "bandswap2_grid_columns.txt").write_text(", ".join(cols) + "  (gen latents, TAESD)\n")
    log("bandswap2: wrote bandswap2_grid.png")
    del lp; M.free(a.device)


def images(a):
    return [(f"b2_{g}_{v}", alt, i, out_dir(a, g, alt, v) / f"{i:04d}.png")
            for g in GROUPS for alt in ALTS for v in VARIANTS for i in range(a.n_eff)]


def stage_score(a):
    fp = a.run / "bandswap2_scores.csv"
    import pandas as pd
    imgs = images(a)
    if fp.exists() and len(pd.read_csv(fp)) >= len(imgs) * 4:
        log("bandswap2 score: already done"); return
    B1.images_saved = B1.images
    B1.images = lambda _a: imgs          # reuse the v1 scorer on the new image list
    try:
        old = a.run / "bandswap_scores.csv"
        keep = old.read_bytes() if old.exists() else None
        if old.exists():
            old.rename(a.run / "bandswap_scores.v1.csv")
        B1.stage_score(a)
        (a.run / "bandswap_scores.csv").rename(fp)
    finally:
        if (a.run / "bandswap_scores.v1.csv").exists():
            (a.run / "bandswap_scores.v1.csv").rename(a.run / "bandswap_scores.csv")
        B1.images = B1.images_saved


def stage_report(a):
    import pandas as pd
    parts = [pd.read_csv(a.run / "scores.csv"), pd.read_csv(a.run / "bandswap2_scores.csv")]
    if (a.run / "aeroblade_scores.csv").exists():
        parts.append(pd.read_csv(a.run / "aeroblade_scores.csv"))
    df = pd.concat(parts)
    q = pd.read_csv(a.run / "bandswap2_quality.csv")
    half = a.n_eff // 2
    L = []; P = lambda *s: L.append(" ".join(str(x) for x in s))
    P("# Band-swap validation (M4 v2)\n")
    P(f"N = {a.n_eff} latents (probes: {a.n_eff - half} held-out). A = original decoder, B = swapped decoder. "
      f"soft_band: A with B's band, raised-cosine edges (±{EDGE} cyc/px). comp_band: B with that band "
      "taken back from A; 'B-band effect' = f(B) - f(comp_band), i.e. what B loses without its own band. "
      "lp_*: blur / noise / JPEG on A at LPIPS matched to LPIPS(A,B). Effects in sd units [95% CI]. "
      "Shares are not exclusive contributions: detectors are nonlinear, so one band can exceed 100%.\n")
    P("## Image quality of the interventions (first 20 latents, mean)\n")
    P("| variant | PSNR vs A | PSNR vs B | LPIPS vs A | LPIPS vs B | LPIPS(A,B) |")
    P("|---|---|---|---|---|---|")
    for v, r in q.groupby("variant")[["psnr_A", "psnr_B", "lpips_A", "lpips_B", "lpips_AB"]].mean().iterrows():
        P(f"| {v} | {r.psnr_A:.1f} | {r.psnr_B:.1f} | {r.lpips_A:.3f} | {r.lpips_B:.3f} | {r.lpips_AB:.3f} |")
    P("")
    for det in sorted(df.detector.unique()):
        d = df[df.detector == det]
        if det.startswith("probe_"):
            d = d[d.idx >= half]
        S = lambda g, dec: d[(d.group == g) & (d.decoder == dec)].sort_values("idx").score.values
        real, gen = S("real", "none"), S("gen", "orig")
        if len(S("b2_gen_soft_low", "taesd")) == 0:
            continue
        sd = np.std(np.concatenate([real, gen])) + 1e-9
        P(f"## `{det}`\n")
        P("| latents | decoder | full B - A | " + " | ".join(f"soft {b}" for b in BANDS) + " | "
          + " | ".join(f"B-band {b}" for b in BANDS) + " | lp blur | lp noise | lp jpeg |")
        P("|---|---|---|" + "---|" * 9)
        for g in GROUPS:
            for alt in ALTS:
                A, Bs = S(g, "orig"), S(g, alt)
                cells = [M.boot_ci((Bs - A) / sd)]
                cells += [M.boot_ci((S(f"b2_{g}_soft_{b}", alt) - A) / sd) for b in BANDS]
                cells += [M.boot_ci((Bs - S(f"b2_{g}_comp_{b}", alt)) / sd) for b in BANDS]
                cells += [M.boot_ci((S(f"b2_{g}_lp_{k}", alt) - A) / sd) for k in ["blur", "noise", "jpeg"]]
                P(f"| {g} | {alt} | " + " | ".join(f"{m:+.2f} [{lo:+.2f}, {hi:+.2f}]" for m, lo, hi in cells) + " |")
        P("")
    P("How to read: if soft-mask effects match the hard-mask ones (report_bandswap.md), ringing was not "
      "driving them. 'soft band' and 'B-band' agreeing means the band acts the same whether added to A or "
      "removed from B. If LPIPS-matched controls still do not reproduce the decoder pattern across "
      "detectors, the 'not ordinary processing' conclusion holds under perceptual matching too. "
      "Check bandswap2_grid.png by eye.\n")
    (a.run / "report_bandswap2.md").write_text("\n".join(L))
    log(f"report: wrote {a.run / 'report_bandswap2.md'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=Path("cache"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--stage", default="all", choices=["all", "make", "score", "report"])
    ap.add_argument("--no-aeroblade", action="store_true")
    ap.add_argument("--fake-models", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    a.device = M.pick_device(a.device)
    a.out = a.run
    meta = json.loads((a.run / "data" / "meta.json").read_text())
    a.n = a.n_eff = len(meta)
    a.res = Image.open(a.run / "data" / "real" / "0000.png").size[0]
    M.SD_ID = "fake" if a.fake_models else M.resolve_sd(None)
    log(f"bandswap2: run={a.run} n={a.n_eff} device={a.device}")
    for s in (["make", "score", "report"] if a.stage == "all" else [a.stage]):
        globals()[f"stage_{s}"](a)


if __name__ == "__main__":
    main()
