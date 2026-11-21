"""E3: LatentLock on SDXL. Same latent, three decoders, the paper's detectors. Pilot first (30 sources), then full (300).

Design (fixed in advance in protocols/PLAN_SDXL.md):
  generator  stabilityai/stable-diffusion-xl-base-1.0 (no refiner), 1024x1024 (native), 30 steps, guidance 5.0,
             default scheduler, fp16 UNet, seed 5000+i, prompt = first COCO caption of the source photo
  decoders   orig   = SDXL VAE (fp32, avoids the known fp16 overflow)
             fp16fix= madebyollin/sdxl-vae-fp16-fix, run in fp32 -> near-identity CONTROL (differs in weights only)
             taesdxl= madebyollin/taesdxl -> the decoder-family INTERVENTION
             every decoder receives the identical saved latent
  scoring    448x448 centre crop of each 1024 image, no resizing: the same preprocessing as the paper's real
             photographs (COCO photos are <= 640 px, so 1024 real photographs do not exist in COCO). Full 1024
             images are kept for figures.
  sources    COCO val2017 photos with shorter side >= 448 that were never used before (frozen/used_coco_ids.json)
  calibration FRESH independent COCO photos (disjoint from everything), same 448 crop; the paper's 1,000
             calibration photos are scored too and reported as a secondary threshold
  detectors  AEROBLADE (paper: SD1.5 AE), AEROBLADE-SDXL (SDXL AE, matched), NPR, UnivFD, CLIP/DINO probes
             (refit exactly as in the paper on the SD1.5 run; needs results of run_e4.sh), Community-Forensics, B-Free
  round-trip the untouched evaluation photos encoded by the SDXL encoder and decoded by each decoder

Stages: data, gen, decode, gate, score, report.   python dod_sdxl.py --out runs/sdxl_pilot --n 30 --calib 200
"""
import argparse, io, json, math, os, random, sys, time, urllib.request, zipfile
from pathlib import Path
import numpy as np, pandas as pd, torch
from PIL import Image
import dod_mac_pilot as M
import dod_v3 as V3
import dod_e4 as E4

SDXL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
FIX_ID = "madebyollin/sdxl-vae-fp16-fix"
TAEXL_ID = "madebyollin/taesdxl"
SF = 0.13025
DECS = ["orig", "fp16fix", "taesdxl"]
RES, CROP = 1024, 448
FROZEN = Path(__file__).resolve().parent / "frozen"
log = M.log


# ---------------------------------------------------------------- data
TRAIN_URL = "http://images.cocodataset.org/train2017/{:012d}.jpg"


def ann_json(a, split):
    cache = a.cache / f"captions_{split}2017.json"
    if not cache.exists():
        a.cache.mkdir(parents=True, exist_ok=True); zp = a.cache / "ann.zip"
        if not zp.exists():
            urllib.request.urlretrieve(M.COCO_ANN_URL, zp)
        with zipfile.ZipFile(zp) as z:
            for sp in ("val", "train"):
                (a.cache / f"captions_{sp}2017.json").write_bytes(z.read(f"annotations/captions_{sp}2017.json"))
    return json.loads(cache.read_text())


def coco(a):
    """Sources: unused COCO val2017 photos (as in the paper). Fresh calibration: COCO train2017 photos (same camera
    population and the same 448 crop, disjoint from val2017 by construction), because val2017 has too few unused photos."""
    ann = ann_json(a, "val")
    caps = {}
    for c in ann["annotations"]:
        caps.setdefault(c["image_id"], c["caption"].strip())
    used = set(json.loads((FROZEN / "used_coco_ids.json").read_text())["ids"])
    pool = sorted(im["id"] for im in ann["images"] if min(im["width"], im["height"]) >= CROP and im["id"] in caps and im["id"] not in used)
    random.Random(2027).shuffle(pool)       # new seed: different from the main (0) and calibration (1) samples
    tr = ann_json(a, "train")
    cpool = sorted(im["id"] for im in tr["images"] if min(im["width"], im["height"]) >= CROP)
    random.Random(2028).shuffle(cpool)
    return pool, caps, cpool


def fetch(iid, url=None):
    raw = urllib.request.urlopen((url or M.COCO_IMG_URL).format(iid), timeout=30).read()
    im = Image.open(io.BytesIO(raw)).convert("RGB"); w, h = im.size
    l, t = (w - CROP) // 2, (h - CROP) // 2
    return im.crop((l, t, l + CROP, t + CROP)), [w, h], [l, t, l + CROP, t + CROP]


def stage_data(a):
    d = a.out / "data"; mp = d / "meta.json"
    if mp.exists() and len(json.loads(mp.read_text())["eval"]) >= a.n:
        log("data: already done"); return
    (d / "real").mkdir(parents=True, exist_ok=True); (d / "calib").mkdir(parents=True, exist_ok=True)
    if a.fake_models:
        rng = np.random.default_rng(0)
        for sub, n in [("real", a.n), ("calib", a.calib)]:
            for k in range(n):
                Image.fromarray(rng.integers(0, 255, (CROP, CROP, 3), dtype=np.uint8)).save(d / sub / f"{k:04d}.png")
        mp.write_text(json.dumps(dict(eval=[dict(k=k, coco_id=k, caption="a photo") for k in range(a.n)],
                                      calib=[dict(k=k, coco_id=10000 + k) for k in range(a.calib)], pool=a.n + a.calib)))
        return
    pool, caps, cpool = coco(a)
    log(f"data: {len(pool)} unused val2017 photos for {a.n} sources; {len(cpool)} train2017 photos for {a.calib} calibration")
    if len(pool) < a.n:
        sys.exit("STOP: not enough unused val2017 photos")
    ev, ca = [], []
    for lst, sub, n, src, url in [(ev, "real", a.n, pool, None), (ca, "calib", a.calib, cpool, TRAIN_URL)]:
        for iid in src:
            if len(lst) >= n:
                break
            try:
                im, size, box = fetch(iid, url)
            except Exception as e:
                log("skip", iid, e); continue
            k = len(lst); im.save(d / sub / f"{k:04d}.png")
            lst.append(dict(k=k, coco_id=iid, split="val2017" if url is None else "train2017", caption=caps.get(iid, ""), orig_size=size, crop_box=box))
    mp.write_text(json.dumps(dict(eval=ev, calib=ca, pool=len(pool)), indent=1))
    assert not ({m["coco_id"] for m in ev} & {m["coco_id"] for m in ca})
    log(f"data: {len(ev)} sources, {len(ca)} fresh calibration photos")


# ---------------------------------------------------------------- generation
def stage_gen(a):
    meta = json.loads((a.out / "data" / "meta.json").read_text())["eval"]
    zd = a.out / "latents"; zd.mkdir(parents=True, exist_ok=True)
    todo = [m for m in meta if not (zd / f"{m['k']:04d}.pt").exists()]
    if not todo:
        log("gen: already done"); return
    if a.fake_models:
        for m in todo:
            torch.save(torch.randn(1, 4, RES // 8, RES // 8, generator=torch.Generator().manual_seed(5000 + m["k"])), zd / f"{m['k']:04d}.pt")
        return
    from diffusers import StableDiffusionXLPipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(SDXL_ID, torch_dtype=torch.float16, variant="fp16").to(a.device)
    pipe.set_progress_bar_config(disable=True)
    t0 = time.time()
    for j, m in enumerate(todo):
        g = torch.Generator(a.device).manual_seed(5000 + m["k"])
        z = pipe(prompt=m["caption"], height=RES, width=RES, num_inference_steps=30, guidance_scale=5.0,
                 generator=g, output_type="latent").images          # scaled latent (what the UNet sees)
        assert torch.isfinite(z).all()
        torch.save(z.float().cpu(), zd / f"{m['k']:04d}.pt")
        if j % 10 == 0:
            log(f"gen {j}/{len(todo)} ({(time.time() - t0) / (j + 1):.1f} s/image)")
    (a.out / "gen_info.json").write_text(json.dumps(dict(model=SDXL_ID, steps=30, guidance=5.0, res=RES, seed="5000+k",
                                                         scheduler=type(pipe.scheduler).__name__,
                                                         s_per_image=(time.time() - t0) / len(todo))))
    del pipe; M.free(a.device)


# ---------------------------------------------------------------- decoders
def load_dec(name, dev, fake):
    from diffusers import AutoencoderKL, AutoencoderTiny
    if fake:
        torch.manual_seed(DECS.index(name))
        if name == "taesdxl":
            return AutoencoderTiny(encoder_block_out_channels=(8,) * 4, decoder_block_out_channels=(8,) * 4,
                                   num_encoder_blocks=(1,) * 4, num_decoder_blocks=(1,) * 4).to(dev).eval()
        return AutoencoderKL(block_out_channels=(8,) * 4, layers_per_block=1, down_block_types=("DownEncoderBlock2D",) * 4,
                             up_block_types=("UpDecoderBlock2D",) * 4, latent_channels=4, norm_num_groups=4,
                             scaling_factor=SF).to(dev).eval()
    if name == "orig":
        return AutoencoderKL.from_pretrained(SDXL_ID, subfolder="vae", torch_dtype=torch.float32).to(dev).eval()
    if name == "fp16fix":
        return AutoencoderKL.from_pretrained(FIX_ID, torch_dtype=torch.float32).to(dev).eval()
    return AutoencoderTiny.from_pretrained(TAEXL_ID, torch_dtype=torch.float32).to(dev).eval()


@torch.no_grad()
def dec(vae, name, zs, tiny_scaled=True):
    """zs: scaled latent. AutoencoderKL gets zs / 0.13025; TAESDXL gets the convention chosen by the scale check."""
    if name == "taesdxl":
        x = vae.decode(zs if tiny_scaled else zs / SF).sample
    else:
        x = vae.decode(zs / vae.config.scaling_factor).sample
    return x.float().clamp(-1, 1)


@torch.no_grad()
def enc(vae, x):
    return vae.encode(x).latent_dist.mean * vae.config.scaling_factor   # scaled latent


def crop(img):
    w, h = img.size; l, t = (w - CROP) // 2, (h - CROP) // 2
    return img.crop((l, t, l + CROP, t + CROP))


def psnr(x, y):
    return float(10 * np.log10(4.0 / torch.mean((x.float() - y.float()) ** 2).item()))


def stage_decode(a):
    meta = json.loads((a.out / "data" / "meta.json").read_text())["eval"]
    O = a.out; chk = {}
    # TAESDXL latent convention: pick the one closest to the original decoder on the first 5 latents
    zs5 = [torch.load(O / "latents" / f"{m['k']:04d}.pt").to(a.device) for m in meta[:5]]
    vo = load_dec("orig", a.device, a.fake_models); ref = [dec(vo, "orig", z) for z in zs5]; del vo
    vt = load_dec("taesdxl", a.device, a.fake_models)
    ps = {conv: float(np.mean([psnr(dec(vt, "taesdxl", z, conv), r) for z, r in zip(zs5, ref)])) for conv in [True, False]}
    del vt, ref; M.free(a.device)
    a.tiny_scaled = max(ps, key=ps.get); chk["taesdxl_scale_psnr"] = {"scaled": ps[True], "unscaled": ps[False]}
    chk["taesdxl_takes_scaled_latent"] = a.tiny_scaled
    real = [Image.open(O / "data" / "real" / f"{m['k']:04d}.png").convert("RGB") for m in meta]
    vo = load_dec("orig", a.device, a.fake_models)
    zr = [enc(vo, M.to_tensor(im).to(a.device)) for im in real]          # SDXL encoder for the round-trip
    for name in DECS:
        v = vo if name == "orig" else load_dec(name, a.device, a.fake_models)
        for m, z in zip(meta, zr):
            k = m["k"]; p1 = O / "png1024" / "gen" / name / f"{k:04d}.png"; p2 = O / "png" / "gen" / name / f"{k:04d}.png"
            p3 = O / "png" / "real_enc" / name / f"{k:04d}.png"
            for p in (p1, p2, p3):
                p.parent.mkdir(parents=True, exist_ok=True)
            if not p2.exists():
                x = dec(v, name, torch.load(O / "latents" / f"{k:04d}.pt").to(a.device), a.tiny_scaled)
                assert torch.isfinite(x).all() and (a.fake_models or float(x.std()) > 0.02), f"bad decode {name} {k}"
                im = M.to_pil(x); im.save(p1); crop(im).save(p2)
            if not p3.exists():
                M.to_pil(dec(v, name, z, a.tiny_scaled)).save(p3)
        if name != "orig":
            del v
        M.free(a.device); log(f"decode: {name} done")
    del vo; M.free(a.device)
    # near-identity control and intervention size, on the 1024 images
    for name in ["fp16fix", "taesdxl"]:
        chk[f"psnr_{name}_vs_orig_1024"] = float(np.mean([M.psnr(O / "png1024" / "gen" / name / f"{m['k']:04d}.png",
                                                                 O / "png1024" / "gen" / "orig" / f"{m['k']:04d}.png") for m in meta]))
    chk["psnr_roundtrip_orig_vs_real"] = float(np.mean([M.psnr(O / "png" / "real_enc" / "orig" / f"{m['k']:04d}.png",
                                                               O / "data" / "real" / f"{m['k']:04d}.png") for m in meta]))
    (O / "decode_check.json").write_text(json.dumps(chk, indent=1)); log(f"decode check: {chk}")


# ---------------------------------------------------------------- detectors
class AeroSDXL:
    """AEROBLADE with the SDXL autoencoder (fp32), LPIPS-VGG layer 2, score = -distance (as V3.AeroAE)."""
    def __init__(self, a):
        import lpips
        self.dev = a.device; self.vae = load_dec("orig", a.device, a.fake_models)
        self.lp = (lambda x, r, retPerLayer: (None, [None, ((x - r) ** 2).mean(1, keepdim=True)])) if a.fake_models \
            else lpips.LPIPS(net="vgg", verbose=False).to(self.dev).eval()

    @torch.no_grad()
    def __call__(self, img):
        x = M.to_tensor(img).to(self.dev)
        r = self.vae.decode(self.vae.encode(x).latent_dist.mean).sample.float().clamp(-1, 1)
        _, layers = self.lp(x, r, retPerLayer=True)
        return -float(layers[1].mean().cpu())


def items(a, with_paper_calib=True):
    O = a.out; meta = json.loads((O / "data" / "meta.json").read_text())
    it = [("gen", d, m["k"], O / "png" / "gen" / d / f"{m['k']:04d}.png") for d in DECS for m in meta["eval"]]
    it += [("real_enc", d, m["k"], O / "png" / "real_enc" / d / f"{m['k']:04d}.png") for d in DECS for m in meta["eval"]]
    it += [("real", "none", m["k"], O / "data" / "real" / f"{m['k']:04d}.png") for m in meta["eval"]]
    it += [("calib", "none", m["k"], O / "data" / "calib" / f"{m['k']:04d}.png") for m in meta["calib"]]
    if with_paper_calib:
        pc = a.sd15 / "data" / "calib"
        it += [("paper_calib", "none", k, pc / f"{k:04d}.png") for k in range(1000) if (pc / f"{k:04d}.png").exists()]
    return it


def stage_gate(a):
    """Technical gates only (no gate looks at a decoder effect)."""
    O = a.out; g = json.loads((O / "decode_check.json").read_text()); res = dict(decode=g)
    res["taesdxl_ok"] = max(g["taesdxl_scale_psnr"].values()) >= 18 or a.fake_models
    res["fp16fix_near_identity"] = g["psnr_fp16fix_vs_orig_1024"] >= 30
    # detector reproduction on 20 SD1.5 images of the paper (proves weights + preprocessing are the paper's)
    if not a.fake_models:
        fr = pd.read_csv(FROZEN / "main_scores_e4.csv"); rep = {}
        sd15 = [("gen", "orig", i, a.sd15 / "png" / "gen" / "orig" / f"{i:04d}.png") for i in range(20)]
        dets = {"aeroblade": lambda: V3.AeroAE(a, "orig"), "npr": lambda: M.load_npr(a), "commfor": lambda: V3.CommFor(a)}
        for name, mk in dets.items():
            f = mk(); now = np.array([f(Image.open(p)) for *_, p in sd15]); del f; M.free(a.device)
            old = np.array([float(fr[(fr.detector == name) & (fr.group == "gen") & (fr.decoder == "orig") & (fr.idx == i)].score.iloc[0]) for i in range(20)])
            rep[name] = float(np.median(np.abs(now - old)) / fr[(fr.detector == name) & (fr.group == "calib")].score.std())
        res["repro_median_abs_diff_sd"] = rep; res["repro_ok"] = all(v <= 0.01 for v in rep.values())
    else:
        res["repro_ok"] = True
    res["passed"] = bool(res["taesdxl_ok"] and res["repro_ok"])
    (O / "gate.json").write_text(json.dumps(res, indent=1)); log(f"gate: {res}")
    if not res["passed"]:
        sys.exit("STOP: technical gate failed (gate.json). Check the log before starting the full experiment.")


def stage_score(a):
    O = a.out; fp = O / "sdxl_scores.csv"
    it = items(a); rows = pd.read_csv(fp).values.tolist() if fp.exists() else []
    done = {(r[0], r[1], int(r[2]), r[3]) for r in rows}
    def run(name, f, sub):
        nonlocal rows
        todo = [(g, d, i, p) for g, d, i, p in sub if (g, d, i, name) not in done]
        for g, d, i, p in todo:
            v = f(Image.open(p)); assert math.isfinite(v); rows.append([g, d, i, name, float(v)])
        pd.DataFrame(rows, columns=["group", "decoder", "idx", "detector", "score"]).to_csv(fp, index=False)
        log(f"score {name}: {len(todo)} images")
    new = [x for x in it if x[0] != "paper_calib"]
    if a.fake_models:
        for name in ["aeroblade", "aero_sdxl", "npr", "commfor", "univfd", "probe_clip", "probe_dino", "bfree"]:
            run(name, lambda im: float(np.random.default_rng(int(np.asarray(im).sum()) % 999).normal()), it)
        return
    f = V3.AeroAE(a, "orig"); run("aeroblade", f, new); del f; M.free(a.device)
    f = AeroSDXL(a); run("aero_sdxl", f, it); del f; M.free(a.device)     # paper calibration scored too (new AE)
    f = M.load_npr(a); run("npr", f, new); del f; M.free(a.device)
    f = V3.CommFor(a); run("commfor", f, new); del f; M.free(a.device)
    # UnivFD + probes from features (probes refit exactly as in the paper on the SD1.5 run's features)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    idx_p = a.sd15 / "feats_index.csv"
    for name in ["clip", "dino"]:
        todo = [x for x in new if (x[0], x[1], x[2], f"probe_{name}") not in done]
        if not todo:
            continue
        F = E4.extract(a, [p for *_, p in todo], name).astype(np.float64)
        if name == "clip":
            wv, b, _ = E4.univfd_head(a)
            rows += [[g, d, i, "univfd", float(v)] for (g, d, i, _), v in zip(todo, F @ wv + b)]
        if idx_p.exists():
            idx = pd.read_csv(idx_p); Fm = np.load(a.sd15 / f"feats_{name}.npy").astype(np.float64)
            tr = idx[(idx.idx < 150) & ((idx.group == "real") | ((idx.group == "gen") & (idx.decoder == "orig")))]
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=5000)).fit(Fm[tr.index.values], (tr.group == "gen").astype(int).values)
            rows += [[g, d, i, f"probe_{name}", float(v)] for (g, d, i, _), v in zip(todo, clf.decision_function(F))]
        else:
            log("probes skipped: run run_e4.sh first (needs feats_*.npy of the SD1.5 run)")
        pd.DataFrame(rows, columns=["group", "decoder", "idx", "detector", "score"]).to_csv(fp, index=False)
    code = E4.bf_setup(a)
    todo = [x for x in new if (x[0], x[1], x[2], "bfree") not in done]
    if todo:
        sc = E4.run_bf(a, code, [p for *_, p in todo], "sdxl_bf")
        rows += [[g, d, i, "bfree", float(v)] for (g, d, i, _), v in zip(todo, sc)]
    pd.DataFrame(rows, columns=["group", "decoder", "idx", "detector", "score"]).to_csv(fp, index=False)
    log(f"score: {len(rows)} rows")


# ---------------------------------------------------------------- report
def stage_report(a):
    O = a.out; df = pd.read_csv(O / "sdxl_scores.csv").drop_duplicates(["group", "decoder", "idx", "detector"], keep="last")
    fr = pd.concat([pd.read_csv(FROZEN / "main_scores_e4.csv"), pd.read_csv(FROZEN / "bfree_scores_v4.csv")])
    rng = np.random.default_rng(0); B = 2000
    def S(det, g, d="none"):
        return df[(df.detector == det) & (df.group == g) & (df.decoder == d)].sort_values("idx").score.values
    def paper_cal(det):
        x = S(det, "paper_calib")
        return x if len(x) else fr[(fr.detector == det) & (fr.group == "calib")].score.values
    gc = json.loads((O / "gate.json").read_text())
    L = [f"# E3 SDXL ({'pilot' if a.n <= 50 else 'full'}): N = {a.n} sources, {len(S('npr', 'calib'))} fresh calibration photos\n",
         f"Gate: {json.dumps(gc)}\n", f"Generation: {(O / 'gen_info.json').read_text() if (O / 'gen_info.json').exists() else 'n/a'}\n",
         "TPR at nominal 1% FPR (threshold: fresh calibration). 95% CIs: bootstrap over sources and calibration photos.",
         "Drop = TPR(orig) - TPR(alt), paired. Flip = share of sources whose decision differs across the 3 decoders.\n",
         "| detector | TPR orig | TPR fp16fix (control) | TPR taesdxl | drop taesdxl [CI] | drop fp16fix [CI] | AUROC orig/taesdxl | FPR untouched | flip | round-trip orig / taesdxl flagged | thr fresh vs paper cal. |",
         "|---" * 11 + "|"]
    summ = {}
    for det in ["aeroblade", "aero_sdxl", "npr", "univfd", "probe_clip", "probe_dino", "commfor", "bfree"]:
        cal = S(det, "calib"); G = np.stack([S(det, "gen", d) for d in DECS], 1) if len(S(det, "gen", "orig")) else None
        if G is None or not len(cal):
            continue
        Rt = np.stack([S(det, "real_enc", d) for d in DECS], 1); U = S(det, "real")
        def st(c, g, r, u):
            t = np.quantile(c, 0.99); dg = g > t
            return np.r_[dg.mean(0), dg[:, 0].mean() - dg[:, 2].mean(), dg[:, 0].mean() - dg[:, 1].mean(), (u > t).mean(),
                         (dg.any(1) & ~dg.all(1)).mean(), (r > t).mean(0)]
        est = st(cal, G, Rt, U); n = len(G)
        bs = np.array([st(cal[rng.integers(0, len(cal), len(cal))], *(lambda i: (G[i], Rt[i], U[i]))(rng.integers(0, n, n))) for _ in range(B)])
        lo, hi = np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0)
        au = [float((G[:, j][:, None] > U[None, :]).mean() + 0.5 * (G[:, j][:, None] == U[None, :]).mean()) for j in (0, 2)]
        tp = np.quantile(paper_cal(det), 0.99)
        L.append(f"| {det} | {est[0]:.2f} | {est[1]:.2f} | {est[2]:.2f} | {100*est[3]:+.0f} [{100*lo[3]:+.0f},{100*hi[3]:+.0f}] | "
                 f"{100*est[4]:+.0f} [{100*lo[4]:+.0f},{100*hi[4]:+.0f}] | {au[0]:.3f}/{au[1]:.3f} | {est[5]:.3f} | {est[6]:.2f} | "
                 f"{est[7]:.2f} / {est[9]:.2f} | {np.quantile(cal, 0.99):.3g} vs {tp:.3g} |")
        summ[det] = dict(tpr=dict(zip(DECS, est[:3].tolist())), drop_taesdxl=[est[3], lo[3], hi[3]], drop_fp16fix=[est[4], lo[4], hi[4]],
                         fpr_untouched=est[5], flip=est[6], roundtrip=dict(zip(DECS, est[7:10].tolist())), auroc_orig=au[0], auroc_taesdxl=au[1])
    usable = [d for d, s in summ.items() if s["tpr"]["orig"] >= 0.5 and d != "aeroblade"]
    big = [d for d in usable if abs(summ[d]["drop_taesdxl"][0]) >= 0.10 and (summ[d]["drop_taesdxl"][1] > 0 or summ[d]["drop_taesdxl"][2] < 0)]
    small = [d for d in usable if abs(summ[d]["drop_taesdxl"][0]) < 0.05 and summ[d]["drop_taesdxl"][1] > -0.10 and summ[d]["drop_taesdxl"][2] < 0.10]
    if a.n <= 50:
        v = "PILOT: technical check only; no reading is made from the pilot numbers"
    elif len(big) >= 2:
        v = f"DECODER EFFECT PRESENT on SDXL (|drop| >= 10 pts, CI excludes 0) for {big}"
    elif len(small) == len(usable) and usable:
        v = "NO DECODER EFFECT on SDXL for the usable detectors (|drop| < 5 pts, CIs within +-10)"
    else:
        v = f"MIXED (effect in {big}; usable = {usable})"
    ctrl = [d for d in usable if abs(summ[d]["drop_fp16fix"][0]) >= 0.05]
    L += [f"\nUsable detectors (TPR orig >= 0.5 at 1% FPR; AEROBLADE with the SD1.5 AE is reported but not counted): {usable}",
          f"\n**Planned reading:** {v}.",
          f"\nControl (fp16-fix, near-identity): detectors with |drop| >= 5 pts: {ctrl or 'none'}"]
    summ["verdict"] = v; summ["usable"] = usable
    (O / "report_sdxl.md").write_text("\n".join(L)); (O / "sdxl_summary.json").write_text(json.dumps(summ, indent=1, default=float))
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--sd15", type=Path, default=Path("runs/n300"))
    ap.add_argument("--device", default="cuda"); ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--calib", type=int, default=200); ap.add_argument("--cache", type=Path, default=Path("cache"))
    ap.add_argument("--stage", default="all", choices=["all", "data", "gen", "decode", "gate", "score", "report"])
    ap.add_argument("--fake-models", action="store_true")
    a = ap.parse_args(); a.n_eff = a.n; a.run = a.out; a.tiny_scaled = True; M.SD_ID = "fake" if a.fake_models else M.resolve_sd(None)
    if (a.out / "decode_check.json").exists():
        a.tiny_scaled = json.loads((a.out / "decode_check.json").read_text())["taesdxl_takes_scaled_latent"]
    for s in (["data", "gen", "decode", "gate", "score", "report"] if a.stage == "all" else [a.stage]):
        globals()[f"stage_{s}"](a)


if __name__ == "__main__":
    main()
