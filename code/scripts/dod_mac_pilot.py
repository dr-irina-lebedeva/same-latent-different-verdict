#!/usr/bin/env python3
"""
Decoder or Denoiser? -- small pilot for a Mac (Apple Silicon, MPS) or any CUDA/CPU machine.

Exp 1: same latent, 3 decoders (SD1.5 original VAE, sd-vae-ft-mse, TAESD),
       for real-photo latents (VAE-encoded COCO) and generated latents (SD1.5 txt2img).
Exp 2: real-photo latents after partial denoising (img2img, strength 0.5),
       decoded with the original VAE and ft-MSE.
Then scores every PNG with the detectors and writes report.md with the 4 checks.

Stages (run all with --stage all, or one at a time):
  data     download N COCO val2017 photos + captions, center-crop to RES (no resizing)
  latents  make real / generated / img2img latents (saved as .pt)
  decode   decode every latent with every decoder -> PNG
  score    run detectors on every PNG -> scores.csv
  report   metrics + checks -> report.md

Example (smoke test, ~10-20 min on an M4 Air):
  python dod_mac_pilot.py --n 4 --steps 10 --out runs/smoke
Pilot:
  python dod_mac_pilot.py --n 100 --out runs/pilot
"""
import argparse, io, json, math, os, random, subprocess, sys, zipfile, urllib.request
from pathlib import Path

import numpy as np
import warnings
# numpy 2.0 + Apple Accelerate prints false "overflow/divide by zero in matmul" warnings;
# we check every feature, weight and score for NaN/inf explicitly instead.
warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)
import torch
from PIL import Image

# ---------------------------------------------------------------- config
SD_ID = None  # resolved at start: first candidate that exists on the Hub (or --sd-id)
SD_CANDIDATES = ["sd-legacy/stable-diffusion-v1-5",
                 "stable-diffusion-v1-5/stable-diffusion-v1-5",
                 "benjamin-paine/stable-diffusion-v1-5",
                 "runwayml/stable-diffusion-v1-5"]
FTMSE_ID = "stabilityai/sd-vae-ft-mse"
TAESD_ID = "madebyollin/taesd"
CLIP_ID = "openai/clip-vit-large-patch14"
DINO_ID = "facebook/dinov2-base"
UNIVFD_FC_URL = ("https://github.com/WisconsinAIVision/UniversalFakeDetect/raw/main/"
                 "pretrained_weights/fc_weights.pth")
NPR_REPO = "https://github.com/chuangchuangtan/NPR-DeepfakeDetection.git"
COCO_ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_IMG_URL = "http://images.cocodataset.org/val2017/{:012d}.jpg"

DECODERS_EXP1 = ["orig", "ftmse", "taesd"]
DECODERS_EXP2 = ["orig", "ftmse"]
ALT = ["ftmse", "taesd"]        # decoders compared against "orig"
HI = "ftmse"                     # high-fidelity swap, used for Exp 2 and the quality check
TINY = {"taesd", "taef1"}        # tiny distilled decoders (lower PSNR limit, own latent convention)
FAMILY = "sd15"
FLUX_ID = "black-forest-labs/FLUX.1-schnell"
TAEF1_ID = "madebyollin/taef1"


def set_family(fam):
    """sd15: SD1.5 + orig / ft-MSE / TAESD.  flux: FLUX.1-schnell + FLUX VAE ("orig") / TAEF1."""
    global DECODERS_EXP1, DECODERS_EXP2, ALT, HI, GROUPS, FAMILY
    FAMILY = fam
    if fam == "flux":
        DECODERS_EXP1, DECODERS_EXP2, ALT, HI = ["orig", "taef1"], ["orig", "taef1"], ["taef1"], "taef1"
    GROUPS = {"real_enc": DECODERS_EXP1, "gen": DECODERS_EXP1, "i2i": DECODERS_EXP2}
# group name -> decoders used for that group
GROUPS = {"real_enc": DECODERS_EXP1, "gen": DECODERS_EXP1, "i2i": DECODERS_EXP2}


def pick_device(arg):
    if arg != "auto":
        return arg
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def log(*a):
    print("[dod]", *a, flush=True)


def free(dev):
    import gc
    gc.collect()
    if dev == "mps":
        torch.mps.empty_cache()
    elif dev == "cuda":
        torch.cuda.empty_cache()


def to_tensor(img):  # PIL -> [1,3,H,W] in [-1,1]
    a = np.asarray(img.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(a).permute(2, 0, 1)[None]


def to_pil(x):  # [1,3,H,W] in [-1,1] -> PIL
    x = ((x[0].clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(x)


# ---------------------------------------------------------------- stage: data
def stage_data(a):
    d = a.out / "data"
    (d / "real").mkdir(parents=True, exist_ok=True)
    meta_p = d / "meta.json"
    if meta_p.exists() and len(json.loads(meta_p.read_text())) >= a.n:
        log("data: already done"); return
    if a.fake_models:  # offline test data
        rng = np.random.default_rng(0); meta = []
        for k in range(a.n):
            x = rng.normal(size=(a.res // 16, a.res // 16, 3))
            im = Image.fromarray(((x - x.min()) / np.ptp(x) * 255).astype(np.uint8)).resize((a.res, a.res), Image.BICUBIC)
            im.save(d / "real" / f"{k:04d}.png"); meta.append({"k": k, "coco_id": k, "caption": "test"})
        meta_p.write_text(json.dumps(meta)); return
    cache = a.cache / "captions_val2017.json"
    if not cache.exists():
        a.cache.mkdir(parents=True, exist_ok=True)
        log("downloading COCO annotations (~240 MB, once)")
        zp = a.cache / "ann.zip"
        urllib.request.urlretrieve(COCO_ANN_URL, zp)
        with zipfile.ZipFile(zp) as z:
            cache.write_bytes(z.read("annotations/captions_val2017.json"))
        zp.unlink()
    ann = json.loads(cache.read_text())
    imgs = {im["id"]: im for im in ann["images"]}
    caps = {}
    for c in ann["annotations"]:
        caps.setdefault(c["image_id"], c["caption"].strip())
    # only photos big enough to crop RES x RES without any resizing
    ids = sorted(i for i, im in imgs.items()
                 if min(im["width"], im["height"]) >= a.res and i in caps)
    random.Random(0).shuffle(ids)
    ids = ids[getattr(a, 'photo_offset', 0):]   # --photo-offset: skip photos used by an earlier run
    meta = []
    for iid in ids:
        if len(meta) >= a.n:
            break
        try:
            raw = urllib.request.urlopen(COCO_IMG_URL.format(iid), timeout=30).read()
            im = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as e:
            log("skip", iid, e); continue
        w, h = im.size
        l, t = (w - a.res) // 2, (h - a.res) // 2
        if min(w, h) < a.res:
            log("skip (too small)", iid); continue
        im = im.crop((l, t, l + a.res, t + a.res))
        assert im.size == (a.res, a.res)
        k = len(meta)
        im.save(d / "real" / f"{k:04d}.png")
        meta.append({"k": k, "coco_id": iid, "caption": caps[iid], "orig_size": [w, h],
                     "crop_box": [l, t, l + a.res, t + a.res], "resized": False})
    meta_p.write_text(json.dumps(meta, indent=1))
    log(f"data: {len(meta)} photos at {a.res}px")


# ---------------------------------------------------------------- models
def load_vae(name, dev, fake=False):
    from diffusers import AutoencoderKL, AutoencoderTiny
    if fake and FAMILY == "sd15":  # tiny random models, only for testing the plumbing
        if name == "taesd":
            return AutoencoderTiny(encoder_block_out_channels=(8, 8, 8, 8),
                                   decoder_block_out_channels=(8, 8, 8, 8),
                                   num_encoder_blocks=(1, 1, 1, 1),
                                   num_decoder_blocks=(1, 1, 1, 1)).to(dev).eval()
        torch.manual_seed(0 if name == "orig" else 1)
        return AutoencoderKL(block_out_channels=(8, 8, 8, 8), layers_per_block=1,
                             down_block_types=("DownEncoderBlock2D",) * 4,
                             up_block_types=("UpDecoderBlock2D",) * 4,
                             latent_channels=4, norm_num_groups=4,
                             scaling_factor=0.18215).to(dev).eval()
    if fake and FAMILY == "flux":
        if name == "taef1":
            return AutoencoderTiny(latent_channels=16, encoder_block_out_channels=(8,) * 4,
                                   decoder_block_out_channels=(8,) * 4, num_encoder_blocks=(1,) * 4,
                                   num_decoder_blocks=(1,) * 4).to(dev).eval()
        return AutoencoderKL(block_out_channels=(8,) * 4, layers_per_block=1,
                             down_block_types=("DownEncoderBlock2D",) * 4,
                             up_block_types=("UpDecoderBlock2D",) * 4, latent_channels=16,
                             norm_num_groups=4, scaling_factor=0.3611, shift_factor=0.1159).to(dev).eval()
    if FAMILY == "flux":
        if name == "orig":
            return AutoencoderKL.from_pretrained(FLUX_ID, subfolder="vae").to(dev).eval()
        if name == "taef1":
            return AutoencoderTiny.from_pretrained(TAEF1_ID).to(dev).eval()
    if name == "orig":
        return AutoencoderKL.from_pretrained(SD_ID, subfolder="vae").to(dev).eval()
    if name == "ftmse":
        return AutoencoderKL.from_pretrained(FTMSE_ID).to(dev).eval()
    if name == "taesd":
        return AutoencoderTiny.from_pretrained(TAESD_ID).to(dev).eval()
    raise ValueError(name)


@torch.no_grad()
def decode(vae, name, z):
    """z = latent in the *scaled* SD space (what the UNet sees; = encoder_mean * 0.18215).
    AutoencoderKL expects z / scaling_factor. Diffusers' AutoencoderTiny (TAESD) has
    scaling_factor 1.0 and takes the scaled latent directly, returning [-1, 1]."""
    if name in TINY:
        x = vae.decode(z).sample
    else:
        sh = getattr(vae.config, "shift_factor", None) or 0.0
        x = vae.decode(z / vae.config.scaling_factor + sh).sample
    return x.float().clamp(-1, 1)


# ---------------------------------------------------------------- stage: latents
def stage_latents(a):
    meta = json.loads((a.out / "data" / "meta.json").read_text())
    L = a.out / "latents"; L.mkdir(exist_ok=True)
    dev = a.device
    todo = [g for g in GROUPS if not (L / f"{g}.pt").exists()
            or len(torch.load(L / f"{g}.pt")) != len(meta)]
    if not todo:
        log("latents: already done"); return
    real = torch.cat([to_tensor(Image.open(a.out / "data" / "real" / f"{m['k']:04d}.png"))
                      for m in meta])
    if "real_enc" in todo:
        vae = load_vae("orig", dev, a.fake_models)
        zs = []
        with torch.no_grad():
            for i in range(len(meta)):
                # use the posterior mean (deterministic) as the encoded latent
                sh = getattr(vae.config, "shift_factor", None) or 0.0
                zs.append((vae.encode(real[i:i + 1].to(dev)).latent_dist.mean - sh)
                          * vae.config.scaling_factor)
        torch.save(torch.cat(zs).cpu(), L / "real_enc.pt")
        del vae; free(dev); log("latents: real_enc done")
    if "gen" in todo or "i2i" in todo:
        if a.fake_models:
            g = torch.Generator().manual_seed(0)
            base = torch.load(L / "real_enc.pt")
            torch.save(torch.randn(base.shape, generator=g) * base.std(), L / "gen.pt")
            torch.save(0.5 * base + 0.5 * torch.randn(base.shape, generator=g) * base.std(),
                       L / "i2i.pt")
            log("latents: fake gen / i2i done"); return
        if FAMILY == "flux":
            return flux_latents(a, meta, L, todo)
        from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
        pipe = StableDiffusionPipeline.from_pretrained(
            SD_ID, torch_dtype=torch.float32, safety_checker=None,
            requires_safety_checker=False).to(dev)
        pipe.enable_attention_slicing()
        pipe.set_progress_bar_config(disable=True)
        if "gen" in todo:
            zs = []
            for i, m in enumerate(meta):
                g = torch.Generator("cpu").manual_seed(1000 + i)
                z = pipe(m["caption"], height=a.res, width=a.res,
                         num_inference_steps=a.steps, guidance_scale=7.5,
                         generator=g, output_type="latent").images
                zs.append(z.float().cpu())
                if i % 10 == 0: log(f"gen {i + 1}/{len(meta)}")
            torch.save(torch.cat(zs), L / "gen.pt")
        if "i2i" in todo:
            p2 = StableDiffusionImg2ImgPipeline(**pipe.components)
            p2.set_progress_bar_config(disable=True)
            zs = []
            for i, m in enumerate(meta):
                g = torch.Generator("cpu").manual_seed(2000 + i)
                init = Image.open(a.out / "data" / "real" / f"{m['k']:04d}.png")
                z = p2(m["caption"], image=init, strength=a.strength,
                       num_inference_steps=a.steps, guidance_scale=7.5,
                       generator=g, output_type="latent").images
                zs.append(z.float().cpu())
                if i % 10 == 0: log(f"i2i {i + 1}/{len(meta)}")
            torch.save(torch.cat(zs), L / "i2i.pt")
        (L / "gen_settings.json").write_text(json.dumps({
            "model": SD_ID, "scheduler": type(pipe.scheduler).__name__,
            "scheduler_config": dict(pipe.scheduler.config), "steps": a.steps,
            "guidance_scale": 7.5, "res": a.res, "img2img_strength": a.strength,
            "txt2img_seeds": "1000+k", "img2img_seeds": "2000+k",
            "img2img_init": "untouched real photo k, encoded by the pipeline's original SD VAE encoder",
            "real_enc": "posterior mean of the original SD1.5 VAE encoder x 0.18215; one encode, "
                        "reused for all decoders"}, indent=1, default=str))
        del pipe; free(dev)
    log("latents: done")


def flux_latents(a, meta, L, todo):
    """FLUX.1-schnell (4 steps, guidance 0). output_type='latent' returns packed tokens; unpack to
    (B,16,H/8,W/8) = the scaled+shifted space that FluxPipeline decodes with z/scale + shift."""
    from diffusers import FluxPipeline, FluxImg2ImgPipeline
    dev = a.device
    dt = torch.bfloat16 if dev == "cuda" else torch.float32
    pipe = FluxPipeline.from_pretrained(FLUX_ID, torch_dtype=dt)
    if dev == "cuda" and torch.cuda.get_device_properties(0).total_memory < 40e9:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(dev)
    pipe.set_progress_bar_config(disable=True)
    unpack = lambda z: pipe._unpack_latents(z, a.res, a.res, pipe.vae_scale_factor).float().cpu()
    steps = a.steps if a.steps <= 8 else 4
    if "gen" in todo:
        zs = []
        for i, m in enumerate(meta):
            g = torch.Generator("cpu").manual_seed(1000 + i)
            z = pipe(m["caption"], height=a.res, width=a.res, num_inference_steps=steps,
                     guidance_scale=0.0, max_sequence_length=256, generator=g,
                     output_type="latent").images
            zs.append(unpack(z))
            if i % 10 == 0: log(f"flux gen {i + 1}/{len(meta)}")
        torch.save(torch.cat(zs), L / "gen.pt")
    if "i2i" in todo:
        p2 = FluxImg2ImgPipeline(**pipe.components)
        p2.set_progress_bar_config(disable=True)
        zs = []
        for i, m in enumerate(meta):
            g = torch.Generator("cpu").manual_seed(2000 + i)
            init = Image.open(a.out / "data" / "real" / f"{m['k']:04d}.png")
            z = p2(m["caption"], image=init, strength=a.strength, height=a.res, width=a.res,
                   num_inference_steps=steps, guidance_scale=0.0, max_sequence_length=256,
                   generator=g, output_type="latent").images
            zs.append(unpack(z))
            if i % 10 == 0: log(f"flux i2i {i + 1}/{len(meta)}")
        torch.save(torch.cat(zs), L / "i2i.pt")
    (L / "gen_settings.json").write_text(json.dumps({
        "model": FLUX_ID, "scheduler": type(pipe.scheduler).__name__, "steps": steps,
        "guidance_scale": 0.0, "res": a.res, "img2img_strength": a.strength,
        "txt2img_seeds": "1000+k", "img2img_seeds": "2000+k",
        "real_enc": "(posterior mean - shift) x scale of the FLUX VAE encoder; reused for all decoders"},
        indent=1, default=str))
    del pipe; free(dev); log("latents: flux done")


# ---------------------------------------------------------------- stage: decode
def stage_decode(a):
    L, P = a.out / "latents", a.out / "png"
    dev = a.device
    for dec in DECODERS_EXP1:
        jobs = [g for g, ds in GROUPS.items() if dec in ds]
        if all((P / g / dec).exists() and len(list((P / g / dec).glob("*.png")))
               == len(torch.load(L / f"{g}.pt")) for g in jobs):
            continue
        vae = load_vae(dec, dev, a.fake_models)
        for g in jobs:
            zs = torch.load(L / f"{g}.pt")
            o = P / g / dec; o.mkdir(parents=True, exist_ok=True)
            for i in range(len(zs)):
                to_pil(decode(vae, dec, zs[i:i + 1].to(dev))).save(o / f"{i:04d}.png")
        del vae; free(dev); log(f"decode: {dec} done")
    check_images(a)
    make_grid(a)
    if not a.fake_models:
        taesd_scale_check(a)
    # sanity check of latent conventions: real photo round-trip must look like the photo
    if not a.fake_models:
        bad = []
        for dec in DECODERS_EXP1:
            ps = [psnr(P / "real_enc" / dec / f"{i:04d}.png",
                       a.out / "data" / "real" / f"{i:04d}.png") for i in range(min(4, a.n))]
            log(f"sanity PSNR {dec}: {np.mean(ps):.1f} dB")
            if np.mean(ps) < (16 if dec in TINY else 20):
                bad.append(dec)
        if bad:
            sys.exit(f"STOP: decoder(s) {bad} give too low PSNR on real photos (need 20 dB, TAESD 16). "
                     "Latent scaling is wrong; send me the log.")


def check_images(a):
    """every decoded PNG: right size, not black / saturated / flat"""
    bad = []
    for g, ds in GROUPS.items():
        for dec in ds:
            for i in range(a.n_eff):
                p = a.out / "png" / g / dec / f"{i:04d}.png"
                x = np.asarray(Image.open(p), dtype=np.float32)
                if x.shape != (a.res, a.res, 3) or not np.isfinite(x).all():
                    bad.append((str(p), "shape")); continue
                sat = np.mean((x <= 1) | (x >= 254))
                if x.std() < 3 or sat > 0.5:
                    bad.append((str(p), f"std {x.std():.1f}, saturated {sat:.2f}"))
    log(f"image check: {len(bad)} suspicious images")
    (a.out / "image_check.json").write_text(json.dumps(bad, indent=1))
    if not a.fake_models and len(bad) > 0.05 * a.n_eff * 8:
        sys.exit("STOP: many black / saturated / flat images. See image_check.json and send me the log.")


def make_grid(a, k=4):
    """visual check: rows = first k examples; cols = real | real_enc x 3 | gen x 3 | i2i x 2"""
    cols = [("data/real", "real")] + [(f"png/real_enc/{d}", f"re_{d}") for d in DECODERS_EXP1] \
        + [(f"png/gen/{d}", f"gen_{d}") for d in DECODERS_EXP1] \
        + [(f"png/i2i/{d}", f"i2i_{d}") for d in DECODERS_EXP2]
    t = 192
    G = Image.new("RGB", (t * len(cols), t * min(k, a.n_eff)), "white")
    for r in range(min(k, a.n_eff)):
        for c, (sub, _) in enumerate(cols):
            im = Image.open(a.out / sub / f"{r:04d}.png").resize((t, t), Image.LANCZOS)
            G.paste(im, (c * t, r * t))
    G.save(a.out / "grid.png")
    (a.out / "grid_columns.txt").write_text(" | ".join(n for _, n in cols))
    log(f"visual check: open {a.out / 'grid.png'} (columns in grid_columns.txt)")


@torch.no_grad()
def taesd_scale_check(a):
    """TAESD's own encoder should give latents on the same scale as the SD encoder x 0.18215.
    If the convention were wrong, the std ratio would be ~5.5 or ~0.18 instead of ~1."""
    dev = a.device
    sd = torch.load(a.out / "latents" / "real_enc.pt")[:4]
    tiny = [d for d in DECODERS_EXP1 if d in TINY][0]
    tv = load_vae(tiny, dev, a.fake_models)
    xs = torch.cat([to_tensor(Image.open(a.out / "data" / "real" / f"{i:04d}.png"))
                    for i in range(min(4, a.n_eff))]).to(dev)
    zt = tv.encode(xs).latents.float().cpu()
    ratio = float(zt.std() / sd.std())
    corr = float(np.corrcoef(zt.flatten(), sd[:len(zt)].flatten())[0, 1])
    log(f"TAESD scale check: std ratio {ratio:.2f} (want 0.7-1.4), correlation {corr:.2f} (want > 0.8)")
    (a.out / "taesd_check.json").write_text(json.dumps({"std_ratio": ratio, "corr": corr}))
    del tv; free(dev)
    if not (0.7 < ratio < 1.4):
        sys.exit("STOP: TAESD latent scale does not match the SD latent scale. Send me the log.")


def psnr(p1, p2):
    x = np.asarray(Image.open(p1), dtype=np.float64)
    y = np.asarray(Image.open(p2), dtype=np.float64)
    mse = ((x - y) ** 2).mean()
    return 99.0 if mse == 0 else 10 * math.log10(255 ** 2 / mse)


# ---------------------------------------------------------------- detectors
def all_images(a):
    """list of (group, decoder, index, path). 'real' = untouched photos."""
    out = [("real", "none", i, a.out / "data" / "real" / f"{i:04d}.png") for i in range(a.n_eff)]
    for g, ds in GROUPS.items():
        for d in ds:
            out += [(g, d, i, a.out / "png" / g / d / f"{i:04d}.png") for i in range(a.n_eff)]
    return out


class ClipFeat:
    def __init__(self, dev, fake=False):
        from transformers import CLIPVisionModelWithProjection, CLIPVisionConfig
        if fake:
            cfg = CLIPVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                                   num_attention_heads=2, image_size=224, patch_size=32,
                                   projection_dim=768)
            self.m = CLIPVisionModelWithProjection(cfg).to(dev).eval()
        else:
            self.m = CLIPVisionModelWithProjection.from_pretrained(CLIP_ID).to(dev).eval()
        self.dev = dev
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)

    @torch.no_grad()
    def __call__(self, img):
        # UnivFD protocol: center crop 224 at native resolution (no resize), CLIP norm
        x = (to_tensor(img) + 1) / 2
        h = x.shape[-1]; o = (h - 224) // 2
        x = x[..., o:o + 224, o:o + 224]
        x = ((x - self.mean) / self.std).to(self.dev)
        return self.m(pixel_values=x).image_embeds[0].float().cpu().numpy()


class DinoFeat:
    def __init__(self, dev, fake=False):
        from transformers import AutoModel, Dinov2Config, Dinov2Model
        if fake:
            self.m = Dinov2Model(Dinov2Config(hidden_size=32, num_hidden_layers=1,
                                              num_attention_heads=2, intermediate_size=64,
                                              patch_size=14, image_size=224)).to(dev).eval()
        else:
            self.m = AutoModel.from_pretrained(DINO_ID).to(dev).eval()
        self.dev = dev
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    @torch.no_grad()
    def __call__(self, img):
        x = (to_tensor(img) + 1) / 2
        h = x.shape[-1]; o = (h - 224) // 2
        x = x[..., o:o + 224, o:o + 224]
        x = ((x - self.mean) / self.std).to(self.dev)
        return self.m(pixel_values=x).last_hidden_state[0, 0].float().cpu().numpy()


def load_npr(a):
    """Best effort: clone the official repo and load its weights. Returns callable or None."""
    try:
        repo = a.cache / "NPR-DeepfakeDetection"
        if not repo.exists():
            subprocess.run(["git", "clone", "--depth", "1", NPR_REPO, str(repo)], check=True)
        sys.path.insert(0, str(repo))
        from networks.resnet import resnet50  # from the NPR repo
        cands = sorted(repo.rglob("*.pth"))
        if not cands:
            log("NPR: no .pth weights found in repo; download NPR.pth from the repo README "
                f"into {repo} and rerun --stage score"); return None
        net = resnet50(num_classes=1)
        sd = torch.load(cands[0], map_location="cpu")
        sd = sd.get("model", sd)
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        net.load_state_dict(sd, strict=True)
        net = net.to(a.device).eval()
        log(f"NPR: loaded {cands[0].name}")
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        @torch.no_grad()
        def f(img):
            x = ((to_tensor(img) + 1) / 2 - mean) / std  # full image, no resize
            return float(net(x.to(a.device)).flatten()[0].cpu())
        return f
    except Exception as e:
        log(f"NPR: could not load ({e}). Skipping NPR; the probes still run.")
        return None


def stage_score(a):
    import pandas as pd
    out_p = a.out / "scores.csv"
    imgs = all_images(a)
    rows = []
    dev = a.device

    # 1) features for probes + UnivFD (CLIP), and DINOv2 probe
    feats = {}
    for name, cls in [("clip", ClipFeat), ("dino", DinoFeat)]:
        fp = a.out / f"feat_{name}.npy"
        if fp.exists() and len(np.load(fp)) == len(imgs):
            feats[name] = np.load(fp)
        else:
            fx = cls(dev, a.fake_models)
            feats[name] = np.stack([fx(Image.open(p)) for _, _, _, p in imgs])
            np.save(fp, feats[name]); del fx; free(dev)
        feats[name] = feats[name].astype(np.float64)
        if not np.isfinite(feats[name]).all():
            sys.exit(f"STOP: {name} features contain NaN/inf. Delete {fp} and rerun --stage score.")
        log(f"features: {name} done (max |x| = {np.abs(feats[name]).max():.2f}, all finite)")

    # 2) UnivFD = official linear head on CLIP ViT-L/14 image features
    w = None
    if not a.fake_models:
        try:
            fcp = a.cache / "univfd_fc_weights.pth"
            if not fcp.exists():
                urllib.request.urlretrieve(UNIVFD_FC_URL, fcp)
            w = torch.load(fcp, map_location="cpu")
        except Exception as e:
            log(f"UnivFD: could not load head ({e}); skipping UnivFD")
    if w is not None:
        wv = w["weight"].double().numpy().reshape(-1)
        s = np.einsum("ij,j->i", feats["clip"], wv) + float(w["bias"].reshape(-1)[0])
        assert np.isfinite(s).all(), "UnivFD scores not finite"
        rows += [(g, d, i, "univfd", float(v)) for (g, d, i, _), v in zip(imgs, s)]

    # 3) linear probes: train on first half (untouched real vs gen/orig), score everything
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    half = a.n_eff // 2
    for name in ["clip", "dino"]:
        F = feats[name]
        tr = [j for j, (g, d, i, _) in enumerate(imgs)
              if i < half and ((g == "real") or (g == "gen" and d == "orig"))]
        y = [0 if imgs[j][0] == "real" else 1 for j in tr]
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=5000))
        clf.fit(F[tr], y)
        lr = clf[-1]
        assert np.isfinite(lr.coef_).all() and np.isfinite(lr.intercept_).all(), "probe weights not finite"
        s = clf.decision_function(F)
        assert np.isfinite(s).all(), f"probe_{name} scores not finite"
        rows += [(g, d, i, f"probe_{name}", float(v)) for (g, d, i, _), v in zip(imgs, s)]

    # 4) NPR
    npr = None if a.fake_models else load_npr(a)
    if npr is not None:
        for (g, d, i, p) in imgs:
            v = npr(Image.open(p)); assert math.isfinite(v), "NPR score not finite"
            rows.append((g, d, i, "npr", v))
        free(dev)

    status = {"univfd": "ok" if w is not None else "SKIPPED (head did not load)",
              "npr": "ok" if npr is not None else "SKIPPED (weights did not load)",
              "probe_clip": "ok", "probe_dino": "ok",
              "preprocessing": {"univfd/probe_clip": "center crop 224 at native res, no resize, CLIP norm",
                                "probe_dino": "center crop 224 at native res, no resize, ImageNet norm",
                                "npr": "full 448 image, no resize, ImageNet norm"},
              "probe_split": "split by source index: k < N/2 train, k >= N/2 test; all variants of "
                             "one source stay in the same split; trained on untouched real vs gen/orig only"}
    (a.out / "detector_status.json").write_text(json.dumps(status, indent=1))
    df = pd.DataFrame(rows, columns=["group", "decoder", "idx", "detector", "score"])
    df.to_csv(out_p, index=False)
    log(f"score: wrote {out_p} ({len(df)} rows)")


# ---------------------------------------------------------------- stage: report
def auroc(neg, pos):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score([0] * len(neg) + [1] * len(pos), list(neg) + list(pos)))


def boot_ci(x, n=2000, seed=0):
    x = np.asarray(x, float)
    r = np.random.default_rng(seed)
    m = [r.choice(x, len(x)).mean() for _ in range(n)]
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def auroc_drop_ci(real, g0, g1, n=2000, seed=0):
    """paired bootstrap over source index k: AUROC(real, g0) - AUROC(real, g1)"""
    r = np.random.default_rng(seed); k = len(real); out = []
    for _ in range(n):
        idx = r.integers(0, k, k)
        if len(set(idx)) < 3:
            continue
        out.append(auroc(real[idx], g0[idx]) - auroc(real[idx], g1[idx]))
    if not out:   # too few sources (smoke test)
        return float("nan"), float("nan")
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def quality_table(a):
    rows = []
    lp = None
    try:
        import lpips
        lp = lpips.LPIPS(net="alex", verbose=False)
    except Exception:
        pass
    for dec in DECODERS_EXP1:
        ps, ls, hf = [], [], []
        for i in range(a.n_eff):
            p = a.out / "png" / "real_enc" / dec / f"{i:04d}.png"
            r = a.out / "data" / "real" / f"{i:04d}.png"
            ps.append(psnr(p, r))
            if lp is not None:
                with torch.no_grad():
                    ls.append(float(lp(to_tensor(Image.open(p)), to_tensor(Image.open(r)))))
            hf.append(hf_ratio(p) / max(hf_ratio(r), 1e-12))
        rows.append((dec, np.mean(ps), np.mean(ls) if ls else float("nan"), np.mean(hf)))
    return rows


def hf_ratio(p):
    """share of spectral energy in the top half of radial frequencies (gray image)"""
    x = np.asarray(Image.open(p).convert("L"), dtype=np.float64)
    F = np.abs(np.fft.fftshift(np.fft.fft2(x - x.mean()))) ** 2
    h, w = F.shape
    yy, xx = np.mgrid[:h, :w]
    r = np.hypot(yy - h / 2, xx - w / 2)
    rmax = min(h, w) / 2
    return F[(r > rmax / 2) & (r <= rmax)].sum() / F[r <= rmax].sum()


def per_image_quality(a):
    """one row per (group, decoder, k): PSNR/LPIPS vs photo (real_enc only), LPIPS vs the orig
    decoding of the same latent, and HF energy share. Cached in per_image_quality.csv."""
    import pandas as pd
    fp = a.out / "per_image_quality.csv"
    if fp.exists():
        q = pd.read_csv(fp)
        if len(q) == a.n_eff * 2 * len(DECODERS_EXP1):
            return q
    try:
        import lpips
        lp = lpips.LPIPS(net="alex", verbose=False)
        L = lambda x, y: float(lp(to_tensor(Image.open(x)), to_tensor(Image.open(y))))
    except Exception:
        L = lambda x, y: float("nan")
    rows = []
    with torch.no_grad():
        for g in ["real_enc", "gen"]:
            for dec in DECODERS_EXP1:
                for k in range(a.n_eff):
                    pth = a.out / "png" / g / dec / f"{k:04d}.png"
                    ref = a.out / "data" / "real" / f"{k:04d}.png"
                    o = a.out / "png" / g / "orig" / f"{k:04d}.png"
                    rows.append(dict(group=g, decoder=dec, idx=k, hf=hf_ratio(pth),
                                     psnr_photo=psnr(pth, ref) if g == "real_enc" else np.nan,
                                     lpips_photo=L(pth, ref) if g == "real_enc" else np.nan,
                                     lpips_vs_orig=L(pth, o) if dec != "orig" else 0.0))
    q = pd.DataFrame(rows); q.to_csv(fp, index=False)
    return q


def per_image_section(a, df, P):
    """Does the per-image score change follow per-image quality / spectrum changes?"""
    from scipy.stats import spearmanr
    q = per_image_quality(a).set_index(["group", "decoder", "idx"])
    P("## Per-image analysis: score change vs quality / spectrum change\n")
    P("Spearman rho between the paired score change (decoder vs orig, same latent) and the per-image "
      "change of each quality measure. |rho| near 1 means the measure explains the ordering of the "
      "effect; near 0 means it does not. (The mean shift itself is in the tables above.)\n")
    P("| detector | group | decoder | n | rho ΔHF | rho ΔPSNR | rho ΔLPIPS(photo) | rho LPIPS(vs orig) |")
    P("|---|---|---|---|---|---|---|---|")
    half = a.n_eff // 2
    for det in sorted(df.detector.unique()):
        d = df[df.detector == det].set_index(["group", "decoder", "idx"]).score
        for g in ["real_enc", "gen"]:
            for dec in ALT:
                ks = [k for k in range(a.n_eff) if not det.startswith("probe_") or k >= half]
                ds = np.array([d[(g, dec, k)] - d[(g, "orig", k)] for k in ks])
                def rho(col, diff=True):
                    x = np.array([q.loc[(g, dec, k), col] - (q.loc[(g, "orig", k), col] if diff else 0)
                                  for k in ks], float)
                    if np.isnan(x).all() or np.std(x) == 0 or np.std(ds) == 0:
                        return "—"
                    r, pv = spearmanr(ds, x)
                    return f"{r:+.2f}{'*' if pv < 0.05 else ''}"
                P(f"| {det} | {g} | {dec} | {len(ks)} | {rho('hf')} | {rho('psnr_photo')} | "
                  f"{rho('lpips_photo')} | {rho('lpips_vs_orig', diff=False)} |")
    P("\n(* p < 0.05, uncorrected; exploratory.)\n")


def stage_report(a):
    import pandas as pd
    df = pd.read_csv(a.out / getattr(a, "scores_file", "scores.csv"))
    half = a.n_eff // 2
    L = []
    P = lambda *s: L.append(" ".join(str(x) for x in s))
    P(f"# Decoder or Denoiser? pilot report\n\nN = {a.n_eff} photos/captions, "
      f"res {a.res}, steps {a.steps if FAMILY == 'sd15' else (a.steps if a.steps <= 8 else 4)}"
      f"{', guidance 7.5' if FAMILY == 'sd15' else ', guidance 0 (FLUX.1-schnell)'}, img2img strength {a.strength}.\n")
    P("Probes (probe_*) are trained on the first half; all their numbers use the held-out "
      f"second half only (n = {a.n_eff - half}). Pretrained detectors use all images.\n")

    q = quality_table(a)
    P("## Decoder quality on real-photo round-trip\n\n| decoder | PSNR dB | LPIPS | HF energy vs photo |\n|---|---|---|---|")
    for dec, p, l, h in q:
        P(f"| {dec} | {p:.2f} | {l:.3f} | {h:.2f} |")
    P("")

    verdict = {}
    for det in sorted(df.detector.unique()):
        d = df[df.detector == det]
        if det.startswith("probe_"):
            d = d[d.idx >= half]
        S = lambda g, dec: d[(d.group == g) & (d.decoder == dec)].sort_values("idx").score.values
        real = S("real", "none")
        sd = np.std(np.concatenate([real, S("gen", "orig")])) + 1e-9
        P(f"## Detector `{det}`\n")
        base = auroc(real, S("gen", "orig"))
        P(f"Baseline AUROC (untouched real vs generated+orig decoder): **{base:.3f}**"
          + ("" if base >= 0.80 else "  <- below 0.80: this detector's changes are NOT informative"))
        P("\n| decoder | AUROC gen vs real | AUROC real-enc vs real | mean Δ gen (sd units) [95% CI] "
          "| mean Δ real-enc [95% CI] | interaction I [95% CI] | TPR@5%FPR | FPR on real-enc | flips |")
        P("|---|---|---|---|---|---|---|---|---|")
        thr = np.percentile(real, 95)
        res = {}
        for dec in DECODERS_EXP1:
            g, r = S("gen", dec), S("real_enc", dec)
            dg = (g - S("gen", "orig")) / sd
            dr = (r - S("real_enc", "orig")) / sd
            mg, mr, mi = boot_ci(dg), boot_ci(dr), boot_ci(dg - dr)
            flips = float(np.mean((g > thr) != (S("gen", "orig") > thr)))
            res[dec] = dict(auc=auroc(real, g), auc_ci=auroc_drop_ci(real, S("gen", "orig"), g),
                            mg=mg, mi=mi, tpr=float(np.mean(g > thr)),
                            fpr=float(np.mean(r > thr)), flips=flips)
            P(f"| {dec} | {res[dec]['auc']:.3f} | {auroc(real, r):.3f} | {mg[0]:+.2f} [{mg[1]:+.2f}, {mg[2]:+.2f}] "
              f"| {mr[0]:+.2f} [{mr[1]:+.2f}, {mr[2]:+.2f}] | {mi[0]:+.2f} [{mi[1]:+.2f}, {mi[2]:+.2f}] "
              f"| {res[dec]['tpr']:.2f} | {res[dec]['fpr']:.2f} | {flips:.2f} |")
        # Exp 2: does partial denoising change the decoder effect?
        di = (S("i2i", HI) - S("i2i", "orig")) / sd
        dr = (S("real_enc", HI) - S("real_enc", "orig")) / sd
        i2 = boot_ci(di - dr)
        P(f"\nExp 2 ({HI} vs orig): Δ after img2img minus Δ with no denoising = "
          f"**{i2[0]:+.2f}** [{i2[1]:+.2f}, {i2[2]:+.2f}] (sd units)\n")
        P("AUROC drop vs orig decoder, paired bootstrap 95% CI (points): " + "; ".join(
            f"{dd} {100 * (base - res[dd]['auc']):+.1f} [{100 * res[dd]['auc_ci'][0]:+.1f}, {100 * res[dd]['auc_ci'][1]:+.1f}]"
            for dd in ALT) + "\n")
        P(f"Note: threshold set from only {len(real)} real photos; TPR/FPR at it are exploratory.\n")
        verdict[det] = dict(base=base, res=res, i2=i2)

    try:
        per_image_section(a, df, P)
    except Exception as e:
        P(f"(per-image analysis skipped: {e})\n")

    # ---- the 4 checks (provisional screening, not a publishability decision)
    sig = lambda ci: ci[1] > 0 or ci[2] < 0
    P("## The 4 checks (provisional)\n")
    st = json.loads((a.out / "detector_status.json").read_text())
    P("Prerequisites: " + "; ".join(f"{k}: {v}" for k, v in st.items()
                                    if k in ("univfd", "npr", "probe_clip", "probe_dino")))
    weak = [k for k, v in verdict.items() if v["base"] < 0.80]
    if weak:
        P(f"Excluded (baseline AUROC < 0.80, changes not interpretable): {weak}")
    ok = {k: v for k, v in verdict.items() if v["base"] >= 0.80}
    qd = {dec: (p, l) for dec, p, l, _ in q}
    P(f"Samples: {a.n_eff} sources per group (probes: {a.n_eff - half} held-out). "
      "Effects in sd units of the detector score; [95% bootstrap CI].\n")
    c1 = c2 = c3 = c4 = False
    if not ok:
        P("No usable detector, so nothing can be concluded. Fix the detectors or use more images.")
    else:
        # 1. decoder effect: paired change on identical generated latents
        l1 = []
        for k, v in ok.items():
            for dd in ALT:
                m = v["res"][dd]["mg"]; hit = sig(m) and abs(m[0]) >= 0.5
                c1 |= hit
                l1.append(f"{k}/{dd} {m[0]:+.2f} [{m[1]:+.2f},{m[2]:+.2f}]{' *' if hit else ''}")
        P(f"**1. Decoder effect** (same latent, different decoder; * = CI excludes 0 and |effect| >= 0.5): "
          f"{'; '.join(l1)} -> **{'YES' if c1 else 'NO'}**\n")
        # 2. heterogeneity across detectors
        l2 = []
        for dd in ALT:
            ms = {k: v["res"][dd]["mg"] for k, v in ok.items()}
            if len(ms) < 2:
                continue
            means = [m[0] for m in ms.values()]
            opp = any(sig(x) and sig(y) and x[0] * y[0] < 0 for x in ms.values() for y in ms.values())
            spread = max(means) - min(means)
            drops = {k: 100 * (v["base"] - v["res"][dd]["auc"]) for k, v in ok.items()}
            strong = max(drops.values()) >= 15 and min(drops.values()) <= 5
            hit = spread >= 1.0 or opp or strong
            c2 |= hit
            l2.append(f"{dd}: spread {spread:.2f} sd, opposite signs {opp}, AUROC drops "
                      + ", ".join(f"{k} {x:+.1f}" for k, x in drops.items())
                      + (" (meets the 15/5 rule)" if strong else ""))
        P(f"**2. Detector heterogeneity** (spread >= 1 sd, or significant opposite signs, or 15/5 AUROC rule): "
          f"{'; '.join(l2) or 'needs >= 2 usable detectors'} -> **{'YES' if c2 else 'NO'}**\n")
        # 3. decoder x denoising history (same source photos: img2img vs reconstruction only)
        l3 = [f"{k} {v['i2'][0]:+.2f} [{v['i2'][1]:+.2f},{v['i2'][2]:+.2f}]" for k, v in ok.items()]
        c3 = any(sig(v["i2"]) for v in ok.values())
        sec = sorted({k for k, v in ok.items() for dd in ALT if sig(v["res"][dd]["mi"])})
        P(f"**3. Decoder x history** ({HI} effect after img2img minus effect with no denoising, same photos): "
          f"{'; '.join(l3)} -> **{'YES' if c3 else 'NO'}**. "
          f"Secondary (generated vs real-encoded latents, different content): significant for {sec or 'none'}.\n")
        # 4. quality confound: does the small, high-fidelity swap (ft-MSE) already move scores?
        gap = qd["orig"][0] - qd[HI][0]
        ft_hits = [k for k, v in ok.items() if sig(v["res"][HI]["mg"])]
        c4 = abs(gap) <= 2.0 and bool(ft_hits)
        P(f"**4. Not just quality loss**: PSNR " + " / ".join(f"{d} {qd[d][0]:.1f}" for d in DECODERS_EXP1)
          + " dB; LPIPS " + " / ".join(f"{d} {qd[d][1]:.3f}" for d in DECODERS_EXP1)
          + f". {HI} (gap {gap:+.1f} dB) changes scores for: {ft_hits or 'none'} -> **{'YES' if c4 else 'NO'}**\n")
        # extra: practical consequence
        l5 = []
        for k, v in ok.items():
            t0 = v["res"]["orig"]["tpr"]
            for dd in ALT:
                l5.append(f"{k}/{dd}: TPR {t0:.2f}->{v['res'][dd]['tpr']:.2f}, "
                          f"FPR real-enc {v['res'][dd]['fpr']:.2f}, flips {v['res'][dd]['flips']:.2f}")
        P(f"Extra, fixed threshold (5% FPR on untouched real): {'; '.join(l5)}\n")
        if a.n_eff < 20:
            P(f"**Provisional label: TOO FEW SAMPLES (N={a.n_eff}).** Bootstrap intervals are "
              "unreliable below ~20 sources (and probes see only half). Technical check only.")
        elif c1 and c2 and c3 and c4:
            P("**Provisional label: GO.** All 4 checks pass. Scale up (24 -> 100 -> full study).")
        elif c1 and (c2 or c3):
            P("**Provisional label: INVESTIGATE.** A decoder effect, plus heterogeneity or interaction. "
              "Look at the tables and the grid, then run more images.")
        else:
            P("**Provisional label: NO CLEAR SIGNAL.** Don't conclude the idea is wrong from a small "
              "sample. Check the grid / prerequisites, and rerun with more images before deciding.")
    P(f"\nSmall sample (N={a.n_eff}): this is a screening signal, not publishable evidence.")
    (a.out / getattr(a, "report_file", "report.md")).write_text("\n".join(L))
    log(f"report: {a.out / 'report.md'}")
    print("\n".join(L))


def resolve_sd(user_id):
    """SD1.5 has moved between Hub accounts; use the first id that answers."""
    if user_id:
        return user_id
    from huggingface_hub import model_info
    errs = []
    for c in SD_CANDIDATES:
        try:
            model_info(c)
            log(f"using SD1.5 from {c}")
            return c
        except Exception as e:
            errs.append(f"{c}: {type(e).__name__}")
    sys.exit("STOP: could not reach any SD1.5 repo on the Hub:\n  " + "\n  ".join(errs) +
             "\nCheck internet / run `hf auth login`, or pass --sd-id <id or local folder>.")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="photos (= captions) per group")
    ap.add_argument("--photo-offset", type=int, default=0,
                    help="skip the first K photos of the fixed order (e.g. 200 for a run disjoint from runs/n100)")
    ap.add_argument("--res", type=int, default=448,
                    help="448: COCO photos can be center-cropped with NO resizing")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--strength", type=float, default=0.5)
    ap.add_argument("--out", type=Path, default=Path("runs/pilot"))
    ap.add_argument("--cache", type=Path, default=Path("cache"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--stage", default="all",
                    choices=["all", "data", "latents", "decode", "score", "report"])
    ap.add_argument("--family", default="sd15", choices=["sd15", "flux"],
                    help="flux = FLUX.1-schnell with FLUX VAE vs TAEF1 (needs a big GPU: a cloud GPU)")
    ap.add_argument("--sd-id", default=None, help="Hub id or local folder of SD1.5 (diffusers format)")
    ap.add_argument("--fake-models", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    set_family(a.family)
    a.device = pick_device(a.device)
    global SD_ID
    if a.fake_models:
        SD_ID = "fake"
    elif a.family == "sd15":
        SD_ID = resolve_sd(a.sd_id)
    a.out.mkdir(parents=True, exist_ok=True)
    log(f"device={a.device} n={a.n} res={a.res} out={a.out}")
    torch.manual_seed(0)
    st = ["data", "latents", "decode", "score", "report"] if a.stage == "all" else [a.stage]
    for s in st:
        if s != "data":
            a.n_eff = len(json.loads((a.out / "data" / "meta.json").read_text()))
        globals()[f"stage_{s}"](a)


if __name__ == "__main__":
    main()
