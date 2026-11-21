<!-- saved 2026-11-14 18:45 UTC, before the SDXL pilot and full runs; run instructions removed -->
# E3: LatentLock on SDXL (pilot first, then full)

**Question.** Does a decoder swap on an identical latent change detector verdicts on a second diffusion architecture?

## Design (fixed before any SDXL image exists)

| Item | Choice | Why |
|---|---|---|
| Generator | SDXL base 1.0, no refiner; 1024×1024 (native); 30 steps; guidance 5.0; default scheduler; seed 5000+k | Native operating regime. No reduced resolution. |
| Prompts and sources | First COCO caption of photos never used before (all earlier ids are excluded) | Disjoint from every SD1.5/FLUX set. |
| Decoders on the *same* saved latent | **orig**: SDXL VAE, fp32. **taesdxl**: TAESDXL, the decoder-family **intervention**. **fp16fix**: sdxl-vae-fp16-fix, run in fp32, a **near-identity control** that differs in weights only. | fp16-fix is designed to reproduce the original, so its role is reported separately. |
| Scoring input | 448×448 centre crop of each 1024 image, no resizing | The same preprocessing as the real photographs. COCO photos are ≤ 640 px, so 1024-px real photographs do not exist in COCO, and resizing would add resampling traces. Full 1024 images are kept. |
| Calibration | **Fresh** independent COCO **train2017** photos with the same 448 crop (val2017 has only 387 unused photos): 200 in the pilot, 1,000 in the full run | Primary thresholds (nominal 1%/5% FPR). The paper's 1,000 calibration thresholds are printed alongside as a check. |
| Detectors | AEROBLADE (paper, SD1.5 AE); AEROBLADE-SDXL (SDXL AE, matched); NPR; UnivFD; CLIP/DINO probes (refit exactly as in the paper on the SD1.5 run); Community-Forensics; B-Free | Same weights and preprocessing as the paper. |
| Round-trip | The untouched evaluation photos, through the SDXL encoder, then each decoder | Replicates the denoiser-free round-trip result. |
| Statistics | Bootstrap over sources and calibration photos; paired drops; AUROC against the untouched evaluation photos; actual FPR | As in the paper. |

## Gates (technical only; no gate looks at a decoder effect)

1. Every latent and decoded image is finite and non-blank.
2. **TAESDXL latent convention.** The convention closest to the original decoder is chosen on 5 latents and must reach ≥ 18 dB PSNR. Otherwise: STOP.
3. **Detector reproduction.** AEROBLADE, NPR and Community-Forensics re-score 20 of the paper's SD1.5 images: median |Δ| ≤ 0.01 s.d. Otherwise: STOP. B-Free is checked by the E4 add-on.
4. The fp16-fix near-identity (≥ 30 dB) is **reported**, not gated. If it fails, fp16-fix is described as a second weight change, not a control.
5. **The pilot makes no scientific reading.** `full` refuses to start unless the pilot gate passed.

## Planned reading (full run, nominal 1% FPR)

**Usable detectors** have an original-decoder TPR ≥ 0.5. AEROBLADE with the SD1.5 AE is reported but not counted; AEROBLADE-SDXL is counted if usable.

| Outcome | Change in the paper |
|---|---|
| **Present**: ≥ 2 usable detectors with a TAESDXL drop of \|Δ\| ≥ 10 points, CI excluding 0 | Sec. 6 gains an SDXL paragraph and table rows. Contribution 2 says the decoder flip replicates on SDXL for those detectors. |
| **Absent**: every usable detector has \|Δ\| < 5 points and a CI within ±10 | Reported as a negative result. The paper says the effect did not replicate with TAESDXL on SDXL for these detectors. |
| **Mixed**: anything else | Per-detector reporting, no general claim. |

**The fp16-fix control** is expected to show \|Δ\| < 5 for every detector. Any detector with \|Δ\| ≥ 5 is reported as sensitive to small weight changes.

