# Plan for E2, E5 and the round-trip controls (saved 2026-11-14 17:35 UTC, before these runs; run instructions removed)

## Reading rule (fixed before the run)

### E2 (main reading: TPR at nominal 1% FPR of the matched autoencoder on Consistency-Decoder images; original AE: 0.11)

| outcome | change in the paper |
|---|---|
| ≥ 0.80: full recovery | Fig. 3 gains a Consistency column and row. Sec. 5 says a matched autoencoder restores AEROBLADE for the diffusion decoder as well. |
| 0.30–0.80: partial recovery | The same figure change. Sec. 5 says matching helps but does not fully restore detection for a distilled or diffusion decoder. |
| < 0.30: no recovery | Sec. 5 says that this reconstruction method (SD1.5 encoder + Consistency Decoder, one seed per image, LPIPS layer 2) does not restore AEROBLADE. We draw no conclusion about other reconstruction methods or about whether a fingerprint exists. |

**Best of 4** is reported in Fig. 3 and the supplement in every case. If it reaches ≥ 0.50 for every decoder and ≥ 0.90 for the original, practical guidance (iii) adds "an ensemble that includes the diffusion decoder's autoencoder". Otherwise the paper notes that adding autoencoders tightens the shared threshold.

### E5 (thresholds recalibrated for nominal 1% FPR on JPEG'd photographs)

| outcome | change in the paper |
|---|---|
| **Robust**: TPR ≥ 0.95 for every decoder and q, and decoder swing ≤ 5 points | Limitations: "B-Free was not tested under JPEG" is removed. Sec. 6: both generality-trained detectors keep their decisions under JPEG q95–75. |
| **Decoder-dependent**: drop ≥ 10 points with a CI that excludes 0, at some q | New sentence in Sec. 6: compression exposes a decoder dependence of B-Free that is hidden on uncompressed images. Contribution 3 is qualified. |
| **JPEG-fragile**: original-decoder TPR ≤ 0.80, with no decoder dependence | Reported as a compression limitation of B-Free, not a decoder effect. |
| anything else | Supplement table only. |

The clean-threshold FPR under JPEG goes in the supplement, next to the Community-Forensics drift. The frozen paper numbers are never replaced.

## Add-on: round-trip controls and E4/E10 features (`bash run_e4.sh`)

Nothing new is generated, and the paper's numbers are not touched.

- **Features for E4/E10.** The CLIP ViT-L/14 and DINOv2 features of the paper's probes and UnivFD, with the same preprocessing, for every image of the main run. **Gate:** UnivFD recomputed from these features must reproduce the paper's UnivFD scores (median |Δ| ≤ 0.01 s.d.).
- **Round-trip control.** The 300 untouched photos have their JPEG traces removed *without* a VAE, in three ways: a 4-px grid shift, a bicubic ×0.5 down-up resize, and σ = 2/255 noise. Each is scored by all seven detectors at the paper's thresholds. B-Free must first reproduce its paper scores on 20 photos.
- **Planned reading:**
  - If a detector flags the controls at ≤ 10% (1% FPR) while flagging the VAE round-trip at ≥ 90%, JPEG-trace removal alone does not explain its round-trip verdict.
  - If a control also reaches ≥ 50%, the round-trip result for that detector is reported as confounded with JPEG-trace removal.
  - Anything in between is reported as is.

