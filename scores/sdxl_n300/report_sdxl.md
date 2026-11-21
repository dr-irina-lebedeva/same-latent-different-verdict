# E3 SDXL (full): N = 300 sources, 1000 fresh calibration photos

Gate: {"decode": {"taesdxl_scale_psnr": {"scaled": 29.30917074360657, "unscaled": 10.283967592637277}, "taesdxl_takes_scaled_latent": true, "psnr_fp16fix_vs_orig_1024": 47.08174410874902, "psnr_taesdxl_vs_orig_1024": 30.410911914432404, "psnr_roundtrip_orig_vs_real": 25.227611718488756}, "taesdxl_ok": true, "fp16fix_near_identity": true, "repro_median_abs_diff_sd": {"aeroblade": 0.0, "npr": 0.0, "commfor": 0.0}, "repro_ok": true, "passed": true}

Generation: {"model": "stabilityai/stable-diffusion-xl-base-1.0", "steps": 30, "guidance": 5.0, "res": 1024, "seed": "5000+k", "scheduler": "EulerDiscreteScheduler", "s_per_image": 5.8886802013715105}

TPR at nominal 1% FPR (threshold: fresh calibration). 95% CIs: bootstrap over sources and calibration photos.
Drop = TPR(orig) - TPR(alt), paired. Flip = share of sources whose decision differs across the 3 decoders.

| detector | TPR orig | TPR fp16fix (control) | TPR taesdxl | drop taesdxl [CI] | drop fp16fix [CI] | AUROC orig/taesdxl | FPR untouched | flip | round-trip orig / taesdxl flagged | thr fresh vs paper cal. |
|---|---|---|---|---|---|---|---|---|---|---|
| aeroblade | 0.26 | 0.25 | 0.23 | +3 [-1,+9] | +1 [-1,+3] | 0.948/0.918 | 0.000 | 0.07 | 0.03 / 0.01 | -0.0213 vs -0.0204 |
| aero_sdxl | 0.92 | 0.92 | 0.17 | +75 [+61,+81] | +0 [-1,+1] | 1.000/0.914 | 0.000 | 0.75 | 0.91 / 0.00 | -0.0175 vs -0.0168 |
| npr | 0.52 | 0.51 | 0.37 | +15 [+8,+21] | +0 [-6,+2] | 0.974/0.959 | 0.003 | 0.28 | 0.31 / 0.21 | 9.94 vs 5.36 |
| univfd | 0.22 | 0.22 | 0.71 | -48 [-55,-44] | +1 [-1,+3] | 0.855/0.977 | 0.010 | 0.50 | 0.07 / 0.73 | -0.217 vs -0.136 |
| probe_clip | 0.87 | 0.87 | 0.98 | -11 [-21,-6] | +0 [-2,+3] | 0.996/0.999 | 0.013 | 0.16 | 0.10 / 0.84 | 0.972 vs 1.09 |
| probe_dino | 0.16 | 0.17 | 0.20 | -4 [-9,+0] | -1 [-2,+2] | 0.846/0.885 | 0.013 | 0.15 | 0.03 / 0.05 | 3.4 vs 3.48 |
| commfor | 1.00 | 1.00 | 1.00 | +0 [+0,+0] | +0 [+0,+0] | 1.000/1.000 | 0.020 | 0.00 | 0.88 / 1.00 | -6.01 vs -6.03 |
| bfree | 1.00 | 1.00 | 1.00 | +0 [+0,+0] | +0 [+0,+0] | 1.000/1.000 | 0.027 | 0.00 | 0.98 / 1.00 | -0.28 vs 0.437 |

Usable detectors (TPR orig >= 0.5 at 1% FPR; AEROBLADE with the SD1.5 AE is reported but not counted): ['aero_sdxl', 'npr', 'probe_clip', 'commfor', 'bfree']

**Pre-registered reading:** DECODER EFFECT PRESENT on SDXL (|drop| >= 10 pts, CI excludes 0) for ['aero_sdxl', 'npr', 'probe_clip'].

Control (fp16-fix, near-identity): detectors with |drop| >= 5 pts: none