# E2: AEROBLADE with the Consistency Decoder's autoencoder

N = 300 generated latents per decoder, 1000 calibration photographs, thresholds at their 99th / 95th percentile. TPR [95% CI over sources and calibration photos]; FPR = actual flag rate on the untouched evaluation photos.


## nominal 1% FPR

| reconstruction AE | orig | ftmse | taesd | cons | actual FPR |
|---|---|---|---|---|---|
| aero_orig | 1.00 [0.97, 1.00] | 0.36 [0.23, 0.61] | 0.16 [0.10, 0.31] | 0.11 [0.08, 0.21] | 0.003 |
| aero_ftmse | 0.29 [0.19, 0.52] | 1.00 [0.99, 1.00] | 0.14 [0.10, 0.30] | 0.11 [0.07, 0.22] | 0.003 |
| aero_taesd | 0.18 [0.12, 0.32] | 0.19 [0.13, 0.35] | 0.53 [0.43, 0.73] | 0.12 [0.08, 0.21] | 0.003 |
| aero_cons | 0.13 [0.09, 0.25] | 0.16 [0.10, 0.29] | 0.13 [0.09, 0.23] | 0.11 [0.07, 0.21] | 0.003 |
| aero_best4 | 0.98 [0.94, 1.00] | 1.00 [0.99, 1.00] | 0.21 [0.13, 0.39] | 0.11 [0.07, 0.22] | 0.003 |

## nominal 5% FPR

| reconstruction AE | orig | ftmse | taesd | cons | actual FPR |
|---|---|---|---|---|---|
| aero_orig | 1.00 [1.00, 1.00] | 0.85 [0.77, 0.89] | 0.54 [0.44, 0.60] | 0.39 [0.29, 0.46] | 0.027 |
| aero_ftmse | 0.74 [0.67, 0.83] | 1.00 [1.00, 1.00] | 0.51 [0.44, 0.61] | 0.37 [0.30, 0.47] | 0.027 |
| aero_taesd | 0.57 [0.47, 0.63] | 0.63 [0.54, 0.70] | 0.90 [0.86, 0.94] | 0.39 [0.30, 0.47] | 0.027 |
| aero_cons | 0.42 [0.35, 0.52] | 0.51 [0.43, 0.59] | 0.42 [0.33, 0.49] | 0.34 [0.27, 0.43] | 0.030 |
| aero_best4 | 1.00 [0.99, 1.00] | 1.00 [1.00, 1.00] | 0.64 [0.58, 0.74] | 0.37 [0.30, 0.47] | 0.027 |

Seed check (100 images, second seed): median |score diff| = 0.019 s.d., Spearman 0.999.


## E5: B-Free under JPEG (thresholds recalibrated on the 300 JPEG'd calibration photos)

| q | TPR@1% orig | TPR@1% ftmse | TPR@1% taesd | TPR@1% cons | max drop vs orig [CI] | FPR (recal. 5%) | FPR (clean 5% thr.) |
|---|---|---|---|---|---|---|---|
| 95 | 1.000 [1.00,1.00] | 1.000 [1.00,1.00] | 1.000 [0.99,1.00] | 0.987 [0.94,1.00] | 1 [0,6] (cons) | 0.050 | 0.073 |
| 85 | 1.000 [1.00,1.00] | 1.000 [0.99,1.00] | 1.000 [1.00,1.00] | 0.993 [0.94,1.00] | 1 [0,6] (cons) | 0.047 | 0.097 |
| 75 | 1.000 [1.00,1.00] | 0.997 [0.99,1.00] | 1.000 [1.00,1.00] | 0.980 [0.92,1.00] | 2 [0,8] (cons) | 0.050 | 0.133 |

**Pre-registered reading (E5):** ROBUST (TPR >= 0.95 everywhere, decoder swing <= 5 points).


**Pre-registered reading:** matched AE on Consistency-Decoder images, TPR at 1% FPR = 0.11: NO recovery (< 0.30) with this reconstruction method.
