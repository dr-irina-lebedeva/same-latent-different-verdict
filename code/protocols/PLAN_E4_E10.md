<!-- file timestamp 2026-11-14 18:46:19 UTC (the "18:50" in the header below was typed by hand); written after feature extraction and before any probe was trained on the features; the first analysis output is timestamped 18:46:47 -->
# E4 / E10 analysis plan. Written 2026-11-14 18:50 UTC, BEFORE any E4/E10 number was computed.
(The feature gate and the round-trip control report had already been seen; neither involves training across decoders.)

## E4: held-out-decoder mitigation (linear probes)
- Backbones: CLIP ViT-L/14 image embeddings and DINOv2-base CLS (the paper's features). Probe: StandardScaler + LogisticRegression(C=0.5), exactly as in the paper. No hyper-parameter search.
- Split by source id: train k < 150, test k >= 150. Every image derived from one source stays in its split. Training negatives: untouched real photos k < 150.
- Thresholds: 99th / 95th percentile of the 1,000 independent calibration photos (never used for training). No use of the held-out decoder for anything except testing.
- Conditions:
  - A, orig only: 150 fakes (the paper's probe).
  - B_h, h in {ft-MSE, TAESD, Cons} held out (rotation over all three), train decoders = the other three incl. orig.
    - Matched budget (primary): 150 fakes, each training source contributes one decode from a decoder drawn uniformly from the training decoders (seed 0).
    - Unmatched (secondary): all 450 decodes.
  - C, all four seen (upper-bound diagnostic): matched (150) and unmatched (600).
- Metrics on the 150 test sources: TPR at 1% / 5% FPR per decoder; AUROC vs the 150 test untouched photos; actual FPR on them. Paired bootstrap (2,000; test sources and calibration photos) for the gain B_h(matched) - A on the held-out decoder h, and for the change on the original decoder.
- Success criterion (per backbone, nominal 1% FPR): in >= 2 of the 3 rotations, the held-out-decoder TPR gain is >= 10 points with CI excluding 0, AND the original-decoder TPR falls by <= 5 points. Otherwise the result is "no demonstrated mitigation" (reported as is, incl. when the baseline gap is < 10 points and cannot be closed by 10).

## E10: feature displacement on the decision direction
- For UnivFD (official head w) and the paper's CLIP / DINO probes (effective direction w_eff = coef / scaler.scale_), per test latent and decoder j: dh = h(D_j z) - h(D_orig z); df = w_eff . dh (exact logit change).
- Report per detector x decoder: mean ||dh|| (relative to the median distance between untouched real and orig-generated features), mean cosine(dh, w), share of the variance of df explained by the mean displacement direction, and the same quantities for reconstructed real photos (real_enc j - real_enc orig). Report the cosine between the mean generated displacement and the mean reconstructed-real displacement (links to ReconScreen).
- Descriptive only; no pass/fail. Claim at most: "decoders move features by similar amounts, but the projection on the decision direction differs in sign/size", if the numbers show that.
