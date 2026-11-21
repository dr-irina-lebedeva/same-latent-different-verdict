"""Post hoc: per-latent decision agreement across the three Consistency Decoder seeds
(cons = seed 3000+idx, cons_s1, cons_s2), idx 150-199, at the paper's 1% FPR threshold
(99th percentile of the 1,000 calibration photographs).
Usage: python seed_agreement.py scores/sd15_main_n300 out.csv"""
import sys, pandas as pd, numpy as np
d, out = sys.argv[1], sys.argv[2]
FILES = ("scores.csv", "aeroblade_scores.csv", "extra_scores.csv", "v3_base_scores.csv", "v3_new_scores.csv", "v3_seed_scores.csv")
S = pd.concat([pd.read_csv(f"{d}/{f}") for f in FILES]).drop_duplicates(["group", "decoder", "idx", "detector"])
rows = []
for det in ["aeroblade", "npr", "univfd", "probe_clip", "probe_dino", "commfor"]:
    s = S[S.detector == det]
    thr = np.percentile(s[s.group == "calib"].score, 99)
    g = s[(s.group == "gen") & s.decoder.isin(["cons", "cons_s1", "cons_s2"])]
    p = g.pivot_table(index="idx", columns="decoder", values="score", aggfunc="first").dropna()
    p = p[[c for c in ["cons", "cons_s1", "cons_s2"]]]
    dec = p > thr
    agree = (dec.all(axis=1) | (~dec).all(axis=1))
    rows.append(dict(detector=det, n=len(p), tpr_seed0=dec.cons.mean(), tpr_seed1=dec.cons_s1.mean(),
                     tpr_seed2=dec.cons_s2.mean(), all3_agree=agree.mean(), n_vary=int((~agree).sum())))
pd.DataFrame(rows).to_csv(out, index=False, float_format="%.3f"); print(pd.DataFrame(rows))
