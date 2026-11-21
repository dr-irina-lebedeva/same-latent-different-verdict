"""E4: held-out-decoder probe training on stored CLIP/DINOv2 features (supplement Sec. S17).

Trains linear probes with one decoder left out and tests on that decoder, with thresholds from the
independent calibration photographs. Writes e4_probe_results.csv and e4_gains.csv to the current folder.

    python e4_e10.py scores/sd15_main_n300_e4

CPU only. The plan is in protocols/PLAN_E4_E10.md.
"""
import numpy as np, pandas as pd, torch, json
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import sys
R = sys.argv[1].rstrip("/") + "/"   # run dir with feats_*.npy, feats_index.csv, univfd_fc_weights.pth
idx=pd.read_csv(R+"feats_index.csv"); D=["orig","ftmse","taesd","cons"]
F={n:np.load(R+f"feats_{n}.npy").astype(np.float64) for n in ["clip","dino"]}
def rows(g,d,lo,hi): 
    m=(idx.group==g)&(idx.decoder==d)&(idx.idx>=lo)&(idx.idx<hi); return idx[m].sort_values("idx").index.values
cal=idx[idx.group=="calib"].index.values
rng=np.random.default_rng(0); B=2000
out=[]; e10=[]
for bb in ["clip","dino"]:
    X=F[bb]
    neg=rows("real","none",0,150)
    fk={d:rows("gen",d,0,150) for d in D}
    te={d:rows("gen",d,150,300) for d in D}; tr_real=rows("real","none",150,300)
    def fit(fakes):
        Xt=np.r_[X[neg],X[fakes]]; y=np.r_[np.zeros(len(neg)),np.ones(len(fakes))]
        return make_pipeline(StandardScaler(),LogisticRegression(C=0.5,max_iter=5000)).fit(Xt,y)
    def matched(decs,seed=0):
        r=np.random.default_rng(seed); ch=r.integers(0,len(decs),150)
        return np.array([fk[decs[c]][k] for k,c in enumerate(ch)])
    conds={"A orig only":fit(fk["orig"])}
    for h in ["ftmse","taesd","cons"]:
        tr=[d for d in D if d!=h]
        conds[f"B matched, {h} held out"]=fit(matched(tr))
        conds[f"B all, {h} held out"]=fit(np.concatenate([fk[d] for d in tr]))
    conds["C matched, all seen"]=fit(matched(D)); conds["C all, all seen"]=fit(np.concatenate([fk[d] for d in D]))
    S={c:{"cal":m.decision_function(X[cal]),"real":m.decision_function(X[tr_real]),**{d:m.decision_function(X[te[d]]) for d in D}} for c,m in conds.items()}
    for c,s in S.items():
        for q in [0.99,0.95]:
            t=np.quantile(s["cal"],q)
            r={"backbone":bb,"cond":c,"fpr":f"{round((1-q)*100)}%","FPR_test_real":np.mean(s["real"]>t)}
            for d in D:
                r[f"TPR_{d}"]=np.mean(s[d]>t)
                r[f"AUC_{d}"]=float((s[d][:,None]>s["real"][None,:]).mean())
            out.append(r)
    # bootstrap gains at 1%
    for h in ["ftmse","taesd","cons"]:
        A=S["A orig only"]; Bm=S[f"B matched, {h} held out"]
        def g(ci,ti):
            tA=np.quantile(A["cal"][ci],0.99); tB=np.quantile(Bm["cal"][ci],0.99)
            return [np.mean(Bm[h][ti]>tB)-np.mean(A[h][ti]>tA), np.mean(Bm["orig"][ti]>tB)-np.mean(A["orig"][ti]>tA)]
        est=g(np.arange(len(cal)),np.arange(150))
        bs=np.array([g(rng.integers(0,len(cal),len(cal)),rng.integers(0,150,150)) for _ in range(B)])
        lo,hi=np.percentile(bs,2.5,0),np.percentile(bs,97.5,0)
        e10.append(dict(backbone=bb,held_out=h,gain_heldout=est[0],gain_lo=lo[0],gain_hi=hi[0],orig_change=est[1],orig_lo=lo[1],orig_hi=hi[1]))
res=pd.DataFrame(out); gains=pd.DataFrame(e10)
res.to_csv("e4_probe_results.csv",index=False); gains.to_csv("e4_gains.csv",index=False)
pd.set_option("display.width",250)
print(res[res.fpr=="1%"].drop(columns=[c for c in res.columns if c.startswith("AUC")]).round(3).to_string(index=False))
print(gains.round(3).to_string(index=False))
for bb in ["clip","dino"]:
    g=gains[gains.backbone==bb]
    ok=((g.gain_heldout>=0.10)&(g.gain_lo>0)&(g.orig_change>=-0.05)).sum()
    print(bb,"rotations passing:",ok,"->", "MITIGATION DEMONSTRATED" if ok>=2 else "no demonstrated mitigation")
