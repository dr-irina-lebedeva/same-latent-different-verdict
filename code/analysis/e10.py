"""E10: projection of the decoder shift onto the UnivFD and probe decision directions (supplement Sec. S17).

Uses the stored CLIP/DINOv2 features and the UnivFD linear-head weights. Writes e10_projection.csv to the
current folder.

    python e10.py scores/sd15_main_n300_e4

CPU only.
"""
import numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import sys
R = sys.argv[1].rstrip("/") + "/"   # run dir with feats_*.npy, feats_index.csv, univfd_fc_weights.pth
idx=pd.read_csv(R+"feats_index.csv"); D=["orig","ftmse","taesd","cons"]
F={n:np.load(R+f"feats_{n}.npy").astype(np.float64) for n in ["clip","dino"]}
def rows(g,d,lo=0,hi=300):
    m=(idx.group==g)&(idx.decoder==d)&(idx.idx>=lo)&(idx.idx<hi); return idx[m].sort_values("idx").index.values
cal=idx[idx.group=="calib"].index.values
w=torch.load(R+"univfd_fc_weights.pth",map_location="cpu")
dets={"univfd":("clip",w["weight"].double().numpy().reshape(-1),0)}
for bb in ["clip","dino"]:
    X=F[bb]; tr=np.r_[rows("real","none",0,150),rows("gen","orig",0,150)]; y=np.r_[np.zeros(150),np.ones(150)]
    clf=make_pipeline(StandardScaler(),LogisticRegression(C=0.5,max_iter=5000)).fit(X[tr],y)
    dets[f"probe_{bb}"]=(bb,clf[-1].coef_.reshape(-1)/clf[0].scale_,150)
out=[]
for name,(bb,wv,lo) in dets.items():
    X=F[bb]; sd=np.std(X[cal]@wv)
    sep=np.median(np.linalg.norm(X[rows("gen","orig",lo)]-X[rows("real","none",lo)],axis=1))
    u=wv/np.linalg.norm(wv)
    for d in D[1:]:
        dh=X[rows("gen",d,lo)]-X[rows("gen","orig",lo)]; dr=X[rows("real_enc",d,lo)]-X[rows("real_enc","orig",lo)]
        df=dh@wv/sd; mu=dh.mean(0); mur=dr.mean(0)
        cos_i=(dh@u)/np.linalg.norm(dh,axis=1)
        # variance of df explained by the common (mean) displacement: share of mean^2 in E[df^2]
        out.append(dict(detector=name,decoder=d,rel_norm=np.linalg.norm(dh,axis=1).mean()/sep,
            mean_cos_dh_w=cos_i.mean(),cos_mean_dh_w=mu@u/np.linalg.norm(mu),mean_df_sd=df.mean(),
            common_share=df.mean()**2/np.mean(df**2),
            rec_rel_norm=np.linalg.norm(dr,axis=1).mean()/sep, rec_mean_df_sd=(dr@wv/sd).mean(),
            cos_gen_vs_rec_mean=mu@mur/np.linalg.norm(mu)/np.linalg.norm(mur)))
o=pd.DataFrame(out); o.to_csv("e10_projection.csv",index=False)
pd.set_option("display.width",250); print(o.round(3).to_string(index=False))
