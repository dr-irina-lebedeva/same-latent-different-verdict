"""Supplement Sec. S14 (add-on run): LaTeX tables from the add-on run folder written by dod_v4.py.

    python make_supp_v4.py <add-on run dir> supp_v4.tex

The surrounding prose was edited by hand in the final supplement.
"""
import sys
import pandas as pd, numpy as np, json
R=sys.argv[1].rstrip("/")+"/"
q=pd.read_csv(R+'quality_v4.csv'); rep=open(R+'report_v4.md').read()
L=[]; A=L.append
A(r"\clearpage\section{Add-on run: full reconstruction quality, Community-Forensics under JPEG, B-Free}\label{sec:v4}")
A(r"The main SD1.5 run was regenerated with the same scripts, arguments, photographs, seeds and model revisions (\texttt{run\_v4.sh}, \texttt{dod\_v4.py}). Before any new number was used, a reproduction gate written in the script compared the regenerated run with the stored one: COCO ids and order are identical (300/300); UnivFD and NPR scores on untouched photographs and on generated images (original and TAESD decoders), per-image PSNR of the three conventional decoders, and Community-Forensics scores on Consistency-Decoder images (50 latents) agree exactly (median and maximum absolute difference 0.00\,s.d.\ or dB); all Hugging Face model revisions are unchanged. The new results therefore describe the same images as the paper. The interpretation of every outcome was fixed in the kit's README before the run.")
A(r"\paragraph{Reconstruction quality, all 300 photographs.} Same metrics as Table~1 of the paper (LPIPS-AlexNet, HF ratio relative to the photograph), plus SSIM.")
A(r"\begin{table}[H]\centering\small\begin{tabular}{lcccc}\toprule Decoder & PSNR (dB) & LPIPS & HF ratio & SSIM \\ \midrule")
nm={'orig':'SD1.5 original','ftmse':'sd-vae-ft-MSE','taesd':'TAESD','cons':'Consistency Decoder'}
for d,r in q.groupby('decoder',sort=False):
    A(f"{nm[d]} & {r.psnr.mean():.2f} & {r.lpips.mean():.3f} & {r.hf_ratio.mean():.2f} & {r.ssim.mean():.3f} \\\\")
c=q[q.decoder=='cons'].psnr
A(r"\bottomrule\end{tabular}\caption{Reconstruction quality on all 300 encoded photographs. Consistency Decoder: seed 3000+$k$; first 20 photographs 23.24\,dB, as in the acceptance gate of the main run; all 300: %.2f\,dB.}\end{table}" % c.mean())
def mdtab(title):
    s=rep.split(title,1)[1]; out=[]
    for l in s.split('\n'):
        if not l.startswith('| '): 
            if out: break
            continue
        cells=[x.strip() for x in l.strip().strip('|').split('|')]
        if cells[0]=='q' or len(cells)!=12: continue
        out.append(cells)
    return out
def esc(x): return x.replace('%',r'\%').replace('-',r'$-$').replace('$$','')
def emit(rows,cap):
    A(r"\begin{table}[H]\centering\scriptsize\setlength{\tabcolsep}{2pt}\resizebox{\linewidth}{!}{\begin{tabular}{llcccccccc}\toprule")
    A(r"q & decoder & AUROC & TPR@1\% & drop@1\% [CI] & TPR@5\% & drop@5\% [CI] & caught$\to$missed @1\% & min margin (s.d.) & $\Delta$ vs orig [CI] \\ \midrule")
    for r in rows:
        A(" & ".join(esc(r[i]) for i in [0,1,2,3,4,5,6,7,8,10])+r" \\")
    A(r"\bottomrule\end{tabular}}\caption{"+cap+"}\end{table}")
A(r"\paragraph{Community-Forensics under JPEG.} Same JPEG function and qualities as for the other detectors (JPEG section); thresholds re-calibrated on the first 300 JPEG-compressed independent photographs, clean thresholds from all 1{,}000. Min margin: lowest generated score minus the 1\% threshold, in s.d.\ units of the clean data (negative = some images missed). Actual FPR on untouched evaluation photographs: 0.00--0.01 at 1\%, 0.03--0.05 at 5\%. Planned reading: TPR $\ge0.95$ for every decoder and quality with a decoder swing $\le5$ points means decisions are preserved under JPEG; this holds. The Consistency Decoder still moves up to 5 of 300 images below the 1\% threshold, and the number of generated images within 0.5\,s.d.\ of it grows from 15 (uncompressed) to 39, 65 and 84 at quality 95, 85 and 75. With the clean threshold (a detector unaware of JPEG) TPR stays $\ge0.99$ but the FPR at a nominal 5\% rises to 0.07, 0.12 and 0.17.")
emit(mdtab("## Community-Forensics: clean and JPEG"),"Community-Forensics on identical latents, uncompressed and after JPEG.")
bs=json.load(open(R+'bfree_status.json'))
A(r"\paragraph{B-Free.} Official repository (commit \texttt{%s}), official weights (MD5 \texttt{%s}) and inference script \texttt{main\_bfree.py}, which applies only tensor conversion and normalisation to the whole $448\times448$ image (no crop, no resizing). The official demo reproduces the published demo scores (maximum absolute difference $%.1e$). Score: fake logit. Acceptance gate G1 (as for Community-Forensics): AUROC 1.000 and TPR 1.00 at 5\%% FPR with the original decoder; passed. Actual FPR on untouched evaluation photographs 0.01 and 0.04 at nominal 1\%% and 5\%%. B-Free under JPEG is reported in Sec.~\ref{sec:v19} (E5)." % (bs['repo_commit'][:12], bs['weights_md5'], bs['demo_max_abs_diff']))
emit(mdtab("## B-Free (clean only)"),"B-Free on identical SD1.5 latents (uncompressed).")
A(r"\paragraph{FLUX.1 calibration overlap (checked by COCO id).} We exported the COCO ids of every photograph set from the saved run metadata (\texttt{scores/flux\_manifest.csv}) and intersected them directly. The 100 FLUX.1 source photographs (the pilot photographs) share \textbf{0} ids with the pilot calibration set used for the FLUX.1 NPR results in the paper, and \textbf{57} with the calibration set of the main SD1.5 run, which is why the 0.49 img2img value obtained with the latter is reported only for comparison. The FLUX.1 sources share no id with the 300 main evaluation photographs, and the main calibration set shares no id with the main evaluation photographs. The calibration photographs of the add-on run are the stored ones: Community-Forensics scores on all 1{,}000 match the stored scores exactly (maximum difference 0).")
open(sys.argv[2],'w').write("\n".join(L)+"\n")
print(len(L))
