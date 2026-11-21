"""Build the supplement tables (supp.tex) from existing run reports. No new experiments.
Expects the run-folder layout written by the pipelines:  python build_supp.py"""
import sys, re, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from md2tex import sections, to_tex
from PIL import Image
B = Path(__file__).resolve().parents[2]
R3, R8, FLX, P1 = B / "results_v3/full/runs/n300", B / "results_nov8/runs/n300", B / "results_nov8/runs/flux100", B / "results_pilot_n100"
DIAG = B / "results_v3/diag_reanalysis_nov14/report_diag_reanalysis.md"
OUT = Path(__file__).parent; (OUT / "img").mkdir(exist_ok=True)
rd = lambda p: p.read_text()
def drop_cols(rows, names):
    keep = [j for j, c in enumerate(rows[0]) if not any(n in c for n in names)]
    return [[r[j] for j in keep if j < len(r)] for r in rows]
T = []; A = T.append
def tabs(md, pick=None, cap=lambda h: h, drop=(), label=None, size=r"\scriptsize"):
    for h, rows in sections(md):
        if pick and not pick(h, rows): continue
        if drop: rows = drop_cols(rows, drop)
        A(to_tex(rows, cap(h), size=size))
def jpg(src, dst, crop=None, maxw=1800):
    im = Image.open(src).convert("RGB")
    if crop: im = im.crop(crop)
    if im.width > maxw: im = im.resize((maxw, int(im.height * maxw / im.width)), Image.LANCZOS)
    im.save(OUT / "img" / dst, quality=88); return f"img/{dst}"
NM = {"aeroblade": "AEROBLADE", "npr": "NPR", "univfd": "UnivFD", "probe_clip": "CLIP probe", "probe_dino": "DINO probe", "commfor": "Community-Forensics"}
def nm(h):
    h = h.replace("Detector ", "").replace("`", "").strip()
    w = h.split()[0] if h else h
    return (NM.get(w, w) + h[len(w):]).replace("_", r"\_")

A(r"""\documentclass[10pt,letterpaper]{article}
\usepackage[margin=0.9in]{geometry}
\usepackage{times,amsmath,amssymb,booktabs,graphicx,adjustbox,xcolor,float}
\usepackage[hidelinks]{hyperref}
\renewcommand{\thetable}{S\arabic{table}}\renewcommand{\thefigure}{S\arabic{figure}}\renewcommand{\thesection}{S\arabic{section}}
\title{Supplementary Material\\[2pt]\large Decoder or Denoiser? Same Latent, Different Verdict in AI-Generated Image Detection}
\author{Irina Lebedeva}\date{}
\begin{document}\maketitle
\noindent This supplement contains only results from the runs reported in the paper; no experiment was added. Tables S1--S\ref{tab:last} are generated directly from the run reports by the included script \texttt{paper/supp/build\_supp.py}. Effects $\Delta$ are paired mean score changes (alternative minus original decoder on the same latent) in units of the detector's score s.d.\ over untouched real and original-decoder generated images. Brackets are 95\% percentile bootstrap intervals (2{,}000 replicates, resampling source photographs; for calibrated TPR changes the calibration photographs are resampled too). Code and per-image scores are included in \texttt{code/} and \texttt{scores/}; they will be released publicly upon acceptance.
\tableofcontents
""")
# ---------------- S1 setup
rev = rd(R3 / "model_revisions.txt").splitlines()
vers = json.loads(rd(R3 / "v3_versions.json")); dst = json.loads(rd(R3 / "detector_status.json"))
A(r"""\section{Experimental details}
\paragraph{Photographs and captions.} COCO val2017 photographs with shorter side $\ge448$\,px, centre-cropped to $448\times448$ without resizing; prompt = first COCO caption. Pilot: run indices 0--99. Main SD1.5 run: the next 300 eligible photographs (indices 100--399, disjoint from the pilot). Calibration: 1{,}000 further COCO photographs not in the respective run (same crop, no resizing). For the FLUX.1 NPR thresholds we use the pilot run's 1{,}000 calibration photographs, which share no COCO image id with the 100 FLUX.1 source photographs (checked); the main run's calibration set excludes the 300 main-run photographs but was not checked against the pilot photographs.
\paragraph{Generation.} SD1.5 (\texttt{sd-legacy/stable-diffusion-v1-5}): 30 steps, guidance 7.5, $448\times448$; text-to-image seed $1000+k$ and img2img seed $2000+k$ for run index $k$; img2img strength 0.5 (history sweep 0.3/0.5/0.7, same seeds). Latents are stored and decoded by every decoder, so all decoders see bit-identical latents. FLUX.1-schnell: 4 steps, guidance 0, same seed convention, $N=100$ (pilot photographs). Consistency Decoder: seed $3000+k$ for the main results; $4000+k$ and $5000+k$ for the seed check on 50 latents. Encoded real latents use the posterior mean.
\paragraph{Detectors and preprocessing.} UnivFD: official linear head on CLIP ViT-L/14; NPR: official weights; probes: logistic regression on frozen CLIP ViT-L/14 and DINOv2 ViT-B/14 features, trained on run indices $k<N/2$ (untouched real vs.\ original-decoder generated only) and evaluated on $k\ge N/2$; all variants of a photograph fall in the same split. AEROBLADE: our implementation computes the LPIPS (VGG, second layer) distance between an image and its reconstruction by the SD1.5 autoencoder (score = negative distance); the ``best of 3'' variant takes the minimum distance over the original, ft-MSE and TAESD autoencoders. Community-Forensics: official ViT-S/16 (384\,px) weights and test transform. Preprocessing:
\begin{itemize}\setlength\itemsep{0pt}""")
for k, v in dst["preprocessing"].items(): A(r"\item " + k.replace("_", r"\_") + ": " + v.replace("_", r"\_"))
A(r"\item probes: " + dst["probe_split"].replace("_", r"\_").replace("<", "$<$").replace(">=", "$\\ge$") + r"\end{itemize}")
A(r"""\paragraph{Thresholds and intervals.} $\tau$ is the empirical 99th (95th) percentile of a detector's scores on the 1{,}000 calibration photographs (\texttt{numpy.quantile}, linear interpolation), giving a nominal 1\% (5\%) FPR; it is never re-tuned per decoder. The 1\% threshold rests on the top ten calibration scores. TPR-change intervals resample calibration photographs and evaluation sources jointly and recompute $\tau$ in every replicate. ``Caught$\to$missed'' counts generated images above $\tau$ with the original decoder and below it with the alternative decoder on the same latent, out of those caught with the original decoder.
\paragraph{Planned gates (v3 run).} G1: a new detector is used only if baseline AUROC $\ge0.80$ and TPR $\ge0.50$ at 5\% FPR. G2: the Consistency Decoder is reported only if real-photograph round-trip PSNR $\ge20$\,dB. G3: its effect is reported only if the range of the mean shift over three decoder seeds is $<50\%$ of the mean (or the shift is $<0.10$ s.d.). Diagnostic: see Section~\ref{sec:diag}.
\paragraph{Model identifiers and revisions (main v3 run).}
\begin{center}\scriptsize\begin{tabular}{ll}\toprule identifier & revision / SHA-256 \\ \midrule""")
for l in rev:
    if l.startswith("command:"): continue
    a, b = l.split(maxsplit=1) if " " in l else (l, "")
    if "cache/" in b or len(a) == 64: a, b = b.strip(), a
    A(a.replace("_", r"\_") + r" & \texttt{" + b.strip()[:40] + r"} \\")
A(r"\bottomrule\end{tabular}\end{center}")
A(r"Software: " + ", ".join(f"{k} {v}" for k, v in vers.items() if k in ("python", "torch", "diffusers", "timm")) + r"; Community-Forensics backbone \texttt{" + vers["commfor_timm"].replace("_", r"\_") + r"}; full \texttt{pip freeze} in \texttt{code/}. The first SD1.5 run and its rerun inside the v3 run produced identical scores (max.\ abs.\ difference 0).")
# ---------------- S2 main SD1.5 tables
A(r"\section{Full SD1.5 results with intervals ($N=300$)}")
A(r"Per-detector results of the main run. AUROC columns compare untouched photographs with generated images (gen) or with VAE-reconstructed photographs (real-enc). The ``interaction'' column is the secondary $I^G=\Delta_{\text{gen}}-\Delta_{\text{real-enc}}$ (different content). The primary same-photograph interaction $I^H$ (ft-MSE, img2img $\sigma{=}0.5$ minus encoded photograph) is: AEROBLADE $+0.01$ [$-0.00$, $+0.02$], NPR $-0.14$ [$-0.18$, $-0.09$], CLIP probe $+0.02$ [$-0.05$, $+0.08$], DINO probe $+0.09$ [$+0.05$, $+0.14$], UnivFD $-0.07$ [$-0.14$, $-0.01$]; all strengths in Section~\ref{sec:hist}. TPR/FPR columns of the original report that used thresholds from the evaluation photographs are omitted here; calibrated operating points are in Section~\ref{sec:cal}.")
tabs(rd(R8 / "report_with_aeroblade.md"), pick=lambda h, r: h.startswith("Detector") or h.startswith("Decoder quality"),
     cap=lambda h: ("SD1.5 " + nm(h) + " ($N=300$; probes 150 held-out sources).") if h.startswith("Detector") else "Reconstruction quality on encoded real photographs ($N=300$).",
     drop=("TPR@", "FPR on real-enc", "flips"))
A(r"\clearpage\section{Calibrated operating points}\label{sec:cal}")
tabs(rd(R8 / "report_extras.md"), pick=lambda h, r: h.startswith("Thresholds"),
     cap=lambda h: "Thresholds from 1{,}000 independent photographs (SD1.5 main run). ``FPR on run reals'': actual FPR on the 300 untouched evaluation photographs. real-enc/flux columns: share of VAE-reconstructed real photographs above $\\tau$ (FLUX.1 VAE vs.\\ TAEF1 reconstructions of the same photographs).")
md3 = rd(R3 / "report_v3.md")
tabs(md3, pick=lambda h, r: h.startswith("Detection after"), cap=lambda h: "All detectors and decoders, including the Consistency Decoder (one seed per latent) and AEROBLADE with each reconstruction autoencoder. drop = TPR(orig)$-$TPR(alt) in points [95\\% CI]; caught$\\to$missed $=k/m$.")
A(r"\section{AEROBLADE: generating decoder $\times$ reconstruction autoencoder}")
tabs(md3, pick=lambda h, r: h.startswith("AEROBLADE"), cap=lambda h: "AEROBLADE TPR (AUROC in brackets) at nominal 1\\% FPR (first table) and 5\\% FPR (second table). Each autoencoder configuration has its own threshold from the calibration photographs.")
A(r"\section{Consistency Decoder: gates, seed stability and content}")
tabs(md3, pick=lambda h, r: h.startswith(("Pre-registered", "Planned")), cap=lambda h: "Planned gates. Per-image seed s.d.: how much one image's score moves when only the decoder seed changes; it can be large even when the mean shift is stable, so we report only rates and mean shifts for this decoder.")
A(r"On the 50-latent subset, TPR at 1\% FPR varies by at most 0.06 across the three seeds for every detector (AEROBLADE 0.08 for all seeds, NPR 0.34--0.38). PSNR of the real-photograph round-trip (first 20 photographs, one seed): 23.2\,dB.")
tabs(md3, pick=lambda h, r: h.startswith("Semantic"), cap=lambda h: "CLIP ViT-L/14 cosine similarity between decodings of the same latent.")
# ---------------- history
A(r"\clearpage\section{Decoder $\times$ generative history}\label{sec:hist}")
tabs(rd(R8 / "report_history.md"), cap=lambda h: ("History sweep, " + nm(h) + ". $D$: decoder effect at img2img strength $s$ ($s{=}0$: encoded photograph); $I=D(s)-D(0)$.") if "Joint" not in h else "Per-photograph slope of $D_k(s)$ over $s\\in\\{0,0.3,0.5,0.7\\}$; sign-flip permutation test (10{,}000), Holm-corrected over 10 tests. A slope summarises a trend and does not imply monotonicity.")
# ---------------- diagnostic
A(r"\clearpage\section{Generator-free screen: all cells}\label{sec:diag}")
A(r"Planned rule (fixed before the held-out results were computed): fit a line on the 10 original cells only; on the 14 held-out detector--decoder conditions the screen must (i) beat the majority-sign baseline in direction, (ii) beat the zero-change baseline in MAE, (iii) flag $\ge2/3$ of held-out large losses ($\ge20$ TPR points). Held-out conditions reuse the same photographs and several share a detector or decoder family, so they are not independent.")
tabs(md3, pick=lambda h, r: ("Out-of-sample" in h) or ("TPR loss" in h), cap=lambda h: "Held-out conditions: " + h + " (as computed in the v3 run).")
tabs(rd(DIAG), cap=lambda h: "Re-analysis from existing scores: " + h.replace("%", "\\%") + ".")
# ---------------- FLUX
A(r"\clearpage\section{FLUX.1}")
A(r"FLUX.1-schnell, $N=100$ (pilot photographs). Probes retrained on FLUX.1 images (50 held-out sources). UnivFD does not detect FLUX.1 images (AUROC 0.557), so its shifts are score changes only. NPR at nominal 1\% FPR with the strictly disjoint calibration set: generated 0.99$\to$0.82 (17 points [9, 26]), img2img 1.00$\to$0.55 (45 [25, 66]); 5\% FPR: 11 [4, 17] and 9 [3, 17]. With the main run's calibration set the img2img value is 0.49.")
tabs(rd(FLX / "report.md"), pick=lambda h, r: h.startswith("Detector") or h.startswith("Decoder quality"),
     cap=lambda h: ("FLUX.1 " + nm(h) + " ($N=100$).") if h.startswith("Detector") else "FLUX.1 VAE vs.\\ TAEF1 reconstruction quality.", drop=("TPR@", "FPR on real-enc", "flips"))
tabs(rd(R8 / "report_extras.md"), pick=lambda h, r: h.startswith("FLUX"), cap=lambda h: "FLUX.1 VAE vs.\\ TAEF1 on the 300 main-run photographs encoded with the FLUX.1 VAE (reconstructions of real photographs, not generated images).")
# ---------------- JPEG
A(r"\clearpage\section{JPEG compression}")
A(r"Every image, including calibration photographs, is JPEG-compressed at the same quality. ``clean thr'': threshold from 1{,}000 clean photographs; ``jpeg thr'': re-set on 300 compressed photographs. swing = max$-$min TPR over decoders at 5\% FPR. Community-Forensics and the Consistency Decoder were not part of this run.")
tabs(rd(R8 / "report_jpeg.md"), cap=lambda h: "JPEG, " + nm(h) + ".")
# ---------------- quality correlations
A(r"\section{Score changes vs.\ reconstruction quality}")
tabs(rd(R8 / "report_with_aeroblade.md"), pick=lambda h, r: h.startswith("Per-image"), cap=lambda h: "Spearman $\\rho$ between the per-image paired score change and the per-image change of each quality measure (* $p<0.05$, uncorrected, exploratory).")
# ---------------- frequency
A(r"\clearpage\section{Frequency and processing interventions (pilot, $N=100$)}")
A(r"For each latent, $A$ is the original-decoder image and $B$ the swapped-decoder image. Radial bands: low $<0.08$, mid $0.08$--$0.25$, high $\ge0.25$ cycles/pixel. Hard-mask results (first set) are superseded by soft raised-cosine masks ($\pm0.02$, second set); hard masks inflated AEROBLADE band effects by about 20\%. \emph{spec.\ match}: $A$ with $B$'s radially averaged amplitude spectrum. \emph{B-band}: $f(B)-f(B$ with that band from $A)$. LPIPS-matched controls: blur, Gaussian noise and JPEG applied to $A$ with per-image severity matched to LPIPS$(A,B)$. Band effects are not exclusive contributions. These are exploratory pilot results.")
tabs(rd(P1 / "report_bandswap.md"), cap=lambda h: "Pilot band swap (hard masks), " + nm(h) + ".")
tabs(rd(P1 / "report_bandswap2.md"), cap=lambda h: ("Pilot band swap v2 (soft masks, LPIPS-matched controls), " + nm(h) + ".") if not h.startswith("Image quality") else "Image quality of the interventions (first 20 latents).")
# ---------------- mitigation
A(r"\clearpage\section{Training probes on several decoders (no claim)}")
A(r"A small proof of concept, not a proposed mitigation: linear probes trained on images from one or more decoders. The CLIP probe is already near-invariant, so the test is uninformative for it; for the DINO probe, adding TAESD images shifts TPR on TAESD images. We make no mitigation claim in the paper.")
tabs(rd(R8 / "report_mitigation.md"), cap=lambda h: "Multi-decoder probe training, " + h + " (test sources $k\\ge150$).")
# ---------------- pilot replication
A(r"\clearpage\section{Pilot run ($N=100$, disjoint photographs)}")
A(r"The pilot was used to develop the analysis; all main effect directions replicate in the main run. (The pilot's automatic ``provisional label'' required $|\Delta|\ge0.5$ and is not used in the paper.)")
tabs(rd(P1 / "report_with_aeroblade.md"), pick=lambda h, r: h.startswith("Detector"), cap=lambda h: "Pilot, " + nm(h) + " ($N=100$; probes 50 held-out sources).", drop=("TPR@", "FPR on real-enc", "flips"))
tabs(rd(P1 / "report_history.md"), pick=lambda h, r: "Joint" in h, cap=lambda h: "Pilot history slopes (Holm-corrected).")
# ---------------- images
A(r"\input{supp_v4}")  # add-on run (v4), generated from results_v4
A(r"\input{supp_v13}")  # B-Free screen test + per-image gallery (hand-written)
A(r"\clearpage\section{Image grids}")
g1 = jpg(R3 / "cons_grid.png", "cons_grid_gen.jpg", crop=(0, 0, 896, 896))
A(r"\begin{figure}[H]\centering\includegraphics[width=0.62\linewidth]{" + g1 + r"}\caption{SD1.5 generated latents (run indices 0--3 of the main run) decoded with, left to right, the original VAE, ft-MSE, TAESD and the Consistency Decoder (seed $3000+k$). Images are downscaled to $224^2$ for display.}\end{figure}")
g2 = jpg(R8 / "grid.png", "grid_sd15.jpg")
A(r"\begin{figure}[H]\centering\includegraphics[width=\linewidth]{" + g2 + r"}\caption{Main run, first four sources. Columns: untouched photograph; encoded photograph decoded with original/ft-MSE/TAESD; generated latent decoded with original/ft-MSE/TAESD; img2img ($\sigma=0.5$) latent decoded with original/ft-MSE.}\end{figure}")
g3 = jpg(FLX / "grid.png", "grid_flux.jpg")
A(r"\begin{figure}[H]\centering\includegraphics[width=\linewidth]{" + g3 + r"}\caption{FLUX.1-schnell, first sources. Columns follow the same layout with the FLUX.1 VAE decoder and TAEF1.}\end{figure}")
g4 = jpg(P1 / "bandswap2_grid.png", "grid_bandswap.jpg")
A(r"\begin{figure}[H]\centering\includegraphics[width=\linewidth]{" + g4 + r"}\caption{Pilot frequency interventions on generated latents (TAESD). Columns: " + rd(P1 / "bandswap2_grid_columns.txt").strip().replace("_", r"\_") + r".}\end{figure}")
A(r"""\section{Code and data availability}
The \texttt{code/} folder of the code release contains the generation, decoding, scoring and analysis scripts used for every number in the paper, the figure scripts, and the environment specification; \texttt{scores/} contains all per-image detector scores (see \texttt{README.md}). Images are not included because of their size; they are regenerated deterministically from the listed seeds. Code and data will be made publicly available upon acceptance.
\refstepcounter{table}\label{tab:last}\addtocounter{table}{-1}
\end{document}""")
(OUT / "supp.tex").write_text("\n".join(T)); print("ok", len(T))
